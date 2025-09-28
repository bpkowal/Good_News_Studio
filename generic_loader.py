from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Tuple

# ---- low-memory defaults (Render 512–1024 MB friendly) ----------------------
# Single-thread everything to keep RSS down
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Keep HF caches on the persistent disk (and avoid filling /tmp)
os.environ.setdefault("TRANSFORMERS_CACHE", "/opt/render/data/hf_cache")
os.environ.setdefault("HF_HOME", "/opt/render/data/hf_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/opt/render/data/hf_cache")

# If torch is present, clamp intra-op threads too
try:
    import torch  # noqa: E402
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
except Exception:
    pass
# -----------------------------------------------------------------------------

# Force Hugging Face embeddings on server to avoid OpenAI residency issues.
# If you ever want to switch back, change this assignment to respect env.
os.environ["EP_EMBEDDINGS"] = "hf"
print("[generic_loader] EP_EMBEDDINGS=hf (forced)")

import yaml
from langchain_chroma import Chroma
from chromadb.config import Settings


# In-process cache of Chroma vectorstores to avoid repeated client creation
_STORE_CACHE: Dict[str, Chroma] = {}

def _get_cached_store(name: str) -> Optional[Chroma]:
    store = _STORE_CACHE.get(name)
    if store is not None:
        try:
            print(f"🔁 Reusing cached Chroma store: {name}")
        except Exception:
            pass
    return store


# ---------- Embeddings backend (switchable via env) ----------
def _get_embedder():
    provider = os.getenv("EP_EMBEDDINGS", "hf").lower()
    try:
        print(f"[generic_loader] Using embeddings provider: {provider}")
    except Exception:
        pass
    if provider == "openai":
        # Requires: langchain-openai, openai. Let the integration manage the SDK client.
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("EP_EMBED_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    elif provider in {"hf", "huggingface"}:
        # Use a smaller, RAM-friendly sentence encoder by default
        from langchain_huggingface import HuggingFaceEmbeddings
        model = os.getenv("EP_HF_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
        return HuggingFaceEmbeddings(
            model_name=model,
            # ensure CPU on small instances; avoids accidental CUDA init
            model_kwargs={"device": "cpu"},
            # tiny batches + normalized vectors = lower peak memory & stable sims
            encode_kwargs={"batch_size": 8, "normalize_embeddings": True, "show_progress_bar": False},
        )
    else:
        raise RuntimeError(f"Unknown EP_EMBEDDINGS provider: {provider}")


# ---------- Helpers ----------
def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    if not metadata:
        return safe
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif isinstance(v, list):
            # Chroma metadata must be simple types; stringify lists
            safe[k] = ", ".join(str(x) for x in v)
    return safe


def _iter_markdown_docs(corpus_dir: Path, required_tag: Optional[str]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yields (text, metadata) for each approved .md doc, optionally filtered by tag.
    """
    for file in corpus_dir.glob("*.md"):
        content = file.read_text(encoding="utf-8")

        if not content.startswith("---"):
            print(f"⚠️ Skipped (no metadata): {file.name}")
            continue

        parts = content.split("---", 2)
        if len(parts) < 3:
            print(f"⚠️ Skipped (malformed frontmatter): {file.name}")
            continue

        md = yaml.safe_load(parts[1]) or {}
        if md.get("status") != "approved":
            print(f"⚠️ Skipped (not approved): {file.name}")
            continue

        tags = md.get("tags", [])
        if not isinstance(tags, list) or not tags:
            print(f"⚠️ Skipped (no tags): {file.name}")
            continue

        if required_tag and required_tag not in tags:
            continue

        text = parts[2].strip()
        md["tags"] = tags  # keep original tags
        yield text, _sanitize_metadata(md)


# ---------- Core API ----------
def load_corpus(
    collection_name: str,
    corpus_dir: str | Path,
    *,
    required_tag: Optional[str] = None,
) -> Chroma:
    """
    Render-friendly loader for ANY corpus with persistent Chroma.

    Env controls:
      - CHROMA_PERSIST_DIR (default: "/opt/render/data/chroma")  -> data saved under {base}/{collection}
      - EP_EMBEDDINGS ("openai" | "hf"; default: "openai")
      - EP_EMBED_MODEL (default: "text-embedding-3-small")
      - EP_HF_MODEL (default: "sentence-transformers/all-MiniLM-L6-v2")
      - EP_BUILD_INDEX_ON_BOOT ("1" to (re)index; default: "0")
    """
    # Fast path: reuse already-opened store in this process
    cached = _get_cached_store(collection_name)
    if cached is not None:
        return cached

    # Persist under the Render disk if present; default to /opt/render/data/chroma
    base = os.getenv("CHROMA_PERSIST_DIR", "/opt/render/data/chroma")
    persist_dir = str(Path(base) / collection_name)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    try:
        print(f"[generic_loader] Persist dir for {collection_name}: {persist_dir}")
    except Exception:
        pass

    embedder = _get_embedder()
    # Use explicit client settings compatible with Chroma 1.x (avoid legacy env config)
    client_settings = Settings(anonymized_telemetry=False)
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_dir,
        client_settings=client_settings,
    )

    build = os.getenv("EP_BUILD_INDEX_ON_BOOT", "0") == "1"
    if not build:
        # Just open persisted store
        _STORE_CACHE[collection_name] = vs
        return vs

    # (Re)index from markdown
    texts, metas = [], []
    count = 0
    for text, meta in _iter_markdown_docs(Path(corpus_dir), required_tag):
        texts.append(text)
        metas.append(meta)
        if len(texts) >= 16:  # micro-batch to bound memory
            vs.add_texts(texts, metadatas=metas)
            count += len(texts)
            texts, metas = [], []
    if texts:
        vs.add_texts(texts, metadatas=metas)
        count += len(texts)

    # Chroma 1.x with PersistentClient writes to disk automatically.
    # Older langchain-chroma exposed `persist()`. Guard for either behavior.
    try:
        persist_fn = getattr(vs, "persist", None)
        if callable(persist_fn):
            persist_fn()
    except Exception:
        # Best-effort; not critical on 1.x
        pass

    print(f"✅ [{collection_name}] Loaded {count} file(s) into Chroma → {persist_dir}")
    _STORE_CACHE[collection_name] = vs
    return vs


def load_all_corpora(ep_corpora: Optional[str] = None) -> Dict[str, Chroma]:
    """
    Bulk loader for multiple corpora.

    Pass EP_CORPORA by env or argument in one of two formats:
      1) JSON list of {"collection": "...", "dir": "...", "tag": "...?"}
         Example:
           EP_CORPORA='[
             {"collection":"nozick_ethics","dir":"nozick_corpus"},
             {"collection":"rawls_ethics","dir":"rawls_corpus","tag":"distributive_justice"}
           ]'
      2) Comma list of "collection:dir[:tag]" items
         Example:
           EP_CORPORA="nozick_ethics:nozick_corpus,rawls_ethics:rawls_corpus:justice"

    Returns a dict {collection_name: Chroma}
    """
    config_text = ep_corpora or os.getenv("EP_CORPORA", "")
    items: Iterable[Tuple[str, str, Optional[str]]] = []

    if config_text.strip().startswith("["):
        # JSON format
        try:
            data = json.loads(config_text)
            for it in data:
                items.append((it["collection"], it["dir"], it.get("tag")))
        except Exception as e:
            raise RuntimeError(f"Invalid EP_CORPORA JSON: {e}")
    elif config_text.strip():
        # comma list format
        for part in config_text.split(","):
            segs = [s.strip() for s in part.split(":")]
            if len(segs) < 2:
                raise RuntimeError(f"Invalid EP_CORPORA item (need 'collection:dir[:tag]'): {part}")
            collection, d = segs[0], segs[1]
            tag = segs[2] if len(segs) > 2 else None
            items.append((collection, d, tag))
    else:
        # Nothing specified → return empty (caller can decide defaults)
        return {}

    stores: Dict[str, Chroma] = {}
    for collection, d, tag in items:
        stores[collection] = load_corpus(collection, d, required_tag=tag)
    return stores


# ---------- Convenience: default set of all known corpora ----------
def load_default_corpora() -> Dict[str, Chroma]:
    """
    Convenience wrapper to load all six standard corpora used in the project.
    Equivalent to setting EP_CORPORA to include all of them.

    Collections & dirs (1:1):
      - nozick_corpus
      - care_ethics_corpus
      - deontological_corpus
      - virtue_ethics_corpus
      - utilitarian_corpus
      - rawlsian_ethics_corpus
    """
    defaults = [
        ("nozick_corpus", "nozick_corpus", None),
        ("care_ethics_corpus", "care_ethics_corpus", None),
        ("deontological_corpus", "deontological_corpus", None),
        ("virtue_ethics_corpus", "virtue_ethics_corpus", None),
        ("utilitarian_corpus", "utilitarian_corpus", None),
        ("rawlsian_ethics_corpus", "rawlsian_ethics_corpus", None),
    ]
    stores: Dict[str, Chroma] = {}
    for collection, d, tag in defaults:
        stores[collection] = load_corpus(collection, d, required_tag=tag)
    return stores