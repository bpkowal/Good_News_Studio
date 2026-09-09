"""Shared, typed corpus ingestion and conservative framework retrieval.

The source ethical agents use this module to keep Markdown conventions, evidence
strength, provenance, and vector-store ingestion consistent.  Retrieval strength
is deliberately distinct from philosophical authority: a semantically adjacent
passage may illustrate an argument, but it may not silently become a decisive
premise for the specialist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


RETRIEVAL_VERSION = "typed-framework-rag-v1"


class EvidenceTier(str, Enum):
    CORE = "CORE"
    ADJACENT = "ADJACENT"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class EvidenceThresholds:
    """Similarity bands calibrated for MiniLM scenario-to-framework retrieval."""

    core: float = 0.42
    adjacent: float = 0.30
    tag_boost_cap: float = 0.06

    def __post_init__(self) -> None:
        if not 0.0 <= self.adjacent < self.core <= 1.0:
            raise ValueError("Evidence thresholds must satisfy 0 <= adjacent < core <= 1")


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    framework: str
    source_file: str
    body: str
    metadata: Mapping[str, Any]

    @property
    def tags(self) -> tuple[str, ...]:
        raw = self.metadata.get("tags", ())
        if isinstance(raw, str):
            return tuple(tag.strip() for tag in raw.split(",") if tag.strip())
        if isinstance(raw, Sequence):
            return tuple(str(tag).strip() for tag in raw if str(tag).strip())
        return ()


@dataclass(frozen=True)
class CorpusPassage:
    passage_id: str
    document_id: str
    framework: str
    source_file: str
    text: str
    source_kind: str
    metadata: Mapping[str, Any]

    @property
    def tags(self) -> tuple[str, ...]:
        raw = self.metadata.get("tags", ())
        if isinstance(raw, str):
            return tuple(tag.strip() for tag in raw.split(",") if tag.strip())
        if isinstance(raw, Sequence):
            return tuple(str(tag).strip() for tag in raw if str(tag).strip())
        return ()


@dataclass(frozen=True)
class RetrievedEvidence:
    passage: CorpusPassage
    semantic_score: float
    tag_boost: float
    final_score: float
    tier: EvidenceTier
    framework_score: float = 0.0
    case_score: float = 0.0


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[RetrievedEvidence, ...]
    candidate_count: int
    rejected_count: int
    top_rejected_score: float | None
    expanded_query: str

    @property
    def has_core_evidence(self) -> bool:
        return any(item.tier is EvidenceTier.CORE for item in self.evidence)


def _is_retracted(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _document_id(framework: str, source_file: str) -> str:
    identity = f"{framework}:{source_file}".encode("utf-8")
    return f"{framework}-{sha256(identity).hexdigest()[:16]}"


def load_approved_documents(
    corpus_dir: str | Path,
    *,
    framework: str,
    required_tag: str | None = None,
) -> list[CorpusDocument]:
    """Load approved, non-retracted Markdown documents in stable file order."""

    documents: list[CorpusDocument] = []
    for path in sorted(Path(corpus_dir).glob("*.md"), key=lambda item: item.name):
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            continue
        parts = content.split("---", 2)
        if len(parts) < 3:
            continue
        metadata = yaml.safe_load(parts[1]) or {}
        if not isinstance(metadata, dict):
            continue
        if metadata.get("status") != "approved" or _is_retracted(metadata.get("retracted", False)):
            continue
        tags = metadata.get("tags", [])
        if not isinstance(tags, list) or not tags:
            continue
        if required_tag and required_tag not in tags:
            continue
        body = parts[2].strip()
        if not body:
            continue
        documents.append(
            CorpusDocument(
                document_id=_document_id(framework, path.name),
                framework=framework,
                source_file=path.name,
                body=body,
                metadata=dict(metadata),
            )
        )
    return documents


_OPENING_QUOTES = ('"', "“", "‘")


def _normalize_paragraph(raw: str) -> tuple[str, str]:
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    is_blockquote = bool(lines) and all(line.startswith(">") for line in lines)
    if is_blockquote:
        text = " ".join(line[1:].strip() for line in lines)
        return text, "direct_quote"
    text = " ".join(lines)
    if text.startswith(_OPENING_QUOTES):
        return text, "direct_quote"
    return text, "commentary"


def _split_long_text(text: str, max_chars: int) -> Iterable[str]:
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        break_at = max(window.rfind(". "), window.rfind("; "), window.rfind(", "))
        if break_at < max_chars // 2:
            break_at = window.rfind(" ")
        if break_at <= 0:
            break_at = max_chars
        yield remaining[: break_at + 1].strip()
        remaining = remaining[break_at + 1 :].strip()
    if remaining:
        yield remaining


def document_passages(
    document: CorpusDocument,
    *,
    max_chars: int = 600,
) -> list[CorpusPassage]:
    """Turn both Markdown blockquotes and typographic quotes into typed passages."""

    passages: list[CorpusPassage] = []
    paragraph_index = 0
    for raw in re.split(r"\n\s*\n", document.body):
        text, source_kind = _normalize_paragraph(raw)
        if not text:
            continue
        chunks = (
            (text[:max_chars].rstrip(),)
            if source_kind == "direct_quote" and len(text) > max_chars
            else _split_long_text(text, max_chars=max_chars)
        )
        for chunk in chunks:
            passage_id = f"{document.document_id}:p{paragraph_index}"
            passages.append(
                CorpusPassage(
                    passage_id=passage_id,
                    document_id=document.document_id,
                    framework=document.framework,
                    source_file=document.source_file,
                    text=chunk,
                    source_kind=source_kind,
                    metadata=document.metadata,
                )
            )
            paragraph_index += 1
    return passages


def load_corpus_passages(
    corpus_dir: str | Path,
    *,
    framework: str,
    required_tag: str | None = None,
    max_chars: int = 600,
) -> list[CorpusPassage]:
    passages: list[CorpusPassage] = []
    for document in load_approved_documents(
        corpus_dir, framework=framework, required_tag=required_tag
    ):
        passages.extend(document_passages(document, max_chars=max_chars))
    return passages


def corpus_fingerprint(corpus_dir: str | Path, *, framework: str) -> str:
    digest = sha256(f"{RETRIEVAL_VERSION}:{framework}".encode("utf-8"))
    for path in sorted(Path(corpus_dir).glob("*.md"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def load_explicit_tag_weights(
    scenario_id: str,
    *,
    scenario_path: str | Path | None = None,
    scenario_dir: str | Path = "scenarios",
) -> dict[str, float]:
    """Load authored tag weights without generating a second semantic ontology."""

    path = (
        Path(scenario_path)
        if scenario_path is not None
        else Path(scenario_dir) / f"{scenario_id}.json"
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("tag_expectations", {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(tag): float(weight)
        for tag, weight in raw.items()
        if isinstance(weight, (int, float)) and float(weight) > 0.0
    }


def sanitize_chroma_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    safe: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = ", ".join(str(item) for item in value)
    return safe


def load_chroma_corpus(
    corpus_dir: str | Path,
    *,
    collection_name: str,
    framework: str,
    required_tag: str | None = None,
    embedder: Any | None = None,
) -> Any:
    """Compatibility loader for agents not yet migrated to passage retrieval."""

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_function = embedder or HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    documents = load_approved_documents(
        corpus_dir, framework=framework, required_tag=required_tag
    )
    if documents:
        metadatas = []
        for document in documents:
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "document_id": document.document_id,
                    "framework": framework,
                    "source_file": document.source_file,
                }
            )
            metadatas.append(sanitize_chroma_metadata(metadata))
        vectorstore.add_texts(
            [document.body for document in documents],
            metadatas=metadatas,
            ids=[document.document_id for document in documents],
        )
    return vectorstore


def _cosine_similarity(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=float).reshape(-1)
    right_array = np.asarray(right, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def classify_evidence(
    score: float,
    *,
    identity_compatible: bool,
    thresholds: EvidenceThresholds,
) -> EvidenceTier:
    if score >= thresholds.core and identity_compatible:
        return EvidenceTier.CORE
    if score >= thresholds.adjacent:
        return EvidenceTier.ADJACENT
    return EvidenceTier.REJECTED


def retrieve_framework_evidence(
    passages: Sequence[CorpusPassage],
    *,
    query: str,
    embedder: Any,
    query_lens: str,
    identity_tags: Iterable[str] = (),
    core_evidence_roles: Iterable[str] = (),
    tag_weights: Mapping[str, float] | None = None,
    thresholds: EvidenceThresholds = EvidenceThresholds(),
    limit: int = 3,
    prefer_direct_quotes: bool = False,
    separate_identity_scoring: bool = False,
    framework_weight: float = 0.65,
) -> RetrievalResult:
    """Retrieve diverse evidence while bounding adjacent material's influence."""

    mix = min(1.0, max(0.0, float(framework_weight)))
    expanded_query = f"{query_lens.strip()}\nCase: {query.strip()}".strip()
    if not passages:
        return RetrievalResult((), 0, 0, None, expanded_query)

    query_embedding = embedder.embed_query(expanded_query)
    framework_embedding = (
        embedder.embed_query(query_lens.strip()) if separate_identity_scoring else None
    )
    case_embedding = embedder.embed_query(query.strip()) if separate_identity_scoring else None
    passage_embeddings = embedder.embed_documents([passage.text for passage in passages])
    identity_tag_set = {tag.strip().lower() for tag in identity_tags if tag.strip()}
    core_role_set = {role.strip().lower() for role in core_evidence_roles if role.strip()}
    weights = tag_weights or {}
    candidates: list[RetrievedEvidence] = []
    rejected_scores: list[float] = []

    for passage, embedding in zip(passages, passage_embeddings):
        combined_score = max(0.0, _cosine_similarity(query_embedding, embedding))
        if separate_identity_scoring:
            framework_score = max(
                0.0, _cosine_similarity(framework_embedding, embedding)
            )
            case_score = max(0.0, _cosine_similarity(case_embedding, embedding))
            semantic_score = mix * framework_score + (1.0 - mix) * case_score
            classification_score = framework_score
        else:
            framework_score = combined_score
            case_score = combined_score
            semantic_score = combined_score
            classification_score = combined_score
        raw_tag_support = sum(max(0.0, float(weights.get(tag, 0.0))) for tag in passage.tags)
        tag_boost = min(thresholds.tag_boost_cap, raw_tag_support * 0.04)
        final_score = min(1.0, semantic_score + tag_boost)
        normalized_tags = {tag.lower() for tag in passage.tags}
        identity_compatible = not identity_tag_set or bool(identity_tag_set & normalized_tags)
        if core_role_set:
            framework_role = str(passage.metadata.get("framework_role", "")).strip().lower()
            identity_compatible = identity_compatible and framework_role in core_role_set
        tier = classify_evidence(
            min(1.0, classification_score + tag_boost),
            identity_compatible=identity_compatible,
            thresholds=thresholds,
        )
        item = RetrievedEvidence(
            passage=passage,
            semantic_score=semantic_score,
            tag_boost=tag_boost,
            final_score=final_score,
            tier=tier,
            framework_score=framework_score,
            case_score=case_score,
        )
        if tier is EvidenceTier.REJECTED:
            rejected_scores.append(final_score)
        else:
            candidates.append(item)

    def rank_key(item: RetrievedEvidence) -> tuple[int, float]:
        direct_priority = int(
            prefer_direct_quotes and item.passage.source_kind == "direct_quote"
        )
        return direct_priority, item.final_score

    core = sorted(
        (item for item in candidates if item.tier is EvidenceTier.CORE),
        key=rank_key,
        reverse=True,
    )
    adjacent = sorted(
        (item for item in candidates if item.tier is EvidenceTier.ADJACENT),
        key=rank_key,
        reverse=True,
    )

    selected: list[RetrievedEvidence] = []
    used_documents: set[str] = set()

    def add_diverse(items: Sequence[RetrievedEvidence], slots: int) -> None:
        for item in items:
            if len(selected) >= limit or slots <= 0:
                return
            if item.passage.document_id in used_documents:
                continue
            selected.append(item)
            used_documents.add(item.passage.document_id)
            slots -= 1

    add_diverse(core, limit)
    adjacent_slots = 1 if selected else min(2, limit)
    add_diverse(adjacent, adjacent_slots)

    return RetrievalResult(
        evidence=tuple(selected),
        candidate_count=len(passages),
        rejected_count=len(rejected_scores),
        top_rejected_score=max(rejected_scores) if rejected_scores else None,
        expanded_query=expanded_query,
    )


def format_evidence_context(result: RetrievalResult) -> str:
    if not result.evidence:
        return (
            "[RAG_CONTEXT_UNAVAILABLE] No corpus passage met the adjacent-evidence "
            "threshold. Reason from the framework specification and scenario facts only."
        )

    lines = [
        "Evidence-use rule: CORE passages may ground framework reasoning. ADJACENT passages",
        "are interpretive context only; they may not determine the recommendation, override",
        "scenario facts, or replace the specialist's own framework commitments.",
    ]
    for index, item in enumerate(result.evidence, start=1):
        metadata = item.passage.metadata
        author = str(metadata.get("author", "Unknown author"))
        source = str(metadata.get("source", metadata.get("title", "Unknown source")))
        framework_role = str(metadata.get("framework_role", "unspecified"))
        lines.extend(
            [
                "",
                (
                    f"[E{index} | {item.tier.value} | framework={item.framework_score:.2f} "
                    f"| case={item.case_score:.2f} | rank={item.final_score:.2f} "
                    f"| role={framework_role} | kind={item.passage.source_kind} | {author}, {source}]"
                ),
                item.passage.text,
            ]
        )
    return "\n".join(lines)
