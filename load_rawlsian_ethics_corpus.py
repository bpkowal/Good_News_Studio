from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import yaml

def sanitize_metadata(metadata):
    safe_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            safe_metadata[key] = value
        elif isinstance(value, list):
            # ✅ Chroma requires metadata to be string, so stringify lists
            safe_metadata[key] = ", ".join(str(v) for v in value)
    return safe_metadata

def load_rawlsian_ethics_corpus(required_tag=None):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(collection_name="rawlsian_ethics", embedding_function=embedder)

    corpus_path = Path("rawlsian_ethics_corpus")
    count_loaded = 0

    for file in corpus_path.glob("*.md"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.startswith("---"):
            print(f"⚠️ Skipped (no metadata): {file.name}")
            continue

        parts = content.split("---", 2)
        if len(parts) < 3:
            print(f"⚠️ Skipped (malformed frontmatter): {file.name}")
            continue

        metadata = yaml.safe_load(parts[1])
        if metadata.get("status") != "approved":
            print(f"⚠️ Skipped (not approved): {file.name}")
            continue

        tags = metadata.get("tags", [])
        if not isinstance(tags, list) or not tags:
            print(f"⚠️ Skipped (no tags): {file.name}")
            continue

        if required_tag and required_tag not in tags:
            continue

        text = parts[2].strip()
        metadata["tags"] = tags  # Keep tags before sanitization
        safe_metadata = sanitize_metadata(metadata)

        vectorstore.add_texts([text], metadatas=[safe_metadata])
        count_loaded += 1

    print(f"✅ Loaded {count_loaded} corpus file(s) into Chroma.")
    return vectorstore