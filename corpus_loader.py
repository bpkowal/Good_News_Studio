import os
from pathlib import Path
import yaml

def load_corpus(corpus_dir):
    documents = []

    for path in Path(corpus_dir).glob("*.md"):
        print(f"🔍 Checking file: {path.name}")  # debug

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                _, frontmatter_raw, body = parts
                try:
                    metadata = yaml.safe_load(frontmatter_raw)
                except yaml.YAMLError as e:
                    print(f"⚠️ YAML parse error in {path.name}: {e}")
                    continue

                print(f"✅ Loaded frontmatter: {metadata}")  # debug

                if metadata.get("status") == "approved":
                    documents.append({
                        "text": body.strip(),
                        "metadata": metadata
                    })
            else:
                print(f"⚠️ Malformed frontmatter in {path.name}")
        else:
            print(f"⚠️ Skipping file without frontmatter: {path.name}")

    return documents