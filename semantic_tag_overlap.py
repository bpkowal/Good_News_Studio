import json
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# === Setup ===
model = SentenceTransformer("all-MiniLM-L6-v2")
SCENARIO_DIR = Path("scenarios")
CORPUS_DIR = Path("care_ethics_corpus")

def get_semantic_tag_weights(scenario_id):
    """
    Returns a dict of corpus tags with weights blended from scenario tag expectations
    and semantic similarity to corpus tag content.
    """
    try:
        scenario_path = SCENARIO_DIR / f"{scenario_id}.json"
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)
        tag_expectations = scenario_data.get("tag_expectations", {})
    except Exception as e:
        print(f"❌ Error loading scenario '{scenario_id}': {e}")
        return {}

    print(f"\n🔍 Scenario Tags: {list(tag_expectations)}")

    scenario_vecs = {
        tag: model.encode(tag, convert_to_tensor=True)
        for tag in tag_expectations
    }

    # === Load corpus tag descriptions ===
    tag_descriptions = {}
    for path in CORPUS_DIR.glob("*.md"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.startswith("---"):
                continue
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
            metadata = yaml.safe_load(parts[1])
            tags = metadata.get("tags", [])
            body = parts[2].strip()
            for tag in tags:
                if tag not in tag_descriptions:
                    tag_descriptions[tag] = " ".join(body.split()[:30]) + "..."
        except Exception as e:
            print(f"⚠️ Skipping {path.name} due to error: {e}")
            continue

    # === Semantic scoring ===
    result_weights = dict(tag_expectations)  # Start with scenario-defined weights

    for s_tag, s_vec in scenario_vecs.items():
        s_weight = tag_expectations.get(s_tag, 0.0)
        for c_tag, c_desc in tag_descriptions.items():
            try:
                c_vec = model.encode(c_desc, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(s_vec, c_vec).item()
                if similarity > 0.55:
                    if c_tag not in result_weights:
                        result_weights[c_tag] = 0.0
                    result_weights[c_tag] += s_weight * similarity * 0.5
            except Exception as e:
                print(f"⚠️ Error comparing {s_tag} and {c_tag}: {e}")
                continue

    print(f"✅ Expanded tag weights: {result_weights}")
    return result_weights