from sentence_transformers import SentenceTransformer, util
import json, yaml
from pathlib import Path

SCENARIO_DIR = Path("scenarios")
CORPUS_DIR = Path("care_ethics_corpus")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Safe for low memory

def load_scenario_tags(scenario_id):
    path = SCENARIO_DIR / f"{scenario_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tag_expectations", {}), data.get("tag_descriptions", {})

def load_corpus_tags():
    tag_descriptions = {}
    for path in CORPUS_DIR.glob("*.md"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                metadata = yaml.safe_load(parts[1])
                tags = metadata.get("tags", [])
                body = parts[2].strip()
                for tag in tags:
                    if tag not in tag_descriptions:
                        tag_descriptions[tag] = " ".join(body.split()[:30]) + "..."
    return tag_descriptions

def get_semantic_tag_weights(scenario_id):
    tag_expectations, scenario_tag_descriptions = load_scenario_tags(scenario_id)
    corpus_tag_descriptions = load_corpus_tags()

    weights = {}
    for s_tag, s_weight in tag_expectations.items():
        s_desc = scenario_tag_descriptions.get(s_tag, s_tag)
        s_vec = model.encode(s_desc, convert_to_tensor=True)
        for c_tag, c_desc in corpus_tag_descriptions.items():
            c_vec = model.encode(c_desc, convert_to_tensor=True)
            score = util.pytorch_cos_sim(s_vec, c_vec).item()
            weights[c_tag] = weights.get(c_tag, 0) + s_weight * score

    return dict(sorted(weights.items(), key=lambda x: -x[1]))