import re
import time
import json
import os
from pathlib import Path
from datetime import datetime
from llama_cpp import Llama

# === Config ===
MODEL_PATH = "../llama-2-7b-chat.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=768, n_threads=6, n_gpu_layers=60, verbose=False)

# Define each ethics agent and its folder + prompt prefix
AGENTS = {
    "care": {
        "prefix": "You are a care ethics assistant.",
        "folder": "care_scenarios"
    },
    "deontological": {
        "prefix": "You are a deontological ethics assistant.",
        "folder": "deontology_scenarios"
    },
    "utilitarian": {
        "prefix": "You are a utilitarian ethics assistant.",
        "folder": "utilitarian_scenarios"
    },
    "virtue": {
        "prefix": "You are a virtue ethics assistant.",
        "folder": "virtue_scenarios"
    }
}

BASE_DIR = Path(".")
for agent in AGENTS.values():
    (BASE_DIR / agent["folder"]).mkdir(parents=True, exist_ok=True)

# === Tag extraction helper ===
def extract_tags(raw_output):
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        matches = re.findall(r'\d+\.\s*([\w\s\-]+)', raw_output)
        if not matches:
            matches = re.findall(r'-\s*([\w\s\-]+)', raw_output)
        if matches and len(matches) >= 3:
            tags = [m.strip().lower() for m in matches[:5]]
            return tags
        raise ValueError("Could not extract valid tag list.")

# === Description generator ===
def describe_tag(prefix, tag):
    prompt = f"""{prefix}

Define the ethical tag "{tag}" in 8 words or fewer.
- Use language consistent with the above ethics framework.
- Keep it abstract and reusable.

Definition:"""
    time.sleep(1)
    resp = llm(prompt, max_tokens=16, temperature=0.3)
    return resp["choices"][0]["text"].strip()

# === Scenario creator ===
def generate_and_save_scenario(agent_key, scenario_text):
    agent = AGENTS[agent_key]
    now = datetime.now().strftime("%H%M_%Y%m%d")
    scenario_id = f"auto_{agent_key}_{now}"
    scenario_type = now[-4:]
    folder = BASE_DIR / agent["folder"]

    # Tag generation prompt
    tag_prompt = f"""{agent['prefix']}

Given an ethical scenario, return only a JSON list of 5 short tags relevant to this framework.

Scenario:
{scenario_text}

Tags:
"""
    raw = llm(tag_prompt.strip(), max_tokens=256, temperature=0.3)["choices"][0]["text"].strip()
    try:
        tags = extract_tags(raw)
    except Exception as e:
        print(f"[{agent_key}] Tag extraction error: {e}")
        return

    tag_expectations = {tag: round(2.0 - 0.2 * i, 1) for i, tag in enumerate(tags)}
    tag_descriptions = {}
    for tag in tags:
        desc = describe_tag(agent["prefix"], tag)
        tag_descriptions[tag] = desc

    data = {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "ethical_question": scenario_text.strip(),
        "tags": tags,
        "tag_expectations": tag_expectations,
        "tag_descriptions": tag_descriptions
    }

    out_path = folder / f"{scenario_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[{agent_key}] Scenario saved to {out_path}")

# === Main ===
if __name__ == "__main__":
    print("Enter your ethical scenario (single paragraph):")
    scenario_text = input("> ").strip()
    for key in AGENTS:
        print(f"[{key}] Generating scenario...")
        generate_and_save_scenario(key, scenario_text)