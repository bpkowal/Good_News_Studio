import re
import time
import json
import os
from pathlib import Path
from datetime import datetime
from llama_cpp import Llama

# === Setup ===
MODEL_PATH = "../llama-2-7b-chat.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=768, n_threads=6, n_gpu_layers=60, verbose=False)

SCENARIO_DIR = Path("scenarios")
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

# === Tag Parsing Fallback ===
def extract_tags(raw_output):
    # Try JSON parsing first
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        print("⚠️ Attempting regex fallback...")

        # Try numbered or bulleted list fallback
        matches = re.findall(r'\d+\.\s*([\w\s\-]+)', raw_output)  # handles "1. Trust"
        if not matches:
            matches = re.findall(r'-\s*([\w\s\-]+)', raw_output)   # handles "- Trust"
        if matches and len(matches) >= 3:
            tags = [m.strip().lower() for m in matches[:5]]
            print(f"✅ Fallback extracted tags: {tags}")
            return tags

        raise ValueError("❌ Could not extract valid tag list.")

# === Tag Description Generator ===
def describe_tag(tag, scenario_text):
    prompt = f"""You are a care ethics assistant.

Define the ethical tag "{tag}" in 8 words or fewer.
- Use care ethics language (e.g. empathy, trust, context).
- Do NOT reference the scenario or use examples.
- Keep it abstract and reusable.

Definition:"""
    time.sleep(3)
    response = llm(prompt, max_tokens=16, temperature=0.3)
    return response["choices"][0]["text"].strip()


# === Scenario Submission Handler ===
def generate_and_save_scenario(scenario_text):
    now = datetime.now().strftime("%H%M_%Y%m%d")
    scenario_id = f"auto_care_ethics_{now}"
    scenario_type = now[-4:]

    print("📨 Submitted. LLaMA is processing. Please wait...")

    # === Prompt for Tags ===
    tag_prompt = f"""
You are a care ethics assistant.

Given an ethical scenario, return **only** a list of 5 short, one- or two-word tags relevant to care ethics.

### Example:
Scenario: A nurse must choose between staying late to care for a patient or attending her child’s school play.

Tags:
["care prioritization", "emotional conflict", "responsibility", "relational ethics", "moral attention"]

### Scenario:
{scenario_text}

Tags:
"""

    print("⏳ Generating tags...")
    tag_output = llm(tag_prompt.strip(), max_tokens=256, temperature=0.3)
    raw_output = tag_output["choices"][0]["text"].strip()
    print(f"🧾 Raw output was:\n{raw_output}\n")

    try:
        tags = extract_tags(raw_output)
    except Exception as e:
        print(f"❌ Could not generate valid tags. {e}")
        return

    tag_expectations = {tag: round(2.0 - 0.2 * i, 1) for i, tag in enumerate(tags)}
    tag_descriptions = {}

    print("🧠 Generating tag descriptions...")
    for tag in tags:
        desc = describe_tag(tag, scenario_text)
        print(f"  • {tag}: {desc}")
        tag_descriptions[tag] = desc

    scenario_data = {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "ethical_question": scenario_text.strip(),
        "tags": tags,
        "tag_expectations": tag_expectations,
        "tag_descriptions": tag_descriptions
    }

    output_path = SCENARIO_DIR / f"{scenario_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenario_data, f, indent=2)
    print(f"\n💾 Scenario saved to {output_path.name}")

# === Entry Point ===
if __name__ == "__main__":
    print("📝 Enter your ethical scenario (single paragraph). Press Enter when done.\n")
    scenario_text = input("> ").strip()
    generate_and_save_scenario(scenario_text)