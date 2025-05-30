import re
import time
import json
from pathlib import Path
from datetime import datetime
from llama_cpp import Llama

# === Setup ===
MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=768, n_threads=6, n_gpu_layers=60, verbose=False)

SCENARIO_DIR = Path("scenarios")
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

# === Tag Parsing Fallback ===

def extract_tags(raw_output):
    # Try JSON parse first
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        print("⚠️ Attempting regex fallback...")

        # Match numbered list or dashes or quotes
        matches = re.findall(r'(?:\d+\.\s*|\-\s*|["“”]?)([A-Za-z][A-Za-z\s\-]+)', raw_output)

        # De-duplicate and clean
        cleaned = []
        for tag in matches:
            tag = tag.strip().title()
            if tag not in cleaned and len(tag) > 2:
                cleaned.append(tag)

        if len(cleaned) >= 3:
            print(f"✅ Extracted tags: {cleaned[:5]}")
            return cleaned[:5]

        print("❌ Still could not extract a valid tag list.")
        raise ValueError("Tag parsing failed.")

# === Tag Description Generator ===
def describe_tag(tag):
    prompt = f"""
You are an ethics assistant.

Define the ethical tag "{tag}" in 8 words or fewer.
- Use neutral ethical language.
- Do NOT reference any specific scenario.
- Avoid school-specific terms like utility, duty, or care.

Definition:
"""
    time.sleep(3)
    response = llm(prompt.strip(), max_tokens=16, temperature=0.3)
    return response["choices"][0]["text"].strip()

# === Scenario Submission Handler ===
def generate_and_save_scenario(scenario_text):
    now = datetime.now().strftime("%H%M_%Y%m%d")
    scenario_id = f"auto_ethics_{now}"
    scenario_type = now[-4:]

    print("📨 Submitted. LLaMA is processing. Please wait...")

    tag_prompt = f"""
You are an ethics assistant.

Return exactly 5 short ethical tags relevant to this scenario. Tags should be one or two words only and not reference any particular school of ethics.

Scenario: {scenario_text}

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
        desc = describe_tag(tag)
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

def build_scenario(scenario_text, id_override=None):
    now = datetime.now().strftime("%H%M_%Y%m%d")
    scenario_id = id_override or f"auto_ethics_{now}"
    scenario_type = now[-4:]

    print("📨 Submitted. LLaMA is processing. Please wait...")

    tag_prompt = f"""
You are an ethics assistant.

Return exactly 5 short ethical tags relevant to this scenario. Tags should be one or two words only and not reference any particular school of ethics.

Scenario: {scenario_text}

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
        raise

    tag_expectations = {tag: round(2.0 - 0.2 * i, 1) for i, tag in enumerate(tags)}
    tag_descriptions = {}

    print("🧠 Generating tag descriptions...")
    for tag in tags:
        desc = describe_tag(tag)
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
    return scenario_data


# === Entry Point ===
if __name__ == "__main__":
    print("📝 Enter your ethical scenario (single paragraph). Press Enter when done.\n")
    scenario_text = input("> ").strip()
    generate_and_save_scenario(scenario_text)
    