import re
import time
import json
from pathlib import Path
from datetime import datetime
import argparse
import os
import requests

# === Setup ===
# Switchable tiny-fast API LLM (default: Groq Llama 3.1 8B Instant)
# Configure via env vars:
#   PROVIDER = 'groq' | 'openai'
#   GROQ_API_KEY, OPENAI_API_KEY
#   GROQ_MODEL (default 'llama-3.1-8b-instant')
#   OPENAI_MODEL (default 'gpt-4o-mini')
PROVIDER = os.getenv('PROVIDER', 'openai').lower()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

def llm_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
    """Unified completion wrapper returning the assistant text string."""
    if PROVIDER == 'groq':
        if not GROQ_API_KEY:
            raise RuntimeError("Missing GROQ_API_KEY for provider 'groq'.")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    elif PROVIDER == 'openai':
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY for provider 'openai'.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    else:
        raise ValueError(f"Unknown PROVIDER '{PROVIDER}'. Use 'groq' or 'openai'.")

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
    time.sleep(0.2)  # gentle pacing
    response_text = llm_complete(prompt.strip(), max_tokens=16, temperature=0.3)
    return response_text

# === Scenario Submission Handler ===
def generate_and_save_scenario(scenario_text):
    now = datetime.now().strftime("%H%M_%Y%m%d")
    scenario_id = f"auto_ethics_{now}"
    scenario_type = now[-4:]

    print("📨 Submitted. LLaMA is processing. Please wait...")

    tag_prompt = f"""
You are an ethics assistant.

Return exactly 5 short ethical tags relevant to this scenario. Tags should be one or two words only and not reference any particular school of ethics. Output one tag per line with no numbering or punctuation.

Scenario: {scenario_text}

Tags:
"""

    print("⏳ Generating tags...")
    tag_output = llm_complete(tag_prompt.strip(), max_tokens=64, temperature=0.2)
    raw_output = tag_output
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

Return exactly 5 short ethical tags relevant to this scenario. Tags should be one or two words only and not reference any particular school of ethics. Output one tag per line with no numbering or punctuation.

Scenario: {scenario_text}

Tags:
"""

    print("⏳ Generating tags...")
    tag_output = llm_complete(tag_prompt.strip(), max_tokens=64, temperature=0.2)
    raw_output = tag_output
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
    parser = argparse.ArgumentParser(description="Scenario Builder (interactive or non-interactive)")
    parser.add_argument("--scenario", type=str, help="Scenario text to process (non-interactive)")
    parser.add_argument("--scenario-file", type=str, help="Path to a file containing the scenario")
    parser.add_argument("--id", type=str, default=None, help="Optional scenario_id override")
    args = parser.parse_args()

    scenario_text: str | None = None

    # Priority: --scenario > --scenario-file > interactive prompt
    if args.scenario:
        scenario_text = args.scenario.strip()
    elif args.scenario_file:
        p = Path(args.scenario_file)
        if not p.exists():
            raise FileNotFoundError(f"Scenario file not found: {p}")
        text = p.read_text(encoding="utf-8").strip()
        # If it's JSON, try to pull `ethical_question`; otherwise use raw text
        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(text)
                scenario_text = (obj.get("ethical_question") or obj.get("scenario") or text).strip()
            except json.JSONDecodeError:
                scenario_text = text
        else:
            scenario_text = text
    else:
        print("📝 Enter your ethical scenario (single paragraph). Press Enter when done.\n")
        scenario_text = input("> ").strip()

    if not scenario_text:
        raise SystemExit("No scenario text provided.")

    # Non-interactive and interactive both funnel here. Keep behavior identical.
    # Use build_scenario so we can optionally override id when provided.
    try:
        build_scenario(scenario_text, id_override=args.id)
    except Exception as e:
        print(f"❌ Could not build scenario: {e}")
        raise