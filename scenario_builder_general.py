import re
import time
import json
from pathlib import Path
from datetime import datetime
import argparse
import os
import requests

# === Setup ===
# Switchable tiny-fast API LLM
# Configure via env vars:
#   PROVIDER = 'openai'
#   OPENAI_API_KEY
#   OPENAI_MODEL (default 'gpt-4o-mini')
PROVIDER = os.getenv('PROVIDER', 'openai').lower()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

def llm_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
    """Unified completion wrapper returning the assistant text string."""
    if PROVIDER == 'openai':
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY for provider 'openai'.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    else:
        raise ValueError(f"Unknown PROVIDER '{PROVIDER}'. Use 'openai'.")

SCENARIO_DIR = Path("scenarios")
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

# === Tag Parsing Fallback ===

def extract_tags(raw_output):
    # Try JSON parse first
    try:
        tags = json.loads(raw_output)
        if isinstance(tags, list):
            cleaned = []
            for t in tags:
                if not isinstance(t, str):
                    continue
                # Normalize and sanitize
                t = re.sub(r'[^A-Za-z\s\-]', '', t)
                t = re.sub(r'\s+', ' ', t).strip().title()
                # 1–3 words, reasonable length, dedupe
                if t and 3 <= len(t) <= 32 and 1 <= len(t.split()) <= 3 and t not in cleaned:
                    cleaned.append(t)
            if len(cleaned) >= 3:
                print(f"✅ Extracted tags (json): {cleaned[:5]}")
                return cleaned[:5]
    except json.JSONDecodeError:
        print("⚠️ Attempting line-based fallback...")

    # 1) Line-based parse: one-tag-per-line (with/without bullets)
    lines = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        # Strip leading bullets/numbers/quotes and trailing punctuation
        ln = re.sub(r'^\s*(?:[-•*]\s*|\d+\.\s*|["“”])', '', ln)
        ln = re.sub(r'["“”\.,;:!?#\[\]{}()]+$', '', ln).strip()
        # Keep only letters, spaces, and dashes
        ln = re.sub(r'[^A-Za-z\s\-]', '', ln)
        ln = re.sub(r'\s+', ' ', ln).strip().title()
        # Basic sanity: 1–3 words, length bounds, dedupe
        if ln and 3 <= len(ln) <= 32 and 1 <= len(ln.split()) <= 3 and ln not in cleaned:
            cleaned.append(ln)
    if len(cleaned) >= 3:
        print(f"✅ Extracted tags (line-based): {cleaned[:5]}")
        return cleaned[:5]

    print("⚠️ Attempting regex fallback...")
    # 2) Regex fallback: up to 3-word phrases starting with a letter
    matches = re.findall(r'[A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,2}', raw_output)
    cleaned = []
    for tag in matches:
        t = re.sub(r'\s+', ' ', tag).strip().title()
        if 3 <= len(t) <= 32 and t not in cleaned:
            cleaned.append(t)
    if len(cleaned) >= 3:
        print(f"✅ Extracted tags (regex): {cleaned[:5]}")
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
    response_text = llm_complete(prompt.strip())
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