import re
import time
import json
from pathlib import Path
from datetime import datetime
import argparse
import os
import requests

# === Setup ===
PROVIDER = os.getenv("PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Optional fallback API (if you host a semantic service)
SEMANTIC_API_URL = os.getenv("SEMANTIC_API_URL")  # e.g. "https://semantic-builder.yourdomain.com"

def llm_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
    """Unified completion wrapper returning the assistant text string."""
    if PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY for provider 'openai'.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    else:
        raise ValueError(f"Unknown PROVIDER '{PROVIDER}'. Use 'openai'.")

SCENARIO_DIR = Path("scenarios")
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

def extract_tags(raw_output: str) -> list[str]:
    """Try to robustly parse tags from LLM output (json, lines, regex)."""
    # JSON parse attempt
    try:
        tags = json.loads(raw_output)
        if isinstance(tags, list):
            cleaned = []
            for t in tags:
                if isinstance(t, str):
                    t2 = re.sub(r'[^A-Za-z\s\-]', "", t).strip().title()
                    if 1 <= len(t2.split()) <= 3 and t2 not in cleaned:
                        cleaned.append(t2)
            if len(cleaned) >= 3:
                return cleaned[:5]
    except Exception:
        pass

    # Line-by-line fallback
    lines = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln2 = re.sub(r'^[\-\•\d\.\s]+', "", ln)  # strip numbers/bullets
        ln2 = re.sub(r'[^A-Za-z\s\-]', "", ln2).strip().title()
        if 1 <= len(ln2.split()) <= 3 and ln2 not in cleaned:
            cleaned.append(ln2)
    if len(cleaned) >= 3:
        return cleaned[:5]

    # Regex fallback
    matches = re.findall(r'[A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,2}', raw_output)
    cleaned = []
    for m in matches:
        m2 = re.sub(r'\s+', " ", m).title()
        if m2 not in cleaned and 1 <= len(m2.split()) <= 3:
            cleaned.append(m2)
    return cleaned[:5]

def describe_tag(tag: str) -> str:
    prompt = (
        f"You are an ethics assistant. Define the ethical tag \"{tag}\" in 8 words or fewer, "
        "- Use neutral ethical language. "
        "- Do not reference any specific scenario. "
        "- Avoid terms like 'utility', 'deontology', or 'care'.\n\nDefinition:"
    )
    # small delay to throttle
    time.sleep(0.2)
    return llm_complete(prompt, max_tokens=32, temperature=0.2)

def build_scenario(scenario_text: str, id_override: str | None = None) -> dict:
    now = datetime.now().strftime("%H%M_%Y%m%d")
    scenario_id = id_override or f"auto_ethics_{now}"
    scenario_type = now[-4:]

    print("📨 Submitted. Processing scenario...")

    # Generate tags
    tag_prompt = (
        "You are an ethics assistant.\n"
        "Return exactly 5 short ethical tags relevant to the scenario. "
        "Tags should be one or two words, not reference any specific school. "
        "List one tag per line.\n\n"
        f"Scenario: {scenario_text}\n\nTags:"
    )
    raw = llm_complete(tag_prompt, max_tokens=64, temperature=0.2)
    print("🧾 Raw tags output:\n", raw)

    try:
        tags = extract_tags(raw)
    except Exception as e:
        print("❌ Tag extraction failed:", e)
        tags = []

    # If you rely on vectorstore for semantic expansion (e.g. related documents), try fallback API
    related = []
    if SEMANTIC_API_URL:
        try:
            resp = requests.post(
                SEMANTIC_API_URL,
                json={"scenario": scenario_text, "tags": tags},
                timeout=10,
            )
            resp.raise_for_status()
            obj = resp.json()
            related = obj.get("related", [])  # whatever structure you choose
        except Exception as e:
            print("⚠️ Semantic API fallback failed:", e)

    tag_expectations = {tag: round(2.0 - 0.2 * i, 1) for i, tag in enumerate(tags)}
    tag_descriptions = {}
    for t in tags:
        tag_descriptions[t] = describe_tag(t)

    scenario_data = {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "ethical_question": scenario_text.strip(),
        "tags": tags,
        "tag_expectations": tag_expectations,
        "tag_descriptions": tag_descriptions,
        "related": related,
    }

    out = SCENARIO_DIR / f"{scenario_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scenario_data, f, indent=2)
    print(f"💾 Scenario saved to {out.name}")

    return scenario_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, help="scenario text")
    parser.add_argument("--scenario-file", type=str, help="file path to scenario")
    parser.add_argument("--id", type=str, default=None)
    args = parser.parse_args()

    scenario_text = None
    if args.scenario:
        scenario_text = args.scenario.strip()
    elif args.scenario_file:
        p = Path(args.scenario_file)
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if p.suffix.lower() == ".json":
                try:
                    obj = json.loads(txt)
                    scenario_text = obj.get("ethical_question") or txt
                except:
                    scenario_text = txt
            else:
                scenario_text = txt

    if scenario_text is None:
        print("Please supply a scenario via --scenario or --scenario-file")
        exit(1)

    build_scenario(scenario_text, id_override=args.id)