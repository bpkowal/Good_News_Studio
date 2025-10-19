import os
from pathlib import Path
import subprocess
import sys
import time
import json
import argparse
import asyncio
import re
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime

# --- Securely load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable not set. "
        "Please add it to your environment or .env file."
    )
# -------------------------------------------

# Instantiate async client
client = AsyncOpenAI(api_key=openai.api_key)

# === Ethics Parliament Synthesis Pipeline ===
SCRIPT_DIR = Path(__file__).parent

LABELS = {
    "Utilitarian":   "Utilitarian Response:",
    "Virtue":        "Virtue Ethics Response:",
    "Deontology":    "Deontological Response:",
    "Care":          "Care Ethics Response:",
    "Rawlsian":      "Rawlsian Ethics Response:",
    "Nozick":        "Nozckian (Liberty) Response:",
    "Libertarian":   "Nozckian (Liberty) Response:",
}

SCENARIO_BUILDER = "scenario_builder_general.py"

AGENTS = [
    ("Virtue",      "virtue_ethics_agent_p.py"),
    ("Care",        "care_ethics_agent_p.py"),
    ("Deontology",  "deontological_agent_p.py"),
    ("Utilitarian", "utilitarian_agent_p.py"),
    ("Rawlsian",    "rawlsian_ethics_agent_p.py"),
]

# Mutable list to allow conditional insertion of Nozck/Nozick agent
AGENT_LIST = list(AGENTS)

parser = argparse.ArgumentParser()
parser.add_argument("--debug-agent", default=os.getenv("DEBUG_AGENT", "").strip(),
                    help="Agent name to echo raw stdout for (Virtue, Care, Deontology, Utilitarian, Rawlsian). Case‐insensitive.")
parser.add_argument("--debug-stdout-to-file", action="store_true",
                    help="If set, also write raw stdout to agent_outputs/debug_raw_<agent>_<timestamp>.log")
parser.add_argument("--debug-sample", type=int, default=1200,
                    help="When echoing, print a trimmed preview of the cleaned extraction.")
args = parser.parse_args()

SYNTHESIS_RATINGS_SCRIPT = SCRIPT_DIR / "synthesis_ratings_only.py"
SYNTHESIS_SCRIPT         = SCRIPT_DIR / "synthesis_final_judgment.py"
PROFILE_PATH            = SCRIPT_DIR / "user_ethics_profile.json"

def sanitize_scenario(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text or '').strip()

def compute_steering_from_mfq(
    mfq: dict,
    include_liberty: bool = False,
    libertarian_boost: bool = False
) -> dict:
    try:
        care      = float(mfq.get("Care/Harm",            mfq.get("care_harm", 0.0)))
        fairness  = float(mfq.get("Fairness/Cheating",    mfq.get("fairness_cheating", 0.0)))
        loyalty   = float(mfq.get("Loyalty/Betrayal",     mfq.get("loyalty_betrayal", 0.0)))
        authority = float(mfq.get("Authority/Subversion", mfq.get("authority_subversion", 0.0)))
        purity    = float(mfq.get("Sanctity/Degradation", mfq.get("purity_degradation", 0.0)))
        liberty   = float(mfq.get("Liberty/Oppression",   mfq.get("liberty_oppression", 0.0))) if include_liberty else 0.0
    except Exception:
        care = fairness = loyalty = authority = purity = liberty = 0.0

    w_util  = max(0.0, 0.4 * care + 0.4 * fairness)
    w_deon  = 0.4 * fairness + 0.3 * loyalty + 0.7 * authority + (0.3 * liberty if include_liberty else 0.0)
    w_virt  = 0.75 * loyalty + 1.1 * purity + 0.2 * authority
    w_care  = 1.0 * care
    w_rawls = 0.6 * fairness + 0.3 * authority
    w_nozk  = 1.0 * liberty if include_liberty else 0.0

    profile_label = str(mfq.get("profile_label", "")).strip().lower()

    if profile_label == "conservatives (us)":
        w_deon  *= 1.5
        w_util  *= 0.6
        w_care  *= 0.9
        w_virt  *= 0.9
        w_rawls *= 0.9

    weights = {
        "Utilitarian":   w_util,
        "Deontological": w_deon,
        "Virtue":        w_virt,
        "Care":          w_care,
        "Rawlsian":      w_rawls,
    }
    if include_liberty:
        weights["Nozick"] = w_nozk

    if libertarian_boost and include_liberty:
        weights["Nozick"] *= 1.6
        for k in ("Utilitarian","Rawlsian","Care","Virtue","Deontological"):
            weights[k] *= 0.9

    total = sum(weights.values()) or 1.0
    for k in list(weights.keys()):
        weights[k] = round(weights[k] / total, 3)
    return weights

# 1. Locate latest scenario file
SCENARIO_DIR = Path("scenarios")
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
scenario_files = sorted(SCENARIO_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
SCENARIO_PATH = str(scenario_files[0]) if scenario_files else ""

if not SCENARIO_PATH:
    print(f"\n🛠 Running Scenario Builder...\n{'='*40}")
    try:
        sb_result = subprocess.run(
            ["python", SCENARIO_BUILDER],
            capture_output=True,
            text=True,
            check=True,
        )
        print(sb_result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Scenario Builder Error:\n{e.stderr}")

    scenario_files = sorted(SCENARIO_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    SCENARIO_PATH = str(scenario_files[0]) if scenario_files else ""

if not SCENARIO_PATH:
    print("❌ No scenario file found in 'scenarios/' directory.")
    sys.exit(1)

with open(SCENARIO_PATH, "r", encoding="utf-8") as f:
    scenario_data = json.load(f)

results = {
    "ethical_question": scenario_data.get("ethical_question", ""),
    "agent_responses":   {}
}
results["ethical_question"] = sanitize_scenario(results["ethical_question"])

# 2. Prepare user profile & steering
with open(PROFILE_PATH, "r", encoding="utf-8") as pf:
    user_ethics_profile = json.load(pf)

def _normalize_mfq_scales(p: dict) -> dict:
    vals = [v for v in p.values() if isinstance(v,(int,float))]
    if vals and max(vals) <= 5.0 and min(vals) >= 0.0:
        p = {k:(float(v)+1.0 if isinstance(v,(int,float)) else v) for k,v in p.items()}
    q = {}
    for k,v in p.items():
        if isinstance(v,(int,float)):
            q[k] = max(1.0, min(6.0, float(v)))
        else:
            q[k] = v
    return q

user_ethics_profile = _normalize_mfq_scales(user_ethics_profile)

# Extract profile_label once
raw_label = user_ethics_profile.get("profile_label", "")
profile_label = str(raw_label).strip().lower()
print(f"[liberty] Retrieved profile_label raw={raw_label!r}, normalized={profile_label!r}")

# Determine libertarian flag and adjust profile if needed
libertarian_selected = False
if profile_label == "libertarians (us)":
    libertarian_selected = True
    user_ethics_profile["Liberty/Oppression"] = 4.0
    print(f"[liberty] Detected libertarian profile_label={profile_label}. Setting libertarian_selected=True")
else:
    print(f"[liberty] profile_label={profile_label}. libertarian_selected=False")
    user_ethics_profile.pop("Liberty/Oppression", None)
    user_ethics_profile.pop("liberty_oppression", None)

# Compute steering weights
steering_weights = compute_steering_from_mfq(
    user_ethics_profile,
    include_liberty=libertarian_selected,
    libertarian_boost=libertarian_selected
)
print(f"[steering] weights={steering_weights} (Nozick={steering_weights.get('Nozck') or steering_weights.get('Nozick')})")

# Insert Nozick agent if libertarian
nozick_candidates = ["nozick_ethics_agent_p.py", "Nozick_ethics_agent_p.py"]
nozick_script = next((p for p in nozick_candidates if Path(p).exists()), nozick_candidates[0])
exists = Path(nozick_script).exists()
print(f"[liberty] Candidate Nozick script: {nozick_script} (exists={exists})")

if libertarian_selected:
    if exists:
        if ("Nozick", nozick_script) not in AGENT_LIST:
            AGENT_LIST.insert(0, ("Nozick", nozick_script))
            print("🧩 Included Nozick (Liberty) agent due to libertarian profile.")
        else:
            print("[liberty] Nozick agent already present.")
    else:
        print(f"⚠️ Nozick script not found: {nozick_script}. Agent not included.")
print(f"[pipeline] Final AGENT_LIST order: {[name for name,_ in AGENT_LIST]}")

# 3. Run Each Agent
for name, script in AGENT_LIST:
    print(f"\n🧠 Running {name} Agent...\n{'='*40}")
    try:
        result = subprocess.run(
            ["python", script, "--scenario", SCENARIO_PATH],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = result.stdout

        debug_this = bool(args.debug_agent) and (name.lower() == args.debug_agent.lower())
        if debug_this:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\n===== DEBUG RAW STDOUT ({name}) BEGIN =====")
            print(raw if raw else "[[EMPTY STDOUT]]")
            print(f"===== DEBUG RAW STDOUT ({name}) END =====\n")
            if args.debug_stdout_to_file:
                debug_dir = SCRIPT_DIR / "agent_outputs"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"debug_raw_{name.lower()}_{ts}.log"
                with open(debug_path, "w", encoding="utf-8") as df:
                    df.write(raw)
                print(f"📝 Saved raw stdout to: {debug_path}")

        def _strip_control(s: str) -> str:
            if not isinstance(s, str):
                return ""
            s = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", s)
            s = re.sub(r"[\ufeff\u200b-\u200f\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", s)
            s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
            s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201C", '"').replace("\u201D", '"').replace("\uFF1A", ":")
            s = re.sub(r"[ \t]+", " ", s)
            return s

        label = LABELS.get(name, f"{name} Response:")
        cleaned_raw = _strip_control(raw)

        base = re.escape(label.rstrip(':'))
        alt_base = re.escape(label.replace(" Response:", "").rstrip(':'))
        patterns = [
            rf'^\s*{base}:?\s*(.*)$',
            rf'^\s*\*\*{base}\*\*:?\s*(.*)$',
            rf'^\s*#+\s*{base}:?\s*(.*)$',
            rf'^\s*{alt_base}:?\s*(.*)$',
            rf'^\s*\*\*{alt_base}\*\*:?\s*(.*)$',
            rf'^\s*#+\s*{alt_base}:?\s*(.*)$',
        ]

        m = None
        for pat in patterns:
            m = re.search(pat, cleaned_raw, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if m:
                break

        if debug_this:
            print("----- DEBUG EXTRACTION INFO -----")
            print(f"Label expected: {label!r}")
            print(f"Patterns tried: {patterns}")
            print(f"Match found   : {bool(m)}")

        def _trim_trailing_noise(text: str) -> str:
            noise_starts = (
                "DEBUG", "Loaded ", "Scenario Tags:", "Expanded tag weights:", "Scenario Tag Weights:",
                "Retrieved ", "Normalized Tags:", "Zipf-", "Quote Used", "Trying primary model:",
                "Model returned", "🧠", "⚠️", "✅", "💾", "🔍", "🧪", "🔧", "📘", "🔎", "📝", "ℹ️",
                "Model used:", "(Model used:", "Top Quotes Used:", "Scenario ID:", "Ethical Question:", "stdout:",
                "stderr:", "Traceback", "File \""
            )
            lines = text.splitlines()
            kept = []
            for line in lines:
                stripped = line.strip()
                if any(stripped.startswith(s) for s in noise_starts):
                    break
                kept.append(line)
            return "\n".join(kept).strip()

        if m:
            candidate = m.group(1).strip()
            candidate = re.split(r"\n(?=\s*(?:#{1,6}\s|\*\*[^\n]+\*\*\s*:|[A-Z][A-z ]+:\s*$|```))", candidate, maxsplit=1)[0]
        else:
            candidate = cleaned_raw.strip()
            candidate_lines = candidate.splitlines()
            if candidate_lines and re.match(r"^\s*(?:#{1,6}\s|\*\*.*\*\*\s*:|[A-Z].*?:\s*$)", candidate_lines[0]):
                candidate = "\n".join(candidate_lines[1:]).strip()
            sys.stderr.write(
                f"[collector] Warning: could not find exact label '{label}' in {name} stdout; using full stdout fallback.\n"
            )

        clean = _trim_trailing_noise(candidate)
        results["agent_responses"][label] = clean

    except subprocess.CalledProcessError as e:
        print(f"❌ {name} Agent Error:\n{e.stderr}")
    time.sleep(1)

# Save consolidated agent responses
output_file = SCRIPT_DIR / "latest_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Wrote consolidated results to {output_file}")

# 4. Run Ratings Script
ratings = {}
for name, label in LABELS.items():
    response = results["agent_responses"].get(label, "")
    if not response:
        continue
    try:
        rate_proc = subprocess.run(
            ["python", str(SYNTHESIS_RATINGS_SCRIPT), "--agent-label", label],
            capture_output=True,
            text=True,
            check=True,
        )
        score_text = rate_proc.stdout.strip()
        try:
            ratings[label] = json.loads(score_text)
        except json.JSONDecodeError:
            ratings[label] = {"raw_rating": score_text}
    except subprocess.CalledProcessError as e:
        print(f"❌ Rating Error for {label}:\n{e.stderr}")

ratings_output = {
    "ethical_question": results["ethical_question"],
    "agent_ratings":   ratings
}
ratings_file = SCRIPT_DIR / "latest_ratings.json"
with open(ratings_file, "w", encoding="utf-8") as f:
    json.dump(ratings_output, f, indent=2)
print(f"\n✅ Consolidated ratings:\n{json.dumps(ratings_output, indent=2)}")

# 6. Rebuttal Agents (Utilitarian, Virtue, Deontological, Care)
print(f"\n⚖️ Running Utilitarian Rebuttal Agent...\n{'='*40}")
rebuttal = ""
try:
    rebuttal_proc = subprocess.run(["python", "util_rebuttal_agent.py"],
                                   capture_output=True, text=True, check=True)
    rebuttal = rebuttal_proc.stdout.strip()
except subprocess.CalledProcessError as e:
    print(f"❌ Rebuttal Agent Error:\n{e.stderr}")

rebuttal_output_file = Path("agent_outputs")
rebuttal_files = sorted(rebuttal_output_file.glob("util_rebuttal_*.txt"),
                       key=os.path.getmtime, reverse=True)
if rebuttal_files:
    with open(rebuttal_files[0], "r", encoding="utf-8") as f:
        rebuttal_text = f.read().strip()
else:
    rebuttal_text = "[ERROR] No rebuttal output file found."

rebuttal_json = {
    "ethical_question": results["ethical_question"],
    "utilitarian_rebuttal": rebuttal_text
}
rebuttal_file = SCRIPT_DIR / "latest_rebuttal.json"
with open(rebuttal_file, "w", encoding="utf-8") as f:
    json.dump(rebuttal_json, f, indent=2)
print(f"\n✅ Saved parsed rebuttal to {rebuttal_file}")

print(f"\n⚖️ Running Virtue Ethics Rebuttal Agent...\n{'='*40}")
virtue_rebuttal = ""
try:
    virtue_proc = subprocess.run(["python", "virtue_rebuttal_agent.py"],
                                 capture_output=True, text=True, check=True)
    virtue_rebuttal = virtue_proc.stdout.strip()
except subprocess.CalledProcessError as e:
    print(f"❌ Virtue Rebuttal Agent Error:\n{e.stderr}")

virtue_files = sorted(rebuttal_output_file.glob("virtue_rebuttal_*.txt"),
                      key=os.path.getmtime, reverse=True)
if virtue_files:
    with open(virtue_files[0], "r", encoding="utf-8") as f:
        virtue_text = f.read().strip()
else:
    virtue_text = "[ERROR] No virtue rebuttal output file found.]"

rebuttal_json["virtue_rebuttal"] = virtue_text
with open(rebuttal_file, "w", encoding="utf-8") as f:
    json.dump(rebuttal_json, f, indent=2)
print(f"\n✅ Appended virtue ethics rebuttal to {rebuttal_file}")

print(f"\n⚖️ Running Deontological Rebuttal Agent...\n{'='*40}")
deon_rebuttal = ""
try:
    deon_proc = subprocess.run(["python", "deontology_rebuttal_agent.py"],
                                capture_output=True, text=True, check=True)
    deon_rebuttal = deon_proc.stdout.strip()
except subprocess.CalledProcessError as e:
    print(f"❌ Deontological Rebuttal Agent Error:\n{e.stderr}")

deon_files = sorted(rebuttal_output_file.glob("deon_rebuttal_*.txt"),
                    key=os.path.getmtime, reverse=True)
if deon_files:
    with open(deon_files[0], "r", encoding="utf-8") as f:
        deon_text = f.read().strip()
else:
    deon_text = "[ERROR] No deon rebuttal output file found.]"

rebuttal_json["deontological_rebuttal"] = deon_text
with open(rebuttal_file, "w", encoding="utf-8") as f:
    json.dump(rebuttal_json, f, indent=2)
print(f"\n✅ Appended deontological rebuttal to {rebuttal_file}")

print(f"\n⚖️ Running Care Ethics Rebuttal Agent...\n{'='*40}")
care_rebuttal = ""
try:
    care_proc = subprocess.run(["python", "care_rebuttal_agent.py"],
                                capture_output=True, text=True, check=True)
    care_rebuttal = care_proc.stdout.strip()
except subprocess.CalledProcessError as e:
    print(f"❌ Care Rebuttal Agent Error:\n{e.stderr}")

care_files = sorted(rebuttal_output_file.glob("care_rebuttal_*.txt"),
                    key=os.path.getmtime, reverse=True)
if care_files:
    with open(care_files[0], "r", encoding="utf-8") as f:
        care_text = f.read().strip()
else:
    care_text = "[ERROR] No care rebuttal output file found.]"

rebuttal_json["care_rebuttal"] = care_text
with open(rebuttal_file, "w", encoding="utf-8") as f:
    json.dump(rebuttal_json, f, indent=2)
print(f"\n✅ Appended care ethics rebuttal to {rebuttal_file}")

async def call_o3(messages, model="o3", tool_choice=None, timeout=45):
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tool_choice=tool_choice,
        max_completion_tokens=20048,
        timeout=timeout,
    )
    usage = response.usage
    return response.choices[0].message, usage

# Final synthesis
with open(PROFILE_PATH, "r", encoding="utf-8") as pf:
    # (Re-load if needed; ensure up to date)
    user_ethics_profile = json.load(pf)

async def run_final_synthesis():
    try:
        messages = []
        messages.append({"role": "system", "content": MASTER_PROMPT})
        if libertarian_selected:
            messages.append({"role": "system", "content": 
                             "Profile inference: The MFQ pattern indicates a Libertarian emphasis …"})

        messages.append({"role": "user", "content": f"### Scenario …\n{results['ethical_question']}…"})
        messages.append({"role": "user", "content": f"### User Ethics Profile …\n```json\n{json.dumps(user_ethics_profile, indent=2)}\n```"})
        messages.append({"role": "user", "content": steering_line})
        messages.append({"role": "user", "content": f"### Agent Responses\n```json\n{json.dumps(results['agent_responses'], indent=2)}\n```"})
        messages.append({"role": "user", "content": f"### Agent Ratings\n```json\n{json.dumps(ratings_output['agent_ratings'], indent=2)}\n```"})
        messages.append({"role": "user", "content": f"### Rebuttals\n```json\n{json.dumps(rebuttal_json, indent=2)}\n```"})

        o3_message, o3_usage = await call_o3(messages)
        print("\n🧠 Final Synthesis\n" + "="*40)
        print(o3_message.content)

        synthesis_path = SCRIPT_DIR / "latest_synthesis.txt"
        with open(synthesis_path, "w", encoding="utf-8") as f:
            f.write(o3_message.content)
        print(f"\n✅ Saved final synthesis to {synthesis_path}")
        print(f"📝 o3 usage — prompt: {o3_usage.prompt_tokens}, completion: {o3_usage.completion_tokens}, total: {o3_usage.total_tokens}")
    except Exception as err:
        print(f"❌ o3 Synthesis Error: {err}")

if __name__ == "__main__":
    asyncio.run(run_final_synthesis())