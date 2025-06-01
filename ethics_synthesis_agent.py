import os
from pathlib import Path
import subprocess
import time
import json

# === Ethics Parliament Synthesis Pipeline ===
# Runs scenario builder, collects agent responses, rates each, and synthesizes final judgment.

SCRIPT_DIR = Path(__file__).parent

# Mapping agent names to their response labels
LABELS = {
    "Utilitarian": "Utilitarian Response:",
    "Virtue": "Virtue Ethics Response:",
    "Deontology": "Deontological Response:",
    "Care": "Care Ethics Response:"
}

SCENARIO_BUILDER = "scenario_builder_general.py"

AGENTS = [
    ("Virtue", "virtue_ethics_agent_p.py"),
    ("Care", "care_ethics_agent_p.py"),
    ("Deontology", "deontological_agent_p.py"),
    ("Utilitarian", "utilitarian_agent_p.py"),
]

SYNTHESIS_RATINGS_SCRIPT = SCRIPT_DIR / "synthesis_ratings_only.py"
SYNTHESIS_SCRIPT = SCRIPT_DIR / "synthesis_final_judgment.py"

# === 1. Run Scenario Builder ===
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

# === 2. Find the latest scenario file ===
SCENARIO_DIR = Path("scenarios")
scenario_files = sorted(SCENARIO_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
SCENARIO_PATH = str(scenario_files[0]) if scenario_files else ""
if not SCENARIO_PATH:
    print("❌ No scenario file found in 'scenarios/' directory.")
    exit(1)

with open(SCENARIO_PATH, "r", encoding="utf-8") as f:
    scenario_data = json.load(f)

results = {
    "ethical_question": scenario_data.get("ethical_question", ""),
    "agent_responses": {}
}

# === 3. Run Each Agent ===
for name, script in AGENTS:
    print(f"\n🧠 Running {name} Agent...\n{'='*40}")
    try:
        result = subprocess.run(
            ["python", script, "--scenario", SCENARIO_PATH],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = result.stdout
        label = LABELS.get(name, f"{name} Response:")
        # Extract only the part after the label (if present)
        idx = raw.find(label)
        if idx != -1:
            clean = raw[idx + len(label):].strip()
        else:
            clean = raw.strip()
        results["agent_responses"][label] = clean
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} Agent Error:\n{e.stderr}")
    time.sleep(1)  # small pause between agents

# Save consolidated agent responses
output_file = SCRIPT_DIR / "latest_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Wrote consolidated results to {output_file}")

# === 4. Run Ratings Script for Each Agent ===
ratings = {}
for name, label in LABELS.items():
    response = results["agent_responses"].get(label, "")
    if not response:
        continue  # Skip if agent response missing
    try:
        rate_proc = subprocess.run(
            ["python", str(SYNTHESIS_RATINGS_SCRIPT), "--agent-label", label],
            capture_output=True,
            text=True,
            check=True,
        )
        score_text = rate_proc.stdout.strip()
        ratings[label] = score_text
    except subprocess.CalledProcessError as e:
        print(f"❌ Rating Error for {label}:\n{e.stderr}")

# Save ratings to JSON
ratings_output = {
    "ethical_question": results["ethical_question"],
    "agent_ratings": ratings
}
ratings_file = SCRIPT_DIR / "latest_ratings.json"
with open(ratings_file, "w", encoding="utf-8") as f:
    json.dump(ratings_output, f, indent=2)
print(f"\n✅ Consolidated ratings:\n{json.dumps(ratings_output, indent=2)}")

# === 5. Run Final Synthesis ===
print(f"\n🧠 Synthesizing Final Judgment...\n{'='*40}")
try:
    final_result = subprocess.run(
        ["python", str(SYNTHESIS_SCRIPT), "--scenario", SCENARIO_PATH],
        capture_output=True,
        text=True,
        check=True,
    )
    print(final_result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Final Synthesis Error:\n{e.stderr}")