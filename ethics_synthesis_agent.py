import os
from pathlib import Path
import subprocess
import time
import json
import argparse

from pathlib import Path

#gives path information used later for locating ratings script
SCRIPT_DIR = Path(__file__).parent

# Mapping agent names to their response labels
LABELS = {
    "Utilitarian": "Utilitarian Response:",
    "Virtue": "Virtue Ethics Response:",
    "Deontology": "Deontological Response:",
    "Care": "Care Ethics Response:"
}

# How the semantic tags for the user question are generated
SCENARIO_BUILDER = "scenario_builder_general.py"

#Scripts for each agent
AGENTS = [
    ("Virtue", "virtue_ethics_agent_p.py"),
    ("Care", "care_ethics_agent_p.py"),
    ("Deontology", "deontological_agent_p.py"),
    ("Utilitarian", "utilitarian_agent_p.py"),
]

# Path to the ratings script
SYNTHESIS_RATINGS_SCRIPT = SCRIPT_DIR / "synthesis_ratings_only.py"

# Path to the deontology critic script
DEONTOLOGY_CRITIC_SCRIPT = SCRIPT_DIR / "deontology_critic_p.py"


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

# Recompute scenario file after builder run
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
        # Extract only the part starting from the label
        idx = raw.find(label)
        clean = raw[idx:].strip() if idx != -1 else raw.strip()
        results["agent_responses"][label] = clean
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} Agent Error:\n{e.stderr}")

    time.sleep(1)  # small pause between agents

output_file = SCRIPT_DIR / "latest_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Wrote consolidated results to {output_file}")

# Run synthesis rating script for each agent response
import argparse
ratings = {}
for label, response in results["agent_responses"].items():
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

# Write consolidated ratings to JSON
ratings_output = {
    "ethical_question": results["ethical_question"],
    "agent_ratings": ratings
}
ratings_file = SCRIPT_DIR / "latest_ratings.json"
with open(ratings_file, "w", encoding="utf-8") as f:
    json.dump(ratings_output, f, indent=2)
print(f"\n✅ Consolidated ratings:\n{json.dumps(ratings_output, indent=2)}")

# Run deontology critic for Deontological Response only
custom_ratings = {}
for label, response in results["agent_responses"].items():
    if label != LABELS["Deontology"]:
        continue
    try:
        proc = subprocess.run(
            ["python", str(DEONTOLOGY_CRITIC_SCRIPT),
             "--question", results["ethical_question"],
             "--response", response],
            capture_output=True,
            text=True,
            check=True,
        )
        custom_text = proc.stdout.strip()
        custom_ratings[label] = custom_text
    except subprocess.CalledProcessError as e:
        print(f"❌ Deontology Critic Error for {label}:\n{e.stderr}")

# Write consolidated deontology critic ratings to JSON
custom_output = {
    "ethical_question": results["ethical_question"],
    "deontology_critic": custom_ratings
}
custom_file = SCRIPT_DIR / "latest_custom_ratings.json"
with open(custom_file, "w", encoding="utf-8") as f:
    json.dump(custom_output, f, indent=2)
print(f"\n✅ Consolidated deontology critic results:\n{json.dumps(custom_output, indent=2)}")

# Run final synthesis script
print(f"\n🧠 Synthesizing Final Judgment...\n{'='*40}")
synthesis_script = SCRIPT_DIR / "synthesis_final_judgment.py"
try:
    final_result = subprocess.run(
        ["python", str(synthesis_script), "--scenario", SCENARIO_PATH],
        capture_output=True,
        text=True,
        check=True,
    )
    print(final_result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Final Synthesis Error:\n{e.stderr}")