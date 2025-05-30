import os
from pathlib import Path
import subprocess
import time

SCENARIO_BUILDER = "scenario_builder_general.py"

AGENTS = [
    ("Virtue", "virtue_ethics_agent_p.py"),
    ("Care", "care_ethics_agent_p.py"),
    ("Deontology", "deontological_agent_p.py"),
    ("Utilitarian", "utilitarian_agent_p.py"),
]

SCENARIO_DIR = Path("scenarios")
scenario_files = sorted(SCENARIO_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
SCENARIO_PATH = str(scenario_files[0]) if scenario_files else ""

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

if not SCENARIO_PATH:
    print("❌ No scenario file found in 'scenarios/' directory.")
    exit(1)

for name, script in AGENTS:
    print(f"\n🧠 Running {name} Agent...\n{'='*40}")
    try:
        result = subprocess.run(
            ["python", script, "--scenario", SCENARIO_PATH],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} Agent Error:\n{e.stderr}")

    time.sleep(1)  # small pause between agents


# Run synthesis script for ratings only
print(f"\n📊 Evaluating Agent Responses (Ratings Only)...\n{'='*40}")
RATINGS_SCRIPT = "evaluate_responses.py"

try:
    ratings_result = subprocess.run(
        ["python", RATINGS_SCRIPT, "--scenario", SCENARIO_PATH],
        capture_output=True,
        text=True,
        check=True,
    )
    print(ratings_result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Ratings Synthesis Error:\n{e.stderr}")

# Run final synthesis script with summary judgment and decision
print(f"\n🧠 Synthesizing Final Judgment...\n{'='*40}")
FINAL_SYNTHESIS_SCRIPT = "final_synthesis.py"

try:
    final_result = subprocess.run(
        ["python", FINAL_SYNTHESIS_SCRIPT, "--scenario", SCENARIO_PATH],
        capture_output=True,
        text=True,
        check=True,
    )
    print(final_result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Final Synthesis Error:\n{e.stderr}")
 