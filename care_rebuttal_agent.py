from pathlib import Path
from datetime import datetime
import json
import os

from openai import OpenAI


RESULTS_FILE = Path("latest_results.json")
OUTPUT_DIR = Path("agent_outputs")


def rebut_util_response():
    # Load scenario
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    query = data["ethical_question"]
    scenario_id = "temp_scenario"  # can also hash query if needed
    care_response = data["agent_responses"].get("Care Ethics Response:", "").strip()
    util_response = data["agent_responses"].get("Utilitarian Response:", "").strip()

    if not util_response:
        raise ValueError("❌ No utilitarian response found in latest_results.json.")

    # Create temporary scenario file
    temp_scenario_path = Path(f"scenarios/{scenario_id}.json")
    os.makedirs(temp_scenario_path.parent, exist_ok=True)

    temp_data = {
        "ethical_question": query,
        "temporal_tags": data.get("temporal_tags", []),
        "outcome_values": data.get("outcome_values", []),
        "outcome_distances": data.get("outcome_distances", []),
    }

    with open(temp_scenario_path, "w") as f:
        json.dump(temp_data, f, indent=2)

    client = OpenAI()

    prompt = f"""
<s>[INST] You are a care ethics rebuttal agent. Your goal is to critique a utilitarian response to a specific moral question by comparing it with the ethical reasoning of care ethics. Use relational, emotional, and contextual insights as your guiding framework.

Focus your critique on:
- Whether the utilitarian response overlooks relationships or emotional impacts
- If it treats persons as abstract quantities rather than unique individuals
- How a care ethics approach might offer a more humane or relationally responsive alternative

Avoid summarizing both theories. Stay grounded in the scenario and emphasize attentiveness to care and context.

Ethical Question:
{query}

Utilitarian Response:
\"\"\"{util_response}\"\"\"

Care Ethics Rebuttal:
[/INST]
"""

    print("🧠 Running care ethics rebuttal LLM...")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.5,
        stream=False
    )

    rebuttal = completion.choices[0].message.content.strip()

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = OUTPUT_DIR / f"care_rebuttal_{timestamp}.txt"

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("=== Care Ethics Rebuttal Log ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        f.write("\n>>> Ethical Question:\n" + query + "\n")
        f.write("\n>>> Care Ethics Response:\n" + care_response + "\n")
        f.write("\n>>> Original Utilitarian Response:\n" + util_response + "\n")
        f.write("\n>>> Care Ethics Rebuttal:\n" + rebuttal + "\n")

    print(f"✅ Rebuttal saved to: {outpath.name}")

    return rebuttal


if __name__ == "__main__":
    rebut_util_response()