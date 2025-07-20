from llama_cpp import Llama
from pathlib import Path
from datetime import datetime
import json
import gc
import os
import time

LAST_REBUT_QUERY_PATH = Path("agent_outputs/.last_virtue_rebut_query.txt")
LAST_REBUT_RESPONSE_PATH = Path("agent_outputs/.last_virtue_rebut_response.txt")


MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
RESULTS_FILE = Path("latest_results.json")
OUTPUT_DIR = Path("agent_outputs")


def rebut_utilitarian_response():
    # Load scenario
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    query = data["ethical_question"]
    scenario_id = "temp_scenario"
    util_response = data["agent_responses"].get("Utilitarian Response:", "").strip()

    if not util_response:
        raise ValueError("❌ No utilitarian response found in latest_results.json.")

    # Skip if the query and response haven't changed
    if LAST_REBUT_QUERY_PATH.exists() and LAST_REBUT_RESPONSE_PATH.exists():
        last_input = LAST_REBUT_QUERY_PATH.read_text().strip()
        current_input = f"{query.strip()}\n{util_response.strip()}"
        if current_input == last_input:
            print("⚡ Skipping LLM call — using cached virtue ethics rebuttal.")
            return LAST_REBUT_RESPONSE_PATH.read_text().strip()

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

    # Load LLM
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=3000,
        n_threads=6,
        n_gpu_layers=60,
        n_batch=64,
        verbose=False
    )

    prompt = f"""
<s>[INST] You are a virtue ethics rebuttal agent. Your task is to critique a utilitarian response to a specific ethical question by evaluating how it aligns with or contradicts the cultivation of moral character and human flourishing.

Focus your analysis on:
- Whether the utilitarian response promotes virtuous dispositions (e.g., courage, honesty, temperance, compassion)
- Whether it supports practical wisdom (phronesis) in this specific context
- Whether the action helps individuals and communities flourish over time

Avoid general summaries. Stay grounded in the ethical scenario and the specific utilitarian reasoning.

Ethical Question:
{query}

Utilitarian Response:
\"\"\"{util_response}\"\"\"

Virtue Ethics Rebuttal:
[/INST]
"""

    print("🧠 Running virtue ethics rebuttal LLM...")
    completion = llm(prompt, max_tokens=500, temperature=0.5, stream=False)

    if isinstance(completion, str):
        rebuttal = completion.strip()
    elif isinstance(completion, dict) and "choices" in completion:
        rebuttal = "".join(choice["text"] for choice in completion["choices"]).strip()
    else:
        rebuttal = "[ERROR] Unexpected LLM output."

    # Clean up
    del llm
    gc.collect()
    time.sleep(1)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = OUTPUT_DIR / f"virtue_rebuttal_{timestamp}.txt"

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("=== Virtue Ethics Rebuttal Log ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        f.write("\n>>> Ethical Question:\n" + query + "\n")
        f.write("\n>>> Utilitarian Response:\n" + util_response + "\n")
        f.write("\n>>> Virtue Ethics Rebuttal:\n" + rebuttal + "\n")

    print(f"✅ Rebuttal saved to: {outpath.name}")

    # Cache the inputs and rebuttal
    LAST_REBUT_QUERY_PATH.write_text(f"{query.strip()}\n{util_response.strip()}")
    LAST_REBUT_RESPONSE_PATH.write_text(rebuttal.strip())

    return rebuttal


if __name__ == "__main__":
    rebut_utilitarian_response()