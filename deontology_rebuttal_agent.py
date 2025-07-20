from llama_cpp import Llama
from pathlib import Path
from datetime import datetime
import json
import gc
import os
import time


MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
RESULTS_FILE = Path("latest_results.json")
OUTPUT_DIR = Path("agent_outputs")

LAST_REBUT_QUERY_PATH = Path("agent_outputs/.last_deon_rebut_query.txt")
LAST_REBUT_RESPONSE_PATH = Path("agent_outputs/.last_deon_rebut_response.txt")


def rebut_virtue_response():
    # Load scenario
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    query = data["ethical_question"]
    scenario_id = "temp_scenario"  # can also hash query if needed
    deon_response = data["agent_responses"].get("Deontological Response:", "").strip()
    virtue_response = data["agent_responses"].get("Virtue Ethics Response:", "").strip()

    # Check for cached result
    if LAST_REBUT_QUERY_PATH.exists() and LAST_REBUT_RESPONSE_PATH.exists():
        last_input = LAST_REBUT_QUERY_PATH.read_text().strip()
        current_input = f"{query.strip()}\n{deon_response.strip()}\n{virtue_response.strip()}"
        if current_input == last_input:
            print("⚡ Skipping LLM call — using cached deontological rebuttal.")
            return LAST_REBUT_RESPONSE_PATH.read_text().strip()

    if not deon_response:
        raise ValueError("❌ No deontological response found in latest_results.json.")

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
<s>[INST] You are a deontological rebuttal agent. Your goal is to critique a virtue ethics response to a specific moral question by comparing it with a deontological answer. Use Kantian principles and the duty to respect persons as your guiding framework.

Focus your critique on:
- Whether the virtue-based reasoning leads to actions that can be universalized
- If it risks treating persons as means rather than ends
- How a deontologist would evaluate the same situation based on duties, not character

Avoid summarizing both theories. Stay grounded in the scenario and emphasize principled moral reasoning.

Ethical Question:
{query}

Virtue Ethics Response:
\"\"\"{virtue_response}\"\"\"

Original Deontological Response:
\"\"\"{deon_response}\"\"\"

Deontological Rebuttal:
[/INST]
"""

    print("🧠 Running deontological rebuttal LLM...")
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
    outpath = OUTPUT_DIR / f"deon_rebuttal_{timestamp}.txt"

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("=== Deontological Rebuttal Log ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        f.write("\n>>> Ethical Question:\n" + query + "\n")
        f.write("\n>>> Deontological Response:\n" + deon_response + "\n")
        f.write("\n>>> Original Virtue Ethics Response:\n" + virtue_response + "\n")
        f.write("\n>>> Deontological Rebuttal:\n" + rebuttal + "\n")

    # Cache the inputs and output
    LAST_REBUT_QUERY_PATH.write_text(f"{query.strip()}\n{deon_response.strip()}\n{virtue_response.strip()}")
    LAST_REBUT_RESPONSE_PATH.write_text(rebuttal.strip())

    print(f"✅ Rebuttal saved to: {outpath.name}")

    return rebuttal


if __name__ == "__main__":
    rebut_virtue_response()