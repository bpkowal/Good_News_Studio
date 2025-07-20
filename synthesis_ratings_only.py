import os
import re
import json
import argparse
from pathlib import Path
from llama_cpp import Llama
from datetime import datetime

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CONTEXT_SIZE = 4000
SCENARIO_DIR = "scenarios"
AGENT_OUTPUT_DIR = "agent_outputs"

# LLM setup
llama = Llama(
    model_path=MODEL_PATH,
    n_ctx=CONTEXT_SIZE,
    n_gpu_layers=60,
    n_threads=6,
    temperature=0.7
)

EXPECTED_LABELS = [
    "Virtue Ethics Response:",
    "Care Ethics Response:",
    "Deontological Response:",
    "Utilitarian Response:",
    "Rawlsian Ethics Response:"
]

def normalize_label(label):
    """Lowercase, strip, and remove trailing colon for consistent lookup."""
    return label.strip().lower().replace(":", "")

def load_latest_results():
    latest_results_file = SCRIPT_DIR / "latest_results.json"
    with open(latest_results_file, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(scenario_text, agent_responses, eval_labels):
    prompt = "<s>[INST]\n"
    prompt += f"Scenario Question:\n{scenario_text}\n\nAgent Responses:\n\n"
    for agent_label in eval_labels:
        if agent_label in agent_responses:
            label_clean = agent_label.rstrip(":")
            prompt += f"--- {label_clean} ---\n{agent_responses[agent_label].strip()}\n\n"
    prompt += (
        "END OF RESPONSES.\n\n"
        "For each agent response above, score ONLY the responses shown (in this order) on these four criteria using a numeric scale (1-5):\n"
        "1. Ethical clarity\n"
        "2. Internal consistency\n"
        "3. Relevance to scenario details\n"
        "4. Alignment with its ethical framework\n"
        "For each, assign a score (1-5). Then provide a TOTAL score as the sum of all four ratings.\n\n"
        "Format each section as:\n"
        "--- [Agent Label] ---\n"
        "Ethical Clarity: [SCORE]\n"
        "Internal Consistency: [SCORE]\n"
        "Relevance to Scenario Details: [SCORE]\n"
        "Alignment with Ethical Framework: [SCORE]\n"
        "TOTAL: [SUM]\n"
        "(Repeat for each agent, in the order above. Output nothing else, no explanations or extra agent responses.)\n"
        "[/INST]"
    )
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-label", type=str, help="Only evaluate this agent label")
    args = parser.parse_args()

    # --- Load and clean up the consolidated results ---
    results = load_latest_results()
    scenario_text = results.get("ethical_question", "")
    raw_responses = results.get("agent_responses", {})

    # Filter responses: only EXPECTED_LABELS and non-empty responses
    normalized_expected = {normalize_label(l): l for l in EXPECTED_LABELS}
    filtered_responses = {}
    for key, value in raw_responses.items():
        norm = normalize_label(key)
        if norm in normalized_expected and isinstance(value, str) and value.strip():
            canon_label = normalized_expected[norm]
            filtered_responses[canon_label] = value.strip()
    agent_responses = filtered_responses

    # Decide which labels to rate
    eval_labels = EXPECTED_LABELS
    if args.agent_label:
        label_norm = normalize_label(args.agent_label)
        eval_labels = [l for l in EXPECTED_LABELS if normalize_label(l) == label_norm]
        if not eval_labels:
            raise ValueError(f"Label {args.agent_label} not found among expected labels.")

    # --- Build the Mistral-style prompt ---
    prompt = build_prompt(scenario_text, agent_responses, eval_labels)

    # --- Run the LLM ---
    gen_max_tokens = 1200
    result = llama(prompt, max_tokens=gen_max_tokens)
    evaluation_text = result["choices"][0]["text"].strip()

    # --- Save the raw output for audit trail ---
    os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_out_path = os.path.join(AGENT_OUTPUT_DIR, f"response_eval_{timestamp}.txt")
    with open(raw_out_path, "w", encoding="utf-8") as f:
        f.write(evaluation_text)

    # --- Parse evaluation into sections by label ---
    output_sections = {}
    for label in eval_labels:
        label_clean = label.rstrip(":")
        pattern = re.compile(
            rf"---\s*{re.escape(label_clean)}\s*---\s*(.*?)(?=(?:---\s*\S+\s*---)|\Z)",
            re.DOTALL
        )
        match = pattern.search(evaluation_text)
        if match:
            val = match.group(1)
            # Remove spurious double backslash newlines etc.
            val = val.replace("\\n", "\n").replace("\r", "").strip()
            # Remove any redundant internal JSON
            try:
                json.loads(val)
                # If this is valid JSON, it's a mistake (should be plain text), so skip
                continue
            except Exception:
                pass
            output_sections[label] = val

    # --- Output: If only one agent was requested, print as plain text ---
    if args.agent_label:
        if output_sections:
            print(next(iter(output_sections.values())))
        else:
            print("No ratings found for the requested agent label.")
        return

    # --- Output: All agent ratings as JSON ---
    output_data = {
        "ethical_question": scenario_text,
        "agent_ratings": output_sections
    }
    # Save for pipeline use
    json_path = SCRIPT_DIR / "latest_ratings.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(output_data, jf, indent=2)
    print(json.dumps(output_data, indent=2))

if __name__ == "__main__":
    main()