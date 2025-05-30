# File 1: evaluate_responses.py
# This agent evaluates the four ethical agent responses

import os
import glob
from pathlib import Path
from llama_cpp import Llama
from datetime import datetime
import json
import argparse
import re

SCRIPT_DIR = Path(__file__).parent

MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CONTEXT_SIZE = 4000
SCENARIO_DIR = "scenarios"
AGENT_OUTPUT_DIR = "agent_outputs"

llama = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=60, n_threads=6, temperature=0.7)
# Ensure we can calculate token usage

def get_most_recent_file(directory, pattern="*"):
    files = glob.glob(os.path.join(directory, pattern))
    return max(files, key=os.path.getmtime) if files else None

def load_file_contents(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-label", type=str, help="Only evaluate this agent label")
    args = parser.parse_args()

    # Load consolidated results
    latest_results_file = SCRIPT_DIR / "latest_results.json"
    with open(latest_results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    # Rigorously filter loaded JSON to expected structure
    # Ensure 'ethical_question' is a string
    if not isinstance(results.get("ethical_question"), str):
        raise ValueError("Invalid or missing 'ethical_question' in JSON")
    # Define allowed response labels
    allowed_labels = [
        "Virtue Ethics Response:",
        "Utilitarian Response:",
        "Deontological Response:",
        "Care Ethics Response:"
    ]
    if args.agent_label:
        if args.agent_label not in allowed_labels:
            raise ValueError(f"Unknown agent label: {args.agent_label}")
        allowed_labels = [args.agent_label]

    raw_responses = results.get("agent_responses", {})
    # Keep only entries with allowed labels and non-empty strings
    filtered_responses = {
        label: raw_responses[label].strip()
        for label in allowed_labels
        if label in raw_responses and isinstance(raw_responses[label], str) and raw_responses[label].strip()
    }
    results["agent_responses"] = filtered_responses
    scenario_text = results["ethical_question"]
    agent_responses = results["agent_responses"]

    prompt = f"Scenario Question:\n{scenario_text}\n\n"
    prompt += "Agent Responses:\n"
    for agent, response in agent_responses.items():
        prompt += f"\n--- {agent} ---\n{response}\n"

    prompt += (
     "\nEvaluate the response on these four criteria using a numeric scale (1-5):\n"
     "\nDo not repeat the response in your answer."
     "1. Ethical clarity\n"
     "2. Internal consistency\n"
     "3. Relevance to scenario details\n"
     "4. Alignment with its ethical framework\n"
     "For each criterion, assign a score from 1 (lowest) to 5 (highest). After scoring all four criteria, provide a TOTAL score equal to the sum of the four numeric ratings. Do NOT include any additional commentary or summary."
    )

    # Debug token usage (safe call)
    #try:
    #    prompt_tokens = llama.tokenize(prompt)  # use default signature
    #    prompt_token_count = len(prompt_tokens)
    #    print(f"Prompt token count: {prompt_token_count}")
    #    available_tokens = CONTEXT_SIZE - prompt_token_count
    #    print(f"Available tokens for generation: {available_tokens}")
    #    gen_max_tokens = min(1000, max(0, available_tokens))
    #    if available_tokens < 200:
    #        print("⚠️ Warning: Low available context for generation. Consider trimming the prompt or increasing CONTEXT_SIZE.")
    #except Exception as e:
    #    print(f"⚠️ Tokenization failed ({e}), defaulting generation tokens to 1000.")
    gen_max_tokens = 2000

    # print("\nPrompt Preview:\n", prompt[:2000], "...\n")
    result = llama(prompt, max_tokens=gen_max_tokens)
    evaluation_text = result["choices"][0]["text"].strip()

    # Write raw evaluation to text file
    with open(os.path.join(AGENT_OUTPUT_DIR, f"response_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), "w", encoding="utf-8") as f:
        f.write(evaluation_text)

    # Parse evaluation_text into labeled sections
    pattern = re.compile(r"^---\s*(.+?)\s*---\s*$", re.MULTILINE)
    splits = pattern.split(evaluation_text)
    # pattern.split returns [preamble, label1, content1, label2, content2, ...]
    sections = {}
    it = iter(splits[1:])  # skip preamble
    for label, content in zip(it, it):
        key = label if label.endswith("Response:") else f"{label} Response:"
        if key not in sections:
            sections[key] = content.strip()

    # Build output JSON
    output_data = {
        "ethical_question": scenario_text,
        "agent_ratings": sections
    }

    # Write JSON file
    json_path = SCRIPT_DIR / "latest_ratings.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(output_data, jf, indent=2)

    # Print JSON for caller to capture
    print(json.dumps(output_data))

if __name__ == "__main__":
    main()