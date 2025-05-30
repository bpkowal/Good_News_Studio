# File 1: evaluate_responses.py
# This agent evaluates the four ethical agent responses

import os
import glob
from llama_cpp import Llama
from datetime import datetime

MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CONTEXT_SIZE = 3200
SCENARIO_DIR = "scenarios"
AGENT_OUTPUT_DIR = "agent_outputs"

llama = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=60, n_threads=6, temperature=0.7)

def get_most_recent_file(directory, pattern="*"):
    files = glob.glob(os.path.join(directory, pattern))
    return max(files, key=os.path.getmtime) if files else None

def load_file_contents(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def main():
    scenario_file = get_most_recent_file(SCENARIO_DIR, "*.json")
    scenario_text = load_file_contents(scenario_file)

    response_files = sorted(glob.glob(os.path.join(AGENT_OUTPUT_DIR, "*.txt")), key=os.path.getmtime, reverse=True)[:4]
    agent_responses = {}
    for filepath in response_files:
        content = load_file_contents(filepath)
        for label in ["Virtue Ethics Response:", "Utilitarian Response:", "Deontological Response:", "Care Ethics Response:"]:
            if label in content:
                agent = label.split()[0].replace("Ethics", "")
                agent_responses[agent] = content[content.index(label):].strip()
                break

    prompt = f"Scenario:\n{scenario_text}\n\n"
    prompt += "Agent Responses:\n"
    for agent, response in agent_responses.items():
        prompt += f"\n--- Response from {agent} ---\n{response}\n"

    prompt += ("\n### Evaluate Each Agent Response:\n"
               "Rate the following for each agent:\n"
               "- Ethical clarity\n"
               "- Internal consistency\n"
               "- Relevance to scenario details\n"
               "- Alignment with its ethical framework\n"
               "Summarize each rating in 1-2 sentences.\n")

    print("\nPrompt Preview:\n", prompt[:1000], "...\n")
    result = llama(prompt, max_tokens=1000)
    evaluation_text = result["choices"][0]["text"].strip()

    with open(os.path.join(AGENT_OUTPUT_DIR, f"response_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), "w", encoding="utf-8") as f:
        f.write(evaluation_text)

if __name__ == "__main__":
    main()