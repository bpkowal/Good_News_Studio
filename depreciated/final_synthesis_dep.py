import os
import glob
import time
from datetime import datetime
from llama_cpp import Llama

# Configuration
SCENARIO_DIR = "scenarios"
AGENT_OUTPUT_DIR = "agent_outputs"

MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CONTEXT_SIZE = 3200

llama = Llama(
    model_path=MODEL_PATH,
    n_ctx=CONTEXT_SIZE,
    n_gpu_layers=60,       # adjust based on your GPU capacity
    n_threads=6,
    temperature=0.7           # adjust based on your CPU cores
)



def get_most_recent_file(directory, pattern="*"):
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    most_recent_file = max(files, key=os.path.getmtime)
    return most_recent_file

def load_file_contents(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def main():
    print("Loading most recent scenario file...")
    scenario_file = get_most_recent_file(SCENARIO_DIR, "*.json")
    if not scenario_file:
        print("No scenario files found in", SCENARIO_DIR)
        return
    print(f"Most recent scenario file: {scenario_file}")
    # Track total tokens (initialize variable)
    total_prompt_tokens = 0

    scenario_timestamp = os.path.splitext(os.path.basename(scenario_file))[0].split("_")[-1]

    print("Loading latest 4 agent responses...")
    agent_response_files = sorted(
        glob.glob(os.path.join(AGENT_OUTPUT_DIR, "*.txt")),
        key=os.path.getmtime,
        reverse=True
    )[:4]

    if not agent_response_files:
        print("No agent response files found in", AGENT_OUTPUT_DIR)
        return

    agent_responses = {}
    for filepath in agent_response_files:
        content = load_file_contents(filepath)
        if "Virtue Ethics Response:" in content:
            start = content.index("Virtue Ethics Response:")
            agent_responses["Virtue"] = content[start:].strip()
            print(f"Loaded response from agent 'Virtue'")
        elif "Utilitarian Response:" in content:
            start = content.index("Utilitarian Response:")
            agent_responses["Utilitarian"] = content[start:].strip()
            print(f"Loaded response from agent 'Utilitarian'")
        elif "Deontological Response:" in content:
            start = content.index("Deontological Response:")
            agent_responses["Deontology"] = content[start:].strip()
            print(f"Loaded response from agent 'Deontology'")
        elif "Care Ethics Response:" in content:
            start = content.index("Care Ethics Response:")
            agent_responses["Care"] = content[start:].strip()
            print(f"Loaded response from agent 'Care'")
        else:
            print(f"Skipping file with unrecognized format: {filepath}")

    if not agent_responses:
        print("No valid agent responses loaded. Exiting.")
        return

    scenario_text = load_file_contents(scenario_file)

    print("Constructing synthesis prompt...")
    prompt = f"Scenario:\n{scenario_text}\n\n"
    prompt += "Agent Responses:\n"
    for agent, response in agent_responses.items():
        prompt += f"\n--- Response from {agent} ---\n{response}\n"

    prompt += (
        "\n### Begin Synthesis:\n\n"
        "You are an expert negotiator who can effortlessly find agreement amongst divergent views"
        "Compare two or three of the highest quality arguements from the responses.\n"
        "Explain the best possible resolution reasoning from the responses and what alternatives are still sensible for consideration.\n"
    )

    # Estimate token count (rough approximation: 1 token ≈ 0.75 words)
    total_words = len(prompt.split())
    approx_tokens = int(total_words / 0.75)

    print("\n==== Prompt Preview ====\n")
    print(prompt)
    print("\n==== End of Prompt ====")
    print(f"Approximate token count: {approx_tokens}")
    total_prompt_tokens = approx_tokens

    print("Initializing LLM model...")
    llama = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE)

    print("Calling LLM model for synthesis...")
    response = llama(prompt, max_tokens=1000, temperature=0.8)

    synthesis_text = response.get('choices', [{}])[0].get('text', '').strip()
    print("\nSynthesis Response:\n", synthesis_text)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"response_synthesis_{timestamp_str}.txt"
    output_path = os.path.join(AGENT_OUTPUT_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(synthesis_text)
    print(f"Synthesis response saved to {output_path}")

if __name__ == "__main__":
    main()
