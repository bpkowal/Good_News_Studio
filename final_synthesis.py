import os
import glob
from datetime import datetime
from llama_cpp import Llama

# Configuration
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
    print("Loading most recent evaluation summary file...")
    evaluation_file = get_most_recent_file(AGENT_OUTPUT_DIR, "evaluation_summary_*.txt")
    if not evaluation_file:
        print("No evaluation summary files found in", AGENT_OUTPUT_DIR)
        return
    print(f"Most recent evaluation summary file: {evaluation_file}")

    evaluation_summary = load_file_contents(evaluation_file)

    print("Constructing final synthesis prompt...")
    prompt = (
        f"Evaluation Summary:\n{evaluation_summary}\n\n"
        "Based on the evaluations provided, synthesize a single recommended course of action.\n"
        "Justify it using the ethical reasoning previously rated, and name one top alternative path with its merits.\n"
        "Express epistemological humility in your synthesis.\n"
    )

    # Estimate token count (rough approximation: 1 token ≈ 0.75 words)
    total_words = len(prompt.split())
    approx_tokens = int(total_words / 0.75)

    print("\n==== Prompt Preview ====\n")
    print(prompt)
    print("\n==== End of Prompt ====")
    print(f"Approximate token count: {approx_tokens}")

    print("Initializing LLM model...")
    llama = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE)

    print("Calling LLM model for final synthesis...")
    response = llama(prompt, max_tokens=1000, temperature=0.8)

    synthesis_text = response.get('choices', [{}])[0].get('text', '').strip()
    print("\nFinal Synthesis Response:\n", synthesis_text)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"final_synthesis_{timestamp_str}.txt"
    output_path = os.path.join(AGENT_OUTPUT_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(synthesis_text)
    print(f"Final synthesis response saved to {output_path}")

if __name__ == "__main__":
    main()
