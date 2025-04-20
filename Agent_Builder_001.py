import os
import json
from datetime import datetime
from pathlib import Path
from llama_cpp import Llama
from deontology_critic import run_deontology_critic, build_deontology_graph
import re

# Configuration
MODEL_PATH = "../llama-2-7b-chat.Q4_K_M.gguf"
SCENARIOS_DIR = "scenarios"
LOGS_DIR = "prompt_logs"
ITERATIONS_PER_SCENARIO = 5
MAX_DIRECTIVES = 5
MAX_WORDS = 18

# Initialize LLM and ontology graph
llm = Llama(model_path=MODEL_PATH, n_ctx=1000, n_threads=6, n_gpu_layers=0)
ontology_graph = build_deontology_graph()

# Default directives (must be <= MAX_DIRECTIVES items, < MAX_WORDS words each)
def default_directives():
    return [
        "State each maxim explicitly before testing.",
        "Test each maxim: 'Can this be a universal law?'.",  
        "Label actions 'Permissible' or 'Impermissible'.",  
        "Ignore emotions, outcomes, and social norms."        
    ]

# Load scenario definitions
def load_scenarios():
    return [json.loads(f.read_text()) for f in sorted(Path(SCENARIOS_DIR).glob("*.json"))]

# Log directive changes
def log_prompt_change(scenario_id, iteration, directives, metadata):
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = Path(LOGS_DIR) / f"{scenario_id}_iter{iteration}.json"
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "directives": directives,
        "metadata": metadata
    }
    log_path.write_text(json.dumps(payload, indent=2))

# Format prompt using current directives
def format_prompt(directives, context, query):
    directive_block = "\n".join(f"- {d}" for d in directives)
    template = f"""
You are a strict Kantian ethics assistant. Apply the categorical imperative: reason only by maxims you can will as universal laws. Ignore all special or role-based duties (e.g., filial piety), emotions, or consequences.

Provide at least 150 words in your answer.

{directive_block}

Corpus Materials:
{context}

Ethical Question:
{query}

Deontological Answer:
"""
    return template

# Run the agent and return answer text
def run_agent(prompt):
    chunks = llm(prompt, stream=True, max_tokens=256, temperature=0.7)
    return ''.join(c['choices'][0]['text'] for c in chunks).strip()

# Critique via deontology critic, returns (score, feedback)
def critique(query, answer, scenario_meta):
    return run_deontology_critic(query, answer, ontology_graph, scenario_meta)

# Refine directives only based on critic feedback
def refine_directives(old_directives, feedback):
    prompt = (
        "Given the following list of directives (each <18 words), refine or replace one to correct errors flagged by feedback. "
        f"Feedback: {feedback}\nDirectives:\n" + "\n".join(old_directives) + "\n" +
        "Return exactly the revised list of directives (up to 5 items)."
    )
    resp = llm(prompt, max_tokens=200, temperature=0.3)
    # Parse response as JSON-like list or newline items
    lines = [l.strip('- ').strip() for l in resp['choices'][0]['text'].splitlines() if l.strip()]
    # Enforce limits
    new = [d for d in lines[:MAX_DIRECTIVES] if len(d.split()) <= MAX_WORDS]
    return new if new else old_directives

# Main self-improvement loop

def main():
    scenarios = load_scenarios()
    for scenario in scenarios:
        sid = scenario['scenario_id']
        context = json.dumps(scenario)
        query = scenario['ethical_question']
        directives = default_directives()

        for i in range(1, ITERATIONS_PER_SCENARIO+1):
            prompt_text = format_prompt(directives, context, query)
            answer = run_agent(prompt_text)
            # Debug output: show prompt and agent response
            print("\n" + "="*40)
            print(f"Iteration {i} for scenario {sid}")
            print("\nPrompt Text:")
            print(prompt_text)
            print("\nAgent Response:")
            print(answer)

            metadata = {"iteration": i}
            result = critique(query, answer, scenario)
            if result is None:
                score, feedback = 0.0, ""
            else:
                score, feedback = result
                if score is None:
                    score = 0.0
            
            # Debug output: show critic evaluation
            print("\nCritic Evaluation:")
            print(f"Score: {score}")
            print("Feedback:")
            print(feedback)

            # Ask user whether to continue or quit
            cont = input("\nContinue to next iteration? (y/n): ").strip().lower()
            if cont != 'y':
                print("Exiting loop as per user request.")
                return
            
            metadata['score'] = score
            if score >= 0.95:
                break
            directives = refine_directives(directives, feedback)

if __name__ == '__main__':
    main()
