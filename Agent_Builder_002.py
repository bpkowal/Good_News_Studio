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
MAX_DIRECTIVES = 6
MAX_WORDS = 50

# Debug flags for manual directive insertion
CANNED_DIRECTIVE = "Universalize each maxim as a universal law."
# Debug flags for manual directive insertion
DEBUG_INSERT_CANNED_DIRECTIVE = False
DEBUG_TEST_LLM_CALL = False


# Initialize LLM and ontology graph
llm = Llama(model_path=MODEL_PATH, n_ctx=1000, n_threads=6, n_gpu_layers=0)
ontology_graph = build_deontology_graph()

# Default directives (must be <= MAX_DIRECTIVES items, < MAX_WORDS words each)
def default_directives():
    return [
        "State each maxim explicitly before testing.",
        "Test each maxim: 'Can this be a universal law?'.",  
        "Label actions 'Permissible' or 'Impermissible'.",  
        "Ignore emotions, outcomes, and social norms.",
        "Use only information from the Ethical Question."       
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
def refine_directives(old_directives, feedback,
                      max_directives=5,
                      max_words=50,
                      similarity_threshold=0.7,
                      max_attempts=3):
    """
    Generate one new, non-redundant directive based on feedback, with a post-filter to
    avoid repeating existing directives. Returns an updated list of directives.
    """
    import json, re, difflib

    # 1) Build an enhanced LLM prompt
    prompt = (
        "You have up to 5 existing directives:\n"
        f"{json.dumps(old_directives)}\n\n"
        "Feedback on why these directives failed:\n"
        f"{feedback}\n\n"
        "**Now** generate exactly ONE brand-new directive (≤50 words) that:\n"
        "  1. Does NOT repeat or lightly rephrase any existing directive.\n"
        "  2. Fills a new gap (e.g. link each maxim to the ontology, require an if-then universalization, or demand a mini-justification).\n"
        "  3. Is actionable, concise, and distinct.\n\n"
        "Return only the JSON-quoted string, e.g. \"New directive text.\""
    )

    new_directive = None

    # 2) Try up to max_attempts times to get a valid, non-similar directive
    for _ in range(max_attempts):
        resp = llm(prompt, max_tokens=100, temperature=0.3)
        raw = resp['choices'][0]['text'].strip()

        # Extract quoted string if present
        m = re.search(r'"([^"]+)"', raw)
        candidate = m.group(1) if m else raw
        candidate = candidate.replace('\n', ' ').strip()

        # Length check
        if len(candidate.split()) > max_words:
            continue

        # Similarity check
        if any(
            difflib.SequenceMatcher(None, candidate.lower(), d.lower()).ratio() > similarity_threshold
            for d in old_directives
        ):
            continue

        new_directive = candidate
        break

    # 3) Fallback if no suitable directive was generated
    if not new_directive:
        new_directive = (
            "After labeling each maxim, add a one-sentence link from each label back to the ontology."
        )

    # 4) Insert into the list, replacing the closest existing directive if at capacity
    if len(old_directives) < max_directives:
        updated = old_directives + [new_directive]
    else:
        ratios = [difflib.SequenceMatcher(None, d, new_directive).ratio() for d in old_directives]
        idx = ratios.index(max(ratios))
        updated = old_directives.copy()
        updated[idx] = new_directive

    return updated

# Main self-improvement loop

def main():
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios: {[sc['scenario_id'] for sc in scenarios]}")
    for scenario in scenarios:
        sid = scenario['scenario_id']
        context = json.dumps(scenario)
        query = scenario['ethical_question']
        directives = default_directives()

        print("\n" + "#" * 60)
        print(f"Starting scenario {sid}")
        print("#" * 60 + "\n")

        for i in range(1, ITERATIONS_PER_SCENARIO + 1):
            # 1) Build and send prompt
            prompt_text = format_prompt(directives, context, query)
            answer = run_agent(prompt_text)

            # 2) Show debug info
            print("\n" + "=" * 40)
            print(f"Iteration {i} for scenario {sid}")
            print("\nPrompt Text:")
            print(prompt_text)
            print("\nAgent Response:")
            print(answer)

            # 3) Critique
            metadata = {"iteration": i}
            result = critique(query, answer, scenario)
            if result is None:
                score, feedback = 0.0, ""
            else:
                score, feedback = result
                if score is None:
                    score = 0.0

            # 4) Show evaluation
            print("\nCritic Evaluation:")
            print(f"Score: {score}")
            print("Feedback:")
            print(feedback)

            # 5) Log
            metadata['score'] = score
            log_prompt_change(sid, i, directives, metadata)

            # 6) Check for success
            if score >= 0.95:
                print(f"\n✅ Score threshold reached ({score:.2f}); moving to next scenario.\n")
                break

            # 7) If not last iteration, ask whether to continue refining
            if i < ITERATIONS_PER_SCENARIO:
                cont = input("\nContinue to next iteration? (y/n): ").strip().lower()
                if cont != 'y':
                    print("⏸️  Exiting current scenario's loop per user request.\n")
                    break

                # 8) Refine directives for next round
                directives = refine_directives(directives, feedback)
            else:
                # we've just finished the last allowed iteration without success
                print(f"\n⚠️  Reached max iterations ({ITERATIONS_PER_SCENARIO}) without meeting threshold; moving on.\n")

        # end of per-scenario loop
    # end of all scenarios

if __name__ == '__main__':
    main()
