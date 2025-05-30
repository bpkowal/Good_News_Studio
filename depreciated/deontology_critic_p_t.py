from datetime import datetime
import networkx as nx
from llama_cpp import Llama
import os
import re
from transformers import GPT2TokenizerFast
import argparse
import json

# GPT-2 tokenizer for debug token counts
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# === Model Setup ===
MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=840,
    n_threads=6,
    n_gpu_layers=30,
    verbose=False
)

# === Build Deontology Ontology Graph ===
def build_deontology_graph():
    G = nx.DiGraph()
    # Define core deontological concepts
    concepts = [
        "Duty", "Universality", "Respect", "Persons", 
        "Consent", "Autonomy", "Noncombatant Immunity", 
        "Harm", "Omission", "Role",
        "Honesty", "Transparency", "Informed Consent",
        "Justice"
    ]
    for concept in concepts:
        G.add_node(concept)
    # Define relations between concepts
    edges = [
        ("Duty", "Universality", "requires_universalizable"),
        ("Respect", "Persons", "demands"),
        ("Consent", "Autonomy", "supports"),
        ("Noncombatant Immunity", "Harm", "forbids"),
        ("Duty", "Role", "entails"),
        ("Omission", "Harm", "can_cause"),
        ("Honesty", "Respect", "supports"),
        ("Transparency", "Autonomy", "supports"),
        ("Informed Consent", "Autonomy", "demands"),
        ("Respect", "Autonomy", "supports")
    ]
    for src, tgt, rel in edges:
        G.add_edge(src, tgt, relation=rel)
    return G

def is_valid_path(graph, source, target):
    """ Check if a valid deontological path exists between two concepts. """
    if graph.has_node(source) and graph.has_node(target) and nx.has_path(graph, source, target):
        path = nx.shortest_path(graph, source, target)
        relations = [
            graph.get_edge_data(path[i], path[i+1])["relation"]
            for i in range(len(path)-1)
        ]
        return True, list(zip(path, relations))
    return False, []

# === Load Agent Output ===
def load_agent_output(filepath: str):
    """ Load the ethical question and deontological response from a file. """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # Extract 'Ethical Question' using anchored regex
    q_match = re.search(
        r"(?m)^Ethical Question:\s*(.*?)\s*^Deontological Response:",
        content,
        flags=re.S
    )
    if not q_match:
        raise ValueError("Could not extract 'Ethical Question' from file")
    question = q_match.group(1).strip()
    # Remove any "Top Quotes Used:" section from the extracted question
    if "Top Quotes Used:" in question:
        question = question.split("Top Quotes Used:")[0].strip()

    # Extract 'Deontological Response' using anchored regex (capture all following lines)
    r_match = re.search(
        r"(?m)^Deontological Response:\s*([\s\S]+)$",
        content
    )
    if not r_match:
        raise ValueError("Could not extract 'Deontological Response' from file")
    answer = r_match.group(1).strip()

    return question, answer

# === Deontology Critique Logic ===
def run_deontology_critic(question: str, answer: str, graph: nx.DiGraph, scenario_meta=None, max_tokens=840):
    # 1. Synonym-aware concept matching
    synonyms = {
        "honesty": "Honesty",
        "truth": "Honesty",
        "transparent": "Transparency",
        "informed consent": "Informed Consent",
        "consent": "Informed Consent",
        "justice": "Justice"
    }
    all_concepts = set(graph.nodes)
    found_concepts = set()
    answer_lower = answer.lower()
    for concept in all_concepts:
        if concept.lower() in answer_lower:
            found_concepts.add(concept)
    for syn, canon in synonyms.items():
        if syn in answer_lower:
            found_concepts.add(canon)

    # 2. Automatic ontology-path inconsistencies
    inconsistencies = []
    for src, tgt, rel in graph.edges(data="relation"):
        if src in found_concepts and tgt in found_concepts:
            if not nx.has_path(graph, src, tgt):
                inconsistencies.append(f"No path: {src} → {tgt}")

    # 3. Regex-based baseline checks (case-insensitive, flattened)
    checks = {
        'explicit_naming': bool(re.search(r"Maxim\s*\d*\s*[:\\.]", answer, re.IGNORECASE)),
        'universalization': (
            bool(re.search(r"can this (?:maxim\s*)?be a universal law\?", answer, re.IGNORECASE)) or
            bool(re.search(r"\buniversaliz\w*\b", answer, re.IGNORECASE)) or
            bool(re.search(r"\buniversal law\b", answer, re.IGNORECASE))
        ),
        # Temporarily disabled label check
        # 'labels': bool(re.search(r"label\s*[:\-]\s*(permissible|impermissible)", answer, re.IGNORECASE)),
        'no_new_hypotheticals': not bool(re.search(r"\b(imagine|suppose|what if)\b", answer, re.IGNORECASE)),
    }
    # Ontology links check: pass if any valid path exists between any two found concepts
    ontology_links = False
    for src in found_concepts:
        for tgt in found_concepts:
            if src != tgt:
                valid, _ = is_valid_path(graph, src, tgt)
                if valid:
                    ontology_links = True
                    break
        if ontology_links:
            break
    checks['ontology_links'] = ontology_links
    passed = sum(checks.values())
    total = len(checks)
    baseline_score = round(passed / total, 2)

    # 4. Summary of automatic checks
    checks_summary = "Automatic check results:\n" + "\n".join(
        f"- {name}: {'✅ Passed' if ok else '❌ Failed'}"
        for name, ok in checks.items()
    ) + f"\n\nBaseline score: {baseline_score}/1.0\n\n"

    # Compute check order string
    check_order = list(checks.keys())
    check_order_str = ", ".join(check_order)

    # Debug: show automatic check results
    print("\n🤖 Running Deontology Critique Review...")
    print("📝 Automatic check summary:")
    print(checks_summary)

    # Prepare inconsistency text for prompt
    if inconsistencies:
        inconsistency_text = "\n".join(inconsistencies)
    else:
        inconsistency_text = "None"

    # 6. Ask LLM to justify each check individually
    justifications = {}
    for name, ok in checks.items():
        # Determine criterion description for more specific justification
        if name == 'explicit_naming':
            criteria_text = "explicitly named each maxim with the phrase 'Maxim' and labeled a duty (e.g., 'Duty: Autonomy')"
        elif name == 'universalization':
            criteria_text = "used the phrase 'universal law' or asked 'Can all rational agents in identical circumstances will this maxim as a universal law?'"
        elif name == 'no_new_hypotheticals':
            criteria_text = "did not introduce any new hypothetical phrasing such as 'imagine', 'suppose', or 'what if'"
        elif name == 'ontology_links':
            # pick a sample linked path for illustration
            sample_path = next(
                " → ".join([f"{p}({r})" for p, r in path])
                for src in found_concepts for tgt in found_concepts if src != tgt
                for valid, path in [is_valid_path(graph, src, tgt)] if valid
            )
            criteria_text = f"linked ontology nodes along path {sample_path}"
        else:
            criteria_text = ""
        single_prompt = f"""### System:
You are a deontology critique assistant. The automatic check '{name}' has {'passed' if ok else 'failed'} because the response {criteria_text}.

Ethical Question:
{question}

Deontological Response:
{answer}

Provide one sentence justifying why this check {'passed' if ok else 'failed'} by referencing how the response {criteria_text}.
Justification:"""
        # optional short pause between calls
        import time; time.sleep(0.2)
        # allow more tokens and stop on single newline
        resp = llm(
            single_prompt + " Justification (one complete sentence):",
            max_tokens=100,
            temperature=0.0,
            stop=["\n"]
        )
        print(f"\n---\nPrompt for check '{name}':\n{single_prompt}\n")
        print("Raw LLM response object:", resp)
        print("Raw LLM text:", repr(resp['choices'][0]['text']))
        print("---\n")
        justifications[name] = resp['choices'][0]['text'].strip()

    # 7. Build composite critique text
    lines = []
    for name, ok in checks.items():
        status = '✅ Passed' if ok else '❌ Failed'
        justification = justifications.get(name, '')
        lines.append(f"- {name}: {status} — {justification}")
    lines.append(f"\nFinal Score: {baseline_score}/1.0")
    critique_text = "\n".join(lines)

    # 8. Return score and feedback
    final_score = baseline_score
    feedback = critique_text
    return final_score, feedback

# === Manual Runner ===
if __name__ == "__main__":
    # Parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True, help="Ethical question text")
    parser.add_argument("--response", type=str, required=True, help="Agent's response text")
    args = parser.parse_args()

    # Build graph and run critic
    graph = build_deontology_graph()
    score, feedback = run_deontology_critic(
        args.question,
        args.response,
        graph
    )

    # Output JSON with score and feedback
    output = {
        "score": score,
        "feedback": feedback
    }
    print(json.dumps(output))