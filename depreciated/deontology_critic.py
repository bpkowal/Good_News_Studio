from datetime import datetime
import networkx as nx
from llama_cpp import Llama
import os
import re
from transformers import GPT2TokenizerFast

# GPT-2 tokenizer for debug token counts
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# === Model Setup ===
MODEL_PATH = "../llama-2-7b-chat.Q4_K_M.gguf"

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

    # 5. Build LLM prompt
    # Build prompt with explicit bullets for each automatic check
    prompt_lines = [
        "### System:",
        "You are a deontology critique assistant. You have performed automatic checks with the following results."
    ]
    for name, ok in checks.items():
        status = "✅" if ok else "❌"
        prompt_lines.append(f"{status} {name}")
    prompt_lines.append(f"Final Score: {baseline_score}/1.0")
    prompt_lines.extend([
        "",
        "Ethical Question:",
        question,
        "",
        "Deontological Response:",
        answer,
        "",
        "### Critique:"
    ])
    prompt = "\n".join(prompt_lines)

    # Debug: print prompt token count and max_tokens
    token_count = len(tokenizer.encode(prompt))
    print(f"🔢 Prompt token count: {token_count}")
    print(f"🧪 max_tokens set to: {max_tokens}")
    print("📝 Full LLM prompt below:")
    print(prompt)



    # 6. Call the LLM
    response = llm(prompt, max_tokens=max_tokens, top_p=1, temperature=0.0, stop=None)
    critique_text = response["choices"][0]["text"].strip()
    # Debug: show raw LLM critique text
    print("🟢 Raw critique_text:", repr(critique_text))

    # 7. Parse final score from LLM output
    m = re.search(r"Final Score:\s*([0-9]*\.?[0-9]+)", critique_text)
    final_score = float(m.group(1)) if m else baseline_score

    # 8. Return full critique text as feedback
    feedback = critique_text

    return final_score, feedback

# === Manual Runner ===
if __name__ == "__main__":
    graph = build_deontology_graph()
    OUTPUT_DIR = "agent_outputs"
    files = sorted(os.listdir(OUTPUT_DIR), reverse=True)
    print("📂 Available files:")
    for i, fname in enumerate(files):
        print(f"{i + 1}: {fname}")
    choice = int(input("\nSelect a file to review (number): ")) - 1
    filepath = os.path.join(OUTPUT_DIR, files[choice])
    question, answer = load_agent_output(filepath)
    score, feedback = run_deontology_critic(question, answer, graph)
    print(f"Critic Score: {score}")
    print("Critic Feedback:")
    print(feedback)