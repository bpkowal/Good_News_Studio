from llama_cpp import Llama
import networkx as nx
import re
import argparse
import json

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
    concepts = [
        "Duty", "Universality", "Respect", "Persons", "Consent", "Autonomy",
        "Noncombatant Immunity", "Harm", "Omission", "Role",
        "Honesty", "Transparency", "Informed Consent", "Justice"
    ]
    G.add_nodes_from(concepts)
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

# === Check for Deontological Paths ===
def is_valid_path(graph, source, target):
    if graph.has_node(source) and graph.has_node(target) and nx.has_path(graph, source, target):
        path = nx.shortest_path(graph, source, target)
        relations = [
            graph.edges[path[i], path[i+1]]["relation"]
            for i in range(len(path)-1)
        ]
        return True, list(zip(path, relations))
    return False, []

# === Core Critic Logic ===
def run_deontology_critic(question, answer, graph):
    synonyms = {
        "honesty": "Honesty", "truth": "Honesty", "transparent": "Transparency",
        "informed consent": "Informed Consent", "consent": "Informed Consent", "justice": "Justice"
    }
    answer_lower = answer.lower()
    found = {c for c in graph.nodes if c.lower() in answer_lower}
    for syn, canon in synonyms.items():
        if syn in answer_lower:
            found.add(canon)

    checks = {
        "explicit_naming": bool(re.search(r"Maxim\s*\d*\s*[:\.]", answer, re.I)),
        "universalization": bool(re.search(r"universaliz\w*|universal law", answer, re.I)),
        "no_new_hypotheticals": not bool(re.search(r"\b(imagine|suppose|what if)\b", answer, re.I)),
        "ontology_links": any(
            is_valid_path(graph, s, t)[0] for s in found for t in found if s != t
        )
    }
    score = round(sum(checks.values()) / len(checks), 2)

    criteria_texts = {
        "explicit_naming": "explicitly named each maxim with 'Maxim' and labeled duties",
        "universalization": "used 'universal law' or checked for universalizability",
        "no_new_hypotheticals": "did not introduce hypothetical phrasing like 'imagine' or 'suppose'",
        "ontology_links": "linked ontology nodes via valid deontological paths"
    }
    justifications = {}
    for name, passed in checks.items():
        prompt = f"""### System:
You are a deontology critique assistant. The automatic check '{name}' has {'passed' if passed else 'failed'} because the response {criteria_texts[name]}.

Ethical Question:
{question}

Deontological Response:
{answer}

Provide one sentence justification."""
        resp = llm(prompt, max_tokens=100, temperature=0.0, stop=["\n"])
        justifications[name] = resp['choices'][0]['text'].strip()

    feedback_lines = [
        f"- {n}: {'✅ Passed' if p else '❌ Failed'} — {justifications[n]}"
        for n, p in checks.items()
    ]
    feedback_lines.append(f"\nFinal Score: {score}/1.0")
    feedback = "\n".join(feedback_lines)
    return score, feedback

# === CLI Entry ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    graph = build_deontology_graph()
    score, feedback = run_deontology_critic(args.question, args.response, graph)
    print(json.dumps({"score": score, "feedback": feedback}))