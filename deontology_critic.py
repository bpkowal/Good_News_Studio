
from datetime import datetime
import networkx as nx
from llama_cpp import Llama
import os


# === Model Setup ===
MODEL_PATH = "../llama-2-7b-chat.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=800,
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
        "Harm", "Omission", "Role"
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
        ("Omission", "Harm", "can_cause")
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
    if "Deontological Response:" not in content:
        raise ValueError("File must contain 'Deontological Response:' section")
    question_part, response_part = content.split("Deontological Response:", 1)
    question = question_part.split("Ethical Question:")[-1].strip()
    answer = response_part.strip()
    return question, answer

# === Deontology Critique Logic ===
def run_deontology_critic(question: str, answer: str, graph: nx.DiGraph):
    # Extract potential concept pairs mentioned in the answer
    concepts = list(graph.nodes)
    found_pairs = []
    for src in concepts:
        if src in answer:
            for tgt in concepts:
                if tgt in answer and src != tgt:
                    found_pairs.append((src, tgt))
    # Check for consistency
    inconsistencies = []
    for src, tgt in set(found_pairs):
        valid, path = is_valid_path(graph, src, tgt)
        if not valid:
            inconsistencies.append(f"No valid deontological path: {src} → {tgt}")
    # Construct critique prompt
    prompt = (
        "### System:\n"
        "You are a deontological critique assistant. Evaluate the following response "
        "for consistency with core deontological principles:\n"
        "- Universalizability (Duty → Universality)\n"
        "- Respect for persons (Respect → Persons)\n"
        "- Noncombatant immunity and consent where applicable\n\n"
        "Return:\n"
        "1. A list of any inconsistent reasoning paths found.\n"
        "2. A score (0.0-1.0) where 1.0 = fully consistent with deontology.\n"
        "3. A brief justification.\n\n"
        f"Ethical Question:\n{question}\n\n"
        f"Deontological Response:\n{answer}\n\n"
        f"Ontology Inconsistencies:\n"
    )
    if inconsistencies:
        prompt += "\n".join(inconsistencies)
    else:
        prompt += "None"
    # Run the LLM
    response = llm(prompt, max_tokens=800, temperature=0.2)
    critique = response["choices"][0]["text"].strip()
    print("\n🟢 Deontology Critic Output:")
    print(critique)
    print("=" * 80)

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
    run_deontology_critic(question, answer, graph)