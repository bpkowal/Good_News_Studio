from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from load_deontological_corpus import load_deontological_corpus
from langchain.schema import Document as LangchainDoc
import json
from pathlib import Path
from datetime import datetime
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from get_semantic_tag import get_semantic_tag_weights

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = load_deontological_corpus()

class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def load_scenario_weights(scenario_id):
    print(f"\U0001f9e0 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    return get_semantic_tag_weights(scenario_id, scenario_dir=Path("deontology_scenarios"), corpus_dir=Path("deontological_corpus"))

def normalize_tags(raw_tags):
    if isinstance(raw_tags, str):
        return [t.strip() for t in raw_tags.split(",")]
    elif isinstance(raw_tags, list):
        return [t.strip() for t in raw_tags]
    return []

def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_deontological_quotes(query: str, scenario_id: str, limit_per_quote: int = 250):
    tag_weights = load_scenario_weights(scenario_id)
    print(f"\U0001f527 Scenario Tag Weights: {tag_weights}")

    raw_docs = vectorstore.similarity_search(query, k=10)
    print(f"🔍 Retrieved {len(raw_docs)} documents for similarity search.")
    wrapped_docs = [Document(d.page_content, d.metadata) for d in raw_docs]
    if wrapped_docs:
        print("🔍 Sample document content snippet:", wrapped_docs[0].content[:100])

    def doc_score(doc):
        tags = normalize_tags(doc.metadata.get("tags", []))
        score = 0.0
        for tag in tags:
            tag_weight = tag_weights.get(tag, 0.0)
            if tag_weight > 0:
                print(f"\U0001f50d Tag '{tag}' has semantic weight {tag_weight}")
            score += tag_weight
        print(f"\U0001f9ea Normalized Tags: {tags} → Score: {score:.2f}")
        return score

    doc_scores = {id(doc): doc_score(doc) for doc in wrapped_docs}

    quotes = []
    for doc in wrapped_docs:
        for line in doc.content.split("\n"):
            if line.strip().startswith(">"):
                quote = line.strip()[1:].strip()[:limit_per_quote]
                if quote:
                    quotes.append((quote, doc_scores[id(doc)]))

    if not quotes:
        print("⚠️ No '>'-prefixed quotes found, using document excerpts as fallback quotes.")
        for doc in wrapped_docs[:3]:
            lines = [l for l in doc.content.split("\n") if l.strip()]
            excerpt = lines[0].strip()[:limit_per_quote] if lines else ""
            if excerpt:
                quotes.append((excerpt, doc_scores[id(doc)]))
    if not quotes:
        return "", []

    quote_texts = [q[0] for q in quotes]
    quote_tag_scores = [q[1] for q in quotes]

    query_embedding = embedder.embed_query(query)
    quote_embeddings = embedder.embed_documents(quote_texts)

    ranked = []
    for i in range(len(quote_texts)):
        sim = cosine_similarity(query_embedding, quote_embeddings[i])
        sim = max(0.0, sim)
        tag_score = quote_tag_scores[i] if i < len(quote_tag_scores) else 0.0
        combined_score = tag_score * 2 + 0.2 * sim
        ranked.append((quote_texts[i], combined_score))

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    top_quotes = ranked[:3]

    for quote, score in top_quotes:
        print(f"\U0001f4d8 Quote Used (score: {score:.2f}): {quote}")

    return "\n---\n".join([q for q, _ in top_quotes]), top_quotes

def respond_to_query(query: str, scenario_id: str, temperature: float = 0.7, max_tokens: int = 300) -> str:
    context, top_quotes = retrieve_deontological_quotes(query, scenario_id)
    print(f"🔍 Retrieved context for quotes:\n{context}\n")
    print(f"🔍 Retrieved top_quotes list: {top_quotes}\n")

    from llama_cpp import Llama
    MODEL_PATH = "../llama-2-7b-chat.Q4_K_M.gguf"
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=768,
        n_threads=6,
        n_gpu_layers=60,
        verbose=False
    )

    prompt = f"""### System:
You are a strict Kantian ethics assistant. Apply the categorical imperative: reason only by maxims you can will as universal laws. Ignore all special or role-based duties (e.g., filial piety), emotions, or consequences.

- For each recommendation, state the maxim and test it: “Can this be a universal law for all rational agents in the same situation?” If not, label the action impermissible.
- Focus exclusively on duties: truth-telling, respect, autonomy, justice, and impartiality.
- Name at least one duty and how it applies to the situation.
- Do not appeal to saving the most lives, outcomes, social norms, or emotional bonds.
- Do not introduce any new hypothetical scenarios or details not present in the Ethical Question.

### Relevant Quotes:
{context}

### Ethical Question:
{query}

### Deontological Answer:
"""

    response = []
    for chunk in llm(prompt, max_tokens=max_tokens, temperature=temperature, stream=False):
        response.append(chunk["choices"][0]["text"])
    final_response = "".join(response).strip()

    os.makedirs("agent_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"agent_outputs/response_{timestamp}.txt")

    with open(output_path, "w") as f:
        f.write(f"Ethical Question: {query}\n\n")
        # f.write("Relevant Quotes Context:\n")
        # f.write(f"{context}\n\n")
        f.write("Top Quotes Used:\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        for quote, score in top_quotes:
            f.write(f"- {quote} (score: {score:.2f})\n")
        f.write("\nDeontological Response:\n")
        f.write(final_response + "\n")

    print(f"\U0001f4be Saved output to: {output_path.name}")
    return final_response

if __name__ == "__main__":
    scenario_id = "auto_deontological_1617_20250427"
    scenario_file = json.load(open(f"deontology_scenarios/{scenario_id}.json"))
    query = scenario_file["ethical_question"]
    print("\U0001f9e0 Deontological Response:\n", respond_to_query(query, scenario_id))
