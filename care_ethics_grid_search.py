import time
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from load_care_ethics_corpus import load_care_ethics_corpus

MODEL_PATH = "../llama-2-7b-chat.Q4_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=768,
    n_threads=6,
    n_gpu_layers=60,
    verbose=False
)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = load_care_ethics_corpus()

class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def load_scenario_weights(scenario_id):
    filepath = Path(f"scenarios/{scenario_id}.json")
    if filepath.exists():
        with open(filepath, "r") as f:
            data = json.load(f)
        return data.get("tag_expectations", {})
    return {}

def normalize_tags(raw_tags):
    if isinstance(raw_tags, str):
        return [t.strip() for t in raw_tags.split(",")]
    elif isinstance(raw_tags, list):
        return [t.strip() for t in raw_tags]
    return []

def cosine_sim(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_quotes(query, scenario_id, alpha, beta, limit_per_quote=250):
    tag_weights = load_scenario_weights(scenario_id)
    print(f"\U0001f527 Scenario Tag Weights: {tag_weights}")

    raw_docs = vectorstore.similarity_search(query, k=10)
    wrapped_docs = [Document(d.page_content, d.metadata) for d in raw_docs]

    def doc_score(doc):
        tags = normalize_tags(doc.metadata.get("tags", []))
        score = sum(tag_weights.get(tag, 0.0) for tag in tags)
        print(f"\U0001f50d Tags: {tags} → Score: {score:.2f}")
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
        return "", []

    quote_texts = [q[0] for q in quotes]
    quote_tag_scores = [q[1] for q in quotes]

    query_embedding = embedder.embed_query(query)
    quote_embeddings = embedder.embed_documents(quote_texts)

    ranked = []
    for i in range(len(quote_texts)):
        sim = cosine_sim(query_embedding, quote_embeddings[i])
        sim = max(0.0, sim)
        tag_score = quote_tag_scores[i]
        combined_score = alpha * tag_score + beta * sim
        ranked.append((quote_texts[i], combined_score))

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    top_quotes = ranked[:3]
    for quote, score in top_quotes:
        print(f"\U0001f4d8 Quote Used (score: {score:.2f}): {quote}")
    return "\n---\n".join([q for q, _ in top_quotes]), top_quotes

def respond_to_query(query, scenario_id, alpha, beta, temperature=0.5, max_tokens=300):
    context, top_quotes = retrieve_quotes(query, scenario_id, alpha, beta)
    prompt = f"""
You are a care ethics assistant. Your role is to reason from the perspective of care, prioritizing relationships, emotional resonance, and concrete human contexts.
- Prioritize relational closeness and interdependence over abstract impartiality.
- Emphasize empathy, responsiveness, and moral attention to the specific people involved.
- Avoid utilitarian calculus or rigid principles unless reframed in terms of care.
- Use the following corpus excerpts where helpful.

Corpus Materials:
{context}

Ethical Question:
{query}

Care Ethics Answer:
"""
    response = []
    for chunk in llm(prompt, max_tokens=max_tokens, temperature=temperature, stream=True):
        response.append(chunk["choices"][0]["text"])
    final_response = "".join(response).strip()

    os.makedirs("agent_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"agent_outputs/response_{timestamp}_a{alpha}_b{beta}.txt")

    with open(output_path, "w") as f:
        f.write(f"Ethical Question: {query}\n\n")
        f.write("Top Quotes Used:\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        f.write(f"Alpha (tag weight): {alpha}, Beta (semantic weight): {beta}\n")
        for quote, score in top_quotes:
            f.write(f"- {quote} (score: {score:.2f})\n")
        f.write("\nCare Ethics Response:\n")
        f.write(final_response + "\n")

    print(f"\U0001f4be Saved output to: {output_path.name}")
    return final_response

if __name__ == "__main__":
    scenario_id = "disrupted_relationship_001"
    scenario_file = json.load(open(f"scenarios/{scenario_id}.json"))
    query = scenario_file["ethical_question"]

    alphas = [1.0, 1.2, 1.5, 2.0]
    betas = [0.1, 0.2, 0.4, 0.6]

    for alpha in alphas:
        for beta in betas:
            print(f"\n🚀 Running with alpha={alpha}, beta={beta}")
            respond_to_query(query, scenario_id, alpha=alpha, beta=beta)
            time.sleep(2)
