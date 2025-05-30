from llama_cpp import Llama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from load_utilitarian_corpus import load_utilitarian_corpus
from langchain.schema import Document as LangchainDoc
import json
from pathlib import Path
from datetime import datetime
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from get_semantic_tag import get_semantic_tag_weights
from pathlib import Path
import gc
import atexit


MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"

vectorstore = None
embedder = None




class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def load_scenario_weights(scenario_id):
    print(f"🧠 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    return get_semantic_tag_weights(scenario_id, scenario_dir=Path("utilitarian_scenarios"), corpus_dir=Path("utilitarian_corpus"))

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

def retrieve_utilitarian_quotes(query: str, scenario_id: str, limit_per_quote: int = 250):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from load_utilitarian_corpus import load_utilitarian_corpus

    global vectorstore, embedder
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = load_utilitarian_corpus()

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

def respond_to_query(query: str, scenario_id: str, temperature: float = 0.5, max_tokens: int = 300, llm=None, scenario_path=None) -> str:
   
    if scenario_path is None:
        scenario_path = Path(f"utilitarian_scenarios/{scenario_id}.json")
    
    if not query or not scenario_id:
        raise ValueError("Both 'query' and 'scenario_id' must be provided.")

    if llm is None:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=768,
            n_threads=6,
            n_gpu_layers=60,
            n_batch=64,
            verbose=False
        )

    context, top_quotes = retrieve_utilitarian_quotes(query, scenario_id)

    import time; time.sleep(2)

    prompt = f"""
You are a utilitarian ethics assistant. Your goal is to determine the action best aligned with utilitarian principles, using the corpus excerpts provided.

- Base your decision entirely on consequences.
- Do not assume harm is always wrong—utilitarianism may permit harm if it maximizes net well-being.
- Ignore proximity and immediacy unless they affect outcomes.
- Use the following corpus excerpts in your reasoning. Explicitly reference or paraphrase their logic where applicable.
- If catastrophic consequences outweigh intuition, state so clearly.

Corpus Materials:
{context}

Ethical Question:
{query}

Utilitarian Answer:
"""

    completion = llm(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
    if isinstance(completion, str):
        final_response = completion.strip()
    elif isinstance(completion, dict) and "choices" in completion:
        final_response = "".join(choice["text"] for choice in completion["choices"]).strip()
    else:
        final_response = "[ERROR] Unexpected response format from LLM."

    del llm
    gc.collect()
    import time; time.sleep(1)

    os.makedirs("agent_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"agent_outputs/response_{timestamp}.txt")

    with open(output_path, "w") as f:
        f.write(f"Ethical Question: {query}\n\n")
        f.write("Top Quotes Used:\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        for quote, score in top_quotes:
            f.write(f"- {quote} (score: {score:.2f})\n")
        f.write("\nUtilitarian Response:\n")
        f.write(final_response + "\n")

    print(f"💾 Saved output to: {output_path.name}")
    return final_response

def cleanup_vectorstore():
    global vectorstore, embedder
    vectorstore = None
    embedder = None
    gc.collect()

atexit.register(cleanup_vectorstore)

if __name__ == "__main__":
    scenario_id = "auto_utilitarian_0812_20250424"
    scenario_file = json.load(open(f"utilitarian_scenarios/{scenario_id}.json"))
    query = scenario_file["ethical_question"]
    print("🧠 Utilitarian Response:\n", respond_to_query(query, scenario_id))
