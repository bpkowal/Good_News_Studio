from load_care_ethics_corpus import load_care_ethics_corpus
from langchain.schema import Document as LangchainDoc
import json
from pathlib import Path
from datetime import datetime
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from get_semantic_tag import get_semantic_tag_weights
import atexit
import gc

MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"

class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def load_scenario_weights(scenario_id):
    print(f"🧠 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    return get_semantic_tag_weights(scenario_id, scenario_dir=Path("care_scenarios"), corpus_dir=Path("care_ethics_corpus"))

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

def retrieve_care_ethics_quotes(query: str, scenario_id: str, limit_per_quote: int = 250):
    from langchain_huggingface import HuggingFaceEmbeddings
    from load_care_ethics_corpus import load_care_ethics_corpus
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    global vectorstore
    vectorstore = load_care_ethics_corpus()

    tag_weights = load_scenario_weights(scenario_id)
    print(f"🔧 Scenario Tag Weights: {tag_weights}")

    raw_docs = vectorstore.similarity_search(query, k=10)
    wrapped_docs = [Document(d.page_content, d.metadata) for d in raw_docs]

    def doc_score(doc):
        tags = normalize_tags(doc.metadata.get("tags", []))
        score = 0.0
        for tag in tags:
            tag_weight = tag_weights.get(tag, 0.0)
            if tag_weight > 0:
                print(f"🔍 Tag '{tag}' has semantic weight {tag_weight}")
            score += tag_weight
        print(f"🧪 Normalized Tags: {tags} → Score: {score:.2f}")
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
        del vectorstore
        del embedder
        import gc; gc.collect()
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
        print(f"📘 Quote Used (score: {score:.2f}): {quote}")

    # Cleanup to free memory
    del quote_embeddings
    del quote_texts
    del quote_tag_scores
    del query_embedding
    del wrapped_docs
    del raw_docs
    del doc_scores
    del ranked
    del vectorstore
    del embedder
    import gc; gc.collect()

    return "\n---\n".join([q for q, _ in top_quotes]), top_quotes

def respond_to_query(query: str, scenario_id: str, scenario_path=None, temperature: float = 0.7, max_tokens: int = 300, llm=None) -> str:
        # fallback if scenario_path not provided
    
    if scenario_path is None:
        scenario_path = Path(f"care_scenarios/{scenario_id}.json")
    
    if not query or not scenario_id:
        raise ValueError("Both 'query' and 'scenario_id' must be provided.")

    context, top_quotes = retrieve_care_ethics_quotes(query, scenario_id)

    if llm is None:
        from llama_cpp import Llama  # Deferred import to avoid GPU crash during embeddings
        global MODEL_PATH
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=768,
            n_threads=6,
            n_gpu_layers=60,
            n_batch=64,
            verbose=False
        )

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

    output = llm(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
    if isinstance(output, str):
        final_response = output.strip()
    elif isinstance(output, dict) and "choices" in output:
        final_response = output["choices"][0]["text"].strip()
    else:
        response = []
        for chunk in output:
            if isinstance(chunk, dict) and "choices" in chunk:
                response.append(chunk["choices"][0]["text"])
            elif isinstance(chunk, str):
                response.append(chunk)
        final_response = "".join(response).strip()

    os.makedirs("agent_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"agent_outputs/response_{timestamp}.txt")

    with open(output_path, "w") as f:
        f.write(f"Ethical Question: {query}\n\n")
        f.write("Top Quotes Used:\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        for quote, score in top_quotes:
            f.write(f"- {quote} (score: {score:.2f})\n")
        f.write("\nCare Ethics Response:\n")
        f.write(final_response + "\n")

    print(f"💾 Saved output to: {output_path.name}")
    return final_response

def cleanup_vectorstore():
    global vectorstore
    try:
        del vectorstore
    except NameError:
        pass
    gc.collect()

atexit.register(cleanup_vectorstore)

if __name__ == "__main__":
    scenario_id = "auto_care_0812_20250424"
    scenario_path = Path(f"care_scenarios/{scenario_id}.json")
    if scenario_path.exists():
        with open(scenario_path) as sf:
            scenario_file = json.load(sf)
        query = scenario_file.get("ethical_question", "")
        if query:
            print("🧡 Care Ethics Response:\n", respond_to_query(query, scenario_id))
        else:
            print(f"Error: 'ethical_question' not found in scenario file {scenario_path}")
    else:
        print(f"Error: Scenario file {scenario_path} does not exist.")
