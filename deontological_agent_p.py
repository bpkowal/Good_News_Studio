

from pathlib import Path
LAST_QUERY_PATH = Path("agent_outputs/.last_query.txt")
LAST_RESPONSE_PATH = Path("agent_outputs/.last_response.txt")
import atexit
import gc
import glob
MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
# from load_deontological_corpus import load_deontological_corpus
from langchain.schema import Document as LangchainDoc
import json
from datetime import datetime
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from get_semantic_tag import get_semantic_tag_weights


vectorstore = None
embedder = None

def cleanup_vectorstore():
    global vectorstore, embedder
    if vectorstore is not None:
        del vectorstore
    if embedder is not None:
        del embedder
    gc.collect()

atexit.register(cleanup_vectorstore)

class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def load_scenario_weights(scenario_id):
    print(f"\U0001f9e0 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    return get_semantic_tag_weights(scenario_id, scenario_dir=Path("scenarios"), corpus_dir=Path("deontological_corpus"))

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
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from load_deontological_corpus import load_deontological_corpus
    # Let cleanup_vectorstore handle resource cleanup at exit
    global vectorstore, embedder
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = load_deontological_corpus()

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

def respond_to_query(query: str, scenario_id: str, temperature: float = 0.4, max_tokens: int = 300, llm=None, scenario_path=None) -> str:
    if scenario_path is None:
        scenario_path = Path(f"scenarios/{scenario_id}.json")

    if not query or not scenario_id:
        raise ValueError("Both 'query' and 'scenario_id' must be provided.")

    # Check if query matches last processed query
    if LAST_QUERY_PATH.exists() and LAST_RESPONSE_PATH.exists():
        last_query = LAST_QUERY_PATH.read_text().strip()
        if query.strip() == last_query:
            print("⚡ Skipping LLM call — using cached response.")
            return LAST_RESPONSE_PATH.read_text().strip()

    context, top_quotes = retrieve_deontological_quotes(query, scenario_id)
    print(f"🔍 Retrieved context for quotes:\n{context}\n")
    print(f"🔍 Retrieved top_quotes list: {top_quotes}\n")

    if llm is None:
        from llama_cpp import Llama
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=768,
            n_threads=6,
            n_gpu_layers=60,
            n_batch=64,  # change to 60 for independent runs
            verbose=False
        )

    prompt = f"""<s>[INST]
You are a strict Kantian ethics assistant. Apply the categorical imperative by reasoning only by maxims you can will as universal laws. Do not reference or mention any other ethical frameworks (e.g., consequentialism) or that you are ignoring them.

For each maxim:
1. State the maxim exactly.
2. Test: "Can all rational agents in identical circumstances will this maxim as a universal law?"
3. Name at least one duty (truth-telling, respect, autonomy, justice, or impartiality) and explain how it applies.
4. If the maxim involves coercion or force, always choose "autonomy" as the duty, overriding other duties.
5. Always remember: Respect for persons as ends applies to all, regardless of their moral character, unless the universal law itself justifies an exception.
Focus solely on deontological reasoning consistent with the relevant quotes below.
Do not repeat any instructions in your Deontological Answer; only provide the reasoning itself.

### Relevant Quotes:
{context}

### Ethical Question:
{query}

Provide your answer as a Deontological Answer.
[/INST]
"""

    completion = llm(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
    if isinstance(completion, str):
        final_response = completion.strip()
    elif isinstance(completion, dict) and "choices" in completion:
        final_response = "".join(choice["text"] for choice in completion["choices"]).strip()
    else:
        final_response = "[ERROR] Unexpected response format from LLM."

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

    # Save the current query and response for caching
    LAST_QUERY_PATH.write_text(query.strip())
    LAST_RESPONSE_PATH.write_text(final_response.strip())

    print(f"\U0001f4be Saved output to: {output_path.name}")
    return final_response

if __name__ == "__main__":
    scenario_files = sorted(Path("scenarios").glob("*.json"), key=os.path.getmtime, reverse=True)
    if not scenario_files:
        raise FileNotFoundError("No scenario files found in 'scenarios' directory.")
    latest_scenario_path = scenario_files[0]
    scenario_id = latest_scenario_path.stem
    scenario_file = json.load(open(latest_scenario_path))
    query = scenario_file["ethical_question"]
    print("🧠 Deontological Response:\n", respond_to_query(query, scenario_id, scenario_path=latest_scenario_path))
