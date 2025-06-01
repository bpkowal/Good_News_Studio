MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from load_virtue_ethics_corpus import load_virtue_ethics_corpus
from langchain.schema import Document as LangchainDoc
import json
from pathlib import Path
import glob
from datetime import datetime
import os
import gc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from get_semantic_tag import get_semantic_tag_weights

import atexit

def cleanup_vectorstore():
    global vectorstore
    global embedder
    try:
        del vectorstore
    except NameError:
        pass
    try:
        del embedder
    except NameError:
        pass
    gc.collect()

atexit.register(cleanup_vectorstore)

class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def load_scenario_weights(scenario_id):
    print(f"🧠 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    return get_semantic_tag_weights(scenario_id, scenario_dir=Path("scenarios"), corpus_dir=Path("virtue_ethics_corpus"))


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

def retrieve_virtue_ethics_quotes(query: str, scenario_id: str, limit_per_quote: int = 250):
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

    del tag_weights
    del raw_docs
    del wrapped_docs
    del quote_embeddings
    del quote_texts
    del quote_tag_scores
    del ranked
    gc.collect()

    return "\n---\n".join([q for q, _ in top_quotes]), top_quotes

def respond_to_query(query=None, scenario_id=None, scenario_path=None, temperature: float = 0.7, max_tokens: int = 300, llm=None) -> str:
    # Load vectorstore and embedder only when needed
    from langchain_huggingface import HuggingFaceEmbeddings
    from load_virtue_ethics_corpus import load_virtue_ethics_corpus

    global embedder
    global vectorstore
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = load_virtue_ethics_corpus()

    if scenario_path:
        try:
            with open(scenario_path, "r") as f:
                scenario_data = json.load(f)
                query = scenario_data.get("ethical_question", query)
                scenario_id = Path(scenario_path).stem
        except FileNotFoundError:
            print(f"❌ Provided scenario file not found: {scenario_path}")
            return "[ERROR] Scenario file not found."

    if query is None or scenario_id is None:
        print("⚠️ No query or scenario ID provided to respond_to_query. Aborting.")
        return "[ERROR] Missing input."
    context, top_quotes = retrieve_virtue_ethics_quotes(query, scenario_id)

    # Cleanup RAG components
    del embedder
    del vectorstore
    gc.collect()

    if llm is None:
        from llama_cpp import Llama  # Deferred import to avoid GPU crash during embeddings
        global MODEL_PATH
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=768,
            n_threads=6,
            n_gpu_layers=60,#was 60 just testing
            n_batch=64,
            verbose=False
        )



    prompt = f"""<s>[INST] You are a virtue ethics assistant. Your task is to generate a detailed answer from the perspective of virtue ethics to the ethical scenario provided below. You should reason from the perspective of virtue ethics, focusing on character, habituation, and human flourishing.
    - Prioritize the development of moral character and virtues over rule-based moral frameworks.
    - Use moral exemplars, narrative analogies, and lived experience as sources of ethical insight.
    - Provide a specific course of action consistent with one path a virtue ethicists could recommend.

    Here are the corpus materials for reference:
    {context}
    Ethical Question:
    {query}
    [/INST]
    """

    output = llm(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
    final_response = output["choices"][0]["text"].strip() + "\n</s>"

    del llm
    gc.collect()

    os.makedirs("agent_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"agent_outputs/response_virtue_{timestamp}.txt")

    with open(output_path, "w") as f:
        f.write(f"Ethical Question: {query}\n\n")
        f.write("Top Quotes Used:\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        for quote, score in top_quotes:
            f.write(f"- {quote} (score: {score:.2f})\n")
        f.write("\nVirtue Ethics Response:\n")
        f.write(final_response + "\n")

    print(f"💾 Saved output to: {output_path.name}")

    del context
    del top_quotes
    gc.collect()

    return final_response


if __name__ == "__main__":
    # Auto-detect the most recent scenario file
    scenario_files = sorted(glob.glob("scenarios/*.json"), key=os.path.getmtime, reverse=True)
    if scenario_files:
        scenario_path = scenario_files[0]
        scenario_id = Path(scenario_path).stem
        try:
            with open(scenario_path) as f:
                scenario_file = json.load(f)
                query = scenario_file.get("ethical_question", "")
                if query:
                    print("🧡 Virtue Ethics Response:\n", respond_to_query(query=query, scenario_id=scenario_id, scenario_path=scenario_path))
                else:
                    print("⚠️ No ethical question found in the scenario file.")
        except FileNotFoundError:
            print(f"❌ Scenario file not found: {scenario_path}")
    else:
        print("❌ No scenario files found in 'scenarios/'")
