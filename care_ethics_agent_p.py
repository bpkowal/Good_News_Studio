import glob
from load_care_ethics_corpus import load_care_ethics_corpus
from langchain.schema import Document as LangchainDoc
import json
from pathlib import Path
from datetime import datetime
import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from get_semantic_tag import get_semantic_tag_weights
import atexit
import gc
from openai import OpenAI
from dotenv import load_dotenv


# Move log function here before first use
def log(msg: str) -> None:
    sys.stderr.write(str(msg) + "\n")


os.environ["TOKENIZERS_PARALLELISM"] = "false"


load_dotenv()
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "agent_outputs"
SCENARIOS_DIR = BASE_DIR / "scenarios"
CORPUS_DIR = BASE_DIR / "care_ethics_corpus"
LATEST_RESULTS_PATH = BASE_DIR / "latest_results.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAST_QUERY_PATH = (BASE_DIR / "agent_outputs" / ".last_query_care.txt")
LAST_RESPONSE_PATH = (BASE_DIR / "agent_outputs" / ".last_response_care.txt")


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    log("⚠️ OPENAI_API_KEY is not set; requests will fail.")
log(f"📁 BASE_DIR set to: {BASE_DIR}")
log(f"📂 OUTPUT_DIR: {OUTPUT_DIR}")
log(f"📂 SCENARIOS_DIR: {SCENARIOS_DIR}")

AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-5-mini")

class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def upsert_latest_results(agent_label: str, query: str, scenario_id: str, response_text: str, top_quotes=None):
    """Merge this agent's response into latest_results.json under agent_ratings/agent_responses style keys.
    Creates the file if missing. Writes atomically via temp file + rename.
    """
    top_quotes = top_quotes or []
    payload = {
        "ethical_question": query,
        "scenario_id": scenario_id,
    }
    # Load existing
    try:
        if LATEST_RESULTS_PATH.exists():
            with open(LATEST_RESULTS_PATH, "r", encoding="utf-8") as rf:
                existing = json.load(rf) or {}
        else:
            existing = {}
    except Exception as e:
        log(f"⚠️ Could not read {LATEST_RESULTS_PATH}: {e}; starting fresh.")
        existing = {}

    # Merge keys
    existing.update(payload)
    agent_responses = existing.get("agent_responses", {})
    agent_responses[agent_label] = response_text
    existing["agent_responses"] = agent_responses

    # (Optional) store quotes for transparency
    if top_quotes:
        # store as list of {quote, score}
        existing.setdefault("agent_supporting_quotes", {})[agent_label] = [
            {"quote": q, "score": float(s)} for (q, s) in top_quotes
        ]

    # Atomic write
    tmp_path = LATEST_RESULTS_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as wf:
            json.dump(existing, wf, ensure_ascii=False, indent=2)
        os.replace(tmp_path, LATEST_RESULTS_PATH)
        log(f"📝 Updated {LATEST_RESULTS_PATH} with {agent_label} response.")
    except Exception as e:
        log(f"❌ Failed to write {LATEST_RESULTS_PATH}: {e}")

def load_scenario_weights(scenario_id):
    log(f"🧠 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    return get_semantic_tag_weights(scenario_id, scenario_dir=SCENARIOS_DIR, corpus_dir=CORPUS_DIR)

def normalize_tags(raw_tags):
    if isinstance(raw_tags, str):
        return [t.strip() for t in raw_tags.split(",")]
    elif isinstance(raw_tags, list):
        return [t.strip() for t in raw_tags]
    return []

def cosine_sim(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def retrieve_care_ethics_quotes(query: str, scenario_id: str, limit_per_quote: int = 250):
    from langchain_huggingface import HuggingFaceEmbeddings
    from load_care_ethics_corpus import load_care_ethics_corpus
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    global vectorstore
    vectorstore = load_care_ethics_corpus()

    tag_weights = load_scenario_weights(scenario_id)
    log(f"🔧 Scenario Tag Weights: {tag_weights}")

    raw_docs = vectorstore.similarity_search(query, k=10)
    wrapped_docs = [Document(d.page_content, d.metadata) for d in raw_docs]

    def doc_score(doc):
        tags = normalize_tags(doc.metadata.get("tags", []))
        score = 0.0
        for tag in tags:
            tag_weight = tag_weights.get(tag, 0.0)
            if tag_weight > 0:
                log(f"🔍 Tag '{tag}' has semantic weight {tag_weight}")
            score += tag_weight
        log(f"🧪 Normalized Tags: {tags} → Score: {score:.2f}")
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
        sim = cosine_sim(query_embedding, quote_embeddings[i])
        sim = max(0.0, sim)
        tag_score = quote_tag_scores[i] if i < len(quote_tag_scores) else 0.0
        combined_score = tag_score * 2 + 0.2 * sim
        ranked.append((quote_texts[i], combined_score))

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    top_quotes = ranked[:3]

    for quote, score in top_quotes:
        log(f"📘 Quote Used (score: {score:.2f}): {quote}")

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

def respond_to_query(query: str, scenario_id: str, scenario_path=None, temperature: float = 0.7, max_tokens: int = 500, llm=None) -> str:
        # fallback if scenario_path not provided
    
    if scenario_path is None:
        scenario_path = SCENARIOS_DIR / f"{scenario_id}.json"
    
    if not query or not scenario_id:
        raise ValueError("Both 'query' and 'scenario_id' must be provided.")

    if LAST_QUERY_PATH.exists() and LAST_RESPONSE_PATH.exists():
        last_query = LAST_QUERY_PATH.read_text().strip()
        if query.strip() == last_query:
            cached_response = LAST_RESPONSE_PATH.read_text()
            if cached_response is None:
                cached_response = ""
            cached_response = cached_response.strip()
            if cached_response:
                log("⚡ Cache hit — reusing last non-empty response.")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_DIR / f"response_{timestamp}.txt"
                with open(output_path, "w") as f:
                    f.write(f"Ethical Question: {query}\n\n")
                    f.write("Top Quotes Used:\n")
                    f.write(f"Scenario ID: {scenario_id}\n")
                    f.write("- (cached)\n")
                    f.write("\nCare Ethics Response:\n")
                    f.write(cached_response + "\n")
                log(f"💾 Saved cached output to: {output_path}")
                upsert_latest_results(
                    agent_label="Care Ethics Response:",
                    query=query,
                    scenario_id=scenario_id,
                    response_text=cached_response,
                    top_quotes=[],
                )
                return cached_response
            else:
                log("⚠️ Cache hit but cached response is empty; ignoring cache and calling the model.")

    context, top_quotes = retrieve_care_ethics_quotes(query, scenario_id)

    log(f"🛰️ Calling model: {AGENT_MODEL}")
    messages = [
        {"role": "system", "content": "You are the Care Ethics agent in the Ethical Parliament."},
        {"role": "user", "content": f"""You are a care ethics assistant. Your role is to reason from the perspective of care, prioritizing relationships, emotional resonance, and concrete human contexts.

- Your response must be at least 300 words.
- Prioritize relational closeness and interdependence over abstract impartiality.
- Emphasize empathy, responsiveness, and moral attention to the specific people involved.
- Avoid utilitarian calculus or rigid principles unless reframed in terms of care.
- Use the following corpus excerpts where helpful.

Corpus Materials:
{context}

Ethical Question:
{query}

Care Ethics Answer:
"""}
    ]
    print(f"🔧 DEBUG model={AGENT_MODEL} max_tokens={max_tokens}")
    print("🔎 DEBUG Care Agent sending messages:\n", json.dumps(messages, indent=2))

    # --- Call OpenAI with robust extraction + fallback ---
    def _extract_text(obj):
        # Try Responses API style first
        try:
            txt = getattr(obj, "output_text", None)
            if txt:
                return txt.strip()
        except Exception:
            pass
        # Try Chat Completions style
        try:
            ch = getattr(obj, "choices", None)
            if ch and len(ch) > 0:
                # SDK objects: choices[0].message.content; dicts: choices[0]['message']['content']
                first = ch[0]
                content = None
                try:
                    content = first.message.content
                except Exception:
                    try:
                        content = first.get("message", {}).get("content", "")
                    except Exception:
                        content = None
                if content:
                    return str(content).strip()
        except Exception:
            pass
        return ""

    final_response = ""
    try:
        # Prefer Responses API for GPT‑5 family
        if AGENT_MODEL.startswith("gpt-5"):
            prompt_text = (messages[0]["content"] + "\n\n" + messages[1]["content"])
            log("🛰️ Using Responses API path (gpt‑5 family).")
            resp = client.responses.create(
                model=AGENT_MODEL,
                input=prompt_text,
                max_output_tokens=max_tokens,
            )
            log(f"🧾 Raw Responses object type: {type(resp)}")
            final_response = _extract_text(resp)
        else:
            # Non‑gpt‑5 models via Chat Completions
            log("🛰️ Using Chat Completions path (non gpt‑5).")
            resp = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                max_tokens=max_tokens,
            )
            log(f"🧾 Raw Chat object type: {type(resp)}")
            final_response = _extract_text(resp)
    except Exception as e:
        log(f"❌ Primary model call failed: {e}")
        final_response = ""

    if not final_response:
        # Fallback to a small, cheap model known to work with Chat Completions
        FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
        log(f"⚠️ Empty response from {AGENT_MODEL}. Falling back to {FALLBACK_MODEL}.")
        try:
            resp_fb = client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=messages,
                max_tokens=max_tokens,
            )
            log(f"🧾 Raw Fallback Chat object type: {type(resp_fb)}")
            final_response = _extract_text(resp_fb)
        except Exception as e2:
            log(f"❌ Fallback call failed: {e2}")
            final_response = ""
    if not final_response:
        log("⚠️ Model returned an empty response. Will still update artifacts for debugging, but not caching as valid.")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"response_{timestamp}.txt"
        with open(output_path, "w") as f:
            f.write(f"Ethical Question: {query}\n\n")
            f.write("Top Quotes Used:\n")
            f.write(f"Scenario ID: {scenario_id}\n")
            for quote, score in top_quotes:
                f.write(f"- {quote} (score: {score:.2f})\n")
            f.write("\nCare Ethics Response:\n")
            f.write(final_response + "\n")
        log(f"💾 Saved output to: {output_path}")
    except Exception as e:
        log(f"❌ Failed to write output file: {e}")

    LAST_QUERY_PATH.write_text(query.strip())
    if final_response.strip():
        LAST_RESPONSE_PATH.write_text(final_response.strip())
    else:
        log("ℹ️ Skipping cache update because response is empty.")

    upsert_latest_results(
        agent_label="Care Ethics Response:",
        query=query,
        scenario_id=scenario_id,
        response_text=final_response,
        top_quotes=top_quotes,
    )

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
    scenario_files = sorted(SCENARIOS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    if not scenario_files:
        sys.stderr.write("Error: No scenario files found in 'scenarios' directory.\n")
    else:
        scenario_path = scenario_files[0]
        scenario_id = scenario_path.stem
        with open(scenario_path) as sf:
            scenario_file = json.load(sf)
        query = scenario_file.get("ethical_question", "")
        if query:
            print("Care Ethics Response:\n" + respond_to_query(query, scenario_id, scenario_path=scenario_path))
        else:
            sys.stderr.write(f"Error: 'ethical_question' not found in scenario file {scenario_path}\n")
