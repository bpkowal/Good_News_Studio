import glob
from pathlib import Path
from datetime import datetime
import json
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import atexit, gc

# ---- Caching paths (mirrors Rawls agent) ----
LAST_QUERY_PATH = Path("agent_outputs/.last_query_nozick.txt")
LAST_RESPONSE_PATH = Path("agent_outputs/.last_response_nozick.txt")

# ---- Env + client ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-5-mini")
FALLBACK_MODEL = os.getenv("OPENAI_AGENT_FALLBACK_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Optional: semantic tag expansion (reuse your helper) ----
from get_semantic_tag import get_semantic_tag_weights

# ---- Optional corpus: load a liberty/Nozick corpus if you have one ----
# Expected to expose `load_nozick_ethics_corpus()` returning a vectorstore like the Rawlsian loader.
def _try_load_nozick_corpus():
    try:
        from load_nozick_ethics_corpus import load_nozick_ethics_corpus
        return load_nozick_ethics_corpus()
    except Exception:
        return None

def _cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def _normalize_tags(raw_tags):
    if isinstance(raw_tags, str):
        return [t.strip() for t in raw_tags.split(",") if t.strip()]
    elif isinstance(raw_tags, list):
        return [t.strip() for t in raw_tags if isinstance(t, str) and t.strip()]
    return []

class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def load_scenario_weights(scenario_id):
    print(f"🧠 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    # Point to a liberty/nozick corpus dir if you keep per-agent tags there; fallback to generic.
    corpus_dir = Path("nozick_ethics_corpus") if Path("nozick_ethics_corpus").exists() else Path("corpus")
    return get_semantic_tag_weights(scenario_id, scenario_dir=Path("scenarios"), corpus_dir=corpus_dir)

def retrieve_nozick_quotes(query: str, scenario_id: str, limit_per_quote: int = 250):
    """
    If a Nozick/liberty vectorstore exists, use it; otherwise, gracefully return empty context.
    """
    vectorstore = _try_load_nozick_corpus()
    if vectorstore is None:
        print("ℹ️ No Nozick corpus found. Proceeding without quotes context.")
        return "", []

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"⚠️ Embedding backend unavailable ({e}); proceeding corpus-only.")
        embedder = None

    tag_weights = load_scenario_weights(scenario_id)
    print(f"🔧 Scenario Tag Weights: {tag_weights}")
    raw_docs = vectorstore.similarity_search(query, k=10)
    wrapped_docs = [Document(d.page_content, d.metadata) for d in raw_docs]

    def doc_score(doc):
        tags = _normalize_tags(doc.metadata.get("tags", []))
        score = 0.0
        for tag in tags:
            w = tag_weights.get(tag, 0.0)
            if w > 0:
                print(f"🔍 Tag '{tag}' has semantic weight {w}")
            score += w
        print(f"🧪 Tags: {tags} → Score: {score:.2f}")
        return score

    doc_scores = {id(doc): doc_score(doc) for doc in wrapped_docs}

    quotes = []
    for doc in wrapped_docs:
        for line in doc.content.split("\n"):
            if line.strip().startswith(">"):
                quote = line.strip()[1:].strip()[:limit_per_quote]
                if quote:
                    quotes.append((quote, doc_scores[id(doc)]))

    if not quotes or embedder is None:
        # Cleanup
        try: del vectorstore
        except: pass
        if not quotes:
            return "", []
        # else we have quotes but no embedder; just return top by tag score
        quotes = sorted(quotes, key=lambda x: x[1], reverse=True)[:3]
        return "\n---\n".join([q for q,_ in quotes]), quotes

    # Rank by tags + semantic similarity
    query_emb = embedder.embed_query(query)
    quote_texts = [q for q,_ in quotes]
    quote_embs = embedder.embed_documents(quote_texts)
    ranked = []
    for i, (qt, _) in enumerate(quotes):
        sim = max(0.0, _cosine_similarity(query_emb, quote_embs[i]))
        tag_score = quotes[i][1]
        combined = tag_score * 2 + 0.2 * sim
        ranked.append((qt, combined))
    ranked.sort(key=lambda x: x[1], reverse=True)
    top = ranked[:3]

    # Cleanup
    del quote_embs, quote_texts, query_emb, wrapped_docs, raw_docs, doc_scores
    try: del vectorstore
    except: pass
    gc.collect()

    return "\n---\n".join([q for q,_ in top]), top

# ---- LLM call with fallback (mirrors your Rawls agent) ----
def _to_responses_input(messages):
    blocks = []
    for m in messages:
        blocks.append({"role": m.get("role","user"),
                       "content": [{"type":"text", "text": m.get("content","")}]})
    return blocks

def call_with_fallback(messages, primary_model: str, fallback_model: str, max_tokens: int):
    try:
        print(f"🛰️ Trying primary model: {primary_model}")
        if primary_model.startswith("gpt-5"):
            resp = client.responses.create(
                model=primary_model,
                input=_to_responses_input(messages),
                max_output_tokens=max_tokens,
            )
            text = (resp.output_text or "").strip()
        else:
            resp = client.chat.completions.create(
                model=primary_model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
        if text:
            print(f"✅ Primary model returned {len(text)} chars.")
            return text, primary_model
        print("⚠️ Primary model returned empty; trying fallback…")
    except Exception as e:
        print(f"⚠️ Primary model error: {e}; trying fallback…")

    try:
        print(f"🛰️ Trying fallback model: {fallback_model}")
        resp = client.chat.completions.create(
            model=fallback_model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            print(f"✅ Fallback model returned {len(text)} chars.")
            return text, fallback_model
        print("❌ Fallback model also returned empty.")
        return "[No response generated by model]", fallback_model
    except Exception as e:
        print(f"❌ Fallback model error as well: {e}")
        return "[No response generated by model]", fallback_model

# ---- Nozick agent core ----
def respond_to_query(query: str, scenario_id: str, scenario_path=None,
                     temperature: float = 0.5, max_tokens: int = 300, llm=None) -> str:

    if scenario_path is None:
        scenario_path = Path(f"scenarios/{scenario_id}.json")

    if not query or not scenario_id:
        raise ValueError("Both 'query' and 'scenario_id' must be provided.")

    # trivial cache
    if LAST_QUERY_PATH.exists() and LAST_RESPONSE_PATH.exists():
        last_q = LAST_QUERY_PATH.read_text().strip()
        if query.strip() == last_q:
            print("⚡ Skipping LLM call — using cached Nozick response.")
            return LAST_RESPONSE_PATH.read_text().strip()

    context, top_quotes = retrieve_nozick_quotes(query, scenario_id)

    # Prompt: Nozick’s entitlement theory + side-constraints
    prompt = (
        "You are the Nozickian (Liberty) member of the Ethical Parliament. "
        "Reason from side-constraints and entitlement theory: persons have stringent rights not to be used; "
        "justice concerns holdings—legitimate acquisition, voluntary transfer, and rectification of past violations. "
        "Avoid patterned end-state principles; the state is minimal: prevent force, theft, and fraud; enforce contracts; "
        "rectify rights violations.\n\n"
        "Guidelines:\n"
        "- Begin by identifying any rights at stake and whether they are being violated.\n"
        "- Ask: Are current holdings the result of just acquisition/transfer? If not, specify rectification.\n"
        "- Warn against policies that treat people as means or that re-pattern holdings without rectification.\n"
        "- Prefer voluntary, contractual solutions; justify any coercion strictly as rights-protection.\n"
        "- Where helpful, cite and weave in the corpus excerpts below.\n\n"
        f"Corpus excerpts (may be partial):\n{context}\n\n"
        f"Ethical Question:\n{query}\n\n"
        "Nozickian (Liberty) Answer:\n"
    )

    messages = [
        {"role": "system",
         "content": (
            "You are the Nozick/Liberty ethics agent in a multi-agent parliament. "
            "Defend strong individual rights (side-constraints). Apply entitlement theory: acquisition, transfer, rectification. "
            "Oppose patterned redistributions unless they are rectifying rights violations. "
            "Favor voluntary arrangements and minimal state functions (protect rights, enforce contracts). "
            "Be clear, concrete, and rigorous."
         )},
        {"role": "user", "content": prompt},
    ]

    # Debug preview
    try:
        prev = (prompt[:2000] + ("…" if len(prompt) > 2000 else ""))
        print("\n🔎 NOZICK outbound prompt preview:\n" + prev)
    except Exception:
        pass

    final_response, model_used = call_with_fallback(
        messages=messages,
        primary_model=AGENT_MODEL,
        fallback_model=FALLBACK_MODEL,
        max_tokens=max_tokens,
    )

    os.makedirs("agent_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"agent_outputs/nozick_response_{timestamp}.txt")
    with open(output_path, "w") as f:
        f.write(f"Ethical Question: {query}\n\n")
        f.write("Top Quotes Used:\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        for quote, score in top_quotes:
            f.write(f"- {quote} (score: {score:.2f})\n")
        f.write("\nNozickian (Liberty) Response:\n")
        f.write(final_response + "\n")
        f.write(f"\n(Model used: {model_used})\n")

    LAST_QUERY_PATH.write_text(query.strip())
    LAST_RESPONSE_PATH.write_text(final_response.strip())
    print(f"💾 Saved output to: {output_path}")
    return final_response

def cleanup_vectorstore():
    try:
        del vectorstore  # if existed
    except NameError:
        pass
    gc.collect()

atexit.register(cleanup_vectorstore)

if __name__ == "__main__":
    scenario_files = sorted(Path("scenarios").glob("*.json"), key=os.path.getmtime, reverse=True)
    if not scenario_files:
        print("Error: No scenario files found in 'scenarios' directory.")
    else:
        scenario_path = scenario_files[0]
        scenario_id = scenario_path.stem
        with open(scenario_path) as sf:
            scenario_file = json.load(sf)
        query = scenario_file.get("ethical_question", "")
        if query:
            print("Nozickian Ethics Response:\n" + respond_to_query(query, scenario_id, scenario_path=scenario_path))
        else:
            print(f"Error: 'ethical_question' not found in scenario file {scenario_path}")