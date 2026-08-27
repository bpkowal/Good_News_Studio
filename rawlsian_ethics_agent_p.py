import json
from pathlib import Path
from global_workspace.source_cache import build_source_cache_key
from datetime import datetime
import os
from get_semantic_tag import get_semantic_tag_weights
import atexit
import gc
from global_workspace.framework_retrieval import (
    EvidenceThresholds,
    corpus_fingerprint,
    format_evidence_context,
    load_corpus_passages,
    retrieve_framework_evidence,
)

LAST_QUERY_PATH = Path("agent_outputs/.last_query_rawlsian.txt")
LAST_RESPONSE_PATH = Path("agent_outputs/.last_response_rawlsian.txt")

MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
RAWLS_PROMPT_VERSION = "rawls-comparative-position-v2"
RAWLS_CORPUS_DIR = Path("rawlsian_ethics_corpus")
RAWLS_EVIDENCE_THRESHOLDS = EvidenceThresholds(core=0.40, adjacent=0.28)
RAWLS_IDENTITY_TAGS = {
    "rawlsian",
    "rawls",
    "justice",
    "fairness",
    "original_position",
    "veil_of_ignorance",
    "least_advantaged",
}
RAWLS_QUERY_LENS = (
    "Rawlsian evidence about justice as fairness, the original position, the veil of "
    "ignorance, basic liberties, fair equality of opportunity, primary goods, and "
    "the least advantaged."
)

class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def load_scenario_weights(scenario_id):
    print(f"🧠 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    return get_semantic_tag_weights(scenario_id, scenario_dir=Path("scenarios"), corpus_dir=Path("rawlsian_ethics_corpus"))

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

def retrieve_rawlsian_ethics_quotes(query: str, scenario_id: str, limit_per_quote: int = 250):
    from langchain_huggingface import HuggingFaceEmbeddings

    tag_weights = load_scenario_weights(scenario_id)
    print(f"🔧 Scenario Tag Weights: {tag_weights}")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    passages = load_corpus_passages(
        RAWLS_CORPUS_DIR,
        framework="rawlsian",
        max_chars=limit_per_quote,
    )
    result = retrieve_framework_evidence(
        passages,
        query=query,
        embedder=embedder,
        query_lens=RAWLS_QUERY_LENS,
        identity_tags=RAWLS_IDENTITY_TAGS,
        tag_weights=tag_weights,
        thresholds=RAWLS_EVIDENCE_THRESHOLDS,
        limit=3,
        prefer_direct_quotes=True,
        separate_identity_scoring=True,
    )

    for item in result.evidence:
        print(
            f"📘 {item.tier.value} evidence (framework={item.framework_score:.2f}, "
            f"case={item.case_score:.2f}, rank={item.final_score:.2f}): "
            f"{item.passage.text}"
        )
    if not result.evidence:
        print("⚠️ No Rawlsian corpus evidence met the adjacent threshold.")

    del embedder
    gc.collect()

    return format_evidence_context(result), [
        (item.passage.text, item.final_score) for item in result.evidence
    ]

def respond_to_query(query: str, scenario_id: str, scenario_path=None, temperature: float = 0.7, max_tokens: int = 300, llm=None) -> str:
        # fallback if scenario_path not provided
    
    if scenario_path is None:
        scenario_path = Path(f"scenarios/{scenario_id}.json")
    
    if not query or not scenario_id:
        raise ValueError("Both 'query' and 'scenario_id' must be provided.")

    cache_key = build_source_cache_key(
        RAWLS_PROMPT_VERSION,
        query,
        scenario_id,
        scenario_path,
    )
    reuse_source = (
        os.getenv("ETHICS_LLM_BACKEND", "local") == "local"
        or os.getenv("ETHICS_REUSE_SOURCE_TESTIMONY", "1") != "0"
    )
    if reuse_source and LAST_QUERY_PATH.exists() and LAST_RESPONSE_PATH.exists():
        last_query = LAST_QUERY_PATH.read_text().strip()
        if cache_key == last_query:
            print("⚡ Skipping LLM call — using cached rawlsian ethics response.")
            return LAST_RESPONSE_PATH.read_text().strip()

    context, top_quotes = retrieve_rawlsian_ethics_quotes(query, scenario_id)

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

    prompt = f"""<s>[INST] You are a Rawlsian ethics assistant. Your role is to reason from the perspective of justice as fairness, emphasizing principles of equality, the original position, and the veil of ignorance.

- Treat CORE evidence as the only corpus material that may ground a decisive Rawlsian claim.
- Treat ADJACENT evidence as interpretive context only; it may clarify a comparison, but
  it may not establish who the least advantaged are, create a new liberty difference,
  or override a committed ranking basis.
- Compare EVERY listed action using its exact action label.
- For each action, identify the least-advantaged affected group, its position
  relative to the rival action, and the relevant basic liberty or primary good.
- Distinguish equal basic liberties, fair equality of opportunity, and the
  difference principle instead of treating every benefit as interchangeable.
- Do not claim that an action protects the least advantaged unless a stated fact
  makes that group better off, prevents it becoming worse off, or preserves a
  basic liberty that the rival action impairs.
- State whether numerical magnitude is DECISIVE, SECONDARY, or IRRELEVANT.
  Numbers are decisive only when they compare the position of the least
  advantaged under otherwise compatible basic liberties and institutions; they
  are not a license for aggregate welfare maximization.
- If Rawlsian principles favor different actions and their priority is unresolved,
  say "Rawlsian Status: NORMATIVELY_CONTESTED" and identify the conflict. A
  provisional recommendation is allowed, but it is not a frozen commitment.
- Do not say an action protects the least advantaged unless you can name the
  concrete stated fact that makes that group better off, less badly off, or
  better protected in liberty, opportunity, or primary goods.
- Do not relabel an unresolved liberty comparison as BASIC_INTEREST_SECURITY
  unless the scenario states a concrete security or survival burden that is
  worse for one action than the rival. If you cannot name the comparative fact
  that makes one group better off or less badly off, keep the basis unresolved.
- Keep the full comparison compact enough to fit within 180 words.
- Use the following corpus excerpts where helpful.

Corpus Materials:
{context}

Ethical Question:
{query}

Rawlsian Ethics Answer:
[/INST]
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

    final_response += "\n[/INST]\n</s>"

    os.makedirs("agent_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"agent_outputs/response_{timestamp}.txt")

    with open(output_path, "w") as f:
        f.write(f"Ethical Question: {query}\n\n")
        f.write("Top Quotes Used:\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        for quote, score in top_quotes:
            f.write(f"- {quote} (score: {score:.2f})\n")
        f.write("\nRawlsian Ethics Response:\n")
        f.write(final_response + "\n")

    LAST_QUERY_PATH.write_text(cache_key)
    LAST_RESPONSE_PATH.write_text(final_response.strip())

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
            print("🧡 Rawlsian Ethics Response:\n", respond_to_query(query, scenario_id, scenario_path=scenario_path))
        else:
            print(f"Error: 'ethical_question' not found in scenario file {scenario_path}")
