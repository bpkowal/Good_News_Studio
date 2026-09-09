import json
from pathlib import Path
from global_workspace.source_cache import build_source_cache_key
from datetime import datetime
import os
import atexit
import gc
from global_workspace.framework_retrieval import (
    EvidenceThresholds,
    corpus_fingerprint,
    format_evidence_context,
    load_corpus_passages,
    retrieve_framework_evidence,
)
from global_workspace.retrieval_trace import (
    disabled_retrieval_result, rag_is_disabled, serialize_retrieval_result,
)

LAST_QUERY_PATH = Path("agent_outputs/.last_query_care.txt")
LAST_RESPONSE_PATH = Path("agent_outputs/.last_response_care.txt")

MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CARE_PROMPT_VERSION = "care-relational-comparison-v2"
CARE_CORPUS_DIR = Path("care_ethics_corpus")
CARE_EVIDENCE_THRESHOLDS = EvidenceThresholds(core=0.40, adjacent=0.28)
CARE_IDENTITY_TAGS = {
    "care",
    "care_ethics",
    "relationship",
    "responsibility",
    "dependency",
    "trust",
    "responsiveness",
}
CARE_QUERY_LENS = (
    "Care ethics evidence about relationship, responsibility, dependency, trust, "
    "responsiveness, vulnerability, attentiveness, and moral attention to concrete people."
)
LAST_RETRIEVAL = None

def load_scenario_weights(scenario_id):
    from get_semantic_tag import get_semantic_tag_weights

    print(f"🧠 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    return get_semantic_tag_weights(scenario_id, scenario_dir=Path("scenarios"), corpus_dir=Path("care_ethics_corpus"))

def retrieve_care_ethics_quotes(query: str, scenario_id: str, limit_per_quote: int = 250):
    from langchain_huggingface import HuggingFaceEmbeddings

    tag_weights = load_scenario_weights(scenario_id)
    print(f"🔧 Scenario Tag Weights: {tag_weights}")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    passages = load_corpus_passages(
        CARE_CORPUS_DIR,
        framework="care",
        max_chars=limit_per_quote,
    )
    result = retrieve_framework_evidence(
        passages,
        query=query,
        embedder=embedder,
        query_lens=CARE_QUERY_LENS,
        identity_tags=CARE_IDENTITY_TAGS,
        tag_weights=tag_weights,
        thresholds=CARE_EVIDENCE_THRESHOLDS,
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
        print("⚠️ No Care corpus evidence met the adjacent threshold.")

    del embedder
    gc.collect()

    return format_evidence_context(result), [
        (item.passage.text, item.final_score) for item in result.evidence
    ]

def respond_to_query(query: str, scenario_id: str, scenario_path=None, temperature: float = 0.7, max_tokens: int = 300, llm=None) -> str:
    global LAST_RETRIEVAL
        # fallback if scenario_path not provided
    
    if scenario_path is None:
        scenario_path = Path(f"scenarios/{scenario_id}.json")
    
    if not query or not scenario_id:
        raise ValueError("Both 'query' and 'scenario_id' must be provided.")

    cache_key = build_source_cache_key(
        CARE_PROMPT_VERSION,
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
            print("⚡ Skipping LLM call — using cached care ethics response.")
            return LAST_RESPONSE_PATH.read_text().strip()

    if rag_is_disabled():
        retrieval = disabled_retrieval_result(query, CARE_QUERY_LENS)
        LAST_RETRIEVAL = serialize_retrieval_result(retrieval, mode="disabled")
        context = format_evidence_context(retrieval)
        top_quotes = []
    else:
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

    prompt = f"""<s>[INST] You are a care ethics assistant. Your role is to reason from the perspective of care, prioritizing relationships, emotional resonance, and concrete human contexts.

- Treat CORE evidence as the only corpus material that may ground a decisive care-ethical claim.
- Treat ADJACENT evidence as interpretive context only; it may illustrate a care theme,
  but it may not establish a new dependency, trust relation, or vulnerability claim on
  its own.
- Prioritize relational closeness and interdependence over abstract impartiality.
- Emphasize empathy, responsiveness, and moral attention to the specific people involved.
- Avoid utilitarian calculus or rigid principles unless reframed in terms of care.
- Compare every listed action through the same relational dimensions: direct
  entrustment, dependency, trust, agent-created vulnerability, responsibility,
  and responsiveness. Do not attach one care concept to whichever action first
  feels salient without testing how it applies to the rival action.
- Do not upgrade institutional responsibility, prioritization, or dependency into
  an explicit promise unless the scenario itself states a promise or commitment.
  Speak in terms of care, trust, and entrustment when those are what the scenario
  actually gives you.
- Explicitly state the role of numerical magnitude as exactly one of DECISIVE,
  SECONDARY, or IRRELEVANT. Counts may inform competent and responsive care, but
  they are DECISIVE only when you explain why the competing relational claims are
  otherwise comparable. Never use "more lives" as a complete care-ethical reason.
- If independent care commitments favor different actions and care ethics does
  not clearly rank those commitments, say "Care-Ethics Status:
  NORMATIVELY_CONTESTED". You may give a provisional recommendation, but do not
  disguise the unresolved relational priority as a settled framework commitment.
- Do not claim a relational advantage unless you can point to a concrete action-
  specific fact that changes responsibility, trust, dependency, or attentiveness.
- If both actions carry the same relational burden, say the care comparison is
  unresolved rather than forcing a cleaner answer from a merely adjacent quote.
- Use the following corpus excerpts where helpful.

Corpus Materials:
{context}

Ethical Question:
{query}

Care Ethics Answer:
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
        f.write("\nCare Ethics Response:\n")
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
            print("🧡 Care Ethics Response:\n", respond_to_query(query, scenario_id, scenario_path=scenario_path))
        else:
            print(f"Error: 'ethical_question' not found in scenario file {scenario_path}")
