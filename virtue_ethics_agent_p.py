MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
VIRTUE_PROMPT_VERSION = "virtue-typed-evidence-v3"
import json
from pathlib import Path
from global_workspace.framework_retrieval import (
    EvidenceThresholds,
    corpus_fingerprint,
    format_evidence_context,
    load_explicit_tag_weights,
    load_corpus_passages,
    retrieve_framework_evidence,
)
LAST_QUERY_PATH = Path("agent_outputs/.last_query_virtue.txt")
LAST_RESPONSE_PATH = Path("agent_outputs/.last_response_virtue.txt")
VIRTUE_CORPUS_DIR = Path("virtue_ethics_corpus")
# Calibrated against the current ICU allocation, vehicle, and ordinary-lie cases.
# Below 0.30 the passages were generic enough to add noise; 0.42 separated the
# consistently framework-relevant passages from merely neighboring material.
VIRTUE_EVIDENCE_THRESHOLDS = EvidenceThresholds(core=0.42, adjacent=0.30)
VIRTUE_IDENTITY_TAGS = {
    "virtue_ethics",
    "virtue",
    "character",
    "flourishing",
    "practical_wisdom",
    "golden_mean",
}
VIRTUE_QUERY_LENS = (
    "Virtue ethics evidence about the actor's role, practical wisdom, character, "
    "virtues and vices, moral perception, tragic conflict, habituation, and human flourishing."
)
import glob
from datetime import datetime
import os
import gc

import atexit

def cleanup_vectorstore():
    global embedder
    try:
        del embedder
    except NameError:
        pass
    gc.collect()

atexit.register(cleanup_vectorstore)

def load_scenario_weights(scenario_id, scenario_path=None):
    """Use explicit tags only; do not invent semantic tag authority."""

    return load_explicit_tag_weights(
        scenario_id,
        scenario_path=scenario_path,
    )


def retrieve_virtue_ethics_quotes(
    query: str,
    scenario_id: str,
    limit_per_quote: int = 250,
    *,
    embedder=None,
    scenario_path=None,
):
    """Compatibility name for typed Virtue evidence retrieval."""

    if embedder is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    tag_weights = load_scenario_weights(scenario_id, scenario_path=scenario_path)
    passages = load_corpus_passages(
        VIRTUE_CORPUS_DIR,
        framework="virtue",
        max_chars=limit_per_quote,
    )
    result = retrieve_framework_evidence(
        passages,
        query=query,
        embedder=embedder,
        query_lens=VIRTUE_QUERY_LENS,
        identity_tags=VIRTUE_IDENTITY_TAGS,
        tag_weights=tag_weights,
        thresholds=VIRTUE_EVIDENCE_THRESHOLDS,
        limit=3,
    )
    for item in result.evidence:
        print(
            f"📘 {item.tier.value} evidence (semantic={item.semantic_score:.2f}, "
            f"score={item.final_score:.2f}): {item.passage.text}"
        )
    if not result.evidence:
        print("⚠️ No Virtue corpus evidence met the adjacent threshold.")
    return format_evidence_context(result), result.evidence, result

def respond_to_query(query=None, scenario_id=None, scenario_path=None, temperature: float = 0.7, max_tokens: int = 300, llm=None) -> str:
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
    retrieval_fingerprint = corpus_fingerprint(VIRTUE_CORPUS_DIR, framework="virtue")
    cache_key = (
        f"{VIRTUE_PROMPT_VERSION}\n{retrieval_fingerprint}\n{query.strip()}"
    )
    reuse_source = (
        os.getenv("ETHICS_LLM_BACKEND", "local") == "local"
        or os.getenv("ETHICS_REUSE_SOURCE_TESTIMONY", "1") != "0"
    )
    if reuse_source and LAST_QUERY_PATH.exists() and LAST_RESPONSE_PATH.exists():
        last_query = LAST_QUERY_PATH.read_text().strip()
        if cache_key == last_query:
            print("⚡ Skipping LLM call — using cached virtue ethics response.")
            return LAST_RESPONSE_PATH.read_text().strip()

    # Load RAG only after the source-testimony cache misses.
    from langchain_huggingface import HuggingFaceEmbeddings
    global embedder
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    context, evidence, retrieval = retrieve_virtue_ethics_quotes(
        query,
        scenario_id,
        embedder=embedder,
        scenario_path=scenario_path,
    )

    # Cleanup RAG components
    del embedder
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



    prompt = f"""<s>[INST] You are a virtue ethics assistant. Give a concise answer of at most 180 words from the perspective of virtue ethics, focusing on character, habituation, and human flourishing.
    - Prioritize the development of moral character and virtues over rule-based moral frameworks.
    - Use moral exemplars, narrative analogies, and lived experience as sources of ethical insight.
    - Compare EVERY listed action using its exact action label. For each, identify
      the actor's role, virtues it expresses, vices it risks, and the concrete
      circumstances practical wisdom must notice.
    - State whether numerical magnitude is DECISIVE, SECONDARY, or IRRELEVANT.
      Stakes can matter to practical wisdom, but numbers alone do not constitute
      courage, justice, honesty, compassion, or flourishing.
    - State the central virtue conflict and the practical-wisdom comparison rule.
      If independent virtues favor different actions and phronesis has no grounded
      way to rank them, say "Virtue-Ethics Status: NORMATIVELY_CONTESTED" and make
      any action recommendation explicitly provisional.
    - CORE corpus evidence may ground your analysis. ADJACENT evidence is optional
      interpretive context only: it must not determine your recommendation, override
      scenario facts, or replace virtue-ethical analysis of role, character, and
      practical wisdom. If no evidence qualifies, reason from these framework
      instructions without pretending the corpus supplied support.
    - Do not repeat the scenario or instructions.

    Here are the corpus materials for reference:
    {context}
    Ethical Question:
    {query}
    [/INST]
    """

    output = llm(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
    final_response = output["choices"][0]["text"].strip() + "\n"

    del llm
    gc.collect()

    os.makedirs("agent_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"agent_outputs/response_virtue_{timestamp}.txt")

    with open(output_path, "w") as f:
        f.write(f"Ethical Question: {query}\n\n")
        f.write("Top Corpus Evidence Used:\n")
        f.write(f"Scenario ID: {scenario_id}\n")
        f.write(
            f"Retrieval candidates: {retrieval.candidate_count}; "
            f"rejected: {retrieval.rejected_count}; "
            f"core present: {retrieval.has_core_evidence}\n"
        )
        for item in evidence:
            metadata = item.passage.metadata
            f.write(
                f"- [{item.tier.value}] {metadata.get('author', 'Unknown')}, "
                f"{metadata.get('source', metadata.get('title', 'Unknown'))}; "
                f"kind={item.passage.source_kind}; semantic={item.semantic_score:.2f}; "
                f"score={item.final_score:.2f}: {item.passage.text}\n"
            )
        f.write("\nVirtue Ethics Response:\n")
        f.write(final_response + "\n")

    print(f"💾 Saved output to: {output_path.name}")

    # Save cache for last query/response
    LAST_QUERY_PATH.write_text(cache_key)
    LAST_RESPONSE_PATH.write_text(final_response.strip())

    del context
    del evidence
    del retrieval
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
