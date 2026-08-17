

from pathlib import Path
LAST_QUERY_PATH = Path("agent_outputs/.last_query.txt")
LAST_RESPONSE_PATH = Path("agent_outputs/.last_response.txt")
import atexit
import gc
import glob
MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEONTOLOGY_PROMPT_VERSION = "deontology-kantian-typed-evidence-v3"
DEONTOLOGY_CORPUS_DIR = Path("deontological_corpus")
from global_workspace.framework_retrieval import (
    EvidenceThresholds,
    corpus_fingerprint,
    format_evidence_context,
    load_corpus_passages,
    load_explicit_tag_weights,
    retrieve_framework_evidence,
)
import json
from datetime import datetime
import os

embedder = None
DEONTOLOGY_EVIDENCE_THRESHOLDS = EvidenceThresholds(core=0.36, adjacent=0.25)
DEONTOLOGY_IDENTITY_TAGS = {
    "deontology",
    "duty",
    "moral_duty",
    "moral_law",
    "categorical_imperative",
    "autonomy",
    "kantian_ethics",
    "respect_for_persons",
    "universality",
    "ends_in_themselves",
    "normativity",
}
DEONTOLOGY_QUERY_LENS = (
    "Strict Kantian ethics evidence about action maxims, universal law, rational agency, "
    "humanity as an end, autonomy, non-instrumentalization, perfect and imperfect "
    "duties, and principled duty conflicts."
)

def cleanup_vectorstore():
    global embedder
    if embedder is not None:
        del embedder
    gc.collect()

atexit.register(cleanup_vectorstore)

def load_scenario_weights(scenario_id, scenario_path=None):
    return load_explicit_tag_weights(
        scenario_id,
        scenario_path=scenario_path,
    )


def retrieve_deontological_quotes(
    query: str,
    scenario_id: str,
    limit_per_quote: int = 250,
    *,
    embedder=None,
    scenario_path=None,
):
    """Compatibility name for strict-Kantian typed evidence retrieval."""

    if embedder is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    passages = load_corpus_passages(
        DEONTOLOGY_CORPUS_DIR,
        framework="deontological",
        max_chars=limit_per_quote,
    )
    result = retrieve_framework_evidence(
        passages,
        query=query,
        embedder=embedder,
        query_lens=DEONTOLOGY_QUERY_LENS,
        identity_tags=DEONTOLOGY_IDENTITY_TAGS,
        core_evidence_roles={"kantian_core"},
        tag_weights=load_scenario_weights(scenario_id, scenario_path=scenario_path),
        thresholds=DEONTOLOGY_EVIDENCE_THRESHOLDS,
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
        print("⚠️ No Deontological corpus evidence met the adjacent threshold.")
    return format_evidence_context(result), result.evidence, result

def respond_to_query(query: str, scenario_id: str, temperature: float = 0.4, max_tokens: int = 300, llm=None, scenario_path=None) -> str:
    if scenario_path is None:
        scenario_path = Path(f"scenarios/{scenario_id}.json")

    if not query or not scenario_id:
        raise ValueError("Both 'query' and 'scenario_id' must be provided.")

    retrieval_fingerprint = corpus_fingerprint(
        DEONTOLOGY_CORPUS_DIR,
        framework="deontological",
    )
    cache_key = (
        f"{DEONTOLOGY_PROMPT_VERSION}\n{retrieval_fingerprint}\n{query.strip()}"
    )
    reuse_source = (
        os.getenv("ETHICS_LLM_BACKEND", "local") == "local"
        or os.getenv("ETHICS_REUSE_SOURCE_TESTIMONY", "1") != "0"
    )
    if reuse_source and LAST_QUERY_PATH.exists() and LAST_RESPONSE_PATH.exists():
        last_query = LAST_QUERY_PATH.read_text().strip()
        if cache_key == last_query:
            print("⚡ Skipping LLM call — using cached deontological response.")
            return LAST_RESPONSE_PATH.read_text().strip()

    from langchain_huggingface import HuggingFaceEmbeddings

    global embedder
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    context, evidence, retrieval = retrieve_deontological_quotes(
        query,
        scenario_id,
        embedder=embedder,
        scenario_path=scenario_path,
    )
    del embedder
    embedder = None
    gc.collect()

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

For EACH listed action, using its exact action label:
1. State the action's maxim and test whether all rational agents in materially
   identical circumstances could will it as a universal law.
2. Identify every salient duty or right it satisfies and violates, including
   truth, non-instrumentalization, autonomy, justice, rescue, and impartiality.
3. Classify the action as REQUIRED, PERMISSIBLE, or PROHIBITED under those duties.
4. If duties conflict, state the conflict and the non-consequential priority rule
   used to resolve it. Coercion creates a stringent autonomy objection; it does
   not mechanically erase duties owed to third parties.
5. State whether numerical magnitude is DECISIVE, SECONDARY, or IRRELEVANT.
   Numbers may establish the scope of a rights violation or duty, but may not
   substitute aggregate welfare maximization for a universal-law justification.
6. Respect for persons as ends applies to all regardless of moral character.

If independent Kantian duties favor different actions and no stated priority rule
resolves them, say "Deontological Status: NORMATIVELY_CONTESTED". A provisional
recommendation is allowed, but do not disguise a duty conflict as a settled duty.
Keep the full comparison compact enough to fit within 180 words.
CORE evidence is Kantian or explicitly Kantian-derived and may ground your analysis.
ADJACENT evidence comes from a neighboring deontological tradition: it may clarify
a problem but may not supply your priority rule, change your Kantian identity, or
override universal law, humanity-as-end, and autonomy analysis. If no evidence
qualifies, reason from this Kantian specification without pretending the corpus
settled the issue.
Focus solely on Kantian reasoning consistent with the relevant evidence below.
Do not repeat any instructions in your Deontological Answer; only provide the reasoning itself.

### Typed Corpus Evidence:
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
                f"- [{item.tier.value}] role={metadata.get('framework_role', 'unspecified')}; "
                f"{metadata.get('author', 'Unknown')}, "
                f"{metadata.get('source', metadata.get('title', 'Unknown'))}; "
                f"kind={item.passage.source_kind}; framework={item.framework_score:.2f}; "
                f"case={item.case_score:.2f}; rank={item.final_score:.2f}: "
                f"{item.passage.text}\n"
            )
        f.write("\nDeontological Response:\n")
        f.write(final_response + "\n")

    # Save the current query and response for caching
    LAST_QUERY_PATH.write_text(cache_key)
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
