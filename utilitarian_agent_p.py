from langchain_huggingface import HuggingFaceEmbeddings
import json
from pathlib import Path
from datetime import datetime
import os
from get_semantic_tag import get_semantic_tag_weights
from pathlib import Path
import gc
import atexit
import glob
from global_workspace.framework_retrieval import (
    EvidenceThresholds,
    corpus_fingerprint,
    format_evidence_context,
    load_corpus_passages,
    retrieve_framework_evidence,
)
from global_workspace.source_cache import build_source_cache_key

from horizon_aggregator import (
    horizon_limited_aggregate,
    exponential_kernel,
    hyperbolic_kernel,
    estimate_horizon_from_tags,
)


MODEL_PATH = "../mistral-7b-instruct-v0.2.Q4_K_M.gguf"
UTILITARIAN_PROMPT_VERSION = "utilitarian-action-consequence-table-v2"
UTILITARIAN_CORPUS_DIR = Path("utilitarian_corpus")
UTILITARIAN_EVIDENCE_THRESHOLDS = EvidenceThresholds(core=0.40, adjacent=0.28)
UTILITARIAN_IDENTITY_TAGS = {
    "utilitarian",
    "utility",
    "consequentialism",
    "welfare",
    "expected_value",
    "aggregate_welfare",
}
UTILITARIAN_QUERY_LENS = (
    "Utilitarian evidence about consequences, welfare, benefits, harms, expected "
    "value, probability, magnitude, duration, reversibility, and impartial aggregation."
)

vectorstore = None
embedder = None

LAST_QUERY_PATH = Path("agent_outputs/.last_query_util.txt")
LAST_RESPONSE_PATH = Path("agent_outputs/.last_response_util.txt")



class Document:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}

def load_scenario_weights(scenario_id):
    print(f"🧠 Expanding tag weights with semantic overlap for scenario: {scenario_id}")
    return get_semantic_tag_weights(scenario_id, scenario_dir=Path("scenarios"), corpus_dir=Path("utilitarian_corpus"))

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
    tag_weights = load_scenario_weights(scenario_id)
    print(f"\U0001f527 Scenario Tag Weights: {tag_weights}")

    global embedder
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    passages = load_corpus_passages(
        UTILITARIAN_CORPUS_DIR,
        framework="utilitarian",
        max_chars=limit_per_quote,
    )
    result = retrieve_framework_evidence(
        passages,
        query=query,
        embedder=embedder,
        query_lens=UTILITARIAN_QUERY_LENS,
        identity_tags=UTILITARIAN_IDENTITY_TAGS,
        tag_weights=tag_weights,
        thresholds=UTILITARIAN_EVIDENCE_THRESHOLDS,
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
        print("⚠️ No Utilitarian corpus evidence met the adjacent threshold.")

    return format_evidence_context(result), [
        (item.passage.text, item.final_score) for item in result.evidence
    ]


# ---------------------------------------------------------------------- #
#  Horizon‑limited utility helper                                        #
# ---------------------------------------------------------------------- #
def _compute_horizon_limited_summary(scenario_tags: list[str],
                                     values: list[float],
                                     distances: list[float]) -> str:
    """
    Given the scenario's temporal tags plus parallel lists of `values`
    and their temporal `distances` (same metric everywhere!), return a
    short natural‑language summary for the LLM prompt.  We use a
    hyperbolic kernel by default.
    """
    H = estimate_horizon_from_tags([t.lower() for t in scenario_tags])
    agg = horizon_limited_aggregate(
        values,
        distances,
        horizon=H,
        kernel=hyperbolic_kernel(scale=0.05),
    )
    return (
        f"Horizon‑limited aggregation (H = {H:.0f} units, hyperbolic scale 0.05) "
        f"yields a net utility of {agg:.2f} for the in‑horizon outcomes."
    )


def respond_to_query(query: str, scenario_id: str, temperature: float = 0.5, max_tokens: int = 300, llm=None, scenario_path=None) -> str:
   
    if scenario_path is None:
        scenario_path = Path(f"scenarios/{scenario_id}.json")
    
    if not query or not scenario_id:
        raise ValueError("Both 'query' and 'scenario_id' must be provided.")

    cache_key = build_source_cache_key(
        UTILITARIAN_PROMPT_VERSION,
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
            print("⚡ Skipping LLM call — using cached utilitarian response.")
            return LAST_RESPONSE_PATH.read_text().strip()

    if llm is None:
        from llama_cpp import Llama
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=768,
            n_threads=6,
            n_gpu_layers=60,
            n_batch=64,
            verbose=False
        )

    with open(scenario_path, "r") as f:
        scenario_file = json.load(f)

    # Attempt to build a horizon‑limited summary if the scenario JSON
    # provides `temporal_tags`, `outcome_values`, and `outcome_distances`.
    horizon_summary = ""
    if isinstance(scenario_file, dict):
        tags  = scenario_file.get("temporal_tags", [])
        vals  = scenario_file.get("outcome_values", [])
        dists = scenario_file.get("outcome_distances", [])
        if tags and vals and dists and len(vals) == len(dists):
            try:
                horizon_summary = _compute_horizon_limited_summary(tags, vals, dists)
                print("🪐 Horizon summary added to prompt.")
            except Exception as e:
                print(f"⚠️ Horizon summary skipped: {e}")

    context, top_quotes = retrieve_utilitarian_quotes(query, scenario_id)

    prompt = f"""
<s>[INST] You are a utilitarian ethics assistant. Your goal is to determine the action best aligned with utilitarian principles, using the corpus excerpts provided.

- Base your decision entirely on consequences.
- Do not assume harm is always wrong—utilitarianism may permit harm if it maximizes net well-being.
- Ignore proximity and immediacy unless they affect outcomes.
- Use the following corpus excerpts in your reasoning. Explicitly reference or paraphrase their logic where applicable.
- First compare the concrete consequences of this particular act for every directly
  affected person. Distinguish facts stated in the scenario from assumptions, and
  identify missing information that could reverse the recommendation.
- Build a compact action-consequence table for EVERY listed action. For each
  material consequence state: affected group or scope, benefit or harm,
  probability or UNKNOWN, magnitude, duration, reversibility, and whether its
  support is STATED, INFERRED, or UNKNOWN. Do not omit the losing action.
- Rank consequences by expected impact: probability multiplied by magnitude.
  Concrete and probable effects should normally outweigh guilt, gratitude, rewards,
  stigma, broad social trust, or other speculative effects unless the scenario gives
  evidence that those effects are likely and material.
- Do not multiply one person's act across millions of hypothetical similar acts.
  Discuss a rule-utilitarian effect only when this act plausibly changes, enforces,
  or publicly instantiates a practice, and explain that causal link. Universalizing
  a choice is not itself a consequence of making the choice once.
- If the result depends on unknown consequences, give a conditional judgment rather
  than using speculative effects to create false certainty.
- A statement that an action maximizes one actor's local payoff does not establish
  that it maximizes aggregate welfare. Likewise, cooperation being mutually
  beneficial does not by itself establish how its total compares with unilateral
  gain plus the other party's loss. Do not import a familiar game matrix or assume
  an unstated inequality between gains and losses.
- When the decisive gain-versus-loss comparison is unspecified, classify the result
  as "Utilitarian Status: UNDERDETERMINED". State the inequality that would select each action, but do not
  turn a typical, ordinary, or textbook assumption into a default recommendation.
- Keep the complete comparison within 180 words.

Corpus Materials:
{context}

{horizon_summary}

Ethical Question:
{query}

Utilitarian Answer:
[/INST]
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

    LAST_QUERY_PATH.write_text(cache_key)
    LAST_RESPONSE_PATH.write_text(final_response.strip())

    print(f"💾 Saved output to: {output_path.name}")
    return final_response

def cleanup_vectorstore():
    global vectorstore, embedder
    vectorstore = None
    embedder = None
    gc.collect()

atexit.register(cleanup_vectorstore)

if __name__ == "__main__":
    scenario_files = sorted(Path("scenarios").glob("*.json"), key=os.path.getmtime, reverse=True)
    if not scenario_files:
        raise FileNotFoundError("No scenario files found in 'scenarios' directory.")
    latest_scenario_path = scenario_files[0]
    scenario_id = latest_scenario_path.stem
    with open(latest_scenario_path, "r") as f:
        scenario_file = json.load(f)
    query = scenario_file["ethical_question"]
    print("🧠 Utilitarian Response:\n", respond_to_query(query, scenario_id))
