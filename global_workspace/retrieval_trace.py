"""Serialize framework retrieval into workspace traces and score quote use."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Mapping, Sequence

from global_workspace.framework_retrieval import RETRIEVAL_VERSION, RetrievalResult


RETRIEVAL_MARKER = "WORKSPACE_RETRIEVAL_B64="
_WORD = re.compile(r"[a-z0-9]{4,}")


def rag_is_disabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("ETHICS_DISABLE_RAG", "")).strip().lower() in {"1", "true", "yes"}


def rag_cache_token(*, disabled: bool | None = None) -> str:
    if disabled is None:
        disabled = rag_is_disabled()
    return "rag=off" if disabled else "rag=on"


def disabled_retrieval_result(query: str, query_lens: str) -> RetrievalResult:
    """The authoritative typed RAG-off result used by bridge and source agents."""
    return RetrievalResult(
        (),
        0,
        0,
        None,
        f"{query_lens}\nCase: {query}".strip(),
    )


def serialize_retrieval_result(
    result: RetrievalResult | None,
    *,
    mode: str = "retrieved",
) -> dict[str, Any]:
    if result is None:
        return {
            "mode": mode,
            "retrieval_version": RETRIEVAL_VERSION,
            "expanded_query": "",
            "candidate_count": 0,
            "rejected_count": 0,
            "top_rejected_score": None,
            "has_core_evidence": False,
            "evidence": [],
        }
    evidence = []
    for item in result.evidence:
        metadata = item.passage.metadata
        evidence.append(
            {
                "passage_id": item.passage.passage_id,
                "document_id": item.passage.document_id,
                "source_file": item.passage.source_file,
                "source_kind": item.passage.source_kind,
                "tier": item.tier.value,
                "semantic_score": round(float(item.semantic_score), 4),
                "framework_score": round(float(item.framework_score), 4),
                "case_score": round(float(item.case_score), 4),
                "tag_boost": round(float(item.tag_boost), 4),
                "final_score": round(float(item.final_score), 4),
                "tags": list(item.passage.tags),
                "text": item.passage.text,
                "framework_role": metadata.get("framework_role"),
                "author": metadata.get("author"),
                "source": metadata.get("source") or metadata.get("title"),
            }
        )
    return {
        "mode": mode,
        "retrieval_version": RETRIEVAL_VERSION,
        "expanded_query": result.expanded_query,
        "candidate_count": int(result.candidate_count),
        "rejected_count": int(result.rejected_count),
        "top_rejected_score": (
            None
            if result.top_rejected_score is None
            else round(float(result.top_rejected_score), 4)
        ),
        "has_core_evidence": bool(result.has_core_evidence),
        "evidence": evidence,
    }


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(" ".join(str(text or "").casefold().split())))


def quote_coverage(haystack: str, quote: str) -> float:
    """Fraction of content words from the quote that also appear in haystack."""
    needle = _tokens(quote)
    if not needle:
        return 0.0
    return len(needle & _tokens(haystack)) / len(needle)


def annotate_retrieval_use(
    record: Mapping[str, Any],
    *,
    testimony: str = "",
    cycle_text: str = "",
    cited_threshold: float = 0.25,
) -> dict[str, Any]:
    annotated = dict(record)
    evidence = []
    testimony_hits = 0
    cycle_hits = 0
    coverages: list[float] = []
    for item in list(record.get("evidence") or []):
        row = dict(item)
        text = str(row.get("text") or "")
        testimony_coverage = round(quote_coverage(testimony, text), 4)
        cycle_coverage = round(quote_coverage(cycle_text, text), 4)
        cited_in_testimony = testimony_coverage >= cited_threshold
        cited_in_cycles = cycle_coverage >= cited_threshold
        row["testimony_coverage"] = testimony_coverage
        row["cycle_coverage"] = cycle_coverage
        row["cited_in_testimony"] = cited_in_testimony
        row["cited_in_cycles"] = cited_in_cycles
        evidence.append(row)
        coverages.append(testimony_coverage)
        testimony_hits += int(cited_in_testimony)
        cycle_hits += int(cited_in_cycles)
    annotated["evidence"] = evidence
    annotated["testimony_cited_count"] = testimony_hits
    annotated["cycle_cited_count"] = cycle_hits
    annotated["mean_testimony_coverage"] = (
        round(sum(coverages) / len(coverages), 4) if coverages else 0.0
    )
    return annotated


def skipped_retrieval_record(reason: str) -> dict[str, Any]:
    return serialize_retrieval_result(None, mode=reason)


def specialist_cycle_text(result: Mapping[str, Any], specialist: str) -> str:
    chunks: list[str] = []
    for cycle in result.get("cycles") or []:
        for candidate in cycle.get("candidates") or []:
            if str(candidate.get("specialist") or "") != specialist:
                continue
            for key in (
                "rationale",
                "change_justification",
                "constraint",
                "unresolved",
                "recommended_action",
            ):
                value = candidate.get(key)
                if value:
                    chunks.append(str(value))
    return "\n".join(chunks)


def specialist_candidate_cycle_text(
    cycles: Iterable[Any], specialist: str,
) -> str:
    """Collect retrieval-comparison text without serializing a whole result.

    This deliberately projects only the five fields consumed by retrieval-use
    annotation. It accepts live ``WorkspaceCycle``/``CandidateChunk`` objects,
    so finalization no longer constructs a duplicate full audit dictionary.
    """
    chunks: list[str] = []
    for cycle in cycles:
        for candidate in getattr(cycle, "candidates", ()):
            if str(getattr(candidate, "specialist", "")) != specialist:
                continue
            for field in (
                "rationale",
                "change_justification",
                "constraint",
                "unresolved",
                "recommended_action",
            ):
                value = getattr(candidate, field, "")
                if value:
                    chunks.append(str(value))
    return "\n".join(chunks)
