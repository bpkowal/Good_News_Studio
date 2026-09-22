"""Conservative binding of generated propositions to exact source spans."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import re
from typing import Any


_WRAPPERS = {'"', "'", "“", "”", "‘", "’"}
_STOP = {"a", "an", "and", "are", "be", "but", "is", "of", "or", "the", "to"}


def _fold(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if token not in _STOP
    }


def _exact_slice(needle: str, haystack: str) -> str | None:
    start = haystack.casefold().find(needle.casefold())
    return haystack[start:start + len(needle)] if start >= 0 else None


def bind_exact_provenance_spans(
    skeleton: Mapping[str, Any],
    clauses: Sequence[Mapping[str, Any]],
    generation_contract: Mapping[str, Any],
    *,
    overlap_threshold: float = 0.65,
    overlap_margin: float = 0.15,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a copied skeleton with exact, auditable proposition bindings."""
    result = copy.deepcopy(dict(skeleton))
    clause_by_id = {
        str(row.get("clause_id") or ""): str(row.get("text") or "")
        for row in clauses
    }
    contract_by_id = {
        str(row.get("clause_id") or ""): row
        for row in generation_contract.get("clauses") or []
        if isinstance(row, Mapping)
    }
    records: list[dict[str, Any]] = []
    for proposition in result.get("propositions") or []:
        if not isinstance(proposition, dict):
            continue
        proposition_id = str(proposition.get("proposition_id") or "")
        original = " ".join(str(proposition.get("source_proposition") or "").split())
        cited = [str(value) for value in proposition.get("clause_ids") or []]
        candidate_source = original
        if (
            len(candidate_source) >= 2
            and candidate_source[0] in _WRAPPERS
            and candidate_source[-1] in _WRAPPERS
        ):
            candidate_source = candidate_source[1:-1].strip()
        exact_matches = [
            (clause_id, exact)
            for clause_id in cited
            if (exact := _exact_slice(candidate_source, clause_by_id.get(clause_id, "")))
        ]
        status = "UNRESOLVED"
        method = None
        bound_span = None
        bound_clause_id = None
        score: float | None = None
        if len(exact_matches) == 1:
            bound_clause_id, bound_span = exact_matches[0]
            status = "BOUND"
            method = "EXACT" if candidate_source == original else "STRIP_WRAPPING_QUOTES"
            score = 1.0
        elif not exact_matches:
            semantic_tokens = _tokens(
                f"{candidate_source} {proposition.get('outcome') or ''}"
            )
            ranked: list[tuple[float, str, str]] = []
            for clause_id in cited:
                for span in contract_by_id.get(clause_id, {}).get(
                    "atomic_source_spans", []
                ):
                    span = str(span)
                    span_tokens = _tokens(span)
                    union = semantic_tokens | span_tokens
                    overlap = (
                        len(semantic_tokens & span_tokens) / len(union)
                        if union else 0.0
                    )
                    ranked.append((overlap, clause_id, span))
            ranked.sort(key=lambda row: row[0], reverse=True)
            if ranked:
                best = ranked[0][0]
                runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
                if best >= overlap_threshold and best - runner_up >= overlap_margin:
                    score, bound_clause_id, proposed = ranked[0]
                    bound_span = _exact_slice(
                        proposed, clause_by_id.get(bound_clause_id, "")
                    )
                    if bound_span is not None:
                        status = "BOUND"
                        method = "UNIQUE_ATOMIC_SPAN_OVERLAP"
        if status == "BOUND" and bound_span is not None:
            proposition["generated_source_proposition"] = original
            proposition["source_proposition"] = bound_span
        proposition["provenance_binding"] = {
            "status": status,
            "method": method,
            "original_span": original,
            "bound_span": bound_span,
            "bound_clause_id": bound_clause_id,
            "score": score,
        }
        records.append({
            "proposition_id": proposition_id,
            **proposition["provenance_binding"],
        })
    result["provenance_bindings"] = records
    return result, records
