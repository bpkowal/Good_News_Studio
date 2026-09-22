"""Source-derived obligations supplied to initial world-model generation.

This module does not infer a world. It exposes conservative facts that the
admission validators will later require the candidate to preserve, so the
generator can build toward the same contract on its first attempt.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .world_state import (
    classify_clause_role,
    explicit_likelihood_spans,
    explicit_quantity_spans,
    explicit_scope_spans,
    explicit_temporal_spans,
    extract_quantity_bearing_consequences,
    source_allocation_constraint_signals,
)
from .ellipsis_integrity import (
    build_ethical_context_profile,
    contextualize_ellipsis_obligations,
    detect_ellipsis_obligations,
)

_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "if", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "then", "this", "to", "will", "with", "without", "where", "which",
}
_NEGATION = re.compile(
    r"\b(?:not|never|no\s+longer|without|refrain(?:s|ed|ing)?\s+from|"
    r"does\s+not|do\s+not|is\s+not|are\s+not)\b",
    re.IGNORECASE,
)
_MODAL = re.compile(
    r"\b(?:might|may|could|possibly|probably|likely|unlikely|certainly|"
    r"guaranteed?|exactly|at\s+risk)\b",
    re.IGNORECASE,
)
_CONDITION = re.compile(r"\b(?:if|unless|only\s+if|provided\s+that|except\s+when)\b", re.I)
_CAUSAL = re.compile(
    r"\b(?:because|caus(?:e|es|ed|ing)|lead(?:s|ing)?\s+to|result(?:s|ing)?\s+in|"
    r"enable(?:s|d|ing)?|prevent(?:s|ed|ing)?|move(?:s|d|ing)?|divert(?:s|ed|ing)?|"
    r"leave(?:s|ing)?|remain(?:s|ed|ing)?|stay(?:s|ed|ing)?|shut(?:s|ting)?\s+down)\b",
    re.IGNORECASE,
)
_ELLIPSIS = re.compile(
    r"\b(?:so\s+(?:does|did|do|will|would|is|was|were)|does\s+too|did\s+too|"
    r"the\s+same\s+(?:thing|action|outcome|result)|likewise)\b",
    re.IGNORECASE,
)
_SUBJECT_LIKELIHOOD = re.compile(
    r"(?:^|\bwhile\b|[,;])\s*(?P<subject>[^,;]{1,80}?)\s+has\s+"
    r"(?:an?\s+)?(?P<qualifier>(?:\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s+percent)"
    r"\s+(?:chance|risk|probability|likelihood))\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    return {
        token for token in _WORD.findall(str(text or "").casefold())
        if len(token) > 2 and token not in _STOP
    }


def _action_candidates(text: str, actions: Mapping[str, str]) -> list[str]:
    clause_tokens = _tokens(text)
    scored = []
    for action_id, action in actions.items():
        action_tokens = _tokens(action)
        overlap = len(clause_tokens & action_tokens)
        if overlap:
            scored.append((overlap, action_id))
    if not scored:
        return []
    best = max(score for score, _action_id in scored)
    # Preserve both sides of an explicit either/or clause rather than forcing
    # the entire clause onto whichever action has one extra lexical token.
    if re.search(r"\b(?:either|or|versus|rather\s+than)\b", text, re.I):
        return [action_id for _score, action_id in sorted(scored)]
    return [action_id for score, action_id in sorted(scored) if score == best]


def _atomic_spans(text: str) -> list[str]:
    spans = [
        " ".join(span.strip(" ,;:").split())
        for span in re.split(
            r"(?<=[,;:])\s+|\s+\b(?:and\s+then|thereby|which)\b\s+",
            str(text or ""), flags=re.IGNORECASE,
        )
    ]
    return [span for span in spans if len(_WORD.findall(span)) >= 3]


def build_candidate_generation_contract(
    clauses: Sequence[Mapping[str, Any]],
    actions: Mapping[str, str],
) -> dict[str, Any]:
    """Compile a conservative, typed checklist from source text."""
    rows: list[dict[str, Any]] = []
    quantity_obligations: list[dict[str, Any]] = []
    likelihood_binding_obligations: list[dict[str, Any]] = []
    ellipsis_obligations: list[dict[str, Any]] = []
    previous_fact_clause_id = ""
    for clause in clauses:
        clause_id = str(clause.get("clause_id") or "")
        text = " ".join(str(clause.get("text") or "").split())
        consequences = extract_quantity_bearing_consequences((text,))
        row = {
            "clause_id": clause_id,
            "role": classify_clause_role(text),
            "action_candidates": _action_candidates(text, actions),
            "quantities": list(explicit_quantity_spans(text)),
            "likelihood_qualifiers": list(explicit_likelihood_spans(text)),
            "scope_qualifiers": list(explicit_scope_spans(text)),
            "temporal_qualifiers": list(explicit_temporal_spans(text)),
            "negation_cues": list(dict.fromkeys(match.group(0) for match in _NEGATION.finditer(text))),
            "modality_cues": list(dict.fromkeys(match.group(0) for match in _MODAL.finditer(text))),
            "condition_cues": list(dict.fromkeys(match.group(0) for match in _CONDITION.finditer(text))),
            "causal_cues": list(dict.fromkeys(match.group(0) for match in _CAUSAL.finditer(text))),
            "atomic_source_spans": _atomic_spans(text),
            "discourse_continuation_of": (
                previous_fact_clause_id
                if previous_fact_clause_id
                and (
                    bool(text[:1].islower())
                    or bool(re.match(
                        r"^(?:it|they|he|she|this|that|these|those|the\s+same)\b",
                        text, re.IGNORECASE,
                    ))
                )
                else ""
            ),
        }
        rows.append(row)
        if row["role"] in {"FACT", "MIXED_FACT_COMPARISON"}:
            previous_fact_clause_id = clause_id
        for item in consequences:
            quantity_obligations.append({
                "clause_id": clause_id,
                "action_candidates": row["action_candidates"],
                "consequence_span": item.consequence_span,
                "quantity_spans": list(item.quantity_spans),
                "polarity": item.polarity,
                "side_cue": item.side_cue,
            })
        ellipsis = list(dict.fromkeys(match.group(0) for match in _ELLIPSIS.finditer(text)))
        if ellipsis:
            ellipsis_obligations.append({
                "clause_id": clause_id,
                "cues": ellipsis,
                "instruction": (
                    "Resolve the omitted predicate from the nearest compatible "
                    "antecedent, but preserve this exact clause as provenance."
                ),
            })
        for match in _SUBJECT_LIKELIHOOD.finditer(text):
            subject = " ".join(match.group("subject").split()).strip(" ,;:")
            subject = re.sub(r"^while\s+", "", subject, flags=re.IGNORECASE)
            qualifier = " ".join(match.group("qualifier").split())
            subject_actions = [
                action_id for action_id, action in actions.items()
                if subject.casefold() in str(action).casefold()
            ]
            likelihood_binding_obligations.append({
                "clause_id": clause_id,
                "subject_span": subject,
                "qualifier_span": qualifier,
                "source_span": " ".join(match.group(0).strip(" ,;").split()),
                "action_candidates": subject_actions or row["action_candidates"],
            })
    typed_ellipsis = contextualize_ellipsis_obligations(
        detect_ellipsis_obligations(clauses), clauses, actions,
    )
    return {
        "contract_version": "1.2",
        "policy": (
            "Coverage checklist only: preserve these source features; do not "
            "treat them as permission to invent an effect or action binding."
        ),
        "actions": dict(actions),
        "allocation_constraint_profile": {
            **source_allocation_constraint_signals([
                str(clause.get("text") or "") for clause in clauses
            ]),
            "status": "CANDIDATE_NOT_PROVENANCE",
            "instruction": (
                "Use these source signals only to test whether competing "
                "RESOURCE_TRANSFER actions share one constrained resource. "
                "Commit an exclusive-allocation complement only after the "
                "typed actions identify distinct recipients and either an "
                "explicit constraint or shared-resource choice evidence."
            ),
        },
        "ethical_context_profile": build_ethical_context_profile(clauses, actions),
        "clauses": rows,
        "quantity_consequence_obligations": quantity_obligations,
        "likelihood_binding_obligations": likelihood_binding_obligations,
        "ellipsis_obligations": typed_ellipsis or ellipsis_obligations,
    }
