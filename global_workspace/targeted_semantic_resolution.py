"""Additive, route-specific semantic resolution after neutral extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Any


ROUTE_INSTRUCTIONS = {
    "SHARED_ARGUMENT_CONJUNCTION": (
        "Determine whether a conjunct inherits a subject, object, or predicate and "
        "therefore expresses a missing distinct atomic proposition."
    ),
    "ELLIPSIS_OR_LIGHT_VERB": (
        "Resolve an omitted predicate or arguments only when one antecedent is "
        "uniquely source-licensed; otherwise quarantine the span."
    ),
    "CONDITIONAL_SCOPE": (
        "Determine whether existing propositions preserve antecedent/consequent "
        "scope. Add only a missing atomic proposition; use the annotation for scope."
    ),
    "NEGATED_SUBORDINATE_SCOPE": (
        "Determine exactly what negation scopes over and whether a subordinate event "
        "is asserted, denied, or merely mentioned."
    ),
    "EMBEDDED_CLAUSE": (
        "Determine whether the embedded event is asserted, requested, enabled, "
        "intended, or merely possible, and add it only if missing."
    ),
    "APPOSITIVE_IDENTITY": (
        "Determine whether the appositive identifies an existing party. Do not create "
        "a second event or duplicate party for an identity description."
    ),
    "REPEATED_PREDICATE_IDENTITY": (
        "Disambiguate predicate occurrences by source span and branch. Add a node only "
        "when an occurrence expresses a missing distinct event."
    ),
}


def resolver_schema(party_ids: Sequence[str], clause_ids: Sequence[str]) -> dict[str, Any]:
    strings = {"type": "array", "items": {"type": "string"}}
    proposition = {
        "type": "object",
        "properties": {
            "proposition_id": {"type": "string"},
            "party_id": {"type": "string", "enum": list(party_ids)},
            "outcome": {"type": "string"},
            "polarity": {"type": "string", "enum": [
                "BENEFICIAL", "ADVERSE", "NEUTRAL", "UNRESOLVED",
            ]},
            "directness": {"type": "string", "enum": ["DIRECT", "DOWNSTREAM"]},
            "modality": {"type": "string", "enum": [
                "CERTAIN", "STIPULATED_CONDITIONAL", "PROBABILISTIC", "POSSIBLE", "UNKNOWN",
            ]},
            "effect_kind": {"type": "string"},
            "quantities": strings,
            "source_proposition": {"type": "string", "minLength": 1},
            "clause_ids": {"type": "array", "minItems": 1, "items": {
                "type": "string", "enum": list(clause_ids),
            }},
        },
        "required": [
            "proposition_id", "party_id", "outcome", "polarity", "directness",
            "modality", "effect_kind", "quantities", "source_proposition", "clause_ids",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["NO_CHANGE", "ADD_NODES", "QUARANTINE"]},
            "assertion_status": {"type": "string", "enum": [
                "ASSERTED_SOURCE_EVENT", "UNIQUELY_LICENSED_ELLIPSIS",
                "NOT_ASSERTED", "AMBIGUOUS",
            ]},
            "added_propositions": {"type": "array", "items": proposition},
            "semantic_annotation": {"type": "string"},
            "evidence_spans": strings,
            "unresolved_span": {"type": "string"},
        },
        "required": [
            "status", "assertion_status", "added_propositions", "semantic_annotation",
            "evidence_spans", "unresolved_span",
        ],
        "additionalProperties": False,
    }


def build_resolution_jobs(
    obligation_bundle: Mapping[str, Any], neutral: Mapping[str, Any],
    clauses: Sequence[Mapping[str, Any]], coverage: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Route only cues not demonstrably complete at the predicate level."""
    clause_by_id = {str(row.get("clause_id") or ""): dict(row) for row in clauses}
    coverage_by_id = {str(row.get("obligation_id") or ""): row for row in coverage}
    propositions = [dict(row) for row in neutral.get("propositions") or []]
    jobs = []
    always_review = {
        "ELLIPSIS_OR_LIGHT_VERB", "CONDITIONAL_SCOPE", "NEGATED_SUBORDINATE_SCOPE",
        "APPOSITIVE_IDENTITY", "REPEATED_PREDICATE_IDENTITY",
    }
    for obligation in obligation_bundle.get("obligations") or []:
        obligation_id = str(obligation.get("obligation_id") or "")
        assessment = coverage_by_id.get(obligation_id) or {}
        obligation_type = str(obligation.get("type") or "")
        missing = list(assessment.get("missing_predicate_lemmas") or [])
        if obligation_type not in always_review and not missing:
            continue
        cited = [str(value) for value in obligation.get("clause_ids") or []]
        local = [
            row for row in propositions
            if set(cited) & set(str(value) for value in row.get("clause_ids") or [])
        ]
        # One neighboring clause on either side supplies antecedent context for
        # ellipsis and cross-clause references without exposing the full scenario.
        indexes = [int(value[1:]) for value in cited if value.startswith("C") and value[1:].isdigit()]
        context_ids = set(cited)
        for index in indexes:
            context_ids.update({f"C{index - 1}", f"C{index + 1}"})
        context = [clause_by_id[key] for key in sorted(context_ids) if key in clause_by_id]
        jobs.append({
            "obligation": copy.deepcopy(obligation),
            "instruction": ROUTE_INSTRUCTIONS.get(obligation_type, "Resolve only the stated question."),
            "source_context": context,
            "existing_propositions": local,
            "coverage_before": dict(assessment),
        })
    return jobs


def prompt_for_resolution_job(job: Mapping[str, Any]) -> str:
    import json
    return f"""[INST]
Answer one narrow semantic-resolution question. Existing propositions are immutable:
do not rewrite, delete, fuse, renumber, or repeat them. Return only genuinely missing
atomic propositions. NO_CHANGE is correct when the structure is already represented.
QUARANTINE when the source does not uniquely license a resolution. Every added node
must quote one exact contiguous source_proposition from the supplied source context,
use an existing party_id, and preserve negation, modality, quantity, and identity.
Syntactic cues locate a question but do not supply its semantic answer.
Set assertion_status=NOT_ASSERTED for intentions, qualifications, certifications,
requests, or hypothetical complements that do not themselves occur. ADD_NODES is
permitted only for ASSERTED_SOURCE_EVENT or UNIQUELY_LICENSED_ELLIPSIS.

Route instruction: {job.get('instruction')}
Payload:
{json.dumps(job, ensure_ascii=False, sort_keys=True)}
Return JSON only.
[/INST]"""


def apply_resolution_results(
    neutral: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Append resolver output while preserving an exact immutable audit trail."""
    updated = copy.deepcopy(dict(neutral))
    propositions = list(updated.get("propositions") or [])
    unresolved = list(updated.get("unresolved_source_spans") or [])
    existing_ids = {str(row.get("proposition_id") or "") for row in propositions}
    audit = []
    for job, result in zip(jobs, results):
        status = str(result.get("status") or "").upper()
        assertion_status = str(result.get("assertion_status") or "").upper()
        accepted = []
        rejected = []
        for row in result.get("added_propositions") or []:
            candidate = copy.deepcopy(dict(row))
            proposition_id = str(candidate.get("proposition_id") or "")
            source = " ".join(str(candidate.get("source_proposition") or "").split())
            cited = {str(value) for value in candidate.get("clause_ids") or []}
            source_context = {
                str(value.get("clause_id") or ""): str(value.get("text") or "")
                for value in job.get("source_context") or []
                if isinstance(value, Mapping)
            }
            exact_source = bool(source) and any(
                clause_id in source_context
                and source.casefold() in source_context[clause_id].casefold()
                for clause_id in cited
            )
            if status != "ADD_NODES":
                rejected.append({"candidate": candidate, "reason": "STATUS_NOT_ADD_NODES"})
            elif assertion_status not in {
                "ASSERTED_SOURCE_EVENT", "UNIQUELY_LICENSED_ELLIPSIS",
            }:
                rejected.append({"candidate": candidate, "reason": "EVENT_NOT_ASSERTED"})
            elif not exact_source:
                rejected.append({"candidate": candidate, "reason": "SOURCE_SPAN_NOT_EXACT"})
            elif not proposition_id or proposition_id in existing_ids:
                rejected.append({"candidate": candidate, "reason": "DUPLICATE_OR_EMPTY_ID"})
            elif any(
                candidate.get(key) == existing.get(key)
                for existing in propositions
                for key in ("source_proposition",)
                if candidate.get(key)
            ):
                rejected.append({"candidate": candidate, "reason": "DUPLICATES_EXISTING_SOURCE_SPAN"})
            else:
                candidate["resolver_obligation_id"] = (
                    job.get("obligation") or {}
                ).get("obligation_id")
                propositions.append(candidate)
                existing_ids.add(proposition_id)
                accepted.append(proposition_id)
        unresolved_span = str(result.get("unresolved_span") or "").strip()
        if status == "QUARANTINE" and unresolved_span and unresolved_span not in unresolved:
            unresolved.append(unresolved_span)
        audit.append({
            "obligation_id": (job.get("obligation") or {}).get("obligation_id"),
            "type": (job.get("obligation") or {}).get("type"),
            "status": status,
            "assertion_status": assertion_status,
            "accepted_proposition_ids": accepted,
            "rejected_additions": rejected,
            "semantic_annotation": str(result.get("semantic_annotation") or ""),
            "evidence_spans": list(result.get("evidence_spans") or []),
            "unresolved_span": unresolved_span,
        })
    updated["propositions"] = propositions
    updated["unresolved_source_spans"] = unresolved
    updated["targeted_resolution_audit"] = audit
    return updated, audit
