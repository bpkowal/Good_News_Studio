"""Read-only pairwise relation judgments over a Stage-1 world skeleton.

This module owns no model client and never mutates a topology.  It prepares
independent pair jobs, validates constrained judge responses, and summarizes
agreement with the deterministic scaffold.  Callers may supply an LLM, NLI
model, or future local classifier through the small ``judge`` callback.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import re
from typing import Any


RELATIONS = ("CAUSES", "PREVENTS", "ENABLES", "NONE", "AMBIGUOUS")
CONFIDENCES = ("HIGH", "MEDIUM", "LOW")
SUPPORT_TYPES = (
    "EXPLICIT_SOURCE", "IMPLICIT_SOURCE", "STRUCTURAL_ONLY", "INSUFFICIENT",
)
POSITIVE_RELATIONS = {"CAUSES", "PREVENTS", "ENABLES"}

_CAUSAL_MARKERS = re.compile(
    r"\b(?:cause[sd]?|causing|because|therefore|leads? to|results? in|"
    r"enables?|allows?|lets?|prevents?|blocks?|averts?)\b", re.IGNORECASE,
)
_CONDITIONAL_MARKERS = re.compile(
    r"\b(?:if|unless|provided(?: that)?|only if|when)\b", re.IGNORECASE,
)
_NEGATION_MARKERS = re.compile(
    r"\b(?:no|not|never|neither|without|doesn't|does not|cannot|can't)\b",
    re.IGNORECASE,
)


def build_pairwise_jobs(
    skeleton: Mapping[str, Any],
    clauses: Sequence[Mapping[str, Any]],
    scaffold: Mapping[str, Any],
    *,
    evidence_augmented: bool = False,
) -> list[dict[str, Any]]:
    """Create one unbiased job per ordered within-action proposition pair.

    Scaffold candidates are ordered first so a cost cap preserves direct
    comparisons, but absent and reversed edges are also audited.  This lets the
    shadow layer detect omissions instead of merely confirming proposals.
    """
    propositions = {
        str(row.get("proposition_id") or ""): dict(row)
        for row in skeleton.get("propositions") or []
        if isinstance(row, Mapping) and str(row.get("proposition_id") or "")
    }
    parties = {
        str(row.get("party_id") or ""): dict(row)
        for row in skeleton.get("parties") or []
        if isinstance(row, Mapping) and str(row.get("party_id") or "")
    }
    clause_by_id = {
        str(row.get("clause_id") or ""): str(row.get("text") or "")
        for row in clauses
    }
    pairs: dict[tuple[str, str], set[str]] = {}
    scaffold_evidence: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for edge in scaffold.get("edge_candidates") or []:
        if not isinstance(edge, Mapping):
            continue
        source_id = str(edge.get("source_proposition_id") or "")
        target_id = str(edge.get("target_proposition_id") or "")
        if source_id in propositions and target_id in propositions and source_id != target_id:
            pairs.setdefault((source_id, target_id), set()).add(
                str(edge.get("relation") or "")
            )
            scaffold_evidence.setdefault((source_id, target_id), []).append({
                # The proposed relation is intentionally excluded.
                "support_type": str(edge.get("support_type") or ""),
                "support_span": str(edge.get("support_span") or ""),
                "clause_id": str(edge.get("clause_id") or ""),
                "commitment": str(edge.get("commitment") or ""),
            })
    proposition_rows = list(propositions.values())
    for source in proposition_rows:
        for target in proposition_rows:
            source_id = str(source.get("proposition_id") or "")
            target_id = str(target.get("proposition_id") or "")
            if (
                source_id != target_id
                and source.get("action_id") == target.get("action_id")
            ):
                pairs.setdefault((source_id, target_id), set())
    jobs: list[dict[str, Any]] = []
    def pair_priority(item: tuple[tuple[str, str], set[str]]) -> tuple[Any, ...]:
        (source_id, target_id), relations = item
        source = propositions[source_id]
        target = propositions[target_id]
        source_clause_ids = {
            str(value) for value in source.get("clause_ids") or []
            if str(value).startswith("C")
        }
        target_clause_ids = {
            str(value) for value in target.get("clause_ids") or []
            if str(value).startswith("C")
        }
        common = source_clause_ids & target_clause_ids
        common_text = " ".join(clause_by_id.get(value, "") for value in common)
        if not evidence_augmented:
            return (not bool(relations), source_id, target_id)
        # Source-local evidence outranks generic scaffold guesses. Both
        # directions of a shared clause receive equal priority, preserving the
        # ability to detect reversed causality.
        cues = scaffold_evidence.get((source_id, target_id), [])
        source_licensed = any(
            row.get("commitment") == "SOURCE_LICENSED" for row in cues
        )
        explicit_marker = bool(_CAUSAL_MARKERS.search(common_text))
        return (
            not source_licensed,
            not bool(common),
            not explicit_marker,
            not bool(relations),
            source_id,
            target_id,
        )

    for index, ((source_id, target_id), scaffold_relations) in enumerate(
        sorted(pairs.items(), key=pair_priority),
        start=1,
    ):
        source = propositions[source_id]
        target = propositions[target_id]
        cited = list(dict.fromkeys([
            *[str(value) for value in source.get("clause_ids") or []],
            *[str(value) for value in target.get("clause_ids") or []],
        ]))
        source_clauses = [
            {"clause_id": clause_id, "text": clause_by_id[clause_id]}
            for clause_id in cited if clause_id in clause_by_id
        ]
        common_clause_ids = sorted(
            set(str(value) for value in source.get("clause_ids") or [])
            & set(str(value) for value in target.get("clause_ids") or [])
        )
        evidence_text = " ".join(
            clause_by_id.get(clause_id, "") for clause_id in common_clause_ids
        )
        jobs.append({
            "audit_id": f"PAIR_{index}",
            "action_id": str(source.get("action_id") or ""),
            "source": source,
            "target": target,
            "source_party": parties.get(str(source.get("party_id") or ""), {}),
            "target_party": parties.get(str(target.get("party_id") or ""), {}),
            "source_clauses": source_clauses,
            "scenario_context_clauses": [dict(row) for row in clauses],
            # Retained for post-judgment measurement; never shown to the judge.
            "scaffold_relations": sorted(filter(None, scaffold_relations)),
            "structured_evidence": ({
                "common_clause_ids": common_clause_ids,
                "common_clause_text": evidence_text,
                "causal_marker_present": bool(_CAUSAL_MARKERS.search(evidence_text)),
                "conditional_scope_present": bool(_CONDITIONAL_MARKERS.search(evidence_text)),
                "negation_present": bool(_NEGATION_MARKERS.search(evidence_text)),
                "source_modality": str(source.get("modality") or ""),
                "target_modality": str(target.get("modality") or ""),
                "source_directness": str(source.get("directness") or ""),
                "target_directness": str(target.get("directness") or ""),
                "scaffold_evidence_without_label": scaffold_evidence.get(
                    (source_id, target_id), []
                ),
            } if evidence_augmented else None),
        })
    return jobs


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "relation": {"type": "string", "enum": list(RELATIONS)},
            "confidence": {"type": "string", "enum": list(CONFIDENCES)},
            "support_type": {"type": "string", "enum": list(SUPPORT_TYPES)},
            "support_span": {"type": "string", "maxLength": 300},
            "reason": {"type": "string", "minLength": 4, "maxLength": 300},
        },
        "required": [
            "relation", "confidence", "support_type", "support_span", "reason",
        ],
        "additionalProperties": False,
    }


def prompt_for_job(job: Mapping[str, Any]) -> str:
    """Prompt one local judgment without exposing the scaffold or full graph."""
    visible = {
        "action_id": job.get("action_id"),
        "source_event": job.get("source"),
        "target_event": job.get("target"),
        "source_party": job.get("source_party"),
        "target_party": job.get("target_party"),
        "source_clauses": job.get("source_clauses"),
        "scenario_context_clauses": job.get("scenario_context_clauses"),
    }
    if job.get("structured_evidence") is not None:
        visible["structured_evidence"] = job.get("structured_evidence")
    return f"""[INST]
Judge exactly one directed semantic relation. Do not construct a graph and do
not infer moral value. Decide whether SOURCE EVENT bears one allowed relation
to TARGET EVENT in the stated direction.

Allowed relations:
- CAUSES: the source produces the target.
- PREVENTS: the source blocks or averts the target.
- ENABLES: the source makes the target possible without itself producing it.
- NONE: the source text does not license a directed relation.
- AMBIGUOUS: at least two live relations/directions remain source-compatible.

Use EXPLICIT_SOURCE only for wording that directly states the relation.
IMPLICIT_SOURCE requires a locally licensed but unstated connection.
STRUCTURAL_ONLY means the relation is merely plausible from node types.
INSUFFICIENT means the source cannot support a relation. Use the full scenario
only to resolve identity, ellipsis, action ownership, and cross-clause relations;
do not treat ethical-task expectations as facts. Quote support_span exactly from
a source proposition or scenario clause, or return an empty span when support
is insufficient. Keep source and target party identity distinct. Conditional
and negation flags describe scope that must be preserved: do not collapse a
conditional relation into an unconditional one. Structured scaffold evidence,
when present, contains cues only; it deliberately withholds the proposed label.
Return JSON only.

PAIR:
{json.dumps(visible, ensure_ascii=False, sort_keys=True)}
[/INST]"""


def run_pairwise_audit(
    jobs: Sequence[Mapping[str, Any]],
    judge: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run independent shadow judgments; failures remain audit data only."""
    records: list[dict[str, Any]] = []
    for job in jobs:
        base = {
            "audit_id": str(job.get("audit_id") or ""),
            "action_id": str(job.get("action_id") or ""),
            "source_proposition_id": str(
                (job.get("source") or {}).get("proposition_id") or ""
            ),
            "target_proposition_id": str(
                (job.get("target") or {}).get("proposition_id") or ""
            ),
            "scaffold_relations": list(job.get("scaffold_relations") or []),
        }
        try:
            answer = dict(judge(job))
            relation = str(answer.get("relation") or "").upper()
            confidence = str(answer.get("confidence") or "").upper()
            support_type = str(answer.get("support_type") or "").upper()
            if relation not in RELATIONS:
                raise ValueError(f"unknown relation {relation!r}")
            if confidence not in CONFIDENCES:
                raise ValueError(f"unknown confidence {confidence!r}")
            if support_type not in SUPPORT_TYPES:
                raise ValueError(f"unknown support_type {support_type!r}")
            records.append({
                **base, "status": "JUDGED", "relation": relation,
                "confidence": confidence, "support_type": support_type,
                "support_span": str(answer.get("support_span") or ""),
                "reason": str(answer.get("reason") or ""),
                "agrees_with_scaffold": (
                    relation in base["scaffold_relations"]
                    or (relation == "NONE" and not base["scaffold_relations"])
                ),
            })
        except Exception as exc:  # shadow instrumentation must not block grounding
            records.append({
                **base, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
            })
    judged = [row for row in records if row["status"] == "JUDGED"]
    by_direction = {
        (
            row["action_id"], row["source_proposition_id"],
            row["target_proposition_id"],
        ): row
        for row in judged
    }
    for row in judged:
        reverse = by_direction.get((
            row["action_id"], row["target_proposition_id"],
            row["source_proposition_id"],
        ))
        scaffold = set(row["scaffold_relations"])
        relation = row["relation"]
        reverse_scaffold = set(reverse["scaffold_relations"]) if reverse else set()
        reverse_relation = str(reverse.get("relation") or "") if reverse else ""
        direction_conflict = bool(
            reverse
            and (
                (scaffold & POSITIVE_RELATIONS and relation == "NONE"
                 and reverse_relation in scaffold)
                or (relation in POSITIVE_RELATIONS and not scaffold
                    and relation in reverse_scaffold and reverse_relation == "NONE")
            )
        )
        if relation == "AMBIGUOUS":
            comparison = "AMBIGUOUS"
        elif direction_conflict:
            comparison = "DIRECTION_DISAGREEMENT"
        elif relation in scaffold or (relation == "NONE" and not scaffold):
            comparison = "AGREE"
        elif not scaffold and relation in POSITIVE_RELATIONS:
            comparison = "AUDIT_ONLY"
        elif scaffold and relation == "NONE":
            comparison = "SCAFFOLD_ONLY"
        else:
            comparison = "LABEL_DISAGREEMENT"
        row["comparison"] = comparison
    ambiguous = [row for row in judged if row["relation"] == "AMBIGUOUS"]
    none_rows = [row for row in judged if row["relation"] == "NONE"]
    agreements = [row for row in judged if row["agrees_with_scaffold"]]
    scaffolded = [row for row in judged if row["scaffold_relations"]]
    scaffold_agreements = [row for row in scaffolded if row["agrees_with_scaffold"]]
    unsupported_scaffold = [
        row for row in judged
        if row["scaffold_relations"] and row["relation"] in {"NONE", "AMBIGUOUS"}
    ]
    novel_relations = [
        row for row in judged
        if not row["scaffold_relations"]
        and row["relation"] in {"CAUSES", "PREVENTS", "ENABLES"}
    ]
    comparison_counts: dict[str, int] = {}
    for row in judged:
        comparison = str(row.get("comparison") or "")
        comparison_counts[comparison] = comparison_counts.get(comparison, 0) + 1
    return {
        "audit_version": "1.0",
        "mode": "READ_ONLY_SHADOW",
        "status": (
            "EMPTY_NO_PAIRS" if not records
            else "COMPLETE" if len(judged) == len(records)
            else "PARTIAL" if judged else "FAILED"
        ),
        "pair_count": len(records),
        "judged_count": len(judged),
        "error_count": len(records) - len(judged),
        "ambiguous_count": len(ambiguous),
        "none_count": len(none_rows),
        "unsupported_scaffold_count": len(unsupported_scaffold),
        "novel_relation_count": len(novel_relations),
        "scaffold_agreement_count": len(agreements),
        "scaffold_agreement_rate": (
            len(agreements) / len(judged) if judged else None
        ),
        "scaffolded_pair_count": len(scaffolded),
        "scaffolded_pair_agreement_count": len(scaffold_agreements),
        "scaffolded_pair_agreement_rate": (
            len(scaffold_agreements) / len(scaffolded) if scaffolded else None
        ),
        "scaffold_disagreement_count": len(scaffolded) - len(scaffold_agreements),
        "comparison_counts": comparison_counts,
        "records": records,
    }


def evaluate_against_gold(
    audit: Mapping[str, Any],
    gold_relations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score audit-only recovery against human-authored challenge-set edges."""
    records = {
        (
            str(row.get("action_id") or ""),
            str(row.get("source_proposition_id") or ""),
            str(row.get("target_proposition_id") or ""),
        ): row
        for row in audit.get("records") or []
        if isinstance(row, Mapping) and row.get("status") == "JUDGED"
    }
    missing_gold = 0
    recovered_missing_gold = 0
    correct = 0
    evaluated = 0
    details: list[dict[str, Any]] = []
    for gold in gold_relations:
        key = (
            str(gold.get("action_id") or ""),
            str(gold.get("source_proposition_id") or ""),
            str(gold.get("target_proposition_id") or ""),
        )
        expected = str(gold.get("relation") or "").upper()
        row = records.get(key)
        actual = str((row or {}).get("relation") or "")
        scaffold = set((row or {}).get("scaffold_relations") or [])
        is_missing_from_scaffold = expected in POSITIVE_RELATIONS and expected not in scaffold
        if is_missing_from_scaffold:
            missing_gold += 1
        matched = bool(row) and actual == expected
        if matched:
            correct += 1
            if is_missing_from_scaffold:
                recovered_missing_gold += 1
        evaluated += 1
        details.append({
            "action_id": key[0],
            "source_proposition_id": key[1],
            "target_proposition_id": key[2],
            "gold_relation": expected,
            "audit_relation": actual or None,
            "scaffold_relations": sorted(scaffold),
            "correct": matched,
            "gold_missing_from_scaffold": is_missing_from_scaffold,
        })
    return {
        "evaluated_gold_count": evaluated,
        "correct_gold_count": correct,
        "gold_accuracy": correct / evaluated if evaluated else None,
        "gold_relations_absent_from_scaffold": missing_gold,
        "correct_audit_only_relations": recovered_missing_gold,
        "audit_novelty_rate": (
            recovered_missing_gold / missing_gold if missing_gold else None
        ),
        "details": details,
    }
