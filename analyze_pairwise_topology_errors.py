"""Classify frozen pairwise-challenge topology errors without new model calls.

This analyzer deliberately distinguishes a relation judgment from an absent audit
record.  It also reports whether the scaffold uniquely selected the gold label or
merely included it among competing candidates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _audit_class(gold: str, relation: str | None, record: dict[str, Any] | None) -> str:
    if record is None:
        return "AUDIT_NOT_EVALUATED"
    if relation == gold:
        return "CORRECT"
    if gold == "AMBIGUOUS" and relation not in {None, "AMBIGUOUS"}:
        return "CONDITIONAL_SCOPE_COLLAPSE"
    if relation in {None, "NONE"} and gold != "NONE":
        return "EXCESSIVE_NONE"
    if relation == "AMBIGUOUS" and gold != "AMBIGUOUS":
        return "EXCESSIVE_AMBIGUOUS"
    if gold == "NONE" and relation not in {None, "NONE"}:
        return "FALSE_POSITIVE_OR_DIRECTION_ERROR"
    return "ONTOLOGY_LABEL_CONFUSION"


def _scaffold_class(gold: str, relations: list[str]) -> str:
    unique = sorted(set(relations))
    if gold == "NONE":
        return "CORRECT_ABSENCE" if not unique else "FALSE_POSITIVE_OR_DIRECTION_ERROR"
    if gold == "AMBIGUOUS" and unique:
        return "CONDITIONAL_SCOPE_COLLAPSE"
    if not unique:
        return "MISSING_EDGE"
    if gold in unique:
        return "CORRECT" if len(unique) == 1 else "GOLD_INCLUDED_NONUNIQUE"
    return "ONTOLOGY_LABEL_CONFUSION"


def analyze(report_path: Path) -> dict[str, Any]:
    report = _load(report_path)
    edges: list[dict[str, Any]] = []

    for result in report.get("results", []):
        details = (result.get("metrics") or {}).get("details") or []
        if not details:
            continue
        stage = _load(result["stage_one_artifact"])
        audit = _load(result["audit_artifact"]) if result.get("audit_artifact") else {}
        propositions = {
            item.get("proposition_id"): item
            for item in (stage.get("skeleton") or {}).get("propositions", [])
        }
        candidates = (stage.get("topology_scaffold") or {}).get("edge_candidates", [])
        audit_records = audit.get("records") or []

        for detail in details:
            action_id = detail["action_id"]
            source_id = detail["source_proposition_id"]
            target_id = detail["target_proposition_id"]
            gold = detail["gold_relation"]
            matching_candidates = [
                item for item in candidates
                if item.get("source_proposition_id") == source_id
                and item.get("target_proposition_id") == target_id
            ]
            record = next((
                item for item in audit_records
                if item.get("action_id") == action_id
                and item.get("source_proposition_id") == source_id
                and item.get("target_proposition_id") == target_id
            ), None)
            audit_relation = record.get("relation") if record else None
            if record is not None:
                audit_unavailability_reason = None
            elif result.get("audit_status") == "NOT_RUN":
                audit_unavailability_reason = "UPSTREAM_STAGE_REJECTED"
            elif int(audit.get("truncated_count") or 0) > 0:
                audit_unavailability_reason = "PAIR_BUDGET_TRUNCATION_OR_RANKING"
            else:
                audit_unavailability_reason = "PAIR_NOT_GENERATED"
            scaffold_relations = [item.get("relation") for item in matching_candidates if item.get("relation")]
            source = propositions.get(source_id) or {}
            target = propositions.get(target_id) or {}
            edges.append({
                "case_id": result.get("id"),
                "family": result.get("family"),
                "action_id": action_id,
                "source": {
                    "proposition_id": source_id,
                    "outcome": source.get("outcome"),
                    "source_proposition": source.get("source_proposition"),
                    "modality": source.get("modality"),
                    "clause_ids": source.get("clause_ids") or [],
                },
                "target": {
                    "proposition_id": target_id,
                    "outcome": target.get("outcome"),
                    "source_proposition": target.get("source_proposition"),
                    "modality": target.get("modality"),
                    "clause_ids": target.get("clause_ids") or [],
                },
                "gold_relation": gold,
                "scaffold_relations": sorted(set(scaffold_relations)),
                "scaffold_class": _scaffold_class(gold, scaffold_relations),
                "scaffold_evidence": [{
                    "support_type": item.get("support_type"),
                    "support_span": item.get("support_span"),
                    "clause_id": item.get("clause_id"),
                    "commitment": item.get("commitment"),
                } for item in matching_candidates],
                "audit_relation": audit_relation,
                "audit_class": _audit_class(gold, audit_relation, record),
                "audit_unavailability_reason": audit_unavailability_reason,
                "audit_judgment": ({
                    "confidence": record.get("confidence"),
                    "support_type": record.get("support_type"),
                    "support_span": record.get("support_span"),
                    "reason": record.get("reason"),
                } if record else None),
                "stage_one_status": result.get("stage_one_status"),
                "audit_status": result.get("audit_status"),
            })

    judged = [edge for edge in edges if edge["audit_class"] != "AUDIT_NOT_EVALUATED"]
    audit_counts = Counter(edge["audit_class"] for edge in edges)
    unavailable_counts = Counter(
        edge["audit_unavailability_reason"] for edge in edges
        if edge["audit_unavailability_reason"]
    )
    scaffold_counts = Counter(edge["scaffold_class"] for edge in edges)
    return {
        "source_report": str(report_path.resolve()),
        "policy": {
            "no_new_model_calls": True,
            "missing_audit_record_is_not_none": True,
            "scaffold_gold_included_is_not_unique_correctness": True,
        },
        "summary": {
            "evaluable_gold_edges": len(edges),
            "audit_judgments_available": len(judged),
            "audit_coverage": len(judged) / len(edges) if edges else None,
            "audit_correct_given_judged": sum(edge["audit_class"] == "CORRECT" for edge in judged),
            "audit_accuracy_given_judged": (
                sum(edge["audit_class"] == "CORRECT" for edge in judged) / len(judged)
                if judged else None
            ),
            "audit_end_to_end_correct": sum(edge["audit_class"] == "CORRECT" for edge in edges),
            "audit_end_to_end_accuracy": (
                sum(edge["audit_class"] == "CORRECT" for edge in edges) / len(edges)
                if edges else None
            ),
            "scaffold_unique_correct": sum(edge["scaffold_class"] in {"CORRECT", "CORRECT_ABSENCE"} for edge in edges),
            "scaffold_gold_included_nonunique": scaffold_counts["GOLD_INCLUDED_NONUNIQUE"],
            "scaffold_unique_accuracy": (
                sum(edge["scaffold_class"] in {"CORRECT", "CORRECT_ABSENCE"} for edge in edges) / len(edges)
                if edges else None
            ),
            "audit_failure_modes": dict(sorted(audit_counts.items())),
            "audit_unavailability_reasons": dict(sorted(unavailable_counts.items())),
            "scaffold_failure_modes": dict(sorted(scaffold_counts.items())),
        },
        "edges": edges,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.report.with_name("topology_error_analysis.json")
    analysis = analyze(args.report)
    output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis["summary"], indent=2))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
