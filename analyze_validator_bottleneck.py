"""Separate semantic extraction, transformation survival, and graph admission."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from global_workspace.pairwise_challenge_evaluation import (
    align_gold_events,
    diagnose_neutral_node_inventory,
)
from global_workspace.staged_node_generation import triage_node_admission
from run_pairwise_challenge import canonical_action_mapping, remap_gold_pairs


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = _load(args.report)
    manifest = _load(report["manifest"])
    cases = {str(row["id"]): row for row in manifest.get("cases") or []}
    rows = []
    outcomes: Counter[str] = Counter()
    retriaged_statuses: Counter[str] = Counter()
    retriaged_decisions: Counter[str] = Counter()
    retriaged_aligned_edges = 0
    gold_edges = 0
    for result in report.get("results") or []:
        case = cases[str(result["id"])]
        stage_path = result.get("stage_one_artifact")
        stage = _load(stage_path) if stage_path and Path(stage_path).is_file() else {}
        log_path = Path(str(result.get("log") or ""))
        log = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        mapping = canonical_action_mapping(log, [str(x) for x in case.get("actions") or []])
        gold = remap_gold_pairs([dict(x) for x in case.get("gold_pairs") or []], mapping)
        neutral_diag = diagnose_neutral_node_inventory(stage.get("neutral_skeleton") or {}, gold)
        neutral_alignment = {
            (str(x.get("action_id") or ""), str(x.get("gold_event") or "")): x
            for x in neutral_diag.get("alignments") or []
        }
        graph_alignments = align_gold_events(stage.get("skeleton") or {}, gold)
        graph_alignment = {
            (str(x.get("action_id") or ""), str(x.get("gold_event") or "")): x
            for x in graph_alignments
        }
        propositions = {
            str(x.get("proposition_id") or ""): x
            for x in (stage.get("skeleton") or {}).get("propositions") or []
        }
        retriage = triage_node_admission(
            stage.get("skeleton") or {}, stage.get("errors") or [],
        )
        retriaged_statuses[str(retriage.get("stage_status") or "MISSING")] += 1
        retriaged_decisions.update({
            str(key).upper(): int(value)
            for key, value in (retriage.get("counts") or {}).items()
        })
        retriaged_alignments = align_gold_events(
            retriage.get("admitted_skeleton") or {}, gold,
        )
        retriaged_alignment = {
            (str(x.get("action_id") or ""), str(x.get("gold_event") or "")): x
            for x in retriaged_alignments
        }
        for pair in gold:
            gold_edges += 1
            source_key = (
                str(pair.get("action_id") or ""),
                str(pair.get("source_event") or ""),
            )
            target_key = (
                str(pair.get("action_id") or ""),
                str(pair.get("target_event") or ""),
            )
            if (
                (retriaged_alignment.get(source_key) or {}).get("status") == "MATCHED"
                and (retriaged_alignment.get(target_key) or {}).get("status") == "MATCHED"
            ):
                retriaged_aligned_edges += 1
        ledger = {
            str(x.get("neutral_proposition_id") or ""): x
            for x in stage.get("node_evidence_ledger") or []
        }
        endpoints = []
        for pair in gold:
            for field in ("source_event", "target_event"):
                key = (str(pair.get("action_id") or ""), str(pair.get(field) or ""))
                if key not in endpoints:
                    endpoints.append(key)
        for key in endpoints:
            neutral = neutral_alignment.get(key) or {}
            graph = graph_alignment.get(key) or {}
            neutral_present = neutral.get("status") == "MATCHED"
            graph_present = graph.get("status") == "MATCHED"
            graph_id = str(graph.get("matched_proposition_id") or "")
            neutral_id = str(neutral.get("matched_proposition_id") or "").split("__NEUTRAL__", 1)[0]
            if not neutral_id and graph_id in propositions:
                neutral_id = str(propositions[graph_id].get("neutral_proposition_id") or "")
            ledger_row = ledger.get(neutral_id) or {}
            admitted = str(stage.get("status") or "") == "ADMITTED" and graph_present
            retriaged = retriaged_alignment.get(key) or {}
            retriaged_admitted = retriaged.get("status") == "MATCHED"
            if admitted:
                outcome = "ADMITTED"
            elif graph_present and neutral_present:
                outcome = "VALIDATOR_GLOBAL_REJECTION"
            elif graph_present:
                outcome = "INTRODUCED_AFTER_NEUTRAL_EXTRACTION"
            elif neutral_present:
                outcome = "LOST_DURING_OWNERSHIP_OR_NORMALIZATION"
            else:
                outcome = "TRUE_GENERATION_OR_ALIGNMENT_FAILURE"
            outcomes[outcome] += 1
            rows.append({
                "case_id": result.get("id"), "family": result.get("family"),
                "action_id": key[0], "gold_event": key[1],
                "generated_neutral": neutral_present,
                "neutral_proposition_id": neutral_id or None,
                "survived_to_skeleton": graph_present,
                "graph_proposition_id": graph_id or None,
                "ownership_status": ledger_row.get("ownership_status"),
                "normalization_status": ledger_row.get("normalization_status"),
                "provenance_status": ledger_row.get("provenance_status"),
                "ledger_graph_admission_status": ledger_row.get("graph_admission_status"),
                "stage_status": stage.get("status", "MISSING"),
                "validator_outcome": outcome,
                "retriaged_admitted": retriaged_admitted,
                "retriaged_proposition_id": retriaged.get("matched_proposition_id"),
            })
    total = len(rows)
    semantic = sum(row["generated_neutral"] for row in rows)
    survived = sum(row["survived_to_skeleton"] for row in rows)
    admitted = outcomes["ADMITTED"]
    retriaged_admitted = sum(row["retriaged_admitted"] for row in rows)
    payload = {
        "source_report": str(args.report),
        "aggregate": {
            "gold_node_count": total,
            "semantic_recall_neutral": semantic / total if total else None,
            "semantic_nodes_neutral": semantic,
            "skeleton_presence_recall": survived / total if total else None,
            "skeleton_present_nodes": survived,
            "admitted_graph_recall": admitted / total if total else None,
            "admitted_graph_nodes": admitted,
            "retriaged_admitted_graph_recall": (
                retriaged_admitted / total if total else None
            ),
            "retriaged_admitted_graph_nodes": retriaged_admitted,
            "rescued_admitted_gold_nodes": retriaged_admitted - admitted,
            "retriaged_stage_status_counts": dict(sorted(retriaged_statuses.items())),
            "retriaged_node_decision_counts": dict(sorted(retriaged_decisions.items())),
            "gold_edge_count": gold_edges,
            "retriaged_edges_with_both_endpoints": retriaged_aligned_edges,
            "retriaged_edge_endpoint_availability": (
                retriaged_aligned_edges / gold_edges if gold_edges else None
            ),
            "validator_outcome_counts": dict(sorted(outcomes.items())),
        },
        "nodes": rows,
    }
    path = args.output or args.report.with_name("validator_bottleneck_analysis.json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
