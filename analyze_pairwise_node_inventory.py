"""Score Stage-1 node inventories from an existing frozen challenge report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from global_workspace.pairwise_challenge_evaluation import (
    align_gold_events,
    diagnose_neutral_node_inventory,
    diagnose_node_inventory,
)
from run_pairwise_challenge import canonical_action_mapping, remap_gold_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.report.read_text(encoding="utf-8"))
    manifest_path = Path(str(frozen["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_by_id = {str(row["id"]): row for row in manifest.get("cases") or []}
    results = []
    for prior in frozen.get("results") or []:
        case = case_by_id[str(prior["id"])]
        stage_path = prior.get("stage_one_artifact")
        stage = (
            json.loads(Path(stage_path).read_text(encoding="utf-8"))
            if stage_path and Path(stage_path).is_file() else {}
        )
        skeleton = stage.get("skeleton") or {}
        log_path = Path(str(prior.get("log") or ""))
        log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        action_mapping = canonical_action_mapping(
            log_text, [str(value) for value in case.get("actions") or []],
        )
        gold_pairs = remap_gold_pairs(
            [dict(row) for row in case.get("gold_pairs") or []], action_mapping,
        )
        alignments = align_gold_events(skeleton, gold_pairs)
        diagnostic = diagnose_node_inventory(
            skeleton, gold_pairs, alignments,
        )
        neutral_diagnostic = diagnose_neutral_node_inventory(
            stage.get("neutral_skeleton") or {}, gold_pairs,
        )
        results.append({
            "id": case["id"], "family": case["family"],
            "stage_one_status": stage.get("status", "MISSING"),
            "presentation_to_canonical_action_ids": action_mapping,
            "diagnostic": diagnostic,
            "neutral_diagnostic": neutral_diagnostic,
            "node_preservation_waterfall": stage.get(
                "node_preservation_waterfall"
            ) or (stage.get("skeleton") or {}).get(
                "node_preservation_waterfall"
            ) or {},
        })
    total_gold = sum(row["diagnostic"]["gold_node_count"] for row in results)
    correct = sum(row["diagnostic"]["present_correct_owner_count"] for row in results)
    wrong = sum(row["diagnostic"]["present_wrong_owner_count"] for row in results)
    absent = sum(row["diagnostic"]["absent_or_unaligned_count"] for row in results)
    payload = {
        "analysis_version": "1.0",
        "source_protocol": frozen.get("protocol"),
        "source_report": str(args.report),
        "aggregate": {
            "gold_node_count": total_gold,
            "present_correct_owner_count": correct,
            "present_wrong_owner_count": wrong,
            "absent_or_unaligned_count": absent,
            "node_recall_correct_owner": correct / total_gold if total_gold else None,
            "node_presence_recall_any_owner": (
                (correct + wrong) / total_gold if total_gold else None
            ),
            "fused_gold_node_count": sum(
                row["diagnostic"]["fused_gold_node_count"] for row in results
            ),
            "complete_action_branch_count": sum(
                branch["status"] == "REPRESENTED"
                for row in results
                for branch in row["diagnostic"]["action_branches"]
            ),
            "required_action_branch_count": sum(
                len(row["diagnostic"]["action_branches"]) for row in results
            ),
            "neutral_present_node_count": sum(
                row["neutral_diagnostic"]["present_neutral_node_count"]
                for row in results
            ),
            "neutral_node_recall": (
                sum(
                    row["neutral_diagnostic"]["present_neutral_node_count"]
                    for row in results
                ) / total_gold if total_gold else None
            ),
            "transformation_waterfall": {
                key: sum(
                    int(row["node_preservation_waterfall"].get(key) or 0)
                    for row in results
                )
                for key in (
                    "neutral_nodes_extracted",
                    "nodes_materialized_after_ownership",
                    "nodes_surviving_normalization",
                    "nodes_with_valid_provenance",
                    "nodes_admitted_to_graph",
                )
            },
        },
        "results": results,
    }
    destination = args.output or args.report.with_name("node_inventory_report.json")
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    print(f"Node inventory report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
