"""Re-score saved fixed-neutral resolver artifacts with the current aligner."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from global_workspace.pairwise_challenge_evaluation import diagnose_neutral_node_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("evals/topology_structure_expanded_challenge.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = {str(row["id"]): row for row in manifest.get("cases") or []}
    results = []
    for name in sorted(glob.glob(str(args.artifact_dir / "*.json"))):
        if name.endswith("/report.json") or name.endswith("realigned_analysis.json"):
            continue
        artifact = json.loads(Path(name).read_text(encoding="utf-8"))
        case = cases[str(artifact["id"])]
        before = diagnose_neutral_node_inventory(
            artifact.get("neutral_before") or {}, case.get("gold_pairs") or [],
        )
        after = diagnose_neutral_node_inventory(
            artifact.get("neutral_after") or {}, case.get("gold_pairs") or [],
        )
        accepted = list(artifact.get("accepted_addition_ids") or [])
        before_ids = {
            str(row.get("proposition_id") or "")
            for row in (artifact.get("neutral_before") or {}).get("propositions") or []
        }
        useful = sorted({
            str(row.get("matched_proposition_id") or "").split("__NEUTRAL__", 1)[0]
            for row in after.get("alignments") or []
            if row.get("status") == "MATCHED"
            and str(row.get("matched_proposition_id") or "").split("__NEUTRAL__", 1)[0]
            not in before_ids
        })
        results.append({
            "id": artifact["id"], "gold_node_count": before["gold_node_count"],
            "gold_nodes_before": before["present_neutral_node_count"],
            "gold_nodes_after": after["present_neutral_node_count"],
            "accepted_addition_ids": accepted, "gold_useful_added_ids": useful,
        })
    total = sum(row["gold_node_count"] for row in results)
    before = sum(row["gold_nodes_before"] for row in results)
    after = sum(row["gold_nodes_after"] for row in results)
    accepted = sum(len(row["accepted_addition_ids"]) for row in results)
    useful = sum(len(row["gold_useful_added_ids"]) for row in results)
    output = {
        "protocol": "FIXED_NEUTRAL_RESOLVER_REALIGNED",
        "aggregate": {
            "gold_node_count": total,
            "gold_nodes_before": before, "gold_nodes_after": after,
            "gold_node_recall_before": before / total if total else None,
            "gold_node_recall_after": after / total if total else None,
            "gold_node_gain": after - before,
            "accepted_addition_count": accepted,
            "gold_useful_addition_count": useful,
        },
        "results": results,
    }
    path = args.output or args.artifact_dir / "realigned_analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2))
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
