"""Compare a targeted-resolution run with its ordinary Stage-1 control."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("experimental", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    control = _load(args.control)
    experimental = _load(args.experimental)
    accepted = []
    gold_useful = []
    statuses: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for result in experimental.get("results") or []:
        artifact_path = result.get("stage_one_artifact")
        if not artifact_path or not Path(artifact_path).is_file():
            continue
        artifact = _load(artifact_path)
        aligned = {
            str(row.get("matched_proposition_id") or "")
            for row in result.get("alignments") or []
            if row.get("matched_proposition_id")
        }
        for row in artifact.get("targeted_resolution_audit") or []:
            statuses[str(row.get("status") or "UNKNOWN")] += 1
            routes[str(row.get("type") or "UNKNOWN")] += 1
            for item in row.get("rejected_additions") or []:
                rejected[str(item.get("reason") or "UNKNOWN")] += 1
            for proposition_id in row.get("accepted_proposition_ids") or []:
                record = {
                    "case_id": result.get("id"), "route": row.get("type"),
                    "proposition_id": proposition_id,
                }
                accepted.append(record)
                if proposition_id in aligned:
                    gold_useful.append(record)
    ca = control.get("aggregate") or {}
    ea = experimental.get("aggregate") or {}
    output = {
        "control_report": str(args.control),
        "experimental_report": str(args.experimental),
        "summary": {
            "control_node_recall": ca.get("node_recall"),
            "experimental_node_recall": ea.get("node_recall"),
            "control_aligned_gold_nodes": ca.get("aligned_gold_node_count"),
            "experimental_aligned_gold_nodes": ea.get("aligned_gold_node_count"),
            "control_evaluable_gold_edges": ca.get("aligned_gold_edge_count"),
            "experimental_evaluable_gold_edges": ea.get("aligned_gold_edge_count"),
            "resolver_call_count": sum(statuses.values()),
            "resolver_status_counts": dict(sorted(statuses.items())),
            "resolver_route_counts": dict(sorted(routes.items())),
            "accepted_addition_count": len(accepted),
            "gold_aligned_accepted_addition_count": len(gold_useful),
            "non_gold_aligned_addition_count": len(accepted) - len(gold_useful),
            "resolver_precision": None,
            "resolver_precision_status": "NOT_IDENTIFIED_SPARSE_GOLD",
            "rejected_addition_reasons": dict(sorted(rejected.items())),
        },
        "accepted_additions": accepted,
        "gold_aligned_accepted_additions": gold_useful,
        "interpretation": (
            "Changes in aggregate recall cannot be attributed to the resolver unless "
            "an accepted resolver node aligns to gold. Sparse gold cannot label every "
            "non-aligned addition false."
        ),
    }
    path = args.output or args.experimental.with_name("targeted_resolution_analysis.json")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
