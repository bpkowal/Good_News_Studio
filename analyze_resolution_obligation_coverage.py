"""Recompute conservative spaCy-obligation coverage from saved Stage-1 artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from global_workspace.semantic_resolution_obligations import assess_obligation_coverage
from global_workspace.syntactic_annotation import load_english_parser


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    nlp = load_english_parser()
    cases = []
    statuses: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for result in report.get("results") or []:
        artifact_path = result.get("stage_one_artifact")
        if not artifact_path or not Path(artifact_path).is_file():
            continue
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        obligations = artifact.get("syntactic_resolution_obligations") or {}
        coverage = assess_obligation_coverage(
            obligations, artifact.get("neutral_skeleton") or {}, nlp=nlp,
        )
        for row in coverage:
            statuses[str(row["status"])] += 1
            types[f"{row['type']}::{row['status']}"] += 1
        cases.append({"id": result.get("id"), "coverage": coverage})
    output = {
        "source_report": str(args.report),
        "policy": "SYNTAX_IDENTIFIES_REVIEW_LOCATIONS_BUT_NEVER_ATTESTS_RESOLUTION",
        "summary": {
            "obligation_count": sum(statuses.values()),
            "status_counts": dict(sorted(statuses.items())),
            "type_status_counts": dict(sorted(types.items())),
        },
        "cases": cases,
    }
    path = args.output or args.report.with_name("semantic_resolution_coverage.json")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
