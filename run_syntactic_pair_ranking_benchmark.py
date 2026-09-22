"""Compare gold-pair retention before any pairwise LLM judgment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from global_workspace.pairwise_relation_audit import build_pairwise_jobs
from global_workspace.syntactic_annotation import annotate_text, load_english_parser
from global_workspace.syntactic_pair_ranking import rank_jobs


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _key(job: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(job.get("action_id") or ""),
        str((job.get("source") or {}).get("proposition_id") or ""),
        str((job.get("target") or {}).get("proposition_id") or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=Path("eval_outputs/staged-provenance-challenge-frozen-20260920/report.json"))
    parser.add_argument("--manifest", type=Path, default=Path("evals/pairwise_relation_challenge_set.json"))
    parser.add_argument("--output", type=Path, default=Path("eval_outputs/syntactic-pair-ranking-frozen-20260920/report.json"))
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--exploration-count", type=int, default=3)
    args = parser.parse_args()

    source_report = _load(args.source_report)
    manifest = _load(args.manifest)
    cases_by_id = {str(row["id"]): row for row in manifest.get("cases") or []}
    nlp = load_english_parser()
    results = []
    totals = {name: 0 for name in (
        "gold", "baseline", "evidence", "syntactic", "candidates", "selected", "exploration",
    )}
    for source_result in source_report.get("results") or []:
        details = (source_result.get("metrics") or {}).get("details") or []
        if not details or not source_result.get("stage_one_artifact"):
            continue
        case = cases_by_id[str(source_result["id"])]
        stage = _load(source_result["stage_one_artifact"])
        skeleton = stage.get("skeleton") or {}
        annotation = annotate_text(str(case["scenario"]), nlp=nlp)
        clauses = [
            {"clause_id": f"C{row['sentence_index']}", "text": row["text"]}
            for row in annotation["sentences"]
        ]
        baseline_jobs = build_pairwise_jobs(skeleton, clauses, stage.get("topology_scaffold") or {})
        evidence_jobs = build_pairwise_jobs(
            skeleton, clauses, stage.get("topology_scaffold") or {}, evidence_augmented=True,
        )
        syntactic_jobs, selection = rank_jobs(
            evidence_jobs, annotation, nlp=nlp, budget=args.budget,
            exploration_count=args.exploration_count,
        )
        gold = {
            (str(row["action_id"]), str(row["source_proposition_id"]), str(row["target_proposition_id"]))
            for row in details
        }
        retained = {
            "baseline": gold & {_key(row) for row in baseline_jobs[:args.budget]},
            "evidence": gold & {_key(row) for row in evidence_jobs[:args.budget]},
            "syntactic": gold & {_key(row) for row in syntactic_jobs},
        }
        totals["gold"] += len(gold)
        totals["candidates"] += len(evidence_jobs)
        totals["selected"] += int(selection["budget"])
        totals["exploration"] += int(selection["exploration_count"])
        for name in retained:
            totals[name] += len(retained[name])
        results.append({
            "id": case["id"],
            "gold_pair_count": len(gold),
            "candidate_pair_count": len(evidence_jobs),
            "retained": {name: len(rows) for name, rows in retained.items()},
            "syntactic_selection": selection,
        })
    report = {
        "report_version": "1.0",
        "protocol": "PRE_LLM_GOLD_PAIR_RETENTION",
        "source_report": str(args.source_report),
        "budget": args.budget,
        "exploration_count": args.exploration_count,
        "aggregate": {
            "evaluable_gold_pair_count": totals["gold"],
            "candidate_pair_count": totals["candidates"],
            "selected_pair_count": totals["selected"],
            "candidate_reduction": (
                1 - totals["selected"] / totals["candidates"]
                if totals["candidates"] else None
            ),
            "exploration_pair_count": totals["exploration"],
            **{
                f"{name}_retained_count": totals[name]
                for name in ("baseline", "evidence", "syntactic")
            },
            **{
                f"{name}_retention": totals[name] / totals["gold"] if totals["gold"] else None
                for name in ("baseline", "evidence", "syntactic")
            },
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
