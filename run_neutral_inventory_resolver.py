"""Replay saved neutral inventories through only the targeted semantic resolver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from dotenv import load_dotenv

from global_workspace.openai_backend import OpenAIWorkspaceLLM
from global_workspace.pairwise_challenge_evaluation import diagnose_neutral_node_inventory
from global_workspace.scenario_semantics import segment_scenario_clauses
from global_workspace.semantic_resolution_obligations import (
    assess_obligation_coverage,
    build_resolution_obligations,
)
from global_workspace.structured_io import call_json_llm, extract_json
from global_workspace.syntactic_annotation import load_english_parser
from global_workspace.targeted_semantic_resolution import (
    apply_resolution_results,
    build_resolution_jobs,
    prompt_for_resolution_job,
    resolver_schema,
)


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="o3")
    parser.add_argument("--max-calls-per-case", type=int, default=3)
    args = parser.parse_args()
    load_dotenv()
    llm = OpenAIWorkspaceLLM(args.model, timeout=180.0)
    nlp = load_english_parser()
    report = _load(args.source_report)
    manifest = _load(report["manifest"])
    cases = {str(row["id"]): row for row in manifest.get("cases") or []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source_result in report.get("results") or []:
        case_id = str(source_result["id"])
        case = cases[case_id]
        stage_path = source_result.get("stage_one_artifact")
        if not stage_path or not Path(stage_path).is_file():
            results.append({"id": case_id, "status": "NO_SAVED_NEUTRAL_INVENTORY"})
            continue
        stage = _load(stage_path)
        neutral = stage.get("neutral_skeleton") or {}
        if not neutral.get("propositions"):
            results.append({"id": case_id, "status": "EMPTY_SAVED_NEUTRAL_INVENTORY"})
            continue
        clauses = segment_scenario_clauses(str(case["scenario"]))
        obligations = build_resolution_obligations(clauses, nlp=nlp)
        coverage = assess_obligation_coverage(obligations, neutral, nlp=nlp)
        jobs = build_resolution_jobs(obligations, neutral, clauses, coverage)[
            :max(0, args.max_calls_per_case)
        ]
        party_ids = [
            str(row.get("party_id") or "") for row in neutral.get("parties") or []
            if row.get("party_id")
        ]
        clause_ids = [str(row.get("clause_id") or "") for row in clauses]
        responses = []
        started = time.monotonic()
        for job in jobs:
            output = call_json_llm(
                llm, prompt_for_resolution_job(job), max_tokens=1024,
                temperature=0.0, schema=resolver_schema(party_ids, clause_ids),
                call_kind="fixed_neutral_targeted_resolution",
                call_metadata={
                    "case_id": case_id,
                    "obligation_id": (job.get("obligation") or {}).get("obligation_id"),
                },
            )
            raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
            responses.append(extract_json(raw))
        resolved, audit = apply_resolution_results(neutral, jobs, responses)
        before = diagnose_neutral_node_inventory(neutral, case.get("gold_pairs") or [])
        after = diagnose_neutral_node_inventory(resolved, case.get("gold_pairs") or [])
        before_ids = {
            str(row.get("proposition_id") or "") for row in neutral.get("propositions") or []
        }
        after_alignments = after.get("alignments") or []
        useful_added_ids = sorted({
            str(row.get("matched_proposition_id") or "").split("__NEUTRAL__", 1)[0]
            for row in after_alignments
            if row.get("status") == "MATCHED"
            and str(row.get("matched_proposition_id") or "").split("__NEUTRAL__", 1)[0]
            not in before_ids
        })
        accepted_ids = sorted({
            str(value) for row in audit for value in row.get("accepted_proposition_ids") or []
        })
        artifact = {
            "id": case_id,
            "source_stage_artifact": stage_path,
            "obligations": obligations,
            "coverage_before": coverage,
            "jobs": jobs,
            "responses": responses,
            "audit": audit,
            "neutral_before": neutral,
            "neutral_after": resolved,
            "diagnostic_before": before,
            "diagnostic_after": after,
            "accepted_addition_ids": accepted_ids,
            "gold_useful_added_ids": useful_added_ids,
        }
        artifact_path = args.output_dir / f"{case_id}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        results.append({
            "id": case_id, "status": "COMPLETE", "artifact": str(artifact_path),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "gold_nodes_before": before["present_neutral_node_count"],
            "gold_nodes_after": after["present_neutral_node_count"],
            "accepted_addition_count": len(accepted_ids),
            "gold_useful_addition_count": len(useful_added_ids),
            "resolver_call_count": len(jobs),
        })
        print(json.dumps(results[-1], sort_keys=True), flush=True)
    completed = [row for row in results if row.get("status") == "COMPLETE"]
    gold_before = sum(int(row["gold_nodes_before"]) for row in completed)
    gold_after = sum(int(row["gold_nodes_after"]) for row in completed)
    accepted = sum(int(row["accepted_addition_count"]) for row in completed)
    useful = sum(int(row["gold_useful_addition_count"]) for row in completed)
    output = {
        "protocol": "FIXED_NEUTRAL_INVENTORY_TARGETED_RESOLVER",
        "source_report": str(args.source_report), "model": args.model,
        "aggregate": {
            "case_count": len(results), "completed_case_count": len(completed),
            "resolver_call_count": sum(int(row["resolver_call_count"]) for row in completed),
            "gold_nodes_before": gold_before, "gold_nodes_after": gold_after,
            "gold_node_gain": gold_after - gold_before,
            "accepted_addition_count": accepted,
            "gold_useful_addition_count": useful,
            "resolver_gold_useful_precision_lower_bound": useful / accepted if accepted else None,
            "precision_status": "LOWER_BOUND_SPARSE_GOLD" if accepted else "NO_ACCEPTED_ADDITIONS",
        },
        "results": results,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2), flush=True)
    print(f"Saved: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
