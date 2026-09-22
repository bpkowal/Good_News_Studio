"""Run the frozen end-to-end world-graph preservation benchmark."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import re
from typing import Any

from global_workspace.graph_preservation_evaluation import (
    score_trace,
    trace_from_unadmitted_candidate,
)
from run_pairwise_challenge import canonical_action_mapping


TRACE_PATH = Path("workspace_outputs/semantic_preservation_trace.json")
_DIAGNOSTIC_PATH = re.compile(r"^Full grounding diagnostic:\s*(.+)$", re.MULTILINE)


def _remap_case(case: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    remapped = copy.deepcopy(case)
    for collection in ("gold_facts", "gold_edges", "gold_projection"):
        for row in remapped.get(collection) or []:
            action_id = str(row.get("action_id") or "")
            row["action_id"] = mapping.get(action_id, action_id)
    return remapped


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row["scores"] for row in results if isinstance(row.get("scores"), dict)]
    rejected_scored = [
        row["rejected_candidate_scores"] for row in results
        if isinstance(row.get("rejected_candidate_scores"), dict)
    ]
    fact_total = sum(row["source_to_graph"]["gold_fact_count"] for row in scored)
    fact_present = sum(row["source_to_graph"]["present_count"] for row in scored)
    edge_aligned = sum(row["topology"]["aligned_edge_count"] for row in scored)
    edge_correct = sum(
        sum(item.get("correct") is True for item in row["topology"]["edges"])
        for row in scored if row["topology"]["measurement_status"] == "MEASURED"
    )
    roles = sum(row["projection"]["gold_role_count"] for row in scored)
    roles_correct = sum(
        sum(item["correct"] for item in row["projection"]["roles"])
        for row in scored
    )
    return {
        "case_count": len(results),
        "committed_count": sum(row.get("world_committed", False) for row in results),
        "scored_count": len(scored),
        "gold_fact_count": fact_total,
        "source_to_graph_recall": fact_present / fact_total if fact_total else None,
        "aligned_gold_edge_count": edge_aligned,
        "topology_accuracy_given_aligned_endpoints": (
            edge_correct / edge_aligned if edge_aligned else None
        ),
        "gold_role_count": roles,
        "projection_accuracy_given_aligned_graph": roles_correct / roles if roles else None,
        "rejected_candidate_scored_count": len(rejected_scored),
        "rejected_candidate_source_to_graph_recall": (
            sum(row["source_to_graph"]["present_count"] for row in rejected_scored)
            / sum(row["source_to_graph"]["gold_fact_count"] for row in rejected_scored)
            if rejected_scored else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evals/world_graph_preservation_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/world-graph-preservation"))
    parser.add_argument("--model", default="o3")
    parser.add_argument("--world-escalation", action="store_true")
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = list(manifest.get("cases") or [])
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [row for row in cases if str(row.get("id")) in selected]
    if args.count is not None:
        cases = cases[:max(0, args.count)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        case_id = str(case["id"])
        prior_trace_mtime = TRACE_PATH.stat().st_mtime_ns if TRACE_PATH.is_file() else None
        command = [
            sys.executable, "parliament.py", "--mode", "workspace",
            "--question", str(case["scenario"]), "--backend", "openai",
            "--openai-model", args.model, "--actions",
            *[str(value) for value in case.get("actions") or []],
            "--accept-actions", "--stop-after-world", "--skip-original-agents",
            "--no-rag", "--no-framing-cache",
            "--max-cycles", "1",
            "--performance-output", str(args.output_dir / f"{index:02d}_{case_id}.performance.json"),
        ]
        command.append(
            "--escalate-world-model" if args.world_escalation
            else "--no-world-escalation"
        )
        started = time.monotonic()
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        output = completed.stdout or ""
        (args.output_dir / f"{index:02d}_{case_id}.log").write_text(output, encoding="utf-8")
        mapping = canonical_action_mapping(output, list(case.get("actions") or []))
        trace = None
        trace_target = args.output_dir / f"{index:02d}_{case_id}.semantic_trace.json"
        if TRACE_PATH.is_file():
            current_trace_bytes = TRACE_PATH.read_bytes()
            candidate = json.loads(current_trace_bytes.decode("utf-8"))
            if (
                TRACE_PATH.stat().st_mtime_ns != prior_trace_mtime
                and str(candidate.get("status") or "").upper() == "COMMITTED"
            ):
                trace = candidate
                trace_target.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        scores = score_trace(_remap_case(case, mapping), trace) if trace else None
        rejected_candidate_scores = None
        diagnostic_path = None
        diagnostic_match = _DIAGNOSTIC_PATH.search(output)
        if diagnostic_match is not None:
            diagnostic = Path(diagnostic_match.group(1).strip())
            diagnostic_path = str(diagnostic)
            if diagnostic.is_file():
                payload = json.loads(diagnostic.read_text(encoding="utf-8"))
                grounding = payload.get("grounding") or {}
                candidate = grounding.get("rejected_candidate") or {}
                if isinstance(candidate, dict) and candidate.get("world_model"):
                    rejected_trace = trace_from_unadmitted_candidate(candidate)
                    rejected_candidate_scores = score_trace(
                        _remap_case(case, mapping), rejected_trace,
                    )
        result = {
            "id": case_id, "family": case.get("family"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "exit_code": completed.returncode,
            "world_committed": trace is not None and "status: COMMITTED" in output,
            "presentation_to_canonical_action_ids": mapping,
            "scores": scores,
            "rejected_candidate_scores": rejected_candidate_scores,
            "grounding_diagnostic": diagnostic_path,
            "log": str(args.output_dir / f"{index:02d}_{case_id}.log"),
            "semantic_trace": str(trace_target) if trace else None,
        }
        results.append(result)
        print(json.dumps({
            "id": case_id, "committed": result["world_committed"],
            "source_recall": scores["source_to_graph"]["recall"] if scores else None,
            "topology_accuracy": scores["topology"]["accuracy_given_aligned_endpoints"] if scores else None,
            "projection_accuracy": scores["projection"]["accuracy_given_gold_graph"] if scores else None,
        }, sort_keys=True), flush=True)
    report = {
        "report_version": "1.0", "protocol": "FROZEN_WORLD_GRAPH_PRESERVATION_V01",
        "manifest": str(args.manifest), "model": args.model,
        "world_escalation": bool(args.world_escalation),
        "aggregate": _aggregate(results), "results": results,
    }
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Graph preservation report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
