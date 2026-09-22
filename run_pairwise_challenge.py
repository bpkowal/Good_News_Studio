"""Run the frozen pairwise-topology challenge set and aggregate its metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from global_workspace.pairwise_challenge_evaluation import (
    align_gold_events,
    score_aligned_challenge,
)


_STAGE_PATH = re.compile(r"^(?:Stage-1 audit artifact):\s*(.+)$", re.MULTILINE)
_AUDIT_PATH = re.compile(r"^(?:Pairwise audit artifact):\s*(.+)$", re.MULTILINE)
_CANONICAL_ACTIONS = re.compile(r"^Canonical deliberation IDs:\s*(.+)$", re.MULTILINE)
_WORLD_BLOCK = re.compile(
    r"--- Admitted world ---\s*(.*?)(?=\n--- Figure|\nWorld grounding accepted|\Z)",
    re.DOTALL,
)
_ACTION_HEADING = re.compile(r"^(A\d+):\s+", re.MULTILINE)
_EFFECT_LINE = re.compile(
    r"^\s+([A-Za-z0-9_]+)\s+(?:DIRECT|DOWNSTREAM|FOREGONE)\s+"
    r"(?:[A-Z_]+(?:/[A-Z_]+)*)\s+(.+?)\s+\[[^\]]+\](?:\s+.*)?$"
)
_EDGE_LINE = re.compile(
    r"^\s+([A-Za-z0-9_]+)\s+(CAUSES|ENABLES|PREVENTS)\s+([A-Za-z0-9_]+)"
)


def _fold(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def canonical_action_mapping(
    output: str, presentation_actions: list[str],
) -> dict[str, str]:
    """Map manifest/list-order action IDs onto pipeline canonical IDs."""
    match = _CANONICAL_ACTIONS.search(output)
    canonical: dict[str, str] = {}
    if match:
        for item in match.group(1).split(";"):
            key, separator, label = item.strip().partition("=")
            if separator and key.strip():
                canonical[key.strip()] = label.strip()
    mapping: dict[str, str] = {}
    for index, action in enumerate(presentation_actions):
        presentation_id = f"A{index}"
        folded = _fold(action)
        matches = [
            action_id for action_id, label in canonical.items()
            if folded == _fold(label)
            or folded in _fold(label) or _fold(label) in folded
        ]
        if len(matches) == 1:
            mapping[presentation_id] = matches[0]
    return mapping


def remap_gold_pairs(
    gold_pairs: list[dict[str, Any]], mapping: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {**row, "action_id": mapping.get(str(row.get("action_id") or ""), str(row.get("action_id") or ""))}
        for row in gold_pairs
    ]


def _read_artifact(pattern: re.Pattern[str], output: str) -> tuple[dict[str, Any], str | None]:
    match = pattern.search(output)
    if match is None:
        return {}, None
    path = Path(match.group(1).strip())
    if not path.is_file():
        return {}, str(path)
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def parse_admitted_world_for_evaluation(output: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover the printed admitted graph for benchmark-only scoring."""
    match = _WORLD_BLOCK.search(output)
    if match is None:
        return {"propositions": []}, {"records": []}
    current_action = ""
    propositions: list[dict[str, Any]] = []
    edges: list[tuple[str, str, str, str]] = []
    for line in match.group(1).splitlines():
        heading = _ACTION_HEADING.match(line)
        if heading:
            current_action = heading.group(1)
            continue
        effect = _EFFECT_LINE.match(line)
        if effect and current_action:
            propositions.append({
                "proposition_id": effect.group(1),
                "action_id": current_action,
                "outcome": effect.group(2).strip(),
                "source_proposition": effect.group(2).strip(),
            })
            continue
        edge = _EDGE_LINE.match(line)
        if edge and current_action:
            edges.append((current_action, edge.group(1), edge.group(3), edge.group(2)))
    records = [{
        "status": "JUDGED",
        "action_id": action_id,
        "source_proposition_id": source_id,
        "target_proposition_id": target_id,
        "relation": "NONE",
        "scaffold_relations": [relation],
    } for action_id, source_id, target_id, relation in edges]
    return {"propositions": propositions}, {"records": records}


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["metrics"] for row in results if isinstance(row.get("metrics"), dict)]
    final_metrics = [
        row["final_graph_metrics"] for row in results
        if isinstance(row.get("final_graph_metrics"), dict)
    ]
    totals = {
        key: sum(int(item.get(key) or 0) for item in metrics)
        for key in (
            "gold_node_count", "aligned_gold_node_count", "gold_edge_count",
            "aligned_gold_edge_count", "gold_relations_absent_from_scaffold",
            "correct_audit_only_relations", "audit_only_relation_count",
            "correct_audit_only_relation_count",
        )
    }
    pairwise_correct = sum(
        sum(bool(detail.get("audit_correct")) for detail in item.get("details") or [])
        for item in metrics
    )
    scaffold_correct = sum(
        sum(bool(detail.get("scaffold_correct")) for detail in item.get("details") or [])
        for item in metrics
    )
    aligned_edges = totals["aligned_gold_edge_count"]
    precision_identified = bool(metrics) and all(
        item.get("audit_only_precision_status") == "MEASURED" for item in metrics
    )
    final_gold_nodes = sum(int(item.get("gold_node_count") or 0) for item in final_metrics)
    final_aligned_nodes = sum(
        int(item.get("aligned_gold_node_count") or 0) for item in final_metrics
    )
    final_aligned_edges = sum(
        int(item.get("aligned_gold_edge_count") or 0) for item in final_metrics
    )
    final_correct_edges = sum(
        sum(bool(detail.get("scaffold_correct")) for detail in item.get("details") or [])
        for item in final_metrics
    )
    return {
        "case_count": len(results),
        "world_committed_count": sum(bool(row.get("world_committed")) for row in results),
        "final_graph_gold_node_count": final_gold_nodes,
        "final_graph_aligned_gold_node_count": final_aligned_nodes,
        "final_graph_node_recall": (
            final_aligned_nodes / final_gold_nodes if final_gold_nodes else None
        ),
        "final_graph_aligned_gold_edge_count": final_aligned_edges,
        "final_graph_relation_accuracy_given_aligned_nodes": (
            final_correct_edges / final_aligned_edges if final_aligned_edges else None
        ),
        "stage_one_admitted_count": sum(
            str(row.get("stage_one_status") or "").upper()
            in {"ADMITTED", "PARTIALLY_ADMITTED"}
            for row in results
        ),
        "stage_one_partially_admitted_count": sum(
            str(row.get("stage_one_status") or "").upper()
            == "PARTIALLY_ADMITTED"
            for row in results
        ),
        "audit_complete_count": sum(
            str(row.get("audit_status") or "").upper() == "COMPLETE"
            for row in results
        ),
        **totals,
        "node_recall": (
            totals["aligned_gold_node_count"] / totals["gold_node_count"]
            if totals["gold_node_count"] else None
        ),
        "pairwise_gold_accuracy_given_aligned_nodes": (
            pairwise_correct / aligned_edges if aligned_edges else None
        ),
        "scaffold_gold_accuracy_given_aligned_nodes": (
            scaffold_correct / aligned_edges if aligned_edges else None
        ),
        "audit_novelty_rate": (
            totals["correct_audit_only_relations"]
            / totals["gold_relations_absent_from_scaffold"]
            if totals["gold_relations_absent_from_scaffold"] else None
        ),
        "audit_only_precision": (
            totals["correct_audit_only_relation_count"]
            / totals["audit_only_relation_count"]
            if precision_identified and totals["audit_only_relation_count"] else None
        ),
        "audit_only_precision_status": (
            "MEASURED" if precision_identified
            else "NOT_IDENTIFIED_SPARSE_GOLD"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("evals/pairwise_relation_challenge_set.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("eval_outputs/pairwise-relation-challenge"),
    )
    parser.add_argument("--model", default="o3")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument(
        "--evidence-augmented", action="store_true",
        help="Use label-blind structured evidence and source-local pair ranking",
    )
    parser.add_argument(
        "--syntactic-resolution-obligations", action="store_true",
    )
    parser.add_argument("--targeted-semantic-resolution", action="store_true")
    parser.add_argument("--targeted-resolution-max-calls", type=int, default=6)
    parser.add_argument(
        "--stage-one-guidance-mode",
        choices=("authoritative", "evidence-only", "evidence-review", "raw-text"),
        default="authoritative",
    )
    parser.add_argument(
        "--deterministic-repair-guidance",
        choices=("off", "alone", "combined"),
        default="off",
    )
    parser.add_argument(
        "--case-ids", nargs="*",
        help="Optional exact case IDs; preserves manifest order",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_cases = list(manifest.get("cases") or [])
    if args.case_ids:
        selected = set(args.case_ids)
        all_cases = [case for case in all_cases if str(case.get("id")) in selected]
    start = max(0, args.start)
    cases = all_cases[start:] if args.count is None else all_cases[
        start:start + max(0, args.count)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=start):
        case_id = str(case["id"])
        log_path = args.output_dir / f"{index + 1:02d}_{case_id}.log"
        performance_path = args.output_dir / f"{index + 1:02d}_{case_id}.performance.json"
        command = [
            sys.executable, "parliament.py", "--mode", "workspace",
            "--question", str(case["scenario"]), "--backend", "openai",
            "--openai-model", args.model,
            "--actions", *[str(value) for value in case.get("actions") or []],
            "--accept-actions", "--stop-after-world", "--skip-original-agents",
            "--no-rag", "--no-framing-cache", "--no-world-escalation",
            "--max-cycles", "1",
            "--performance-output", str(performance_path),
        ]
        if args.stage_one_guidance_mode != "raw-text":
            command.extend([
                "--pairwise-relation-audit", "--staged-node-generation",
                "--stage-one-guidance-mode", args.stage_one_guidance_mode,
                "--deterministic-repair-guidance",
                args.deterministic_repair_guidance,
                "--pairwise-audit-max-pairs", str(max(0, args.max_pairs)),
            ])
        if args.evidence_augmented:
            command.append("--pairwise-audit-evidence-augmented")
        if args.syntactic_resolution_obligations:
            command.append("--syntactic-resolution-obligations")
        if args.targeted_semantic_resolution:
            command.append("--targeted-semantic-resolution")
        command.extend([
            "--targeted-resolution-max-calls",
            str(max(0, args.targeted_resolution_max_calls)),
        ])
        started = time.monotonic()
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        elapsed = round(time.monotonic() - started, 3)
        output = completed.stdout or ""
        log_path.write_text(output, encoding="utf-8")
        stage_one, stage_path = _read_artifact(_STAGE_PATH, output)
        audit, audit_path = _read_artifact(_AUDIT_PATH, output)
        # When Stage 1 is admitted, the pairwise artifact is separate and the
        # log intentionally prints only that path. Its skeleton remains in the
        # adjacent stage artifact, which is always persisted in audit mode.
        skeleton = stage_one.get("skeleton") if isinstance(stage_one, dict) else {}
        action_mapping = canonical_action_mapping(
            output, [str(value) for value in case.get("actions") or []],
        )
        gold_pairs = remap_gold_pairs(
            [dict(row) for row in case.get("gold_pairs") or []], action_mapping,
        )
        alignments = align_gold_events(
            skeleton or {}, gold_pairs,
        )
        metrics = score_aligned_challenge(
            audit or {"records": []}, gold_pairs, alignments,
            gold_edge_set_complete=bool(case.get("gold_edge_set_complete", False)),
        )
        final_skeleton, final_relation_records = parse_admitted_world_for_evaluation(output)
        final_alignments = align_gold_events(final_skeleton, gold_pairs)
        final_metrics = score_aligned_challenge(
            final_relation_records, gold_pairs, final_alignments,
            gold_edge_set_complete=bool(case.get("gold_edge_set_complete", False)),
        )
        result = {
            "index": index,
            "id": case_id,
            "family": case.get("family"),
            "elapsed_seconds": elapsed,
            "exit_code": completed.returncode,
            "world_committed": "status: COMMITTED" in output,
            "stage_one_status": stage_one.get("status") if stage_one else "MISSING",
            "audit_status": audit.get("status") if audit else "NOT_RUN",
            "stage_one_artifact": stage_path,
            "audit_artifact": audit_path,
            "log": str(log_path),
            "performance": str(performance_path),
            "presentation_to_canonical_action_ids": action_mapping,
            "remapped_gold_pairs": gold_pairs,
            "alignments": alignments,
            "metrics": metrics,
            "final_graph_alignments": final_alignments,
            "final_graph_metrics": final_metrics,
        }
        results.append(result)
        print(json.dumps({
            "id": case_id, "stage_one": result["stage_one_status"],
            "audit": result["audit_status"], "node_recall": metrics["node_recall"],
            "pairwise_accuracy": metrics["pairwise_gold_accuracy_given_aligned_nodes"],
        }, sort_keys=True), flush=True)

    report = {
        "report_version": "1.0",
        "protocol": (
            "FROZEN_FIRST_EIGHT_CASE_CHALLENGE"
            if start == 0 and len(cases) == len(all_cases)
            else "STAGED_NODE_GENERATION_PILOT"
        ),
        "manifest": str(args.manifest),
        "model": args.model,
        "max_pairs": max(0, args.max_pairs),
        "evidence_augmented": bool(args.evidence_augmented),
        "syntactic_resolution_obligations": bool(
            args.syntactic_resolution_obligations
        ),
        "targeted_semantic_resolution": bool(args.targeted_semantic_resolution),
        "stage_one_guidance_mode": args.stage_one_guidance_mode,
        "deterministic_repair_guidance": args.deterministic_repair_guidance,
        "aggregate": _aggregate(results),
        "results": results,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Challenge report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
