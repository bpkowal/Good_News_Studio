"""Run one Study 1 perturbation family and compare response-state changes only."""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from comparative_ethics_eval import (
    load_cases,
    read_json,
    run_parliament,
    run_solo,
    usage_totals,
    write_json,
)
from study1_metrics import (
    response_distance,
    response_vector_from_parliament,
    response_vector_from_solo,
)


def _case(raw: dict[str, Any], family: str) -> dict[str, Any]:
    return {
        "id": f"{family}__{raw['id']}",
        "question": raw["question"],
        "actions": raw["actions"],
        "explicit_facts": [],
        "forbidden_assumptions": [],
        "catastrophic_risks": [],
        "variant_id": raw.get("id", "canonical"),
        "variant_type": raw.get("type", "CANONICAL"),
        "target": raw.get("target", ""),
    }


def _args(parsed: argparse.Namespace) -> Namespace:
    return Namespace(
        model=parsed.model,
        judge_model=parsed.model,
        max_cycles=parsed.max_cycles,
        time_budget=parsed.time_budget,
        agent_timeout=parsed.agent_timeout,
        workspace_call_timeout=parsed.workspace_call_timeout,
        parliament_process_timeout=parsed.parliament_process_timeout,
        delegate_tokens=parsed.delegate_tokens,
        max_solo_completion_tokens=parsed.max_solo_completion_tokens,
        solo_reasoning_effort=parsed.solo_reasoning_effort,
        force=parsed.force,
        structured_solo=True,
        escalate_world_model=parsed.escalate_world_model,
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evals/study1_trolley_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/study1-trolley-pilot"))
    parser.add_argument("--model", default="o3")
    parser.add_argument("--execute", action="store_true", help="run paid model calls")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reduced", action="store_true", help="Skip the action-order nuisance variant")
    parser.add_argument("--escalate-world-model", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--time-budget", type=float, default=600.0)
    parser.add_argument("--agent-timeout", type=float, default=600.0)
    parser.add_argument("--workspace-call-timeout", type=float, default=90.0)
    parser.add_argument("--parliament-process-timeout", type=float, default=0.0)
    parser.add_argument("--delegate-tokens", type=int, default=128)
    parser.add_argument("--max-solo-completion-tokens", type=int, default=100_000)
    parser.add_argument("--solo-reasoning-effort", choices=("low", "medium", "high"), default="high")
    parsed = parser.parse_args()
    manifest = read_json(parsed.manifest)
    family = str(manifest["family_id"])
    raw_cases = [{"id": "canonical", **manifest["canonical"]}, *manifest["variants"]]
    cases = [_case(item, family) for item in raw_cases]
    if parsed.reduced:
        cases = [case for case in cases if case["variant_id"] != "nuisance_action_order"]
    if not parsed.execute:
        print(f"Validated {family}: {len(cases)} response-state cases; no API calls made.")
        return 0
    args = _args(parsed)
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    vectors: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases, 1):
        case_dir = parsed.output_dir / case["variant_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(cases)}] {case['variant_id']}: Parliament", flush=True)
        parliament, parliament_usage = run_parliament(case, case_dir, args)
        print(f"[{index}/{len(cases)}] {case['variant_id']}: structured solo", flush=True)
        solo, solo_usage, solo_budget = run_solo(case, case_dir, args, parliament_usage)
        parliament_vector = response_vector_from_parliament(parliament, case["actions"])
        solo_vector = response_vector_from_solo(solo["state"], case["actions"])
        vectors[case["variant_id"]] = {
            "type": case["variant_type"],
            "target": case["target"],
            "actions": case["actions"],
            "parliament": parliament_vector.as_dict(),
            "solo": solo_vector.as_dict(),
            "parliament_usage": parliament_usage,
            "solo_usage": solo_usage,
            "solo_budget": solo_budget,
        }
        write_json(case_dir / "response_vectors.json", vectors[case["variant_id"]])
    canonical = vectors["canonical"]
    rows: list[dict[str, Any]] = []
    for variant_id, row in vectors.items():
        if variant_id == "canonical":
            continue
        from study1_metrics import ResponseVector
        parliament_base = ResponseVector(**canonical["parliament"])
        solo_base = ResponseVector(**canonical["solo"])
        parliament_variant = ResponseVector(**row["parliament"])
        solo_variant = ResponseVector(**row["solo"])
        rows.append({
            "variant_id": variant_id,
            "type": row["type"],
            "target": row["target"],
            "parliament_distance": response_distance(parliament_base, parliament_variant),
            "solo_distance": response_distance(solo_base, solo_variant),
        })
    report = {
        "family_id": family,
        "measurement": "within-system response-state distance; no quality judgment",
        "variants": rows,
        "raw_vectors": vectors,
    }
    write_json(parsed.output_dir / "study1_response_state_report.json", report)
    print(f"Saved response-state report: {parsed.output_dir / 'study1_response_state_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
