"""Compare whole-world and independent-branch world generation on one binary case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from dotenv import load_dotenv

from global_workspace.branchwise_world_generation import merge_binary_branch_worlds
from global_workspace.local_specialists import (
    _admit_action_source_rows,
    ground_actions_in_scenario,
)
from global_workspace.openai_backend import OpenAIWorkspaceLLM
from global_workspace.scenario_semantics import segment_scenario_clauses


DEFAULT_SCENARIO = (
    "A rural clinic has one portable oxygen unit and two patients, Eli and Noor. "
    "Each patient will suffer respiratory injury during the next hour without "
    "the unit. The unit cannot be divided, and no replacement can arrive in time. "
    "The clinician must give the oxygen unit either to Eli or to Noor."
)
DEFAULT_ACTIONS = ["Give the oxygen unit to Eli", "Give the oxygen unit to Noor"]


def _world_for_merge(result: dict) -> dict:
    if result.get("status") == "COMMITTED" and isinstance(result.get("world_model"), dict):
        return result["world_model"]
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="o3")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--actions", nargs=2, default=DEFAULT_ACTIONS)
    parser.add_argument(
        "--output", type=Path,
        default=Path("eval_outputs/branchwise-world-head-to-head/report.json"),
    )
    args = parser.parse_args()
    load_dotenv()
    llm = OpenAIWorkspaceLLM(args.model, timeout=600.0)
    actions = [str(value) for value in args.actions]

    whole_started = time.monotonic()
    whole = ground_actions_in_scenario(
        llm, args.scenario, actions, max_tokens=256, max_attempts=3,
        allow_action_text_evidence=True,
        stage_one_guidance_mode="RAW_TEXT",
    )
    whole_seconds = round(time.monotonic() - whole_started, 3)

    branch_results: list[dict] = []
    branch_seconds: list[float] = []
    for action in actions:
        started = time.monotonic()
        result = ground_actions_in_scenario(
            llm, args.scenario, [action], max_tokens=256, max_attempts=3,
            allow_action_text_evidence=True,
            stage_one_guidance_mode="RAW_TEXT",
        )
        branch_results.append(result)
        branch_seconds.append(round(time.monotonic() - started, 3))

    merge_conflicts: list[dict] = []
    merged_admission: dict = {
        "status": "NOT_RUN", "errors": ["one or more branches did not commit"],
    }
    if all(result.get("status") == "COMMITTED" for result in branch_results):
        merged_world, merge_conflicts = merge_binary_branch_worlds([
            _world_for_merge(result) for result in branch_results
        ])
        clauses = segment_scenario_clauses(args.scenario)
        candidate = {
            "actions": {
                f"A{index}": {
                    "clause_ids": [row["clause_id"] for row in clauses],
                    "reason": "Branch generated independently from the complete source scenario.",
                }
                for index in range(2)
            },
            "world_model": merged_world,
            "ellipsis_resolutions": [],
        }
        merged_admission = _admit_action_source_rows(
            candidate, actions, ["A0", "A1"], clauses,
            allow_action_text_evidence=True,
        )

    report = {
        "experiment_version": "1.0",
        "model": args.model,
        "scenario": args.scenario,
        "actions": actions,
        "whole_world": {
            "status": whole.get("status"),
            "repair_attempts": whole.get("repair_attempts"),
            "errors": whole.get("errors") or [],
            "validation_issues": whole.get("validation_issues") or [],
            "elapsed_seconds": whole_seconds,
            "world_model": whole.get("world_model") or {},
        },
        "branchwise": {
            "branch_statuses": [result.get("status") for result in branch_results],
            "branch_repair_attempts": [
                result.get("repair_attempts") for result in branch_results
            ],
            "branch_errors": [result.get("errors") or [] for result in branch_results],
            "branch_elapsed_seconds": branch_seconds,
            "merge_conflicts": merge_conflicts,
            "merged_status": merged_admission.get("status"),
            "merged_errors": merged_admission.get("errors") or [],
            "merged_validation_issues": merged_admission.get("validation_issues") or [],
            "merged_world_model": merged_admission.get("world_model") or {},
        },
        "intentionally_not_compared": [
            "ethical framework deliberation",
            "final ethical recommendation",
            "counterfactual overlay quality",
            "cost-normalized performance beyond this single scenario",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "whole_world_status": report["whole_world"]["status"],
        "branch_statuses": report["branchwise"]["branch_statuses"],
        "merged_status": report["branchwise"]["merged_status"],
        "whole_world_seconds": whole_seconds,
        "branch_seconds": branch_seconds,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
