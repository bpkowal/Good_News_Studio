"""Extract T0-T3 representation artifacts from a saved workspace trace.

This is a read-only diagnostic for deliberation semantics.  It performs no
model calls and does not affect workspace scoring, salience, or execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from global_workspace.models import WorkspaceBroadcast


FRAMEWORK_LEDGER_FIELDS = (
    "utilitarian_ledger_proposal",
    "deontological_ledger_proposal",
    "virtue_character_proposal",
    "care_ledger_proposal",
    "rawls_position_proposal",
)

REASONING_FIELDS = (
    "recommended_action",
    "rationale",
    "decision_rule",
    "framework_action_map",
    "framework_internal_conflicts",
    "framework_specific_open_questions",
    "framework_insights",
    "unsupported_assumption",
    "reversal_condition",
    "factual_reversal_threshold",
    "normative_reversal_threshold",
    "supporting_proposition_ids",
    "decision_critical_proposition_ids",
    "material_empirical_claims",
    "unresolved",
    "assumption_status",
    "choice_status",
    "adjudication_status",
    "landscape_cases",
    "landscape_decisive_axis",
    "landscape_tiebreaker",
    "landscape_tiebreaker_failure",
)


def _select(mapping: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: mapping.get(field) for field in fields if field in mapping}


def _compact_testimony(testimony: str, limit: int = 900) -> str:
    compact = " ".join(str(testimony).split())
    if len(compact) <= limit:
        return compact
    tail_size = max(1, limit // 2)
    head_size = max(1, limit - tail_size - 5)
    return f"{compact[:head_size]} ... {compact[-tail_size:]}"


def _candidate(cycle: dict[str, Any], specialist: str) -> dict[str, Any]:
    return next(
        (
            dict(candidate)
            for candidate in cycle.get("candidates", [])
            if candidate.get("specialist") == specialist
        ),
        {},
    )


def _state_item(
    problem_state: dict[str, Any], collection: str, key: str, specialist: str,
) -> dict[str, Any]:
    return next(
        (
            dict(item)
            for item in problem_state.get(collection, [])
            if item.get(key) == specialist
        ),
        {},
    )


def build_artifacts(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return exact per-specialist T0-T3 views from the first recurrence."""
    base_cycles = [
        dict(cycle) for cycle in trace.get("cycles", [])
        if not cycle.get("is_hypothetical", False)
    ]
    if len(base_cycles) < 2:
        raise ValueError("the trace needs at least two ordinary cycles")
    first, second = base_cycles[:2]
    outgoing = dict(first.get("broadcast", {}))
    problem_state = dict(outgoing.get("problem_state", {}))
    delivered_compact = WorkspaceBroadcast(**outgoing).compact()
    specialists = list(trace.get("active_specialists", [])) or list(
        trace.get("source_testimonies", {})
    )

    artifacts: dict[str, dict[str, Any]] = {}
    for specialist in specialists:
        t1_candidate = _candidate(first, specialist)
        t3_candidate = _candidate(second, specialist)
        testimony = trace.get("source_testimonies", {}).get(specialist, "")
        assigned_agenda = [
            dict(item) for item in outgoing.get("challenge_agenda", [])
            if specialist.casefold() in {
                str(target).casefold()
                for target in item.get("target_specialists", [])
            }
        ]
        ledgers_t1 = {
            field: t1_candidate[field]
            for field in FRAMEWORK_LEDGER_FIELDS
            if t1_candidate.get(field)
        }
        ledgers_t3 = {
            field: t3_candidate[field]
            for field in FRAMEWORK_LEDGER_FIELDS
            if t3_candidate.get(field)
        }
        artifacts[specialist] = {
            "T0": {
                "stage": "original_specialist_testimony",
                "testimony": testimony,
                "testimony_delivered_to_recurrent_prompt": _compact_testimony(testimony),
                "frozen_baseline": trace.get("source_baselines", {}).get(specialist, {}),
            },
            "T1": {
                "stage": "cycle_1_structured_framework_representation",
                "reasoning": _select(t1_candidate, REASONING_FIELDS),
                "framework_ledgers": ledgers_t1,
                "committed_framework_state": t1_candidate.get(
                    "committed_framework_state", {}
                ),
                "committed_native_ledger": t1_candidate.get(
                    "committed_native_ledger", {}
                ),
            },
            "T2": {
                "stage": "cycle_1_outgoing_shared_workspace",
                "structured_agent_position": _state_item(
                    problem_state, "agent_positions", "specialist", specialist
                ),
                "structured_workspace_contribution": _state_item(
                    problem_state, "workspace_contributions", "agent", specialist
                ),
                "delivered_compact_workspace": delivered_compact,
                "assigned_challenge_delivered_separately": assigned_agenda,
                "specialist_name_visible_in_compact": specialist in delivered_compact,
            },
            "T3": {
                "stage": "cycle_2_reconstructed_framework_reasoning",
                "reasoning": _select(t3_candidate, REASONING_FIELDS),
                "framework_ledgers": ledgers_t3,
                "committed_framework_state": t3_candidate.get(
                    "committed_framework_state", {}
                ),
                "committed_native_ledger": t3_candidate.get(
                    "committed_native_ledger", {}
                ),
                "challenge_response": t3_candidate.get("challenge_response", {}),
            },
            "transport_checks": {
                "framework_ledger_exactly_preserved_T1_to_T3": ledgers_t1 == ledgers_t3,
                "committed_state_exactly_preserved_T1_to_T3": (
                    t1_candidate.get("committed_framework_state", {})
                    == t3_candidate.get("committed_framework_state", {})
                ),
                "native_ledger_exactly_preserved_T1_to_T3": (
                    t1_candidate.get("committed_native_ledger", {})
                    == t3_candidate.get("committed_native_ledger", {})
                ),
                "compact_workspace_character_count": len(delivered_compact),
            },
        }
    return artifacts


def write_artifacts(
    artifacts: dict[str, dict[str, Any]], output_dir: Path, trace_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source_trace": str(trace_path.resolve()),
        "specialists": {},
        "interpretation": (
            "Exact equality is a transport check, not a semantic-quality score. "
            "Review proposition survival manually in T0, T1, T2, and T3."
        ),
    }
    for specialist, stages in artifacts.items():
        specialist_dir = output_dir / specialist
        specialist_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for stage in ("T0", "T1", "T2", "T3"):
            path = specialist_dir / f"{stage}.json"
            path.write_text(
                json.dumps(stages[stage], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            paths[stage] = str(path)
        paths["transport_checks"] = stages["transport_checks"]
        manifest["specialists"][specialist] = paths
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-framework T0-T3 proposition-survival artifacts."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    artifacts = build_artifacts(trace)
    write_artifacts(artifacts, args.output_dir, args.trace)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
