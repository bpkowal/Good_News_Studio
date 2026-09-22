"""Diagnose historical EXCLUSIVE_ALLOCATION_BRANCH_MISSING failures."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


_TARGET = "EXCLUSIVE_ALLOCATION_BRANCH_MISSING"
_BRANCH = re.compile(r"\b(A\d+)\b.*?nonrecipient\s+([A-Za-z0-9_]+)", re.I)
_NONRECEIPT = re.compile(
    r"\b(?:does\s+not|do\s+not|did\s+not|will\s+not|is\s+not|was\s+not)\s+"
    r"(?:receive|receives|received|given|treated)|\b(?:untreated|without\s+"
    r"(?:the\s+)?(?:medicine|medication|dose|resource|oxygen|antidote)|"
    r"denied|withheld|receives?\s+(?:no|none))\b",
    re.I,
)
_WELFARE_KINDS = {
    "HEALTH_OUTCOME", "MORTALITY", "PHYSICAL_HARM", "WELFARE_OUTCOME",
    "OTHER",
}


def _codes(attempt: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get("code") or "")
        for row in attempt.get("validation_issues") or []
        if isinstance(row, Mapping)
    }


def _candidate_for(
    grounding: Mapping[str, Any], attempt: Mapping[str, Any],
) -> Mapping[str, Any]:
    number = int(attempt.get("attempt") or 0)
    histories = [
        row for row in grounding.get("rejected_candidate_history") or []
        if isinstance(row, Mapping)
        and int(row.get("attempt") or 0) <= number
        and isinstance(row.get("candidate"), Mapping)
    ]
    if histories:
        return histories[-1]["candidate"]
    candidate = grounding.get("rejected_candidate")
    return candidate if isinstance(candidate, Mapping) else {}


def _world(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    world = candidate.get("world_model")
    return world if isinstance(world, Mapping) else {}


def audit_file(path: Path, *, replay_current: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grounding = payload.get("grounding") or {}
    attempts = [
        row for row in grounding.get("attempts") or []
        if isinstance(row, Mapping)
    ]
    replay_cache: dict[tuple[int, str], list[Mapping[str, Any]] | None] = {}
    rows: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        issues = [
            issue for issue in attempt.get("validation_issues") or []
            if isinstance(issue, Mapping) and issue.get("code") == _TARGET
        ]
        if not issues:
            continue
        candidate = _candidate_for(grounding, attempt)
        replay_key = (
            int(attempt.get("attempt") or index + 1),
            str(attempt.get("repair_scope") or ""),
        )
        if replay_current and replay_key not in replay_cache:
            try:
                from global_workspace.local_specialists import _admit_action_source_rows
                from global_workspace.scenario_semantics import segment_scenario_clauses

                scenario = str(payload.get("canonical_scenario") or "")
                actions = [str(value) for value in payload.get("canonical_actions") or []]
                admitted = _admit_action_source_rows(
                    candidate,
                    actions,
                    [f"A{position}" for position in range(len(actions))],
                    segment_scenario_clauses(scenario),
                    allow_action_text_evidence=True,
                )
                replay_cache[replay_key] = [
                    row
                    for row in admitted.get("validation_issues") or []
                    if isinstance(row, Mapping)
                ]
            except Exception:
                replay_cache[replay_key] = None
        world = _world(candidate)
        effects = [row for row in world.get("effects") or [] if isinstance(row, Mapping)]
        actions = {
            str(row.get("action_id") or ""): row
            for row in world.get("actions") or [] if isinstance(row, Mapping)
        }
        links = [row for row in world.get("causal_links") or [] if isinstance(row, Mapping)]
        next_codes = _codes(attempts[index + 1]) if index + 1 < len(attempts) else set()
        current_codes = _codes(attempt)
        for issue in issues:
            message = str(issue.get("message") or "")
            match = _BRANCH.search(message)
            action_id = match.group(1) if match else str(issue.get("entity_id") or "")
            party_id = match.group(2) if match else ""
            action = actions.get(action_id, {})
            owned = [row for row in effects if str(row.get("action_id") or "") == action_id]
            party_effects = [row for row in owned if str(row.get("party_id") or "") == party_id]
            anchors = [
                row for row in owned
                if str(row.get("effect_kind") or "").upper() == "RESOURCE_TRANSFER"
                and str(row.get("directness") or "").upper() == "DIRECT"
            ]
            nonreceipt = [
                row for row in party_effects
                if _NONRECEIPT.search(" ".join((
                    str(row.get("outcome") or ""),
                    str(row.get("source_proposition") or ""),
                )))
            ]
            adverse = [
                row for row in party_effects
                if str(row.get("polarity") or "").upper() in {"ADVERSE", "UNRESOLVED"}
                and str(row.get("effect_kind") or "").upper() in _WELFARE_KINDS
            ]
            nonreceipt_ids = {str(row.get("effect_id") or "") for row in nonreceipt}
            adverse_ids = {str(row.get("effect_id") or "") for row in adverse}
            linked = any(
                str(link.get("source_id") or "") in nonreceipt_ids
                and str(link.get("target_id") or "") in adverse_ids
                for link in links
            )
            replay_issues = replay_cache.get(replay_key) or []
            replay_target_messages = [
                str(row.get("message") or "")
                for row in replay_issues
                if row.get("code") == _TARGET
            ]
            current_same_branch = any(
                action_id in current and party_id in current
                for current in replay_target_messages
            )
            if nonreceipt:
                shape = "NONRECEIPT_PRESENT_BUT_FLAGGED"
            elif adverse:
                shape = "MISSING_NONRECEIPT_NODE_ADVERSE_PRESENT"
            else:
                shape = "ENTIRE_NONRECIPIENT_BRANCH_MISSING"
            rows.append({
                "diagnostic": str(path),
                "scenario_path": str(payload.get("scenario_path") or ""),
                "attempt": int(attempt.get("attempt") or index + 1),
                "repair_scope": str(attempt.get("repair_scope") or ""),
                "action_id": action_id,
                "nonrecipient_party_id": party_id,
                "shape": shape,
                "allocation_anchor_present": bool(anchors),
                "action_recipient_ids": list(action.get("recipient_party_ids") or []),
                "nonreceipt_present": bool(nonreceipt),
                "adverse_present": bool(adverse),
                "nonreceipt_to_adverse_link_present": linked,
                "persists_next_attempt": _TARGET in next_codes,
                "introduced_next_codes": sorted(next_codes - current_codes),
                "current_validator_target_present": (
                    bool(replay_target_messages)
                    if replay_current and replay_cache.get(replay_key) is not None
                    else None
                ),
                "current_validator_same_branch_present": (
                    current_same_branch
                    if replay_current and replay_cache.get(replay_key) is not None
                    else None
                ),
            })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shape_counts = Counter(row["shape"] for row in rows)
    shape_persistence: dict[str, dict[str, Any]] = {}
    for shape in shape_counts:
        subset = [row for row in rows if row["shape"] == shape]
        opportunities = [row for row in subset if row["persists_next_attempt"] is not None]
        persistent = sum(bool(row["persists_next_attempt"]) for row in opportunities)
        shape_persistence[shape] = {
            "occurrences": len(subset),
            "persistent_next_attempt": persistent,
            "persistence_rate": persistent / len(opportunities) if opportunities else None,
            "missing_allocation_anchor": sum(
                not row["allocation_anchor_present"] for row in subset
            ),
        }
    introduced = Counter(
        code for row in rows for code in row["introduced_next_codes"]
    )
    replayed = [
        row for row in rows
        if row.get("current_validator_target_present") is not None
    ]
    same_branch_replayed = [
        row for row in rows
        if row.get("current_validator_same_branch_present") is not None
    ]
    scenario_paths = {row["scenario_path"] for row in rows if row["scenario_path"]}
    return {
        "audit_schema_version": 1,
        "diagnostic_occurrences": len(rows),
        "distinct_scenarios": len(scenario_paths),
        "shape_summary": dict(sorted(shape_persistence.items())),
        "introduced_issue_codes_after_target": dict(introduced.most_common()),
        "current_validator_replay": {
            "replayed_occurrences": len(replayed),
            "target_still_present": sum(
                bool(row["current_validator_target_present"]) for row in replayed
            ),
            "same_branch_still_present": sum(
                bool(row["current_validator_same_branch_present"])
                for row in same_branch_replayed
            ),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path("workspace_outputs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-current", action="store_true")
    args = parser.parse_args()
    paths = (
        sorted(args.input.rglob("world_grounding_failure_*.json"))
        if args.input.is_dir() else [args.input]
    )
    rows = [
        row for path in paths
        for row in audit_file(path, replay_current=args.replay_current)
    ]
    report = summarize(rows)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
