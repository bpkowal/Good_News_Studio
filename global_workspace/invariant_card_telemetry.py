"""Run-local telemetry for invariant firings and repair-card outcomes.

The grounding loop already records an ordered attempt ledger.  This module
derives transition metrics from that ledger without changing validation or
repair behavior.  A typed issue on attempt N is an *inferred card opportunity*
for the repair that produces attempt N+1; deterministic patches and model
repairs remain distinguishable through the next attempt's repair scope.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


def _issue_rows(attempt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row for row in attempt.get("validation_issues") or []
        if isinstance(row, Mapping)
    ]


def _codes(attempt: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get("code") or "WORLD_VALIDATION_ERROR")
        for row in _issue_rows(attempt)
    }


def build_invariant_card_telemetry(
    attempts: Sequence[Mapping[str, Any]],
    *,
    terminal_status: str = "",
) -> dict[str, Any]:
    """Summarize one ordered grounding attempt ledger.

    Metrics count attempt-level presence, not raw duplicate messages.  This
    prevents one validator from inflating a code merely by reporting several
    affected entities in the same attempt.  Entity-level occurrences remain
    available separately.
    """
    rows = [row for row in attempts if isinstance(row, Mapping)]
    status = str(terminal_status or "UNKNOWN").upper()
    per_code: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "attempt_firings": 0,
        "entity_firings": 0,
        "attempt_indices": [],
        "affected_entity_ids": set(),
        "repair_opportunities": 0,
        "resolved_after_repair": 0,
        "persisted_after_repair": 0,
        "introduced_after_repair": 0,
        "commit_after_target": 0,
        "introduced_issue_codes_while_targeted": set(),
        "repair_scopes": Counter(),
        "terminal_present": False,
    })

    for fallback_index, attempt in enumerate(rows, 1):
        attempt_index = int(attempt.get("attempt") or fallback_index)
        present = _codes(attempt)
        for code in present:
            metric = per_code[code]
            metric["attempt_firings"] += 1
            metric["attempt_indices"].append(attempt_index)
        for issue in _issue_rows(attempt):
            code = str(issue.get("code") or "WORLD_VALIDATION_ERROR")
            metric = per_code[code]
            metric["entity_firings"] += 1
            entity_id = str(issue.get("entity_id") or "").strip()
            if entity_id:
                metric["affected_entity_ids"].add(entity_id)

    transitions: list[dict[str, Any]] = []
    for index in range(max(0, len(rows) - 1)):
        before = rows[index]
        after = rows[index + 1]
        before_codes = _codes(before)
        after_codes = _codes(after)
        resolved = before_codes - after_codes
        persisted = before_codes & after_codes
        introduced = after_codes - before_codes
        repair_scope = str(after.get("repair_scope") or "UNSPECIFIED").upper()
        committed_after = (
            status in {"COMMITTED", "COMMITTED_WITH_QUARANTINE"}
            and index + 1 == len(rows) - 1
            and not after_codes
            and not list(after.get("errors") or [])
        )
        for code in before_codes:
            metric = per_code[code]
            metric["repair_opportunities"] += 1
            metric["repair_scopes"][repair_scope] += 1
            if code in resolved:
                metric["resolved_after_repair"] += 1
            else:
                metric["persisted_after_repair"] += 1
            if committed_after:
                metric["commit_after_target"] += 1
            metric["introduced_issue_codes_while_targeted"].update(introduced)
        for code in introduced:
            per_code[code]["introduced_after_repair"] += 1
        transitions.append({
            "from_attempt": int(before.get("attempt") or index + 1),
            "to_attempt": int(after.get("attempt") or index + 2),
            "repair_scope": repair_scope,
            "target_issue_codes": sorted(before_codes),
            "after_issue_codes": sorted(after_codes),
            "resolved_issue_codes": sorted(resolved),
            "persistent_issue_codes": sorted(persisted),
            "introduced_issue_codes": sorted(introduced),
            "committed_after": committed_after,
        })

    terminal_codes = _codes(rows[-1]) if rows else set()
    for code in terminal_codes:
        per_code[code]["terminal_present"] = True

    rendered: dict[str, Any] = {}
    for code, metric in sorted(per_code.items()):
        opportunities = int(metric["repair_opportunities"])
        rendered[code] = {
            **{
                key: value for key, value in metric.items()
                if key not in {
                    "affected_entity_ids", "repair_scopes",
                    "introduced_issue_codes_while_targeted",
                }
            },
            "affected_entity_ids": sorted(metric["affected_entity_ids"]),
            "repair_scopes": dict(sorted(metric["repair_scopes"].items())),
            "introduced_issue_codes_while_targeted": sorted(
                metric["introduced_issue_codes_while_targeted"]
            ),
            "resolution_rate": (
                metric["resolved_after_repair"] / opportunities
                if opportunities else None
            ),
            "persistence_rate": (
                metric["persisted_after_repair"] / opportunities
                if opportunities else None
            ),
        }

    return {
        "telemetry_schema_version": 1,
        "terminal_status": status,
        "attempt_count": len(rows),
        "repair_transition_count": len(transitions),
        "distinct_issue_codes": len(rendered),
        "terminal_issue_codes": sorted(terminal_codes),
        "definitions": {
            "attempt_firing": "Issue code appears at least once in one attempt.",
            "repair_opportunity": (
                "Issue code precedes another attempt and therefore could target "
                "the next deterministic or model repair."
            ),
            "resolved_after_repair": "Target code is absent from the next attempt.",
            "introduced_after_repair": (
                "Code is absent before a repair transition and present afterward."
            ),
        },
        "invariants": rendered,
        "transitions": transitions,
    }


def aggregate_invariant_card_telemetry(
    runs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate run-local telemetry without conflating runs and attempts."""
    run_rows = [row for row in runs if isinstance(row, Mapping)]
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    runs_with_code: Counter[str] = Counter()
    terminal_runs: Counter[str] = Counter()
    for run in run_rows:
        invariants = run.get("invariants") or {}
        if not isinstance(invariants, Mapping):
            continue
        for code, raw in invariants.items():
            if not isinstance(raw, Mapping):
                continue
            code = str(code)
            runs_with_code[code] += 1
            for field in (
                "attempt_firings", "entity_firings", "repair_opportunities",
                "resolved_after_repair", "persisted_after_repair",
                "introduced_after_repair", "commit_after_target",
            ):
                totals[code][field] += int(raw.get(field) or 0)
            if raw.get("terminal_present"):
                terminal_runs[code] += 1

    rendered: dict[str, Any] = {}
    for code in sorted(totals):
        metric = totals[code]
        opportunities = metric["repair_opportunities"]
        rendered[code] = {
            "runs_with_firing": runs_with_code[code],
            **dict(metric),
            "terminal_rejection_runs": terminal_runs[code],
            "resolution_rate": (
                metric["resolved_after_repair"] / opportunities
                if opportunities else None
            ),
            "persistence_rate": (
                metric["persisted_after_repair"] / opportunities
                if opportunities else None
            ),
        }
    return {
        "telemetry_schema_version": 1,
        "run_count": len(run_rows),
        "invariants": rendered,
    }
