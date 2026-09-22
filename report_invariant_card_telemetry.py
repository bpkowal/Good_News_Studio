"""Aggregate invariant/repair-card telemetry from grounding diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from global_workspace.invariant_card_telemetry import (
    aggregate_invariant_card_telemetry,
    build_invariant_card_telemetry,
)


def _paths(inputs: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for path in inputs:
        if path.is_dir():
            found.extend(sorted(path.rglob("world_grounding_failure_*.json")))
            found.extend(sorted(path.rglob("performance_*.json")))
        elif path.is_file():
            found.append(path)
    return list(dict.fromkeys(found))


def _telemetry(payload: dict[str, Any]) -> dict[str, Any] | None:
    grounding = payload.get("grounding") if isinstance(payload.get("grounding"), dict) else payload
    existing = grounding.get("invariant_card_telemetry")
    if isinstance(existing, dict):
        return existing
    attempts = grounding.get("attempts")
    if isinstance(attempts, list):
        return build_invariant_card_telemetry(
            attempts,
            terminal_status=str(grounding.get("status") or "UNKNOWN"),
        )
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        if event.get("name") != "world_grounding_invariant_card_telemetry":
            continue
        metadata = event.get("metadata") or {}
        telemetry = metadata.get("telemetry") if isinstance(metadata, dict) else None
        if isinstance(telemetry, dict):
            return telemetry
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="*", type=Path, default=[Path("workspace_outputs")],
        help="Diagnostic JSON files or directories (default: workspace_outputs).",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    runs: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for path in _paths(args.inputs):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = _telemetry(payload)
            if row is not None:
                runs.append(row)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            unreadable.append(str(path))
    report = aggregate_invariant_card_telemetry(runs)
    report["source_file_count"] = len(runs)
    report["unreadable_files"] = unreadable
    ranked = sorted(
        report["invariants"].items(),
        key=lambda item: (
            -int(item[1].get("runs_with_firing") or 0),
            -int(item[1].get("terminal_rejection_runs") or 0),
            item[0],
        ),
    )
    report["ranking"] = [
        {"code": code, **metrics}
        for code, metrics in ranked[: max(0, args.limit)]
    ]
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
