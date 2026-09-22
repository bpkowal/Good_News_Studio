"""Aggregate explicit world-grounding repair experiment events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from global_workspace.repair_experiment import aggregate_repair_events


def _paths(inputs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in inputs:
        if path.is_dir():
            found.extend(path.rglob("world_grounding_failure_*.json"))
            found.extend(path.rglob("performance_*.json"))
        elif path.is_file():
            found.append(path)
    return sorted(set(found))


def _events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    grounding = payload.get("grounding")
    if isinstance(grounding, dict):
        return [
            row for row in grounding.get("repair_experiment_events") or []
            if isinstance(row, dict)
        ]
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("name") != "world_grounding_repair_experiments":
            continue
        metadata = event.get("metadata") or {}
        if isinstance(metadata, dict):
            return [
                row for row in metadata.get("repair_events") or []
                if isinstance(row, dict)
            ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[Path("workspace_outputs")])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs: list[list[dict[str, Any]]] = []
    source_files: list[str] = []
    for path in _paths(args.inputs):
        try:
            events = _events(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
        if events:
            runs.append(events)
            source_files.append(str(path))
    report = aggregate_repair_events(runs)
    report["source_files"] = source_files
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
