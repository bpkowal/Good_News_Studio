"""Validate and query the repository's experiment and decision memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
OUTCOMES = {
    "SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE",
    "REGRESSION", "NO_DEMONSTRATED_VALUE",
}
DECISION_STATUSES = {
    "ADOPTED", "ACTIVE", "PAUSE", "REJECTED_FOR_NOW", "SUPERSEDED",
}


def _records(folder: str) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((ROOT / folder).glob("*.json")):
        rows.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def validate_ledger() -> list[str]:
    errors: list[str] = []
    experiments = _records("experiments")
    decisions = _records("decisions")
    experiment_ids: set[str] = set()
    decision_ids: set[str] = set()
    for path, row in experiments:
        record_id = str(row.get("experiment_id") or "")
        if not record_id:
            errors.append(f"{path}: missing experiment_id")
        elif record_id in experiment_ids:
            errors.append(f"{path}: duplicate experiment_id {record_id}")
        experiment_ids.add(record_id)
        for field in (
            "date", "title", "project_state", "question", "hypothesis",
            "change_under_test", "control", "experimental", "held_constant",
            "intentionally_not_compared", "metrics", "result",
            "interpretation", "decision", "limitations", "artifacts",
        ):
            if field not in row:
                errors.append(f"{path}: missing {field}")
        if row.get("result") not in OUTCOMES:
            errors.append(f"{path}: unsupported result {row.get('result')!r}")
    for path, row in decisions:
        record_id = str(row.get("decision_id") or "")
        if not record_id:
            errors.append(f"{path}: missing decision_id")
        elif record_id in decision_ids:
            errors.append(f"{path}: duplicate decision_id {record_id}")
        decision_ids.add(record_id)
        if row.get("status") not in DECISION_STATUSES:
            errors.append(f"{path}: unsupported status {row.get('status')!r}")
        for evidence_id in row.get("evidence") or []:
            if str(evidence_id) not in experiment_ids:
                errors.append(f"{path}: unknown evidence experiment {evidence_id}")
        if "reconsider_if" not in row:
            errors.append(f"{path}: missing reconsider_if")
    return errors


def search_records(terms: Iterable[str]) -> list[dict[str, Any]]:
    needles = [term.casefold() for term in terms if term.strip()]
    matches: list[dict[str, Any]] = []
    for kind, folder, id_field in (
        ("experiment", "experiments", "experiment_id"),
        ("decision", "decisions", "decision_id"),
    ):
        for path, row in _records(folder):
            haystack = json.dumps(row, sort_keys=True).casefold()
            if all(needle in haystack for needle in needles):
                matches.append({
                    "kind": kind,
                    "id": row.get(id_field),
                    "title": row.get("title") or row.get("topic"),
                    "status": row.get("result") or row.get("status"),
                    "path": str(path.relative_to(ROOT.parent)),
                })
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    search = subparsers.add_parser("search")
    search.add_argument("terms", nargs="+")
    args = parser.parse_args()
    if args.command == "validate":
        errors = validate_ledger()
        if errors:
            print("\n".join(errors))
            return 1
        print("research ledger valid")
        return 0
    print(json.dumps(search_records(args.terms), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

