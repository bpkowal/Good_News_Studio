"""Score frozen invariant and repair-card trials without running the pipeline."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _codes(value: Any) -> set[str]:
    return {str(item) for item in value or [] if str(item)}


def score_trials(trials: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate family metrics with explicit, denominator-safe definitions."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[str(trial.get("family") or "UNSPECIFIED")].append(trial)

    def score(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        tp = fp = fn = tn = 0
        repair_denominator = repair_successes = 0
        repaired_trials = new_error_trials = 0
        prior_successes = prior_successes_preserved = 0
        for row in rows:
            expected = bool(row.get("expected_violation"))
            detected = bool(row.get("detected"))
            if expected and detected:
                tp += 1
            elif not expected and detected:
                fp += 1
            elif expected:
                fn += 1
            else:
                tn += 1

            attempted = bool(row.get("repair_attempted"))
            if expected and detected and attempted:
                repair_denominator += 1
                repair_successes += int(bool(row.get("repair_committed")))
            if attempted:
                repaired_trials += 1
                before = _codes(row.get("before_issue_codes"))
                after = _codes(row.get("after_issue_codes"))
                target = _codes(row.get("target_issue_codes"))
                new_error_trials += int(bool(after - before - target))

            if bool(row.get("previous_success")):
                prior_successes += 1
                prior_successes_preserved += int(
                    bool(row.get("previous_success_preserved"))
                )
        return {
            "trial_count": len(rows),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "detection_precision": tp / (tp + fp) if tp + fp else None,
            "detection_recall": tp / (tp + fn) if tp + fn else None,
            "repair_success_rate": (
                repair_successes / repair_denominator if repair_denominator else None
            ),
            "new_error_rate": (
                new_error_trials / repaired_trials if repaired_trials else None
            ),
            "previous_success_preservation": (
                prior_successes_preserved / prior_successes
                if prior_successes else None
            ),
            "denominators": {
                "repair_attempts_on_true_detections": repair_denominator,
                "all_repair_attempts": repaired_trials,
                "previous_success_trials": prior_successes,
            },
        }

    rows = [row for family_rows in grouped.values() for row in family_rows]
    return {
        "evaluation_version": "1.0",
        "definitions": {
            "repair_success": "A true-positive violation was repaired and the candidate committed.",
            "new_error": "The repaired candidate contains an issue code absent before repair and outside the targeted codes.",
            "previous_success_preservation": "A case known to commit under the control still commits without unrelated structural loss.",
        },
        "aggregate": score(rows),
        "families": {
            family: score(family_rows)
            for family, family_rows in sorted(grouped.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.trials.read_text(encoding="utf-8"))
    report = score_trials(payload.get("trials") or [])
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

