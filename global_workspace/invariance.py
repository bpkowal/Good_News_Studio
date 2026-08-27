"""Permutation-invariance measurements for completed Parliament traces.

The evaluator compares semantic action keys, never A0/A1 presentation labels.
It separates ordinal stability, cardinal stability, and orchestration-path stability
so a preserved judgment cannot conceal materially different confidence dynamics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .scenario_semantics import semantic_action_key
from .structured_io import structured_text_error


@dataclass(slots=True)
class InvarianceReport:
    ordinal_invariant: bool
    cardinal_invariant: bool
    dynamic_invariant: bool
    selected_action_keys: tuple[str, str]
    specialist_direction_matches: dict[str, bool] = field(default_factory=dict)
    max_action_score_delta: float = 0.0
    max_preference_strength_delta: float = 0.0
    max_epistemic_confidence_delta: float = 0.0
    final_policy_support_delta: float = 0.0
    published_policy_support_delta: float = 0.0
    published_epistemic_confidence_delta: float = 0.0
    cycle_counts: tuple[int, int] = (0, 0)
    broadcast_paths: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    excluded_reversal_reviews: tuple[int, int] = (0, 0)
    tolerance: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _action_key(trace: dict[str, Any], action: str) -> str:
    for graph in reversed(trace.get("semantic_graphs") or []):
        for node in graph.get("nodes") or []:
            if node.get("kind") != "ACTION":
                continue
            attributes = node.get("attributes") or {}
            if action in {
                node.get("id", ""),
                node.get("label", ""),
                attributes.get("canonical_action_id", ""),
                attributes.get("semantic_action_key", ""),
            }:
                stored = str(attributes.get("semantic_action_key", ""))
                if stored.startswith("action_graph:v1:"):
                    return stored
                # Migrate lexical-key traces in memory. Historical files remain
                # immutable, but comparisons use the current graph identity.
                return semantic_action_key(node.get("label", action))
    return semantic_action_key(action)


def _selected_key(trace: dict[str, Any]) -> str:
    state = trace.get("authoritative_semantic_state") or {}
    stored = str(state.get("selected_action_key", ""))
    if stored.startswith("action_graph:v1:"):
        return stored
    return _action_key(trace, str(trace.get("selected_action", "")))


def _base_cycles(trace: dict[str, Any]) -> list[dict[str, Any]]:
    cycles = list(trace.get("cycles") or [])
    base = [cycle for cycle in cycles if not cycle.get("is_hypothetical", False)]
    return base or cycles


def _candidate_map(trace: dict[str, Any], cycle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate.get("specialist", "")): candidate
        for candidate in cycle.get("candidates") or []
        if candidate.get("schema_valid", True) and candidate.get("specialist")
    }


def _score_vector(trace: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    return {
        _action_key(trace, str(action)): float(score)
        for action, score in (candidate.get("action_scores") or {}).items()
    }


def _policy_support(trace: dict[str, Any], cycle: dict[str, Any], key: str) -> float:
    return next((
        float(score) for action, score in (cycle.get("policy") or {}).items()
        if _action_key(trace, str(action)) == key
    ), 0.0)


def _constraint_path(trace: dict[str, Any]) -> tuple[str, ...]:
    path = []
    for cycle in trace.get("cycles") or []:
        received = cycle.get("received_broadcast") or cycle.get("broadcast") or {}
        path.append(str(received.get("constraint", "")))
    return tuple(path)


def _excluded_reviews(trace: dict[str, Any]) -> int:
    count = 0
    for cycle in trace.get("cycles") or []:
        for candidate in cycle.get("candidates") or []:
            response = candidate.get("reversal_review_response", "NOT_TESTED")
            if response == "NOT_TESTED":
                continue
            justification = candidate.get("reversal_review_justification", "")
            if (
                not candidate.get("reversal_review_valid", True)
                or bool(structured_text_error(justification))
            ):
                count += 1
    return count


def compare_label_permutation_traces(
    first: dict[str, Any], second: dict[str, Any], *, tolerance: float = 0.05,
) -> InvarianceReport:
    """Compare two semantically identical traces whose labels/order may differ."""
    first_key, second_key = _selected_key(first), _selected_key(second)
    first_base, second_base = _base_cycles(first), _base_cycles(second)
    first_open = _candidate_map(first, first_base[0]) if first_base else {}
    second_open = _candidate_map(second, second_base[0]) if second_base else {}
    specialists = sorted(set(first_open) & set(second_open))

    direction_matches: dict[str, bool] = {}
    score_deltas: list[float] = []
    preference_deltas: list[float] = []
    epistemic_deltas: list[float] = []
    for specialist in specialists:
        left, right = first_open[specialist], second_open[specialist]
        left_choice = _action_key(first, str(left.get("recommended_action", "")))
        right_choice = _action_key(second, str(right.get("recommended_action", "")))
        direction_matches[specialist] = left_choice == right_choice
        left_scores, right_scores = _score_vector(first, left), _score_vector(second, right)
        for key in set(left_scores) & set(right_scores):
            score_deltas.append(abs(left_scores[key] - right_scores[key]))
        preference_deltas.append(abs(
            float(left.get("preference_strength", 0.0))
            - float(right.get("preference_strength", 0.0))
        ))
        epistemic_deltas.append(abs(
            float(left.get("epistemic_confidence", left.get("confidence", 0.0)))
            - float(right.get("epistemic_confidence", right.get("confidence", 0.0)))
        ))

    final_support_delta = 0.0
    if first_base and second_base and first_key == second_key:
        final_support_delta = abs(
            _policy_support(first, first_base[-1], first_key)
            - _policy_support(second, second_base[-1], second_key)
        )
    max_score = max(score_deltas, default=0.0)
    max_preference = max(preference_deltas, default=0.0)
    max_epistemic = max(epistemic_deltas, default=0.0)
    published_support_delta = abs(
        float(first.get("confidence", 0.0)) - float(second.get("confidence", 0.0))
    )
    published_epistemic_delta = abs(
        float(first.get("epistemic_confidence", 0.0))
        - float(second.get("epistemic_confidence", 0.0))
    )
    ordinal = bool(specialists) and first_key == second_key and all(
        direction_matches.values()
    )
    cardinal = ordinal and all(
        delta <= tolerance
        for delta in (
            max_score, max_preference, max_epistemic, final_support_delta,
            published_support_delta, published_epistemic_delta,
        )
    )
    first_path, second_path = _constraint_path(first), _constraint_path(second)
    return InvarianceReport(
        ordinal_invariant=ordinal,
        cardinal_invariant=cardinal,
        dynamic_invariant=first_path == second_path,
        selected_action_keys=(first_key, second_key),
        specialist_direction_matches=direction_matches,
        max_action_score_delta=max_score,
        max_preference_strength_delta=max_preference,
        max_epistemic_confidence_delta=max_epistemic,
        final_policy_support_delta=final_support_delta,
        published_policy_support_delta=published_support_delta,
        published_epistemic_confidence_delta=published_epistemic_delta,
        cycle_counts=(len(first.get("cycles") or []), len(second.get("cycles") or [])),
        broadcast_paths=(first_path, second_path),
        excluded_reversal_reviews=(_excluded_reviews(first), _excluded_reviews(second)),
        tolerance=tolerance,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_trace", type=Path)
    parser.add_argument("second_trace", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    first = json.loads(args.first_trace.read_text())
    second = json.loads(args.second_trace.read_text())
    report = compare_label_permutation_traces(
        first, second, tolerance=max(0.0, args.tolerance)
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
