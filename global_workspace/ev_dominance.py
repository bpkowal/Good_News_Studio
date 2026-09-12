from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import statistics
from typing import Sequence

from .expected_value import EV_ARITHMETIC_VERIFIED
from .models import CandidateChunk


@dataclass(slots=True)
class EVDominanceAssessment:
    activated: bool
    leading_action: str
    majority_count: int
    valid_delegate_count: int
    ratio: float
    direction: str = ""
    unit: str = ""
    aggregate_values: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def assess_ev_dominance(
    candidates: Sequence[CandidateChunk], actions: Sequence[str], *,
    ratio_threshold: float = 5.0, majority_fraction: float = 0.6,
    minimum_majority: int = 3,
) -> EVDominanceAssessment:
    valid = [candidate for candidate in candidates if candidate.schema_valid]
    recommendations = [candidate.recommended_action for candidate in valid]
    if not recommendations:
        return EVDominanceAssessment(False, "", 0, 0, 0.0, reason="no valid delegates")
    leader = max(set(recommendations), key=recommendations.count)
    majority = recommendations.count(leader)
    required = max(minimum_majority, math.ceil(majority_fraction * len(valid)))
    if majority < required:
        return EVDominanceAssessment(
            False, leader, majority, len(valid), 0.0,
            reason=f"majority {majority}/{len(valid)} is below required {required}",
        )
    usable = []
    for candidate in valid:
        if candidate.expected_value_validation_status != EV_ARITHMETIC_VERIFIED:
            continue
        estimates = candidate.expected_value_estimates
        if set(estimates) != set(actions):
            continue
        if not all(
            bool(value.get("grounded"))
            and value.get("validation_status") == EV_ARITHMETIC_VERIFIED
            for value in estimates.values()
        ):
            continue
        units = {str(value.get("unit", "")).upper() for value in estimates.values()}
        directions = {str(value.get("direction", "")).upper() for value in estimates.values()}
        if len(units) == len(directions) == 1 and units != {""} and directions <= {"BENEFIT", "HARM"}:
            usable.append(candidate)
    if len(usable) < required:
        return EVDominanceAssessment(
            False, leader, majority, len(usable), 0.0,
            reason=f"only {len(usable)} delegates supplied comparable grounded EV estimates",
        )
    unit = str(next(iter(usable[0].expected_value_estimates.values()))["unit"]).upper()
    direction = str(next(iter(usable[0].expected_value_estimates.values()))["direction"]).upper()
    if any(
        str(next(iter(candidate.expected_value_estimates.values()))["unit"]).upper() != unit
        or str(next(iter(candidate.expected_value_estimates.values()))["direction"]).upper() != direction
        for candidate in usable
    ):
        return EVDominanceAssessment(
            False, leader, majority, len(usable), 0.0,
            reason="delegates used incomparable EV units or directions",
        )
    aggregate = {
        action: statistics.median(
            float(candidate.expected_value_estimates[action]["value"])
            for candidate in usable
        )
        for action in actions
    }
    rivals = [aggregate[action] for action in actions if action != leader]
    if not rivals:
        return EVDominanceAssessment(False, leader, majority, len(usable), 0.0, reason="no rival action")
    if direction == "BENEFIT":
        rival = max(rivals)
        ratio = math.inf if rival == 0 and aggregate[leader] > 0 else (
            aggregate[leader] / rival if rival > 0 else 0.0
        )
    else:
        rival = min(rivals)
        ratio = math.inf if aggregate[leader] == 0 and rival > 0 else (
            rival / aggregate[leader] if aggregate[leader] > 0 else 0.0
        )
    activated = ratio > ratio_threshold
    return EVDominanceAssessment(
        activated, leader, majority, len(usable), ratio, direction, unit, aggregate,
        reason=(
            f"{leader} has {ratio:.2f}x {direction.lower()} EV dominance with "
            f"{majority}/{len(valid)} delegate support"
            if activated else f"EV ratio {ratio:.2f} does not exceed {ratio_threshold:.2f}x"
        ),
    )
