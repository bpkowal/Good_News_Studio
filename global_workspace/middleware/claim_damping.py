from __future__ import annotations

from dataclasses import dataclass


SPECULATIVE_DIRECTION_RETENTION = 0.55
SPECULATIVE_EPISTEMIC_CAP = 0.50


@dataclass(frozen=True, slots=True)
class ClaimDampingOutcome:
    scores: dict[str, float]
    unresolved: str
    epistemic_cap: float | None = None
    applied: bool = False


def apply_symmetric_claim_damping(
    scores: dict[str, float],
    evidence_basis: str,
    unresolved: str,
    *,
    retention: float = SPECULATIVE_DIRECTION_RETENTION,
) -> ClaimDampingOutcome:
    """Contract disclosed speculation toward neutrality without reversing it."""
    if evidence_basis != "UNSTATED_FACTS":
        return ClaimDampingOutcome(dict(scores), unresolved)
    bounded_retention = max(0.0, min(1.0, float(retention)))
    damped = {
        action: 0.5 + (float(score) - 0.5) * bounded_retention
        for action, score in scores.items()
    }
    return ClaimDampingOutcome(
        damped,
        unresolved if unresolved != "NONE" else "VERIFY_FACTS",
        epistemic_cap=SPECULATIVE_EPISTEMIC_CAP,
        applied=True,
    )
