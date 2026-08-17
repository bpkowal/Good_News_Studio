from __future__ import annotations

from collections.abc import Sequence

from ..models import CycleRecord


def collect_moral_residue(
    cycles: Sequence[CycleRecord],
    current_plurality: str,
) -> list[str]:
    """Collect persistent opposed constraints without interpreting correctness."""
    # Residue is evaluated against the final policy. A specialist who dissented
    # from an earlier cycle winner but ultimately supports the final policy is
    # part of its provenance, not an unsatisfied claim against it.
    return sorted({
        candidate.constraint
        for cycle in cycles
        if not cycle.is_hypothetical
        for candidate in cycle.candidates
        if current_plurality
        and candidate.schema_valid
        and candidate.action_scores.get(current_plurality, 0.0) < 0.5
    })


def collect_reopen_conditions(cycles: Sequence[CycleRecord]) -> list[str]:
    """Collect unresolved base-scenario conditions, excluding hypothetical cycles."""
    return sorted({
        candidate.unresolved
        for cycle in cycles
        if not cycle.is_hypothetical
        for candidate in cycle.candidates
        if candidate.unresolved != "NONE"
    })
