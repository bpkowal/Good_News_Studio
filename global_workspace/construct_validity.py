"""Observational metadata for constructs that must not be conflated with votes.

These records describe a run after the fact. They deliberately do not alter policy
scores, workspace admission, stopping, or delegate confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from .models import CycleRecord, WorkspaceAccessDecision


@dataclass(slots=True)
class TerminationAssessment:
    termination_type: str
    resource_censored: bool
    endogenous_stop: bool
    convergence_evidence: float
    stable_cycles_observed: int
    rationale: str


@dataclass(slots=True)
class MoralResidueRecord:
    constraint: str
    claim_type: str
    affected_action: str
    source_specialists: list[str] = field(default_factory=list)
    observed_cycles: list[int] = field(default_factory=list)
    compatible_with_recommendation: bool = True
    resolved: bool = False


@dataclass(slots=True)
class AccessConstructRecord:
    cycle: int
    content_type: str
    admitted: bool
    decision_relevance: float
    evidential_support: float
    novelty: float
    neglected_coverage: float
    redundancy: float
    observational_only: bool = True
    calibrated: bool = False


@dataclass(slots=True)
class FurtherDeliberationEstimate:
    action_change_signal: float
    new_material_constraint_signal: float
    information_value_signal: float
    calibrated: bool = False
    affects_stopping: bool = False
    rationale: str = ""


def assess_termination(
    halted_by: str,
    cycles: Sequence[CycleRecord],
    stable_cycles_required: int,
) -> TerminationAssessment:
    actual = [cycle for cycle in cycles if not cycle.is_hypothetical]
    final = actual[-1] if actual else None
    stable = final.stable_cycles if final is not None else 0
    entropy = final.entropy if final is not None else 1.0
    stability_component = min(1.0, stable / max(1, stable_cycles_required))
    entropy_component = max(0.0, 1.0 - entropy)
    evidence = round(0.55 * stability_component + 0.45 * entropy_component, 3)
    mapping = {
        "convergence": "ENDOGENOUS_CONVERGENCE",
        "ev_dominance": "EV_DOMINANCE_STOP",
        "time_budget": "RESOURCE_CENSORED",
        "cycle_budget": "RESOURCE_CENSORED",
        "insufficient_valid_candidates": "INSUFFICIENT_VALID_RESPONSES",
    }
    termination_type = mapping.get(halted_by, halted_by.upper() or "UNKNOWN")
    resource_censored = halted_by in {"time_budget", "cycle_budget"}
    endogenous = halted_by in {"convergence", "ev_dominance"}
    rationale = (
        "Observation ended at an external compute limit; the last preference is not "
        "evidence that the deliberative trajectory naturally terminated."
        if resource_censored else
        "The run stopped under an endogenous workspace criterion."
        if endogenous else
        "The run ended without an endogenous convergence determination."
    )
    return TerminationAssessment(
        termination_type, resource_censored, endogenous, evidence, stable, rationale
    )


def collect_typed_residue(
    cycles: Sequence[CycleRecord], current_plurality: str
) -> list[MoralResidueRecord]:
    records: dict[tuple[str, str], MoralResidueRecord] = {}
    for cycle in cycles:
        if cycle.is_hypothetical:
            continue
        for candidate in cycle.candidates:
            if not candidate.schema_valid or not current_plurality:
                continue
            opposed = candidate.action_scores.get(current_plurality, 0.0) < 0.5
            if not opposed:
                continue
            constraint = candidate.constraint or "UNSPECIFIED_CONSTRAINT"
            affected = candidate.recommended_action or "UNRESOLVED_ALTERNATIVE"
            key = (constraint, affected)
            record = records.setdefault(key, MoralResidueRecord(
                constraint=constraint,
                claim_type="PRESERVED_NORMATIVE_CLAIM",
                affected_action=affected,
            ))
            if candidate.specialist not in record.source_specialists:
                record.source_specialists.append(candidate.specialist)
            if cycle.cycle not in record.observed_cycles:
                record.observed_cycles.append(cycle.cycle)
    return sorted(records.values(), key=lambda item: (item.constraint, item.affected_action))


def describe_access(decision: WorkspaceAccessDecision) -> AccessConstructRecord:
    signals = set(decision.signals)
    relevance = 0.85 if decision.admitted else 0.40
    evidential = 0.45
    if "delegate_uncertainty_present" in signals:
        evidential += 0.15
    if "conditional_source_testimony" in signals:
        evidential += 0.10
    novelty = 0.75 if decision.cycle <= 2 else 0.50
    neglected = 0.80 if decision.content_type in {
        "VISIBILITY_AUDIT", "AUTONOMY_AUDIT", "REVERSAL_AUDIT"
    } else 0.55
    redundancy = 0.65 if "homogeneous_score_vectors" in signals else 0.25
    return AccessConstructRecord(
        decision.cycle, decision.content_type, decision.admitted,
        min(1.0, relevance), min(1.0, evidential), novelty, neglected, redundancy,
    )


def estimate_further_deliberation(
    cycles: Sequence[CycleRecord], termination: TerminationAssessment
) -> FurtherDeliberationEstimate:
    actual = [cycle for cycle in cycles if not cycle.is_hypothetical]
    if not actual:
        return FurtherDeliberationEstimate(
            1.0, 1.0, 1.0, rationale="No valid base deliberation was observed."
        )
    final = actual[-1]
    dissent = 1.0 if final.dissent is not None else 0.0
    unresolved = sum(
        candidate.unresolved != "NONE"
        for candidate in final.candidates if candidate.schema_valid
    ) / max(1, sum(candidate.schema_valid for candidate in final.candidates))
    instability = 1.0 - min(1.0, termination.stable_cycles_observed / 3.0)
    entropy = min(1.0, max(0.0, final.entropy))
    change = min(0.95, max(0.02,
        0.35 * entropy + 0.25 * dissent + 0.25 * unresolved + 0.15 * instability
    ))
    new_constraint = min(0.95, max(0.03,
        0.40 * unresolved + 0.30 * entropy + 0.30 * instability
    ))
    information_value = 1.0 - math.prod((1.0 - change, 1.0 - new_constraint))
    return FurtherDeliberationEstimate(
        round(change, 3), round(new_constraint, 3), round(information_value, 3),
        calibrated=False, affects_stopping=False,
        rationale=(
            "Uncalibrated 0-1 trace signals only; they are neither probabilities nor "
            "measures of moral correctness and do not currently control stopping."
        ),
    )
