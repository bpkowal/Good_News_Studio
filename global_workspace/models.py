from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class WorkspaceBroadcast:
    constraint: str = "OPEN_DELIBERATION"
    intent: str = "identify_a_defensible_action"
    urgency: float = 0.5
    danger_probability: float = 0.5
    unresolved: str = "NONE"
    contingency_question: str = ""
    reformulation_context: str = ""
    branch_kind: str = "BASE"
    branch_origin_action: str = ""
    branch_condition: str = ""
    branch_fallback: str = ""
    reversal_challenge: str = ""

    def __post_init__(self) -> None:
        self.constraint = self.constraint.strip().upper()[:48] or "OPEN_DELIBERATION"
        self.intent = self.intent.strip().lower()[:80] or "identify_a_defensible_action"
        self.unresolved = self.unresolved.strip().upper()[:48] or "NONE"
        self.contingency_question = " ".join(self.contingency_question.split())[:240]
        self.reformulation_context = " ".join(self.reformulation_context.split())[:600]
        self.branch_kind = self.branch_kind.strip().upper()[:32] or "BASE"
        self.branch_origin_action = " ".join(self.branch_origin_action.split())[:120]
        self.branch_condition = " ".join(self.branch_condition.split())[:240]
        self.branch_fallback = " ".join(self.branch_fallback.split())[:180]
        self.reversal_challenge = " ".join(self.reversal_challenge.split())[:360]
        self.urgency = clamp(self.urgency)
        self.danger_probability = clamp(self.danger_probability)

    def compact(self) -> str:
        return (
            f"constraint={self.constraint}; intent={self.intent}; "
            f"urgency={self.urgency:.2f}; danger={self.danger_probability:.2f}; "
            f"unresolved={self.unresolved}; "
            f"contingency={self.contingency_question or 'NONE'}; "
            f"reformulation={self.reformulation_context or 'NONE'}; "
            f"branch={self.branch_kind}; origin={self.branch_origin_action or 'NONE'}; "
            f"condition={self.branch_condition or 'NONE'}; "
            f"fallback={self.branch_fallback or 'NONE'}"
            f"; reversal_challenge={self.reversal_challenge or 'NONE'}"
        )


@dataclass(slots=True)
class CandidateChunk:
    specialist: str
    constraint: str
    action_scores: dict[str, float]
    surprise: float
    friction: float
    confidence: float
    unresolved: str = "NONE"
    rationale: str = ""
    salience: float = 0.0
    schema_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    recommended_action: str = ""
    baseline_action: str = ""
    testimony_alignment: str = "UNCLEAR"
    previous_action: str = ""
    position_changed: bool = False
    change_justification: str = ""
    conformity_penalty: float = 0.0
    previous_confidence: float = 0.0
    confidence_drift: float = 0.0
    confidence_drift_penalty: float = 0.0
    assumption_status: str = "NOT_AUDITED"
    unsupported_assumption: str = ""
    reversal_condition: str = ""
    boundary_position: str = "NOT_TESTED"
    decisive_axis: str = ""
    boundary_switch_condition: str = ""
    evidence_basis: str = "STATED_FACTS"
    speculative_claim: str = ""
    landscape_cases: dict[str, str] = field(default_factory=dict)
    landscape_decisive_axis: str = ""
    landscape_tiebreaker: str = ""
    landscape_tiebreaker_failure: str = ""
    landscape_search_complete: bool = False
    landscape_search_attempted: bool = False
    landscape_semantic_valid: bool = True
    landscape_validation_errors: list[str] = field(default_factory=list)
    independence_bonus: float = 0.0
    # Preference is how decisively this framework ranks the actions. Epistemic
    # confidence is how likely that ranking is to survive new facts/scrutiny.
    # `confidence` remains a compatibility alias for epistemic_confidence.
    preference_strength: float = -1.0
    epistemic_confidence: float = -1.0
    previous_preference_strength: float = 0.0
    preference_drift: float = 0.0
    preference_drift_penalty: float = 0.0
    decision_rule: str = ""
    factual_reversal_threshold: str = "NONE"
    normative_reversal_threshold: str = "NONE"
    reversal_review_response: str = "NOT_TESTED"
    reversal_review_justification: str = ""
    revised_reversal_condition: str = ""

    def __post_init__(self) -> None:
        self.specialist = self.specialist.strip()[:32]
        self.constraint = self.constraint.strip().upper()[:48] or "UNSPECIFIED"
        self.unresolved = self.unresolved.strip().upper()[:48] or "NONE"
        self.rationale = " ".join(self.rationale.split())[:180]
        self.surprise = clamp(self.surprise)
        self.friction = clamp(self.friction)
        if self.preference_strength < 0:
            ordered = sorted(self.action_scores.values(), reverse=True)
            self.preference_strength = (
                ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
            ) if ordered else 0.0
        if self.epistemic_confidence < 0:
            self.epistemic_confidence = self.confidence
        self.preference_strength = clamp(self.preference_strength)
        self.epistemic_confidence = clamp(self.epistemic_confidence)
        self.previous_preference_strength = clamp(self.previous_preference_strength)
        self.preference_drift = max(-1.0, min(1.0, float(self.preference_drift)))
        self.preference_drift_penalty = clamp(self.preference_drift_penalty)
        self.confidence = self.epistemic_confidence
        self.conformity_penalty = clamp(self.conformity_penalty)
        self.previous_confidence = clamp(self.previous_confidence)
        self.confidence_drift = max(-1.0, min(1.0, float(self.confidence_drift)))
        self.confidence_drift_penalty = clamp(self.confidence_drift_penalty)
        self.change_justification = " ".join(self.change_justification.split())[:180]
        self.assumption_status = self.assumption_status.strip().upper()[:24] or "NOT_AUDITED"
        self.unsupported_assumption = " ".join(self.unsupported_assumption.split())[:180]
        self.reversal_condition = " ".join(self.reversal_condition.split())[:180]
        self.boundary_position = self.boundary_position.strip().upper()[:16] or "NOT_TESTED"
        self.decisive_axis = " ".join(self.decisive_axis.split())[:100]
        self.boundary_switch_condition = " ".join(self.boundary_switch_condition.split())[:180]
        self.evidence_basis = self.evidence_basis.strip().upper()[:24] or "STATED_FACTS"
        self.speculative_claim = " ".join(self.speculative_claim.split())[:180]
        self.landscape_cases = {
            str(action): " ".join(str(reason).split())[:180]
            for action, reason in self.landscape_cases.items()
            if str(action).strip() and " ".join(str(reason).split())
        }
        self.landscape_decisive_axis = " ".join(
            self.landscape_decisive_axis.split()
        )[:120]
        self.landscape_tiebreaker = " ".join(self.landscape_tiebreaker.split())[:180]
        self.landscape_tiebreaker_failure = " ".join(
            self.landscape_tiebreaker_failure.split()
        )[:180]
        if (
            self.landscape_cases
            or self.landscape_decisive_axis
            or self.landscape_tiebreaker
            or self.landscape_tiebreaker_failure
        ):
            self.landscape_search_attempted = True
        self.landscape_validation_errors = [
            " ".join(str(error).split())[:180]
            for error in self.landscape_validation_errors
            if " ".join(str(error).split())
        ][:8]
        self.independence_bonus = clamp(self.independence_bonus)
        self.decision_rule = " ".join(self.decision_rule.split())[:180]
        self.factual_reversal_threshold = (
            " ".join(self.factual_reversal_threshold.split())[:180] or "NONE"
        )
        self.normative_reversal_threshold = (
            " ".join(self.normative_reversal_threshold.split())[:180] or "NONE"
        )
        response = self.reversal_review_response.strip().upper()
        self.reversal_review_response = (
            response if response in {"NOT_TESTED", "ACCEPT", "REVISE", "REJECT"}
            else "NOT_TESTED"
        )
        self.reversal_review_justification = " ".join(
            self.reversal_review_justification.split()
        )[:180]
        self.revised_reversal_condition = " ".join(
            self.revised_reversal_condition.split()
        )[:180]
        self.action_scores = {str(k): clamp(v) for k, v in self.action_scores.items()}


@dataclass(slots=True)
class CycleRecord:
    cycle: int
    broadcast: WorkspaceBroadcast
    candidates: list[CandidateChunk]
    winner: CandidateChunk
    dissent: CandidateChunk | None
    policy: dict[str, float]
    entropy: float
    stable_cycles: int
    elapsed_seconds: float
    received_broadcast: WorkspaceBroadcast | None = None
    is_hypothetical: bool = False


@dataclass(slots=True)
class PlanningBranchEvaluation:
    cycle: int
    origin_action: str
    condition: str
    fallback: str
    selected_action: str
    confidence: float
    policy: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cycle = max(1, int(self.cycle))
        self.origin_action = " ".join(self.origin_action.split())[:120]
        self.condition = " ".join(self.condition.split())[:240]
        self.fallback = " ".join(self.fallback.split())[:180]
        self.selected_action = " ".join(self.selected_action.split())[:120]
        self.confidence = clamp(self.confidence)
        self.policy = {str(action): clamp(score) for action, score in self.policy.items()}


@dataclass(slots=True)
class SynthesisProposal:
    action: str
    grounded_in: list[str]
    addressed_constraints: list[str]
    feasibility: float
    rationale: str
    executable: bool = True
    non_evasive: bool = True
    accepted: bool = False
    rejection_reason: str = ""
    introduced_requirements: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.action = " ".join(self.action.split())[:120]
        self.grounded_in = list(dict.fromkeys(str(item).strip().lower() for item in self.grounded_in if str(item).strip()))[:5]
        self.addressed_constraints = list(dict.fromkeys(str(item).strip().upper() for item in self.addressed_constraints if str(item).strip()))[:5]
        self.feasibility = clamp(self.feasibility)
        self.rationale = " ".join(self.rationale.split())[:240]
        self.rejection_reason = " ".join(self.rejection_reason.split())[:200]
        self.introduced_requirements = list(dict.fromkeys(
            " ".join(str(item).split())[:100]
            for item in self.introduced_requirements
            if " ".join(str(item).split())
        ))[:8]


@dataclass(slots=True)
class FailureCondition:
    synthesis_action: str
    necessary_condition: str
    failure_condition: str
    contingency_question: str
    valid: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        self.synthesis_action = " ".join(self.synthesis_action.split())[:120]
        self.necessary_condition = " ".join(self.necessary_condition.split())[:180]
        self.failure_condition = " ".join(self.failure_condition.split())[:180]
        self.contingency_question = " ".join(self.contingency_question.split())[:240]
        self.error = " ".join(self.error.split())[:200]


@dataclass(slots=True)
class PlanningAssessment:
    """Non-normative implementation analysis for a currently leading action."""

    target_action: str
    activation_reason: str
    feasibility: float
    necessary_condition: str
    failure_condition: str
    fallback: str
    actor_constraints: list[str] = field(default_factory=list)
    resource_constraints: list[str] = field(default_factory=list)
    strategic_forces: list[str] = field(default_factory=list)
    broadcast_worthy: bool = False
    valid: bool = True
    error: str = ""
    grounded_evidence: str = ""
    fallback_available: bool = False
    fallback_availability_reason: str = ""

    def __post_init__(self) -> None:
        self.target_action = " ".join(self.target_action.split())[:120]
        self.activation_reason = " ".join(self.activation_reason.split())[:100]
        self.feasibility = clamp(self.feasibility)
        self.necessary_condition = " ".join(self.necessary_condition.split())[:180]
        self.failure_condition = " ".join(self.failure_condition.split())[:180]
        self.fallback = " ".join(self.fallback.split())[:180]
        self.actor_constraints = self._clean_list(self.actor_constraints)
        self.resource_constraints = self._clean_list(self.resource_constraints)
        self.strategic_forces = self._clean_list(self.strategic_forces)
        self.error = " ".join(self.error.split())[:200]
        self.grounded_evidence = " ".join(self.grounded_evidence.split())[:240]
        self.fallback_availability_reason = " ".join(
            self.fallback_availability_reason.split()
        )[:180]

    @staticmethod
    def _clean_list(values: list[str]) -> list[str]:
        return list(dict.fromkeys(
            " ".join(str(value).split())[:120]
            for value in values
            if " ".join(str(value).split())
        ))[:5]


@dataclass(slots=True)
class WorkspaceAccessDecision:
    """Traceable decision by the capacity-limited workspace access gate."""

    cycle: int
    content_type: str
    admitted: bool
    signals: list[str]
    question: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        self.cycle = max(1, int(self.cycle))
        self.content_type = self.content_type.strip().upper()[:40]
        self.signals = list(dict.fromkeys(
            str(signal).strip().lower().replace(" ", "_")[:64]
            for signal in self.signals
            if str(signal).strip()
        ))[:10]
        self.question = " ".join(self.question.split())[:300]
        self.rationale = " ".join(self.rationale.split())[:240]


@dataclass(slots=True)
class VisibilityAssessment:
    """Non-voting audit of action advantages created by unequal observability."""

    low_observability: bool
    endogenous: bool
    affected_group: str
    mechanism: str
    evidence_quote: str
    action_multipliers: dict[str, float] = field(default_factory=dict)
    activated: bool = False
    valid: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        self.affected_group = " ".join(self.affected_group.split())[:120]
        self.mechanism = " ".join(self.mechanism.split())[:220]
        self.evidence_quote = " ".join(self.evidence_quote.split())[:240]
        # Visibility reduces confidence in an apparent evidential advantage; it
        # cannot veto an action or manufacture support for its alternative.
        self.action_multipliers = {
            " ".join(str(action).split())[:120]: max(0.65, min(1.0, float(value)))
            for action, value in self.action_multipliers.items()
        }
        self.error = " ".join(self.error.split())[:240]


@dataclass(slots=True)
class CalibrationOutcome:
    action: str
    dimension: str
    direction: str
    description: str
    probability: float
    magnitude: float
    unit: str
    horizon: str

    def __post_init__(self) -> None:
        self.action = " ".join(self.action.split())[:120]
        self.dimension = " ".join(self.dimension.split())[:80]
        self.direction = self.direction.strip().upper()[:12]
        self.description = " ".join(self.description.split())[:140]
        self.probability = clamp(self.probability)
        self.magnitude = max(0.0, float(self.magnitude))
        unit = " ".join(self.unit.split())
        # Direction belongs in ``direction``. Keeping it in the unit prevents
        # legitimate within-unit comparisons (for example lives saved/lost).
        unit = re.sub(
            r"\s+(?:gained|lost|added|avoided|saved|prevented|protected|foregone)$",
            "",
            unit,
            flags=re.IGNORECASE,
        )
        self.unit = unit[:48]
        self.horizon = " ".join(self.horizon.split())[:64]


@dataclass(slots=True)
class CategoricalAxis:
    name: str
    action_values: dict[str, str]
    ethical_relevance: str
    fixed_by_scenario: bool = True

    def __post_init__(self) -> None:
        self.name = " ".join(self.name.split())[:80]
        self.action_values = {
            " ".join(str(action).split())[:120]: " ".join(str(value).split())[:140]
            for action, value in self.action_values.items()
            if " ".join(str(action).split()) and " ".join(str(value).split())
        }
        self.ethical_relevance = " ".join(self.ethical_relevance.split())[:180]


@dataclass(slots=True)
class NumericComparison:
    dimension: str
    unit: str
    action_values: dict[str, float]
    absolute_gap: float
    relative_gap: float

    def __post_init__(self) -> None:
        self.dimension = " ".join(self.dimension.split())[:80]
        self.unit = " ".join(self.unit.split())[:48]
        self.action_values = {
            " ".join(str(action).split())[:120]: float(value)
            for action, value in self.action_values.items()
        }
        self.absolute_gap = max(0.0, float(self.absolute_gap))
        self.relative_gap = max(0.0, float(self.relative_gap))


@dataclass(slots=True)
class ProblemReformulation:
    unknowns: list[str]
    outcomes: list[CalibrationOutcome]
    switch_condition: str
    residual_tension: str
    question: str
    grounded_in: list[str]
    fixed_facts: list[str] = field(default_factory=list)
    categorical_axes: list[CategoricalAxis] = field(default_factory=list)
    changed_fixed_facts: list[str] = field(default_factory=list)
    numeric_comparisons: list[NumericComparison] = field(default_factory=list)
    hypothetical: bool = True
    accepted: bool = False
    rejection_reason: str = ""
    probe_result: str = "UNTESTED"
    switch_claim_valid: bool = False

    def __post_init__(self) -> None:
        self.unknowns = list(dict.fromkeys(
            " ".join(str(value).split())[:140]
            for value in self.unknowns
            if " ".join(str(value).split())
        ))[:5]
        self.outcomes = self.outcomes[:6]
        self.switch_condition = " ".join(self.switch_condition.split())[:220]
        self.residual_tension = " ".join(self.residual_tension.split())[:220]
        self.question = " ".join(self.question.split())[:300]
        self.grounded_in = list(dict.fromkeys(
            str(value).strip().lower()[:32]
            for value in self.grounded_in
            if str(value).strip()
        ))[:8]
        self.fixed_facts = self._clean(self.fixed_facts, 140, 8)
        self.categorical_axes = self.categorical_axes[:6]
        self.changed_fixed_facts = self._clean(self.changed_fixed_facts, 140, 8)
        self.numeric_comparisons = self.numeric_comparisons[:8]
        self.rejection_reason = " ".join(self.rejection_reason.split())[:240]
        self.probe_result = self.probe_result.strip().upper()[:32] or "UNTESTED"

    @staticmethod
    def _clean(values: list[str], limit: int, count: int) -> list[str]:
        return list(dict.fromkeys(
            " ".join(str(value).split())[:limit]
            for value in values
            if " ".join(str(value).split())
        ))[:count]

    def compact(self) -> str:
        stakes = "; ".join(
            f"{outcome.dimension}/{outcome.action}: {outcome.probability:.0%} chance of "
            f"{outcome.magnitude:g} {outcome.unit} {outcome.direction.lower()} "
            f"over {outcome.horizon}"
            for outcome in self.outcomes
        )
        axes = "; ".join(
            f"{axis.name}: " + " / ".join(axis.action_values.values())
            for axis in self.categorical_axes
        )
        return " ".join(
            f"HYPOTHETICAL CALIBRATION — {stakes}. Categorical axes: {axes or 'none'}. Remaining tension: "
            f"{self.residual_tension}. {self.question}".split()
        )[:600]


@dataclass(slots=True)
class WorkspaceResult:
    scenario: str
    actions: list[str]
    source_testimonies: dict[str, str] = field(default_factory=dict)
    source_errors: dict[str, str] = field(default_factory=dict)
    source_baselines: dict[str, dict[str, str]] = field(default_factory=dict)
    scenario_facts: dict[str, Any] = field(default_factory=dict)
    cycles: list[CycleRecord] = field(default_factory=list)
    selected_action: str = ""
    confidence: float = 0.0
    halted_by: str = ""
    moral_residue: list[str] = field(default_factory=list)
    compressed_rule: str = ""
    reopen_conditions: list[str] = field(default_factory=list)
    synthesis_proposals: list[SynthesisProposal] = field(default_factory=list)
    failure_conditions: list[FailureCondition] = field(default_factory=list)
    planning_assessments: list[PlanningAssessment] = field(default_factory=list)
    planning_branches: list[PlanningBranchEvaluation] = field(default_factory=list)
    access_decisions: list[WorkspaceAccessDecision] = field(default_factory=list)
    visibility_assessments: list[VisibilityAssessment] = field(default_factory=list)
    problem_reformulations: list[ProblemReformulation] = field(default_factory=list)
    judgment_status: str = "ACTION_RECOMMENDATION"
    current_plurality: str = ""
    epistemic_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
