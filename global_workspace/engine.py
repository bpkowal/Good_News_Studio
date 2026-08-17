from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .middleware.moral_residue import collect_moral_residue, collect_reopen_conditions
from .construct_validity import (
    assess_termination, collect_typed_residue, describe_access,
    estimate_further_deliberation,
)
from .middleware.reversal_audit import (
    build_reversal_audit_request,
    dissent_reversal_condition,
)
from .models import AutonomyAssessment, CandidateChunk, ContingencyFeasibilityAssessment, CycleRecord, FailureCondition, PlanningAssessment, PlanningBranchEvaluation, ProblemReformulation, SynthesisProposal, SynthesisViabilityAssessment, VisibilityAssessment, WorkspaceAccessDecision, WorkspaceBroadcast, WorkspaceResult, clamp
from .semantic_invariants import (
    SemanticProposition,
    compile_preference_rule,
    validate_transformation,
)
from .trace_health import audit_trace_health
from .decision_boundaries import select_collective_reversal_boundary
from .graph_transactions import SemanticGraphStore
from .rawls_ledger import (
    apply_rawls_ledger_transaction,
    committed_rawls_positions,
)
from .utilitarian_ledger import (
    apply_utilitarian_ledger_transaction,
    committed_utilitarian_consequences,
)
from .deontology_ledger import (
    apply_deontological_ledger_transaction,
    committed_deontological_assessments,
)
from .virtue_ledger import (
    apply_virtue_ledger_transaction,
    committed_virtue_assessments,
)
from .scenario_semantics import compile_scenario_graph
from .scenario_semantics import compile_action_burdens, compile_execution_obstacles
from .semantic_state import (
    project_authoritative_semantic_state,
    select_committed_reversal_boundary,
)
from .ev_dominance import assess_ev_dominance
from .contingency_graph import (
    certify_fallback_availability, validate_contingency_graph_dict,
)
from .structured_io import (
    ModelCallBudgetExceeded, ModelCallUnavailable, begin_model_call_cycle,
)


SynthesisCallback = Callable[
    [str, Sequence[str], Sequence[CandidateChunk], WorkspaceBroadcast],
    SynthesisProposal | None,
]
ExtensionCallback = Callable[[WorkspaceResult], int]
ContingencyCallback = Callable[[WorkspaceResult], FailureCondition]
ContingencyFeasibilityCallback = Callable[
    [FailureCondition], ContingencyFeasibilityAssessment
]
PlanningCallback = Callable[
    [str, Sequence[str], str, WorkspaceBroadcast, Sequence[CandidateChunk], str],
    PlanningAssessment,
]
ReformulationCallback = Callable[
    [str, Sequence[str], Sequence[CandidateChunk]],
    ProblemReformulation,
]
VisibilityCallback = Callable[[str, Sequence[str]], VisibilityAssessment]
AutonomyCallback = Callable[[str, Sequence[str]], AutonomyAssessment]
CheckpointCallback = Callable[[WorkspaceResult], None]


class Specialist(Protocol):
    name: str

    def evaluate(
        self,
        scenario: str,
        actions: Sequence[str],
        broadcast: WorkspaceBroadcast,
    ) -> CandidateChunk: ...


def _apply_framework_ledger_uncertainty(
    candidate: CandidateChunk,
    errors: Sequence[str],
    penalty: float = 0.35,
    *,
    state_status: str = "COMMITTED_WITH_UNCERTAINTY",
) -> None:
    """Damp one unsupported framework update without dropping its visible vote."""
    if candidate.framework_grounding_penalty < penalty:
        retention = 1.0 - penalty
        candidate.action_scores = {
            action: 0.5 + (score - 0.5) * retention
            for action, score in candidate.action_scores.items()
        }
        ordered = sorted(candidate.action_scores.values(), reverse=True)
        candidate.preference_strength = (
            ordered[0] - ordered[1] if len(ordered) > 1
            else (ordered[0] if ordered else 0.0)
        )
        candidate.friction = candidate.preference_strength
    candidate.framework_grounding_penalty = max(
        candidate.framework_grounding_penalty, penalty
    )
    candidate.framework_validation_errors.extend(
        error for error in errors
        if error not in candidate.framework_validation_errors
    )
    candidate.epistemic_confidence = min(candidate.epistemic_confidence, 0.55)
    candidate.confidence = candidate.epistemic_confidence
    if state_status == "UPDATE_REJECTED":
        # The transaction did not alter authoritative state; the rejected
        # proposal is visible in the diagnostics but is not framework loss.
        candidate.framework_constraint_retained = True
        candidate.framework_retention_status = "UPDATE_REJECTED"
    else:
        candidate.framework_constraint_retained = False
        candidate.framework_retention_status = "COMMITTED_WITH_UNCERTAINTY"


@dataclass(slots=True)
class WorkspaceConfig:
    max_cycles: int = 4
    high_urgency_cycles: int = 2
    time_budget_seconds: float = 180.0
    entropy_threshold: float = 0.62
    stable_cycles_required: int = 2
    surprise_weight: float = 0.28
    urgency_weight: float = 0.24
    friction_weight: float = 0.28
    minority_weight: float = 0.12
    redundancy_weight: float = 0.18
    min_valid_specialists: int = 2
    enable_synthesis: bool = True
    synthesis_after_cycle: int = 1
    synthesis_min_entropy: float = 0.55
    max_cycle_extensions: int = 1
    enable_planning: bool = True
    planning_entropy_threshold: float = 0.55
    max_planning_branches: int = 2
    enable_consensus_audit: bool = True
    consensus_audit_max_entropy: float = 0.50
    consensus_audit_min_signals: int = 3
    enable_reversal_audit: bool = True
    graph_rejection_policy: str = "RETAIN_VOTE"
    enable_ev_dominance_breaker: bool = True
    ev_dominance_ratio: float = 5.0
    ev_majority_fraction: float = 0.6


class WorkspaceEngine:
    def __init__(self, specialists: Sequence[Specialist], config: WorkspaceConfig | None = None):
        if not specialists:
            raise ValueError("At least one specialist is required")
        self.specialists = list(specialists)
        self.config = config or WorkspaceConfig()

    def _salience(
        self,
        candidate: CandidateChunk,
        broadcast: WorkspaceBroadcast,
        constraint_counts: dict[str, int],
        minority_bonus: float,
    ) -> float:
        if not candidate.schema_valid:
            return 0.0
        cfg = self.config
        redundancy = min(1.0, constraint_counts.get(candidate.constraint, 0) / 2)
        value = (
            cfg.surprise_weight * candidate.surprise
            + cfg.urgency_weight * broadcast.urgency
            + cfg.friction_weight * candidate.friction
            + cfg.minority_weight * minority_bonus
            - cfg.redundancy_weight * redundancy
        ) * (0.5 + 0.5 * candidate.epistemic_confidence)
        if candidate.landscape_search_attempted and not candidate.landscape_semantic_valid:
            value *= 0.55
        return value

    @staticmethod
    def _policy(
        candidates: Sequence[CandidateChunk],
        actions: Sequence[str],
        visibility_multipliers: dict[str, float] | None = None,
        visibility_review: bool = True,
    ) -> dict[str, float]:
        candidates = [candidate for candidate in candidates if candidate.schema_valid]
        if not candidates:
            return {action: 1.0 / len(actions) for action in actions}
        totals = {action: 0.0 for action in actions}
        weight_sum = 0.0
        for candidate in candidates:
            weight = max(0.05, candidate.epistemic_confidence)
            weight_sum += weight
            for action in actions:
                score = candidate.action_scores.get(action, 0.0)
                # Visibility first changes the shared causal representation. Its
                # confidence adjustment is secondary and applies only when a
                # delegate fails to incorporate the admitted upward-harm probe.
                incorporated_visibility = (
                    visibility_review
                    and candidate.visibility_response in {"ACCEPT", "QUALIFY"}
                    and candidate.visibility_harm_revision == "UPWARD"
                )
                multiplier = (
                    (visibility_multipliers or {}).get(action, 1.0)
                    if visibility_review and not incorporated_visibility else 1.0
                )
                if score > 0.5 and multiplier < 1.0:
                    score = 0.5 + (score - 0.5) * multiplier
                totals[action] += weight * score
        means = {action: score / weight_sum for action, score in totals.items()}
        temperature = 0.25
        exps = {action: math.exp(score / temperature) for action, score in means.items()}
        denominator = sum(exps.values()) or 1.0
        return {action: value / denominator for action, value in exps.items()}

    @staticmethod
    def _normalized_entropy(policy: dict[str, float]) -> float:
        if len(policy) <= 1:
            return 0.0
        entropy = -sum(p * math.log(p) for p in policy.values() if p > 0)
        return entropy / math.log(len(policy))

    @staticmethod
    def _dissent(candidates: Sequence[CandidateChunk], selected_action: str) -> CandidateChunk | None:
        alternatives = [
            candidate
            for candidate in candidates
            if candidate.schema_valid
            and candidate.action_scores
            and max(candidate.action_scores.values())
            - candidate.action_scores.get(selected_action, 0.0) > 0.15
        ]
        if not alternatives:
            return None
        return max(alternatives, key=lambda c: c.friction * c.epistemic_confidence)

    @staticmethod
    def _dissent_reversal_condition(
        dissent: CandidateChunk | None, selected_action: str
    ) -> str:
        """Compatibility wrapper for the standalone reversal middleware."""
        return dissent_reversal_condition(dissent, selected_action)

    @staticmethod
    def _assess_synthesis_viability(
        result: WorkspaceResult,
    ) -> SynthesisViabilityAssessment | None:
        """Require post-review support before spending calls on a failure branch."""
        proposal = next(
            (item for item in reversed(result.synthesis_proposals) if item.accepted),
            None,
        )
        if proposal is None:
            return None
        review = next(
            (
                cycle for cycle in reversed(result.cycles)
                if (cycle.received_broadcast or cycle.broadcast).constraint
                == "SYNTHESIS_REVIEW"
                and proposal.action in cycle.policy
            ),
            None,
        )
        if review is None:
            return SynthesisViabilityAssessment(
                proposal.action, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, False,
                "no completed specialist review of the admitted synthesis",
            )
        valid = [candidate for candidate in review.candidates if candidate.schema_valid]
        recommendation_count = sum(
            candidate.recommended_action == proposal.action for candidate in valid
        )
        admissible_names = {
            candidate.specialist for candidate in valid
            if candidate.recommended_action == proposal.action
            or candidate.action_admissibility.get(proposal.action)
            in {"REQUIRED", "PERMISSIBLE"}
        }
        rejection_count = sum(
            candidate.action_admissibility.get(proposal.action) == "REJECTED"
            for candidate in valid
        )
        mean_score = (
            sum(candidate.action_scores.get(proposal.action, 0.0) for candidate in valid)
            / len(valid)
            if valid else 0.0
        )
        policy_support = review.policy.get(proposal.action, 0.0)
        leader_support = max(review.policy.values(), default=0.0)
        majority = len(valid) // 2 + 1
        cross_framework_support = (
            recommendation_count >= 1 and len(admissible_names) >= 2
        )
        broad_acceptability = (
            len(admissible_names) >= majority and mean_score >= 0.50
        )
        viable = bool(valid) and rejection_count < majority and (
            cross_framework_support or broad_acceptability
        )
        if viable:
            reason = (
                "post-review synthesis remains decision-relevant: "
                f"recommended by {recommendation_count}; admissible to "
                f"{len(admissible_names)}/{len(valid)}"
            )
        else:
            reason = (
                "post-review synthesis lacks actionable support: "
                f"recommended by {recommendation_count}; admissible to "
                f"{len(admissible_names)}/{len(valid)}; mean score={mean_score:.2f}"
            )
        return SynthesisViabilityAssessment(
            proposal.action,
            review.cycle,
            len(valid),
            recommendation_count,
            len(admissible_names),
            rejection_count,
            mean_score,
            policy_support,
            leader_support,
            viable,
            reason,
        )

    @staticmethod
    def _snapshot_specialist_state(specialists: Sequence[Specialist]) -> list[tuple[Specialist, dict]]:
        """Capture recurrent state that a hypothetical branch must not overwrite."""
        names = (
            "previous_recommendation_id", "previous_confidence", "previous_context", "assumption_status",
            "unsupported_assumption", "reversal_condition", "epistemic_commitments",
            "previous_framework_state",
        )
        return [
            (specialist, {
                name: (
                    list(getattr(specialist, name))
                    if isinstance(getattr(specialist, name), list)
                    else dict(getattr(specialist, name))
                    if isinstance(getattr(specialist, name), dict)
                    else getattr(specialist, name)
                )
                for name in names if hasattr(specialist, name)
            })
            for specialist in specialists
        ]

    @staticmethod
    def _restore_specialist_state(snapshot: Sequence[tuple[Specialist, dict]]) -> None:
        for specialist, values in snapshot:
            for name, value in values.items():
                setattr(specialist, name, value)

    @staticmethod
    def _explicit_implementation_obstacle(
        scenario: str, broadcast: WorkspaceBroadcast, actions: Sequence[str],
        target_action: str = "",
    ) -> bool:
        """Require a typed obstacle linked to an action, not generic danger words."""
        context = " ".join(
            (scenario, broadcast.contingency_question, broadcast.reformulation_context)
        )
        facts = compile_execution_obstacles(context, actions)
        if not target_action:
            return bool(facts)
        try:
            target_id = f"A{list(actions).index(target_action)}"
        except ValueError:
            return False
        return any(fact.affected_action_node_id == target_id for fact in facts)

    @staticmethod
    def _validate_planning_assessment(
        assessment: PlanningAssessment,
        scenario: str,
        broadcast: WorkspaceBroadcast,
        actions: Sequence[str],
    ) -> PlanningAssessment:
        if not assessment.valid:
            return assessment
        evidence = assessment.grounded_evidence.casefold().strip()
        errors: list[str] = []
        try:
            expected_target_node = f"A{list(actions).index(assessment.target_action)}"
        except ValueError:
            expected_target_node = ""
        # Identity and grounding are structural. The engine has independently
        # established that the scenario/workspace contains an implementation
        # obstacle before invoking planning; the planner is system-bound to the
        # canonical ActionNode. Its prose is retained for audit, not reinterpreted
        # as an identifier or subjected to brittle substring matching.
        if not expected_target_node:
            errors.append("planning target is not an existing action")
        elif not assessment.target_action_node_id:
            # Callbacks created before graph-addressed planning do not carry the
            # redundant ID. Bind it from the system-owned target action rather
            # than making generated prose reconstruct identity.
            assessment.target_action_node_id = expected_target_node
        elif assessment.target_action_node_id != expected_target_node:
            errors.append("planning target does not match the canonical ActionNode")
        if not WorkspaceEngine._explicit_implementation_obstacle(
            scenario, broadcast, actions, assessment.target_action
        ):
            errors.append("planning activation has no scenario-grounded implementation obstacle")
        if assessment.broadcast_worthy and len(evidence.split()) < 2:
            errors.append("material implementation obstacle lacks an audit provenance note")
        if assessment.fallback not in actions or assessment.fallback == assessment.target_action:
            errors.append("fallback is not a distinct existing action")
        if not assessment.fallback_available:
            errors.append("fallback is not physically available after the failure")
        if len(assessment.fallback_availability_reason.split()) < 3:
            errors.append("fallback availability is not explained")

        failure = assessment.failure_condition.casefold()
        target = assessment.target_action.casefold()
        fallback = assessment.fallback.casefold()
        shared_control_failure = re.search(
            r"\b(?:loss|lose|loses|lost|jam|jammed|failure|fails|failed|unable|"
            r"unavailable|incapacitated|dies|death)\b.*\b(?:control|steer|actuat|"
            r"decid|choose|capacity|authority)\w*",
            failure,
        )
        target_words = set(re.findall(r"[a-z]{4,}", target))
        fallback_words = set(re.findall(r"[a-z]{4,}", fallback))
        if shared_control_failure and target_words & fallback_words:
            errors.append("failure disables a capability shared by target and fallback")
        if errors:
            assessment.valid = False
            assessment.broadcast_worthy = False
            assessment.error = "; ".join(errors)
        return assessment

    def _consensus_access_decision(
        self,
        cycle_number: int,
        scenario: str,
        actions: Sequence[str],
        candidates: Sequence[CandidateChunk],
        selected_action: str,
        entropy: float,
        dissent: CandidateChunk | None,
        scenario_facts: dict | None,
        source_testimonies: dict[str, str] | None,
    ) -> WorkspaceAccessDecision | None:
        """Detect agreement that may be produced by shared unsupported assumptions."""
        if (
            not self.config.enable_consensus_audit
            or source_testimonies is None
            or cycle_number > 2
            or dissent is not None
            or entropy > self.config.consensus_audit_max_entropy
        ):
            return None
        valid = [candidate for candidate in candidates if candidate.schema_valid]
        if len(valid) < self.config.min_valid_specialists:
            return None

        signals: list[str] = ["rapid_consensus"]
        recommended = [candidate.recommended_action for candidate in valid]
        if recommended and all(action == selected_action for action in recommended):
            signals.append("unanimous_recommendation")

        vectors = [
            tuple(round(candidate.action_scores.get(action, 0.0), 3) for action in actions)
            for candidate in valid
        ]
        largest_score_cluster = max(
            (
                sum(
                    all(abs(first - second) <= 0.10 for first, second in zip(anchor, vector))
                    for vector in vectors
                )
                for anchor in vectors
            ),
            default=0,
        )
        if vectors and largest_score_cluster / len(vectors) >= 0.8:
            signals.append("homogeneous_score_vectors")

        uncertainty_language = re.search(
            r"\b(?:could|might|may|risks?|risky|uncertain|uncertainty|"
            r"unspecified|depends?|likelihood|probability|severity|extent)\b|"
            r"\bunknown\s+(?:probability|likelihood|number|extent|severity|duration|outcome)\b",
            scenario,
            flags=re.IGNORECASE,
        )
        comparative_uncertainty = re.search(
            r"\b(?:greater|lesser|more|less|relative|comparative)\s+(?:harm|risk|cost|benefit)\b|"
            r"\b(?:outweighs?|trade[- ]?off|which\s+is\s+(?:the\s+)?greater\s+harm|"
            r"how\s+(?:many|likely|severe|long))\b",
            scenario,
            flags=re.IGNORECASE,
        )
        if comparative_uncertainty:
            signals.append("comparative_magnitude_unresolved")
        if not (scenario_facts or {}) and uncertainty_language:
            signals.append("sparse_facts_with_uncertainty")

        conditional_pattern = re.compile(
            r"\b(?:if|unless|depends?|conditional|uncertain|unspecified|missing information|could flip)\b",
            flags=re.IGNORECASE,
        )
        conditional_sources = sum(
            bool(conditional_pattern.search(testimony))
            for testimony in source_testimonies.values()
            if testimony
        )
        if conditional_sources >= max(1, math.ceil(len(source_testimonies) / 3)):
            signals.append("conditional_source_testimony")

        if all(candidate.unresolved == "NONE" for candidate in valid) and (
            uncertainty_language or conditional_sources
        ):
            signals.append("uncertainty_erased_by_delegates")
        if any(candidate.unresolved != "NONE" for candidate in valid):
            signals.append("delegate_uncertainty_present")

        immediate_actions = {
            index for index, action in enumerate(actions)
            if re.search(r"\b(?:immediately|always|now)\b", action, re.IGNORECASE)
        }
        permanent_actions = {
            index for index, action in enumerate(actions)
            if re.search(r"\b(?:indefinitely|never|permanently)\b", action, re.IGNORECASE)
        }
        opposing_extremes = bool(
            immediate_actions and permanent_actions
            and any(first != second for first in immediate_actions for second in permanent_actions)
        )
        if opposing_extremes:
            signals.append("asymmetric_action_extremity")

        structural_signal = any(signal in signals for signal in (
            "homogeneous_score_vectors",
            "sparse_facts_with_uncertainty",
            "comparative_magnitude_unresolved",
            "asymmetric_action_extremity",
            "delegate_uncertainty_present",
        ))
        conditionality_erased = (
            "conditional_source_testimony" in signals
            and "uncertainty_erased_by_delegates" in signals
            and "unanimous_recommendation" in signals
        )
        if conditionality_erased:
            signals.append("conditionality_collapsed_into_consensus")
            structural_signal = True
        admitted = structural_signal and len(signals) >= self.config.consensus_audit_min_signals
        return WorkspaceAccessDecision(
            cycle=cycle_number,
            content_type="CONSENSUS_AUDIT",
            admitted=admitted,
            signals=signals,
            question=(
                f"The apparent dominance of '{selected_action}' may depend on unsupported "
                "or asymmetric assumptions. Which assumption is unstated, and what plausible "
                "facts would reverse the recommendation?"
                if admitted else ""
            ),
            rationale=(
                "Rapid agreement requires one epistemic challenge before convergence."
                if admitted else "Consensus did not meet the suspicious-access threshold."
            ),
        )

    def run(
        self,
        scenario: str,
        actions: Sequence[str],
        initial_broadcast: WorkspaceBroadcast | None = None,
        progress: Callable[[str], None] | None = None,
        synthesize: SynthesisCallback | None = None,
        request_extension: ExtensionCallback | None = None,
        analyze_contingency: ContingencyCallback | None = None,
        verify_contingency_feasibility: ContingencyFeasibilityCallback | None = None,
        analyze_plan: PlanningCallback | None = None,
        scenario_facts: dict | None = None,
        source_action_legend: dict[str, str] | None = None,
        presentation_actions: Sequence[str] | None = None,
        source_testimonies: dict[str, str] | None = None,
        reformulate_problem: ReformulationCallback | None = None,
        assess_visibility: VisibilityCallback | None = None,
        assess_autonomy: AutonomyCallback | None = None,
        checkpoint: CheckpointCallback | None = None,
    ) -> WorkspaceResult:
        clean_actions = list(dict.fromkeys(a.strip() for a in actions if a.strip()))[:5]
        if len(clean_actions) < 2:
            raise ValueError("At least two distinct actions are required")

        broadcast = initial_broadcast or WorkspaceBroadcast()
        result = WorkspaceResult(
            scenario=scenario,
            actions=clean_actions,
            presentation_actions=list(presentation_actions or clean_actions),
            source_action_legend=dict(source_action_legend or {}),
        )
        # Canonical actions and explicit observability facts are system-owned graph
        # state. Delegate transactions may extend this graph but cannot redefine
        # the identity of the action currently being planned.
        graph_store = SemanticGraphStore(compile_scenario_graph(scenario, clean_actions))
        visibility_multipliers = {action: 1.0 for action in clean_actions}
        visibility: VisibilityAssessment | None = None
        if assess_visibility is not None:
            if progress:
                progress("Auditing whether unequal observability creates epistemic exclusion...")
            try:
                visibility = assess_visibility(scenario, tuple(clean_actions))
            except Exception as exc:
                visibility = VisibilityAssessment(
                    False, False, "", "", "", visibility_multipliers,
                    valid=False, error=f"visibility audit unavailable: {exc}",
                )
            result.visibility_assessments.append(visibility)
            if visibility.valid and visibility.activated:
                visibility_multipliers.update(visibility.action_multipliers)
                if progress:
                    penalized = [
                        f"{action}×{value:.2f}"
                        for action, value in visibility_multipliers.items()
                        if value < 1.0
                    ]
                    progress(
                        "Visibility audit activated: "
                        + ", ".join(penalized)
                        + f"; mechanism={visibility.mechanism}"
                    )
            elif progress and not visibility.valid:
                progress(f"Visibility audit ignored: {visibility.error}")
        autonomy: AutonomyAssessment | None = None
        if assess_autonomy is not None:
            if progress:
                progress("Auditing candidate actions for coercion and competent refusal...")
            try:
                autonomy = assess_autonomy(scenario, tuple(clean_actions))
            except Exception as exc:
                autonomy = AutonomyAssessment(
                    {action: "NONE" for action in clean_actions},
                    {action: False for action in clean_actions},
                    {action: "" for action in clean_actions},
                    valid=False, error=f"autonomy audit unavailable: {exc}",
                )
            result.autonomy_assessments.append(autonomy)
            if progress and autonomy.activated:
                tagged = [
                    f"{action}={tag}"
                    for action, tag in autonomy.action_tags.items() if tag != "NONE"
                ]
                progress("Autonomy audit activated: " + ", ".join(tagged))
            elif progress and not autonomy.valid:
                progress(f"Autonomy audit ignored: {autonomy.error}")
        started = time.monotonic()
        previous_action = ""
        stable_cycles = 0
        constraint_counts: dict[str, int] = {}
        previous_dissent: CandidateChunk | None = None
        synthesis_attempted = False
        planned_contexts: set[tuple[str, str]] = set()
        planning_branches_used = 0
        planning_resume_broadcast: WorkspaceBroadcast | None = None
        reformulation_resume_broadcast: WorkspaceBroadcast | None = None
        reversal_resume_broadcast: WorkspaceBroadcast | None = None
        contingency_resume_broadcast: WorkspaceBroadcast | None = None
        consensus_audit_attempted = False
        access_signal_signatures: set[tuple[str, ...]] = set()
        reformulation_attempted = False
        reversal_audit_attempted = False
        visibility_broadcast_attempted = False
        reformulation_baseline_action = ""
        extensions_used = 0
        cycle_limit = (
            min(self.config.max_cycles, self.config.high_urgency_cycles)
            if broadcast.urgency >= 0.8
            else self.config.max_cycles
        )

        cycle_number = 1
        while cycle_number <= cycle_limit:
            begin_model_call_cycle(cycle_number)
            received_broadcast = broadcast
            is_planning_branch = received_broadcast.branch_kind == "PLANNING_CONTINGENCY"
            is_reformulation_probe = received_broadcast.constraint == "PROBLEM_REFORMULATION"
            is_reversal_probe = received_broadcast.constraint == "REVERSAL_AUDIT"
            is_contingency_probe = received_broadcast.constraint == "CONTINGENCY_REVIEW"
            is_counterfactual = (
                is_planning_branch or is_reformulation_probe or is_reversal_probe
                or is_contingency_probe
            )
            cycle_actions = (
                list(received_broadcast.contingency_fallback_actions)
                if is_contingency_probe
                and len(received_broadcast.contingency_fallback_actions) == 2
                else clean_actions
            )
            specialist_snapshot = (
                self._snapshot_specialist_state(self.specialists) if is_counterfactual else []
            )
            if progress:
                progress(f"Cycle {cycle_number}/{cycle_limit} — broadcast {broadcast.constraint}")
            candidates = []
            terminal_model_failure = False
            for index, specialist in enumerate(self.specialists, start=1):
                call_started = time.monotonic()
                if progress:
                    progress(
                        f"  [{index}/{len(self.specialists)}] {specialist.name} delegate thinking..."
                    )
                try:
                    candidate = specialist.evaluate(scenario, cycle_actions, broadcast)
                except ModelCallBudgetExceeded as exc:
                    result.halted_by = "model_call_budget"
                    if progress:
                        progress(f"  halting before next model call: {exc}")
                    break
                except ModelCallUnavailable as exc:
                    # Provider transport failure is a missing observation, not a
                    # malformed moral judgment. Exclude only this response and
                    # continue when later delegates may still provide quorum.
                    candidate = CandidateChunk(
                        specialist=specialist.name,
                        constraint="MODEL_UNAVAILABLE",
                        action_scores={action: 0.5 for action in cycle_actions},
                        surprise=0.0,
                        friction=0.0,
                        confidence=0.0,
                        unresolved="RETRY_MODEL_CALL",
                        rationale=f"Delegate unavailable after {exc.category} failure.",
                        schema_valid=False,
                        validation_errors=[str(exc)[:300]],
                    )
                    terminal_model_failure = exc.terminal
                    if exc.terminal:
                        result.halted_by = "model_backend_unavailable"
                if is_contingency_probe and candidate.schema_valid:
                    contingency_errors = []
                    if candidate.contingency_choice not in cycle_actions:
                        contingency_errors.append(
                            "contingency response did not select a typed fallback"
                        )
                    if candidate.recommended_action != candidate.contingency_choice:
                        contingency_errors.append(
                            "contingency recommendation differs from typed fallback choice"
                        )
                    if len(candidate.contingency_justification.split()) < 3:
                        contingency_errors.append(
                            "contingency response lacks conditional justification"
                        )
                    if contingency_errors:
                        candidate.schema_valid = False
                        candidate.contingency_response_valid = False
                        candidate.contingency_response_error = "; ".join(
                            contingency_errors
                        )
                        candidate.validation_errors.extend(contingency_errors)
                if autonomy is not None and autonomy.valid and candidate.schema_valid:
                    tag = autonomy.action_tags.get(candidate.recommended_action, "NONE")
                    candidate.coercion_tag = tag
                    if (
                        tag != "NONE"
                        and not autonomy.catastrophic_harm_threshold.get(
                            candidate.recommended_action, False
                        )
                    ):
                        candidate.coercion_surcharge = 1.0 - autonomy.surcharge_multiplier
                        candidate.epistemic_confidence *= autonomy.surcharge_multiplier
                        candidate.confidence = candidate.epistemic_confidence
                candidates.append(candidate)
                if (
                    candidate.schema_valid
                    and candidate.graph_update_proposal
                    and not is_counterfactual
                ):
                    transaction = graph_store.apply(
                        candidate.graph_update_proposal,
                        cycle=cycle_number,
                        specialist=candidate.specialist,
                        expected_from_action=candidate.recommended_action,
                        allowed_actions=tuple(clean_actions),
                        expected_source_text=candidate.factual_reversal_threshold,
                        current_action_values=candidate.expected_value_estimates,
                        rejection_policy=(
                            "DROP_VOTE"
                            if self.config.graph_rejection_policy == "DROP_VOTE"
                            else "RETAIN_VOTE"
                        ),
                    )
                    if transaction.vote_disposition == "DROPPED":
                        candidate.schema_valid = False
                        candidate.validation_errors.append(
                            "decision-critical graph update rejected; vote dropped"
                        )
                    result.graph_transactions = graph_store.transaction_dicts()
                    result.semantic_graphs = (
                        [graph_store.graph_dict()] if graph_store.graph.nodes else []
                    )
                    if progress and transaction.status == "REJECTED":
                        progress(
                            f"    graph update rejected; previous state preserved: "
                            + " | ".join(transaction.errors[:2])
                        )
                if (
                    candidate.schema_valid
                    and candidate.specialist == "utilitarian"
                    and candidate.utilitarian_ledger_proposal
                    and not is_counterfactual
                ):
                    util_transaction = apply_utilitarian_ledger_transaction(
                        graph_store,
                        candidate.utilitarian_ledger_proposal,
                        cycle=cycle_number,
                        specialist=candidate.specialist,
                        allowed_actions=tuple(clean_actions),
                    )
                    result.utilitarian_consequence_ledger = (
                        committed_utilitarian_consequences(graph_store.graph)
                    )
                    result.graph_transactions = graph_store.transaction_dicts()
                    result.semantic_graphs = [graph_store.graph_dict()]
                    if util_transaction.status == "COMMITTED_WITH_UNCERTAINTY":
                        _apply_framework_ledger_uncertainty(
                            candidate, util_transaction.errors
                        )
                    if progress and util_transaction.status != "COMMITTED":
                        progress(
                            "    Utilitarian consequence ledger "
                            f"{util_transaction.status.lower()}: "
                            + " | ".join(util_transaction.errors[:2])
                        )
                if (
                    candidate.schema_valid
                    and candidate.specialist == "deontological"
                    and candidate.deontological_ledger_proposal
                    and not is_counterfactual
                ):
                    deon_transaction = apply_deontological_ledger_transaction(
                        graph_store,
                        candidate.deontological_ledger_proposal,
                        cycle=cycle_number,
                        specialist=candidate.specialist,
                        allowed_actions=tuple(clean_actions),
                    )
                    result.deontological_duty_ledger = (
                        committed_deontological_assessments(graph_store.graph)
                    )
                    result.graph_transactions = graph_store.transaction_dicts()
                    result.semantic_graphs = [graph_store.graph_dict()]
                    if deon_transaction.status == "COMMITTED_WITH_UNCERTAINTY":
                        _apply_framework_ledger_uncertainty(
                            candidate, deon_transaction.errors
                        )
                    if progress and deon_transaction.status != "COMMITTED":
                        progress(
                            "    Deontological duty ledger "
                            f"{deon_transaction.status.lower()}: "
                            + " | ".join(deon_transaction.errors[:2])
                        )
                if (
                    candidate.schema_valid
                    and candidate.specialist == "rawlsian"
                    and candidate.rawls_position_proposal
                    and not is_counterfactual
                ):
                    rawls_transaction = apply_rawls_ledger_transaction(
                        graph_store,
                        candidate.rawls_position_proposal,
                        cycle=cycle_number,
                        specialist=candidate.specialist,
                        allowed_actions=tuple(clean_actions),
                    )
                    result.graph_transactions = graph_store.transaction_dicts()
                    result.semantic_graphs = (
                        [graph_store.graph_dict()] if graph_store.graph.nodes else []
                    )
                    result.rawlsian_position_ledger = committed_rawls_positions(
                        graph_store.graph
                    )
                    if (
                        rawls_transaction.status.startswith("COMMITTED")
                        and hasattr(specialist, "previous_framework_state")
                        and result.rawlsian_position_ledger
                    ):
                        committed_positions = result.rawlsian_position_ledger
                        specialist.previous_framework_state = {
                            "ranking_basis": committed_positions[0].get(
                                "ranking_basis", "UNRESOLVED"
                            ),
                            "liberty_status": {
                                str(item.get("canonical_action_id", "")): item.get(
                                    "liberty_status", "UNKNOWN"
                                )
                                for item in committed_positions
                            },
                            # Recurrent prompts need the stable moral relation,
                            # not graph plumbing, provenance, or evidence IDs.
                            "positions": [
                                {
                                    "action_id": item.get("canonical_action_id", ""),
                                    "group": item.get("group", ""),
                                    "dimension": item.get("dimension", "UNKNOWN"),
                                    "effect": item.get("effect", "UNCERTAIN"),
                                    "proposed_effect": item.get(
                                        "proposed_effect", item.get("effect", "UNCERTAIN")
                                    ),
                                }
                                for item in committed_positions
                            ],
                        }
                    if rawls_transaction.status == "COMMITTED_WITH_UNCERTAINTY":
                        _apply_framework_ledger_uncertainty(
                            candidate, rawls_transaction.errors
                        )
                    if progress and rawls_transaction.status != "COMMITTED":
                        progress(
                            "    Rawls position ledger "
                            f"{rawls_transaction.status.lower()}: "
                            + " | ".join(rawls_transaction.errors[:2])
                        )
                if (
                    candidate.schema_valid
                    and candidate.specialist == "virtue"
                    and candidate.virtue_character_proposal
                    and not is_counterfactual
                ):
                    virtue_transaction = apply_virtue_ledger_transaction(
                        graph_store,
                        candidate.virtue_character_proposal,
                        cycle=cycle_number,
                        specialist=candidate.specialist,
                        allowed_actions=tuple(clean_actions),
                    )
                    result.virtue_character_ledger = committed_virtue_assessments(
                        graph_store.graph
                    )
                    result.graph_transactions = graph_store.transaction_dicts()
                    result.semantic_graphs = [graph_store.graph_dict()]
                    if virtue_transaction.status == "REJECTED":
                        _apply_framework_ledger_uncertainty(
                            candidate, virtue_transaction.errors,
                            state_status="UPDATE_REJECTED",
                        )
                    if (
                        virtue_transaction.status == "COMMITTED"
                        and hasattr(specialist, "previous_framework_state")
                        and result.virtue_character_ledger
                    ):
                        committed_character = result.virtue_character_ledger
                        specialist.previous_framework_state = {
                            "ranking_basis": committed_character[0].get(
                                "ranking_basis", "UNRESOLVED"
                            ),
                            "assessments": [
                                {
                                    "action_id": item.get("canonical_action_id", ""),
                                    "verdict": item.get("verdict", "UNCERTAIN"),
                                    "actor_role": item.get("actor_role", ""),
                                }
                                for item in committed_character
                            ],
                        }
                    if progress and virtue_transaction.status != "COMMITTED":
                        progress(
                            "    Virtue character ledger "
                            f"{virtue_transaction.status.lower()}: "
                            + " | ".join(virtue_transaction.errors[:2])
                        )
                if progress:
                    validation_note = (
                        f"; error={candidate.validation_errors[0]}"
                        if not candidate.schema_valid and candidate.validation_errors
                        else ""
                    )
                    if candidate.conformity_penalty:
                        validation_note += (
                            f"; conformity_penalty={candidate.conformity_penalty:.2f}; "
                            f"previous={candidate.previous_action}"
                        )
                    if candidate.confidence_drift_penalty:
                        validation_note += (
                            f"; preference_drift_penalty={candidate.preference_drift_penalty:.2f}; "
                            f"preference_drift={candidate.preference_drift:+.2f}; "
                            f"epistemic={candidate.epistemic_confidence:.2f}"
                        )
                    if candidate.evidence_basis == "UNSTATED_FACTS":
                        validation_note += (
                            f"; speculative_claim={candidate.speculative_claim}; "
                            f"evidence_tier={candidate.evidence_calibration_tier}; "
                            f"retention={candidate.evidence_direction_retention:.2f}; "
                            "vote_damped"
                        )
                    if candidate.coercion_surcharge:
                        validation_note += (
                            f"; coercion_surcharge={candidate.coercion_surcharge:.2f}; "
                            f"tag={candidate.coercion_tag}"
                        )
                    if candidate.landscape_search_attempted and not candidate.landscape_semantic_valid:
                        validation_note += (
                            "; landscape_penalty="
                            + " | ".join(candidate.landscape_validation_errors[:2])
                        )
                    if candidate.independence_bonus:
                        validation_note += "; grounded_nonconsensus_bonus=1.00"
                    if candidate.contingency_choice:
                        validation_note += (
                            f"; contingency_choice={candidate.contingency_choice}; "
                            f"conditional_why={candidate.contingency_justification}"
                        )
                    progress(
                        f"  [{index}/{len(self.specialists)}] {specialist.name} returned "
                        f"{candidate.constraint}; recommends={candidate.recommended_action or 'unknown'}; "
                        f"preference={candidate.preference_strength:.2f}; "
                        f"epistemic={candidate.epistemic_confidence:.2f}; "
                        f"alignment={candidate.testimony_alignment}; "
                        f"why={candidate.rationale}{validation_note} "
                        f"in {time.monotonic() - call_started:.1f}s"
                    )
                if terminal_model_failure:
                    break
            if result.halted_by in {"model_call_budget", "model_backend_unavailable"}:
                if specialist_snapshot:
                    self._restore_specialist_state(specialist_snapshot)
                break
            if specialist_snapshot:
                self._restore_specialist_state(specialist_snapshot)
            minority_name = previous_dissent.specialist if previous_dissent else ""
            recommendation_counts: dict[str, int] = {}
            for candidate in candidates:
                if not candidate.schema_valid or not candidate.action_scores:
                    continue
                recommendation = candidate.recommended_action or max(
                    candidate.action_scores, key=candidate.action_scores.get
                )
                recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
            largest_coalition = max(recommendation_counts.values(), default=0)
            for candidate in candidates:
                recommendation = (
                    candidate.recommended_action
                    or (max(candidate.action_scores, key=candidate.action_scores.get) if candidate.action_scores else "")
                )
                stable_or_justified = (
                    not candidate.position_changed
                    or (
                        candidate.change_justification.casefold() != "none"
                        and len(candidate.change_justification.split()) >= 3
                    )
                )
                procedural_randomizer = re.search(
                    r"\b(?:random|coin|lottery)\b",
                    candidate.landscape_tiebreaker,
                    flags=re.IGNORECASE,
                )
                independent_quality = bool(
                    recommendation
                    and candidate.evidence_basis != "UNSTATED_FACTS"
                    and candidate.landscape_semantic_valid
                    and stable_or_justified
                    and not procedural_randomizer
                )
                grounded_minority = bool(
                    independent_quality
                    and recommendation_counts.get(recommendation, 0) < largest_coalition
                )
                candidate.independence_bonus = 1.0 if grounded_minority else 0.0
                bonus = max(
                    candidate.independence_bonus,
                    1.0 if candidate.specialist == minority_name and independent_quality else 0.0,
                )
                candidate.salience = self._salience(candidate, broadcast, constraint_counts, bonus)

            valid_candidates = [candidate for candidate in candidates if candidate.schema_valid]
            winner_pool = valid_candidates or candidates
            winner = max(
                winner_pool,
                key=lambda c: (c.salience, c.epistemic_confidence, c.specialist),
            )
            if not is_counterfactual:
                constraint_counts[winner.constraint] = constraint_counts.get(winner.constraint, 0) + 1
            policy = self._policy(
                candidates,
                cycle_actions,
                visibility_multipliers,
                visibility_review=(received_broadcast.constraint == "VISIBILITY_AUDIT"),
            )
            selected_action = max(policy, key=policy.get)
            if not is_counterfactual:
                stable_cycles = stable_cycles + 1 if selected_action == previous_action else 1
                previous_action = selected_action
            entropy = self._normalized_entropy(policy)
            dissent = self._dissent(candidates, selected_action)
            elapsed = time.monotonic() - started
            visibility_review_pending = bool(
                visibility is not None and visibility.valid and visibility.activated
                and not visibility_broadcast_attempted
            )
            ev_assessment = assess_ev_dominance(
                valid_candidates, clean_actions,
                ratio_threshold=self.config.ev_dominance_ratio,
                majority_fraction=self.config.ev_majority_fraction,
            ) if (
                self.config.enable_ev_dominance_breaker
                and not is_counterfactual
                and not visibility_review_pending
            ) else None
            if ev_assessment is not None:
                result.ev_dominance_assessments.append(ev_assessment.to_dict())

            if (
                broadcast.constraint == "PROBLEM_REFORMULATION"
                and result.problem_reformulations
                and result.problem_reformulations[-1].probe_result == "UNTESTED"
            ):
                proposal = result.problem_reformulations[-1]
                calibrated_recommendations = {
                    candidate.recommended_action or max(candidate.action_scores, key=candidate.action_scores.get)
                    for candidate in valid_candidates
                    if candidate.action_scores
                }
                explicit_split = any(
                    candidate.boundary_position == "SPLIT"
                    for candidate in valid_candidates
                )
                structured_boundary_responses = all(
                    candidate.boundary_position != "NOT_TESTED"
                    and candidate.decisive_axis
                    and candidate.boundary_switch_condition
                    for candidate in valid_candidates
                )
                if not structured_boundary_responses:
                    proposal.probe_result = "INVALID_BOUNDARY_RESPONSES"
                    proposal.switch_claim_valid = False
                    proposal.accepted = False
                    proposal.rejection_reason = (
                        "specialists did not identify decisive axes and switch conditions"
                    )
                elif explicit_split or len(calibrated_recommendations) >= 2:
                    proposal.probe_result = "SPLIT_OBSERVED"
                    proposal.switch_claim_valid = True
                elif selected_action != reformulation_baseline_action:
                    proposal.probe_result = "SWITCH_OBSERVED"
                    proposal.switch_claim_valid = True
                else:
                    proposal.probe_result = "NO_SWITCH"
                    proposal.switch_claim_valid = False
                    proposal.accepted = False
                    proposal.rejection_reason = (
                        "calibration did not move or split the specialist coalition"
                    )
                if progress:
                    progress(
                        f"  reformulation probe result: {proposal.probe_result}; "
                        f"recommendations={', '.join(sorted(calibrated_recommendations))}"
                    )

            next_broadcast = WorkspaceBroadcast(
                constraint=winner.constraint,
                intent=f"evaluate_{selected_action}",
                urgency=broadcast.urgency,
                danger_probability=broadcast.danger_probability,
                unresolved=(dissent.unresolved if dissent and dissent.unresolved != "NONE" else winner.unresolved),
            )
            result.cycles.append(
                CycleRecord(
                    cycle=cycle_number,
                    broadcast=next_broadcast,
                    candidates=candidates,
                    winner=winner,
                    dissent=dissent,
                    policy=policy,
                    entropy=entropy,
                    stable_cycles=stable_cycles,
                    elapsed_seconds=elapsed,
                    received_broadcast=received_broadcast,
                    is_hypothetical=is_counterfactual,
                )
            )
            if checkpoint is not None:
                checkpoint(result)
            if ev_assessment is not None and ev_assessment.activated:
                result.halted_by = "ev_dominance"
                if progress:
                    progress(f"  EV dominance circuit breaker: {ev_assessment.reason}")
                break
            if is_planning_branch:
                result.planning_branches.append(PlanningBranchEvaluation(
                    cycle=cycle_number,
                    origin_action=received_broadcast.branch_origin_action,
                    condition=received_broadcast.branch_condition,
                    fallback=received_broadcast.branch_fallback,
                    selected_action=selected_action,
                    confidence=policy[selected_action],
                    policy=dict(policy),
                ))
                broadcast = planning_resume_broadcast or WorkspaceBroadcast(
                    urgency=received_broadcast.urgency,
                    danger_probability=received_broadcast.danger_probability,
                )
                planning_resume_broadcast = None
                result.cycles[-1].broadcast = broadcast
                if progress:
                    progress(
                        f"  planning branch result: if {received_broadcast.branch_condition}, "
                        f"prefer {selected_action} ({policy[selected_action]:.2f}); resuming base case"
                    )
                cycle_number += 1
                continue
            if is_reformulation_probe:
                broadcast = reformulation_resume_broadcast or WorkspaceBroadcast(
                    urgency=received_broadcast.urgency,
                    danger_probability=received_broadcast.danger_probability,
                )
                reformulation_resume_broadcast = None
                result.cycles[-1].broadcast = broadcast
                if progress:
                    progress("  reformulation probe recorded; restored base specialist state")
                cycle_number += 1
                continue
            if is_reversal_probe:
                broadcast = reversal_resume_broadcast or WorkspaceBroadcast(
                    urgency=received_broadcast.urgency,
                    danger_probability=received_broadcast.danger_probability,
                )
                reversal_resume_broadcast = None
                result.cycles[-1].broadcast = broadcast
                if progress:
                    progress("  reversal review recorded; restored base specialist state")
                cycle_number += 1
                continue
            if is_contingency_probe:
                broadcast = contingency_resume_broadcast or WorkspaceBroadcast(
                    urgency=received_broadcast.urgency,
                    danger_probability=received_broadcast.danger_probability,
                )
                contingency_resume_broadcast = None
                result.cycles[-1].broadcast = broadcast
                if progress:
                    answered = sum(
                        candidate.schema_valid and bool(candidate.contingency_choice)
                        for candidate in candidates
                    )
                    progress(
                        f"  contingency review recorded: {answered}/{len(candidates)} "
                        "delegates selected a typed fallback; restored base specialist state"
                    )
                cycle_number += 1
                continue

            previous_dissent = dissent
            planning_reason = ""
            if broadcast.constraint == "SYNTHESIS_REVIEW":
                planning_reason = "synthesis review"
            elif broadcast.constraint == "CONTINGENCY_REVIEW":
                planning_reason = "contingency review"
            elif any(
                candidate.schema_valid and (
                    candidate.constraint == "FEASIBILITY"
                    or candidate.unresolved == "CHECK_FEASIBILITY"
                )
                for candidate in candidates
            ):
                planning_reason = "delegate feasibility concern"
            elif dissent is not None and entropy >= self.config.planning_entropy_threshold:
                planning_reason = "unresolved policy competition"

            planning_key = (selected_action.casefold(), planning_reason)
            planning_broadcast = False
            audit_broadcast = False
            reformulation_broadcast = False
            reversal_audit_broadcast = False
            visibility_broadcast = False
            if (
                visibility is not None
                and
                visibility.valid
                and visibility.activated
                and not visibility_broadcast_attempted
                and broadcast.constraint != "VISIBILITY_AUDIT"
            ):
                visibility_targets = [
                    action for action, value in visibility.action_multipliers.items()
                    if value < 0.999
                ]
                visibility_target = visibility_targets[0] if visibility_targets else ""
                visibility_record = validate_transformation(
                    "VISIBILITY_TO_BROADCAST",
                    SemanticProposition(
                        actor="visibility auditor",
                        action=visibility_target,
                        relation="INCREASES",
                        consequence="estimated harm",
                        condition=visibility.mechanism,
                        affected_party=visibility.affected_group,
                        epistemic_status="SCENARIO_GROUNDED",
                        context="VISIBILITY_AUDIT",
                        provenance=("visibility_audit", visibility.evidence_quote),
                        source_text=visibility.proposition,
                    ),
                    visibility.proposition,
                    required_fragments=(visibility_target, "downward-biased"),
                )
                result.semantic_invariants.append(visibility_record)
                if not visibility_record.valid:
                    if progress:
                        progress(
                            "  visibility broadcast withheld by semantic invariant layer: "
                            + "; ".join(visibility_record.errors)
                        )
                    visibility_broadcast_attempted = True
                else:
                    visibility_broadcast_attempted = True
                    next_broadcast = WorkspaceBroadcast(
                        constraint="VISIBILITY_AUDIT",
                        intent="evaluate_visibility_bias",
                        urgency=broadcast.urgency,
                        danger_probability=broadcast.danger_probability,
                        unresolved="VERIFY_ASSUMPTIONS",
                        contingency_question=(
                            f"Proposition P: {visibility.proposition} Does P alter your "
                            "reasoning or estimated harm? Why or why not?"
                        ),
                    )
                    visibility_broadcast = True
                    cycle_limit += 1
                    result.access_decisions.append(WorkspaceAccessDecision(
                        cycle=cycle_number,
                        content_type="VISIBILITY_AUDIT",
                        admitted=True,
                        signals=["endogenous_low_observability", "action_confidence_bias"],
                        question=visibility.proposition,
                        rationale=(
                            "A grounded visibility proposition receives recurrent review "
                            "instead of acting only as an external confidence multiplier."
                        ),
                    ))
                    if progress:
                        progress("  workspace access gate admitted VISIBILITY_AUDIT: " + visibility.proposition)
            if not consensus_audit_attempted and not visibility_broadcast:
                access_decision = self._consensus_access_decision(
                    cycle_number,
                    scenario,
                    tuple(clean_actions),
                    tuple(candidates),
                    selected_action,
                    entropy,
                    dissent,
                    scenario_facts,
                    source_testimonies,
                )
                if access_decision is not None:
                    signal_signature = tuple(sorted(access_decision.signals))
                    is_new_signal_state = signal_signature not in access_signal_signatures
                    if is_new_signal_state:
                        access_signal_signatures.add(signal_signature)
                        result.access_decisions.append(access_decision)
                    if (
                        access_decision.admitted
                        and is_new_signal_state
                        and cycle_number < cycle_limit
                    ):
                        consensus_audit_attempted = True
                        audit_question = access_decision.question
                        if (
                            autonomy is not None
                            and autonomy.valid
                            and autonomy.action_tags.get(selected_action, "NONE") != "NONE"
                        ):
                            alternative = autonomy.voluntary_alternative
                            audit_question = (
                                "Autonomy reversal probe: Does the coercive preference assume "
                                "voluntary or less-coercive alternatives are ineffective? "
                                f"Test: {alternative}. {audit_question}"
                            )
                        next_broadcast = WorkspaceBroadcast(
                            constraint="CONSENSUS_AUDIT",
                            intent=f"audit_{selected_action}",
                            urgency=broadcast.urgency,
                            danger_probability=broadcast.danger_probability,
                            unresolved="VERIFY_ASSUMPTIONS",
                            contingency_question=audit_question,
                        )
                        audit_broadcast = True
                        if progress:
                            progress(
                                "  workspace access gate admitted CONSENSUS_AUDIT: "
                                + ", ".join(access_decision.signals)
                            )

            audited_uncertain = [
                candidate
                for candidate in valid_candidates
                if candidate.assumption_status in {"CONDITIONAL", "UNDERDETERMINED"}
            ]
            can_reformulate = (
                reformulate_problem is not None
                and not reformulation_attempted
                and broadcast.constraint == "CONSENSUS_AUDIT"
                and len(audited_uncertain) / max(1, len(valid_candidates)) >= 0.60
                and cycle_number < cycle_limit
                and elapsed < self.config.time_budget_seconds
            )
            if can_reformulate:
                reformulation_attempted = True
                if progress:
                    progress("  audited underdetermination detected; calibrating a switch-point case...")
                try:
                    reformulation = reformulate_problem(
                        scenario, tuple(clean_actions), tuple(candidates)
                    )
                except Exception as exc:
                    reformulation = ProblemReformulation(
                        [], [], "", "", "", [], accepted=False,
                        rejection_reason=f"reformulation unavailable: {exc}",
                    )
                result.problem_reformulations.append(reformulation)
                if reformulation.accepted:
                    reformulation_baseline_action = selected_action
                    reformulation_resume_broadcast = next_broadcast
                    next_broadcast = WorkspaceBroadcast(
                        constraint="PROBLEM_REFORMULATION",
                        intent="evaluate_hypothetical_switch_point",
                        urgency=broadcast.urgency,
                        danger_probability=broadcast.danger_probability,
                        unresolved="RESOLVE_VALUE_TENSION",
                        contingency_question=reformulation.question,
                        reformulation_context=reformulation.compact(),
                    )
                    reformulation_broadcast = True
                    cycle_limit += 1
                    if progress:
                        progress(
                            "  workspace access gate admitted PROBLEM_REFORMULATION: "
                            + reformulation.compact()
                        )
                elif progress:
                    progress(
                        f"  problem reformulation rejected: {reformulation.rejection_reason}"
                    )

            # A grounded opposing specialist acts as the critic. Its explicit
            # rule and typed switch condition are broadcast once, before a
            # stable plurality can treat dissent as resolved by repetition.
            if (
                self.config.enable_reversal_audit
                and not reversal_audit_attempted
                and not audit_broadcast
                and not reformulation_broadcast
                and not visibility_broadcast
                and cycle_number < cycle_limit
                and elapsed < self.config.time_budget_seconds
            ):
                competing_action = (
                    dissent.recommended_action
                    if dissent is not None and dissent.schema_valid else ""
                )
                committed_boundary = select_committed_reversal_boundary(
                    graph_store.graph,
                    leading_action=selected_action,
                    competing_action=competing_action,
                ) if competing_action else None
                audit_request = build_reversal_audit_request(
                    dissent, selected_action, committed_boundary
                )
                if audit_request is not None:
                    reversal_record = validate_transformation(
                        "DISSENT_TO_REVERSAL_AUDIT",
                        SemanticProposition(
                            actor="workspace_graph",
                            action=audit_request.leading_action,
                            relation="REVERSES_IF",
                            consequence=audit_request.competing_action,
                            condition=audit_request.proposed_condition,
                            alternative=audit_request.competing_action,
                            epistemic_status="CONDITIONAL",
                            context="REVERSAL_AUDIT",
                            provenance=tuple(dict.fromkeys((
                                *audit_request.boundary_sources,
                                audit_request.critic,
                            ))),
                            source_text=audit_request.proposed_condition,
                        ),
                        audit_request.challenge,
                        required_fragments=(
                            audit_request.leading_action,
                            audit_request.competing_action,
                            audit_request.proposed_condition,
                        ),
                    )
                    result.semantic_invariants.append(reversal_record)
                    if not reversal_record.valid:
                        if progress:
                            progress(
                                "  reversal audit withheld by semantic invariant layer: "
                                + "; ".join(reversal_record.errors)
                            )
                        audit_request = None
                if audit_request is not None:
                    reversal_audit_attempted = True
                    reversal_resume_broadcast = next_broadcast
                    burden_facts = compile_action_burdens(scenario, clean_actions)
                    burdened_ids = {fact.affected_action_node_id for fact in burden_facts}
                    symmetric_burden = len(burdened_ids) >= 2
                    burden_probe = (
                        " Symmetry probe: Does the leading action avoid the burden or "
                        "transfer a comparable burden to another group? Apply the same "
                        "rule to both."
                        if symmetric_burden else ""
                    )
                    next_broadcast = WorkspaceBroadcast(
                        constraint="REVERSAL_AUDIT",
                        intent=f"test_reversal_of_{selected_action}",
                        urgency=broadcast.urgency,
                        danger_probability=broadcast.danger_probability,
                        unresolved="TEST_REVERSAL",
                        contingency_question=(
                            "Does committed boundary P reverse the ranking? "
                            "Accept it, revise it to the smallest valid condition, or reject it."
                            + burden_probe
                        ),
                        reversal_challenge=audit_request.challenge,
                    )
                    reversal_audit_broadcast = True
                    cycle_limit += 1
                    result.access_decisions.append(WorkspaceAccessDecision(
                        cycle=cycle_number,
                        content_type="REVERSAL_AUDIT",
                        admitted=True,
                        signals=[
                            "substantive_dissent", "committed_reversal_boundary",
                            *(["symmetric_burden_substitution"] if symmetric_burden else []),
                        ],
                        question=audit_request.challenge + burden_probe,
                        rationale=(
                            "A committed switch boundary toward a grounded dissenting "
                            "action must be tested before convergence."
                        ),
                    ))
                    if progress:
                        progress(
                            "  adversarial reversal review admitted: "
                            + audit_request.challenge
                        )
            if (
                self.config.enable_planning
                and analyze_plan is not None
                and not audit_broadcast
                and not reformulation_broadcast
                and not reversal_audit_broadcast
                and not visibility_broadcast
                and planning_reason
                and planning_key not in planned_contexts
                and planning_branches_used < self.config.max_planning_branches
                and self._explicit_implementation_obstacle(
                    scenario, broadcast, tuple(clean_actions), selected_action
                )
                and elapsed < self.config.time_budget_seconds
            ):
                planned_contexts.add(planning_key)
                if progress:
                    progress(f"  planning system activated: {planning_reason}...")
                try:
                    assessment = analyze_plan(
                        scenario,
                        tuple(clean_actions),
                        selected_action,
                        broadcast,
                        tuple(candidates),
                        planning_reason,
                    )
                except Exception as exc:
                    assessment = PlanningAssessment(
                        selected_action, planning_reason, 0.0, "", "", "",
                        valid=False, error=f"planning unavailable: {exc}",
                    )
                assessment = self._validate_planning_assessment(
                    assessment, scenario, broadcast, tuple(clean_actions)
                )
                result.planning_assessments.append(assessment)
                if assessment.valid and assessment.broadcast_worthy:
                    planning_text = (
                        f"If {assessment.failure_condition}, reconsider "
                        f"{assessment.target_action}; fallback: {assessment.fallback}."
                    )
                    planning_record = validate_transformation(
                        "PLANNING_TO_BRANCH",
                        SemanticProposition(
                            actor="planning system",
                            action=assessment.target_action,
                            relation="DISABLES",
                            consequence=assessment.fallback,
                            condition=assessment.failure_condition,
                            alternative=assessment.fallback,
                            epistemic_status="HYPOTHETICAL",
                            context="PLANNING_COUNTERFACTUAL",
                            provenance=(assessment.activation_reason, assessment.grounded_evidence),
                            source_text=assessment.failure_condition,
                        ),
                        planning_text,
                        required_fragments=(
                            assessment.target_action,
                            assessment.failure_condition,
                            assessment.fallback,
                        ),
                    )
                    result.semantic_invariants.append(planning_record)
                    if not planning_record.valid:
                        assessment.valid = False
                        assessment.broadcast_worthy = False
                        assessment.error = (
                            "semantic invariant failure: "
                            + "; ".join(planning_record.errors)
                        )
                if assessment.valid and assessment.broadcast_worthy:
                    planning_resume_broadcast = next_broadcast
                    next_broadcast = WorkspaceBroadcast(
                        constraint="PLANNING_REVIEW",
                        intent=f"evaluate_{selected_action}",
                        urgency=broadcast.urgency,
                        danger_probability=broadcast.danger_probability,
                        unresolved="CHECK_FEASIBILITY",
                        contingency_question=(
                            f"If {assessment.failure_condition}, should the policy change?"
                        ),
                        branch_kind="PLANNING_CONTINGENCY",
                        branch_origin_action=selected_action,
                        branch_condition=assessment.failure_condition,
                        branch_fallback=assessment.fallback,
                    )
                    planning_broadcast = True
                    planning_branches_used += 1
                    cycle_limit += 1
                    if progress:
                        progress(
                            f"  planning broadcast: failure={assessment.failure_condition}; "
                            f"fallback={assessment.fallback}; feasibility={assessment.feasibility:.2f}"
                        )
                elif progress and assessment.valid:
                    progress(
                        f"  planning assessment retained privately: "
                        f"feasibility={assessment.feasibility:.2f}"
                    )
                elif progress:
                    progress(f"  planning assessment unavailable: {assessment.error}")

            result.cycles[-1].broadcast = next_broadcast
            broadcast = next_broadcast
            if progress:
                progress(
                    f"  cycle policy: {selected_action} "
                    f"({policy[selected_action]:.2f}); entropy={entropy:.2f}; "
                    f"winner={winner.specialist}:{winner.constraint}"
                )

            if len(valid_candidates) < self.config.min_valid_specialists:
                result.halted_by = "insufficient_valid_candidates"
                if progress:
                    progress(
                        f"  halting: only {len(valid_candidates)} valid delegate response(s); "
                        f"need {self.config.min_valid_specialists}"
                    )
                break

            if audit_broadcast:
                cycle_number += 1
                continue

            if visibility_broadcast:
                cycle_number += 1
                continue

            if reformulation_broadcast:
                cycle_number += 1
                continue

            if reversal_audit_broadcast:
                cycle_number += 1
                continue

            # A material planning failure gets one full recurrent response before
            # synthesis or convergence can absorb it.
            if planning_broadcast and cycle_number < cycle_limit:
                cycle_number += 1
                continue

            can_synthesize = (
                self.config.enable_synthesis
                and synthesize is not None
                and not synthesis_attempted
                and cycle_number >= self.config.synthesis_after_cycle
                and cycle_number < cycle_limit
                and dissent is not None
                and entropy >= self.config.synthesis_min_entropy
                and len(clean_actions) < 5
                and elapsed < self.config.time_budget_seconds
            )
            if can_synthesize:
                synthesis_attempted = True
                if progress:
                    progress("  unresolved competition detected; attempting grounded synthesis...")
                try:
                    proposal = synthesize(scenario, tuple(clean_actions), tuple(candidates), next_broadcast)
                except Exception as exc:
                    proposal = SynthesisProposal(
                        "", [], [], 0.0, "", accepted=False,
                        rejection_reason=f"synthesis unavailable: {exc}",
                    )
                if proposal is not None:
                    result.synthesis_proposals.append(proposal)
                    if proposal.accepted:
                        synthesis_text = (
                            f"{proposal.action}; addresses "
                            f"{', '.join(proposal.addressed_constraints)}; grounded in "
                            f"{', '.join(proposal.grounded_in)}."
                        )
                        synthesis_record = validate_transformation(
                            "WORKSPACE_TO_SYNTHESIS",
                            SemanticProposition(
                                actor="workspace synthesis",
                                action=proposal.action,
                                relation="RESOLVES",
                                consequence=", ".join(proposal.addressed_constraints),
                                condition=", ".join(proposal.introduced_requirements) or "NONE",
                                epistemic_status="CONDITIONAL",
                                context="SYNTHESIS_REVIEW",
                                provenance=tuple(proposal.grounded_in),
                                source_text=proposal.rationale,
                            ),
                            synthesis_text,
                            required_fragments=(
                                proposal.action,
                                *proposal.addressed_constraints,
                                *proposal.grounded_in,
                            ),
                        )
                        result.semantic_invariants.append(synthesis_record)
                        if not synthesis_record.valid:
                            proposal.accepted = False
                            proposal.rejection_reason = (
                                "semantic invariant failure: "
                                + "; ".join(synthesis_record.errors)
                            )
                    if proposal.accepted:
                        clean_actions.append(proposal.action)
                        extension_record = graph_store.apply_action_extension(
                            compile_scenario_graph(scenario, clean_actions),
                            cycle=cycle_number,
                            source="workspace_synthesis",
                        )
                        result.graph_transactions = graph_store.transaction_dicts()
                        if extension_record.status == "REJECTED":
                            clean_actions.pop()
                            proposal.accepted = False
                            proposal.rejection_reason = (
                                "canonical action-graph extension failed: "
                                + "; ".join(extension_record.errors)
                            )
                            if progress:
                                progress(
                                    "  synthesis rejected: " + proposal.rejection_reason
                                )
                            cycle_number += 1
                            continue
                        result.actions = list(clean_actions)
                        stable_cycles = 0
                        previous_action = ""
                        broadcast = WorkspaceBroadcast(
                            constraint="SYNTHESIS_REVIEW",
                            intent=f"evaluate_{proposal.action}",
                            urgency=broadcast.urgency,
                            danger_probability=broadcast.danger_probability,
                            unresolved="CHECK_FEASIBILITY",
                        )
                        if progress:
                            progress(
                                f"  synthesis admitted for specialist review: {proposal.action} "
                                f"(feasibility={proposal.feasibility:.2f}; "
                                f"grounded in {', '.join(proposal.grounded_in)})"
                            )
                        cycle_number += 1
                        continue
                    if progress:
                        progress(f"  synthesis rejected: {proposal.rejection_reason}")

            dissent_addressed = dissent is None or (
                cycle_number > 1
                and result.cycles[-2].broadcast.unresolved == dissent.unresolved
            )
            if elapsed >= self.config.time_budget_seconds:
                result.halted_by = "time_budget"
                break
            epistemically_resolved = all(
                candidate.assumption_status in {
                    "NOT_AUDITED", "SUPPORTED", "NORMATIVELY_CONTESTED",
                }
                for candidate in valid_candidates
            )
            if (
                entropy < self.config.entropy_threshold
                and stable_cycles >= self.config.stable_cycles_required
                and dissent_addressed
                and epistemically_resolved
            ):
                result.halted_by = "convergence"
                break
            if cycle_number >= cycle_limit:
                valid_ratio = len(valid_candidates) / max(1, len(candidates))
                synthesis_viability = self._assess_synthesis_viability(result)
                if (
                    synthesis_viability is not None
                    and not any(
                        item.synthesis_action == synthesis_viability.synthesis_action
                        and item.review_cycle == synthesis_viability.review_cycle
                        for item in result.synthesis_viability_assessments
                    )
                ):
                    result.synthesis_viability_assessments.append(synthesis_viability)
                good_recurrent_deliberation = (
                    request_extension is not None
                    and analyze_contingency is not None
                    and verify_contingency_feasibility is not None
                    and extensions_used < self.config.max_cycle_extensions
                    and synthesis_viability is not None
                    and synthesis_viability.viable
                    and dissent is not None
                    and entropy >= self.config.synthesis_min_entropy
                    and valid_ratio >= 0.8
                    and broadcast.urgency < 0.8
                )
                added_cycles = 0
                if (
                    synthesis_viability is not None
                    and not synthesis_viability.viable
                    and progress
                ):
                    progress(
                        "  contingency not activated: " + synthesis_viability.reason
                    )
                if good_recurrent_deliberation:
                    analysis = analyze_contingency(result)
                    result.failure_conditions.append(analysis)
                    if analysis.valid:
                        graph_errors = validate_contingency_graph_dict(
                            analysis.semantic_graph,
                            analysis.synthesis_action,
                            analysis.fallback_actions,
                            require_fallback_availability=False,
                        )
                        if graph_errors:
                            analysis.valid = False
                            analysis.error = (
                                "contingency graph validation failed: "
                                + "; ".join(graph_errors)
                            )
                    if analysis.valid:
                        try:
                            feasibility = verify_contingency_feasibility(analysis)
                        except ModelCallBudgetExceeded as exc:
                            result.halted_by = "model_call_budget"
                            feasibility = ContingencyFeasibilityAssessment(
                                analysis.synthesis_action,
                                analysis.predicate_label,
                                {}, {}, {}, False, valid=False, approved=False,
                                error=f"independent feasibility verifier unavailable: {exc}",
                            )
                        except Exception as exc:
                            feasibility = ContingencyFeasibilityAssessment(
                                analysis.synthesis_action,
                                analysis.predicate_label,
                                {}, {}, {}, False, valid=False, approved=False,
                                error=f"independent feasibility verifier unavailable: {exc}",
                            )
                        result.contingency_feasibility_assessments.append(feasibility)
                        if feasibility.valid and feasibility.approved:
                            certified_graph, certification_errors = (
                                certify_fallback_availability(
                                    analysis.semantic_graph,
                                    analysis.synthesis_action,
                                    analysis.fallback_actions,
                                    feasibility.fallback_statuses,
                                    feasibility.fallback_reasons,
                                )
                            )
                            if certification_errors:
                                analysis.valid = False
                                analysis.error = (
                                    "fallback certification failed: "
                                    + "; ".join(certification_errors)
                                )
                            else:
                                analysis.semantic_graph = certified_graph
                                analysis.fallback_availability = dict(
                                    feasibility.fallback_statuses
                                )
                                analysis.fallback_availability_reasons = dict(
                                    feasibility.fallback_reasons
                                )
                        else:
                            analysis.valid = False
                            analysis.error = (
                                "independent fallback feasibility not established: "
                                + (feasibility.error or "verification did not approve both fallbacks")
                            )
                    if analysis.valid:
                        contingency_text = (
                            f"If {analysis.failure_condition}, "
                            f"{analysis.synthesis_action} no longer satisfies "
                            f"{analysis.necessary_condition}. {analysis.contingency_question}"
                        )
                        contingency_record = validate_transformation(
                            "SYNTHESIS_TO_FAILURE_CONDITION",
                            SemanticProposition(
                                actor="contingency analyzer",
                                action=analysis.synthesis_action,
                                relation="DISABLES",
                                consequence=analysis.necessary_condition,
                                condition=analysis.failure_condition,
                                epistemic_status="HYPOTHETICAL",
                                context="PLANNING_COUNTERFACTUAL",
                                provenance=(analysis.synthesis_action,),
                                source_text=analysis.failure_condition,
                            ),
                            contingency_text,
                            required_fragments=(
                                analysis.synthesis_action,
                                analysis.failure_condition,
                                analysis.necessary_condition,
                            ),
                        )
                        result.semantic_invariants.append(contingency_record)
                        if not contingency_record.valid:
                            analysis.valid = False
                            analysis.error = (
                                "semantic invariant failure: "
                                + "; ".join(contingency_record.errors)
                            )
                    if analysis.valid:
                        contingency_resume_broadcast = broadcast
                        broadcast = WorkspaceBroadcast(
                            constraint="CONTINGENCY_REVIEW",
                            intent="evaluate_synthesis_failure_fallback",
                            urgency=broadcast.urgency,
                            danger_probability=broadcast.danger_probability,
                            unresolved="CHECK_FEASIBILITY",
                            contingency_question=analysis.contingency_question,
                            contingency_synthesis_action=analysis.synthesis_action,
                            contingency_failure_condition=analysis.failure_condition,
                            contingency_predicate=analysis.predicate_label,
                            contingency_failure_truth=analysis.failure_truth,
                            contingency_fallback_actions=tuple(
                                analysis.fallback_actions or result.actions[:2]
                            ),
                            branch_kind="SYNTHESIS_CONTINGENCY",
                        )
                        requested_cycles = max(0, int(request_extension(result)))
                        # Do not begin more full five-delegate rounds than the
                        # observed run rate can finish. A reserve covers final
                        # synthesis, serialization, and normal call variance.
                        elapsed_now = time.monotonic() - started
                        observed_cycle_cost = elapsed_now / max(1, len(result.cycles))
                        reserve = max(15.0, observed_cycle_cost * 0.50)
                        affordable_cycles = int(
                            max(0.0, self.config.time_budget_seconds - elapsed_now - reserve)
                            // max(1.0, observed_cycle_cost)
                        )
                        added_cycles = min(requested_cycles, affordable_cycles)
                        if progress and added_cycles < requested_cycles:
                            progress(
                                f"  extension capped by remaining time budget: "
                                f"requested={requested_cycles}; affordable={added_cycles}"
                            )
                    elif progress:
                        progress(f"  contingency analysis unavailable: {analysis.error}")
                if added_cycles:
                    extensions_used += 1
                    cycle_limit += added_cycles
                    if progress:
                        progress(
                            f"  cycle budget extended by {added_cycles}; "
                            f"new limit={cycle_limit}"
                        )
                else:
                    result.halted_by = result.halted_by or "cycle_budget"
                    break
            cycle_number += 1

        if not result.halted_by:
            result.halted_by = "cycle_budget"
        actual_cycles = [cycle for cycle in result.cycles if not cycle.is_hypothetical]
        if not actual_cycles:
            result.selected_action = "INCONCLUSIVE"
            result.current_plurality = ""
            result.judgment_status = "INCONCLUSIVE"
            result.confidence = 0.0
            result.epistemic_confidence = 0.0
            result.compressed_rule = (
                f"Unavailable: deliberation halted by {result.halted_by}."
            )
            result.termination_assessment = assess_termination(
                result.halted_by, result.cycles, self.config.stable_cycles_required
            )
            result.access_construct_records = [
                describe_access(decision) for decision in result.access_decisions
            ]
            result.further_deliberation_estimate = estimate_further_deliberation(
                result.cycles, result.termination_assessment
            )
            result.semantic_graphs = (
                [graph_store.graph_dict()] if graph_store.graph.nodes else []
            )
            result.graph_transactions = graph_store.transaction_dicts()
            result.rawlsian_position_ledger = committed_rawls_positions(
                graph_store.graph
            )
            result.utilitarian_consequence_ledger = (
                committed_utilitarian_consequences(graph_store.graph)
            )
            result.deontological_duty_ledger = (
                committed_deontological_assessments(graph_store.graph)
            )
            result.virtue_character_ledger = committed_virtue_assessments(
                graph_store.graph
            )
            result.authoritative_semantic_state = project_authoritative_semantic_state(
                graph_store.graph
            ).to_dict()
            result.trace_health = audit_trace_health(result)
            return result
        final = actual_cycles[-1]
        result.current_plurality = max(final.policy, key=final.policy.get)
        if result.halted_by == "insufficient_valid_candidates":
            result.selected_action = "INCONCLUSIVE"
            result.judgment_status = "INCONCLUSIVE"
            result.confidence = 0.0
            result.epistemic_confidence = 0.0
        else:
            valid_final = [candidate for candidate in final.candidates if candidate.schema_valid]
            underdetermined_count = sum(
                candidate.assumption_status == "UNDERDETERMINED"
                for candidate in valid_final
            )
            uncertain_count = sum(
                candidate.assumption_status in {"CONDITIONAL", "UNDERDETERMINED"}
                for candidate in valid_final
            )
            if underdetermined_count / max(1, len(valid_final)) >= 0.50:
                result.judgment_status = "UNDERDETERMINED"
                result.selected_action = "UNDERDETERMINED"
            elif uncertain_count / max(1, len(valid_final)) >= 0.50:
                result.judgment_status = "CONDITIONAL"
                result.selected_action = "CONDITIONAL"
            elif (
                result.halted_by == "cycle_budget"
                and final.dissent is not None
                and final.entropy >= self.config.entropy_threshold
            ):
                # Failure to converge is not failure to judge. Preserve the
                # leading action as a defeasible, contested recommendation;
                # epistemic underdetermination and insufficient valid evidence
                # are handled by the branches above.
                result.judgment_status = "CONTESTED_RECOMMENDATION"
                result.selected_action = result.current_plurality
            else:
                result.judgment_status = "ACTION_RECOMMENDATION"
                result.selected_action = result.current_plurality
            result.confidence = clamp(final.policy[result.current_plurality])
            supporting_final = [
                candidate for candidate in valid_final
                if (
                    candidate.recommended_action
                    or max(candidate.action_scores, key=candidate.action_scores.get)
                ) == result.current_plurality
            ]
            epistemic_weight = sum(
                max(0.05, candidate.preference_strength)
                for candidate in supporting_final
            )
            result.epistemic_confidence = clamp(
                sum(
                    max(0.05, candidate.preference_strength)
                    * candidate.epistemic_confidence
                    for candidate in supporting_final
                ) / max(0.05, epistemic_weight)
            )
            if final.stable_cycles < self.config.stable_cycles_required:
                result.epistemic_confidence *= 0.80
            if result.judgment_status == "UNDERDETERMINED":
                result.confidence = min(result.confidence, 0.50)
                result.epistemic_confidence = min(result.epistemic_confidence, 0.35)
            elif result.judgment_status == "CONDITIONAL":
                result.confidence = min(result.confidence, 0.65)
                result.epistemic_confidence = min(result.epistemic_confidence, 0.50)
            elif result.judgment_status == "CONTESTED_RECOMMENDATION":
                result.confidence = min(result.confidence, 0.65)
                result.epistemic_confidence = min(result.epistemic_confidence, 0.65)
        result.moral_residue = collect_moral_residue(
            actual_cycles, result.current_plurality
        )
        result.moral_residue_records = collect_typed_residue(
            actual_cycles, result.current_plurality
        )
        result.reopen_conditions = collect_reopen_conditions(actual_cycles)
        boundary = select_collective_reversal_boundary(
            final.candidates, result.current_plurality, clean_actions
        )
        if boundary is not None:
            transition = boundary.transition()
            # Compatibility fallback for delegates produced before typed graph
            # proposals were introduced. Do not overwrite a committed graph.
            if not graph_store.graph.nodes:
                result.semantic_graphs = [boundary.graph().to_dict()]
            reversal = (
                f"If {transition['condition']}, prefer {transition['to_action']}"
            )
            if reversal not in result.reopen_conditions:
                result.reopen_conditions.append(reversal)
        else:
            dissent_reversal = self._dissent_reversal_condition(
                final.dissent, result.current_plurality
            )
            if dissent_reversal and dissent_reversal not in result.reopen_conditions:
                result.reopen_conditions.append(dissent_reversal)
        publishable_judgment = (
            result.judgment_status == "CONTESTED_RECOMMENDATION"
            or (
                result.judgment_status == "ACTION_RECOMMENDATION"
                and result.halted_by in {"convergence", "cycle_budget", "ev_dominance"}
            )
        )
        if not final.winner.schema_valid or not publishable_judgment:
            result.compressed_rule = f"Unavailable: deliberation halted by {result.halted_by}."
        else:
            qualifier = (
                "contestedly prefer"
                if result.judgment_status == "CONTESTED_RECOMMENDATION"
                else (
                    "provisionally prefer"
                    if result.halted_by == "cycle_budget"
                    else "prefer"
                )
            )
            governing_candidate = (
                final.winner
                if final.winner.schema_valid
                and final.winner.recommended_action == result.current_plurality
                and final.winner.decision_rule
                else next(
                    (
                        candidate for candidate in final.candidates
                        if candidate.schema_valid
                        and candidate.recommended_action == result.current_plurality
                        and candidate.decision_rule
                    ),
                    None,
                )
            )
            governing = (
                governing_candidate.decision_rule if governing_candidate else ""
            )
            preserved_constraints = set(result.moral_residue)
            preserved_objections: list[str] = []
            for candidate in final.candidates:
                if (
                    not candidate.schema_valid
                    or candidate.recommended_action == result.current_plurality
                    or candidate.constraint not in preserved_constraints
                ):
                    continue
                framework_case = candidate.framework_action_map.get(
                    candidate.recommended_action, ""
                )
                objection = framework_case or candidate.landscape_decisive_axis
                if not objection:
                    continue
                preserved_objections.append(
                    f"{candidate.specialist}/{candidate.constraint}: "
                    + " ".join(objection.split())[:120]
                )
            compiled, rule_record = compile_preference_rule(
                result.selected_action,
                governing,
                qualifier,
                result.reopen_conditions,
                tuple(
                    candidate.specialist for candidate in final.candidates
                    if candidate.schema_valid
                    and candidate.recommended_action == result.current_plurality
                ),
                governing_constraint=(
                    governing_candidate.constraint if governing_candidate else ""
                ),
                preserved_objections=tuple(preserved_objections),
            )
            result.semantic_invariants.append(rule_record)
            result.compressed_rule = (
                compiled if rule_record.valid
                else "Unavailable: semantic invariant validation failed; source judgment preserved."
            )
        result.termination_assessment = assess_termination(
            result.halted_by, result.cycles, self.config.stable_cycles_required
        )
        result.access_construct_records = [
            describe_access(decision) for decision in result.access_decisions
        ]
        result.further_deliberation_estimate = estimate_further_deliberation(
            result.cycles, result.termination_assessment
        )
        result.semantic_graphs = (
            [graph_store.graph_dict()] if graph_store.graph.nodes else []
        )
        result.graph_transactions = graph_store.transaction_dicts()
        result.rawlsian_position_ledger = committed_rawls_positions(graph_store.graph)
        result.utilitarian_consequence_ledger = committed_utilitarian_consequences(
            graph_store.graph
        )
        result.deontological_duty_ledger = committed_deontological_assessments(
            graph_store.graph
        )
        result.virtue_character_ledger = committed_virtue_assessments(
            graph_store.graph
        )
        result.authoritative_semantic_state = project_authoritative_semantic_state(
            graph_store.graph, selected_action=result.current_plurality
        ).to_dict()
        result.trace_health = audit_trace_health(result)
        return result
