from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .models import CandidateChunk, CycleRecord, FailureCondition, PlanningAssessment, PlanningBranchEvaluation, ProblemReformulation, SynthesisProposal, VisibilityAssessment, WorkspaceAccessDecision, WorkspaceBroadcast, WorkspaceResult, clamp


SynthesisCallback = Callable[
    [str, Sequence[str], Sequence[CandidateChunk], WorkspaceBroadcast],
    SynthesisProposal | None,
]
ExtensionCallback = Callable[[WorkspaceResult], int]
ContingencyCallback = Callable[[WorkspaceResult], FailureCondition]
PlanningCallback = Callable[
    [str, Sequence[str], str, WorkspaceBroadcast, Sequence[CandidateChunk], str],
    PlanningAssessment,
]
ReformulationCallback = Callable[
    [str, Sequence[str], Sequence[CandidateChunk]],
    ProblemReformulation,
]
VisibilityCallback = Callable[[str, Sequence[str]], VisibilityAssessment]


class Specialist(Protocol):
    name: str

    def evaluate(
        self,
        scenario: str,
        actions: Sequence[str],
        broadcast: WorkspaceBroadcast,
    ) -> CandidateChunk: ...


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
                multiplier = (visibility_multipliers or {}).get(action, 1.0)
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
        """Make preserved substantive dissent operational after convergence."""
        if dissent is None or not dissent.schema_valid or not dissent.rationale:
            return ""
        alternative = dissent.recommended_action or (
            max(dissent.action_scores, key=dissent.action_scores.get)
            if dissent.action_scores else ""
        )
        if not alternative or alternative == selected_action:
            return ""
        if re.search(r"\b(?:random|coin|lottery)\b", dissent.landscape_tiebreaker, re.I):
            return ""
        typed_factual = (
            dissent.factual_reversal_threshold
            if dissent.factual_reversal_threshold.casefold() != "none"
            else ""
        )
        factual = " ".join(
            (dissent.revised_reversal_condition or typed_factual or dissent.reversal_condition).split()
        ).strip(" .")
        if factual.casefold() == "none":
            factual = ""
        if factual:
            return f"If {factual[0].lower() + factual[1:]}, prefer {alternative}"
        normative = " ".join(dissent.normative_reversal_threshold.split()).strip(" .")
        if normative and normative.casefold() != "none":
            return f"If {normative[0].lower() + normative[1:]}, prefer {alternative}"
        axis = " ".join(dissent.landscape_decisive_axis.replace("_", " ").split()).strip(" .")
        if axis:
            return (
                f"If {axis[0].lower() + axis[1:]} is judged overriding rather than "
                f"the leading consideration, prefer {alternative}"
            )
        rationale = dissent.rationale.strip(" .")
        return (
            f"If the {dissent.constraint.lower().replace('_', ' ')} objection that "
            f"{rationale[0].lower() + rationale[1:]} is judged overriding, "
            f"prefer {alternative}"
        )

    @staticmethod
    def _snapshot_specialist_state(specialists: Sequence[Specialist]) -> list[tuple[Specialist, dict]]:
        """Capture recurrent state that a hypothetical branch must not overwrite."""
        names = (
            "previous_recommendation_id", "previous_confidence", "assumption_status",
            "unsupported_assumption", "reversal_condition",
        )
        return [
            (specialist, {name: getattr(specialist, name) for name in names if hasattr(specialist, name)})
            for specialist in specialists
        ]

    @staticmethod
    def _restore_specialist_state(snapshot: Sequence[tuple[Specialist, dict]]) -> None:
        for specialist, values in snapshot:
            for name, value in values.items():
                setattr(specialist, name, value)

    @staticmethod
    def _explicit_implementation_obstacle(
        scenario: str, broadcast: WorkspaceBroadcast
    ) -> bool:
        """Require an obstacle stated by the problem, not inferred by a delegate."""
        context = " ".join((scenario, broadcast.contingency_question, broadcast.reformulation_context))
        return bool(re.search(
            r"\b(?:cannot|can't|unable|impossible|imminent|suicid\w*|before\b|"
            r"without\b|lacks?\b|loss\b|loses?\b|fail(?:s|ed|ure)?\b|"
            r"refus(?:e|es|ed|al)\b|unwilling\b|unavailable\b|deadline\b|"
            r"limited\b|only\b|risk(?:s|ed)?\b|breakdown\b|jam(?:s|med)?\b)",
            context,
            flags=re.IGNORECASE,
        ))

    @staticmethod
    def _validate_planning_assessment(
        assessment: PlanningAssessment,
        scenario: str,
        broadcast: WorkspaceBroadcast,
        actions: Sequence[str],
    ) -> PlanningAssessment:
        if not assessment.valid:
            return assessment
        grounding_context = " ".join(
            (scenario, broadcast.contingency_question, broadcast.reformulation_context)
        ).casefold()
        evidence = assessment.grounded_evidence.casefold().strip()
        errors: list[str] = []
        if len(evidence.split()) < 2 or evidence not in grounding_context:
            errors.append("implementation obstacle lacks an exact grounded quote")
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
        analyze_plan: PlanningCallback | None = None,
        scenario_facts: dict | None = None,
        source_testimonies: dict[str, str] | None = None,
        reformulate_problem: ReformulationCallback | None = None,
        assess_visibility: VisibilityCallback | None = None,
    ) -> WorkspaceResult:
        clean_actions = list(dict.fromkeys(a.strip() for a in actions if a.strip()))[:5]
        if len(clean_actions) < 2:
            raise ValueError("At least two distinct actions are required")

        broadcast = initial_broadcast or WorkspaceBroadcast()
        result = WorkspaceResult(scenario=scenario, actions=clean_actions)
        visibility_multipliers = {action: 1.0 for action in clean_actions}
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
        consensus_audit_attempted = False
        access_signal_signatures: set[tuple[str, ...]] = set()
        reformulation_attempted = False
        reversal_audit_attempted = False
        reformulation_baseline_action = ""
        extensions_used = 0
        cycle_limit = (
            min(self.config.max_cycles, self.config.high_urgency_cycles)
            if broadcast.urgency >= 0.8
            else self.config.max_cycles
        )

        cycle_number = 1
        while cycle_number <= cycle_limit:
            received_broadcast = broadcast
            is_planning_branch = received_broadcast.branch_kind == "PLANNING_CONTINGENCY"
            is_reformulation_probe = received_broadcast.constraint == "PROBLEM_REFORMULATION"
            is_reversal_probe = received_broadcast.constraint == "REVERSAL_AUDIT"
            is_counterfactual = (
                is_planning_branch or is_reformulation_probe or is_reversal_probe
            )
            specialist_snapshot = (
                self._snapshot_specialist_state(self.specialists) if is_counterfactual else []
            )
            if progress:
                progress(f"Cycle {cycle_number}/{cycle_limit} — broadcast {broadcast.constraint}")
            candidates = []
            for index, specialist in enumerate(self.specialists, start=1):
                call_started = time.monotonic()
                if progress:
                    progress(
                        f"  [{index}/{len(self.specialists)}] {specialist.name} delegate thinking..."
                    )
                candidate = specialist.evaluate(scenario, clean_actions, broadcast)
                candidates.append(candidate)
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
                            "vote_damped"
                        )
                    if candidate.landscape_search_attempted and not candidate.landscape_semantic_valid:
                        validation_note += (
                            "; landscape_penalty="
                            + " | ".join(candidate.landscape_validation_errors[:2])
                        )
                    if candidate.independence_bonus:
                        validation_note += "; grounded_nonconsensus_bonus=1.00"
                    progress(
                        f"  [{index}/{len(self.specialists)}] {specialist.name} returned "
                        f"{candidate.constraint}; recommends={candidate.recommended_action or 'unknown'}; "
                        f"preference={candidate.preference_strength:.2f}; "
                        f"epistemic={candidate.epistemic_confidence:.2f}; "
                        f"alignment={candidate.testimony_alignment}; "
                        f"why={candidate.rationale}{validation_note} "
                        f"in {time.monotonic() - call_started:.1f}s"
                    )
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
            policy = self._policy(candidates, clean_actions, visibility_multipliers)
            selected_action = max(policy, key=policy.get)
            if not is_counterfactual:
                stable_cycles = stable_cycles + 1 if selected_action == previous_action else 1
                previous_action = selected_action
            entropy = self._normalized_entropy(policy)
            dissent = self._dissent(candidates, selected_action)
            elapsed = time.monotonic() - started

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
            if not consensus_audit_attempted:
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
                        next_broadcast = WorkspaceBroadcast(
                            constraint="CONSENSUS_AUDIT",
                            intent=f"audit_{selected_action}",
                            urgency=broadcast.urgency,
                            danger_probability=broadcast.danger_probability,
                            unresolved="VERIFY_ASSUMPTIONS",
                            contingency_question=access_decision.question,
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
                and dissent is not None
                and not audit_broadcast
                and not reformulation_broadcast
                and cycle_number < cycle_limit
                and elapsed < self.config.time_budget_seconds
                and bool(dissent.decision_rule)
                and any(
                    value and value.casefold() != "none"
                    for value in (
                        dissent.factual_reversal_threshold,
                        dissent.normative_reversal_threshold,
                    )
                )
            ):
                alternative = dissent.recommended_action or (
                    max(dissent.action_scores, key=dissent.action_scores.get)
                    if dissent.action_scores else ""
                )
                factual = dissent.factual_reversal_threshold
                normative = dissent.normative_reversal_threshold
                proposed = factual if factual.casefold() != "none" else normative
                if proposed.casefold() == "none":
                    proposed = dissent.reversal_condition or dissent.decision_rule
                challenge = (
                    f"{dissent.specialist} challenges '{selected_action}' with '{alternative}'. "
                    f"Rule: {dissent.decision_rule or dissent.landscape_tiebreaker}. "
                    f"Proposed reversal: {proposed or dissent.rationale}"
                )
                if len(challenge.split()) >= 8:
                    reversal_audit_attempted = True
                    reversal_resume_broadcast = next_broadcast
                    next_broadcast = WorkspaceBroadcast(
                        constraint="REVERSAL_AUDIT",
                        intent=f"test_reversal_of_{selected_action}",
                        urgency=broadcast.urgency,
                        danger_probability=broadcast.danger_probability,
                        unresolved="TEST_REVERSAL",
                        contingency_question=(
                            "Does the critic's condition reverse the ranking? "
                            "Accept it, revise it to the smallest valid condition, or reject it."
                        ),
                        reversal_challenge=challenge,
                    )
                    reversal_audit_broadcast = True
                    cycle_limit += 1
                    result.access_decisions.append(WorkspaceAccessDecision(
                        cycle=cycle_number,
                        content_type="REVERSAL_AUDIT",
                        admitted=True,
                        signals=["substantive_dissent", "explicit_competing_rule"],
                        question=challenge,
                        rationale="A grounded critic must test the leading rule before convergence.",
                    ))
                    if progress:
                        progress(f"  adversarial reversal review admitted: {challenge}")
            if (
                self.config.enable_planning
                and analyze_plan is not None
                and not audit_broadcast
                and not reformulation_broadcast
                and not reversal_audit_broadcast
                and planning_reason
                and planning_key not in planned_contexts
                and planning_branches_used < self.config.max_planning_branches
                and self._explicit_implementation_obstacle(scenario, broadcast)
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
                        clean_actions.append(proposal.action)
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
                candidate.assumption_status in {"NOT_AUDITED", "SUPPORTED"}
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
                good_recurrent_deliberation = (
                    request_extension is not None
                    and analyze_contingency is not None
                    and extensions_used < self.config.max_cycle_extensions
                    and any(proposal.accepted for proposal in result.synthesis_proposals)
                    and dissent is not None
                    and entropy >= self.config.synthesis_min_entropy
                    and valid_ratio >= 0.8
                    and broadcast.urgency < 0.8
                )
                added_cycles = 0
                if good_recurrent_deliberation:
                    analysis = analyze_contingency(result)
                    result.failure_conditions.append(analysis)
                    if analysis.valid:
                        broadcast = WorkspaceBroadcast(
                            constraint="CONTINGENCY_REVIEW",
                            intent="evaluate_synthesis_failure_fallback",
                            urgency=broadcast.urgency,
                            danger_probability=broadcast.danger_probability,
                            unresolved="CHECK_FEASIBILITY",
                            contingency_question=analysis.contingency_question,
                        )
                        added_cycles = max(0, int(request_extension(result)))
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
                    result.halted_by = "cycle_budget"
                    break
            cycle_number += 1

        if not result.halted_by:
            result.halted_by = "cycle_budget"
        actual_cycles = [cycle for cycle in result.cycles if not cycle.is_hypothetical]
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
        result.moral_residue = sorted({
            candidate.constraint
            for cycle in actual_cycles
            for candidate in cycle.candidates
            if result.current_plurality
            and candidate.schema_valid
            and candidate.action_scores.get(result.current_plurality, 0.0) < 0.5
        } | {
            cycle.dissent.constraint
            for cycle in actual_cycles
            if cycle.dissent is not None and cycle.dissent.schema_valid
        })
        result.reopen_conditions = sorted({
            candidate.unresolved
            for cycle in actual_cycles
            for candidate in cycle.candidates
            if candidate.unresolved != "NONE"
        })
        dissent_reversal = self._dissent_reversal_condition(
            final.dissent, result.current_plurality
        )
        if dissent_reversal and dissent_reversal not in result.reopen_conditions:
            result.reopen_conditions.append(dissent_reversal)
        publishable_judgment = (
            result.judgment_status == "CONTESTED_RECOMMENDATION"
            or (
                result.judgment_status == "ACTION_RECOMMENDATION"
                and result.halted_by == "convergence"
            )
        )
        if not final.winner.schema_valid or not publishable_judgment:
            result.compressed_rule = f"Unavailable: deliberation halted by {result.halted_by}."
        else:
            qualifier = (
                "contestedly prefer"
                if result.judgment_status == "CONTESTED_RECOMMENDATION"
                else "prefer"
            )
            governing = next(
                (
                    candidate.decision_rule
                    for candidate in final.candidates
                    if candidate.schema_valid
                    and candidate.recommended_action == result.current_plurality
                    and candidate.decision_rule
                ),
                "",
            )
            result.compressed_rule = (
                f"Rule: {governing}. " if governing else ""
            ) + (
                f"{qualifier} {result.selected_action}; reopen if "
                f"{', '.join(result.reopen_conditions) or 'material facts change'}."
            )
        return result
