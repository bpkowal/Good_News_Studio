from __future__ import annotations

import copy
import math
import re
import textwrap
import time
from dataclasses import dataclass, replace
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
from .models import ArgumentChallenge, AutonomyAssessment, CandidateChunk, ContingencyFeasibilityAssessment, CycleRecord, FailureCondition, PlanningAssessment, PlanningBranchEvaluation, ProblemReformulation, SynthesisProposal, SynthesisViabilityAssessment, VisibilityAssessment, WorkspaceAccessDecision, WorkspaceBroadcast, WorkspaceResult, clamp
from .specialist_authority import (
    CONTESTED_RECOMMENDATION,
    GOVERNED_RECOMMENDATION,
    REOPEN_PRIORITY_THRESHOLD,
    UNRESOLVED,
    apply_investigative_authority,
    apply_specialist_authority,
    apply_validator_governance_gate,
    can_supply_governing_basis,
    claim_ref,
    classify_terminal_judgment,
    derive_broadcast_authority,
    evidence_fingerprint_for,
    has_unique_policy_leader,
    select_governing_claim,
)
from .semantic_invariants import (
    SemanticProposition,
    compile_preference_rule,
    validate_transformation,
)
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode
from .scenario_semantics import (
    classify_planning_failure_grounding,
    project_grounded_action_effects,
)
from .trace_health import audit_trace_health
from .decision_boundaries import select_collective_reversal_boundary
from .deliberative_state import (
    apply_verified_challenge_supersession,
    build_deliberative_problem_state,
    observe_broadcast_influence,
    opening_problem_state,
    update_broadcast_influence_persistence,
)
from .graph_transactions import SemanticGraphStore
from .epistemic_ledger import (
    apply_side_premise_audit,
    attach_candidate_dependencies,
    focus_proposition_ids,
    ledger_projection,
    seed_proposition_ledger,
    shared_unresolved_dependency_projection,
)
from .resolved_questions import (
    commit_question_resolution,
    resolve_audited_question,
    settled_question_keys,
)
from .rawls_ledger import (
    apply_rawls_ledger_transaction,
    committed_rawls_positions,
)
from .utilitarian_ledger import (
    apply_utilitarian_ledger_transaction,
    committed_utilitarian_consequences,
)
from .deontology_ledger import (
    active_calibration_errors,
    apply_deontological_ledger_transaction,
    calibration_issue_kinds_for_challenge,
    classify_deontological_authority,
    committed_deontological_assessments,
    harm_relation_conflicts_with_graph,
    latest_assessment_by_action,
    omission_classified_as_perfect_negative_violation,
    supersede_calibration_issues,
)
from .virtue_ledger import (
    apply_virtue_ledger_transaction,
    committed_virtue_assessments,
)
from .care_ledger import apply_care_ledger_transaction, committed_care_assessments
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
from .local_specialists import _invalid_candidate


SynthesisCallback = Callable[
    [str, Sequence[str], Sequence[CandidateChunk], WorkspaceBroadcast],
    SynthesisProposal | None,
]


def _preserve_problem_state_after_invalid_cycle(
    previous_state: dict[str, object] | None,
    candidates: Sequence[CandidateChunk],
) -> dict[str, object]:
    """Retain deliberative knowledge when a whole model cycle is unusable."""
    state = copy.deepcopy(previous_state or {})
    if not state:
        return state
    failed_specialists = sorted({
        candidate.specialist for candidate in candidates if not candidate.schema_valid
    })
    failure_question = {
        "question_key": "QUESTION:MODEL_OUTPUT_FAILURE",
        "source_specialist": "workspace_runtime",
        "raised_by": failed_specialists or ["workspace_runtime"],
        "category": "REVIEW_MODEL_OUTPUT",
        "uncertainty_kind": "MODEL_OUTPUT_UNCERTAINTY",
        "question": (
            "The latest delegate cycle produced insufficient valid structured responses; "
            "review model output before treating prior unresolved issues as resolved."
        ),
        "grounded_in": [],
        "source_type": "SYSTEM_OPERATIONAL_FAILURE",
    }
    questions = list(state.get("unresolved_questions", []) or [])
    if not any(
        item.get("question_key") == failure_question["question_key"]
        for item in questions if isinstance(item, dict)
    ):
        questions.append(failure_question)
    state["unresolved_questions"] = questions
    categories = list(state.get("unresolved_categories", []) or [])
    if "REVIEW_MODEL_OUTPUT" not in categories:
        categories.append("REVIEW_MODEL_OUTPUT")
    state["unresolved_categories"] = categories
    state["primary_unresolved"] = "REVIEW_MODEL_OUTPUT"
    delta = dict(state.get("problem_delta", {}) or {})
    new_questions = list(delta.get("new_questions", []) or [])
    if not any(
        item.get("question_key") == failure_question["question_key"]
        for item in new_questions if isinstance(item, dict)
    ):
        new_questions.append(failure_question)
    delta["new_questions"] = new_questions
    state["problem_delta"] = delta
    return state
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
PremiseAuditCallback = Callable[
    [Sequence[dict[str, object]], Sequence[CandidateChunk]], dict[str, object]
]


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
    if (
        state_status != "UPDATE_REJECTED"
        and candidate.framework_grounding_penalty < penalty
    ):
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
    elif state_status == "CALIBRATED_CONTESTED":
        # The framework survived; calibration established that its current
        # conclusion remains contested. This is an admissible judgment, not
        # framework loss.
        candidate.framework_constraint_retained = True
        candidate.framework_retention_status = "CALIBRATED_CONTESTED"
    else:
        candidate.framework_constraint_retained = False
        candidate.framework_retention_status = "COMMITTED_WITH_UNCERTAINTY"


def _framework_retention_failed(candidate: CandidateChunk) -> bool:
    return bool(
        not candidate.framework_constraint_retained
        or candidate.framework_retention_status.upper() in {
            "LOST", "UPDATE_REJECTED",
        }
    )


_FRAMEWORK_STATE_FIELDS = (
    "constraint", "action_scores", "recommended_action", "rationale",
    "decision_rule", "framework_action_map", "framework_numerical_role",
    "framework_numerical_justification", "action_admissibility",
    "comparison_complete", "evidence_sufficient_for_action", "interim_action",
    "factual_reversal_threshold", "normative_reversal_threshold",
    "utilitarian_consequence_table", "utilitarian_ledger_proposal",
    "rawls_position_proposal", "deontological_ledger_proposal",
    "virtue_character_proposal", "care_ledger_proposal", "care_relational_map", "care_numerical_role",
    "care_numerical_justification",
)


def _framework_state_projection(candidate: CandidateChunk) -> dict[str, object]:
    """Snapshot only the framework-authoritative state subject to rollback."""
    return {
        field: copy.deepcopy(getattr(candidate, field))
        for field in _FRAMEWORK_STATE_FIELDS
    }


def _record_committed_native_ledger(
    candidate: CandidateChunk,
    *,
    ledger_kind: str,
    transaction_status: str,
    records: Sequence[dict[str, object]],
) -> None:
    """Attach only graph-committed native records to the operative candidate."""
    if not str(transaction_status).startswith("COMMITTED") or not records:
        return
    candidate.committed_native_ledger = {
        "ledger_kind": ledger_kind,
        "transaction_status": str(transaction_status),
        "records": copy.deepcopy(list(records)),
    }


def _merge_unique(values: Sequence[object], additions: Sequence[object], limit: int) -> list:
    merged: list = []
    for value in [*values, *additions]:
        if value not in merged:
            merged.append(copy.deepcopy(value))
        if len(merged) >= limit:
            break
    return merged


def _preserve_current_cycle_components(
    restored: CandidateChunk, candidate: CandidateChunk,
) -> list[str]:
    """Overlay facts and diagnostics that a normative transaction cannot reject."""
    preserved: list[str] = []

    restored.supporting_proposition_ids = _merge_unique(
        restored.supporting_proposition_ids,
        candidate.supporting_proposition_ids,
        24,
    )
    restored.decision_critical_proposition_ids = _merge_unique(
        restored.decision_critical_proposition_ids,
        candidate.decision_critical_proposition_ids,
        12,
    )
    restored.decision_critical_dependency_claims = _merge_unique(
        restored.decision_critical_dependency_claims,
        candidate.decision_critical_dependency_claims,
        4,
    )
    restored.material_empirical_claims = _merge_unique(
        restored.material_empirical_claims,
        candidate.material_empirical_claims,
        12,
    )
    restored.epistemic_binding_notes = _merge_unique(
        restored.epistemic_binding_notes,
        candidate.epistemic_binding_notes,
        12,
    )
    epistemic_rank = {
        "REJECTED": 0, "HYPOTHETICAL": 1, "UNRESOLVED": 2,
        "DERIVED": 3, "ESTABLISHED": 4,
    }
    restored.weakest_decision_critical_status = min(
        (
            restored.weakest_decision_critical_status,
            candidate.weakest_decision_critical_status,
        ),
        key=lambda status: epistemic_rank.get(str(status).upper(), 0),
    )
    if candidate.side_premise_audit_status != "NOT_RUN":
        restored.side_premise_audit_status = candidate.side_premise_audit_status
        restored.side_premise_audit_findings = copy.deepcopy(
            candidate.side_premise_audit_findings
        )
    if candidate.assumption_status != "NOT_AUDITED":
        restored.assumption_status = candidate.assumption_status
    if candidate.unsupported_assumption:
        restored.unsupported_assumption = candidate.unsupported_assumption
    if candidate.unresolved != "NONE":
        restored.unresolved = candidate.unresolved
    restored.epistemic_confidence = min(
        restored.epistemic_confidence, candidate.epistemic_confidence,
    )
    restored.confidence = restored.epistemic_confidence
    if candidate.selection_status == "PROVISIONAL":
        restored.selection_status = "PROVISIONAL"
    restored.comparison_complete = (
        restored.comparison_complete and candidate.comparison_complete
    )
    restored.evidence_sufficient_for_action = (
        restored.evidence_sufficient_for_action
        and candidate.evidence_sufficient_for_action
    )
    if candidate.evidence_basis == "UNSTATED_FACTS":
        restored.evidence_basis = candidate.evidence_basis
    if candidate.speculative_claim:
        restored.speculative_claim = candidate.speculative_claim
    preserved.append("EPISTEMIC_DEPENDENCIES")

    restored.validation_errors = _merge_unique(
        restored.validation_errors, candidate.validation_errors, 12,
    )
    restored.framework_validation_errors = _merge_unique(
        restored.framework_validation_errors,
        [f"rejected update: {error}" for error in candidate.framework_validation_errors],
        16,
    )
    restored.framework_insights = _merge_unique(
        restored.framework_insights, candidate.framework_insights, 6,
    )
    restored.framework_internal_conflicts = _merge_unique(
        restored.framework_internal_conflicts,
        candidate.framework_internal_conflicts,
        3,
    )
    restored.framework_specific_open_questions = _merge_unique(
        restored.framework_specific_open_questions,
        [
            *candidate.framework_specific_open_questions,
            *(
                str(item.get("proposition", ""))
                for item in candidate.framework_insights
                if str(item.get("proposition", "")).strip()
            ),
        ],
        3,
    )
    if candidate.challenge_response:
        restored.challenge_response = copy.deepcopy(candidate.challenge_response)
        preserved.append("CHALLENGE_RESPONSE")
    restored.landscape_validation_errors = _merge_unique(
        restored.landscape_validation_errors,
        candidate.landscape_validation_errors,
        8,
    )
    preserved.extend(["FRAMEWORK_DIAGNOSTICS", "OPEN_QUESTIONS"])

    restored.self_reported_broadcast_dependence = (
        candidate.self_reported_broadcast_dependence
    )
    if (
        candidate.workspace_proposition_response != "NOT_APPLICABLE"
        or candidate.workspace_reasoning_effect != "NONE"
    ):
        restored.workspace_proposition_response = candidate.workspace_proposition_response
        restored.workspace_reasoning_effect = candidate.workspace_reasoning_effect
        restored.framework_application = candidate.framework_application
    if candidate.audit_participation != "NOT_TESTED":
        restored.audit_participation = candidate.audit_participation
        restored.audit_internal_effect = candidate.audit_internal_effect
        restored.audit_framework_explanation = candidate.audit_framework_explanation
        restored.audit_variable = copy.deepcopy(candidate.audit_variable)
    if candidate.visibility_response != "NOT_TESTED":
        restored.visibility_response = candidate.visibility_response
        restored.visibility_justification = candidate.visibility_justification
        restored.visibility_harm_revision = candidate.visibility_harm_revision
        restored.visibility_magnitude_status = candidate.visibility_magnitude_status
        restored.visibility_magnitude_overreach = candidate.visibility_magnitude_overreach
    if candidate.reversal_review_response != "NOT_TESTED":
        restored.reversal_review_response = candidate.reversal_review_response
        restored.reversal_review_justification = candidate.reversal_review_justification
        restored.revised_reversal_condition = candidate.revised_reversal_condition
        restored.reversal_review_valid = candidate.reversal_review_valid
        restored.reversal_review_error = candidate.reversal_review_error
    preserved.append("CURRENT_CYCLE_MEASUREMENTS")
    return preserved


def _operative_framework_candidates(
    candidates: Sequence[CandidateChunk],
    last_valid: dict[str, CandidateChunk],
    *,
    remember: bool,
) -> list[CandidateChunk]:
    """Use accepted framework state, never a rejected mutation, deliberatively.

    Rejected recurrent proposals remain in graph transactions and trace-health
    output. Policy, salience, ProblemState, and broadcasts instead receive the
    last accepted candidate snapshot. A structurally valid first state is never
    treated as a failed mutation merely because there is no prior snapshot: it
    remains operative, with its existing grounding penalty and diagnostics.
    """
    operative: list[CandidateChunk] = []
    for candidate in candidates:
        if not candidate.schema_valid:
            operative.append(candidate)
            continue
        if _framework_retention_failed(candidate):
            previous = last_valid.get(candidate.specialist)
            if previous is None:
                admitted = copy.deepcopy(candidate)
                if candidate.framework_retention_status == "UPDATE_REJECTED":
                    # A rejected first ledger is diagnostic only; it cannot
                    # become this cycle's policy baseline.
                    admitted.adjudication_status = "CONTESTED_NO_LEANING"
                    admitted.governing_eligible = False
                    admitted.broadcast_authority = "INVESTIGATIVE"
                    admitted.policy_weight_factor = 0.0
                    admitted.recommended_action = ""
                    if admitted.action_scores:
                        admitted.action_scores = {
                            action: 0.5 for action in admitted.action_scores
                        }
                    admitted.preference_strength = 0.0
                    admitted.framework_constraint_retained = True
                    admitted.framework_retention_status = (
                        "FIRST_STATE_REJECTED_NOT_OPERATIVE"
                    )
                    state = _framework_state_projection(admitted)
                    admitted.proposed_framework_state = _framework_state_projection(
                        candidate
                    )
                    admitted.committed_framework_state = copy.deepcopy(state)
                    operative.append(admitted)
                    if remember:
                        last_valid[admitted.specialist] = copy.deepcopy(admitted)
                    continue
                admitted.framework_constraint_retained = True
                admitted.framework_retention_status = (
                    "FIRST_STATE_ADMITTED_WITH_WARNINGS"
                    if admitted.framework_validation_errors
                    or admitted.framework_grounding_penalty > 0.0
                    else "FIRST_STATE_ADMITTED"
                )
                state = _framework_state_projection(admitted)
                admitted.proposed_framework_state = copy.deepcopy(state)
                admitted.committed_framework_state = copy.deepcopy(state)
                operative.append(admitted)
                if remember:
                    last_valid[admitted.specialist] = copy.deepcopy(admitted)
                continue
            restored = copy.deepcopy(previous)
            restored.proposed_framework_state = _framework_state_projection(candidate)
            restored.committed_framework_state = _framework_state_projection(previous)
            restored.framework_retention_status = "PRESERVED_AFTER_REJECTED_UPDATE"
            restored.framework_constraint_retained = True
            restored.preserved_current_cycle_components = (
                _preserve_current_cycle_components(restored, candidate)
            )
            operative.append(restored)
            if remember:
                last_valid[restored.specialist] = copy.deepcopy(restored)
            continue
        previous = last_valid.get(candidate.specialist)
        if not candidate.committed_native_ledger and previous is not None:
            candidate.committed_native_ledger = copy.deepcopy(
                previous.committed_native_ledger
            )
        operative.append(candidate)
        state = _framework_state_projection(candidate)
        candidate.proposed_framework_state = copy.deepcopy(state)
        candidate.committed_framework_state = copy.deepcopy(state)
        if remember:
            last_valid[candidate.specialist] = copy.deepcopy(candidate)
    return operative


_LEGACY_DECISION_VARIABLE_PROBES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\bthird[- ]party\b|\bthird parties\b", re.IGNORECASE),
        "third_party_status",
        "Does the recommendation depend on whether {term} counts as a third party or as an authorized internal agent?",
    ),
    (
        re.compile(r"\bauthoriz\w*|\bpermission\b|\bconsent\b", re.IGNORECASE),
        "authorization_scope",
        "Does the recommendation depend on whether permission or consent covers this specific disclosure or only a narrower use?",
    ),
    (
        re.compile(r"\binternal\b|\binternal agent\b|\bexternal\b|\bon behalf of\b|\bagent\b|\bemploye\w*\b", re.IGNORECASE),
        "organizational_boundary",
        "Does the recommendation depend on whether {term} is inside or outside the relevant organizational boundary?",
    ),
    (
        re.compile(r"\bprivate\b|\bprivacy\b|\bconfidential\w*\b", re.IGNORECASE),
        "privacy_scope",
        "Does 'keep it private' extend to this disclosure, or only to third-party sharing?",
    ),
)


def _graph_first_latent_variable_probe(
    graph: SemanticGraph,
    selected_action: str,
) -> tuple[list[str], str, dict[str, object]] | None:
    """Derive a consensus audit variable from committed graph structure.

    This is intentionally graph-first: it prefers already-committed semantic
    state such as Rawlsian tradeoffs and problem-shape relations before it ever
    falls back to legacy text heuristics.
    """
    state = project_authoritative_semantic_state(graph, selected_action=selected_action)
    if state.decision_variables:
        variable = state.decision_variables[0]
        label = " ".join(str(variable.get("label", "")).split()).strip()
        label = label or "unresolved decision variable"
        relation = str(variable.get("probe_kind", "MISSING_DECISION_VARIABLE")).upper()
        entity = label
        question = (
            f"Which value of the graph-committed variable '{label}' would change the "
            f"recommendation for '{selected_action}'?"
        )
        payload = {
            "entity": entity,
            "relation": relation,
            "possible_values": [
                "NO_CHANGE",
                "WEAKENS",
                "REVERSES",
                "UNRESOLVED",
            ],
            "focus_action": selected_action,
            "question": question,
            "required_response": {
                "counterfactual_anchor": label,
                "allowed_effects": [
                    "NO_CHANGE",
                    "WEAKENS",
                    "REVERSES",
                    "UNRESOLVED",
                ],
            },
        }
        return (
            ["graph_decision_variable", "graph_latent_variable"],
            question,
            payload,
        )

    rawls_positions = [
        position for position in state.rawlsian_positions
        if str(position.get("effect", "")).upper() in {"MIXED", "UNCERTAIN"}
    ]
    if rawls_positions:
        chosen = next(
            (
                position for position in rawls_positions
                if str(position.get("canonical_action_id", "")).strip() == selected_action
            ),
            rawls_positions[0],
        )
        subject = " ".join(
            str(chosen.get("subject") or chosen.get("affected_subject") or "the affected subject").split()
        ).strip()
        dimension = " ".join(
            str(chosen.get("dimension") or "primary goods").split()
        ).strip()
        rival = " ".join(
            str(chosen.get("compared_to_action_id") or "the rival action").split()
        ).strip()
        effect = str(chosen.get("effect", "UNCERTAIN")).upper()
        relation = (
            "DISTRIBUTIVE_TRADEOFF"
            if effect == "MIXED"
            else "RAWLSIAN_DIRECTION_UNCERTAIN"
        )
        distributive_shape = next((
            item for item in state.problem_shape_relations
            if str(item.relation).upper() == "AGGREGATE_VS_DISTRIBUTIVE"
        ), None)
        if distributive_shape is not None:
            possessive_subject = f"{subject}'" if subject.casefold().endswith("s") else f"{subject}'s"
            entity = f"{possessive_subject} overall primary-goods position"
            question = (
                f"{selected_action} improves the stated opportunity, access, and living "
                f"conditions for {subject}, while {rival} offers the higher aggregate gain. "
                f"Could the aggregate loss under {selected_action} become large enough to "
                f"leave {subject} worse off overall?"
            )
        else:
            readable_dimension = (
                "overall primary-goods position"
                if dimension in {"OTHER_PRIMARY_GOOD", "UNKNOWN"}
                else dimension.lower().replace("_", " ")
            )
            entity = f"{subject} on {readable_dimension}"
            question = (
                f"Which action better protects {subject}'s {readable_dimension}, "
                f"and what stated fact would make {rival} preferable instead?"
            )
        payload = {
            "entity": entity,
            "relation": relation,
            "possible_values": [
                "HIGHER_MINIMUM_POSITION",
                "HIGHER_AGGREGATE_GAIN",
                "UNRESOLVED",
            ],
            "focus_action": selected_action,
            "question": question,
            "required_response": {
                "counterfactual_anchor": entity,
                "allowed_effects": [
                    "NO_CHANGE",
                    "WEAKENS",
                    "REVERSES",
                    "UNRESOLVED",
                ],
            },
        }
        return (
            [
                "graph_rawlsian_tradeoff" if effect == "MIXED" else "graph_rawlsian_direction",
                "graph_latent_variable",
            ],
            question,
            payload,
        )

    priority_relations = {
        "AGGREGATE_VS_DISTRIBUTIVE",
        "DECISION_BOUNDARY",
        "DECISION_CRITICAL_UNKNOWN",
        "TEMPORAL_RISK_ASYMMETRY",
        "EPISTEMIC_ASYMMETRY",
        "ASYMMETRIC_COST",
    }
    relation = next(
        (
            item for item in state.problem_shape_relations
            if str(item.relation).upper() in priority_relations
        ),
        None,
    )
    if relation is not None:
        relation_name = str(relation.relation).upper()
        statement = " ".join(str(relation.statement).split())
        if relation_name == "DECISION_BOUNDARY":
            from .uncertainty_types import build_boundary_audit_fields
            boundary = relation.statement.split(" if ", 1)[-1] if " if " in relation.statement.lower() else relation.statement
            entity = boundary.strip() or "decision boundary"
            condition = boundary.strip() or statement
            question = (
                f"Does the graph-committed boundary '{statement}' actually change the recommendation "
                f"for '{selected_action}'?"
            )
            counterfactual_anchor = boundary.strip() or statement or selected_action
            payload = build_boundary_audit_fields(
                condition=condition,
                expected_effect="REVERSES_FRAMEWORK_PREFERENCE",
                boundary_status="UNRESOLVED",
            )
            payload.update({
                "entity": entity,
                "focus_action": selected_action,
                "question": question,
                "required_response": {
                    "counterfactual_anchor": counterfactual_anchor,
                    "allowed_effects": [
                        "NO_CHANGE",
                        "WEAKENS",
                        "REVERSES",
                        "REVERSES_FRAMEWORK_PREFERENCE",
                        "UNRESOLVED",
                    ],
                },
            })
            return (
                [f"graph_{relation_name.lower()}", "graph_latent_variable"],
                question,
                payload,
            )
        elif relation_name == "TEMPORAL_RISK_ASYMMETRY":
            entity = "present-versus-delayed risk tradeoff"
            possible_values = [
                "IMMEDIATE_HARM_DOMINATES",
                "DELAYED_RISK_DOMINATES",
                "UNRESOLVED",
            ]
            question = (
                f"The graph says {statement}. Does the delayed catastrophe really outweigh the "
                f"immediate harm for '{selected_action}'?"
            )
            counterfactual_anchor = "delayed catastrophe versus immediate harm"
        elif relation_name == "EPISTEMIC_ASYMMETRY":
            entity = "certainty-versus-uncertainty tradeoff"
            possible_values = [
                "CERTAIN_HARM_DOMINATES",
                "UNCERTAIN_HARM_DOMINATES",
                "UNRESOLVED",
            ]
            question = (
                f"The graph says {statement}. Which side is actually more decision-critical for "
                f"'{selected_action}'?"
            )
            counterfactual_anchor = "certainty versus uncertainty"
        elif relation_name == "ASYMMETRIC_COST":
            entity = "burden distribution tradeoff"
            possible_values = [
                "MORE_COSTLY_LEFT",
                "MORE_COSTLY_RIGHT",
                "UNRESOLVED",
            ]
            question = (
                f"The graph says {statement}. Which option bears the larger burden for "
                f"'{selected_action}'?"
            )
            counterfactual_anchor = "relative burden distribution"
        else:
            entity = statement or relation_name.lower().replace("_", " ")
            possible_values = [
                "NO_CHANGE",
                "WEAKENS",
                "REVERSES",
                "UNRESOLVED",
            ]
            question = (
                f"The graph identifies {statement}. Which value of this unresolved variable would "
                f"change the recommendation for '{selected_action}'?"
            )
            counterfactual_anchor = statement or relation_name
        payload = {
            "entity": entity,
            "relation": relation_name,
            "possible_values": possible_values,
            "focus_action": selected_action,
            "question": question,
            "required_response": {
                "counterfactual_anchor": counterfactual_anchor,
                "allowed_effects": [
                    "NO_CHANGE",
                    "WEAKENS",
                    "REVERSES",
                    "UNRESOLVED",
                ],
            },
        }
        return (
            [f"graph_{relation_name.lower()}", "graph_latent_variable"],
            question,
            payload,
        )

    return None


def _legacy_missing_decision_variable_probe(
    scenario: str,
    selected_action: str,
    source_testimonies: dict[str, str] | None,
    scenario_facts: dict | None,
) -> tuple[list[str], str, dict[str, object]]:
    """Identify unresolved boundary facts that may drive consensus."""
    # Testimony may expose uncertainty, but it is not an authority for adding
    # entities or relations to this run's scenario ontology.  In particular,
    # stale framework output must not instantiate a privacy/agency probe in an
    # unrelated scenario.
    probe_text = " ".join([scenario, selected_action]).casefold()
    fact_text = " ".join(map(str, (scenario_facts or {}).values())).casefold()
    signals: list[str] = []
    questions: list[str] = []
    payload: dict[str, object] = {}
    for pattern, signal, template in _LEGACY_DECISION_VARIABLE_PROBES:
        match = pattern.search(probe_text)
        if not match:
            continue
        if signal in {"third_party_status", "organizational_boundary"} and re.search(
            r"\b(?:authorized|internal|external|employee|agent|third[- ]party)\b",
            fact_text,
        ):
            continue
        term = " ".join(match.group(0).split())
        signals.append(signal)
        if signal == "third_party_status":
            question = (
                f"Classify the entity '{term}' as EXTERNAL_THIRD_PARTY "
                f"(third party), AUTHORIZED_INTERNAL_AGENT (authorized internal agent), "
                f"or UNKNOWN, then state whether the "
                f"recommendation for '{selected_action}' changes under the "
                f"AUTHORIZED_INTERNAL_AGENT reading."
            )
        elif signal == "organizational_boundary":
            question = (
                f"Classify '{term}' relative to the relevant organization as "
                f"INTERNAL, EXTERNAL, or UNKNOWN, then state whether that boundary "
                f"changes the recommendation for '{selected_action}'."
            )
        elif signal == "authorization_scope":
            question = (
                f"Does permission or consent cover the disclosure in '{selected_action}', "
                f"or is the scope narrower? State whether the recommendation changes."
            )
        else:
            question = (
                f"Does 'keep it private' extend to '{selected_action}', or only to "
                f"third-party sharing? State whether the recommendation changes."
            )
        questions.append(question)
        if not payload:
            relation = signal.upper()
            if signal in {"third_party_status", "organizational_boundary"}:
                possible_values = [
                    "EXTERNAL_THIRD_PARTY",
                    "AUTHORIZED_INTERNAL_AGENT",
                    "UNKNOWN",
                ]
            elif signal == "authorization_scope":
                possible_values = [
                    "COVERS_DISCLOSURE",
                    "DOES_NOT_COVER_DISCLOSURE",
                    "UNKNOWN",
                ]
            else:
                possible_values = [
                    "EXTENDS_TO_DISCLOSURE",
                    "LIMITED_TO_PRIVATE_SCOPE",
                    "UNKNOWN",
                ]
            payload = {
                "entity": term,
                "relation": relation,
                "possible_values": possible_values,
                "focus_action": selected_action,
                "question": question,
                "required_response": {
                    "counterfactual_anchor": (
                        "AUTHORIZED_INTERNAL_AGENT"
                        if relation in {"THIRD_PARTY_STATUS", "ORGANIZATIONAL_BOUNDARY"}
                        else "COVERS_DISCLOSURE"
                        if relation == "AUTHORIZATION_SCOPE"
                        else "EXTENDS_TO_DISCLOSURE"
                    ),
                    "allowed_effects": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
                },
            }
    if len(questions) > 1:
        question = " ".join(questions[:2])
    else:
        question = questions[0] if questions else ""
    return signals, question, payload


def _audit_variable_has_current_run_provenance(
    payload: dict[str, object],
    scenario: str,
    actions: Sequence[str],
    graph: SemanticGraph | None = None,
) -> bool:
    """Admit an audit entity only when the current problem names it.

    Possible values are explicit hypothetical branches and therefore need not
    already be facts.  Their entity anchor does: framework testimony and prior
    audit output cannot supply it.
    """
    if not payload:
        return True
    entity = " ".join(str(payload.get("entity", "")).casefold().split()).strip()
    if not entity:
        return False
    current_problem_parts = [
        " ".join(str(value).casefold().split())
        for value in (scenario, *actions)
    ]
    if graph is not None:
        for node in graph.nodes.values():
            if "consensus_access_gate" in node.provenance:
                continue
            current_problem_parts.append(" ".join(node.label.casefold().split()))
            aliases = node.attributes.get("aliases", [])
            if isinstance(aliases, (list, tuple, set)):
                current_problem_parts.extend(
                    " ".join(str(alias).casefold().split()) for alias in aliases
                )
    current_problem_text = " ".join(current_problem_parts)
    return bool(re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", current_problem_text))


def _missing_decision_variable_probe(
    graph: SemanticGraph,
    scenario: str,
    selected_action: str,
    source_testimonies: dict[str, str] | None,
    scenario_facts: dict | None,
) -> tuple[list[str], str, dict[str, object]]:
    graph_probe = _graph_first_latent_variable_probe(graph, selected_action)
    if graph_probe is not None:
        return graph_probe
    return _legacy_missing_decision_variable_probe(
        scenario, selected_action, source_testimonies, scenario_facts,
    )


def _problem_state_audit_probe(
    problem_state: dict[str, object] | None,
    selected_action: str,
    *,
    require_clause_grounding: bool = False,
) -> tuple[list[str], str, dict[str, object]]:
    """Select an audit target exclusively from the live ProblemState.

    Callers that manufacture an audit from aggregate signals rather than from
    an explicitly persistent issue require a clause-grounded target, so a
    question the graph cannot trace cannot become evidence that the agreement
    needs reviewing.
    """
    state = dict(problem_state or {})
    questions = [
        *(state.get("audit_candidates", []) or []),
        *(state.get("argument_challenge_candidates", []) or []),
    ]
    positions = {
        str(item.get("specialist", "")): item
        for item in state.get("agent_positions", []) or []
    }
    ranked: list[tuple[int, dict[str, object]]] = []
    for question in questions:
        if str(question.get("status", "")).upper() in {
            "RESOLVED", "REFINED", "SUPERSEDED", "WITHDRAWN",
        }:
            continue
        issue_id = str(
            question.get("issue_id", question.get("question_key", "")) or ""
        ).strip()
        # ProblemState audit candidates must carry their stable current-run
        # identity. Never stringify a missing key into an authoritative
        # "None" audit variable.
        if not issue_id.startswith(("QUESTION:", "CHALLENGE:")):
            continue
        grounded_in = list(question.get("grounded_in", []) or [])
        if not grounded_in:
            continue
        if (
            require_clause_grounding
            and question.get("grounding_status", "CLAUSE_GROUNDED") not in {
                "CLAUSE_GROUNDED", "PROPOSITION_GROUNDED",
            }
        ):
            continue
        proposition = " ".join(str(
            question.get("proposition", question.get("question", ""))
        ).split())
        category = str(question.get("category", "UNRESOLVED_QUESTION")).upper()
        raised_by = list(question.get("raised_by", []) or [])
        source = str(
            raised_by[0] if len(raised_by) == 1
            else question.get("about_specialist", "")
        )
        score = 1
        if re.search(
            r"\b(?:relative|compare|versus|outweigh|magnitude|utility|trade[- ]?off)\b",
            proposition, re.IGNORECASE,
        ):
            score += 8
        if positions.get(source, {}).get("choice_status") in {
            "UNDERDETERMINED", "CONDITIONAL",
        }:
            score += 4
        if question.get("status") == "PERSISTENT_UNRESOLVED":
            score += 4
        if category == "ACTION_SET_ADEQUACY":
            score += 6
        if question.get("grounding_status") == "UNGROUNDED":
            score -= 6
        score += int(float(question.get("priority", 0.0) or 0.0) * 10)
        ranked.append((score, question))
    if not ranked:
        return [], "", {}

    _score, chosen = max(
        ranked,
        key=lambda item: (item[0], str(
            item[1].get("issue_id", item[1].get("question_key", ""))
        )),
    )
    issue_id = str(chosen.get("issue_id", chosen.get("question_key", ""))).strip()
    proposition = " ".join(str(
        chosen.get("proposition", chosen.get("question", ""))
    ).split())
    category = str(chosen.get("category", "UNRESOLVED_QUESTION")).upper()
    raised_by = list(chosen.get("raised_by", []) or [])
    target_specialists = list(chosen.get("target_specialists", []) or [])
    persistent = chosen.get("status") == "PERSISTENT_UNRESOLVED"
    relation = (
        "COMPARATIVE_MAGNITUDE"
        if re.search(
            r"\b(?:relative|compare|versus|outweigh|magnitude|utility|trade[- ]?off)\b",
            proposition, re.IGNORECASE,
        )
        else category
    )
    status = "PERSISTENT_UNRESOLVED" if persistent else "UNRESOLVED"
    question = (
        f"Audit {issue_id}: {proposition} State whether resolving this issue would "
        f"leave, weaken, or reverse the recommendation for '{selected_action}'."
    )
    payload = {
        "issue_id": issue_id,
        "source": str(chosen.get("source", "problem_state.unresolved_questions")),
        "generated_by": str(chosen.get("generated_by", "")),
        "about_specialist": str(chosen.get("about_specialist", "")),
        "proposition": proposition,
        "grounded_in": list(chosen.get("grounded_in", []) or []),
        "grounding_status": str(chosen.get("grounding_status", "CLAUSE_GROUNDED")),
        "raised_by": raised_by,
        "target_framework": str(
            (target_specialists or [chosen.get("target_framework", "")])[0]
        ),
        "target_specialists": target_specialists,
        "challenge_kind": str(chosen.get(
            "challenge_kind", chosen.get("uncertainty_kind", "")
        )),
        "trigger_fields": list(chosen.get("trigger_fields", []) or []),
        "status": status,
        "entity": proposition,
        "relation": relation,
        "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
        "focus_action": selected_action,
        "question": question,
        "required_response": {
            "counterfactual_anchor": f"issue {issue_id}",
            "allowed_effects": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
        },
    }
    # grounding_status is workspace-internal ranking metadata. Sending it as
    # part of the admitted av object made every delegate fail schema checks
    # for "unsupported keys" when they echoed the authoritative variable.
    return [
        "problem_state_audit_candidate",
        f"problem_state_{relation.casefold()}",
        "persistent_unresolved" if persistent else "live_unresolved",
    ], question, payload


_DECISIVE_OUTCOME_MARKERS: dict[str, re.Pattern[str]] = {
    "DEATH_OR_SURVIVAL": re.compile(
        r"\b(?:death|die|dying|fatal|mortal|life[- ]saving|survival)\b", re.I,
    ),
    "CERTAIN_OUTCOME": re.compile(
        r"\b(?:certain(?:ly)?|guarantee[ds]?|inevitabl\w*|necessarily)\b", re.I,
    ),
}
_EFFECT_DOWNGRADE_MARKER = re.compile(
    r"\b(?:non[- ]vital|mere(?:ly)?\s+(?:benefit|beneficence|beneficent)|"
    r"optional\s+(?:benefit|welfare)|ordinary\s+beneficence)\b",
    re.I,
)


def _argument_challenge_candidates(
    candidates: Sequence[CandidateChunk],
    proposition_ledger: dict[str, object],
    graph: SemanticGraph,
    selected_action: str,
) -> list[dict[str, object]]:
    """Compile questions about inferential bridges without asserting answers.

    These are stress tests for an otherwise redundant recurrence. A challenge
    may increase attention to a premise, but does not change its epistemic
    status or the current policy by itself.
    """
    challenges: list[dict[str, object]] = []
    known_proposition_ids = set(proposition_ledger)
    proposition_ids_by_support: dict[str, list[str]] = {}
    for proposition_id, record in proposition_ledger.items():
        for support_id in getattr(record, "support_ids", []) or []:
            proposition_ids_by_support.setdefault(str(support_id), []).append(
                proposition_id
            )

    def add(
        specialist: str,
        kind: str,
        question: str,
        *,
        grounded_in: Sequence[str] = (),
        trigger_fields: Sequence[str] = (),
        priority: float = 0.5,
    ) -> None:
        cleaned = " ".join(str(question).split())[:240]
        if not cleaned:
            return
        supports: list[str] = []
        for item in grounded_in:
            reference = str(item).strip()
            if reference in known_proposition_ids:
                supports.append(reference)
            supports.extend(proposition_ids_by_support.get(reference, []))
        supports = list(dict.fromkeys(supports))[:6]
        challenge = ArgumentChallenge(
            challenge_kind=kind,
            question=cleaned,
            about_specialist=specialist,
            target_specialists=(specialist,),
            grounded_in=tuple(supports or [
                node.id for node in graph.nodes.values() if node.kind == "ACTION"
            ]),
            trigger_fields=tuple(trigger_fields),
            focus_action=selected_action,
            priority=priority,
            generated_by="WORKSPACE_ARGUMENT_AUDITOR",
            raised_by=(),
            grounding_status=(
                "PROPOSITION_GROUNDED" if supports else "UNGROUNDED"
            ),
        ).as_dict()
        if any(item.get("issue_id") == challenge["issue_id"] for item in challenges):
            return
        challenges.append(challenge)

    established_claims = {
        proposition_id: str(getattr(record, "claim", ""))
        for proposition_id, record in proposition_ledger.items()
        if str(getattr(record, "epistemic_status", "")).upper() == "ESTABLISHED"
    }
    world_effects = project_grounded_action_effects(graph)

    for candidate in candidates:
        if not candidate.schema_valid:
            continue
        cited = list(dict.fromkeys((
            *candidate.supporting_proposition_ids,
            *candidate.decision_critical_proposition_ids,
        )))
        cited_text = " ".join(established_claims.get(item, "") for item in cited)

        if candidate.specialist == "utilitarian":
            rows = [
                row
                for action_rows in candidate.utilitarian_consequence_table.values()
                for row in action_rows
                if isinstance(row, dict)
            ]
            material_rows = [
                row for row in rows
                if str(row.get("importance", "")).upper() in {"HIGH", "CRITICAL"}
            ]
            unknown_material = [
                row for row in material_rows
                if any(str(row.get(field, "UNKNOWN")).upper() == "UNKNOWN" for field in (
                    "probability", "magnitude", "duration",
                ))
            ]
            if candidate.utilitarian_decision_depends_on_unknown:
                missing = candidate.utilitarian_missing_comparison or (
                    "the probability, magnitude, duration, or relative importance of the competing effects"
                )
                add(
                    candidate.specialist,
                    "UTILITY_UNRESOLVED_COMPARISON",
                    f"Which grounded comparison would resolve the utilitarian ranking: {missing}?",
                    grounded_in=cited,
                    trigger_fields=("cd", "cm", "ct"),
                    priority=0.97,
                )
            elif unknown_material and candidate.comparison_complete:
                effect_names = list(dict.fromkeys(
                    str(row.get("outcome", row.get("effect_id", "effect")))
                    for row in unknown_material
                ))
                add(
                    candidate.specialist,
                    "UTILITY_UNCERTAINTY_TREATMENT",
                    "How can the ranking be complete while decision-relevant probability, magnitude, or duration remains unknown for "
                    + ", ".join(effect_names[:2]) + "?",
                    grounded_in=cited,
                    trigger_fields=("ct.*.wi", "ct.*.probability", "ct.*.magnitude", "ct.*.duration", "cc"),
                    priority=0.95,
                )
            if any(str(row.get("importance", "")).upper() == "UNKNOWN" for row in rows):
                add(
                    candidate.specialist,
                    "UTILITY_UNKNOWN_WEIGHT",
                    "Would resolving the UNKNOWN utilitarian importance assigned to a grounded effect weaken or reverse the current ranking?",
                    grounded_in=cited,
                    trigger_fields=("ct.*.wi",),
                    priority=0.90,
                )
            directions = {
                str(row.get("direction", "")).upper() for row in material_rows
                if str(row.get("direction", "")).upper() in {
                    "BENEFIT", "HARM", "OPPORTUNITY_COST",
                }
            }
            scopes = {
                " ".join(str(row.get("scope", "")).casefold().split())
                for row in material_rows if str(row.get("scope", "")).strip()
            }
            grounded_ev = bool(candidate.expected_value_estimates) and all(
                bool(item.get("grounded"))
                for item in candidate.expected_value_estimates.values()
                if isinstance(item, dict)
            )
            if len(material_rows) >= 2 and (len(directions) > 1 or len(scopes) > 1) and not grounded_ev:
                add(
                    candidate.specialist,
                    "UTILITY_COMMENSURATION",
                    "What common comparison rule combines the material effects across different directions, affected scopes, and time horizons without treating salience as weight?",
                    grounded_in=cited,
                    trigger_fields=("ct", "dr", "ev"),
                    priority=0.89,
                )
            if any("valuations must reference every grounded effect" in error for error in candidate.framework_validation_errors):
                add(
                    candidate.specialist,
                    "UTILITY_EFFECT_COVERAGE",
                    "Which grounded action effect is missing from or duplicated in the utilitarian valuation ledger?",
                    grounded_in=cited,
                    trigger_fields=("ct.*.eid",),
                    priority=0.96,
                )

        if candidate.specialist == "deontological":
            assessments = list(
                (candidate.deontological_ledger_proposal or {}).get("assessments", [])
            )
            for item in assessments:
                if not isinstance(item, dict) or item.get("resolution_status") != "RESOLVED":
                    continue
                decisive_text = " ".join(str(item.get(key, "")) for key in (
                    "norm", "priority_rule", "public_justification", "reason",
                ))
                for category, marker in _DECISIVE_OUTCOME_MARKERS.items():
                    if marker.search(decisive_text) and not marker.search(cited_text):
                        add(
                            candidate.specialist,
                            "EPISTEMIC_STRENGTH_BRIDGE",
                            f"Do the established effects justify the stronger {category.lower().replace('_', ' ')} premise used to classify {item.get('norm', 'the duty')}, or is that premise unestablished?",
                            grounded_in=cited,
                            trigger_fields=("dp.*.n", "dp.*.pr", "dp.*.pj", "dp.*.rs"),
                            priority=0.98,
                        )
                if _EFFECT_DOWNGRADE_MARKER.search(decisive_text):
                    add(
                        candidate.specialist,
                        "EFFECT_MORAL_CLASSIFICATION",
                        "What grounded feature makes the competing effect non-vital or merely beneficent rather than a basic-interest claim over the relevant time horizon?",
                        grounded_in=cited,
                        trigger_fields=("dp.*.cn", "dp.*.crs", "dp.*.pr"),
                        priority=0.93,
                    )
                if (
                    item.get("duty_type") == "PERFECT_POSITIVE"
                    and item.get("special_obligation_status") != "ESTABLISHED"
                ):
                    add(
                        candidate.specialist,
                        "DUTY_PERFECTION_BASIS",
                        f"What makes {item.get('norm', 'the positive duty')} perfect rather than imperfect when no special obligation is established?",
                        grounded_in=cited,
                        trigger_fields=("dp.*.dt", "dp.*.so", "dp.*.sob"),
                        priority=0.94,
                    )
                if omission_classified_as_perfect_negative_violation(item):
                    add(
                        candidate.specialist,
                        "DOING_ALLOWING_CLASSIFICATION",
                        f"Does the classified omission under {item.get('action_id', 'this action')} violate a perfect negative duty, or does that conclusion require a separate right-correlative premise?",
                        grounded_in=cited,
                        trigger_fields=("dp.*.dt", "dp.*.hr", "dp.*.pb"),
                        priority=0.96,
                    )
                elif harm_relation_conflicts_with_graph(graph, item):
                    add(
                        candidate.specialist,
                        "DOING_ALLOWING_CLASSIFICATION",
                        f"Does the harm under {item.get('action_id', 'this action')} follow the admitted graph's causal path, or is doing-versus-allowing being relabeled to fit the verdict?",
                        grounded_in=cited,
                        trigger_fields=("dp.*.dt", "dp.*.hr", "dp.*.pb"),
                        priority=0.96,
                    )
                if item.get("means_relation") == "INTENDED_AS_MEANS":
                    add(
                        candidate.specialist,
                        "MEANS_CAUSAL_PATH",
                        f"Is the burden under {item.get('action_id', 'this action')} causally used to produce the chosen end, or is it a foreseen or counterfactual side effect?",
                        grounded_in=cited,
                        trigger_fields=("dp.*.mr",),
                        priority=0.92,
                    )

        if candidate.specialist == "rawlsian":
            proposal = candidate.rawls_position_proposal or {}
            positions = [
                item for item in proposal.get("positions", [])
                if isinstance(item, dict)
            ]
            represented = " ".join(str(item.get("subject", "")) for item in positions)
            generic_subject_words = {
                "people", "person", "persons", "group", "groups", "resident",
                "residents", "individual", "individuals", "population",
                "populations", "community", "communities", "stakeholder",
                "stakeholders",
            }
            represented_words = {
                token for token in re.findall(r"[a-z0-9]+", represented.casefold())
                if token not in generic_subject_words
            }
            omitted: list[str] = []
            omitted_support: list[str] = []
            for effect in world_effects:
                subject = " ".join(str(effect.affected_subject).split())
                words = {
                    token for token in re.findall(r"[a-z0-9]+", subject.casefold())
                    if len(token) >= 4 and token not in generic_subject_words
                }
                if subject and words and not words & represented_words:
                    omitted.append(subject)
                    omitted_support.append(
                        f"PROP:WORLD:{effect.effect_id}"
                        if str(effect.effect_id).startswith("E") else effect.consequence_id
                    )
            omitted = list(dict.fromkeys(omitted))
            if omitted:
                add(
                    candidate.specialist,
                    "POSITION_COVERAGE",
                    "Does the Rawlsian comparison remain complete after representing the materially affected position of "
                    + ", ".join(omitted[:2]) + " over the same relevant horizon?",
                    grounded_in=omitted_support,
                    trigger_fields=("rp.*.s", "rp.*.d", "rp.*.ca"),
                    priority=0.90,
                )
            dimensions = {str(item.get("dimension", "")) for item in positions}
            ranking = str(proposal.get("ranking_basis", ""))
            if ranking == "MAXIMIN_PRIMARY_GOODS" and dimensions == {"BASIC_INTEREST_SECURITY"}:
                add(
                    candidate.specialist,
                    "RAWLS_RANKING_STAGE",
                    "Does this concern invoke maximin over primary goods, basic-interest security, or an original-position public-rule choice, and what establishes that priority?",
                    grounded_in=cited,
                    trigger_fields=("rb", "rbc", "lpj", "rp.*.d"),
                    priority=0.91,
                )

        if candidate.specialist == "virtue":
            proposal = candidate.virtue_character_proposal or {}
            assessments = [
                item for item in proposal.get("assessments", [])
                if isinstance(item, dict)
            ]
            ranking = str(proposal.get("ranking_basis", ""))
            if ranking == "UNRESOLVED" and candidate.comparison_complete:
                add(
                    candidate.specialist,
                    "VIRTUE_RANKING_GAP",
                    "What practical-wisdom, role-fidelity, flourishing, or exemplar principle resolves the competing virtues rather than merely selecting an action?",
                    grounded_in=cited,
                    trigger_fields=("vb", "cc"),
                    priority=0.96,
                )
            unsettled = [
                item for item in assessments
                if item.get("verdict") in {"MIXED", "UNCERTAIN"}
            ]
            if unsettled and candidate.selection_status == "SELECTED":
                add(
                    candidate.specialist,
                    "VIRTUE_CONFLICT_RESOLUTION",
                    "Which circumstance-sensitive practical-wisdom judgment resolves the recorded mixed or uncertain virtues strongly enough to support a settled selection?",
                    grounded_in=cited,
                    trigger_fields=("vl.*.v", "vl.*.vs", "vl.*.x", "vl.*.c", "ss"),
                    priority=0.94,
                )
            roles = {
                " ".join(str(item.get("actor_role", "")).casefold().split())
                for item in assessments if str(item.get("actor_role", "")).strip()
            }
            if len(roles) > 1:
                add(
                    candidate.specialist,
                    "VIRTUE_ROLE_CONSISTENCY",
                    "Is the same decision-maker legitimately acting under different roles across the options, or has role redescription changed the comparison?",
                    grounded_in=cited,
                    trigger_fields=("vl.*.r",),
                    priority=0.91,
                )
            effects_by_action = {
                effect.action_id for effect in world_effects
            }
            unsupported_scenario_rows = [
                item for item in assessments
                if item.get("evidence_basis") in {"ACTION_GRAPH", "SCENARIO"}
                and str(item.get("action_id", "")) not in effects_by_action
            ]
            if unsupported_scenario_rows:
                add(
                    candidate.specialist,
                    "VIRTUE_CIRCUMSTANCE_GROUNDING",
                    "Which grounded action effect supports the circumstance used to infer the claimed virtue or vice?",
                    grounded_in=cited,
                    trigger_fields=("vl.*.c", "vl.*.g", "vl.*.rs"),
                    priority=0.93,
                )
            virtue_text = " ".join(
                str(item.get(key, ""))
                for item in assessments
                for key in ("virtues", "vice_risk", "circumstance", "reason")
            ) + " " + candidate.decision_rule + " " + candidate.rationale
            for category, marker in _DECISIVE_OUTCOME_MARKERS.items():
                if marker.search(virtue_text) and not marker.search(cited_text):
                    add(
                        candidate.specialist,
                        "VIRTUE_EPISTEMIC_STRENGTH_BRIDGE",
                        f"Does grounded evidence establish the stronger {category.lower().replace('_', ' ')} premise used in the character judgment, or should the virtue assessment remain conditional?",
                        grounded_in=cited,
                        trigger_fields=("vl.*.c", "vl.*.rs"),
                        priority=0.98,
                    )
        if candidate.specialist == "care" and candidate.care_ledger_proposal:
            proposal = candidate.care_ledger_proposal or {}
            assessments = [
                item for item in proposal.get("assessments", [])
                if isinstance(item, dict)
            ]
            ranking = str(proposal.get("ranking_basis", ""))
            if ranking == "UNRESOLVED" and candidate.comparison_complete:
                add(
                    candidate.specialist, "CARE_RANKING_GAP",
                    "Which relational basis resolves the competing Care claims rather than merely selecting one dependent party?",
                    grounded_in=cited,
                    trigger_fields=("cb", "cl.*.cc", "cl.*.res", "cc"),
                    priority=0.97,
                )
            if any(
                item.get("relationship_type") == "UNRESOLVED"
                or item.get("need_kind") == "UNRESOLVED"
                for item in assessments
            ) and candidate.selection_status == "SELECTED":
                add(
                    candidate.specialist, "CARE_RELATIONAL_GROUNDING",
                    "What establishes the dependency, relationship, and responsibility used to give one Care claim priority?",
                    grounded_in=cited,
                    trigger_fields=("cl.*.rt", "cl.*.ds", "cl.*.rb", "cl.*.nk", "ss"),
                    priority=0.96,
                )
            if any(
                item.get("feasibility") in {"CONDITIONAL", "UNKNOWN"}
                or item.get("responsiveness") == "UNKNOWN"
                for item in assessments
            ) and candidate.evidence_sufficient_for_action:
                add(
                    candidate.specialist, "CARE_RESPONSIVENESS_FEASIBILITY",
                    "Can the proposed response be treated as ethically responsive before its practical feasibility and mode of response are established?",
                    grounded_in=cited,
                    trigger_fields=("cl.*.rr", "cl.*.f", "esa"),
                    priority=0.95,
                )
            if ranking == "ENTRUSTED_RESPONSIBILITY" and not any(
                item.get("relationship_type") in {"ENTRUSTED", "CARE_ROLE"}
                for item in assessments
            ):
                add(
                    candidate.specialist, "CARE_ENTRUSTMENT_BASIS",
                    "What action-grounded entrustment or care role supports using entrusted responsibility as the governing basis?",
                    grounded_in=cited,
                    trigger_fields=("cb", "cl.*.rt", "cl.*.ds", "cl.*.rb"),
                    priority=0.94,
                )
            if ranking == "ACUTE_DEPENDENCY" and not any(
                item.get("need_urgency") == "IMMEDIATE" for item in assessments
            ):
                add(
                    candidate.specialist, "CARE_URGENCY_BASIS",
                    "What establishes acute dependency when no affected party is classified as having an immediate need?",
                    grounded_in=cited,
                    trigger_fields=("cb", "cl.*.nk", "cl.*.nu"),
                    priority=0.94,
                )
            represented = " ".join(
                str(item.get("affected_party", "")) for item in assessments
            ).casefold()
            omitted_parties = list(dict.fromkeys(
                effect.affected_subject for effect in world_effects
                if effect.affected_subject and not any(
                    token in represented
                    for token in re.findall(
                        r"[a-z0-9]+", effect.affected_subject.casefold()
                    )
                    if len(token) >= 5 and token not in {
                        "people", "person", "persons", "group", "groups",
                        "resident", "residents", "population", "community",
                    }
                )
            ))
            if omitted_parties:
                add(
                    candidate.specialist, "CARE_CLAIM_COVERAGE",
                    "Does the Care comparison remain complete after representing the dependency claim of "
                    + ", ".join(omitted_parties[:2]) + "?",
                    grounded_in=[
                        effect.consequence_id for effect in world_effects
                        if effect.affected_subject in omitted_parties
                    ],
                    trigger_fields=("cl.*.p", "cl.*.cc"),
                    priority=0.92,
                )
            care_text = " ".join(
                str(item.get(key, ""))
                for item in assessments
                for key in (
                    "dependency_source", "responsibility_basis", "need_kind",
                    "need_urgency", "reason",
                )
            ) + " " + candidate.decision_rule + " " + candidate.rationale
            for category, marker in _DECISIVE_OUTCOME_MARKERS.items():
                if marker.search(care_text) and not marker.search(cited_text):
                    add(
                        candidate.specialist, "CARE_EPISTEMIC_STRENGTH_BRIDGE",
                        f"Does grounded evidence establish the stronger {category.lower().replace('_', ' ')} premise used to rank the Care claims, or should urgency remain conditional?",
                        grounded_in=cited,
                        trigger_fields=("cl.*.nk", "cl.*.nu", "cl.*.rs"),
                        priority=0.98,
                    )

        for threshold, kind in (
            (candidate.factual_reversal_threshold, "FACTUAL_DEFEATER"),
            (candidate.normative_reversal_threshold, "NORMATIVE_DEFEATER"),
        ):
            if threshold and threshold.strip().upper() != "NONE":
                add(
                    candidate.specialist,
                    kind,
                    f"Would this stated defeat condition weaken or reverse the current argument: {threshold}?",
                    grounded_in=cited,
                    trigger_fields=(
                        "ft" if kind == "FACTUAL_DEFEATER" else "nt",
                    ),
                    priority=0.55,
                )

    settled = set(settled_question_keys(graph))
    return sorted(
        [item for item in challenges if str(item.get("issue_id", "")) not in settled],
        key=lambda item: (-float(item.get("priority", 0.0)), str(item.get("issue_id", ""))),
    )[:12]


_CHALLENGE_AGENDA_PROTECTED_CONSTRAINTS = {
    "VISIBILITY_AUDIT", "AUTONOMY_AUDIT", "COERCION_AUDIT",
    "CONSENSUS_AUDIT", "PROBLEM_STATE_AUDIT", "REVERSAL_AUDIT",
    "CONTINGENCY_REVIEW", "PROBLEM_REFORMULATION", "PROPOSAL_REVIEW",
    "PLANNING_REVIEW", "SYNTHESIS_REVIEW",
}
_ACTIVE_CHALLENGE_STATUSES = {
    "UNTESTED", "ASSIGNED", "UNANSWERED", "UNRESOLVED",
}


def _graph_party_labels(graph: SemanticGraph) -> list[str]:
    return list(dict.fromkeys(
        node.label
        for node in graph.nodes.values()
        if node.kind == "TARGET" and str(node.label or "").strip()
    ))


def _content_token_overlap(left: str, right: str) -> bool:
    ignored = {"the", "and", "for", "with", "from", "that", "this", "group"}
    left_words = {
        word for word in re.findall(r"[a-z0-9]+", str(left or "").casefold())
        if len(word) > 3 and word not in ignored
    }
    right_words = {
        word for word in re.findall(r"[a-z0-9]+", str(right or "").casefold())
        if len(word) > 3 and word not in ignored
    }
    return bool(left_words and right_words and (left_words & right_words))


def _graph_parties_mentioned(text: str, graph: SemanticGraph) -> list[str]:
    blob = str(text or "")
    return [
        label for label in _graph_party_labels(graph)
        if _content_token_overlap(label, blob)
    ]


def _world_effect_rows(graph: SemanticGraph) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    incoming: dict[str, str] = {}
    for edge in graph.edges:
        if edge.relation != "HAS_CONSEQUENCE":
            continue
        source = graph.nodes.get(edge.source)
        if source is None or source.kind != "ACTION":
            continue
        incoming[edge.target] = str(source.attributes.get("canonical_action_id", source.id))
    for node in graph.nodes.values():
        if node.kind != "CONSEQUENCE" or node.attributes.get("framework"):
            continue
        if not str(node.attributes.get("world_effect_id", "")).strip():
            continue
        parties = [
            graph.nodes[edge.target].label
            for edge in graph.outgoing(node.id, "AFFECTS")
            if edge.target in graph.nodes
        ]
        quantities = []
        for value in (
            list(node.attributes.get("quantities", []) or [])
            + list(node.attributes.get("party_quantities", []) or [])
        ):
            if isinstance(value, dict):
                text = str(value.get("raw") or value.get("canonical") or "").strip()
            else:
                text = str(value).strip()
            if text:
                quantities.append(text)
        rows.append({
            "action_id": incoming.get(node.id, ""),
            "outcome": node.label,
            "parties": parties,
            "polarity": str(node.attributes.get("polarity", "")).upper(),
            "directness": str(node.attributes.get("directness", "")).upper(),
            "quantities": quantities,
        })
    return rows


def _answer_transfers_unshared_world_effect(
    answer: str, graph: SemanticGraph,
) -> str:
    """Reject cross-action copies and mortality conversions of admitted effects."""
    from .epistemic_ledger import _quantity_keys, _text_has_mortality

    blob = str(answer or "")
    if not blob.strip():
        return ""
    rows = _world_effect_rows(graph)
    folded = blob.casefold()
    both = bool(re.search(
        r"\bboth\s+(?:plans|actions|options)\b|"
        r"\bunder\s+both\b|"
        r"\bequal\b[^.]{0,60}\bunder\s+both\b",
        folded,
    ))
    actual = [
        row for row in rows
        if str(row.get("polarity") or "").upper() != "FOREGONE"
        and str(row.get("directness") or "").upper() != "FOREGONE"
    ]
    if both:
        by_party: dict[str, dict[str, list[str]]] = {}
        for row in actual:
            for party in row.get("parties") or []:
                key = " ".join(str(party).casefold().split())
                if not key:
                    continue
                by_party.setdefault(key, {}).setdefault(
                    str(row.get("action_id") or ""), []
                ).append(str(row.get("outcome") or ""))
        for party, actions in by_party.items():
            if not _content_token_overlap(party, blob):
                continue
            if len(actions) > 1:
                continue
            outcomes = next(iter(actions.values()), [])
            if any(_content_token_overlap(outcome, blob) for outcome in outcomes if outcome):
                return (
                    "answer transfers an action-scoped world effect onto a plan "
                    "that does not admit it"
                )
    if _text_has_mortality(blob):
        answer_quantities = _quantity_keys(blob)
        mentioned_actions = {
            match.group(0).upper()
            for match in re.finditer(r"\bA\d+\b", blob, re.IGNORECASE)
        }
        for row in actual:
            action_id = str(row.get("action_id") or "").upper()
            if mentioned_actions and action_id and action_id not in mentioned_actions:
                continue
            if _text_has_mortality(str(row.get("outcome") or "")):
                continue
            parties = [str(value) for value in row.get("parties") or []]
            if not any(_content_token_overlap(party, blob) for party in parties):
                continue
            record_quantities = _quantity_keys(" ".join(
                [str(row.get("outcome") or ""), *list(row.get("quantities") or [])]
            ))
            if answer_quantities and record_quantities and (
                {value for value, _flag in answer_quantities}
                & {value for value, _flag in record_quantities}
            ):
                return (
                    "answer converts an admitted non-mortality outcome into deaths"
                )
    return ""


def _challenge_resolution_supported(
    candidate: CandidateChunk,
    challenge: dict[str, object],
    response: dict[str, object],
    graph: SemanticGraph | None = None,
) -> tuple[bool, str]:
    """Verify a self-reported resolution against the operative native ledger."""
    kind = str(challenge.get("challenge_kind", "")).upper()
    current_effect = str(
        response.get("current_position_effect", response.get("effect", "UNRESOLVED"))
    ).upper()
    boundary_effect = str(response.get("boundary_effect", "NOT_APPLICABLE")).upper()

    # Defeater questions ask for a counterfactual decision boundary, not an
    # immediate change of view. A typed answer can settle that question without
    # mutating the current framework ledger.
    if kind in {"FACTUAL_DEFEATER", "NORMATIVE_DEFEATER"}:
        if boundary_effect in {"NO_SWITCH", "MAY_SWITCH", "SWITCHES"}:
            return True, "counterfactual boundary was explicitly classified"
        return False, "counterfactual boundary remains unclassified"

    native = dict(candidate.committed_native_ledger or {})
    records = [
        dict(item) for item in native.get("records", []) or []
        if isinstance(item, dict)
    ]
    if not records:
        return False, "no operative graph-committed framework ledger is available"

    action_ids = {
        str(item.get("canonical_action_id", "")) for item in records
        if str(item.get("canonical_action_id", ""))
    }
    if kind == "UTILITY_UNRESOLVED_COMPARISON":
        resolved = bool(
            not candidate.utilitarian_decision_depends_on_unknown
            and candidate.comparison_complete
            and candidate.evidence_sufficient_for_action
        )
        return (
            resolved,
            "utilitarian comparison is complete"
            if resolved else "the operative utilitarian comparison remains underdetermined",
        )
    if kind == "DUTY_PERFECTION_BASIS":
        relevant = [
            item for item in records
            if str(item.get("duty_type", "")).startswith("PERFECT_POSITIVE")
            or str(item.get("duty_type", "")) == "RIGHT_CORRELATIVE"
        ]
        unsupported = any(
            "perfect positive duty lacks" in error.casefold()
            for item in relevant
            for error in active_calibration_errors(item)
        )
        supported = bool(relevant and not unsupported)
        return (
            supported,
            "perfect-duty basis survives ledger calibration"
            if supported else "perfect-duty basis remains unestablished after calibration",
        )
    if kind == "DOING_ALLOWING_CLASSIFICATION":
        still_claimed = any(
            omission_classified_as_perfect_negative_violation(item)
            for item in records
        )
        still_errored = any(
            "perfect negative-duty violation" in error.casefold()
            or "agent-caused settled welfare harm" in error.casefold()
            for item in records
            for error in active_calibration_errors(item)
        )
        still_misaligned = any(
            harm_relation_conflicts_with_graph(graph, item)
            for item in records
        ) if graph is not None else False
        resolved = (
            bool(records)
            and not still_claimed
            and not still_errored
            and not still_misaligned
        )
        return (
            resolved,
            "omission is not committed as a perfect negative-duty violation"
            if resolved else
            "the operative ledger still treats the omission as a perfect negative-duty violation"
            if still_claimed or still_errored else
            "the operative ledger still relabels graph-supported allowing as doing",
        )
    if kind == "MEANS_CAUSAL_PATH":
        unsupported = any(
            "causal path" in error.casefold()
            for item in records
            for error in active_calibration_errors(item)
        )
        return (
            not unsupported,
            "means relation survives causal-path calibration"
            if not unsupported else "means classification still lacks a causal path",
        )
    if kind == "CARE_CLAIM_COVERAGE":
        complete = bool(
            len(action_ids) >= 2
            and all(
                str(item.get("affected_party", "")).strip()
                and str(item.get("relationship_type", "")).strip()
                and str(item.get("dependency_source", "")).strip()
                and str(item.get("competing_care_claim", "")).strip()
                for item in records
            )
        )
        return (
            complete,
            "both actions retain parties, dependencies, and competing Care claims"
            if complete else "the committed Care comparison still omits a relational claim",
        )
    if kind == "POSITION_COVERAGE":
        complete = bool(
            len(action_ids) >= 2
            and all(
                str(item.get("subject", item.get("affected_subject", ""))).strip()
                and str(item.get("dimension", "")).strip()
                and str(item.get("institutional_relation", "")).strip()
                for item in records
            )
        )
        if not complete:
            return (
                False,
                "the committed Rawlsian position comparison is incomplete",
            )
        if graph is not None:
            named = _graph_parties_mentioned(
                str(challenge.get("question") or challenge.get("proposition") or ""),
                graph,
            )
            subjects = [
                str(item.get("subject") or item.get("affected_subject") or "")
                for item in records
            ]
            missing = [
                party for party in named
                if not any(_content_token_overlap(party, subject) for subject in subjects)
            ]
            if missing:
                return (
                    False,
                    "committed Rawlsian positions omit named parties: "
                    + ", ".join(missing[:3]),
                )
            transfer = _answer_transfers_unshared_world_effect(
                str(response.get("answer") or ""), graph,
            )
            if transfer:
                return False, transfer
        return True, "committed Rawlsian positions cover both actions"

    rejected = candidate.framework_retention_status in {
        "UPDATE_REJECTED", "PRESERVED_AFTER_REJECTED_UPDATE",
    }
    if rejected:
        return False, "the proposed framework update was rejected"
    if current_effect == "UNRESOLVED":
        return False, "the answer reports an unresolved effect"
    return True, "answer is backed by an operative graph-committed ledger"


def _verification_follow_up(
    candidate: CandidateChunk,
    challenge: dict[str, object],
    reason: str,
) -> dict[str, object]:
    diagnostic = next((
        " ".join(str(value).split())
        for value in candidate.framework_validation_errors
        if " ".join(str(value).split())
    ), reason)
    question = (
        "What additional framework-specific ground would resolve this validation "
        f"result: {diagnostic}"
    )[:240]
    return ArgumentChallenge(
        challenge_kind="VALIDATION_FOLLOW_UP",
        question=question,
        about_specialist=candidate.specialist,
        target_specialists=(candidate.specialist,),
        grounded_in=tuple(challenge.get("grounded_in", []) or []),
        trigger_fields=tuple(challenge.get("trigger_fields", []) or []),
        focus_action=str(challenge.get("focus_action", "")),
        priority=max(0.6, float(challenge.get("priority", 0.5)) - 0.01),
        generated_by="FRAMEWORK_LEDGER_VALIDATOR",
        raised_by=(candidate.specialist,),
        grounding_status=str(challenge.get("grounding_status", "UNGROUNDED")),
    ).as_dict()


def _render_deliberation_progress(
    progress: Callable[[str], None] | None,
    *,
    cycle: int,
    actions: Sequence[str],
    candidates: Sequence[CandidateChunk],
    received_challenges: Sequence[dict[str, object]],
    next_agenda: Sequence[dict[str, object]],
) -> None:
    """Render new deliberative events; collapse unchanged framework state."""
    if progress is None:
        return
    action_ids = {action: f"A{index}" for index, action in enumerate(actions)}
    questions = {
        str(item.get("issue_id", "")): " ".join(
            str(item.get("question", item.get("proposition", ""))).split()
        )
        for item in received_challenges
    }
    progress(f"  Deliberation updates — cycle {cycle}")

    def native_lines(candidate: CandidateChunk) -> list[str]:
        records = [
            dict(item)
            for item in (candidate.committed_native_ledger or {}).get("records", []) or []
            if isinstance(item, dict)
        ]
        if candidate.specialist == "utilitarian":
            material = [
                item for item in records
                if str(item.get("importance", "")).upper() in {"HIGH", "CRITICAL"}
            ]
            by_action: dict[str, list[dict[str, object]]] = {}
            for item in material:
                by_action.setdefault(str(item.get("canonical_action_id", "?")), []).append(item)
            return [
                f"{action_id}: " + "; ".join(
                    f"{item.get('outcome', 'effect')} [{item.get('direction', 'UNKNOWN')}, "
                    f"{item.get('probability', 'UNKNOWN')}] — {item.get('valuation_reason', '')}"
                    for item in values[:3]
                )
                for action_id, values in sorted(by_action.items())
            ]
        if candidate.specialist == "deontological":
            return [
                f"{item.get('canonical_action_id', '?')}: {item.get('verdict', 'UNKNOWN')} — "
                f"{item.get('norm', 'unnamed norm')}; duty={item.get('duty_type', 'UNRESOLVED')}; "
                f"relation={item.get('relation', 'UNCERTAIN')}; {item.get('reason', '')}"
                for item in records
            ]
        if candidate.specialist == "virtue":
            return [
                f"{item.get('canonical_action_id', '?')}: {item.get('verdict', 'UNKNOWN')} — "
                f"virtues={item.get('virtues', '')}; vice risk={item.get('vice_risk', '')}; "
                f"judgment={item.get('reason', '')}"
                for item in records
            ]
        if candidate.specialist == "care":
            return [
                f"{item.get('canonical_action_id', '?')}: {item.get('verdict', 'UNKNOWN')} — "
                f"{item.get('affected_party', '')}; relation={item.get('relationship_type', '')}; "
                f"dependency={item.get('dependency_source', '')}; {item.get('reason', '')}"
                for item in records
            ]
        if candidate.specialist == "rawlsian":
            return [
                f"{item.get('canonical_action_id', '?')}: {item.get('effect', 'UNCERTAIN')} — "
                f"subject={item.get('subject', '')}; dimension={item.get('dimension', '')}; "
                f"relation={item.get('institutional_relation', '')}; {item.get('reason', '')}"
                for item in records
            ]
        return []

    for candidate in candidates:
        action_id = action_ids.get(candidate.recommended_action, "?")
        response = dict(candidate.challenge_response or {})
        if not response:
            reason = " ".join(
                value for value in (candidate.decision_rule, candidate.rationale)
                if value and value.casefold() != "none"
            )
            progress(
                f"    {candidate.specialist}: {action_id} "
                f"[{candidate.adjudication_status}] — {reason or 'no reason supplied'}"
            )
            for line in native_lines(candidate):
                progress(f"      Framework analysis: {line}")
            continue
        issue_id = str(response.get("issue_id", ""))
        progress(
            f"    {candidate.specialist}: {action_id} "
            f"[{candidate.adjudication_status}]"
        )
        progress(f"      Question: {questions.get(issue_id, issue_id)}")
        answer_lines = textwrap.wrap(str(response.get("answer", "")), width=100)
        for line_index, line in enumerate(answer_lines):
            progress(
                f"      Answer: {line}"
                if line_index == 0 else f"              {line}"
            )
        progress(
            "      Outcome: "
            f"{response.get('operative_disposition', response.get('disposition', 'UNRESOLVED'))}; "
            f"current={response.get('current_position_effect', response.get('effect', 'UNRESOLVED'))}; "
            f"boundary={response.get('boundary_effect', 'NOT_APPLICABLE')}; "
            f"verification={response.get('verification_status', 'NOT_VERIFIED')}"
        )
        verification_reason = str(response.get("verification_reason", ""))
        if verification_reason:
            progress(f"      Ledger: {verification_reason}")
        if candidate.framework_retention_status in {
            "UPDATE_REJECTED", "PRESERVED_AFTER_REJECTED_UPDATE",
        }:
            progress("      Operative framework state: unchanged from previous cycle")
        else:
            reason = " ".join(
                value for value in (candidate.decision_rule, candidate.rationale)
                if value and value.casefold() != "none"
            )
            progress(f"      Position rationale: {reason or 'no reason supplied'}")
            for line in native_lines(candidate):
                progress(f"      Framework analysis: {line}")
    if next_agenda:
        progress("  Next-cycle agenda")
        for item in next_agenda:
            targets = ",".join(str(value) for value in item.get("target_specialists", []))
            progress(
                f"    {targets or 'unassigned'}: "
                + " ".join(str(item.get("question", "")).split())
            )


def _apply_verified_resolution_to_native_ledgers(
    retained_challenges: Sequence[dict[str, object]],
    candidates: Sequence[CandidateChunk],
    graph_store: SemanticGraphStore,
    result: WorkspaceResult | None = None,
) -> None:
    """A verified repair must supersede the matching complaint on the ledger."""
    kinds = [
        kind
        for item in retained_challenges
        if str(
            (dict(item.get("last_response") or {})).get("verification_status", "")
        ).upper() == "VERIFIED_RESOLVED"
        for kind in calibration_issue_kinds_for_challenge(
            str(item.get("challenge_kind", ""))
        )
    ]
    kinds = [kind for kind in dict.fromkeys(kinds) if kind]
    if not kinds:
        return
    if not supersede_calibration_issues(graph_store.graph, kinds=kinds):
        return
    records = committed_deontological_assessments(graph_store.graph)
    if result is not None:
        result.deontological_duty_ledger = records
    for candidate in candidates:
        native = candidate.committed_native_ledger
        if (
            candidate.specialist == "deontological"
            and isinstance(native, dict)
            and str(native.get("ledger_kind", "")) == "DEONTOLOGICAL_DUTY_LEDGER"
        ):
            native["records"] = copy.deepcopy(records)


def _advance_argument_challenge_agenda(
    *,
    previous_challenges: Sequence[dict[str, object]],
    generated_challenges: Sequence[dict[str, object]],
    candidates: Sequence[CandidateChunk],
    next_cycle: int,
    next_constraint: str,
    active_specialists: Sequence[str],
    settled_issue_ids: Sequence[str] = (),
    graph: SemanticGraph | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Retain challenge history and assign one live question per framework."""
    settled = {str(value) for value in settled_issue_ids}
    retained: dict[str, dict[str, object]] = {}
    for raw in previous_challenges:
        item = copy.deepcopy(dict(raw))
        issue_id = str(item.get("issue_id", ""))
        if not issue_id.startswith("CHALLENGE:"):
            continue
        if issue_id in settled:
            item["status"] = "RESOLVED"
        if str(item.get("status", "")).upper() == "ASSIGNED":
            item["status"] = "UNANSWERED"
        retained[issue_id] = item

    followups: list[dict[str, object]] = []
    for candidate in candidates:
        response = dict(getattr(candidate, "challenge_response", {}) or {})
        issue_id = str(response.get("issue_id", ""))
        if issue_id not in retained:
            continue
        item = retained[issue_id]
        targets = {
            str(value).casefold()
            for value in item.get("target_specialists", []) or []
        }
        if candidate.specialist.casefold() not in targets:
            continue
        reported_disposition = str(
            response.get("disposition", "UNRESOLVED")
        ).upper()
        if reported_disposition not in {"RESOLVED", "REFINED", "UNRESOLVED"}:
            reported_disposition = "UNRESOLVED"
        reported_current_effect = str(
            response.get("current_position_effect", response.get("effect", "UNRESOLVED"))
        ).upper()
        current_effect = reported_current_effect
        boundary_effect = str(response.get("boundary_effect", "")).upper()
        if not boundary_effect:
            # Backward-compatible interpretation for older saved responses.
            if str(item.get("challenge_kind", "")).upper() in {
                "FACTUAL_DEFEATER", "NORMATIVE_DEFEATER",
            }:
                boundary_effect = {
                    "REVERSES": "SWITCHES",
                    "WEAKENS": "MAY_SWITCH",
                    "NO_CHANGE": "NO_SWITCH",
                }.get(current_effect, "UNRESOLVED")
                current_effect = "NO_CHANGE" if boundary_effect != "UNRESOLVED" else "UNRESOLVED"
            else:
                boundary_effect = "NOT_APPLICABLE"
        # A current reversal is an observable state transition, not a label the
        # specialist may assign to a merely hypothetical switch condition.
        if candidate.position_changed:
            current_effect = "REVERSES"
        elif current_effect == "REVERSES":
            current_effect = "NO_CHANGE"

        verified, verification_reason = _challenge_resolution_supported(
            candidate, item, {
                **response,
                "current_position_effect": current_effect,
                "boundary_effect": boundary_effect,
            },
            graph=graph,
        )
        operative_disposition = reported_disposition
        verification_status = "ANSWER_RECORDED"
        if reported_disposition == "RESOLVED":
            if verified:
                verification_status = "VERIFIED_RESOLVED"
            else:
                # The attempted answer remains in history, while a validator-
                # authored narrower question becomes the live continuation.
                operative_disposition = "REFINED"
                verification_status = "RESOLUTION_REJECTED"
                followups.append(_verification_follow_up(
                    candidate, item, verification_reason,
                ))
        elif reported_disposition == "REFINED":
            verification_status = "VERIFIED_REFINED"
        else:
            verification_status = "VERIFIED_UNRESOLVED"

        response.update({
            "reported_disposition": reported_disposition,
            "reported_current_position_effect": reported_current_effect,
            "operative_disposition": operative_disposition,
            "current_position_effect": current_effect,
            "effect": current_effect,
            "boundary_effect": boundary_effect,
            "verification_status": verification_status,
            "verification_reason": verification_reason,
        })
        candidate.challenge_response = response
        item["status"] = operative_disposition
        item["last_response"] = {
            "specialist": candidate.specialist,
            "reported_disposition": reported_disposition,
            "reported_current_position_effect": reported_current_effect,
            "operative_disposition": operative_disposition,
            "current_position_effect": current_effect,
            "boundary_effect": boundary_effect,
            "verification_status": verification_status,
            "verification_reason": verification_reason,
            "answer": " ".join(str(response.get("answer", "")).split())[:900],
            "cycle": max(1, next_cycle - 1),
        }
        follow_up = " ".join(
            str(response.get("follow_up_question", "")).split()
        )[:240]
        if (
            operative_disposition == "REFINED"
            and follow_up and follow_up.upper() != "NONE"
        ):
            followups.append(ArgumentChallenge(
                challenge_kind="SPECIALIST_FOLLOW_UP",
                question=follow_up,
                about_specialist=candidate.specialist,
                target_specialists=(candidate.specialist,),
                grounded_in=tuple(item.get("grounded_in", []) or []),
                trigger_fields=tuple(item.get("trigger_fields", []) or []),
                focus_action=str(item.get("focus_action", "")),
                priority=max(0.5, float(item.get("priority", 0.5)) - 0.01),
                generated_by=f"{candidate.specialist.upper()}_SPECIALIST",
                raised_by=(candidate.specialist,),
                grounding_status=str(item.get("grounding_status", "UNGROUNDED")),
            ).as_dict())

    for raw in [*generated_challenges, *followups]:
        item = copy.deepcopy(dict(raw))
        issue_id = str(item.get("issue_id", ""))
        if not issue_id.startswith("CHALLENGE:"):
            continue
        prior = retained.get(issue_id)
        if prior is None:
            item["status"] = str(item.get("status", "UNTESTED")).upper()
            item["first_seen_cycle"] = max(1, next_cycle - 1)
            retained[issue_id] = item
        else:
            prior["priority"] = max(
                float(prior.get("priority", 0.0)),
                float(item.get("priority", 0.0)),
            )
            prior["last_seen_cycle"] = max(1, next_cycle - 1)

    ordered = sorted(
        retained.values(),
        key=lambda item: (
            str(item.get("status", "")).upper() == "RESOLVED",
            -float(item.get("priority", 0.0)),
            int(item.get("first_seen_cycle", next_cycle)),
            str(item.get("issue_id", "")),
        ),
    )[:40]
    if str(next_constraint).upper() in _CHALLENGE_AGENDA_PROTECTED_CONSTRAINTS:
        return ordered, []

    active = {str(name).casefold() for name in active_specialists}
    assigned_targets: set[str] = set()
    agenda: list[dict[str, object]] = []
    for item in ordered:
        if str(item.get("status", "UNTESTED")).upper() not in _ACTIVE_CHALLENGE_STATUSES:
            continue
        targets = [
            str(value).casefold()
            for value in item.get("target_specialists", []) or []
            if str(value).casefold() in active
        ]
        target = next((value for value in targets if value not in assigned_targets), "")
        if not target:
            continue
        assigned_targets.add(target)
        assigned = copy.deepcopy(item)
        assigned["status"] = "ASSIGNED"
        assigned["assigned_cycle"] = max(1, next_cycle)
        agenda.append(assigned)
    return ordered, agenda


def _commit_access_variable_node(
    graph_store: SemanticGraphStore,
    selected_action: str,
    access_decision: WorkspaceAccessDecision,
    *,
    cycle_number: int,
) -> tuple[str, str, str]:
    """Store the unresolved audit variable as typed graph state."""
    action_id = graph_store._resolve_action_ref(selected_action)
    probe_terms = [
        signal for signal in access_decision.signals
        if signal in {
            "third_party_status",
            "authorization_scope",
            "organizational_boundary",
            "privacy_scope",
        }
    ]
    node_label = access_decision.question or "unresolved decision variable"
    node_id = f"ACCESS_VARIABLE:{cycle_number}:{len(graph_store.graph.nodes)}"
    graph_store.graph.add_node(SemanticNode(
        node_id,
        "CONDITION",
        node_label,
        (f"cycle:{cycle_number}", "consensus_access_gate"),
        {
            "probe_kind": "MISSING_DECISION_VARIABLE",
            "probe_signals": probe_terms,
            "audit_variable": dict(access_decision.audit_variable or {}),
            "selected_action": selected_action,
            "selected_action_id": action_id or "NONE",
            "admitted": True,
        },
    ))
    if action_id:
        graph_store.graph.add_edge(SemanticEdge(
            action_id,
            "HAS_CONSTRAINT",
            node_id,
            justification=access_decision.question,
            provenance=(f"cycle:{cycle_number}", "consensus_access_gate"),
        ))
    return node_id, "CONDITION", node_label


@dataclass(slots=True)
class WorkspaceConfig:
    max_cycles: int = 4
    high_urgency_cycles: int = 2
    time_budget_seconds: float = 180.0
    entropy_threshold: float = 0.62
    stable_cycles_required: int = 2
    # Stop after one unchallenged unanimous cycle by default. Deployments or
    # tests that require two-observation stability can explicitly disable it;
    # when a challenge exists, recurrence remains targeted rather than skipped.
    stop_redundant_consensus_cycles: bool = True
    surprise_weight: float = 0.28
    urgency_weight: float = 0.24
    friction_weight: float = 0.28
    minority_weight: float = 0.12
    redundancy_weight: float = 0.18
    unresolved_tension_weight: float = 0.16
    min_valid_specialists: int = 2
    enable_synthesis: bool = True
    synthesis_after_cycle: int = 1
    synthesis_min_entropy: float = 0.55
    max_cycle_extensions: int = 1
    enable_planning: bool = True
    planning_entropy_threshold: float = 0.55
    max_planning_branches: int = 2
    enable_consensus_audit: bool = True
    enable_problem_state_audit: bool = True
    consensus_audit_max_entropy: float = 0.50
    consensus_audit_min_signals: int = 3
    enable_reversal_audit: bool = True
    graph_rejection_policy: str = "RETAIN_VOTE"
    enable_ev_dominance_breaker: bool = True
    ev_dominance_ratio: float = 5.0
    ev_majority_fraction: float = 0.6
    # Attention bonus for unresolved but consequential claims that may interrupt
    # without becoming governing rules. Tunable; starts conservative.
    investigative_attention_weight: float = 0.40


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
            + cfg.unresolved_tension_weight * candidate.tension_engagement
            - cfg.redundancy_weight * redundancy
        ) * (0.5 + 0.5 * candidate.epistemic_confidence)
        if candidate.landscape_search_attempted and not candidate.landscape_semantic_valid:
            value *= 0.55
        # Investigative claims: unresolved consequential arguments interrupt.
        # Prefer explicit investigative_priority over the deprecated authority enum.
        investigative = float(getattr(candidate, "investigative_priority", 0.0) or 0.0)
        if investigative > 0.0 or candidate.reopen_eligible:
            value += cfg.investigative_attention_weight * max(
                investigative, candidate.tension_engagement, 0.55,
            )
            if candidate.reopen_eligible:
                value += 0.20
            elif candidate.unresolved in {
                "NORMATIVE_ADJUDICATION", "RESOLVE_NORMATIVE_TENSION",
            }:
                value += 0.15
            elif candidate.unresolved == "DECISION_BOUNDARY":
                value += 0.12
        return value

    @staticmethod
    def _tension_engagement(
        candidate: CandidateChunk,
        problem_state: dict[str, object] | None,
    ) -> float:
        """Measure typed engagement with live, attributed deliberative tension."""
        state = dict(problem_state or {})
        questions = list(state.get("unresolved_questions", []) or [])
        conflicts = list(state.get("active_conflicts", []) or [])
        if not questions and not conflicts:
            return 0.0
        own_questions = [
            item for item in questions
            if item.get("source_specialist") == candidate.specialist
        ]
        involved_conflicts = [
            item for item in conflicts
            if candidate.specialist in item.get("specialists", [])
        ]
        candidate.tension_target_keys = list(dict.fromkeys(
            str(item.get("question_key") or item.get("conflict_key"))
            for item in (*own_questions, *involved_conflicts)
            if item.get("question_key") or item.get("conflict_key")
        ))
        value = 0.0
        # Continuing an attributed question matters, but repetition alone gets
        # little credit. Typed factual/normative uptake supplies most weight.
        if own_questions:
            value += 0.15
            if candidate.unresolved == "NONE":
                value += 0.20
        if involved_conflicts:
            value += 0.15
        if candidate.workspace_reasoning_effect in {"FACTUAL", "NORMATIVE", "BOTH"}:
            value += 0.35
        if candidate.workspace_proposition_response in {"ACCEPT", "QUALIFY", "REJECT"}:
            value += 0.15
        if candidate.position_changed and candidate.change_justification:
            value += 0.10
        if (
            candidate.evidence_basis == "UNSTATED_FACTS"
            or not candidate.landscape_semantic_valid
            or not candidate.framework_constraint_retained
        ):
            return 0.0
        return clamp(value)

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
            # Provisional / incomplete adjudications attenuate vote strength
            # without erasing directional information (factor 0 zeros them).
            weight *= max(0.0, float(candidate.policy_weight_factor))
            if weight <= 0.0:
                continue
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
        if weight_sum <= 0.0:
            return {action: 1.0 / len(actions) for action in actions}
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
    def _select_governing_candidate(
        candidates: Sequence[CandidateChunk],
        plurality: str,
        preferred: CandidateChunk | None = None,
    ) -> CandidateChunk | None:
        """Pick a justificatory rule source — not merely the broadcast focus."""
        return select_governing_claim(candidates, plurality, preferred=preferred)

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
    def _proposal_has_review_potential(proposal: SynthesisProposal) -> bool:
        """True when the proposal could reverse policy or improve a Pareto frontier."""
        if float(proposal.feasibility) >= 0.70:
            return True
        if len(proposal.addressed_constraints) >= 2:
            return True
        if len(proposal.source_agents) >= 2:
            return True
        return False

    @staticmethod
    def _proposal_review_pass_completed(
        proposal: SynthesisProposal,
        result: WorkspaceResult,
    ) -> bool:
        """True once a dedicated PROPOSAL_REVIEW cycle has run for this proposal."""
        proposal_id = str(proposal.proposal_id or "").strip()
        action = str(proposal.action or "").strip()
        for cycle in result.cycles:
            received = cycle.received_broadcast or cycle.broadcast
            if received.constraint != "PROPOSAL_REVIEW":
                continue
            intent = str(received.intent or "")
            context = str(received.reformulation_context or "")
            if proposal_id and (
                proposal_id.casefold() in intent.casefold()
                or proposal_id in context
            ):
                return True
            if action and action in context:
                return True
        return False

    @staticmethod
    def _proposal_review_complete(
        proposal: SynthesisProposal,
        expected_specialists: Sequence[str],
    ) -> bool:
        """True once the admitted proposal has received a specialist review pass."""
        if not proposal.framework_reviews:
            return False
        expected = [name for name in expected_specialists if str(name).strip()]
        if not expected:
            return True
        reviewed = {
            name for name, review in proposal.framework_reviews.items()
            if isinstance(review, dict)
        }
        missing = set(expected) - reviewed
        if not missing:
            return True
        # Partial receipt still counts once a majority of frameworks have spoken.
        return len(reviewed) >= max(2, (len(expected) + 1) // 2)

    def _pending_proposal_for_guaranteed_review(
        self,
        result: WorkspaceResult,
    ) -> SynthesisProposal | None:
        """Select an admitted proposal that still lacks its dedicated review pass."""
        expected = [specialist.name for specialist in self.specialists]
        for proposal in reversed(result.synthesis_proposals):
            if not proposal.accepted:
                continue
            if proposal.promotion_status not in {"UNDER_REVIEW", "PROPOSED"}:
                continue
            if str(proposal.admission_status).upper() in {
                "WITHHOLD_FROM_REVIEW", "REJECTED",
            }:
                continue
            if self._proposal_review_pass_completed(proposal, result):
                continue
            if self._proposal_review_complete(proposal, expected):
                continue
            if not self._proposal_has_review_potential(proposal):
                continue
            return proposal
        return None

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
            "previous_framework_state", "proposition_ledger",
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
        has_obstacle, _, _ = WorkspaceEngine._planning_obstacle_status(
            scenario, broadcast, actions, target_action
        )
        return has_obstacle

    @staticmethod
    def _planning_obstacle_status(
        scenario: str,
        broadcast: WorkspaceBroadcast,
        actions: Sequence[str],
        target_action: str,
    ) -> tuple[bool, bool, str]:
        """Return whether the target is obstructed and whether any fallback survives."""
        context = " ".join(
            (scenario, broadcast.contingency_question, broadcast.reformulation_context)
        )
        facts = compile_execution_obstacles(context, actions)
        try:
            target_index = list(actions).index(target_action)
        except ValueError:
            return False, False, "target action is not in the workspace action set"
        target_id = f"A{target_index}"
        affected_ids = {
            fact.affected_action_node_id
            for fact in facts
            if fact.affected_action_node_id
        }
        has_obstacle = target_id in affected_ids
        fallback_ids = [
            f"A{index}"
            for index in range(len(actions))
            if f"A{index}" != target_id
        ]
        surviving_fallbacks = [
            action_id for action_id in fallback_ids if action_id not in affected_ids
        ]
        if surviving_fallbacks:
            return (
                has_obstacle,
                True,
                f"{surviving_fallbacks[0]} remains physically executable",
            )
        if has_obstacle:
            return (
                True,
                False,
                "no other action remains physically executable after the obstacle",
            )
        return False, False, "no action-linked implementation obstacle was detected"

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

        grounding_status, unsupported_details = classify_planning_failure_grounding(
            assessment.necessary_condition,
            assessment.failure_condition,
            scenario,
            actions,
            assessment.grounded_evidence,
        )
        assessment.failure_grounding_status = grounding_status
        if unsupported_details:
            errors.append(
                "planning introduced concrete external details without current-run provenance: "
                + ", ".join(unsupported_details)
            )
        elif grounding_status == "PLAUSIBLE_HYPOTHETICAL_FAILURE_CONDITION":
            assessment.broadcast_worthy = False

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
        graph: SemanticGraph,
        scenario: str,
        actions: Sequence[str],
        candidates: Sequence[CandidateChunk],
        selected_action: str,
        entropy: float,
        dissent: CandidateChunk | None,
        scenario_facts: dict | None,
        source_testimonies: dict[str, str] | None,
        problem_state: dict[str, object] | None = None,
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

        # Consensus audits may focus only an issue already admitted to the live
        # ProblemState. Keyword probes remain available to explicit legacy
        # audits, but cannot manufacture a variable for consensus review.
        missing_variable_signals, missing_variable_question, missing_variable_payload = (
            _problem_state_audit_probe(
                problem_state, selected_action, require_clause_grounding=True,
            )
        )
        if missing_variable_signals:
            signals.extend(missing_variable_signals)

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
            "third_party_status",
            "authorization_scope",
            "organizational_boundary",
            "privacy_scope",
        ))
        conditionality_erased = (
            "conditional_source_testimony" in signals
            and "uncertainty_erased_by_delegates" in signals
            and "unanimous_recommendation" in signals
        )
        if conditionality_erased:
            signals.append("conditionality_collapsed_into_consensus")
            structural_signal = True
        if missing_variable_signals:
            structural_signal = True
        admitted = bool(missing_variable_payload) and structural_signal and len(signals) >= self.config.consensus_audit_min_signals
        return WorkspaceAccessDecision(
            cycle=cycle_number,
            content_type="CONSENSUS_AUDIT",
            admitted=admitted,
            signals=signals,
            question=(
                (
                    missing_variable_question
                    or f"The apparent dominance of '{selected_action}' may depend on unsupported "
                    "or asymmetric assumptions. Which assumption is unstated, and what plausible "
                    "facts would reverse the recommendation?"
                )
                if admitted else ""
            ),
            rationale=(
                "Rapid agreement requires one epistemic challenge before convergence."
                if admitted else "Consensus did not meet the suspicious-access threshold."
            ),
            audit_variable=missing_variable_payload,
        )

    def _investigative_reopen_decision(
        self,
        cycle_number: int,
        selected_action: str,
        candidates: Sequence[CandidateChunk],
        problem_state: dict[str, object] | None,
        fired_keys: dict[str, str],
    ) -> WorkspaceAccessDecision | None:
        """Admit a reopen-eligible investigative interrupt with one-shot budget."""
        if not self.config.enable_problem_state_audit:
            return None
        eligible = [
            candidate for candidate in candidates
            if candidate.schema_valid and candidate.reopen_eligible
        ]
        if not eligible:
            return None
        chosen = max(
            eligible,
            key=lambda c: (
                c.investigative_priority,
                c.epistemic_confidence,
                c.specialist,
            ),
        )
        question_key = chosen.reopen_question_key or ""
        if not question_key:
            return None
        fingerprint = evidence_fingerprint_for(chosen, problem_state)
        payload = {
            "entity": question_key,
            "relation": "INVESTIGATIVE_INTERRUPT",
            "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
            "focus_action": selected_action,
            "question": chosen.reopen_reason or chosen.investigative_claim,
            "status": "REOPEN_REQUIRED",
            "question_key": question_key,
            "raised_by": [chosen.specialist],
            "priority": chosen.investigative_priority,
        }
        return WorkspaceAccessDecision(
            cycle=cycle_number,
            content_type="PROBLEM_STATE_AUDIT",
            admitted=True,
            signals=[
                "investigative_interrupt",
                "reopen_eligible",
                f"priority>={REOPEN_PRIORITY_THRESHOLD:.2f}",
            ],
            question=payload["question"],
            rationale=(
                "A grounded, novel investigative claim with high reversal potential "
                "forces reopen before finalization; it does not itself reverse policy."
            ),
            audit_variable=payload,
        )

    def _problem_state_access_decision(
        self,
        cycle_number: int,
        selected_action: str,
        problem_state: dict[str, object] | None,
    ) -> WorkspaceAccessDecision | None:
        """Admit one grounded issue without pretending consensus."""
        if not self.config.enable_problem_state_audit or cycle_number < 2:
            return None
        signals, question, payload = _problem_state_audit_probe(
            problem_state, selected_action,
        )
        if not payload:
            return None
        # Persistence normally earns the audit: an issue that survived a cycle
        # is more likely to be live than one raised once. But when every
        # framework already prefers the same action, another cycle cannot
        # resolve a disagreement that does not exist, so the remaining
        # deliberative value is in stress-testing the agreement rather than
        # waiting a cycle to confirm the question is still there.
        already_agreed = str(
            (problem_state or {}).get("surface_consensus", "")
        ).upper() == "UNANIMOUS"
        persistent = payload.get("status") == "PERSISTENT_UNRESOLVED"
        # Unanimity substitutes for persistence, not for grounding. A question
        # the graph cannot trace to a clause is already the weakest audit
        # target available, and spending the untested-agreement slot on it
        # would stack two separate relaxations on one audit.
        if not persistent and not (
            already_agreed
            and payload.get("grounding_status") == "CLAUSE_GROUNDED"
        ):
            return None
        return WorkspaceAccessDecision(
            cycle=cycle_number,
            content_type="PROBLEM_STATE_AUDIT",
            admitted=True,
            signals=list(dict.fromkeys([
                *signals,
                "persistent_grounded_issue" if persistent
                else "unanimous_agreement_untested",
                "deliberative_focus",
            ])),
            question=question,
            rationale=(
                "A persistent grounded issue receives focused review even when "
                "the Parliament is contested rather than converged."
                if persistent else
                "Every framework already prefers this action, so the cycle is "
                "spent testing the agreement rather than re-confirming it."
            ),
            audit_variable={
                key: value for key, value in payload.items()
                if key != "grounding_status"
            },
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
        action_source_grounding: dict[str, Any] | None = None,
        presentation_actions: Sequence[str] | None = None,
        canonical_action_records: Sequence[dict[str, Any]] | None = None,
        source_testimonies: dict[str, str] | None = None,
        reformulate_problem: ReformulationCallback | None = None,
        assess_visibility: VisibilityCallback | None = None,
        assess_autonomy: AutonomyCallback | None = None,
        checkpoint: CheckpointCallback | None = None,
        audit_side_premises: PremiseAuditCallback | None = None,
    ) -> WorkspaceResult:
        clean_actions = list(dict.fromkeys(a.strip() for a in actions if a.strip()))[:5]
        if len(clean_actions) < 2:
            raise ValueError("At least two distinct actions are required")

        # Recurrent delegate state is scoped to one ethical problem. Abstract
        # schemas live in code; concrete parties, claims, and adjudications must
        # be regenerated from the current scenario graph on every run.
        for specialist in self.specialists:
            if not hasattr(specialist, "scenario_graph"):
                continue
            for name, value in (
                ("previous_recommendation_id", ""),
                ("previous_confidence", None),
                ("previous_context", ""),
                ("assumption_status", "NOT_AUDITED"),
                ("unsupported_assumption", ""),
                ("reversal_condition", ""),
                ("epistemic_commitments", []),
                ("previous_framework_state", {}),
                ("private_framework_contribution", {}),
                ("proposition_ledger", []),
            ):
                if hasattr(specialist, name):
                    setattr(specialist, name, copy.deepcopy(value))

        broadcast = initial_broadcast or WorkspaceBroadcast()
        result = WorkspaceResult(
            scenario=scenario,
            actions=clean_actions,
            active_specialists=[specialist.name for specialist in self.specialists],
            presentation_actions=list(presentation_actions or clean_actions),
            source_action_legend=dict(source_action_legend or {}),
            action_source_grounding=dict(action_source_grounding or {}),
            canonical_action_records=[
                dict(record) for record in (canonical_action_records or [])
            ],
        )
        # Canonical actions and explicit observability facts are system-owned graph
        # state. Delegate transactions may extend this graph but cannot redefine
        # the identity of the action currently being planned.
        grounded_actions = dict((action_source_grounding or {}).get("actions", {}))
        graph_store = SemanticGraphStore(compile_scenario_graph(
            scenario, clean_actions, grounded_actions,
            world_model=dict((action_source_grounding or {}).get("world_model", {})),
        ))
        proposition_ledger = seed_proposition_ledger(graph_store.graph)
        result.proposition_ledger = ledger_projection(proposition_ledger)
        # The opening cycle is the only one whose broadcast has no prior cycle
        # to describe, but the shared world already exists by this point. Hand
        # over that world without a position so cycle 1 is an informed
        # independent read rather than an uninformed one.
        if not broadcast.problem_state:
            broadcast = replace(
                broadcast,
                problem_state=opening_problem_state(
                    clean_actions, scenario, graph_store.graph,
                    world_model=(action_source_grounding or {}).get("world_model"),
                ),
            )
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
            elif visibility.valid and visibility.mechanism_provenance != "SCENARIO_GROUNDED":
                if progress:
                    progress(
                        "Visibility audit recorded as "
                        f"{visibility.mechanism_provenance.lower()}; "
                        "no confidence penalty applied"
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
        incumbent_governing_specialist = ""
        stable_cycles = 0
        constraint_counts: dict[str, int] = {}
        previous_dissent: CandidateChunk | None = None
        last_valid_framework_candidates: dict[str, CandidateChunk] = {}
        synthesis_attempted = False
        proposal_review_forced = False
        planned_contexts: set[tuple[str, str]] = set()
        planning_branches_used = 0
        planning_resume_broadcast: WorkspaceBroadcast | None = None
        reformulation_resume_broadcast: WorkspaceBroadcast | None = None
        reversal_resume_broadcast: WorkspaceBroadcast | None = None
        contingency_resume_broadcast: WorkspaceBroadcast | None = None
        consensus_audit_attempted = False
        access_signal_signatures: set[tuple[str, ...]] = set()
        # question_key -> evidence fingerprint for one forced reopen per key,
        # with an escape hatch when materially new grounded evidence appears.
        fired_reopen_keys: dict[str, str] = {}
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
            working_proposition_ledger = (
                copy.deepcopy(proposition_ledger) if is_counterfactual
                else proposition_ledger
            )
            cycle_proposition_projection = ledger_projection(
                working_proposition_ledger
            )
            specialist_snapshot = (
                self._snapshot_specialist_state(self.specialists) if is_counterfactual else []
            )
            if progress:
                progress(
                    f"Cycle {cycle_number}/{cycle_limit} — broadcast {broadcast.trace_summary()}"
                )
            candidates = []
            terminal_model_failure = False
            for index, specialist in enumerate(self.specialists, start=1):
                if hasattr(specialist, "scenario_graph"):
                    specialist.scenario_graph = graph_store.graph
                if hasattr(specialist, "proposition_ledger"):
                    specialist.proposition_ledger = copy.deepcopy(
                        cycle_proposition_projection
                    )
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
                        recommended_action="",
                        adjudication_status="CONTESTED_NO_LEANING",
                        governing_eligible=False,
                        broadcast_authority="INVESTIGATIVE",
                        policy_weight_factor=0.0,
                        selection_status="UNSELECTED",
                        delegate_status="MODEL_ERROR",
                        error_type="MODEL_ERROR",
                        validation_errors=[str(exc)[:300]],
                    )
                    terminal_model_failure = exc.terminal
                    if exc.terminal:
                        result.halted_by = "model_backend_unavailable"
                except Exception as exc:
                    cause = getattr(exc, "cause", exc)
                    exception_type = type(cause).__name__
                    failure_stage = str(
                        getattr(exc, "stage", "SPECIALIST_EVALUATION")
                    ).strip().upper()
                    diagnostic = (
                        f"specialist evaluation failed at {failure_stage}: "
                        f"{exception_type}: {cause}"
                    )
                    candidate = _invalid_candidate(
                        specialist.name,
                        cycle_actions,
                        diagnostic,
                        delegate_status_override="SPECIALIST_INTERNAL_ERROR",
                        error_type_override="SPECIALIST_INTERNAL_ERROR",
                        exception_type=exception_type,
                        failure_stage=failure_stage,
                    )
                    if progress:
                        progress(
                            f"    {specialist.name} delegate failed softly at "
                            f"{failure_stage}: {exception_type}: {cause}"
                        )
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
                    action_tags = {
                        autonomy.action_tags.get(action, "NONE")
                        for action in clean_actions
                    }
                    burden_discriminates = len(action_tags) > 1
                    if (
                        tag != "NONE"
                        and burden_discriminates
                        and not autonomy.catastrophic_harm_threshold.get(
                            candidate.recommended_action, False
                        )
                    ):
                        candidate.coercion_surcharge = 1.0 - autonomy.surcharge_multiplier
                        # Keep the normative surcharge separate from epistemic
                        # confidence. The audit records that the action carries a
                        # coercion/rights-breach burden, but the delegate's
                        # epistemic reliability should only change when its own
                        # reasoning is uncertain.
                if candidate.schema_valid:
                    attach_candidate_dependencies(
                        working_proposition_ledger, candidate,
                    )
                    if not is_counterfactual:
                        result.proposition_ledger = ledger_projection(
                            proposition_ledger
                        )
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
                    _record_committed_native_ledger(
                        candidate,
                        ledger_kind="UTILITARIAN_CONSEQUENCE_LEDGER",
                        transaction_status=util_transaction.status,
                        records=result.utilitarian_consequence_ledger,
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
                    _record_committed_native_ledger(
                        candidate,
                        ledger_kind="DEONTOLOGICAL_DUTY_LEDGER",
                        transaction_status=deon_transaction.status,
                        records=result.deontological_duty_ledger,
                    )
                    result.graph_transactions = graph_store.transaction_dicts()
                    result.semantic_graphs = [graph_store.graph_dict()]
                    if deon_transaction.status == "COMMITTED_WITH_UNCERTAINTY":
                        calibrated_contested = any(
                            item.get("calibration_errors")
                            for item in result.deontological_duty_ledger
                        )
                        _apply_framework_ledger_uncertainty(
                            candidate, deon_transaction.errors,
                            state_status=(
                                "CALIBRATED_CONTESTED"
                                if calibrated_contested
                                else "COMMITTED_WITH_UNCERTAINTY"
                            ),
                        )
                    recommended_id = (
                        f"A{clean_actions.index(candidate.recommended_action)}"
                        if candidate.recommended_action in clean_actions else ""
                    )
                    committed_by_action = latest_assessment_by_action(
                        result.deontological_duty_ledger
                    )
                    rendered_assessment = committed_by_action.get(recommended_id)
                    # The rival adjudications are part of this verdict, not
                    # separate ones. Preferring the recommended action rests on
                    # the rivals being prohibited, so a rival prohibition the
                    # calibration downgraded has to constrain the claim.
                    rival_assessments = [
                        item
                        for action_id, item in sorted(committed_by_action.items())
                        if action_id != recommended_id
                        and action_id in {
                            f"A{index}" for index in range(len(clean_actions))
                        }
                    ]
                    if rendered_assessment is not None:
                        authority = classify_deontological_authority(
                            rendered_assessment,
                            rival_assessments,
                            recommended_action=candidate.recommended_action,
                            preference_strength=candidate.preference_strength,
                        )
                        candidate.rationale = authority.rationale
                        candidate.decision_rule = authority.decision_rule
                        candidate.adjudication_status = authority.adjudication_status
                        candidate.broadcast_authority = authority.broadcast_authority
                        candidate.governing_eligible = authority.governing_eligible
                        candidate.policy_weight_factor = authority.policy_weight_factor
                        candidate.investigative_claim = authority.investigative_claim
                        internal_conflicts = list(authority.internal_conflicts)
                        open_questions = list(authority.open_questions)
                        candidate.framework_internal_conflicts = list(dict.fromkeys([
                            *candidate.framework_internal_conflicts,
                            *internal_conflicts,
                        ]))[:3]
                        candidate.framework_specific_open_questions = list(dict.fromkeys([
                            *candidate.framework_specific_open_questions,
                            *open_questions,
                        ]))[:3]
                        if open_questions:
                            candidate.unresolved = "NORMATIVE_ADJUDICATION"
                            candidate.assumption_status = "NORMATIVELY_CONTESTED"
                            candidate.selection_status = "PROVISIONAL"
                            candidate.comparison_complete = False
                            candidate.evidence_sufficient_for_action = False
                    if (
                        deon_transaction.status.startswith("COMMITTED")
                        and candidate.framework_constraint_retained
                        and hasattr(specialist, "previous_framework_state")
                        and result.deontological_duty_ledger
                    ):
                        specialist.previous_framework_state = {
                            "adjudication_form": "KANTIAN_CLAIM_COERCION_RESOLUTION",
                            "assessments": [{
                                "action_id": item.get("canonical_action_id", ""),
                                "verdict": item.get("verdict", "CONFLICTED"),
                                "norm_kind": item.get("norm_kind", "UNKNOWN"),
                                "relation": item.get("relation", "UNCERTAIN"),
                                "competing_norm_kind": item.get(
                                    "competing_norm_kind", "UNKNOWN"
                                ),
                                "competing_relation": item.get(
                                    "competing_relation", "UNCERTAIN"
                                ),
                                "duty_type": item.get("duty_type", "UNRESOLVED"),
                                "harm_relation": item.get("harm_relation", "UNRESOLVED"),
                                "special_obligation_status": item.get(
                                    "special_obligation_status", "UNKNOWN"
                                ),
                                "special_obligation_basis": item.get(
                                    "special_obligation_basis", ""
                                ),
                                "means_relation": item.get("means_relation", "UNRESOLVED"),
                                "governing_norm": item.get("governing_norm", "UNRESOLVED"),
                                "priority_basis": item.get("priority_basis", "UNRESOLVED"),
                                "protected_standing": item.get("protected_standing", "UNKNOWN"),
                                "competing_protected_standing": item.get(
                                    "competing_protected_standing", "UNKNOWN"
                                ),
                                "coercion_kind": item.get("coercion_kind", "UNKNOWN"),
                                "authorization_status": item.get(
                                    "authorization_status", "UNKNOWN"
                                ),
                                "derivation": item.get("derivation", "UNRESOLVED"),
                                "resolution_status": item.get(
                                    "resolution_status", "UNKNOWN"
                                ),
                                "norm": item.get("norm", ""),
                                "protected_party": item.get("protected_party", ""),
                                "competing_norm": item.get("competing_norm", ""),
                                "competing_protected_party": item.get(
                                    "competing_protected_party", ""
                                ),
                            } for item in result.deontological_duty_ledger],
                        }
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
                    _record_committed_native_ledger(
                        candidate,
                        ledger_kind="RAWLSIAN_POSITION_LEDGER",
                        transaction_status=rawls_transaction.status,
                        records=result.rawlsian_position_ledger,
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
                            "ranking_classification_justification": committed_positions[0].get(
                                "ranking_classification_justification", ""
                            ),
                            "lexical_priority_justification": committed_positions[0].get(
                                "lexical_priority_justification", ""
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
                                    "subject": item.get(
                                        "subject", item.get("affected_subject", "")
                                    ),
                                    "subject_kind": item.get("subject_kind", "UNKNOWN"),
                                    "dimension": item.get("dimension", "UNKNOWN"),
                                    "basic_liberty_kind": item.get(
                                        "basic_liberty_kind", "UNRESOLVED"
                                    ),
                                    "institutional_relation": item.get(
                                        "institutional_relation", "UNRESOLVED"
                                    ),
                                    "additional_dimensions": list(
                                        item.get("additional_dimensions", [])
                                    ),
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
                    _record_committed_native_ledger(
                        candidate,
                        ledger_kind="VIRTUE_CHARACTER_LEDGER",
                        transaction_status=virtue_transaction.status,
                        records=result.virtue_character_ledger,
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
                if (
                    candidate.schema_valid
                    and candidate.specialist == "care"
                    and candidate.care_ledger_proposal
                    and not is_counterfactual
                ):
                    care_transaction = apply_care_ledger_transaction(
                        graph_store,
                        candidate.care_ledger_proposal,
                        cycle=cycle_number,
                        specialist=candidate.specialist,
                        allowed_actions=tuple(clean_actions),
                    )
                    result.care_relationship_ledger = committed_care_assessments(
                        graph_store.graph
                    )
                    _record_committed_native_ledger(
                        candidate,
                        ledger_kind="CARE_RELATIONSHIP_LEDGER",
                        transaction_status=care_transaction.status,
                        records=result.care_relationship_ledger,
                    )
                    result.graph_transactions = graph_store.transaction_dicts()
                    result.semantic_graphs = [graph_store.graph_dict()]
                    if care_transaction.status == "REJECTED":
                        _apply_framework_ledger_uncertainty(
                            candidate, care_transaction.errors,
                            state_status="UPDATE_REJECTED",
                        )
                    elif care_transaction.status == "COMMITTED_WITH_UNCERTAINTY":
                        _apply_framework_ledger_uncertainty(
                            candidate, care_transaction.errors,
                        )
                    if (
                        care_transaction.status.startswith("COMMITTED")
                        and hasattr(specialist, "previous_framework_state")
                        and result.care_relationship_ledger
                    ):
                        committed_care = result.care_relationship_ledger
                        specialist.previous_framework_state = {
                            "ranking_basis": committed_care[0].get(
                                "ranking_basis", "UNRESOLVED"
                            ),
                            "assessments": [
                                {
                                    "action_id": item.get("canonical_action_id", ""),
                                    "verdict": item.get("verdict", "UNCERTAIN"),
                                    "affected_party": item.get("affected_party", ""),
                                    "relationship_type": item.get("relationship_type", "UNRESOLVED"),
                                    "dependency_source": item.get("dependency_source", ""),
                                    "responsibility_basis": item.get("responsibility_basis", ""),
                                    "need_kind": item.get("need_kind", "UNRESOLVED"),
                                    "need_urgency": item.get("need_urgency", "UNKNOWN"),
                                    "trust_effect": item.get("trust_effect", "UNKNOWN"),
                                    "responsiveness": item.get("responsiveness", "UNKNOWN"),
                                    "feasibility": item.get("feasibility", "UNKNOWN"),
                                    "competing_care_claim": item.get("competing_care_claim", ""),
                                    "resolution_status": item.get("resolution_status", "UNKNOWN"),
                                }
                                for item in committed_care
                            ],
                        }
                    if progress and care_transaction.status != "COMMITTED":
                        progress(
                            "    Care relationship ledger "
                            f"{care_transaction.status.lower()}: "
                            + " | ".join(care_transaction.errors[:2])
                        )
                if progress:
                    returned_action = (
                        f"A{cycle_actions.index(candidate.recommended_action)}"
                        if candidate.recommended_action in cycle_actions else "?"
                    )
                    progress(
                        f"  [{index}/{len(self.specialists)}] {specialist.name} returned; "
                        f"position={returned_action}; constraint={candidate.constraint}; "
                        f"preference={candidate.preference_strength:.2f}; "
                        f"epistemic={candidate.epistemic_confidence:.2f}; "
                        f"status={candidate.adjudication_status} "
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
            if received_broadcast.constraint == "PROPOSAL_REVIEW":
                review_proposals = [
                    proposal for proposal in result.synthesis_proposals
                    if proposal.promotion_status == "UNDER_REVIEW"
                ]
                if len(review_proposals) == 1:
                    reviewed_proposal = review_proposals[0]
                    accepted_predictions: list[dict[str, object]] = []
                    for candidate in candidates:
                        review = candidate.proposal_review
                        if review is None or review.proposal_id != reviewed_proposal.proposal_id:
                            continue
                        review.framework_retained = candidate.framework_constraint_retained
                        if not candidate.framework_constraint_retained:
                            review.valid = False
                            if "proposal review failed framework-retention validation" not in review.validation_errors:
                                review.validation_errors.append(
                                    "proposal review failed framework-retention validation"
                                )
                        reviewed_proposal.framework_reviews[candidate.specialist] = review.to_dict()
                        if review.valid:
                            accepted_predictions.extend({
                                **dict(consequence),
                                "source_specialist": candidate.specialist,
                                "framework_status": review.framework_status,
                            } for consequence in review.predicted_consequences)
                    reviewed_proposal.predicted_consequences = accepted_predictions
                    if self._proposal_review_complete(
                        reviewed_proposal,
                        [specialist.name for specialist in self.specialists],
                    ):
                        # Reviewed ≠ promoted: ADMISSIBLE means specialists spoke;
                        # the proposal still stays outside the live action set.
                        reviewed_proposal.promotion_status = "ADMISSIBLE"
                        reviewed_proposal.admission_status = "ADMISSIBLE"
                    proposal_node = graph_store.graph.nodes.get(reviewed_proposal.proposal_id)
                    if proposal_node is not None:
                        graph_store.graph.add_node(SemanticNode(
                            proposal_node.id,
                            proposal_node.kind,
                            proposal_node.label,
                            proposal_node.provenance,
                            {
                                **proposal_node.attributes,
                                "framework_reviews": dict(reviewed_proposal.framework_reviews),
                                "predicted_consequences": list(
                                    reviewed_proposal.predicted_consequences
                                ),
                                "promotion_status": reviewed_proposal.promotion_status,
                            },
                        ))
            if audit_side_premises is not None and any(
                candidate.schema_valid for candidate in candidates
            ):
                try:
                    premise_audit = audit_side_premises(
                        ledger_projection(working_proposition_ledger), candidates,
                    )
                except Exception as error:
                    premise_audit = {
                        "status": "UNAVAILABLE", "findings": [],
                        "error": f"{type(error).__name__}: {error}"[:240],
                    }
                apply_side_premise_audit(
                    working_proposition_ledger, candidates, premise_audit,
                )
                audit_record = {
                    "cycle": cycle_number,
                    "is_hypothetical": is_counterfactual,
                    "status": str(premise_audit.get("status", "UNAVAILABLE")),
                    "findings": list(premise_audit.get("findings", []) or []),
                    "error": str(premise_audit.get("error", "")),
                }
                if not is_counterfactual:
                    result.side_premise_audits.append(audit_record)
                    result.proposition_ledger = ledger_projection(proposition_ledger)
                if progress:
                    progress(
                        "  side-premise audit="
                        f"{audit_record['status']}; findings="
                        f"{len(audit_record['findings'])}"
                    )

            candidates = _operative_framework_candidates(
                candidates,
                last_valid_framework_candidates,
                remember=not is_counterfactual,
            )
            # Framework-general authority typing: policy weight, investigative
            # attention, and governing eligibility are independent dimensions.
            settled_keys = list(settled_question_keys(graph_store.graph))
            settled_keys.extend(
                str(item.get("question_key") or item.get("issue_id") or "")
                for item in (
                    (received_broadcast.problem_state or {}).get("resolved_questions", [])
                    or []
                )
            )
            for candidate in candidates:
                apply_specialist_authority(candidate)
                if candidate.schema_valid:
                    apply_investigative_authority(
                        candidate,
                        plurality=previous_action,
                        policy=None,
                        problem_state=received_broadcast.problem_state,
                        fired_keys=fired_reopen_keys,
                        settled_keys=settled_keys,
                    )
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
                candidate.tension_engagement = self._tension_engagement(
                    candidate, received_broadcast.problem_state,
                )
                candidate.salience = self._salience(candidate, broadcast, constraint_counts, bonus)

            for candidate in candidates:
                if candidate.schema_valid:
                    continue
                candidate.constraint = "NONE"
                if candidate.delegate_status == "VALID":
                    candidate.delegate_status = "SEMANTIC_VALIDATION_ERROR"
                    candidate.error_type = "SEMANTIC_VALIDATION_ERROR"
            valid_candidates = [candidate for candidate in candidates if candidate.schema_valid]
            broadcast_focus = max(
                valid_candidates,
                key=lambda c: (c.salience, c.epistemic_confidence, c.specialist),
            ) if valid_candidates else None
            cycle_has_quorum = len(valid_candidates) >= self.config.min_valid_specialists
            recorded_focus = broadcast_focus
            # A neutral internal placeholder keeps policy/finalization arithmetic
            # total; it is never serialized as a deliberative broadcast focus.
            focus = broadcast_focus or max(
                candidates,
                key=lambda c: (c.salience, c.epistemic_confidence, c.specialist),
            )
            if not is_counterfactual and recorded_focus is not None:
                constraint_counts[focus.constraint] = constraint_counts.get(focus.constraint, 0) + 1
            policy = self._policy(
                candidates,
                cycle_actions,
                visibility_multipliers,
                visibility_review=(received_broadcast.constraint == "VISIBILITY_AUDIT"),
            )
            selected_action = max(policy, key=policy.get)
            for candidate in valid_candidates:
                apply_investigative_authority(
                    candidate,
                    plurality=selected_action,
                    policy=policy,
                    problem_state=received_broadcast.problem_state,
                    fired_keys=fired_reopen_keys,
                    settled_keys=settled_keys,
                )
            proposed_governing_claim = self._select_governing_candidate(
                valid_candidates,
                selected_action,
                preferred=(
                    focus
                    if not incumbent_governing_specialist
                    and recorded_focus is not None
                    and focus.recommended_action == selected_action
                    else None
                ),
            )
            incumbent_candidate = next((
                candidate for candidate in valid_candidates
                if candidate.specialist == incumbent_governing_specialist
                and candidate.recommended_action == selected_action
                and candidate.governing_eligible
                and candidate.decision_rule
            ), None)
            governing_claim = incumbent_candidate or proposed_governing_claim
            transition_reason = ""
            if not is_counterfactual:
                if incumbent_governing_specialist and incumbent_candidate is None:
                    transition_reason = "INCUMBENT_INVALIDATED_OR_POLICY_CHANGED"
                elif (
                    incumbent_candidate is not None
                    and proposed_governing_claim is not None
                    and proposed_governing_claim.specialist != incumbent_candidate.specialist
                    and received_broadcast.constraint in {
                        "CONSENSUS_AUDIT", "PROBLEM_STATE_AUDIT", "REVERSAL_AUDIT",
                    }
                    and incumbent_candidate.audit_internal_effect in {"WEAKENS", "REVERSES"}
                    and proposed_governing_claim.audit_participation != "NOT_TESTED"
                ):
                    governing_claim = proposed_governing_claim
                    transition_reason = "TARGETED_CHALLENGE_CHANGED_ADJUDICATION"
                elif not incumbent_governing_specialist and governing_claim is not None:
                    transition_reason = "INITIAL_GOVERNING_ADJUDICATION"

                new_specialist = (
                    governing_claim.specialist if governing_claim is not None else ""
                )
                if new_specialist != incumbent_governing_specialist:
                    result.governing_authority_transitions.append({
                        "cycle": cycle_number,
                        "from_specialist": incumbent_governing_specialist or "NONE",
                        "to_specialist": new_specialist or "NONE",
                        "action": selected_action,
                        "reason": transition_reason or "NO_ELIGIBLE_GOVERNING_CLAIM",
                        "trigger_constraint": received_broadcast.constraint,
                        "argumentative_event": transition_reason in {
                            "INCUMBENT_INVALIDATED_OR_POLICY_CHANGED",
                            "TARGETED_CHALLENGE_CHANGED_ADJUDICATION",
                        },
                    })
                incumbent_governing_specialist = new_specialist
            for candidate in valid_candidates:
                candidate.broadcast_authority = derive_broadcast_authority(
                    governing_eligible=candidate.governing_eligible,
                    is_governing_focus=(
                        governing_claim is not None and candidate is governing_claim
                    ),
                    reopen_eligible=candidate.reopen_eligible,
                    investigative_priority=candidate.investigative_priority,
                )
            # Deprecated alias: winner == broadcast_focus for one migration cycle.
            deliberative_winner = recorded_focus
            recorded_winner = recorded_focus
            winner = focus
            if not is_counterfactual:
                stable_cycles = stable_cycles + 1 if selected_action == previous_action else 1
                previous_action = selected_action
            entropy = self._normalized_entropy(policy)
            dissent = self._dissent(candidates, selected_action)
            divergent_recommendations = {
                candidate.recommended_action
                for candidate in valid_candidates
                if candidate.recommended_action
            }
            synthesis_pressure = bool(
                dissent is not None or len(divergent_recommendations) >= 2
            )
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

            # An audited question that came back with agreement is answered.
            # Record it before projecting the next ProblemState so the audit
            # slot in later cycles goes to something still open, and so a
            # later change to the evidence behind the answer can reopen it.
            audited_issue = dict(received_broadcast.audit_variable or {})
            if not is_counterfactual and str(
                audited_issue.get("issue_id", "")
            ).startswith(("QUESTION:", "CHALLENGE:")):
                question_resolution = resolve_audited_question(
                    valid_candidates,
                    question_key=str(audited_issue.get("issue_id", "")),
                    proposition=str(audited_issue.get("proposition", "")),
                    cycle=cycle_number,
                    grounded_in=list(audited_issue.get("grounded_in", []) or []),
                    graph=graph_store.graph,
                )
                if question_resolution is not None:
                    resolution_record = commit_question_resolution(
                        graph_store, question_resolution,
                    )
                    if progress and resolution_record.status == "COMMITTED":
                        progress(
                            f"  audited issue {question_resolution.question_key} "
                            f"settled as {question_resolution.resolution} by "
                            + ", ".join(question_resolution.responders)
                        )

            deliberative_state = build_deliberative_problem_state(
                cycle_number,
                clean_actions,
                valid_candidates,
                selected_action,
                winner,
                received_broadcast.problem_state,
                graph_store.graph,
                scenario,
                result.synthesis_proposals,
            )
            if not is_counterfactual:
                private_by_agent = {
                    item.agent: item.to_dict()
                    for item in deliberative_state.workspace_contributions
                }
                for specialist in self.specialists:
                    if hasattr(specialist, "private_framework_contribution"):
                        specialist.private_framework_contribution = copy.deepcopy(
                            private_by_agent.get(specialist.name, {})
                        )
            if not is_counterfactual:
                update_broadcast_influence_persistence(
                    result.broadcast_influence_records,
                    deliberative_state,
                )
                result.broadcast_influence_records.extend(
                    record.to_dict() for record in observe_broadcast_influence(
                        received_broadcast.problem_state,
                        deliberative_state,
                        valid_candidates,
                    )
                )
            next_problem_state = (
                deliberative_state.to_dict()
                if recorded_winner is not None
                else _preserve_problem_state_after_invalid_cycle(
                    received_broadcast.problem_state, candidates,
                )
            )
            cycle_focus_proposition_ids = focus_proposition_ids(
                working_proposition_ledger, valid_candidates,
            )
            shared_dependencies = shared_unresolved_dependency_projection(
                working_proposition_ledger, valid_candidates,
            )
            next_problem_state["focus_proposition_ids"] = list(
                cycle_focus_proposition_ids
            )
            next_problem_state["shared_unresolved_dependencies"] = shared_dependencies
            argument_challenges = _argument_challenge_candidates(
                valid_candidates,
                working_proposition_ledger,
                graph_store.graph,
                selected_action,
            )
            retained_challenges, challenge_agenda = _advance_argument_challenge_agenda(
                previous_challenges=(
                    received_broadcast.problem_state.get(
                        "argument_challenge_candidates", [],
                    )
                    if received_broadcast.problem_state else []
                ),
                generated_challenges=argument_challenges,
                candidates=valid_candidates,
                next_cycle=cycle_number + 1,
                next_constraint=(
                    winner.constraint
                    if recorded_winner is not None
                    else received_broadcast.constraint
                ),
                active_specialists=[specialist.name for specialist in self.specialists],
                settled_issue_ids=settled_keys,
                graph=graph_store.graph,
            )
            _apply_verified_resolution_to_native_ledgers(
                retained_challenges, valid_candidates, graph_store, result,
            )
            apply_verified_challenge_supersession(
                next_problem_state, retained_challenges,
            )
            if not is_counterfactual:
                private_by_agent = {
                    str(item.get("agent", "")): dict(item)
                    for item in next_problem_state.get("workspace_contributions", []) or []
                    if isinstance(item, dict) and str(item.get("agent", ""))
                }
                for specialist in self.specialists:
                    if hasattr(specialist, "private_framework_contribution"):
                        specialist.private_framework_contribution = copy.deepcopy(
                            private_by_agent.get(specialist.name, {})
                        )
            next_problem_state["argument_challenge_candidates"] = retained_challenges
            for candidate in valid_candidates:
                apply_validator_governance_gate(candidate)
            prior_governing = governing_claim
            if (
                governing_claim is None
                or not can_supply_governing_basis(governing_claim, selected_action)
            ):
                governing_claim = self._select_governing_candidate(
                    valid_candidates, selected_action,
                )
            if (
                not is_counterfactual
                and claim_ref(prior_governing) != claim_ref(governing_claim)
            ):
                result.governing_authority_transitions.append({
                    "cycle": cycle_number,
                    "from_specialist": (
                        prior_governing.specialist if prior_governing is not None else "NONE"
                    ),
                    "to_specialist": (
                        governing_claim.specialist if governing_claim is not None else "NONE"
                    ),
                    "action": selected_action,
                    "reason": "VALIDATOR_RESOLUTION_REJECTED",
                    "trigger_constraint": received_broadcast.constraint,
                    "argumentative_event": True,
                })
                incumbent_governing_specialist = (
                    governing_claim.specialist if governing_claim is not None else ""
                )
            for candidate in valid_candidates:
                candidate.broadcast_authority = derive_broadcast_authority(
                    governing_eligible=candidate.governing_eligible,
                    is_governing_focus=(
                        governing_claim is not None and candidate is governing_claim
                    ),
                    reopen_eligible=candidate.reopen_eligible,
                    investigative_priority=candidate.investigative_priority,
                )
            _render_deliberation_progress(
                progress,
                cycle=cycle_number,
                actions=clean_actions,
                candidates=valid_candidates,
                received_challenges=received_broadcast.challenge_agenda,
                next_agenda=challenge_agenda,
            )
            if not is_counterfactual:
                result.shared_unresolved_dependencies = shared_dependencies
            next_broadcast = WorkspaceBroadcast(
                constraint=(
                    winner.constraint
                    if recorded_winner is not None
                    else received_broadcast.constraint
                ),
                intent=f"evaluate_{selected_action}",
                salient_specialist=(winner.specialist if recorded_winner is not None else ""),
                salient_action=(selected_action if recorded_winner is not None else ""),
                salient_claim=(
                    (
                        winner.investigative_claim
                        or " ".join((winner.decision_rule, winner.rationale)).strip()
                    )
                    if recorded_winner is not None
                    and winner.broadcast_authority == "INVESTIGATIVE"
                    else (
                        " ".join((winner.decision_rule, winner.rationale)).strip()
                        if recorded_winner is not None else ""
                    )
                ),
                broadcast_authority=(
                    winner.broadcast_authority if recorded_winner is not None else ""
                ),
                adjudication_status=(
                    winner.adjudication_status if recorded_winner is not None else ""
                ),
                urgency=broadcast.urgency,
                danger_probability=broadcast.danger_probability,
                unresolved=(
                    deliberative_state.primary_unresolved
                    if recorded_winner is not None else "REVIEW_MODEL_OUTPUT"
                ),
                focus_proposition_ids=cycle_focus_proposition_ids,
                challenge_agenda=tuple(challenge_agenda),
                problem_state=next_problem_state,
            )
            result.cycles.append(
                CycleRecord(
                    cycle=cycle_number,
                    broadcast=next_broadcast,
                    candidates=candidates,
                    winner=recorded_winner,
                    dissent=dissent,
                    policy=policy,
                    entropy=entropy,
                    stable_cycles=stable_cycles,
                    elapsed_seconds=elapsed,
                    received_broadcast=received_broadcast,
                    is_hypothetical=is_counterfactual,
                    execution_status=("VALID" if cycle_has_quorum else "SYSTEM_ERROR"),
                    system_error=("NONE" if cycle_has_quorum else "INSUFFICIENT_VALID_DELEGATES"),
                    policy_leader=selected_action if recorded_winner is not None else "",
                    governing_claim=(
                        governing_claim if recorded_winner is not None else None
                    ),
                    broadcast_focus=recorded_winner,
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
                    focus_proposition_ids=received_broadcast.focus_proposition_ids,
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
                    focus_proposition_ids=received_broadcast.focus_proposition_ids,
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
                    focus_proposition_ids=received_broadcast.focus_proposition_ids,
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
                    focus_proposition_ids=received_broadcast.focus_proposition_ids,
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
            if broadcast.constraint in {"SYNTHESIS_REVIEW", "PROPOSAL_REVIEW"}:
                planning_reason = "proposal review"
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
                        epistemic_status=(
                            "SCENARIO_GROUNDED"
                            if visibility.mechanism_provenance == "SCENARIO_GROUNDED"
                            else "HYPOTHETICAL"
                        ),
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
                        salient_specialist=next_broadcast.salient_specialist,
                        salient_action=next_broadcast.salient_action,
                        salient_claim=next_broadcast.salient_claim,
                        urgency=broadcast.urgency,
                        danger_probability=broadcast.danger_probability,
                        unresolved="VERIFY_ASSUMPTIONS",
                        contingency_question=(
                            f"Proposition P: {visibility.proposition} Does P alter your "
                            "reasoning or estimated harm? Why or why not?"
                        ),
                        focus_proposition_ids=next_broadcast.focus_proposition_ids,
                        problem_state=dict(next_broadcast.problem_state),
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
                access_decision = None
                if (
                    not is_counterfactual
                    and received_broadcast.constraint not in {
                        "CONSENSUS_AUDIT", "PROBLEM_STATE_AUDIT",
                    }
                ):
                    access_decision = self._investigative_reopen_decision(
                        cycle_number,
                        selected_action,
                        valid_candidates,
                        next_broadcast.problem_state,
                        fired_reopen_keys,
                    )
                if access_decision is None:
                    access_decision = self._consensus_access_decision(
                        cycle_number,
                        graph_store.graph,
                        scenario,
                        tuple(clean_actions),
                        tuple(candidates),
                        selected_action,
                        entropy,
                        dissent,
                        scenario_facts,
                        source_testimonies,
                        next_broadcast.problem_state,
                    )
                if (
                    (access_decision is None or not access_decision.admitted)
                    and not is_counterfactual
                    and received_broadcast.constraint not in {
                        "CONSENSUS_AUDIT", "PROBLEM_STATE_AUDIT",
                    }
                ):
                    focused_decision = self._problem_state_access_decision(
                        cycle_number,
                        selected_action,
                        next_broadcast.problem_state,
                    )
                    if focused_decision is not None:
                        access_decision = focused_decision
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
                        if "investigative_interrupt" in access_decision.signals:
                            key = str(
                                (access_decision.audit_variable or {}).get(
                                    "question_key", ""
                                )
                            )
                            if key:
                                fired_reopen_keys[key] = evidence_fingerprint_for(
                                    next(
                                        (
                                            candidate for candidate in valid_candidates
                                            if candidate.reopen_question_key == key
                                        ),
                                        CandidateChunk(
                                            "system", "NONE", {}, 0, 0, 0,
                                        ),
                                    ),
                                    next_broadcast.problem_state,
                                )
                        consensus_audit_attempted = True
                        semantic_node_id, semantic_node_kind, semantic_node_label = _commit_access_variable_node(
                            graph_store,
                            selected_action,
                            access_decision,
                            cycle_number=cycle_number,
                        )
                        access_decision.semantic_node_id = semantic_node_id
                        access_decision.semantic_node_kind = semantic_node_kind
                        access_decision.semantic_node_label = semantic_node_label
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
                            constraint=access_decision.content_type,
                            intent=f"audit_{selected_action}",
                            salient_specialist=next_broadcast.salient_specialist,
                            salient_action=next_broadcast.salient_action,
                            salient_claim=next_broadcast.salient_claim,
                            urgency=broadcast.urgency,
                            danger_probability=broadcast.danger_probability,
                            unresolved="VERIFY_ASSUMPTIONS",
                            contingency_question=audit_question,
                            audit_variable=dict(access_decision.audit_variable),
                            focus_proposition_ids=tuple(dict.fromkeys((
                                *next_broadcast.focus_proposition_ids,
                                *(
                                    str(item) for item in
                                    (access_decision.audit_variable or {}).get("grounded_in", [])
                                    if str(item).startswith("PROP:")
                                ),
                            ))),
                            problem_state=dict(next_broadcast.problem_state),
                        )
                        audit_broadcast = True
                        if progress:
                            progress(
                                f"  workspace access gate admitted {access_decision.content_type}: "
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
                and broadcast.constraint in {"CONSENSUS_AUDIT", "PROBLEM_STATE_AUDIT"}
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
                        salient_specialist=next_broadcast.salient_specialist,
                        salient_action=next_broadcast.salient_action,
                        salient_claim=next_broadcast.salient_claim,
                        urgency=broadcast.urgency,
                        danger_probability=broadcast.danger_probability,
                        unresolved="RESOLVE_VALUE_TENSION",
                        contingency_question=reformulation.question,
                        reformulation_context=reformulation.compact(),
                        focus_proposition_ids=next_broadcast.focus_proposition_ids,
                        problem_state=dict(next_broadcast.problem_state),
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
                        salient_specialist=next_broadcast.salient_specialist,
                        salient_action=next_broadcast.salient_action,
                        salient_claim=next_broadcast.salient_claim,
                        urgency=broadcast.urgency,
                        danger_probability=broadcast.danger_probability,
                        unresolved="TEST_REVERSAL",
                        contingency_question=(
                            "Does committed boundary P reverse the ranking? "
                            "Accept it, revise it to the smallest valid condition, or reject it."
                            + burden_probe
                        ),
                        reversal_challenge=audit_request.challenge,
                        focus_proposition_ids=next_broadcast.focus_proposition_ids,
                        problem_state=dict(next_broadcast.problem_state),
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
                obstacle_supported, fallback_available, fallback_reason = (
                    self._planning_obstacle_status(
                        scenario, broadcast, tuple(clean_actions), selected_action
                    )
                )
                if not obstacle_supported:
                    if progress:
                        progress(
                            "  planning skipped: no scenario-grounded implementation obstacle"
                        )
                elif not fallback_available:
                    planned_contexts.add(planning_key)
                    assessment = PlanningAssessment(
                        selected_action,
                        planning_reason,
                        0.0,
                        "",
                        "",
                        "",
                        valid=False,
                        error="planning skipped: no physically available fallback",
                        fallback_available=False,
                        fallback_availability_reason=fallback_reason,
                    )
                    result.planning_assessments.append(assessment)
                    if progress:
                        progress(
                            "  planning skipped: " + fallback_reason
                        )
                else:
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
                            salient_specialist=next_broadcast.salient_specialist,
                            salient_action=next_broadcast.salient_action,
                            salient_claim=next_broadcast.salient_claim,
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
                            focus_proposition_ids=next_broadcast.focus_proposition_ids,
                            problem_state=dict(next_broadcast.problem_state),
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
                if recorded_winner is not None:
                    claim_text = (
                        winner.investigative_claim
                        if winner.broadcast_authority == "INVESTIGATIVE"
                        else ""
                    ) or winner.decision_rule or winner.rationale or "NONE"
                    claim_text = " ".join(str(claim_text).split())[:120]
                    progress(
                        f"  cycle policy_leader={selected_action} "
                        f"({policy[selected_action]:.2f}); entropy={entropy:.2f}; "
                        f"governing_claim={claim_ref(governing_claim) or 'NONE'} | "
                        f"broadcast_focus={winner.specialist}:{winner.constraint} | "
                        f"authority={winner.broadcast_authority or 'NONE'} | "
                        f"claim={claim_text}"
                    )
                else:
                    progress(
                        f"  cycle policy_leader={selected_action} "
                        f"({policy[selected_action]:.2f}); entropy={entropy:.2f}; "
                        "system_status=INSUFFICIENT_VALID_DELEGATES; "
                        "broadcast_focus=NONE"
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
                and synthesis_pressure
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
                    proposal.proposal_id = f"P{len(result.synthesis_proposals)}"
                    proposal.source_agents = [
                        candidate.specialist for candidate in valid_candidates
                        if candidate.constraint in proposal.addressed_constraints
                    ]
                    proposal.feasibility_status = (
                        "PLAUSIBLE" if proposal.feasibility >= 0.70 else "UNCERTAIN"
                    )
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
                            proposal.grounding_status = "REJECTED"
                            proposal.promotion_status = "REJECTED"
                            proposal.rejection_reason = (
                                "semantic invariant failure: "
                                + "; ".join(synthesis_record.errors)
                            )
                    if proposal.accepted:
                        proposal.grounding_status = "GROUNDED"
                        proposal.promotion_status = "UNDER_REVIEW"
                        proposal.admission_status = "UNDER_REVIEW"
                        graph_store.graph.add_node(SemanticNode(
                            proposal.proposal_id,
                            "PROPOSAL",
                            proposal.action,
                            tuple(dict.fromkeys((
                                "workspace_synthesis", "synthesis_proposal",
                                *proposal.grounded_in,
                            ))),
                            {
                                "source_type": "SYNTHESIS_PROPOSAL",
                                "proposal_id": proposal.proposal_id,
                                "proposal_text": proposal.action,
                                "source_constraints": list(proposal.addressed_constraints),
                                "source_agents": list(proposal.source_agents),
                                "grounding_status": proposal.grounding_status,
                                "feasibility_status": proposal.feasibility_status,
                                "promotion_status": proposal.promotion_status,
                                "unresolved_claims": [
                                    *proposal.introduced_requirements, "feasibility",
                                ],
                            },
                        ))
                        synthesis_problem_state = build_deliberative_problem_state(
                            cycle_number,
                            clean_actions,
                            valid_candidates,
                            selected_action,
                            winner,
                            next_broadcast.problem_state,
                            graph_store.graph,
                            scenario,
                            result.synthesis_proposals,
                        )
                        broadcast = WorkspaceBroadcast(
                            constraint="PROPOSAL_REVIEW",
                            intent=f"review_{proposal.proposal_id.casefold()}",
                            urgency=broadcast.urgency,
                            danger_probability=broadcast.danger_probability,
                            unresolved="CHECK_FEASIBILITY",
                            reformulation_context=(
                                f"Review proposal {proposal.proposal_id}: {proposal.action}. "
                                "It is not a live action and cannot be selected."
                            ),
                            focus_proposition_ids=next_broadcast.focus_proposition_ids,
                            problem_state=synthesis_problem_state.to_dict(),
                        )
                        result.access_decisions.append(WorkspaceAccessDecision(
                            cycle=cycle_number,
                            content_type="PROPOSAL_REVIEW",
                            admitted=True,
                            signals=[
                                "admitted_synthesis_proposal",
                                "reversal_or_pareto_potential",
                            ],
                            question=proposal.action,
                            rationale=(
                                "An admitted proposal with reversal or Pareto-improvement "
                                "potential receives one targeted specialist review before "
                                "ordinary finalization."
                            ),
                        ))
                        if progress:
                            progress(
                                f"  synthesis stored as {proposal.proposal_id} for proposal review: {proposal.action} "
                                f"(feasibility={proposal.feasibility:.2f}; "
                                f"grounded in {', '.join(proposal.grounded_in)})"
                            )
                        cycle_number += 1
                        continue
                    if progress:
                        progress(f"  synthesis rejected: {proposal.rejection_reason}")
                    if not proposal.accepted:
                        proposal.promotion_status = "REJECTED"
                        if proposal.grounding_status == "PENDING":
                            proposal.grounding_status = "REJECTED"

            dissent_addressed = dissent is None or (
                cycle_number > 1
                and result.cycles[-2].broadcast.unresolved == dissent.unresolved
            )
            pending_proposal = self._pending_proposal_for_guaranteed_review(result)
            hard_resource_exhausted = (
                elapsed >= self.config.time_budget_seconds
                or result.halted_by in {
                    "model_call_budget",
                    "model_backend_unavailable",
                    "insufficient_valid_candidates",
                }
            )
            if (
                pending_proposal is not None
                and not proposal_review_forced
                and not hard_resource_exhausted
                and self.config.enable_synthesis
            ):
                # Mirror PROBLEM_STATE_AUDIT privilege: admitted proposals get one
                # dedicated review before ordinary termination.
                proposal_review_forced = True
                if cycle_number >= cycle_limit:
                    cycle_limit += 1
                synthesis_problem_state = build_deliberative_problem_state(
                    cycle_number,
                    clean_actions,
                    valid_candidates,
                    selected_action,
                    winner,
                    next_broadcast.problem_state,
                    graph_store.graph,
                    scenario,
                    result.synthesis_proposals,
                )
                broadcast = WorkspaceBroadcast(
                    constraint="PROPOSAL_REVIEW",
                    intent=f"review_{pending_proposal.proposal_id.casefold()}",
                    urgency=broadcast.urgency,
                    danger_probability=broadcast.danger_probability,
                    unresolved="CHECK_FEASIBILITY",
                    reformulation_context=(
                        f"Review proposal {pending_proposal.proposal_id}: "
                        f"{pending_proposal.action}. "
                        "It is not a live action and cannot be selected."
                    ),
                    focus_proposition_ids=next_broadcast.focus_proposition_ids,
                    problem_state=synthesis_problem_state.to_dict(),
                )
                result.access_decisions.append(WorkspaceAccessDecision(
                    cycle=cycle_number,
                    content_type="PROPOSAL_REVIEW",
                    admitted=True,
                    signals=[
                        "guaranteed_proposal_review",
                        "reversal_or_pareto_potential",
                    ],
                    question=pending_proposal.action,
                    rationale=(
                        "Pre-finalization guarantee: an admitted proposal with "
                        "sufficient reversal or Pareto-improvement potential must "
                        "receive one targeted specialist review before halt."
                    ),
                ))
                if progress:
                    progress(
                        f"  guaranteeing proposal review for {pending_proposal.proposal_id} "
                        f"before finalization: {pending_proposal.action}"
                    )
                cycle_number += 1
                continue
            if elapsed >= self.config.time_budget_seconds:
                result.halted_by = "time_budget"
                break
            no_op_consensus = bool(
                self.config.stop_redundant_consensus_cycles
                and cycle_number == 1
                and cycle_number < cycle_limit
                and dissent is None
                and entropy < self.config.entropy_threshold
                and len(divergent_recommendations) <= 1
                and all(
                    candidate.assumption_status not in {
                        "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
                    }
                    for candidate in valid_candidates
                )
                and not next_problem_state.get("argument_challenge_candidates")
                and not next_problem_state.get("audit_candidates")
            )
            if no_op_consensus:
                result.halted_by = "convergence"
                if progress:
                    progress(
                        "  halting after one cycle: consensus is stable enough to finalize "
                        "and no grounded argument challenge remains to test"
                    )
                break
            if (
                entropy < self.config.entropy_threshold
                and stable_cycles >= self.config.stable_cycles_required
                and dissent_addressed
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
                            focus_proposition_ids=broadcast.focus_proposition_ids,
                            problem_state=dict(broadcast.problem_state),
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
            result.judgment_status = UNRESOLVED
            result.governing_justification_status = "NONE"
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
            result.care_relationship_ledger = committed_care_assessments(
                graph_store.graph
            )
            result.authoritative_semantic_state = project_authoritative_semantic_state(
                graph_store.graph
            ).to_dict()
            result.trace_health = audit_trace_health(result)
            return result
        final = actual_cycles[-1]
        judgment_cycle = final
        if result.halted_by == "insufficient_valid_candidates":
            previous_state = (
                final.received_broadcast.problem_state
                if final.received_broadcast is not None else {}
            )
            result.deliberative_problem_state = _preserve_problem_state_after_invalid_cycle(
                previous_state, final.candidates,
            )
            final.broadcast.problem_state = copy.deepcopy(result.deliberative_problem_state)
            final.broadcast.unresolved = "REVIEW_MODEL_OUTPUT"
            recovered = [
                cycle for cycle in actual_cycles
                if cycle.execution_status == "VALID"
                and cycle.winner is not None
                and any(candidate.schema_valid for candidate in cycle.candidates)
            ]
            if not recovered:
                result.current_plurality = max(final.policy, key=final.policy.get)
                result.selected_action = "INCONCLUSIVE"
                result.judgment_status = UNRESOLVED
                result.governing_justification_status = "NONE"
                result.confidence = 0.0
                result.epistemic_confidence = 0.0
            else:
                # A failed audit cycle interrupted observation. It does not erase
                # the last cycle that actually produced a valid parliament.
                judgment_cycle = recovered[-1]
                result.current_plurality = max(
                    judgment_cycle.policy, key=judgment_cycle.policy.get,
                )
        else:
            result.current_plurality = max(final.policy, key=final.policy.get)

        if result.judgment_status not in {UNRESOLVED}:
            if not result.deliberative_problem_state:
                result.deliberative_problem_state = build_deliberative_problem_state(
                    judgment_cycle.cycle,
                    result.actions,
                    judgment_cycle.candidates,
                    result.current_plurality,
                    judgment_cycle.winner,
                    (
                        judgment_cycle.received_broadcast.problem_state
                        if judgment_cycle.received_broadcast else {}
                    ),
                    graph_store.graph,
                    scenario,
                    result.synthesis_proposals,
                ).to_dict()
            valid_final = [
                candidate for candidate in judgment_cycle.candidates
                if candidate.schema_valid
            ]
            for candidate in valid_final:
                apply_investigative_authority(
                    candidate,
                    plurality=result.current_plurality,
                    policy=judgment_cycle.policy,
                    problem_state=(
                        judgment_cycle.received_broadcast.problem_state
                        if judgment_cycle.received_broadcast else {}
                    ),
                    fired_keys=fired_reopen_keys,
                    settled_keys=list(settled_question_keys(graph_store.graph)),
                )
                apply_validator_governance_gate(candidate)
            stored_governing = judgment_cycle.governing_claim
            governing_candidate = (
                stored_governing
                if stored_governing is not None
                and can_supply_governing_basis(
                    stored_governing, result.current_plurality,
                )
                else self._select_governing_candidate(
                    valid_final,
                    result.current_plurality,
                    preferred=(
                        (judgment_cycle.broadcast_focus or judgment_cycle.winner)
                        if (
                            (judgment_cycle.broadcast_focus or judgment_cycle.winner)
                            and (
                                judgment_cycle.broadcast_focus or judgment_cycle.winner
                            ).recommended_action == result.current_plurality
                        )
                        else None
                    ),
                )
            )
            for candidate in valid_final:
                candidate.broadcast_authority = derive_broadcast_authority(
                    governing_eligible=candidate.governing_eligible,
                    is_governing_focus=(candidate is governing_candidate),
                    reopen_eligible=candidate.reopen_eligible,
                    investigative_priority=candidate.investigative_priority,
                )
            has_stable_plurality = has_unique_policy_leader(
                result.current_plurality, judgment_cycle.policy,
            )
            terminal = classify_terminal_judgment(
                plurality=result.current_plurality,
                governing=governing_candidate,
                candidates=valid_final,
                halted_by=result.halted_by,
                has_stable_plurality=has_stable_plurality,
                problem_state=(
                    judgment_cycle.broadcast.problem_state
                    if judgment_cycle.broadcast is not None else {}
                ) or (
                    judgment_cycle.received_broadcast.problem_state
                    if judgment_cycle.received_broadcast else {}
                ),
                policy=judgment_cycle.policy,
            )
            result.judgment_status = terminal.status
            result.governing_justification_status = terminal.governing_justification_status
            result.governing_attack_reason = terminal.governing_attack_reason
            if terminal.status == UNRESOLVED:
                result.selected_action = "UNRESOLVED"
            else:
                result.selected_action = terminal.policy_direction
            # Keep cycle governing_claim aligned with the terminal pick.
            judgment_cycle.governing_claim = governing_candidate
            result.confidence = clamp(
                judgment_cycle.policy.get(result.current_plurality, 0.0)
            )
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
            if judgment_cycle.stable_cycles < self.config.stable_cycles_required:
                result.epistemic_confidence *= 0.80
            if result.judgment_status == UNRESOLVED:
                result.confidence = min(result.confidence, 0.50)
                result.epistemic_confidence = min(result.epistemic_confidence, 0.35)
            elif result.judgment_status == CONTESTED_RECOMMENDATION:
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
            judgment_cycle.candidates, result.current_plurality, clean_actions
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
                judgment_cycle.dissent, result.current_plurality
            )
            if dissent_reversal and dissent_reversal not in result.reopen_conditions:
                result.reopen_conditions.append(dissent_reversal)
        publishable_judgment = result.judgment_status in {
            GOVERNED_RECOMMENDATION,
            CONTESTED_RECOMMENDATION,
        }
        governing_focus = (
            judgment_cycle.broadcast_focus or judgment_cycle.winner
        )
        if (
            governing_focus is None
            or not governing_focus.schema_valid
            or not publishable_judgment
        ):
            result.compressed_rule = f"Unavailable: deliberation halted by {result.halted_by}."
        else:
            qualifier = (
                "contestedly prefer"
                if result.judgment_status == CONTESTED_RECOMMENDATION
                else (
                    "provisionally prefer"
                    if result.halted_by in {"cycle_budget", "insufficient_valid_candidates"}
                    else "prefer"
                )
            )
            governing_candidate = judgment_cycle.governing_claim
            governing = (
                governing_candidate.decision_rule
                if governing_candidate is not None
                and result.governing_justification_status in {"ADMISSIBLE", "CONTESTED"}
                else ""
            )
            if (
                result.judgment_status == CONTESTED_RECOMMENDATION
                and result.governing_justification_status == "UNDER_ATTACK"
            ):
                result.compressed_rule = (
                    f"Current plurality leans toward {result.selected_action}, but the "
                    "available governing justification is under a live reopen-eligible "
                    "attack and cannot yet finalize the recommendation."
                )
                if result.governing_attack_reason:
                    result.compressed_rule += (
                        " Attack: " + result.governing_attack_reason
                    )
            elif result.judgment_status == CONTESTED_RECOMMENDATION and not governing:
                result.compressed_rule = (
                    f"Current plurality leans toward {result.selected_action}, but no "
                    "framework has yet supplied a sufficiently adjudicated governing "
                    "justification. The recommendation therefore remains provisional."
                )
            else:
                preserved_constraints = set(result.moral_residue)
                preserved_objections: list[str] = []
                for candidate in judgment_cycle.candidates:
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
                        candidate.specialist for candidate in judgment_cycle.candidates
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
        result.care_relationship_ledger = committed_care_assessments(
            graph_store.graph
        )
        result.authoritative_semantic_state = project_authoritative_semantic_state(
            graph_store.graph, selected_action=result.current_plurality
        ).to_dict()
        world_admission = dict(
            ((result.action_source_grounding.get("world_model") or {}).get("admission") or {})
        )
        if world_admission.get("status") == "USER_ACCEPTED_WITH_QUARANTINE":
            result.judgment_status = "PROVISIONAL_RECOMMENDATION_DEGRADED_WORLD_STATE"
            warning = (
                "The user continued after contradictory direct world effects were "
                "quarantined; reconsider after those facts are resolved."
            )
            if warning not in result.reopen_conditions:
                result.reopen_conditions.append(warning)
            result.compressed_rule = (
                result.compressed_rule + " " + warning
            ).strip()
        result.trace_health = audit_trace_health(result)
        return result
