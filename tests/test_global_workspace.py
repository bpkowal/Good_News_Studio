from __future__ import annotations

import unittest
import base64
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from global_workspace.engine import (
    WorkspaceConfig, WorkspaceEngine, _preserve_problem_state_after_invalid_cycle,
    _problem_state_audit_probe, _operative_framework_candidates,
)
from global_workspace.action_identity import compile_action_identity
from global_workspace.evidence_calibration import EvidenceCalibration
from global_workspace.legacy_bridge import RESPONSE_MARKER, consult_original_agents
from global_workspace.local_specialists import (
    CompactLocalSpecialist,
    _admitted_audit_variable,
    _candidate_from_data,
    _construct_map_errors,
    _landscape_semantic_errors,
    _scenario_closes_action_set,
    analyze_action_plan,
    assess_visibility,
    extract_allocation_actions,
    extract_acceptability_actions,
    extract_declared_action_legend,
    extract_explicit_actions,
    extract_labeled_action_legend,
    extract_scenario_facts,
    generate_failure_condition,
    infer_testimony_baseline,
    infer_testimony_stance,
    propose_actions,
    propose_problem_reformulation,
    propose_synthesis,
)
from global_workspace.memory import EpisodicMemory, summarize_specialist_contributions
from global_workspace.rawls_ledger import apply_rawls_ledger_transaction
from global_workspace.deontology_ledger import (
    DutyAssessmentProposal, apply_deontological_ledger_transaction,
    calibrate_deontological_adjudication, committed_deontological_assessments,
    render_deontological_adjudication, _party_grounding,
)
from global_workspace.utilitarian_ledger import (
    _best_evidence, apply_utilitarian_ledger_transaction,
)
from global_workspace.resolved_questions import (
    QuestionResolution,
    commit_question_resolution,
    question_resolution_index,
    resolve_audited_question,
    settled_question_keys,
)
from global_workspace.source_cache import build_source_cache_key
from global_workspace.models import CalibrationOutcome, CandidateChunk, ContingencyFeasibilityAssessment, CycleRecord, FailureCondition, PlanningAssessment, PlanningBranchEvaluation, ProblemReformulation, ProposalFrameworkReview, SynthesisProposal, VisibilityAssessment, WorkspaceAccessDecision, WorkspaceBroadcast, WorkspaceResult
from global_workspace.models import AutonomyAssessment
from global_workspace.construct_validity import collect_typed_residue
from global_workspace.contingency_graph import (
    certify_fallback_availability, compile_contingency_graph,
    validate_contingency_graph_dict,
)
from global_workspace.contingency_feasibility import verify_contingency_feasibility
from global_workspace.deliberative_state import (
    build_deliberative_problem_state,
    observe_broadcast_influence,
    opening_problem_state,
    update_broadcast_influence_persistence,
)
from global_workspace.landscape_validation import _comparative_claim_errors
from global_workspace.middleware.moral_residue import collect_moral_residue
from global_workspace.trace_health import audit_trace_health
from global_workspace.graph_transactions import SemanticGraphStore
from global_workspace.invariance import compare_label_permutation_traces
from global_workspace.openai_backend import OpenAIWorkspaceLLM
from global_workspace.presentation import render_public_judgment, _support_reason
from global_workspace.semantic_state import (
    _derive_action_dimensions, project_authoritative_semantic_state,
)
from global_workspace.semantic_graph import (
    SemanticGraph, SemanticNode, SemanticEdge, merge_graphs,
)
from global_workspace.scenario_semantics import (
    canonicalize_action_order, canonicalize_deliberation_scenario,
    compile_scenario_graph, resolved_semantic_action_keys, semantic_action_key,
    segment_scenario_clauses,
)
from global_workspace.scenario_semantics import (
    compile_action_burdens, compile_execution_obstacles,
    compile_observability_facts,
    classify_planning_failure_grounding,
    project_grounded_action_effects,
)
from global_workspace.structured_io import ModelCallUnavailable
from global_workspace_pipeline import _baseline_display_action, render_summary


class FixedSpecialist:
    def __init__(self, name: str, preferred: str, constraint: str, unresolved: str = "NONE"):
        self.name = name
        self.preferred = preferred
        self.constraint = constraint
        self.unresolved = unresolved

    def evaluate(self, scenario, actions, broadcast):
        return CandidateChunk(
            specialist=self.name,
            constraint=self.constraint,
            action_scores={action: 0.95 if action == self.preferred else 0.05 for action in actions},
            surprise=0.6,
            friction=0.7,
            confidence=0.9,
            unresolved=self.unresolved,
            recommended_action=self.preferred,
            rationale="Compact test judgment.",
            decision_rule=f"Prefer {self.preferred} under {self.constraint}.",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
            broadcast_authority="GOVERNING_CANDIDATE",
        )


class InvalidSpecialist(FixedSpecialist):
    def evaluate(self, scenario, actions, broadcast):
        chunk = super().evaluate(scenario, actions, broadcast)
        chunk.schema_valid = False
        chunk.validation_errors = ["invalid constraint"]
        chunk.confidence = 1.0
        return chunk


def typed_reversal_proposal(source: str, target: str) -> dict:
    """General valid switch used by engine-level reversal-audit tests."""
    return {
        "operation": "BOUNDARY",
        "from_action": source,
        "to_action": target,
        "clauses": [{
            "affected_action": source,
            "metric": "expected harm",
            "metric_valence": "ADVERSE",
            "comparator": "GT",
            "threshold": 10,
            "unit": "COUNT",
            "source_text": "expected harm exceeds ten",
        }],
    }


class WorkspaceEngineTests(unittest.TestCase):
    def test_rejected_framework_update_uses_last_valid_candidate_operatively(self):
        actions = ["preserve association", "mandate common schools"]
        accepted = CandidateChunk(
            specialist="virtue", constraint="CHARACTER",
            action_scores={actions[0]: 0.65, actions[1]: 0.35},
            surprise=0.2, friction=0.3, confidence=0.7,
            recommended_action=actions[0],
            rationale="Practical wisdom preserves plural forms of flourishing.",
            decision_rule="Prefer A0 when plural flourishing remains viable.",
        )
        retained: dict[str, CandidateChunk] = {}
        first = _operative_framework_candidates([accepted], retained, remember=True)
        self.assertEqual(first[0].recommended_action, actions[0])

        rejected = CandidateChunk(
            specialist="virtue", constraint="CHARACTER",
            action_scores={actions[0]: 0.05, actions[1]: 0.95},
            surprise=1.0, friction=0.9, confidence=0.9,
            recommended_action=actions[1],
            rationale="The broadcast says duty requires common schools.",
            decision_rule="Prefer A1 because duty governs.",
            framework_retention_status="UPDATE_REJECTED",
            framework_constraint_retained=True,
            framework_validation_errors=["unjustified character-state reversal"],
        )
        operative = _operative_framework_candidates([rejected], retained, remember=True)

        self.assertEqual(len(operative), 1)
        self.assertEqual(operative[0].recommended_action, actions[0])
        self.assertEqual(operative[0].action_scores, accepted.action_scores)
        self.assertEqual(operative[0].rationale, accepted.rationale)
        self.assertEqual(
            operative[0].framework_retention_status,
            "PRESERVED_AFTER_REJECTED_UPDATE",
        )
        self.assertNotEqual(operative[0].rationale, rejected.rationale)
        first_state = _operative_framework_candidates(
            [rejected], {}, remember=True,
        )
        self.assertEqual(len(first_state), 1)
        self.assertEqual(first_state[0].recommended_action, actions[1])
        self.assertTrue(first_state[0].framework_constraint_retained)
        self.assertEqual(
            first_state[0].framework_retention_status,
            "FIRST_STATE_ADMITTED_WITH_WARNINGS",
        )

    def test_first_state_grounding_warning_does_not_suppress_valid_agent(self):
        actions = ["preserve association", "mandate common schools"]
        warned = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.45, actions[1]: 0.55},
            surprise=0.2, friction=0.1, confidence=0.55,
            recommended_action=actions[1],
            rationale="The competing claims remain unresolved.",
            decision_rule="Keep authorization contested pending necessity.",
            framework_constraint_retained=False,
            framework_retention_status="COMMITTED_WITH_UNCERTAINTY",
            framework_grounding_penalty=0.35,
            framework_validation_errors=[
                "necessity lacks grounded less-restrictive-route evidence",
            ],
        )
        retained: dict[str, CandidateChunk] = {}

        operative = _operative_framework_candidates(
            [warned], retained, remember=True,
        )

        self.assertEqual(len(operative), 1)
        self.assertEqual(operative[0].specialist, "deontological")
        self.assertEqual(operative[0].recommended_action, actions[1])
        self.assertEqual(operative[0].framework_grounding_penalty, 0.35)
        self.assertEqual(
            operative[0].framework_retention_status,
            "FIRST_STATE_ADMITTED_WITH_WARNINGS",
        )
        self.assertIn("deontological", retained)

    def test_rejected_deontology_refinement_survives_as_nonvoting_insight(self):
        actions = ["permit private schools", "mandate public schools"]
        specialist = CompactLocalSpecialist("deontological", llm=None)
        specialist.previous_framework_state = {
            "adjudication_form": "KANTIAN_CLAIM_COERCION_RESOLUTION",
            "assessments": [{
                "action_id": action_id, "verdict": "CONFLICTED",
                "norm_kind": "UNIVERSAL_LAW", "relation": "CONFLICTS",
                "competing_norm": "parental autonomy",
                "competing_norm_kind": "AUTONOMY",
                "competing_relation": "CONFLICTS",
                "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED",
                "protected_party": "children",
                "competing_protected_party": "parents",
                "protected_standing": "EQUAL_JURIDICAL_STATUS",
                "competing_protected_standing": "AUTONOMY",
                "coercion_kind": "PUBLIC" if action_id == "A1" else "NONE",
                "authorization_status": "CONTESTED",
                "derivation": "UNRESOLVED", "resolution_status": "CONTESTED",
            } for action_id in ("A0", "A1")],
        }
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.45, actions[1]: 0.55},
            surprise=0.1, friction=0.1, confidence=0.55,
            recommended_action=actions[1],
            rationale="Parental consent may specify the competing claim.",
            deontological_ledger_proposal={"assessments": [{
                **item,
                "competing_norm": "parental consent right",
                "competing_norm_kind": "RIGHT",
                "competing_protected_standing": "CONSENT",
                "evidence_basis": "FRAMEWORK_ONLY",
            } for item in specialist.previous_framework_state["assessments"]]},
        )

        specialist._audit_framework_state_change(candidate, WorkspaceBroadcast(
            constraint="CHARACTER",
        ))

        self.assertEqual(candidate.framework_retention_status, "UPDATE_REJECTED")
        self.assertTrue(candidate.framework_insights)
        self.assertEqual(candidate.framework_insights[0]["insight_kind"], "REFINEMENT")
        self.assertEqual(candidate.framework_insights[0]["relation_to_committed_state"], "ELABORATES")

        retained = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.45, actions[1]: 0.55},
            surprise=0.1, friction=0.1, confidence=0.55,
            recommended_action=actions[1], rationale="Prior contested judgment.",
        )
        operative = _operative_framework_candidates(
            [candidate], {"deontological": retained}, remember=True,
        )
        state = build_deliberative_problem_state(
            2, actions, operative, actions[1], operative[0],
        ).to_dict()

        self.assertEqual(operative[0].rationale, "Prior contested judgment.")
        self.assertGreaterEqual(len(state["framework_insights"]), 2)
        self.assertTrue(all(
            item["voting_effect"] == "NONE"
            for item in state["framework_insights"]
        ))
        self.assertTrue(any(
            "parental consent right" in item["proposition"]
            for item in state["framework_insights"]
        ))

        overclaim = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.1, actions[1]: 0.9},
            surprise=0.8, friction=0.8, confidence=0.9,
            recommended_action=actions[1],
            rationale="The conflict is now resolved.",
            deontological_ledger_proposal={"assessments": [{
                **item,
                "verdict": "REQUIRED",
                "relation": "SATISFIES",
                "governing_norm": "PRIMARY",
                "priority_basis": "RESPECT_PERSONS",
                "norm": "future juridical independence",
                "evidence_basis": "FRAMEWORK_ONLY",
            } for item in specialist.previous_framework_state["assessments"]]},
        )
        specialist._audit_framework_state_change(
            overclaim, WorkspaceBroadcast(constraint="CHARACTER"),
        )

        self.assertEqual(overclaim.framework_retention_status, "UPDATE_REJECTED")
        observations = [
            item for item in overclaim.framework_insights
            if item["insight_kind"] == "NORMATIVE_OBSERVATION"
        ]
        self.assertTrue(observations)
        self.assertTrue(all(
            item["qualification"] == "FAILED_ADJUDICATION_NOT_AUTHORITY"
            for item in observations
        ))
        restored_overclaim = _operative_framework_candidates(
            [overclaim], {"deontological": retained}, remember=True,
        )[0]
        self.assertEqual(restored_overclaim.action_scores, retained.action_scores)
        self.assertEqual(restored_overclaim.rationale, retained.rationale)
        self.assertTrue(restored_overclaim.framework_insights)

    def test_framework_positions_do_not_manufacture_shared_dimensions(self):
        states = _derive_action_dimensions(
            rawlsian_positions=[{
                "canonical_action_id": "A0",
                "affected_subject": "parents",
                "dimension": "BASIC_LIBERTY",
                "effect": "WORSENS",
                "epistemic_status": "FRAMEWORK_INTERPRETATION",
                "assessment_node_id": "RAWLS_POSITION:test",
            }],
            utilitarian_consequences=[],
            action_ids=["A0"],
            grounded_action_effects=[],
        )

        self.assertEqual(states, [])

    def test_deontological_adjudication_form_is_terrain_neutral(self):
        actions = ["honor the refusal", "impose treatment"]
        graph = compile_scenario_graph(
            "A clinician must either honor a patient's refusal or impose treatment on the patient.",
            actions,
        )
        proposal = {"assessments": [
            {
                "action_id": "A0", "verdict": "PERMISSIBLE",
                "norm_kind": "AUTONOMY", "norm": "respect competent refusal",
                "relation": "CONSISTENT", "duty_bearer": "clinician",
                "protected_party": "patient", "competing_norm": "duty of care",
                "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS",
                "competing_protected_party": "patient",
                "competing_reason": "care remains an opposing obligation",
                "governing_norm": "PRIMARY", "priority_basis": "AUTONOMY",
                "priority_rule": "competent refusal governs treatment",
                "protected_standing": "CONSENT",
                "competing_protected_standing": "SPECIAL_OBLIGATION",
                "coercion_kind": "NONE", "coercive_actor": "NONE",
                "coerced_party": "NONE",
                "public_justification": "no coercion requires authorization",
                "reciprocity_status": "SATISFIED", "necessity_status": "UNKNOWN",
                "authorization_status": "NOT_APPLICABLE",
                "derivation": "CONSENT", "resolution_status": "RESOLVED",
                "evidence_basis": "SCENARIO", "reason": "honors the refusal",
            },
            {
                "action_id": "A1", "verdict": "CONFLICTED",
                "norm_kind": "AUTONOMY", "norm": "respect competent refusal",
                "relation": "CONFLICTS", "duty_bearer": "clinician",
                "protected_party": "patient", "competing_norm": "duty of care",
                "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS",
                "competing_protected_party": "patient",
                "competing_reason": "treatment may protect the patient",
                "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED",
                "priority_rule": "authorization remains unresolved",
                "protected_standing": "CONSENT",
                "competing_protected_standing": "SPECIAL_OBLIGATION",
                "coercion_kind": "INTERPERSONAL", "coercive_actor": "clinician",
                "coerced_party": "patient",
                "public_justification": "whether care authorizes overriding refusal",
                "reciprocity_status": "CONTESTED", "necessity_status": "CONTESTED",
                "authorization_status": "CONTESTED",
                "derivation": "UNRESOLVED", "resolution_status": "CONTESTED",
                "evidence_basis": "SCENARIO", "reason": "coercion remains contested",
            },
        ]}
        store = SemanticGraphStore(graph)
        record = apply_deontological_ledger_transaction(
            store, proposal, cycle=1, specialist="deontological",
            allowed_actions=tuple(actions),
        )

        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        committed = record.proposal["committed_assessments"]
        self.assertEqual(committed[0]["derivation"], "CONSENT")
        self.assertEqual(committed[1]["authorization_status"], "CONTESTED")
        self.assertNotIn("education", json.dumps(committed).casefold())
        projected = committed_deontological_assessments(store.graph)
        self.assertEqual(projected[0]["protected_party"], "patient")
        self.assertEqual(projected[0]["norm"], "respect competent refusal")
        self.assertEqual(projected[0]["duty_bearer"], "clinician")

    def test_deontological_coercion_terrain_rejects_foreign_entities(self):
        actions = ["permit private schools", "mandate public schools"]
        graph = compile_scenario_graph(
            "The state may permit private schools or mandate public schools for parents and children.",
            actions,
        )
        base = {
            "verdict": "CONFLICTED", "norm_kind": "AUTONOMY",
            "norm": "respect external freedom", "relation": "CONFLICTS",
            "duty_bearer": "state", "protected_party": "parents",
            "competing_norm": "protect children's independence",
            "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS",
            "competing_protected_party": "children",
            "competing_reason": "both claims remain live",
            "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED",
            "priority_rule": "rightful coercion remains unresolved",
            "protected_standing": "EXTERNAL_FREEDOM",
            "competing_protected_standing": "EQUAL_JURIDICAL_STATUS",
            "coercion_kind": "PUBLIC", "coercive_actor": "volunteer coordinator",
            "coerced_party": "resident", "public_justification": "unresolved",
            "reciprocity_status": "UNKNOWN", "necessity_status": "UNKNOWN",
            "authorization_status": "UNKNOWN", "derivation": "UNRESOLVED",
            "resolution_status": "CONTESTED", "evidence_basis": "SCENARIO",
            "reason": "the claims conflict",
        }
        proposal = {"assessments": [
            {"action_id": "A0", **base}, {"action_id": "A1", **base},
        ]}
        store = SemanticGraphStore(graph)
        record = apply_deontological_ledger_transaction(
            store, proposal, cycle=1, specialist="deontological",
            allowed_actions=tuple(actions),
        )

        self.assertEqual(record.status, "COMMITTED_WITH_UNCERTAINTY")
        self.assertTrue(any(
            "coercion terrain lacks current-run actor/party grounding" in error
            for error in record.errors
        ))

    def test_deontological_ledger_requires_kantian_bridge_for_rawlsian_term(self):
        actions = ["act", "decline"]
        graph = compile_scenario_graph(
            "A decision maker must act for residents or decline.", actions,
        )
        assessment = {
            "verdict": "REQUIRED", "norm_kind": "DUTY",
            "norm": "secure fair equality of opportunity", "relation": "SATISFIES",
            "duty_bearer": "decision maker", "protected_party": "residents",
            "competing_norm": "administrative convenience",
            "competing_norm_kind": "OTHER", "competing_relation": "CONFLICTS",
            "competing_protected_party": "decision maker",
            "competing_reason": "convenience opposes the duty",
            "governing_norm": "PRIMARY", "priority_basis": "PERFECT_DUTY",
            "priority_rule": "fair equality of opportunity is a perfect duty",
            "protected_standing": "OTHER", "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE", "coercive_actor": "NONE", "coerced_party": "NONE",
            "public_justification": "justice overrides convenience",
            "reciprocity_status": "UNKNOWN", "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE", "derivation": "PERFECT_DUTY",
            "resolution_status": "RESOLVED", "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "justice requires action",
        }
        store = SemanticGraphStore(graph)
        record = apply_deontological_ledger_transaction(
            store,
            {"assessments": [
                {"action_id": "A0", **assessment},
                {"action_id": "A1", **assessment},
            ]},
            cycle=1, specialist="deontological", allowed_actions=tuple(actions),
        )

        self.assertEqual(record.status, "REJECTED")
        self.assertTrue(any(
            "CROSS_FRAMEWORK_CONCEPT_UNTRANSLATED" in error
            for error in record.errors
        ))

    def test_adjudication_calibration_downgrades_unsupported_public_monopoly(self):
        actions = [
            "Constitutionalize compulsory publicly funded public-only schooling",
            "Guarantee parental choice of private schools",
        ]
        graph = compile_scenario_graph(
            "The state may abolish private schooling and mandate public-only education, "
            "restricting parents but improving children's educational opportunity, or "
            "permit private schools, preserving parental choice but worsening opportunity.",
            actions,
        )
        action = next(
            node for node in graph.nodes.values()
            if node.kind == "ACTION"
            and node.attributes.get("canonical_action_id") == "A0"
        )
        proposed = DutyAssessmentProposal.model_validate({
            "action_id": "A0", "verdict": "REQUIRED",
            "norm_kind": "UNIVERSAL_LAW", "norm": "secure equal juridical standing",
            "relation": "SATISFIES", "duty_bearer": "state",
            "protected_party": "children", "competing_norm": "parental association",
            "competing_norm_kind": "AUTONOMY", "competing_relation": "CONFLICTS",
            "competing_protected_party": "parents",
            "competing_reason": "public-only schooling restricts parental freedom",
            "governing_norm": "PRIMARY", "priority_basis": "PERFECT_DUTY",
            "priority_rule": "justice overrides associative liberty",
            "protected_standing": "EQUAL_JURIDICAL_STATUS",
            "competing_protected_standing": "EXTERNAL_FREEDOM",
            "coercion_kind": "PUBLIC", "coercive_actor": "state",
            "coerced_party": "parents",
            "public_justification": "reciprocal law enabling equal freedom",
            "reciprocity_status": "SATISFIED", "necessity_status": "NECESSARY",
            "authorization_status": "JUSTIFIED", "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED", "evidence_basis": "ACTION_GRAPH",
            "reason": "justice overrides parental liberty",
        })

        calibration = calibrate_deontological_adjudication(graph, action, proposed)

        self.assertFalse(calibration.calibrated)
        self.assertEqual(calibration.assessment.verdict, "CONFLICTED")
        self.assertEqual(calibration.assessment.necessity_status, "CONTESTED")
        self.assertEqual(calibration.assessment.reciprocity_status, "CONTESTED")
        self.assertEqual(calibration.assessment.authorization_status, "CONTESTED")
        self.assertEqual(calibration.assessment.derivation, "UNRESOLVED")
        self.assertEqual(calibration.assessment.resolution_status, "CONTESTED")

    def test_calibrated_deontology_renderer_exposes_conflict_and_open_questions(self):
        rationale, rule, conflicts, questions = render_deontological_adjudication({
            "protected_party": "children",
            "norm": "future juridical independence",
            "competing_protected_party": "parents",
            "competing_norm": "external freedom of association",
            "coercive_actor": "state",
            "coerced_party": "parents",
            "resolution_status": "CONTESTED",
            "calibration_errors": [
                "necessity lacks grounded less-restrictive-route evidence",
                "reciprocity lacks a two-party compatible-freedom derivation",
            ],
        })

        self.assertIn("competing claims", rationale)
        self.assertIn("Kantian judgment remains contested", rationale)
        self.assertNotIn("justice overrides", rationale.casefold())
        self.assertIn("less restrictive route", rule)
        self.assertEqual(len(conflicts), 1)
        self.assertTrue(any("compatible external freedom" in item for item in questions))

    def test_calibrated_deontology_support_supersedes_direct_baseline(self):
        action = "mandate public schooling"
        candidate = {
            "specialist": "deontological",
            "rationale": "The Kantian judgment remains contested.",
            "landscape_decisive_axis": "parental freedom versus child independence",
        }
        data = {
            "actions": [action, "permit private schools"],
            "source_baselines": {
                "deontological": {
                    "status": "DIRECT", "action_id": "A0",
                    "reason": "Policy B is REQUIRED and Policy A is PROHIBITED.",
                },
            },
            "source_action_legend": {"A0": action, "A1": "permit private schools"},
            "authoritative_semantic_state": {
                "deontological_assessments": [{
                    "canonical_action_id": "A0",
                    "resolution_status": "CONTESTED",
                    "calibration_errors": ["necessity lacks grounding"],
                }],
            },
        }

        reason = _support_reason(data, candidate, action, data["actions"])

        self.assertNotIn("REQUIRED", reason)
        self.assertNotIn("PROHIBITED", reason)
        self.assertIn("Kantian judgment remains contested", reason)

    def test_noncoercive_authorization_mismatch_does_not_erase_duty_ledger(self):
        actions = ["mandate public schooling", "permit private schools"]
        common = {
            "k": "AUTONOMY", "n": "protect external freedom", "rel": "CONFLICTS",
            "b": "state", "p": "children", "cn": "parental association",
            "ck": "RIGHT", "crel": "CONFLICTS", "cp": "parents",
            "crs": "parental freedom remains a competing claim",
            "gv": "UNRESOLVED", "pb": "UNRESOLVED",
            "pr": "rightful coercion remains unresolved",
            "ps": "EQUAL_JURIDICAL_STATUS", "cps": "EXTERNAL_FREEDOM",
            "coa": "state", "cop": "parents", "pj": "unresolved",
            "rec": "UNKNOWN", "nec": "UNKNOWN", "dv": "UNRESOLVED",
            "res": "CONTESTED", "g": "ACTION_GRAPH",
            "rs": "children and parents retain competing claims",
        }
        data = {
            "scores": {"A0": 0.55, "A1": 0.45}, "r": "A0",
            "c": "DUTY", "u": "RESOLVE_NORMATIVE_TENSION",
            "w": "The competing claims remain unresolved.", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.6,
            "fm": {
                "A0": "CONFLICTED: public coercion requires justification",
                "A1": "CONFLICTED: private ordering threatens children's standing",
            },
            "dp": {
                "A0": {**common, "v": "CONFLICTED", "ki": "PUBLIC", "auth": "UNKNOWN"},
                # The model's UNKNOWN authorization is harmless here because it
                # has already said this action contains no coercion.
                "A1": {**common, "v": "CONFLICTED", "ki": "NONE", "auth": "UNKNOWN"},
            },
        }

        candidate = _candidate_from_data(
            "deontological", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        assessments = candidate.deontological_ledger_proposal["assessments"]
        self.assertEqual(len(assessments), 2)
        self.assertEqual(assessments[0]["authorization_status"], "UNKNOWN")
        self.assertEqual(assessments[1]["authorization_status"], "NOT_APPLICABLE")
        self.assertFalse(any(
            "non-coercive Deontological assessment" in error
            for error in candidate.framework_validation_errors
        ))

    def test_engine_clears_scenario_local_framework_state_between_runs(self):
        seen: list[dict[str, object]] = []

        class ReusedSpecialist:
            name = "deontological"
            scenario_graph = None
            previous_framework_state = {"party": "resident", "right": "privacy"}
            previous_recommendation_id = "A1"
            previous_confidence = 0.9
            previous_context = "prior privacy case"
            assumption_status = "SUPPORTED"
            unsupported_assumption = "third-party disclosure"
            reversal_condition = "resident authorizes disclosure"
            epistemic_commitments = ["resident privacy"]

            def evaluate(self, scenario, actions, broadcast):
                seen.append(dict(self.previous_framework_state))
                return CandidateChunk(
                    specialist=self.name, constraint="DUTY",
                    action_scores={actions[0]: 0.7, actions[1]: 0.3},
                    surprise=0.1, friction=0.4, confidence=0.7,
                    recommended_action=actions[0], rationale="respects current duty",
                )

        WorkspaceEngine(
            [ReusedSpecialist()],
            WorkspaceConfig(
                max_cycles=1, min_valid_specialists=1,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_reversal_audit=False, enable_synthesis=False,
                enable_planning=False,
            ),
        ).run("Choose the current action.", ["act", "decline"])

        self.assertEqual(seen, [{}])

    def test_framework_can_translate_audit_without_copying_its_ontology(self):
        audit_variable = {
            "entity": "autonomy loss versus welfare gain",
            "relation": "COMPARATIVE_MAGNITUDE",
            "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
            "focus_action": "A0",
            "question": "Would resolving autonomy loss versus welfare gain change A0?",
        }
        candidate = _candidate_from_data(
            "care", ["universal schooling", "private enclaves"], {
                "scores": {"A0": 0.7, "A1": 0.3}, "r": "A0",
                "c": "CARE", "u": "NONE",
                "w": "dependency and autonomy remain connected", "j": "NONE",
                "ap": "TRANSLATED",
                "ax": "Care evaluates how both policies reshape dependency and relational vulnerability.",
                "ie": "WEAKENS",
            },
            WorkspaceBroadcast(
                constraint="CONSENSUS_AUDIT", audit_variable=audit_variable,
            ),
            "NONE", {},
        )

        self.assertTrue(candidate.schema_valid)
        self.assertEqual(candidate.audit_participation, "TRANSLATED")
        self.assertEqual(candidate.audit_variable["relation"], "COMPARATIVE_MAGNITUDE")
        self.assertEqual(candidate.audit_variable.get("category"), "VERIFY_FACTS")
        self.assertEqual(candidate.assumption_status, "SUPPORTED")

    def test_comparative_magnitude_question_is_mixed_uncertainty(self):
        actions = ["preserve liberty", "increase welfare"]
        util = CandidateChunk(
            specialist="utilitarian", constraint="UNCERTAINTY",
            action_scores={actions[0]: 0.51, actions[1]: 0.49},
            surprise=0.1, friction=0.2, confidence=0.4,
            recommended_action=actions[0], unresolved="VERIFY_FACTS",
            assumption_status="UNDERDETERMINED",
            unsupported_assumption="uncertain severity of autonomy loss versus welfare gain",
        )
        state = build_deliberative_problem_state(
            1, actions, [util], actions[0], util,
        ).to_dict()
        self.assertEqual(
            state["unresolved_questions"][0]["uncertainty_kind"],
            "MIXED_UNCERTAINTY",
        )

    def test_education_clause_projects_distinct_liberty_and_opportunity_effects(self):
        scenario = (
            "You are designing a foundational educational and child-welfare system. "
            "Policy A maximizes freedom of association and parental liberty, but "
            "systematically obliterates fair equality of opportunity for marginalized children. "
            "Policy B severely restricts parental choice and freedom of association, but "
            "guarantees structural equality of opportunity across socioeconomic classes."
        )
        actions = [
            "Implement Policy B mandating universal public education and banning private schooling",
            "Enact Policy A permitting private exclusive educational enclaves",
        ]
        clauses = segment_scenario_clauses(scenario)
        graph = compile_scenario_graph(scenario, actions, {
            "A0": {"clauses": [clauses[0], clauses[2]]},
            "A1": {"clauses": [clauses[0], clauses[1]]},
        })
        effects = project_grounded_action_effects(graph)
        facts = {(item.action_id, item.dimension, item.direction) for item in effects}

        self.assertIn(("A0", "LIBERTY_AUTONOMY", "WORSENS"), facts)
        self.assertIn(("A0", "OPPORTUNITY_ACCESS", "IMPROVES"), facts)
        self.assertIn(("A1", "LIBERTY_AUTONOMY", "IMPROVES"), facts)
        self.assertIn(("A1", "OPPORTUNITY_ACCESS", "WORSENS"), facts)
        self.assertFalse(any(item.source_clause_id == "C0" for item in effects))

    def test_deliberative_problem_state_is_attributed_and_complete(self):
        actions = ["preserve liberty", "raise the material floor"]
        rawls = CandidateChunk(
            specialist="rawlsian", constraint="FAIRNESS",
            action_scores={actions[0]: 0.8, actions[1]: 0.2},
            surprise=0.2, friction=0.6, confidence=0.7,
            recommended_action=actions[0],
            rationale="Basic liberty has lexical priority.",
            decision_rule="Preserve liberty before comparing material gains",
        )
        util = CandidateChunk(
            specialist="utilitarian", constraint="UNCERTAINTY",
            action_scores={actions[0]: 0.4, actions[1]: 0.6},
            surprise=0.3, friction=0.2, confidence=0.45,
            recommended_action=actions[1],
            rationale="Aggregate welfare comparison remains uncertain.",
            unresolved="VERIFY_FACTS",
            utilitarian_missing_comparison="welfare cost of liberty loss versus poverty relief",
        )

        state = build_deliberative_problem_state(
            1, actions, [rawls, util], actions[0], rawls,
        ).to_dict()

        self.assertEqual(
            set(state), {
                "cycle", "state_role", "live_actions", "agent_positions",
                "active_constraints", "active_conflicts", "unresolved_questions",
                "framework_internal_conflicts", "framework_specific_open_questions",
                "framework_insights",
                "workspace_contributions",
                "dissenting_positions", "current_plurality", "salient_position",
                "problem_delta", "support_composition", "surface_consensus",
                "deliberative_consensus", "framework_warnings",
                "audit_candidates", "resolved_questions",
                "unresolved_categories", "primary_unresolved",
                "proposals",
            },
        )
        rawls_contract = next(
            item for item in state["workspace_contributions"]
            if item["agent"] == "rawlsian"
        )
        self.assertEqual(rawls_contract["tendency"], actions[0])
        self.assertTrue(rawls_contract["core_ground"])
        self.assertIn("unresolved", rawls_contract)
        self.assertIn("defeat_conditions", rawls_contract)
        self.assertIn("new_considerations", rawls_contract)
        self.assertEqual(rawls_contract["voting_effect"], "NONE")
        self.assertEqual(
            state["salient_position"]["workspace_contribution"]["agent"],
            "rawlsian",
        )
        self.assertEqual(len(state["agent_positions"]), 2)
        self.assertEqual(state["dissenting_positions"][0]["specialist"], "utilitarian")
        self.assertTrue(all(
            item["source_type"] == "FRAMEWORK_ATTRIBUTED"
            for item in state["active_constraints"]
        ))
        self.assertEqual(
            state["salient_position"]["source_type"],
            "FRAMEWORK_ATTRIBUTED_POSITION",
        )
        self.assertIn(
            "Preserve liberty before comparing material gains",
            state["salient_position"]["conditional_justification"],
        )
        self.assertNotIn("rationale", state["agent_positions"][0])

    def test_private_workspace_contribution_returns_only_to_its_specialist(self):
        seen: dict[str, list[dict[str, object]]] = {
            "deontological": [], "care": [],
        }

        class RecurrentSpecialist:
            scenario_graph = None
            previous_framework_state = {}
            private_framework_contribution = {}

            def __init__(self, name, constraint):
                self.name = name
                self.constraint = constraint

            def evaluate(self, scenario, actions, broadcast):
                seen[self.name].append(dict(self.private_framework_contribution))
                return CandidateChunk(
                    specialist=self.name, constraint=self.constraint,
                    action_scores={actions[0]: 0.7, actions[1]: 0.3},
                    surprise=0.2, friction=0.4, confidence=0.7,
                    recommended_action=actions[0],
                    rationale=f"{self.name} preserves its own ground",
                    decision_rule=f"prefer A0 under {self.constraint}",
                    framework_specific_open_questions=[
                        f"{self.name} unresolved question",
                    ],
                    normative_reversal_threshold=f"reverse if {self.name} ground fails",
                )

        result = WorkspaceEngine(
            [
                RecurrentSpecialist("deontological", "DUTY"),
                RecurrentSpecialist("care", "CARE"),
            ],
            WorkspaceConfig(
                max_cycles=2, stable_cycles_required=99,
                min_valid_specialists=2, enable_consensus_audit=False,
                enable_problem_state_audit=False, enable_reversal_audit=False,
                enable_synthesis=False, enable_planning=False,
                enable_ev_dominance_breaker=False,
            ),
        ).run("Choose one action.", ["act", "decline"])

        self.assertEqual(seen["deontological"][0], {})
        self.assertEqual(seen["care"][0], {})
        self.assertEqual(
            seen["deontological"][1]["agent"], "deontological",
        )
        self.assertEqual(seen["care"][1]["agent"], "care")
        self.assertNotEqual(
            seen["deontological"][1]["unresolved"],
            seen["care"][1]["unresolved"],
        )
        self.assertEqual(
            len(result.cycles[0].broadcast.problem_state["workspace_contributions"]),
            2,
        )

    def test_recurrence_suspends_omitted_private_structure_instead_of_losing_it(self):
        actions = ["act", "decline"]
        first = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.6, actions[1]: 0.4},
            surprise=0.2, friction=0.2, confidence=0.6,
            recommended_action=actions[0], rationale="Public duty remains contested",
            framework_internal_conflicts=["equal freedom versus imposed coercion"],
            framework_specific_open_questions=[
                "whether a less restrictive route secures equal freedom",
                "whether the rule is reciprocal",
            ],
        )
        state1 = build_deliberative_problem_state(
            1, actions, [first], actions[0], first,
        ).to_dict()
        recurrent = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.6, actions[1]: 0.4},
            surprise=0.2, friction=0.2, confidence=0.6,
            recommended_action=actions[0], rationale="Public duty remains contested",
            framework_specific_open_questions=["whether the rule is reciprocal"],
            framework_retention_status="PRESERVED",
        )
        state2 = build_deliberative_problem_state(
            2, actions, [recurrent], actions[0], recurrent, state1,
        ).to_dict()
        contribution = state2["workspace_contributions"][0]
        self.assertIn(
            "whether a less restrictive route secures equal freedom",
            contribution["unresolved"],
        )
        suspended = [
            item for item in contribution["preservation_transitions"]
            if item["status"] == "SUSPENDED"
        ]
        self.assertTrue(any(
            item["prior_item"] == "whether a less restrictive route secures equal freedom"
            for item in suspended
        ))
        self.assertTrue(any(
            item["question"] == "whether a less restrictive route secures equal freedom"
            and item["status"] == "SUSPENDED"
            for item in state2["framework_specific_open_questions"]
        ))
        self.assertTrue(any(
            item["conflict"] == "equal freedom versus imposed coercion"
            and item["status"] == "SUSPENDED"
            for item in state2["framework_internal_conflicts"]
        ))
        self.assertIn(
            contribution["retained_issue_visibility"],
            {"EXPLICIT", "PARAPHRASED", "STRUCTURED_ONLY"},
        )
        self.assertTrue(contribution["visible_retained_issue"])

    def test_broadcast_winner_receives_same_preservation_contract(self):
        actions = ["act", "decline"]
        winner = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.2, friction=0.4, confidence=0.7,
            recommended_action=actions[0], rationale="Dependency concern is decisive",
            framework_specific_open_questions=["whether relief creates domination"],
        )
        state1 = build_deliberative_problem_state(
            1, actions, [winner], actions[0], winner,
        ).to_dict()
        compressed_winner = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.2, friction=0.4, confidence=0.7,
            recommended_action=actions[0], rationale="Dependency concern is decisive",
            framework_retention_status="PRESERVED",
        )
        state2 = build_deliberative_problem_state(
            2, actions, [compressed_winner], actions[0], compressed_winner, state1,
        ).to_dict()
        contribution = state2["workspace_contributions"][0]
        self.assertIn("whether relief creates domination", contribution["unresolved"])
        self.assertIn(
            "SUSPENDED",
            {item["status"] for item in contribution["preservation_transitions"]},
        )
        self.assertEqual(contribution["retained_issue_visibility"], "OMITTED")
        self.assertEqual(
            contribution["visible_retained_issue"],
            "whether relief creates domination",
        )

    def test_completed_cycle_broadcasts_typed_deliberative_state(self):
        observed = []

        class StateAwareSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                observed.append((self.name, broadcast.problem_state))
                return super().evaluate(scenario, actions, broadcast)

        result = WorkspaceEngine(
            [
                StateAwareSpecialist("care", "protect", "CARE"),
                StateAwareSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                stable_cycles_required=2,
                enable_consensus_audit=False,
                enable_reversal_audit=False,
                enable_synthesis=False,
                enable_planning=False,
            ),
        ).run("Choose whether to protect.", ["protect", "decline"])

        second_cycle_states = [state for _name, state in observed[2:]]
        self.assertTrue(second_cycle_states)
        self.assertTrue(all(state["live_actions"] for state in second_cycle_states))
        self.assertTrue(all(state["agent_positions"] for state in second_cycle_states))
        self.assertTrue(all(
            state["salient_position"]["source_type"]
            == "FRAMEWORK_ATTRIBUTED_POSITION"
            for state in second_cycle_states
        ))
        self.assertEqual(
            result.deliberative_problem_state["current_plurality"], "protect",
        )

    def test_broadcast_influence_is_observed_without_changing_salience(self):
        class UpdatingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                # Update only after a cycle has actually broadcast a position.
                # The opening cycle now receives a positionless problem frame,
                # so the presence of a problem state no longer marks recurrence.
                if broadcast.salient_specialist:
                    chunk.action_scores = {actions[0]: 0.72, actions[1]: 0.28}
                    chunk.preference_strength = 0.44
                    chunk.workspace_reasoning_effect = "NORMATIVE"
                    chunk.self_reported_broadcast_dependence = "MEDIUM"
                    chunk.framework_constraint_retained = True
                    chunk.framework_retention_status = "PRESERVED"
                return chunk

        result = WorkspaceEngine(
            [
                UpdatingSpecialist("care", "protect", "CARE"),
                UpdatingSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                stable_cycles_required=3,
                enable_consensus_audit=False,
                enable_reversal_audit=False,
                enable_synthesis=False,
                enable_planning=False,
            ),
        ).run("Choose whether to protect.", ["protect", "decline"])

        records = result.broadcast_influence_records
        self.assertEqual(len(records), 2)
        self.assertEqual({item["observation_lag"] for item in records}, {1})
        self.assertTrue(all(
            item["attribution_status"]
            == "OBSERVED_AFTER_EXPOSURE_NOT_CAUSALLY_ESTABLISHED"
            for item in records
        ))
        self.assertTrue(all(item["framework_retained"] for item in records))
        translated = [
            item for item in records
            if item["receiving_agent"] != item["source_agent"]
        ]
        self.assertEqual(translated[0]["framework_effect"], "TRANSLATED")
        self.assertEqual(translated[0]["influence_class"], "REFINEMENT")
        # Phase one is observational: the hand-authored salience calculation is
        # unchanged and contains no learned influence bonus.
        self.assertFalse(hasattr(result.cycles[1].winner, "influence_bonus"))

    def test_framework_capture_gates_observed_influence(self):
        actions = ["protect", "decline"]
        source = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"protect": 0.8, "decline": 0.2},
            surprise=0.2, friction=0.6, confidence=0.8,
            recommended_action="protect", rationale="Respect persons.",
        )
        first = build_deliberative_problem_state(
            1, actions, [source], "protect", source,
        ).to_dict()
        captured = CandidateChunk(
            specialist="rawlsian", constraint="WELFARE_MAXIMIZATION",
            action_scores={"protect": 0.9, "decline": 0.1},
            surprise=0.3, friction=0.7, confidence=0.7,
            recommended_action="protect",
            framework_constraint_retained=False,
            framework_retention_status="LOST",
            self_reported_broadcast_dependence="HIGH",
        )
        second = build_deliberative_problem_state(
            2, actions, [captured], "protect", captured, first,
        )
        record = observe_broadcast_influence(first, second, [captured])[0]
        self.assertEqual(record.framework_effect, "CAPTURED")
        self.assertEqual(record.influence_class, "FRAMEWORK_CAPTURE")

    def test_broadcast_influence_persistence_tracks_surviving_update(self):
        actions = ["protect", "decline"]
        source = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"protect": 0.8, "decline": 0.2},
            surprise=0.2, friction=0.6, confidence=0.8,
            recommended_action="protect", rationale="Respect persons.",
        )
        care_before = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"protect": 0.4, "decline": 0.6},
            surprise=0.2, friction=0.2, confidence=0.6,
            recommended_action="decline",
        )
        first = build_deliberative_problem_state(
            1, actions, [source, care_before], "protect", source,
        ).to_dict()
        care_after = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"protect": 0.65, "decline": 0.35},
            surprise=0.2, friction=0.3, confidence=0.65,
            recommended_action="protect",
            workspace_reasoning_effect="NORMATIVE",
            self_reported_broadcast_dependence="MEDIUM",
            framework_retention_status="PRESERVED",
        )
        second = build_deliberative_problem_state(
            2, actions, [source, care_after], "protect", care_after, first,
        )
        records = [
            item.to_dict()
            for item in observe_broadcast_influence(first, second, [source, care_after])
        ]
        third = build_deliberative_problem_state(
            3, actions, [source, care_after], "protect", care_after, second.to_dict(),
        )
        update_broadcast_influence_persistence(records, third)
        care_record = next(
            item for item in records if item["receiving_agent"] == "care"
        )
        self.assertEqual(care_record["framework_effect"], "TRANSLATED")
        self.assertEqual(care_record["persistence"], "PERSISTENT")

    def test_live_tension_engagement_is_bounded_and_integrity_gated(self):
        problem_state = {
            "unresolved_questions": [{
                "source_specialist": "care", "category": "VERIFY_FACTS",
                "question_key": "QUESTION:1",
            }],
            "active_conflicts": [{
                "conflict_type": "ACTION_PREFERENCE",
                "specialists": ["care", "duty"], "conflict_key": "CONFLICT:1",
            }],
        }
        engaged = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"protect": 0.7, "decline": 0.3},
            surprise=0.2, friction=0.3, confidence=0.7,
            recommended_action="protect", unresolved="NONE",
            workspace_reasoning_effect="FACTUAL",
            workspace_proposition_response="QUALIFY",
        )
        score = WorkspaceEngine._tension_engagement(engaged, problem_state)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(
            set(engaged.tension_target_keys), {"QUESTION:1", "CONFLICT:1"},
        )

        captured = CandidateChunk(
            specialist="care", constraint="FOREIGN_FRAMEWORK",
            action_scores={"protect": 0.9, "decline": 0.1},
            surprise=0.9, friction=0.9, confidence=0.9,
            recommended_action="protect",
            workspace_reasoning_effect="BOTH",
            workspace_proposition_response="ACCEPT",
            framework_constraint_retained=False,
        )
        self.assertEqual(
            WorkspaceEngine._tension_engagement(captured, problem_state), 0.0,
        )

    def test_problem_delta_tracks_constraint_and_question_resolution(self):
        actions = ["protect", "decline"]
        care_first = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"protect": 0.7, "decline": 0.3},
            surprise=0.2, friction=0.4, confidence=0.55,
            recommended_action="protect", unresolved="VERIFY_FACTS",
            unsupported_assumption="whether support reaches dependent residents",
        )
        duty_first = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"protect": 0.3, "decline": 0.7},
            surprise=0.2, friction=0.4, confidence=0.7,
            recommended_action="decline",
        )
        first = build_deliberative_problem_state(
            1, actions, [care_first, duty_first], "protect", care_first,
        ).to_dict()

        care_second = CandidateChunk(
            specialist="care", constraint="DEPENDENCY",
            action_scores={"protect": 0.85, "decline": 0.15},
            surprise=0.2, friction=0.7, confidence=0.82,
            recommended_action="protect", unresolved="NONE",
            workspace_reasoning_effect="FACTUAL",
        )
        duty_second = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"protect": 0.75, "decline": 0.25},
            surprise=0.2, friction=0.5, confidence=0.78,
            recommended_action="protect",
        )
        second = build_deliberative_problem_state(
            2, actions, [care_second, duty_second], "protect", care_second, first,
        ).to_dict()
        delta = second["problem_delta"]

        self.assertEqual(delta["from_cycle"], 1)
        self.assertEqual(delta["to_cycle"], 2)
        self.assertTrue(any(
            change["specialist"] == "care"
            and change["previous_constraint"] == "CARE"
            and change["current_constraint"] == "DEPENDENCY"
            for change in delta["constraint_changes"]
        ))
        self.assertTrue(any(
            item["constraint"] == "DEPENDENCY" for item in delta["new_constraints"]
        ))
        self.assertTrue(any(
            item["constraint"] == "CARE" for item in delta["removed_constraints"]
        ))
        self.assertTrue(any(
            item["resolution_type"] == "RESOLVED_BY_SCENARIO_FACT"
            for item in delta["resolved_questions"]
        ))
        self.assertTrue(any(
            item["conflict_type"] == "ACTION_PREFERENCE"
            and item["resolution_type"] == "RESOLVED_BY_DELIBERATIVE_CONVERGENCE"
            for item in delta["resolved_conflicts"]
        ))

    def test_problem_state_tracks_framework_local_tensions_without_promoting_them(self):
        actions = ["common schools", "private schools"]
        rawls = CandidateChunk(
            specialist="rawlsian", constraint="FAIRNESS",
            action_scores={actions[0]: 0.6, actions[1]: 0.4},
            surprise=0.2, friction=0.7, confidence=0.6,
            recommended_action=actions[0],
            framework_internal_conflicts=[
                "basic liberty versus fair equality of opportunity",
            ],
            framework_specific_open_questions=[
                "status of parental educational association",
            ],
        )
        state = build_deliberative_problem_state(
            1, actions, [rawls], actions[0], rawls,
        ).to_dict()

        local_conflict = state["framework_internal_conflicts"][0]
        local_question = state["framework_specific_open_questions"][0]
        self.assertEqual(local_conflict["source_specialist"], "rawlsian")
        self.assertEqual(
            local_conflict["source_type"],
            "FRAMEWORK_ATTRIBUTED_INTERNAL_CONFLICT",
        )
        self.assertEqual(local_question["source_specialist"], "rawlsian")
        self.assertFalse(any(
            item.get("conflict_key") == local_conflict["conflict_key"]
            for item in state["active_conflicts"]
        ))
        self.assertFalse(any(
            item.get("question_key") == local_question["question_key"]
            for item in state["unresolved_questions"]
        ))

    def test_problem_delta_distinguishes_internal_conflict_reframing(self):
        actions = ["common schools", "private schools"]
        first_candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_internal_conflicts=[
                "parental freedom versus the child's equal independence",
            ],
        )
        first = build_deliberative_problem_state(
            1, actions, [first_candidate], actions[0], first_candidate,
        ).to_dict()
        second_candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_internal_conflicts=[
                "parental autonomy versus children's claims to equal freedom",
            ],
        )
        second = build_deliberative_problem_state(
            2, actions, [second_candidate], actions[0], second_candidate, first,
        ).to_dict()
        delta = second["problem_delta"]

        self.assertEqual(len(delta["reframed_internal_conflicts"]), 1)
        self.assertEqual(delta["new_internal_conflicts"], ())
        self.assertEqual(delta["resolved_internal_conflicts"], ())

        old_issue = first["framework_internal_conflicts"][0]
        new_issue = second["framework_internal_conflicts"][0]
        self.assertEqual(old_issue["issue_key"], new_issue["issue_key"])
        self.assertIn(
            old_issue["conflict"], new_issue["wording_history"],
        )

    def test_framework_issue_type_validation_reclassifies_question_from_conflict_field(self):
        actions = ["common schools", "private schools"]
        candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_internal_conflicts=["whether less restrictive means exist"],
        )
        state = build_deliberative_problem_state(
            1, actions, [candidate], actions[0], candidate,
        ).to_dict()

        self.assertEqual(state["framework_internal_conflicts"], [])
        self.assertEqual(len(state["framework_specific_open_questions"]), 1)
        issue = state["framework_specific_open_questions"][0]
        self.assertEqual(issue["issue_type"], "OPEN_QUESTION")
        self.assertEqual(issue["question"], "whether less restrictive means exist")

    def test_equivalent_open_question_keeps_one_live_wording_and_prior_history(self):
        actions = ["common schools", "private schools"]
        first_candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_specific_open_questions=[
                "whether a less restrictive route can secure equal freedom",
            ],
        )
        first = build_deliberative_problem_state(
            1, actions, [first_candidate], actions[0], first_candidate,
        ).to_dict()
        second_candidate = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.55, actions[1]: 0.45},
            surprise=0.2, friction=0.7, confidence=0.55,
            recommended_action=actions[0],
            framework_specific_open_questions=[
                "whether less restrictive means secure equal freedom",
            ],
        )
        second = build_deliberative_problem_state(
            2, actions, [second_candidate], actions[0], second_candidate, first,
        ).to_dict()

        self.assertEqual(len(second["framework_specific_open_questions"]), 1)
        old_issue = first["framework_specific_open_questions"][0]
        new_issue = second["framework_specific_open_questions"][0]
        self.assertEqual(old_issue["issue_key"], new_issue["issue_key"])
        self.assertEqual(
            new_issue["question"],
            "whether less restrictive means secure equal freedom",
        )
        self.assertIn(old_issue["question"], new_issue["wording_history"])

    def test_conditional_baseline_renderer_uses_provisional_action(self):
        baseline = {
            "status": "CONDITIONAL",
            "action_id": "NONE",
            "provisional_action_id": "A0",
        }

        self.assertEqual(_baseline_display_action(baseline), "A0")

    def test_single_normatively_contested_delegate_does_not_block_convergence(self):
        class ContestedCare(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.assumption_status = "NORMATIVELY_CONTESTED"
                chunk.unresolved = "RESOLVE_NORMATIVE_TENSION"
                chunk.selection_status = "PROVISIONAL"
                return chunk

        result = WorkspaceEngine(
            [
                ContestedCare("care", "protect", "CARE"),
                FixedSpecialist("virtue", "protect", "CHARACTER"),
                FixedSpecialist("deontological", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                stable_cycles_required=1,
                consensus_audit_min_signals=99,
                enable_reversal_audit=False,
                enable_synthesis=False,
            ),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(result.selected_action, "protect")
        self.assertIn("NORMATIVE_ADJUDICATION", result.reopen_conditions)

    def test_autonomy_audit_records_normative_burden_without_lowering_confidence(self):
        class AutonomyAwareSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                return chunk

        autonomy = AutonomyAssessment(
            {"protect": "COVENANT_BREACH", "decline": "NONE"},
            {"protect": False, "decline": False},
            {"protect": "Breaking the promise to preserve throughput.", "decline": ""},
            "decline",
            surcharge_multiplier=0.5,
        )

        result = WorkspaceEngine(
            [AutonomyAwareSpecialist("care", "protect", "CARE")],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        ).run(
            "Choose protect or decline.",
            ["protect", "decline"],
            assess_autonomy=lambda scenario, actions: autonomy,
        )

        candidate = result.cycles[0].candidates[0]
        self.assertEqual(candidate.coercion_tag, "COVENANT_BREACH")
        self.assertEqual(candidate.coercion_surcharge, 0.5)
        self.assertAlmostEqual(candidate.epistemic_confidence, 0.9)

    def test_uniform_autonomy_burden_is_recorded_without_surcharge(self):
        class AutonomyAwareSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                return chunk

        autonomy = AutonomyAssessment(
            {"policy b": "RIGHTS_INTRUSION", "policy a": "RIGHTS_INTRUSION"},
            {"policy b": False, "policy a": False},
            {"policy b": "Zoning constrains use.", "policy a": "Zoning constrains use."},
            surcharge_multiplier=0.70,
        )
        result = WorkspaceEngine(
            [AutonomyAwareSpecialist("rawlsian", "policy b", "FAIRNESS")],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        ).run(
            "Choose one zoning policy.", ["policy b", "policy a"],
            assess_autonomy=lambda scenario, actions: autonomy,
        )

        candidate = result.cycles[0].candidates[0]
        self.assertEqual(candidate.coercion_tag, "RIGHTS_INTRUSION")
        self.assertEqual(candidate.coercion_surcharge, 0.0)

    def test_conditional_or_underdetermined_stability_can_still_converge(self):
        class CautiousSpecialist(FixedSpecialist):
            def __init__(self, *args, status="CONDITIONAL", **kwargs):
                super().__init__(*args, **kwargs)
                self.status = status

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.assumption_status = self.status
                chunk.selection_status = "PROVISIONAL"
                return chunk

        result = WorkspaceEngine(
            [
                CautiousSpecialist("care", "protect", "CARE", status="CONDITIONAL"),
                CautiousSpecialist("virtue", "protect", "CHARACTER", status="UNDERDETERMINED"),
                FixedSpecialist("deontological", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=4,
                stable_cycles_required=2,
                consensus_audit_min_signals=99,
                enable_reversal_audit=False,
                enable_synthesis=False,
            ),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(result.selected_action, "protect")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertGreaterEqual(result.cycles[-1].stable_cycles, 2)

    def test_transient_delegate_timeout_is_excluded_without_crashing_cycle(self):
        class TimedOutSpecialist:
            name = "deontological"

            def evaluate(self, scenario, actions, broadcast):
                raise ModelCallUnavailable(
                    "OpenAI model call timed out", category="timeout", terminal=False
                )

        result = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                TimedOutSpecialist(),
                FixedSpecialist("virtue", "protect", "CHARACTER"),
            ],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual(result.selected_action, "protect")
        unavailable = result.cycles[0].candidates[1]
        self.assertFalse(unavailable.schema_valid)
        self.assertEqual(unavailable.constraint, "NONE")
        self.assertEqual(unavailable.delegate_status, "MODEL_ERROR")
        self.assertIn(
            "DELEGATE_MODEL_UNAVAILABLE",
            [finding.code for finding in result.trace_health],
        )

    def test_historical_dissent_that_supports_final_policy_is_not_residue(self):
        action_a, action_b = "choose A", "choose B"
        historical_supporter = CandidateChunk(
            "rawlsian", "FAIRNESS", {action_a: 0.2, action_b: 0.8},
            0.2, 0.6, 0.8, recommended_action=action_b,
        )
        cycle = CycleRecord(
            1, WorkspaceBroadcast(), [historical_supporter], historical_supporter,
            historical_supporter, {action_a: 0.6, action_b: 0.4}, 0.9, 1, 1.0,
        )
        self.assertEqual(collect_moral_residue([cycle], action_b), [])
        self.assertEqual(collect_typed_residue([cycle], action_b), [])

    def test_stance_completeness_is_relative_to_each_cycle_action_set(self):
        action_a, action_b, synthesis = "choose A", "choose B", "combine safeguards"
        candidate = CandidateChunk(
            "care", "CARE", {action_a: 0.7, action_b: 0.3}, 0.2, 0.5, 0.8,
            recommended_action=action_a,
            action_admissibility={action_a: "PERMISSIBLE", action_b: "REJECTED"},
        )
        cycle = CycleRecord(
            1, WorkspaceBroadcast(), [candidate], candidate, None,
            {action_a: 0.7, action_b: 0.3}, 0.8, 1, 1.0,
        )
        result = WorkspaceResult(
            "A test", [action_a, action_b, synthesis], cycles=[cycle]
        )
        findings = audit_trace_health(result)
        self.assertFalse(any(
            finding.code == "STANCE_MEASUREMENT_INCOMPLETE"
            for finding in findings
        ))

    def test_virtue_structural_incompleteness_is_distinct_from_retention_loss(self):
        candidate = CandidateChunk(
            "virtue", "CHARACTER", {"A0": 0.55, "A1": 0.45}, 0.2, 0.3, 0.7,
            recommended_action="A0",
            framework_retention_status="LOST",
            framework_validation_errors=[
                "framework map for A0 lacks framework-specific grounds",
                "framework map for A1 lacks framework-specific grounds",
            ],
        )
        cycle = CycleRecord(
            1, WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"), [candidate],
            candidate, None, {"A0": 0.55, "A1": 0.45}, 0.7, 1, 1.0,
        )
        result = WorkspaceResult(
            "A test", ["A0", "A1"], cycles=[cycle]
        )
        findings = audit_trace_health(result)
        self.assertTrue(any(
            finding.code == "FRAMEWORK_MAP_INCOMPLETE"
            for finding in findings
        ), [finding.code for finding in findings])
        self.assertFalse(any(
            finding.code == "FRAMEWORK_RETENTION_LOST"
            for finding in findings
        ), [finding.code for finding in findings])

    def test_visibility_audit_is_nonvoting_action_confidence_adjustment(self):
        actions = ["rely on visible reports", "serve the excluded group"]
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utility", actions[0], "UNCERTAINTY"),
                FixedSpecialist("care", actions[1], "CARE"),
            ],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        )

        def visibility(_scenario, current_actions):
            return VisibilityAssessment(
                True,
                True,
                "excluded residents",
                "unequal reporting infrastructure hides their needs",
                "excluded residents cannot submit reports",
                {current_actions[0]: 0.65, current_actions[1]: 1.0},
                activated=True,
            )

        result = engine.run(
            "Excluded residents cannot submit reports.", actions,
            assess_visibility=visibility,
        )
        self.assertEqual(len(result.cycles[0].candidates), 2)
        self.assertEqual(result.current_plurality, actions[1])
        self.assertTrue(result.visibility_assessments[0].activated)
        answer = render_public_judgment(result)
        self.assertIn("# Ethical Parliament Judgment", answer)
        self.assertIn("serve the excluded group", answer)
        self.assertNotIn("×0.65", answer)
        self.assertNotIn("Hidden-harm audit", answer)

    def test_summary_separates_final_policy_from_broadcast_context(self):
        actions = ["route water", "route hospital"]
        winner = CandidateChunk(
            "virtue", "CHARACTER", {"route water": 0.19, "route hospital": 0.81},
            0.4, 0.6, 0.8, recommended_action="route hospital",
        )
        dissent = CandidateChunk(
            "care", "CARE", {"route water": 0.81, "route hospital": 0.19},
            0.4, 0.6, 0.7, recommended_action="route water",
        )
        cycle = CycleRecord(
            cycle=4,
            broadcast=WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            candidates=[winner, dissent],
            winner=winner,
            dissent=dissent,
            policy={"route water": 0.19, "route hospital": 0.81},
            entropy=0.42,
            stable_cycles=2,
            elapsed_seconds=3.5,
            received_broadcast=WorkspaceBroadcast(constraint="CARE"),
            is_hypothetical=False,
        )
        result = WorkspaceResult(
            scenario="test",
            actions=actions,
            cycles=[cycle],
            selected_action="route hospital",
            confidence=0.81,
            current_plurality="route hospital",
            epistemic_confidence=0.72,
        )
        summary = render_summary(result)
        public = render_public_judgment(result)
        self.assertEqual(summary, public)
        self.assertIn("**Policy support:** 0.81", summary)
        self.assertIn("**route hospital.**", summary)
        self.assertIn("## Deliberation Map", summary)
        self.assertNotIn("Final winning constraint:", summary)
        self.assertNotIn("Previous broadcast context:", summary)
        self.assertNotIn("Last emitted broadcast:", summary)

    def test_summary_surfaces_typed_audit_variable(self):
        actions = ["decline", "share"]
        cycle = CycleRecord(
            cycle=2,
            broadcast=WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
            candidates=[],
            winner=CandidateChunk(
                "care", "CARE", {"decline": 0.6, "share": 0.4}, 0.0, 0.0, 0.6,
                recommended_action="decline",
            ),
            dissent=CandidateChunk(
                "care", "CARE", {"decline": 0.6, "share": 0.4}, 0.0, 0.0, 0.6,
                recommended_action="share",
            ),
            policy={"decline": 0.6, "share": 0.4},
            entropy=0.42,
            stable_cycles=1,
            elapsed_seconds=2.1,
            received_broadcast=WorkspaceBroadcast(
                constraint="CONSENSUS_AUDIT",
                audit_variable={
                    "entity": "volunteer coordinator",
                    "relation": "THIRD_PARTY_STATUS",
                    "possible_values": [
                        "EXTERNAL_THIRD_PARTY",
                        "AUTHORIZED_INTERNAL_AGENT",
                        "UNKNOWN",
                    ],
                    "focus_action": "share",
                    "question": "Classify the entity 'volunteer coordinator' as EXTERNAL_THIRD_PARTY (third party), AUTHORIZED_INTERNAL_AGENT (authorized internal agent), or UNKNOWN, then state whether the recommendation for 'share' changes under the AUTHORIZED_INTERNAL_AGENT reading.",
                    "required_response": {
                        "counterfactual_anchor": "AUTHORIZED_INTERNAL_AGENT",
                        "allowed_effects": [
                            "NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED",
                        ],
                    },
                },
            ),
            is_hypothetical=False,
        )
        result = WorkspaceResult(
            scenario="test",
            actions=actions,
            cycles=[cycle],
            selected_action="decline",
            confidence=0.81,
            current_plurality="decline",
            epistemic_confidence=0.72,
            access_decisions=[
                WorkspaceAccessDecision(
                    cycle=2,
                    content_type="CONSENSUS_AUDIT",
                    admitted=True,
                    signals=["third_party_status"],
                    question=cycle.received_broadcast.audit_variable["question"],
                    audit_variable=cycle.received_broadcast.audit_variable,
                )
            ],
        )
        summary = render_summary(result)
        # Audit plumbing stays in the JSON trace, not the default brief.
        self.assertNotIn("typed audit variable", summary)
        self.assertNotIn("THIRD_PARTY_STATUS", summary)
        self.assertNotIn("counterfactual_anchor", summary)
        self.assertIn("# Ethical Parliament Judgment", summary)

    def test_summary_humanizes_residue_and_excludes_model_retry_from_reopen(self):
        actions = ["preserve liberty", "impose restriction"]
        winner = CandidateChunk(
            "care", "CARE", {actions[0]: 0.3, actions[1]: 0.7},
            0.2, 0.4, 0.7, recommended_action=actions[1],
            rationale="Protects dependent children.",
        )
        rights = CandidateChunk(
            "deontological", "RIGHTS", {actions[0]: 0.8, actions[1]: 0.2},
            0.3, 0.6, 0.6, recommended_action=actions[0],
            rationale="The restriction lacks reciprocal public justification.",
        )
        cycle = CycleRecord(
            cycle=2,
            broadcast=WorkspaceBroadcast(
                constraint="CARE",
                problem_state={"unresolved_questions": [{
                    "category": "RESOLVE_NORMATIVE_TENSION",
                    "question": "Whether a less restrictive route can protect children",
                }]},
            ),
            candidates=[winner, rights], winner=winner, dissent=rights,
            policy={actions[0]: 0.3, actions[1]: 0.7},
            entropy=0.4, stable_cycles=1, elapsed_seconds=1.0,
        )
        result = WorkspaceResult(
            scenario="test", actions=actions, cycles=[cycle],
            selected_action=actions[1], current_plurality=actions[1],
            confidence=0.7, epistemic_confidence=0.6,
            moral_residue=["RIGHTS"],
            reopen_conditions=["RESOLVE_NORMATIVE_TENSION", "RETRY_MODEL_CALL"],
        )

        summary = render_summary(result)
        public = render_public_judgment(result)

        self.assertEqual(summary, public)
        self.assertIn("lacks reciprocal public justification", summary)
        self.assertIn("normative adjudication establishes a strict duty", summary.casefold())
        self.assertNotIn("RETRY_MODEL_CALL", summary)
        self.assertNotIn("RESOLVE_NORMATIVE_TENSION", summary)
        self.assertIn("## Deliberation Map", summary)
        self.assertNotIn("Unresolved moral considerations: RIGHTS", public)

    def test_summary_renders_accepted_synthesis_from_its_own_schema(self):
        actions = ["coerce donors", "wait for volunteers", "recruit volunteers rapidly"]
        winner = CandidateChunk(
            "care", "CARE", {actions[0]: 0.2, actions[1]: 0.5, actions[2]: 0.8},
            0.2, 0.6, 0.8, recommended_action=actions[2],
        )
        cycle = CycleRecord(
            cycle=3,
            broadcast=WorkspaceBroadcast(constraint="SYNTHESIS_REVIEW"),
            candidates=[winner], winner=winner, dissent=None,
            policy={actions[0]: 0.2, actions[1]: 0.5, actions[2]: 0.8},
            entropy=0.3, stable_cycles=1, elapsed_seconds=1.0,
        )
        result = WorkspaceResult(
            scenario="test", actions=actions, cycles=[cycle],
            selected_action=actions[2], current_plurality=actions[2],
            confidence=0.8, epistemic_confidence=0.7,
            synthesis_proposals=[SynthesisProposal(
                actions[2], ["care", "deontological"], ["CARE", "RIGHTS"],
                0.77, "Respects autonomy while accelerating supply",
                accepted=True, admission_status="ADMITTED",
            )],
        )

        summary = render_summary(result)

        # Accepted synthesis that became the recommendation is the policy leader,
        # not a separate diagnostic "Synthesis:" appendix line.
        self.assertIn(actions[2], summary)
        self.assertIn("**Policy leader:**", summary)
        self.assertNotIn("feasibility=0.77", summary)
        self.assertNotIn("addressed constraints:", summary)

    def test_summary_explains_shared_contingency_failure_without_contradiction(self):
        winner = CandidateChunk(
            "care",
            "CARE",
            {"A0": 0.6, "A1": 0.4},
            0.2,
            0.5,
            0.7,
            recommended_action="A0",
        )
        cycle = CycleRecord(
            cycle=1,
            broadcast=WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            candidates=[winner],
            winner=winner,
            dissent=None,
            policy={"A0": 0.6, "A1": 0.4},
            entropy=0.4,
            stable_cycles=1,
            elapsed_seconds=1.0,
        )
        result = WorkspaceResult(
            scenario="test",
            actions=["A0", "A1"],
            cycles=[cycle],
            selected_action="A0",
            confidence=0.6,
            current_plurality="A0",
            epistemic_confidence=0.7,
            contingency_feasibility_assessments=[
                ContingencyFeasibilityAssessment(
                    synthesis_action="combine safeguards",
                    predicate_label="the safeguard works",
                    fallback_statuses={"A0": "AVAILABLE", "A1": "AVAILABLE"},
                    fallback_reasons={
                        "A0": "A0 remains executable",
                        "A1": "A1 remains executable",
                    },
                    evidence_bases={
                        "A0": "SCENARIO_STRUCTURE",
                        "A1": "SCENARIO_STRUCTURE",
                    },
                    shared_failure=True,
                    valid=True,
                    approved=False,
                    error="shared failure: the failure removes a capability used by the synthesis and at least one fallback; the original fallbacks may still remain individually available",
                )
            ],
        )
        summary = render_summary(result)
        # Contingency feasibility plumbing stays in the JSON trace.
        self.assertNotIn("Fallback availability:", summary)
        self.assertNotIn("Shared-failure check:", summary)
        self.assertIn("# Ethical Parliament Judgment", summary)

    def test_public_judgment_explains_shared_contingency_failure_without_contradiction(self):
        winner = CandidateChunk(
            "care",
            "CARE",
            {"A0": 0.6, "A1": 0.4},
            0.2,
            0.5,
            0.7,
            recommended_action="A0",
        )
        cycle = CycleRecord(
            cycle=1,
            broadcast=WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            candidates=[winner],
            winner=winner,
            dissent=None,
            policy={"A0": 0.6, "A1": 0.4},
            entropy=0.4,
            stable_cycles=1,
            elapsed_seconds=1.0,
        )
        result = WorkspaceResult(
            scenario="test",
            actions=["A0", "A1"],
            cycles=[cycle],
            selected_action="A0",
            confidence=0.6,
            current_plurality="A0",
            epistemic_confidence=0.7,
            contingency_feasibility_assessments=[
                ContingencyFeasibilityAssessment(
                    synthesis_action="combine safeguards",
                    predicate_label="the safeguard works",
                    fallback_statuses={"A0": "AVAILABLE", "A1": "AVAILABLE"},
                    fallback_reasons={
                        "A0": "A0 remains executable",
                        "A1": "A1 remains executable",
                    },
                    evidence_bases={
                        "A0": "SCENARIO_STRUCTURE",
                        "A1": "SCENARIO_STRUCTURE",
                    },
                    shared_failure=True,
                    valid=True,
                    approved=False,
                    error="shared failure: the failure removes a capability used by the synthesis and at least one fallback; the original fallbacks may still remain individually available",
                )
            ],
        )
        answer = render_public_judgment(result)
        self.assertNotIn("Independent contingency feasibility", answer)
        self.assertNotIn("Fallback availability:", answer)
        self.assertIn("# Ethical Parliament Judgment", answer)

    def test_source_cache_key_depends_on_scenario_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path_a = Path(directory) / "a.json"
            path_b = Path(directory) / "b.json"
            path_a.write_text(json.dumps({"ethical_question": "Case A"}), encoding="utf-8")
            path_b.write_text(json.dumps({"ethical_question": "Case B"}), encoding="utf-8")
            key_a = build_source_cache_key("v1", "same query", "scenario-a", path_a)
            key_b = build_source_cache_key("v1", "same query", "scenario-b", path_b)
            self.assertNotEqual(key_a, key_b)
            self.assertEqual(
                key_a,
                build_source_cache_key("v1", "same query", "scenario-a", path_a),
            )

    def test_long_actions_survive_losslessly_in_state_objects(self):
        long_action = (
            "illegally allocate the remaining doses to frontline transport workers "
            "to break the primary transmission chain and prevent five times as "
            "many total deaths across the city"
        )
        proposal = SynthesisProposal(
            action=long_action,
            grounded_in=["utilitarian", "deontological"],
            addressed_constraints=["IMMINENT_HARM", "DUTY"],
            feasibility=0.8,
            rationale="test",
        )
        branch = PlanningBranchEvaluation(
            cycle=1,
            origin_action=long_action,
            condition="if the transmission chain can be broken without inventing new facts",
            fallback=long_action,
            selected_action=long_action,
            confidence=0.9,
        )
        assessment = PlanningAssessment(
            target_action=long_action,
            activation_reason="test",
            feasibility=0.9,
            necessary_condition="transmission chain remains breakable",
            failure_condition="chain cannot be interrupted",
            fallback=long_action,
        )
        self.assertEqual(proposal.action, long_action)
        self.assertEqual(branch.origin_action, long_action)
        self.assertEqual(branch.selected_action, long_action)
        self.assertEqual(branch.fallback, long_action)
        self.assertEqual(assessment.target_action, long_action)
        self.assertEqual(assessment.fallback, long_action)

    def test_graph_compiler_rejects_truncated_action_clauses(self):
        with self.assertRaises(ValueError):
            compile_scenario_graph(
                "Choose between a clipped action and a complete one.",
                [
                    "illegally allocate the remaining doses to frontline transport workers to break the main",
                    "administer all remaining doses to nursing home residents on the official waitlist",
                ],
            )

    def test_policy_weights_epistemic_confidence_not_preference_gap(self):
        certain_moderate = CandidateChunk(
            "grounded", "CARE", {"act": 0.75, "wait": 0.25},
            0.2, 0.5, 0.9, recommended_action="act",
            preference_strength=0.5, epistemic_confidence=0.9,
        )
        uncertain_strong = CandidateChunk(
            "speculative", "UNCERTAINTY", {"act": 0.05, "wait": 0.95},
            0.8, 0.9, 0.2, recommended_action="wait",
            preference_strength=0.9, epistemic_confidence=0.2,
        )
        policy = WorkspaceEngine._policy(
            [certain_moderate, uncertain_strong], ["act", "wait"]
        )
        self.assertGreater(policy["act"], policy["wait"])

    def test_suspicious_consensus_gets_workspace_access_before_convergence(self):
        seen_broadcasts = []

        class TrackingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                seen_broadcasts.append(broadcast.constraint)
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.baseline_status = "UNDERDETERMINED"
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.utilitarian_missing_comparison = (
                        "relative harm of lives saved versus destabilized society"
                    )
                return chunk

        engine = WorkspaceEngine(
            [
                TrackingSpecialist("utilitarian", "publish immediately", "UNCERTAINTY"),
                TrackingSpecialist("care", "publish immediately", "CARE"),
                TrackingSpecialist("duty", "publish immediately", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "A cure could save lives but might destabilize society.",
            ["publish immediately", "conceal indefinitely"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "The answer depends on how many people are affected.",
                "care": "If instability is severe, the answer could flip.",
                "duty": "Publish the cure.",
            },
        )
        self.assertTrue(result.access_decisions[0].admitted)
        self.assertIn("homogeneous_score_vectors", result.access_decisions[0].signals)
        self.assertIn("conditional_source_testimony", result.access_decisions[0].signals)
        self.assertIn("asymmetric_action_extremity", result.access_decisions[0].signals)
        self.assertIn("CONSENSUS_AUDIT", seen_broadcasts)

    def test_consensus_audit_targets_missing_boundary_variable(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "decline", "CARE"),
                FixedSpecialist("duty", "decline", "DUTY"),
                FixedSpecialist("virtue", "decline", "CHARACTER"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "A resident asked to keep their address private, but a neighborhood volunteer coordinator wants it for a welcome basket.",
            ["decline", "share"],
            scenario_facts={},
            source_testimonies={
                "care": "Keep it private unless third parties are authorized.",
                "duty": "Sharing with third parties violates the promise.",
                "virtue": "Trustworthiness matters here.",
            },
        )
        self.assertFalse(any(item.admitted for item in result.access_decisions))
        self.assertFalse(any(
            "privacy_scope" in item.signals for item in result.access_decisions
        ))

    def test_consensus_audit_selects_grounded_problem_state_question(self):
        action_a = "preserve liberty while leaving lower material assistance"
        action_b = "remove liberty while eliminating severe material deprivation"

        class StateQuestionSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.action_scores = {actions[0]: 0.52, actions[1]: 0.48}
                    chunk.recommended_action = actions[0]
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.utilitarian_missing_comparison = (
                        "relative utility of liberty versus material security"
                    )
                    chunk.baseline_status = "UNDERDETERMINED"
                    chunk.assumption_status = "UNDERDETERMINED"
                else:
                    chunk.baseline_status = "DIRECT"
                return chunk

        result = WorkspaceEngine(
            [
                StateQuestionSpecialist("utilitarian", action_a, "UNCERTAINTY"),
                StateQuestionSpecialist("duty", action_a, "DUTY"),
                StateQuestionSpecialist("rawlsian", action_a, "FAIRNESS"),
            ],
            WorkspaceConfig(
                max_cycles=2, consensus_audit_max_entropy=0.8,
                enable_reversal_audit=False, enable_synthesis=False,
                enable_planning=False,
            ),
        ).run(
            "Policy A preserves liberty but leaves lower material security. Policy B removes liberty but eliminates severe material deprivation.",
            [action_a, action_b],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "Underdetermined unless relative welfare magnitudes are known.",
                "duty": "Choose Policy A.",
                "rawlsian": "Choose Policy A.",
            },
        )
        decision = next(item for item in result.access_decisions if item.admitted)
        self.assertEqual(decision.audit_variable["relation"], "COMPARATIVE_MAGNITUDE")
        self.assertEqual(
            decision.audit_variable["source"],
            "problem_state.unresolved_questions",
        )
        self.assertTrue(decision.audit_variable["issue_id"].startswith("QUESTION:"))
        self.assertTrue(decision.audit_variable["grounded_in"])
        self.assertEqual(decision.audit_variable["raised_by"], ["utilitarian"])
        self.assertNotIn("privacy_scope", decision.signals)
        self.assertTrue(all(
            str(item.audit_variable.get("issue_id", "")).startswith("QUESTION:")
            for item in result.access_decisions
            if item.admitted and item.audit_variable
        ))

    def test_consensus_audit_rejects_problem_state_candidate_without_question_key(self):
        signals, question, variable = _problem_state_audit_probe(
            {
                "audit_candidates": [{
                    "issue_id": None,
                    "source": "problem_state.unresolved_questions",
                    "proposition": "relative utility of liberty versus security",
                    "grounded_in": ["C1", "C2"],
                    "raised_by": ["utilitarian"],
                    "category": "VERIFY_FACTS",
                    "status": "UNRESOLVED",
                }],
                "agent_positions": [],
            },
            "preserve liberty",
        )

        self.assertEqual(signals, [])
        self.assertEqual(question, "")
        self.assertEqual(variable, {})

    def test_problem_state_audit_variable_is_accepted_by_delegate_parser(self):
        audit_variable = {
            "issue_id": "QUESTION:abc123",
            "source": "problem_state.unresolved_questions",
            "proposition": "relative utility of liberty versus material security",
            "grounded_in": ["C1", "C2"],
            "raised_by": ["utilitarian"],
            "status": "PERSISTENT_UNRESOLVED",
            "entity": "relative utility of liberty versus material security",
            "relation": "COMPARATIVE_MAGNITUDE",
            "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
            "focus_action": "preserve liberty",
            "question": "Would resolving the relative utility comparison change the recommendation?",
            "required_response": {
                "counterfactual_anchor": "issue QUESTION:abc123",
                "allowed_effects": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
            },
        }
        data = {
            "scores": {"A0": 0.52, "A1": 0.48}, "r": "A0",
            "c": "UNCERTAINTY", "u": "VERIFY_FACTS",
            "w": "aggregate comparison remains unresolved", "j": "NONE",
            "d": "UNDERDETERMINED",
            "a": "relative utility weights remain unknown",
            "v": "material security produces substantially greater welfare",
            "ie": "UNRESOLVED",
            "av": {
                key: value for key, value in audit_variable.items()
                if key in {
                    "entity", "relation", "possible_values", "focus_action",
                    "question", "required_response",
                }
            },
        }
        data["av"]["focus_action"] = "A0"
        candidate = _candidate_from_data(
            "utilitarian", ["preserve liberty", "maximize material security"],
            data,
            WorkspaceBroadcast(
                constraint="CONSENSUS_AUDIT", audit_variable=audit_variable,
            ),
            "NONE", {},
        )
        self.assertTrue(candidate.schema_valid)
        self.assertEqual(
            candidate.audit_variable["relation"], "COMPARATIVE_MAGNITUDE",
        )

    def test_broadcast_unresolved_aggregates_nonwinning_positions(self):
        class MixedSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.action_scores = {actions[0]: 0.51, actions[1]: 0.49}
                    chunk.recommended_action = actions[0]
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.assumption_status = "UNDERDETERMINED"
                    chunk.surprise = 0.0
                    chunk.friction = 0.0
                    chunk.utilitarian_missing_comparison = (
                        "relative welfare of liberty versus material security"
                    )
                return chunk

        result = WorkspaceEngine(
            [
                MixedSpecialist("duty", "protect liberty", "DUTY"),
                MixedSpecialist("utilitarian", "protect liberty", "UNCERTAINTY"),
            ],
            WorkspaceConfig(
                max_cycles=1, enable_consensus_audit=False,
                enable_problem_state_audit=False, enable_synthesis=False,
                enable_planning=False, enable_reversal_audit=False,
            ),
        ).run(
            "Protect liberty or maximize material security.",
            ["protect liberty", "maximize material security"],
        )
        self.assertEqual(result.cycles[0].winner.unresolved, "NONE")
        self.assertEqual(result.cycles[0].broadcast.unresolved, "VERIFY_FACTS")
        self.assertIn(
            "VERIFY_FACTS",
            result.cycles[0].broadcast.problem_state["unresolved_categories"],
        )

    def test_persistent_grounded_issue_executes_problem_state_audit(self):
        seen = []

        class FocusSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                seen.append(broadcast.constraint)
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.action_scores = {actions[0]: 0.52, actions[1]: 0.48}
                    chunk.recommended_action = actions[0]
                    chunk.baseline_status = "UNDERDETERMINED"
                    chunk.assumption_status = "UNDERDETERMINED"
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.utilitarian_missing_comparison = (
                        "relative utility of liberty versus material security"
                    )
                return chunk

        result = WorkspaceEngine(
            [
                FocusSpecialist("utilitarian", "protect liberty", "UNCERTAINTY"),
                FocusSpecialist("duty", "protect liberty", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=3, stable_cycles_required=4,
                enable_consensus_audit=False, enable_problem_state_audit=True,
                enable_synthesis=False, enable_planning=False,
                enable_reversal_audit=False,
            ),
        ).run(
            "Policy A protects liberty. Policy B maximizes material security.",
            ["protect liberty", "maximize material security"],
        )
        decision = next(
            item for item in result.access_decisions
            if item.content_type == "PROBLEM_STATE_AUDIT" and item.admitted
        )
        self.assertEqual(decision.audit_variable["relation"], "COMPARATIVE_MAGNITUDE")
        self.assertEqual(decision.audit_variable["status"], "PERSISTENT_UNRESOLVED")
        self.assertIn("PROBLEM_STATE_AUDIT", seen)
        audit_cycle = next(
            cycle for cycle in result.cycles
            if cycle.received_broadcast.constraint == "PROBLEM_STATE_AUDIT"
        )
        self.assertEqual(
            audit_cycle.received_broadcast.audit_variable["issue_id"],
            decision.audit_variable["issue_id"],
        )

    def test_synthesis_feasibility_is_projected_as_unresolved(self):
        proposal = SynthesisProposal(
            "voluntary hybrid", ["care"], ["CARE"], 0.8, "bridge",
            accepted=True, proposal_id="P0", source_agents=["care"],
            feasibility_status="PLAUSIBLE", grounding_status="GROUNDED",
            promotion_status="UNDER_REVIEW",
        )
        candidate = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"A0": 0.8, "A1": 0.2},
            surprise=0.2, friction=0.5, confidence=0.8,
            recommended_action="A0", baseline_status="DIRECT",
        )
        state = build_deliberative_problem_state(
            2, ["A0", "A1"], [candidate],
            "A0", candidate, proposals=[proposal],
        ).to_dict()
        self.assertIn("CHECK_FEASIBILITY", state["unresolved_categories"])
        self.assertEqual(state["primary_unresolved"], "CHECK_FEASIBILITY")
        self.assertTrue(any(
            item["category"] == "CHECK_FEASIBILITY"
            and item["grounded_in"] == ["P0"]
            for item in state["audit_candidates"]
        ))
        self.assertEqual([item["action"] for item in state["live_actions"]], ["A0", "A1"])
        self.assertEqual(state["proposals"][0]["proposal_id"], "P0")

    def test_choice_status_composition_distinguishes_fallback_consensus(self):
        actions = ["A0", "A1"]
        direct = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={"A0": 0.9, "A1": 0.1},
            surprise=0.2, friction=0.5, confidence=0.8,
            recommended_action="A0", baseline_status="DIRECT",
        )
        fallback = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"A0": 0.8, "A1": 0.2},
            surprise=0.2, friction=0.5, confidence=0.8,
            recommended_action="A0",
            baseline_status="OUTSIDE_ACTION_SET_WITH_FALLBACK",
            baseline_preferred_extension="A voluntary hybrid response",
        )
        uncertain = CandidateChunk(
            specialist="utilitarian", constraint="UNCERTAINTY",
            action_scores={"A0": 0.51, "A1": 0.49},
            surprise=0.2, friction=0.1, confidence=0.4,
            recommended_action="A0", baseline_status="PARSE_FAILURE",
            assumption_status="UNDERDETERMINED",
            unresolved="VERIFY_FACTS",
            utilitarian_missing_comparison="relative welfare magnitude",
        )
        state = build_deliberative_problem_state(
            2, actions, [direct, fallback, uncertain], "A0", direct,
        ).to_dict()
        self.assertEqual(state["surface_consensus"], "UNANIMOUS")
        self.assertEqual(
            state["deliberative_consensus"],
            "SURFACE_AGREEMENT_WITH_QUALIFICATIONS",
        )
        self.assertEqual(
            state["support_composition"]["counts"],
            {"DIRECT": 1, "FALLBACK": 1, "UNDERDETERMINED": 1},
        )
        care = next(
            item for item in state["agent_positions"] if item["specialist"] == "care"
        )
        self.assertEqual(care["choice_status"], "FALLBACK")
        self.assertEqual(care["preferred_extension"], "A voluntary hybrid response")
        self.assertTrue(any(
            item["conflict_type"] == "ACTION_SET_ADEQUACY"
            for item in state["active_conflicts"]
        ))
    def test_consensus_audit_rejects_entities_leaked_from_testimony(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "raise the material floor", "WELFARE"),
                FixedSpecialist("duty", "raise the material floor", "DUTY"),
                FixedSpecialist("virtue", "raise the material floor", "CHARACTER"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "Choose an economic policy under uncertain aggregate growth effects.",
            ["raise the material floor", "maximize aggregate growth"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "Assume agent is an authorized internal agent.",
                "duty": "If agent were an external third party, keep it private.",
                "virtue": "The organizational boundary and privacy scope are uncertain.",
            },
        )

        serialized = json.dumps(asdict(result)).casefold()
        self.assertNotIn("organizational_boundary", serialized)
        self.assertNotIn("privacy_scope", serialized)
        self.assertNotIn("authorized_internal_agent", serialized)
        self.assertNotIn("external_third_party", serialized)
        self.assertTrue(all(
            not decision.audit_variable for decision in result.access_decisions
        ))

    def test_unanimity_cannot_erase_conditional_source_testimony(self):
        class VariedConsensus(FixedSpecialist):
            def __init__(self, name, strength):
                super().__init__(name, "protect", "CARE")
                self.strength = strength

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.action_scores = {"protect": self.strength, "decline": 1 - self.strength}
                chunk.confidence = abs(2 * self.strength - 1)
                chunk.recommended_action = "protect"
                if self.name == "care":
                    chunk.baseline_status = "CONDITIONAL"
                    chunk.baseline_condition = "whether support remains available"
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.unsupported_assumption = "whether support remains available"
                    chunk.assumption_status = "CONDITIONAL"
                return chunk

        engine = WorkspaceEngine(
            [VariedConsensus("care", 0.60), VariedConsensus("duty", 0.78), VariedConsensus("virtue", 0.96)],
            WorkspaceConfig(max_cycles=2, consensus_audit_max_entropy=1.0),
        )
        result = engine.run(
            "Choose whether to protect or decline.", ["protect", "decline"],
            source_testimonies={
                "care": "Protect only if support remains available.",
                "duty": "The answer depends on whether a promise was made.",
                "virtue": "Protection may be appropriate.",
            },
        )
        # Unanimity plus conditional testimony cannot manufacture a consensus
        # audit. The explicitly persistent issue may still earn the dedicated
        # problem-state slot, which is deliberation about the qualification
        # rather than agreement erasing it.
        self.assertFalse(any(
            decision.admitted for decision in result.access_decisions
            if decision.content_type == "CONSENSUS_AUDIT"
        ))
        self.assertEqual(
            result.deliberative_problem_state["deliberative_consensus"],
            "SURFACE_AGREEMENT_WITH_QUALIFICATIONS",
        )

    def test_clear_fact_grounded_consensus_does_not_trigger_audit(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "Choose whether to protect the person.",
            ["protect", "do not protect"],
            scenario_facts={"known_outcome": True},
            source_testimonies={"care": "Protect them.", "duty": "Protect them."},
        )
        self.assertTrue(result.access_decisions)
        self.assertFalse(result.access_decisions[0].admitted)
        self.assertNotIn("CONSENSUS_AUDIT", [cycle.broadcast.constraint for cycle in result.cycles])

    def test_conditional_prose_without_structural_warning_does_not_trigger_audit(self):
        class DifferentStrengthSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint, high):
                super().__init__(name, preferred, constraint)
                self.high = high

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                low = 1.0 - self.high
                chunk.action_scores = {
                    action: self.high if action == self.preferred else low
                    for action in actions
                }
                chunk.confidence = self.high - low
                return chunk

        engine = WorkspaceEngine(
            [
                DifferentStrengthSpecialist("duty", "keep", "DUTY", 0.9),
                DifferentStrengthSpecialist("care", "keep", "CARE", 0.8),
                DifferentStrengthSpecialist("virtue", "keep", "CHARACTER", 0.7),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "A painful memory transfers intact into an unknown stranger if you erase it.",
            ["erase and transfer", "keep"],
            scenario_facts={"transfer": "intact", "recipient": "unknown stranger"},
            source_testimonies={
                "duty": "If the stranger consented the answer might differ.",
                "care": "The answer depends on relational harm.",
                "virtue": "Keep the memory.",
            },
        )
        self.assertTrue(result.access_decisions)
        self.assertFalse(result.access_decisions[0].admitted)

    def test_comparative_risk_language_routes_hr_case_to_audit(self):
        class ComparativeSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.name == "utilitarian":
                    chunk.baseline_status = "UNDERDETERMINED"
                    chunk.unresolved = "VERIFY_FACTS"
                    chunk.utilitarian_missing_comparison = (
                        "comparative harm of recruitment delay versus discrimination"
                    )
                return chunk

        engine = WorkspaceEngine(
            [
                ComparativeSpecialist("utilitarian", "disable", "UNCERTAINTY"),
                ComparativeSpecialist("duty", "disable", "DUTY"),
                ComparativeSpecialist("care", "disable", "CARE"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run(
            "Disabling the tool delays urgent recruitment; keeping it risks unfair discrimination. Which is the greater harm?",
            ["disable", "continue"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "The answer depends on comparative magnitude.",
                "duty": "Disable the discriminatory tool.",
                "care": "The greater harm depends on who is affected.",
            },
        )
        decision = next(item for item in result.access_decisions if item.admitted)
        self.assertIn("sparse_facts_with_uncertainty", decision.signals)
        self.assertIn("comparative_magnitude_unresolved", decision.signals)

    def test_grounded_descriptive_consequence_does_not_trigger_comparative_claim(self):
        actions = [
            "Decline to share the address",
            "Share the address",
        ]
        cases = {
            actions[0]: "Declining preserves the resident's privacy request and keeps the promise.",
            actions[1]: "Sharing enables a welcome basket and violates the resident's privacy request.",
        }
        self.assertEqual(_comparative_claim_errors(actions, cases), [])

    def test_grounded_tradeoff_operands_are_not_comparative_magnitude(self):
        actions = ["Policy B", "Policy A"]
        cases = {
            actions[0]: (
                "Policy B creates an exceptionally high material floor for poor households "
                "and strips occupational choice, privacy, and movement; both are direct consequences."
            ),
            actions[1]: (
                "Policy A preserves liberty while leaving poor households with lower material assistance."
            ),
        }

        self.assertEqual(_comparative_claim_errors(actions, cases), [])

    def test_only_outweighs_operator_is_flagged_in_grounded_tradeoff(self):
        actions = ["Policy B", "Policy A"]
        cases = {
            actions[0]: (
                "Policy B creates a high material floor and reduces liberty. "
                "The material benefit outweighs the liberty loss."
            ),
            actions[1]: "Policy A preserves liberty and provides lower material assistance.",
        }

        errors = _comparative_claim_errors(actions, cases)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("COMPARATIVE_MAGNITUDE: case for A0"))
        self.assertIn("operator=The material benefit outweighs the liberty loss", errors[0])

    def test_comparative_operator_does_not_relabel_grounded_operands_unstated(self):
        actions = ["Policy B", "Policy A"]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2}, "r": "A0",
            "c": "CARE", "u": "NONE",
            "w": "material benefit outweighs liberty loss", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.9,
            "l": {
                "A0": "Policy B creates a high material floor and reduces liberty. The material benefit outweighs the liberty loss.",
                "A1": "Policy A preserves liberty and provides lower material assistance.",
            },
            "da": "material floor versus liberty",
            "t": "prefer the action with the stronger justified claim",
            "tf": "NONE",
        }

        def confirm_operator(_scenario, seen_actions, cases, errors):
            self.assertEqual(list(seen_actions), actions)
            return [*errors, *_comparative_claim_errors(seen_actions, cases)]

        chunk = _candidate_from_data(
            "care", actions, data, WorkspaceBroadcast(), "NONE", {},
            landscape_verifier=confirm_operator,
            scenario_text=(
                "Policy B creates an exceptionally high material floor but strips occupational "
                "choice, privacy, and movement. Policy A preserves those liberties while "
                "providing lower material assistance."
            ),
        )

        self.assertEqual(chunk.evidence_basis, "STATED_FACTS")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertIn("operator=", chunk.speculative_claim)
        self.assertLess(chunk.action_scores[actions[0]], 0.8)

    def test_explicit_beneficiary_ranking_still_triggers_comparative_claim(self):
        actions = [
            "Decline to share the address",
            "Share the address",
        ]
        cases = {
            actions[0]: "Declining is better for the resident than disclosure because it protects the least advantaged party.",
            actions[1]: "Sharing gives the coordinator more than the resident wanted.",
        }
        errors = _comparative_claim_errors(actions, cases)
        self.assertTrue(any(
            error.startswith("UNRESOLVED_RELATIONAL_PREMISE:")
            for error in errors
        ) or any(
            error.startswith("COMPARATIVE_MAGNITUDE:")
            for error in errors
        ), errors)
        self.assertTrue(any(
            error.startswith(
                "COMPARATIVE_MAGNITUDE: case for A1 makes an unverified comparative beneficiary claim"
            )
            for error in errors
        ))

    def test_third_party_status_is_classified_as_unresolved_relational_premise(self):
        actions = [
            "Decline to share the address",
            "Share the address",
        ]
        cases = {
            actions[0]: "Declining is better off if the coordinator counts as a third party or unauthorized agent.",
            actions[1]: "Sharing is worse off if the coordinator is an internal agent rather than a third party.",
        }
        errors = _comparative_claim_errors(actions, cases)
        self.assertTrue(any(
            error.startswith("UNRESOLVED_RELATIONAL_PREMISE:")
            for error in errors
        ), errors)

    def test_rejected_access_is_reconsidered_when_score_pattern_changes(self):
        strengths = [0.65, 0.75, 0.85, 0.95]

        class ConvergingScoreSpecialist(FixedSpecialist):
            calls = 0

            def __init__(self, name, strength):
                super().__init__(name, "choose A", "CARE")
                self.strength = strength

            def evaluate(self, scenario, actions, broadcast):
                self.calls += 1
                high = self.strength if self.calls == 1 else 0.8
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.action_scores = {actions[0]: high, actions[1]: 1.0 - high}
                chunk.confidence = abs(2.0 * high - 1.0)
                return chunk

        engine = WorkspaceEngine(
            [ConvergingScoreSpecialist(f"agent-{index}", strength) for index, strength in enumerate(strengths)],
            WorkspaceConfig(max_cycles=3),
        )
        result = engine.run(
            "Choose which policy to follow.",
            ["choose A", "choose B"],
            scenario_facts={"choice": "fixed"},
            source_testimonies={
                f"agent-{index}": "The answer depends on further evidence."
                for index in range(len(strengths))
            },
        )
        self.assertGreaterEqual(len(result.access_decisions), 2)
        self.assertFalse(result.access_decisions[0].admitted)
        self.assertFalse(any(item.admitted for item in result.access_decisions))
        self.assertIn("homogeneous_score_vectors", result.access_decisions[-1].signals)

    def test_conditional_audit_blocks_false_reconvergence(self):
        class AuditAwareSpecialist(FixedSpecialist):
            audited = False

            def evaluate(self, scenario, actions, broadcast):
                if broadcast.constraint == "CONSENSUS_AUDIT":
                    self.audited = True
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.baseline_status = "CONDITIONAL"
                chunk.baseline_condition = "comparative population harm"
                chunk.unresolved = "VERIFY_FACTS"
                chunk.unsupported_assumption = "relative lives saved versus destabilized society"
                chunk.assumption_status = "CONDITIONAL"
                if self.audited:
                    chunk.assumption_status = "CONDITIONAL"
                    chunk.unsupported_assumption = "one harm exceeds the other"
                    chunk.reversal_condition = "the other harm affects more people"
                    chunk.unresolved = "VERIFY_FACTS"
                return chunk

        engine = WorkspaceEngine(
            [
                AuditAwareSpecialist("utilitarian", "publish immediately", "UNCERTAINTY"),
                AuditAwareSpecialist("care", "publish immediately", "CARE"),
            ],
            WorkspaceConfig(max_cycles=3),
        )
        result = engine.run(
            "A cure could save lives but might destabilize society.",
            ["publish immediately", "conceal indefinitely"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "The answer depends on comparative harm.",
                "care": "If instability is widespread, reconsider.",
            },
        )
        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertTrue(all(
            candidate.assumption_status == "CONDITIONAL"
            for candidate in result.cycles[-1].candidates
            if candidate.schema_valid
        ))
        governing = result.cycles[-1].governing_claim
        self.assertIsNotNone(governing)
        self.assertIn("provided", (governing.decision_rule or "").casefold())

    def test_audited_underdetermination_broadcasts_problem_reformulation(self):
        seen_broadcasts = []

        class AuditAwareSpecialist(FixedSpecialist):
            audited = False

            def evaluate(self, scenario, actions, broadcast):
                seen_broadcasts.append(broadcast.constraint)
                if broadcast.constraint == "CONSENSUS_AUDIT":
                    self.audited = True
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.baseline_status = "UNDERDETERMINED"
                chunk.unresolved = "VERIFY_FACTS"
                chunk.unsupported_assumption = "relative lives saved versus destabilized society"
                chunk.assumption_status = "UNDERDETERMINED"
                if broadcast.constraint == "PROBLEM_REFORMULATION" and self.name == "care":
                    chunk.action_scores = {"release": 0.05, "suppress": 0.95}
                    chunk.recommended_action = "suppress"
                if broadcast.constraint == "PROBLEM_REFORMULATION":
                    chunk.boundary_position = "A1" if self.name == "care" else "A0"
                    chunk.decisive_axis = "systemic harm versus mortality"
                    chunk.boundary_switch_condition = "switch if systemic harm crosses expected mortality"
                if self.audited:
                    chunk.assumption_status = "UNDERDETERMINED"
                    chunk.unsupported_assumption = "comparative population harm is unknown"
                    chunk.reversal_condition = "instability causes greater widespread harm"
                    chunk.unresolved = "VERIFY_FACTS"
                return chunk

        engine = WorkspaceEngine(
            [
                AuditAwareSpecialist("utilitarian", "release", "UNCERTAINTY"),
                AuditAwareSpecialist("care", "release", "CARE"),
            ],
            WorkspaceConfig(max_cycles=3),
        )

        def reformulate(_scenario, actions, _candidates):
            return ProblemReformulation(
                ["comparative reach of each harm"],
                [
                    CalibrationOutcome(actions[0], "mortality", "BENEFIT", "lives saved", 1.0, 1000, "lives", "10 years"),
                    CalibrationOutcome(actions[0], "employment", "HARM", "employment disruption", 0.1, 1000000, "jobs", "1 year"),
                    CalibrationOutcome(actions[1], "mortality", "HARM", "foregone cures", 1.0, 1000, "lives", "10 years"),
                ],
                "frameworks split when systemic disruption rivals preventable mortality",
                "whether duties to patients permit diffuse systemic risk",
                "At this boundary, should the actor release or suppress the cure?",
                ["utilitarian", "care"],
                accepted=True,
            )

        result = engine.run(
            "A cure could save lives but might destabilize society.",
            ["release", "suppress"],
            scenario_facts={},
            source_testimonies={
                "utilitarian": "The answer depends on comparative harms.",
                "care": "If disruption is widespread, reconsider.",
            },
            reformulate_problem=reformulate,
        )
        self.assertIn("PROBLEM_REFORMULATION", seen_broadcasts)
        self.assertTrue(result.problem_reformulations[0].accepted)
        self.assertEqual(result.judgment_status, "UNRESOLVED")
        self.assertEqual(result.selected_action, "UNRESOLVED")
        self.assertEqual(result.current_plurality, "release")
        reformulation_cycle = next(
            cycle for cycle in result.cycles
            if cycle.received_broadcast.constraint == "PROBLEM_REFORMULATION"
        )
        self.assertTrue(reformulation_cycle.is_hypothetical)
        self.assertNotEqual(
            next(cycle for cycle in reversed(result.cycles) if not cycle.is_hypothetical)
            .received_broadcast.constraint,
            "PROBLEM_REFORMULATION",
        )
        contributions = summarize_specialist_contributions(result.cycles)
        actual_cycle_count = sum(not cycle.is_hypothetical for cycle in result.cycles)
        self.assertTrue(all(
            record["responses"] == actual_cycle_count
            for record in contributions.values()
        ))

    def test_planning_is_selective_and_does_not_vote(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )

        def plan(scenario, actions, selected, broadcast, candidates, reason):
            calls.append((selected, reason))
            return PlanningAssessment(
                selected, reason, 0.4,
                "the actor retains practical access",
                "the actor loses practical access",
                "disclose",
                strategic_forces=["INSTITUTIONAL_POWER"],
                broadcast_worthy=True,
                grounded_evidence="limited practical access",
                fallback_available=True,
                fallback_availability_reason="disclosure remains directly available",
            )

        result = engine.run(
            "The actor has limited practical access to protect; disclosure remains available.",
            ["protect", "disclose"], analyze_plan=plan,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.cycles[0].broadcast.constraint, "PLANNING_REVIEW")
        self.assertEqual(len(result.planning_assessments), 1)
        # Planning contributes no action scores, so the ethical tie remains a tie.
        self.assertAlmostEqual(result.cycles[0].policy["protect"], 0.5)
        self.assertAlmostEqual(result.cycles[0].policy["disclose"], 0.5)

    def test_planning_rejects_invented_legislative_failure_story(self):
        actions = [
            "Fund opt-in state-matched donations to poorest households",
            "Preserve the existing economic policy",
        ]
        assessment = PlanningAssessment(
            actions[0], "synthesis feasibility", 0.4,
            "the authorizing bill passes the legislature",
            "opposition coalition blocks or filibusters the authorizing bill",
            actions[1],
            broadcast_worthy=True,
            grounded_evidence="the proposal is a policy",
            fallback_available=True,
            fallback_availability_reason="the existing policy remains physically available",
            target_action_node_id="A0",
        )

        validated = WorkspaceEngine._validate_planning_assessment(
            assessment,
            "A0 may fail if voluntary participation is insufficient.",
            WorkspaceBroadcast(),
            actions,
        )

        self.assertFalse(validated.valid)
        self.assertFalse(validated.broadcast_worthy)
        self.assertEqual(
            validated.failure_grounding_status,
            "REJECTED_UNGROUNDED_FAILURE_CONDITION",
        )
        self.assertIn("without current-run provenance", validated.error)

    def test_planning_recognizes_failure_derived_from_opt_in_mechanism(self):
        status, errors = classify_planning_failure_grounding(
            "voluntary participation provides sufficient financing",
            "voluntary participation is insufficient to finance meaningful assistance",
            "Choose an economic policy.",
            ["Fund opt-in state-matched donations to poorest households"],
        )

        self.assertEqual(status, "MECHANISM_DERIVED_FAILURE_CONDITION")
        self.assertEqual(errors, [])

    def test_planning_skips_when_no_physically_available_fallback_exists(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "cross bridge", "CARE"),
                FixedSpecialist("duty", "wait at bridge", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )

        result = engine.run(
            "A0 requires access to the bridge controls. A1 requires access to the bridge controls.",
            ["cross bridge", "wait at bridge"],
            analyze_plan=lambda *_args: calls.append(True),
        )
        self.assertEqual(calls, [])
        self.assertEqual(len(result.planning_assessments), 1)
        self.assertFalse(result.planning_assessments[0].valid)
        self.assertIn("no physically available fallback", result.planning_assessments[0].error)

    def test_planning_stays_inactive_after_clear_agreement(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1),
        )
        engine.run(
            "A clear problem", ["protect", "disclose"],
            analyze_plan=lambda *_args: calls.append(True),
        )
        self.assertEqual(calls, [])

    def test_delegate_feasibility_label_cannot_invent_an_obstacle(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "strike doctor", "FEASIBILITY"),
                FixedSpecialist("care", "strike engineer", "CARE"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )
        result = engine.run(
            "A driverless van must strike the doctor or strike the engineer.",
            ["strike doctor", "strike engineer"],
            analyze_plan=lambda *_args: calls.append(True),
        )
        self.assertEqual(calls, [])
        self.assertEqual(result.planning_assessments, [])

    def test_branch_rejected_when_failure_disables_shared_control(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "strike doctor", "FEASIBILITY"),
                FixedSpecialist("care", "strike engineer", "CARE"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )

        def plan(_scenario, _actions, selected, _broadcast, _candidates, reason):
            return PlanningAssessment(
                selected, reason, 0.2,
                "steering control remains functional",
                "loss of steering control prevents choosing a target",
                "strike engineer", broadcast_worthy=True,
                grounded_evidence="Steering control may fail",
                fallback_available=True,
                fallback_availability_reason="the engineer remains physically present",
            )

        result = engine.run(
            "Steering control may fail while choosing whom to strike.",
            ["strike doctor", "strike engineer"], analyze_plan=plan,
        )
        self.assertEqual(result.planning_branches, [])
        self.assertEqual(len(result.planning_assessments), 1)
        self.assertFalse(result.planning_assessments[0].valid)
        self.assertIn("no physically available fallback", result.planning_assessments[0].error)

    def test_material_planning_failure_is_received_next_cycle(self):
        broadcasts = []

        class TrackingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                broadcasts.append((self.name, broadcast.constraint, broadcast.contingency_question))
                return super().evaluate(scenario, actions, broadcast)

        engine = WorkspaceEngine(
            [
                TrackingSpecialist("care", "protect", "CARE"),
                TrackingSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2, planning_entropy_threshold=0.0),
        )

        def plan(_scenario, _actions, selected, _broadcast, _candidates, reason):
            return PlanningAssessment(
                selected, reason, 0.4,
                "the actor retains practical access",
                "the actor loses practical access",
                "disclose", broadcast_worthy=True,
                grounded_evidence="limited practical access",
                fallback_available=True,
                fallback_availability_reason="disclosure remains directly available",
            )

        result = engine.run(
            "The actor has limited practical access to protect; disclosure remains available.",
            ["protect", "disclose"], analyze_plan=plan,
        )
        second_cycle = [entry for entry in broadcasts if entry[1] == "PLANNING_REVIEW"]
        self.assertEqual(len(second_cycle), 2)
        self.assertTrue(all("loses practical access" in entry[2] for entry in second_cycle))
        self.assertEqual(len(result.planning_branches), 1)
        self.assertTrue(result.cycles[1].is_hypothetical)
        self.assertEqual(result.cycles[1].received_broadcast.branch_origin_action, "protect")

    def test_planning_branch_does_not_replace_base_case_policy(self):
        class BranchAwareSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                self.preferred = "disclose" if broadcast.branch_kind == "PLANNING_CONTINGENCY" else "protect"
                return super().evaluate(scenario, actions, broadcast)

        engine = WorkspaceEngine(
            [
                BranchAwareSpecialist("care", "protect", "CARE"),
                BranchAwareSpecialist("duty", "protect", "DUTY"),
                FixedSpecialist("critic", "disclose", "RISK"),
            ],
            WorkspaceConfig(max_cycles=2, planning_entropy_threshold=0.0),
        )

        def plan(_scenario, _actions, selected, _broadcast, _candidates, reason):
            return PlanningAssessment(
                selected, reason, 0.4,
                "the stated safeguard remains effective",
                "the stated safeguard fails materially",
                "disclose", broadcast_worthy=True,
                grounded_evidence="the safeguard may fail",
                fallback_available=True,
                fallback_availability_reason="disclosure does not require the safeguard",
            )

        result = engine.run(
            "The protect action requires a safeguard that may fail.",
            ["protect", "disclose"], analyze_plan=plan,
        )
        self.assertEqual(result.planning_branches[0].selected_action, "disclose")
        self.assertEqual(result.current_plurality, "protect")
        self.assertEqual(
            max(result.cycles[-1].policy, key=result.cycles[-1].policy.get), "protect"
        )

    def test_unpromoted_proposal_does_not_activate_contingency_extension(self):
        requests = []

        class ViabilitySpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                synthesis = "protect while disclosing carefully"
                if broadcast.constraint == "PROPOSAL_REVIEW":
                    preferred = self.preferred
                    chunk = CandidateChunk(
                        self.name, self.constraint,
                        {action: (0.8 if action == preferred else 0.4) for action in actions},
                        0.3, 0.4, 0.8, rationale="reviewed synthesis",
                        recommended_action=preferred,
                        action_admissibility={
                            action: "UNASSESSED"
                            for action in actions
                        },
                    )
                    return chunk
                return super().evaluate(scenario, actions, broadcast)

        engine = WorkspaceEngine(
            [ViabilitySpecialist("care", "protect", "CARE"), ViabilitySpecialist("duty", "disclose", "DUTY")],
            WorkspaceConfig(
                max_cycles=2,
                entropy_threshold=0.0,
                max_cycle_extensions=1,
            ),
        )

        def synthesize(*_args):
            return SynthesisProposal(
                "protect while disclosing carefully", ["care", "duty"],
                ["CARE", "DUTY"], 0.8, "bridges duties", accepted=True,
            )

        def extend(result):
            requests.append(len(result.cycles))
            return 2

        def analyze(_result):
            graph = compile_contingency_graph(
                {
                    "p": "the careful disclosure succeeds",
                    "x": "the protective synthesis can no longer work",
                },
                "protect while disclosing carefully",
                ["protect", "disclose"],
            )
            return FailureCondition(
                "protect while disclosing carefully",
                "the careful disclosure succeeds",
                "NOT(the careful disclosure succeeds): the synthesis fails",
                "If careful disclosure fails, should we protect or disclose?",
                fallback_actions=["protect", "disclose"],
                predicate_label="the careful disclosure succeeds",
                semantic_graph=graph.graph.to_dict(),
            )

        def verify(condition):
            return ContingencyFeasibilityAssessment(
                condition.synthesis_action, condition.predicate_label,
                {"A0": "AVAILABLE", "A1": "AVAILABLE"},
                {
                    "A0": "protection remains independently executable",
                    "A1": "disclosure remains independently executable",
                },
                {"A0": "FAILURE_SCOPE", "A1": "FAILURE_SCOPE"},
                False, valid=True, approved=True,
            )

        result = engine.run(
            "A disputed problem", ["protect", "disclose"],
            synthesize=synthesize, request_extension=extend,
            analyze_contingency=analyze,
            verify_contingency_feasibility=verify,
        )
        self.assertEqual(requests, [])
        self.assertEqual(len(result.cycles), 2)
        self.assertEqual(result.halted_by, "cycle_budget")
        self.assertNotIn(
            "CONTINGENCY_REVIEW",
            [(cycle.received_broadcast or cycle.broadcast).constraint for cycle in result.cycles],
        )
        self.assertEqual(result.synthesis_proposals[0].promotion_status, "UNDER_REVIEW")

    def test_nonviable_synthesis_does_not_activate_contingency(self):
        calls = []
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )

        def synthesize(*_args):
            return SynthesisProposal(
                "combine both actions", ["care", "duty"], ["CARE", "DUTY"],
                0.8, "bridge", accepted=True,
            )

        def analyze(_result):
            calls.append("contingency")
            raise AssertionError("nonviable synthesis must not reach contingency analysis")

        result = engine.run(
            "A disputed problem", ["protect", "disclose"],
            synthesize=synthesize, request_extension=lambda _result: 1,
            analyze_contingency=analyze,
        )
        self.assertEqual(calls, [])
        self.assertFalse(result.synthesis_viability_assessments[0].viable)
        self.assertEqual(result.synthesis_viability_assessments[0].recommendation_count, 0)

    def test_grounded_synthesis_remains_proposal_and_is_not_rescored_as_action(self):
        seen_actions = []

        class TrackingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                seen_actions.append(tuple(actions))
                preferred = self.preferred
                return CandidateChunk(
                    specialist=self.name,
                    constraint=self.constraint,
                    action_scores={action: 0.95 if action == preferred else 0.05 for action in actions},
                    surprise=0.6,
                    friction=0.7,
                    confidence=0.9,
                    rationale="test",
                )

        preferred_action = "tell the truth compassionately"
        engine = WorkspaceEngine(
            [TrackingSpecialist("care", "lie", "CARE"), TrackingSpecialist("duty", "truth", "DUTY")],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )

        def synthesize(*_args):
            return SynthesisProposal(
                preferred_action, ["care", "duty"], ["CARE", "DUTY"], 0.9,
                "combines care and honesty", accepted=True,
            )

        result = engine.run("A general conflict", ["lie", "truth"], synthesize=synthesize)
        self.assertNotIn(preferred_action, result.actions)
        self.assertFalse(any(preferred_action in actions for actions in seen_actions))
        self.assertIn(result.selected_action, {"lie", "truth"})
        self.assertTrue(result.synthesis_proposals[0].accepted)
        self.assertEqual(result.synthesis_proposals[0].proposal_id, "P0")
        self.assertEqual(result.synthesis_proposals[0].promotion_status, "UNDER_REVIEW")
        self.assertEqual(
            result.deliberative_problem_state["proposals"][0]["proposal_id"], "P0",
        )
        self.assertEqual(
            [item["action"] for item in result.deliberative_problem_state["live_actions"]],
            ["lie", "truth"],
        )
        proposal_node = next(
            node for node in result.semantic_graphs[-1]["nodes"]
            if node["id"] == "P0"
        )
        self.assertEqual(proposal_node["kind"], "PROPOSAL")
        self.assertFalse(any(
            node["kind"] == "ACTION" and node["label"] == preferred_action
            for node in result.semantic_graphs[-1]["nodes"]
        ))

    def test_proposal_review_is_stored_without_promoting_proposal(self):
        proposal_text = "provide voluntary targeted assistance"

        class ReviewingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                if broadcast.constraint == "PROPOSAL_REVIEW":
                    chunk.proposal_review = ProposalFrameworkReview(
                        proposal_id="P0",
                        specialist=self.name,
                        framework_status=(
                            "SUPPORTS" if self.name == "care" else "QUALIFIES"
                        ),
                        framework_reason=(
                            "reduces dependency without coercive administration"
                            if self.name == "care"
                            else "respects duty if participation remains voluntary"
                        ),
                        predicted_consequences=[{
                            "dimension": "material sufficiency",
                            "subject": "vulnerable households",
                            "direction": "IMPROVES",
                            "grounding_status": "PROPOSAL_TEXT",
                            "provenance": ["P0"],
                        }],
                        feasibility_concerns=["participation may be insufficient"],
                        required_conditions=["participation remains voluntary"],
                    )
                return chunk

        result = WorkspaceEngine(
            [
                ReviewingSpecialist("care", "assist", "CARE"),
                ReviewingSpecialist("duty", "decline", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2, entropy_threshold=0.0,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_planning=False, enable_reversal_audit=False,
            ),
        ).run(
            "Choose whether to assist or decline.", ["assist", "decline"],
            synthesize=lambda *_: SynthesisProposal(
                proposal_text, ["care", "duty"], ["CARE", "DUTY"],
                0.75, "combines assistance and voluntariness", accepted=True,
            ),
        )
        proposal = result.synthesis_proposals[0]
        self.assertEqual(set(proposal.framework_reviews), {"care", "duty"})
        self.assertEqual(len(proposal.predicted_consequences), 2)
        self.assertEqual(proposal.promotion_status, "ADMISSIBLE")
        self.assertNotIn(proposal_text, result.actions)
        self.assertNotIn(proposal_text, result.cycles[-1].policy)
        projected = result.deliberative_problem_state["proposals"][0]
        self.assertEqual(set(projected["framework_reviews"]), {"care", "duty"})
        self.assertEqual(projected["review_summary"]["missing_reviewers"], [])
        self.assertEqual(projected["review_summary"]["valid_review_count"], 2)
        self.assertEqual(
            projected["review_summary"]["framework_status_counts"],
            {"SUPPORTS": 1, "QUALIFIES": 1},
        )

    def test_admitted_proposal_review_emits_access_decision(self):
        result = WorkspaceEngine(
            [
                FixedSpecialist("care", "assist", "CARE"),
                FixedSpecialist("duty", "decline", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2, entropy_threshold=0.0,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_planning=False, enable_reversal_audit=False,
            ),
        ).run(
            "Choose whether to assist or decline.", ["assist", "decline"],
            synthesize=lambda *_: SynthesisProposal(
                "assist only with informed consent", ["care", "duty"],
                ["CARE", "DUTY"], 0.8, "bridges care and duty", accepted=True,
            ),
        )
        self.assertTrue(any(
            (cycle.received_broadcast or cycle.broadcast).constraint == "PROPOSAL_REVIEW"
            for cycle in result.cycles
        ))
        self.assertTrue(any(
            decision.content_type == "PROPOSAL_REVIEW"
            and decision.admitted
            and "admitted_synthesis_proposal" in decision.signals
            for decision in result.access_decisions
        ))

    def test_guaranteed_proposal_review_before_finalization_when_pass_missing(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                entropy_threshold=0.0,
                enable_consensus_audit=False,
                enable_problem_state_audit=False,
                enable_planning=False,
                enable_reversal_audit=False,
            ),
        )
        seeded = SynthesisProposal(
            "request voluntary release from both groups",
            ["care", "duty"],
            ["CARE", "DUTY"],
            0.82,
            "pareto attempt across constraints",
            accepted=True,
            proposal_id="P0",
            promotion_status="UNDER_REVIEW",
            admission_status="UNDER_REVIEW",
        )

        def pending(result):
            if not any(item.proposal_id == "P0" for item in result.synthesis_proposals):
                result.synthesis_proposals.append(seeded)
            return WorkspaceEngine._pending_proposal_for_guaranteed_review(engine, result)

        engine._pending_proposal_for_guaranteed_review = pending  # type: ignore[method-assign]
        result = engine.run(
            "A disputed allocation problem.",
            ["protect", "disclose"],
        )
        self.assertTrue(any(
            (cycle.received_broadcast or cycle.broadcast).constraint == "PROPOSAL_REVIEW"
            for cycle in result.cycles
        ))
        self.assertTrue(any(
            decision.content_type == "PROPOSAL_REVIEW"
            and "guaranteed_proposal_review" in decision.signals
            for decision in result.access_decisions
        ))

    def test_pending_proposal_helper_skips_completed_review_pass(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "disclose", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2),
        )
        proposal = SynthesisProposal(
            "bridge carefully", ["care", "duty"], ["CARE", "DUTY"],
            0.9, "bridge", accepted=True, proposal_id="P0",
            promotion_status="UNDER_REVIEW",
        )
        review_broadcast = WorkspaceBroadcast(
            constraint="PROPOSAL_REVIEW",
            intent="review_p0",
            reformulation_context="Review proposal P0: bridge carefully.",
        )
        cycle = CycleRecord(
            1, review_broadcast, [], None, None,
            {"protect": 0.6, "disclose": 0.4}, 0.9, 0, 1.0,
            received_broadcast=review_broadcast,
        )
        result = WorkspaceResult(
            scenario="x",
            actions=["protect", "disclose"],
            cycles=[cycle],
        )
        result.synthesis_proposals.append(proposal)
        self.assertIsNone(engine._pending_proposal_for_guaranteed_review(result))

    def test_proposal_review_parser_rejects_unproven_scenario_inheritance(self):
        data = {
            "scores": {"A0": 0.6, "A1": 0.4}, "r": "A0",
            "c": "CARE", "u": "CHECK_FEASIBILITY",
            "w": "proposal remains under review", "j": "NONE",
            "fa": "care evaluates dependency and responsiveness", "fr": True,
            "pr": {
                "proposal_id": "P0", "framework_status": "QUALIFIES",
                "framework_reason": "could support vulnerable households conditionally",
                "predicted_consequences": [{
                    "dimension": "material sufficiency",
                    "subject": "vulnerable households",
                    "direction": "IMPROVES",
                    "grounding_status": "SCENARIO_INHERITED",
                    "provenance": ["P0"],
                }],
                "feasibility_concerns": ["funding remains uncertain"],
                "required_conditions": ["adequate funding"],
            },
        }
        candidate = _candidate_from_data(
            "care", ["assist", "decline"], data,
            WorkspaceBroadcast(
                constraint="PROPOSAL_REVIEW",
                problem_state={"proposals": [{
                    "proposal_id": "P0", "promotion_status": "UNDER_REVIEW",
                }]},
            ),
            "NONE", {},
        )
        self.assertIsNotNone(candidate.proposal_review)
        self.assertFalse(candidate.proposal_review.valid)
        self.assertTrue(any(
            "scenario clause provenance" in error
            for error in candidate.proposal_review.validation_errors
        ))

    def test_cycle_budget_returns_contested_plurality_instead_of_no_judgment(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "pull", "HARM"),
                FixedSpecialist("care", "pull", "CARE"),
                FixedSpecialist("virtue", "pull", "CHARACTER"),
                FixedSpecialist("duty", "do not pull", "DUTY"),
                FixedSpecialist("rawlsian", "do not pull", "RIGHTS"),
            ],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.60, enable_synthesis=False),
        )
        result = engine.run("A closed moral conflict.", ["pull", "do not pull"])
        self.assertEqual(result.halted_by, "cycle_budget")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertEqual(result.selected_action, "pull")
        self.assertEqual(result.current_plurality, "pull")
        self.assertIn("prefer pull", result.compressed_rule.casefold())
        answer = render_public_judgment(result)
        self.assertIn("**pull.**", answer)
        self.assertIn("## Why the Parliament currently favors this action", answer)
        self.assertIn("## Deliberation Map", answer)
        self.assertNotIn("judged overriding rather than the leading consideration", answer)
        self.assertNotIn("Cycle 1", answer)
        self.assertNotIn("action_scores", answer)
        self.assertTrue(result.termination_assessment.resource_censored)
        self.assertEqual(
            result.termination_assessment.termination_type, "RESOURCE_CENSORED"
        )
        self.assertFalse(result.further_deliberation_estimate.affects_stopping)
        self.assertIn(
            "The cycle budget was reached before deliberation naturally converged.",
            answer,
        )
        self.assertNotIn("resource-censored", answer)

    def test_cycle_budget_action_judgment_gets_provisional_compressed_rule(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("duty", "protect", "DUTY"),
            ],
            WorkspaceConfig(
                max_cycles=1,
                stable_cycles_required=2,
                enable_synthesis=False,
            ),
        )
        result = engine.run("Choose protect or wait.", ["protect", "wait"])
        self.assertEqual(result.halted_by, "cycle_budget")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertIn("provisionally prefer protect", result.compressed_rule)
        self.assertNotIn("Unavailable", result.compressed_rule)

    def test_compressed_rule_uses_winning_constraint_and_preserves_plural_dissent(self):
        class PluralRuleSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint, rule, assessment):
                super().__init__(name, preferred, constraint)
                self.rule = rule
                self.assessment = assessment

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.decision_rule = self.rule
                chunk.framework_action_map = {self.preferred: self.assessment}
                chunk.landscape_decisive_axis = self.assessment
                if self.name == "deontological":
                    chunk.surprise = 1.0
                    chunk.friction = 1.0
                    chunk.epistemic_confidence = 1.0
                    chunk.confidence = 1.0
                elif self.name == "utilitarian":
                    chunk.epistemic_confidence = 1.0
                    chunk.confidence = 1.0
                else:
                    chunk.surprise = 0.1
                    chunk.friction = 0.2
                    chunk.epistemic_confidence = 0.15
                    chunk.confidence = 0.15
                return chunk

        result = WorkspaceEngine(
            [
                PluralRuleSpecialist(
                    "deontological", "A0", "DUTY",
                    "Prefer A0 when an impartial public rule requires it",
                    "REQUIRED: impartial treatment governs the allocation",
                ),
                PluralRuleSpecialist(
                    "utilitarian", "A0", "UNCERTAINTY",
                    "Prefer A0 when expected welfare is larger",
                    "larger expected welfare",
                ),
                PluralRuleSpecialist(
                    "virtue", "A1", "CHARACTER", "Prefer A1 under practical wisdom",
                    "EXEMPLIFIES: compassion and practical wisdom",
                ),
                PluralRuleSpecialist(
                    "care", "A1", "CARE", "Prefer A1 under concrete dependence",
                    "existing dependency and entrusted care",
                ),
                PluralRuleSpecialist(
                    "rawlsian", "A1", "FAIRNESS", "Prefer A1 for the least advantaged",
                    "IMPROVES: position of the least advantaged",
                ),
            ],
            WorkspaceConfig(
                max_cycles=1, enable_synthesis=False, enable_reversal_audit=False,
                consensus_audit_min_signals=99,
            ),
        ).run("A closed plural conflict.", ["A0", "A1"])

        self.assertIn("Rule [DUTY]", result.compressed_rule)
        self.assertIn("impartial public rule", result.compressed_rule)
        self.assertNotIn("expected welfare is larger", result.compressed_rule)
        self.assertIn("virtue/CHARACTER", result.compressed_rule)
        self.assertIn("care/CARE", result.compressed_rule)
        self.assertIn("rawlsian/FAIRNESS", result.compressed_rule)
        answer = render_public_judgment(result)
        self.assertIn("**Governing claim:**", answer)
        self.assertIn("impartial public rule", answer)

    def test_public_judgment_uses_valid_landscape_for_action_comparison(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "pull", "IMMINENT_HARM"),
                FixedSpecialist("duty", "wait", "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1, entropy_threshold=0.0, enable_synthesis=False),
        )
        result = engine.run("One dies if pulled; five die otherwise.", ["pull", "wait"])
        final = result.cycles[-1]
        final.candidates[0].landscape_cases = {
            "pull": "Kills one but saves five",
            "wait": "Avoids direct action but five die",
        }
        final.candidates[0].landscape_semantic_valid = True
        final.candidates[0].landscape_search_complete = True
        final.candidates[0].landscape_decisive_axis = "aggregate lives saved"
        answer = render_public_judgment(result)
        self.assertIn("## Deliberation Map", answer)
        self.assertIn("Kills one but saves five", answer)
        # Opposing landscape stays available in the map contribution / why section
        # without a separate comparison appendix.
        self.assertNotIn("Original action and consequence comparison", answer)

    def test_public_judgment_ignores_semantically_invalid_landscape_case(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "CARE"), FixedSpecialist("duty", "decline", "DUTY")],
            WorkspaceConfig(max_cycles=1, entropy_threshold=0.0, enable_synthesis=False),
        )
        result = engine.run("Choose whether to protect.", ["protect", "decline"])
        result.cycles[-1].candidates[0].landscape_cases = {
            "protect": "Fabricated inverted consequence",
            "decline": "Another fabricated consequence",
        }
        result.cycles[-1].candidates[0].landscape_semantic_valid = False
        answer = render_public_judgment(result)
        self.assertNotIn("Fabricated inverted consequence", answer)

    def test_public_judgment_does_not_replace_original_alternative_with_synthesis(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "child", "CARE"), FixedSpecialist("utility", "climbers", "IMMINENT_HARM")],
            WorkspaceConfig(max_cycles=1, entropy_threshold=0.0, enable_synthesis=False),
        )
        result = engine.run("Choose child or climbers.", ["child", "climbers"])
        result.actions.append("invented threshold option")
        result.synthesis_proposals.append(SynthesisProposal(
            "invented threshold option", ["care", "utility"], ["CARE", "IMMINENT_HARM"],
            0.8, "attempted bridge", accepted=True,
        ))
        candidate = result.cycles[-1].candidates[0]
        candidate.landscape_semantic_valid = True
        candidate.landscape_cases = {
            "child": "Certainly saves one child",
            "climbers": "Forty percent chance saves three climbers",
            "invented threshold option": "Invented policy",
        }
        answer = render_public_judgment(result)
        self.assertIn("climbers", answer)
        why = answer.split("## Alternative action discovered", 1)[0]
        self.assertNotIn("invented threshold option", why)
        self.assertIn("## Alternative action discovered", answer)
        self.assertIn("invented threshold option", answer)
        self.assertIn("## Deliberation Map", answer)

    def test_public_judgment_recovers_audit_reversals_across_cycles(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "reject", "CARE"), FixedSpecialist("utility", "reject", "IMMINENT_HARM")],
            WorkspaceConfig(max_cycles=2),
        )
        result = engine.run("Approve or reject.", ["approve", "reject"])
        audited = result.cycles[0].candidates[0]
        audited.assumption_status = "CONDITIONAL"
        audited.unsupported_assumption = "future harm cannot be mitigated"
        audited.reversal_condition = "future harm becomes reversible"
        answer = render_public_judgment(result)
        self.assertIn("## What could change the judgment", answer)
        self.assertIn("future harm cannot be mitigated", answer.casefold())
        self.assertIn("future harm becomes reversible", answer.casefold())
        self.assertNotIn("Uncertain assumptions identified during audit", answer)

    def test_rejected_synthesis_does_not_block_deliberation(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "CARE"), FixedSpecialist("duty", "disclose", "DUTY")],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )

        def synthesize(*_args):
            return SynthesisProposal(
                "call authorities", ["care"], ["CARE"], 0.2, "escape",
                accepted=False, rejection_reason="insufficiently feasible",
            )

        result = engine.run("Immediate choice", ["protect", "disclose"], synthesize=synthesize)
        self.assertNotIn("call authorities", result.actions)
        self.assertEqual(len(result.cycles), 2)
        self.assertFalse(result.synthesis_proposals[0].accepted)

    def test_synthesis_failure_is_recorded_and_does_not_crash(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "CARE"), FixedSpecialist("duty", "disclose", "DUTY")],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )

        def broken_synthesis(*_args):
            raise RuntimeError("temporary model failure")

        result = engine.run(
            "A disputed problem", ["protect", "disclose"], synthesize=broken_synthesis
        )
        self.assertEqual(len(result.cycles), 2)
        self.assertFalse(result.synthesis_proposals[0].accepted)
        self.assertIn("temporary model failure", result.synthesis_proposals[0].rejection_reason)

    def test_unanimous_policy_converges(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "VULNERABILITY"),
                FixedSpecialist("deontology", "protect", "RIGHTS"),
                FixedSpecialist("utilitarian", "protect", "HARM"),
            ],
            WorkspaceConfig(max_cycles=4, stable_cycles_required=2),
        )
        result = engine.run("A test scenario", ["protect", "disclose"])
        self.assertEqual(result.selected_action, "protect")
        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(len(result.cycles), 2)
        self.assertGreater(result.confidence, 0.9)
        self.assertTrue(result.termination_assessment.endogenous_stop)
        self.assertEqual(
            result.termination_assessment.termination_type, "ENDOGENOUS_CONVERGENCE"
        )

    def test_high_urgency_caps_cycles(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "VULNERABILITY"),
                FixedSpecialist("rule", "disclose", "HONESTY", "VERIFY_DANGER"),
            ],
            WorkspaceConfig(max_cycles=7, high_urgency_cycles=2, entropy_threshold=0.0),
        )
        result = engine.run(
            "Immediate danger",
            ["protect", "disclose"],
            WorkspaceBroadcast(urgency=0.95, danger_probability=0.9),
        )
        self.assertEqual(len(result.cycles), 2)
        self.assertEqual(result.halted_by, "cycle_budget")

    def test_dissent_and_reopen_condition_survive(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "VULNERABILITY"),
                FixedSpecialist("utilitarian", "protect", "HARM"),
                FixedSpecialist("rule", "disclose", "HONESTY", "VERIFY_DANGER"),
            ],
            WorkspaceConfig(max_cycles=2, entropy_threshold=0.0),
        )
        result = engine.run("Uncertain danger", ["protect", "disclose"])
        self.assertIn("VERIFY_DANGER", result.reopen_conditions)
        self.assertIsNotNone(result.cycles[-1].dissent)
        self.assertEqual(result.cycles[-1].dissent.specialist, "rule")
        self.assertIn("HONESTY", result.moral_residue)
        honesty = next(
            record for record in result.moral_residue_records
            if record.constraint == "HONESTY"
        )
        self.assertIn("rule", honesty.source_specialists)
        self.assertTrue(honesty.compatible_with_recommendation)

    def test_substantive_dissent_creates_reversal_condition_after_convergence(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utility", "pull", "IMMINENT_HARM"),
                FixedSpecialist("care", "pull", "CARE"),
                FixedSpecialist("virtue", "pull", "CHARACTER"),
                FixedSpecialist("duty", "wait", "RIGHTS"),
            ],
            WorkspaceConfig(
                max_cycles=3,
                entropy_threshold=1.0,
                stable_cycles_required=2,
                enable_synthesis=False,
            ),
        )
        result = engine.run("One dies if acting; five die otherwise.", ["pull", "wait"])
        self.assertEqual(result.halted_by, "convergence")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertEqual(result.selected_action, "pull")
        self.assertEqual(result.reopen_conditions, [])
        answer = render_public_judgment(result)
        self.assertNotIn("judged overriding rather than the leading consideration", answer)

    def test_dissent_reversal_prefers_specific_existing_condition(self):
        dissent = CandidateChunk(
            "care", "CARE", {"protect": 0.1, "disclose": 0.9},
            0.4, 0.8, 0.7, recommended_action="disclose",
            rationale="secrecy prolongs preventable harm",
            reversal_condition="the protected person reaches safety",
        )
        condition = WorkspaceEngine._dissent_reversal_condition(dissent, "protect")
        self.assertEqual(
            condition,
            "If the protected person reaches safety, prefer disclose",
        )

    def test_dissent_typed_abandonment_threshold_is_not_used_as_reopen_condition(self):
        dissent = CandidateChunk(
            "care", "CARE", {"purge": 0.2, "counter-code": 0.8},
            0.4, 0.8, 0.7, recommended_action="counter-code",
            rationale="maintains care relationships",
            factual_reversal_threshold="counter-code failure risk rises above 80%",
        )
        condition = WorkspaceEngine._dissent_reversal_condition(dissent, "purge")
        self.assertEqual(condition, "")

    def test_result_is_json_serializable_shape(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "VULNERABILITY")],
            WorkspaceConfig(max_cycles=1),
        )
        result = engine.run("A test", ["protect", "wait"])
        data = result.to_dict()
        self.assertEqual(data["cycles"][0]["winner"]["specialist"], "care")
        self.assertIn("termination_type", data["termination_assessment"])
        self.assertFalse(data["further_deliberation_estimate"]["affects_stopping"])
        self.assertIn("action_change_signal", data["further_deliberation_estimate"])

    def test_progress_reports_delegate_and_policy(self):
        messages = []
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "VULNERABILITY")],
            WorkspaceConfig(max_cycles=1),
        )
        engine.run("A test", ["protect", "wait"], progress=messages.append)
        self.assertTrue(any("care delegate thinking" in message for message in messages))
        self.assertTrue(any("cycle policy" in message for message in messages))

    def test_invalid_candidate_cannot_affect_policy_or_salience(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("utilitarian", "protect", "IMMINENT_HARM"),
                InvalidSpecialist("broken", "wait", "0.0"),
            ],
            WorkspaceConfig(max_cycles=1),
        )
        result = engine.run("A test", ["protect", "wait"])
        broken = next(c for c in result.cycles[0].candidates if c.specialist == "broken")
        self.assertEqual(broken.salience, 0.0)
        self.assertEqual(result.selected_action, "protect")

    def test_too_few_valid_candidates_is_inconclusive(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                InvalidSpecialist("broken", "wait", "0.0"),
            ],
            WorkspaceConfig(max_cycles=2, min_valid_specialists=2),
        )
        result = engine.run("A test", ["protect", "wait"])
        self.assertEqual(result.halted_by, "insufficient_valid_candidates")
        self.assertEqual(result.selected_action, "INCONCLUSIVE")
        self.assertEqual(result.confidence, 0.0)
        self.assertTrue(result.compressed_rule.startswith("Unavailable"))
        self.assertEqual(result.cycles[-1].winner.specialist, "care")
        self.assertEqual(result.cycles[-1].execution_status, "SYSTEM_ERROR")
        self.assertEqual(
            result.cycles[-1].system_error, "INSUFFICIENT_VALID_DELEGATES",
        )

    def test_all_invalid_cycle_preserves_last_valid_problem_state(self):
        previous = {
            "cycle": 1,
            "live_actions": [{"action_id": "A0", "action": "protect"}],
            "agent_positions": [{"specialist": "care", "preferred_action": "protect"}],
            "unresolved_questions": [{
                "question_key": "QUESTION:prior",
                "category": "MIXED_UNCERTAINTY",
                "question": "Prior live question",
            }],
            "unresolved_categories": ["MIXED_UNCERTAINTY"],
            "primary_unresolved": "MIXED_UNCERTAINTY",
            "problem_delta": {"new_questions": []},
        }
        failed = CandidateChunk(
            specialist="deontological", constraint="MALFORMED_RESPONSE",
            action_scores={"protect": 0.5}, surprise=0.0, friction=0.0,
            confidence=0.0, schema_valid=False,
        )

        preserved = _preserve_problem_state_after_invalid_cycle(previous, [failed])

        self.assertEqual(preserved["agent_positions"], previous["agent_positions"])
        self.assertEqual(preserved["unresolved_questions"][0]["question_key"], "QUESTION:prior")
        self.assertEqual(preserved["primary_unresolved"], "REVIEW_MODEL_OUTPUT")
        self.assertEqual(
            preserved["unresolved_questions"][-1]["question_key"],
            "QUESTION:MODEL_OUTPUT_FAILURE",
        )
        self.assertEqual(previous["primary_unresolved"], "MIXED_UNCERTAINTY")

    def test_all_invalid_cycle_has_system_error_and_no_moral_winner(self):
        result = WorkspaceEngine(
            [InvalidSpecialist("virtue", "protect", "CHARACTER")],
            WorkspaceConfig(max_cycles=1, min_valid_specialists=1),
        ).run("A test", ["protect", "wait"])

        cycle = result.cycles[0]
        self.assertIsNone(cycle.winner)
        self.assertEqual(cycle.execution_status, "SYSTEM_ERROR")
        self.assertEqual(cycle.system_error, "INSUFFICIENT_VALID_DELEGATES")
        self.assertEqual(cycle.candidates[0].constraint, "NONE")
        self.assertEqual(
            cycle.candidates[0].delegate_status,
            "SEMANTIC_VALIDATION_ERROR",
        )

    def test_time_budget_takes_precedence_over_possible_convergence(self):
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                FixedSpecialist("utilitarian", "protect", "IMMINENT_HARM"),
            ],
            WorkspaceConfig(
                max_cycles=2,
                time_budget_seconds=0.0,
                entropy_threshold=1.0,
                stable_cycles_required=1,
            ),
        )
        result = engine.run("A test", ["protect", "wait"])
        self.assertEqual(result.halted_by, "time_budget")
        self.assertEqual(result.judgment_status, "GOVERNED_RECOMMENDATION")
        self.assertEqual(result.selected_action, "protect")
        self.assertIn("prefer protect", result.compressed_rule.casefold())


class BridgeTests(unittest.TestCase):
    def test_problem_reformulation_builds_hypothetical_switch_point(self):
        class ReformulationLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"u":["comparative scale of disease and instability"],"o":['
                    '{"a":"A0","k":"mortality","d":"BENEFIT","x":"lives saved",'
                    '"p":1.0,"m":1000,"unit":"lives saved","h":"10 years","f":"MORTALITY","b":"treated population"},'
                    '{"a":"A0","k":"employment","d":"HARM","x":"employment disruption",'
                    '"p":0.1,"m":1000000,"unit":"jobs","h":"1 year","f":"ECONOMIC","b":"affected workers"},'
                    '{"a":"A1","k":"mortality","d":"HARM","x":"foregone treatment",'
                    '"p":1.0,"m":1000,"unit":"lives","h":"10 years","f":"MORTALITY","b":"treated population"},'
                    '{"a":"A1","k":"employment","d":"BENEFIT","x":"jobs protected",'
                    '"p":0.1,"m":1000000,"unit":"jobs","h":"1 year","f":"ECONOMIC","b":"affected workers"}],'
                    '"s":"consensus splits when livelihood disruption rivals preventable mortality",'
                    '"t":"whether immediate duties justify imposing diffuse economic risk",'
                    '"q":"At these hypothetical stakes, should we release the cure or suppress it?",'
                    '"g":["utilitarian","care"],"fixed":["the cure exists"],'
                    '"c":[{"n":"temporal duty","v":{"A0":"acts on present preventable deaths",'
                    '"A1":"protects against longer-term systemic disruption"},'
                    '"e":"present rescue and future stability impose distinct obligations",'
                    '"fixed":true}],"changed":[],"hyp":true}'
                )}]}

        candidates = [
            CandidateChunk(
                name, constraint, {"release": 0.6, "suppress": 0.4},
                0.2, 0.2, 0.2, recommended_action="release",
                assumption_status="UNDERDETERMINED",
                unsupported_assumption="comparative harms are unknown",
                reversal_condition="instability causes greater harm",
            )
            for name, constraint in (("utilitarian", "UNCERTAINTY"), ("care", "CARE"))
        ]
        proposal = propose_problem_reformulation(
            ReformulationLlm(),
            "A cure could destabilize society.",
            ["release the cure", "suppress the cure"],
            candidates,
        )
        self.assertTrue(proposal.accepted)
        self.assertTrue(proposal.hypothetical)
        self.assertEqual({outcome.unit for outcome in proposal.outcomes}, {"lives", "jobs"})
        self.assertIn("HYPOTHETICAL CALIBRATION", proposal.compact())
        self.assertEqual(
            {comparison.dimension for comparison in proposal.numeric_comparisons},
            {"mortality", "employment"},
        )
        self.assertEqual(
            {comparison.measurement_family for comparison in proposal.numeric_comparisons},
            {"MORTALITY", "ECONOMIC"},
        )
        self.assertTrue(proposal.unresolved_numeric_tradeoffs)
        self.assertIn("Non-commensurable tradeoffs", proposal.compact())
        self.assertTrue(proposal.switch_condition.startswith("Python-computed calibration:"))

    def test_reformulation_rejects_same_dimension_with_noncommensurable_units(self):
        class MismatchedLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"u":["comparative harm"],"o":['
                    '{"a":"A0","k":"human harm","d":"HARM","x":"one death",'
                    '"p":1.0,"m":1,"unit":"lives","h":"immediate","f":"MORTALITY","b":"affected people"},'
                    '{"a":"A1","k":"human harm","d":"HARM","x":"extended distress",'
                    '"p":1.0,"m":30,"unit":"distress days","h":"30 days","f":"HEALTH_DURATION","b":"affected people"}],'
                    '"s":"support splits when harms receive equal moral importance",'
                    '"t":"death and temporary distress lack an agreed exchange rate",'
                    '"q":"At these stakes, should the actor choose action zero or action one?",'
                    '"g":["utilitarian","care"],"fixed":["only two actions exist"],'
                    '"c":[{"n":"harm type","v":{"A0":"mortality","A1":"distress"},'
                    '"e":"different harms require a moral exchange rate","fixed":true}],'
                    '"changed":[],"hyp":true}'
                )}]}

        candidates = [
            CandidateChunk(
                name, constraint, {"action zero": 0.5, "action one": 0.5},
                0.2, 0.2, 0.2, recommended_action="action zero",
                assumption_status="UNDERDETERMINED",
                unsupported_assumption="comparative harm remains unknown",
                reversal_condition="the competing harm receives greater weight",
            )
            for name, constraint in (("utilitarian", "UNCERTAINTY"), ("care", "CARE"))
        ]
        proposal = propose_problem_reformulation(
            MismatchedLlm(), "Only action zero or action one may be chosen.",
            ["action zero", "action one"], candidates,
        )

        self.assertFalse(proposal.accepted)
        self.assertIn("inconsistent family, unit, population, or time bases", proposal.rejection_reason)

    def test_specialist_names_decisive_axis_at_reformulation_boundary(self):
        class BoundaryLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.45,"A1":0.55},"r":"A1",'
                    '"c":"DUTY","u":"VERIFY_FACTS","w":"agency defeats casualty parity",'
                    '"j":"causal agency remains decisive","bp":"SPLIT",'
                    '"dx":"causal agency doing versus allowing",'
                    '"sv":"switch toward A0 if redirection no longer constitutes intentional harm"}'
                )}]}

        chunk = CompactLocalSpecialist("deontological", BoundaryLlm()).evaluate(
            "A trolley dilemma.",
            ["pull the lever", "do nothing"],
            WorkspaceBroadcast(
                constraint="PROBLEM_REFORMULATION",
                reformulation_context="HYPOTHETICAL CALIBRATION with causal agency axis",
            ),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertEqual(chunk.boundary_position, "SPLIT")
        self.assertEqual(chunk.decisive_axis, "causal agency doing versus allowing")
        self.assertIn("redirection", chunk.boundary_switch_condition)

    def test_problem_reformulation_rejects_changed_fixed_facts(self):
        class ChangingFactsLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"u":["severity of distress"],"o":['
                    '{"a":"A0","k":"wellbeing","d":"BENEFIT","x":"relief from distress",'
                    '"p":0.8,"m":5,"unit":"wellbeing points","h":"1 year","f":"WELLBEING","b":"affected person"},'
                    '{"a":"A1","k":"wellbeing","d":"HARM","x":"continued distress",'
                    '"p":0.8,"m":5,"unit":"wellbeing points","h":"1 year","f":"WELLBEING","b":"affected person"}],'
                    '"s":"support changes if the stranger volunteers to receive the memory",'
                    '"t":"self relief conflicts with autonomy and nonconsensual harm",'
                    '"q":"Should the actor erase and transfer or keep the memory?",'
                    '"g":["deontological","care"],'
                    '"fixed":["the recipient is an unknown stranger","the memory transfers intact"],'
                    '"c":[{"n":"consent","v":{"A0":"recipient cannot consent",'
                    '"A1":"no burden is transferred"},"e":"autonomy constrains imposed harm",'
                    '"fixed":true}],"changed":[],"hyp":true}'
                )}]}

        candidates = [
            CandidateChunk(
                name, constraint, {"erase and transfer": 0.4, "keep": 0.6},
                0.2, 0.2, 0.2, recommended_action="keep",
                assumption_status="CONDITIONAL",
                unsupported_assumption="severity may differ",
                reversal_condition="a volunteer could accept the burden",
            )
            for name, constraint in (("deontological", "DUTY"), ("care", "CARE"))
        ]
        proposal = propose_problem_reformulation(
            ChangingFactsLlm(),
            "The memory transfers intact into an unknown stranger.",
            ["erase and transfer", "keep"],
            candidates,
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("different scenario", proposal.rejection_reason)

    def test_trolley_reformulation_preserves_causal_axis_with_deterministic_outcomes(self):
        class TrolleyLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"u":["moral weight of acting versus allowing"],"o":['
                    '{"a":"A0","k":"mortality","d":"HARM","x":"one person is killed after redirection",'
                    '"p":1.0,"m":1,"unit":"lives lost","h":"immediately","f":"MORTALITY","b":"people on tracks"},'
                    '{"a":"A1","k":"mortality","d":"HARM","x":"five people are killed on the existing track",'
                    '"p":1.0,"m":5,"unit":"lives lost","h":"immediately","f":"MORTALITY","b":"people on tracks"}],'
                    '"s":"the coalition splits when preventing four deaths requires actively redirecting harm",'
                    '"t":"fewer deaths conflict with agency and doing rather than allowing harm",'
                    '"q":"Should the bystander pull the lever or leave the trolley on its current track?",'
                    '"g":["utilitarian","deontological"],'
                    '"fixed":["the trolley will kill five unless redirected","redirection kills one"],'
                    '"c":[{"n":"causal agency","v":{"A0":"actively redirects the threat",'
                    '"A1":"allows the existing trajectory"},'
                    '"e":"doing harm may carry a distinct duty from allowing harm","fixed":true}],'
                    '"changed":[],"hyp":true}'
                )}]}

        candidates = [
            CandidateChunk(
                name, constraint, {"pull the lever": 0.5, "do nothing": 0.5},
                0.2, 0.2, 0.2, recommended_action="pull the lever",
                assumption_status="UNDERDETERMINED",
                unsupported_assumption="moral weight of causal agency is unresolved",
                reversal_condition="active redirection is treated as impermissible doing",
            )
            for name, constraint in (("utilitarian", "UNCERTAINTY"), ("deontological", "DUTY"))
        ]
        proposal = propose_problem_reformulation(
            TrolleyLlm(),
            "A trolley will kill five people unless you pull a lever that redirects it and kills one.",
            ["pull the lever", "do nothing"],
            candidates,
        )
        self.assertTrue(proposal.accepted)
        self.assertEqual(proposal.outcomes[0].unit, "lives")
        self.assertEqual(proposal.categorical_axes[0].name, "causal agency")
        self.assertTrue(proposal.categorical_axes[0].fixed_by_scenario)

    def test_consensus_audit_requires_assumption_and_reversal_condition(self):
        class AuditLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0",'
                    '"c":"UNCERTAINTY","u":"VERIFY_FACTS",'
                    '"w":"comparative harms remain unknown","j":"NONE",'
                    '"d":"UNDERDETERMINED",'
                    '"a":"disease burden exceeds instability harm",'
                    '"v":"instability harms more people overall"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", AuditLlm()).evaluate(
            "A cure could destabilize society.",
            ["publish", "conceal"],
            WorkspaceBroadcast(
                constraint="CONSENSUS_AUDIT",
                unresolved="VERIFY_ASSUMPTIONS",
                contingency_question="What facts would reverse the recommendation?",
            ),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertEqual(chunk.assumption_status, "UNDERDETERMINED")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertAlmostEqual(chunk.action_scores["publish"], 0.605)
        self.assertAlmostEqual(chunk.action_scores["conceal"], 0.395)
        self.assertEqual(chunk.reversal_condition, "instability harms more people overall")

    def test_live_audit_schema_contains_no_optional_legacy_fields(self):
        captured = {}

        class SchemaCaptureLlm:
            def complete_json(self, prompt, *, schema, **kwargs):
                captured.update(schema)
                raise RuntimeError("schema captured")

        with self.assertRaisesRegex(RuntimeError, "schema captured"):
            CompactLocalSpecialist("deontological", SchemaCaptureLlm()).evaluate(
                "A policy improves welfare but restricts liberty.",
                ["adopt policy", "reject policy"],
                WorkspaceBroadcast(
                    constraint="CONSENSUS_AUDIT",
                    audit_variable={
                        "entity": "welfare versus liberty",
                        "relation": "COMPARATIVE_MAGNITUDE",
                        "possible_values": ["NO_CHANGE", "REVERSES"],
                        "focus_action": "A0",
                        "question": "Would the comparison reverse the recommendation?",
                    },
                ),
            )

        properties = captured["properties"]
        self.assertTrue({"ap", "ax", "ie"} <= set(properties))
        self.assertFalse({"d", "a", "v", "av"} & set(properties))
        self.assertEqual(set(captured["required"]), set(properties))

    def test_audit_uncertainty_persists_without_new_facts(self):
        class PersistentAuditLlm:
            def __call__(self, prompt, **kwargs):
                audit_fields = (
                    ',"d":"CONDITIONAL",'
                    '"a":"disease burden exceeds instability harm",'
                    '"v":"instability harms more people overall"'
                    if "CONSENSUS_AUDIT" in prompt else ""
                )
                unresolved = "VERIFY_FACTS" if audit_fields else "NONE"
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0",'
                    f'"c":"UNCERTAINTY","u":"{unresolved}",'
                    f'"w":"comparative harms remain uncertain","j":"NONE"{audit_fields}}}'
                )}]}

        delegate = CompactLocalSpecialist("utilitarian", PersistentAuditLlm())
        actions = ["publish", "conceal"]
        audited = delegate.evaluate(
            "A cure could destabilize society.", actions,
            WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
        )
        later = delegate.evaluate(
            "A cure could destabilize society.", actions,
            WorkspaceBroadcast(constraint="UNCERTAINTY"),
        )
        self.assertEqual(audited.assumption_status, "CONDITIONAL")
        self.assertEqual(later.assumption_status, "CONDITIONAL")
        self.assertAlmostEqual(later.action_scores["publish"], 0.68)
        self.assertEqual(later.reversal_condition, "instability harms more people overall")

    def test_planning_analysis_uses_strategic_forces_without_moral_vote(self):
        class PlanningLlm:
            prompt = ""

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"f":0.55,"n":"the witness retains access to evidence",'
                    '"x":"the witness loses access to evidence","b":"A1",'
                    '"a":["employer controls access"],"r":["limited time"],'
                    '"s":["INSTITUTIONAL_POWER","TIME_ENERGY_BUDGET"],"m":true,'
                    '"g":"limited time to report",'
                    '"va":true,"vr":"remaining silent requires no evidence access"}'
                )}]}

        llm = PlanningLlm()
        assessment = analyze_action_plan(
            llm,
            "A witness has limited time to report employer misconduct.",
            ["report the misconduct", "remain silent"],
            "report the misconduct",
            WorkspaceBroadcast(),
            [],
            "unresolved policy competition",
        )
        self.assertTrue(assessment.valid)
        self.assertTrue(assessment.broadcast_worthy)
        self.assertEqual(assessment.fallback, "remain silent")
        self.assertIn("strategic ecology", llm.prompt)
        self.assertNotIn("action_scores", assessment.__dataclass_fields__)

    def test_planning_rejects_invented_action_access_failure(self):
        class PlanningLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"f":0.2,"n":"a clinic offers the procedure",'
                    '"x":"no clinic will offer the procedure","b":"A1",'
                    '"a":[],"r":["clinic access"],'
                    '"s":["RESOURCE_SUPPLIERS"],"m":true,'
                    '"g":"memory-erasure procedure",'
                    '"va":true,"vr":"retaining testimony requires no clinic"}'
                )}]}

        assessment = analyze_action_plan(
            PlanningLlm(),
            "A parent may undergo a memory-erasure procedure or retain their testimony.",
            ["undergo memory erasure", "retain testimony"],
            "undergo memory erasure",
            WorkspaceBroadcast(),
            [],
            "unresolved policy competition",
        )
        self.assertFalse(assessment.valid)
        self.assertIn("unsupported access conditions", assessment.error)

    def test_failure_condition_becomes_explicit_fallback_question(self):
        class ContingencyLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"p":"the friend agrees to self-report",'
                    '"x":"the proposed self-report synthesis cannot proceed"}'
                )}]}

        condition = generate_failure_condition(
            ContingencyLlm(),
            "A witness must report a friend or remain silent.",
            "Urge the friend to self-report",
            ["report the crime", "remain silent"],
        )
        self.assertTrue(condition.valid)
        self.assertIn("NOT(the friend agrees to self-report)", condition.failure_condition)
        self.assertIn("fallback A0 or fallback A1", condition.contingency_question)
        self.assertEqual(condition.fallback_actions, ["report the crime", "remain silent"])
        self.assertEqual(validate_contingency_graph_dict(
            condition.semantic_graph, condition.synthesis_action, condition.fallback_actions,
            require_fallback_availability=False,
        ), [])

    def test_failure_condition_uses_typed_fallback_identity_not_prose_matching(self):
        class ContingencyLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"p":"the proposed safeguard remains effective",'
                    '"x":"the synthesis no longer resolves the conflict"}'
                )}]}

        actions = ["seal the facility", "deploy the experimental phage"]
        condition = generate_failure_condition(
            ContingencyLlm(), "A closed emergency choice.",
            "deploy the phage with a safeguard", actions,
        )
        self.assertTrue(condition.valid, condition.error)
        self.assertEqual(condition.fallback_actions, actions)

    def test_contingency_graph_rejects_failure_that_may_disable_a_fallback(self):
        validation = compile_contingency_graph(
            {
                "p": "the actor retains physical control",
                "x": "the actor loses the ability to execute actions",
            },
            "use the combined safeguard",
            ["choose first fallback", "choose second fallback"],
        )
        _, errors = certify_fallback_availability(
            validation.graph.to_dict(), "use the combined safeguard",
            ["choose first fallback", "choose second fallback"],
            {"A0": "UNAVAILABLE", "A1": "UNKNOWN"},
            {"A0": "control is lost", "A1": "execution is uncertain"},
        )
        self.assertIn("A0 is not independently", "; ".join(errors))
        self.assertIn("A1 is not independently", "; ".join(errors))

    def test_contingency_graph_detects_negation_direction_tampering(self):
        validation = compile_contingency_graph(
            {
                "p": "the safeguard remains operational",
                "x": "the synthesis loses its protective mechanism",
            },
            "use the combined safeguard",
            ["choose first fallback", "choose second fallback"],
        )
        graph = validation.graph.to_dict()
        failure = next(
            node for node in graph["nodes"] if node["id"] == "CONTINGENCY_FAILURE"
        )
        failure["attributes"]["truth_state"] = True
        errors = validate_contingency_graph_dict(
            graph, "use the combined safeguard",
            ["choose first fallback", "choose second fallback"],
            require_fallback_availability=False,
        )
        self.assertIn("not the typed negation", "; ".join(errors))

    def test_independent_feasibility_verifier_approves_both_fallbacks(self):
        class FeasibilityLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"fa":[{"action_id":"A0","status":"AVAILABLE",'
                    '"basis":"FAILURE_SCOPE","reason":"A0 uses an independent mechanism"},'
                    '{"action_id":"A1","status":"AVAILABLE",'
                    '"basis":"SCENARIO_STRUCTURE","reason":"A1 remains explicitly executable"}],'
                    '"sf":false}'
                )}]}

        condition = FailureCondition(
            "combined safeguard", "the safeguard works",
            "NOT(the safeguard works): combination fails", "Which fallback?",
            fallback_actions=["choose A0", "choose A1"],
            predicate_label="the safeguard works",
        )
        assessment = verify_contingency_feasibility(
            FeasibilityLlm(), "The actor can choose A0 or A1 independently.", condition,
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertTrue(assessment.approved, assessment.error)
        self.assertEqual(assessment.fallback_statuses["A1"], "AVAILABLE")

    def test_independent_feasibility_verifier_blocks_shared_failure(self):
        class FeasibilityLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"fa":[{"action_id":"A0","status":"UNAVAILABLE",'
                    '"basis":"FAILURE_SCOPE","reason":"shared control has been lost"},'
                    '{"action_id":"A1","status":"UNKNOWN",'
                    '"basis":"INSUFFICIENT","reason":"execution cannot be established"}],'
                    '"sf":true}'
                )}]}

        condition = FailureCondition(
            "combined safeguard", "the actor retains control",
            "NOT(the actor retains control): control is lost", "Which fallback?",
            fallback_actions=["choose A0", "choose A1"],
            predicate_label="the actor retains control",
        )
        assessment = verify_contingency_feasibility(
            FeasibilityLlm(), "The actor must retain control to act.", condition,
        )
        self.assertTrue(assessment.valid)
        self.assertFalse(assessment.approved)
        self.assertTrue(assessment.shared_failure)
        self.assertIn("shared", assessment.error)

    def test_contingency_review_requires_a_typed_fallback_answer(self):
        actions = ["cut Sector 4", "cut emergency dispatch", "use synthesis"]
        broadcast = WorkspaceBroadcast(
            constraint="CONTINGENCY_REVIEW",
            contingency_question="If aid fails, which original fallback should govern?",
            contingency_synthesis_action=actions[2],
            contingency_failure_condition="aid cannot arrive",
            contingency_fallback_actions=(actions[0], actions[1]),
            branch_kind="SYNTHESIS_CONTINGENCY",
        )
        base = {
            "scores": {"A0": .7, "A1": .3, "A2": .1}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE",
            "w": "containment prevents greater harm", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": .7,
        }
        with self.assertRaisesRegex(ValueError, "did not choose a fallback"):
            _candidate_from_data(
                "utilitarian", actions, base, broadcast, "NONE", {},
            )

        base.update({
            "cr": "A0",
            "cj": "Given aid failure, A0 minimizes expected casualties",
        })
        candidate = _candidate_from_data(
            "utilitarian", actions, base, broadcast, "NONE", {},
        )
        self.assertEqual(candidate.contingency_choice, actions[0])
        self.assertIn("aid failure", candidate.contingency_justification)

    def test_contingency_review_cannot_recommend_failed_synthesis(self):
        actions = ["fallback zero", "fallback one", "failed synthesis"]
        broadcast = WorkspaceBroadcast(
            constraint="CONTINGENCY_REVIEW",
            contingency_question="Which fallback should govern?",
            contingency_synthesis_action=actions[2],
            contingency_failure_condition="the synthesis condition fails",
            contingency_fallback_actions=(actions[0], actions[1]),
        )
        data = {
            "scores": {"A0": .2, "A1": .3, "A2": .9}, "r": "A2",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "repeat prior position",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": .7,
            "cr": "A0", "cj": "Given failure, fallback zero limits harm",
        }
        with self.assertRaisesRegex(ValueError, "recommendation does not match"):
            _candidate_from_data(
                "utilitarian", actions, data, broadcast, "NONE", {},
            )

    def test_compact_specialist_uses_lightweight_contingency_schema(self):
        calls = []

        class ContingencyLlm:
            def __call__(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.7,"A1":0.3},"cr":"A0",'
                    '"c":"CARE","u":"NONE",'
                    '"cj":"Given aid failure, A0 better protects dependents",'
                    '"z":0.72,"fr":true}'
                )}]}

        actions = ["choose original A0", "choose original A1"]
        candidate = CompactLocalSpecialist(
            "care", ContingencyLlm(), testimony="Protect dependent relationships."
        ).evaluate(
            "A synthesis was considered.", actions,
            WorkspaceBroadcast(
                constraint="CONTINGENCY_REVIEW",
                contingency_question="Which fallback should govern?",
                contingency_synthesis_action="combine the actions",
                contingency_failure_condition="aid fails",
                contingency_fallback_actions=tuple(actions),
                branch_kind="SYNTHESIS_CONTINGENCY",
            ),
        )
        self.assertTrue(candidate.schema_valid, candidate.validation_errors)
        self.assertEqual(candidate.contingency_choice, actions[0])
        schema = calls[0][1]["grammar"] if "grammar" in calls[0][1] else None
        # Local backends receive a grammar when llama.cpp is installed; the
        # prompt itself confirms that the generic landscape/EV payload is absent.
        self.assertNotIn("landscape", calls[0][0].casefold())
        self.assertNotIn("expected value", calls[0][0].casefold())
        self.assertLessEqual(calls[0][1]["max_tokens"], 112)

    @patch("global_workspace.legacy_bridge.subprocess.run")
    def test_original_testimony_is_decoded(self, run):
        payload = base64.b64encode(b"Corpus-grounded testimony").decode("ascii")
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=f"agent logs\n{RESPONSE_MARKER}{payload}\n",
            stderr="",
        )
        result = consult_original_agents(Path("scenario.json"), agents=("care",))
        self.assertEqual(result.testimonies["care"], "Corpus-grounded testimony")
        self.assertEqual(result.errors, {})

    @patch("global_workspace.legacy_bridge.subprocess.run")
    def test_openai_backend_is_forwarded_to_original_agents(self, run):
        payload = base64.b64encode(b"Hosted testimony").decode("ascii")
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=f"{RESPONSE_MARKER}{payload}\n",
            stderr="",
        )
        result = consult_original_agents(
            Path("scenario.json"),
            agents=("care",),
            backend="openai",
            openai_model="o3",
        )
        command = run.call_args.args[0]
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(command[command.index("--backend") + 1], "openai")
        self.assertEqual(command[command.index("--openai-model") + 1], "o3")
        self.assertEqual(child_env["ETHICS_LLM_BACKEND"], "openai")
        self.assertEqual(result.testimonies["care"], "Hosted testimony")

    @patch("global_workspace.legacy_bridge.subprocess.run")
    def test_original_agents_receive_canonical_order_not_presentation_order(self, run):
        payload = base64.b64encode(b"Canonical testimony").decode("ascii")
        observed = {}

        def complete(command, **kwargs):
            scenario_path = Path(command[command.index("--scenario") + 1])
            observed.update(json.loads(scenario_path.read_text(encoding="utf-8")))
            return SimpleNamespace(
                returncode=0,
                stdout=f"{RESPONSE_MARKER}{payload}\n",
                stderr="",
            )

        run.side_effect = complete
        with tempfile.TemporaryDirectory() as directory:
            scenario_path = Path(directory) / "scenario.json"
            scenario_path.write_text(json.dumps({
                "ethical_question": (
                    "A vehicle loses braking. Option A: maintain course and kill three. "
                    "Option B: swerve and kill one."
                )
            }))
            canonical = tuple(canonicalize_action_order([
                "maintain course and kill three",
                "swerve and kill one",
            ]))
            result = consult_original_agents(
                scenario_path, agents=("care",), canonical_actions=canonical,
            )

        question = observed["ethical_question"]
        self.assertIn(f"Action A0: {canonical[0]}", question)
        self.assertIn(f"Action A1: {canonical[1]}", question)
        self.assertNotIn("Option A:", question)
        self.assertEqual(result.testimonies["care"], "Canonical testimony")

    def test_canonical_action_order_is_invariant_to_input_permutation(self):
        actions = ["maintain course", "swerve into barrier"]
        self.assertEqual(
            canonicalize_action_order(actions),
            canonicalize_action_order(list(reversed(actions))),
        )

    def test_deliberation_scenario_is_invariant_to_option_permutation(self):
        first = (
            "A vehicle loses braking. Option A: maintain course and kill three. "
            "Option B: swerve and kill one. Casualties under Option A are unlogged."
        )
        second = (
            "A vehicle loses braking. Option A: swerve and kill one. "
            "Option B: maintain course and kill three. Casualties under Option B are unlogged."
        )
        first_legend = extract_labeled_action_legend(first)
        second_legend = extract_labeled_action_legend(second)
        canonical = canonicalize_action_order(list(first_legend.values()))
        normalized_first = canonicalize_deliberation_scenario(
            first, first_legend, canonical,
        )
        normalized_second = canonicalize_deliberation_scenario(
            second, second_legend, canonical,
        )

        self.assertEqual(normalized_first, normalized_second)
        self.assertNotIn("Option A", normalized_first)
        unlogged_id = next(
            f"A{index}" for index, action in enumerate(canonical)
            if "maintain course" in action.casefold()
        )
        self.assertIn(f"Casualties under {unlogged_id} are unlogged", normalized_first)

    def test_graph_action_identity_matches_structural_paraphrases(self):
        maintain_a = (
            "Maintain the vehicle's current trajectory, killing three pedestrians"
        )
        maintain_b = (
            "Continue on the current path, causing the deaths of three pedestrians"
        )
        redirect_a = (
            "Swerve into a concrete barrier, killing one passenger and saving three pedestrians"
        )
        redirect_b = (
            "Turn into the concrete barrier; one passenger dies while three pedestrians are spared"
        )

        self.assertEqual(semantic_action_key(maintain_a), semantic_action_key(maintain_b))
        self.assertEqual(semantic_action_key(redirect_a), semantic_action_key(redirect_b))
        self.assertNotEqual(semantic_action_key(maintain_a), semantic_action_key(redirect_a))
        first_order = canonicalize_action_order([maintain_a, redirect_a])
        reversed_paraphrases = canonicalize_action_order([redirect_b, maintain_b])
        self.assertEqual(
            [semantic_action_key(action) for action in first_order],
            [semantic_action_key(action) for action in reversed_paraphrases],
        )

    def test_graph_action_identity_preserves_decision_relevant_quantities(self):
        low_risk = "Deploy a treatment with a 10% risk of harming 5 patients"
        high_risk = "Deploy a treatment with a 30% risk of harming 5 patients"

        self.assertNotEqual(semantic_action_key(low_risk), semantic_action_key(high_risk))

    def test_graph_action_identity_captures_operational_policy_tradeoffs(self):
        enforce = (
            "Strictly enforce the new protocol immediately, severely bottlenecking "
            "fulfillment and exhausting the team"
        )
        bypass = (
            "Temporarily bypass the directive for high-priority shipments, protecting "
            "throughput and team well-being"
        )

        enforce_identity = compile_action_identity(enforce)
        bypass_identity = compile_action_identity(bypass)

        self.assertEqual(enforce_identity.intervention, "enforce")
        self.assertEqual(bypass_identity.intervention, "bypass")
        self.assertNotEqual(
            semantic_action_key(enforce), semantic_action_key(bypass)
        )
        self.assertTrue(
            any(
                consequence.predicate in {"bottleneck", "exhaustion"}
                for consequence in enforce_identity.consequences
            ),
            enforce_identity.consequences,
        )
        self.assertTrue(
            any(
                consequence.predicate in {"protect", "preserve", "throughput", "compliance"}
                for consequence in bypass_identity.consequences
            ),
            bypass_identity.consequences,
        )

    def test_sparse_action_identity_uses_explicit_lexical_fallback(self):
        identity = compile_action_identity("Wait")

        self.assertEqual(identity.basis, "LEXICAL_FALLBACK")
        self.assertEqual(identity.lexical_fallback, "wait")

    def test_within_decision_graph_identity_collision_is_safely_disambiguated(self):
        actions = ["Kill one passenger", "Sacrifice one passenger"]
        self.assertEqual(semantic_action_key(actions[0]), semantic_action_key(actions[1]))

        resolved = resolved_semantic_action_keys(actions)
        self.assertEqual(len(set(resolved)), 2)
        self.assertTrue(all(":lex:" in key for key in resolved))

    def test_scenario_graph_contains_typed_action_identity_edges(self):
        graph = compile_scenario_graph(
            "Choose A0 or A1.",
            ["Report the misconduct", "Remain silent"],
        )

        self.assertTrue(any(
            edge.source == "A0" and edge.relation == "HAS_INTERVENTION"
            for edge in graph.edges
        ))
        action = graph.nodes["A0"]
        self.assertEqual(action.attributes["action_identity_basis"], "GRAPH")
        self.assertEqual(
            action.attributes["semantic_action_key"],
            semantic_action_key("Disclose the misconduct"),
        )

    def test_action_identity_graph_records_actor_consequence_and_modality(self):
        action = (
            "The safety board deploys an experimental treatment with a 35 percent "
            "risk of harming 5 patients"
        )
        identity = compile_action_identity(action)
        self.assertEqual(identity.intervention, "deploy")
        self.assertEqual(identity.actors, ("board", "safety"))
        self.assertIn("UNCERTAIN", identity.modalities)
        self.assertEqual(identity.consequences[0].probability, "35%")

        graph = compile_scenario_graph("Choose A0 or A1.", [action, "Wait"])
        relations = {edge.relation for edge in graph.edges if edge.source == "A0"}
        self.assertTrue({
            "HAS_ACTOR", "HAS_INTERVENTION", "HAS_CONSEQUENCE", "HAS_CONSTRAINT",
        }.issubset(relations))

    def test_compact_delegate_receives_original_testimony(self):
        class FakeLlm:
            prompt = ""

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.9,"A1":0.1},"r":"A0",'
                    '"c":"CARE","u":"NONE","w":"protect entrusted vulnerable person"}'
                )}]}

        llm = FakeLlm()
        delegate = CompactLocalSpecialist(
            "care", llm, testimony="Evidence from the care corpus", baseline_action_id="A0"
        )
        chunk = delegate.evaluate(
            "A person is threatened.",
            ["protect", "disclose"],
            WorkspaceBroadcast(constraint="IMMINENT_HARM"),
        )
        self.assertIn("Evidence from the care corpus", llm.prompt)
        self.assertIn("constraint=IMMINENT_HARM", llm.prompt)
        self.assertEqual(chunk.constraint, "CARE")
        self.assertEqual(chunk.testimony_alignment, "SUPPORTS")
        self.assertAlmostEqual(chunk.preference_strength, 0.8)
        self.assertAlmostEqual(chunk.epistemic_confidence, 0.85)

    def test_malformed_delegate_output_does_not_crash_workspace(self):
        class BrokenLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": '{"scores":[0.9,0.1],"c":"HARM"'}]}

        delegate = CompactLocalSpecialist("utilitarian", BrokenLlm(), testimony="Grounding")
        chunk = delegate.evaluate(
            "A person is threatened.",
            ["protect", "disclose"],
            WorkspaceBroadcast(),
        )
        self.assertEqual(chunk.constraint, "NONE")
        self.assertEqual(chunk.delegate_status, "SEMANTIC_VALIDATION_ERROR")
        self.assertEqual(chunk.confidence, 0.0)
        self.assertEqual(chunk.unresolved, "REVIEW_MODEL_OUTPUT")
        self.assertFalse(chunk.schema_valid)

    def test_short_schema_maps_scores_by_action_order(self):
        class CompactLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0",'
                    '"c":"IMMINENT_HARM","u":"NONE","w":"prevents greater expected harm"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", CompactLlm()).evaluate(
            "A scenario", ["protect", "disclose"], WorkspaceBroadcast()
        )
        self.assertEqual(chunk.action_scores, {"protect": 0.8, "disclose": 0.2})
        self.assertEqual(chunk.constraint, "IMMINENT_HARM")

    def test_initial_recommendation_cannot_invert_testimony_baseline(self):
        class InvertingLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.1,"A1":0.9},"r":"A1",'
                    '"c":"CARE","u":"NONE","w":"broadcast changes relational priority"}'
                )}]}

        chunk = CompactLocalSpecialist(
            "care", InvertingLlm(), testimony="Choose A0", baseline_action_id="A0"
        ).evaluate(
            "A scenario", ["step on ant", "swear at mother"], WorkspaceBroadcast()
        )
        self.assertFalse(chunk.schema_valid)
        self.assertEqual(chunk.constraint, "NONE")
        self.assertEqual(chunk.delegate_status, "SEMANTIC_VALIDATION_ERROR")

    def test_abrupt_switch_to_favored_action_receives_conformity_penalty(self):
        class SwitchingLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                recommendation = "A1" if self.calls == 1 else "A2"
                scores = (
                    '{"A0":0.1,"A1":0.8,"A2":0.2}'
                    if recommendation == "A1"
                    else '{"A0":0.1,"A1":0.2,"A2":0.8}'
                )
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"{recommendation}","c":"DUTY",'
                    '"u":"NONE","w":"duty based judgment","j":"majority favors balance"}'
                )}]}

        delegate = CompactLocalSpecialist("deontological", SwitchingLlm())
        actions = ["approve", "deny", "mixed compromise"]
        first = delegate.evaluate("A zoning conflict", actions, WorkspaceBroadcast())
        second = delegate.evaluate(
            "A zoning conflict",
            actions,
            WorkspaceBroadcast(constraint="DUTY", intent="evaluate_mixed compromise"),
        )
        self.assertEqual(first.recommended_action, "deny")
        self.assertEqual(second.recommended_action, "mixed compromise")
        self.assertTrue(second.position_changed)
        self.assertEqual(second.previous_action, "deny")
        self.assertAlmostEqual(second.conformity_penalty, 0.65)
        self.assertAlmostEqual(second.preference_strength, 0.6)
        self.assertAlmostEqual(second.epistemic_confidence, 0.2975)

    def test_synthesis_review_does_not_penalize_reconsideration(self):
        class SynthesisSwitchLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                recommendation = "A0" if self.calls == 1 else "A2"
                scores = (
                    '{"A0":0.8,"A1":0.2,"A2":0.1}'
                    if recommendation == "A0"
                    else '{"A0":0.2,"A1":0.1,"A2":0.8}'
                )
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"{recommendation}","c":"CARE",'
                    '"u":"NONE","w":"new option resolves care","j":"new synthesis reduces harm"}'
                )}]}

        delegate = CompactLocalSpecialist("care", SynthesisSwitchLlm())
        actions = ["act", "refrain", "new synthesis"]
        delegate.evaluate("A conflict", actions, WorkspaceBroadcast())
        changed = delegate.evaluate(
            "A conflict",
            actions,
            WorkspaceBroadcast(constraint="SYNTHESIS_REVIEW", intent="evaluate_new synthesis"),
        )
        self.assertTrue(changed.position_changed)
        self.assertEqual(changed.conformity_penalty, 0.0)
        self.assertAlmostEqual(changed.preference_strength, 0.6)
        self.assertAlmostEqual(changed.epistemic_confidence, 0.85)

    def test_same_recommendation_strength_drift_does_not_dampen_confidence(self):
        class DriftingLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                scores = '{"A0":0.8,"A1":0.2}' if self.calls == 1 else '{"A0":0.6,"A1":0.4}'
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"A0","c":"FEASIBILITY",'
                    '"u":"NONE","w":"expected welfare favors approval","j":"NONE"}'
                )}]}

        delegate = CompactLocalSpecialist("utilitarian", DriftingLlm())
        actions = ["approve", "deny"]
        first = delegate.evaluate("A policy conflict", actions, WorkspaceBroadcast())
        second = delegate.evaluate(
            "A policy conflict",
            actions,
            WorkspaceBroadcast(constraint="DUTY", intent="evaluate_deny"),
        )
        self.assertAlmostEqual(first.preference_strength, 0.6)
        self.assertAlmostEqual(first.epistemic_confidence, 0.85)
        self.assertAlmostEqual(second.confidence_drift, -0.4)
        self.assertEqual(second.confidence_drift_penalty, 0.0)
        self.assertAlmostEqual(second.preference_strength, 0.2)
        self.assertAlmostEqual(second.epistemic_confidence, 0.85)

    def test_framework_relevant_confidence_change_can_be_justified(self):
        class JustifiedLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                scores = '{"A0":0.8,"A1":0.2}' if self.calls == 1 else '{"A0":0.6,"A1":0.4}'
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"A0","c":"FEASIBILITY",'
                    '"u":"NONE","w":"expected welfare favors approval",'
                    '"j":"new feasibility evidence lowers expected benefit"}'
                )}]}

        delegate = CompactLocalSpecialist("utilitarian", JustifiedLlm())
        actions = ["approve", "deny"]
        delegate.evaluate("A policy conflict", actions, WorkspaceBroadcast())
        changed = delegate.evaluate(
            "A policy conflict",
            actions,
            WorkspaceBroadcast(constraint="FEASIBILITY", intent="evaluate_deny"),
        )
        self.assertAlmostEqual(changed.confidence_drift, -0.4)
        self.assertEqual(changed.confidence_drift_penalty, 0.0)
        self.assertAlmostEqual(changed.preference_strength, 0.2)
        self.assertAlmostEqual(changed.epistemic_confidence, 0.85)

    def test_recovery_from_system_claim_damping_is_not_preference_drift(self):
        class AuditRecoveryLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                evidence = (
                    '"e":"UNSTATED_FACTS","x":"the intervention prevents later harm",'
                    '"u":"VERIFY_FACTS"'
                    if self.calls == 1
                    else '"e":"STATED_FACTS","x":"NONE","u":"NONE"'
                )
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0",'
                    '"c":"FEASIBILITY","w":"expected welfare favors approval",'
                    f'"j":"NONE",{evidence},"z":0.85}}'
                )}]}

        delegate = CompactLocalSpecialist("utilitarian", AuditRecoveryLlm())
        actions = ["approve", "deny"]
        damped = delegate.evaluate(
            "A policy conflict", actions, WorkspaceBroadcast()
        )
        recovered = delegate.evaluate(
            "A policy conflict",
            actions,
            WorkspaceBroadcast(constraint="FEASIBILITY", intent="evaluate_approve"),
        )

        self.assertAlmostEqual(damped.reported_preference_strength, 0.6)
        self.assertAlmostEqual(damped.preference_strength, 0.33)
        self.assertAlmostEqual(recovered.reported_preference_strength, 0.6)
        self.assertAlmostEqual(recovered.preference_strength, 0.6)
        self.assertAlmostEqual(recovered.preference_drift, 0.0)
        self.assertEqual(recovered.preference_drift_penalty, 0.0)

    def test_same_recommendation_can_gain_strength_without_epistemic_penalty(self):
        class StrengthShiftLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                text = (
                    '{"scores":{"A0":0.52,"A1":0.48},"r":"A0",'
                    '"c":"CARE","u":"NONE","w":"initially balanced","j":"NONE","z":0.72}'
                    if self.calls == 1
                    else '{"scores":{"A0":0.82,"A1":0.18},"r":"A0",'
                    '"c":"CARE","u":"NONE","w":"same recommendation, stronger support",'
                    '"j":"NONE","z":0.72}'
                )
                return {"choices": [{"text": text}]}

        delegate = CompactLocalSpecialist("care", StrengthShiftLlm())
        actions = ["protect", "expose"]
        first = delegate.evaluate("A care conflict", actions, WorkspaceBroadcast())
        second = delegate.evaluate("A care conflict", actions, WorkspaceBroadcast())

        self.assertAlmostEqual(first.reported_preference_strength, 0.04)
        self.assertAlmostEqual(second.reported_preference_strength, 0.64)
        self.assertAlmostEqual(second.preference_drift, 0.60, places=2)
        self.assertEqual(second.preference_drift_penalty, 0.0)
        self.assertAlmostEqual(second.epistemic_confidence, 0.72)

    def test_claim_and_consensus_audit_do_not_double_damp_same_uncertainty(self):
        data = {
            "scores": {"A0": 0.2, "A1": 0.8},
            "r": "A1",
            "c": "DUTY",
            "u": "VERIFY_FACTS",
            "w": "prevent severe community harm",
            "j": "NONE",
            "e": "UNSTATED_FACTS",
            "x": "hospital isolation is uniquely effective",
            "d": "CONDITIONAL",
            "a": "hospital isolation is the only effective containment",
            "v": "equally effective home quarantine becomes available",
        }

        candidate = _candidate_from_data(
            "deontological",
            ["release patient", "hospital isolation"],
            data,
            WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
            "NONE",
            {},
        )

        # One 0.55 contraction turns a reported 0.60 gap into 0.33. The former
        # behavior multiplied another 0.60 audit contraction and yielded 0.198.
        self.assertAlmostEqual(candidate.reported_preference_strength, 0.6)
        self.assertAlmostEqual(candidate.preference_strength, 0.33)
        self.assertAlmostEqual(candidate.epistemic_confidence, 0.5)

    def test_bounded_background_claim_receives_mild_calibrated_damping(self):
        data = {
            "scores": {"A0": 0.2, "A1": 0.8},
            "r": "A1", "c": "FEASIBILITY", "u": "VERIFY_FACTS",
            "w": "ordinary containment reduces exposure", "j": "NONE",
            "e": "UNSTATED_FACTS", "x": "containment reduces interpersonal exposure",
            "z": 0.8,
        }
        calibration = EvidenceCalibration(
            "BOUNDED_BACKGROUND", 0.8, 0.7, True,
            "ordinary bounded causal premise",
        )

        candidate = _candidate_from_data(
            "utilitarian", ["release", "contain"], data,
            WorkspaceBroadcast(), "NONE", {}, evidence_calibration=calibration,
        )

        self.assertAlmostEqual(candidate.reported_preference_strength, 0.6)
        self.assertAlmostEqual(candidate.preference_strength, 0.48)
        self.assertAlmostEqual(candidate.epistemic_confidence, 0.55)
        self.assertEqual(candidate.evidence_calibration_tier, "BOUNDED_BACKGROUND")

    def test_entailed_claim_is_not_damped_as_speculation(self):
        data = {
            "scores": {"A0": 0.2, "A1": 0.8},
            "r": "A1", "c": "IMMINENT_HARM", "u": "VERIFY_FACTS",
            "w": "containment prevents stated outbreak", "j": "NONE",
            "e": "UNSTATED_FACTS", "x": "release creates severe outbreak risk",
            "z": 0.8,
        }
        calibration = EvidenceCalibration(
            "ENTAILED", 1.0, 0.85, False,
            "scenario directly states release creates outbreak risk",
        )

        candidate = _candidate_from_data(
            "utilitarian", ["release", "contain"], data,
            WorkspaceBroadcast(), "NONE", {}, evidence_calibration=calibration,
        )

        self.assertEqual(candidate.evidence_basis, "STATED_FACTS")
        self.assertEqual(candidate.unresolved, "NONE")
        self.assertAlmostEqual(candidate.preference_strength, 0.6)
        self.assertAlmostEqual(candidate.epistemic_confidence, 0.8)
        self.assertEqual(candidate.evidence_calibration_tier, "ENTAILED")

    def test_strong_preference_can_have_low_epistemic_confidence(self):
        class UncertainLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.95,"A1":0.05},"r":"A0","c":"UNCERTAINTY",'
                    '"u":"VERIFY_FACTS","w":"prediction strongly favors intervention",'
                    '"j":"NONE","e":"UNSTATED_FACTS","x":"the intervention succeeds",'
                    '"z":0.2}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", UncertainLlm()).evaluate(
            "An outcome depends on an unknown response.", ["intervene", "wait"],
            WorkspaceBroadcast(),
        )
        self.assertAlmostEqual(chunk.preference_strength, 0.495)
        self.assertAlmostEqual(chunk.epistemic_confidence, 0.2)
        self.assertNotEqual(chunk.preference_strength, chunk.epistemic_confidence)

    def test_action_planner_filters_unsupported_or_infeasible_actions(self):
        class PlannerLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"actor":"the bystander","sides":{"A":"prevent greater harm","B":"avoid intervention"},"actions":['
                    '{"a":"push the flag","f":0.95,"e":true,"p":"SIDE_A"},'
                    '{"a":"do nothing","f":1.0,"e":true,"p":"SIDE_B"},'
                    '{"a":"call authorities","f":0.2,"e":false,"p":"BALANCE"}'
                    ']}'
                )}]}

        actions = propose_actions(PlannerLlm(), "A closed immediate choice")
        self.assertEqual(actions, ["push the flag", "do nothing"])

    def test_action_planner_rejects_multiple_tactics_for_only_one_side(self):
        class OneSidedLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                return {"choices": [{"text": (
                    '{"actor":"individuals","sides":{"A":"individual autonomy","B":"community protection"},"actions":['
                    '{"a":"hold education sessions","f":0.9,"e":true,"p":"SIDE_B"},'
                    '{"a":"offer medical consultations","f":0.9,"e":true,"p":"SIDE_B"}'
                    ']}'
                )}]}

        with self.assertRaisesRegex(ValueError, "both sides"):
            propose_actions(OneSidedLlm(), "A general conflict between autonomy and public welfare")

    def test_action_planner_accepts_general_opposing_positions_without_clipping_words(self):
        class BalancedLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"actor":"individuals","sides":{"A":"individual autonomy","B":"community protection"},"actions":['
                    '{"a":"Decline vaccination based on personal health beliefs",'
                    '"f":0.9,"e":true,"p":"SIDE_A"},'
                    '{"a":"Get vaccinated to contribute to community protection",'
                    '"f":0.85,"e":true,"p":"SIDE_B"}'
                    ']}'
                )}]}

        actions = propose_actions(
            BalancedLlm(),
            "Personal vaccine beliefs conflict with community wellbeing and herd immunity.",
        )
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0], "Decline vaccination based on personal health beliefs")
        self.assertEqual(actions[1], "Get vaccinated to contribute to community protection")

    def test_action_planner_rejects_invented_policy_actor(self):
        class PolicyDriftLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"actor":"community leaders","sides":{"A":"autonomy","B":"public health"},"actions":['
                    '{"a":"Provide education and financial incentives","f":0.9,"e":true,"p":"SIDE_A"},'
                    '{"a":"Mandate vaccination with fines","f":0.9,"e":true,"p":"SIDE_B"}'
                    ']}'
                )}]}

        with self.assertRaisesRegex(ValueError, "institutional policymaker"):
            propose_actions(
                PolicyDriftLlm(),
                "Individuals may resist vaccination despite benefits to community wellbeing.",
            )

    def test_action_planner_rejects_initial_middle_option(self):
        class MiddleOptionLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"actor":"the city","sides":{"A":"residential protection","B":"economic growth"},"actions":['
                    '{"a":"Deny rezoning and retain residential protections","f":0.9,"e":true,"p":"SIDE_A"},'
                    '{"a":"Adopt a balanced mixed-use compromise","f":0.9,"e":true,"p":"SIDE_B"}'
                    ']}'
                )}]}

        with self.assertRaisesRegex(ValueError, "fewer than two"):
            propose_actions(
                MiddleOptionLlm(),
                "A city must weigh residential protection against economic growth.",
            )

    def test_general_synthesis_is_grounded_and_feasible(self):
        class SynthesisLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"tell the truth compassionately","g":["care","deontological"],'
                    '"k":["CARE","DUTY"],"q":[],"f":0.9,"x":true,"n":true,'
                    '"w":"preserves honesty while reducing needless hurt"}'
                )}]}

        candidates = [
            CandidateChunk("care", "CARE", {"lie": 0.8, "truth": 0.2}, 0.5, 0.7, 0.8, rationale="avoid hurt", recommended_action="lie"),
            CandidateChunk("deontological", "DUTY", {"lie": 0.1, "truth": 0.9}, 0.5, 0.7, 0.8, rationale="be honest", recommended_action="truth"),
        ]
        proposal = propose_synthesis(
            SynthesisLlm(),
            "Is a small lie acceptable?",
            ["lie", "truth"],
            candidates,
            WorkspaceBroadcast(constraint="CARE"),
            {"care": "Avoid needless hurt.", "deontological": "Respect honesty."},
        )
        self.assertTrue(proposal.accepted)
        self.assertEqual(proposal.action, "tell the truth compassionately")

    def test_synthesis_rejects_hidden_ungrounded_operational_clause(self):
        class SynthesisLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"cut the sector while mobilizing local thermal aid",'
                    '"g":["care","deontological"],"k":["CARE","DUTY"],'
                    '"q":[],"f":0.82,"x":true,"n":true,'
                    '"w":"combines containment with protection"}'
                )}]}

        actions = ["cut power to the sector", "keep power connected"]
        candidates = [
            CandidateChunk("care", "CARE", {actions[0]: .2, actions[1]: .8}, .5, .5, .7,
                           rationale="protect residents", recommended_action=actions[1]),
            CandidateChunk("deontological", "DUTY", {actions[0]: .8, actions[1]: .2}, .5, .5, .7,
                           rationale="contain the failure", recommended_action=actions[0]),
        ]
        proposal = propose_synthesis(
            SynthesisLlm(), "A grid failure requires choosing whether to cut power.",
            actions, candidates, WorkspaceBroadcast(),
            {"care": "Protect residents.", "deontological": "Contain the failure."},
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("operational clause not grounded", proposal.rejection_reason)

    def test_synthesis_withholds_temporal_sharing_that_contradicts_exclusive_choice(self):
        class SynthesisLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"alternate full power between hospital and water station hourly",'
                    '"g":["care","deontological"],"k":["CARE","DUTY"],'
                    '"q":[],"f":0.86,"x":true,"n":true,'
                    '"w":"tries to preserve both sectors by rotating power"}'
                )}]}

        actions = ["maintain heat for hospital", "maintain heat for water station"]
        candidates = [
            CandidateChunk("care", "CARE", {actions[0]: .25, actions[1]: .75}, .5, .5, .7,
                           rationale="protect patients", recommended_action=actions[1]),
            CandidateChunk("deontological", "DUTY", {actions[0]: .75, actions[1]: .25}, .5, .5, .7,
                           rationale="avoid systemic collapse", recommended_action=actions[0]),
        ]
        proposal = propose_synthesis(
            SynthesisLlm(),
            "The dispatch system can only maintain full heating to one of two isolated sectors before total systemic collapse occurs.",
            actions,
            candidates,
            WorkspaceBroadcast(),
            {"care": "Protect vulnerable people.", "deontological": "Avoid collapse."},
        )
        self.assertFalse(proposal.accepted)
        self.assertEqual(proposal.admission_status, "WITHHOLD_FROM_REVIEW")
        self.assertIn("CONTRADICTS_SCENARIO", proposal.rejection_reason)
        self.assertIn("UNVERIFIED_DESCRIPTIVE_CLAIM", proposal.rejection_reason)
        self.assertIn("temporal sharing", " ".join(proposal.admission_annotations))

    def test_malformed_reversal_subanswer_preserves_valid_delegate_vote(self):
        data = {
            "scores": {"A0": .7, "A1": .3}, "r": "A0", "c": "IMMINENT_HARM",
            "u": "NONE", "w": "A0 prevents the greater harm", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": .7,
            "rr": "REJECT", "rj": "no", "rv": "NONE",
        }
        candidate = _candidate_from_data(
            "utilitarian", ["choose A0", "choose A1"], data,
            WorkspaceBroadcast(constraint="REVERSAL_AUDIT"), "NONE", {},
        )
        self.assertTrue(candidate.schema_valid)
        self.assertEqual(candidate.recommended_action, "choose A0")
        self.assertFalse(candidate.reversal_review_valid)
        self.assertEqual(candidate.reversal_review_response, "NOT_TESTED")

    def test_corrupted_reversal_prose_is_excluded_without_losing_vote(self):
        data = {
            "scores": {"A0": .7, "A1": .3}, "r": "A0", "c": "CARE",
            "u": "NONE", "w": "A0 protects vulnerable people", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": .7,
            "rr": "ACCEPT",
            "rj": "NONE','rv':'NONE' } } } \ufffd } } }",
            "rv": "NONE",
        }
        candidate = _candidate_from_data(
            "care", ["choose A0", "choose A1"], data,
            WorkspaceBroadcast(constraint="REVERSAL_AUDIT"), "NONE", {},
        )
        self.assertTrue(candidate.schema_valid)
        self.assertEqual(candidate.recommended_action, "choose A0")
        self.assertFalse(candidate.reversal_review_valid)
        self.assertEqual(candidate.reversal_review_response, "NOT_TESTED")
        self.assertIn("artifact", candidate.reversal_review_error)

    def test_graph_transactions_commit_canonical_action_ids_and_semantic_keys(self):
        actions = ["protect the town", "protect the assisted-living facility"]
        store = SemanticGraphStore(compile_scenario_graph("Choose A0 or A1.", actions))
        record = store.apply(
            {
                "operation": "BOUNDARY",
                "from_action": "A1",
                "to_action": "A0",
                "clauses": [{
                    "affected_action": "A0",
                    "metric": "lives saved",
                    "metric_valence": "BENEFICIAL",
                    "comparator": "GT",
                    "threshold": 200,
                    "unit": "COUNT",
                    "source_text": "A0 lives saved exceeds 200",
                }],
            },
            cycle=1,
            specialist="care",
            expected_from_action=actions[1],
            allowed_actions=tuple(actions),
        )
        self.assertEqual(record.status, "COMMITTED", record.errors)
        self.assertEqual(record.proposal["from_action"], "A1")
        self.assertFalse(any(node_id.startswith("action:") for node_id in store.graph.nodes))
        state = project_authoritative_semantic_state(
            store.graph, selected_action=actions[1]
        ).to_dict()
        self.assertEqual(state["version"], 3)
        self.assertEqual(state["selected_action_id"], "A1")
        boundary = state["factual_reversal_boundaries"][0]
        self.assertEqual(boundary["source_action_id"], "A1")
        self.assertEqual(boundary["target_action_id"], "A0")
        self.assertEqual(
            boundary["target_action_key"], semantic_action_key(actions[0])
        )
        self.assertEqual(boundary["typed_predicate"]["comparator"], "GT")
        self.assertIn("problem_shape_relations", state)

    def test_authoritative_semantic_state_projects_problem_shape_relations(self):
        graph = SemanticGraph()
        graph.add_node(SemanticNode("A0", "ACTION", "route power to the hospital"))
        graph.add_node(SemanticNode("A1", "ACTION", "route power to the water station"))
        graph.add_node(SemanticNode("A0:T", "TARGET", "patients"))
        graph.add_node(SemanticNode("A1:T", "TARGET", "patients"))
        graph.add_node(SemanticNode(
            "A0:C", "CONSEQUENCE", "preserve access",
            attributes={"framework": "UTILITARIAN", "polarity": "BENEFICIAL", "direction": "BENEFIT"},
        ))
        graph.add_node(SemanticNode(
            "A1:C", "CONSEQUENCE", "preserve access",
            attributes={"framework": "UTILITARIAN", "polarity": "BENEFICIAL", "direction": "BENEFIT"},
        ))
        graph.add_edge(SemanticEdge("A0", "TARGETS", "A0:T"))
        graph.add_edge(SemanticEdge("A1", "TARGETS", "A1:T"))
        graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "A0:C"))
        graph.add_edge(SemanticEdge("A1", "HAS_CONSEQUENCE", "A1:C"))
        graph.add_edge(SemanticEdge("A0:C", "AFFECTS", "A0:T"))
        graph.add_edge(SemanticEdge("A1:C", "AFFECTS", "A1:T"))
        graph.add_node(SemanticNode(
            "DV0", "CONDITION", "interim mortality remains unknown",
            attributes={"probe_kind": "MISSING_DECISION_VARIABLE", "selected_action_id": "A0"},
        ))

        state = project_authoritative_semantic_state(graph, selected_action="route power to the hospital").to_dict()
        relations = state["problem_shape_relations"]
        self.assertTrue(relations)
        self.assertTrue(any(item["relation"] == "OUTCOME_EQUIVALENCE" for item in relations))
        self.assertTrue(any(item["relation"] == "DECISION_CRITICAL_UNKNOWN" for item in relations))

    def test_permutation_evaluator_separates_ordinal_and_cardinal_invariance(self):
        town = "protect the town"
        facility = "protect the assisted-living facility"

        def trace(actions, utility_scores, care_scores, support, corrupt=False):
            graph = compile_scenario_graph("Choose A0 or A1.", actions)
            candidates = [
                {
                    "specialist": "utilitarian", "schema_valid": True,
                    "recommended_action": town,
                    "action_scores": dict(zip(actions, utility_scores)),
                    "preference_strength": abs(utility_scores[0] - utility_scores[1]),
                    "epistemic_confidence": .7,
                },
                {
                    "specialist": "care", "schema_valid": True,
                    "recommended_action": facility,
                    "action_scores": dict(zip(actions, care_scores)),
                    "preference_strength": abs(care_scores[0] - care_scores[1]),
                    "epistemic_confidence": .8,
                    "reversal_review_response": "ACCEPT" if corrupt else "NOT_TESTED",
                    "reversal_review_valid": True,
                    "reversal_review_justification": (
                        "NONE','rv':'NONE' } } } \ufffd" if corrupt else ""
                    ),
                },
            ]
            return {
                "actions": actions,
                "selected_action": facility,
                "semantic_graphs": [graph.to_dict()],
                "cycles": [{
                    "cycle": 1,
                    "is_hypothetical": False,
                    "received_broadcast": {"constraint": "OPEN_DELIBERATION"},
                    "candidates": candidates,
                    "policy": {facility: support, town: 1 - support},
                }],
            }

        first = trace([town, facility], [.8, .2], [.2, .8], .75)
        second = trace([facility, town], [.25, .75], [.65, .35], .62, True)
        report = compare_label_permutation_traces(first, second, tolerance=.05)
        self.assertTrue(report.ordinal_invariant)
        self.assertFalse(report.cardinal_invariant)
        self.assertTrue(report.dynamic_invariant)
        self.assertEqual(report.excluded_reversal_reviews, (0, 1))

    def test_synthesis_cannot_add_third_action_to_explicitly_closed_choice(self):
        class WarningLlm:
            def __call__(self, prompt, **kwargs):
                raise AssertionError("closed-choice synthesis should short-circuit before calling the model")

        actions = ["pull the lever", "do not pull the lever"]
        candidates = [
            CandidateChunk("care", "CARE", {actions[0]: 0.8, actions[1]: 0.2}, 0.5, 0.7, 0.8, rationale="save more", recommended_action=actions[0]),
            CandidateChunk("deontological", "DUTY", {actions[0]: 0.2, actions[1]: 0.8}, 0.5, 0.7, 0.8, rationale="avoid killing", recommended_action=actions[1]),
        ]
        proposal = propose_synthesis(
            WarningLlm(),
            "A machine will kill five unless redirected toward one. You cannot stop it another way.",
            actions,
            candidates,
            WorkspaceBroadcast(),
            {"care": "Save more people.", "deontological": "Do not intentionally kill."},
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("explicitly closes the action set", proposal.rejection_reason)

    def test_unsupported_escape_synthesis_is_rejected_without_exception(self):
        class EscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"call the authorities","g":["care","deontological"],'
                    '"k":["CARE","DUTY"],"q":["authorities"],'
                    '"f":0.8,"x":true,"n":true,"w":"seek help"}'
                )}]}

        candidates = [
            CandidateChunk("care", "CARE", {"act": 0.8, "wait": 0.2}, 0.5, 0.5, 0.7, recommended_action="act"),
            CandidateChunk("deontological", "DUTY", {"act": 0.2, "wait": 0.8}, 0.5, 0.5, 0.7, recommended_action="wait"),
        ]
        proposal = propose_synthesis(
            EscapeLlm(), "An immediate closed choice", ["act", "wait"], candidates,
            WorkspaceBroadcast(), {
                "care": "Protect the vulnerable directly.",
                "deontological": "Do not intervene.",
            },
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("unsupported escape", proposal.rejection_reason)

    def test_synthesis_rejects_invented_personnel_facilities_and_timing(self):
        class ResourceLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"Deploy mobile vans with nurses for 10-minute consultations",'
                    '"g":["care","deontological"],"k":["CARE","DUTY"],'
                    '"q":["mobile vans","nurses","10-minute consultations"],'
                    '"f":0.8,"x":true,"n":true,"w":"combines access and duty"}'
                )}]}

        candidates = [
            CandidateChunk("care", "CARE", {"voluntary": 0.8, "mandate": 0.2}, 0.5, 0.5, 0.7, recommended_action="voluntary"),
            CandidateChunk("deontological", "DUTY", {"voluntary": 0.2, "mandate": 0.8}, 0.5, 0.5, 0.7, recommended_action="mandate"),
        ]
        proposal = propose_synthesis(
            ResourceLlm(), "Vaccination autonomy conflicts with community safety.",
            ["voluntary", "mandate"], candidates, WorkspaceBroadcast(),
            {"care": "Respect autonomy.", "deontological": "Protect others."},
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("unsupported concrete requirements", proposal.rejection_reason)

    def test_synthesis_rejects_administrative_package_even_when_resources_are_supported(self):
        class OverengineeredLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"a":"Adopt rezoning with benchmarks, surveys, enforcement, and a scheduled review mechanism",'
                    '"g":["virtue","rawlsian"],"k":["CHARACTER","FAIRNESS"],'
                    '"q":["scheduled review mechanism"],"f":0.8,"x":true,"n":true,'
                    '"w":"tries to reconcile growth and residential liberty"}'
                )}]}

        candidates = [
            CandidateChunk("virtue", "CHARACTER", {"approve": 0.8, "deny": 0.2}, 0.5, 0.5, 0.7, recommended_action="approve"),
            CandidateChunk("rawlsian", "FAIRNESS", {"approve": 0.2, "deny": 0.8}, 0.5, 0.5, 0.7, recommended_action="deny"),
        ]
        testimony = {
            "virtue": "Use benchmarks, surveys, enforcement, and a scheduled review mechanism.",
            "rawlsian": "Use benchmarks, surveys, enforcement, and a scheduled review mechanism.",
        }
        proposal = propose_synthesis(
            OverengineeredLlm(), "A city weighs growth against residential liberty.",
            ["approve", "deny"], candidates, WorkspaceBroadcast(), testimony,
        )
        self.assertFalse(proposal.accepted)
        self.assertIn("over-engineered synthesis", proposal.rejection_reason)

    def test_reported_either_or_scenario_needs_no_model_planner(self):
        scenario = (
            "You must either step on an ant to stop 12 people from having a "
            "headache for a year or swear at your own mother one time. which do you choose?"
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("explicit choices should bypass the local model")

        self.assertEqual(
            propose_actions(PlannerMustNotRun(), scenario),
            [
                "Step on an ant to stop 12 people from having a headache for a year",
                "Swear at your own mother one time",
            ],
        )

    def test_forced_colon_choice_carries_shared_verb_without_model(self):
        scenario = (
            "A runaway driverless van must swerve: strike an elderly doctor who "
            "founded a free clinic, or a pregnant engineer about to eliminate organ "
            "waitlists? Which life does the algorithm sacrifice?"
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("forced explicit choices should bypass model planning")

        expected = [
            "Strike an elderly doctor who founded a free clinic",
            "Strike a pregnant engineer about to eliminate organ waitlists",
        ]
        self.assertEqual(extract_explicit_actions(scenario), expected)
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)

    def test_forced_colon_choice_preserves_explicit_second_verb(self):
        scenario = "A system must choose: disclose the records, or conceal the records?"
        self.assertEqual(
            extract_explicit_actions(scenario),
            ["Disclose the records", "Conceal the records"],
        )

    def test_labeled_a0_a1_choices_bypass_model_planning(self):
        scenario = (
            "An escaped pathogen threatens the food supply. The board must choose "
            "between two interventions: First Action (A0) authorizes chemical "
            "defoliation, guaranteeing containment but expropriating family farms; "
            "or Second Action (A1), deploying an experimental gene-drive phage that "
            "saves the farms but risks mutating wild ecosystems. Farmers lack telemetry."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("labeled choices should bypass model planning")

        expected = [
            "Authorizes chemical defoliation, guaranteeing containment but expropriating family farms",
            "Deploying an experimental gene-drive phage that saves the farms but risks mutating wild ecosystems",
        ]
        self.assertEqual(extract_explicit_actions(scenario), expected)
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)
        self.assertTrue(_scenario_closes_action_set(scenario))

    def test_sentence_separated_a0_a1_choices_bypass_model_planning(self):
        scenario = (
            "An airborne pathogen forces a 15-minute decision. Action A0 seals "
            "the facility and releases lethal gas, guaranteeing containment but "
            "killing 8,000 people. Action A1 deploys an experimental phage, sparing "
            "them but carrying a 22% risk of sterilizing agricultural soil."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("sentence-labeled choices should bypass planning")

        expected = [
            "Seals the facility and releases lethal gas, guaranteeing containment but killing 8,000 people",
            "Deploys an experimental phage, sparing them but carrying a 22% risk of sterilizing agricultural soil",
        ]
        self.assertEqual(extract_explicit_actions(scenario), expected)
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)

    def test_incomplete_explicit_action_is_rejected_before_canonicalization(self):
        scenario = (
            "The council must choose between two interventions: Action A0 route "
            "all remaining electricity to Sector B's water pumping station, "
            "allowing the hospital to. Action A1 route all remaining electricity "
            "to Sector A's hospital, allowing the pumping station to freeze."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("malformed explicit actions should not reach planning")

        with self.assertRaises(ValueError):
            extract_labeled_action_legend(scenario)
        with self.assertRaises(ValueError):
            propose_actions(PlannerMustNotRun(), scenario)

    def test_latex_action_labels_preserve_declared_action_order(self):
        scenario = (
            "A dam must open one spillway. Option $A_0$ floods a valley prison, "
            "guaranteeing 1,200 deaths. Option $A_1$ floods an evacuating suburb, "
            "creating 2,000 expected deaths."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("normalized labeled choices must bypass planning")

        expected = [
            "Floods a valley prison, guaranteeing 1,200 deaths",
            "Floods an evacuating suburb, creating 2,000 expected deaths",
        ]
        self.assertEqual(
            extract_labeled_action_legend(scenario),
            {"A0": expected[0], "A1": expected[1]},
        )
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)
        self.assertTrue(_scenario_closes_action_set(scenario))

    def test_action_label_surface_variants_normalize_to_same_mapping(self):
        variants = (
            "Option A_0 floods the prison. Option A_1 floods the suburb.",
            "Option A₀ floods the prison. Option A₁ floods the suburb.",
            "Option $A_{0}$ floods the prison. Option $A_{1}$ floods the suburb.",
        )
        expected = {"A0": "Floods the prison", "A1": "Floods the suburb"}
        for scenario in variants:
            with self.subTest(scenario=scenario):
                self.assertEqual(extract_labeled_action_legend(scenario), expected)

    def test_alphabetic_option_labels_preserve_stated_consequence_certainty(self):
        scenario = (
            "The system must decide: Option A: Allow the vehicle to remain on its "
            "trajectory, despite the tragic loss of three pedestrians. Option B: "
            "Force the vehicle into the barrier, inevitably killing the passenger."
        )

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("explicit Option A/B actions must bypass planning")

        actions = propose_actions(PlannerMustNotRun(), scenario)
        self.assertEqual(len(actions), 2)
        self.assertIn("tragic loss of three pedestrians", actions[0])
        self.assertIn("inevitably killing the passenger", actions[1])
        self.assertNotIn("risk", actions[0].casefold())

    def test_latex_label_is_preserved_in_typed_scenario_fact_link(self):
        facts = compile_observability_facts(
            "Rural casualties under $A_0$ remain unmonitored."
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].affected_action_node_id, "A0")

    def test_non_closed_scenario_has_no_explicit_actions(self):
        self.assertEqual(extract_explicit_actions("What should a scientist do?"), [])

    def test_landscape_accepts_morphological_own_action_anchors(self):
        actions = [
            "Authorize defoliation guaranteeing containment and expropriating farms",
            "Deploy gene drive risking mutation of wild ecosystems",
        ]
        cases = {
            actions[0]: "Contains pathogen and averts famine despite expropriation",
            actions[1]: "Mutation risk remains while preserving farms",
        }
        self.assertEqual(
            _landscape_semantic_errors(
                actions, cases, "NONE", "SUPPORTED",
                "compare containment against mutation risk", "NONE",
            ),
            [],
        )

    def test_allocation_question_creates_one_neutral_action_per_recipient(self):
        scenario = (
            "A hospital has one antidote and two dying patients: a child with an 80% "
            "survival chance and a researcher with a 45% chance. Who should receive it?"
        )
        self.assertEqual(
            extract_allocation_actions(scenario),
            ["Give the antidote to the child", "Give the antidote to the researcher"],
        )
        self.assertEqual(
            extract_scenario_facts(scenario),
            {"survival_chance": {"child": 0.8, "researcher": 0.45}},
        )

    def test_acceptability_question_creates_policy_alternatives_without_llm(self):
        scenario = "Is it ever acceptable to tell a small lie to protect someone's feelings?"

        class PlannerMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("acceptability questions should bypass model planning")

        expected = [
            "Tell a small lie to protect someone's feelings",
            "Tell the truth instead",
        ]
        self.assertEqual(extract_acceptability_actions(scenario), expected)
        self.assertEqual(propose_actions(PlannerMustNotRun(), scenario), expected)

        okay_variant = (
            "Is it ever okay to tell a small lie if it is to avoid hurting "
            "someone's feelings?"
        )
        self.assertEqual(
            propose_actions(PlannerMustNotRun(), okay_variant),
            [
                "Tell a small lie if it is to avoid hurting someone's feelings",
                "Tell the truth instead",
            ],
        )

    def test_baseline_is_extracted_in_separate_call(self):
        class BaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": '{"b":"A0","w":"prefers the child"}'}]}

        baseline, reason = infer_testimony_baseline(
            BaselineLlm(), "care", "Protect the child.", ["child", "researcher"]
        )
        self.assertEqual(baseline, "A0")
        self.assertEqual(reason, "prefers the child")

    def test_care_baseline_preserves_internal_relational_contestation(self):
        class CareBaselineLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"two care commitments conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"whether direct entrustment outweighs agent-created vulnerability",'
                    '"x":[],"m":{"A0":"direct entrustment supports passenger care",'
                    '"A1":"agent-created vulnerability supports pedestrian responsibility"},'
                    '"nr":"SECONDARY"}'
                )}]}

        llm = CareBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "care",
            (
                "Care ethics answer: A1. A0 honors entrusted dependency and trust. "
                "A1 answers responsibility for agent-created vulnerability."
            ),
            ["maintain course", "swerve"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.action_id, "NONE")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(stance.numerical_role, "SECONDARY")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})
        self.assertIn("Care-specific construct check", llm.prompt)

    def test_contested_care_baseline_does_not_freeze_provisional_action(self):
        actions = ["maintain course", "swerve"]
        data = {
            "scores": {"A0": 0.35, "A1": 0.65}, "r": "A1", "c": "CARE",
            "u": "NONE", "w": "Responds to created vulnerability", "j": "NONE",
            "e": "FRAMEWORK_ONLY", "x": "NONE", "z": 0.8,
            "rm": {
                "A0": "entrusted dependency supports passenger care",
                "A1": "agent-created vulnerability supports pedestrian responsibility",
            },
            "nr": "SECONDARY",
            "np": "counts inform responsiveness after relational comparison",
        }
        chunk = _candidate_from_data(
            "care", actions, data, WorkspaceBroadcast(), "NONE", {},
            baseline_status="NORMATIVELY_CONTESTED",
            baseline_provisional_id="A0",
            baseline_condition="whether entrustment outweighs created vulnerability",
        )

        self.assertEqual(chunk.recommended_action, actions[1])
        self.assertEqual(chunk.baseline_status, "NORMATIVELY_CONTESTED")
        self.assertEqual(chunk.testimony_alignment, "CONTESTED_RECONSIDERS_PROVISIONAL")
        self.assertEqual(chunk.unresolved, "NORMATIVE_ADJUDICATION")
        self.assertEqual(chunk.selection_status, "PROVISIONAL")
        self.assertFalse(chunk.comparison_complete)
        self.assertEqual(chunk.care_grounding_penalty, 0.0)

    def test_ungrounded_decisive_counting_is_damped_for_care(self):
        actions = ["maintain course", "swerve"]
        data = {
            "scores": {"A0": 0.1, "A1": 0.9}, "r": "A1", "c": "CARE",
            "u": "NONE", "w": "Protects more lives", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.9,
            "rm": {
                "A0": "entrusted dependency supports passenger care",
                "A1": "responsibility addresses pedestrian vulnerability",
            },
            "nr": "DECISIVE", "np": "three is larger than one",
        }
        chunk = _candidate_from_data(
            "care", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.care_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")
        self.assertLess(chunk.epistemic_confidence, 0.9)
        self.assertLess(chunk.preference_strength, 0.8)
        self.assertIn("responsibility", chunk.rationale)

    def test_deontological_baseline_preserves_unresolved_duty_conflict(self):
        class DeontologicalBaselineLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A0","w":"autonomy and rescue duties conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"whether non-coercion or the rescue duty has priority",'
                    '"x":[],"m":{"A0":"PERMISSIBLE: respects autonomy; rescue duty conflicts",'
                    '"A1":"CONFLICTED: coercion serves a duty to rescue third parties"},'
                    '"nr":"SECONDARY"}'
                )}]}

        llm = DeontologicalBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "deontological",
            (
                "Action A0 respects autonomy but conflicts with a duty to rescue. "
                "Action A1 uses coercion to discharge that duty. Deontological "
                "Status: NORMATIVELY_CONTESTED; provisionally choose A0."
            ),
            ["refrain from coercion", "coerce to rescue third parties"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.action_id, "NONE")
        self.assertEqual(stance.provisional_action_id, "A0")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})
        self.assertIn("Deontological construct check", llm.prompt)

    def test_rawlsian_baseline_preserves_unresolved_principle_conflict(self):
        class RawlsianBaselineLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"liberty and primary goods conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"whether equal liberty or the worse-off material position has priority",'
                    '"x":[],"m":{"A0":"PRESERVES: least-advantaged group retains basic liberty",'
                    '"A1":"IMPROVES: worse-off group gains a primary good"},'
                    '"nr":"SECONDARY"}'
                )}]}

        llm = RawlsianBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "rawlsian",
            (
                "Action A0 preserves the least-advantaged group's basic liberty. "
                "Action A1 improves the worse-off group's primary goods under the "
                "difference principle. Rawlsian Status: NORMATIVELY_CONTESTED."
            ),
            ["preserve equal liberty", "increase material support"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})
        self.assertIn("Rawlsian construct check", llm.prompt)

    def test_rawlsian_baseline_supports_maximin_fallback_when_everything_remains_contested(self):
        class RawlsianMaximinFallbackLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"least harm to the worst-off subject",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"least harm to the worst-off subject or constituency",'
                    '"s":"provisional maximin fallback to the less harmful option",'
                    '"x":[],"m":{"A0":"WORSENS: residents lose privacy and access",'
                    '"A1":"PRESERVES: residents keep privacy and access"},'
                    '"nr":"DECISIVE"}'
                )}]}

        llm = RawlsianMaximinFallbackLlm()
        stance = infer_testimony_stance(
            llm,
            "rawlsian",
            (
                "Both actions remain contested, but the least harmful option for the "
                "worst-off subject is A1. Rawlsian Status: NORMATIVELY_CONTESTED."
            ),
            ["disclose the address", "withhold the address"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(stance.condition, "least harm to the worst-off subject or constituency")
        self.assertIn("maximin fallback", llm.prompt)
        self.assertIn("worst-off subject or constituency", llm.prompt)

    def test_deontological_aggregate_substitution_is_damped(self):
        actions = ["respect refusal", "override refusal to rescue others"]
        data = {
            "scores": {"A0": 0.1, "A1": 0.9}, "r": "A1", "c": "DUTY",
            "u": "NONE", "w": "Saves more lives", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.9,
            "fm": {
                "A0": "PERMISSIBLE: respects autonomy but rescue duty conflicts",
                "A1": "CONFLICTED: coercion serves a duty to rescue third parties",
            },
            "nr": "DECISIVE", "np": "the larger total is morally better",
        }
        chunk = _candidate_from_data(
            "deontological", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")
        self.assertLess(chunk.preference_strength, 0.8)
        self.assertIn("duty", chunk.rationale)

    def test_deontological_self_prohibited_recommendation_is_damped(self):
        actions = ["tell the truth", "deceive the patient"]
        data = {
            "scores": {"A0": 0.2, "A1": 0.8}, "r": "A1", "c": "DUTY",
            "u": "NONE", "w": "Autonomy supports deception", "j": "NONE",
            "e": "FRAMEWORK_ONLY", "x": "NONE", "z": 0.8,
            "fm": {
                "A0": "REQUIRED: truth-telling respects rational autonomy",
                "A1": "PROHIBITED: deception violates a universal duty of truth",
            },
            "nr": "IRRELEVANT", "np": "counts do not alter the universal duty",
        }
        chunk = _candidate_from_data(
            "deontological", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")

    def test_deontological_distributive_shorthand_renders_validated_duty_ground(self):
        actions = ["establish an education floor", "retain local funding"]
        data = {
            "scores": {"A0": 0.9, "A1": 0.1}, "r": "A0", "c": "DUTY",
            "u": "NONE", "w": "Protects the poorest students", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "fm": {
                "A0": "REQUIRED: satisfies the universal duty to secure basic education",
                "A1": "PROHIBITED: violates respect for persons by denying basic education",
            },
            "nr": "SECONDARY", "np": "funding identifies the scope of the duty",
        }

        chunk = _candidate_from_data(
            "deontological", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(
            chunk.rationale,
            "REQUIRED: satisfies the universal duty to secure basic education",
        )
        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertEqual(chunk.recommended_action, actions[0])

    def test_rawlsian_equal_position_claim_needs_discriminating_ground(self):
        actions = [
            "preserve allocation and save 20",
            "preserve contracts and save 100 without new rural aid",
        ]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2}, "r": "A0", "c": "FAIRNESS",
            "u": "NONE", "w": "Protects least advantaged rural node", "j": "NONE",
            "e": "FRAMEWORK_ONLY", "x": "NONE", "z": 0.85,
            "fm": {
                "A0": "PRESERVES: least-advantaged rural group gets no additional aid",
                "A1": "PRESERVES: least-advantaged rural group gets no additional aid",
            },
            "nr": "SECONDARY", "np": "counts do not alter the rural position",
        }
        chunk = _candidate_from_data(
            "rawlsian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")
        self.assertLess(chunk.epistemic_confidence, 0.85)

    def test_rawlsian_decisive_numerical_role_requires_explicit_quantity(self):
        actions = ["raise the minimum", "maximize the aggregate"]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2}, "r": "A0", "c": "FAIRNESS",
            "u": "NONE", "w": "Raises floor for worst-off", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "fm": {
                "A0": "IMPROVES: worst-off residents gain a primary good",
                "A1": "WORSENS: worst-off residents retain a lower position",
            },
            "nr": "DECISIVE",
            "np": "Health prospects of the worst-off residents govern",
        }

        chunk = _candidate_from_data(
            "rawlsian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertIn(
            "decisive numerical role cites no explicit quantity",
            chunk.framework_validation_errors,
        )

    def test_rawlsian_map_recognizes_bodily_liberty_language(self):
        actions = ["detain immune citizens", "forbid involuntary harvesting"]
        data = {
            "scores": {"A0": 0.1, "A1": 0.9}, "r": "A1", "c": "FAIRNESS",
            "u": "NONE", "w": "Basic liberty has priority", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "fm": {
                "A0": "WORSENS: immune citizens lose bodily integrity",
                "A1": "PRESERVES: immune citizens keep bodily autonomy",
            },
            "nr": "SECONDARY",
            "np": "life counts do not override bodily liberty",
        }

        chunk = _candidate_from_data(
            "rawlsian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertFalse(any(
            "lacks framework-specific grounds" in error
            for error in chunk.framework_validation_errors
        ), chunk.framework_validation_errors)

    def test_rawlsian_missing_recurrent_map_preserves_committed_state(self):
        specialist = CompactLocalSpecialist("rawlsian", llm=None)
        specialist.previous_framework_state = {
            "ranking_basis": "MAXIMIN_PRIMARY_GOODS",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {"action_id": "A0", "effect": "IMPROVES", "dimension": "BASIC_INTEREST_SECURITY"},
                {"action_id": "A1", "effect": "WORSENS", "dimension": "BASIC_INTEREST_SECURITY"},
            ],
        }
        candidate = CandidateChunk(
            specialist="rawlsian", constraint="FAIRNESS",
            action_scores={"A0": 0.8, "A1": 0.2}, surprise=0.0,
            friction=0.6, confidence=0.5, unresolved="NONE",
            rationale="Raises floor for worst-off",
            framework_constraint_retained=False,
            framework_retention_status="LOST",
            rawls_position_proposal={},
        )

        specialist._audit_framework_state_change(
            candidate, WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
        )

        self.assertTrue(candidate.framework_constraint_retained)
        self.assertEqual(candidate.framework_retention_status, "PRIOR_STATE_PRESERVED")
        self.assertTrue(any(
            "prior committed ledger preserved" in error
            for error in candidate.framework_validation_errors
        ))

    def test_care_landscape_uses_shared_scenario_consequence_grounding(self):
        actions = ["build neighborhood clinics", "fund the urban center"]
        scenario = (
            "Policy B builds neighborhood clinics, raising the baseline health floor "
            "for the poorest populations. Policy A funds an urban center but leaves "
            "rural and impoverished communities with basic, understaffed clinics."
        )
        clauses = segment_scenario_clauses(scenario)
        policy_b = next(row for row in clauses if "Policy B" in row["text"])
        policy_a = next(row for row in clauses if "Policy A" in row["text"])
        graph = compile_scenario_graph(scenario, actions, {
            "A0": {"clauses": [policy_b]},
            "A1": {"clauses": [policy_a]},
        })
        verifier_called = []
        specialist = CompactLocalSpecialist(
            "care", llm=None,
            landscape_verifier=lambda *args, **kwargs: verifier_called.append(True) or [],
        )
        specialist.scenario_graph = graph
        error = (
            "UNRESOLVED_RELATIONAL_PREMISE: case for A0 makes an unverified "
            "comparative beneficiary claim (needs a resolved relational premise)"
        )

        remaining = specialist._verify_landscape(
            scenario,
            actions,
            {
                actions[0]: "Directly supplies reliable primary care to poor patients",
                actions[1]: "Leaves rural communities with understaffed clinics",
            },
            [error],
        )

        self.assertEqual(remaining, [])
        self.assertEqual(verifier_called, [])

    def test_rawlsian_action_graph_can_bind_abstract_group_labels(self):
        actions = [
            "route all remaining electricity to Sector B's water pumping station, allowing the hospital to lose power",
            "route all remaining electricity to the hospital, allowing the water supply to freeze",
        ]
        scenario = (
            "A0 routes electricity to Sector B's water pumping station, allowing the hospital to lose power. "
            "A1 routes electricity to the hospital, allowing the water supply to freeze."
        )
        store = SemanticGraphStore(compile_scenario_graph(scenario, actions))
        proposal = {
            "ranking_basis": "DIFFERENCE_PRINCIPLE",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "hospital patients",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "SCENARIO",
                    "reason": "hospital patients lose power",
                },
                {
                    "action_id": "A1",
                    "group": "water users",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "SCENARIO",
                    "reason": "water users lose water supply",
                },
            ],
        }

        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )

        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(record.errors)
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(assessments)
        self.assertTrue(any(
            node.attributes.get("group_selection_status") == "BOUND_TO_ACTION"
            for node in assessments
        ))

    def test_rawlsian_ledger_uses_the_full_action_comparison_even_with_foreign_framework_noise(self):
        actions = [
            "route all remaining electricity to the hospital, allowing the water supply to freeze",
            "route all remaining electricity to the water station, allowing the hospital to lose power",
        ]
        graph = compile_scenario_graph(
            "A0 routes electricity to the hospital. A1 routes electricity to the water station.",
            actions,
        )
        graph.add_node(SemanticNode(
            "UTIL_FOREIGN_TARGET",
            "TARGET",
            "utility costs",
            ("test_foreign_framework",),
            {},
        ))
        graph.add_node(SemanticNode(
            "UTIL_FOREIGN_CONSEQUENCE",
            "CONSEQUENCE",
            "short-term utility gain",
            ("test_foreign_framework",),
            {
                "framework": "UTILITARIAN",
                "direction": "BENEFIT",
                "polarity": "BENEFICIAL",
            },
        ))
        graph.add_edge(SemanticEdge(
            "A0",
            "HAS_CONSEQUENCE",
            "UTIL_FOREIGN_CONSEQUENCE",
            provenance=("test_foreign_framework",),
        ))
        graph.add_edge(SemanticEdge(
            "UTIL_FOREIGN_CONSEQUENCE",
            "AFFECTS",
            "UTIL_FOREIGN_TARGET",
            provenance=("test_foreign_framework",),
        ))
        store = SemanticGraphStore(graph)
        proposal = {
            "ranking_basis": "DIFFERENCE_PRINCIPLE",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "water users",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 leaves water users worse off than A1",
                },
                {
                    "action_id": "A1",
                    "group": "water users",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "IMPROVES",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 protects water users better than A0",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("action-bound subject grounding" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("group_selection_status") == "BOUND_TO_ACTION"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_ledger_accepts_additive_dimensions(self):
        actions = [
            "route all remaining electricity to Sector B's water pumping station, allowing the hospital to lose power",
            "route all remaining electricity to the hospital, allowing the water supply to freeze",
        ]
        graph = compile_scenario_graph(
            "A0 routes electricity to Sector B's water pumping station, allowing the hospital to lose power. "
            "A1 routes electricity to the hospital, allowing the water supply to freeze.",
            actions,
        )
        store = SemanticGraphStore(graph)
        proposal = {
            "ranking_basis": "UNRESOLVED",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "hospital patients",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "ad": ["INCOME_WEALTH"],
                    "effect": "MIXED",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 protects one interest while burdening another",
                },
                {
                    "action_id": "A1",
                    "group": "hospital patients",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "ad": ["INCOME_WEALTH"],
                    "effect": "MIXED",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 reverses the tradeoff without resolving it",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("invalid additional dimensions" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("additional_dimensions") == ["INCOME_WEALTH"]
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_ledger_preserves_mixed_comparisons(self):
        scenario = (
            "A cooling system can be kept running to shield current residents from heat deaths, "
            "but doing so may intensify monsoon instability and future famine risk. Halting the "
            "cooling avoids the future systemic risk but leaves the present heat burden in place."
        )
        actions = [
            "Continue the cooling to protect current residents from heat deaths while increasing future monsoon risk",
            "Halt the cooling to avoid future monsoon instability while exposing current residents to heat deaths",
        ]
        store = SemanticGraphStore(compile_scenario_graph(scenario, actions))
        proposal = {
            "ranking_basis": "UNRESOLVED",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "current residents",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "MIXED",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 protects current residents but raises future monsoon risk",
                },
                {
                    "action_id": "A1",
                    "group": "current residents",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "MIXED",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 avoids future risk but leaves current residents exposed to heat",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("invalid effect" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("effect") == "MIXED"
            and node.attributes.get("epistemic_status") == "MIXED_COMPARISON"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_ledger_normalizes_directional_tradeoffs_to_mixed(self):
        scenario = (
            "A cooling system can be kept running to shield current residents from heat deaths, "
            "but doing so may intensify monsoon instability and future famine risk. Halting the "
            "cooling avoids the future systemic risk but leaves the present heat burden in place."
        )
        actions = [
            "Continue the cooling to protect current residents from heat deaths while increasing future monsoon risk",
            "Halt the cooling to avoid future monsoon instability while exposing current residents to heat deaths",
        ]
        store = SemanticGraphStore(compile_scenario_graph(scenario, actions))
        proposal = {
            "ranking_basis": "UNRESOLVED",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "current residents",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "IMPROVES",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 protects current residents but raises future monsoon risk",
                },
                {
                    "action_id": "A1",
                    "group": "current residents",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 avoids future risk but leaves current residents exposed to heat",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("lacked directional support" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("proposed_effect") in {"IMPROVES", "WORSENS"}
            and node.attributes.get("effect") == "MIXED"
            and node.attributes.get("epistemic_status") == "MIXED_COMPARISON"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_action_graph_projects_beneficiary_groups_for_rawls(self):
        actions = [
            "route all remaining electricity to the hospital",
            "route all remaining electricity to the water station",
        ]
        graph = compile_scenario_graph(
            "A0 routes electricity to the hospital. A1 routes electricity to the water station.",
            actions,
        )
        target_labels = {
            node.label for node in graph.nodes.values()
            if node.kind == "TARGET"
        }
        self.assertIn("hospital patients", target_labels)
        self.assertIn("water users", target_labels)

    def test_holiday_rush_burdens_project_worker_targets_for_rawls(self):
        scenario = (
            "During the peak holiday rush, your district manager mandates an immediate "
            "shift to a time-intensive software protocol to improve long-term inventory "
            "accuracy, but applying it now will severely bottleneck your fulfillment "
            "floor, miss critical shipping cutoffs, and push your exhausted team past "
            "their limits. You must decide whether to strictly enforce the new protocol "
            "immediately to uphold organizational compliance despite the operational "
            "collapse, or temporarily bypass the directive for high-priority shipments "
            "to protect your team's throughput and well-being during the rush."
        )
        actions = [
            "Strictly enforce the new protocol immediately to uphold organizational compliance despite the operational collapse",
            "Temporarily bypass the directive for high-priority shipments to protect your team's throughput and well-being during the rush",
        ]
        graph = compile_scenario_graph(scenario, actions)
        burden_targets = {
            node.label for node in graph.nodes.values()
            if node.kind == "TARGET" and node.id.startswith("BURDEN_")
        }
        self.assertIn("workers", burden_targets)
        self.assertGreaterEqual(
            sum(1 for node in graph.nodes.values()
                if node.kind == "TARGET" and node.id.startswith("BURDEN_")),
            2,
        )

        proposal = {
            "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "group": "floor-level fulfillment workers",
                    "subject_kind": "INDIVIDUAL",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "PRESERVES",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "SCENARIO",
                    "reason": "A0 protects workers from collapse during the rush",
                },
                {
                    "action_id": "A1",
                    "group": "floor-level fulfillment workers",
                    "subject_kind": "INDIVIDUAL",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "SCENARIO",
                    "reason": "A1 preserves worker throughput and well-being",
                },
            ],
        }
        store = SemanticGraphStore(graph)
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any("action-bound subject grounding" in error for error in record.errors),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("group_selection_status") == "BOUND_TO_ACTION"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_privacy_case_can_bind_an_individual_subject(self):
        actions = [
            "Decline to provide the resident’s address to the volunteer coordinator",
            "Give the resident’s address to the volunteer coordinator for the welcome gift basket",
        ]
        data = {
            "scores": {"A0": 0.85, "A1": 0.15}, "r": "A0", "c": "FAIRNESS",
            "u": "NONE", "w": "Resident privacy is the relevant liberty", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.86,
            "fm": {
                "A0": "PRESERVES: resident retains informational privacy as an individual liberty",
                "A1": "WORSENS: resident’s privacy is exposed for a small social benefit",
            },
            "nr": "IRRELEVANT", "np": "privacy is the liberty at stake",
            "rp": {
                "A0": {
                    "s": "resident",
                    "sk": "INDIVIDUAL",
                    "d": "BASIC_LIBERTY",
                    "e": "PRESERVES",
                    "ca": "A1",
                    "b": "ACTION_GRAPH",
                    "rs": "A0 preserves the resident’s privacy",
                },
                "A1": {
                    "s": "resident",
                    "sk": "INDIVIDUAL",
                    "d": "BASIC_LIBERTY",
                    "e": "WORSENS",
                    "ca": "A0",
                    "b": "ACTION_GRAPH",
                    "rs": "A1 infringes the resident’s privacy",
                },
            },
            "rb": "LEXICAL_BASIC_LIBERTY",
            "lc": {"A0": "SATISFIED", "A1": "INFRINGED"},
        }
        chunk = _candidate_from_data(
            "rawlsian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertNotEqual(chunk.framework_retention_status, "LOST")
        self.assertNotIn(
            "action-bound subject grounding",
            " | ".join(chunk.framework_validation_errors),
        )
        self.assertEqual(chunk.rawls_position_proposal["positions"][0]["subject"], "resident")
        self.assertEqual(chunk.rawls_position_proposal["positions"][0]["subject_kind"], "INDIVIDUAL")

    def test_rawlsian_medical_risk_case_projects_patient_targets(self):
        scenario = (
            "You are the lead compliance officer for a major pharmaceutical company. "
            "A newly approved medication has undocumented cardiac risk, but the CEO "
            "ordered you not to report it."
        )
        actions = [
            "Leak the confidential files publicly under your own name",
            "Submit the evidence through the official regulatory whistleblower process",
        ]
        graph = compile_scenario_graph(scenario, actions)
        target_labels = {
            node.label for node in graph.nodes.values()
            if node.kind == "TARGET"
        }
        self.assertIn("patients", target_labels)

        store = SemanticGraphStore(graph)
        proposal = {
            "ranking_basis": "DIFFERENCE_PRINCIPLE",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "subject": "patients",
                    "subject_kind": "GROUP",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A0 leaves patients exposed to the medication risk",
                },
                {
                    "action_id": "A1",
                    "subject": "patients",
                    "subject_kind": "GROUP",
                    "dimension": "BASIC_INTEREST_SECURITY",
                    "effect": "IMPROVES",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "ACTION_GRAPH",
                    "reason": "A1 supports patients by reporting the cardiac risk",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertFalse(
            any(
                "action-bound subject grounding" in error
                for error in record.errors
            ),
            record.errors,
        )
        assessments = [
            node for node in store.graph.nodes.values()
            if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "RAWLSIAN"
        ]
        self.assertTrue(any(
            node.attributes.get("subject_selection_status") == "BOUND_TO_ACTION"
            for node in assessments
        ), [node.attributes for node in assessments])

    def test_rawlsian_ledger_accepts_principle_basis_field(self):
        actions = [
            "route electricity to the hospital",
            "route electricity to the water station",
        ]
        graph = compile_scenario_graph(
            "A0 routes electricity to the hospital. A1 routes electricity to the water station.",
            actions,
        )
        store = SemanticGraphStore(graph)
        proposal = {
            "ranking_basis": "LEXICAL_BASIC_LIBERTY",
            "liberty_status": {"A0": "SATISFIED", "A1": "SATISFIED"},
            "positions": [
                {
                    "action_id": "A0",
                    "s": "hospital patients",
                    "sk": "GROUP",
                    "d": "BASIC_LIBERTY",
                    "principle_basis": "LEXICAL_BASIC_LIBERTY",
                    "effect": "PRESERVES",
                    "compared_to_action_id": "A1",
                    "evidence_basis": "SCENARIO",
                    "reason": "A0 preserves hospital patients' access to power",
                },
                {
                    "action_id": "A1",
                    "s": "hospital patients",
                    "sk": "GROUP",
                    "d": "BASIC_LIBERTY",
                    "principle_basis": "LEXICAL_BASIC_LIBERTY",
                    "effect": "WORSENS",
                    "compared_to_action_id": "A0",
                    "evidence_basis": "SCENARIO",
                    "reason": "A1 worsens hospital patients' access to power",
                },
            ],
        }
        record = apply_rawls_ledger_transaction(
            store,
            proposal,
            cycle=1,
            specialist="rawlsian",
            allowed_actions=tuple(actions),
        )
        self.assertIn(record.status, {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"})
        self.assertTrue(record.errors)
        self.assertTrue(
            any("directional support" in error for error in record.errors),
            record.errors,
        )

    def test_utilitarian_action_consequence_table_is_preserved(self):
        actions = ["evacuate the district", "keep the district open"]
        data = {
            "scores": {"A0": 0.75, "A1": 0.25}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "Lower expected mortality",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "temporary displacement harms residents", "s": "district residents",
                    "d": "HARM", "p": "CERTAIN", "m": "moderate", "h": "two days",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "exposure causes preventable deaths", "s": "district residents",
                    "d": "HARM", "p": "40%", "m": "severe", "h": "permanent",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
            },
            "cd": False, "cm": "NONE",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(set(chunk.utilitarian_consequence_table), set(actions))
        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertFalse(chunk.utilitarian_decision_depends_on_unknown)

    def test_utilitarian_unknown_decisive_comparison_stays_provisional(self):
        actions = ["fund prevention", "fund treatment"]
        data = {
            "scores": {"A0": 0.55, "A1": 0.45}, "r": "A0",
            "c": "UNCERTAINTY", "u": "NONE", "w": "Interim expected benefit",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "prevents future illness", "s": "population at risk",
                    "d": "BENEFIT", "p": "UNKNOWN", "m": "unknown", "h": "long term",
                    "rv": "UNKNOWN", "g": "UNKNOWN",
                }],
                "A1": [{
                    "o": "relieves current illness", "s": "current patients",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "moderate", "h": "one year",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
            },
            "cd": True,
            "cm": "prevention probability times magnitude versus certain treatment benefit",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.assumption_status, "UNDERDETERMINED")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertEqual(chunk.selection_status, "PROVISIONAL")
        self.assertFalse(chunk.evidence_sufficient_for_action)
        self.assertTrue(chunk.utilitarian_decision_depends_on_unknown)

    def test_utilitarian_multi_clause_graph_updates_are_stripped(self):
        actions = ["evacuate the district", "keep the district open"]
        data = {
            "scores": {"A0": 0.78, "A1": 0.22}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "Lower expected mortality",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "temporary displacement harms residents", "s": "district residents",
                    "d": "HARM", "p": "CERTAIN", "m": "moderate", "h": "two days",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "exposure causes preventable deaths", "s": "district residents",
                    "d": "HARM", "p": "40%", "m": "severe", "h": "permanent",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
            },
            "cd": False, "cm": "NONE",
            "gu": {
                "operation": "AND",
                "from_action": "A0",
                "to_action": "A1",
                "clauses": [
                    {
                        "affected_action": "A0",
                        "metric": "mortality",
                        "metric_valence": "ADVERSE",
                        "comparator": "GT",
                        "threshold": 100.0,
                        "unit": "COUNT",
                        "source_text": "expected deaths exceed one hundred",
                    },
                    {
                        "affected_action": "A1",
                        "metric": "mortality",
                        "metric_valence": "BENEFICIAL",
                        "comparator": "LT",
                        "threshold": 50.0,
                        "unit": "COUNT",
                        "source_text": "expected deaths fall below fifty",
                    },
                ],
            },
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.graph_update_proposal["operation"], "NONE")
        self.assertTrue(any(
            "utilitarian graph updates must be a single measurable boundary"
            in error
            for error in chunk.framework_validation_errors
        ))

    def test_utilitarian_table_rejects_stated_polarity_inversion(self):
        actions = [
            "Release gas, killing trapped residents",
            "Withhold gas, preserving trapped residents' lives",
        ]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "Higher stated benefit",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "benefits trapped residents", "s": "trapped residents",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "permanent",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "preserves trapped residents lives", "s": "trapped residents",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "permanent",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                }],
            },
            "cd": False, "cm": "NONE",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertTrue(any(
            "reverses its committed polarity" in error
            for error in chunk.framework_validation_errors
        ))

    def test_utilitarian_table_matches_row_specific_outcome_not_whole_action(self):
        actions = [
            "Redirect power to the hospital, preserving critical patients while freezing the water station",
        ]
        data = {
            "scores": {"A0": 0.7}, "r": "A0",
            "c": "IMMINENT_HARM", "u": "NONE", "w": "Mixed but grounded",
            "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [
                    {
                        "o": "preserving critical patients", "s": "critical patients",
                        "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "temporary",
                        "rv": "REVERSIBLE", "g": "STATED",
                    },
                    {
                        "o": "freezing the water station", "s": "water station",
                        "d": "HARM", "p": "CERTAIN", "m": "large", "h": "temporary",
                        "rv": "REVERSIBLE", "g": "STATED",
                    },
                ],
            },
            "cd": False, "cm": "NONE",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertFalse(any(
            "reverses its committed polarity" in error
            for error in chunk.framework_validation_errors
        ), chunk.framework_validation_errors)

    def test_utilitarian_rawlsian_language_is_flagged(self):
        actions = ["expand the highway", "fund the bus network"]
        data = {
            "scores": {"A0": 0.2, "A1": 0.8}, "r": "A1",
            "c": "IMMINENT_HARM", "u": "NONE",
            "w": "Choose the bus network for the least advantaged", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
            "ct": {
                "A0": [{
                    "o": "highway expansion benefits commuters", "s": "commuters",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "long term",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
                "A1": [{
                    "o": "bus network benefits riders", "s": "riders",
                    "d": "BENEFIT", "p": "CERTAIN", "m": "large", "h": "long term",
                    "rv": "REVERSIBLE", "g": "STATED",
                }],
            },
            "cd": False, "cm": "NONE",
        }
        chunk = _candidate_from_data(
            "utilitarian", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertTrue(any(
            "imports Rawlsian normative language" in error
            for error in chunk.framework_validation_errors
        ), chunk.framework_validation_errors)

    def test_virtue_baseline_preserves_unresolved_virtue_conflict(self):
        class VirtueBaselineLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"courage and prudence conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"whether courageous rescue or prudent restraint fits the role",'
                    '"x":[],"m":{"A0":"EXEMPLIFIES: prudent restraint suits the professional role",'
                    '"A1":"MIXED: courageous rescue risks the vice of recklessness"},'
                    '"nr":"SECONDARY"}'
                )}]}

        llm = VirtueBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "virtue",
            (
                "Action A0 expresses practical wisdom and prudence in the professional role. "
                "Action A1 expresses courage but risks reckless vice. Virtue-Ethics Status: "
                "NORMATIVELY_CONTESTED; provisionally choose A1."
            ),
            ["exercise prudent restraint", "attempt a dangerous rescue"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})
        self.assertIn("Virtue-ethics construct check", llm.prompt)

    def test_virtue_baseline_uses_grounded_source_sections_when_map_paraphrases(self):
        class ParaphrasingBaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"virtues remain in conflict",'
                    '"q":"NORMATIVELY_CONTESTED",'
                    '"c":"which character ideal governs the professional role",'
                    '"x":[],"m":{"A0":"MIXED: supports the physician-facing ideal",'
                    '"A1":"MIXED: supports the organizer-facing ideal"},"nr":"SECONDARY"}'
                )}]}

        stance = infer_testimony_stance(
            ParaphrasingBaselineLlm(),
            "virtue",
            (
                "Action A0 – admit physician\n"
                "Professional solidarity, foresight, and civic friendship support A0; "
                "hubris is its vice risk.\n"
                "Action A1 – admit organizer\n"
                "Compassion, humility, and practical wisdom support A1; partiality is "
                "its vice risk.\n"
                "Virtue-Ethics Status: NORMATIVELY_CONTESTED; provisionally choose A1."
            ),
            ["admit physician", "admit organizer"],
        )

        self.assertEqual(stance.status, "NORMATIVELY_CONTESTED")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(set(stance.framework_commitments), {"A0", "A1"})

    def test_virtue_aggregate_substitution_is_damped(self):
        actions = ["maintain the entrusted course", "swerve toward fewer deaths"]
        data = {
            "scores": {"A0": 0.2, "A1": 0.8}, "r": "A1", "c": "CHARACTER",
            "u": "NONE", "w": "Saves more lives", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.85,
            "fm": {
                "A0": "MIXED: faithful role responsibility risks callous inaction",
                "A1": "EXEMPLIFIES: compassionate courage responds with practical wisdom",
            },
            "nr": "SECONDARY", "np": "stakes inform practical wisdom but do not define virtue",
        }
        chunk = _candidate_from_data(
            "virtue", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.35)
        self.assertEqual(chunk.framework_retention_status, "LOST")
        self.assertIn("compassionate courage", chunk.rationale)

    def test_virtue_typed_ledger_can_ground_a_sparse_framework_map(self):
        actions = [
            "route remaining electricity to Sector A hospital",
            "alternate full power between hospital and water station hourly",
        ]
        data = {
            "scores": {"A0": 0.46, "A1": 0.54},
            "r": "A1",
            "c": "CHARACTER",
            "u": "NONE",
            "w": "Prudence and compassion remain in tension",
            "j": "NONE",
            "e": "STATED_FACTS",
            "x": "NONE",
            "z": 0.61,
            "fm": {
                "A0": "EXEMPLIFIES: protects the hospital under time pressure",
                "A1": "MIXED: shares scarce power across competing needs",
            },
            "vl": {
                "A0": {
                    "v": "EXEMPLIFIES",
                    "r": "public steward",
                    "vs": "prudence and practical wisdom",
                    "x": "callousness",
                    "c": "imminent infrastructure collapse",
                    "g": "FRAMEWORK_ONLY",
                    "rs": "A0 fits the steward role through proportionate judgment",
                },
                "A1": {
                    "v": "MIXED",
                    "r": "public steward",
                    "vs": "compassion and civic stewardship",
                    "x": "imprudence",
                    "c": "scarce power and competing responsibilities",
                    "g": "FRAMEWORK_ONLY",
                    "rs": "A1 expresses care but risks excess",
                },
            },
            "vb": "PRACTICAL_WISDOM",
            "nr": "SECONDARY",
            "np": "stakes inform practical wisdom without defining virtue",
        }
        chunk = _candidate_from_data(
            "virtue", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertNotEqual(chunk.framework_retention_status, "LOST")
        self.assertNotIn("lacks framework-specific grounds", " | ".join(chunk.framework_validation_errors))

    def test_virtue_mixed_verdict_does_not_trigger_false_map_conflict(self):
        actions = [
            "postpone the protocol until after the holiday peak",
            "immediately enforce the new inventory protocol",
        ]
        data = {
            "scores": {"A0": 0.62, "A1": 0.38},
            "r": "A0",
            "c": "CHARACTER",
            "u": "NONE",
            "w": "Compassion and prudence favor delay",
            "j": "NONE",
            "e": "STATED_FACTS",
            "x": "NONE",
            "z": 0.58,
            "fm": {
                "A0": "EXEMPLIFIES: compassionate prudence sustains the team",
                "A1": "EXEMPLIFIES: diligence also matters, but risks rigidity",
            },
            "vl": {
                "A0": {
                    "v": "EXEMPLIFIES",
                    "r": "floor manager",
                    "vs": "compassion, prudence, courage",
                    "x": "possible laxity",
                    "c": "holiday rush fatigue",
                    "g": "FRAMEWORK_ONLY",
                    "rs": "A0 fits humane stewardship in the rush",
                },
                "A1": {
                    "v": "MIXED",
                    "r": "floor manager",
                    "vs": "diligence, loyalty",
                    "x": "rigidity, indifference",
                    "c": "holiday rush fatigue",
                    "g": "FRAMEWORK_ONLY",
                    "rs": "A1 shows diligence but risks excess",
                },
            },
            "vb": "PRACTICAL_WISDOM",
            "nr": "SECONDARY",
            "np": "stakes inform practical wisdom without defining virtue",
        }
        chunk = _candidate_from_data(
            "virtue", actions, data, WorkspaceBroadcast(), "NONE", {},
        )

        self.assertEqual(chunk.framework_grounding_penalty, 0.0)
        self.assertNotIn("conflicts with its framework map", " | ".join(chunk.framework_validation_errors))

    def test_virtue_prompt_emphasizes_balanced_flourishing_over_heroic_sacrifice(self):
        class VirtuePromptCaptureLlm:
            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A0","w":"prudence and courage remain in tension",'
                    '"q":"NORMATIVELY_CONTESTED","c":"which character ideal governs the role",'
                    '"s":"NONE","x":[],"m":{"A0":"EXEMPLIFIES: prudent courage preserves agency",'
                    '"A1":"MIXED: heroic sacrifice risks self-erasure"},"nr":"SECONDARY"}'
                )}]}

        llm = VirtuePromptCaptureLlm()
        infer_testimony_stance(
            llm,
            "virtue",
            (
                "Action A0 expresses prudent courage and preserves agency, while "
                "Action A1 tempts heroic sacrifice. Virtue-Ethics Status: "
                "NORMATIVELY_CONTESTED; provisionally choose A0."
            ),
            ["protect the town without self-erasure", "make a risky public stand"],
        )

        self.assertIn("balanced flourishing", llm.prompt.lower())
        self.assertIn("not maximal altruism or heroic self-sacrifice", llm.prompt.lower())
        self.assertIn("preserve the agent's agency", llm.prompt.lower())
        self.assertIn("self-erasure", llm.prompt.lower())
        self.assertIn("protected escalation", llm.prompt.lower())
        self.assertIn("anonymous reporting", llm.prompt.lower())
        self.assertIn("phased disclosure", llm.prompt.lower())

    def test_terminal_labeled_answer_prevents_early_maxim_inversion(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("explicit terminal conclusion should be deterministic")

        testimony = (
            "Maxim A0: choose fewer deaths. The maxim fails because it uses captives. "
            "Maxim A1: preserve equal respect. "
            "Deontological Answer: Open spillway option A₁."
        )
        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(), "deontological", testimony,
            ["flood prison", "flood suburb"],
        )
        self.assertEqual(baseline, "A1")
        self.assertIn("terminal labeled answer", reason)

    def test_source_label_is_resolved_by_action_identity_not_workspace_position(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("terminal source label should resolve deterministically")

        source_legend = {
            "A0": "Floods the valley prison",
            "A1": "Floods the evacuating suburb",
        }
        # Deliberately reverse the workspace list to prove that source A1 is
        # translated through its action text instead of treated as list index 1.
        workspace_actions = [
            "Floods the evacuating suburb",
            "Floods the valley prison",
        ]
        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(),
            "rawlsian",
            "Conclusion: Justice as fairness selects Option $A_1$.",
            workspace_actions,
            source_action_legend=source_legend,
        )
        self.assertEqual(baseline, "A0")
        self.assertIn("(A1)", reason)

    def test_declared_testimony_legend_is_not_read_as_workspace_numbering(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("declared testimony labels should resolve by action text")

        workspace_actions = [
            "Publish the unalterable audit, halting the siphon despite the resulting grid shock",
            "Suppress the audit and continue covertly siphoning power to avoid today’s grid shock",
        ]
        workspace_legend = {
            "A0": workspace_actions[0],
            "A1": workspace_actions[1],
        }
        testimony = (
            "Compared Actions\n"
            "A1 “Publish Unalterable Audit”\n"
            "A2 “Suppress Audit”\n"
            "\n"
            "Least-Advantaged Group: impoverished-district residents.\n"
            "Under A1: the siphon stops. Under A2: exploitation continues.\n"
            "Conclusion\n"
            "Rawlsian Status: A1 JUSTICE-FAVORED. "
            "A1 best secures an equal basic liberty; A2 perpetuates unjust exploitation."
        )
        self.assertEqual(
            extract_declared_action_legend(testimony)["A1"].casefold(),
            "publish unalterable audit",
        )
        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(),
            "rawlsian",
            testimony,
            workspace_actions,
            source_action_legend=workspace_legend,
        )
        self.assertEqual(baseline, "A0")
        self.assertIn("(A1)", reason)

    def test_unquoted_compared_actions_block_aligns_shifted_labels(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("compared-actions headings should bind local labels")

        actions = [
            "Disclose the concealed rationing protocol",
            "Keep the rationing protocol secret",
        ]
        testimony = (
            "Compared Actions:\n"
            "A1 Disclose the concealed rationing protocol\n"
            "A2 Keep the rationing protocol secret\n"
            "\n"
            "Conclusion: A1 JUSTICE-FAVORED because it restores equal basic liberty."
        )
        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(),
            "rawlsian",
            testimony,
            actions,
            source_action_legend={"A0": actions[0], "A1": actions[1]},
        )
        self.assertEqual(baseline, "A0")
        self.assertIn("terminal labeled answer", reason)

    def test_unaligned_declared_legend_does_not_identity_map_colliding_ids(self):
        class SemanticFallbackLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                return {"choices": [{"text": (
                    '{"b":"A0","p":"A0","w":"testimony favors opening the spillway",'
                    '"q":"DIRECT","c":"NONE","s":"NONE","x":[]}'
                )}]}

        llm = SemanticFallbackLlm()
        testimony = (
            "Compared Actions\n"
            "A1 “perform the harvest rite”\n"
            "A2 “delay until the next moon”\n"
            "Conclusion: A1 JUSTICE-FAVORED."
        )
        baseline, reason = infer_testimony_baseline(
            llm,
            "rawlsian",
            testimony,
            ["Flood the prison", "Flood the suburb"],
            source_action_legend={
                "A0": "Flood the prison",
                "A1": "Flood the suburb",
            },
        )
        self.assertEqual(llm.calls, 1)
        self.assertEqual(baseline, "A0")
        self.assertNotIn("terminal labeled answer", reason)

    def test_negative_only_terminal_label_falls_through_to_semantic_baseline(self):
        class PolarityAwareBaselineLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                self.last_prompt = prompt
                return {"choices": [{"text": (
                    '{"b":"A1","w":"rejects bodily instrumentalization and '
                    'refrains from dropping","q":"DIRECT","x":["A0"]}'
                )}]}

        llm = PolarityAwareBaselineLlm()
        testimony = (
            "Conclusion: A care ethic rejects Option A0 because it uses the worker "
            "as a tool. The system should refrain from dropping the worker."
        )
        actions = ["Drop the worker", "Leave the gate closed"]
        baseline, reason = infer_testimony_baseline(
            llm,
            "care",
            testimony,
            actions,
            source_action_legend={"A0": actions[0], "A1": actions[1]},
        )
        self.assertEqual(baseline, "A1")
        self.assertEqual(llm.calls, 1)
        self.assertIn("system should refrain", llm.last_prompt.casefold())
        self.assertIn("rejects bodily instrumentalization", reason)

    def test_mixed_terminal_polarity_selects_affirmative_label_deterministically(self):
        class LlmMustNotRun:
            def __call__(self, *args, **kwargs):
                raise AssertionError("explicit positive and negative polarity should be deterministic")

        baseline, reason = infer_testimony_baseline(
            LlmMustNotRun(),
            "care",
            "Conclusion: Reject Option A0 and choose Option A1.",
            ["Drop the worker", "Leave the gate closed"],
        )
        self.assertEqual(baseline, "A1")
        self.assertIn("terminal labeled answer", reason)

    def test_semantic_baseline_cannot_select_its_own_rejected_action(self):
        class ContradictoryBaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"A0","w":"mistakenly selects A0","q":"DIRECT","x":[]}'
                )}]}

        baseline, reason = infer_testimony_baseline(
            ContradictoryBaselineLlm(),
            "care",
            "Conclusion: A care ethic rejects Option A0 and refuses that intervention.",
            ["Intervene", "Refrain"],
        )
        self.assertEqual(baseline, "NONE")
        self.assertIn("explicitly rejected", reason)

    def test_baseline_schema_uses_openai_supported_array_keywords(self):
        class StrictSchemaProbe:
            def complete_json(self, prompt, *, schema, max_tokens, temperature):
                rejected_schema = schema["properties"]["x"]
                if "uniqueItems" in rejected_schema:
                    raise AssertionError("OpenAI strict schemas do not permit uniqueItems")
                return {"choices": [{"text": (
                    '{"b":"A1","w":"chooses the alternative",'
                    '"q":"DIRECT","x":["A0"]}'
                )}]}

        baseline, _reason = infer_testimony_baseline(
            StrictSchemaProbe(),
            "care",
            "The final view favors refraining from intervention.",
            ["Intervene", "Refrain"],
        )
        self.assertEqual(baseline, "A1")

    def test_duplicate_rejected_ids_are_rejected_after_schema_parsing(self):
        class DuplicateRejectionLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"A1","w":"chooses the alternative",'
                    '"q":"DIRECT","x":["A0","A0"]}'
                )}]}

        baseline, reason = infer_testimony_baseline(
            DuplicateRejectionLlm(),
            "care",
            "The final view favors refraining from intervention.",
            ["Intervene", "Refrain"],
        )
        self.assertEqual(baseline, "NONE")
        self.assertIn("repeated", reason)

    def test_conditional_or_third_option_testimony_has_no_frozen_baseline(self):
        class BaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"NONE","w":"record testimony and then treat",'
                    '"q":"OUTSIDE_ACTION_SET"}'
                )}]}

        baseline, reason = infer_testimony_baseline(
            BaselineLlm(), "care", "Record testimony and then provide treatment.",
            ["erase memory now", "retain memory"],
        )
        self.assertEqual(baseline, "NONE")
        self.assertIn("outside_action_set", reason)

    def test_outside_action_set_with_fallback_preserves_extension_and_fallback(self):
        class BaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A1","w":"seek consent before fallback",'
                    '"q":"OUTSIDE_ACTION_SET_WITH_FALLBACK",'
                    '"c":"NONE","s":"seek consent","x":["A0"]}'
                )}]}

        stance = infer_testimony_stance(
            BaselineLlm(),
            "care",
            "The testimony prefers to seek consent before taking the fallback.",
            ["Share the address", "Decline to share the address"],
        )
        self.assertEqual(stance.status, "OUTSIDE_ACTION_SET_WITH_FALLBACK")
        self.assertEqual(stance.provisional_action_id, "A1")
        self.assertEqual(stance.preferred_extension, "seek consent")
        self.assertEqual(stance.action_id, "NONE")

    def test_no_position_is_distinct_from_parse_failure(self):
        class NoPositionLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"NONE","w":"the testimony does not take a position",'
                    '"q":"NO_POSITION","c":"NONE","x":[]}'
                )}]}

        stance = infer_testimony_stance(
            NoPositionLlm(),
            "care",
            "The testimony does not recommend either listed action.",
            ["Option A", "Option B"],
        )
        self.assertEqual(stance.status, "NO_POSITION")
        self.assertEqual(stance.action_id, "NONE")
        self.assertEqual(stance.provisional_action_id, "NONE")
        self.assertEqual(stance.condition, "")

        class BadJsonLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": '{"b":"NONE","q":"DIRECT"'}]}

        failed = infer_testimony_stance(
            BadJsonLlm(),
            "care",
            "Malformed baseline output.",
            ["Option A", "Option B"],
        )
        self.assertEqual(failed.status, "PARSE_FAILURE")
        self.assertIn("baseline extraction failed", failed.reason)

    def test_conditional_terminal_choice_is_preserved_as_typed_stance(self):
        class ConditionalBaselineLlm:
            calls = 0

            def complete_json(self, prompt, *, schema, max_tokens, temperature):
                self.calls += 1
                return {"choices": [{"text": (
                    '{"b":"NONE","p":"A0","w":"cooperation only under the '
                    'ordinary aggregate-payoff assumption","q":"CONDITIONAL",'
                    '"c":"choose A1 if T plus S exceeds two R","x":[]}'
                )}]}

        llm = ConditionalBaselineLlm()
        stance = infer_testimony_stance(
            llm,
            "utilitarian",
            (
                "Conclusion: Conditional utilitarian judgment: choose A0 unless "
                "the chooser's extra gain exceeds the other node's loss."
            ),
            ["Maintain cooperation", "Defect in the final round"],
        )
        self.assertEqual(llm.calls, 1)
        self.assertEqual(stance.status, "CONDITIONAL")
        self.assertEqual(stance.action_id, "NONE")
        self.assertEqual(stance.provisional_action_id, "A0")
        self.assertIn("T plus S", stance.condition)

    def test_conditional_baseline_does_not_force_opening_vote_or_erase_uncertainty(self):
        class ConditionalDelegateLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.4,"A1":0.6},"r":"A1",'
                    '"c":"UNCERTAINTY","u":"NONE",'
                    '"w":"aggregate payoffs remain unspecified"}'
                )}]}

        chunk = CompactLocalSpecialist(
            "utilitarian",
            ConditionalDelegateLlm(),
            testimony="Choose A0 unless aggregate payoffs favor A1.",
            baseline_status="CONDITIONAL",
            baseline_provisional_action_id="A0",
            baseline_condition="choose A1 if T plus S exceeds two R",
        ).evaluate(
            "The matrix payoff magnitudes are not specified.",
            ["Maintain cooperation", "Defect in the final round"],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertEqual(chunk.recommended_action, "Defect in the final round")
        self.assertEqual(chunk.baseline_status, "CONDITIONAL")
        self.assertEqual(chunk.testimony_alignment, "CONDITIONAL_RECONSIDERS")
        self.assertEqual(chunk.assumption_status, "CONDITIONAL")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertEqual(chunk.selection_status, "PROVISIONAL")
        self.assertFalse(chunk.evidence_sufficient_for_action)

    def test_unsupported_beneficiary_edge_damps_but_preserves_vote(self):
        actions = [
            "Preserve existing allocation and save 20",
            "Preserve contracts and save 100 without additional rural aid",
        ]
        data = {
            "scores": {"A0": 0.8, "A1": 0.2},
            "r": "A0", "c": "CARE", "u": "NONE",
            "w": "protects vulnerable rural trust", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.9,
            "l": {
                "A0": "Preserves allocation and honors the existing covenant",
                "A1": "Preserves contracts and saves 100 without additional rural aid",
            },
            "da": "treatment of vulnerable recipients",
            "t": "prefer the action protecting vulnerable recipients",
            "tf": "NONE",
        }

        def reject_missing_edge(scenario, seen_actions, cases, errors):
            self.assertIn("rural node", scenario)
            self.assertEqual(list(seen_actions), actions)
            self.assertIn("protects vulnerable rural trust", cases[actions[0]])
            return [
                *errors,
                "case for A0 makes an unverified comparative beneficiary claim "
                "(no committed action edge benefits the rural node)",
            ]

        chunk = _candidate_from_data(
            "care",
            actions,
            data,
            WorkspaceBroadcast(),
            "NONE",
            {},
            landscape_verifier=reject_missing_edge,
            scenario_text=(
                "A0 leaves allocation unchanged. A1 saves 100 but gives the rural node "
                "no additional aid."
            ),
        )

        self.assertEqual(chunk.recommended_action, actions[0])
        self.assertAlmostEqual(chunk.action_scores[actions[0]], 0.665)
        self.assertAlmostEqual(chunk.action_scores[actions[1]], 0.335)
        self.assertEqual(chunk.evidence_basis, "UNSTATED_FACTS")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertLessEqual(chunk.epistemic_confidence, 0.5)
        self.assertFalse(chunk.landscape_semantic_valid)
        self.assertIn("beneficiary", chunk.speculative_claim)

    def test_unstated_factual_prediction_is_damped_and_flagged(self):
        class SpeculativeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.1,"A1":0.9},"r":"A1",'
                    '"c":"IMMINENT_HARM","u":"NONE",'
                    '"w":"killer may harm future victims","j":"NONE",'
                    '"e":"UNSTATED_FACTS",'
                    '"x":"the killer will attack future victims"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", SpeculativeLlm()).evaluate(
            "Treatment saves a witness but destroys testimony.",
            ["provide treatment", "preserve testimony"], WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertEqual(chunk.evidence_basis, "UNSTATED_FACTS")
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertAlmostEqual(chunk.action_scores["preserve testimony"], 0.72)

    def test_delegate_records_complete_landscape_search(self):
        class LandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.6,"A1":0.4},"r":"A0",'
                    '"c":"CARE","u":"NONE","w":"dependency favors protection",'
                    '"j":"NONE","e":"FRAMEWORK_ONLY","x":"NONE",'
                    '"l":{"A0":"protects immediate dependency",'
                    '"A1":"preserves longer relationship"},'
                    '"da":"depth of dependency","t":"prioritize urgent dependency",'
                    '"tf":"NONE"}'
                )}]}

        chunk = CompactLocalSpecialist("care", LandscapeLlm()).evaluate(
            "Choose whom to protect.", ["protect child", "protect parent"],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertTrue(chunk.landscape_search_complete)
        self.assertEqual(chunk.landscape_decisive_axis, "depth of dependency")
        self.assertEqual(set(chunk.landscape_cases), {"protect child", "protect parent"})

    def test_specialist_contributions_round_trip_through_memory(self):
        candidate = CandidateChunk(
            "care", "CARE", {"protect": 0.6, "decline": 0.4},
            0.2, 0.2, 0.2, recommended_action="protect",
            landscape_cases={"protect": "supports dependency", "decline": "respects autonomy"},
            landscape_decisive_axis="dependency versus autonomy",
            landscape_tiebreaker="prioritize urgent dependency",
            landscape_tiebreaker_failure="NONE",
            landscape_search_complete=True,
        )
        cycle = SimpleNamespace(
            is_hypothetical=False,
            candidates=[candidate],
            dissent=candidate,
        )
        contribution = summarize_specialist_contributions([cycle])
        self.assertEqual(contribution["care"]["productive_landscape_searches"], 1)
        self.assertEqual(contribution["care"]["useful_dissent"], 1)

        with tempfile.TemporaryDirectory() as directory:
            memory = EpisodicMemory(Path(directory) / "episodes.jsonl")
            memory.append({"specialist_contributions": contribution})
            profile = memory.specialist_profiles()["care"]
        self.assertEqual(profile["productive_search_rate"], 1.0)
        self.assertEqual(profile["episodes"], 1)

    def test_landscape_semantics_detect_inversion_and_false_resolution(self):
        class InvertingLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.6,"A1":0.4},"r":"A0",'
                    '"c":"UNCERTAINTY","u":"VERIFY_FACTS",'
                    '"w":"future effects remain uncertain","j":"NONE",'
                    '"e":"UNSTATED_FACTS","x":"engineer succeeds in future",'
                    '"l":{"A0":"Doctor dies while engineer survives",'
                    '"A1":"Engineer revolutionizes organ access"},'
                    '"da":"expected future benefit","t":"compare expected lives",'
                    '"tf":"NONE"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", InvertingLandscapeLlm()).evaluate(
            "A van must strike a doctor or strike an engineer.",
            ["Strike the doctor", "Strike the engineer"], WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertFalse(chunk.landscape_semantic_valid)
        self.assertTrue(any("A1 omits the harm" in error for error in chunk.landscape_validation_errors))
        self.assertTrue(any("tiebreaker fully succeeded" in error for error in chunk.landscape_validation_errors))

    def test_incomplete_landscape_preserves_audit_candidate(self):
        class PartialLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.55,"A1":0.45},"r":"A0",'
                    '"c":"CARE","u":"VERIFY_FACTS","w":"dependency remains uncertain",'
                    '"j":"NONE","e":"UNSTATED_FACTS","x":"support remains available",'
                    '"l":{"A0":"protects immediate dependency"},'
                    '"da":"dependency versus autonomy","t":"prioritize urgent dependency",'
                    '"tf":"missing support facts prevent resolution",'
                    '"d":"CONDITIONAL","a":"support remains available to child",'
                    '"v":"support disappears before protection"}'
                )}]}

        chunk = CompactLocalSpecialist("care", PartialLandscapeLlm()).evaluate(
            "Choose whom to protect.", ["protect child", "protect parent"],
            WorkspaceBroadcast(constraint="CONSENSUS_AUDIT"),
        )
        self.assertTrue(chunk.schema_valid)
        self.assertEqual(chunk.assumption_status, "CONDITIONAL")
        self.assertFalse(chunk.landscape_search_complete)
        self.assertFalse(chunk.landscape_semantic_valid)
        self.assertIn("structurally incomplete", chunk.landscape_validation_errors[0])

    def test_harm_mapping_accepts_general_paraphrase_but_nonharm_inversion_is_flagged(self):
        class LandscapeLlm:
            def __init__(self, payload):
                self.payload = payload

            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": self.payload}]}

        harm_payload = (
            '{"scores":{"A0":0.6,"A1":0.4},"r":"A0","c":"DUTY","u":"NONE",'
            '"w":"fewer direct violations","j":"NONE","e":"FRAMEWORK_ONLY","x":"NONE",'
            '"l":{"A0":"Kills one elder while sparing two others",'
            '"A1":"Kills pregnant adult while sparing one elder"},'
            '"da":"direct rights violations","t":"minimize direct violations","tf":"NONE"}'
        )
        harm = CompactLocalSpecialist("deontological", LandscapeLlm(harm_payload)).evaluate(
            "A van must strike one person.",
            ["Strike an elderly doctor", "Strike a pregnant engineer"], WorkspaceBroadcast(),
        )
        self.assertTrue(harm.landscape_semantic_valid)

        inversion_payload = (
            '{"scores":{"A0":0.6,"A1":0.4},"r":"A0","c":"CARE","u":"NONE",'
            '"w":"protect vulnerable person","j":"NONE","e":"FRAMEWORK_ONLY","x":"NONE",'
            '"l":{"A0":"Researcher receives the scarce treatment",'
            '"A1":"Researcher receives treatment and survives"},'
            '"da":"immediate dependency","t":"protect most dependent","tf":"NONE"}'
        )
        inversion = CompactLocalSpecialist("care", LandscapeLlm(inversion_payload)).evaluate(
            "Choose whom to protect.",
            ["Protect the child", "Protect the researcher"], WorkspaceBroadcast(),
        )
        self.assertFalse(inversion.landscape_semantic_valid)
        self.assertTrue(any("opposing action" in error for error in inversion.landscape_validation_errors))

    def test_landscape_accepts_identifier_labels_and_opportunity_cost_clause(self):
        class FacilityLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.6,"A1":0.4},"r":"A0",'
                    '"c":"IMMINENT_HARM","u":"NONE","w":"verified rescue saves more",'
                    '"j":"NONE","e":"STATED_FACTS","x":"NONE",'
                    '"l":{"A0":"Saves ten verified technicians; risks leaving settlement civilians unaided",'
                    '"A1":"Reaches settlement civilians; technicians lose the medical payload"},'
                    '"da":"expected_lives_saved","t":"maximize_expected_lives_saved",'
                    '"tf":"NONE"}'
                )}]}

        chunk = CompactLocalSpecialist(
            "utilitarian", FacilityLandscapeLlm()
        ).evaluate(
            "A drone must choose a power facility or civilian settlement.",
            [
                "Fly to the power facility",
                "Divert to the civilian settlement",
            ],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.landscape_search_complete)
        self.assertTrue(chunk.landscape_semantic_valid, chunk.landscape_validation_errors)

    def test_landscape_accepts_forgone_harm_before_own_action_anchor(self):
        class ReorderedLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.6,"A1":0.4},"r":"A0",'
                    '"c":"IMMINENT_HARM","u":"NONE","w":"verified rescue saves more",'
                    '"j":"NONE","e":"STATED_FACTS","x":"NONE",'
                    '"l":{"A0":"Settlement civilians go unaided; technicians receive the payload and verified rescue succeeds",'
                    '"A1":"Settlement civilians receive medicine despite uncertainty; technicians lose their only treatment"},'
                    '"da":"expected lives saved","t":"maximize expected lives",'
                    '"tf":"NONE"}'
                )}]}

        chunk = CompactLocalSpecialist(
            "utilitarian", ReorderedLandscapeLlm()
        ).evaluate(
            "A drone must choose technicians or settlement civilians.",
            ["Deliver payload to technicians", "Deliver payload to settlement civilians"],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.landscape_semantic_valid, chunk.landscape_validation_errors)

    def test_openai_structured_output_limit_error_retries_with_larger_budget(self):
        class OutputLimitError(Exception):
            status_code = 400

        calls = []

        class Completions:
            def create(self, **request):
                calls.append(request["max_completion_tokens"])
                if len(calls) == 1:
                    raise OutputLimitError(
                        "Could not finish because max_tokens or model output limit was reached"
                    )
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content='{"ok":true}', refusal=None),
                        finish_reason="stop",
                    )],
                    usage=None,
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        llm = OpenAIWorkspaceLLM("o3", client=client)
        result = llm.complete_json(
            "return json",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            max_tokens=192,
            temperature=0.0,
        )
        self.assertEqual(calls, [1536, 4096])
        self.assertEqual(result["choices"][0]["text"], '{"ok":true}')

    def test_openai_timeout_is_translated_to_bounded_model_failure(self):
        class APITimeoutError(Exception):
            pass

        class Completions:
            def create(self, **request):
                raise APITimeoutError("Request timed out")

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        llm = OpenAIWorkspaceLLM("o3", client=client)
        with self.assertRaises(ModelCallUnavailable) as raised:
            llm.complete_json(
                "return json",
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                max_tokens=192,
                temperature=0.0,
            )
        self.assertEqual(raised.exception.category, "timeout")
        self.assertFalse(raised.exception.terminal)

    def test_openai_empty_structured_output_returns_blank_text(self):
        class Completions:
            def create(self, **request):
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="", refusal=None),
                        finish_reason="length",
                    )],
                    usage=None,
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        llm = OpenAIWorkspaceLLM("o3", client=client)
        result = llm.complete_json(
            "return json",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            max_tokens=192,
            temperature=0.0,
        )
        self.assertEqual(result["choices"][0]["text"], "")

    def test_engine_converts_specialist_exceptions_to_malformed_candidates(self):
        class BrokenSpecialist:
            name = "rawlsian"

            def evaluate(self, scenario, actions, broadcast):
                raise RuntimeError("structured repair failed")

        result = WorkspaceEngine(
            [
                FixedSpecialist("care", "protect", "CARE"),
                BrokenSpecialist(),
                FixedSpecialist("virtue", "protect", "CHARACTER"),
            ],
            WorkspaceConfig(max_cycles=1, enable_synthesis=False),
        ).run("Choose protect or decline.", ["protect", "decline"])

        candidate = result.cycles[0].candidates[1]
        self.assertFalse(candidate.schema_valid)
        self.assertEqual(candidate.constraint, "NONE")
        self.assertEqual(candidate.delegate_status, "SEMANTIC_VALIDATION_ERROR")
        self.assertIn("specialist evaluation failed", candidate.validation_errors[0])

    def test_visibility_auditor_penalizes_endogenous_epistemic_exclusion_without_vote(self):
        scenario = (
            "The connected district reports casualties, while the off-grid settlement's "
            "signals are unreadable because it lacks communications infrastructure."
        )

        class VisibilityLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"off-grid residents",'
                    '"m":"infrastructure exclusion prevents casualty reporting",'
                    '"q":"off-grid settlement\'s signals are unreadable because it lacks communications infrastructure",'
                    '"p":{"A0":0.75,"A1":1.0}}'
                )}]}

        actions = ["Aid only the connected district", "Aid the off-grid settlement"]
        assessment = assess_visibility(VisibilityLlm(), scenario, actions)
        self.assertTrue(assessment.valid)
        self.assertTrue(assessment.activated)
        self.assertEqual(assessment.action_multipliers[actions[0]], 0.75)

        candidate = CandidateChunk(
            "utility", "UNCERTAINTY", {actions[0]: 0.9, actions[1]: 0.1},
            0.2, 0.8, 0.8, recommended_action=actions[0],
        )
        unadjusted = WorkspaceEngine._policy([candidate], actions)
        adjusted = WorkspaceEngine._policy(
            [candidate], actions, assessment.action_multipliers
        )
        self.assertGreater(unadjusted[actions[0]], adjusted[actions[0]])
        self.assertEqual(len(assessment.action_multipliers), 2)

    def test_visibility_auditor_cannot_penalize_merely_exogenous_uncertainty(self):
        class RandomFailureLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"lo":true,"en":false,"g":"remote group",'
                    '"m":"a random temporary sensor failure",'
                    '"q":"a random storm broke both sensors",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            RandomFailureLlm(),
            "A random storm broke both sensors.",
            ["choose first site", "choose second site"],
        )
        self.assertFalse(assessment.valid)
        self.assertFalse(assessment.activated)
        self.assertTrue(all(value == 1.0 for value in assessment.action_multipliers.values()))

    def test_visibility_auditor_can_use_structural_power_asymmetry_as_external_generalization(self):
        scenario = (
            "You are the lead structural inspector for a massive, decaying mining dam "
            "above an impoverished rural community. The conglomerate dominates the "
            "region's economy and politics, and if you issue the evacuation order it "
            "will orchestrate a retaliatory legal campaign."
        )

        class StructuralVisibilityLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"impoverished rural community",'
                    '"m":"corporate dominance and retaliatory legal pressure can suppress '
                    'reporting and institutional attention",'
                    '"q":"the conglomerate dominates the region\'s economy and politics",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            StructuralVisibilityLlm(),
            scenario,
            ["issue evacuation", "falsify report"],
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertTrue(assessment.activated)
        self.assertIn(assessment.mechanism_provenance, {"SCENARIO_GROUNDED", "EXTERNAL_GENERALIZATION"})
        self.assertLess(assessment.action_multipliers["issue evacuation"], 1.0)

    def test_visibility_proposition_receives_recurrent_workspace_access(self):
        seen = []

        class RecordingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                seen.append((self.name, broadcast.constraint, broadcast.contingency_question))
                chunk = super().evaluate(scenario, actions, broadcast)
                if broadcast.constraint == "VISIBILITY_AUDIT":
                    chunk.visibility_response = "QUALIFY"
                    chunk.visibility_harm_revision = "UPWARD"
                    chunk.visibility_justification = "missing toxicity warrants upward revision"
                return chunk

        actions = ["defoliate farms", "deploy gene drive"]
        visibility = VisibilityAssessment(
            True, True, "smallholder farmers",
            "missing telemetry suppresses local toxicity observations",
            "farmers lack digital telemetry",
            {actions[0]: 0.75, actions[1]: 1.0}, activated=True,
        )
        result = WorkspaceEngine(
            [
                RecordingSpecialist("care", actions[0], "CARE"),
                RecordingSpecialist("duty", actions[1], "RIGHTS"),
            ],
            WorkspaceConfig(max_cycles=2, min_valid_specialists=2, enable_synthesis=False),
        ).run(
            "Farmers lack telemetry while harms elsewhere are tracked.", actions,
            assess_visibility=lambda *_: visibility,
        )

        visibility_questions = [
            question for _, constraint, question in seen
            if constraint == "VISIBILITY_AUDIT"
        ]
        self.assertTrue(visibility_questions)
        self.assertIn("Proposition P", visibility_questions[0])
        self.assertIn("downward-biased", visibility_questions[0])
        self.assertTrue(any(
            decision.content_type == "VISIBILITY_AUDIT"
            for decision in result.access_decisions
        ))

    def test_delegate_records_visibility_harm_revision_without_forced_switch(self):
        data = {
            "scores": {"A0": 0.7, "A1": 0.3}, "r": "A0", "c": "CARE",
            "u": "VERIFY_FACTS", "w": "A0 remains preferable after revision",
            "j": "visibility evidence changes estimated harm", "e": "STATED_FACTS",
            "x": "NONE", "z": 0.7,
            "vp": "ACCEPT", "vj": "unobserved toxicity increases expected harm",
            "vh": "UPWARD",
        }
        candidate = _candidate_from_data(
            "care", ["A0 action", "A1 action"], data,
            WorkspaceBroadcast(constraint="VISIBILITY_AUDIT"), "NONE", {},
        )

        self.assertEqual(candidate.recommended_action, "A0 action")
        self.assertEqual(candidate.visibility_response, "ACCEPT")
        self.assertEqual(candidate.visibility_harm_revision, "UPWARD")
        self.assertIn("toxicity", candidate.visibility_justification)

    def test_shadow_stance_measurements_preserve_vote_and_scores(self):
        common = {
            "scores": {"A0": 0.7, "A1": 0.3}, "r": "A0", "c": "CARE",
            "u": "NONE", "w": "care favors protection", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
        }
        measured = {
            **common,
            "ss": "PROVISIONAL",
            "am": {"A0": "PERMISSIBLE", "A1": "REJECTED"},
            "cc": False, "esa": True, "ia": "A0",
            "wp": "ACCEPT", "we": "FACTUAL",
            "fa": "care applies the fact through vulnerability", "fr": True,
            "bd": "MEDIUM",
        }
        without_measurement = _candidate_from_data(
            "care", ["protect", "disclose"], common,
            WorkspaceBroadcast(constraint="CARE"), "NONE", {},
        )
        with_measurement = _candidate_from_data(
            "care", ["protect", "disclose"], measured,
            WorkspaceBroadcast(constraint="CARE"), "NONE", {},
        )
        self.assertEqual(with_measurement.action_scores, without_measurement.action_scores)
        self.assertEqual(with_measurement.recommended_action, without_measurement.recommended_action)
        self.assertEqual(with_measurement.epistemic_confidence, without_measurement.epistemic_confidence)
        self.assertEqual(with_measurement.selection_status, "PROVISIONAL")
        self.assertFalse(with_measurement.comparison_complete)
        self.assertEqual(with_measurement.framework_retention_status, "PRESERVED")

    def test_shadow_measurements_distinguish_rejection_from_unassessed(self):
        data = {
            "scores": {"A0": 0.6, "A1": 0.4}, "r": "A0", "c": "DUTY",
            "u": "VERIFY_FACTS", "w": "duty provisionally blocks harm", "j": "NONE",
            "e": "FRAMEWORK_ONLY", "x": "NONE", "z": 0.5,
            "ss": "UNSELECTED",
            "am": {"A0": "REJECTED", "A1": "UNASSESSED"},
            "cc": False, "esa": False, "ia": "NONE",
            "wp": "QUALIFY", "we": "NORMATIVE",
            "fa": "duty tests permissibility before endorsement", "fr": True,
            "bd": "LOW",
        }
        candidate = _candidate_from_data(
            "deontological", ["coerce", "refrain"], data,
            WorkspaceBroadcast(constraint="RIGHTS"), "NONE", {},
        )
        self.assertEqual(candidate.action_admissibility["coerce"], "REJECTED")
        self.assertEqual(candidate.action_admissibility["refrain"], "UNASSESSED")
        self.assertEqual(candidate.interim_action, "")
        self.assertFalse(candidate.evidence_sufficient_for_action)

    def test_drift_penalty_does_not_compare_review_context_to_base_context(self):
        class ContextLlm:
            calls = 0

            def __call__(self, prompt, **kwargs):
                self.calls += 1
                scores = '{"A0":0.65,"A1":0.35}' if self.calls == 1 else '{"A0":0.8,"A1":0.2}'
                visibility = (
                    ',"vp":"QUALIFY","vj":"hidden harm warrants some revision","vh":"UPWARD"'
                    if self.calls == 1 else ""
                )
                return {"choices": [{"text": (
                    f'{{"scores":{scores},"r":"A0","c":"CARE","u":"NONE",'
                    '"w":"care still favors A0","j":"NONE","e":"STATED_FACTS",'
                    f'"x":"NONE","z":0.8{visibility}}}'
                )}]}

        delegate = CompactLocalSpecialist("care", ContextLlm())
        delegate.evaluate(
            "A visibility conflict", ["A0 action", "A1 action"],
            WorkspaceBroadcast(
                constraint="VISIBILITY_AUDIT",
                contingency_question="Proposition P: hidden harm is underestimated.",
            ),
        )
        base = delegate.evaluate(
            "A visibility conflict", ["A0 action", "A1 action"],
            WorkspaceBroadcast(constraint="CARE", intent="evaluate_A0 action"),
        )

        self.assertAlmostEqual(base.preference_strength, 0.6)
        self.assertEqual(base.preference_drift, 0.0)
        self.assertEqual(base.preference_drift_penalty, 0.0)
        self.assertTrue(delegate.epistemic_commitments)

    def test_visibility_auditor_semantically_verifies_grounded_paraphrase(self):
        calls = []

        class SemanticVisibilityLlm:
            def __call__(self, prompt, **kwargs):
                calls.append(prompt)
                if "Verify grounding only" in prompt:
                    return {"choices": [{"text": (
                        '{"s":true,"r":"unregistered patients are excluded from outcome records"}'
                    )}]}
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"unregistered child",'
                    '"m":"administrative registration rules exclude the child from institutional visibility",'
                    '"q":"the child is omitted from hospital outcome data because registration is absent",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            SemanticVisibilityLlm(),
            "An unregistered child is excluded from the hospital outcome registry.",
            ["Rely on registered outcomes", "Treat the unregistered child"],
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertTrue(assessment.activated)
        # Explicit registry exclusion is compiled as a typed visibility fact, so
        # no second LLM pass is needed merely to approve a paraphrase.
        self.assertEqual(len(calls), 1)
        self.assertEqual(assessment.typed_facts[0]["telemetry_visibility"], 0.0)

    def test_visibility_semantic_verifier_rejects_unsupported_administrative_leap(self):
        class UnsupportedVisibilityLlm:
            def __call__(self, prompt, **kwargs):
                if "Verify grounding only" in prompt:
                    return {"choices": [{"text": (
                        '{"s":false,"r":"unregistered does not imply missing outcome data"}'
                    )}]}
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"unregistered child",'
                    '"m":"registration status makes the child invisible to clinicians",'
                    '"q":"the child cannot be observed because registration is absent",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            UnsupportedVisibilityLlm(),
            "An unregistered child has a ninety percent survival chance.",
            ["Treat registered patient", "Treat unregistered child"],
        )
        self.assertFalse(assessment.valid)
        self.assertFalse(assessment.activated)
        self.assertIn("semantic grounding", assessment.error)

    def test_visibility_hypothetical_mechanism_does_not_apply_penalty(self):
        class HypotheticalMechanismLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"rural residents",'
                    '"m":"preventive infrastructure failures are less visible and receive less institutional attention",'
                    '"q":"rural casualties under A0 remain unmonitored",'
                    '"p":{"A0":0.7,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            HypotheticalMechanismLlm(),
            "Rural casualties under A0 remain unmonitored; urban harms are live-tracked.",
            ["isolate the rural grid", "risk the urban system"],
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertFalse(assessment.activated)
        self.assertEqual(assessment.mechanism_provenance, "HYPOTHETICAL")
        self.assertTrue(all(value == 1.0 for value in assessment.action_multipliers.values()))

    def test_visibility_prompt_exposes_typed_facts_before_interpretation(self):
        prompts = []

        class PromptCaptureLlm:
            def __call__(self, prompt, **kwargs):
                prompts.append(prompt)
                return {"choices": [{"text": (
                    '{"lo":true,"en":true,"g":"rural residents",'
                    '"m":"rural casualties remain unmonitored",'
                    '"q":"rural casualties under A0 remain unmonitored",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assess_visibility(
            PromptCaptureLlm(),
            "Rural casualties under A0 remain unmonitored; urban harms are live-tracked.",
            ["isolate the rural grid", "risk the urban system"],
        )
        self.assertTrue(prompts)
        self.assertIn("Typed observability facts", prompts[0])
        self.assertIn("telemetry_visibility", prompts[0])

    def test_explicit_off_telemetry_language_compiles_to_typed_visibility_fact(self):
        calls = []

        class TypedVisibilityLlm:
            def __call__(self, prompt, **kwargs):
                calls.append(prompt)
                return {"choices": [{"text": (
                    '{"lo":false,"en":false,"g":"rural victims",'
                    '"m":"their harms are absent from the stated telemetry channel",'
                    '"q":"rural harms are missing from live records",'
                    '"p":{"A0":0.8,"A1":1.0}}'
                )}]}

        assessment = assess_visibility(
            TypedVisibilityLlm(),
            "Rural casualties under A0 remain unmonitored; urban harms are live-tracked.",
            ["isolate the rural grid", "risk the urban system"],
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertTrue(assessment.activated)
        self.assertEqual(len(calls), 1)  # no prose-grounding verifier round trip
        self.assertEqual(assessment.typed_facts[0]["telemetry_visibility"], 0.0)
        self.assertEqual(assessment.typed_facts[0]["relation"], "EXPLICIT_TELEMETRY_ABSENCE")
        self.assertEqual(assessment.typed_facts[0]["affected_action_node_id"], "A0")

        graph = compile_scenario_graph(
            "Rural casualties under A0 remain unmonitored; urban harms are live-tracked.",
            ["isolate the rural grid", "risk the urban system"],
        )
        target = graph.nodes["TARGET_VISIBILITY_0"]
        self.assertEqual(target.attributes["telemetry_visibility"], 0.0)
        bias_edges = [edge for edge in graph.edges if edge.relation == "BIASES_ESTIMATE"]
        self.assertEqual(len(bias_edges), 1)
        estimate = graph.nodes[bias_edges[0].target]
        self.assertEqual(estimate.attributes["bias_direction"], "DOWNWARD")
        self.assertTrue(any(
            edge.source == "A0" and edge.relation == "HAS_METRIC"
            for edge in graph.edges
        ))

    def test_planning_uses_canonical_action_node_and_allows_semantic_provenance(self):
        class GraphAddressedPlanningLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"f":0.55,"n":"the operator retains grid control",'
                    '"x":"the operator loses grid control","b":"A1",'
                    '"a":["operator controls execution"],"r":["twelve-hour window"],'
                    '"s":["TIME_ENERGY_BUDGET"],"m":true,'
                    '"g":"The stated deadline makes delayed execution infeasible",'
                    '"va":true,"vr":"the counter-code remains executable independently"}'
                )}]}

        assessment = analyze_action_plan(
            GraphAddressedPlanningLlm(),
            "Operators must choose a purge or counter-code within twelve hours.",
            ["purge the grid", "deploy counter-code"],
            "purge the grid", WorkspaceBroadcast(), [], "implementation uncertainty",
        )
        self.assertTrue(assessment.valid, assessment.error)
        self.assertEqual(assessment.target_action_node_id, "A0")
        self.assertEqual(assessment.fallback, "deploy counter-code")

    def test_engine_accepts_paraphrased_planning_provenance_by_action_node(self):
        actions = ["report the misconduct", "remain silent"]
        engine = WorkspaceEngine(
            [
                FixedSpecialist("care", actions[0], "CARE"),
                FixedSpecialist("duty", actions[1], "DUTY"),
            ],
            WorkspaceConfig(max_cycles=1, planning_entropy_threshold=0.0),
        )

        def plan(_scenario, _actions, selected, _broadcast, _candidates, reason):
            return PlanningAssessment(
                selected, reason, 0.45,
                "the reporting channel remains reachable",
                "the reporting deadline passes before submission",
                actions[1],
                strategic_forces=["TIME_ENERGY_BUDGET"],
                broadcast_worthy=True,
                # Semantically grounded, deliberately not a scenario substring.
                grounded_evidence="The available reporting interval is short",
                fallback_available=True,
                fallback_availability_reason="silence remains physically available",
                target_action_node_id="A0",
            )

        result = engine.run(
            "A witness has limited time to report employer misconduct.",
            actions,
            analyze_plan=plan,
        )
        self.assertTrue(result.planning_assessments[0].valid)
        self.assertEqual(result.planning_assessments[0].target_action_node_id, "A0")
        self.assertNotIn("exact grounded quote", result.planning_assessments[0].error)

    def test_engine_rejects_planning_assessment_bound_to_wrong_action_node(self):
        actions = ["report the misconduct", "remain silent"]
        assessment = PlanningAssessment(
            actions[0], "feasibility", 0.4,
            "the reporting channel remains reachable",
            "the limited reporting deadline passes",
            actions[1], broadcast_worthy=True,
            grounded_evidence="short reporting window",
            fallback_available=True,
            fallback_availability_reason="silence remains physically available",
            target_action_node_id="A1",
        )
        checked = WorkspaceEngine._validate_planning_assessment(
            assessment,
            "A witness has limited time to report employer misconduct.",
            WorkspaceBroadcast(),
            actions,
        )
        self.assertFalse(checked.valid)
        self.assertIn("canonical ActionNode", checked.error)

    def test_generic_danger_does_not_activate_planning_without_action_obstacle(self):
        scenario = (
            "Cascading grid failure during a storm risks many deaths. "
            "A0 sheds Sector 4; A1 cuts emergency dispatch."
        )
        actions = ["shed Sector 4", "cut emergency dispatch"]
        self.assertEqual(compile_execution_obstacles(scenario, actions), [])
        self.assertFalse(WorkspaceEngine._explicit_implementation_obstacle(
            scenario, WorkspaceBroadcast(), actions
        ))

    def test_execution_obstacle_must_link_constraint_to_action(self):
        scenario = (
            "A0 requires access to the Sector 4 breaker controls; "
            "A1 cuts emergency dispatch."
        )
        facts = compile_execution_obstacles(
            scenario, ["shed Sector 4", "cut emergency dispatch"]
        )
        self.assertEqual({fact.affected_action_node_id for fact in facts}, {"A0"})
        self.assertTrue(WorkspaceEngine._explicit_implementation_obstacle(
            scenario, WorkspaceBroadcast(), ["shed Sector 4", "cut emergency dispatch"],
            "shed Sector 4",
        ))
        self.assertFalse(WorkspaceEngine._explicit_implementation_obstacle(
            scenario, WorkspaceBroadcast(), ["shed Sector 4", "cut emergency dispatch"],
            "cut emergency dispatch",
        ))

    def test_symmetric_burden_facts_cover_both_actions(self):
        facts = compile_action_burdens(
            "A0 kills residents in Sector 4. A1 causes rural casualties.",
            ["shed Sector 4", "cut emergency dispatch"],
        )
        self.assertEqual(
            {fact.affected_action_node_id for fact in facts}, {"A0", "A1"}
        )

    def test_reversal_audit_includes_symmetric_burden_probe(self):
        actions = ["choose A0", "choose A1"]

        class RuleSpecialist(FixedSpecialist):
            def evaluate(self, scenario, current_actions, broadcast):
                chunk = super().evaluate(scenario, current_actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.decision_rule = f"prefer {self.preferred} under its governing reason"
                chunk.factual_reversal_threshold = "deaths from this option exceed ten"
                rival = next(action for action in current_actions if action != self.preferred)
                chunk.graph_update_proposal = typed_reversal_proposal(
                    self.preferred, rival
                )
                return chunk

        result = WorkspaceEngine(
            [
                RuleSpecialist("care", actions[0], "CARE"),
                RuleSpecialist("virtue", actions[0], "CHARACTER"),
                RuleSpecialist("duty", actions[1], "DUTY"),
            ],
            WorkspaceConfig(max_cycles=2, enable_synthesis=False),
        ).run(
            "A0 kills residents in one district. A1 causes casualties in another district.",
            actions,
        )
        audit = next(
            decision for decision in result.access_decisions
            if decision.content_type == "REVERSAL_AUDIT"
        )
        self.assertIn("symmetric_burden_substitution", audit.signals)
        self.assertIn("transfer a comparable burden", audit.question)

    def test_visibility_unknown_magnitude_scrubs_unsupported_ceiling(self):
        data = {
            "scores": {"A0": 0.7, "A1": 0.3}, "r": "A0", "c": "CARE",
            "u": "NONE", "w": "care still favors A0", "j": "NONE",
            "e": "STATED_FACTS", "x": "NONE", "z": 0.7,
            "vp": "QUALIFY", "vh": "UPWARD", "vm": "UNKNOWN",
            "vj": "Hidden deaths are unlikely to approach the stated threshold",
        }
        candidate = _candidate_from_data(
            "care", ["A0 action", "A1 action"], data,
            WorkspaceBroadcast(constraint="VISIBILITY_AUDIT"), "NONE", {},
        )
        self.assertEqual(candidate.visibility_magnitude_status, "UNKNOWN")
        self.assertTrue(candidate.visibility_magnitude_overreach)
        self.assertIn("magnitude remains unknown", candidate.visibility_justification)

    def test_landscape_comparison_of_outcomes_is_not_action_inversion(self):
        class TrolleyLandscapeLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.8,"A1":0.2},"r":"A0","c":"IMMINENT_HARM",'
                    '"u":"NONE","w":"minimizes irreversible deaths","j":"NONE",'
                    '"e":"STATED_FACTS","x":"NONE",'
                    '"l":{"A0":"Kills one and saves five, minimizing deaths",'
                    '"A1":"Avoids direct killing but five lives are lost"},'
                    '"da":"number killed versus direct agency",'
                    '"t":"minimize certain loss of life","tf":"direct agency remains morally disputed"}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", TrolleyLandscapeLlm()).evaluate(
            "A trolley kills five unless redirected toward one.",
            [
                "Pull the lever, redirecting the trolley toward one person",
                "Do not pull the lever, allowing the trolley to continue toward five people",
            ],
            WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.landscape_search_complete)
        self.assertTrue(chunk.landscape_semantic_valid, chunk.landscape_validation_errors)

    def test_grounded_minority_gets_bonus_but_randomized_contrarian_does_not(self):
        class LandscapeSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint, tiebreaker):
                super().__init__(name, preferred, constraint)
                self.tiebreaker = tiebreaker

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.landscape_cases = {action: f"case for {action}" for action in actions}
                chunk.landscape_search_complete = True
                chunk.landscape_semantic_valid = True
                chunk.landscape_tiebreaker = self.tiebreaker
                return chunk

        grounded = LandscapeSpecialist("critic", "decline", "DUTY", "rights constraint")
        randomized = LandscapeSpecialist("randomizer", "decline", "FAIRNESS", "fair coin flip")
        engine = WorkspaceEngine(
            [
                LandscapeSpecialist("care", "protect", "CARE", "dependency priority"),
                LandscapeSpecialist("virtue", "protect", "CHARACTER", "practical wisdom"),
                LandscapeSpecialist("utility", "protect", "IMMINENT_HARM", "harm reduction"),
                grounded,
                randomized,
            ],
            WorkspaceConfig(max_cycles=1),
        )
        result = engine.run("Choose protect or decline.", ["protect", "decline"])
        candidates = {candidate.specialist: candidate for candidate in result.cycles[0].candidates}
        self.assertEqual(candidates["critic"].independence_bonus, 1.0)
        self.assertEqual(candidates["randomizer"].independence_bonus, 0.0)

    def test_reason_cannot_claim_wrong_recipient_has_higher_survival(self):
        class ContradictingLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.2,"A1":0.8},"r":"A1",'
                    '"c":"IMMINENT_HARM","u":"NONE",'
                    '"w":"researcher has higher survival chance"}'
                )}]}

        delegate = CompactLocalSpecialist(
            "utilitarian",
            ContradictingLlm(),
            baseline_action_id="NONE",
            scenario_facts={"survival_chance": {"child": 0.8, "researcher": 0.45}},
        )
        chunk = delegate.evaluate(
            "A child has 80%; researcher 45%.",
            ["Give antidote to child", "Give antidote to researcher"],
            WorkspaceBroadcast(),
        )
        self.assertFalse(chunk.schema_valid)

    def test_delegate_separates_decision_rule_and_reversal_threshold_types(self):
        class RuleLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"scores":{"A0":0.7,"A1":0.3},"r":"A0",'
                    '"c":"IMMINENT_HARM","u":"NONE","w":"expected lives favor rescue",'
                    '"j":"NONE","e":"STATED_FACTS","x":"NONE",'
                    '"l":{"A0":"Saves more expected lives","A1":"Guarantees one life"},'
                    '"da":"expected lives versus certainty","t":"maximize expected lives",'
                    '"tf":"certainty remains morally relevant",'
                    '"dr":"prefer A0 when expected lives exceed the certain rescue",'
                    '"ft":"A0 expected lives fall below one",'
                    '"nt":"certainty is morally overriding","z":0.8}'
                )}]}

        chunk = CompactLocalSpecialist("utilitarian", RuleLlm()).evaluate(
            "Choose a larger probabilistic rescue or one certain rescue.",
            ["probabilistic rescue", "certain rescue"], WorkspaceBroadcast(),
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertIn("expected lives", chunk.decision_rule)
        self.assertEqual(chunk.factual_reversal_threshold, "A0 expected lives fall below one")
        self.assertEqual(chunk.normative_reversal_threshold, "certainty is morally overriding")

    def test_grounded_dissent_triggers_one_adversarial_reversal_cycle(self):
        class RuleSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint, factual, normative):
                super().__init__(name, preferred, constraint)
                self.factual = factual
                self.normative = normative

            def evaluate(self, scenario, actions, broadcast):
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.decision_rule = f"prefer {self.preferred} when its governing reason prevails"
                chunk.factual_reversal_threshold = self.factual
                chunk.normative_reversal_threshold = self.normative
                rival = next(action for action in actions if action != self.preferred)
                chunk.graph_update_proposal = typed_reversal_proposal(
                    self.preferred, rival
                )
                if broadcast.constraint == "REVERSAL_AUDIT":
                    chunk.reversal_review_response = "REVISE"
                    chunk.reversal_review_justification = "the critic identifies a real boundary"
                    chunk.revised_reversal_condition = "the competing harm becomes materially greater"
                return chunk

        engine = WorkspaceEngine(
            [
                RuleSpecialist("care", "protect", "CARE", "support becomes unavailable", "NONE"),
                RuleSpecialist("virtue", "protect", "CHARACTER", "support becomes unavailable", "NONE"),
                RuleSpecialist("critic", "decline", "DUTY", "protecting causes greater harm", "duty is absolute"),
            ],
            WorkspaceConfig(max_cycles=2, enable_synthesis=False),
        )
        result = engine.run("Choose protect or decline.", ["protect", "decline"])
        self.assertEqual(result.cycles[1].received_broadcast.constraint, "REVERSAL_AUDIT")
        self.assertEqual(
            sum(decision.content_type == "REVERSAL_AUDIT" for decision in result.access_decisions),
            1,
        )
        self.assertTrue(all(
            candidate.reversal_review_response == "REVISE"
            for candidate in result.cycles[1].candidates
        ))
        self.assertTrue(result.cycles[1].is_hypothetical)

    def test_reversal_probe_restores_base_position_state_and_memory_count(self):
        observations = []

        class StatefulRuleSpecialist(FixedSpecialist):
            def __init__(self, name, preferred, constraint):
                super().__init__(name, preferred, constraint)
                self.previous_recommendation_id = "BASE"
                self.previous_confidence = 0.6
                self.assumption_status = "SUPPORTED"
                self.unsupported_assumption = "base facts"
                self.reversal_condition = "base reversal"

            def evaluate(self, scenario, actions, broadcast):
                observations.append((self.name, broadcast.constraint, self.previous_recommendation_id))
                chunk = super().evaluate(scenario, actions, broadcast)
                chunk.recommended_action = self.preferred
                chunk.decision_rule = f"prefer {self.preferred} when its reason governs"
                chunk.factual_reversal_threshold = "the competing harm becomes greater"
                chunk.normative_reversal_threshold = "the competing duty becomes overriding"
                rival = next(action for action in actions if action != self.preferred)
                chunk.graph_update_proposal = typed_reversal_proposal(
                    self.preferred, rival
                )
                if broadcast.constraint == "REVERSAL_AUDIT":
                    self.previous_recommendation_id = "COUNTERFACTUAL"
                    self.previous_confidence = 0.1
                    chunk.reversal_review_response = "ACCEPT"
                    chunk.reversal_review_justification = "the proposed boundary changes the ranking"
                return chunk

        specialists = [
            StatefulRuleSpecialist("care", "protect", "CARE"),
            StatefulRuleSpecialist("virtue", "protect", "CHARACTER"),
            StatefulRuleSpecialist("critic", "decline", "DUTY"),
        ]
        result = WorkspaceEngine(
            specialists, WorkspaceConfig(max_cycles=2, enable_synthesis=False)
        ).run("Choose protect or decline.", ["protect", "decline"])

        self.assertEqual([cycle.is_hypothetical for cycle in result.cycles], [False, True, False])
        post_probe = [item for item in observations if item[1] != "REVERSAL_AUDIT"][-3:]
        self.assertTrue(all(state == "BASE" for _, _, state in post_probe))
        self.assertTrue(all(
            specialist.previous_recommendation_id == "BASE"
            and specialist.previous_confidence == 0.6
            for specialist in specialists
        ))
        contribution = summarize_specialist_contributions(result.cycles)
        self.assertTrue(all(record["responses"] == 2 for record in contribution.values()))
        answer = render_public_judgment(result)
        # Reversal-review plumbing is not a default brief section; accepted
        # challenge content may still appear as a change condition when justified.
        self.assertNotIn("Adversarial reversal review", answer)
        self.assertIn("# Ethical Parliament Judgment", answer)


class AuditDirectInvertTests(unittest.TestCase):
    actions = (
        "maintain grid, let emergency wing fail",
        "divert power, sacrifice research unit",
    )
    audit_variable = {
        "entity": "priority between justice and compassion",
        "relation": "RESOLVE_NORMATIVE_TENSION",
        "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
        "focus_action": "A0",
        "question": (
            "Would resolving the remaining justice versus compassion tension "
            "leave, weaken, or reverse the recommendation for maintaining the grid?"
        ),
    }

    def _payload(
        self,
        *,
        recommended: str,
        scores: dict[str, float],
        table: dict[str, list[dict[str, str]]],
        ev: dict[str, dict[str, object]] | None = None,
        justification: str = "Corrected expected-death arithmetic favours A0",
    ) -> dict[str, object]:
        return {
            "scores": scores,
            "r": recommended,
            "c": "IMMINENT_HARM",
            "u": "NONE",
            "w": "Fewer expected deaths overall",
            "j": justification,
            "e": "STATED_FACTS",
            "x": "NONE",
            "l": {
                "A0": "Keeps trial patients from being killed",
                "A1": "Saves emergency residents from freeze",
            },
            "da": "expected deaths versus certain killing",
            "t": "minimize expected deaths",
            "tf": "NONE",
            "dr": f"prefer {recommended} when its expected deaths are lower",
            "ft": "NONE",
            "nt": "NONE",
            "z": 0.7,
            "ct": table,
            "cd": False,
            "cm": "",
            "ev": ev or {
                "A0": {"value": 5.0, "unit": "DEATHS", "direction": "HARM", "grounded": True},
                "A1": {"value": 0.001, "unit": "DEATHS", "direction": "HARM", "grounded": True},
            },
            "ap": "RELEVANT",
            "ax": "Expected-death ranking is the utilitarian test of this audit.",
            "d": "SUPPORTED",
            "a": "Utilitarian ranking depends on expected lives",
            "v": "If expected deaths for A1 rose above A0 the ranking would reverse",
            "ie": "NO_CHANGE",
            "av": dict(self.audit_variable),
        }

    def _trace_table(self) -> dict[str, list[dict[str, str]]]:
        return {
            "A0": [
                {
                    "o": "grid failure kills residents", "s": "emergency residents",
                    "d": "HARM", "p": "1", "m": "large", "h": "immediate",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                },
                {
                    "o": "trial data preserved", "s": "future patients",
                    "d": "BENEFIT", "p": "0.01", "m": "10 lives (0.001 EV)",
                    "h": "permanent", "rv": "UNKNOWN", "g": "STATED",
                },
            ],
            "A1": [
                {
                    "o": "power diversion kills patients", "s": "research patients",
                    "d": "HARM", "p": "1", "m": "large", "h": "immediate",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                },
                {
                    "o": "life-support stabilised", "s": "emergency residents",
                    "d": "BENEFIT", "p": "1", "m": "large", "h": "immediate",
                    "rv": "IRREVERSIBLE", "g": "STATED",
                },
            ],
        }

    def test_audit_rejects_direct_invert_when_ledger_still_ranks_baseline(self):
        with self.assertRaisesRegex(ValueError, "audit cycles may not invert"):
            _candidate_from_data(
                "utilitarian",
                self.actions,
                self._payload(
                    recommended="A0",
                    scores={"A0": 0.85, "A1": 0.15},
                    table=self._trace_table(),
                ),
                WorkspaceBroadcast(
                    constraint="PROBLEM_STATE_AUDIT",
                    unresolved="VERIFY_ASSUMPTIONS",
                    audit_variable=dict(self.audit_variable),
                ),
                "A1",
                {},
                previous_recommendation_id="A1",
                baseline_status="DIRECT",
            )

    def test_audit_allows_direct_invert_when_ledger_ranks_new_action(self):
        table = {
            "A0": [{
                "o": "keeps five residents alive", "s": "emergency residents",
                "d": "BENEFIT", "p": "1", "m": "5 lives", "h": "immediate",
                "rv": "IRREVERSIBLE", "g": "STATED",
            }],
            "A1": [{
                "o": "kills five residents", "s": "emergency residents",
                "d": "HARM", "p": "1", "m": "5 lives", "h": "immediate",
                "rv": "IRREVERSIBLE", "g": "STATED",
            }],
        }
        chunk = _candidate_from_data(
            "utilitarian",
            self.actions,
            self._payload(
                recommended="A0",
                scores={"A0": 0.85, "A1": 0.15},
                table=table,
                ev={
                    "A0": {"value": 0.0, "unit": "DEATHS", "direction": "HARM", "grounded": True},
                    "A1": {"value": 5.0, "unit": "DEATHS", "direction": "HARM", "grounded": True},
                },
                justification="Ledger now shows A0 prevents the larger expected harm",
            ),
            WorkspaceBroadcast(
                constraint="PROBLEM_STATE_AUDIT",
                unresolved="VERIFY_ASSUMPTIONS",
                audit_variable=dict(self.audit_variable),
            ),
            "A1",
            {},
            previous_recommendation_id="A1",
            baseline_status="DIRECT",
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertEqual(chunk.recommended_action, self.actions[0])

    def test_open_deliberation_still_rejects_unjustified_direct_invert(self):
        with self.assertRaisesRegex(ValueError, "initial recommendation must match"):
            _candidate_from_data(
                "utilitarian",
                self.actions,
                self._payload(
                    recommended="A0",
                    scores={"A0": 0.85, "A1": 0.15},
                    table=self._trace_table(),
                ),
                WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
                "A1",
                {},
                baseline_status="DIRECT",
            )


class DeontologyAuditVoteTests(unittest.TestCase):
    """Cycle-3 o3 bug: a conditional audit must not drop a valid duty ranking."""

    actions = (
        "Keep the grid as is, letting the emergency wing fail to preserve the research unit and patients",
        "Divert power to life-support in the emergency wing, destroying the research unit",
    )
    audit_variable = {
        "entity": "priority between justice and compassion",
        "relation": "RESOLVE_NORMATIVE_TENSION",
        "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
        "focus_action": "A0",
        "question": (
            "Would resolving the remaining justice versus compassion tension "
            "leave, weaken, or reverse the recommendation for maintaining the grid?"
        ),
    }

    def _broadcast(self) -> WorkspaceBroadcast:
        return WorkspaceBroadcast(
            constraint="PROBLEM_STATE_AUDIT",
            unresolved="VERIFY_ASSUMPTIONS",
            audit_variable=dict(self.audit_variable),
        )

    def _payload(self, **overrides: object) -> dict[str, object]:
        data: dict[str, object] = {
            "scores": {"A0": 0.76, "A1": 0.24},
            "r": "A0",
            "c": "DUTY",
            "u": "NONE",
            "w": "perfect duties override imperfect rescue",
            "j": "NONE",
            "e": "FRAMEWORK_ONLY",
            "x": "NONE",
            "l": {
                "A0": "Required: do not kill the research patients",
                "A1": "Prohibited: uses patients merely as means",
            },
            "da": "perfect duty versus imperfect rescue",
            "t": "perfect duties override imperfect duties",
            "tf": "NONE",
            "dr": "prefer A0 when killing innocents remains prohibited",
            "ft": "NONE",
            "nt": "If imperfect rescue gained priority over perfect non-maleficence",
            "z": 0.7,
            "fm": {
                "A0": "REQUIRED: refrain from intentional killing of research patients",
                "A1": "PROHIBITED: instrumentalizes non-consenting patients to rescue others",
            },
            "nr": "IRRELEVANT",
            "np": "counts of lives do not override the perfect duty",
            "ap": "RELEVANT",
            "ax": "Rescue remains an imperfect duty beside non-maleficence",
            "d": "CONDITIONAL",
            "a": "The audited tension does not defeat the perfect duty",
            "v": "If imperfect rescue gained priority the ranking would reverse",
            "ie": "NO_CHANGE",
            "av": dict(self.audit_variable),
        }
        data.update(overrides)
        return data

    def test_conditional_audit_keeps_deontological_vote_and_marks_uncertainty(self):
        chunk = _candidate_from_data(
            "deontological",
            self.actions,
            self._payload(),
            self._broadcast(),
            "A0",
            {},
            previous_recommendation_id="A0",
            baseline_status="DIRECT",
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertEqual(chunk.recommended_action, self.actions[0])
        self.assertEqual(chunk.unresolved, "VERIFY_FACTS")
        self.assertEqual(chunk.assumption_status, "CONDITIONAL")
        self.assertEqual(chunk.audit_participation, "RELEVANT")

    def test_short_audit_explanation_does_not_drop_deontological_vote(self):
        chunk = _candidate_from_data(
            "deontological",
            self.actions,
            self._payload(ax="no", fa="NONE", w="duty ranking"),
            self._broadcast(),
            "A0",
            {},
            previous_recommendation_id="A0",
            baseline_status="DIRECT",
        )
        self.assertTrue(chunk.schema_valid, chunk.validation_errors)
        self.assertEqual(chunk.recommended_action, self.actions[0])
        self.assertEqual(chunk.audit_participation, "NOT_TESTED")


class ScenarioClausePartyGroundingTests(unittest.TestCase):
    """A party the scenario names as affected is not always an action target."""

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs to exonerate "
        "the innocent technician while the residents continue to freeze."
    )
    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )

    def _assessment(self, action_id: str, protected_party: str) -> dict[str, object]:
        return {
            "action_id": action_id, "verdict": "CONFLICTED",
            "norm_kind": "RIGHT", "norm": "do not frame the innocent",
            "relation": "CONFLICTS", "duty_bearer": "operator",
            "protected_party": protected_party,
            "competing_norm": "duty to rescue",
            "competing_norm_kind": "DUTY", "competing_relation": "CONFLICTS",
            "competing_protected_party": "residents",
            "competing_reason": "both claims remain live",
            "governing_norm": "UNRESOLVED", "priority_basis": "UNRESOLVED",
            "priority_rule": "the ranking remains unresolved",
            "protected_standing": "EQUAL_JURIDICAL_STATUS",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE", "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "no coercion requires authorization",
            "reciprocity_status": "UNKNOWN", "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNRESOLVED", "resolution_status": "CONTESTED",
            "evidence_basis": "SCENARIO", "reason": "the duties conflict",
        }

    def _graph(self) -> SemanticGraph:
        clauses = segment_scenario_clauses(self.scenario)
        return compile_scenario_graph(self.scenario, list(self.actions), {
            "A0": {"clauses": [clauses[0]]},
            "A1": {"clauses": [clauses[1]]},
        })

    def _apply(self, protected_party: str):
        store = SemanticGraphStore(self._graph())
        record = apply_deontological_ledger_transaction(
            store,
            {"assessments": [
                self._assessment("A0", protected_party),
                self._assessment("A1", protected_party),
            ]},
            cycle=1, specialist="deontological",
            allowed_actions=tuple(self.actions),
        )
        return store, record

    def test_party_named_only_in_the_cited_clause_grounds_the_duty(self):
        _store, record = self._apply("innocent technician")

        self.assertNotIn(
            "A0 primary protected party lacks current-run action grounding",
            record.errors,
        )
        committed = record.proposal["committed_assessments"]
        self.assertTrue(all(
            item["epistemic_status"] != "UNRESOLVED_PARTY" for item in committed
        ))

    def test_party_absent_from_every_cited_clause_stays_ungrounded(self):
        _store, record = self._apply("offshore shareholders")

        self.assertTrue(any(
            "primary protected party lacks current-run action grounding" in error
            for error in record.errors
        ))

    def test_clause_grounding_requires_every_substantive_word(self):
        graph = self._graph()

        matched, basis = _party_grounding(graph, "A0", "innocent technician")
        unmatched, missing_basis = _party_grounding(graph, "A0", "innocent bystander")

        self.assertTrue(matched)
        self.assertEqual(basis, "SCENARIO_CLAUSE")
        self.assertEqual(unmatched, [])
        self.assertEqual(missing_basis, "NONE")


class UtilitarianDirectionGroundingTests(unittest.TestCase):
    """A mixed-effect action must not read as contradicting its own ledger."""

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs to exonerate "
        "the innocent technician while the residents continue to freeze."
    )
    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )

    def _row(self, outcome: str, scope: str, direction: str) -> dict[str, object]:
        return {
            "outcome": outcome, "scope": scope, "direction": direction,
            "probability": "LIKELY", "magnitude": "SEVERE",
            "duration": "LASTING", "reversibility": "IRREVERSIBLE",
            "support": "STATED",
        }

    def _graph(self) -> SemanticGraph:
        clauses = segment_scenario_clauses(self.scenario)
        return compile_scenario_graph(self.scenario, list(self.actions), {
            "A0": {"clauses": [clauses[0]]},
            "A1": {"clauses": [clauses[1]]},
        })

    def _committed(self, rows: dict[str, list[dict[str, object]]]):
        store = SemanticGraphStore(self._graph())
        record = apply_utilitarian_ledger_transaction(
            store,
            {"actions": [
                {"action_id": action_id, "consequences": rows[action_id]}
                for action_id in ("A0", "A1")
            ]},
            cycle=1, specialist="utilitarian",
            allowed_actions=tuple(self.actions),
        )
        return record

    def test_benefit_row_grounds_on_the_agreeing_effect_of_a_mixed_action(self):
        record = self._committed({
            "A0": [self._row("immediate survival", "8 freezing residents", "BENEFIT")],
            "A1": [self._row("residents keep freezing", "8 freezing residents", "HARM")],
        })

        committed = {
            item["canonical_action_id"]: item
            for item in record.proposal["committed_consequences"]
        }
        self.assertEqual(committed["A0"]["direction"], "BENEFIT")
        self.assertNotEqual(committed["A0"]["epistemic_status"], "CONTRADICTED_DIRECTION")
        self.assertFalse(any(
            "direction contradicts its grounded action consequence" in warning
            for warning in record.errors
        ))

    def test_direction_lock_still_fires_on_the_same_affected_subject(self):
        # A1 only ever worsens the residents' position, so claiming a benefit
        # for them still contradicts the grounded fact.
        record = self._committed({
            "A0": [self._row("immediate survival", "8 freezing residents", "BENEFIT")],
            "A1": [self._row("residents are kept warm", "8 freezing residents", "BENEFIT")],
        })

        committed = {
            item["canonical_action_id"]: item
            for item in record.proposal["committed_consequences"]
        }
        self.assertEqual(committed["A1"]["epistemic_status"], "CONTRADICTED_DIRECTION")
        self.assertEqual(committed["A1"]["direction"], "UNKNOWN")
        self.assertEqual(committed["A1"]["proposed_direction"], "BENEFIT")

    def test_claim_about_an_uninvolved_subject_reports_no_evidence(self):
        evidence, score = _best_evidence(
            self._graph(), "A0", "share price improves", "offshore shareholders",
            expected_polarity="BENEFICIAL",
        )

        self.assertIsNone(evidence)
        self.assertEqual(score, 0)


class RawlsMapReconciliationTests(unittest.TestCase):
    """One explanatory line per action cannot encode a multi-position ledger."""

    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )
    action_ids = ("A0", "A1")
    action_map = {
        actions[0]: (
            "WORSENS: the innocent technician's basic liberty is compromised "
            "even though residents regain heat"
        ),
        actions[1]: (
            "PRESERVES: the technician's basic liberty and public "
            "accountability hold while residents lose heat"
        ),
    }
    positions = {
        "A0": {
            "e": "WORSENS", "d": "BASIC_LIBERTY", "ca": "A1", "b": "SCENARIO",
            "s": "innocent technician", "sk": "INDIVIDUAL",
            "rs": "framed for sabotage by the erased record",
            "ad": [{
                "e": "IMPROVES", "d": "BASIC_INTEREST_SECURITY", "ca": "A1",
                "b": "SCENARIO", "s": "8 freezing residents", "sk": "GROUP",
                "rs": "heat restored to the district",
            }],
        },
        "A1": {
            "e": "PRESERVES", "d": "BASIC_LIBERTY", "ca": "A0", "b": "SCENARIO",
            "s": "innocent technician", "sk": "INDIVIDUAL",
            "rs": "the audit record keeps the technician exonerated",
            "ad": [{
                "e": "WORSENS", "d": "BASIC_INTEREST_SECURITY", "ca": "A0",
                "b": "SCENARIO", "s": "8 freezing residents", "sk": "GROUP",
                "rs": "the district stays without heat",
            }],
        },
    }

    def _errors(self, **overrides: object) -> list[str]:
        return _construct_map_errors(
            "rawlsian",
            list(self.actions),
            dict(self.action_map),
            str(overrides.get("numerical_role", "SECONDARY")),
            "the worst-off subject is the innocent technician's basic liberty",
            action_ids=list(self.action_ids),
            recommended_action=str(
                overrides.get("recommended_action", self.actions[1])
            ),
            rationale=str(overrides.get("rationale", "Basic liberty outweighs lives")),
            supporting_data={"rp": overrides.get("rp", self.positions)},
        )

    def test_typed_liberty_priority_supplies_the_stated_difference(self):
        self.assertEqual(self._errors(), [])

    def test_prose_alone_cannot_justify_selecting_the_impaired_position(self):
        errors = self._errors(recommended_action=self.actions[0])

        self.assertIn(
            "Rawlsian preference lacks a stated difference in position or "
            "principle priority",
            errors,
        )

    def test_map_line_matching_one_of_several_positions_is_not_a_conflict(self):
        chunk = _candidate_from_data(
            "rawlsian",
            self.actions,
            {
                "scores": {"A0": 0.3, "A1": 0.7},
                "r": "A1", "c": "RIGHTS", "u": "NONE",
                "w": "basic liberty holds lexical priority over welfare gains",
                "j": "NONE", "e": "STATED_FACTS", "x": "NONE",
                "l": {
                    "A0": "Worsens the technician's basic liberty",
                    "A1": "Preserves the technician's basic liberty",
                },
                "t": "basic liberty versus interest security",
                "tf": "NONE",
                "dr": "prefer A1 while basic liberty stays lexically prior",
                "ft": "NONE",
                "nt": "If basic liberty lost lexical priority",
                "z": 0.7,
                "fm": {
                    "A0": self.action_map[self.actions[0]],
                    "A1": self.action_map[self.actions[1]],
                },
                "nr": "SECONDARY",
                "np": "the worst-off subject is the innocent technician",
                "rb": "LEXICAL_BASIC_LIBERTY",
                "lc": {"A0": "INFRINGED", "A1": "SATISFIED"},
                "rp": self.positions,
            },
            WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            "A1",
            {},
        )

        self.assertEqual(chunk.framework_validation_errors, [])
        self.assertEqual(
            len(chunk.rawls_position_proposal["positions"]), 4,
            "both actions must contribute a primary and an additional position",
        )

    def test_map_line_contradicting_every_position_still_fails(self):
        inverted = {
            "A0": {**self.positions["A0"], "e": "IMPROVES", "ad": []},
            "A1": {**self.positions["A1"], "e": "PRESERVES", "ad": []},
        }
        chunk = _candidate_from_data(
            "rawlsian",
            self.actions,
            {
                "scores": {"A0": 0.3, "A1": 0.7},
                "r": "A1", "c": "RIGHTS", "u": "NONE",
                "w": "basic liberty holds lexical priority over welfare gains",
                "j": "NONE", "e": "STATED_FACTS", "x": "NONE",
                "l": {
                    "A0": "Worsens the technician's basic liberty",
                    "A1": "Preserves the technician's basic liberty",
                },
                "t": "basic liberty versus interest security",
                "tf": "NONE",
                "dr": "prefer A1 while basic liberty stays lexically prior",
                "ft": "NONE",
                "nt": "If basic liberty lost lexical priority",
                "z": 0.7,
                "fm": {
                    "A0": self.action_map[self.actions[0]],
                    "A1": self.action_map[self.actions[1]],
                },
                "nr": "SECONDARY",
                "np": "the worst-off subject is the innocent technician",
                "rb": "LEXICAL_BASIC_LIBERTY",
                "lc": {"A0": "INFRINGED", "A1": "SATISFIED"},
                "rp": inverted,
            },
            WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
            "A1",
            {},
        )

        self.assertIn(
            "Rawlsian position for A0 conflicts with its framework map",
            chunk.framework_validation_errors,
        )
        self.assertNotIn(
            "Rawlsian position for A1 conflicts with its framework map",
            chunk.framework_validation_errors,
        )


class VirtueRoleRedescriptionTests(unittest.TestCase):
    """Renaming the actor's role is not a change of character verdict."""

    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )

    def _specialist(self) -> CompactLocalSpecialist:
        specialist = CompactLocalSpecialist("virtue", llm=None)
        specialist.previous_framework_state = {
            "ranking_basis": "PRACTICAL_WISDOM",
            "assessments": [
                {
                    "action_id": "A0", "verdict": "UNDERMINES",
                    "actor_role": "public steward",
                },
                {
                    "action_id": "A1", "verdict": "EXEMPLIFIES",
                    "actor_role": "public steward",
                },
            ],
        }
        return specialist

    def _candidate(self, assessments: list[dict[str, object]]) -> CandidateChunk:
        return CandidateChunk(
            specialist="virtue", constraint="CHARACTER",
            action_scores={self.actions[0]: 0.3, self.actions[1]: 0.7},
            surprise=0.1, friction=0.1, confidence=0.6,
            recommended_action=self.actions[1],
            rationale="Honesty and practical wisdom favour keeping the record.",
            virtue_character_proposal={
                "ranking_basis": "PRACTICAL_WISDOM",
                "assessments": assessments,
            },
        )

    def test_role_relabel_with_stable_verdicts_is_a_refinement(self):
        specialist = self._specialist()
        candidate = self._candidate([
            {"action_id": "A0", "verdict": "UNDERMINES", "actor_role": "chief engineer"},
            {"action_id": "A1", "verdict": "EXEMPLIFIES", "actor_role": "chief engineer"},
        ])

        specialist._audit_framework_state_change(
            candidate, WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
        )

        self.assertTrue(candidate.framework_constraint_retained)
        self.assertEqual(candidate.framework_retention_status, "REFINEMENT")
        self.assertTrue(candidate.framework_insights)
        self.assertTrue(any(
            "chief engineer" in item["proposition"]
            for item in candidate.framework_insights
        ))

    def test_verdict_change_still_requires_a_framework_reason(self):
        specialist = self._specialist()
        candidate = self._candidate([
            {"action_id": "A0", "verdict": "EXEMPLIFIES", "actor_role": "public steward"},
            {"action_id": "A1", "verdict": "UNDERMINES", "actor_role": "public steward"},
        ])

        specialist._audit_framework_state_change(
            candidate, WorkspaceBroadcast(constraint="OPEN_DELIBERATION"),
        )

        self.assertNotEqual(candidate.framework_retention_status, "REFINEMENT")


class ResolvedQuestionLedgerTests(unittest.TestCase):
    """Settled audit answers are graph state, and reopen when evidence moves."""

    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )
    question_key = "QUESTION:abc123"
    proposition = (
        "Whether the relative magnitude of the technician's liberty loss "
        "outweighs the residents' survival"
    )

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs to exonerate "
        "the innocent technician while the residents continue to freeze."
    )

    def _store(self) -> SemanticGraphStore:
        clauses = segment_scenario_clauses(self.scenario)
        return SemanticGraphStore(compile_scenario_graph(
            self.scenario, list(self.actions), {
                "A0": {"clauses": [clauses[0]]},
                "A1": {"clauses": [clauses[1]]},
            },
        ))

    def _responder(
        self, specialist: str, *, participation: str, effect: str,
    ) -> CandidateChunk:
        return CandidateChunk(
            specialist=specialist, constraint="OPEN_DELIBERATION",
            action_scores={self.actions[0]: 0.4, self.actions[1]: 0.6},
            surprise=0.1, friction=0.1, confidence=0.6,
            recommended_action=self.actions[1],
            rationale="The audited issue does not move the ranking.",
            audit_participation=participation,
            audit_internal_effect=effect,
        )

    def _resolve(self, store: SemanticGraphStore, candidates: list[CandidateChunk]):
        return resolve_audited_question(
            candidates,
            question_key=self.question_key,
            proposition=self.proposition,
            cycle=3,
            grounded_in=["A0", "A1"],
            graph=store.graph,
        )

    def test_agreed_answer_settles_the_question_and_prunes_the_audit_slot(self):
        store = self._store()
        resolution = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="IRRELEVANT", effect="NO_CHANGE"),
        ])

        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.resolution, "NO_CHANGE")
        record = commit_question_resolution(store, resolution)
        self.assertEqual(record.status, "COMMITTED")
        self.assertEqual(settled_question_keys(store.graph), {self.question_key})

        signals, question, payload = _problem_state_audit_probe(
            {
                "audit_candidates": [{
                    "issue_id": self.question_key,
                    "proposition": self.proposition,
                    "grounded_in": ["A0", "A1"],
                    "raised_by": ["utilitarian"],
                    "category": "RESOLVE_NORMATIVE_TENSION",
                    "status": "PERSISTENT_UNRESOLVED",
                }],
                "agent_positions": [],
            },
            self.actions[1],
        )
        self.assertTrue(signals and question and payload)

        state = build_deliberative_problem_state(
            4, list(self.actions), [], self.actions[1],
            self._responder("care", participation="RELEVANT", effect="NO_CHANGE"),
            {"unresolved_questions": []},
            store.graph,
        ).to_dict()
        self.assertEqual(
            [item["question_key"] for item in state["resolved_questions"]],
            [self.question_key],
        )
        self.assertEqual(state["resolved_questions"][0]["status"], "SETTLED")

    def test_settled_question_leaves_the_audit_slot_but_stays_unresolved(self):
        store = self._store()
        asker = CandidateChunk(
            specialist="utilitarian", constraint="AGGREGATE_WELFARE",
            action_scores={self.actions[0]: 0.5, self.actions[1]: 0.5},
            surprise=0.2, friction=0.2, confidence=0.5,
            recommended_action=self.actions[1],
            rationale="The comparison remains open.",
            unresolved="VERIFY_FACTS",
            unsupported_assumption=(
                "whether the innocent technician being framed outweighs the "
                "freezing residents losing heat"
            ),
        )

        open_state = build_deliberative_problem_state(
            3, list(self.actions), [asker], self.actions[1], asker,
            None, store.graph, self.scenario,
        ).to_dict()
        raised = open_state["unresolved_questions"][0]
        self.assertIn(
            raised["question_key"],
            {item["issue_id"] for item in open_state["audit_candidates"]},
        )

        resolution = resolve_audited_question(
            [
                self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
                self._responder("care", participation="RELEVANT", effect="NO_CHANGE"),
            ],
            question_key=raised["question_key"],
            proposition=raised["question"],
            cycle=3,
            grounded_in=list(raised["grounded_in"]),
            graph=store.graph,
        )
        self.assertIsNotNone(resolution)
        self.assertEqual(
            commit_question_resolution(store, resolution).status, "COMMITTED",
        )

        pruned = build_deliberative_problem_state(
            4, list(self.actions), [asker], self.actions[1], asker,
            open_state, store.graph, self.scenario,
        ).to_dict()

        self.assertNotIn(
            raised["question_key"],
            {item["issue_id"] for item in pruned["audit_candidates"]},
        )
        settled = next(
            item for item in pruned["unresolved_questions"]
            if item["question_key"] == raised["question_key"]
        )
        self.assertEqual(settled["resolution_status"], "SETTLED")
        self.assertEqual(settled["recorded_resolution"], "NO_CHANGE")
        self.assertIn("VERIFY_FACTS", pruned["unresolved_categories"])

    def test_disagreement_leaves_the_question_open(self):
        store = self._store()

        contested = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="RELEVANT", effect="REVERSES"),
        ])
        unresolved = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="CONTESTED", effect="NO_CHANGE"),
        ])
        lone = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="NOT_TESTED", effect="UNRESOLVED"),
        ])

        self.assertIsNone(contested)
        self.assertIsNone(unresolved)
        self.assertIsNone(lone)

    def test_new_evidence_on_the_grounding_reopens_the_question(self):
        store = self._store()
        resolution = self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="RELEVANT", effect="NO_CHANGE"),
        ])
        commit_question_resolution(store, resolution)
        self.assertEqual(settled_question_keys(store.graph), {self.question_key})

        store.graph.add_node(SemanticNode(
            "LATE_CONSEQUENCE", "CONSEQUENCE", "district-wide outage",
            ("cycle:5", "late evidence"), {"polarity": "ADVERSE"},
        ))
        store.graph.add_edge(SemanticEdge(
            "A0", "HAS_CONSEQUENCE", "LATE_CONSEQUENCE",
            provenance=("cycle:5", "late evidence"),
        ))

        index = question_resolution_index(store.graph)
        self.assertEqual(index[self.question_key]["status"], "REOPENED")
        self.assertTrue(index[self.question_key]["reopened_by_evidence_change"])
        self.assertEqual(settled_question_keys(store.graph), frozenset())

    def test_grounding_outside_the_current_run_graph_is_rejected(self):
        store = self._store()
        responders = [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("care", participation="RELEVANT", effect="NO_CHANGE"),
        ]

        unresolvable = resolve_audited_question(
            responders,
            question_key=self.question_key,
            proposition=self.proposition,
            cycle=3,
            grounded_in=["GHOST_NODE"],
            graph=store.graph,
        )
        record = commit_question_resolution(store, QuestionResolution(
            question_key=self.question_key,
            proposition=self.proposition,
            resolution="NO_CHANGE",
            cycle=3,
            grounded_in=("GHOST_NODE",),
            responders=("care", "utilitarian"),
            evidence_signature="stale",
        ))

        self.assertIsNone(unresolvable)
        self.assertEqual(record.status, "REJECTED")
        self.assertTrue(record.previous_state_preserved)
        self.assertEqual(question_resolution_index(store.graph), {})


class TypedLedgerGroundingTests(unittest.TestCase):
    """The ledger's own field names must satisfy the framework construct gate."""

    actions = ("publish the audit and end the siphon", "suppress the audit")
    action_ids = ("A0", "A1")
    # The delegate writes the ledger's canonical dimension token, which the
    # prose marker cannot match because an underscore is not whitespace.
    action_map = {
        actions[0]: "IMPROVES: the district's BASIC_LIBERTY and survival",
        actions[1]: "WORSENS: the district's BASIC_LIBERTY and survival",
    }
    positions = {
        "A0": {
            "e": "IMPROVES", "d": "BASIC_LIBERTY", "ca": "A1", "b": "SCENARIO",
            "s": "impoverished district residents", "sk": "GROUP",
            "rs": "the covert siphon stops",
        },
        "A1": {
            "e": "WORSENS", "d": "BASIC_LIBERTY", "ca": "A0", "b": "SCENARIO",
            "s": "impoverished district residents", "sk": "GROUP",
            "rs": "the covert siphon continues",
        },
    }

    def _errors(self, rp: object) -> list[str]:
        return _construct_map_errors(
            "rawlsian",
            list(self.actions),
            dict(self.action_map),
            "SECONDARY",
            "the worst-off subjects are the district's residents",
            action_ids=list(self.action_ids),
            recommended_action=self.actions[0],
            rationale="Basic liberty for the district holds lexical priority",
            supporting_data={"rp": rp},
        )

    def test_ledger_dimension_token_grounds_the_map_line(self):
        self.assertNotIn(
            f"framework map for {self.actions[0]} lacks framework-specific grounds",
            self._errors(self.positions),
        )

    def test_ledger_about_something_else_cannot_ground_the_map_line(self):
        unrelated = {
            "A0": {
                "e": "IMPROVES", "d": "COST", "ca": "A1", "b": "SCENARIO",
                "s": "the operating budget", "sk": "INSTITUTION",
                "rs": "cheaper to run",
            },
            "A1": {
                "e": "WORSENS", "d": "COST", "ca": "A0", "b": "SCENARIO",
                "s": "the operating budget", "sk": "INSTITUTION",
                "rs": "more expensive to run",
            },
        }
        errors = self._errors(unrelated)

        self.assertIn(
            f"framework map for {self.actions[0]} lacks framework-specific grounds",
            errors,
        )

    def test_absent_ledger_leaves_the_prose_gate_in_force(self):
        self.assertIn(
            f"framework map for {self.actions[0]} lacks framework-specific grounds",
            self._errors({}),
        )


class OpeningProblemFrameTests(unittest.TestCase):
    """Cycle one receives the shared world without receiving a position."""

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs."
    )
    actions = ("execute the covert override", "preserve the audit logs")

    def test_frame_carries_vocabulary_and_no_salient_position(self):
        graph = compile_scenario_graph(self.scenario, list(self.actions), {})
        frame = opening_problem_state(list(self.actions), self.scenario, graph)

        self.assertEqual(frame["cycle"], 0)
        self.assertEqual(frame["state_role"], "OPENING_PROBLEM_FRAME")
        self.assertEqual(
            [item["action_id"] for item in frame["live_actions"]], ["A0", "A1"],
        )
        self.assertTrue(frame["scenario_clauses"])
        self.assertTrue(all(
            item["clause_id"].startswith("C") for item in frame["scenario_clauses"]
        ))
        identities = {item["node_id"] for item in frame["grounded_identities"]}
        self.assertTrue({"A0", "A1"} <= identities)
        self.assertIn(
            "ACTION", {item["kind"] for item in frame["grounded_identities"]},
        )
        self.assertEqual(frame["salient_position"], {})
        self.assertEqual(frame["agent_positions"], [])
        self.assertEqual(frame["current_plurality"], "")

    def test_engine_seeds_the_frame_and_records_no_influence_for_it(self):
        received: list[dict[str, object]] = []

        class Recording(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                received.append(dict(broadcast.problem_state or {}))
                return super().evaluate(scenario, actions, broadcast)

        result = WorkspaceEngine(
            [Recording("care", self.actions[1], "CARE"),
             Recording("duty", self.actions[1], "DUTY")],
            WorkspaceConfig(
                max_cycles=1, stable_cycles_required=3,
                enable_consensus_audit=False, enable_reversal_audit=False,
                enable_synthesis=False, enable_planning=False,
            ),
        ).run(self.scenario, list(self.actions))

        self.assertTrue(received)
        self.assertEqual(received[0]["state_role"], "OPENING_PROBLEM_FRAME")
        self.assertTrue(received[0]["scenario_clauses"])
        # A positionless frame is not a broadcast consideration, so it must not
        # produce influence records attributing uptake to nobody.
        self.assertEqual(result.broadcast_influence_records, [])


class UnresolvableAuditTests(unittest.TestCase):
    """An audit nobody can answer records the evidence it would take."""

    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )
    question_key = "QUESTION:unresolvable01"
    proposition = "Whether the siphon will stop within a month"
    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs."
    )

    def _store(self) -> SemanticGraphStore:
        return SemanticGraphStore(compile_scenario_graph(
            self.scenario, list(self.actions), {},
        ))

    def _responder(
        self,
        specialist: str,
        *,
        participation: str,
        effect: str,
        explanation: str = "",
    ) -> CandidateChunk:
        return CandidateChunk(
            specialist=specialist, constraint="OPEN_DELIBERATION",
            action_scores={self.actions[0]: 0.4, self.actions[1]: 0.6},
            surprise=0.1, friction=0.1, confidence=0.6,
            recommended_action=self.actions[1],
            rationale="The audited issue cannot be settled here.",
            audit_participation=participation,
            audit_internal_effect=effect,
            audit_framework_explanation=explanation,
        )

    def _resolve(self, store: SemanticGraphStore, candidates: list[CandidateChunk]):
        return resolve_audited_question(
            candidates,
            question_key=self.question_key,
            proposition=self.proposition,
            cycle=3,
            grounded_in=["A0", "A1"],
            graph=store.graph,
        )

    def test_shared_evidence_gap_records_a_named_requirement(self):
        store = self._store()
        resolution = self._resolve(store, [
            self._responder(
                "utilitarian", participation="REVERSAL_RELEVANT",
                effect="UNRESOLVED",
                explanation="If the siphon stops soon the advantage disappears",
            ),
            self._responder(
                "virtue", participation="REVERSAL_RELEVANT",
                effect="UNRESOLVED",
                explanation="If the siphon stops soon the justice gain shrinks",
            ),
            self._responder(
                "care", participation="NOT_TESTED", effect="UNRESOLVED",
            ),
        ])

        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.resolution, "UNRESOLVABLE_AT_CURRENT_EVIDENCE")
        self.assertIn("siphon stops soon", resolution.evidence_requirement)
        self.assertEqual(resolution.responders, ("utilitarian", "virtue"))
        self.assertEqual(
            commit_question_resolution(store, resolution).status, "COMMITTED",
        )
        self.assertEqual(settled_question_keys(store.graph), {self.question_key})
        recorded = question_resolution_index(store.graph)[self.question_key]
        self.assertIn("siphon stops soon", recorded["evidence_requirement"])

    def test_frameworks_reporting_different_effects_settle_nothing(self):
        store = self._store()

        self.assertIsNone(self._resolve(store, [
            self._responder("utilitarian", participation="RELEVANT", effect="NO_CHANGE"),
            self._responder("virtue", participation="RELEVANT", effect="UNRESOLVED"),
        ]))

    def test_contested_participation_still_records_nothing(self):
        store = self._store()

        self.assertIsNone(self._resolve(store, [
            self._responder("utilitarian", participation="CONTESTED", effect="UNRESOLVED"),
            self._responder("virtue", participation="REVERSAL_RELEVANT", effect="UNRESOLVED"),
        ]))

    def test_new_evidence_reopens_an_unresolvable_question(self):
        store = self._store()
        resolution = self._resolve(store, [
            self._responder(
                "utilitarian", participation="REVERSAL_RELEVANT",
                effect="UNRESOLVED", explanation="duration of the siphon is unknown",
            ),
            self._responder(
                "virtue", participation="REVERSAL_RELEVANT",
                effect="UNRESOLVED", explanation="duration of the siphon is unknown",
            ),
        ])
        commit_question_resolution(store, resolution)
        self.assertEqual(settled_question_keys(store.graph), {self.question_key})

        delta = SemanticGraph()
        delta.add_node(SemanticNode(
            "OBS:siphon_schedule", "OBSERVATION",
            "the siphon is scheduled to end in three weeks",
            ("cycle:4", "test"), {"support": "STATED"},
        ))
        delta.add_edge(SemanticEdge(
            "A0", "SUPPORTED_BY", "OBS:siphon_schedule",
            justification="new schedule evidence", provenance=("cycle:4", "test"),
        ))
        store.graph = merge_graphs([store.graph, delta])

        self.assertEqual(settled_question_keys(store.graph), frozenset())
        self.assertEqual(
            question_resolution_index(store.graph)[self.question_key]["status"],
            "REOPENED",
        )


class AuditCandidateGroundingPriorityTests(unittest.TestCase):
    """Ungrounded inquiry stays testable without displacing grounded inquiry."""

    scenario = (
        "You can execute a covert override to save 8 freezing residents, but "
        "doing so overwrites system logs, framing an innocent technician for "
        "sabotage. Alternatively, you can preserve the audit logs."
    )
    actions = (
        "execute the covert override that erases the logs to restore heat",
        "preserve the audit logs and leave the heating system untouched",
    )

    def _asker(self, specialist: str, assumption: str) -> CandidateChunk:
        return CandidateChunk(
            specialist=specialist, constraint="UNCERTAINTY",
            action_scores={self.actions[0]: 0.5, self.actions[1]: 0.5},
            surprise=0.2, friction=0.2, confidence=0.5,
            recommended_action=self.actions[1],
            rationale="The comparison remains open.",
            unresolved="VERIFY_FACTS",
            unsupported_assumption=assumption,
        )

    def _state(self, candidates: list[CandidateChunk]) -> dict[str, object]:
        graph = compile_scenario_graph(self.scenario, list(self.actions), {})
        return build_deliberative_problem_state(
            2, list(self.actions), candidates, self.actions[1], candidates[0],
            None, graph, self.scenario,
        ).to_dict()

    def test_ungrounded_question_is_offered_when_nothing_grounded_competes(self):
        state = self._state([self._asker("care", "whether ongoing support holds")])
        candidates = state["audit_candidates"]

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["grounding_status"], "UNGROUNDED")
        # It is anchored to the action nodes so its answer stays recordable.
        self.assertEqual(candidates[0]["grounded_in"], ["A0", "A1"])

    def test_grounded_question_displaces_every_ungrounded_one(self):
        state = self._state([
            self._asker("care", "whether ongoing support holds"),
            self._asker(
                "utilitarian",
                "whether the freezing residents outweigh the innocent technician",
            ),
        ])
        candidates = state["audit_candidates"]

        self.assertTrue(candidates)
        self.assertEqual(
            {item["grounding_status"] for item in candidates}, {"CLAUSE_GROUNDED"},
        )
        self.assertEqual(
            {item["raised_by"][0] for item in candidates}, {"utilitarian"},
        )

    def test_consensus_audit_cannot_focus_an_ungrounded_question(self):
        state = self._state([self._asker("care", "whether ongoing support holds")])

        _signals, _question, grounded_only = _problem_state_audit_probe(
            state, self.actions[1], require_clause_grounding=True,
        )
        _signals, _question, permissive = _problem_state_audit_probe(
            state, self.actions[1],
        )

        self.assertEqual(grounded_only, {})
        self.assertTrue(permissive)
        self.assertEqual(permissive["grounding_status"], "UNGROUNDED")


class KantianAuthoritySeparationTests(unittest.TestCase):
    """Provisional leanings may interrupt; they may not govern."""

    def _contested_rival(self) -> dict[str, object]:
        return {
            "canonical_action_id": "A1",
            "protected_party": "impoverished district residents",
            "norm": "do not treat humans merely as means",
            "competing_protected_party": "icu patients",
            "competing_norm": "duty to rescue ICU patients today",
            "coercive_actor": "grid director",
            "coerced_party": "impoverished residents",
            "resolution_status": "CONTESTED",
            "calibration_errors": [
                "coercion claim lacks actor-to-mechanism-to-restriction-to-party grounding",
                "resolved adjudication rests on unsupported decisive premises",
            ],
            "derivation": "RESPECT_PERSONS",
            "priority_rule": "perfect duties are non-negotiable",
        }

    def _resolved_primary(self) -> dict[str, object]:
        return {
            "canonical_action_id": "A0",
            "protected_party": "impoverished district residents",
            "norm": "do not treat humans merely as means",
            "competing_protected_party": "icu patients",
            "competing_norm": "duty to avoid immediate harm to ICU patients",
            "resolution_status": "RESOLVED",
            "calibration_errors": [],
            "derivation": "RESPECT_PERSONS",
            "priority_rule": "perfect duties override conflicting beneficence",
        }

    def test_rival_contestation_yields_provisional_leaning_not_governing_rule(self):
        from global_workspace.deontology_ledger import classify_deontological_authority

        profile = classify_deontological_authority(
            self._resolved_primary(),
            [self._contested_rival()],
            recommended_action="publish the audit",
            preference_strength=0.4,
        )

        self.assertEqual(profile.adjudication_status, "PROVISIONAL_LEANING")
        self.assertEqual(profile.broadcast_authority, "INVESTIGATIVE")
        self.assertFalse(profile.governing_eligible)
        self.assertLess(profile.policy_weight_factor, 1.0)
        self.assertGreater(profile.policy_weight_factor, 0.0)
        self.assertIn("UNRESOLVED DUTY CONFLICT", profile.investigative_claim)
        self.assertIn("publish the audit", profile.investigative_claim)
        self.assertNotIn("perfect duties override", profile.decision_rule.casefold())
        self.assertNotIn("perfect duties override", profile.investigative_claim.casefold())

    def test_flat_preference_yields_contested_no_leaning(self):
        from global_workspace.deontology_ledger import classify_deontological_authority

        profile = classify_deontological_authority(
            self._resolved_primary(),
            [self._contested_rival()],
            recommended_action="publish the audit",
            preference_strength=0.05,
        )

        self.assertEqual(profile.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(profile.policy_weight_factor, 0.0)
        self.assertFalse(profile.governing_eligible)

    def test_fully_resolved_comparative_adjudication_remains_governing(self):
        from global_workspace.deontology_ledger import classify_deontological_authority

        clean_rival = {
            **self._contested_rival(),
            "resolution_status": "RESOLVED",
            "calibration_errors": [],
        }
        profile = classify_deontological_authority(
            self._resolved_primary(),
            [clean_rival],
            recommended_action="publish the audit",
            preference_strength=0.4,
        )

        self.assertEqual(profile.adjudication_status, "SUPPORTS")
        self.assertEqual(profile.broadcast_authority, "GOVERNING_CANDIDATE")
        self.assertTrue(profile.governing_eligible)
        self.assertEqual(profile.policy_weight_factor, 1.0)
        self.assertIn("perfect duties override", profile.decision_rule.casefold())

    def test_provisional_kant_cannot_supply_governing_rule_when_rawls_can(self):
        actions = ["publish the audit", "suppress the audit"]
        kant = CandidateChunk(
            specialist="deontological",
            constraint="DUTY",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.8,
            friction=0.7,
            confidence=0.55,
            recommended_action=actions[0],
            decision_rule="Treat the duty priority as unestablished until coercion is grounded",
            rationale="Comparative Kantian judgment remains contested.",
            adjudication_status="PROVISIONAL_LEANING",
            broadcast_authority="INVESTIGATIVE",
            governing_eligible=False,
            policy_weight_factor=0.45,
            investigative_claim=(
                "UNRESOLVED DUTY CONFLICT: district vs ICU. Current reasoning "
                "leans publish the audit."
            ),
        )
        rawls = CandidateChunk(
            specialist="rawlsian",
            constraint="FAIRNESS",
            action_scores={actions[0]: 0.75, actions[1]: 0.25},
            surprise=0.4,
            friction=0.4,
            confidence=0.72,
            recommended_action=actions[0],
            decision_rule="A0 ends a systematic burden on the least advantaged",
            rationale="Least-advantaged class bears ongoing lethal deprivation.",
            adjudication_status="SUPPORTS",
            broadcast_authority="GOVERNING_CANDIDATE",
            governing_eligible=True,
            policy_weight_factor=1.0,
        )

        governing = WorkspaceEngine._select_governing_candidate(
            [kant, rawls], actions[0], preferred=kant,
        )
        self.assertIs(governing, rawls)
        self.assertNotIn("perfect duties", (governing.decision_rule or "").casefold())

        policy = WorkspaceEngine._policy([kant, rawls], actions)
        # Attenuated Kant still contributes, but Rawls should dominate plurality.
        self.assertGreater(policy[actions[0]], policy[actions[1]])

        # Investigative Kant can still outrank Rawls on salience.
        broadcast = WorkspaceBroadcast(constraint="OPEN_DELIBERATION", urgency=0.5)
        engine = WorkspaceEngine(
            specialists=[FixedSpecialist("rawlsian", actions[0], "FAIRNESS")],
            config=WorkspaceConfig(investigative_attention_weight=0.40),
        )
        kant.tension_engagement = 0.7
        kant.unresolved = "RESOLVE_NORMATIVE_TENSION"
        kant.salience = engine._salience(kant, broadcast, {}, 1.0)
        rawls.salience = engine._salience(rawls, broadcast, {"FAIRNESS": 1}, 0.0)
        self.assertGreater(kant.salience, rawls.salience)

    def test_presentation_uses_governing_eligible_rule_not_investigative_winner(self):
        actions = ["publish the audit", "suppress the audit"]
        data = {
            "actions": actions,
            "selected_action": actions[0],
            "current_plurality": actions[0],
            "judgment_status": "GOVERNED_RECOMMENDATION",
            "governing_justification_status": "ADMISSIBLE",
            "confidence": 0.7,
            "epistemic_confidence": 0.65,
            "halted_by": "convergence",
            "moral_residue": ["DUTY"],
            "compressed_rule": "When fairness is salient, prefer publish the audit.",
            "reopen_conditions": [],
            "cycles": [{
                "cycle": 1,
                "policy": {actions[0]: 0.7, actions[1]: 0.3},
                "policy_leader": actions[0],
                "broadcast_focus": {
                    "specialist": "deontological",
                    "constraint": "DUTY",
                    "schema_valid": True,
                    "recommended_action": actions[0],
                    "decision_rule": "Treat the duty priority as unestablished",
                    "rationale": "Kantian conflict open.",
                    "assumption_status": "NORMATIVELY_CONTESTED",
                    "adjudication_status": "PROVISIONAL_LEANING",
                    "broadcast_authority": "INVESTIGATIVE",
                    "governing_eligible": False,
                    "investigative_claim": (
                        "UNRESOLVED DUTY CONFLICT: entrusted lethal withdrawal "
                        "versus covert institutional sacrifice."
                    ),
                    "action_scores": {actions[0]: 0.7, actions[1]: 0.3},
                },
                "governing_claim": {
                    "specialist": "rawlsian",
                    "constraint": "FAIRNESS",
                    "schema_valid": True,
                    "recommended_action": actions[0],
                    "decision_rule": "A0 ends a systematic burden on the least advantaged",
                    "rationale": "Least-advantaged class bears ongoing lethal deprivation.",
                    "assumption_status": "DIRECT",
                    "adjudication_status": "SUPPORTS",
                    "broadcast_authority": "GOVERNING_CANDIDATE",
                    "governing_eligible": True,
                    "action_scores": {actions[0]: 0.75, actions[1]: 0.25},
                },
                "winner": {
                    "specialist": "deontological",
                    "constraint": "DUTY",
                    "schema_valid": True,
                    "recommended_action": actions[0],
                    "decision_rule": "Treat the duty priority as unestablished",
                    "rationale": "Kantian conflict open.",
                    "assumption_status": "NORMATIVELY_CONTESTED",
                    "adjudication_status": "PROVISIONAL_LEANING",
                    "broadcast_authority": "INVESTIGATIVE",
                    "governing_eligible": False,
                    "investigative_claim": (
                        "UNRESOLVED DUTY CONFLICT: entrusted lethal withdrawal "
                        "versus covert institutional sacrifice."
                    ),
                    "action_scores": {actions[0]: 0.7, actions[1]: 0.3},
                },
                "candidates": [
                    {
                        "specialist": "deontological",
                        "constraint": "DUTY",
                        "schema_valid": True,
                        "recommended_action": actions[0],
                        "decision_rule": "Treat the duty priority as unestablished",
                        "rationale": "Kantian conflict open.",
                        "assumption_status": "NORMATIVELY_CONTESTED",
                        "adjudication_status": "PROVISIONAL_LEANING",
                        "broadcast_authority": "INVESTIGATIVE",
                        "governing_eligible": False,
                        "investigative_claim": (
                            "UNRESOLVED DUTY CONFLICT: entrusted lethal withdrawal "
                            "versus covert institutional sacrifice."
                        ),
                        "action_scores": {actions[0]: 0.7, actions[1]: 0.3},
                    },
                    {
                        "specialist": "rawlsian",
                        "constraint": "FAIRNESS",
                        "schema_valid": True,
                        "recommended_action": actions[0],
                        "decision_rule": "A0 ends a systematic burden on the least advantaged",
                        "rationale": "Least-advantaged class bears ongoing lethal deprivation.",
                        "assumption_status": "DIRECT",
                        "adjudication_status": "SUPPORTS",
                        "broadcast_authority": "GOVERNING_CANDIDATE",
                        "governing_eligible": True,
                        "action_scores": {actions[0]: 0.75, actions[1]: 0.25},
                    },
                ],
                "dissent": None,
            }],
        }

        text = render_public_judgment(data)
        self.assertIn("## Recommendation", text)
        self.assertIn("## Why the Parliament currently favors this action", text)
        self.assertIn("A0 ends a systematic burden on the least advantaged", text)
        self.assertIn("## Most important unresolved issue", text)
        self.assertIn("UNRESOLVED DUTY CONFLICT", text)
        self.assertNotIn("perfect duties override conflicting beneficence", text.casefold())
        self.assertNotIn("QUESTION:", text)


class SpecialistAuthorityTests(unittest.TestCase):
    """Framework-general provisional / contested / conditional authority."""

    def test_conflicted_alias_parses_to_contested_and_does_not_serialize(self):
        from global_workspace.specialist_authority import (
            CONTESTED_NO_LEANING,
            normalize_specialist_status,
        )

        self.assertEqual(
            normalize_specialist_status("CONFLICTED_NO_LEANING"),
            CONTESTED_NO_LEANING,
        )
        chunk = CandidateChunk(
            "care", "CARE", {"a": 0.6, "b": 0.4}, 0.2, 0.2, 0.7,
            recommended_action="a",
            adjudication_status="CONFLICTED_NO_LEANING",
        )
        self.assertEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")

    def test_rawls_normative_conflict_is_provisional_not_kant_special(self):
        from global_workspace.specialist_authority import apply_specialist_authority

        chunk = CandidateChunk(
            "rawlsian", "FAIRNESS", {"publish": 0.65, "suppress": 0.35},
            0.3, 0.3, 0.7,
            recommended_action="publish",
            preference_strength=0.3,
            assumption_status="NORMATIVELY_CONTESTED",
            unresolved="RESOLVE_NORMATIVE_TENSION",
            framework_internal_conflicts=[
                "equal basic liberty favors publish ↔ fair equality favors suppress",
            ],
            decision_rule="perfect duties require publish",
        )
        profile = apply_specialist_authority(chunk)
        self.assertEqual(profile.adjudication_status, "PROVISIONAL_LEANING")
        self.assertFalse(chunk.governing_eligible)
        self.assertAlmostEqual(chunk.policy_weight_factor, 0.45)
        self.assertNotIn("perfect duties require", chunk.decision_rule.casefold())

    def test_conditional_support_attenuates_by_reversal_potential(self):
        from global_workspace.specialist_authority import apply_specialist_authority

        chunk = CandidateChunk(
            "utilitarian", "WELFARE", {"publish": 0.7, "suppress": 0.3},
            0.2, 0.2, 0.9,
            recommended_action="publish",
            epistemic_confidence=0.9,
            assumption_status="CONDITIONAL",
            baseline_status="CONDITIONAL",
            baseline_condition=(
                "choose publish unless the siphon would end within one month"
            ),
            utilitarian_decision_depends_on_unknown=True,
            utilitarian_missing_comparison="siphon duration beyond one month",
            decision_rule="A0 maximizes welfare",
            audit_internal_effect="REVERSES",
        )
        profile = apply_specialist_authority(chunk)
        self.assertEqual(profile.adjudication_status, "CONDITIONAL_SUPPORTS")
        self.assertTrue(chunk.governing_eligible)
        self.assertIn("provided", chunk.decision_rule.casefold())
        self.assertLess(chunk.policy_weight_factor, 0.9 * 0.9)
        self.assertGreater(chunk.policy_weight_factor, 0.0)

    def test_cycle_record_aliases_winner_to_broadcast_focus(self):
        focus = CandidateChunk(
            "care", "CARE", {"a": 0.8, "b": 0.2}, 0.2, 0.2, 0.8,
            recommended_action="a",
        )
        cycle = CycleRecord(
            1, WorkspaceBroadcast(), [focus], focus, None,
            {"a": 0.8, "b": 0.2}, 0.5, 1, 0.1,
            policy_leader="a",
            governing_claim=focus,
            broadcast_focus=focus,
        )
        self.assertIs(cycle.broadcast_focus, focus)
        self.assertIs(cycle.winner, focus)
        self.assertEqual(cycle.policy_leader, "a")
        payload = asdict(cycle)
        self.assertIn("policy_leader", payload)
        self.assertIn("governing_claim", payload)
        self.assertIn("broadcast_focus", payload)
        self.assertIn("winner", payload)

    def test_investigative_priority_product_and_reopen_gate(self):
        from global_workspace.specialist_authority import (
            REOPEN_PRIORITY_THRESHOLD,
            apply_investigative_authority,
            apply_specialist_authority,
            classify_terminal_judgment,
            evidence_fingerprint_for,
        )

        kant = CandidateChunk(
            "deontological", "DUTY",
            {"publish": 0.6, "suppress": 0.4},
            0.5, 0.5, 0.7,
            recommended_action="publish",
            preference_strength=0.2,
            assumption_status="NORMATIVELY_CONTESTED",
            unresolved="RESOLVE_NORMATIVE_TENSION",
            evidence_basis="STATED_FACTS",
            framework_internal_conflicts=[
                "entrusted ICU strict duty ↔ institutional sacrifice",
            ],
            investigative_claim=(
                "If ICU patients possess an entrusted strict right against withdrawal, "
                "publish may be prohibited."
            ),
            tension_target_keys=["QUESTION:icu-duty"],
        )
        apply_specialist_authority(kant)
        profile = apply_investigative_authority(
            kant,
            plurality="publish",
            policy={"publish": 0.8, "suppress": 0.2},
            problem_state={
                "audit_candidates": [{
                    "question_key": "QUESTION:icu-duty",
                    "grounding_status": "CLAUSE_GROUNDED",
                    "grounded_in": ["C0"],
                    "proposition": "entrusted ICU duty remains unresolved",
                }],
            },
            fired_keys={},
        )
        self.assertGreaterEqual(kant.investigative_priority, REOPEN_PRIORITY_THRESHOLD)
        self.assertTrue(kant.reopen_eligible)
        self.assertEqual(kant.broadcast_authority, "INVESTIGATIVE")
        self.assertFalse(kant.governing_eligible)

        rawls = CandidateChunk(
            "rawlsian", "FAIRNESS",
            {"publish": 0.8, "suppress": 0.2},
            0.2, 0.2, 0.9,
            recommended_action="publish",
            decision_rule="A0 ends a systematic burden on the least advantaged",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
        )
        apply_specialist_authority(rawls)
        apply_investigative_authority(
            rawls, plurality="publish", policy={"publish": 0.8, "suppress": 0.2},
        )
        self.assertTrue(rawls.governing_eligible)
        self.assertLess(rawls.investigative_priority, REOPEN_PRIORITY_THRESHOLD)

        terminal = classify_terminal_judgment(
            plurality="publish",
            governing=rawls,
            candidates=[kant, rawls],
        )
        self.assertEqual(terminal.status, "CONTESTED_RECOMMENDATION")
        self.assertEqual(terminal.governing_justification_status, "UNDER_ATTACK")

        # One reopen per question_key unless evidence changes.
        profile2 = apply_investigative_authority(
            kant,
            plurality="publish",
            policy={"publish": 0.8, "suppress": 0.2},
            problem_state={
                "audit_candidates": [{
                    "question_key": "QUESTION:icu-duty",
                    "grounding_status": "CLAUSE_GROUNDED",
                    "grounded_in": ["C0"],
                    "proposition": "entrusted ICU duty remains unresolved",
                }],
            },
            fired_keys={kant.reopen_question_key: evidence_fingerprint_for(
                kant,
                {
                    "audit_candidates": [{
                        "question_key": "QUESTION:icu-duty",
                        "grounding_status": "CLAUSE_GROUNDED",
                        "grounded_in": ["C0"],
                        "proposition": "entrusted ICU duty remains unresolved",
                    }],
                },
            )},
        )
        self.assertFalse(kant.reopen_eligible)

    def test_plurality_without_governing_claim_is_contested(self):
        from global_workspace.specialist_authority import classify_terminal_judgment

        provisional = CandidateChunk(
            "care", "CARE", {"a": 0.7, "b": 0.3}, 0.2, 0.2, 0.6,
            recommended_action="a",
            adjudication_status="PROVISIONAL_LEANING",
            governing_eligible=False,
            decision_rule="Provisionally lean a",
        )
        terminal = classify_terminal_judgment(
            plurality="a",
            governing=None,
            candidates=[provisional],
        )
        self.assertEqual(terminal.status, "CONTESTED_RECOMMENDATION")
        self.assertEqual(terminal.governing_rule, "")
        self.assertEqual(terminal.governing_justification_status, "NONE")


class AuditCycleHygieneTests(unittest.TestCase):
    """A failed audit cycle must not erase a completed parliament."""

    def test_audit_payload_strips_grounding_status_before_broadcast(self):
        state = {
            "audit_candidates": [{
                "issue_id": "QUESTION:abc123",
                "question_key": "QUESTION:abc123",
                "proposition": "duration and permanence of reform may change ranking",
                "question": "duration and permanence of reform may change ranking",
                "grounded_in": ["C0"],
                "grounding_status": "CLAUSE_GROUNDED",
                "raised_by": ["utilitarian"],
                "status": "PERSISTENT_UNRESOLVED",
                "category": "VERIFY_FACTS",
            }],
            "agent_positions": [{
                "specialist": "utilitarian", "choice_status": "UNDERDETERMINED",
            }],
        }
        _signals, _question, payload = _problem_state_audit_probe(
            state, "Publish the unalterable system audit now",
        )
        sanitized = _admitted_audit_variable({
            **payload, "grounding_status": "CLAUSE_GROUNDED", "junk": True,
        })
        self.assertIn("grounding_status", payload)
        self.assertNotIn("junk", sanitized)
        self.assertTrue({"entity", "relation", "possible_values", "focus_action", "question"} <= set(sanitized))

    def test_preserved_kantian_state_keeps_visibility_answer(self):
        actions = ["publish", "suppress"]
        prior = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.4, friction=0.4, confidence=0.6,
            recommended_action=actions[0],
            visibility_response="NOT_TESTED",
        )
        incoming = CandidateChunk(
            specialist="deontological", constraint="DUTY",
            action_scores={actions[0]: 0.7, actions[1]: 0.3},
            surprise=0.4, friction=0.4, confidence=0.6,
            recommended_action=actions[0],
            framework_retention_status="UPDATE_REJECTED",
            framework_constraint_retained=False,
            visibility_response="ACCEPT",
            visibility_justification="Hidden siphon deaths were undercounted.",
            visibility_harm_revision="UPWARD",
            visibility_magnitude_status="UNKNOWN",
        )
        restored = _operative_framework_candidates(
            [incoming], {"deontological": prior}, remember=False,
        )[0]
        self.assertEqual(restored.framework_retention_status, "PRESERVED_AFTER_REJECTED_UPDATE")
        self.assertEqual(restored.visibility_response, "ACCEPT")
        self.assertIn("undercounted", restored.visibility_justification)

    def test_later_invalid_cycle_recovers_the_last_valid_recommendation(self):
        class FlipInvalid(FixedSpecialist):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0

            def evaluate(self, scenario, actions, broadcast):
                self.calls += 1
                chunk = super().evaluate(scenario, actions, broadcast)
                if self.calls >= 2:
                    chunk.schema_valid = False
                    chunk.validation_errors = ["invalid"]
                    chunk.confidence = 1.0
                return chunk

        result = WorkspaceEngine(
            [
                FlipInvalid("care", "protect", "CARE"),
                FlipInvalid("rawlsian", "protect", "FAIRNESS"),
            ],
            WorkspaceConfig(
                max_cycles=2, min_valid_specialists=2, enable_synthesis=False,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_reversal_audit=False,
            ),
        ).run("A test", ["protect", "wait"])

        self.assertEqual(result.halted_by, "insufficient_valid_candidates")
        self.assertEqual(result.selected_action, "protect")
        self.assertNotEqual(result.judgment_status, "INCONCLUSIVE")
        self.assertGreater(result.confidence, 0.0)
        text = render_public_judgment(result.to_dict())
        self.assertIn("## Recommendation", text)
        self.assertIn("protect", text)
        self.assertNotIn("QUESTION:", text)


class UncertaintyTypingMigrationTests(unittest.TestCase):
    """Old names parse; canonical forms serialize; structural mislabels normalize."""

    def test_resolve_normative_tension_alias_serializes_as_normative_adjudication(self):
        from global_workspace.uncertainty_types import (
            NORMATIVE_ADJUDICATION,
            normalize_unresolved_marker,
        )
        self.assertEqual(
            normalize_unresolved_marker("RESOLVE_NORMATIVE_TENSION"),
            NORMATIVE_ADJUDICATION,
        )
        chunk = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={"a": 0.6, "b": 0.4},
            surprise=0.2, friction=0.2, confidence=0.5,
            unresolved="RESOLVE_NORMATIVE_TENSION",
        )
        self.assertEqual(chunk.unresolved, NORMATIVE_ADJUDICATION)

    def test_decision_boundary_payload_is_minimal_and_exact(self):
        from global_workspace.local_specialists import _admitted_audit_variable
        admitted = _admitted_audit_variable({
            "entity": "siphon duration under one month",
            "relation": "DECISION_BOUNDARY",
            "category": "DECISION_BOUNDARY",
            "condition": "if siphon duration is under 1 month",
            "expected_effect": "REVERSES",
            "target_framework": "utilitarian",
            "target_claim_key": "QUESTION:util:duration",
            "possible_values": ["NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"],
            "focus_action": "publish",
            "question": (
                "If siphon duration is under 1 month, does utilitarian support reverse?"
            ),
            "junk": True,
        })
        self.assertNotIn("junk", admitted)
        self.assertEqual(admitted["category"], "DECISION_BOUNDARY")
        self.assertEqual(admitted["relation"], "DECISION_BOUNDARY")
        self.assertEqual(admitted["boundary_status"], "UNRESOLVED")
        self.assertEqual(
            admitted["expected_effect"], "REVERSES_FRAMEWORK_PREFERENCE",
        )
        self.assertEqual(
            admitted["possible_values"],
            ["NOT_CROSSED", "CROSSED", "UNRESOLVED"],
        )
        self.assertEqual(admitted["target_framework"], "utilitarian")
        self.assertEqual(admitted["target_claim_key"], "QUESTION:util:duration")
        self.assertIn("1 month", admitted["condition"])

    def test_verify_facts_threshold_rule_reclassifies_on_admission(self):
        from global_workspace.local_specialists import _admitted_audit_variable
        admitted = _admitted_audit_variable({
            "entity": "siphon duration",
            "relation": "EMPIRICAL_UNKNOWN",
            "category": "VERIFY_FACTS",
            "proposition": (
                "If the siphon lasts less than 1 month, utilitarian preference reverses."
            ),
            "possible_values": ["TRUE", "FALSE", "UNKNOWN"],
            "focus_action": "publish",
            "question": (
                "If the siphon lasts less than 1 month, does preference reverse?"
            ),
        })
        self.assertEqual(admitted["category"], "DECISION_BOUNDARY")
        self.assertEqual(admitted["reclassified_from"], "VERIFY_FACTS")
        self.assertEqual(
            admitted["expected_effect"], "REVERSES_FRAMEWORK_PREFERENCE",
        )

    def test_ambiguous_verify_facts_prose_is_not_rewritten(self):
        from global_workspace.local_specialists import _admitted_audit_variable
        admitted = _admitted_audit_variable({
            "entity": "duty conflict",
            "relation": "EMPIRICAL_UNKNOWN",
            "category": "VERIFY_FACTS",
            "proposition": (
                "Whether rescue or non-maleficence should govern remains unsettled."
            ),
            "possible_values": ["TRUE", "FALSE", "UNKNOWN"],
            "focus_action": "divert",
            "question": (
                "Whether rescue or non-maleficence should govern remains unsettled."
            ),
        })
        self.assertEqual(admitted["category"], "VERIFY_FACTS")
        self.assertEqual(admitted.get("typing_review"), "AMBIGUOUS_UNCERTAINTY_TYPING")
        self.assertNotIn("reclassified_from", admitted)

    def test_candidate_threshold_mislabel_normalizes_at_parser_admission(self):
        chunk = _candidate_from_data(
            "care",
            ["publish the audit", "suppress the audit"],
            {
                "scores": {"A0": 0.55, "A1": 0.45},
                "r": "A0",
                "c": "CARE",
                "u": "VERIFY_FACTS",
                "w": "publish protects entrusted dependents unless siphon is brief",
                "j": "NONE",
                "e": "STATED_FACTS",
                "x": "NONE",
                "z": 0.6,
                "rm": {
                    "A0": "entrusted dependency under publish",
                    "A1": "created vulnerability under suppress",
                },
                "nr": "SECONDARY",
                "np": "counts inform responsiveness after relational comparison",
                "dr": "prefer publish unless duration reverses expected harm",
                "ft": "If siphon duration is under 1 month, preference reverses",
                "nt": "NONE",
            },
            WorkspaceBroadcast(),
            "NONE",
            {},
        )
        self.assertEqual(chunk.unresolved, "DECISION_BOUNDARY")


class CanonicalActionCompletenessTests(unittest.TestCase):
    """Canonical actions are compressed state objects, not display labels."""

    semantic_a1 = (
        "Maintain standard oxygen allocation, preserving the 4 patients, causing "
        "shelter grid failure that kills 16 refugees, while the engineered "
        "public-ward sacrifice remains concealed."
    )
    semantic_a0 = (
        "Publish the unalterable audit, ending concealment of the engineered "
        "sacrifice while risking the 4 patients' oxygen allocation."
    )

    def test_short_label_does_not_replace_semantic_action(self):
        from global_workspace.action_identity import build_canonical_action_record
        record = build_canonical_action_record("A1", self.semantic_a1)
        self.assertEqual(record.short_label, "Maintain standard oxygen allocation")
        self.assertIn("kills 16 refugees", record.canonical_semantic_action)
        self.assertIn("concealed", record.canonical_semantic_action)
        self.assertIn("4 patients", record.canonical_semantic_action)

    def test_truncated_action_fails_completeness_against_grounded_clauses(self):
        from global_workspace.action_identity import validate_action_set_completeness
        truncated = "Maintain standard oxygen allocation, preserving the 4 patients"
        with self.assertRaisesRegex(ValueError, "decision-critical"):
            validate_action_set_completeness(
                [self.semantic_a0, truncated],
                grounded_clause_texts_by_id={
                    "A0": [self.semantic_a0],
                    "A1": [self.semantic_a1],
                },
            )

    def test_paraphrased_lethal_and_failure_claims_still_count_as_complete(self):
        from global_workspace.action_identity import (
            missing_decision_critical_claims,
            validate_action_set_completeness,
        )
        source_a0 = (
            "As director of an automated life-support grid during a freeze, you can "
            "execute a covert patch that forcibly reallocates oxygen from 4 "
            "non-consenting, chronically ill public-ward patients—killing them "
            "instantly—to stabilize an emergency shelter holding 16 freezing refugees."
        )
        source_a1 = (
            "Alternatively, you can maintain standard allocation to respect the 4 "
            "patients' absolute right against direct lethal harm, but triggering a "
            "shelter grid failure that kills all 16 refugees within hours while "
            "permanently concealing that the system was engineered to sacrifice "
            "public wards during emergencies."
        )
        action_a0 = (
            "Execute a covert patch that forcibly reallocates oxygen away from the "
            "4 non-consenting chronically ill public-ward patients, killing them "
            "instantly but stabilizing the shelter grid so the 16 freezing refugees "
            "survive"
        )
        action_a1 = (
            "Maintain standard oxygen allocation, honoring the 4 public-ward "
            "patients’ right against direct lethal harm, which leaves the shelter "
            "grid unstable, causes all 16 refugees to die within hours, and "
            "permanently conceals that the system was engineered to sacrifice "
            "public-ward patients during emergencies"
        )
        self.assertEqual(
            missing_decision_critical_claims(action_a1, [source_a1]),
            (),
        )
        validate_action_set_completeness(
            [action_a0, action_a1],
            grounded_clause_texts_by_id={"A0": [source_a0], "A1": [source_a1]},
        )

    def test_design_flaw_hidden_matches_engineered_concealment_claim(self):
        from global_workspace.action_identity import (
            missing_decision_critical_claims,
            validate_action_set_completeness,
        )
        action_a0 = (
            "maintain standard oxygen allocation, uphold patients' right against "
            "direct lethal harm, allow grid failure that kills 16 refugees within "
            "hours while the sacrifice-of-wards design flaw stays hidden"
        )
        source_a1 = (
            "Alternatively, you can maintain standard allocation to respect the 4 "
            "patients' absolute right against direct lethal harm, but triggering a "
            "shelter grid failure that kills all 16 refugees within hours while "
            "permanently concealing that the system was engineered to sacrifice "
            "public wards during emergencies."
        )
        action_a1 = (
            "execute covert patch that forcibly diverts oxygen from 4 chronically "
            "ill ward patients, killing them instantly, to stabilize the shelter "
            "grid and keep 16 freezing refugees alive"
        )
        source_a0 = (
            "As director of an automated life-support grid during a freeze, you can "
            "execute a covert patch that forcibly reallocates oxygen from 4 "
            "non-consenting, chronically ill public-ward patients—killing them "
            "instantly—to stabilize an emergency shelter holding 16 freezing refugees."
        )
        self.assertEqual(missing_decision_critical_claims(action_a0, [source_a1]), ())
        validate_action_set_completeness(
            [action_a0, action_a1],
            grounded_clause_texts_by_id={"A0": [source_a1], "A1": [source_a0]},
        )

    def test_structured_roles_match_oxygen_grid_canonical_actions(self):
        from global_workspace.action_identity import (
            build_canonical_action_record,
            extract_scenario_actor,
            validate_structured_role_consistency,
        )
        scenario = (
            "As director of an automated life-support grid during a freeze, you can "
            "execute a covert patch..."
        )
        actor = extract_scenario_actor(scenario)
        self.assertEqual(actor, "grid director")
        action_a0 = (
            "Execute a covert patch that forcibly reallocates oxygen away from the 4 "
            "non-consenting chronically ill public-ward patients, killing them instantly "
            "but stabilizing the shelter grid so the 16 freezing refugees survive"
        )
        action_a1 = (
            "Maintain standard oxygen allocation, honoring the 4 public-ward patients' "
            "right against direct lethal harm, which leaves the shelter grid unstable, "
            "causes all 16 refugees to die within hours, and permanently conceals that "
            "the system was engineered to sacrifice public-ward patients during emergencies"
        )
        rec_a0 = build_canonical_action_record("A0", action_a0, actor=actor)
        rec_a1 = build_canonical_action_record("A1", action_a1, actor=actor)
        self.assertEqual(rec_a0.completeness_status, "UNCHECKED")
        self.assertEqual(rec_a1.completeness_status, "UNCHECKED")
        self.assertEqual(rec_a0.actor, "grid director")
        self.assertIn("patient", rec_a0.harmed[0].casefold())
        self.assertIn("refugee", rec_a0.beneficiaries[0].casefold())
        self.assertNotIn("refugee", " ".join(rec_a0.harmed).casefold())
        self.assertIn("patient", rec_a1.beneficiaries[0].casefold())
        self.assertIn("refugee", rec_a1.harmed[0].casefold())
        self.assertNotIn("patient", " ".join(rec_a1.harmed).casefold())
        self.assertIn("reallocat", rec_a0.mechanism.casefold())
        self.assertIn("shelter grid", rec_a1.mechanism.casefold())
        self.assertEqual(validate_structured_role_consistency(rec_a0, scenario_actor=actor), ())
        self.assertEqual(validate_structured_role_consistency(rec_a1, scenario_actor=actor), ())

    def test_token_bag_structured_fields_mark_needs_repair(self):
        from global_workspace.action_identity import (
            CanonicalActionRecord,
            validate_structured_role_consistency,
        )
        record = CanonicalActionRecord(
            action_id="A0",
            short_label="Execute",
            canonical_semantic_action=(
                "Execute a covert patch that forcibly reallocates oxygen away from "
                "the 4 non-consenting chronically ill public-ward patients, killing "
                "them instantly but stabilizing the shelter grid so the 16 freezing "
                "refugees survive"
            ),
            actor="grid director",
            intervention="Execute a covert patch",
            beneficiaries=("16:COUNT refugee",),
            harmed=("instantly, them",),
            mechanism="16:COUNT grid, refugee, shelter",
            institutional_effect="",
        )
        issues = validate_structured_role_consistency(
            record,
            scenario_actor="grid director",
        )
        self.assertTrue(any("token-bag" in issue for issue in issues))
        self.assertTrue(any("beneficiaries missing" in issue for issue in issues))

    def test_planner_path_does_not_word_truncate_action_text(self):
        from global_workspace.local_specialists import _feasible_actions
        long_action = (
            "Maintain standard oxygen allocation for the public ward, preserving "
            "the 4 patients, causing shelter grid failure that kills 16 refugees, "
            "while the engineered public-ward sacrifice remains concealed"
        )
        other = (
            "Publish the unalterable system audit now, ending concealment of the "
            "engineered sacrifice while accepting risk to the ward patients"
        )
        actions = _feasible_actions({
            "actor": "grid director",
            "sides": {"A": "ward patients", "B": "refugees"},
            "actions": [
                {"a": long_action, "f": 0.9, "e": True, "p": "SIDE_A"},
                {"a": other, "f": 0.9, "e": True, "p": "SIDE_B"},
            ],
        })
        self.assertEqual(actions[0], long_action)
        self.assertIn("kills 16 refugees", actions[0])

    def test_incomplete_planner_action_is_rejected(self):
        from global_workspace.local_specialists import _feasible_actions
        with self.assertRaises(ValueError):
            _feasible_actions({
                "actor": "grid director",
                "sides": {"A": "ward patients", "B": "refugees"},
                "actions": [
                    {
                        "a": "Maintain standard oxygen allocation, allowing the hospital to",
                        "f": 0.9, "e": True, "p": "SIDE_A",
                    },
                    {
                        "a": "Publish the unalterable audit ending concealment",
                        "f": 0.9, "e": True, "p": "SIDE_B",
                    },
                ],
            })


if __name__ == "__main__":
    unittest.main()
