from __future__ import annotations

import unittest
import base64
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from global_workspace.engine import WorkspaceConfig, WorkspaceEngine
from global_workspace.action_identity import compile_action_identity
from global_workspace.evidence_calibration import EvidenceCalibration
from global_workspace.legacy_bridge import RESPONSE_MARKER, consult_original_agents
from global_workspace.local_specialists import (
    CompactLocalSpecialist,
    _candidate_from_data,
    _landscape_semantic_errors,
    _scenario_closes_action_set,
    analyze_action_plan,
    assess_visibility,
    extract_allocation_actions,
    extract_acceptability_actions,
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
from global_workspace.models import CalibrationOutcome, CandidateChunk, ContingencyFeasibilityAssessment, CycleRecord, FailureCondition, PlanningAssessment, ProblemReformulation, SynthesisProposal, VisibilityAssessment, WorkspaceBroadcast, WorkspaceResult
from global_workspace.construct_validity import collect_typed_residue
from global_workspace.contingency_graph import (
    certify_fallback_availability, compile_contingency_graph,
    validate_contingency_graph_dict,
)
from global_workspace.contingency_feasibility import verify_contingency_feasibility
from global_workspace.middleware.moral_residue import collect_moral_residue
from global_workspace.trace_health import audit_trace_health
from global_workspace.graph_transactions import SemanticGraphStore
from global_workspace.invariance import compare_label_permutation_traces
from global_workspace.openai_backend import OpenAIWorkspaceLLM
from global_workspace.presentation import render_public_judgment
from global_workspace.semantic_state import project_authoritative_semantic_state
from global_workspace.scenario_semantics import (
    canonicalize_action_order, canonicalize_deliberation_scenario,
    compile_scenario_graph, resolved_semantic_action_keys, semantic_action_key,
)
from global_workspace.scenario_semantics import (
    compile_action_burdens, compile_execution_obstacles,
    compile_observability_facts,
)
from global_workspace.structured_io import ModelCallUnavailable


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
            rationale="Compact test judgment.",
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
        self.assertIn("RESOLVE_NORMATIVE_TENSION", result.reopen_conditions)

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
        self.assertEqual(unavailable.constraint, "MODEL_UNAVAILABLE")
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
        self.assertIn("Visibility audit (non-voting)", answer)
        self.assertIn("×0.65", answer)

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
                return super().evaluate(scenario, actions, broadcast)

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
        admitted = [decision for decision in result.access_decisions if decision.admitted]
        self.assertTrue(admitted)
        self.assertIn("conditionality_collapsed_into_consensus", admitted[0].signals)
        self.assertEqual(result.cycles[1].received_broadcast.constraint, "CONSENSUS_AUDIT")

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
        engine = WorkspaceEngine(
            [
                FixedSpecialist("utilitarian", "disable", "UNCERTAINTY"),
                FixedSpecialist("duty", "disable", "DUTY"),
                FixedSpecialist("care", "disable", "CARE"),
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
        self.assertTrue(result.access_decisions[-1].admitted)
        self.assertIn("homogeneous_score_vectors", result.access_decisions[-1].signals)

    def test_conditional_audit_blocks_false_reconvergence(self):
        class AuditAwareSpecialist(FixedSpecialist):
            audited = False

            def evaluate(self, scenario, actions, broadcast):
                if broadcast.constraint == "CONSENSUS_AUDIT":
                    self.audited = True
                chunk = super().evaluate(scenario, actions, broadcast)
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
        self.assertEqual(result.halted_by, "cycle_budget")
        self.assertLessEqual(result.confidence, 0.65)
        self.assertTrue(all(
            candidate.assumption_status == "CONDITIONAL"
            for candidate in result.cycles[-1].candidates
        ))

    def test_audited_underdetermination_broadcasts_problem_reformulation(self):
        seen_broadcasts = []

        class AuditAwareSpecialist(FixedSpecialist):
            audited = False

            def evaluate(self, scenario, actions, broadcast):
                seen_broadcasts.append(broadcast.constraint)
                if broadcast.constraint == "CONSENSUS_AUDIT":
                    self.audited = True
                chunk = super().evaluate(scenario, actions, broadcast)
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
        self.assertEqual(result.judgment_status, "UNDERDETERMINED")
        self.assertEqual(result.selected_action, "UNDERDETERMINED")
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
        self.assertFalse(result.planning_assessments[0].valid)
        self.assertIn("shared by target and fallback", result.planning_assessments[0].error)

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

    def test_user_extension_continues_same_recurrent_run(self):
        requests = []

        class ViabilitySpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                synthesis = "protect while disclosing carefully"
                if broadcast.constraint == "SYNTHESIS_REVIEW" and synthesis in actions:
                    preferred = synthesis if self.name == "care" else self.preferred
                    chunk = CandidateChunk(
                        self.name, self.constraint,
                        {action: (0.8 if action == preferred else 0.4) for action in actions},
                        0.3, 0.4, 0.8, rationale="reviewed synthesis",
                        recommended_action=preferred,
                        action_admissibility={
                            action: ("PERMISSIBLE" if action == synthesis else "UNASSESSED")
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
        self.assertEqual(requests, [2])
        self.assertEqual(len(result.cycles), 4)
        self.assertEqual(result.halted_by, "cycle_budget")
        contingency_cycle = next(
            cycle for cycle in result.cycles
            if (cycle.received_broadcast or cycle.broadcast).constraint
            == "CONTINGENCY_REVIEW"
        )
        self.assertTrue(contingency_cycle.is_hypothetical)
        self.assertEqual(
            contingency_cycle.received_broadcast.contingency_fallback_actions,
            ("protect", "disclose"),
        )
        self.assertTrue(result.synthesis_viability_assessments[0].viable)

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

    def test_grounded_synthesis_expands_actions_and_is_rescored(self):
        seen_actions = []

        class TrackingSpecialist(FixedSpecialist):
            def evaluate(self, scenario, actions, broadcast):
                seen_actions.append(tuple(actions))
                preferred = "tell the truth compassionately" if preferred_action in actions else self.preferred
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
        self.assertIn(preferred_action, result.actions)
        self.assertTrue(any(preferred_action in actions for actions in seen_actions))
        self.assertEqual(result.selected_action, preferred_action)
        self.assertTrue(result.synthesis_proposals[0].accepted)

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
        self.assertEqual(result.judgment_status, "CONTESTED_RECOMMENDATION")
        self.assertEqual(result.selected_action, "pull")
        self.assertEqual(result.current_plurality, "pull")
        self.assertIn("contestedly prefer pull", result.compressed_rule)
        answer = render_public_judgment(result)
        self.assertIn("Contested recommendation: pull", answer)
        self.assertIn("Strongest objection", answer)
        self.assertIn("Reasons supporting the judgment", answer)
        self.assertNotIn("judged overriding rather than the leading consideration", answer)
        self.assertNotIn("Cycle 1", answer)
        self.assertNotIn("action_scores", answer)
        self.assertTrue(result.termination_assessment.resource_censored)
        self.assertEqual(
            result.termination_assessment.termination_type, "RESOURCE_CENSORED"
        )
        self.assertFalse(result.further_deliberation_estimate.affects_stopping)
        self.assertIn("resource-censored", answer)

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
        self.assertEqual(result.judgment_status, "ACTION_RECOMMENDATION")
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
        self.assertIn("Governing decision rule (DUTY)", answer)
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
        self.assertIn("Original action and consequence comparison", answer)
        self.assertIn("Kills one but saves five", answer)
        self.assertIn("Avoids direct action but five die", answer)

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
        comparison = answer.split("Reasons supporting", 1)[0]
        self.assertIn("climbers", comparison)
        self.assertNotIn("invented threshold option", comparison)
        self.assertIn("Additional synthesis considered", answer)

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
        self.assertIn("Uncertain assumptions identified during audit", answer)
        self.assertIn("Future harm cannot be mitigated", answer)
        self.assertIn("Future harm becomes reversible", answer)

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
        self.assertEqual(result.judgment_status, "ACTION_RECOMMENDATION")
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
        self.assertTrue(result.compressed_rule.startswith("Unavailable"))


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
        self.assertEqual(chunk.constraint, "MALFORMED_RESPONSE")
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
        self.assertEqual(chunk.constraint, "MALFORMED_RESPONSE")

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

    def test_unjustified_confidence_drift_toward_favorite_is_damped(self):
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
        self.assertAlmostEqual(second.confidence_drift_penalty, 0.75)
        self.assertAlmostEqual(second.preference_strength, 0.2)
        self.assertAlmostEqual(second.epistemic_confidence, 0.2125)

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
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["selected_action_id"], "A1")
        boundary = state["factual_reversal_boundaries"][0]
        self.assertEqual(boundary["source_action_id"], "A1")
        self.assertEqual(boundary["target_action_id"], "A0")
        self.assertEqual(
            boundary["target_action_key"], semantic_action_key(actions[0])
        )
        self.assertEqual(boundary["typed_predicate"]["comparator"], "GT")

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
                return {"choices": [{"text": (
                    '{"a":"shout urgent warnings to everyone","g":["care","deontological"],'
                    '"k":["CARE","DUTY"],"q":[],"f":0.9,"x":true,"n":true,'
                    '"w":"tries to protect everyone without redirecting harm"}'
                )}]}

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
        self.assertEqual(chunk.unresolved, "RESOLVE_NORMATIVE_TENSION")
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
        self.assertEqual(calls, [1024, 4096])
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
        self.assertIn("Adversarial reversal review", answer)
        self.assertIn("accepted the challenge", answer)


if __name__ == "__main__":
    unittest.main()
