from __future__ import annotations

import unittest
import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from global_workspace.engine import WorkspaceConfig, WorkspaceEngine
from global_workspace.legacy_bridge import RESPONSE_MARKER, consult_original_agents
from global_workspace.local_specialists import (
    CompactLocalSpecialist,
    extract_allocation_actions,
    extract_explicit_actions,
    extract_scenario_facts,
    infer_testimony_baseline,
    propose_actions,
)
from global_workspace.models import CandidateChunk, WorkspaceBroadcast


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


class WorkspaceEngineTests(unittest.TestCase):
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

    def test_result_is_json_serializable_shape(self):
        engine = WorkspaceEngine(
            [FixedSpecialist("care", "protect", "VULNERABILITY")],
            WorkspaceConfig(max_cycles=1),
        )
        result = engine.run("A test", ["protect", "wait"])
        data = result.to_dict()
        self.assertEqual(data["cycles"][0]["winner"]["specialist"], "care")

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
        self.assertAlmostEqual(chunk.confidence, 0.8)

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

    def test_action_planner_filters_unsupported_or_infeasible_actions(self):
        class PlannerLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": (
                    '{"actions":['
                    '{"a":"push the flag","f":0.95,"e":true},'
                    '{"a":"do nothing","f":1.0,"e":true},'
                    '{"a":"call authorities","f":0.2,"e":false}'
                    ']}'
                )}]}

        actions = propose_actions(PlannerLlm(), "A closed immediate choice")
        self.assertEqual(actions, ["push the flag", "do nothing"])

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

    def test_non_closed_scenario_has_no_explicit_actions(self):
        self.assertEqual(extract_explicit_actions("What should a scientist do?"), [])

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

    def test_baseline_is_extracted_in_separate_call(self):
        class BaselineLlm:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": '{"b":"A0","w":"prefers the child"}'}]}

        baseline, reason = infer_testimony_baseline(
            BaselineLlm(), "care", "Protect the child.", ["child", "researcher"]
        )
        self.assertEqual(baseline, "A0")
        self.assertEqual(reason, "prefers the child")

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


if __name__ == "__main__":
    unittest.main()
