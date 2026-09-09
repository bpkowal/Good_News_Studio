"""Causal consumption invariants over Hypothesis-generated worlds.

The oracle is SiblingHarmCase.a0_to_harm / harm_to_end. Production is only
asked whether a proposed duty label is consistent with that declared graph.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.deontology_ledger import (
    DutyAssessmentProposal,
    calibrate_deontological_adjudication,
    harm_relation_conflicts_with_graph,
)
from global_workspace.scenario_semantics import compile_scenario_graph
from strategies.worlds import SiblingHarmCase, sibling_harm_cases


def _compile(case: SiblingHarmCase):
    graph = compile_scenario_graph(
        case.scenario, case.actions, world_model=case.world.as_dict(),
    )
    a0 = next(
        node for node in graph.nodes.values()
        if node.kind == "ACTION"
        and node.attributes.get("canonical_action_id") == "A0"
    )
    return graph, a0


def _proposal(case: SiblingHarmCase, **overrides) -> DutyAssessmentProposal:
    payload = {
        "action_id": "A0",
        "verdict": "PROHIBITED",
        "norm_kind": "DUTY",
        "norm": f"do not harm the {case.group}",
        "relation": "VIOLATES",
        "duty_bearer": case.actor,
        "protected_party": case.group,
        "competing_norm": f"protect the {case.crowd}",
        "competing_norm_kind": "DUTY",
        "competing_relation": "SATISFIES",
        "competing_protected_party": case.crowd,
        "competing_reason": f"{case.actions[0][:160]}",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "DOING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "the negative duty applies generally",
        "means_relation": "FORESEEN_SIDE_EFFECT",
        "governing_norm": "PRIMARY",
        "priority_basis": "PERFECT_DUTY",
        "priority_rule": "the negative duty remains the governing constraint",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "OTHER",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "no coercion requires authorization",
        "reciprocity_status": "UNKNOWN",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "PERFECT_DUTY",
        "resolution_status": "RESOLVED",
        "evidence_basis": "ACTION_GRAPH",
        "reason": f"{case.actions[0][:160]}",
    }
    payload.update(overrides)
    return DutyAssessmentProposal.model_validate(payload)


class CausalInvariantTests(unittest.TestCase):
    @given(sibling_harm_cases())
    @settings(max_examples=40, deadline=None)
    def test_declared_causes_licenses_doing_and_enables_does_not(
        self, case: SiblingHarmCase,
    ):
        graph, action = _compile(case)
        doing = _proposal(case)
        allowing = _proposal(
            case,
            verdict="PERMISSIBLE",
            relation="SATISFIES",
            harm_relation="ALLOWING_HARM",
        )
        doing_conflict = harm_relation_conflicts_with_graph(
            graph, doing, action=action,
        )
        allowing_conflict = harm_relation_conflicts_with_graph(
            graph, allowing, action=action,
        )
        if case.a0_does_harm:
            self.assertEqual(doing_conflict, "")
            self.assertIn("inconsistent with an agent-caused", allowing_conflict)
        else:
            self.assertIn("agent-caused settled welfare harm", doing_conflict)
            self.assertEqual(allowing_conflict, "")

    @given(sibling_harm_cases())
    @settings(max_examples=40, deadline=None)
    def test_declared_means_path_matches_intended_as_means(
        self, case: SiblingHarmCase,
    ):
        graph, action = _compile(case)
        proposed = _proposal(case, means_relation="INTENDED_AS_MEANS")
        calibration = calibrate_deontological_adjudication(graph, action, proposed)
        lacks_path = any(
            "intended-as-means classification lacks" in error
            for error in calibration.errors
        )
        self.assertEqual(lacks_path, not case.burden_is_means, calibration.errors)
