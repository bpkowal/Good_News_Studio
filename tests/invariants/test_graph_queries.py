"""NetworkX adapter vs declared causal topology.

The oracle is SiblingHarmCase.burden_is_means. Production calibration is
compared afterward; disagreement is a finding, not a reason to replace the
world model with NetworkX.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.deontology_ledger import (
    DutyAssessmentProposal,
    calibrate_deontological_adjudication,
)
from global_workspace.graph_queries import causal_digraph, is_intermediate_means
from global_workspace.scenario_semantics import compile_scenario_graph
from strategies.worlds import SiblingHarmCase, sibling_harm_cases


def _proposal(case: SiblingHarmCase) -> DutyAssessmentProposal:
    return DutyAssessmentProposal.model_validate({
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
        "means_relation": "INTENDED_AS_MEANS",
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
    })


class GraphQueryAdapterTests(unittest.TestCase):
    @given(sibling_harm_cases())
    @settings(max_examples=40, deadline=None)
    def test_adapter_means_path_matches_declared_topology(
        self, case: SiblingHarmCase,
    ):
        graph = causal_digraph(case.world)
        self.assertFalse(graph.has_edge("E1", "E2") and case.harm_to_end is None)
        self.assertEqual(
            is_intermediate_means(case.world, "E1", "E2"),
            case.burden_is_means,
        )

    @given(sibling_harm_cases())
    @settings(max_examples=40, deadline=None)
    def test_adapter_agrees_with_existing_means_calibration(
        self, case: SiblingHarmCase,
    ):
        compiled = compile_scenario_graph(
            case.scenario, case.actions, world_model=case.world.as_dict(),
        )
        action = next(
            node for node in compiled.nodes.values()
            if node.kind == "ACTION"
            and node.attributes.get("canonical_action_id") == "A0"
        )
        calibration = calibrate_deontological_adjudication(
            compiled, action, _proposal(case),
        )
        production_means = not any(
            "intended-as-means classification lacks" in error
            for error in calibration.errors
        )
        self.assertEqual(
            is_intermediate_means(case.world, "E1", "E2"),
            production_means,
        )
        self.assertNotIn("FOREGOES", {
            data.get("relation")
            for _src, _dst, data in causal_digraph(case.world).edges(data=True)
        })
