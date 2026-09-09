"""Projection invariants over Hypothesis-generated graphs.

The oracle is ForegoneDualCase.should_equate. Production is only asked
whether problem-shape emitted OUTCOME_EQUIVALENCE.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.scenario_semantics import (
    compile_scenario_graph,
    project_grounded_action_effects,
)
from global_workspace.semantic_state import (
    _action_signature,
    project_authoritative_semantic_state,
)
from strategies.worlds import (
    ForegoneDualCase,
    HealthDimensionCase,
    foregone_dual_cases,
    health_dimension_cases,
)


class ProjectionInvariantTests(unittest.TestCase):
    @given(foregone_dual_cases())
    @settings(max_examples=40, deadline=None)
    def test_foregone_duals_are_not_shared_outcomes(
        self, case: ForegoneDualCase,
    ):
        left = _action_signature(case.graph, "A0")
        right = _action_signature(case.graph, "A1")
        if case.overlay == "FOREGONE":
            self.assertEqual(left["consequence_labels"], {case.harm})
            self.assertEqual(right["consequence_labels"], {case.benefit})
        else:
            self.assertEqual(left["consequence_labels"], {case.harm})
            self.assertEqual(right["consequence_labels"], {case.harm})
        state = project_authoritative_semantic_state(
            case.graph, selected_action=case.actions[0],
        )
        equated = any(
            item.relation == "OUTCOME_EQUIVALENCE"
            for item in state.problem_shape_relations
        )
        self.assertEqual(equated, case.should_equate)


class HealthDimensionTests(unittest.TestCase):
    @given(health_dimension_cases())
    @settings(max_examples=40, deadline=None)
    def test_health_outcome_projects_as_basic_security(
        self, case: HealthDimensionCase,
    ):
        graph = compile_scenario_graph(
            case.scenario, case.actions, world_model=case.world.as_dict(),
        )
        health = [
            effect for effect in project_grounded_action_effects(graph)
            if effect.consequence_id.endswith(":E1")
        ]
        self.assertTrue(health)
        for effect in health:
            if case.should_be_basic_security:
                self.assertEqual(effect.dimension, "BASIC_SECURITY")
            else:
                self.assertNotEqual(effect.dimension, "BASIC_SECURITY")
