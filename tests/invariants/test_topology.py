"""Action-mediated vs exogenous process over Hypothesis-generated worlds.

The oracle is TopologyCase.should_admit. Production is only asked whether
that declared graph is complete. Ancestry and tautological-if gates stay.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.world_state import (
    parse_world_model,
    validate_world_completeness,
    validate_world_model,
    world_model_as_parse_payload,
)
from strategies.topology import TopologyCase, topology_cases


def _admission_errors(world) -> list[str]:
    structural, _contradictions = validate_world_model(world, action_ids=["A0"])
    complete = validate_world_completeness(world, action_ids=["A0"])
    return [*structural, *complete]


class ActionMediatedVersusExogenousTests(unittest.TestCase):
    @given(topology_cases())
    @settings(max_examples=40, deadline=None)
    def test_declared_topology_matches_admission(self, case: TopologyCase):
        errors = _admission_errors(case.world)
        if case.should_admit:
            self.assertEqual(errors, [])
            payload = world_model_as_parse_payload(case.world)
            parsed = parse_world_model(
                payload,
                clauses=[{"clause_id": "C0", "text": case.source}],
                action_ids=["A0"],
            )
            self.assertEqual(_admission_errors(parsed), [])
            return
        self.assertTrue(errors)
        joined = " ".join(errors)
        self.assertTrue(
            "ancestry never reaches a DIRECT act" in joined
            or "independent stochastic" in joined
            or "inbound path" in joined
            or "do not invent" in joined.lower()
            or "event_effect_id" in joined
            or "ordinary causal parent" in joined,
            errors,
        )
        if case.kind == "FALSE_CAUSE":
            self.assertIn("stochastic", joined)
            self.assertIn("Do not parent", joined)
        if case.kind == "ACTION_CAUSED_UNLINKED":
            self.assertIn("action-caused", joined)
            self.assertIn("inbound", joined)
