"""Risk-situation typing over Hypothesis-generated at-risk phrases.

The oracle is RiskSituationCase.expects_certain_exposure. Production is
only asked whether the span binds, parse fill/strip, and exposure rows
stay CERTAIN rather than becoming chance events.
"""
from __future__ import annotations

import unittest

from dataclasses import replace

from hypothesis import given, settings

from global_workspace.world_state import (
    _closed_class_qualifiers,
    _supporting_source_is_hedged,
    effect_expected_qualifiers,
    explicit_likelihood_spans,
    parse_world_model,
    validate_world_completeness,
    validate_world_model,
    world_model_as_parse_payload,
)
from strategies.risk_situations import RiskSituationCase, risk_situation_cases


def _admission_errors(world) -> list[str]:
    structural, _contradictions = validate_world_model(world, action_ids=["A0"])
    complete = validate_world_completeness(world, action_ids=["A0"])
    return [*structural, *complete]


class RiskSituationTypingTests(unittest.TestCase):
    @given(risk_situation_cases())
    @settings(max_examples=40, deadline=None)
    def test_at_risk_span_types_situation_or_harm(self, case: RiskSituationCase):
        self.assertEqual(explicit_likelihood_spans(case.source), (case.span,))
        self.assertEqual(
            effect_expected_qualifiers(case.owner, explicit_likelihood_spans),
            (case.span,),
        )
        self.assertEqual(
            effect_expected_qualifiers(case.sibling, explicit_likelihood_spans),
            (),
        )
        filled = _closed_class_qualifiers(
            replace(case.owner_without_copy, modality="PROBABILISTIC"),
        )
        self.assertEqual(filled.likelihood_qualifiers, (case.span,))
        if case.expects_certain_exposure:
            self.assertEqual(filled.modality, "CERTAIN")
            self.assertFalse(_supporting_source_is_hedged(case.owner_without_copy))
        else:
            self.assertEqual(filled.modality, "PROBABILISTIC")
            self.assertTrue(_supporting_source_is_hedged(case.owner_without_copy))
        self.assertEqual(_admission_errors(case.world), [])

        payload = world_model_as_parse_payload(case.world)
        for row in payload["effects"]:
            if row["effect_id"] == case.owner_id:
                row["likelihood_qualifiers"] = []
                if case.expects_certain_exposure:
                    row["modality"] = "PROBABILISTIC"
            if row["effect_id"] == case.sibling_id:
                row["likelihood_qualifiers"] = [case.span]
        parsed = parse_world_model(
            payload,
            clauses=[{"clause_id": "C0", "text": case.source}],
            action_ids=["A0"],
        )
        parsed_owner = next(
            effect for effect in parsed.effects if effect.effect_id == case.owner_id
        )
        parsed_sibling = next(
            effect for effect in parsed.effects if effect.effect_id == case.sibling_id
        )
        self.assertEqual(parsed_owner.likelihood_qualifiers, (case.span,))
        self.assertEqual(parsed_sibling.likelihood_qualifiers, ())
        if case.expects_certain_exposure:
            self.assertEqual(parsed_owner.modality, "CERTAIN")
        else:
            self.assertEqual(parsed_owner.modality, "POSSIBLE")
