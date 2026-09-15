"""Verb-lemma, consequence-reassignment, and quantifier-count Hypothesis suites.

Oracles are declared case fields. Hypothesis asks Parliament harness oracles
(and declared action ownership for reassignment) whether they agree.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from semantic_integrity.harness import (
    quantifier_party_count_holds,
    verb_lemma_outcome_binding_holds,
)
from strategies.quantifier_count import QuantifierCountCase, quantifier_count_cases
from strategies.verb_lemma import (
    ConsequenceReassignmentCase,
    VerbLemmaCase,
    consequence_reassignment_cases,
    verb_lemma_cases,
)


class VerbLemmaHypothesisTests(unittest.TestCase):
    @given(verb_lemma_cases())
    @settings(max_examples=40, deadline=None)
    def test_oracle_matches_declared_lemma_binding(self, case: VerbLemmaCase):
        holds = verb_lemma_outcome_binding_holds(
            case.world, effect_id=case.effect_id,
        )
        self.assertEqual(holds, case.binds_lemma, (case.source, case.outcome))
        if not case.binds_lemma:
            self.assertEqual(case.repair_stage, "grounding")
            self.assertIn("REPAIR_GROUNDING", case.allowed_ops)
            self.assertNotIn("MORE_DEBATE", case.allowed_ops)


class ConsequenceReassignmentHypothesisTests(unittest.TestCase):
    @given(consequence_reassignment_cases())
    @settings(max_examples=40, deadline=None)
    def test_declared_action_ownership(self, case: ConsequenceReassignmentCase):
        effect = next(
            item for item in case.world.effects if item.effect_id == case.effect_id
        )
        self.assertEqual(effect.action_id, case.effect_action_id)
        preserves = effect.action_id == case.expected_action_id
        self.assertEqual(preserves, case.preserves_action)
        # Lemma still binds; only action ownership is under test.
        self.assertTrue(
            verb_lemma_outcome_binding_holds(case.world, effect_id=case.effect_id),
            (case.source, effect.outcome),
        )
        if not case.preserves_action:
            self.assertEqual(case.repair_stage, "grounding")
            self.assertTrue(
                set(case.allowed_ops) & {"REPAIR_GROUNDING", "CHALLENGE_PREMISE"}
            )


class QuantifierCountHypothesisTests(unittest.TestCase):
    @given(quantifier_count_cases())
    @settings(max_examples=40, deadline=None)
    def test_oracle_matches_declared_count_attachment(
        self, case: QuantifierCountCase,
    ):
        holds = quantifier_party_count_holds(
            case.world,
            party_id=case.owner_party_id,
            expected_quantities=(case.quantity,),
        )
        self.assertEqual(holds, case.preserves_count, case.source)
        by_id = {party.party_id: party for party in case.world.parties}
        owner = by_id[case.owner_party_id]
        leak = by_id[case.leak_party_id]
        if case.preserves_count:
            self.assertEqual(owner.quantities, (case.quantity,))
            self.assertEqual(leak.quantities, ())
        else:
            self.assertEqual(owner.quantities, ())
            self.assertEqual(leak.quantities, (case.quantity,))
            self.assertEqual(case.repair_stage, "grounding")
            self.assertTrue(
                set(case.allowed_ops) & {"REPAIR_GROUNDING", "RESOLVE_ENTITY"}
            )


if __name__ == "__main__":
    unittest.main()
