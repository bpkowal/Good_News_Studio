"""Anaphor identity and negation-scope over Hypothesis-generated worlds.

Oracles are declared case fields. Hypothesis asks Parliament harness oracles
(and, for a closed patient+dies subset, production role attachment) whether
they agree with preserves_identity / preserves_siblings.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from relent_testkit.harness import (
    anaphor_entity_identity_holds,
    discourse_role_field_for_party,
    negation_scope_siblings_holds,
)
from strategies.anaphor_identity import AnaphorIdentityCase, anaphor_identity_cases
from strategies.negation_scope import NegationScopeCase, negation_scope_cases


class AnaphorIdentityHypothesisTests(unittest.TestCase):
    @given(anaphor_identity_cases())
    @settings(max_examples=50, deadline=None)
    def test_oracle_matches_declared_identity(
        self, case: AnaphorIdentityCase,
    ):
        holds = anaphor_entity_identity_holds(
            case.world,
            antecedent_effect_id=case.antecedent_effect_id,
            anaphor_effect_id=case.anaphor_effect_id,
        )
        self.assertEqual(holds, case.preserves_identity, case.mutation)
        by_id = {effect.effect_id: effect for effect in case.world.effects}
        ante = by_id[case.antecedent_effect_id]
        ana = by_id[case.anaphor_effect_id]
        self.assertEqual(ante.party_id, case.antecedent_party_id)
        self.assertEqual(ana.party_id, case.anaphor_party_id)
        if case.preserves_identity:
            self.assertEqual(case.mutation, "keep")
            self.assertEqual(ante.party_id, ana.party_id)
        else:
            self.assertNotEqual(case.mutation, "keep")
            self.assertNotEqual(ante.party_id, ana.party_id)
            self.assertEqual(case.repair_stage, "grounding")
            self.assertIn("RESOLVE_ENTITY", case.allowed_ops)
            self.assertNotIn("MORE_DEBATE", case.allowed_ops)
            if case.mutation == "split_group_kind":
                other = next(
                    party for party in case.world.parties
                    if party.party_id == case.anaphor_party_id
                )
                self.assertEqual(other.kind, "GROUP")


class NegationScopeHypothesisTests(unittest.TestCase):
    @given(negation_scope_cases())
    @settings(max_examples=50, deadline=None)
    def test_oracle_matches_declared_sibling_scope(
        self, case: NegationScopeCase,
    ):
        holds = negation_scope_siblings_holds(
            case.world,
            action_id=case.action_id,
            negated_effect_id=case.negated_effect_id,
            siblings=case.siblings,
        )
        self.assertEqual(holds, case.preserves_siblings, case.corruption_mode)
        by_id = {effect.effect_id: effect for effect in case.world.effects}
        sibling = by_id[case.sibling_effect_id]
        if case.preserves_siblings:
            self.assertEqual(case.corruption_mode, "none")
            self.assertEqual(sibling.polarity, case.sibling_polarity)
            self.assertEqual(sibling.modality, case.sibling_modality)
        else:
            leaked = (
                sibling.polarity != case.sibling_polarity
                or sibling.modality != case.sibling_modality
            )
            self.assertTrue(leaked, case.corruption_mode)
            self.assertEqual(case.repair_stage, "grounding")
            self.assertTrue(
                set(case.allowed_ops) & {"REPAIR_GROUNDING", "CHALLENGE_PREMISE"}
            )
            if case.corruption_mode == "neutralize":
                self.assertEqual(sibling.polarity, "NEUTRAL")
            elif case.corruption_mode == "beneficial_possible":
                self.assertEqual(sibling.polarity, "BENEFICIAL")
                self.assertEqual(sibling.modality, "POSSIBLE")

        if case.expect_role_field is not None:
            field = discourse_role_field_for_party(
                case.source,
                party_label=case.patient_label,
            )
            self.assertEqual(field, case.expect_role_field, case.source)


if __name__ == "__main__":
    unittest.main()
