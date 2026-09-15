"""FraCaS discourse-family Hypothesis suites (plural through attitudes).

Oracles are declared case fields. Hypothesis asks Parliament harness oracles
whether they agree with preserves_* flags.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from semantic_integrity.harness import (
    adjective_modifier_binding_holds,
    attitude_factivity_holds,
    ellipsis_predicate_resolution_holds,
    plural_member_distinctness_holds,
    temporal_order_consistency_holds,
)
from strategies.adjective_modifiers import (
    AdjectiveModifierCase,
    adjective_modifier_cases,
)
from strategies.attitude_factivity import (
    AttitudeFactivityCase,
    attitude_factivity_cases,
)
from strategies.ellipsis_predicate import (
    EllipsisPredicateCase,
    ellipsis_predicate_cases,
)
from strategies.plural_members import PluralMemberCase, plural_member_cases
from strategies.temporal_order import TemporalOrderCase, temporal_order_cases


class PluralMemberHypothesisTests(unittest.TestCase):
    @given(plural_member_cases())
    @settings(max_examples=50, deadline=None)
    def test_oracle_matches_declared_distinctness(self, case: PluralMemberCase):
        holds = plural_member_distinctness_holds(
            case.world, member_party_ids=case.member_party_ids,
        )
        self.assertEqual(holds, case.preserves_distinct, case.mutation)
        if case.preserves_distinct:
            self.assertEqual(case.mutation, "keep")
        else:
            self.assertNotEqual(case.mutation, "keep")
            self.assertIn("RESOLVE_ENTITY", case.allowed_ops)
            party_ids = {party.party_id for party in case.world.parties}
            if case.mutation == "merge_group":
                self.assertIn("P_group", party_ids)
                self.assertFalse({"P_a", "P_b", "P_c"} <= party_ids)
            elif case.mutation == "drop_one":
                self.assertTrue({"P_a", "P_b"} <= party_ids)
                self.assertNotIn("P_c", party_ids)
            elif case.mutation == "single_person":
                self.assertIn("P_one", party_ids)
                self.assertFalse({"P_a", "P_b", "P_c"} <= party_ids)


class EllipsisPredicateHypothesisTests(unittest.TestCase):
    @given(ellipsis_predicate_cases())
    @settings(max_examples=40, deadline=None)
    def test_oracle_matches_declared_predicate(self, case: EllipsisPredicateCase):
        holds = ellipsis_predicate_resolution_holds(
            case.world,
            antecedent_effect_id=case.antecedent_effect_id,
            elliptical_effect_id=case.elliptical_effect_id,
        )
        self.assertEqual(holds, case.preserves_predicate)
        if not case.preserves_predicate:
            self.assertIn("REPAIR_GROUNDING", case.allowed_ops)


class AdjectiveModifierHypothesisTests(unittest.TestCase):
    @given(adjective_modifier_cases())
    @settings(max_examples=40, deadline=None)
    def test_oracle_matches_declared_binding(self, case: AdjectiveModifierCase):
        holds = adjective_modifier_binding_holds(
            case.world,
            effect_id=case.effect_id,
            expected_temporal_qualifiers=case.temporal_qualifiers,
            sibling_effect_id=case.sibling_effect_id,
            party_id=case.party_id,
            expected_party_kind=case.expected_party_kind,
            label_must_contain=case.label_must_contain,
        )
        self.assertEqual(holds, case.preserves_binding, case.source)
        if not case.preserves_binding:
            self.assertIn("REPAIR_GROUNDING", case.allowed_ops)


class TemporalOrderHypothesisTests(unittest.TestCase):
    @given(temporal_order_cases())
    @settings(max_examples=40, deadline=None)
    def test_oracle_matches_declared_order(self, case: TemporalOrderCase):
        holds = temporal_order_consistency_holds(
            case.world,
            earlier_effect_id=case.earlier_effect_id,
            later_effect_id=case.later_effect_id,
            later_marker=case.later_marker,
        )
        self.assertEqual(holds, case.preserves_order, case.source)
        if not case.preserves_order:
            self.assertIn("REPAIR_GROUNDING", case.allowed_ops)


class AttitudeFactivityHypothesisTests(unittest.TestCase):
    @given(attitude_factivity_cases())
    @settings(max_examples=40, deadline=None)
    def test_oracle_matches_declared_factivity(self, case: AttitudeFactivityCase):
        holds = attitude_factivity_holds(
            case.world,
            complement_effect_id=case.complement_effect_id,
            factive=case.factive,
        )
        self.assertEqual(holds, case.preserves_factivity, case.source)
        self.assertEqual(case.repair_stage, "epistemic_binding")
        if not case.preserves_factivity:
            self.assertTrue(
                set(case.allowed_ops)
                & {"VERIFY_PROVENANCE", "CHALLENGE_PREMISE"}
            )


if __name__ == "__main__":
    unittest.main()
