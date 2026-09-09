"""Likelihood-span normalization over Hypothesis-generated hedges.

The oracle is LikelihoodSpanCase.canonical. Production is only asked whether
that span is the one extractor, qualifier-binding, and admission record.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.world_state import (
    _HIGH_CONFIDENCE_LIKELIHOOD,
    _effect_expected_qualifiers,
    _supporting_source_is_hedged,
    assigned_party_quantities,
    canonical_likelihood_span,
    explicit_likelihood_span_records,
    explicit_likelihood_spans,
    explicit_quantity_spans,
    validate_world_completeness,
    validate_world_model,
)
from strategies.likelihoods import (
    ChancePercentCase,
    LikelihoodSpanCase,
    chance_percent_cases,
    likelihood_span_cases,
)


def _admission_errors(world) -> list[str]:
    structural, _contradictions = validate_world_model(world, action_ids=["A0"])
    complete = validate_world_completeness(world, action_ids=["A0"])
    return [*structural, *complete]


class LikelihoodSpanNormalizationTests(unittest.TestCase):
    @given(likelihood_span_cases())
    @settings(max_examples=50, deadline=None)
    def test_hedge_is_one_canonical_span(self, case: LikelihoodSpanCase):
        spans = explicit_likelihood_spans(case.source)
        self.assertEqual(spans, (case.canonical,))
        if case.nested_inner:
            self.assertNotIn(case.nested_inner, spans)
        self.assertEqual(
            _effect_expected_qualifiers(
                case.hedged_effect, explicit_likelihood_spans,
            ),
            (case.canonical,),
        )
        self.assertEqual(
            _effect_expected_qualifiers(
                case.unhedged_effect, explicit_likelihood_spans,
            ),
            (),
        )
        self.assertTrue(_supporting_source_is_hedged(case.hedged_without_copy))
        self.assertFalse(_supporting_source_is_hedged(case.unhedged_effect))
        self.assertEqual(case.world.effects[-1].likelihood_qualifiers, (case.canonical,))
        self.assertEqual(_admission_errors(case.world), [])
        self.assertEqual(
            canonical_likelihood_span(case.canonical), case.canonical,
        )
        self.assertEqual(
            canonical_likelihood_span(canonical_likelihood_span(case.canonical)),
            canonical_likelihood_span(case.canonical),
        )
        self.assertEqual(
            explicit_likelihood_spans(case.canonical),
            (case.canonical,),
        )
        self.assertEqual(
            bool(_HIGH_CONFIDENCE_LIKELIHOOD.search(case.canonical)),
            case.high_confidence,
        )


class LikelihoodIsNotAPartyQuantityTests(unittest.TestCase):
    @given(chance_percent_cases())
    @settings(max_examples=50, deadline=None)
    def test_chance_percent_is_not_a_party_quantity(self, case: ChancePercentCase):
        quantity_spans = explicit_quantity_spans(case.source)
        assigned = assigned_party_quantities(case.world.parties)["P2"]
        if case.binds_to_party:
            self.assertIn(case.quantity_span, quantity_spans)
            self.assertEqual(assigned, (case.quantity_span,))
            self.assertEqual(case.world.parties[-1].quantities, (case.quantity_span,))
            self.assertEqual(case.world.effects[-1].quantities, (case.quantity_span,))
            self.assertEqual(case.world.effects[-1].likelihood_qualifiers, ())
        else:
            self.assertNotIn(case.quantity_span, quantity_spans)
            self.assertNotIn(
                case.quantity_span.casefold(),
                {item.casefold() for item in assigned},
            )
            self.assertEqual(case.world.parties[-1].quantities, ())
            self.assertEqual(case.world.effects[-1].quantities, ())
            self.assertEqual(
                case.world.effects[-1].likelihood_qualifiers, (case.chance_span,),
            )
            self.assertEqual(
                explicit_likelihood_spans(case.source), (case.chance_span,),
            )
            records = explicit_likelihood_span_records(case.source)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].canonical, case.chance_span)
            self.assertEqual(records[0].literal, case.literal_span)
            self.assertEqual(
                case.source[records[0].start:records[0].end],
                case.literal_span,
            )
            self.assertEqual(
                canonical_likelihood_span(records[0].literal),
                case.chance_span,
            )
        self.assertEqual(_admission_errors(case.world), [])
