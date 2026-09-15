"""QUANTITY_FIELD_TYPING: magnitude spans only in recorded quantities."""
from __future__ import annotations

import unittest

from hypothesis import given, settings, strategies as st

from relent.quantity_typing import (
    classify_quantity_span,
    magnitude_quantity_spans,
    pseudo_quantity_errors,
    sanitize_recorded_quantities,
)


class QuantityFieldTypingTests(unittest.TestCase):
    def test_catastrophic_loss_is_pseudo_outcome(self):
        self.assertEqual(
            classify_quantity_span("catastrophic loss of life"),
            "PSEUDO_OUTCOME",
        )
        self.assertEqual(
            sanitize_recorded_quantities(
                ("catastrophic loss of life", "thousands of lives"),
                source_texts=(
                    "risks thousands of lives through catastrophic loss of life",
                ),
            ),
            ("thousands of lives",),
        )

    def test_incomplete_decades_of_completes_from_source(self):
        self.assertEqual(classify_quantity_span("decades of"), "INCOMPLETE")
        self.assertEqual(
            sanitize_recorded_quantities(
                ("decades of",),
                source_texts=(
                    "permanently erase decades of medical research",
                ),
            ),
            ("decades of medical research",),
        )

    def test_bare_thousands_is_retained(self):
        self.assertEqual(
            sanitize_recorded_quantities(
                ("thousands",),
                source_texts=("A cyberattack risks thousands of lives.",),
            ),
            ("thousands",),
        )

    def test_incomplete_without_source_unit_becomes_head(self):
        self.assertEqual(
            sanitize_recorded_quantities(("thousands of",)),
            ("thousands",),
        )

    def test_math_filter_drops_pseudo(self):
        self.assertEqual(
            magnitude_quantity_spans(
                ("catastrophic loss of life", "thousands", "decades of"),
            ),
            ("thousands",),
        )

    def test_pseudo_errors_name_issue(self):
        errors = pseudo_quantity_errors(
            ("catastrophic loss of life",),
            prefix="F1",
        )
        self.assertTrue(errors)
        self.assertIn("QUANTITY_FIELD_TYPING", errors[0])
        self.assertIn("F1", errors[0])


class QuantityFieldTypingHypothesisTests(unittest.TestCase):
    @settings(max_examples=40, deadline=None)
    @given(st.sampled_from((
        "catastrophic loss of life",
        "immediate survival",
        "mass casualties",
        "fatal injury",
    )))
    def test_outcome_phrases_never_survive_sanitize(self, span: str):
        self.assertEqual(classify_quantity_span(span), "PSEUDO_OUTCOME")
        self.assertEqual(sanitize_recorded_quantities((span,)), ())

    @settings(max_examples=40, deadline=None)
    @given(st.sampled_from((
        "thousands",
        "thousands of lives",
        "hundreds",
        "500+",
        "over five hundred",
        "decades",
    )))
    def test_magnitude_phrases_remain_magnitude(self, span: str):
        self.assertEqual(classify_quantity_span(span), "MAGNITUDE")
        kept = sanitize_recorded_quantities((span,))
        self.assertTrue(kept)
        self.assertIn(classify_quantity_span(kept[0]), {"MAGNITUDE", "INCOMPLETE"})


if __name__ == "__main__":
    unittest.main()
