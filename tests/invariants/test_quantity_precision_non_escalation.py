"""QUANTITY_PRECISION_NON_ESCALATION over Hypothesis-generated claims."""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.world_state import quantity_precision_escalation_errors
from strategies.quantity_precision import (
    QuantityPrecisionCase,
    quantity_precision_cases,
)


class QuantityPrecisionNonEscalationTests(unittest.TestCase):
    def test_thousands_may_not_become_10000(self):
        errors = quantity_precision_escalation_errors(
            source_texts=(
                "A cyberattack risks thousands of lives through failure.",
            ),
            claim_text="The purge would directly save about 10,000 people.",
        )
        self.assertTrue(errors)
        self.assertIn("QUANTITY_PRECISION_NON_ESCALATION", errors[0])

    def test_licensed_exact_numeral_is_not_escalation(self):
        errors = quantity_precision_escalation_errors(
            source_texts=(
                "The outbreak risks thousands of lives; exactly 10000 beds are ready.",
            ),
            claim_text="About 10,000 people are affected.",
        )
        self.assertEqual(errors, [])

    def test_vague_paraphrase_without_digits_is_allowed(self):
        errors = quantity_precision_escalation_errors(
            source_texts=(
                "A cyberattack risks thousands of lives through failure.",
            ),
            claim_text="Thousands of lives remain at stake under refrain.",
        )
        self.assertEqual(errors, [])


class QuantityPrecisionHypothesisTests(unittest.TestCase):
    @given(quantity_precision_cases())
    @settings(max_examples=60, deadline=None)
    def test_oracle_agrees_with_detector(self, case: QuantityPrecisionCase):
        errors = quantity_precision_escalation_errors(
            source_texts=(case.source,),
            claim_text=case.claim,
        )
        if case.expect_escalation:
            self.assertTrue(errors, (case.source, case.claim))
        else:
            self.assertEqual(errors, [], (case.source, case.claim))


if __name__ == "__main__":
    unittest.main()
