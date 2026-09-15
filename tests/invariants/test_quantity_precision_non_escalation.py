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

    def test_life_year_product_from_thousands_is_escalation(self):
        errors = quantity_precision_escalation_errors(
            source_texts=(
                "A cyberattack risks thousands of lives through failure.",
            ),
            claim_text=(
                "~8 million life-years preserved "
                "(100 000 residents × 80-year average remaining life)"
            ),
        )
        self.assertTrue(errors)
        self.assertIn("QUANTITY_PRECISION_NON_ESCALATION", errors[0])
        self.assertIn("life-year", errors[0].casefold())

    def test_spaced_range_headcount_is_escalation(self):
        errors = quantity_precision_escalation_errors(
            source_texts=(
                "A cyberattack risks thousands of lives through failure.",
            ),
            claim_text=(
                "approximately 5–10 000 city residents would be saved "
                "by executing the purge"
            ),
        )
        self.assertTrue(errors)
        self.assertIn("QUANTITY_PRECISION_NON_ESCALATION", errors[0])

    def test_challenge_refined_answer_rejects_life_year_invention(self):
        from global_workspace.engine import _advance_argument_challenge_agenda
        from global_workspace.models import CandidateChunk

        challenge = {
            "issue_id": "CHALLENGE:util000000000001",
            "generated_by": "WORKSPACE_ARGUMENT_AUDITOR",
            "about_specialist": "utilitarian",
            "raised_by": [],
            "target_specialists": ["utilitarian"],
            "question": "Which comparison would resolve the welfare ranking?",
            "proposition": "Which comparison would resolve the welfare ranking?",
            "challenge_kind": "UTILITY_UNRESOLVED_COMPARISON",
            "grounded_in": [],
            "grounding_status": "PROPOSITION_GROUNDED",
            "status": "ASSIGNED",
            "priority": 0.97,
        }
        candidate = CandidateChunk(
            specialist="utilitarian",
            constraint="WELFARE",
            action_scores={"execute the purge": 0.7, "refrain": 0.3},
            surprise=0.2,
            friction=0.4,
            confidence=0.5,
            recommended_action="execute the purge",
            challenge_response={
                "issue_id": challenge["issue_id"],
                "disposition": "REFINED",
                "current_position_effect": "NO_CHANGE",
                "answer": (
                    "~8 million life-years preserved "
                    "(100 000 residents × 80-year average remaining life)"
                ),
            },
        )
        retained, _agenda = _advance_argument_challenge_agenda(
            previous_challenges=[challenge],
            generated_challenges=[],
            candidates=[candidate],
            next_cycle=2,
            next_constraint="WELFARE",
            active_specialists=["utilitarian"],
            source_texts=(
                "A cyberattack risks thousands of lives through failure.",
                "execute the purge",
                "refrain",
            ),
        )
        verified = next(
            item for item in retained if item["issue_id"] == challenge["issue_id"]
        )
        self.assertEqual(
            verified["last_response"]["verification_status"],
            "PRECISION_REJECTED",
        )
        self.assertEqual(verified["status"], "UNRESOLVED")
        self.assertIn(
            "QUANTITY_PRECISION_NON_ESCALATION",
            verified["last_response"]["verification_reason"],
        )


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
