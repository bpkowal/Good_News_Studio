"""Decision state vs authority state over Hypothesis-generated labels.

The oracle is PreferenceSupportCase.should_uniquely_support. Production is
only asked whether equal scores with no recommendation uniquely SUPPORT.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.models import CandidateChunk
from global_workspace.specialist_authority import apply_specialist_authority
from strategies.consumption import PreferenceSupportCase, preference_support_cases


class PreferenceSupportTests(unittest.TestCase):
    @given(preference_support_cases())
    @settings(max_examples=40, deadline=None)
    def test_tied_scores_without_recommendation_cannot_uniquely_support(
        self, case: PreferenceSupportCase,
    ):
        chunk = CandidateChunk(
            specialist=case.specialist,
            constraint="IMMINENT_HARM",
            action_scores=case.action_scores,
            surprise=0.1,
            friction=0.0 if case.ranking_kind == "TIED" else 0.44,
            confidence=0.8,
            recommended_action=case.recommended_action,
            preference_strength=0.0 if case.ranking_kind == "TIED" else 0.44,
            epistemic_confidence=0.8,
            schema_valid=True,
            decision_rule="prefer the second action" if case.should_uniquely_support else "",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
        )
        profile = apply_specialist_authority(chunk)
        uniquely = (
            profile.governing_eligible
            and profile.adjudication_status == "SUPPORTS"
        )
        self.assertEqual(uniquely, case.should_uniquely_support)
        if not case.should_uniquely_support:
            self.assertEqual(profile.adjudication_status, "CONTESTED_NO_LEANING")
            self.assertEqual(profile.policy_weight_factor, 0.0)
            self.assertFalse(bool(chunk.recommended_action.strip()))
