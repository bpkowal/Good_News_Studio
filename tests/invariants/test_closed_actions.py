"""Closed action-set invariants over Hypothesis-generated pairs.

The oracle is ClosedActionSetCase.should_withhold. Production is only asked
whether synthesis withheld a third action as closing the set.
"""
from __future__ import annotations

import json
import unittest

from hypothesis import given, settings

from global_workspace.local_specialists import propose_synthesis
from global_workspace.models import CandidateChunk, WorkspaceBroadcast
from strategies.actions import ClosedActionSetCase, closed_action_set_cases


class _ShareSplitLlm:
    def __init__(self, third_action: str):
        self.third_action = third_action

    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": json.dumps({
            "a": self.third_action,
            "g": ["care", "deontological"],
            "k": ["CARE", "DUTY"],
            "q": [],
            "f": 0.86,
            "x": True,
            "n": True,
            "w": "tries to preserve both by splitting the pair",
        })}]}


def _candidates(case: ClosedActionSetCase) -> list[CandidateChunk]:
    return [
        CandidateChunk(
            "care", "CARE",
            {case.actions[0]: 0.8, case.actions[1]: 0.2},
            0.5, 0.7, 0.8,
            rationale="protect the first option",
            recommended_action=case.actions[0],
        ),
        CandidateChunk(
            "deontological", "DUTY",
            {case.actions[0]: 0.2, case.actions[1]: 0.8},
            0.5, 0.7, 0.8,
            rationale="avoid the first option",
            recommended_action=case.actions[1],
        ),
    ]


class ClosedActionSetTests(unittest.TestCase):
    @given(closed_action_set_cases())
    @settings(max_examples=40, deadline=None)
    def test_exclusive_pair_does_not_admit_share_split_synthesis(
        self, case: ClosedActionSetCase,
    ):
        proposal = propose_synthesis(
            _ShareSplitLlm(case.third_action),
            case.scenario,
            case.actions,
            _candidates(case),
            WorkspaceBroadcast(),
            {
                "care": "Protect the first option.",
                "deontological": "Do not take the first option.",
            },
        )
        withheld = (
            not proposal.accepted
            and "closes the action set" in proposal.rejection_reason
        )
        self.assertEqual(withheld, case.should_withhold)
        if case.should_withhold:
            self.assertEqual(proposal.admission_status, "WITHHOLD_FROM_REVIEW")
