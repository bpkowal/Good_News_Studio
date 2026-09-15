"""Admitted vote vs directional lean must stay separate in public judgment."""
from __future__ import annotations

import unittest

from global_workspace.presentation import render_public_judgment


def _purge_like_payload() -> dict:
    actions = [
        "execute the irreversible emergency purge",
        "refrain from triggering the purge",
    ]
    recommendation = actions[0]
    return {
        "actions": actions,
        "selected_action": recommendation,
        "current_plurality": recommendation,
        "judgment_status": "CONTESTED_RECOMMENDATION",
        "confidence": 0.72,
        "epistemic_confidence": 0.55,
        "halted_by": "cycle_budget",
        "convergence_status": "UNRESOLVED",
        "canonical_action_records": [],
        "proposition_ledger": [],
        "cycles": [{
            "cycle": 3,
            "is_hypothetical": False,
            "constraint": "DUTY",
            "policy": {recommendation: 0.72, actions[1]: 0.28},
            "selected_action": recommendation,
            "current_plurality": recommendation,
            "candidates": [
                {
                    "specialist": "utilitarian",
                    "constraint": "WELFARE",
                    "schema_valid": True,
                    "recommended_action": recommendation,
                    "decision_rule": "Maximize expected life-years under uncertainty",
                    "rationale": "Directional lean toward purge on expected harm.",
                    "assumption_status": "DIRECT",
                    "adjudication_status": "SUPPORTS",
                    "framework_vote_status": "ABSTAIN",
                    "framework_vote_reason": (
                        "utilitarian ledger did not complete a same-unit ranking"
                    ),
                    "action_scores": {recommendation: 0.62, actions[1]: 0.38},
                },
                {
                    "specialist": "deontological",
                    "constraint": "DUTY",
                    "schema_valid": True,
                    "recommended_action": recommendation,
                    "decision_rule": "Rescue duty remains structurally unsettled",
                    "rationale": "Directional lean while duty conflict stays open.",
                    "assumption_status": "NORMATIVELY_CONTESTED",
                    "adjudication_status": "PROVISIONAL_LEANING",
                    "framework_vote_status": "ABSTAIN",
                    "framework_vote_reason": (
                        "deontological duty conflict is not resolved"
                    ),
                    "action_scores": {recommendation: 0.58, actions[1]: 0.42},
                },
                {
                    "specialist": "virtue",
                    "constraint": "CHARACTER",
                    "schema_valid": True,
                    "recommended_action": recommendation,
                    "decision_rule": "Prudence favors preserving city survival",
                    "rationale": "Courageous stewardship under emergency.",
                    "assumption_status": "DIRECT",
                    "adjudication_status": "SUPPORTS",
                    "framework_vote_status": "FULL",
                    "framework_vote_reason": (
                        "framework ranking is supported by its committed typed ledger"
                    ),
                    "action_scores": {recommendation: 0.8, actions[1]: 0.2},
                },
                {
                    "specialist": "care",
                    "constraint": "CARE",
                    "schema_valid": True,
                    "recommended_action": recommendation,
                    "decision_rule": "Urgent dependent need favors survival",
                    "rationale": "Attenuated care support.",
                    "assumption_status": "CONDITIONAL",
                    "adjudication_status": "CONDITIONAL_SUPPORTS",
                    "framework_vote_status": "ATTENUATED",
                    "framework_vote_reason": (
                        "framework compared every live action but reports residual uncertainty"
                    ),
                    "action_scores": {recommendation: 0.7, actions[1]: 0.3},
                },
                {
                    "specialist": "rawlsian",
                    "constraint": "FAIRNESS",
                    "schema_valid": True,
                    "recommended_action": recommendation,
                    "decision_rule": "Maximin favors the survival of the least secure",
                    "rationale": "Worst-off security.",
                    "assumption_status": "DIRECT",
                    "adjudication_status": "SUPPORTS",
                    "framework_vote_status": "FULL",
                    "framework_vote_reason": (
                        "framework ranking is supported by its committed typed ledger"
                    ),
                    "action_scores": {recommendation: 0.78, actions[1]: 0.22},
                },
            ],
            "dissent": None,
        }],
    }


class VoteReportSplitTests(unittest.TestCase):
    def test_abstain_never_reads_as_favors_despite_directional_lean(self):
        text = render_public_judgment(_purge_like_payload())
        self.assertIn("Virtue and Rawlsian support this action", text)
        self.assertIn("attenuated support", text.casefold())
        self.assertIn("Virtue favors this action", text)
        self.assertIn("Rawlsian favors this action", text)
        self.assertIn("Care conditionally supports this action", text)
        self.assertNotIn("Utilitarian favors this action", text)
        self.assertNotIn("Deontological favors this action", text)
        self.assertNotIn("Deontological provisionally favors this action", text)
        self.assertIn("Directional leans without admitted vote", text)
        self.assertIn("Utilitarian leans toward", text)
        self.assertIn("abstains", text.casefold())
        self.assertIn("| Directional position | Admitted vote |", text)
        self.assertIn("ABSTAIN", text)

    def test_full_vote_still_reported_as_favor(self):
        payload = _purge_like_payload()
        # Strip abstainers so only FULL/ATTENUATED remain.
        cycle = payload["cycles"][0]
        cycle["candidates"] = [
            row for row in cycle["candidates"]
            if row["framework_vote_status"] != "ABSTAIN"
        ]
        text = render_public_judgment(payload)
        self.assertIn("Virtue favors this action", text)
        self.assertNotIn("Directional leans without admitted vote", text)


if __name__ == "__main__":
    unittest.main()
