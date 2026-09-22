from __future__ import annotations

import unittest

from research.projection_ab_experiment import compare_projection_arms


class ProjectionABExperimentTests(unittest.TestCase):
    def test_enhancement_explains_omissions_without_changing_roles_or_graph(self):
        trace = {
            "status": "COMMITTED",
            "compact_role_assignments": [{
                "action_id": "A0", "effect_id": "E3", "bucket": "harmed",
                "label": "David", "reason": "certain health harm",
            }],
            "chains": [
                {"canonical_action": {"action_id": "A0"}, "world": {
                    "effect_id": "E1", "party_label": "Maria",
                    "outcome": "receives the medicine", "directness": "DIRECT",
                    "effect_kind": "RESOURCE_TRANSFER", "polarity": "BENEFICIAL",
                }, "compact_role": None},
                {"canonical_action": {"action_id": "A0"}, "world": {
                    "effect_id": "E3", "party_label": "David", "outcome": "dies",
                    "directness": "DOWNSTREAM", "effect_kind": "HEALTH_OUTCOME",
                    "polarity": "ADVERSE",
                }, "compact_role": {
                    "action_id": "A0", "effect_id": "E3", "bucket": "harmed",
                    "label": "David", "reason": "certain health harm",
                }},
            ],
        }
        result = compare_projection_arms(trace, case_id="medicine")
        self.assertTrue(result["comparison"]["graph_unchanged"])
        self.assertTrue(result["comparison"]["role_assignments_unchanged"])
        self.assertEqual(
            result["arm_b"]["projection_dispositions"][0]["disposition"],
            "EXCLUDED_RESOURCE_TRANSFER",
        )
        self.assertEqual(result["arm_b"]["audit"]["status"], "COMPLETE")

    def test_unknown_polarity_exclusion_is_sent_to_review(self):
        trace = {
            "status": "COMMITTED", "compact_role_assignments": [],
            "chains": [{"canonical_action": {"action_id": "A0"}, "world": {
                "effect_id": "E9", "party_label": "Resident",
                "outcome": "loses access", "directness": "DOWNSTREAM",
                "effect_kind": "ACCESS_OUTCOME", "polarity": "ADVERSE",
            }, "compact_role": None}],
        }
        result = compare_projection_arms(trace, case_id="unknown")
        self.assertEqual(result["arm_b"]["audit"]["status"], "REVIEW")
        self.assertEqual(result["arm_b"]["audit"]["unresolved_effect_ids"], ["E9"])


if __name__ == "__main__":
    unittest.main()
