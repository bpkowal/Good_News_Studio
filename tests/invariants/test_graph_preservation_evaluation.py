from __future__ import annotations

import unittest

from global_workspace.graph_preservation_evaluation import (
    score_trace,
    trace_from_unadmitted_candidate,
)


class GraphPreservationEvaluationTests(unittest.TestCase):
    def test_missing_source_fact_is_separate_from_correct_projection(self):
        case = {
            "id": "case", "gold_inventory_exhaustive": False,
            "gold_facts": [
                {"fact_id":"death","action_id":"A0","event":"Jo dies","party":"Jo",
                 "required_term_groups":["Jo","dies death"],"polarity":"ADVERSE"},
                {"fact_id":"chance","action_id":"A0","event":"Jo has 40% survival chance","party":"Jo",
                 "required_term_groups":["Jo","40","survival chance"]},
            ],
            "gold_edges": [],
            "gold_projection": [{"action_id":"A0","fact_id":"death","bucket":"harmed"}],
        }
        trace = {
            "chains": [{"canonical_action":{"action_id":"A0"},"world":{
                "effect_id":"E1","party_label":"Jo","outcome":"dies","polarity":"ADVERSE",
                "modality":"CERTAIN","directness":"DOWNSTREAM","effect_kind":"HEALTH_OUTCOME",
                "quantities":[],"likelihood_qualifiers":[]}}],
            "compact_role_assignments": [{
                "action_id":"A0","effect_id":"E1","bucket":"harmed","label":"Jo",
            }],
            "topology_edges": [],
        }
        result = score_trace(case, trace)
        self.assertEqual(result["source_to_graph"]["recall"], 0.5)
        self.assertEqual(result["projection"]["accuracy_given_gold_graph"], 1.0)

    def test_legacy_trace_without_edges_does_not_score_topology_as_wrong(self):
        case = {
            "id":"legacy", "gold_facts":[
                {"fact_id":"a","action_id":"A0","event":"gate opens","party":"gate",
                 "required_term_groups":["gate","open"]},
                {"fact_id":"b","action_id":"A0","event":"water drains","party":"water",
                 "required_term_groups":["water","drain"]},
            ], "gold_edges":[{"action_id":"A0","source_fact_id":"a","target_fact_id":"b","relation":"CAUSES"}],
        }
        trace = {"chains":[
            {"canonical_action":{"action_id":"A0"},"world":{"effect_id":"E1","party_label":"gate","outcome":"opens"}},
            {"canonical_action":{"action_id":"A0"},"world":{"effect_id":"E2","party_label":"water","outcome":"drains"}},
        ],"compact_role_assignments":[]}
        result = score_trace(case, trace)
        self.assertIsNone(result["topology"]["accuracy_given_aligned_endpoints"])
        self.assertEqual(result["topology"]["measurement_status"], "NOT_MEASURED_ARTIFACT_LACKS_EDGES")

    def test_rejected_candidate_scores_semantics_without_admitting_projection(self):
        candidate = {"world_model": {
            "parties": [{"party_id":"P1","label":"Jo","kind":"PERSON"}],
            "actions": [{"action_id":"A0","intervention":"withhold antidote"}],
            "effects": [{
                "effect_id":"E1","action_id":"A0","party_id":"P1",
                "outcome":"dies","polarity":"ADVERSE","modality":"CERTAIN",
                "directness":"DOWNSTREAM","effect_kind":"HEALTH_OUTCOME",
            }],
            "causal_links": [],
        }}
        case = {"id":"rejected","gold_facts":[{
            "fact_id":"death","action_id":"A0","event":"Jo dies","party":"Jo",
            "required_term_groups":["Jo","dies death"],"polarity":"ADVERSE",
        }],"gold_projection":[{"action_id":"A0","fact_id":"death","bucket":"harmed"}]}
        result = score_trace(case, trace_from_unadmitted_candidate(candidate))
        self.assertEqual(result["source_to_graph"]["recall"], 1.0)
        self.assertEqual(
            result["projection"]["measurement_status"],
            "NOT_MEASURED_UNADMITTED_CANDIDATE",
        )
        self.assertIsNone(result["projection"]["accuracy_given_gold_graph"])

    def test_explicit_intermediate_counts_as_valid_causal_path(self):
        case = {"id":"path","gold_facts":[
            {"fact_id":"a","action_id":"A0","event":"medicine received","party":"patient",
             "required_term_groups":["patient","medicine","receive receives received"]},
            {"fact_id":"b","action_id":"A0","event":"other patient receives no medicine","party":"other patient",
             "required_term_groups":["other","no not","medicine"]},
        ],"gold_edges":[{"action_id":"A0","source_fact_id":"a","target_fact_id":"b","relation":"CAUSES"}]}
        trace = {"chains":[
            {"canonical_action":{"action_id":"A0"},"world":{"effect_id":"E1","party_label":"patient","outcome":"receives medicine"}},
            {"canonical_action":{"action_id":"A0"},"world":{"effect_id":"E2","party_label":"resource","outcome":"is unavailable"}},
            {"canonical_action":{"action_id":"A0"},"world":{"effect_id":"E3","party_label":"other patient","outcome":"does not receive medicine"}},
        ],"compact_role_assignments":[],"topology_edges":[
            {"action_id":"A0","source_id":"E1","target_id":"E2","relation":"CAUSES"},
            {"action_id":"A0","source_id":"E2","target_id":"E3","relation":"CAUSES"},
        ]}
        result = score_trace(case, trace)
        edge = result["topology"]["edges"][0]
        self.assertTrue(edge["correct"])
        self.assertEqual(edge["match_type"], "VALID_CAUSAL_PATH_CORRECT")


if __name__ == "__main__":
    unittest.main()
