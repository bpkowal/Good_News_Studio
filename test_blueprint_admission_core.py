import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from blueprint_admission_core import supported_core
from run_blueprint_parliament import _admit_candidates


def example():
    return {"candidate": {"world_model": {
        "actions": [{"action_id": "A0", "effect_ids": ["E1", "E2", "AV1", "E3"]}],
        "effects": [
            {"effect_id": "E1", "outcome": "acts", "source_effect_ids": []},
            {"effect_id": "E2", "outcome": "dies", "source_effect_ids": ["E1"]},
            {"effect_id": "AV1", "outcome": "survives", "action_id": "A0",
             "party_id": "P1", "derivation_operation": "AVERTED_ALTERNATIVE_HARM",
             "derivation_assumptions": ["unproved"], "source_effect_ids": ["E2"]},
            {"effect_id": "E3", "outcome": "recovers", "source_effect_ids": ["AV1"]},
        ],
        "causal_links": [{"source_id": "E1", "target_id": "E2", "relation": "CAUSES"},
                         {"source_id": "AV1", "target_id": "E3", "relation": "ENABLES"}],
        "temporal_relations": [{"source_id": "E1", "target_id": "E2", "relation": "BEFORE"}],
        "counterfactual_links": [{"source_effect_id": "AV1", "alternative_effect_id": "E2"}],
    }}, "assignment": ["act"], "admission_authorized": True}


class CoreAdmissionTests(unittest.TestCase):
    def test_projection_keeps_evidence_original_and_extra_supported_links(self):
        original = example()
        before = copy.deepcopy(original)
        core, overlay = supported_core(original)
        world = core["candidate"]["world_model"]
        self.assertEqual(original, before)
        self.assertEqual([e["effect_id"] for e in world["effects"]], ["E1", "E2"])
        self.assertEqual(world["actions"][0]["effect_ids"], ["E1", "E2"])
        self.assertEqual(world["causal_links"], before["candidate"]["world_model"]["causal_links"][:1])
        self.assertEqual(world["temporal_relations"], before["candidate"]["world_model"]["temporal_relations"])
        self.assertEqual(world["counterfactual_links"], [])
        self.assertEqual(overlay["reasons"]["E3"], ["depends_on_excluded_effect"])
        self.assertEqual(len(overlay["effects"]), 2)
        self.assertEqual(overlay["hypotheses"], [])
        self.assertTrue(all(x["status"] == "UNKNOWN" for x in overlay["unknown_branch_outcomes"]))

    def test_explicit_benefit_and_non_template_relation_survive(self):
        original = example()
        world = original["candidate"]["world_model"]
        world["effects"][2].update(derivation_operation="DIRECT_COPY", derivation_assumptions=[],
                                  outcome_type_transformation="PRESERVED", source_effect_ids=[])
        world["causal_links"].append({"source_id": "E2", "target_id": "AV1", "relation": "PREVENTS"})
        core, overlay = supported_core(original)
        self.assertEqual(core, original)
        self.assertEqual(overlay["effects"], [])

    @mock.patch("run_blueprint_parliament.validate_proposal", return_value=[])
    @mock.patch("run_blueprint_parliament.subprocess.run")
    def test_failed_preferred_candidate_falls_back_and_tests_all(self, run, validate):
        run.side_effect = [
            mock.Mock(returncode=1, stdout="", stderr="bad source binding"),
            mock.Mock(returncode=0, stdout=json.dumps({"status": "COMMITTED"}), stderr=""),
            mock.Mock(returncode=0, stdout=json.dumps({"status": "COMMITTED"}), stderr=""),
        ]
        blueprint = {"candidate_attempts": [
            {"rank": i, "blueprint_id": f"b{i}", "proposal": example(), "selected": i == 0}
            for i in range(3)]}
        with tempfile.TemporaryDirectory() as tmp:
            chosen, records = _admit_candidates(blueprint, "scenario", {}, Path(tmp),
                                                Path(tmp), Path("python"))
            self.assertEqual(chosen["attempt"]["blueprint_id"], "b1")
            self.assertEqual([r["status"] for r in records], ["REJECTED", "ADMITTED", "ADMITTED"])
            self.assertEqual(run.call_count, 3)
            self.assertEqual(len(json.loads(Path(records[1]["overlay_path"]).read_text())["effects"]), 2)

    def test_unlicensed_resource_transfer_is_not_committed(self):
        original = example()
        world = original["candidate"]["world_model"]
        world["parties"] = [{"party_id": "P1", "label": "Ada", "kind": "PERSON",
                             "kind_origin": "STRUCTURALLY_DERIVED"}]
        world["effects"][0]["effect_kind"] = "RESOURCE_TRANSFER"
        core, overlay = supported_core(original)
        self.assertNotIn("E1", [row["effect_id"] for row in core["candidate"]["world_model"]["effects"]])
        self.assertIn("unlicensed_resource_transfer", overlay["reasons"]["E1"])


if __name__ == "__main__":
    unittest.main()
