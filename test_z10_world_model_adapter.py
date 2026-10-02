import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import parsing_game_Z10 as z10
from z10_world_model_adapter import (
    enumerate_candidate_world_models,
    segment_source_clauses,
)


class Z10WorldModelAdapterTests(unittest.TestCase):
    def test_dr_title_is_not_a_clause_boundary(self):
        rows = segment_source_clauses("Dr. Chen has one vial. Imani waits.")
        self.assertEqual([row["text"] for row in rows],
                         ["Dr. Chen has one vial.", "Imani waits."])

    def test_heldout_allocation_produces_a_concrete_provisional_world(self):
        with open("diagnostics/z10_heldout_antivenom_packet.json", encoding="utf-8") as handle:
            package = json.load(handle)["package"]
        result = enumerate_candidate_world_models(
            package,
            ["administer the antivenom to Imani", "administer the antivenom to Pavel"],
        )
        self.assertTrue(result["drafts"], result)
        draft = result["drafts"][0]
        self.assertTrue(draft["selection_validation"]["contract_valid"])
        self.assertFalse(draft["admission_authorized"])
        world = draft["world_model"]
        self.assertEqual(world["schema_version"], "1.3")
        self.assertEqual([row["action_id"] for row in world["actions"]], ["A0", "A1"])
        self.assertEqual(len(world["conditions"]), 2)
        self.assertGreaterEqual(len(world["effects"]), 4)
        self.assertEqual(world["causal_links"], [])
        self.assertTrue(any(row["quantities"] == ["1 vial"] for row in world["parties"]))
        codes = {row["code"] for row in draft["construction_problems"]}
        self.assertIn("conditional_causation_unresolved", codes)
        self.assertIn("outcome_semantics_unresolved", codes)
        self.assertIn("resource_quantity_identity_unresolved", codes)
        self.assertIn("choice_exclusivity_unresolved", codes)

    def test_stripping_role_ambiguity_becomes_three_candidate_worlds(self):
        text = "Lila gives Omar the medicine, but not Nora."
        package = z10.export_candidate_graph(text, package_id="adapter_stripping")
        result = enumerate_candidate_world_models(
            package, ["give the medicine to Omar", "give the medicine to Nora"])
        self.assertEqual(len(result["drafts"]), 3, result)
        assignments = {tuple(row["proposition_ids"]) for row in result["drafts"]}
        self.assertEqual(len(assignments), 3)
        self.assertTrue(all(row["selection_validation"]["contract_valid"]
                            for row in result["drafts"]))
        # Each draft is a real Parliament-shaped model, but the negative
        # reconstruction remains an operator-scope problem rather than an
        # invented positive occurrence.
        for draft in result["drafts"]:
            self.assertEqual(len(draft["world_model"]["actions"]), 2)
            self.assertIn("operator_scope_unresolved",
                          {row["code"] for row in draft["construction_problems"]})
            self.assertFalse(draft["admission_authorized"])
        role_alignment = [
            "action_role_alignment_unresolved" in
            {row["code"] for row in draft["construction_problems"]}
            for draft in result["drafts"]
        ]
        self.assertEqual(role_alignment.count(False), 1)
        self.assertEqual(role_alignment.count(True), 2)

    def test_unmatched_action_returns_an_explicit_problem(self):
        package = z10.export_candidate_graph("Maria sleeps.", package_id="unmatched")
        result = enumerate_candidate_world_models(package, ["allocate a ventilator"])
        self.assertEqual(result["drafts"], [])
        self.assertEqual(result["construction_problems"][0]["code"],
                         "action_alignment_failed")

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_generated_candidates_parse_in_real_parliament_schema(self):
        text = "Lila gives Omar the medicine, but not Nora."
        package = z10.export_candidate_graph(text, package_id="adapter_real_schema")
        result = enumerate_candidate_world_models(
            package, ["give the medicine to Omar", "give the medicine to Nora"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            script = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.world_state import parse_world_model
result = json.load(open(sys.argv[2]))
for draft in result["drafts"]:
    parse_world_model(
        draft["world_model"], clauses=draft["evidence_binding"]["clauses"],
        action_ids=["A0", "A1"],
        action_texts={"A0": result["actions"][0], "A1": result["actions"][1]},
        require_completeness=False,
    )
print(len(result["drafts"]))
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", script,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), "3")

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_current_raw_admission_commits_even_with_adapter_blockers(self):
        text = ("A clinic has one dose of serum. "
                "Ada can give the serum to Ben or Cara, but not both. "
                "If Ada gives the serum to Ben, Ben will recover. "
                "If Ada gives the serum to Cara, Cara will recover.")
        actions = ["give the serum to Ben", "give the serum to Cara"]
        package = z10.export_candidate_graph(text, package_id="adapter_live_pipeline")
        result = enumerate_candidate_world_models(package, actions)
        self.assertEqual(len(result["drafts"]), 1)
        draft = result["drafts"][0]
        self.assertFalse(draft["admission_authorized"])
        self.assertTrue(draft["construction_problems"])
        payload = {"actions": actions, "draft": draft}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            script = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.local_specialists import _admit_action_source_rows
x = json.load(open(sys.argv[2])); d = x["draft"]
candidate = {"actions": d["evidence_binding"]["actions"],
             "world_model": d["world_model"], "ellipsis_resolutions": []}
result = _admit_action_source_rows(
    candidate, x["actions"], ["A0", "A1"], d["evidence_binding"]["clauses"])
assert result["status"] == "COMMITTED", result
assert result["world_model_status"] == "COMMITTED", result
assert result["errors"] == [], result
print(result["status"])
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", script,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), "COMMITTED")


if __name__ == "__main__":
    unittest.main()
