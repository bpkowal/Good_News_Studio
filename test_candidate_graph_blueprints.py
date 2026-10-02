import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import parsing_game_Z10 as z10
from candidate_graph_blueprints import (
    instantiate_exclusive_allocation,
    match_exclusive_allocation,
)


SERUM = ("A clinic has one dose of serum. "
         "Ada can give the serum to Ben or Cara, but not both. "
         "If Ada gives the serum to Ben, Ben will recover. "
         "If Ada gives the serum to Cara, Cara will recover.")
ACTIONS = ["give the serum to Ben", "give the serum to Cara"]
MEDICINE = (
    "A clinic has one dose of medicine. "
    "Ada can give the medicine to Ben or Cara, but not both. "
    "If Ada gives the medicine to Ben, Ben has a 95% chance of survival. "
    "If Ada gives the medicine to Cara, Cara has a 5% chance of survival. "
    "The patient who does not get the medicine will die."
)
MEDICINE_ACTIONS = ["give the medicine to Ben", "give the medicine to Cara"]
GROUPED_MEDICINE = (
    "A clinic has one dose of antiviral. "
    "Dr. Rivera must give the antiviral to either one child or three adult patients, "
    "but not both. "
    "If Dr. Rivera gives the antiviral to the child, the child has an 80% chance "
    "of survival. "
    "If Dr. Rivera gives the antiviral to the three adult patients, each of the "
    "three adult patients has a 60% chance of survival. "
    "The patients who do not receive the antiviral will die."
)
GROUPED_ACTIONS = [
    "give the antiviral to the child",
    "give the antiviral to the three adult patients",
]


class ExclusiveAllocationBlueprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = z10.export_candidate_graph(SERUM, package_id="blueprint_test")

    def test_match_fills_all_required_slots(self):
        match = match_exclusive_allocation(self.package, ACTIONS)
        self.assertTrue(match["matched"], match)
        self.assertTrue(all(match["required_slots"].values()))
        self.assertEqual(match["unfilled_required_slots"], [])
        self.assertEqual(len(match["assignments"]), 1)

    def test_proposal_is_complete_and_selection_valid(self):
        result = instantiate_exclusive_allocation(self.package, ACTIONS)
        self.assertEqual(result["status"], "FILLED")
        proposal = result["proposals"][0]
        self.assertEqual(proposal["unfilled_required_slots"], [])
        self.assertTrue(proposal["selection_validation"]["contract_valid"])
        world = proposal["candidate"]["world_model"]
        self.assertEqual(len(world["actions"]), 2)
        self.assertEqual(len(world["effects"]), 6)
        self.assertEqual(len(world["causal_links"]), 2)
        resource = next(row for row in world["parties"] if row["kind"] == "RESOURCE")
        self.assertEqual(resource["quantities"], ["1 dose"])
        self.assertEqual(
            [row["predicate"] for row in world["effects"]],
            ["RECEIVES", "NOT_RECEIVES", "SURVIVES",
             "RECEIVES", "NOT_RECEIVES", "SURVIVES"],
        )

    def test_nonallocation_does_not_match(self):
        package = z10.export_candidate_graph("Maria leaves.", package_id="not_allocation")
        match = match_exclusive_allocation(package, ["Maria leaves"])
        self.assertFalse(match["matched"])
        self.assertTrue(match["unfilled_required_slots"])

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_proposal_passes_parliament_validation_and_admission(self):
        result = instantiate_exclusive_allocation(self.package, ACTIONS)
        proposal = result["proposals"][0]
        payload = {"actions": ACTIONS, "proposal": proposal}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            script = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.local_specialists import _admit_action_source_rows
from global_workspace.world_state import parse_world_model
x = json.load(open(sys.argv[2])); p = x["proposal"]
ids = ["A0", "A1"]
parse_world_model(
    p["candidate"]["world_model"], clauses=p["clauses"], action_ids=ids,
    action_texts=dict(zip(ids, x["actions"])), require_completeness=True)
result = _admit_action_source_rows(
    p["candidate"], x["actions"], ids, p["clauses"])
assert result["status"] == "COMMITTED", result
assert result["world_model_status"] == "COMMITTED", result
assert result["errors"] == [], result
assert len(result["world_model"]["effects"]) == 6, result
print("COMMITTED")
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", script,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), "COMMITTED")


class ChanceAndDeathAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = z10.export_candidate_graph(MEDICINE, package_id="chance_death")

    def test_chance_and_death_fill_schema_1_3(self):
        result = instantiate_exclusive_allocation(self.package, MEDICINE_ACTIONS)
        self.assertEqual(result["status"], "FILLED")
        self.assertEqual(result["blueprint_version"], "candidate-graph-blueprints/0.4")
        proposal = result["proposals"][0]
        self.assertTrue(proposal["selection_validation"]["contract_valid"])
        world = proposal["candidate"]["world_model"]
        self.assertEqual(world["schema_version"], "1.3")
        self.assertEqual(len(world["effects"]), 8)
        self.assertEqual(len(world["causal_links"]), 4)
        by_action = {}
        for effect in world["effects"]:
            by_action.setdefault(effect["action_id"], []).append(effect)
        ben = {row["predicate"]: row for row in by_action["A0"]}
        cara = {row["predicate"]: row for row in by_action["A1"]}
        self.assertEqual(ben["survival"]["modality"], "PROBABILISTIC")
        self.assertEqual(ben["survival"]["likelihood_qualifiers"], ["95% chance"])
        self.assertIn("95% chance of survival", ben["survival"]["source_proposition"])
        self.assertEqual(cara["survival"]["likelihood_qualifiers"], ["5% chance"])
        self.assertEqual(ben["die"]["modality"], "CERTAIN")
        self.assertEqual(ben["die"]["polarity"], "ADVERSE")
        self.assertEqual(ben["die"]["party_id"], "P3")
        self.assertEqual(cara["die"]["party_id"], "P2")
        self.assertIn("will die", ben["die"]["source_proposition"])
        death_links = [row for row in world["causal_links"] if row["target_id"] in {"E4", "E8"}]
        self.assertEqual(len(death_links), 2)
        self.assertTrue(all(row["modality"] == "CERTAIN" for row in death_links))

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_chance_and_death_pass_parliament_admission(self):
        result = instantiate_exclusive_allocation(self.package, MEDICINE_ACTIONS)
        proposal = result["proposals"][0]
        payload = {"actions": MEDICINE_ACTIONS, "proposal": proposal}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            script = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.local_specialists import _admit_action_source_rows
from global_workspace.world_state import parse_world_model
x = json.load(open(sys.argv[2])); p = x["proposal"]
ids = ["A0", "A1"]
parse_world_model(
    p["candidate"]["world_model"], clauses=p["clauses"], action_ids=ids,
    action_texts=dict(zip(ids, x["actions"])), require_completeness=True)
result = _admit_action_source_rows(
    p["candidate"], x["actions"], ids, p["clauses"])
assert result["status"] == "COMMITTED", result["errors"]
assert result["world_model_status"] == "COMMITTED", result
assert len(result["world_model"]["effects"]) == 8, result
print("COMMITTED")
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", script,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), "COMMITTED")


class GroupedAllocationGeneralizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = z10.export_candidate_graph(
            GROUPED_MEDICINE, package_id="grouped_chance_death")

    def test_group_headcount_stays_on_its_branch(self):
        result = instantiate_exclusive_allocation(
            self.package, GROUPED_ACTIONS)
        self.assertEqual(result["status"], "FILLED", result)
        world = result["proposals"][0]["candidate"]["world_model"]
        adult = next(
            row for row in world["parties"]
            if "adult patients" in row["label"])
        child = next(row for row in world["parties"] if "child" in row["label"])
        self.assertEqual(adult["kind"], "HUMAN_GROUP")
        self.assertEqual(adult["quantities"], ["three"])
        self.assertEqual(child["kind"], "PERSON")
        adult_effects = [
            row for row in world["effects"] if row["party_id"] == adult["party_id"]
        ]
        self.assertTrue(any(row["quantities"] == ["three"] for row in adult_effects))
        child_complement = next(
            row for row in world["effects"]
            if row["party_id"] == child["party_id"]
            and row["derivation_operation"] == "EXCLUSIVE_ALLOCATION_COMPLEMENT"
        )
        self.assertEqual(child_complement["quantities"], ["1 dose"])
        self.assertNotIn("three", child_complement["quantities"])
        self.assertEqual(
            child_complement["clause_ids"],
            result["proposals"][0]["slot_bindings"]["resource"][
                "quantity_clause_ids"],
        )

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_grouped_candidate_admits_and_replays_idempotently(self):
        proposal = instantiate_exclusive_allocation(
            self.package, GROUPED_ACTIONS)["proposals"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.json"
            path.write_text(json.dumps(proposal), encoding="utf-8")
            script = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.local_specialists import _admit_action_source_rows
from global_workspace.world_admission import restore_admitted_world
from global_workspace.world_state import parse_world_model
p = json.load(open(sys.argv[2]))
w = p["candidate"]["world_model"]
ids = [a["action_id"] for a in w["actions"]]
actions = [a["intervention"] for a in w["actions"]]
compiled = parse_world_model(
    w, clauses=p["clauses"], action_ids=ids,
    action_texts=dict(zip(ids, actions)), require_completeness=True)
restore_admitted_world(compiled.as_dict())
r = _admit_action_source_rows(p["candidate"], actions, ids, p["clauses"])
assert r["status"] == "COMMITTED", r.get("errors")
restore_admitted_world(r["world_model"])
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", script,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(
            completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
