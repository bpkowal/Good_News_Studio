import os
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

import parsing_game_Z10 as z10
from blueprint_ensemble import blank_blueprints, choose_blueprint


MEDICINE = (
    "A clinic has one dose of medicine. "
    "Ada can give the medicine to Ben or Cara, but not both. "
    "If Ada gives the medicine to Ben, Ben has a 95% chance of survival. "
    "If Ada gives the medicine to Cara, Cara has a 5% chance of survival. "
    "The patient who does not get the medicine will die."
)
RESCUE = (
    "A child and a dog are in the water. "
    "Maria can save the child, but not the dog. "
    "If Maria saves the child, the child will live."
)
TROLLEY = (
    "Five workers are on the track. One worker is on the side track. "
    "Maria can pull the lever, but not the brake. "
    "If Maria pulls the lever, one worker will die. "
    "If Maria does not pull the lever, five workers will die."
)


class BlueprintEnsembleTests(unittest.TestCase):
    def test_blank_plans_have_empty_slots(self):
        plans = blank_blueprints()
        self.assertEqual(
            [row["blueprint_id"] for row in plans],
            ["exclusive_allocation", "conditional_outcome", "rescue_contrast", "omission_harm"],
        )
        for plan in plans:
            self.assertTrue(plan["required_slots"])
            self.assertTrue(all(value is None for value in plan["required_slots"].values()))
            self.assertTrue(all(value is None for value in plan["optional_slots"].values()))

    def test_chooser_prefers_allocation_for_the_scarce_dose(self):
        package = z10.export_candidate_graph(MEDICINE, package_id="ensemble_medicine")
        choice = choose_blueprint(package, ["give the medicine to Ben", "give the medicine to Cara"])
        self.assertEqual(choice["chosen_blueprint_id"], "exclusive_allocation")
        by_id = {row["blueprint_id"]: row for row in choice["considered"]}
        self.assertEqual(by_id["exclusive_allocation"]["status"], "FILLED")
        self.assertEqual(by_id["rescue_contrast"]["status"], "NO_MATCH")
        self.assertIn("rescue_action", by_id["rescue_contrast"]["match"]["unfilled_required_slots"])
        world = by_id["exclusive_allocation"]["proposals"][0]["candidate"]["world_model"]
        self.assertEqual(len(world["effects"]), 8)

    def test_chooser_prefers_rescue_contrast_and_does_not_kill_the_dog(self):
        package = z10.export_candidate_graph(RESCUE, package_id="ensemble_rescue")
        choice = choose_blueprint(package)
        self.assertEqual(choice["chosen_blueprint_id"], "rescue_contrast")
        proposal = next(row for row in choice["considered"] if row["blueprint_id"] == "rescue_contrast")
        world = proposal["proposals"][0]["candidate"]["world_model"]
        labels = {row["label"] for row in world["parties"]}
        self.assertIn("the dog", labels)
        self.assertTrue(any(row["predicate"] == "live" for row in world["effects"]))
        self.assertFalse(any(row["predicate"] == "die" for row in world["effects"]))
        self.assertFalse(proposal["optional_slots"]["foregone_harm"])
        self.assertEqual(
            next(row for row in choice["considered"] if row["blueprint_id"] == "exclusive_allocation")["status"],
            "NO_MATCH",
        )

    def test_chooser_prefers_omission_harm_for_the_lever(self):
        package = z10.export_candidate_graph(TROLLEY, package_id="ensemble_trolley")
        choice = choose_blueprint(package)
        self.assertEqual(choice["chosen_blueprint_id"], "omission_harm")
        proposal = next(row for row in choice["considered"] if row["blueprint_id"] == "omission_harm")
        world = proposal["proposals"][0]["candidate"]["world_model"]
        interventions = [row["intervention"] for row in world["actions"]]
        self.assertEqual(interventions, ["pull the lever", "do not pull the lever"])
        deaths = [row for row in world["effects"] if row["predicate"] == "die"]
        self.assertEqual(len(deaths), 2)
        self.assertTrue(all(row["polarity"] == "ADVERSE" for row in deaths))
        self.assertTrue(proposal["optional_slots"]["instrument_contrast"])

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_rescue_and_trolley_graphs_pass_parliament_admission(self):
        graphs = []
        for text, package_id in ((RESCUE, "admit_rescue"), (TROLLEY, "admit_trolley")):
            choice = choose_blueprint(z10.export_candidate_graph(text, package_id=package_id))
            chosen = next(row for row in choice["considered"]
                          if row["blueprint_id"] == choice["chosen_blueprint_id"])
            proposal = chosen["proposals"][0]
            actions = [row["intervention"] for row in proposal["candidate"]["world_model"]["actions"]]
            graphs.append({"actions": actions, "proposal": proposal,
                           "clauses": [{"clause_id": c["clause_id"], "text": c["text"]}
                                       for c in __import__("z10_world_model_adapter").segment_source_clauses(text)]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graphs.json"
            path.write_text(json.dumps(graphs), encoding="utf-8")
            script = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.local_specialists import _admit_action_source_rows
from global_workspace.world_state import parse_world_model
for graph in json.load(open(sys.argv[2])):
    proposal = graph["proposal"]
    world = proposal["candidate"]["world_model"]
    ids = [row["action_id"] for row in world["actions"]]
    parse_world_model(world, clauses=graph["clauses"], action_ids=ids,
                      action_texts=dict(zip(ids, graph["actions"])), require_completeness=True)
    result = _admit_action_source_rows(proposal["candidate"], graph["actions"], ids, graph["clauses"])
    assert result["status"] == "COMMITTED", result["errors"]
print("COMMITTED")
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", script,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
