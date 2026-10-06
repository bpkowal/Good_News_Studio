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
    "Maria can save either the child or the dog, but not both. "
    "If Maria saves the child, the child will live and the dog will drown. "
    "If Maria saves the dog, the dog will live and the child will drown."
)
TROLLEY = (
    "Five workers are on the track. One worker is on the side track. "
    "Maria can pull the lever, but not the brake. "
    "If Maria pulls the lever, one worker will die. "
    "If Maria does not pull the lever, five workers will die."
)
DIVERSION = (
    "Maria can divert the trolley toward one worker, and one worker will die."
)
RISK = "Ada may administer medicine to Ben, and Ben could die."
CONDITIONAL = "If Ada gives medicine to Ben, Ben will live."


class BlueprintEnsembleTests(unittest.TestCase):
    def test_blank_plans_have_empty_slots(self):
        plans = blank_blueprints()
        self.assertEqual(
            [row["blueprint_id"] for row in plans],
            [
                "exclusive_allocation", "conditional_outcome", "rescue_contrast",
                "omission_harm", "ability_permission", "diversion_redirection",
                "deontic_rule", "promise_reliance", "uncertain_risk",
                "disputed_report",
            ],
        )
        for plan in plans:
            self.assertTrue(plan["required_slots"])
            self.assertTrue(all(value is None for value in plan["required_slots"].values()))
            self.assertTrue(all(value is None for value in plan["optional_slots"].values()))
        self.assertEqual(
            [row["graph_builder"] for row in plans].count("implemented"), 6)
        self.assertEqual(
            [row["graph_builder"] for row in plans].count("discourse"), 4)

    def test_chooser_prefers_allocation_for_the_scarce_dose(self):
        package = z10.export_candidate_graph(MEDICINE, package_id="ensemble_medicine")
        choice = choose_blueprint(package, ["give the medicine to Ben", "give the medicine to Cara"])
        self.assertEqual(choice["chosen_blueprint_id"], "exclusive_allocation")
        by_id = {row["blueprint_id"]: row for row in choice["considered"]}
        self.assertEqual(by_id["exclusive_allocation"]["status"], "FILLED")
        self.assertEqual(by_id["rescue_contrast"]["status"], "NO_MATCH")
        self.assertIn(
            "two_rescue_actions",
            by_id["rescue_contrast"]["match"]["unfilled_required_slots"],
        )
        world = by_id["exclusive_allocation"]["proposals"][0]["candidate"]["world_model"]
        self.assertEqual(len(world["effects"]), 8)

    def test_chooser_prefers_rescue_contrast_and_does_not_kill_the_dog(self):
        package = z10.export_candidate_graph(RESCUE, package_id="ensemble_rescue")
        choice = choose_blueprint(package)
        self.assertEqual(choice["chosen_blueprint_id"], "rescue_contrast")
        proposal = next(row for row in choice["considered"] if row["blueprint_id"] == "rescue_contrast")
        world = proposal["proposals"][0]["candidate"]["world_model"]
        envelope = proposal["proposals"][0]
        self.assertIn("selection", envelope)
        self.assertIn("slot_bindings", envelope)
        self.assertIn("clauses", envelope)
        labels = {row["label"] for row in world["parties"]}
        self.assertIn("the dog", labels)
        self.assertEqual(len(world["actions"]), 2)
        self.assertTrue(any(row["predicate"] == "live" for row in world["effects"]))
        self.assertFalse(any(row["predicate"] == "die" for row in world["effects"]))
        self.assertFalse(proposal["optional_slots"]["foregone_harms"])
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

    def test_discourse_families_emit_authorized_or_incomplete_envelopes(self):
        package = z10.export_candidate_graph(RESCUE, package_id="ensemble_discourse")
        choice = choose_blueprint(package)
        rows = [row for row in choice["considered"] if row["blueprint_id"] in {
            "ability_permission", "deontic_rule", "promise_reliance", "disputed_report",
        }]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(row["proposals"]) == 1 for row in rows))
        filled = [row for row in rows if row["status"] == "FILLED"]
        incomplete = [row for row in rows if row["status"] != "FILLED"]
        self.assertTrue(filled)
        for row in filled:
            proposal = row["proposals"][0]
            self.assertTrue(proposal["admission_authorized"])
            world = proposal["candidate"]["world_model"]
            self.assertEqual(world["schema_version"], "1.4")
            self.assertEqual(world["effects"], [])
            self.assertFalse(proposal["world_withheld"])
        for row in incomplete:
            proposal = row["proposals"][0]
            self.assertIsNone(proposal["candidate"])
            self.assertFalse(proposal["admission_authorized"])
            self.assertTrue(proposal["world_withheld"])

    def test_diversion_and_risk_build_source_grounded_candidates(self):
        for text, blueprint_id in (
                (DIVERSION, "diversion_redirection"),
                (RISK, "uncertain_risk")):
            with self.subTest(blueprint_id=blueprint_id):
                choice = choose_blueprint(
                    z10.export_candidate_graph(text, package_id=blueprint_id))
                self.assertEqual(choice["chosen_blueprint_id"], blueprint_id)
                row = next(item for item in choice["considered"]
                           if item["blueprint_id"] == blueprint_id)
                proposal = row["proposals"][0]
                self.assertTrue(proposal["admission_authorized"])
                self.assertTrue(proposal["pre_world_assessment"][
                    "eligible_for_world_state"])
                self.assertTrue(proposal["accepted_evidence"])

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_rescue_and_trolley_graphs_pass_parliament_admission(self):
        graphs = []
        for text, package_id in (
                (RESCUE, "admit_rescue"),
                (TROLLEY, "admit_trolley"),
                (CONDITIONAL, "admit_conditional"),
                (DIVERSION, "admit_diversion"),
                (RISK, "admit_risk")):
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
from global_workspace.world_admission import restore_admitted_world
from global_workspace.world_state import parse_world_model
for graph in json.load(open(sys.argv[2])):
    proposal = graph["proposal"]
    world = proposal["candidate"]["world_model"]
    ids = [row["action_id"] for row in world["actions"]]
    compiled = parse_world_model(
        world, clauses=graph["clauses"], action_ids=ids,
        action_texts=dict(zip(ids, graph["actions"])), require_completeness=True)
    restore_admitted_world(compiled.as_dict())
    result = _admit_action_source_rows(proposal["candidate"], graph["actions"], ids, graph["clauses"])
    assert result["status"] in {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"}, result["errors"]
    restore_admitted_world(result["world_model"])
print("COMMITTED")
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", script,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
