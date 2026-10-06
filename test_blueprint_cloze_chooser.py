import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from blueprint_allocation_invariants import group_quantity
from blueprint_kind_license import license_kind
from blueprint_cloze_chooser import _implied_sentence, _link, assess_question, choose_by_cloze


WATER = (
    "An AI bot must decide whether to devote water to a farm that can sustain a city's dietary needs "
    "or a small town that will lose 100 people to drowning."
)
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
CONDITIONAL = (
    "If Maria pulls the lever, one worker will live. "
    "If Maria does not pull the lever, five workers will live."
)


class Script:
    def __init__(self, ranking, sheets):
        self.ranking = ranking
        self.sheets = sheets
        self.cloze_ids = []
        self.prompts = []

    def __call__(self, messages):
        content = messages[-1]["content"]
        self.prompts.append(content)
        if content.startswith("Which three templates"):
            return json.dumps({"templates": self.ranking})
        for blueprint_id, answers in self.sheets.items():
            if f"Template: {blueprint_id}" in content:
                self.cloze_ids.append(blueprint_id)
                return json.dumps({"answers": answers})
        from blueprint_cloze_chooser import _BY_ID
        for blueprint_id in _BY_ID:
            if f"Template: {blueprint_id}" in content:
                self.cloze_ids.append(blueprint_id)
                return json.dumps({"answers": _none_sheet(blueprint_id)})
        raise AssertionError(content[:240])


def _none_sheet(blueprint_id):
    from blueprint_cloze_chooser import _BY_ID
    return {item["id"]: "NONE" for item in _BY_ID[blueprint_id]["items"]}


class ClozeChooserTests(unittest.TestCase):
    def _allocation_run(self, text, answers):
        sheet = _none_sheet("exclusive_allocation")
        sheet.update(answers)
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": sheet,
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        return choose_by_cloze(text, script)

    def test_z10_recovers_synonymous_exclusivity_and_postposed_conditions(self):
        text = (
            "A hospital has one vial of antidote. Dr. Lee must administer the antidote "
            "either to Ana or to Bo, but cannot treat both. Ana has an 85% chance of "
            "recovering if Dr. Lee administers it to Ana. Bo has a 40% chance of "
            "recovering if Dr. Lee administers it to Bo. Whoever is untreated will die."
        )
        result = self._allocation_run(text, {
            "decider": "Dr. Lee", "resource": "antidote",
            "assignment": "administer the antidote",
            "first_recipient": "Ana", "second_recipient": "Bo",
            "first_outcome": "Ana has an 85% chance of recovering",
            "second_outcome": "Bo has a 40% chance of recovering",
        })
        winner = result["considered"][0]
        self.assertEqual(winner["slots"]["exclusivity"], "cannot treat both")
        self.assertEqual(winner["slots"]["quantity"], "one vial")
        self.assertEqual(winner["slots"]["nonreceipt"], "Whoever is untreated will die")
        self.assertIn("administers it to Ana", winner["slots"]["first_transfer"])
        world = result["graph"]["candidate"]["world_model"]
        self.assertEqual(len([e for e in world["effects"] if e["predicate"] == "die"]), 2)
        self.assertEqual(result["question"]["exclusivity_proof"]["status"], "EXPLICIT")

    def test_z10_recovers_passive_quantity_and_without_outcome(self):
        text = (
            "There is a single dose of serum. It can be given by Priya to either Malik "
            "or Rosa, but not both. If Malik receives the serum, he has a 70% chance "
            "of surviving. If Rosa receives the serum, she has a 60% chance of surviving. "
            "The person left without serum will die."
        )
        result = self._allocation_run(text, {
            "decider": "Priya", "resource": "serum", "assignment": "given by Priya",
            "first_recipient": "Malik", "second_recipient": "Rosa",
            "exclusivity": "not both",
            "first_outcome": "he has a 70% chance of surviving",
            "second_outcome": "she has a 60% chance of surviving",
        })
        winner = result["considered"][0]
        self.assertEqual(winner["slots"]["quantity"], "a single dose")
        self.assertEqual(winner["slots"]["nonreceipt"], "The person left without serum will die")
        world = result["graph"]["candidate"]["world_model"]
        self.assertEqual({e["predicate"] for e in world["effects"]
                          if e["effect_kind"] == "RESOURCE_TRANSFER"}, {"receive"})
        self.assertEqual(len([e for e in world["effects"] if e["predicate"] == "die"]), 2)

    def test_one_dose_and_or_does_not_license_exclusivity(self):
        text = (
            "A clinic has one indivisible dose of medicine. Ada may give the dose to "
            "Ben or Cara. If Ben receives it, Ben will recover. If Cara receives it, "
            "Cara will recover."
        )
        result = self._allocation_run(text, {
            "decider": "Ada", "resource": "the dose", "assignment": "give the dose",
            "first_recipient": "Ben", "second_recipient": "Cara",
            "first_outcome": "Ben will recover", "second_outcome": "Cara will recover",
        })
        self.assertEqual(result["question"]["exclusivity_proof"]["status"], "HYPOTHESIZED")
        self.assertNotIn("exclusivity", result["considered"][0]["slots"])
        self.assertIsNone(result["graph"])
        self.assertTrue(any("exclusive" in reason.lower() for reason in result["world_withheld"]))

    def test_relation_generation_uses_explicit_cues_and_conservative_default(self):
        host = {"clause_id": "C0", "text": "Pulling the lever causes the trolley to move."}
        self.assertEqual(_link("A0", "E1", "E2", "CERTAIN", host)["link_relation"],
                         "CAUSES")
        host["text"] = "If Maria pulls the lever, the trolley moves."
        self.assertEqual(_link("A0", "E1", "E2", "CERTAIN", host)["link_relation"],
                         "ENABLES")

    def test_allocation_recovers_assignment_from_a_copied_transfer(self):
        sheet = _none_sheet("exclusive_allocation")
        sheet.update({
            "decider": "Ada", "resource": "one dose of medicine",
            "assignment": "give", "first_transfer": "give the medicine to Ben",
            "quantity": "one dose", "first_recipient": "Ben",
            "second_recipient": "Cara", "exclusivity": "but not both",
        })
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "rescue_contrast"],
            {
                "exclusive_allocation": sheet,
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "rescue_contrast": _none_sheet("rescue_contrast"),
            },
        )
        result = choose_by_cloze(MEDICINE, script)
        winner = result["considered"][0]
        assignment = next(row for row in winner["items"] if row["id"] == "assignment")
        self.assertEqual(assignment["reason"], "recovered_from_copied_transfer")
        self.assertEqual(winner["slots"]["assignment"], "give the medicine to Ben")

    def test_water_sentence_fills_the_allocation_blanks_it_actually_states(self):
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "rescue_contrast"],
            {
                "exclusive_allocation": {
                    "decider": "An AI bot must decide",
                    "resource": "water",
                    "assignment": "decide whether to devote water",
                    "first_transfer": "devote water",
                    "second_transfer": "devote water to a small town",
                    "quantity": "100 people",
                    "first_recipient": "a farm that can sustain a city's dietary needs",
                    "second_recipient": "a small town that will lose 100 people to drowning",
                    "exclusivity": "whether to devote water to a farm that can sustain a city's dietary needs or a small town that will lose 100 people to drowning",
                    "first_outcome": "can sustain a city's dietary needs",
                    "first_hedge": "can",
                    "second_outcome": "will lose 100 people to drowning",
                    "second_hedge": "NONE",
                    "first_implied_process": "providing water for crops to be grown",
                    "second_implied_process": "flooding",
                    "survival_chance": "NONE",
                    "nonreceipt": "will lose 100 people to drowning",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "rescue_contrast": _none_sheet("rescue_contrast"),
            },
        )
        result = choose_by_cloze(WATER, script)
        self.assertEqual(script.cloze_ids, [
            "exclusive_allocation", "exclusive_allocation",
            "conditional_outcome", "rescue_contrast"])
        self.assertIn("omission_harm", result["left_out"])
        self.assertIn("disputed_report", result["left_out"])
        self.assertEqual(result["chosen_blueprint_id"], "exclusive_allocation")
        self.assertEqual(result["status"], "WITHHELD")
        winner = result["considered"][0]
        self.assertIn("quantity", winner["unfilled"])
        self.assertIn("exclusivity", winner["unfilled"])
        self.assertIn("nonreceipt", winner["unfilled"])
        rejected = {row["id"]: row["reason"] for row in winner["items"] if row["verdict"] == "rejected"}
        self.assertEqual(rejected["quantity"], "does_not_answer")
        self.assertEqual(rejected["exclusivity"], "too_long")
        self.assertEqual(rejected["first_transfer"], "does_not_answer")
        self.assertEqual(rejected["first_hedge"], "does_not_answer")
        self.assertEqual(winner["slots"]["assignment"], "devote water")
        self.assertEqual(winner["slots"]["decider"], "An AI bot")
        self.assertEqual(
            winner["slots"]["first_recipient"],
            "a farm that can sustain a city's dietary needs",
        )
        self.assertEqual(
            winner["slots"]["second_recipient"],
            "a small town that will lose 100 people to drowning",
        )
        self.assertIsNone(result["graph"])
        self.assertEqual(result["proposals"][0]["candidate"], None)
        self.assertFalse(result["proposals"][0]["admission_authorized"])
        self.assertFalse(result["proposals"][0]["pre_world_assessment"][
            "eligible_for_world_state"])
        self.assertEqual(result["question"]["exclusivity"], "unspecified")
        self.assertFalse(result["question"]["eligible_for_world_state"])
        mentions = {row["mention"] for row in result["question"]["participants"]}
        self.assertIn("city's dietary needs", mentions)
        self.assertTrue(any("averted-alternative" in reason for reason in result["world_withheld"]))
        self.assertTrue(any("city's dietary needs" in reason for reason in result["world_withheld"]))
        self.assertEqual(winner["slots"]["first_implied_process"], "providing water for crops to be grown")
        self.assertEqual(winner["slots"]["second_implied_process"], "flooding")
        implied = next(row for row in winner["items"] if row["id"] == "first_implied_process")
        second = next(row for row in winner["items"] if row["id"] == "second_implied_process")
        self.assertEqual(implied["reason"], "implied")
        self.assertIn(
            'how "water" leads to "can sustain a city\'s dietary needs"',
            implied["sentence"],
        )
        self.assertIn(
            'how "water" leads to "will lose 100 people to drowning"',
            second["sentence"],
        )
        self.assertNotIn("decide whether", implied["sentence"])
        long_outcome = _implied_sentence({
            "resource": "water",
            "second_recipient": "a small town",
            "second_outcome": "a small town that will lose 100 people to drowning",
        }, "second_implied_process")
        self.assertIn('how "water" leads to "will lose 100 people to drowning"', long_outcome)
        self.assertTrue(any(
            'how "water" leads to "can sustain a city\'s dietary needs"' in prompt
            for prompt in script.prompts
        ))
        self.assertIn("providing water for crops to be grown", winner["slots"]["first_implied_process"])
        self.assertIn("flooding", winner["slots"]["second_implied_process"])

    def test_medicine_blanks_keep_the_chance_and_the_stated_death(self):
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": {
                    "decider": "Ada",
                    "resource": "the medicine",
                    "assignment": "give the medicine",
                    "quantity": "one dose",
                    "first_recipient": "Ben",
                    "second_recipient": "Cara",
                    "exclusivity": "but not both",
                    "first_outcome": "Ben has a 95% chance of survival",
                    "second_outcome": "Cara has a 5% chance of survival",
                    "survival_chance": "95% chance",
                    "nonreceipt": "The patient who does not get the medicine will die",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        result = choose_by_cloze(MEDICINE, script)
        attempts = result["candidate_attempts"]
        self.assertEqual(len(attempts), 3)
        self.assertEqual(attempts[0]["blueprint_id"], "exclusive_allocation")
        self.assertTrue(attempts[0]["selected"])
        self.assertTrue(attempts[0]["contract_valid"])
        self.assertIsNotNone(attempts[0]["proposal"]["candidate"])
        self.assertTrue(all("unfilled_slots" in row for row in attempts))
        world = result["graph"]["candidate"]["world_model"]
        chances = [row["likelihood_qualifiers"] for row in world["effects"] if row["modality"] == "PROBABILISTIC"]
        self.assertEqual(chances, [["95% chance"], ["5% chance"]])
        deaths = [row for row in world["effects"] if row["predicate"] == "die"]
        self.assertEqual(len(deaths), 2)
        self.assertTrue(all(row["modality"] == "CERTAIN" and row["polarity"] == "ADVERSE" for row in deaths))
        complements = [
            row for row in world["effects"]
            if row["derivation_operation"] == "EXCLUSIVE_ALLOCATION_COMPLEMENT"
        ]
        self.assertEqual(len(complements), 2)
        self.assertTrue(all(row["predicate"] == "NOT_RECEIVES" for row in complements))
        self.assertTrue(result["graph"]["pre_world_assessment"][
            "eligible_for_world_state"])

    def test_group_recipient_headcount_stays_on_transfer_and_out_of_complement(self):
        text = (
            "A clinic has one dose of antiviral. Dr. Rivera must give the antiviral "
            "to either one child or three adult patients, but not both. If Dr. Rivera "
            "gives the antiviral to the child, the child has an 80% chance of survival. "
            "If Dr. Rivera gives the antiviral to the three adult patients, each of "
            "the three adult patients has a 60% chance of survival. The patients who "
            "do not receive the antiviral will die."
        )
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": {
                    "decider": "Dr. Rivera", "resource": "antiviral",
                    "assignment": "give the antiviral", "quantity": "one dose",
                    "first_recipient": "one child",
                    "second_recipient": "three adult patients",
                    "exclusivity": "but not both",
                    "first_outcome": "the child has an 80% chance of survival",
                    "second_outcome": (
                        "each of the three adult patients has a 60% chance of survival"
                    ),
                    "first_hedge": "80% chance",
                    "second_hedge": "60% chance",
                    "first_branch_sentence": (
                        "If Dr. Rivera gives the antiviral to the child, the child "
                        "has an 80% chance of survival"
                    ),
                    "second_branch_sentence": (
                        "If Dr. Rivera gives the antiviral to the three adult patients, "
                        "each of the three adult patients has a 60% chance of survival"
                    ),
                    "nonreceipt": (
                        "The patients who do not receive the antiviral will die"
                    ),
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        question = {
            "ethical_question": "give to the child / give to the adult patients",
            "scenario_options": [],
            "participants": [],
            "exclusivity": "evidenced",
            "eligible_for_world_state": True,
        }
        with mock.patch("blueprint_cloze_chooser.assess_question",
                        return_value=question):
            proposal = choose_by_cloze(text, script)["graph"]
        world = proposal["candidate"]["world_model"]
        kinds = {row["label"]: row["kind"] for row in world["parties"]}
        self.assertEqual(kinds["Dr. Rivera"], "PERSON")
        self.assertEqual(kinds["antiviral"], "RESOURCE")
        self.assertEqual(kinds["one child"], "PERSON")
        self.assertEqual(kinds["three adult patients"], "HUMAN_GROUP")
        self.assertTrue(all(row.get("kind_origin") != "UNRESOLVED"
                            for row in world["parties"]))
        self.assertEqual(
            [row["intervention"] for row in world["actions"]],
            [
                "gives the antiviral to the child",
                "gives the antiviral to the three adult patients",
            ],
        )
        transfers = [
            row for row in world["effects"]
            if row["effect_kind"] == "RESOURCE_TRANSFER"
        ]
        adult_transfer = next(row for row in transfers if row["party_id"] == "P4")
        self.assertEqual(adult_transfer["quantities"], ["three"])
        complements = [
            row for row in world["effects"]
            if row["derivation_operation"] == "EXCLUSIVE_ALLOCATION_COMPLEMENT"
        ]
        child_complement = next(
            row for row in complements if row["party_id"] == "P3"
        )
        self.assertEqual(child_complement["quantities"], ["one"])
        self.assertNotIn("three", child_complement["quantities"])
        root = os.environ.get("PARLIAMENT_SMOKE_ROOT")
        python = os.environ.get("PARLIAMENT_SMOKE_PYTHON")
        if root and python:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "world.json"
                path.write_text(json.dumps(proposal), encoding="utf-8")
                code = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.world_admission import restore_admitted_world
from global_workspace.world_state import parse_world_model
p = json.load(open(sys.argv[2]))
w = p["candidate"]["world_model"]; w["schema_version"] = "1.3"
ids = [a["action_id"] for a in w["actions"]]
actions = [a["intervention"] for a in w["actions"]]
compiled = parse_world_model(
    w, clauses=p["clauses"], action_ids=ids,
    action_texts=dict(zip(ids, actions)), require_completeness=True)
restore_admitted_world(compiled.as_dict())
'''
                completed = subprocess.run(
                    [python, "-c", code, root, str(path)],
                    text=True, capture_output=True, timeout=60)
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr)

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_normalized_medicine_cloze_candidate_passes_parliament_admission(self):
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": {
                    "decider": "Ada", "resource": "the medicine",
                    "assignment": "give the medicine", "quantity": "one dose",
                    "first_recipient": "Ben", "second_recipient": "Cara",
                    "exclusivity": "but not both",
                    "first_outcome": "Ben has a 95% chance of survival",
                    "second_outcome": "Cara has a 5% chance of survival",
                    "survival_chance": "95% chance",
                    "nonreceipt": "The patient who does not get the medicine will die",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        proposal = choose_by_cloze(MEDICINE, script)["graph"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.json"
            path.write_text(json.dumps(proposal), encoding="utf-8")
            code = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.local_specialists import _admit_action_source_rows
from global_workspace.world_state import parse_world_model
p = json.load(open(sys.argv[2]))
w = p["candidate"]["world_model"]; w["schema_version"] = "1.3"
ids = [a["action_id"] for a in w["actions"]]
actions = [a["intervention"] for a in w["actions"]]
parse_world_model(w, clauses=p["clauses"], action_ids=ids,
                  action_texts=dict(zip(ids, actions)), require_completeness=True)
r = _admit_action_source_rows(p["candidate"], actions, ids, p["clauses"])
assert r["status"] == "COMMITTED", r.get("errors")
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", code,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_rescue_blanks_build_two_exclusive_rescue_branches(self):
        script = Script(
            ["rescue_contrast", "exclusive_allocation", "conditional_outcome"],
            {
                "rescue_contrast": {
                    "rescuer": "Maria",
                    "first_saved": "the child",
                    "second_saved": "the dog",
                    "rescue_exclusivity": "but not both",
                    "first_rescue_action": "Maria saves the child",
                    "second_rescue_action": "Maria saves the dog",
                    "first_benefit": "the child will live",
                    "first_harm": "the dog will drown",
                    "second_benefit": "the dog will live",
                    "second_harm": "the child will drown",
                    "scene": "NONE",
                },
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
                "conditional_outcome": _none_sheet("conditional_outcome"),
            },
        )
        result = choose_by_cloze(RESCUE, script)
        self.assertEqual(result["chosen_blueprint_id"], "rescue_contrast")
        world = result["graph"]["candidate"]["world_model"]
        labels = {row["label"]: row["kind"] for row in world["parties"]}
        self.assertEqual(labels["the dog"], "ANIMAL")
        self.assertEqual(len(world["actions"]), 2)
        self.assertEqual(len(world["effects"]), 6)
        self.assertEqual(len(world["causal_links"]), 4)
        self.assertEqual(
            [row["intervention"] for row in world["actions"]],
            ["Maria saves the child", "Maria saves the dog"],
        )
        self.assertEqual(len([row for row in world["effects"] if row["predicate"] == "live"]), 2)
        self.assertEqual(len([row for row in world["effects"] if row["predicate"] == "drown"]), 2)
        self.assertEqual(
            result["graph"]["construction_problems"][0]["code"],
            "rescue_harm_missing_named_process",
        )

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_rescue_process_gap_is_an_explicit_parliament_rejection(self):
        script = Script(
            ["rescue_contrast", "conditional_outcome", "exclusive_allocation"],
            {
                "rescue_contrast": {
                    "rescuer": "Maria", "first_saved": "the child",
                    "second_saved": "the dog", "rescue_exclusivity": "but not both",
                    "first_rescue_action": "Maria saves the child",
                    "second_rescue_action": "Maria saves the dog",
                    "first_benefit": "the child will live",
                    "first_harm": "the dog will drown",
                    "second_benefit": "the dog will live",
                    "second_harm": "the child will drown", "scene": "NONE",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        proposal = choose_by_cloze(RESCUE, script)["graph"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.json"
            path.write_text(json.dumps(proposal), encoding="utf-8")
            code = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.world_state import parse_world_model
p = json.load(open(sys.argv[2]))
w = p["candidate"]["world_model"]; w["schema_version"] = "1.3"
ids = [a["action_id"] for a in w["actions"]]
actions = [a["intervention"] for a in w["actions"]]
parse_world_model(w, clauses=p["clauses"], action_ids=ids,
                  action_texts=dict(zip(ids, actions)), require_completeness=True)
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", code,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("insert a PROCESS", completed.stderr)

    def test_rescue_without_stated_harms_does_not_invent_them(self):
        text = (
            "Maria can save either the child or the dog, but not both. "
            "If Maria saves the child, the child will live. "
            "If Maria saves the dog, the dog will live."
        )
        script = Script(
            ["rescue_contrast", "conditional_outcome", "exclusive_allocation"],
            {
                "rescue_contrast": {
                    "rescuer": "Maria",
                    "first_saved": "the child", "second_saved": "the dog",
                    "rescue_exclusivity": "but not both",
                    "first_rescue_action": "Maria saves the child",
                    "second_rescue_action": "Maria saves the dog",
                    "first_benefit": "the child will live", "first_harm": "NONE",
                    "second_benefit": "the dog will live", "second_harm": "NONE",
                    "scene": "NONE",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        result = choose_by_cloze(text, script)
        world = result["graph"]["candidate"]["world_model"]
        self.assertEqual(len(world["actions"]), 2)
        self.assertEqual(len(world["effects"]), 4)
        self.assertFalse(any(row["polarity"] == "ADVERSE" for row in world["effects"]))

    def test_lever_blanks_keep_both_harms_and_leave_the_brake_off_the_actions(self):
        script = Script(
            ["omission_harm", "rescue_contrast", "exclusive_allocation"],
            {
                "omission_harm": {
                    "actor": "Maria",
                    "done": "pull the lever",
                    "omitted": "NONE",
                    "harm_done": "one worker will die",
                    "harm_omitted": "five workers will die",
                    "done_hedge": "If",
                    "omitted_hedge": "If Maria does not pull the lever",
                    "group_counts": "Five workers",
                    "instrument": "the brake",
                },
                "rescue_contrast": _none_sheet("rescue_contrast"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        question = {
            "ethical_question": "pulls / not pull",
            "scenario_options": [],
            "participants": [],
            "exclusivity": "unspecified",
        }
        with mock.patch("blueprint_cloze_chooser.assess_question",
                        return_value=question):
            result = choose_by_cloze(TROLLEY, script)
        self.assertEqual(result["chosen_blueprint_id"], "omission_harm")
        world = result["graph"]["candidate"]["world_model"]
        self.assertEqual(
            [row["intervention"] for row in world["actions"]],
            ["pulls the lever", "does not pull the lever"],
        )
        self.assertEqual(len([row for row in world["effects"] if row["predicate"] == "die"]), 2)
        deaths = [row for row in world["effects"] if row["predicate"] == "die"]
        self.assertTrue(all(row["modality"] == "CERTAIN" for row in deaths))
        one = next(row for row in deaths if "one worker" in row["outcome"])
        self.assertEqual(one["quantities"], ["one"])
        aversions = [
            row for row in world["effects"]
            if row["derivation_operation"] == "AVERTED_ALTERNATIVE_HARM"
        ]
        self.assertEqual(len(aversions), 2)
        self.assertTrue(all(row["derivation_assumptions"] for row in aversions))
        self.assertTrue(all(
            row["outcome_type_transformation"] == "POLARITY_INVERTED"
            for row in aversions
        ))
        self.assertTrue(all(
            row["derivation_operation"] == "SOURCE_STIPULATED_CAUSAL"
            for row in deaths
        ))
        pull = next(row for row in world["effects"] if "lever" in row["outcome"]
                    and "not" not in row["outcome"] and row["directness"] == "DIRECT")
        self.assertEqual(pull["source_proposition"], "pulls the lever")
        self.assertNotIn("one", pull["quantities"])
        process_party = next(row for row in world["parties"]
                             if row["party_id"] == pull["party_id"])
        self.assertEqual((process_party["label"], process_party["kind"]),
                         ("the lever", "PROCESS"))
        self.assertEqual(
            [(row["description"], row["polarity"]) for row in world["conditions"]],
            [
                ("If Maria pulls the lever", "POSITIVE"),
                ("If Maria does not pull the lever", "NEGATED"),
            ],
        )
        self.assertTrue(all(not row["condition_ids"] for row in deaths))
        process_states = [row for row in world["effects"]
                          if row["effect_kind"] == "PHYSICAL_STATE"]
        self.assertEqual(len(process_states), 2)
        self.assertTrue(all(row["party_id"] == process_party["party_id"]
                            for row in process_states))
        self.assertTrue(any("the brake" in note for note in result["graph"]["notes"]))
        self.assertNotIn("the brake", [row["intervention"] for row in world["actions"]])
        omitted = next(row for row in result["considered"][0]["items"]
                       if row["id"] == "omitted")
        self.assertEqual(omitted["reason"], "recovered_from_z10_conditional_structure")

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_trolley_omission_cloze_candidate_passes_parliament_admission(self):
        script = Script(
            ["omission_harm", "rescue_contrast", "exclusive_allocation"],
            {
                "omission_harm": {
                    "actor": "Maria",
                    "done": "pulls the lever",
                    "omitted": "does not pull the lever",
                    "harm_done": "one worker will die",
                    "harm_omitted": "five workers will die",
                    "done_hedge": "If Maria pulls the lever",
                    "omitted_hedge": "If Maria does not pull the lever",
                    "group_counts": "Five workers",
                    "instrument": "the brake",
                },
                "rescue_contrast": _none_sheet("rescue_contrast"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        question = {
            "ethical_question": "pulls / not pull",
            "scenario_options": [],
            "participants": [],
            "exclusivity": "unspecified",
        }
        with mock.patch("blueprint_cloze_chooser.assess_question",
                        return_value=question):
            proposal = choose_by_cloze(TROLLEY, script)["graph"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.json"
            path.write_text(json.dumps(proposal), encoding="utf-8")
            code = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.local_specialists import _admit_action_source_rows
from global_workspace.world_state import parse_world_model
p = json.load(open(sys.argv[2]))
w = p["candidate"]["world_model"]; w["schema_version"] = "1.3"
ids = [a["action_id"] for a in w["actions"]]
actions = [a["intervention"] for a in w["actions"]]
model = parse_world_model(
    w, clauses=p["clauses"], action_ids=ids,
    action_texts=dict(zip(ids, actions)), require_completeness=True)
assert model.admission.status in {
    "COMMITTED", "COMMITTED_WITH_UNCERTAINTY"}, model.admission.status
result = _admit_action_source_rows(p["candidate"], actions, ids, p["clauses"])
assert result["status"] in {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"}, result
assert result["world_model_status"] in {
    "COMMITTED", "COMMITTED_WITH_UNCERTAINTY"}, result
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", code,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_children_count_as_a_human_group(self):
        group = license_kind("two children", role="bearer")
        self.assertEqual(group["kind"], "HUMAN_GROUP")
        self.assertEqual(group["origin"], "STRUCTURALLY_DERIVED")
        self.assertEqual(group_quantity("two children"), ["two"])
        one = license_kind("one child", role="bearer")
        self.assertEqual(one["kind"], "PERSON")
        self.assertEqual(group_quantity("one child"), [])

    def test_conditional_template_keeps_two_branches_separate(self):
        script = Script(
            ["conditional_outcome", "omission_harm", "exclusive_allocation"],
            {
                "conditional_outcome": {
                    "actor": "Maria",
                    "condition": "If Maria pulls the lever",
                    "bearer": "one worker",
                    "outcome": "one worker will live",
                    "second_condition": "If Maria does not pull the lever",
                    "second_bearer": "five workers",
                    "second_outcome": "five workers will live",
                    "chance": "NONE",
                },
                "omission_harm": _none_sheet("omission_harm"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        result = choose_by_cloze(CONDITIONAL, script)
        world = result["graph"]["candidate"]["world_model"]
        self.assertEqual(len(world["actions"]), 2)
        self.assertEqual(
            [row["intervention"] for row in world["actions"]],
            ["pulls the lever", "does not pull the lever"],
        )
        outcomes = [row for row in world["effects"]
                    if row["effect_kind"] == "HEALTH_OUTCOME"]
        labels = {row["party_id"]: row["label"] for row in world["parties"]}
        self.assertEqual([labels[row["party_id"]] for row in outcomes],
                         ["one worker", "five workers"])
        self.assertEqual(
            [(row["description"], row["polarity"]) for row in world["conditions"]],
            [
                ("If Maria pulls the lever", "POSITIVE"),
                ("If Maria does not pull the lever", "NEGATED"),
            ],
        )
        self.assertTrue(all(row["link_relation"] == "ENABLES"
                            for row in world["causal_links"]))
        self.assertTrue(all(not row["condition_ids"]
                            for row in world["causal_links"]))
        self.assertTrue(all(row["modality"] == "CERTAIN" for row in outcomes))
        alternatives = result["graph"]["relation_alternatives"]
        self.assertTrue(all(row["status"] == "UNRESOLVED" for row in alternatives))
        self.assertTrue(all(row["alternatives"] == ["ENABLES", "CAUSES"]
                            for row in alternatives))
        self.assertEqual(
            result["graph"]["unresolved_readings"][0]["kind"],
            "conditional_relation",
        )
        self.assertEqual(len(result["graph"]["clauses"]), 2)

    def test_z10_recovers_postposed_conditionals_when_cloze_returns_none(self):
        text = (
            "One worker will live if Maria pulls the lever. "
            "Five workers will live if Maria does not pull the lever."
        )
        script = Script(
            ["conditional_outcome", "exclusive_allocation", "uncertain_risk"],
            {
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
                "uncertain_risk": _none_sheet("uncertain_risk"),
            },
        )
        result = choose_by_cloze(text, script)
        self.assertEqual(result["chosen_blueprint_id"], "conditional_outcome")
        world = result["graph"]["candidate"]["world_model"]
        self.assertEqual(
            [row["intervention"] for row in world["actions"]],
            ["pulls the lever", "does not pull the lever"],
        )
        self.assertEqual(
            [row["description"] for row in world["conditions"]],
            ["if Maria pulls the lever", "if Maria does not pull the lever"],
        )
        outcomes = [row for row in world["effects"]
                    if row["effect_kind"] == "HEALTH_OUTCOME"]
        self.assertEqual([row["quantities"] for row in outcomes],
                         [["One"], ["Five"]])
        recovered = result["considered"][0]["semantic_recoveries"]
        self.assertEqual(recovered["condition"]["producer"], "parsing_game_Z10")

    def test_z10_recovers_passive_trolley_actor_and_process(self):
        text = (
            "If the lever is pulled by Maria, one worker will die. "
            "If the lever is not pulled by Maria, five workers will die."
        )
        script = Script(
            ["omission_harm", "conditional_outcome", "exclusive_allocation"],
            {
                "omission_harm": _none_sheet("omission_harm"),
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        result = choose_by_cloze(text, script)
        self.assertEqual(result["chosen_blueprint_id"], "omission_harm")
        world = result["graph"]["candidate"]["world_model"]
        parties = {row["party_id"]: row for row in world["parties"]}
        self.assertTrue(all(parties[row["actor_party_id"]]["label"] == "Maria"
                            for row in world["actions"]))
        recipients = [parties[row["recipient_party_ids"][0]] for row in world["actions"]]
        self.assertTrue(all((row["label"], row["kind"]) == ("the lever", "PROCESS")
                            for row in recipients))
        self.assertEqual(
            [row["intervention"] for row in world["actions"]],
            ["the lever is pulled by Maria", "the lever is not pulled by Maria"],
        )

    def test_conditional_chance_stays_on_its_own_outcome(self):
        text = "If Ada gives medicine to Ben, Ben has a 95% chance of survival."
        script = Script(
            ["conditional_outcome", "exclusive_allocation", "uncertain_risk"],
            {
                "conditional_outcome": {
                    "actor": "Ada", "condition": "If Ada gives medicine to Ben",
                    "bearer": "Ben", "outcome": "Ben has a 95% chance of survival",
                    "second_condition": "NONE", "second_bearer": "NONE",
                    "second_outcome": "NONE", "chance": "95% chance",
                },
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
                "uncertain_risk": _none_sheet("uncertain_risk"),
            },
        )
        result = choose_by_cloze(text, script)
        outcome = result["graph"]["candidate"]["world_model"]["effects"][1]
        self.assertEqual(outcome["modality"], "PROBABILISTIC")
        self.assertEqual(outcome["likelihood_qualifiers"], ["95% chance"])

    @unittest.skipUnless(os.environ.get("PARLIAMENT_SMOKE_ROOT") and
                         os.environ.get("PARLIAMENT_SMOKE_PYTHON"),
                         "requires the isolated patched Parliament checkout")
    def test_conditional_candidate_passes_parliament_admission(self):
        script = Script(
            ["conditional_outcome", "exclusive_allocation", "uncertain_risk"],
            {
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
                "uncertain_risk": _none_sheet("uncertain_risk"),
            },
        )
        proposal = choose_by_cloze(CONDITIONAL, script)["graph"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.json"
            path.write_text(json.dumps(proposal), encoding="utf-8")
            code = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from global_workspace.local_specialists import _admit_action_source_rows
from global_workspace.world_state import parse_world_model
p = json.load(open(sys.argv[2]))
w = p["candidate"]["world_model"]; w["schema_version"] = "1.3"
ids = [a["action_id"] for a in w["actions"]]
actions = [a["intervention"] for a in w["actions"]]
model = parse_world_model(
    w, clauses=p["clauses"], action_ids=ids,
    action_texts=dict(zip(ids, actions)), require_completeness=True)
assert model.admission.status in {
    "COMMITTED", "COMMITTED_WITH_UNCERTAINTY"}, model.admission.status
result = _admit_action_source_rows(p["candidate"], actions, ids, p["clauses"])
assert result["status"] in {"COMMITTED", "COMMITTED_WITH_UNCERTAINTY"}, result
'''
            completed = subprocess.run(
                [os.environ["PARLIAMENT_SMOKE_PYTHON"], "-c", code,
                 os.environ["PARLIAMENT_SMOKE_ROOT"], str(path)],
                text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_promise_reliance_admits_commitment_not_delivery(self):
        text = "Ada promised Ben that Ada would deliver the medicine."
        script = Script(
            ["promise_reliance", "conditional_outcome", "exclusive_allocation"],
            {
                "promise_reliance": {
                    "promisor": "Ada",
                    "commitment_event": "promised",
                    "commitment_content": "Ada would deliver the medicine",
                    "promisee": "Ben",
                    "reliance": "NONE",
                    "breach": "NONE",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        result = choose_by_cloze(text, script)
        self.assertEqual(result["chosen_blueprint_id"], "promise_reliance")
        self.assertEqual(result["status"], "FILLED")
        self.assertIsNotNone(result["graph"])
        self.assertEqual(result["considered"][0]["graph_builder"], "discourse")
        self.assertEqual(result["world_withheld"], [])
        proposal = result["proposals"][0]
        self.assertTrue(proposal["admission_authorized"])
        world = proposal["candidate"]["world_model"]
        self.assertEqual(world["schema_version"], "1.4")
        self.assertEqual(world["effects"], [])
        self.assertEqual(world["commitments"][0]["promisor_party_id"], "P1")
        self.assertEqual(world["propositions"][0]["status"], "COMMITTED_CONTENT")
        self.assertNotIn("deliver", " ".join(row.get("predicate") or "" for row in world["effects"]))
        self.assertEqual(
            proposal["unresolved_readings"][0]["kind"], "commitment_status")
        self.assertEqual(
            proposal["accepted_evidence"]["commitment_event"], "promised")

    def test_remaining_semantic_plans_admit_discourse_without_occurrence(self):
        cases = (
            ("ability_permission", "Ada can give medicine to Ben.", {
                "actor": "Ada", "modal_action": "can give medicine", "target": "Ben",
                "modal_words": "can", "outcome": "NONE", "duty_or_prohibition": "NONE",
            }, "modal_operators", "GIVE"),
            ("deontic_rule", "Ada must deliver medicine to Ben.", {
                "deontic_words": "must", "governed_action": "deliver medicine to Ben",
                "bearer": "Ada", "authority": "NONE", "exception": "NONE", "sanction": "NONE",
            }, "normative_propositions", "DELIVER"),
            ("disputed_report", "Ada claims the medicine is safe.", {
                "source": "Ada", "report_words": "claims",
                "reported_content": "the medicine is safe",
                "competing_report": "NONE", "reliability": "NONE", "confirmation": "NONE",
            }, "reports", "SAFE"),
        )
        for blueprint_id, text, sheet, collection, forbidden in cases:
            with self.subTest(blueprint_id=blueprint_id):
                script = Script(
                    [blueprint_id, "conditional_outcome", "exclusive_allocation"],
                    {
                        blueprint_id: sheet,
                        "conditional_outcome": _none_sheet("conditional_outcome"),
                        "exclusive_allocation": _none_sheet("exclusive_allocation"),
                    },
                )
                result = choose_by_cloze(text, script)
                self.assertEqual(result["chosen_blueprint_id"], blueprint_id)
                self.assertEqual(result["status"], "FILLED")
                self.assertIsNotNone(result["graph"])
                proposal = result["proposals"][0]
                self.assertTrue(proposal["admission_authorized"])
                world = proposal["candidate"]["world_model"]
                self.assertEqual(world["schema_version"], "1.4")
                self.assertTrue(world[collection])
                self.assertEqual(world["effects"], [])
                blob = json.dumps(world["effects"]).casefold()
                self.assertNotIn(forbidden.casefold(), blob)
                self.assertTrue(proposal["slot_bindings"]["copied_spans"])
                self.assertTrue(proposal["unresolved_readings"])

    def test_diversion_and_risk_emit_normalized_candidates(self):
        cases = (
            ("diversion_redirection",
             "Maria can divert the trolley toward one worker, and one worker will die.", {
                 "actor": "Maria", "controllable_process": "the trolley",
                 "intervention": "divert the trolley",
                 "affected_party": "one worker", "outcome": "one worker will die",
                 "alternative_route": "NONE", "omission_branch": "NONE",
                 "uncertainty": "NONE",
             }),
            ("uncertain_risk", "Ada may administer medicine to Ben, and Ben could die.", {
                "actor": "Ada", "action": "administer medicine to Ben",
                "possible_outcome": "Ben could die", "affected_party": "Ben",
                "likelihood": "could", "second_action": "NONE",
                "second_affected_party": "NONE", "second_outcome": "NONE",
                "second_likelihood": "NONE", "expected_quantity": "NONE",
            }),
        )
        for blueprint_id, text, sheet in cases:
            with self.subTest(blueprint_id=blueprint_id):
                script = Script(
                    [blueprint_id, "conditional_outcome", "exclusive_allocation"],
                    {
                        blueprint_id: sheet,
                        "conditional_outcome": _none_sheet("conditional_outcome"),
                        "exclusive_allocation": _none_sheet("exclusive_allocation"),
                    },
                )
                result = choose_by_cloze(text, script)
                self.assertEqual(result["status"], "FILLED")
                self.assertIsNotNone(result["graph"])
                self.assertTrue(result["graph"]["admission_authorized"])
                self.assertEqual(result["proposals"], [result["graph"]])

    def test_no_accepted_blank_passes_nothing_forward(self):
        script = Script(
            ["exclusive_allocation", "rescue_contrast", "omission_harm"],
            {blueprint_id: _none_sheet(blueprint_id) for blueprint_id in
             ("exclusive_allocation", "rescue_contrast", "omission_harm")},
        )
        result = choose_by_cloze("The sky is blue today.", script)
        self.assertIsNone(result["chosen_blueprint_id"])
        self.assertEqual(result["status"], "NO_MATCH")
        self.assertIsNone(result["graph"])

    def test_none_and_composite_are_explicit_withheld_ranking_choices(self):
        for option in ("none", "composite"):
            with self.subTest(option=option):
                script = Script([option, "conditional_outcome", "exclusive_allocation"], {
                    "conditional_outcome": _none_sheet("conditional_outcome"),
                    "exclusive_allocation": _none_sheet("exclusive_allocation"),
                })
                result = choose_by_cloze("Ada described a difficult situation.", script)
                self.assertEqual(result["chosen_blueprint_id"], option)
                self.assertEqual(result["status"], "WITHHELD")
                self.assertIsNone(result["graph"])
                self.assertEqual(result["candidate_attempts"][0]["blueprint_id"], option)

    def test_exclusivity_proof_distinguishes_explicit_hypothesized_and_unknown(self):
        explicit = assess_question("Ada can give one dose to Ben or Cara, but not both.")
        hypothesized = assess_question("Ada can give one dose to Ben or Cara.")
        unknown = assess_question("Ada may help Ben and Cara.")
        self.assertEqual(explicit["exclusivity_proof"]["status"], "EXPLICIT")
        self.assertEqual(hypothesized["exclusivity_proof"]["status"], "HYPOTHESIZED")
        self.assertEqual(unknown["exclusivity_proof"]["status"], "UNKNOWN")

    def test_constructed_atoms_and_relations_have_external_provenance(self):
        script = Script(
            ["conditional_outcome", "omission_harm", "exclusive_allocation"],
            {
                "conditional_outcome": {
                    "actor": "Maria", "condition": "If Maria pulls the lever",
                    "bearer": "one worker", "outcome": "one worker will live",
                    "second_condition": "NONE", "second_bearer": "NONE",
                    "second_outcome": "NONE", "chance": "NONE",
                },
                "omission_harm": _none_sheet("omission_harm"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        proposal = choose_by_cloze(CONDITIONAL.split(". ")[0] + ".", script)["graph"]
        ids = {row["atom_id"] for row in proposal["construction_provenance"]}
        self.assertIn("action:A0", ids)
        self.assertIn("effect:E2", ids)
        relation = next(row for row in proposal["construction_provenance"]
                        if row["atom_id"] == "relation:0")
        self.assertEqual(relation["origin"], "UNRESOLVED")

    def test_must_deliver_is_deontic_even_when_allocation_ranks_first(self):
        text = "Ada must deliver water to Ben."
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "deontic_rule": {
                    "deontic_words": "must",
                    "governed_action": "deliver water",
                    "bearer": "Ada",
                    "target": "Ben",
                    "authority": "NONE",
                    "exception": "NONE",
                    "sanction": "NONE",
                },
                "exclusive_allocation": {
                    **_none_sheet("exclusive_allocation"),
                    "decider": "Ada",
                    "resource": "water",
                    "assignment": "deliver water",
                    "first_recipient": "Ben",
                    "first_transfer": "deliver water",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        result = choose_by_cloze(text, script)
        self.assertEqual(result["ranking"][0], "deontic_rule")
        self.assertEqual(result["chosen_blueprint_id"], "deontic_rule")
        self.assertEqual(result["status"], "FILLED")
        world = result["graph"]["candidate"]["world_model"]
        self.assertEqual(world["effects"], [])
        self.assertTrue(world["normative_propositions"])
        self.assertIn("Ben", {row["label"] for row in world["parties"]})
        self.assertIn("Ben", world["propositions"][0]["predication"])

    def test_must_decide_with_not_both_stays_allocation(self):
        text = (
            "An AI bot must decide whether to devote the only water tanker to a child "
            "or five elderly patients, but not both."
        )
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": {
                    **_none_sheet("exclusive_allocation"),
                    "decider": "An AI bot",
                    "resource": "the only water tanker",
                    "assignment": "devote the only water tanker",
                    "first_recipient": "a child",
                    "second_recipient": "five elderly patients",
                    "exclusivity": "but not both",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        result = choose_by_cloze(text, script)
        self.assertNotEqual(result["ranking"][0], "deontic_rule")
        self.assertEqual(result["chosen_blueprint_id"], "exclusive_allocation")

    def test_ability_copy_includes_the_target(self):
        text = "Ada can give water to Ben."
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "ability_permission": {
                    "actor": "Ada",
                    "modal_action": "give water",
                    "target": "Ben",
                    "modal_words": "can",
                    "outcome": "NONE",
                    "duty_or_prohibition": "NONE",
                },
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        result = choose_by_cloze(text, script)
        self.assertEqual(result["chosen_blueprint_id"], "ability_permission")
        world = result["graph"]["candidate"]["world_model"]
        self.assertEqual(world["propositions"][0]["predication"], "give water to Ben")
        self.assertIn("Ben", {row["label"] for row in world["parties"]})

    def test_commitment_event_keeps_the_verb_not_the_promisee(self):
        text = "Ada promised Ben she would deliver the water."
        script = Script(
            ["promise_reliance", "conditional_outcome", "exclusive_allocation"],
            {
                "promise_reliance": {
                    "promisor": "Ada",
                    "commitment_event": "Ada promised Ben",
                    "commitment_content": "she would deliver the water",
                    "promisee": "Ben",
                    "reliance": "NONE",
                    "breach": "NONE",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        result = choose_by_cloze(text, script)
        self.assertEqual(result["graph"]["accepted_evidence"]["commitment_event"], "promised")
        self.assertEqual(
            result["graph"]["candidate"]["world_model"]["commitments"][0]["commitment_event"],
            "promised",
        )

    def test_nonreceipt_death_parents_off_complement_without_quantity(self):
        text = (
            "An AI bot must decide whether to devote the only water tanker to a child "
            "or five elderly patients, but not both. If the child gets the water, "
            "the child will live. If the five elderly patients get the water, "
            "the five elderly patients will live. The people who do not get the water will die."
        )
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": {
                    **_none_sheet("exclusive_allocation"),
                    "decider": "An AI bot",
                    "resource": "the only water tanker",
                    "assignment": "devote the only water tanker",
                    "first_recipient": "a child",
                    "second_recipient": "five elderly patients",
                    "exclusivity": "but not both",
                    "first_outcome": "the child will live",
                    "second_outcome": "the five elderly patients will live",
                    "first_transfer": "the child gets the water",
                    "second_transfer": "the five elderly patients get the water",
                    "first_branch_sentence": (
                        "If the child gets the water, the child will live"
                    ),
                    "second_branch_sentence": (
                        "If the five elderly patients get the water, "
                        "the five elderly patients will live"
                    ),
                    "nonreceipt": "The people who do not get the water will die",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        result = choose_by_cloze(text, script)
        world = result["graph"]["candidate"]["world_model"]
        states = {
            row["effect_id"]: row for row in world["effects"]
            if row.get("predicate") == "NOT_RECEIVES"
        }
        self.assertEqual(len(states), 2)
        self.assertFalse(any(
            row.get("derivation_operation") == "EXCLUSIVE_ALLOCATION_COMPLEMENT"
            for row in states.values()
        ))
        deaths = [row for row in world["effects"] if row["predicate"] == "die"]
        self.assertEqual(len(deaths), 2)
        parties = {row["party_id"]: row for row in world["parties"]}
        effects = {row["effect_id"]: row for row in world["effects"]}
        for death in deaths:
            self.assertEqual(len(death["source_effect_ids"]), 1)
            parent = states[death["source_effect_ids"][0]]
            self.assertNotEqual(parent.get("directness"), "DIRECT")
            self.assertEqual(len(parent.get("source_effect_ids") or []), 1)
            resource_state = effects[parent["source_effect_ids"][0]]
            self.assertEqual(parties[resource_state["party_id"]]["kind"], "RESOURCE")
            self.assertNotEqual(resource_state.get("directness"), "DIRECT")
            self.assertEqual(len(resource_state.get("source_effect_ids") or []), 1)
            self.assertEqual(
                effects[resource_state["source_effect_ids"][0]].get("directness"),
                "DIRECT",
            )
        self.assertEqual(len(world["actions"]), 2)

    def test_allocation_interventions_are_branch_copies_not_glued_or_clauses(self):
        text = (
            "An AI bot must decide whether to devote the only water tanker to a child "
            "or five elderly patients, but not both. If the child gets the water, "
            "the child will live. If the five elderly patients get the water, "
            "the five elderly patients will live. The people who do not get the water will die."
        )
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": {
                    **_none_sheet("exclusive_allocation"),
                    "decider": "An AI bot",
                    "resource": "the only water tanker",
                    "assignment": "devote the only water tanker",
                    "first_recipient": "a child",
                    "second_recipient": "five elderly patients",
                    "exclusivity": "but not both",
                    "first_outcome": "the child will live",
                    "second_outcome": "the five elderly patients will live",
                    "first_branch_sentence": (
                        "If the child gets the water, the child will live"
                    ),
                    "second_branch_sentence": (
                        "If the five elderly patients get the water, "
                        "the five elderly patients will live"
                    ),
                    "nonreceipt": "The people who do not get the water will die",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        result = choose_by_cloze(text, script)
        world = result["graph"]["candidate"]["world_model"]
        interventions = [row["intervention"] for row in world["actions"]]
        self.assertEqual(len(interventions), 2)
        self.assertTrue(all(span in text for span in interventions), interventions)
        self.assertFalse(any("not both" in span.casefold() for span in interventions),
                         interventions)
        joined = " ".join(interventions).casefold()
        self.assertIn("child", joined)
        self.assertIn("elderly", joined)


if __name__ == "__main__":
    unittest.main()
