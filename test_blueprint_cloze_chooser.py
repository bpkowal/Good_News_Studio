import json
import unittest

from blueprint_cloze_chooser import _implied_sentence, choose_by_cloze


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
        raise AssertionError(content[:240])


def _none_sheet(blueprint_id):
    from blueprint_cloze_chooser import _BY_ID
    return {item["id"]: "NONE" for item in _BY_ID[blueprint_id]["items"]}


class ClozeChooserTests(unittest.TestCase):
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
        self.assertEqual(result["left_out"], ["omission_harm"])
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
        world = result["graph"]["candidate"]["world_model"]
        chances = [row["likelihood_qualifiers"] for row in world["effects"] if row["modality"] == "PROBABILISTIC"]
        self.assertEqual(chances, [["95% chance"], ["5% chance"]])
        deaths = [row for row in world["effects"] if row["predicate"] == "die"]
        self.assertEqual(len(deaths), 2)
        self.assertTrue(all(row["modality"] == "CERTAIN" and row["polarity"] == "ADVERSE" for row in deaths))

    def test_rescue_blanks_keep_the_dog_and_do_not_add_a_death(self):
        script = Script(
            ["rescue_contrast", "exclusive_allocation", "conditional_outcome"],
            {
                "rescue_contrast": {
                    "rescuer": "Maria",
                    "saved": "the child",
                    "not_saved": "the dog",
                    "survival": "the child will live",
                    "scene": "A child and a dog",
                    "foregone": "the dog",
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
        self.assertTrue(any(row["predicate"] == "live" for row in world["effects"]))
        self.assertFalse(any(row["predicate"] == "die" for row in world["effects"]))
        foregone = next(row for row in result["considered"][0]["items"] if row["id"] == "foregone")
        self.assertEqual(foregone["verdict"], "rejected")

    def test_lever_blanks_keep_both_harms_and_leave_the_brake_off_the_actions(self):
        script = Script(
            ["omission_harm", "rescue_contrast", "exclusive_allocation"],
            {
                "omission_harm": {
                    "actor": "Maria",
                    "done": "pull the lever",
                    "omitted": "does not pull the lever",
                    "harm_done": "one worker",
                    "harm_omitted": "five workers",
                    "instrument": "the brake",
                },
                "rescue_contrast": _none_sheet("rescue_contrast"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        result = choose_by_cloze(TROLLEY, script)
        self.assertEqual(result["chosen_blueprint_id"], "omission_harm")
        world = result["graph"]["candidate"]["world_model"]
        self.assertEqual(
            [row["intervention"] for row in world["actions"]],
            ["pull the lever", "does not pull the lever"],
        )
        self.assertEqual(len([row for row in world["effects"] if row["predicate"] == "die"]), 2)
        self.assertTrue(any("the brake" in note for note in result["graph"]["notes"]))
        self.assertNotIn("the brake", [row["intervention"] for row in world["actions"]])

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


if __name__ == "__main__":
    unittest.main()
