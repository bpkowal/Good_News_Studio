import json
import unittest

import ellipsis_proposer as proposer

SPELLED = "If Maria does not pull the lever, five workers will die."
GAPPED = "If Maria pulls the lever, one worker will die. If she does not, five workers will die."
COORDINATION = "Sam can save the child or the dog."
BARE = "John didn't go to the wedding, but Karen did."
SAME = "Maria pulls the lever. Anna does the same."


class AlignmentTests(unittest.TestCase):
    def test_the_spelled_out_words_are_the_training_span(self):
        spans = proposer.missing_spans(
            "John didn't go to the wedding, but Karen did.",
            "John didn't go to the wedding, but Karen did go to the wedding.")
        self.assertIn("go to the wedding", spans)


class ProposerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = proposer.load_proposer()

    def test_a_spelled_out_sentence_gets_no_copy(self):
        result = self.model.propose(SPELLED)
        self.assertFalse(result["present"])
        self.assertEqual(result["proposals"], [])
        self.assertIsNone(result["choice"])

    def test_coordination_without_a_gap_gets_no_copy(self):
        result = self.model.propose(COORDINATION)
        self.assertFalse(result["present"])
        self.assertEqual(result["proposals"], [])

    def test_a_bare_auxiliary_offers_the_antecedent_phrase(self):
        result = self.model.propose(GAPPED)
        self.assertTrue(result["present"])
        self.assertEqual(result["gate"], "verb_phrase")
        copied = [item["text"] for item in result["proposals"]]
        self.assertEqual(copied, ["pulls the lever"])
        self.assertEqual(result["proposals"][0]["status"], "unresolved")
        self.assertTrue(result["proposals"][0]["antecedent_copy"])
        self.assertIsNone(result["choice"])

    def test_other_remnants_stay_silent(self):
        silent = (
            "Someone will die, but Maria does not know who.",
            "Nora needs the medicine more than Omar.",
            "Lila gives the medicine to Omar and Nora the bandage to Sam.",
            "Five on the track. One on the side.",
            "Susan works at night, and Bill too.",
        )
        for text in silent:
            result = self.model.propose(text)
            self.assertEqual(result["proposals"], [], text)
            self.assertNotEqual(result.get("gate"), "stripping_not_nominal", text)

    def test_a_following_verb_stays_on_the_verb_phrase_gate(self):
        result = self.model.propose("Maria can save the child, but not if the dog is there.")
        self.assertEqual(result["gate"], "verb_phrase")
        self.assertEqual([item["text"] for item in result["proposals"]], ["save the child"])

    def test_but_not_copies_the_verb_onto_the_remnant(self):
        result = self.model.propose("Maria can save the child, but not the dog.")
        self.assertTrue(result["present"])
        self.assertEqual(result["gate"], "stripping_not_nominal")
        self.assertEqual(len(result["proposals"]), 1)
        copy = result["proposals"][0]
        self.assertEqual(copy["text"], "save")
        self.assertEqual(copy["role"], "object")
        self.assertEqual(copy["remnant"], "the dog")
        self.assertEqual(copy["subject"], "Maria")
        self.assertEqual(copy["partner"], "the child")
        self.assertEqual(copy["polarity"], "negative")
        self.assertEqual(copy["status"], "unresolved")
        self.assertTrue(copy["antecedent_copy"])
        self.assertIsNone(result["choice"])

    def test_a_prepositional_remnant_keeps_its_role(self):
        result = self.model.propose("John went to the movie theater but not to the park.")
        self.assertEqual(result["gate"], "stripping_not_nominal")
        self.assertEqual(len(result["proposals"]), 1)
        copy = result["proposals"][0]
        self.assertEqual(copy["verb"], "went")
        self.assertEqual(copy["role"], "destination")
        self.assertEqual(copy["remnant"], "to the park")
        self.assertEqual(copy["subject"], "John")
        self.assertEqual(copy["partner"], "to the movie theater")
        self.assertEqual(copy["polarity"], "negative")
        self.assertIsNone(result["choice"])

    def test_a_bare_remnant_with_two_roles_stays_a_choice(self):
        result = self.model.propose("Lila gives Omar the medicine, but not Nora.")
        self.assertEqual(result["gate"], "stripping_not_nominal")
        self.assertEqual(sorted(item["role"] for item in result["proposals"]), ["destination", "object"])
        self.assertTrue(all(item["remnant"] == "Nora" and item["polarity"] == "negative"
                            and item["subject"] == "Lila" for item in result["proposals"]))
        self.assertEqual(result["choice"]["kind"], "interpretation")
        self.assertEqual(result["choice"]["selection_rule"], "at_most_one")
        self.assertFalse(result["choice"]["exhaustive"])

    def test_a_bare_remnant_matches_the_object_not_a_conjunct(self):
        result = self.model.propose("Lila gives the medicine to Omar, but not the bandage.")
        self.assertEqual(len(result["proposals"]), 1)
        copy = result["proposals"][0]
        self.assertEqual(copy["role"], "object")
        self.assertEqual(copy["remnant"], "the bandage")
        self.assertEqual(copy["partner"], "the medicine")
        self.assertEqual(copy["subject"], "Lila")
        self.assertIsNone(result["choice"])

    def test_a_subject_remnant_is_the_new_subject(self):
        result = self.model.propose("Susan works at night, but not Bill.")
        self.assertEqual(len(result["proposals"]), 1)
        copy = result["proposals"][0]
        self.assertEqual(copy["role"], "subject")
        self.assertEqual(copy["remnant"], "Bill")
        self.assertEqual(copy["subject"], "Bill")
        self.assertEqual(copy["partner"], "Susan")
        self.assertEqual(copy["verb"], "works")
        self.assertEqual(copy["polarity"], "negative")
        self.assertIsNone(result["choice"])

    def test_a_stripping_sentence_survives_the_rest_of_the_scenario(self):
        hidden = self.model.propose(
            "Susan works at night, and Bill too. Maria can save the child, but not the dog.")
        self.assertEqual([(item["remnant"], item["role"]) for item in hidden["stripping"]],
                         [("the dog", "object")])
        self.assertNotIn("Bill", [item["remnant"] for item in hidden["stripping"]])
        mixed = self.model.propose("Maria can save the child, but not the dog. Anna does the same.")
        self.assertEqual(mixed["gate"], "verb_phrase")
        self.assertEqual([item["text"] for item in mixed["proposals"]], ["save the child"])
        self.assertEqual([(item["remnant"], item["role"]) for item in mixed["stripping"]],
                         [("the dog", "object")])
        self.assertIsNone(mixed["choice"])

    def test_a_remnant_in_the_middle_of_the_sentence_still_copies(self):
        frames = (
            ("Alex can eat the apple, but not the pear, if Jordan locks the gate.",
             "eat", "object", "the pear", "Alex"),
            ("Alex can eat the apple but not the pear and Jordan will call the office.",
             "eat", "object", "the pear", "Alex"),
            ("If Alex eats the apple but not the pear, Jordan will call the office.",
             "eats", "object", "the pear", "Alex"),
            ("Alex went to the office, but not to the park, after Jordan locks the gate.",
             "went", "destination", "to the park", "Alex"),
            ("If Maria pulls the lever but not the brake, one worker will die.",
             "pulls", "object", "the brake", "Maria"),
        )
        for text, verb, role, remnant, subject in frames:
            result = self.model.propose(text)
            self.assertEqual(result["gate"], "stripping_not_nominal", text)
            self.assertEqual(len(result["proposals"]), 1, result["proposals"])
            copy = result["proposals"][0]
            self.assertEqual(copy["verb"], verb, text)
            self.assertEqual(copy["role"], role, text)
            self.assertEqual(copy["remnant"], remnant, text)
            self.assertEqual(copy["subject"], subject, text)
            self.assertEqual(copy["polarity"], "negative", text)
        blocked = self.model.propose("Alex can eat the apple, but not if Jordan locks the gate.")
        self.assertNotEqual(blocked.get("gate"), "stripping_not_nominal")
        self.assertEqual(blocked.get("stripping"), [])

    def test_bare_both_closes_an_alternative_set_without_stripping(self):
        alternatives = (
            "Lila can give the medicine to either Omar or Nora, but not both.",
            "Dr. Reed must give the injector to either Sora or Malik, but not both.",
            "The team selected either plan A or plan B, but not both.",
            "Lila can give the medicine to Omar or Nora; but not both.",
            "Lila can give the medicine to either Omar or Nora,\nbut not both.",
        )
        for text in alternatives:
            result = self.model.propose(text)
            self.assertNotEqual(result.get("gate"), "stripping_not_nominal", text)
            self.assertEqual(result.get("stripping"), [], text)

        nominal = self.model.propose(
            "Maria invited Ana and Ben, but not both parents."
        )
        self.assertEqual(nominal["gate"], "stripping_not_nominal")
        self.assertEqual(nominal["proposals"][0]["remnant"], "both parents")

    def test_does_the_same_copies_the_earlier_phrase(self):
        result = self.model.propose(SAME)
        self.assertEqual([item["text"] for item in result["proposals"]], ["pulls the lever"])

    def test_a_participle_does_not_replace_the_main_phrase(self):
        result = self.model.propose(
            "By evening, melatonin levels begin to increase, leading to tiredness. Sam does, too.")
        copied = [item["text"] for item in result["proposals"]]
        self.assertTrue(any("begin" in text and "increase" in text for text in copied), copied)
        self.assertFalse(any("leading" in text for text in copied), copied)
        self.assertIsNone(result["choice"])

    def test_the_matching_auxiliary_is_the_copy(self):
        nearer = self.model.propose("Alex can eat the apple if Jordan locks the gate. Sam can, too.")
        inner = self.model.propose("Jordan will open the window after Alex can eat the apple. Sam can, too.")
        matched = self.model.propose(
            "Many of us might seek physical therapists if we are recovering from surgery. Sam might, too.")
        for result in (nearer, inner, matched):
            copied = [item["text"] for item in result["proposals"]]
            self.assertEqual(len(copied), 1, copied)
            self.assertIsNone(result["choice"])
        self.assertIn("eat the apple", nearer["proposals"][0]["text"])
        self.assertIn("eat the apple", inner["proposals"][0]["text"])
        self.assertIn("seek", matched["proposals"][0]["text"])
        self.assertNotIn("recovering", matched["proposals"][0]["text"])

    def test_karen_did_copies_the_wedding_trip(self):
        result = self.model.propose(BARE)
        self.assertEqual([item["text"] for item in result["proposals"]], ["go to the wedding"])

    def test_saved_holdout_keeps_the_gold_copy_in_a_small_set(self):
        payload = json.loads(proposer.MODEL_PATH.read_text(encoding="utf-8"))
        self.assertFalse(payload["rewrites_sentence"])
        self.assertTrue(payload["probe_not_used_for_fitting"])
        self.assertEqual(payload["world_state_commitment"], "not_authorized")
        self.assertEqual(payload["independent_of"], "parsing_game four-action relation CEM")
        self.assertEqual(payload["phrase_gate"], ["bare_auxiliary_clause", "too_either", "do_the_same"])
        self.assertEqual(payload["copy_kind"], "verb_and_nominal")
        self.assertIn("aux_match", payload["feature_names"])
        held = payload["metrics"]["synthetic_dev"]
        self.assertGreaterEqual(held["exact_when_recoverable"], 0.8)
        self.assertGreaterEqual(held["gold_in_set_when_recoverable"], 0.9)


if __name__ == "__main__":
    unittest.main()
