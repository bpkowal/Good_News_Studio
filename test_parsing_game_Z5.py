import unittest

import parsing_game_Z4 as z4
import parsing_game_Z5 as z5
from candidate_validation import empty_selection, validate_candidate_selection

UNCHANGED = (
    "Five workers are on the track. One worker is on the side track. "
    "If Maria pulls the lever, one worker will die. "
    "If Maria does not pull the lever, five workers will die.",
    "Sam can save the child or the dog.",
    "Someone will die, but Maria does not know who.",
    "Nora needs the medicine more than Omar.",
    "Five on the track. One on the side.",
    "Lila gives the medicine to Omar and Nora the bandage to Sam.",
    "Maria pulls the lever. Anna does the same.",
    "John didn't go to the wedding, but Karen did.",
)


class StrippingExportTests(unittest.TestCase):
    def package(self, text):
        result = z5.export_candidate_graph(text, package_id="pkg_z5_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        self.assertEqual(result["producer"]["name"], "parsing_game_Z5")
        self.assertIn("stripping_not_nominal_is_an_unresolved_copy", result["coverage"]["limitations"])
        return result

    def nodes(self, package):
        return {node["id"]: node for node in package["nodes"]}

    def stripping(self, package):
        return [item for item in package["candidates"]
                if item["provenance"][0]["method"] == "stripping_not_nominal"]

    def test_sentences_without_this_remnant_match_z4(self):
        for text in UNCHANGED:
            left = z4.export_candidate_graph(text, package_id="pkg_same")
            right = z5.export_candidate_graph(text, package_id="pkg_same")
            self.assertEqual(left["candidates"], right["candidates"], text)
            self.assertEqual(left["nodes"], right["nodes"], text)
            self.assertEqual(left["evidence"], right["evidence"], text)
            self.assertEqual(left["choice_sets"], right["choice_sets"], text)
            self.assertEqual(left["open_questions"], right["open_questions"], text)
            self.assertEqual(right["producer"]["version"], "Z5")

    def test_but_not_the_dog_is_a_negative_object_copy(self):
        package = self.package("Maria can save the child, but not the dog.")
        nodes = self.nodes(package)
        copies = self.stripping(package)
        predicates = [item for item in copies if item["type"] == "PREDICATION"]
        self.assertEqual(len(predicates), 1)
        pred = predicates[0]
        self.assertEqual(pred["scope"]["polarity"], "negative")
        self.assertEqual(pred["assessment"]["status"], "unresolved")
        self.assertEqual(nodes[pred["arguments"]["proposition"]]["predicate"], "save")
        roles = [(item["value"], nodes[item["arguments"]["mention"]]["label"], item)
                 for item in copies if item["type"] == "PARTICIPANT"]
        by_role = {role: (label, item) for role, label, item in roles}
        self.assertEqual(set(by_role), {"subject", "object"})
        self.assertEqual(by_role["subject"][0], "Maria")
        self.assertEqual(by_role["object"][0], "the dog")
        self.assertTrue(all(item["scope"]["polarity"] == "negative" and item["assessment"]["status"] == "unresolved"
                            for _, _, item in roles))
        spoken = [item for item in package["candidates"]
                  if item["type"] == "PARTICIPANT" and item["value"] == "object"
                  and item["provenance"][0]["method"] != "stripping_not_nominal"]
        self.assertEqual([nodes[item["arguments"]["mention"]]["label"] for item in spoken], ["the child"])
        self.assertTrue(all(item["scope"]["polarity"] == "positive" for item in spoken))
        role_ids = {by_role["subject"][1]["id"], by_role["object"][1]["id"]}
        self.assertFalse(any(role_ids.intersection(item["candidate_ids"]) for item in package["choice_sets"]))
        question = next(item for item in package["open_questions"]
                        if pred["id"] in item["candidate_ids"])
        self.assertEqual(question["kind"], "missing_representation")
        self.assertEqual(question["blocking_for"], [])
        self.assertIn(by_role["object"][1]["id"], question["candidate_ids"])

    def test_a_prepositional_remnant_is_a_destination(self):
        package = self.package("John went to the movie theater but not to the park.")
        nodes = self.nodes(package)
        copies = [item for item in self.stripping(package) if item["type"] == "PARTICIPANT"]
        self.assertEqual([(item["value"], nodes[item["arguments"]["mention"]]["label"]) for item in copies],
                         [("subject", "John"), ("destination", "the park")])
        self.assertTrue(all(item["scope"]["polarity"] == "negative" for item in copies))
        self.assertFalse(any(item["kind"] == "interpretation" and item["selection_rule"] == "at_most_one"
                             and any(copy["id"] in item["candidate_ids"] for copy in copies)
                             for item in package["choice_sets"]))

    def test_two_roles_stay_an_interpretation_choice(self):
        package = self.package("Lila gives Omar the medicine, but not Nora.")
        nodes = self.nodes(package)
        copies = [item for item in self.stripping(package) if item["type"] == "PARTICIPANT" and item["value"] != "subject"]
        self.assertEqual(sorted(item["value"] for item in copies), ["destination", "object"])
        self.assertTrue(all(nodes[item["arguments"]["mention"]]["label"] == "Nora" for item in copies))
        self.assertEqual(copies[0]["exclusive_with"], [copies[1]["id"]])
        self.assertEqual(copies[1]["exclusive_with"], [copies[0]["id"]])
        group = next(item for item in package["choice_sets"]
                     if set(item["candidate_ids"]) == {copy["id"] for copy in copies})
        self.assertEqual(group["kind"], "interpretation")
        self.assertEqual(group["selection_rule"], "at_most_one")
        self.assertFalse(group["exhaustive"])

    def test_a_subject_remnant_stays_unresolved(self):
        package = self.package("Susan works at night, but not Bill.")
        nodes = self.nodes(package)
        copies = [item for item in self.stripping(package) if item["type"] == "PARTICIPANT"]
        self.assertEqual([(item["value"], nodes[item["arguments"]["mention"]]["label"]) for item in copies],
                         [("subject", "Bill")])
        self.assertEqual(copies[0]["scope"]["polarity"], "negative")
        spoken = [item for item in package["candidates"]
                  if item["type"] == "PARTICIPANT" and item["value"] == "subject"
                  and nodes[item["arguments"]["mention"]]["label"] == "Susan"]
        self.assertEqual(len(spoken), 1)
        self.assertEqual(spoken[0]["scope"]["polarity"], "positive")
        self.assertNotEqual(spoken[0]["provenance"][0]["method"], "stripping_not_nominal")


RESCUE = (
    "A child and a dog are in the water. "
    "Maria can save the child, but not the dog. "
    "If Maria saves the child, the child will live."
)
MEDICINE = (
    "Lila has one dose of medicine. Omar needs the medicine. Nora needs the medicine. "
    "Lila can give the medicine to Omar, but not to Nora. "
    "If Lila gives the medicine to Omar, Omar will live. "
    "If Lila gives the medicine to Nora, Nora will live."
)
TROLLEY = (
    "Five workers are on the track. One worker is on the side track. "
    "Maria can pull the lever, but not the brake. "
    "If Maria pulls the lever, one worker will die. "
    "If Maria does not pull the lever, five workers will die."
)


class ScenarioStrippingTests(unittest.TestCase):
    def package(self, text):
        result = z5.export_candidate_graph(text, package_id="pkg_scenario")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        self.assertEqual(report["world_state_commitment"], "not_authorized")
        return result

    def nodes(self, package):
        return {node["id"]: node for node in package["nodes"]}

    def stripping(self, package):
        return [item for item in package["candidates"]
                if item["provenance"][0]["method"] == "stripping_not_nominal"]

    def roles(self, package):
        nodes = self.nodes(package)
        found = []
        for item in self.stripping(package):
            if item["type"] != "PARTICIPANT":
                continue
            prop = nodes[item["arguments"]["proposition"]]
            found.append((prop.get("predicate"), item["value"],
                          nodes[item["arguments"]["mention"]]["label"], item))
        return found

    def assert_rest_matches_z4(self, text, package):
        left = z4.export_candidate_graph(text, package_id="pkg_scenario")
        kept = [item for item in package["candidates"]
                if item["provenance"][0]["method"] != "stripping_not_nominal"]
        self.assertEqual(kept, left["candidates"])
        self.assertEqual(package["nodes"], left["nodes"])
        for evidence in package["evidence"]:
            self.assertEqual(text[evidence["start"]:evidence["end"]], evidence["text"])

    def test_rescue_keeps_the_child_and_adds_the_dog(self):
        package = self.package(RESCUE)
        self.assert_rest_matches_z4(RESCUE, package)
        roles = self.roles(package)
        self.assertEqual([(pred, role, label) for pred, role, label, _ in roles],
                         [("save", "subject", "Maria"), ("save", "object", "the dog")])
        self.assertTrue(all(item["scope"]["polarity"] == "negative" and item["assessment"]["status"] == "unresolved"
                            for *_, item in roles))
        nodes = self.nodes(package)
        spoken = [nodes[item["arguments"]["mention"]]["label"]
                  for item in package["candidates"]
                  if item["type"] == "PARTICIPANT" and item["value"] == "object"
                  and item["scope"]["polarity"] == "positive"
                  and nodes[item["arguments"]["proposition"]].get("predicate") == "save"]
        self.assertIn("the child", spoken)
        self.assertNotIn("the dog", spoken)

    def test_medicine_keeps_both_gifts_and_marks_nora(self):
        package = self.package(MEDICINE)
        self.assert_rest_matches_z4(MEDICINE, package)
        self.assertEqual([(pred, role, label) for pred, role, label, _ in self.roles(package)],
                         [("give", "subject", "Lila"), ("give", "destination", "Nora")])
        groups = [item for item in package["choice_sets"] if item["kind"] == "scenario_option"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["selection_rule"], "any_subset")
        self.assertFalse(groups[0]["exhaustive"])
        nodes = self.nodes(package)
        gifts = [item for item in package["candidates"]
                 if item["type"] == "PARTICIPANT" and item["value"] == "destination"
                 and item["provenance"][0]["method"] != "stripping_not_nominal"
                 and nodes[item["arguments"]["mention"]]["label"] == "Nora"]
        self.assertTrue(gifts)
        self.assertTrue(all(item["provenance"][0]["method"] != "stripping_not_nominal" for item in gifts))

    def test_trolley_keeps_the_lever_choice_and_adds_the_brake(self):
        package = self.package(TROLLEY)
        self.assert_rest_matches_z4(TROLLEY, package)
        self.assertEqual([(pred, role, label) for pred, role, label, _ in self.roles(package)],
                         [("pull", "subject", "Maria"), ("pull", "object", "the brake")])
        groups = [item for item in package["choice_sets"] if item["kind"] == "scenario_option"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["selection_rule"], "any_subset")
        self.assertFalse(groups[0]["exhaustive"])
        nodes = self.nodes(package)
        places = [nodes[item["arguments"]["mention"]]["label"]
                  for item in package["candidates"]
                  if item["type"] == "PARTICIPANT" and item["value"] == "location"]
        self.assertIn("the track", places)
        self.assertIn("the side track", places)

    def test_a_verb_phrase_in_the_same_scenario_is_not_rewritten(self):
        text = "Maria can save the child, but not the dog. Anna does the same."
        package = self.package(text)
        self.assert_rest_matches_z4(text, package)
        self.assertEqual([(pred, role, label) for pred, role, label, _ in self.roles(package)],
                         [("save", "subject", "Maria"), ("save", "object", "the dog")])

    def test_coordination_stays_untouched_inside_a_scenario(self):
        text = (
            "A child and a dog are in the water. Maria can save the child or the dog. "
            "If Maria saves the child, the child will live."
        )
        package = self.package(text)
        self.assertEqual(self.stripping(package), [])
        self.assert_rest_matches_z4(text, package)

    def test_a_remnant_before_a_later_clause_is_still_a_copy(self):
        text = "Five workers are on the track. If Maria pulls the lever but not the brake, one worker will die."
        package = self.package(text)
        self.assert_rest_matches_z4(text, package)
        self.assertEqual([(pred, role, label) for pred, role, label, _ in self.roles(package)],
                         [("pull", "subject", "Maria"), ("pull", "object", "the brake")])
        nodes = self.nodes(package)
        deaths = [item for item in package["candidates"]
                  if item["type"] == "PREDICATION"
                  and nodes[item["arguments"]["proposition"]].get("predicate") == "die"]
        self.assertTrue(deaths)
        self.assertTrue(all(item["provenance"][0]["method"] != "stripping_not_nominal" for item in deaths))
        self.assertTrue(all(item["scope"]["polarity"] == "positive" for item in deaths))

    def test_two_stripping_sentences_are_both_kept(self):
        text = ("Maria can save the child, but not the dog. "
                "Sam can move the lever, but not the brake.")
        package = self.package(text)
        self.assert_rest_matches_z4(text, package)
        found = [(pred, role, label) for pred, role, label, _ in self.roles(package)]
        self.assertEqual(found, [
            ("save", "subject", "Maria"), ("save", "object", "the dog"),
            ("move", "subject", "Sam"), ("move", "object", "the brake"),
        ])
        copies = [item for item in self.stripping(package) if item["type"] == "PARTICIPANT" and item["value"] == "object"]
        self.assertEqual(copies[0]["exclusive_with"], [])
        self.assertEqual(copies[1]["exclusive_with"], [])


if __name__ == "__main__":
    unittest.main()
