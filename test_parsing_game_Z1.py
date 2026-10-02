import unittest

import parsing_game_Z1 as z
from candidate_validation import empty_selection, validate_candidate_selection

MEDICINE = (
    "Lila has one dose of medicine. Omar needs the medicine. Nora needs the medicine. "
    "If Lila gives the medicine to Omar, Omar will live. "
    "If Lila gives the medicine to Nora, Nora will live."
)
TROLLEY = (
    "Five workers are on the track. One worker is on the side track. Maria can pull the lever. "
    "If Maria pulls the lever, one worker will die. "
    "If Maria does not pull the lever, five workers will die."
)


class ConditionalChoiceTests(unittest.TestCase):
    def package(self, text):
        result = z.export_candidate_graph(text, package_id="pkg_z1_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def scenario_sets(self, package):
        return [item for item in package["choice_sets"] if item["kind"] == "scenario_option"]

    def options(self, package, group):
        nodes = {node["id"]: node for node in package["nodes"]}
        found = []
        for ident in group["candidate_ids"]:
            item = next(candidate for candidate in package["candidates"] if candidate["id"] == ident)
            prop = nodes[item["arguments"]["proposition"]]
            found.append((prop.get("label"), item["scope"]["polarity"], item))
        return found

    def test_two_gifts_are_one_nonexclusive_choice(self):
        package = self.package(MEDICINE)
        groups = self.scenario_sets(package)
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group["selection_rule"], "any_subset")
        self.assertFalse(group["exhaustive"])
        self.assertEqual(len(group["candidate_ids"]), 2)
        labels = []
        for _, polarity, item in self.options(package, group):
            self.assertEqual(polarity, "positive")
            self.assertEqual(item["provenance"][0]["method"], "attested_conditional_alternatives")
            self.assertEqual(item["exclusive_with"], [])
            required = [candidate for candidate in package["candidates"] if candidate["id"] in item["requires"]]
            self.assertTrue(any(candidate["type"] == "PREDICATION" for candidate in required))
            self.assertTrue(any(candidate["type"] == "CONDITIONAL_ON" for candidate in required))
            labels.append(item["arguments"]["proposition"])
        self.assertEqual(len(set(labels)), 2)
        point = next(node for node in package["nodes"] if node["id"] == self.options(package, group)[0][2]["arguments"]["choice_point"])
        self.assertEqual(point["kind"], "choice_point")
        self.assertNotIn("not act", point["label"].casefold())

    def test_one_if_does_not_invent_a_second_branch(self):
        package = self.package("If Maria pulls the lever, the trolley will stop.")
        self.assertEqual(self.scenario_sets(package), [])

    def test_trolley_keeps_the_negative_pull_as_its_own_option(self):
        package = self.package(TROLLEY)
        groups = self.scenario_sets(package)
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0]["exhaustive"])
        self.assertEqual(groups[0]["selection_rule"], "any_subset")
        found = sorted((label, polarity) for label, polarity, _ in self.options(package, groups[0]))
        self.assertEqual(found, [("pull", "negative"), ("pulls", "positive")])
        nodes = {node["id"]: node for node in package["nodes"]}
        places = [nodes[item["arguments"]["mention"]]["label"]
                  for item in package["candidates"] if item["type"] == "PARTICIPANT" and item["value"] == "location"]
        self.assertIn("the track", places)
        self.assertIn("the side track", places)

    def test_different_subjects_are_not_one_choice(self):
        package = self.package(
            "If Maria pulls the lever, one worker will die. If Anna waits, the train will stop.")
        self.assertEqual(self.scenario_sets(package), [])

    def test_plural_subjects_are_not_treated_as_one_name(self):
        package = self.package(
            "If the worker leaves, the dog will die. If the workers return, the child will live.")
        self.assertEqual(self.scenario_sets(package), [])

    def test_decide_whether_stays_a_single_textual_option(self):
        package = self.package("Maria must decide whether to act.")
        groups = self.scenario_sets(package)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["candidate_ids"]), 1)
        self.assertFalse(groups[0]["exhaustive"])
        option = next(item for item in package["candidates"] if item["id"] == groups[0]["candidate_ids"][0])
        self.assertEqual(option["provenance"][0]["method"], "bounded_dependency_rules")

    def test_can_sentences_are_not_added_as_options(self):
        package = self.package(
            "Sam can save the child. Sam can save the dog. "
            "If Sam saves the child, the dog will die. If Sam saves the dog, the child will die.")
        groups = self.scenario_sets(package)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["candidate_ids"]), 2)
        conditions = {item["arguments"]["condition"] for item in package["candidates"] if item["type"] == "CONDITIONAL_ON"}
        for _, _, option in self.options(package, groups[0]):
            self.assertIn(option["arguments"]["proposition"], conditions)
            self.assertEqual(option["scope"]["polarity"], "positive")
