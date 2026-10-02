import unittest

import parsing_game_Z3 as z
from candidate_validation import empty_selection, validate_candidate_selection


class CoordinationTests(unittest.TestCase):
    def package(self, text):
        result = z.export_candidate_graph(text, package_id="pkg_z3_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def roles(self, package, value, predicate=None):
        nodes = {node["id"]: node for node in package["nodes"]}
        found = []
        for item in package["candidates"]:
            if item["type"] != "PARTICIPANT" or item["value"] != value:
                continue
            prop = nodes[item["arguments"]["proposition"]]
            if predicate is not None and prop.get("predicate") != predicate:
                continue
            found.append((nodes[item["arguments"]["mention"]]["label"], item["provenance"][0]["method"], item))
        return found

    def test_and_shares_the_subject_and_the_destination(self):
        lived = self.package("Omar and Nora will live.")
        subjects = self.roles(lived, "subject", "live")
        self.assertEqual([label for label, _, _ in subjects], ["Omar", "Nora"])
        self.assertEqual(subjects[1][1], "coordinated_nominal_shares_head_role")
        self.assertEqual({node["label"] for node in lived["nodes"] if node["kind"] == "mention"}, {"Omar", "Nora"})
        given = self.package("Lila gives the medicine to Omar and Nora.")
        destinations = self.roles(given, "destination", "give")
        self.assertEqual([label for label, _, _ in destinations], ["Omar", "Nora"])
        self.assertEqual([label for label, _, _ in self.roles(given, "object", "give")], ["the medicine"])
        self.assertEqual(destinations[1][1], "coordinated_nominal_shares_head_role")

    def test_or_and_a_middle_conjunct_are_included(self):
        package = self.package("Omar or Nora will live.")
        self.assertEqual([label for label, _, _ in self.roles(package, "subject", "live")], ["Omar", "Nora"])
        seen = self.package("She saw Omar, Nora, and Sam.")
        self.assertEqual([label for label, _, _ in self.roles(seen, "object", "see")], ["Omar", "Nora", "Sam"])

    def test_a_comma_list_and_the_false_conjunct_stay_unshared(self):
        bought = self.package("I bought apples, oranges.")
        self.assertEqual([label for label, _, _ in self.roles(bought, "object", "buy")], ["apples"])
        left = self.package("If Lila helps Omar, Omar will leave.")
        subjects = self.roles(left, "subject", "leave")
        self.assertEqual([label for label, _, _ in subjects], ["Omar"])
        self.assertNotEqual(subjects[0][1], "coordinated_nominal_shares_head_role")

    def test_shared_roles_keep_the_proposition_polarity(self):
        package = self.package("Maria does not see Omar and Nora.")
        objects = self.roles(package, "object", "see")
        self.assertEqual([label for label, _, _ in objects], ["Omar", "Nora"])
        self.assertTrue(all(item["scope"]["polarity"] == "negative" for _, _, item in objects))

    def test_different_quantities_stay_unlinked(self):
        package = self.package("Five workers and one manager are on the track.")
        self.assertEqual([label for label, _, _ in self.roles(package, "subject", "be")],
                         ["Five workers", "one manager"])
        nodes = {node["id"]: node for node in package["nodes"]}
        linked = False
        for item in package["candidates"]:
            if item["type"] != "SAME_REFERENT":
                continue
            labels = {nodes[item["arguments"]["mention_a"]]["label"],
                      nodes[item["arguments"]["mention_b"]]["label"]}
            linked = linked or labels == {"Five workers", "one manager"}
        self.assertFalse(linked)
        places = [label for label, _, _ in self.roles(package, "location", "be")]
        self.assertIn("the track", places)

    def test_trolley_choice_and_worker_links_remain(self):
        text = (
            "Five workers are on the track. One worker is on the side track. "
            "If Maria pulls the lever, one worker will die. "
            "If Maria does not pull the lever, five workers will die."
        )
        package = self.package(text)
        groups = [group for group in package["choice_sets"] if group["kind"] == "scenario_option"]
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0]["exhaustive"])
        nodes = {node["id"]: node for node in package["nodes"]}
        pairs = []
        for item in package["candidates"]:
            if item["provenance"][0]["method"] != "quantity_matched_indefinite_description":
                continue
            self.assertEqual(item["assessment"]["status"], "unresolved")
            pairs.append({nodes[item["arguments"]["mention_a"]]["label"],
                          nodes[item["arguments"]["mention_b"]]["label"]})
        self.assertIn({"Five workers", "five workers"}, pairs)
        self.assertIn({"One worker", "one worker"}, pairs)
