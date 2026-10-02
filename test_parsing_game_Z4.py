import unittest

import parsing_game_Z4 as z
from candidate_validation import empty_selection, validate_candidate_selection

TROLLEY = (
    "Five workers are on the track. One worker is on the side track. "
    "If Maria pulls the lever, one worker will die. "
    "If Maria does not pull the lever, five workers will die."
)


class ClauseBoundaryTests(unittest.TestCase):
    def package(self, text):
        result = z.export_candidate_graph(text, package_id="pkg_z4_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def roles(self, package, value, predicate):
        nodes = {node["id"]: node for node in package["nodes"]}
        found = []
        for item in package["candidates"]:
            if item["type"] != "PARTICIPANT" or item["value"] != value:
                continue
            prop = nodes[item["arguments"]["proposition"]]
            if prop.get("predicate") != predicate:
                continue
            found.append((nodes[item["arguments"]["mention"]]["label"], item))
        return found

    def test_a_new_clause_does_not_inherit_the_recipient_role(self):
        package = self.package(
            "If Lila gives the medicine to Omar and Nora, Omar and Nora will live.")
        destinations = [label for label, _ in self.roles(package, "destination", "give")]
        subjects = self.roles(package, "subject", "live")
        self.assertEqual(destinations, ["Omar", "Nora"])
        self.assertEqual([label for label, _ in subjects], ["Omar", "Nora"])
        self.assertTrue(all(item["provenance"][0]["method"] == "clause_boundary_subject" for _, item in subjects))
        self.assertTrue(all(any(ctx["kind"] == "conditional" for ctx in item["scope"]["contexts"])
                            for _, item in subjects))

    def test_one_recipient_then_a_new_subject_stays_split(self):
        package = self.package("If Lila gives the medicine to Omar, Omar will live.")
        self.assertEqual([label for label, _ in self.roles(package, "destination", "give")], ["Omar"])
        subjects = self.roles(package, "subject", "live")
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0][0], "Omar")
        self.assertEqual(subjects[0][1]["provenance"][0]["method"], "comma_clause_subject_not_conjunct")

    def test_a_boundary_name_keeps_both_attachments(self):
        package = self.package("If Lila helps Omar, Omar will leave.")
        nodes = {node["id"]: node for node in package["nodes"]}
        objects = self.roles(package, "object", "help")
        subjects = self.roles(package, "subject", "leave")
        self.assertEqual([label for label, _ in objects], ["Omar"])
        self.assertEqual([label for label, _ in subjects], ["Omar", "Omar"])
        obj = objects[0][1]
        first, second = (item for _, item in subjects)
        self.assertEqual(obj["assessment"]["status"], "unresolved")
        self.assertEqual(first["assessment"]["status"], "unresolved")
        self.assertEqual(second["assessment"]["status"], "unresolved")
        self.assertIn(obj["id"], first["exclusive_with"])
        self.assertIn(first["id"], obj["exclusive_with"])
        self.assertIn(second["id"], obj["requires"])
        self.assertIn(first["id"], second["exclusive_with"])
        group = next(item for item in package["choice_sets"]
                     if obj["id"] in item["candidate_ids"] and first["id"] in item["candidate_ids"])
        self.assertEqual(group["kind"], "interpretation")
        self.assertEqual(group["selection_rule"], "at_most_one")
        self.assertFalse(group["exhaustive"])
        blocked = {ident for question in package["open_questions"] for ident in question["blocking_for"]}
        predicates = [item["id"] for item in package["candidates"] if item["type"] == "PREDICATION"
                      and nodes[item["arguments"]["proposition"]].get("predicate") in {"help", "leave"}]
        self.assertTrue(predicates)
        self.assertTrue(all(ident not in blocked for ident in predicates))

    def test_a_real_later_subject_stays_beside_the_boundary_choice(self):
        package = self.package("If Lila helps Omar, Nora will leave.")
        objects = self.roles(package, "object", "help")
        subjects = self.roles(package, "subject", "leave")
        self.assertEqual([label for label, _ in objects], ["Omar"])
        self.assertEqual(objects[0][1]["assessment"]["status"], "unresolved")
        by_label = {label: item for label, item in subjects}
        self.assertEqual(by_label["Omar"]["assessment"]["status"], "unresolved")
        self.assertEqual(by_label["Nora"]["assessment"]["status"], "proposed")
        self.assertIn(objects[0][1]["id"], by_label["Omar"]["exclusive_with"])
        self.assertNotIn(by_label["Nora"]["id"], objects[0][1]["exclusive_with"])

    def test_lists_and_pairs_still_share_one_role(self):
        lived = self.package("Omar and Nora will live.")
        self.assertEqual([label for label, _ in self.roles(lived, "subject", "live")], ["Omar", "Nora"])
        given = self.package("Lila gives the medicine to Omar and Nora.")
        self.assertEqual([label for label, _ in self.roles(given, "destination", "give")], ["Omar", "Nora"])
        seen = self.package("She saw Omar, Nora, and Sam.")
        self.assertEqual([label for label, _ in self.roles(seen, "object", "see")], ["Omar", "Nora", "Sam"])
        left = self.package("Omar, Nora, and Sam left.")
        self.assertEqual([label for label, _ in self.roles(left, "subject", "leave")], ["Omar", "Sam"])
        bought = self.package("I bought apples, oranges.")
        self.assertEqual([label for label, _ in self.roles(bought, "object", "buy")], ["apples"])
        hidden = self.package("Maria does not see Omar and Nora.")
        objects = self.roles(hidden, "object", "see")
        self.assertEqual([label for label, _ in objects], ["Omar", "Nora"])
        self.assertTrue(all(item["scope"]["polarity"] == "negative" for _, item in objects))

    def test_coordination_inside_one_clause_is_not_a_boundary_choice(self):
        package = self.package("If Lila helps Omar and Nora, they will live.")
        self.assertEqual([label for label, _ in self.roles(package, "object", "help")], ["Omar", "Nora"])
        self.assertEqual([label for label, _ in self.roles(package, "subject", "live")], ["they"])
        self.assertFalse(any(item["provenance"][0]["method"] == "clause_boundary_attachment"
                             for item in package["candidates"]))

    def test_trolley_choice_and_counts_remain(self):
        package = self.package(TROLLEY)
        groups = [group for group in package["choice_sets"] if group["kind"] == "scenario_option"]
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0]["exhaustive"])
        self.assertEqual(groups[0]["selection_rule"], "any_subset")
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
        places = [label for label, _ in self.roles(package, "location", "be")]
        self.assertIn("the track", places)
        self.assertIn("the side track", places)
