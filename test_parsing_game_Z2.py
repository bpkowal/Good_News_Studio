import unittest

import parsing_game_Z2 as z
from candidate_validation import empty_selection, validate_candidate_selection

TROLLEY = (
    "Five workers are on the track. One worker is on the side track. Maria can pull the lever. "
    "If Maria pulls the lever, one worker will die. "
    "If Maria does not pull the lever, five workers will die."
)


class QuantityIdentityTests(unittest.TestCase):
    def package(self, text):
        result = z.export_candidate_graph(text, package_id="pkg_z2_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def links(self, package):
        nodes = {node["id"]: node for node in package["nodes"]}
        blocked = {ident for question in package["open_questions"] for ident in question["blocking_for"]}
        found = []
        for item in package["candidates"]:
            if item["type"] != "SAME_REFERENT":
                continue
            labels = {nodes[item["arguments"]["mention_a"]]["label"],
                      nodes[item["arguments"]["mention_b"]]["label"]}
            found.append((tuple(sorted(labels)), item["assessment"]["status"],
                          item["provenance"][0]["method"], item))
        return found, nodes, blocked

    def test_repeated_counts_stay_unresolved_candidates(self):
        package = self.package(TROLLEY)
        found, nodes, blocked = self.links(package)
        matched = [item for labels, status, method, item in found
                   if method == "quantity_matched_indefinite_description"]
        pairs = []
        for item in matched:
            self.assertEqual(item["assessment"]["status"], "unresolved")
            self.assertEqual(item["exclusive_with"], [])
            pairs.append({nodes[item["arguments"]["mention_a"]]["label"],
                          nodes[item["arguments"]["mention_b"]]["label"]})
            self.assertNotEqual(item["arguments"]["mention_a"], item["arguments"]["mention_b"])
        self.assertIn({"Five workers", "five workers"}, pairs)
        self.assertIn({"One worker", "one worker"}, pairs)
        self.assertFalse(any(pair == {"Five workers", "one worker"} or pair == {"five workers", "One worker"}
                             or pair == {"Five workers", "One worker"} or pair == {"five workers", "one worker"}
                             for pair in pairs))
        for item in matched:
            for mention_id in item["arguments"].values():
                roles = [role["id"] for role in package["candidates"]
                         if role["type"] == "PARTICIPANT" and role["arguments"].get("mention") == mention_id]
                self.assertTrue(roles)
                self.assertTrue(all(role_id not in blocked for role_id in roles))
        groups = [group for group in package["choice_sets"] if group["kind"] == "scenario_option"]
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0]["exhaustive"])

    def test_different_numbers_and_bare_indefinites_do_not_link(self):
        split = self.package("Five workers rest. One worker leaves.")
        found, _, _ = self.links(split)
        self.assertFalse(any(method == "quantity_matched_indefinite_description" for _, _, method, _ in found))
        dogs = self.package("A dog barked. A dog slept.")
        found, _, _ = self.links(dogs)
        self.assertFalse(any(method == "quantity_matched_indefinite_description" for _, _, method, _ in found))

    def test_word_and_digit_forms_of_the_same_amount_link(self):
        package = self.package("Five workers stand. 5 workers leave.")
        found, nodes, _ = self.links(package)
        self.assertTrue(any(
            {nodes[item["arguments"]["mention_a"]]["label"],
             nodes[item["arguments"]["mention_b"]]["label"]} == {"Five workers", "5 workers"}
            and status == "unresolved" and method == "quantity_matched_indefinite_description"
            for _, status, method, item in found))

    def test_a_definite_number_stays_on_the_reference_pass(self):
        package = self.package("The five workers waited. The five workers left.")
        found, _, _ = self.links(package)
        self.assertFalse(any(method == "quantity_matched_indefinite_description" for _, _, method, _ in found))
        self.assertTrue(any(method == "bounded_number_animacy_recency_role_ranking" for _, _, method, _ in found))

    def test_one_if_still_does_not_invent_a_choice(self):
        package = self.package("If Maria pulls the lever, the trolley will stop.")
        self.assertFalse(any(group["kind"] == "scenario_option" for group in package["choice_sets"]))
