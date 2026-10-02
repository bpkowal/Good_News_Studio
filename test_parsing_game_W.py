import unittest

import parsing_game_W as w
from candidate_validation import empty_selection, validate_candidate_selection


class ClauseSubjectRepairTests(unittest.TestCase):
    def package(self, text):
        result = w.export_candidate_graph(text, package_id="pkg_w_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def subjects_of(self, package, predicate):
        nodes = {node["id"]: node for node in package["nodes"]}
        found = []
        for item in package["candidates"]:
            if item["type"] != "PARTICIPANT" or item["value"] != "subject":
                continue
            prop = nodes[item["arguments"]["proposition"]]
            if prop.get("predicate") == predicate:
                found.append((nodes[item["arguments"]["mention"]]["label"], item["provenance"][0]["method"]))
        return found

    def test_repeated_name_after_comma_is_the_clause_subject(self):
        package = self.package("If Lila gives the medicine to Omar, Omar will live.")
        self.assertIn(("Omar", "comma_clause_subject_not_conjunct"), self.subjects_of(package, "live"))

    def test_already_correct_subject_is_not_repaired(self):
        package = self.package("If Lila gives the medicine to Nora, Nora will live.")
        subjects = self.subjects_of(package, "live")
        self.assertIn(("Nora", "bounded_dependency_rules"), subjects)
        self.assertFalse(any(method == "comma_clause_subject_not_conjunct" for _, method in subjects))

    def test_coordinator_keeps_a_real_conjunct(self):
        package = self.package("Omar and Nora will live.")
        subjects = self.subjects_of(package, "live")
        self.assertEqual(subjects, [("Omar", "bounded_dependency_rules")])

    def test_comma_list_without_a_following_verb_is_not_a_subject(self):
        package = self.package("I bought apples, oranges.")
        labels = [node["label"] for node in package["nodes"] if node["kind"] == "mention"]
        self.assertIn("oranges", labels)
        self.assertEqual(self.subjects_of(package, "buy"), [("I", "bounded_dependency_rules")])
        self.assertFalse(any(item["provenance"][0]["method"] == "comma_clause_subject_not_conjunct"
                             for item in package["candidates"]))
