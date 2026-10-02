import unittest

import parsing_game_X as x
from candidate_validation import empty_selection, validate_candidate_selection


class SameNameBlockTests(unittest.TestCase):
    def package(self, text):
        result = x.export_candidate_graph(text, package_id="pkg_x_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def blocked_roles(self, package):
        nodes = {node["id"]: node for node in package["nodes"]}
        candidates = {item["id"]: item for item in package["candidates"]}
        found = []
        for question in package["open_questions"]:
            if question["kind"] != "reference":
                continue
            for ident in question["blocking_for"]:
                item = candidates[ident]
                mention = nodes[item["arguments"]["mention"]]["label"]
                prop = nodes[item["arguments"]["proposition"]]
                found.append((mention, item["value"], prop.get("predicate")))
        return found

    def test_article_and_plural_count_as_one_name(self):
        self.assertTrue(x.same_mention_name("the medicine", "medicine"))
        self.assertTrue(x.same_mention_name("worker", "workers"))
        self.assertTrue(x.same_mention_name("the worker", "workers"))
        self.assertFalse(x.same_mention_name("the trolley", "a runaway trolley"))
        self.assertFalse(x.same_mention_name("She", "Maria"))

    def test_same_name_role_is_not_blocked(self):
        text = ("Lila has one dose of medicine. Omar needs the medicine. "
                "If Lila gives the medicine to Omar, Omar will live.")
        package = self.package(text)
        blocked = self.blocked_roles(package)
        self.assertNotIn(("Omar", "subject", "live"), blocked)
        self.assertNotIn(("the medicine", "object", "need"), blocked)
        self.assertNotIn(("Lila", "subject", "give"), blocked)
        identity = [item for item in package["candidates"] if item["type"] == "SAME_REFERENT"]
        self.assertTrue(identity)
        self.assertTrue(all(item["assessment"]["status"] in {"unresolved", "preferred"} for item in identity))

    def test_different_names_and_empty_questions_stay_blocked(self):
        people = self.package("Maria saw Anna. She left.")
        self.assertIn(("She", "subject", "leave"), self.blocked_roles(people))
        lever = self.package("If Maria pulls the lever, the trolley will stop.")
        blocked = self.blocked_roles(lever)
        self.assertIn(("the lever", "object", "pull"), blocked)
        self.assertIn(("the trolley", "subject", "stop"), blocked)

    def test_modified_noun_stays_blocked(self):
        package = self.package("A runaway trolley approached. The trolley stopped.")
        self.assertTrue(any(mention == "The trolley" and predicate == "stop"
                            for mention, _, predicate in self.blocked_roles(package)))
