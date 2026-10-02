import unittest

import parsing_game_Y as y
from candidate_validation import empty_selection, validate_candidate_selection


class RecipientTests(unittest.TestCase):
    def package(self, text):
        result = y.export_candidate_graph(text, package_id="pkg_y_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def roles(self, package, predicate, value):
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

    def destinations(self, text, predicate):
        return self.roles(self.package(text), predicate, "destination")

    def test_dative_to_is_the_gift_recipient(self):
        package = self.package("If Lila gives the medicine to Omar, Omar will live.")
        recipients = self.roles(package, "give", "destination")
        self.assertEqual([label for label, _ in recipients], ["Omar"])
        _, item = recipients[0]
        self.assertEqual(item["provenance"][0]["method"], "dative_or_person_transfer_to")
        self.assertTrue(any(ctx["kind"] == "hypothetical" for ctx in item["scope"]["contexts"]))
        self.assertEqual([label for label, _ in self.roles(package, "give", "object")], ["the medicine"])
        self.assertIn("Omar", [label for label, _ in self.roles(package, "live", "subject")])

    def test_bare_dative_and_dative_to_without_an_object(self):
        self.assertEqual([label for label, _ in self.destinations("Lila gives Omar the medicine.", "give")], ["Omar"])
        self.assertEqual([label for label, _ in self.destinations("Lila gives to Omar.", "give")], ["Omar"])
        self.assertEqual([item["provenance"][0]["method"] for _, item in self.destinations("Maria sent the letter to Anna.", "send")],
                         ["dative_or_person_transfer_to"])

    def test_prep_to_keeps_the_single_existing_destination(self):
        for text, predicate, label in (
            ("She gave the book to the child.", "give", "the child"),
            ("She handed the dose to Nora.", "hand", "Nora"),
            ("They walked to the shelter.", "walk", "the shelter"),
            ("Lila talks to Omar.", "talk", "Omar"),
            ("She moved the chair to the wall.", "move", "the wall"),
            ("She compared the plan to the budget.", "compare", "the budget"),
            ("Rain leads to flooding.", "lead", "flooding"),
            ("He went to work.", "go", "work"),
        ):
            found = self.destinations(text, predicate)
            self.assertEqual([name for name, _ in found], [label], text)
            self.assertEqual([item["provenance"][0]["method"] for _, item in found], ["bounded_dependency_rules"], text)

    def test_infinitive_and_unrelated_verbs_gain_no_destination(self):
        self.assertEqual(self.destinations("The worker tried to leave.", "try"), [])
        self.assertEqual(self.destinations("The worker tried to leave.", "leave"), [])
        self.assertEqual(self.destinations("If Maria pulls the lever, the trolley will stop.", "pull"), [])

    def test_second_coordinated_recipient_is_not_invented(self):
        found = self.destinations("Lila gives the medicine to Omar and Nora.", "give")
        self.assertEqual([label for label, _ in found], ["Omar"])
        self.assertEqual(found[0][1]["provenance"][0]["method"], "dative_or_person_transfer_to")

    def test_same_name_recipient_is_not_blocked(self):
        text = "Omar needs the medicine. If Lila gives the medicine to Omar, Omar will live."
        package = self.package(text)
        nodes = {node["id"]: node for node in package["nodes"]}
        candidates = {item["id"]: item for item in package["candidates"]}
        blocked = set()
        for question in package["open_questions"]:
            blocked.update(question["blocking_for"])
        recipient = next(item for item in package["candidates"]
                         if item["type"] == "PARTICIPANT" and item["value"] == "destination")
        self.assertEqual(nodes[recipient["arguments"]["mention"]]["label"], "Omar")
        self.assertNotIn(recipient["id"], blocked)
        self.assertIn(recipient["id"], candidates)
