import unittest

import parsing_game_Z as z
from candidate_validation import empty_selection, validate_candidate_selection


class DefiniteSetupTests(unittest.TestCase):
    def package(self, text):
        result = z.export_candidate_graph(text, package_id="pkg_z_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def roles(self, package, value):
        nodes = {node["id"]: node for node in package["nodes"]}
        blocked = set()
        for question in package["open_questions"]:
            blocked.update(question["blocking_for"])
        found = []
        for item in package["candidates"]:
            if item["type"] != "PARTICIPANT" or item["value"] != value:
                continue
            prop = nodes[item["arguments"]["proposition"]]
            found.append((nodes[item["arguments"]["mention"]]["label"], prop.get("predicate"),
                          item["id"] in blocked, item["provenance"][0]["method"]))
        return found

    def test_on_the_track_is_a_location(self):
        package = self.package("Five workers are on the track. One worker is on the side track.")
        places = [(label, predicate) for label, predicate, blocked, method in self.roles(package, "location")]
        self.assertIn(("the track", "be"), places)
        self.assertIn(("the side track", "be"), places)
        self.assertTrue(all(not blocked and method == "definite_locative_on_in_at"
                            for _, _, blocked, method in self.roles(package, "location")))
        links = [item for item in package["candidates"] if item["type"] == "SAME_REFERENT"]
        nodes = {node["id"]: node for node in package["nodes"]}
        self.assertTrue(any(nodes[item["arguments"]["mention_a"]]["label"] == "the side track"
                            and nodes[item["arguments"]["mention_b"]]["label"] == "the track"
                            and item["assessment"]["status"] == "unresolved" for item in links))

    def test_in_the_house_is_a_location_and_monday_is_not(self):
        self.assertEqual([label for label, _, _, _ in self.roles(self.package("They are in the house."), "location")],
                         ["the house"])
        self.assertEqual(self.roles(self.package("They meet on Monday."), "location"), [])
        walked = self.package("They walked to the shelter.")
        self.assertEqual(self.roles(walked, "location"), [])
        self.assertEqual([label for label, _, _, _ in self.roles(walked, "destination")], ["the shelter"])

    def test_first_definite_role_stays_and_the_question_remains(self):
        package = self.package("Maria can pull the lever.")
        lever = [item for item in self.roles(package, "object") if item[0] == "the lever"]
        self.assertEqual(len(lever), 1)
        self.assertFalse(lever[0][2])
        self.assertTrue(any(question["kind"] == "reference" and not question["candidate_ids"]
                            for question in package["open_questions"]))
        nodes = {node["id"]: node for node in package["nodes"]}
        blocked = {ident for question in package["open_questions"] for ident in question["blocking_for"]}
        pull = next(item for item in package["candidates"]
                    if item["type"] == "PREDICATION" and nodes[item["arguments"]["proposition"]]["predicate"] == "pull")
        self.assertIn(pull["id"], blocked)

    def test_pronoun_without_an_antecedent_stays_blocked(self):
        package = self.package("She left.")
        subjects = self.roles(package, "subject")
        self.assertEqual([label for label, _, _, _ in subjects], ["She"])
        self.assertTrue(subjects[0][2])

    def test_modified_definite_keeps_the_role_and_the_identity_question(self):
        package = self.package("A runaway trolley approached. The trolley stopped.")
        stopped = [item for item in self.roles(package, "subject") if item[0] == "The trolley" and item[1] == "stop"]
        self.assertEqual(len(stopped), 1)
        self.assertFalse(stopped[0][2])
        nodes = {node["id"]: node for node in package["nodes"]}
        self.assertTrue(any(item["type"] == "SAME_REFERENT"
                            and {nodes[item["arguments"]["mention_a"]]["label"],
                                 nodes[item["arguments"]["mention_b"]]["label"]} == {"The trolley", "A runaway trolley"}
                            for item in package["candidates"]))
        labels = [node["label"] for node in package["nodes"] if node["kind"] == "mention"]
        self.assertIn("The trolley", labels)
        self.assertIn("A runaway trolley", labels)

    def test_the_heart_stays_a_need_and_an_identity_candidate(self):
        package = self.package("The doctor has one heart. Omar needs the heart.")
        self.assertTrue(any(label == "The doctor" and predicate == "have" and not blocked
                            for label, predicate, blocked, _ in self.roles(package, "subject")))
        self.assertTrue(any(label == "the heart" and predicate == "need" and not blocked
                            for label, predicate, blocked, _ in self.roles(package, "object")))
        nodes = {node["id"]: node for node in package["nodes"]}
        self.assertTrue(any(item["type"] == "SAME_REFERENT"
                            and nodes[item["arguments"]["mention_a"]]["label"] == "the heart"
                            and nodes[item["arguments"]["mention_b"]]["label"] == "one heart"
                            for item in package["candidates"]))
