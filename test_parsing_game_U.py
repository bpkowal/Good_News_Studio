import unittest

import parsing_game_U as u
from candidate_validation import empty_selection, validate_candidate_selection


SENTENCE = "If Maria decides to pull the lever, the trolley will stop."


class ConditionComplementTests(unittest.TestCase):
    def package(self, text=SENTENCE):
        result = u.export_candidate_graph(text, package_id="pkg_u_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def test_pull_inherits_hypothetical_scope(self):
        package = self.package()
        nodes = {node["id"]: node for node in package["nodes"]}
        pull = next(node["id"] for node in package["nodes"] if node.get("predicate") == "pull")
        predication = next(item for item in package["candidates"]
                           if item["type"] == "PREDICATION" and item["arguments"]["proposition"] == pull)
        lever = next(item for item in package["candidates"]
                     if item["type"] == "PARTICIPANT" and item["value"] == "object"
                     and item["arguments"]["proposition"] == pull)
        self.assertEqual([ctx["kind"] for ctx in predication["scope"]["contexts"]], ["hypothetical"])
        self.assertEqual([ctx["kind"] for ctx in lever["scope"]["contexts"]], ["hypothetical"])
        self.assertEqual(nodes[pull]["predicate"], "pull")

    def test_conditional_requires_the_pull_complement(self):
        package = self.package()
        nodes = {node["id"]: node for node in package["nodes"]}
        candidates = {item["id"]: item for item in package["candidates"]}
        link = next(item for item in package["candidates"] if item["type"] == "CONDITIONAL_ON")
        self.assertEqual(nodes[link["arguments"]["condition"]]["predicate"], "decide")
        self.assertEqual(nodes[link["arguments"]["consequence"]]["predicate"], "stop")
        required = [candidates[ident] for ident in link["requires"] if candidates[ident]["type"] == "EVENT_LINK"]
        self.assertEqual(len(required), 1)
        self.assertEqual(required[0]["value"], "complement")
        self.assertEqual(nodes[required[0]["arguments"]["child"]]["predicate"], "pull")
        self.assertEqual(required[0]["assessment"]["status"], "proposed")
        self.assertFalse(any(question["kind"] == "attachment" and required[0]["id"] in question["candidate_ids"]
                             for question in package["open_questions"]))

    def test_maria_is_a_controller_candidate_of_pull(self):
        package = self.package()
        nodes = {node["id"]: node for node in package["nodes"]}
        pull = next(node["id"] for node in package["nodes"] if node.get("predicate") == "pull")
        controller = next(item for item in package["candidates"]
                          if item["type"] == "PARTICIPANT" and item["value"] == "controller"
                          and item["arguments"]["proposition"] == pull)
        self.assertEqual(nodes[controller["arguments"]["mention"]]["label"], "Maria")
        self.assertEqual(controller["assessment"]["status"], "unresolved")
        self.assertIn("hypothetical", [ctx["kind"] for ctx in controller["scope"]["contexts"]])
        question = next(item for item in package["open_questions"] if controller["id"] in item["candidate_ids"])
        self.assertEqual(question["kind"], "reference")
        self.assertNotIn(controller["id"], question["blocking_for"])

    def test_plain_if_clause_is_unchanged(self):
        package = self.package("If Maria pulls the lever, the trolley will stop.")
        nodes = {node["id"]: node for node in package["nodes"]}
        link = next(item for item in package["candidates"] if item["type"] == "CONDITIONAL_ON")
        self.assertEqual(nodes[link["arguments"]["condition"]]["predicate"], "pull")
        self.assertFalse(any(item["type"] == "EVENT_LINK" and item["id"] in link["requires"]
                             for item in package["candidates"]))
        self.assertFalse(any(item["value"] == "controller" for item in package["candidates"]))

    def test_object_control_does_not_offer_the_matrix_subject(self):
        package = self.package("The storm forced the school to close.")
        nodes = {node["id"]: node for node in package["nodes"]}
        controllers = [item for item in package["candidates"] if item["value"] == "controller"]
        labels = {nodes[item["arguments"]["mention"]]["label"] for item in controllers}
        self.assertNotIn("The storm", labels)
        self.assertTrue(any("school" in label for label in labels))


if __name__ == "__main__":
    unittest.main()
