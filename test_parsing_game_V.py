import unittest

import parsing_game_V as v
from candidate_validation import empty_selection, validate_candidate_selection


class VocabularyExportTests(unittest.TestCase):
    def package(self, text):
        result = v.export_candidate_graph(text, package_id="pkg_v_test")
        report = validate_candidate_selection(result, empty_selection(result))
        self.assertTrue(report["contract_valid"], report["errors"])
        return result

    def test_try_is_an_attempt_without_child_occurrence(self):
        package = self.package("The worker tried to leave.")
        nodes = {node["id"]: node for node in package["nodes"]}
        link = next(item for item in package["candidates"] if item["type"] == "EVENT_LINK")
        self.assertEqual(link["value"], "attempt")
        child = nodes[link["arguments"]["child"]]
        self.assertEqual(child["predicate"], "leave")
        predication = next(item for item in package["candidates"]
                           if item["type"] == "PREDICATION" and item["arguments"]["proposition"] == child["id"])
        self.assertEqual(predication["scope"]["contexts"], [])
        self.assertNotIn("occurrence_status", link)
        self.assertTrue(any(resource["id"] == "verbnet_try-61.1" for resource in package["producer"]["resources"]))

    def test_intend_stays_a_complement(self):
        package = self.package("The worker intended to leave.")
        link = next(item for item in package["candidates"] if item["type"] == "EVENT_LINK")
        self.assertEqual(link["value"], "complement")
        self.assertFalse(any(resource["id"] == "verbnet_try-61.1" for resource in package["producer"]["resources"]))

    def test_decide_whether_is_one_nonexhaustive_option(self):
        package = self.package("Maria must decide whether to act.")
        nodes = {node["id"]: node for node in package["nodes"]}
        point = next(node for node in package["nodes"] if node["kind"] == "choice_point")
        option = next(item for item in package["candidates"] if item["type"] == "OPTION_OF")
        self.assertEqual(option["arguments"]["choice_point"], point["id"])
        self.assertEqual(nodes[option["arguments"]["proposition"]]["predicate"], "act")
        group = next(item for item in package["choice_sets"] if option["id"] in item["candidate_ids"])
        self.assertEqual(group["kind"], "scenario_option")
        self.assertEqual(group["selection_rule"], "any_subset")
        self.assertFalse(group["exhaustive"])
        self.assertEqual(group["candidate_ids"], [option["id"]])
        modalities = [item for item in package["candidates"] if item["type"] == "MODALITY"]
        self.assertTrue(modalities)
        self.assertTrue(all(nodes[item["arguments"]["proposition"]]["predicate"] == "decide" for item in modalities))
        self.assertNotIn(option["id"], {item["id"] for item in modalities})

    def test_asked_whether_is_not_a_choice_point(self):
        package = self.package("Maria asked whether to act.")
        self.assertFalse(any(node["kind"] == "choice_point" for node in package["nodes"]))

    def test_explicit_quantity(self):
        package = self.package("Five workers will die.")
        nodes = {node["id"]: node for node in package["nodes"]}
        quantity = next(item for item in package["candidates"] if item["type"] == "QUANTITY")
        self.assertEqual(quantity["value"]["operator"], "exact")
        self.assertEqual(quantity["value"]["amount"], 5)
        self.assertEqual(quantity["value"]["unit"], "worker")
        self.assertIn("worker", nodes[quantity["arguments"]["mention"]]["label"].lower())

    def test_all_is_a_question_not_a_quantity(self):
        package = self.package("All of them were late to work.")
        self.assertFalse(any(item["type"] == "QUANTITY" for item in package["candidates"]))
        self.assertTrue(any(question["kind"] == "missing_representation" and "quantif" in question["question"]
                            for question in package["open_questions"]))

    def test_u_condition_behavior_remains(self):
        package = self.package("If Maria decides to pull the lever, the trolley will stop.")
        nodes = {node["id"]: node for node in package["nodes"]}
        pull = next(node["id"] for node in package["nodes"] if node.get("predicate") == "pull")
        predication = next(item for item in package["candidates"]
                           if item["type"] == "PREDICATION" and item["arguments"]["proposition"] == pull)
        self.assertEqual([ctx["kind"] for ctx in predication["scope"]["contexts"]], ["hypothetical"])
        link = next(item for item in package["candidates"] if item["type"] == "CONDITIONAL_ON")
        candidates = {item["id"]: item for item in package["candidates"]}
        required = [candidates[ident] for ident in link["requires"] if candidates[ident]["type"] == "EVENT_LINK"]
        self.assertEqual(nodes[required[0]["arguments"]["child"]]["predicate"], "pull")
        controller = next(item for item in package["candidates"]
                          if item["value"] == "controller" and item["arguments"]["proposition"] == pull)
        self.assertEqual(nodes[controller["arguments"]["mention"]]["label"], "Maria")


if __name__ == "__main__":
    unittest.main()
