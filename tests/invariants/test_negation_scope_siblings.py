"""NEGATION_SCOPE_SIBLINGS — structured and grounding lanes."""
from __future__ import annotations

import unittest

from relent_testkit.coverage import load_map, load_taxonomy
from relent_testkit.harness import (
    admit_world_from_discourse,
    discourse_role_field_for_party,
    load_seed,
    negation_scope_siblings_holds,
    structured_world_from_seed,
)
from global_workspace.world_validation import WorldModelValidationError
from invariants.catalog import invariant_by_id


class NegationScopeSiblingsTests(unittest.TestCase):
    def test_taxonomy_marks_parliament_extension(self):
        taxonomy = load_taxonomy()
        row = next(
            item for item in taxonomy["phenomena"]
            if item["id"] == "negation_scope_siblings"
        )
        self.assertEqual(row["source"]["type"], "parliament_extension")
        self.assertIn("structured", row["lanes"])
        self.assertIn("grounding", row["lanes"])

    def test_catalog_entry(self):
        item = invariant_by_id()["NEGATION_SCOPE_SIBLINGS"]
        self.assertEqual(item.source_type, "parliament_extension")
        self.assertIsNotNone(item.integrity_layers)
        self.assertTrue(item.integrity_layers.structured_property)
        self.assertTrue(item.integrity_layers.grounding_property)
        self.assertFalse(item.integrity_layers.production_validator)

    def test_structured_lane_preserves_sibling_polarity(self):
        seed = load_seed("negation_treatment_dies_structured.yaml")
        self.assertEqual(seed["lane"], "structured")
        self.assertIsNone(seed["fracas"].get("gold"))
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            negation_scope_siblings_holds(
                model,
                action_id=expected["action_id"],
                negated_effect_id=expected["negated_effect_id"],
                siblings=expected["siblings"],
            )
        )

    def test_structured_lane_detects_flipped_sibling(self):
        seed = load_seed("negation_treatment_dies_structured.yaml")
        model = structured_world_from_seed(
            seed, world_key="flipped_sibling_world",
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            negation_scope_siblings_holds(
                model,
                action_id=expected["action_id"],
                negated_effect_id=expected["negated_effect_id"],
                siblings=expected["siblings"],
            )
        )

    def test_grounding_lane_role_attachment_marks_harm(self):
        seed = load_seed("negation_treatment_dies_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        expected = seed["parliament_expectation"]["expected"]
        # Production attachment: "no treatment and dies" → patient is harmed.
        field = discourse_role_field_for_party(
            discourse,
            party_label=expected["role_party_label"],
        )
        self.assertEqual(field, expected["role_field"])

    def test_grounding_lane_preserves_sibling_polarity(self):
        seed = load_seed("negation_treatment_dies_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            negation_scope_siblings_holds(
                model,
                action_id=expected["action_id"],
                negated_effect_id=expected["negated_effect_id"],
                siblings=expected["siblings"],
            )
        )

    def test_grounding_lane_detects_flipped_sibling(self):
        seed = load_seed("negation_treatment_dies_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        try:
            model = admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["flipped_sibling_world"]),
            )
        except WorldModelValidationError as exc:
            self.fail(f"flipped-sibling fixture should admit for oracle check: {exc}")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            negation_scope_siblings_holds(
                model,
                action_id=expected["action_id"],
                negated_effect_id=expected["negated_effect_id"],
                siblings=expected["siblings"],
            )
        )

    def test_map_lists_negation_seeds(self):
        mapping = load_map()
        entry = next(
            row for row in mapping["entries"]
            if row["phenomenon_id"] == "negation_scope_siblings"
        )
        self.assertEqual(entry["invariant_id"], "NEGATION_SCOPE_SIBLINGS")
        self.assertEqual(len(entry["seeds"]), 2)
        self.assertEqual(entry["enforcement"], "cataloged")


if __name__ == "__main__":
    unittest.main()
