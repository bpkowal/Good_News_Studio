"""ADJECTIVE_MODIFIER_BINDING — FraCaS §5 structured and grounding lanes."""
from __future__ import annotations

import unittest

from semantic_integrity.coverage import load_map, load_taxonomy
from semantic_integrity.harness import (
    admit_world_from_discourse,
    adjective_modifier_binding_holds,
    load_seed,
    production_temporal_qualifiers_for_effect,
    structured_world_from_seed,
)
from global_workspace.world_validation import WorldModelValidationError
from invariants.catalog import invariant_by_id


class AdjectiveModifierBindingTests(unittest.TestCase):
    def test_taxonomy_marks_fracas_section_five(self):
        taxonomy = load_taxonomy()
        row = next(
            item for item in taxonomy["phenomena"]
            if item["id"] == "adjective_modifier_binding"
        )
        self.assertEqual(row["source"]["type"], "fracas")
        self.assertEqual(row["source"]["section"], 5)

    def test_catalog_entry(self):
        item = invariant_by_id()["ADJECTIVE_MODIFIER_BINDING"]
        self.assertEqual(item.source_type, "fracas")
        self.assertTrue(item.integrity_layers.structured_property)
        self.assertTrue(item.integrity_layers.grounding_property)

    def test_structured_lane_binds_stacked_modifiers(self):
        seed = load_seed("adjective_stacked_harm_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            adjective_modifier_binding_holds(
                model,
                effect_id=expected["effect_id"],
                expected_temporal_qualifiers=expected["temporal_qualifiers"],
                sibling_effect_id=expected["sibling_effect_id"],
                party_id=expected["party_id"],
                expected_party_kind=expected["expected_party_kind"],
                label_must_contain=expected["label_must_contain"],
            )
        )

    def test_structured_lane_detects_leaked_modifiers(self):
        seed = load_seed("adjective_stacked_harm_structured.yaml")
        model = structured_world_from_seed(seed, world_key="leaked_modifier_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            adjective_modifier_binding_holds(
                model,
                effect_id=expected["effect_id"],
                expected_temporal_qualifiers=expected["temporal_qualifiers"],
                sibling_effect_id=expected["sibling_effect_id"],
                party_id=expected["party_id"],
                expected_party_kind=expected["expected_party_kind"],
                label_must_contain=expected["label_must_contain"],
            )
        )

    def test_grounding_lane_binds_stacked_modifiers(self):
        seed = load_seed("adjective_stacked_harm_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            adjective_modifier_binding_holds(
                model,
                effect_id=expected["effect_id"],
                expected_temporal_qualifiers=expected["temporal_qualifiers"],
                sibling_effect_id=expected["sibling_effect_id"],
                party_id=expected["party_id"],
                expected_party_kind=expected["expected_party_kind"],
                label_must_contain=expected["label_must_contain"],
            )
        )

    def test_grounding_lane_detects_leaked_modifiers(self):
        seed = load_seed("adjective_stacked_harm_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        try:
            model = admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["leaked_modifier_world"]),
            )
        except WorldModelValidationError as exc:
            self.fail(f"leaked fixture should admit for oracle check: {exc}")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            adjective_modifier_binding_holds(
                model,
                effect_id=expected["effect_id"],
                expected_temporal_qualifiers=expected["temporal_qualifiers"],
                sibling_effect_id=expected["sibling_effect_id"],
                party_id=expected["party_id"],
                expected_party_kind=expected["expected_party_kind"],
                label_must_contain=expected["label_must_contain"],
            )
        )

    def test_grounding_lane_production_binds_temporals_to_harm(self):
        seed = load_seed("adjective_stacked_harm_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        harm = {
            item.casefold()
            for item in production_temporal_qualifiers_for_effect(
                model, effect_id="E_harm",
            )
        }
        process = {
            item.casefold()
            for item in production_temporal_qualifiers_for_effect(
                model, effect_id="E_process",
            )
        }
        self.assertEqual(harm, {"immediate", "prolonged"})
        self.assertFalse(process & {"immediate", "prolonged"})

    def test_map_lists_adjective_seeds(self):
        mapping = load_map()
        entry = next(
            row for row in mapping["entries"]
            if row["phenomenon_id"] == "adjective_modifier_binding"
        )
        self.assertEqual(len(entry["seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
