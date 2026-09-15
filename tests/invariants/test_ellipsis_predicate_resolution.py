"""ELLIPSIS_PREDICATE_RESOLUTION — FraCaS §4 structured and grounding lanes."""
from __future__ import annotations

import unittest

from semantic_integrity.coverage import load_map, load_taxonomy
from semantic_integrity.harness import (
    admit_world_from_discourse,
    ellipsis_predicate_resolution_holds,
    load_seed,
    production_snap_ellipsis_proposition,
    structured_world_from_seed,
)
from global_workspace.world_validation import WorldModelValidationError
from invariants.catalog import invariant_by_id


class EllipsisPredicateResolutionTests(unittest.TestCase):
    def test_taxonomy_marks_fracas_section_four(self):
        taxonomy = load_taxonomy()
        row = next(
            item for item in taxonomy["phenomena"]
            if item["id"] == "ellipsis_predicate_resolution"
        )
        self.assertEqual(row["source"]["type"], "fracas")
        self.assertEqual(row["source"]["section"], 4)

    def test_catalog_entry(self):
        item = invariant_by_id()["ELLIPSIS_PREDICATE_RESOLUTION"]
        self.assertEqual(item.source_type, "fracas")
        self.assertTrue(item.integrity_layers.structured_property)
        self.assertTrue(item.integrity_layers.grounding_property)

    def test_structured_lane_resolves_same_predicate(self):
        seed = load_seed("ellipsis_so_did_rescue_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            ellipsis_predicate_resolution_holds(
                model,
                antecedent_effect_id=expected["antecedent_effect_id"],
                elliptical_effect_id=expected["elliptical_effect_id"],
            )
        )

    def test_structured_lane_detects_wrong_predicate(self):
        seed = load_seed("ellipsis_so_did_rescue_structured.yaml")
        model = structured_world_from_seed(seed, world_key="wrong_predicate_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            ellipsis_predicate_resolution_holds(
                model,
                antecedent_effect_id=expected["antecedent_effect_id"],
                elliptical_effect_id=expected["elliptical_effect_id"],
            )
        )

    def test_grounding_lane_resolves_same_predicate(self):
        seed = load_seed("ellipsis_so_did_rescue_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            ellipsis_predicate_resolution_holds(
                model,
                antecedent_effect_id=expected["antecedent_effect_id"],
                elliptical_effect_id=expected["elliptical_effect_id"],
            )
        )

    def test_grounding_lane_detects_wrong_predicate(self):
        seed = load_seed("ellipsis_so_did_rescue_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        try:
            model = admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["wrong_predicate_world"]),
            )
        except WorldModelValidationError as exc:
            self.fail(f"wrong-predicate fixture should admit for oracle check: {exc}")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            ellipsis_predicate_resolution_holds(
                model,
                antecedent_effect_id=expected["antecedent_effect_id"],
                elliptical_effect_id=expected["elliptical_effect_id"],
            )
        )

    def test_grounding_lane_production_snaps_ellipsis_span(self):
        seed = load_seed("ellipsis_so_did_rescue_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        snapped = production_snap_ellipsis_proposition(
            model, effect_id="E_ellip",
        )
        self.assertIn("rescued", snapped.casefold())
        self.assertIn("residents", snapped.casefold())
        self.assertNotIn("...", snapped)

    def test_map_lists_ellipsis_seeds(self):
        mapping = load_map()
        entry = next(
            row for row in mapping["entries"]
            if row["phenomenon_id"] == "ellipsis_predicate_resolution"
        )
        self.assertEqual(len(entry["seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
