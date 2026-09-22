"""TEMPORAL_ORDER_CONSISTENCY — FraCaS §7 structured and grounding lanes."""
from __future__ import annotations

import unittest

from relent_testkit.coverage import load_map, load_taxonomy
from relent_testkit.harness import (
    admit_world_from_discourse,
    load_seed,
    structured_world_from_seed,
    temporal_order_consistency_holds,
)
from global_workspace.world_validation import WorldModelValidationError
from invariants.catalog import invariant_by_id


class TemporalOrderConsistencyTests(unittest.TestCase):
    def test_taxonomy_marks_fracas_section_seven(self):
        row = next(
            item for item in load_taxonomy()["phenomena"]
            if item["id"] == "temporal_order_consistency"
        )
        self.assertEqual(row["source"]["type"], "fracas")
        self.assertEqual(row["source"]["section"], 7)

    def test_catalog_entry(self):
        item = invariant_by_id()["TEMPORAL_ORDER_CONSISTENCY"]
        self.assertEqual(item.source_type, "fracas")
        self.assertTrue(item.integrity_layers.grounding_property)

    def test_structured_lane_keeps_after_on_later_event(self):
        seed = load_seed("temporal_after_before_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            temporal_order_consistency_holds(
                model,
                earlier_effect_id=expected["earlier_effect_id"],
                later_effect_id=expected["later_effect_id"],
                later_marker=expected["later_marker"],
            )
        )

    def test_structured_lane_detects_swapped_order(self):
        seed = load_seed("temporal_after_before_structured.yaml")
        model = structured_world_from_seed(seed, world_key="swapped_order_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            temporal_order_consistency_holds(
                model,
                earlier_effect_id=expected["earlier_effect_id"],
                later_effect_id=expected["later_effect_id"],
                later_marker=expected["later_marker"],
            )
        )

    def test_grounding_lane_keeps_after_on_later_event(self):
        seed = load_seed("temporal_after_before_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            temporal_order_consistency_holds(
                model,
                earlier_effect_id=expected["earlier_effect_id"],
                later_effect_id=expected["later_effect_id"],
                later_marker=expected["later_marker"],
            )
        )

    def test_grounding_lane_detects_swapped_order(self):
        seed = load_seed("temporal_after_before_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        try:
            model = admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["swapped_order_world"]),
            )
        except WorldModelValidationError as exc:
            self.fail(f"swapped fixture should admit for oracle check: {exc}")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            temporal_order_consistency_holds(
                model,
                earlier_effect_id=expected["earlier_effect_id"],
                later_effect_id=expected["later_effect_id"],
                later_marker=expected["later_marker"],
            )
        )

    def test_map_lists_temporal_seeds(self):
        entry = next(
            row for row in load_map()["entries"]
            if row["phenomenon_id"] == "temporal_order_consistency"
        )
        self.assertEqual(len(entry["seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
