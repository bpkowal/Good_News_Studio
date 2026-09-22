"""ATTITUDE_FACTIVITY — FraCaS §9 structured and grounding lanes."""
from __future__ import annotations

import unittest

from relent_testkit.coverage import load_map, load_taxonomy
from relent_testkit.harness import (
    admit_world_from_discourse,
    attitude_factivity_holds,
    load_seed,
    structured_world_from_seed,
)
from global_workspace.world_validation import WorldModelValidationError
from invariants.catalog import invariant_by_id


class AttitudeFactivityTests(unittest.TestCase):
    def test_taxonomy_marks_fracas_section_nine(self):
        row = next(
            item for item in load_taxonomy()["phenomena"]
            if item["id"] == "attitude_factivity"
        )
        self.assertEqual(row["source"]["type"], "fracas")
        self.assertEqual(row["source"]["section"], 9)

    def test_catalog_entry(self):
        item = invariant_by_id()["ATTITUDE_FACTIVITY"]
        self.assertEqual(item.source_type, "fracas")

    def test_structured_lane_know_licenses_certain_complement(self):
        seed = load_seed("attitude_know_believe_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            attitude_factivity_holds(
                model,
                complement_effect_id=expected["complement_effect_id"],
                factive=bool(expected["factive"]),
            )
        )

    def test_structured_lane_detects_noncertain_under_know(self):
        seed = load_seed("attitude_know_believe_structured.yaml")
        model = structured_world_from_seed(
            seed, world_key="nonfactive_under_know_world",
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            attitude_factivity_holds(
                model,
                complement_effect_id=expected["complement_effect_id"],
                factive=bool(expected["factive"]),
            )
        )

    def test_grounding_lane_know_licenses_certain_complement(self):
        seed = load_seed("attitude_know_believe_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            attitude_factivity_holds(
                model,
                complement_effect_id=expected["complement_effect_id"],
                factive=bool(expected["factive"]),
            )
        )

    def test_grounding_lane_detects_noncertain_under_know(self):
        seed = load_seed("attitude_know_believe_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        try:
            model = admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["nonfactive_under_know_world"]),
            )
        except WorldModelValidationError as exc:
            self.fail(f"nonfactive fixture should admit for oracle check: {exc}")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            attitude_factivity_holds(
                model,
                complement_effect_id=expected["complement_effect_id"],
                factive=bool(expected["factive"]),
            )
        )

    def test_grounding_lane_believe_allows_noncertain_complement(self):
        seed = load_seed("attitude_know_believe_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["believe_ok_world"]),
        )
        expected = seed["parliament_expectation"]["believe_expected"]
        self.assertTrue(
            attitude_factivity_holds(
                model,
                complement_effect_id=expected["complement_effect_id"],
                factive=bool(expected["factive"]),
            )
        )

    def test_map_lists_attitude_seeds(self):
        entry = next(
            row for row in load_map()["entries"]
            if row["phenomenon_id"] == "attitude_factivity"
        )
        self.assertEqual(len(entry["seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
