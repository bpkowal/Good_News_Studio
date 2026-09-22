"""VERB_ASPECT_CULMINATION — FraCaS §8 structured and grounding lanes."""
from __future__ import annotations

import unittest

from relent_testkit.coverage import load_map, load_taxonomy
from relent_testkit.harness import (
    admit_world_from_discourse,
    load_seed,
    structured_world_from_seed,
    verb_aspect_culmination_holds,
)
from global_workspace.world_validation import WorldModelValidationError
from invariants.catalog import invariant_by_id


class VerbAspectCulminationTests(unittest.TestCase):
    def test_taxonomy_marks_fracas_section_eight(self):
        row = next(
            item for item in load_taxonomy()["phenomena"]
            if item["id"] == "verb_aspect_culmination"
        )
        self.assertEqual(row["source"]["type"], "fracas")
        self.assertEqual(row["source"]["section"], 8)

    def test_catalog_entry(self):
        item = invariant_by_id()["VERB_ASPECT_CULMINATION"]
        self.assertEqual(item.source_type, "fracas")

    def test_structured_lane_perfective_licenses_certain_finish(self):
        seed = load_seed("verb_aspect_built_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            verb_aspect_culmination_holds(
                model,
                culmination_effect_id=expected["culmination_effect_id"],
                expects_certain=bool(expected["expects_certain"]),
            )
        )

    def test_structured_lane_detects_uncertain_finish_under_perfective(self):
        seed = load_seed("verb_aspect_built_structured.yaml")
        model = structured_world_from_seed(seed, world_key="uncertain_finish_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            verb_aspect_culmination_holds(
                model,
                culmination_effect_id=expected["culmination_effect_id"],
                expects_certain=bool(expected["expects_certain"]),
            )
        )

    def test_grounding_lane_perfective_licenses_certain_finish(self):
        seed = load_seed("verb_aspect_built_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            verb_aspect_culmination_holds(
                model,
                culmination_effect_id=expected["culmination_effect_id"],
                expects_certain=bool(expected["expects_certain"]),
            )
        )

    def test_grounding_lane_detects_uncertain_finish_under_perfective(self):
        seed = load_seed("verb_aspect_built_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        try:
            model = admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["uncertain_finish_world"]),
            )
        except WorldModelValidationError as exc:
            self.fail(f"uncertain fixture should admit for oracle check: {exc}")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            verb_aspect_culmination_holds(
                model,
                culmination_effect_id=expected["culmination_effect_id"],
                expects_certain=bool(expected["expects_certain"]),
            )
        )

    def test_grounding_lane_progressive_allows_noncertain_finish(self):
        seed = load_seed("verb_aspect_built_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["progressive_ok_world"]),
        )
        expected = seed["parliament_expectation"]["progressive_expected"]
        self.assertTrue(
            verb_aspect_culmination_holds(
                model,
                culmination_effect_id=expected["culmination_effect_id"],
                expects_certain=bool(expected["expects_certain"]),
            )
        )

    def test_map_lists_verb_seeds(self):
        entry = next(
            row for row in load_map()["entries"]
            if row["phenomenon_id"] == "verb_aspect_culmination"
        )
        self.assertEqual(len(entry["seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
