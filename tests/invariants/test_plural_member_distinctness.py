"""PLURAL_MEMBER_DISTINCTNESS — FraCaS §2 structured and grounding lanes."""
from __future__ import annotations

import unittest

from semantic_integrity.coverage import load_map, load_taxonomy
from semantic_integrity.harness import (
    admit_world_from_discourse,
    load_seed,
    plural_member_distinctness_holds,
    structured_world_from_seed,
)
from global_workspace.world_validation import WorldModelValidationError
from invariants.catalog import invariant_by_id


class PluralMemberDistinctnessTests(unittest.TestCase):
    def test_taxonomy_marks_fracas_section_two(self):
        taxonomy = load_taxonomy()
        row = next(
            item for item in taxonomy["phenomena"]
            if item["id"] == "plural_member_distinctness"
        )
        self.assertEqual(row["source"]["type"], "fracas")
        self.assertEqual(row["source"]["section"], 2)

    def test_catalog_entry(self):
        item = invariant_by_id()["PLURAL_MEMBER_DISTINCTNESS"]
        self.assertEqual(item.source_type, "fracas")
        self.assertTrue(item.integrity_layers.structured_property)
        self.assertTrue(item.integrity_layers.grounding_property)

    def test_structured_lane_keeps_members_distinct(self):
        seed = load_seed("plural_conjoined_members_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            plural_member_distinctness_holds(
                model,
                member_party_ids=expected["member_party_ids"],
            )
        )

    def test_structured_lane_detects_silent_merge(self):
        seed = load_seed("plural_conjoined_members_structured.yaml")
        model = structured_world_from_seed(seed, world_key="merged_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            plural_member_distinctness_holds(
                model,
                member_party_ids=expected["member_party_ids"],
            )
        )

    def test_grounding_lane_keeps_members_distinct(self):
        seed = load_seed("plural_conjoined_members_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            plural_member_distinctness_holds(
                model,
                member_party_ids=expected["member_party_ids"],
            )
        )

    def test_grounding_lane_detects_silent_merge(self):
        seed = load_seed("plural_conjoined_members_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        try:
            model = admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["merged_world"]),
            )
        except WorldModelValidationError as exc:
            self.fail(f"merged fixture should admit for oracle check: {exc}")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            plural_member_distinctness_holds(
                model,
                member_party_ids=expected["member_party_ids"],
            )
        )

    def test_map_lists_plural_seeds(self):
        mapping = load_map()
        entry = next(
            row for row in mapping["entries"]
            if row["phenomenon_id"] == "plural_member_distinctness"
        )
        self.assertEqual(len(entry["seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
