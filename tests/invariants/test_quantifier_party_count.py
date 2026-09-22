"""QUANTIFIER_PARTY_COUNT — FraCaS §1 structured and grounding lanes."""
from __future__ import annotations

import unittest

from relent_testkit.coverage import load_map, load_taxonomy
from relent_testkit.harness import (
    admit_world_from_discourse,
    load_seed,
    production_assigned_quantities,
    quantifier_party_count_holds,
    structured_world_from_seed,
)
from global_workspace.world_validation import WorldModelValidationError
from invariants.catalog import invariant_by_id


class QuantifierPartyCountTests(unittest.TestCase):
    def test_taxonomy_marks_fracas_section_one(self):
        taxonomy = load_taxonomy()
        row = next(
            item for item in taxonomy["phenomena"]
            if item["id"] == "quantifier_party_count"
        )
        self.assertEqual(row["source"]["type"], "fracas")
        self.assertEqual(row["source"]["section"], 1)

    def test_catalog_entry(self):
        item = invariant_by_id()["QUANTIFIER_PARTY_COUNT"]
        self.assertEqual(item.source_type, "fracas")
        self.assertTrue(item.integrity_layers.structured_property)
        self.assertTrue(item.integrity_layers.grounding_property)

    def test_structured_lane_preserves_count_on_party(self):
        seed = load_seed("quantifier_thirty_residents_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            quantifier_party_count_holds(
                model,
                party_id=expected["party_id"],
                expected_quantities=expected["quantities"],
            )
        )

    def test_structured_lane_detects_leaked_quantity(self):
        seed = load_seed("quantifier_thirty_residents_structured.yaml")
        model = structured_world_from_seed(seed, world_key="leaked_quantity_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            quantifier_party_count_holds(
                model,
                party_id=expected["party_id"],
                expected_quantities=expected["quantities"],
            )
        )

    def test_grounding_lane_preserves_count_on_party(self):
        seed = load_seed("quantifier_thirty_residents_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            quantifier_party_count_holds(
                model,
                party_id=expected["party_id"],
                expected_quantities=expected["quantities"],
            )
        )

    def test_grounding_lane_closed_class_recovers_leaked_quantity(self):
        """Production admit strips facility leak and rebinds thirty to residents."""
        seed = load_seed("quantifier_thirty_residents_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        try:
            model = admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["leaked_quantity_world"]),
            )
        except WorldModelValidationError as exc:
            self.fail(f"leaked fixture should admit for recovery check: {exc}")
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            quantifier_party_count_holds(
                model,
                party_id=expected["party_id"],
                expected_quantities=expected["quantities"],
            ),
            "closed_class_party_quantities must restore the declared count",
        )

    def test_grounding_lane_production_binder_recovers_count(self):
        seed = load_seed("quantifier_thirty_residents_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["leaked_quantity_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        assigned = production_assigned_quantities(model)
        got = {
            item.casefold()
            for item in assigned.get(expected["party_id"].casefold(), ())
        }
        self.assertEqual(got, {"thirty"})

    def test_map_lists_quantifier_seeds(self):
        mapping = load_map()
        entry = next(
            row for row in mapping["entries"]
            if row["phenomenon_id"] == "quantifier_party_count"
        )
        self.assertEqual(len(entry["seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
