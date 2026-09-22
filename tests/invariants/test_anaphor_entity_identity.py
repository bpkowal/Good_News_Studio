"""ANAPHOR_ENTITY_IDENTITY — structured and grounding lanes."""
from __future__ import annotations

import unittest

from relent_testkit.coverage import coverage_report, load_map, load_taxonomy
from relent_testkit.harness import (
    admit_world_from_discourse,
    anaphor_entity_identity_holds,
    load_seed,
    structured_world_from_seed,
)
from global_workspace.scenario_semantics import segment_scenario_clauses
from global_workspace.world_validation import WorldModelValidationError
from invariants.catalog import format_integrity_layers_report, invariant_by_id


class AnaphorEntityIdentityTests(unittest.TestCase):
    def test_taxonomy_marks_anaphora_as_fracas_section_three(self):
        taxonomy = load_taxonomy()
        row = next(
            item for item in taxonomy["phenomena"]
            if item["id"] == "anaphor_entity_identity"
        )
        self.assertEqual(row["source"]["type"], "fracas")
        self.assertEqual(row["source"]["section"], 3)
        self.assertIn("grounding", row["lanes"])
        neg = next(
            item for item in taxonomy["phenomena"]
            if item["id"] == "negation_scope_siblings"
        )
        self.assertEqual(neg["source"]["type"], "parliament_extension")

    def test_catalog_integrity_layers_include_grounding(self):
        item = invariant_by_id()["ANAPHOR_ENTITY_IDENTITY"]
        self.assertEqual(item.source_type, "fracas")
        self.assertIsNotNone(item.integrity_layers)
        self.assertTrue(item.integrity_layers.grounding_property)
        self.assertFalse(item.integrity_layers.production_validator)
        report = format_integrity_layers_report()
        self.assertIn("ANAPHOR_ENTITY_IDENTITY", report)
        self.assertIn("grounding", coverage_report().casefold())

    def test_structured_lane_preserves_party_identity(self):
        seed = load_seed("anaphor_smith_workstation_structured.yaml")
        self.assertEqual(seed["lane"], "structured")
        self.assertEqual(
            seed["parliament_expectation"]["invariant"],
            "ANAPHOR_ENTITY_IDENTITY",
        )
        # FraCaS gold is not the oracle for this lane.
        self.assertIsNone(seed["fracas"].get("gold"))
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            anaphor_entity_identity_holds(
                model,
                antecedent_effect_id=expected["antecedent_effect_id"],
                anaphor_effect_id=expected["anaphor_effect_id"],
            )
        )

    def test_grounding_lane_segments_discourse_and_preserves_party_identity(self):
        seed = load_seed("anaphor_smith_workstation_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        clauses = segment_scenario_clauses(discourse)
        self.assertGreaterEqual(len(clauses), 4)
        self.assertIn("He used", clauses[1]["text"])
        self.assertIn("Smith owns", clauses[0]["text"])

        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        # Anaphor wording survived into the admitted model.
        ana = next(e for e in model.effects if e.effect_id == "E_ana")
        self.assertIn("he used", ana.source_proposition.casefold())

        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            anaphor_entity_identity_holds(
                model,
                antecedent_effect_id=expected["antecedent_effect_id"],
                anaphor_effect_id=expected["anaphor_effect_id"],
            ),
            "anaphor effect must share party_id with antecedent after admit",
        )

    def test_grounding_lane_detects_split_identity(self):
        seed = load_seed("anaphor_smith_workstation_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        # Split-identity candidates may still parse; the Parliament oracle fails.
        try:
            model = admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["split_identity_world"]),
            )
        except WorldModelValidationError as exc:
            self.fail(f"split-identity fixture should admit for oracle check: {exc}")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            anaphor_entity_identity_holds(
                model,
                antecedent_effect_id=expected["antecedent_effect_id"],
                anaphor_effect_id=expected["anaphor_effect_id"],
            )
        )

    def test_map_lists_anaphor_seeds(self):
        mapping = load_map()
        entry = next(
            row for row in mapping["entries"]
            if row["phenomenon_id"] == "anaphor_entity_identity"
        )
        self.assertEqual(entry["invariant_id"], "ANAPHOR_ENTITY_IDENTITY")
        self.assertEqual(len(entry["seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
