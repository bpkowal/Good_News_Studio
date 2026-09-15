"""Semantic-integrity coverage CI: enforced regressions fail; cataloged stays green."""
from __future__ import annotations

import importlib
import unittest

from semantic_integrity.coverage import (
    coverage_report,
    enforced_invariant_ids,
    load_map,
    load_taxonomy,
)
from semantic_integrity.harness import seed_dir
from invariants.catalog import INVARIANTS, invariant_by_id


class SemanticIntegrityCoverageTests(unittest.TestCase):
    def test_coverage_report_includes_integrity_invariants(self):
        report = coverage_report().casefold()
        self.assertIn("anaphor_entity_identity", report)
        self.assertIn("negation_scope_siblings", report)
        self.assertIn("quantifier_party_count", report)
        self.assertIn("plural_member_distinctness", report)
        self.assertIn("ellipsis_predicate_resolution", report)
        self.assertIn("adjective_modifier_binding", report)
        self.assertIn("temporal_order_consistency", report)
        self.assertIn("verb_aspect_culmination", report)
        self.assertIn("verb_lemma_outcome_binding", report)
        self.assertIn("attitude_factivity", report)
        self.assertIn("outcome_predicate_completeness", report)
        self.assertIn("source_stipulated_outcome_preservation", report)
        self.assertIn("quantity_bearing_consequence_preservation", report)
        self.assertIn("repair_provenance_minimality", report)
        self.assertIn("averted_alternative_harm", report)
        self.assertIn("likelihood_qualifier_preservation", report)
        self.assertIn("temporal_qualifier_preservation", report)
        self.assertIn("scope_qualifier_preservation", report)
        self.assertIn("semantic_status_conservation", report)

    def test_taxonomy_and_map_agree_on_phenomena(self):
        taxonomy_ids = {
            str(row["id"]) for row in load_taxonomy().get("phenomena") or []
        }
        map_ids = {
            str(row["phenomenon_id"]) for row in load_map().get("entries") or []
        }
        self.assertEqual(taxonomy_ids, map_ids)

    def test_cataloged_or_enforced_map_entries_have_seeds_and_tests(self):
        seed_root = seed_dir()
        for entry in load_map().get("entries") or []:
            enforcement = str(entry.get("enforcement") or "")
            if enforcement not in {"cataloged", "enforced"}:
                continue
            seeds = list(entry.get("seeds") or [])
            self.assertTrue(
                seeds,
                f"{entry.get('phenomenon_id')} ({enforcement}) needs seeds",
            )
            for name in seeds:
                self.assertTrue(
                    (seed_root / name).is_file(),
                    f"missing seed file: {name}",
                )
            module_name = entry.get("test_module")
            self.assertTrue(
                module_name,
                f"{entry.get('phenomenon_id')} needs test_module",
            )
            importlib.import_module(str(module_name))

    def test_enforced_invariants_have_integrity_layers_and_tests(self):
        """CI gate: enforcement=enforced regressions must fail closed."""
        by_id = invariant_by_id()
        for invariant_id in enforced_invariant_ids():
            item = by_id[invariant_id]
            self.assertIsNotNone(
                item.integrity_layers,
                f"{invariant_id} enforced without integrity_layers",
            )
            self.assertTrue(
                item.tests,
                f"{invariant_id} enforced without named tests",
            )
            self.assertEqual(item.production, "pass")

    def test_map_invariants_are_in_semantic_integrity_catalog(self):
        catalog_si = {
            item.id for item in INVARIANTS
            if item.layer == "semantic_integrity"
        }
        for entry in load_map().get("entries") or []:
            invariant_id = str(entry.get("invariant_id") or "")
            if not invariant_id:
                continue
            self.assertIn(invariant_id, catalog_si)


if __name__ == "__main__":
    unittest.main()
