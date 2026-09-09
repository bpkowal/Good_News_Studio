from __future__ import annotations

import unittest

from .catalog import INVARIANTS, format_coverage_report, invariant_by_id, production_status


class InvariantCatalogTests(unittest.TestCase):
    def test_ids_are_unique_and_stable(self):
        ids = [item.id for item in INVARIANTS]
        self.assertEqual(ids, list(dict.fromkeys(ids)))
        self.assertEqual(set(invariant_by_id()), set(ids))

    def test_every_invariant_has_a_layer_and_a_one_line_description(self):
        for item in INVARIANTS:
            self.assertTrue(item.layer.strip(), item.id)
            self.assertGreaterEqual(len(item.description.split()), 8, item.id)
            self.assertIn(item.coverage, {"story", "structural", "none"})
            self.assertIn(item.oracle_risk, {"independent", "coupled", "unknown"})
            self.assertIn(
                production_status(item),
                {"pass", "fail", "story", "untested"},
                item.id,
            )
            if item.coverage == "none":
                self.assertEqual(item.tests, ())
                self.assertEqual(production_status(item), "untested", item.id)
            else:
                self.assertTrue(item.tests, item.id)

    def test_coverage_report_lists_every_invariant(self):
        report = format_coverage_report()
        self.assertIn("Invariant", report)
        self.assertIn("Generation", report)
        self.assertIn("Production", report)
        for item in INVARIANTS:
            self.assertIn(item.id, report)
            self.assertIn(item.coverage, report.split(item.id, 1)[1][:80])
        for invariant_id in (
            "CHANCE_IS_NOT_AN_OUTCOME",
            "COMPOSITIONAL_HYPOTHESIS_INFLUENCE",
            "PRESENTATION_PRESERVES_MODALITY",
            "MODALITY_BLIND_COST_SHAPE",
            "COMPOSITIONAL_UNGROUNDED_THRESHOLD",
            "FOREGONE_IS_NOT_OBTAINED",
            "NO_PREFERENCE_NO_SUPPORT",
            "AVERTED_RISK_IS_NOT_OBTAINED_BENEFIT",
            "BOUND_QUANTITY_NORMALIZATION",
            "LIKELIHOOD_SPAN_NORMALIZATION",
            "LIKELIHOOD_IS_NOT_A_PARTY_QUANTITY",
            "QUALIFIER_HEAD_BINDING",
            "QUANTITY_PARTY_UNIQUENESS",
            "RISK_SITUATION_TYPING",
            "ACTION_MEDIATED_VS_EXOGENOUS",
        ):
            self.assertEqual(production_status(invariant_by_id()[invariant_id]), "pass")
