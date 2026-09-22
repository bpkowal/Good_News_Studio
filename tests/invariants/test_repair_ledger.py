"""Repair-ledger integrity and promoted production validators."""
from __future__ import annotations

import unittest

from global_workspace.world_state import (
    validate_world_model,
)
from global_workspace.world_validation import (
    repair_guidance_cards,
    validation_issues_from_messages,
)
from invariants.catalog import invariant_by_id
from relent_testkit.coverage import load_map
from relent_testkit.repair_ledger import load_repair_ledger
from strategies.quantifier_count import quantifier_count_cases
from strategies.verb_lemma import verb_lemma_cases
from hypothesis import given, settings


class RepairLedgerTests(unittest.TestCase):
    def test_ledger_entries_reference_catalog_invariants(self):
        catalog = invariant_by_id()
        ledger = load_repair_ledger()
        for row in ledger["entries"]:
            self.assertIn(row["invariant_id"], catalog, row["failure_class"])
            self.assertIn(
                row["promote"],
                {"enforced", "cataloged_hold", "deferred"},
                row["failure_class"],
            )
            self.assertTrue(row["issue_code"])
            self.assertTrue(row["allowed_ops"])
            self.assertTrue(row["repair_stage"])

    def test_promoted_families_match_catalog_and_map(self):
        ledger = load_repair_ledger()
        catalog = invariant_by_id()
        map_by_invariant = {
            str(row["invariant_id"]): row
            for row in load_map()["entries"]
        }
        for row in ledger["entries"]:
            if row["promote"] != "enforced":
                continue
            item = catalog[row["invariant_id"]]
            self.assertTrue(
                item.integrity_layers.production_validator,
                row["invariant_id"],
            )
            self.assertEqual(item.enforcement, "enforced", row["invariant_id"])
            mapped = map_by_invariant[row["invariant_id"]]
            self.assertEqual(mapped["enforcement"], "enforced", row["invariant_id"])


class PromotedQuantifierValidatorTests(unittest.TestCase):
    @given(quantifier_count_cases())
    @settings(max_examples=25, deadline=None)
    def test_production_rejects_leaked_quantity(self, case):
        errors, _ = validate_world_model(case.world, action_ids=["A0"])
        leak_errors = [
            error for error in errors
            if "records quantity" in error and "binds to" in error
        ]
        if case.preserves_count:
            self.assertFalse(leak_errors, errors)
        else:
            self.assertTrue(leak_errors, errors)

    def test_repair_card_for_quantifier_leak(self):
        messages = [
            "P_fac records quantity 'thirty' that source binds to P_pop; "
            "move the span to P_pop or clear it here"
        ]
        issues = validation_issues_from_messages(messages)
        self.assertEqual(issues[0].code, "QUANTIFIER_PARTY_LEAK")
        cards = repair_guidance_cards(issues, {"world_model": {"parties": []}})
        self.assertEqual(cards[0]["code"], "QUANTIFIER_PARTY_LEAK")
        ops = {patch["op"] for patch in cards[0]["concrete_patches"]}
        self.assertIn("move_quantity", ops)
        self.assertIn("clear_quantity", ops)


class PromotedVerbLemmaValidatorTests(unittest.TestCase):
    @given(verb_lemma_cases())
    @settings(max_examples=25, deadline=None)
    def test_production_rejects_wrong_lemma(self, case):
        errors, _ = validate_world_model(case.world, action_ids=["A0"])
        lemma_errors = [
            error for error in errors if "outcome lemma does not bind" in error
        ]
        if case.binds_lemma:
            self.assertFalse(lemma_errors, errors)
        else:
            self.assertTrue(lemma_errors, errors)

    def test_repair_card_for_lemma_mismatch(self):
        messages = [
            "E0 outcome lemma does not bind to source_proposition; "
            "rewrite the outcome to a morphological variant of the source "
            "event or omit the derived world effect"
        ]
        issues = validation_issues_from_messages(messages)
        self.assertEqual(issues[0].code, "VERB_LEMMA_MISMATCH")
        cards = repair_guidance_cards(
            issues,
            {
                "world_model": {
                    "effects": [{
                        "effect_id": "E0",
                        "action_id": "A0",
                        "outcome": "is destroyed",
                        "source_proposition": "deploy an emergency purge",
                        "clause_ids": ["C0"],
                    }],
                },
            },
            clauses=[{
                "clause_id": "C0",
                "text": "Engineer can deploy an emergency purge to secure the city.",
            }],
        )
        self.assertEqual(cards[0]["code"], "VERB_LEMMA_MISMATCH")
        self.assertIn("replace_outcome", cards[0]["allowed_operations"])


if __name__ == "__main__":
    unittest.main()
