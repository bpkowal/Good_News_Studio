"""OUTCOME_PREDICATE_COMPLETENESS — reject dangling-copula outcome fragments."""
from __future__ import annotations

import unittest

from global_workspace.world_state import (
    WorldModelValidationError,
    outcome_predicate_is_incomplete,
    validate_world_model,
)
from global_workspace.world_validation import (
    repair_guidance_cards,
    validation_issues_from_messages,
)
from invariants.catalog import invariant_by_id
from semantic_integrity.coverage import load_map, load_taxonomy
from semantic_integrity.harness import (
    admit_world_from_discourse,
    load_seed,
    outcome_predicate_completeness_holds,
    structured_world_from_seed,
)


class OutcomePredicateCompletenessTests(unittest.TestCase):
    def test_taxonomy_marks_parliament_extension(self):
        row = next(
            item for item in load_taxonomy()["phenomena"]
            if item["id"] == "outcome_predicate_completeness"
        )
        self.assertEqual(row["source"]["type"], "parliament_extension")

    def test_catalog_entry(self):
        item = invariant_by_id()["OUTCOME_PREDICATE_COMPLETENESS"]
        self.assertEqual(item.source_type, "parliament_extension")
        self.assertEqual(item.enforcement, "enforced")
        self.assertTrue(item.integrity_layers.production_validator)

    def test_structured_lane_complete_outcome_holds(self):
        seed = load_seed("outcome_predicate_purge_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            outcome_predicate_completeness_holds(
                model, effect_id=expected["effect_id"],
            )
        )

    def test_structured_lane_detects_dangling_copula(self):
        seed = load_seed("outcome_predicate_purge_structured.yaml")
        model = structured_world_from_seed(seed, world_key="incomplete_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            outcome_predicate_completeness_holds(
                model, effect_id=expected["effect_id"],
            )
        )

    def test_grounding_lane_rejects_dangling_copula(self):
        seed = load_seed("outcome_predicate_purge_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        with self.assertRaises(WorldModelValidationError) as ctx:
            admit_world_from_discourse(
                discourse,
                actions=list(seed["actions"]),
                world_candidate=dict(seed["incomplete_world"]),
            )
        messages = " ".join(ctx.exception.messages).casefold()
        self.assertIn("incomplete predicate", messages)
        self.assertIn("purge is", messages)
        codes = {issue.code for issue in ctx.exception.issues}
        self.assertIn("OUTCOME_PREDICATE_INCOMPLETE", codes)

    def test_grounding_lane_admits_finished_predicate(self):
        seed = load_seed("outcome_predicate_purge_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            outcome_predicate_completeness_holds(
                model, effect_id=expected["effect_id"],
            )
        )

    def test_production_validator_rejects_purge_is(self):
        self.assertTrue(outcome_predicate_is_incomplete("purge is"))
        self.assertFalse(outcome_predicate_is_incomplete("is purged"))
        self.assertFalse(outcome_predicate_is_incomplete("IS"))
        self.assertTrue(outcome_predicate_is_incomplete("sent to"))
        seed = load_seed("outcome_predicate_purge_structured.yaml")
        model = structured_world_from_seed(seed, world_key="incomplete_world")
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertTrue(
            any("incomplete predicate" in error for error in errors),
            errors,
        )

    def test_repair_card_for_incomplete_outcome(self):
        messages = [
            "E_purge outcome 'purge is' is an incomplete predicate; "
            "finish the state or event (not a dangling copula or preposition)"
        ]
        issues = validation_issues_from_messages(messages)
        cards = repair_guidance_cards(
            issues,
            {
                "world_model": {
                    "effects": [{
                        "effect_id": "E_purge",
                        "action_id": "A0",
                        "outcome": "purge is",
                        "source_proposition": "deploy an emergency purge",
                        "clause_ids": ["C0"],
                    }],
                },
            },
            clauses=[{
                "clause_id": "C0",
                "text": (
                    "Engineer Aris Thorne can deploy an emergency purge "
                    "to secure the city."
                ),
            }],
        )
        self.assertEqual(cards[0]["code"], "OUTCOME_PREDICATE_INCOMPLETE")
        patches = cards[0].get("concrete_patches") or []
        values = {
            str(patch.get("value") or "")
            for patch in patches
            if patch.get("op") == "replace_outcome"
        }
        self.assertTrue(values, patches)
        self.assertNotIn("is purge", values)
        self.assertTrue(
            any("purge" in value.casefold() for value in values),
            values,
        )

    def test_map_lists_predicate_seeds(self):
        entry = next(
            row for row in load_map()["entries"]
            if row["phenomenon_id"] == "outcome_predicate_completeness"
        )
        self.assertEqual(len(entry["seeds"]), 2)
        self.assertEqual(entry["enforcement"], "enforced")


if __name__ == "__main__":
    unittest.main()
