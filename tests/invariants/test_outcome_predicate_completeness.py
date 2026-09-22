"""OUTCOME_PREDICATE_COMPLETENESS — reject dangling-copula outcome fragments."""
from __future__ import annotations

import unittest
from dataclasses import replace

from global_workspace.world_state import (
    WorldEffect,
    WorldModelValidationError,
    _unstated_negated_benefit,
    outcome_predicate_is_incomplete,
    outcome_polarity_contradiction,
    validate_world_model,
)
from global_workspace.world_validation import (
    repair_guidance_cards,
    validation_issues_from_messages,
)
from invariants.catalog import invariant_by_id
from relent_testkit.coverage import load_map, load_taxonomy
from relent_testkit.harness import (
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
        self.assertTrue(outcome_predicate_is_incomplete("is hit by the"))
        self.assertTrue(outcome_predicate_is_incomplete("the side track"))
        self.assertFalse(outcome_predicate_is_incomplete("is hit by the trolley"))
        self.assertFalse(outcome_predicate_is_incomplete("the patient survives"))
        self.assertFalse(outcome_predicate_is_incomplete(
            "give the medicine to Patient A"
        ))
        self.assertFalse(outcome_predicate_is_incomplete(
            "allocate the ventilator to Group B"
        ))
        seed = load_seed("outcome_predicate_purge_structured.yaml")
        model = structured_world_from_seed(seed, world_key="incomplete_world")
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertTrue(
            any("incomplete predicate" in error for error in errors),
            errors,
        )

    def test_resource_transfer_cannot_be_only_a_resource_noun(self):
        seed = load_seed("outcome_predicate_purge_structured.yaml")
        model = structured_world_from_seed(seed)
        effect = model.effects[0]
        malformed = replace(
            model,
            effects=(replace(
                effect,
                outcome="the medicine",
                relation="RECEIVES_MEDICINE",
                effect_kind="RESOURCE_TRANSFER",
            ),),
        )
        errors, _ = validate_world_model(malformed, action_ids=["A0"])
        self.assertTrue(any(
            "incomplete predicate for RESOURCE_TRANSFER" in error
            for error in errors
        ), errors)

    def test_resource_transfer_accepts_complete_giving_event(self):
        seed = load_seed("outcome_predicate_purge_structured.yaml")
        model = structured_world_from_seed(seed)
        effect = model.effects[0]
        complete = replace(
            model,
            effects=(replace(
                effect,
                outcome="give the medicine to Patient A",
                relation="GIVE",
                effect_kind="RESOURCE_TRANSFER",
            ),),
        )
        errors, _ = validate_world_model(complete, action_ids=["A0"])
        self.assertFalse(any(
            "incomplete predicate for RESOURCE_TRANSFER" in error
            for error in errors
        ), errors)

    def test_outcome_polarity_rejects_unnegated_harm_as_benefit(self):
        effect = WorldEffect(
            "E_hit", "A0", "P1", "is hit by the trolley", "IS",
            "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
        )
        self.assertIn("adverse event", outcome_polarity_contradiction(effect))

    def test_outcome_polarity_accepts_negated_or_prevented_harm(self):
        for outcome in ("is not hit by the trolley", "averts being hit"):
            effect = WorldEffect(
                "E_safe", "A0", "P1", outcome, "IS", "BENEFICIAL",
                "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
            )
            self.assertEqual("", outcome_polarity_contradiction(effect), outcome)

    def test_without_resource_does_not_negate_matrix_harm(self):
        effect = WorldEffect(
            "E_death", "A0", "P1", "dies without the medicine", "DIES",
            "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
        )
        self.assertEqual("", outcome_polarity_contradiction(effect))

    def test_without_adverse_complement_negates_that_harm(self):
        effect = WorldEffect(
            "E_harm", "A0", "P1", "leaves without harm", "LEAVES",
            "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
        )
        self.assertEqual("negates an adverse event", outcome_polarity_contradiction(effect))

    def test_polarity_error_gets_targeted_repair_code(self):
        issues = validation_issues_from_messages([
            "E_hit outcome 'is hit by the trolley' contradicts polarity "
            "BENEFICIAL: it states an adverse event without negating or "
            "preventing it"
        ])
        self.assertEqual("OUTCOME_POLARITY_CONTRADICTION", issues[0].code)

    def test_positive_harm_source_cannot_be_inverted_into_actual_benefit(self):
        inverted = WorldEffect(
            "E_spared", "A0", "P500", "are not hit by the trolley", "SPARED",
            "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
            quantities=("1",),
            source_proposition="moving the trolley where it will hit exactly 1 person",
            source_effect_ids=("E_move",),
            derivation_operation="SOURCE_STIPULATED_CAUSAL",
        )
        self.assertTrue(_unstated_negated_benefit(inverted))
        stated = replace(
            inverted,
            source_proposition="500 people are not hit by the trolley",
            quantities=("500",),
        )
        self.assertFalse(_unstated_negated_benefit(stated))

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

    def test_transfer_fragment_repair_card_quotes_transfer_event(self):
        issues = validation_issues_from_messages([
            "E0 outcome 'the medicine' is an incomplete predicate for "
            "RESOURCE_TRANSFER; state the source-licensed giving event"
        ])
        cards = repair_guidance_cards(
            issues,
            {"world_model": {"effects": [{
                "effect_id": "E0", "action_id": "A0",
                "effect_kind": "RESOURCE_TRANSFER",
                "outcome": "the medicine",
                "source_proposition": "give the medicine to Patient A",
                "clause_ids": ["C0"],
            }]}},
            clauses=[{
                "clause_id": "C0",
                "text": (
                    "The doctor must choose to give the medicine to Patient A, "
                    "or give it to Patient B."
                ),
            }],
        )
        values = [
            str(patch.get("value") or "")
            for patch in cards[0].get("concrete_patches") or []
            if patch.get("op") == "replace_outcome"
        ]
        self.assertIn("give the medicine to Patient A", values)

    def test_map_lists_predicate_seeds(self):
        entry = next(
            row for row in load_map()["entries"]
            if row["phenomenon_id"] == "outcome_predicate_completeness"
        )
        self.assertEqual(len(entry["seeds"]), 2)
        self.assertEqual(entry["enforcement"], "enforced")


if __name__ == "__main__":
    unittest.main()
