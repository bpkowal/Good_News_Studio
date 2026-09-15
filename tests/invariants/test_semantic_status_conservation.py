"""SEMANTIC_STATUS_CONSERVATION — family integration for purge grounding."""
from __future__ import annotations

import unittest

from global_workspace.world_state import (
    outcome_predicate_is_incomplete,
    validate_world_completeness,
    validate_world_model,
)
from global_workspace.world_validation import (
    format_repair_guidance_for_prompt,
    repair_guidance_cards,
    repair_patch_contract,
    validation_issues_from_messages,
)
from invariants.catalog import invariant_by_id
from semantic_integrity.coverage import load_map, load_taxonomy
from semantic_integrity.harness import (
    admit_world_from_discourse,
    load_seed,
    outcome_predicate_completeness_holds,
    quantity_bearing_consequence_preservation_holds,
    source_stipulated_outcome_preservation_holds,
    structured_world_from_seed,
)


_FAMILY_CODES = {
    "OUTCOME_PREDICATE_INCOMPLETE",
    "SOURCE_STIPULATED_OUTCOME_MISSING",
    "QUANTITY_BEARING_CONSEQUENCE_MISSING",
}


class SemanticStatusConservationTests(unittest.TestCase):
    def test_taxonomy_marks_family(self):
        row = next(
            item for item in load_taxonomy()["phenomena"]
            if item["id"] == "semantic_status_conservation"
        )
        self.assertEqual(row["source"]["type"], "parliament_extension")
        self.assertEqual(row["family"], "status_conservation")

    def test_catalog_umbrella_lists_members(self):
        item = invariant_by_id()["SEMANTIC_STATUS_CONSERVATION"]
        self.assertEqual(item.source_type, "parliament_extension")
        self.assertEqual(item.enforcement, "enforced")
        notes = item.notes.casefold()
        self.assertIn("outcome_predicate_completeness".casefold(), notes)
        self.assertIn("source_stipulated_outcome_preservation".casefold(), notes)
        self.assertIn("quantity_bearing_consequence_preservation".casefold(), notes)

    def test_map_lists_family_seeds(self):
        entry = next(
            row for row in load_map()["entries"]
            if row["phenomenon_id"] == "semantic_status_conservation"
        )
        self.assertEqual(len(entry["seeds"]), 2)
        self.assertEqual(entry["enforcement"], "enforced")

    def test_incomplete_evening_world_fails_all_three_gates(self):
        seed = load_seed("status_conservation_purge_structured.yaml")
        model = structured_world_from_seed(seed, world_key="incomplete_world")
        structural, _ = validate_world_model(model, action_ids=["A0", "A1"])
        completeness = validate_world_completeness(
            model, action_ids=["A0", "A1"],
        )
        issues = validation_issues_from_messages([*structural, *completeness])
        codes = {issue.code for issue in issues}
        self.assertTrue(
            _FAMILY_CODES <= codes,
            f"expected {_FAMILY_CODES}, got {codes}\n"
            f"structural={structural}\ncompleteness={completeness}",
        )
        self.assertTrue(
            any(outcome_predicate_is_incomplete(effect.outcome)
                for effect in model.effects)
        )

    def test_repaired_structured_world_passes_family_oracles(self):
        seed = load_seed("status_conservation_purge_structured.yaml")
        model = structured_world_from_seed(seed, world_key="repaired_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            outcome_predicate_completeness_holds(
                model, effect_id=expected["finished_effect_id"],
            )
        )
        self.assertTrue(
            source_stipulated_outcome_preservation_holds(
                model,
                survival_effect_id=expected["survival_effect_id"],
                loss_effect_id=expected["loss_effect_id"],
            )
        )
        self.assertTrue(
            quantity_bearing_consequence_preservation_holds(
                model,
                life_effect_id=expected["loss_effect_id"],
                research_effect_id=expected["research_effect_id"],
                life_quantity=expected["life_quantity"],
                research_quantity=expected["research_quantity"],
            )
        )
        structural, _ = validate_world_model(model, action_ids=["A0", "A1"])
        completeness = validate_world_completeness(
            model, action_ids=["A0", "A1"],
        )
        family_msgs = [
            msg for msg in [*structural, *completeness]
            if any(
                needle in msg
                for needle in (
                    "incomplete predicate",
                    "source-stipulated outcome",
                    "quantity-bearing consequence",
                )
            )
        ]
        self.assertEqual(family_msgs, [], family_msgs)

    def test_grounding_lane_admits_repaired_purge_world(self):
        seed = load_seed("status_conservation_purge_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["repaired_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            outcome_predicate_completeness_holds(
                model, effect_id=expected["finished_effect_id"],
            )
        )
        self.assertTrue(
            source_stipulated_outcome_preservation_holds(
                model,
                survival_effect_id=expected["survival_effect_id"],
                loss_effect_id=expected["loss_effect_id"],
            )
        )
        self.assertTrue(
            quantity_bearing_consequence_preservation_holds(
                model,
                life_effect_id=expected["loss_effect_id"],
                research_effect_id=expected["research_effect_id"],
                life_quantity=expected["life_quantity"],
                research_quantity=expected["research_quantity"],
            )
        )
        completeness = validate_world_completeness(
            model, action_ids=["A0", "A1"],
        )
        family_msgs = [
            msg for msg in completeness
            if any(
                needle in msg
                for needle in (
                    "source-stipulated outcome",
                    "quantity-bearing consequence",
                )
            )
        ]
        self.assertEqual(family_msgs, [], family_msgs)

    def test_family_repair_cards_cover_all_issue_codes(self):
        messages = [
            "E_fragment outcome 'purge is' is an incomplete predicate; "
            "finish the state or event (not a dangling copula or preposition)",
            "A1 omits source-stipulated outcome 'immediate survival' "
            "from binary contrast; admit a matching effect "
            "(or UNRESOLVED quarantine)",
            "E_loss omits source quantity 'thousands' from quantity-bearing "
            "consequence 'risks thousands of lives'; copy the span onto "
            "the effect (or its population party)",
        ]
        issues = validation_issues_from_messages(messages)
        codes = {issue.code for issue in issues}
        self.assertEqual(codes, _FAMILY_CODES)
        candidate = {
            "world_model": {
                "effects": [
                    {
                        "effect_id": "E_fragment",
                        "action_id": "A1",
                        "outcome": "purge is",
                        "source_proposition": "execute the purge",
                        "clause_ids": ["C3"],
                    },
                    {
                        "effect_id": "E_loss",
                        "action_id": "A0",
                        "outcome": "catastrophic loss of life",
                        "quantities": [],
                        "clause_ids": ["C0"],
                    },
                ],
                "actions": [
                    {"action_id": "A0"},
                    {"action_id": "A1"},
                ],
            },
        }
        clauses = [
            {
                "clause_id": "C0",
                "text": (
                    "A cyberattack on a city's central AI risks thousands "
                    "of lives through imminent infrastructure failure."
                ),
            },
            {
                "clause_id": "C3",
                "text": (
                    "Thorne faces a binary choice: execute the purge to "
                    "guarantee immediate survival, or refrain to preserve "
                    "invaluable knowledge at the cost of catastrophic loss "
                    "of life."
                ),
            },
        ]
        contract = repair_patch_contract(
            issues, candidate=candidate, clauses=clauses,
        )
        card_codes = {card["code"] for card in contract["guidance_cards"]}
        self.assertEqual(card_codes, _FAMILY_CODES)
        prompt = format_repair_guidance_for_prompt(contract["guidance_cards"])
        self.assertIn("purge is", prompt)
        self.assertIn("immediate survival", prompt)
        self.assertIn("thousands", prompt)
        ops = {
            patch.get("op")
            for card in contract["guidance_cards"]
            for patch in (card.get("concrete_patches") or [])
        }
        self.assertTrue({"replace_outcome", "add_effect", "add_quantity"} & ops, ops)


if __name__ == "__main__":
    unittest.main()
