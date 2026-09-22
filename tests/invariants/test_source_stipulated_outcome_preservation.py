"""SOURCE_STIPULATED_OUTCOME_PRESERVATION — binary-contrast life stakes."""
from __future__ import annotations

import unittest

from global_workspace.world_state import (
    extract_binary_contrast_stipulations,
    validate_world_completeness,
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
    source_stipulated_outcome_preservation_holds,
    structured_world_from_seed,
)


class SourceStipulatedOutcomePreservationTests(unittest.TestCase):
    def test_taxonomy_marks_parliament_extension(self):
        row = next(
            item for item in load_taxonomy()["phenomena"]
            if item["id"] == "source_stipulated_outcome_preservation"
        )
        self.assertEqual(row["source"]["type"], "parliament_extension")

    def test_catalog_entry(self):
        item = invariant_by_id()["SOURCE_STIPULATED_OUTCOME_PRESERVATION"]
        self.assertEqual(item.source_type, "parliament_extension")
        self.assertEqual(item.enforcement, "enforced")
        self.assertTrue(item.integrity_layers.production_validator)

    def test_production_extractor_reads_binary_choice(self):
        text = (
            "Thorne faces a binary choice: execute the purge to guarantee "
            "immediate survival, or refrain to preserve invaluable knowledge "
            "at the cost of catastrophic loss of life."
        )
        stipulations = extract_binary_contrast_stipulations([text])
        spans = {item.consequence_span.casefold() for item in stipulations}
        polarities = {item.polarity for item in stipulations}
        self.assertIn("immediate survival", spans)
        self.assertIn("catastrophic loss of life", spans)
        self.assertEqual(polarities, {"BENEFICIAL", "ADVERSE"})

    def test_structured_lane_preserves_both_stakes(self):
        seed = load_seed("source_stipulated_purge_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            source_stipulated_outcome_preservation_holds(
                model,
                survival_effect_id=expected["survival_effect_id"],
                loss_effect_id=expected["loss_effect_id"],
            )
        )
        errors = validate_world_completeness(
            model, action_ids=["A0", "A1"],
        )
        self.assertFalse(
            any("source-stipulated outcome" in error for error in errors),
            errors,
        )

    def test_structured_lane_detects_omission(self):
        seed = load_seed("source_stipulated_purge_structured.yaml")
        model = structured_world_from_seed(seed, world_key="incomplete_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            source_stipulated_outcome_preservation_holds(
                model,
                survival_effect_id=expected["survival_effect_id"],
                loss_effect_id=expected["loss_effect_id"],
            )
        )
        errors = validate_world_completeness(
            model, action_ids=["A0", "A1"],
        )
        stip_errors = [
            error for error in errors if "source-stipulated outcome" in error
        ]
        self.assertGreaterEqual(len(stip_errors), 2, errors)
        blob = " ".join(stip_errors).casefold()
        self.assertIn("immediate survival", blob)
        self.assertIn("catastrophic loss of life", blob)

    def test_grounding_lane_completeness_rejects_omission(self):
        seed = load_seed("source_stipulated_purge_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["incomplete_world"]),
        )
        errors = validate_world_completeness(
            model, action_ids=["A0", "A1"],
        )
        stip_errors = [
            error for error in errors if "source-stipulated outcome" in error
        ]
        self.assertTrue(stip_errors, errors)
        issues = validation_issues_from_messages(stip_errors)
        self.assertTrue(
            any(issue.code == "SOURCE_STIPULATED_OUTCOME_MISSING" for issue in issues)
        )

    def test_grounding_lane_completeness_accepts_both_stakes(self):
        seed = load_seed("source_stipulated_purge_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            source_stipulated_outcome_preservation_holds(
                model,
                survival_effect_id=expected["survival_effect_id"],
                loss_effect_id=expected["loss_effect_id"],
            )
        )
        errors = validate_world_completeness(
            model, action_ids=["A0", "A1"],
        )
        self.assertFalse(
            any("source-stipulated outcome" in error for error in errors),
            errors,
        )

    def test_unresolved_quarantine_satisfies_preservation(self):
        from dataclasses import replace

        seed = load_seed("source_stipulated_purge_structured.yaml")
        model = structured_world_from_seed(seed)
        effects = tuple(
            replace(effect, polarity="UNRESOLVED")
            if effect.effect_id in {"E_surv", "E_loss"}
            else effect
            for effect in model.effects
        )
        model = replace(model, effects=effects)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            source_stipulated_outcome_preservation_holds(
                model,
                survival_effect_id=expected["survival_effect_id"],
                loss_effect_id=expected["loss_effect_id"],
            )
        )
        errors = validate_world_completeness(
            model, action_ids=["A0", "A1"],
        )
        self.assertFalse(
            any("source-stipulated outcome" in error for error in errors),
            errors,
        )

    def test_repair_card_for_missing_stipulation(self):
        messages = [
            "A1 omits source-stipulated outcome 'immediate survival' "
            "from binary contrast; admit a matching effect "
            "(or UNRESOLVED quarantine)"
        ]
        issues = validation_issues_from_messages(messages)
        cards = repair_guidance_cards(
            issues,
            {"world_model": {"effects": [], "actions": [{"action_id": "A1"}]}},
            clauses=[{
                "clause_id": "C3",
                "text": (
                    "Thorne faces a binary choice: execute the purge to "
                    "guarantee immediate survival, or refrain to preserve "
                    "invaluable knowledge at the cost of catastrophic loss "
                    "of life."
                ),
            }],
        )
        self.assertEqual(cards[0]["code"], "SOURCE_STIPULATED_OUTCOME_MISSING")
        patches = cards[0].get("concrete_patches") or []
        self.assertTrue(
            any(patch.get("op") == "add_effect" for patch in patches),
            patches,
        )
        outcomes = [
            (patch.get("value") or {}).get("outcome")
            for patch in patches
            if patch.get("op") == "add_effect"
        ]
        self.assertIn("immediate survival", outcomes)

    def test_map_lists_stipulation_seeds(self):
        entry = next(
            row for row in load_map()["entries"]
            if row["phenomenon_id"] == "source_stipulated_outcome_preservation"
        )
        self.assertEqual(len(entry["seeds"]), 2)
        self.assertEqual(entry["enforcement"], "enforced")


if __name__ == "__main__":
    unittest.main()
