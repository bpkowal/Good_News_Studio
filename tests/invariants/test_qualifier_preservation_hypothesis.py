"""Qualifier-preservation invariants over Hypothesis-generated worlds.

Oracles are declared case fields. Production is only asked whether
validate_world_model agrees with records_qualifier for the bound channel.
"""
from __future__ import annotations

import unittest
from dataclasses import replace

from hypothesis import given, settings

from global_workspace.world_state import ScenarioWorldModel, validate_world_model
from global_workspace.world_validation import (
    apply_deterministic_local_patches,
    repair_guidance_cards,
    validation_issues_from_messages,
)
from strategies.qualifier_preservation import (
    QualifierPreservationCase,
    qualifier_preservation_cases,
)


def _candidate_from_world(world: ScenarioWorldModel) -> dict[str, object]:
    raw = world.as_dict()
    effects = []
    for row in raw.get("effects") or []:
        item = dict(row)
        item["clause_ids"] = [
            str(ref.get("clause_id") or "")
            for ref in (item.get("provenance") or [])
            if isinstance(ref, dict) and ref.get("clause_id")
        ]
        effects.append(item)
    return {"world_model": {"effects": effects, "schema_version": "1.3"}}


class QualifierPreservationHypothesisTests(unittest.TestCase):
    @given(qualifier_preservation_cases())
    @settings(max_examples=60, deadline=None)
    def test_production_matches_declared_qualifier_oracle(
        self, case: QualifierPreservationCase,
    ):
        errors, _ = validate_world_model(case.world, action_ids=["A0"])
        channel_errors = [
            error for error in errors
            if f"source-grounded {case.channel} qualifiers" in error
        ]
        effect = next(
            item for item in case.world.effects if item.effect_id == case.effect_id
        )
        sibling = next(
            item for item in case.world.effects
            if item.effect_id == case.sibling_effect_id
        )
        recorded = getattr(effect, case.field)
        sibling_recorded = getattr(sibling, case.field)
        if case.records_qualifier:
            self.assertEqual(case.placement, "effect")
            self.assertIn(case.span, recorded)
            self.assertFalse(channel_errors, errors)
        else:
            self.assertFalse(recorded)
            self.assertTrue(channel_errors, errors)
            blob = " ".join(channel_errors).casefold()
            self.assertIn(case.span.casefold(), blob)
            self.assertEqual(case.repair_stage, "grounding")
            self.assertTrue(case.allowed_ops)
            self.assertNotIn("MORE_DEBATE", case.allowed_ops)
            if case.placement == "sibling":
                self.assertIn(case.span, sibling_recorded)
            else:
                self.assertFalse(sibling_recorded)

    @given(
        qualifier_preservation_cases().filter(
            lambda case: case.placement in {"omit", "sibling"},
        ),
    )
    @settings(max_examples=40, deadline=None)
    def test_det_attaches_missing_qualifier_without_sibling_spray(
        self, case: QualifierPreservationCase,
    ):
        errors, _ = validate_world_model(case.world, action_ids=["A0"])
        channel_errors = [
            error for error in errors
            if f"source-grounded {case.channel} qualifiers" in error
        ]
        self.assertTrue(channel_errors, errors)
        candidate = _candidate_from_world(case.world)
        before_sibling = next(
            row for row in candidate["world_model"]["effects"]
            if row["effect_id"] == case.sibling_effect_id
        )
        before_spans = list(before_sibling.get(case.field) or [])
        patched, applied = apply_deterministic_local_patches(
            candidate,
            validation_issues_from_messages(channel_errors),
        )
        self.assertTrue(applied, (case.placement, case.channel, channel_errors))
        self.assertEqual(
            {row.get("effect_id") for row in applied},
            {case.effect_id},
            applied,
        )
        target = next(
            row for row in patched["world_model"]["effects"]
            if row["effect_id"] == case.effect_id
        )
        sibling = next(
            row for row in patched["world_model"]["effects"]
            if row["effect_id"] == case.sibling_effect_id
        )
        self.assertIn(case.span, target.get(case.field) or [])
        self.assertEqual(list(sibling.get(case.field) or []), before_spans)
        # Typed world after DET should clear the channel error.
        effect = next(e for e in case.world.effects if e.effect_id == case.effect_id)
        repaired_effect = replace(effect, **{case.field: (case.span,)})
        if case.field == "likelihood_qualifiers":
            repaired_effect = replace(repaired_effect, modality="PROBABILISTIC")
        repaired_world = replace(
            case.world,
            effects=tuple(
                repaired_effect if e.effect_id == case.effect_id else e
                for e in case.world.effects
            ),
        )
        remaining, _ = validate_world_model(repaired_world, action_ids=["A0"])
        still = [
            error for error in remaining
            if f"source-grounded {case.channel} qualifiers" in error
        ]
        self.assertFalse(still, still)


class QualifierPreservationRepairTests(unittest.TestCase):
    def test_deterministic_patch_attaches_missing_likelihood(self):
        messages = [
            "E_loss omits source-grounded likelihood qualifiers: "
            "['near-certain']"
        ]
        issues = validation_issues_from_messages(messages)
        self.assertEqual(issues[0].code, "LIKELIHOOD_QUALIFIER_MISSING")
        candidate = {
            "world_model": {
                "effects": [{
                    "effect_id": "E_loss",
                    "action_id": "A0",
                    "outcome": "catastrophic loss of life",
                    "modality": "CERTAIN",
                    "likelihood_qualifiers": [],
                }],
            },
        }
        cards = repair_guidance_cards(issues, candidate)
        self.assertEqual(cards[0]["code"], "LIKELIHOOD_QUALIFIER_MISSING")
        patched, applied = apply_deterministic_local_patches(candidate, issues)
        self.assertEqual(len(applied), 1, applied)
        effect = patched["world_model"]["effects"][0]
        self.assertIn("near-certain", effect["likelihood_qualifiers"])
        self.assertEqual(effect["modality"], "PROBABILISTIC")


if __name__ == "__main__":
    unittest.main()
