"""Contract tests for the Parliament-owned admission facade."""
from __future__ import annotations

import unittest

from global_workspace.world_admission import (
    admit_world_extension,
    normalize_admitted_world,
    restore_admitted_world,
    serialize_world_for_admission,
    validate_admitted_world,
)
from global_workspace.world_state import (
    CausalLink,
    ScenarioWorldModel,
    SourceRef,
    WorldAction,
    WorldCondition,
    WorldEffect,
    WorldParty,
)


def _conditional_world() -> ScenarioWorldModel:
    ref = (SourceRef("C0", "Act now; patients recover if the backup starts."),)
    effects = (
        WorldEffect(
            "E0", "A0", "P1", "backup started", "STATE_CHANGE",
            "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
            provenance=ref, source_proposition="backup starts",
            derivation_operation="DIRECT_COPY",
        ),
        WorldEffect(
            "E1", "A0", "P2", "patients recover", "RECOVERS",
            "BENEFICIAL", "DOWNSTREAM", "STIPULATED_CONDITIONAL",
            "HEALTH_OUTCOME", condition_ids=("COND0",), provenance=ref,
            source_effect_ids=("E0",),
            source_proposition="patients recover if the backup starts",
            derivation_operation="SOURCE_STIPULATED_CAUSAL",
            derivation_explanation=(
                "The source explicitly conditions patient recovery on the backup."
            ),
        ),
    )
    return ScenarioWorldModel(
        schema_version="1.3",
        parties=(
            WorldParty("P0", "operator", "HUMAN", ref),
            WorldParty("P1", "backup", "PROCESS", ref),
            WorldParty("P2", "patients", "POPULATION", ref),
        ),
        actions=(WorldAction(
            "A0", "start the backup", "P0", ("P1",), ("E0", "E1"), ref,
        ),),
        effects=effects,
        conditions=(WorldCondition(
            "COND0", "the backup starts", provenance=ref,
            event_effect_id="E0",
        ),),
        causal_links=(CausalLink(
            "E0", "CAUSES", "E1", "CERTAIN", (), ref, "A0",
        ),),
    )


class WorldAdmissionFacadeTests(unittest.TestCase):
    def test_restore_accepts_typed_world_without_copying(self):
        world = _conditional_world()
        self.assertIs(restore_admitted_world(world), world)

    def test_normalization_synchronizes_target_gates_without_mutating_input(self):
        world = _conditional_world()
        normalized = normalize_admitted_world(world)
        self.assertEqual(world.causal_links[0].condition_ids, ())
        self.assertEqual(normalized.causal_links[0].condition_ids, ("COND0",))
        self.assertEqual(
            normalized.causal_links[0].modality,
            "STIPULATED_CONDITIONAL",
        )

    def test_validation_phase_is_explicit_and_side_effect_free(self):
        world = normalize_admitted_world(_conditional_world())
        report = validate_admitted_world(
            world, action_ids=("A0",), require_completeness=False,
        )
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.contradictions, ())

    def test_serialization_produces_readmission_payload(self):
        payload = serialize_world_for_admission(_conditional_world())
        self.assertEqual(payload["actions"][0]["effect_ids"], ("E0", "E1"))
        self.assertEqual(payload["conditions"][0]["event_effect_id"], "E0")
        self.assertNotIn("provenance", payload["effects"][0])

    def test_extension_admission_enforces_append_only_identity(self):
        with self.assertRaisesRegex(ValueError, "append-only"):
            admit_world_extension(
                _conditional_world(),
                {"effects": [{"effect_id": "E0"}]},
                clauses=[{
                    "clause_id": "C0",
                    "text": "Act now; patients recover if the backup starts.",
                }],
                action_ids=("A0",),
            )


if __name__ == "__main__":
    unittest.main()
