from __future__ import annotations

from dataclasses import replace
import unittest

from global_workspace.world_state import (
    CausalLink,
    ScenarioWorldModel,
    WorldAction,
    WorldCompilationStage,
    WorldEffect,
    WorldParty,
    _completion_errors_with_compiler_witnesses,
    _exclusive_allocation_nonreceipt_errors,
    world_compilation_loss_telemetry,
)
from global_workspace.world_validation import (
    format_repair_guidance_for_prompt,
    repair_guidance_cards,
    validation_issues_from_messages,
)


SOURCE = [
    "There is one dose of medicine.",
    "The doctor must give the dose to Maria or to David.",
    "Maria will die without it. David will die without it.",
]


def effect(
    effect_id: str, action_id: str, party_id: str, outcome: str,
    *, polarity: str = "NEUTRAL", kind: str = "PHYSICAL_STATE",
    directness: str = "DOWNSTREAM", operation: str = "DIRECT_COPY",
) -> WorldEffect:
    return WorldEffect(
        effect_id, action_id, party_id, outcome, "STATE", polarity,
        directness, "CERTAIN", kind,
        source_proposition=outcome,
        derivation_operation=operation,
    )


def allocation_world(*, a0_parent_party: str = "P2", include_a0_nonreceipt: bool = True) -> ScenarioWorldModel:
    effects = [
        effect("A0_GIVE", "A0", "P1", "David receives the medicine", kind="RESOURCE_TRANSFER", directness="DIRECT"),
        effect("A0_DEATH", "A0", "P2", "Maria dies", polarity="ADVERSE", kind="HEALTH_OUTCOME"),
        effect("A1_GIVE", "A1", "P2", "Maria receives the medicine", kind="RESOURCE_TRANSFER", directness="DIRECT"),
        effect("A1_NORECV", "A1", "P1", "David does not receive the medicine", operation="EXCLUSIVE_ALLOCATION_COMPLEMENT"),
        effect("A1_DEATH", "A1", "P1", "David dies", polarity="ADVERSE", kind="HEALTH_OUTCOME"),
    ]
    links = [
        CausalLink("A1_NORECV", "CAUSES", "A1_DEATH", "CERTAIN", action_id="A1"),
    ]
    if include_a0_nonreceipt:
        effects.append(effect(
            "A0_NORECV", "A0", a0_parent_party,
            "Maria does not receive the medicine",
            operation="EXCLUSIVE_ALLOCATION_COMPLEMENT",
        ))
        links.append(CausalLink("A0_NORECV", "CAUSES", "A0_DEATH", "CERTAIN", action_id="A0"))
    else:
        links.append(CausalLink("A0_GIVE", "CAUSES", "A0_DEATH", "CERTAIN", action_id="A0"))
    effect_ids = {action: tuple(row.effect_id for row in effects if row.action_id == action) for action in ("A0", "A1")}
    return ScenarioWorldModel(
        parties=(
            WorldParty("P0", "doctor", "PERSON"),
            WorldParty("P1", "David", "PERSON"),
            WorldParty("P2", "Maria", "PERSON"),
        ),
        actions=(
            WorldAction("A0", "give the medicine to David", "P0", ("P1",), effect_ids["A0"]),
            WorldAction("A1", "give the medicine to Maria", "P0", ("P2",), effect_ids["A1"]),
        ),
        effects=tuple(effects), causal_links=tuple(links), schema_version="1.3",
    )


class ValidatorWitnessForensicsTests(unittest.TestCase):
    def test_same_party_nonreceipt_immediately_parenting_death_passes(self):
        self.assertEqual(_exclusive_allocation_nonreceipt_errors(allocation_world(), SOURCE), [])

    def test_receipt_directly_parenting_other_partys_death_fails(self):
        errors = _exclusive_allocation_nonreceipt_errors(
            allocation_world(include_a0_nonreceipt=False), SOURCE,
        )
        self.assertTrue(any("A0 omits exclusive-allocation branch" in row for row in errors), errors)
        self.assertTrue(any("A0_DEATH is harm" in row for row in errors), errors)

    def test_wrong_party_nonreceipt_does_not_satisfy_death_parent(self):
        errors = _exclusive_allocation_nonreceipt_errors(
            allocation_world(a0_parent_party="P1"), SOURCE,
        )
        self.assertTrue(any("nonrecipient P2" in row for row in errors), errors)
        self.assertTrue(any("A0_DEATH is harm" in row for row in errors), errors)

    def test_compiler_removal_is_named_in_invariant_witness(self):
        before = allocation_world()
        before = replace(before, effects=tuple(
            replace(row, derivation_operation="SOURCE_STIPULATED_CAUSAL")
            if row.effect_id == "A0_NORECV" else row
            for row in before.effects
        ))
        after = replace(
            before,
            effects=tuple(row for row in before.effects if row.effect_id != "A0_NORECV"),
            causal_links=(
                CausalLink("A0_GIVE", "CAUSES", "A0_DEATH", "CERTAIN", action_id="A0"),
                *tuple(row for row in before.causal_links if row.action_id == "A1"),
            ),
        )
        stages = (WorldCompilationStage(
            "SOURCE_BINDING", "before", "after", len(after.effects),
            len(after.causal_links), 0, removed_effect_ids=("A0_NORECV",),
        ),)
        errors = _completion_errors_with_compiler_witnesses(
            ["A0 omits exclusive-allocation branch for nonrecipient P2: missing nonreceipt state"],
            before=before, after=after, stages=stages,
        )
        self.assertIn("A0_NORECV derivation contract operation mismatch", errors[0])
        self.assertIn("semantically valid nonreceipt effect", errors[0])
        self.assertIn("during SOURCE_BINDING", errors[0])
        self.assertIn("SOURCE_STIPULATED_CAUSAL", errors[0])
        self.assertIn("EXCLUSIVE_ALLOCATION_COMPLEMENT", errors[0])

        issues = validation_issues_from_messages(errors)
        self.assertEqual(
            issues[0].code, "DERIVATION_CONTRACT_OPERATION_MISMATCH",
        )
        self.assertEqual(issues[0].entity_id, "A0_NORECV")
        self.assertEqual(issues[0].entity_kind, "effect")
        cards = repair_guidance_cards(
            issues,
            {"world_model": before.as_dict()},
            clauses=[
                {"clause_id": "C0", "text": "There is one dose of medicine."},
                {"clause_id": "C1", "text": "Maria will die without it."},
            ],
        )
        patch = cards[0]["concrete_patches"][0]
        self.assertEqual(patch["op"], "repair_derivation_contract_metadata")
        self.assertEqual(
            patch["set"]["derivation_operation"],
            "EXCLUSIVE_ALLOCATION_COMPLEMENT",
        )
        self.assertEqual(
            patch["set"]["source_proposition"],
            "There is one dose of medicine.",
        )
        self.assertEqual(patch["set"]["clause_ids"], ["C0"])
        self.assertEqual(patch["set"]["quantities"], ["one"])
        self.assertIn("causal_links", patch["preserve"])
        self.assertFalse(cards[0]["permits_removal"])
        prompt = format_repair_guidance_for_prompt(cards)
        self.assertIn("preserve node A0_NORECV and every causal link", prompt)
        self.assertNotIn("restore the branch", prompt)

        telemetry = world_compilation_loss_telemetry(before, after, stages)
        self.assertEqual(telemetry["candidate_effect_count"], 6)
        self.assertEqual(telemetry["compiled_effect_count"], 5)
        self.assertEqual(telemetry["compiler_loss_count"], 1)
        self.assertEqual(
            telemetry["loss_reason_counts"],
            {"INVALID_DERIVATION_OPERATION": 1},
        )
        self.assertEqual(telemetry["losses"][0]["failure_class"], "ENCODING_FAILURE")

    def test_metadata_card_does_not_invent_exclusivity_provenance(self):
        message = (
            "E9 derivation contract operation mismatch: semantically valid "
            "nonreceipt effect was removed because derivation_operation="
            "SOURCE_STIPULATED_CAUSAL"
        )
        cards = repair_guidance_cards(
            validation_issues_from_messages([message]),
            {"world_model": {"effects": [{
                "effect_id": "E9",
                "derivation_operation": "SOURCE_STIPULATED_CAUSAL",
                "outcome": "Lee does not receive a dose",
            }]}},
            clauses=[{
                "clause_id": "C0",
                "text": "Several doses may be available from another clinic.",
            }],
        )
        values = cards[0]["concrete_patches"][0]["set"]
        self.assertEqual(values.get("clause_ids"), None)
        self.assertEqual(values.get("quantities"), None)
        self.assertEqual(values.get("source_proposition"), None)
        self.assertEqual(
            cards[0]["concrete_patches"][0]["op"],
            "quarantine_unlicensed_allocation_complement",
        )


if __name__ == "__main__":
    unittest.main()
