from __future__ import annotations

from dataclasses import replace
import unittest

from global_workspace.world_state import (
    CausalLink,
    ScenarioWorldModel,
    SourceRef,
    WorldAction,
    WorldEffect,
    WorldParty,
    WorldStateAdmission,
    quarantine_unsupported_comparative_causalizations,
)


def comparative_world(source: str, *, linked: bool = True) -> ScenarioWorldModel:
    effects = (
        WorldEffect(
            "E0", "A0", "P1", "Maria receives medicine", "STATE",
            "NEUTRAL", "DIRECT", "CERTAIN", "RESOURCE_TRANSFER",
            provenance=(SourceRef("C0", "Maria receives medicine"),),
            source_proposition="Maria receives medicine",
            derivation_operation="DIRECT_COPY",
        ),
        WorldEffect(
            "E1", "A0", "P1", "Maria has a slightly better chance of survival",
            "STATE", "BENEFICIAL", "DOWNSTREAM", "PROBABILISTIC",
            "HEALTH_OUTCOME", provenance=(SourceRef("C1", source),),
            source_proposition=source,
            source_effect_ids=("E0",) if linked else (),
            derivation_operation="SOURCE_STIPULATED_CAUSAL",
        ),
    )
    return ScenarioWorldModel(
        parties=(WorldParty("P1", "Maria", "PERSON"),),
        actions=(WorldAction("A0", "give medicine to Maria", effect_ids=("E0", "E1")),),
        effects=effects,
        causal_links=(
            (CausalLink("E0", "CAUSES", "E1", "CERTAIN", action_id="A0"),)
            if linked else ()
        ),
        admission=WorldStateAdmission(
            status="COMMITTED", admitted_effect_ids=("E0", "E1"),
        ),
        schema_version="1.3",
    )


class ImplicitComparativeQuarantineTests(unittest.TestCase):
    def test_unsupported_action_causalization_is_quarantined(self) -> None:
        model = quarantine_unsupported_comparative_causalizations(
            comparative_world("Maria has a slightly better chance of survival."),
        )
        self.assertEqual(model.admission.status, "COMMITTED_WITH_QUARANTINE")
        self.assertEqual(model.admission.admitted_effect_ids, ("E0",))
        self.assertEqual(
            model.admission.quarantined_effects[0].contradiction_type,
            "IMPLICIT_COMPARATIVE_CAUSALIZATION_UNRESOLVED",
        )
        self.assertEqual([row.effect_id for row in model.effects_for("A0")], ["E0"])
        self.assertEqual(len(model.effects), 2, "quarantine must preserve audit evidence")

    def test_explicit_comparison_standard_is_not_quarantined(self) -> None:
        model = quarantine_unsupported_comparative_causalizations(
            comparative_world(
                "Maria has a slightly better chance of survival than David."
            ),
        )
        self.assertEqual(model.admission.status, "COMMITTED")

    def test_source_action_conditioning_is_not_mistaken_for_invention(self) -> None:
        model = quarantine_unsupported_comparative_causalizations(
            comparative_world(
                "If Maria receives the medicine, she has a better chance of survival."
            ),
        )
        self.assertEqual(model.admission.status, "COMMITTED")

    def test_contextual_comparative_without_added_causal_edge_is_preserved(self) -> None:
        model = quarantine_unsupported_comparative_causalizations(
            comparative_world(
                "Maria has a slightly better chance of survival.", linked=False,
            ),
        )
        self.assertEqual(model.admission.status, "COMMITTED")


if __name__ == "__main__":
    unittest.main()
