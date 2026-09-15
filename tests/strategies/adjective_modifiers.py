"""Adjective / stacked-modifier Hypothesis cases with declared oracles.

The oracle is AdjectiveModifierCase.preserves_binding (stacked temporal
modifiers stay on the harm head; sibling process does not inherit them;
Adj+N facility label/kind remain intact).
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from global_workspace.world_state import (
    ScenarioWorldModel,
    SourceRef,
    WorldAction,
    WorldEffect,
    WorldParty,
)


_MODIFIER_PAIRS = (
    ("immediate", "prolonged"),
    ("sudden", "lasting"),
    ("acute", "chronic"),
)
_FACILITY_NOUNS = (
    "care home",
    "clinic",
    "shelter",
)


@dataclass(frozen=True, slots=True)
class AdjectiveModifierCase:
    """Stacked modifiers on harm. preserves_binding is declared."""

    source: str
    effect_id: str
    sibling_effect_id: str
    temporal_qualifiers: tuple[str, str]
    party_id: str
    expected_party_kind: str
    label_must_contain: str
    preserves_binding: bool
    world: ScenarioWorldModel
    issue_code: str = "ADJECTIVE_MODIFIER_LEAK"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("REPAIR_GROUNDING",)
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE",)


def _adjective_world(
    *,
    source: str,
    facility: str,
    left: str,
    right: str,
    preserves_binding: bool,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    owner_temps = (left, right)
    sibling_temps = () if preserves_binding else (left, right)
    facility_kind = "FACILITY" if preserves_binding else "POPULATION"
    facility_label = f"remote {facility}" if preserves_binding else "remote group"
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P_subjects", "subjects", "POPULATION", ref),
            WorldParty("P_home", facility_label, facility_kind, ref),
            WorldParty("P_power", "power", "PROCESS", ref),
            WorldParty("P_actor", "coordinator", "PERSON", ref),
        ),
        actions=(
            WorldAction(
                "A0", "cut power",
                "P_actor", ("P_power",), ("E_process", "E_harm"), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E_process", "A0", "P_power", "power fails", "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                temporal_qualifiers=sibling_temps,
                provenance=ref,
                source_proposition="when power fails",
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E_harm", "A0", "P_subjects", "harm", "STATE_CHANGE",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                temporal_qualifiers=owner_temps,
                provenance=ref,
                source_proposition=source,
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def adjective_modifier_cases(draw) -> AdjectiveModifierCase:
    facility = draw(st.sampled_from(_FACILITY_NOUNS))
    left, right = draw(st.sampled_from(_MODIFIER_PAIRS))
    preserves = draw(st.booleans())
    source = (
        f"Subjects at the remote {facility} suffer {left} and {right} "
        f"harm when power fails."
    )
    return AdjectiveModifierCase(
        source=source,
        effect_id="E_harm",
        sibling_effect_id="E_process",
        temporal_qualifiers=(left, right),
        party_id="P_home",
        expected_party_kind="FACILITY",
        label_must_contain=facility,
        preserves_binding=preserves,
        world=_adjective_world(
            source=source,
            facility=facility,
            left=left,
            right=right,
            preserves_binding=preserves,
        ),
    )
