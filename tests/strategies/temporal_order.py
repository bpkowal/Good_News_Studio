"""Temporal before/after order Hypothesis cases with declared oracles.

The oracle is TemporalOrderCase.preserves_order (later event carries the
order marker; earlier event does not).
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


_NAME_PAIRS = (
    ("Jones", "Smith"),
    ("Lee", "Chen"),
    ("Park", "Ng"),
)
_MARKERS = ("after", "before")


@dataclass(frozen=True, slots=True)
class TemporalOrderCase:
    """Earlier/later events. preserves_order is declared."""

    source: str
    earlier_effect_id: str
    later_effect_id: str
    later_marker: str
    preserves_order: bool
    world: ScenarioWorldModel
    issue_code: str = "TEMPORAL_ORDER_SWAPPED"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("REPAIR_GROUNDING",)
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE",)


def _temporal_world(
    *,
    earlier_name: str,
    later_name: str,
    marker: str,
    preserves_order: bool,
) -> ScenarioWorldModel:
    earlier_source = f"{earlier_name} left."
    later_source = f"{later_name} left {marker} {earlier_name} left."
    source = f"{earlier_source} {later_source}"
    earlier_ref = (SourceRef("C0", earlier_source),)
    later_ref = (SourceRef("C1", later_source),)
    if preserves_order:
        earlier_temps: tuple[str, ...] = ()
        later_temps = (marker,)
        earlier_prop = earlier_source
        later_prop = later_source
    else:
        # Marker wrongly lands on the earlier event.
        earlier_temps = (marker,)
        later_temps = ()
        earlier_prop = f"{earlier_name} left {marker}."
        later_prop = f"{later_name} left."
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P_earlier", earlier_name, "PERSON", earlier_ref),
            WorldParty("P_later", later_name, "PERSON", later_ref),
            WorldParty("P_actor", "coordinator", "PERSON", earlier_ref),
        ),
        actions=(
            WorldAction(
                "A0", "record departures",
                "P_actor", ("P_earlier", "P_later"), ("E_earlier", "E_later"),
                earlier_ref + later_ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E_earlier", "A0", "P_earlier", "left", "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "PHYSICAL_STATE",
                temporal_qualifiers=earlier_temps,
                provenance=earlier_ref,
                source_proposition=earlier_prop,
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E_later", "A0", "P_later", "left", "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "PHYSICAL_STATE",
                temporal_qualifiers=later_temps,
                provenance=later_ref,
                source_proposition=later_prop,
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def temporal_order_cases(draw) -> TemporalOrderCase:
    earlier_name, later_name = draw(st.sampled_from(_NAME_PAIRS))
    marker = draw(st.sampled_from(_MARKERS))
    preserves = draw(st.booleans())
    source = (
        f"{earlier_name} left. {later_name} left {marker} {earlier_name} left."
    )
    return TemporalOrderCase(
        source=source,
        earlier_effect_id="E_earlier",
        later_effect_id="E_later",
        later_marker=marker,
        preserves_order=preserves,
        world=_temporal_world(
            earlier_name=earlier_name,
            later_name=later_name,
            marker=marker,
            preserves_order=preserves,
        ),
    )
