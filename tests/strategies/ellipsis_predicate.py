"""VP-ellipsis predicate-resolution Hypothesis cases with declared oracles.

The oracle is EllipsisPredicateCase.preserves_predicate (elliptical effect
shares the antecedent outcome on a distinct party).
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
    ("Alex", "Jordan"),
    ("Sam", "Riley"),
    ("Morgan", "Casey"),
)
_PREDICATES = (
    ("rescued the residents", "abandoned the residents"),
    ("evacuated the ward", "sealed the ward"),
    ("secured the clinic", "cleared the clinic"),
)


@dataclass(frozen=True, slots=True)
class EllipsisPredicateCase:
    """Antecedent + elliptical effects. preserves_predicate is declared."""

    ante_source: str
    ellip_source: str
    antecedent_effect_id: str
    elliptical_effect_id: str
    preserves_predicate: bool
    world: ScenarioWorldModel
    issue_code: str = "ELLIPSIS_WRONG_PREDICATE"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("REPAIR_GROUNDING",)
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE",)


def _ellipsis_world(
    *,
    ante_source: str,
    ellip_source: str,
    ante_name: str,
    ellip_name: str,
    ante_outcome: str,
    ellip_outcome: str,
) -> ScenarioWorldModel:
    ante_ref = (SourceRef("C0", ante_source),)
    ellip_ref = (SourceRef("C1", ellip_source),)
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P_ante", ante_name, "PERSON", ante_ref),
            WorldParty("P_ellip", ellip_name, "PERSON", ellip_ref),
            WorldParty("P_group", "residents", "POPULATION", ante_ref),
            WorldParty("P_actor", "coordinator", "PERSON", ante_ref),
        ),
        actions=(
            WorldAction(
                "A0", "record the rescue",
                "P_actor", ("P_group",), ("E_ante", "E_ellip"),
                ante_ref + ellip_ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E_ante", "A0", "P_ante", ante_outcome, "STATE_CHANGE",
                "BENEFICIAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ante_ref,
                source_proposition=ante_source,
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E_ellip", "A0", "P_ellip", ellip_outcome, "STATE_CHANGE",
                "BENEFICIAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ellip_ref,
                source_proposition=ellip_source,
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def ellipsis_predicate_cases(draw) -> EllipsisPredicateCase:
    ante_name, ellip_name = draw(st.sampled_from(_NAME_PAIRS))
    good, bad = draw(st.sampled_from(_PREDICATES))
    preserves = draw(st.booleans())
    ante_source = f"{ante_name} {good}."
    ellip_source = f"So did {ellip_name}."
    ellip_outcome = good if preserves else bad
    return EllipsisPredicateCase(
        ante_source=ante_source,
        ellip_source=ellip_source,
        antecedent_effect_id="E_ante",
        elliptical_effect_id="E_ellip",
        preserves_predicate=preserves,
        world=_ellipsis_world(
            ante_source=ante_source,
            ellip_source=ellip_source,
            ante_name=ante_name,
            ellip_name=ellip_name,
            ante_outcome=good,
            ellip_outcome=ellip_outcome,
        ),
    )
