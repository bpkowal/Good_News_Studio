"""Quantifier party-count Hypothesis cases with declared oracles.

The oracle is QuantifierCountCase.preserves_count (cardinal stays only on
the owning population party). Hypothesis asks the Parliament harness oracle
whether keep/leak worlds agree with that flag.

Do not import production quantity binders here. A generated count that
silently migrates onto a facility is the failure this module exists to find.
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
from strategies.worlds import unique_tokens


_CARDINALS = (
    "thirty",
    "twelve",
    "forty",
    "eighty",
    "three hundred",
)
_POPULATION_NOUNS = (
    "residents",
    "patients",
    "workers",
    "families",
    "households",
)
_FACILITY_NOUNS = (
    "care home",
    "clinic",
    "depot",
    "shelter",
    "ward",
)
_TOKEN_STOP = frozenset({
    "at", "care", "clinic", "depot", "eighty", "families", "forty", "home",
    "households", "hundred", "patients", "protect", "residents", "shelter",
    "stranded", "the", "thirty", "three", "twelve", "ward", "workers",
})


@dataclass(frozen=True, slots=True)
class QuantifierCountCase:
    """One cardinal on a population NP. preserves_count is declared."""

    source: str
    quantity: str
    owner_party_id: str
    leak_party_id: str
    preserves_count: bool
    world: ScenarioWorldModel
    issue_code: str = "QUANTIFIER_PARTY_LEAK"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("REPAIR_GROUNDING", "RESOLVE_ENTITY")
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "COMPARE_FRAMEWORKS")


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _count_world(
    *,
    source: str,
    quantity: str,
    actor: str,
    population: str,
    facility: str,
    preserves_count: bool,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    owner_qty = (quantity,) if preserves_count else ()
    leak_qty = () if preserves_count else (quantity,)
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P_pop", population, "POPULATION", ref, quantities=owner_qty),
            WorldParty("P_fac", facility, "FACILITY", ref, quantities=leak_qty),
            WorldParty("P_actor", actor, "PERSON", ref),
        ),
        actions=(
            WorldAction(
                "A0", f"protect the {facility}",
                "P_actor", ("P_fac",), ("E0",), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P_pop", f"stranded at the {facility}", "STATE_CHANGE",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                provenance=ref,
                source_proposition=source,
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def quantifier_count_cases(draw) -> QuantifierCountCase:
    """Count on population owner vs silent facility leak."""
    actor = _labels(draw, 1)[0]
    quantity = draw(st.sampled_from(_CARDINALS))
    population = draw(st.sampled_from(_POPULATION_NOUNS))
    facility = draw(st.sampled_from(_FACILITY_NOUNS))
    preserves = draw(st.booleans())
    source = f"{quantity.capitalize()} {population} are stranded at the {facility}."
    return QuantifierCountCase(
        source=source,
        quantity=quantity,
        owner_party_id="P_pop",
        leak_party_id="P_fac",
        preserves_count=preserves,
        world=_count_world(
            source=source,
            quantity=quantity,
            actor=actor,
            population=population,
            facility=facility,
            preserves_count=preserves,
        ),
    )
