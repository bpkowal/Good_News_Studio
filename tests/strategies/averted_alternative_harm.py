"""Averted-alternative-harm Hypothesis cases with declared oracles.

Quantity may sit only on the CERTAIN ADVERSE alternative. Copying that span
onto the survival row as DIRECT_COPY source support is forbidden. A licensed
derived claim uses AVERTED_ALTERNATIVE_HARM + counterfactual edge +
source_effect_ids pointing at the adverse effect.

Oracles are declared case fields. Distinct from AVERTED_RISK_IS_NOT_OBTAINED_BENEFIT
(unsettled opposed harm).
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from global_workspace.world_state import (
    CounterfactualLink,
    ScenarioWorldModel,
    SourceRef,
    WorldAction,
    WorldEffect,
    WorldParty,
)
from strategies.worlds import unique_tokens


_LIFE_QUANTITIES = ("thousands", "hundreds", "millions")
_MUTATIONS = (
    "adverse_only",
    "silent_copy_survival",
    "derived_averted",
)
_TOKEN_STOP = frozenset({
    "averts", "catastrophic", "cyberattack", "immediate", "life", "lives",
    "loss", "purge", "survival", "thousands",
})


@dataclass(frozen=True, slots=True)
class AvertedAlternativeHarmCase:
    """Paired survival/death worlds. Oracles declare copy vs derivation."""

    life_source: str
    choice_source: str
    life_quantity: str
    mutation: str
    survival_effect_id: str
    death_effect_id: str
    permits_silent_source_copy: bool
    expects_licensed_averted_quantity: bool
    world: ScenarioWorldModel
    issue_code: str = "AVERTED_ALTERNATIVE_QUANTITY_UNLICENSED"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = (
        "add_averted_alternative_harm",
        "add_counterfactual_link",
    )
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "COMPARE_FRAMEWORKS")


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _averted_world(
    *,
    life_source: str,
    choice_source: str,
    actor: str,
    facility: str,
    life_quantity: str,
    mutation: str,
) -> ScenarioWorldModel:
    life_ref = (SourceRef("C0", life_source),)
    choice_ref = (SourceRef("C3", choice_source),)
    survival_qty: tuple[str, ...] = ()
    death_qty = (life_quantity,)
    derived: list[WorldEffect] = []
    links: list[CounterfactualLink] = []
    effect_ids_a0 = ["E_surv"]
    if mutation == "silent_copy_survival":
        # Wrong: pretends the survival clause / C3 states the magnitude.
        survival_qty = (life_quantity,)
    elif mutation == "derived_averted":
        derived.append(WorldEffect(
            "E_avert", "A0", "P2",
            f"averts catastrophic loss of {life_quantity} of lives",
            "STATE_CHANGE",
            "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
            quantities=death_qty,
            provenance=(),
            source_proposition=life_source,
            source_effect_ids=("E_death",),
            derivation_operation="AVERTED_ALTERNATIVE_HARM",
            derivation_explanation=(
                "A0 precludes CERTAIN alternative harm E_death; "
                "quantity inherited from the opposed adverse effect"
            ),
        ))
        effect_ids_a0.append("E_avert")
        links.append(CounterfactualLink(
            "A0",
            "E_avert",
            "PRECLUDES_ALTERNATIVE_EFFECT",
            "A1",
            "E_death",
            "CERTAIN",
            (),
            life_ref,
        ))
    return ScenarioWorldModel(
        schema_version="1.3",
        parties=(
            WorldParty("P0", actor, "HUMAN", choice_ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", choice_ref),
            WorldParty("P2", "city populace", "POPULATION", life_ref),
        ),
        actions=(
            WorldAction(
                "A0", "execute emergency purge",
                "P0", ("P1", "P2"), tuple(effect_ids_a0), choice_ref + life_ref,
            ),
            WorldAction(
                "A1", "withhold emergency purge",
                "P0", ("P2",), ("E_death",), choice_ref + life_ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E_surv", "A0", "P2", "immediate survival guaranteed",
                "STATE_CHANGE",
                "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                quantities=survival_qty,
                provenance=choice_ref,
                source_proposition="immediate survival guaranteed",
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E_death", "A1", "P2", "catastrophic loss of life", "DIES",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                quantities=death_qty,
                provenance=life_ref,
                source_proposition=life_source,
                derivation_operation="DIRECT_COPY",
            ),
            *derived,
        ),
        counterfactual_links=tuple(links),
    )


@st.composite
def averted_alternative_harm_cases(draw) -> AvertedAlternativeHarmCase:
    """Survival/death pairs with declared copy vs derivation oracles."""
    actor, facility = _labels(draw, 2)
    life_quantity = draw(st.sampled_from(_LIFE_QUANTITIES))
    mutation = draw(st.sampled_from(_MUTATIONS))
    life_source = (
        f"A cyberattack on the {facility} risks {life_quantity} of lives "
        f"through imminent infrastructure failure."
    )
    choice_source = (
        "binary choice: execute the purge for immediate survival, or "
        "withhold at the cost of catastrophic loss of life"
    )
    return AvertedAlternativeHarmCase(
        life_source=life_source,
        choice_source=choice_source,
        life_quantity=life_quantity,
        mutation=mutation,
        survival_effect_id="E_surv",
        death_effect_id="E_death",
        permits_silent_source_copy=False,
        expects_licensed_averted_quantity=(mutation == "derived_averted"),
        world=_averted_world(
            life_source=life_source,
            choice_source=choice_source,
            actor=actor,
            facility=facility,
            life_quantity=life_quantity,
            mutation=mutation,
        ),
    )
