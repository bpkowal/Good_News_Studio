"""At-risk situation spans with a declared typing oracle.

The oracle is RiskSituationCase.expects_certain_exposure, not production
modality. 'at moderate risk' on an exposure row is a stipulated situation.
'at high risk of dying' is a chance hedge on that harm.

Do not import production hedge lists here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from hypothesis import strategies as st

from global_workspace.world_state import (
    CausalLink,
    ScenarioWorldModel,
    SourceRef,
    WorldAction,
    WorldEffect,
    WorldParty,
)
from strategies.worlds import unique_tokens


# Independent of world_state._AT_RISK_MAGNITUDES. Situation families first.
RISK_SITUATIONS = (
    ("at risk", "SITUATION"),
    ("at moderate risk", "SITUATION"),
    ("at high risk", "HARM"),
)
_POPULATION_NOUNS = ("residents", "patients", "families")
_TOKEN_STOP = frozenset({
    "at", "chance", "death", "dying", "exposed", "face", "high", "moderate",
    "remain", "risk", "stay",
})


@dataclass(frozen=True, slots=True)
class RiskSituationCase:
    """One clause. expects_certain_exposure is written by the strategy."""

    source: str
    span: str
    family: str
    world: ScenarioWorldModel
    owner_id: str
    sibling_id: str

    @property
    def expects_certain_exposure(self) -> bool:
        return self.family == "SITUATION"

    @property
    def owner(self) -> WorldEffect:
        return next(
            effect for effect in self.world.effects
            if effect.effect_id == self.owner_id
        )

    @property
    def sibling(self) -> WorldEffect:
        return next(
            effect for effect in self.world.effects
            if effect.effect_id == self.sibling_id
        )

    @property
    def owner_without_copy(self) -> WorldEffect:
        return replace(self.owner, likelihood_qualifiers=())


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


@st.composite
def risk_situation_cases(draw) -> RiskSituationCase:
    actor, facility, group, verb = _labels(draw, 4)
    span, family = draw(st.sampled_from(RISK_SITUATIONS))
    noun = draw(st.sampled_from(_POPULATION_NOUNS))
    label = f"{group} {noun}"
    if family == "HARM":
        source = (
            f"{actor} performs {verb} on the {facility}. "
            f"The {label} are {span} of dying."
        )
        owner_outcome = f"{label} death"
        owner_modality = "POSSIBLE"
        owner_kind = "HEALTH_OUTCOME"
    else:
        source = (
            f"{actor} performs {verb} on the {facility}. "
            f"The {label} remain {span}."
        )
        owner_outcome = "exposed"
        owner_modality = "CERTAIN"
        owner_kind = "HEALTH_OUTCOME"
    ref = (SourceRef("C0", source),)
    sibling = WorldEffect(
        "E1", "A0", "P1", f"{facility} held", "STATE_CHANGE",
        "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
        provenance=ref,
    )
    owner = WorldEffect(
        "E2", "A0", "P2", owner_outcome, "EXPERIENCES",
        "ADVERSE", "DOWNSTREAM", owner_modality, owner_kind,
        likelihood_qualifiers=(span,),
        provenance=ref,
    )
    world = ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "FACILITY", ref),
            WorldParty("P2", label, "POPULATION", ref),
        ),
        actions=(
            WorldAction(
                "A0", f"perform {verb} on the {facility}",
                "P0", ("P1",), ("E0", "E1", "E2"), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P1", f"{verb} on the {facility}", "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ref,
            ),
            sibling,
            owner,
        ),
        causal_links=(
            CausalLink(
                "E0", "ENABLES", "E1", "CERTAIN", provenance=ref, action_id="A0",
            ),
            CausalLink(
                "E1", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0",
            ),
        ),
    )
    return RiskSituationCase(
        source=source,
        span=span,
        family=family,
        world=world,
        owner_id="E2",
        sibling_id="E1",
    )
