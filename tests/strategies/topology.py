"""Action-mediated vs exogenous process, with a declared admit/reject oracle.

The oracle is TopologyCase.should_admit, not the production ancestry walk.
Production is only asked whether that declared topology is complete.

Do not import live-dilemma nouns here. Exogeneity is the source's causal
structure, not a likelihood hedge: a hedged collapse may still be
action-caused. The independent row is a background process failure.
CONJUNCTIVE gates the mediated path with condition.event_effect_id; the
independent event is not a second ordinary causal parent.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from global_workspace.world_state import (
    CausalLink,
    ScenarioWorldModel,
    SourceRef,
    WorldAction,
    WorldCondition,
    WorldEffect,
    WorldParty,
)
from strategies.worlds import unique_tokens


# CONJUNCTIVE, MEDIATED_ONLY, and ACTION_CAUSED must admit.
TOPOLOGY_KINDS = (
    "CONJUNCTIVE",
    "MEDIATED_ONLY",
    "ACTION_CAUSED",
    "EXOGENOUS_ONLY",
    "FALSE_CAUSE",
    "LINEAR_STOCHASTIC",
    "ACTION_CAUSED_UNLINKED",
)
_ACTION_CAUSED_KINDS = frozenset({"ACTION_CAUSED", "ACTION_CAUSED_UNLINKED"})
_POPULATION_NOUNS = ("residents", "patients", "families")
_TOKEN_STOP = frozenset({
    "about", "bring", "brings", "cause", "caused", "causes", "causing",
    "chance", "certain", "collapse", "collapses", "death", "face", "fail",
    "fails", "held", "holding", "lead", "leads", "near", "perform",
    "performing", "process", "result", "results", "risk", "state",
})


@dataclass(frozen=True, slots=True)
class TopologyCase:
    """One shared clause. should_admit is written by the strategy."""

    kind: str
    source: str
    world: ScenarioWorldModel

    @property
    def should_admit(self) -> bool:
        return self.kind in {"CONJUNCTIVE", "MEDIATED_ONLY", "ACTION_CAUSED"}


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _independent_links(kind: str, ref: tuple[SourceRef, ...]) -> tuple[CausalLink, ...]:
    mediated = CausalLink(
        "E0", "ENABLES", "E1", "CERTAIN", provenance=ref, action_id="A0",
    )
    mediated_to_harm = CausalLink(
        "E1", "CAUSES", "E3", "CERTAIN", provenance=ref, action_id="A0",
    )
    exogenous_to_harm = CausalLink(
        "E2", "CAUSES", "E3", "CERTAIN", provenance=ref, action_id="A0",
    )
    if kind == "CONJUNCTIVE":
        return (mediated, mediated_to_harm)
    if kind == "MEDIATED_ONLY":
        return (mediated, mediated_to_harm)
    if kind == "EXOGENOUS_ONLY":
        return (exogenous_to_harm,)
    if kind == "FALSE_CAUSE":
        return (
            CausalLink(
                "E0", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0",
            ),
            exogenous_to_harm,
        )
    return (
        mediated,
        CausalLink(
            "E1", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0",
        ),
        exogenous_to_harm,
    )


def _independent_world(
    *,
    kind: str,
    source: str,
    actor: str,
    facility: str,
    label: str,
    verb: str,
    process: str,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    gated = kind == "CONJUNCTIVE"
    harm_modality = "STIPULATED_CONDITIONAL" if gated else "POSSIBLE"
    harm_conditions = ("COND1",) if gated else ()
    conditions = (
        (
            WorldCondition(
                "COND1",
                f"the {process} fails",
                provenance=ref,
                event_effect_id="E2",
            ),
        )
        if gated else ()
    )
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "FACILITY", ref),
            WorldParty("P2", label, "POPULATION", ref),
            WorldParty("P3", process, "PROCESS", ref),
        ),
        actions=(
            WorldAction(
                "A0", f"perform {verb} on the {facility}",
                "P0", ("P1",), ("E0", "E1", "E2", "E3"), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P1", f"{verb} on the {facility}", "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ref,
            ),
            WorldEffect(
                "E1", "A0", "P1", f"{facility} held", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                provenance=ref,
            ),
            WorldEffect(
                "E2", "A0", "P3", "fails", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
                likelihood_qualifiers=("20% chance",),
                provenance=ref,
            ),
            WorldEffect(
                "E3", "A0", "P2", f"{label} death", "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", harm_modality, "HEALTH_OUTCOME",
                condition_ids=harm_conditions,
                likelihood_qualifiers=("near-certain",),
                provenance=ref,
            ),
        ),
        conditions=conditions,
        causal_links=_independent_links(kind, ref),
    )


def _action_caused_world(
    *,
    kind: str,
    source: str,
    actor: str,
    facility: str,
    label: str,
    verb: str,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    caused = CausalLink(
        "E0", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0",
    )
    harm = CausalLink(
        "E2", "CAUSES", "E3", "CERTAIN", provenance=ref, action_id="A0",
    )
    links = (caused, harm) if kind == "ACTION_CAUSED" else (harm,)
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "FACILITY", ref),
            WorldParty("P2", label, "POPULATION", ref),
        ),
        actions=(
            WorldAction(
                "A0", f"perform {verb} on the {facility}",
                "P0", ("P1",), ("E0", "E2", "E3"), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P1", f"{verb} on the {facility}", "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ref,
            ),
            WorldEffect(
                "E2", "A0", "P1", f"{facility} collapse", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
                likelihood_qualifiers=("20% chance",),
                provenance=ref,
            ),
            WorldEffect(
                "E3", "A0", "P2", f"{label} death", "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "POSSIBLE", "HEALTH_OUTCOME",
                likelihood_qualifiers=("near-certain",),
                provenance=ref,
            ),
        ),
        causal_links=links,
    )


@st.composite
def topology_cases(draw) -> TopologyCase:
    actor, facility, group, verb, process = _labels(draw, 5)
    kind = draw(st.sampled_from(TOPOLOGY_KINDS))
    noun = draw(st.sampled_from(_POPULATION_NOUNS))
    label = f"{group} {noun}"
    if kind in _ACTION_CAUSED_KINDS:
        source = (
            f"{actor} performs {verb} on the {facility}. There is a 20% chance "
            f"that performing {verb} causes the {facility} to collapse. The "
            f"{label} face near-certain death."
        )
        world = _action_caused_world(
            kind=kind, source=source, actor=actor, facility=facility,
            label=label, verb=verb,
        )
    else:
        source = (
            f"{actor} performs {verb} on the {facility}. The {facility} is held. "
            f"There is a 20% chance the {process} fails. The {label} face "
            "near-certain death."
        )
        world = _independent_world(
            kind=kind, source=source, actor=actor, facility=facility,
            label=label, verb=verb, process=process,
        )
    return TopologyCase(kind=kind, source=source, world=world)
