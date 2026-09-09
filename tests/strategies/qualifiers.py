"""One clause, two effects, one qualifier, a declared owner.

The oracle is QualifierHeadCase.owner_id, not effect_expected_qualifiers.
Production is only asked whether that span binds to the owner, not the
sibling process or welfare row that cites the same clause.

Do not import production hedge lists here. A generated attachment that the
binder still copies onto the sibling is the failure this module exists to find.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

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


# Independent of world_state extractors. Chance-complement families first so
# shrinking reports the verb-only process owner, then attributive hedges,
# then scope/time owners on the process conjunct.
# kind: LIKELIHOOD / SCOPE / TEMPORAL. family selects the clause template.
HEAD_QUALIFIERS = (
    ("LIKELIHOOD", "20% chance", "PERCENT_THE"),
    ("LIKELIHOOD", "10% chance", "PERCENT_OF"),
    ("LIKELIHOOD", "10% chance", "PERCENT_OF_NOUN"),
    ("LIKELIHOOD", "near-certain", "ADJECTIVE"),
    ("LIKELIHOOD", "a chance", "CHANCE"),
    ("LIKELIHOOD", "could", "MODAL"),
    ("LIKELIHOOD", "likely", "POST"),
    ("SCOPE", "widespread", "SCOPE"),
    ("SCOPE", "localized", "SCOPE"),
    ("TEMPORAL", "immediate", "TEMPORAL"),
    ("TEMPORAL", "prolonged", "TEMPORAL"),
)
_POPULATION_NOUNS = ("residents", "patients", "families")
_KIND_FIELD = {
    "LIKELIHOOD": "likelihood_qualifiers",
    "SCOPE": "scope_qualifiers",
    "TEMPORAL": "temporal_qualifiers",
}
_PERCENT_FAMILIES = frozenset({"PERCENT_THE", "PERCENT_OF", "PERCENT_OF_NOUN"})
_TOKEN_STOP = frozenset({
    "about", "across", "almost", "allows", "being", "blackouts", "blockage",
    "blocked",
    "chance", "certain", "choice", "contaminate", "could", "death", "escape",
    "face", "faces", "fail", "fails", "grid", "harm", "held", "holding",
    "immediate", "keeping", "leak", "likely", "localized", "lose", "lost",
    "might", "mine", "near", "open", "power", "prolonged", "remain",
    "rolling", "route", "spared", "sparing", "spread", "spreads", "state",
    "supply", "triggers", "used", "while", "widespread",
})


@dataclass(frozen=True, slots=True)
class QualifierHeadCase:
    """One shared clause. owner_id is written by the strategy."""

    source: str
    kind: str
    span: str
    family: str
    owner_id: str
    sibling_id: str
    world: ScenarioWorldModel

    @property
    def field(self) -> str:
        return _KIND_FIELD[self.kind]

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
        return replace(self.owner, **{self.field: ()})

    @property
    def sibling_with_leak(self) -> WorldEffect:
        return replace(self.sibling, **{self.field: (self.span,)})


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _clause(
    *,
    family: str,
    span: str,
    actor: str,
    facility: str,
    group: str,
    noun: str,
    verb: str,
) -> str:
    label = f"{group} {noun}"
    if family == "PERCENT_THE":
        return (
            f"{actor} performs {verb} on the {facility}. There is a {span} "
            f"the {facility} fails, and the {label} face death."
        )
    if family == "PERCENT_OF":
        return (
            f"{actor} performs {verb} on the {facility}. There is a {span} "
            f"of being blocked, and the {label} face death."
        )
    if family == "PERCENT_OF_NOUN":
        return (
            f"{actor} performs {verb} on the {facility}. There is a {span} "
            f"of blockage, and the {label} face death."
        )
    if family == "ADJECTIVE":
        return (
            f"{actor} performs {verb} on the {facility}. "
            f"The {label} lose {facility} power and face {span} death."
        )
    if family == "CHANCE":
        return (
            f"{actor} performs {verb} on the {facility}. "
            f"Keeping the {facility} connected allows the {label} {span} "
            f"to escape, but leaves an open route for the leak to spread."
        )
    if family == "MODAL":
        return (
            f"{actor} performs {verb} on the {facility}. "
            f"The leak spreads and {span} contaminate the supply used by "
            f"the {label}."
        )
    if family == "POST":
        return (
            f"{actor} performs {verb} on the {facility}. "
            f"The {facility} fails; {label} death is {span}."
        )
    if family == "SCOPE":
        return (
            f"{actor} performs {verb} on the {facility}. "
            f"The choice triggers {span} rolling blackouts across the grid "
            f"while the {label} face death."
        )
    return (
        f"{actor} performs {verb} on the {facility}, {span} holding the "
        f"{facility}, and sparing the {label}."
    )


def _qualifiers(kind: str, span: str, owner_id: str) -> dict[str, tuple[str, ...]]:
    empty = {
        "likelihood_qualifiers": (),
        "scope_qualifiers": (),
        "temporal_qualifiers": (),
    }
    if owner_id:
        empty[_KIND_FIELD[kind]] = (span,)
    return empty


def _head_world(
    *,
    source: str,
    kind: str,
    span: str,
    family: str,
    actor: str,
    facility: str,
    group: str,
    noun: str,
    verb: str,
) -> tuple[ScenarioWorldModel, str, str]:
    ref = (SourceRef("C0", source),)
    label = f"{group} {noun}"
    percent_family = family in _PERCENT_FAMILIES
    process_on_crowd = family == "ADJECTIVE"
    process_owns = kind != "LIKELIHOOD" or percent_family
    owner_id = "E1" if process_owns else "E2"
    sibling_id = "E2" if process_owns else "E1"
    process_outcome = (
        "fails" if family == "PERCENT_THE"
        else "blocked" if family in {"PERCENT_OF", "PERCENT_OF_NOUN"}
        else f"{label} lose {facility} power" if process_on_crowd
        else "rolling blackouts" if family == "SCOPE"
        else f"{facility} held" if family == "TEMPORAL"
        else f"{facility} fails" if family == "POST"
        else "leak spreads"
    )
    health_outcome = (
        f"{label} death" if family in {"ADJECTIVE", "POST", "SCOPE"} or percent_family
        else f"{label} escape" if family == "CHANCE"
        else f"{label} spared" if family == "TEMPORAL"
        else "supply contaminated"
    )
    health_polarity = (
        "BENEFICIAL" if family in {"CHANCE", "TEMPORAL"} else "ADVERSE"
    )
    process_kwargs = _qualifiers(kind, span, "E1" if owner_id == "E1" else "")
    health_kwargs = _qualifiers(kind, span, "E2" if owner_id == "E2" else "")
    process = WorldEffect(
        "E1", "A0", "P2" if process_on_crowd else "P1", process_outcome,
        "STATE_CHANGE", "NEUTRAL" if not percent_family else "ADVERSE",
        "DOWNSTREAM",
        "POSSIBLE" if percent_family else "CERTAIN", "PHYSICAL_STATE",
        provenance=ref, **process_kwargs,
    )
    health = WorldEffect(
        "E2", "A0", "P2", health_outcome, "EXPERIENCES",
        health_polarity, "DOWNSTREAM",
        (
            "STIPULATED_CONDITIONAL" if percent_family
            else "POSSIBLE" if (kind == "LIKELIHOOD" and not percent_family)
            else "CERTAIN"
        ),
        "HEALTH_OUTCOME",
        condition_ids=("COND1",) if percent_family else (),
        provenance=ref, **health_kwargs,
    )
    mediated = WorldEffect(
        "E3", "A0", "P1", f"{facility} held", "STATE_CHANGE",
        "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
        provenance=ref,
    )
    conditions = (
        (
            WorldCondition(
                "COND1", process_outcome, provenance=ref, event_effect_id="E1",
            ),
        )
        if percent_family else ()
    )
    links = (
        CausalLink("E0", "ENABLES", "E3", "CERTAIN", provenance=ref, action_id="A0"),
        CausalLink("E3", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
    ) if percent_family else (
        CausalLink("E0", "ENABLES", "E1", "CERTAIN", provenance=ref, action_id="A0"),
        CausalLink("E1", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
    )
    world = ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", ref),
            WorldParty("P2", label, "POPULATION", ref),
        ),
        actions=(
            WorldAction(
                "A0", f"perform {verb} on the {facility}",
                "P0", ("P1",),
                ("E0", "E1", "E2", "E3") if percent_family else ("E0", "E1", "E2"),
                ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P1", f"{verb} on the {facility}", "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ref,
            ),
            process,
            health,
            *((mediated,) if percent_family else ()),
        ),
        conditions=conditions,
        causal_links=links,
    )
    return world, owner_id, sibling_id


@st.composite
def qualifier_head_cases(draw) -> QualifierHeadCase:
    actor, facility, group, verb = _labels(draw, 4)
    kind, span, family = draw(st.sampled_from(HEAD_QUALIFIERS))
    noun = draw(st.sampled_from(_POPULATION_NOUNS))
    source = _clause(
        family=family, span=span, actor=actor, facility=facility,
        group=group, noun=noun, verb=verb,
    )
    world, owner_id, sibling_id = _head_world(
        source=source, kind=kind, span=span, family=family,
        actor=actor, facility=facility, group=group, noun=noun, verb=verb,
    )
    return QualifierHeadCase(
        source=source,
        kind=kind,
        span=span,
        family=family,
        owner_id=owner_id,
        sibling_id=sibling_id,
        world=world,
    )
