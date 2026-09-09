"""Likelihood hedges with a declared canonical span.

The oracle is LikelihoodSpanCase.canonical, not explicit_likelihood_spans.
Production is only asked whether it recognizes that span, binds it to the
modified outcome, leaves an unhedged sibling CERTAIN, and treats nested
chance-phrases as one span.

Do not import production hedge lists here. A generated hedge that the
extractor does not know is the failure this module exists to find.
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


# Independent of world_state._LIKELIHOOD_HEDGES. Known live-run hedges first,
# then gaps so shrinking reports the first unseen phrase.
# family: CHANCE / MODAL / ADJECTIVE. nested_inner is the shorter span the
# extractor must not emit beside the canonical phrase.
LIKELIHOOD_HEDGES = (
    ("almost no chance", "CHANCE", True, "chance"),
    ("a chance", "CHANCE", False, "chance"),
    ("could", "MODAL", False, ""),
    ("might", "MODAL", False, ""),
    ("near-certain", "ADJECTIVE", True, ""),
    ("likely", "ADJECTIVE", False, ""),
    ("no chance", "CHANCE", True, "chance"),
    ("remote chance", "CHANCE", False, "chance"),
    ("some chance", "CHANCE", False, "chance"),
    ("probably", "ADJECTIVE", False, ""),
    ("possibly", "ADJECTIVE", False, ""),
    ("nearly certain", "ADJECTIVE", True, ""),
    ("may", "MODAL", False, ""),
)
_OUTCOME = "escape"
_POPULATION_NOUNS = ("residents", "patients", "families")
_TOKEN_STOP = frozenset({
    "about", "almost", "certain", "chance", "could", "death", "drown",
    "escape", "group", "harm", "likely", "may", "might", "mine", "near",
    "nearly", "odds", "percent", "possible", "possibly", "probable",
    "probably", "probability", "remote", "risk", "some", "state", "unknown",
})
# Independent of world_state._PERCENT_CHANCE_CUES. A generated cue the
# quantity extractor still treats as a count is the failure this finds.
CHANCE_PERCENT_CUES = ("chance", "risk", "probability", "odds")
_CHANCE_PERCENTS = ("10%", "25%", "30%", "40%", "75%")
_COUNT_CARDINALS = ("12", "30", "300", "three hundred")
_PARTY_KINDS = ("CHANCE", "COUNT_PERCENT", "COUNT_CARDINAL")


@dataclass(frozen=True, slots=True)
class LikelihoodSpanCase:
    """One source hedge. canonical is written by the strategy."""

    source: str
    canonical: str
    family: str
    high_confidence: bool
    nested_inner: str
    group: str
    world: ScenarioWorldModel
    hedged_effect: WorldEffect
    unhedged_effect: WorldEffect

    @property
    def hedged_without_copy(self) -> WorldEffect:
        return replace(self.hedged_effect, likelihood_qualifiers=())


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _hedged_clause(span: str, family: str, group: str, noun: str) -> str:
    if family == "CHANCE":
        return f"The {group} {noun} have {span} of {_OUTCOME}."
    if family == "MODAL":
        return f"The {group} {noun} {span} {_OUTCOME}."
    return f"The {group} {noun} face {span} {_OUTCOME}."


def _likelihood_world(
    *,
    source: str,
    canonical: str,
    actor: str,
    facility: str,
    group: str,
    noun: str,
    verb: str,
) -> tuple[ScenarioWorldModel, WorldEffect, WorldEffect]:
    ref = (SourceRef("C0", source),)
    label = f"{group} {noun}"
    unhedged = WorldEffect(
        "E1", "A0", "P1", f"{facility} changes state", "STATE_CHANGE",
        "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
        provenance=ref,
    )
    hedged = WorldEffect(
        "E2", "A0", "P2", f"{label} {_OUTCOME}", "EXPERIENCES",
        "BENEFICIAL", "DOWNSTREAM", "POSSIBLE", "HEALTH_OUTCOME",
        likelihood_qualifiers=(canonical,),
        provenance=ref,
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
                "P0", ("P1",), ("E0", "E1", "E2"), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P1", f"{verb} on the {facility}", "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ref,
            ),
            unhedged,
            hedged,
        ),
        causal_links=(
            CausalLink("E0", "ENABLES", "E1", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E1", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
        ),
    )
    return world, hedged, unhedged


@st.composite
def likelihood_span_cases(draw) -> LikelihoodSpanCase:
    actor, facility, group, verb = _labels(draw, 4)
    canonical, family, high_confidence, nested_inner = draw(
        st.sampled_from(LIKELIHOOD_HEDGES)
    )
    noun = draw(st.sampled_from(_POPULATION_NOUNS))
    clause = _hedged_clause(canonical, family, group, noun)
    source = (
        f"{actor} performs {verb} on the {facility}. "
        f"The {facility} changes state. {clause}"
    )
    world, hedged, unhedged = _likelihood_world(
        source=source,
        canonical=canonical,
        actor=actor,
        facility=facility,
        group=group,
        noun=noun,
        verb=verb,
    )
    return LikelihoodSpanCase(
        source=source,
        canonical=canonical,
        family=family,
        high_confidence=high_confidence,
        nested_inner=nested_inner,
        group=group,
        world=world,
        hedged_effect=hedged,
        unhedged_effect=unhedged,
    )


@dataclass(frozen=True, slots=True)
class ChancePercentCase:
    """A percent or count. binds_to_party is written by the strategy."""

    kind: str
    source: str
    quantity_span: str
    chance_span: str
    literal_span: str
    cue: str
    group: str
    world: ScenarioWorldModel

    @property
    def binds_to_party(self) -> bool:
        return self.kind != "CHANCE"


def _percent_world(
    *,
    source: str,
    actor: str,
    facility: str,
    group: str,
    noun: str,
    verb: str,
    party_quantities: tuple[str, ...],
    effect_quantities: tuple[str, ...],
    likelihood: tuple[str, ...],
    modality: str,
    outcome: str,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    label = f"{group} {noun}"
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", ref),
            WorldParty("P2", label, "POPULATION", ref, quantities=party_quantities),
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
            WorldEffect(
                "E1", "A0", "P1", f"{facility} changes state", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                provenance=ref,
            ),
            WorldEffect(
                "E2", "A0", "P2", outcome, "EXPERIENCES",
                "BENEFICIAL", "DOWNSTREAM", modality, "HEALTH_OUTCOME",
                quantities=effect_quantities,
                likelihood_qualifiers=likelihood,
                provenance=ref,
            ),
        ),
        causal_links=(
            CausalLink("E0", "ENABLES", "E1", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E1", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
        ),
    )


@st.composite
def chance_percent_cases(draw) -> ChancePercentCase:
    actor, facility, group, verb = _labels(draw, 4)
    kind = draw(st.sampled_from(_PARTY_KINDS))
    noun = draw(st.sampled_from(_POPULATION_NOUNS))
    label = f"{group} {noun}"
    cue = ""
    chance_span = ""
    literal_span = ""
    if kind == "CHANCE":
        quantity_span = draw(st.sampled_from(_CHANCE_PERCENTS))
        cue = draw(st.sampled_from(CHANCE_PERCENT_CUES))
        chance_span = f"{quantity_span} {cue}"
        emitted = chance_span.replace(" ", "") if draw(st.booleans()) else chance_span
        literal_span = emitted
        source = (
            f"{actor} performs {verb} on the {facility}. "
            f"There is a {emitted} the {label} {_OUTCOME}."
        )
        world = _percent_world(
            source=source,
            actor=actor,
            facility=facility,
            group=group,
            noun=noun,
            verb=verb,
            party_quantities=(),
            effect_quantities=(),
            likelihood=(chance_span,),
            modality="POSSIBLE",
            outcome=f"{label} {_OUTCOME}",
        )
    elif kind == "COUNT_PERCENT":
        quantity_span = draw(st.sampled_from(_CHANCE_PERCENTS))
        source = (
            f"{actor} performs {verb} on the {facility}. "
            f"{quantity_span} of the {label} {_OUTCOME}."
        )
        world = _percent_world(
            source=source,
            actor=actor,
            facility=facility,
            group=group,
            noun=noun,
            verb=verb,
            party_quantities=(quantity_span,),
            effect_quantities=(quantity_span,),
            likelihood=(),
            modality="CERTAIN",
            outcome=f"{quantity_span} of the {label} {_OUTCOME}",
        )
    else:
        quantity_span = draw(st.sampled_from(_COUNT_CARDINALS))
        source = (
            f"{actor} performs {verb} on the {facility}. "
            f"{quantity_span} {label} {_OUTCOME}."
        )
        world = _percent_world(
            source=source,
            actor=actor,
            facility=facility,
            group=group,
            noun=noun,
            verb=verb,
            party_quantities=(quantity_span,),
            effect_quantities=(quantity_span,),
            likelihood=(),
            modality="CERTAIN",
            outcome=f"{quantity_span} {label} {_OUTCOME}",
        )
    return ChancePercentCase(
        kind=kind,
        source=source,
        quantity_span=quantity_span,
        chance_span=chance_span,
        literal_span=literal_span,
        cue=cue,
        group=group,
        world=world,
    )
