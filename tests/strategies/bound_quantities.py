"""Bound and plain quantity phrases with a declared canonical span.

The oracle is BoundQuantityCase.canonical / PlainQuantityCase.canonical,
not explicit_quantity_spans. Production is only asked whether it recognizes
that span, keeps a declared upper bound, admits the party/effect record of
it, and treats that single record as non-nested.

Do not import production prefix lists here. A generated hedge that the
extractor does not know is the failure this module exists to find.
"""
from __future__ import annotations

from dataclasses import dataclass

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


# Independent of world_state._QUANTITY_BOUND_PREFIXES. Longer phrases first
# so shrinking prefers the live floodgate hedge, then the gaps it revealed.
UPPER_BOUND_PREFIXES = (
    "as many as",
    "up to",
    "at most",
    "no more than",
    "not more than",
)
_CARDINALS = (
    "three hundred",
    "five hundred",
    "eight thousand",
    "300",
    "4,000",
    "twenty-five",
    "twenty five",
    "fifty-thousand",
    "fifty thousand",
)
_POPULATION_NOUNS = (
    "residents", "patients", "families", "workers", "households",
)
_TOKEN_STOP = frozenset({
    "about", "almost", "death", "dozen", "drown", "eight", "fewer",
    "fifty", "five", "flood", "forty", "four", "group", "harm",
    "household", "households", "hundred", "kill", "least", "less",
    "many", "mine", "more", "most", "nearly", "over", "people",
    "person", "score", "tens", "than", "thousand", "three", "twelve",
    "twenty", "workers",
})
_PLAIN_FORMS = (
    ("three hundred", True),
    ("five hundred", True),
    ("a hundred", True),
    ("4", True),
    ("300", True),
    ("tens of thousands", True),
    ("dozen", True),
    ("35%", True),
    ("twenty-five", True),
    ("twenty five", True),
    ("fifty-thousand", True),
    ("fifty thousand", True),
)
_UNIQUENESS_CARDINALS = (
    ("forty", ""),
    ("six hundred", ""),
    ("three hundred", ""),
    ("twenty-five", "five"),
    ("twenty five", "five"),
    ("fifty-thousand", "thousand"),
    ("fifty thousand", "thousand"),
    ("eight", ""),
    ("12", ""),
)


@dataclass(frozen=True, slots=True)
class BoundQuantityCase:
    """One upper-bound phrase. canonical is written by the strategy."""

    source: str
    prefix: str
    cardinal: str
    canonical: str
    noun: str
    group: str
    world: ScenarioWorldModel

    @property
    def bound_kind(self) -> str:
        return "UPPER"

    @property
    def core(self) -> str:
        return self.cardinal


@dataclass(frozen=True, slots=True)
class PlainQuantityCase:
    """An unbound quantity. canonical is the phrase itself."""

    source: str
    canonical: str
    binds_to_party: bool
    noun: str
    group: str
    world: ScenarioWorldModel


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _quantity_world(
    *,
    source: str,
    canonical: str,
    actor: str,
    facility: str,
    group: str,
    noun: str,
    verb: str,
    party_quantities: tuple[str, ...],
    effect_quantities: tuple[str, ...],
    outcome_repeats_quantity: bool,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    label = f"{group} {noun}"
    outcome = (
        f"{canonical} {label} are harmed"
        if outcome_repeats_quantity
        else f"{label} are harmed"
    )
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
                "E1", "A0", "P1", f"{facility} state changes", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                provenance=ref,
            ),
            WorldEffect(
                "E2", "A0", "P2", outcome, "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                quantities=effect_quantities,
                provenance=ref,
            ),
        ),
        causal_links=(
            CausalLink("E0", "ENABLES", "E1", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E1", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
        ),
    )


@st.composite
def bound_quantity_cases(draw) -> BoundQuantityCase:
    actor, facility, group, verb = _labels(draw, 4)
    prefix = draw(st.sampled_from(UPPER_BOUND_PREFIXES))
    cardinal = draw(st.sampled_from(_CARDINALS))
    noun = draw(st.sampled_from(_POPULATION_NOUNS))
    canonical = f"{prefix} {cardinal}"
    source = (
        f"{actor} performs {verb} on the {facility}. "
        f"{canonical} {group} {noun} are harmed."
    )
    return BoundQuantityCase(
        source=source,
        prefix=prefix,
        cardinal=cardinal,
        canonical=canonical,
        noun=noun,
        group=group,
        world=_quantity_world(
            source=source,
            canonical=canonical,
            actor=actor,
            facility=facility,
            group=group,
            noun=noun,
            verb=verb,
            party_quantities=(canonical,),
            effect_quantities=(canonical,),
            outcome_repeats_quantity=True,
        ),
    )


@st.composite
def plain_quantity_cases(draw) -> PlainQuantityCase:
    actor, facility, group, verb = _labels(draw, 4)
    canonical, binds_to_party = draw(st.sampled_from(_PLAIN_FORMS))
    noun = draw(st.sampled_from(_POPULATION_NOUNS))
    if binds_to_party:
        source = (
            f"{actor} performs {verb} on the {facility}. "
            f"{canonical} {group} {noun} are harmed."
        )
        party_quantities = (canonical,)
        effect_quantities = (canonical,)
        outcome_repeats_quantity = True
    else:
        source = (
            f"{actor} performs {verb} on the {facility} at {canonical} risk. "
            f"The {group} {noun} are harmed."
        )
        party_quantities = ()
        effect_quantities = ()
        outcome_repeats_quantity = False
    return PlainQuantityCase(
        source=source,
        canonical=canonical,
        binds_to_party=binds_to_party,
        noun=noun,
        group=group,
        world=_quantity_world(
            source=source,
            canonical=canonical,
            actor=actor,
            facility=facility,
            group=group,
            noun=noun,
            verb=verb,
            party_quantities=party_quantities,
            effect_quantities=effect_quantities,
            outcome_repeats_quantity=outcome_repeats_quantity,
        ),
    )


@dataclass(frozen=True, slots=True)
class QuantityPartyCase:
    """Two spans in one clause. expected is written by the strategy."""

    source: str
    left_span: str
    right_span: str
    left_id: str
    right_id: str
    generic_id: str
    nested_inners: tuple[str, ...]
    world: ScenarioWorldModel

    @property
    def expected(self) -> dict[str, tuple[str, ...]]:
        return {
            self.left_id: (self.left_span,),
            self.right_id: (self.right_span,),
            self.generic_id: (),
        }


def _uniqueness_world(
    *,
    source: str,
    actor: str,
    facility: str,
    left_group: str,
    left_noun: str,
    left_span: str,
    right_group: str,
    right_noun: str,
    right_span: str,
    verb: str,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    left_label = f"{left_group} {left_noun}"
    right_label = f"{right_group} {right_noun}"
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", ref),
            WorldParty("P2", left_label, "GROUP", ref, quantities=(left_span,)),
            WorldParty("P3", right_label, "HOUSEHOLD", ref, quantities=(right_span,)),
            WorldParty("P4", "people", "POPULATION", ref, quantities=()),
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
                "E1", "A0", "P1", f"{facility} state changes", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                provenance=ref,
            ),
            WorldEffect(
                "E2", "A0", "P2", f"{left_label} lose pay", "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME",
                provenance=ref,
            ),
            WorldEffect(
                "E3", "A0", "P3", f"{right_label} lose water", "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME",
                provenance=ref,
            ),
        ),
        causal_links=(
            CausalLink("E0", "ENABLES", "E1", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E1", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E1", "CAUSES", "E3", "CERTAIN", provenance=ref, action_id="A0"),
        ),
    )


@st.composite
def quantity_party_cases(draw) -> QuantityPartyCase:
    actor, facility, left_group, right_group, verb = _labels(draw, 5)
    left_noun = draw(st.sampled_from(_POPULATION_NOUNS))
    right_noun = draw(st.sampled_from(_POPULATION_NOUNS))
    left_row, right_row = draw(
        st.lists(st.sampled_from(_UNIQUENESS_CARDINALS), min_size=2, max_size=2, unique=True)
    )
    left_span, left_inner = left_row
    right_span, right_inner = right_row
    source = (
        f"{actor} performs {verb} on the {facility}. "
        f"{left_span} {left_group} {left_noun} lose pay while "
        f"{right_span} {right_group} {right_noun} lose water."
    )
    nested = tuple(dict.fromkeys(
        inner for inner in (left_inner, right_inner) if inner
    ))
    return QuantityPartyCase(
        source=source,
        left_span=left_span,
        right_span=right_span,
        left_id="P2",
        right_id="P3",
        generic_id="P4",
        nested_inners=nested,
        world=_uniqueness_world(
            source=source,
            actor=actor,
            facility=facility,
            left_group=left_group,
            left_noun=left_noun,
            left_span=left_span,
            right_group=right_group,
            right_noun=right_noun,
            right_span=right_span,
            verb=verb,
        ),
    )
