"""Verb-lemma and consequence-reassignment Hypothesis cases.

Oracles are declared case fields:
- VerbLemmaCase.binds_lemma — outcome shares a verbal lemma with source
- ConsequenceReassignmentCase.preserves_action — effect stays on the action
  whose intervention owns the consequence

Do not import production stem tables here. A generated morphological variant
or action reassignment the binder mishandles is the failure these strategies
exist to find.
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


# Independent lemma bank. Good outcomes are morphological variants of the
# source event; bad outcomes are distinct predicates.
_LEMMA_ROWS = (
    ("deploy an emergency purge", "is purged", "is destroyed"),
    ("secure the city", "is secured", "is abandoned"),
    ("evacuate the ward", "is evacuated", "is flooded"),
    ("quarantine the lab", "is quarantined", "is opened"),
    ("isolate the reactor", "is isolated", "is connected"),
)

_TOKEN_STOP = frozenset({
    "abandoned", "city", "connected", "deploy", "destroyed", "emergency",
    "evacuated", "flooded", "isolate", "isolated", "lab", "opened", "purge",
    "purged", "quarantine", "quarantined", "reactor", "secure", "secured",
    "the", "ward",
})


@dataclass(frozen=True, slots=True)
class VerbLemmaCase:
    """One effect. binds_lemma is written by the strategy."""

    source: str
    outcome: str
    binds_lemma: bool
    effect_id: str
    world: ScenarioWorldModel
    issue_code: str = "VERB_LEMMA_MISMATCH"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("REPAIR_GROUNDING",)
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "COMPARE_FRAMEWORKS")


@dataclass(frozen=True, slots=True)
class ConsequenceReassignmentCase:
    """Consequence text owned by A0; effect may be wrongly hung on A1."""

    source: str
    expected_action_id: str
    effect_action_id: str
    preserves_action: bool
    effect_id: str
    world: ScenarioWorldModel
    issue_code: str = "CONSEQUENCE_REASSIGNED"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("REPAIR_GROUNDING", "CHALLENGE_PREMISE")
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "VIRTUE_REFRAME")


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _lemma_world(
    *,
    source: str,
    outcome: str,
    actor: str,
    facility: str,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", ref),
        ),
        actions=(
            WorldAction(
                "A0", source,
                "P0", ("P1",), ("E0",), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P1", outcome, "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ref,
                source_proposition=source,
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def verb_lemma_cases(draw) -> VerbLemmaCase:
    """Morphological bind vs wrong-lemma paraphrase."""
    actor, facility = _labels(draw, 2)
    source, good_outcome, bad_outcome = draw(st.sampled_from(_LEMMA_ROWS))
    binds = draw(st.booleans())
    outcome = good_outcome if binds else bad_outcome
    clause = f"{actor} can {source} on the {facility}."
    return VerbLemmaCase(
        source=source,
        outcome=outcome,
        binds_lemma=binds,
        effect_id="E0",
        world=_lemma_world(
            source=source,
            outcome=outcome,
            actor=actor,
            facility=facility,
        ),
    )


def _reassignment_world(
    *,
    source: str,
    outcome: str,
    actor: str,
    facility: str,
    alt_verb: str,
    effect_action_id: str,
) -> ScenarioWorldModel:
    owned_ref = (SourceRef("C0", source),)
    alt_ref = (SourceRef("C1", f"{actor} may {alt_verb} the {facility}."),)
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", owned_ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", owned_ref),
        ),
        actions=(
            WorldAction(
                "A0", source,
                "P0", ("P1",), ("E0",) if effect_action_id == "A0" else (),
                owned_ref,
            ),
            WorldAction(
                "A1", f"{alt_verb} the {facility}",
                "P0", ("P1",), ("E0",) if effect_action_id == "A1" else (),
                alt_ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E0", effect_action_id, "P1", outcome, "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=owned_ref,
                source_proposition=source,
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def consequence_reassignment_cases(draw) -> ConsequenceReassignmentCase:
    """Effect hung on the owning action vs the sibling action."""
    actor, facility, alt_verb = _labels(draw, 3)
    source, good_outcome, _bad = draw(st.sampled_from(_LEMMA_ROWS))
    preserves = draw(st.booleans())
    effect_action_id = "A0" if preserves else "A1"
    return ConsequenceReassignmentCase(
        source=source,
        expected_action_id="A0",
        effect_action_id=effect_action_id,
        preserves_action=preserves,
        effect_id="E0",
        world=_reassignment_world(
            source=source,
            outcome=good_outcome,
            actor=actor,
            facility=facility,
            alt_verb=alt_verb,
            effect_action_id=effect_action_id,
        ),
    )
