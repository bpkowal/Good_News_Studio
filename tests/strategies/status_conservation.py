"""Status-conservation Hypothesis cases with declared oracles.

Oracles live on the case fields (expect_incomplete, preserves_stakes,
records_quantities). Production validators are only asked whether they
agree. Do not import production dangling/contrast/quantity regexes here —
a generated form the extractor or admit gate does not catch is the failure
these strategies exist to find.

Templates below are independent closed-class phrase banks. They intentionally
overlap known working purge patterns so the first pass stresses systematic
omissions, not English coverage.
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


# Independent of world_state dangling regexes. expect_incomplete is written
# by the strategy, not inferred from production.
_OUTCOME_FORMS = (
    ("{head} is", True),
    ("{head} are", True),
    ("{head} was", True),
    ("{head} were", True),
    ("sent to", True),
    ("moved into", True),
    ("routed toward", True),
    ("is {head}", False),
    ("{head} executed", False),
    ("{head} completed", False),
    ("IS", False),
)

_CONTRAST_FRAMES = (
    "binary choice:",
    "choice:",
    "either",
)

# Declared welfare stakes. Fixed so the case oracle stays independent of
# production extractor wording while still matching known life-stake cues.
_SURVIVAL_SPAN = "immediate survival"
_LOSS_SPAN = "catastrophic loss of life"

# (source noun-phrase after "risks "/"endangers ", recorded quantity span).
# Closed-class bank: collectives, bounds, approximations, ranges, and
# population nouns. Placement modes exercise effect vs party vs omit vs misassign.
_LIFE_QUANTITY_FORMS = (
    ("hundreds of lives", "hundreds"),
    ("thousands of lives", "thousands"),
    ("millions of lives", "millions"),
    ("dozens of workers", "dozens"),
    ("at most 20 patients", "at most 20"),
    ("about 300 residents", "about 300"),
    ("roughly 300 homes", "roughly 300"),
    ("up to five hundred residents", "up to five hundred"),
    ("as many as three hundred patients", "as many as three hundred"),
    ("between 10 and 15 shelters", "between 10 and 15"),
)
_RESEARCH_QUANTITY_FORMS = (
    ("decades of medical and scientific research", "decades"),
    ("centuries of medical and scientific research", "centuries"),
)
_QUANTITY_PLACEMENTS = ("effect", "party", "omit", "misassign")
# Licensing topologies for quantity repair Hypothesis. Placement still
# decides whether the span is recorded; topology decides which provenance
# edges exist around the matched life effect.
_LICENSING_TOPOLOGIES = (
    "effect_has_clause",
    "party_has_clause",
    "both",
    "neither",
    "wrong_clause",
    "sibling_clause",
    "paraphrased_population",
    "competing_parties",
)

_TOKEN_STOP = frozenset({
    "about", "are", "as", "at", "been", "being", "between", "binary",
    "catastrophic", "centuries", "choice", "completed", "cost", "decades",
    "dozens", "either", "erase", "erased", "execute", "executed", "faces",
    "five", "from", "guarantee", "homes", "hundred", "hundreds",
    "immediate", "into", "is", "knowledge", "life", "lives", "loss",
    "many", "medical", "millions", "most", "moved", "or", "patients",
    "preserve", "purge", "refrain", "research", "residents", "risks",
    "roughly", "routed", "sent", "shelters", "survival", "the",
    "thousand", "thousands", "three", "to", "toward", "up", "was",
    "were", "withhold", "workers",
})


@dataclass(frozen=True, slots=True)
class OutcomePredicateCase:
    """One outcome string. expect_incomplete is written by the strategy."""

    source: str
    outcome: str
    expect_incomplete: bool
    effect_id: str
    world: ScenarioWorldModel
    issue_code: str = "OUTCOME_PREDICATE_INCOMPLETE"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("replace_outcome",)
    forbidden_ops: tuple[str, ...] = ("add_effect", "MORE_DEBATE")


@dataclass(frozen=True, slots=True)
class BinaryStipulationCase:
    """Binary-contrast source with declared keep/omit of life stakes."""

    source: str
    survival_span: str
    loss_span: str
    preserves_stakes: bool
    omit_survival: bool
    omit_loss: bool
    survival_effect_id: str
    loss_effect_id: str
    world: ScenarioWorldModel
    issue_code: str = "SOURCE_STIPULATED_OUTCOME_MISSING"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("add_effect",)
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "VIRTUE_REFRAME")


@dataclass(frozen=True, slots=True)
class QuantityConsequenceCase:
    """Quantity-bearing life/research stakes with declared retention."""

    life_source: str
    research_source: str
    life_quantity: str
    research_quantity: str
    placement: str
    records_quantities: bool
    life_effect_id: str
    research_effect_id: str
    world: ScenarioWorldModel
    licensing_topology: str = "effect_has_clause"
    expected_licensing_clause_id: str = "C0"
    issue_code: str = "QUANTITY_BEARING_CONSEQUENCE_MISSING"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = (
        "add_quantity", "add_provenance", "add_effect",
    )
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "COMPARE_FRAMEWORKS")


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _outcome_world(
    *,
    source: str,
    outcome: str,
    actor: str,
    facility: str,
    verb: str,
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
                "A0", f"execute {verb} on the {facility}",
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
def outcome_predicate_cases(draw) -> OutcomePredicateCase:
    """Fragment vs finished outcomes. expect_incomplete is declared."""
    actor, facility, verb, head = _labels(draw, 4)
    template, expect_incomplete = draw(st.sampled_from(_OUTCOME_FORMS))
    outcome = template.format(head=head)
    source = (
        f"{actor} may execute {verb} on the {facility}. "
        f"The admitted outcome must finish the predicate."
    )
    return OutcomePredicateCase(
        source=source,
        outcome=outcome,
        expect_incomplete=expect_incomplete,
        effect_id="E0",
        world=_outcome_world(
            source=source,
            outcome=outcome,
            actor=actor,
            facility=facility,
            verb=verb,
        ),
    )


def _contrast_clause(
    *,
    frame: str,
    actor: str,
    verb: str,
) -> str:
    execute_side = (
        f"execute the {verb} to guarantee {_SURVIVAL_SPAN}"
    )
    refrain_side = (
        f"refrain to preserve invaluable knowledge at the cost of {_LOSS_SPAN}"
    )
    if frame == "either":
        return (
            f"{actor} faces either {execute_side}, or {refrain_side}."
        )
    return (
        f"{actor} faces a {frame} {execute_side}, or {refrain_side}."
    )


def _binary_world(
    *,
    source: str,
    actor: str,
    facility: str,
    verb: str,
    omit_survival: bool,
    omit_loss: bool,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    effects: list[WorldEffect] = [
        WorldEffect(
            "E_exec", "A1", "P1", f"{verb} executed", "STATE_CHANGE",
            "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
            provenance=ref,
            source_proposition=f"execute the {verb}",
            derivation_operation="DIRECT_COPY",
        ),
    ]
    effect_ids_a0: list[str] = []
    effect_ids_a1: list[str] = ["E_exec"]
    if not omit_survival:
        effects.append(WorldEffect(
            "E_surv", "A1", "P2", _SURVIVAL_SPAN, "STATE_CHANGE",
            "BENEFICIAL", "DIRECT", "CERTAIN", "HEALTH_OUTCOME",
            provenance=ref,
            source_proposition=_SURVIVAL_SPAN,
            derivation_operation="DIRECT_COPY",
        ))
        effect_ids_a1.append("E_surv")
    if not omit_loss:
        effects.append(WorldEffect(
            "E_loss", "A0", "P2", _LOSS_SPAN, "DIES",
            "ADVERSE", "DIRECT", "CERTAIN", "HEALTH_OUTCOME",
            provenance=ref,
            source_proposition=_LOSS_SPAN,
            derivation_operation="DIRECT_COPY",
        ))
        effect_ids_a0.append("E_loss")
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", ref),
            WorldParty("P2", "populace", "POPULATION", ref),
        ),
        actions=(
            WorldAction(
                "A0", f"withhold the {verb}",
                "P0", ("P2",), tuple(effect_ids_a0), ref,
            ),
            WorldAction(
                "A1", f"execute the {verb}",
                "P0", ("P1",), tuple(effect_ids_a1), ref,
            ),
        ),
        effects=tuple(effects),
    )


@st.composite
def binary_stipulation_cases(draw) -> BinaryStipulationCase:
    """Binary-contrast worlds. preserves_stakes is declared."""
    actor, facility, verb = _labels(draw, 3)
    frame = draw(st.sampled_from(_CONTRAST_FRAMES))
    preserves = draw(st.booleans())
    if preserves:
        omit_survival = False
        omit_loss = False
    else:
        mode = draw(st.sampled_from(("omit_survival", "omit_loss", "omit_both")))
        omit_survival = mode in {"omit_survival", "omit_both"}
        omit_loss = mode in {"omit_loss", "omit_both"}
    source = _contrast_clause(frame=frame, actor=actor, verb=verb)
    return BinaryStipulationCase(
        source=source,
        survival_span=_SURVIVAL_SPAN,
        loss_span=_LOSS_SPAN,
        preserves_stakes=preserves,
        omit_survival=omit_survival,
        omit_loss=omit_loss,
        survival_effect_id="E_surv",
        loss_effect_id="E_loss",
        world=_binary_world(
            source=source,
            actor=actor,
            facility=facility,
            verb=verb,
            omit_survival=omit_survival,
            omit_loss=omit_loss,
        ),
    )


def _quantity_world(
    *,
    life_source: str,
    research_source: str,
    actor: str,
    facility: str,
    life_quantity: str,
    research_quantity: str,
    placement: str,
    licensing_topology: str = "effect_has_clause",
) -> ScenarioWorldModel:
    life_ref = (SourceRef("C0", life_source),)
    research_ref = (SourceRef("C1", research_source),)
    sibling_ref = (
        SourceRef(
            "C3",
            "binary choice: execute the purge, or refrain at the cost of "
            "catastrophic loss of life",
        ),
    )
    wrong_ref = (SourceRef("C9", "an unrelated logistics clause"),)
    life_effect_qty: tuple[str, ...] = ()
    research_effect_qty: tuple[str, ...] = ()
    life_party_qty: tuple[str, ...] = ()
    research_party_qty: tuple[str, ...] = ()
    if placement == "effect":
        life_effect_qty = (life_quantity,)
        research_effect_qty = (research_quantity,)
    elif placement == "party":
        life_party_qty = (life_quantity,)
        research_party_qty = (research_quantity,)
    elif placement == "misassign":
        # Span present, but on the wrong effect/party — still a semantic loss
        # for the matched consequence.
        life_effect_qty = ()
        research_effect_qty = (life_quantity,)
        life_party_qty = (research_quantity,)
    # omit: leave all empty

    if licensing_topology in {"effect_has_clause", "both"}:
        e_loss_prov = life_ref
    elif licensing_topology == "sibling_clause":
        e_loss_prov = sibling_ref
    elif licensing_topology == "wrong_clause":
        e_loss_prov = wrong_ref
    elif licensing_topology == "neither":
        e_loss_prov = ()
    elif licensing_topology == "party_has_clause":
        e_loss_prov = ()
    else:
        # paraphrased_population / competing_parties: effect cites sibling only.
        e_loss_prov = sibling_ref

    if licensing_topology in {
        "party_has_clause",
        "both",
        "effect_has_clause",
        "paraphrased_population",
        "competing_parties",
        "sibling_clause",
        "wrong_clause",
    }:
        p2_prov = life_ref
    else:
        # neither: party does not license; action still carries C0 for extraction.
        p2_prov = ()

    if licensing_topology == "paraphrased_population":
        populace_label = "city populace"
    else:
        populace_label = "populace"

    parties = [
        WorldParty("P0", actor, "HUMAN", life_ref),
        WorldParty("P1", facility, "INFRASTRUCTURE", research_ref),
        WorldParty(
            "P2", populace_label, "POPULATION", p2_prov,
            quantities=life_party_qty,
        ),
        WorldParty(
            "P3", "central archive", "FACILITY", research_ref,
            quantities=research_party_qty,
        ),
    ]
    if licensing_topology == "competing_parties":
        parties.append(
            WorldParty("P4", "tourist group", "POPULATION", sibling_ref),
        )

    # Action always carries the life clause so extractors see the source text;
    # topology varies effect/party licensing only.
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=tuple(parties),
        actions=(
            WorldAction(
                "A0", "withhold emergency purge",
                "P0", ("P2",), ("E_loss",), life_ref,
            ),
            WorldAction(
                "A1", "execute emergency purge",
                "P0", ("P1", "P3"), ("E_erase",), research_ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E_loss", "A0", "P2", "catastrophic loss of life", "DIES",
                "ADVERSE", "DIRECT", "CERTAIN", "HEALTH_OUTCOME",
                quantities=life_effect_qty,
                provenance=e_loss_prov,
                source_proposition=life_source,
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E_erase", "A1", "P3", "research erased", "STATE_CHANGE",
                "ADVERSE", "DIRECT", "CERTAIN", "OTHER",
                quantities=research_effect_qty,
                provenance=research_ref,
                source_proposition=research_source,
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def quantity_consequence_cases(draw) -> QuantityConsequenceCase:
    """Life/research quantity worlds. placement + licensing topology."""
    actor, facility = _labels(draw, 2)
    life_phrase, life_quantity = draw(st.sampled_from(_LIFE_QUANTITY_FORMS))
    research_phrase, research_quantity = draw(
        st.sampled_from(_RESEARCH_QUANTITY_FORMS)
    )
    placement = draw(st.sampled_from(_QUANTITY_PLACEMENTS))
    licensing_topology = draw(st.sampled_from(_LICENSING_TOPOLOGIES))
    records = placement in {"effect", "party"}
    life_source = (
        f"A cyberattack on the {facility} risks {life_phrase} "
        f"through imminent infrastructure failure."
    )
    research_source = (
        f"Executing the purge will permanently erase {research_phrase}."
    )
    return QuantityConsequenceCase(
        life_source=life_source,
        research_source=research_source,
        life_quantity=life_quantity,
        research_quantity=research_quantity,
        placement=placement,
        records_quantities=records,
        life_effect_id="E_loss",
        research_effect_id="E_erase",
        licensing_topology=licensing_topology,
        expected_licensing_clause_id="C0",
        world=_quantity_world(
            life_source=life_source,
            research_source=research_source,
            actor=actor,
            facility=facility,
            life_quantity=life_quantity,
            research_quantity=research_quantity,
            placement=placement,
            licensing_topology=licensing_topology,
        ),
    )
