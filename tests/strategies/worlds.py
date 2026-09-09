"""Strategies that build typed worlds with a declared causal topology.

`st.from_type(ScenarioWorldModel)` is not enough: polarity, modality, and
kind are bare strings, and nothing ties effect.action_id to an action.
The live JSON Schema in local_specialists is also the wrong generator —
it is bound to that scenario's action_ids and clause_ids, and it does
not encode referential integrity or NEUTRAL infrastructure INTERVENTION.

hypothesis-jsonschema can wait. Closed-class constants plus a declared
link field are the thin layer Hypothesis cannot supply.
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


# Agent-caused settled harm. ENABLES is allowing, not doing.
_DOING_RELATIONS = frozenset({"CAUSES", "ACCELERATES"})
_A0_HARM_RELATIONS = ("CAUSES", "ENABLES")


@dataclass(frozen=True, slots=True)
class SiblingHarmCase:
    """Two-action world whose oracle is the declared A0 harm link.

    `a0_to_harm` and `harm_to_end` are written by the strategy, not inferred
    from the production path-finder.
    """

    world: ScenarioWorldModel
    actions: tuple[str, str]
    scenario: str
    a0_to_harm: str
    harm_to_end: str | None
    actor: str
    facility: str
    group: str
    crowd: str

    @property
    def a0_does_harm(self) -> bool:
        return self.a0_to_harm in _DOING_RELATIONS

    @property
    def burden_is_means(self) -> bool:
        return self.harm_to_end == "CAUSES"


_TOKEN = st.from_regex(r"[a-z]{4,8}", fullmatch=True)


def unique_tokens(count: int):
    """Count distinct lowercase labels. Shared by causal and epistemic strategies."""
    return st.lists(_TOKEN, min_size=count, max_size=count, unique=True)


@st.composite
def sibling_harm_cases(draw) -> SiblingHarmCase:
    """NEUTRAL facility INTERVENTION; sibling or intermediate welfare rows."""
    actor, facility, group, crowd, verb0, verb1, harm, benefit = draw(
        unique_tokens(8)
    )
    a0_to_harm = draw(st.sampled_from(_A0_HARM_RELATIONS))
    harm_to_end = None
    if a0_to_harm == "CAUSES":
        harm_to_end = draw(st.sampled_from([None, "CAUSES"]))
    scenario = (
        f"An {actor} may {verb0} the {facility}, {harm} the {group} "
        f"while {benefit} the {crowd}, or {verb1} the {facility}."
    )
    actions = (
        f"{verb0} the {facility}, {harm} the {group} while {benefit} the {crowd}",
        f"{verb1} the {facility}, leaving the {group} to {harm}",
    )
    ref = (SourceRef("C0", scenario),)
    links = [
        CausalLink("E0", a0_to_harm, "E1", "CERTAIN", provenance=ref, action_id="A0"),
        CausalLink("E0", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
        CausalLink("E3", "ENABLES", "E4", "CERTAIN", provenance=ref, action_id="A1"),
    ]
    if harm_to_end:
        links.append(CausalLink(
            "E1", harm_to_end, "E2", "CERTAIN", provenance=ref, action_id="A0",
        ))
    world = ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", ref),
            WorldParty("P2", group, "GROUP", ref),
            WorldParty("P3", crowd, "POPULATION", ref),
        ),
        actions=(
            WorldAction("A0", actions[0], "P0", ("P1",), ("E0", "E1", "E2"), ref),
            WorldAction("A1", actions[1], "P0", ("P1",), ("E3", "E4"), ref),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P1", verb0.upper(), "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
            ),
            WorldEffect(
                "E1", "A0", "P2", harm.upper(), "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                provenance=ref,
            ),
            WorldEffect(
                "E2", "A0", "P3", benefit.upper(), "EXPERIENCES",
                "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME",
                provenance=ref,
            ),
            WorldEffect(
                "E3", "A1", "P1", verb1.upper(), "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
            ),
            WorldEffect(
                "E4", "A1", "P2", harm.upper(), "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                provenance=ref,
            ),
        ),
        causal_links=tuple(links),
    )
    return SiblingHarmCase(
        world=world,
        actions=actions,
        scenario=scenario,
        a0_to_harm=a0_to_harm,
        harm_to_end=harm_to_end,
        actor=actor,
        facility=facility,
        group=group,
        crowd=crowd,
    )


_HEALTH_KINDS = ("HEALTH_OUTCOME", "WELFARE_OUTCOME")
_DIMENSION_CUES = frozenset({
    "food", "health", "housing", "medical", "safety", "security", "shelter",
    "survival", "water", "choice", "consent", "freedom", "liberty", "privacy",
    "speech", "vote", "rights", "access", "career", "office", "school",
    "training", "transit", "income", "money", "poverty", "wealth", "assets",
    "funding", "wages", "bodily", "detain",
})


@dataclass(frozen=True, slots=True)
class HealthDimensionCase:
    """One downstream row whose declared kind is or is not HEALTH_OUTCOME."""

    world: ScenarioWorldModel
    actions: tuple[str, str]
    scenario: str
    effect_kind: str

    @property
    def should_be_basic_security(self) -> bool:
        return self.effect_kind == "HEALTH_OUTCOME"


@st.composite
def health_dimension_cases(draw) -> HealthDimensionCase:
    """Same NEUTRAL-facility topology; only E1's declared kind varies."""
    actor, facility, group, crowd, verb0, verb1, harm, benefit = draw(
        unique_tokens(8).filter(
            lambda tokens: not any(token in _DIMENSION_CUES for token in tokens)
        )
    )
    effect_kind = draw(st.sampled_from(_HEALTH_KINDS))
    scenario = (
        f"An {actor} may {verb0} the {facility}, {harm} the {group} "
        f"while {benefit} the {crowd}, or {verb1} the {facility}."
    )
    actions = (
        f"{verb0} the {facility}, {harm} the {group} while {benefit} the {crowd}",
        f"{verb1} the {facility}, leaving the {group} to {harm}",
    )
    ref = (SourceRef("C0", scenario),)
    world = ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", ref),
            WorldParty("P2", group, "GROUP", ref),
            WorldParty("P3", crowd, "POPULATION", ref),
        ),
        actions=(
            WorldAction("A0", actions[0], "P0", ("P1",), ("E0", "E1", "E2"), ref),
            WorldAction("A1", actions[1], "P0", ("P1",), ("E3", "E4"), ref),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P1", verb0.upper(), "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
            ),
            WorldEffect(
                "E1", "A0", "P2", harm.upper(), "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", effect_kind,
                provenance=ref,
            ),
            WorldEffect(
                "E2", "A0", "P3", benefit.upper(), "EXPERIENCES",
                "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME",
                provenance=ref,
            ),
            WorldEffect(
                "E3", "A1", "P1", verb1.upper(), "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
            ),
            WorldEffect(
                "E4", "A1", "P2", harm.upper(), "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                provenance=ref,
            ),
        ),
        causal_links=(
            CausalLink("E0", "CAUSES", "E1", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E0", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E3", "ENABLES", "E4", "CERTAIN", provenance=ref, action_id="A1"),
        ),
    )
    return HealthDimensionCase(
        world=world,
        actions=actions,
        scenario=scenario,
        effect_kind=effect_kind,
    )


_OVERLAYS = ("FOREGONE", "SHARED")
_UNINFORMATIVE_OUTCOMES = frozenset({
    "action", "choice", "decision", "option", "outcome", "policy",
    "program", "proposal", "system",
})


@dataclass(frozen=True, slots=True)
class ForegoneDualCase:
    """Two actions whose actual/foregone rows are declared, not inferred."""

    graph: object
    actions: tuple[str, str]
    harm: str
    benefit: str
    overlay: str

    @property
    def should_equate(self) -> bool:
        return self.overlay == "SHARED"


@st.composite
def foregone_dual_cases(draw) -> ForegoneDualCase:
    """A0/A1 swap, or the same actual outcome on both actions."""
    from global_workspace.semantic_graph import SemanticEdge, SemanticGraph, SemanticNode

    harm, benefit, verb0, verb1 = draw(
        st.lists(
            _TOKEN.filter(lambda token: token not in _UNINFORMATIVE_OUTCOMES),
            min_size=4,
            max_size=4,
            unique=True,
        )
    )
    overlay = draw(st.sampled_from(_OVERLAYS))
    actions = (f"{verb0} the first option", f"{verb1} the second option")
    graph = SemanticGraph()
    graph.add_node(SemanticNode(
        "A0", "ACTION", actions[0], attributes={"canonical_action_id": "A0"},
    ))
    graph.add_node(SemanticNode(
        "A1", "ACTION", actions[1], attributes={"canonical_action_id": "A1"},
    ))
    if overlay == "FOREGONE":
        graph.add_node(SemanticNode(
            "A0:C", "CONSEQUENCE", harm,
            attributes={"polarity": "ADVERSE", "directness": "DOWNSTREAM"},
        ))
        graph.add_node(SemanticNode(
            "A0:F", "CONSEQUENCE", benefit,
            attributes={"polarity": "FOREGONE", "directness": "FOREGONE"},
        ))
        graph.add_node(SemanticNode(
            "A1:C", "CONSEQUENCE", benefit,
            attributes={"polarity": "BENEFICIAL", "directness": "DOWNSTREAM"},
        ))
        graph.add_node(SemanticNode(
            "A1:F", "CONSEQUENCE", harm,
            attributes={"polarity": "FOREGONE", "directness": "FOREGONE"},
        ))
        graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "A0:C"))
        graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "A0:F"))
        graph.add_edge(SemanticEdge("A1", "HAS_CONSEQUENCE", "A1:C"))
        graph.add_edge(SemanticEdge("A1", "HAS_CONSEQUENCE", "A1:F"))
    else:
        graph.add_node(SemanticNode(
            "A0:C", "CONSEQUENCE", harm,
            attributes={"polarity": "ADVERSE", "directness": "DOWNSTREAM"},
        ))
        graph.add_node(SemanticNode(
            "A1:C", "CONSEQUENCE", harm,
            attributes={"polarity": "ADVERSE", "directness": "DOWNSTREAM"},
        ))
        graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "A0:C"))
        graph.add_edge(SemanticEdge("A1", "HAS_CONSEQUENCE", "A1:C"))
    return ForegoneDualCase(
        graph=graph,
        actions=actions,
        harm=harm,
        benefit=benefit,
        overlay=overlay,
    )


_LIVE_TRACE_LABELS = frozenset({
    "mine", "explosion", "suffocation", "suffocate", "tunnel",
})
_AVERTED_RISK_KINDS = ("AVERTED", "DIRECT")
_AVERTED_RISK_CHANCE = ("POSSIBLE", "PROBABILISTIC")
_AVERTED_RISK_CHANCE_HINT = {
    "POSSIBLE": ("possible",),
    "PROBABILISTIC": ("likely",),
}


@dataclass(frozen=True, slots=True)
class AvertedRiskCase:
    """Two-action world whose oracle is the declared survive-row kind.

    AVERTED: CERTAIN BENEFICIAL welfare only as the dual of an opposed
    chance-harm, via a process-prevention parent. DIRECT: CERTAIN welfare
    caused by a DIRECT act on that same party. Labels are incidental.
    """

    world: ScenarioWorldModel
    actions: tuple[str, str]
    scenario: str
    row_kind: str
    group: str
    survive: str
    process: str

    @property
    def should_count_as_obtained(self) -> bool:
        return self.row_kind == "DIRECT"


@st.composite
def averted_risk_cases(draw) -> AvertedRiskCase:
    """NEUTRAL facility INTERVENTION; survive-row is averted risk or direct."""
    actor, facility, group, process, verb0, verb1, survive, harm, hold = draw(
        unique_tokens(9).filter(
            lambda tokens: not any(
                token in _DIMENSION_CUES or token in _LIVE_TRACE_LABELS
                for token in tokens
            )
        )
    )
    row_kind = draw(st.sampled_from(_AVERTED_RISK_KINDS))
    chance_modality = draw(st.sampled_from(_AVERTED_RISK_CHANCE))
    hedge = _AVERTED_RISK_CHANCE_HINT[chance_modality][0]
    scenario = (
        f"An {actor} may {verb0} the {facility} and {hold} then {survive} "
        f"the {group} via the {process}, or {verb1} the {facility} and "
        f"{hedge} {harm} the {group}. The {process} is {hedge}."
    )
    actions = (
        f"{verb0} the {facility}, {survive} the {group}",
        f"{verb1} the {facility}, leaving the {group} to {harm}",
    )
    ref = (SourceRef("C0", scenario),)
    if row_kind == "AVERTED":
        a0_mid = WorldEffect(
            "E1", "A0", "P3", process.upper(), "IS_PREVENTED",
            "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
            provenance=ref,
        )
        a0_links = (
            CausalLink("E0", "CAUSES", "E1", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E1", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
        )
    else:
        a0_mid = WorldEffect(
            "E1", "A0", "P2", hold.upper(), "PERFORMS",
            "BENEFICIAL", "DIRECT", "CERTAIN", "INSTITUTIONAL_OUTCOME",
            provenance=ref,
        )
        a0_links = (
            CausalLink("E0", "CAUSES", "E1", "CERTAIN", provenance=ref, action_id="A0"),
            CausalLink("E1", "CAUSES", "E2", "CERTAIN", provenance=ref, action_id="A0"),
        )
    world = ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P1", facility, "INFRASTRUCTURE", ref),
            WorldParty("P2", group, "GROUP", ref),
            WorldParty("P3", process, "PROCESS", ref),
        ),
        actions=(
            WorldAction("A0", actions[0], "P0", ("P1",), ("E0", "E1", "E2"), ref),
            WorldAction("A1", actions[1], "P0", ("P1",), ("E3", "E4", "E5"), ref),
        ),
        effects=(
            WorldEffect(
                "E0", "A0", "P1", verb0.upper(), "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
            ),
            a0_mid,
            WorldEffect(
                "E2", "A0", "P2", survive.upper(), "EXPERIENCES",
                "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                provenance=ref,
            ),
            WorldEffect(
                "E3", "A1", "P1", verb1.upper(), "PERFORMS",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
            ),
            WorldEffect(
                "E4", "A1", "P2", harm.upper(), "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", chance_modality, "HEALTH_OUTCOME",
                provenance=ref,
                likelihood_qualifiers=_AVERTED_RISK_CHANCE_HINT[chance_modality],
            ),
            WorldEffect(
                "E5", "A1", "P3", process.upper(), "OCCURS",
                "ADVERSE", "DOWNSTREAM", chance_modality, "PHYSICAL_STATE",
                provenance=ref,
                likelihood_qualifiers=_AVERTED_RISK_CHANCE_HINT[chance_modality],
            ),
        ),
        causal_links=a0_links + (
            CausalLink("E3", "CAUSES", "E4", "CERTAIN", provenance=ref, action_id="A1"),
            CausalLink("E3", "CAUSES", "E5", "CERTAIN", provenance=ref, action_id="A1"),
        ),
    )
    return AvertedRiskCase(
        world=world,
        actions=actions,
        scenario=scenario,
        row_kind=row_kind,
        group=group,
        survive=survive,
        process=process,
    )
