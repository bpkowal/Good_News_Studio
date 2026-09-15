"""CAUSES/BEFORE directionality Hypothesis cases with declared oracles.

Edges are abstract effect/event ids. Oracles declare whether reverse or
skip-links must be absent, and whether BEFORE may transit. No discourse clock.
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


_MUTATIONS = (
    "causes_chain_no_skip",
    "causes_no_reverse",
    "before_transits",
    "before_antisymmetric",
)


@dataclass(frozen=True, slots=True)
class DirectionalityCase:
    mutation: str
    licensed_causes_edges: tuple[dict, ...]
    licensed_before_edges: tuple[dict, ...]
    expect_causes_skip: bool
    expect_causes_reverse: bool
    expect_before_transit: bool
    expect_relational_errors: bool
    world: ScenarioWorldModel
    issue_code: str = "RELATIONAL_DIRECTIONALITY"
    repair_stage: str = "relational"
    allowed_ops: tuple[str, ...] = ("REJECT_FREE_CAUSAL_HOP",)
    forbidden_ops: tuple[str, ...] = ("FREE_COMPOSITE", "INVENT_REVERSE")


def _edge(src: str, tgt: str, rel: str) -> dict:
    return {
        "source": src,
        "target": tgt,
        "relation": rel,
        "derived_marked": False,
    }


def _chain_world(*, with_skip: bool, with_reverse: bool) -> ScenarioWorldModel:
    ref = (SourceRef("C0", "process then outcome then harm"),)
    parties = (
        WorldParty("P0", "actor", "PERSON", ref),
        WorldParty("P1", "patient", "PERSON", ref),
    )
    effects = (
        WorldEffect(
            "E_a", "A0", "P0", "intervenes", "STATE_CHANGE",
            "NEUTRAL", "DIRECT", "CERTAIN", "PHYSICAL_STATE",
            provenance=ref, source_proposition="intervenes",
            derivation_operation="DIRECT_COPY",
        ),
        WorldEffect(
            "E_b", "A0", "P1", "process unfolds", "STATE_CHANGE",
            "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
            provenance=ref, source_proposition="process unfolds",
            derivation_operation="DIRECT_COPY",
        ),
        WorldEffect(
            "E_c", "A0", "P1", "outcome obtains", "STATE_CHANGE",
            "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
            provenance=ref, source_proposition="outcome obtains",
            derivation_operation="DIRECT_COPY",
        ),
    )
    links = [
        CausalLink("E_a", "CAUSES", "E_b", "CERTAIN", (), ref, "A0"),
        CausalLink("E_b", "CAUSES", "E_c", "CERTAIN", (), ref, "A0"),
    ]
    if with_skip:
        links.append(CausalLink("E_a", "CAUSES", "E_c", "CERTAIN", (), ref, "A0"))
    if with_reverse:
        links.append(CausalLink("E_b", "CAUSES", "E_a", "CERTAIN", (), ref, "A0"))
    return ScenarioWorldModel(
        schema_version="1.3",
        parties=parties,
        actions=(
            WorldAction(
                "A0", "intervene", "P0", ("P1",),
                ("E_a", "E_b", "E_c"), ref,
            ),
        ),
        effects=effects,
        causal_links=tuple(links),
    )


def _empty_world() -> ScenarioWorldModel:
    ref = (SourceRef("C0", "temporal markers only"),)
    return ScenarioWorldModel(
        schema_version="1.3",
        parties=(WorldParty("P0", "actor", "PERSON", ref),),
        actions=(
            WorldAction("A0", "wait", "P0", (), (), ref),
        ),
        effects=(),
    )


@st.composite
def directionality_cases(draw) -> DirectionalityCase:
    mutation = draw(st.sampled_from(_MUTATIONS))
    if mutation == "causes_chain_no_skip":
        world = _chain_world(with_skip=False, with_reverse=False)
        return DirectionalityCase(
            mutation=mutation,
            licensed_causes_edges=(),
            licensed_before_edges=(),
            expect_causes_skip=False,
            expect_causes_reverse=False,
            expect_before_transit=False,
            expect_relational_errors=False,
            world=world,
        )
    if mutation == "causes_no_reverse":
        world = _chain_world(with_skip=False, with_reverse=False)
        return DirectionalityCase(
            mutation=mutation,
            licensed_causes_edges=(),
            licensed_before_edges=(),
            expect_causes_skip=False,
            expect_causes_reverse=False,
            expect_before_transit=False,
            expect_relational_errors=False,
            world=world,
            allowed_ops=("REJECT_REVERSE_CAUSES",),
        )
    if mutation == "before_transits":
        edges = (
            _edge("T0", "T1", "BEFORE"),
            _edge("T1", "T2", "BEFORE"),
        )
        return DirectionalityCase(
            mutation=mutation,
            licensed_causes_edges=(),
            licensed_before_edges=edges,
            expect_causes_skip=False,
            expect_causes_reverse=False,
            expect_before_transit=True,
            expect_relational_errors=False,
            world=_empty_world(),
            allowed_ops=("DERIVE_LICENSED",),
            issue_code="RELATIONAL_TRANSITIVITY",
        )
    # before_antisymmetric
    edges = (
        _edge("T0", "T1", "BEFORE"),
        _edge("T1", "T0", "BEFORE"),
    )
    return DirectionalityCase(
        mutation=mutation,
        licensed_causes_edges=(),
        licensed_before_edges=edges,
        expect_causes_skip=False,
        expect_causes_reverse=False,
        expect_before_transit=False,
        expect_relational_errors=True,
        world=_empty_world(),
        allowed_ops=("REJECT_SYMMETRIC_BEFORE",),
    )
