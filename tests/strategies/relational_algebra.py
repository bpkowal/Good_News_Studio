"""RELATIONAL_ENTAILMENT Hypothesis cases over synthetic edge graphs.

Oracle: ``expect_errors``. Production closure / transfer only asked whether
they agree. Nodes are abstract A/B/C — no live-dilemma nouns.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from relent.algebra import RelEdge


@dataclass(frozen=True, slots=True)
class RelationalAlgebraCase:
    mode: str
    edges: tuple[dict, ...]
    expect_derived: tuple[dict, ...]
    forbid_derived: tuple[dict, ...]
    transfer_relation: str
    transfer_kind: str
    transfer_derivation_marked: bool
    transfer_source_action: str
    transfer_target_action: str
    expect_transfer_ok: bool
    expect_errors: bool
    issue_code: str = "RELATIONAL_ENTAILMENT"
    repair_stage: str = "relational"
    allowed_ops: tuple[str, ...] = ("DERIVE_LICENSED",)
    forbidden_ops: tuple[str, ...] = ("FREE_COMPOSITE", "CROSS_BRANCH_HARM")


def _edge(src: str, tgt: str, rel: str, *, derived: bool = False) -> dict:
    return {
        "source": src,
        "target": tgt,
        "relation": rel,
        "derived_marked": derived,
    }


@st.composite
def relational_algebra_cases(draw) -> RelationalAlgebraCase:
    mode = draw(st.sampled_from((
        "identity_closure",
        "alternative_symmetry",
        "causes_no_free_transitivity",
        "alternative_not_identity",
        "quantity_along_identity",
        "quantity_alt_needs_derivation",
        "harm_no_cross_alternative",
        "harm_no_cross_identity",
        "before_transits",
        "before_antisymmetric",
    )))
    if mode == "identity_closure":
        edges = (
            _edge("A", "B", "SAME_ENTITY_AS"),
            _edge("B", "C", "EQUIVALENT_TO"),
        )
        return RelationalAlgebraCase(
            mode=mode,
            edges=edges,
            expect_derived=(_edge("A", "C", "SAME_ENTITY_AS"),),
            forbid_derived=(),
            transfer_relation="SAME_ENTITY_AS",
            transfer_kind="quantity",
            transfer_derivation_marked=False,
            transfer_source_action="",
            transfer_target_action="",
            expect_transfer_ok=True,
            expect_errors=False,
        )
    if mode == "alternative_symmetry":
        edges = (_edge("A0", "A1", "ALTERNATIVE_OF"),)
        return RelationalAlgebraCase(
            mode=mode,
            edges=edges,
            expect_derived=(_edge("A1", "A0", "ALTERNATIVE_OF"),),
            forbid_derived=(),
            transfer_relation="ALTERNATIVE_OF",
            transfer_kind="quantity",
            transfer_derivation_marked=False,
            transfer_source_action="A0",
            transfer_target_action="A1",
            expect_transfer_ok=False,
            expect_errors=False,
        )
    if mode == "causes_no_free_transitivity":
        edges = (
            _edge("A", "B", "CAUSES"),
            _edge("B", "C", "CAUSES"),
        )
        return RelationalAlgebraCase(
            mode=mode,
            edges=edges,
            expect_derived=(),
            forbid_derived=(_edge("A", "C", "CAUSES"),),
            transfer_relation="CAUSES",
            transfer_kind="status",
            transfer_derivation_marked=False,
            transfer_source_action="",
            transfer_target_action="",
            expect_transfer_ok=False,
            expect_errors=False,
            allowed_ops=("REJECT_FREE_CAUSAL_HOP",),
        )
    if mode == "alternative_not_identity":
        edges = (
            _edge("A0", "A1", "ALTERNATIVE_OF"),
            _edge("A0", "A1", "SAME_ENTITY_AS"),
        )
        return RelationalAlgebraCase(
            mode=mode,
            edges=edges,
            expect_derived=(),
            forbid_derived=(),
            transfer_relation="ALTERNATIVE_OF",
            transfer_kind="harm_under_action",
            transfer_derivation_marked=False,
            transfer_source_action="A0",
            transfer_target_action="A1",
            expect_transfer_ok=False,
            expect_errors=True,
            allowed_ops=("REJECT_ALT_AS_IDENTITY",),
        )
    if mode == "quantity_along_identity":
        edges = (_edge("P2", "residents", "SAME_ENTITY_AS"),)
        return RelationalAlgebraCase(
            mode=mode,
            edges=edges,
            expect_derived=(_edge("residents", "P2", "SAME_ENTITY_AS"),),
            forbid_derived=(),
            transfer_relation="SAME_ENTITY_AS",
            transfer_kind="quantity",
            transfer_derivation_marked=False,
            transfer_source_action="A1",
            transfer_target_action="A1",
            expect_transfer_ok=True,
            expect_errors=False,
        )
    if mode == "quantity_alt_needs_derivation":
        edges = (_edge("A0", "A1", "ALTERNATIVE_OF"),)
        return RelationalAlgebraCase(
            mode=mode,
            edges=edges,
            expect_derived=(_edge("A1", "A0", "ALTERNATIVE_OF"),),
            forbid_derived=(),
            transfer_relation="ALTERNATIVE_OF",
            transfer_kind="quantity",
            transfer_derivation_marked=False,
            transfer_source_action="A1",
            transfer_target_action="A0",
            expect_transfer_ok=False,
            expect_errors=False,
        )
    if mode == "harm_no_cross_identity":
        edges = (_edge("P2", "residents", "SAME_ENTITY_AS"),)
        return RelationalAlgebraCase(
            mode=mode,
            edges=edges,
            expect_derived=(_edge("residents", "P2", "SAME_ENTITY_AS"),),
            forbid_derived=(),
            transfer_relation="SAME_ENTITY_AS",
            transfer_kind="harm_under_action",
            transfer_derivation_marked=False,
            transfer_source_action="A1",
            transfer_target_action="A0",
            expect_transfer_ok=False,
            expect_errors=False,
            allowed_ops=("REJECT_CROSS_BRANCH_HARM",),
        )
    if mode == "before_transits":
        edges = (
            _edge("T0", "T1", "BEFORE"),
            _edge("T1", "T2", "BEFORE"),
        )
        return RelationalAlgebraCase(
            mode=mode,
            edges=edges,
            expect_derived=(_edge("T0", "T2", "BEFORE"),),
            forbid_derived=(_edge("T1", "T0", "BEFORE"),),
            transfer_relation="BEFORE",
            transfer_kind="status",
            transfer_derivation_marked=False,
            transfer_source_action="",
            transfer_target_action="",
            expect_transfer_ok=False,
            expect_errors=False,
            allowed_ops=("DERIVE_LICENSED",),
        )
    if mode == "before_antisymmetric":
        edges = (
            _edge("T0", "T1", "BEFORE"),
            _edge("T1", "T0", "BEFORE"),
        )
        return RelationalAlgebraCase(
            mode=mode,
            edges=edges,
            expect_derived=(),
            forbid_derived=(),
            transfer_relation="BEFORE",
            transfer_kind="status",
            transfer_derivation_marked=False,
            transfer_source_action="",
            transfer_target_action="",
            expect_transfer_ok=False,
            expect_errors=True,
            allowed_ops=("REJECT_SYMMETRIC_BEFORE",),
        )
    # harm_no_cross_alternative
    edges = (_edge("A0", "A1", "ALTERNATIVE_OF"),)
    return RelationalAlgebraCase(
        mode=mode,
        edges=edges,
        expect_derived=(),
        forbid_derived=(),
        transfer_relation="ALTERNATIVE_OF",
        transfer_kind="harm_under_action",
        transfer_derivation_marked=True,
        transfer_source_action="A1",
        transfer_target_action="A0",
        expect_transfer_ok=False,
        expect_errors=False,
        allowed_ops=("REJECT_CROSS_BRANCH_HARM",),
    )


def edges_from_case(case: RelationalAlgebraCase) -> tuple[RelEdge, ...]:
    return tuple(
        RelEdge(
            source=row["source"],
            target=row["target"],
            relation=row["relation"],
            derived_marked=bool(row.get("derived_marked")),
        )
        for row in case.edges
    )
