"""SAME_ENTITY_AS identity-closure Hypothesis cases with declared oracles.

Party aliases are data (party_id edges / shared labels), not discourse parses.
Oracle fields declare expected closure and whether cross-action harm transfer
is forbidden. RelEnt stays portable; Parliament adapters project edges.
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


_MUTATIONS = (
    "alias_pair",
    "chain_three",
    "quantity_along_identity",
    "harm_cross_branch",
)


@dataclass(frozen=True, slots=True)
class IdentityClosureCase:
    """Licensed identity graph over party_ids. Oracles are declared fields."""

    mutation: str
    licensed_identity_edges: tuple[dict, ...]
    expect_closed_pairs: tuple[tuple[str, str], ...]
    expect_quantity_transfer_ok: bool
    expect_harm_cross_action_ok: bool
    expect_relational_errors: bool
    world: ScenarioWorldModel
    issue_code: str = "RELATIONAL_IDENTITY_CLOSURE"
    repair_stage: str = "relational"
    allowed_ops: tuple[str, ...] = ("DERIVE_LICENSED",)
    forbidden_ops: tuple[str, ...] = ("CROSS_BRANCH_HARM", "SPLIT_IDENTITY")


def _edge(src: str, tgt: str, rel: str = "SAME_ENTITY_AS") -> dict:
    return {
        "source": src,
        "target": tgt,
        "relation": rel,
        "derived_marked": False,
    }


def _identity_world(
    *,
    mutation: str,
    qty: str,
    alias_label: str,
) -> tuple[ScenarioWorldModel, tuple[dict, ...], tuple[tuple[str, str], ...]]:
    ref = (SourceRef("C0", f"city populace also called {alias_label}"),)
    choice = (SourceRef("C1", "binary choice between purge and withhold"),)
    if mutation == "chain_three":
        parties = (
            WorldParty("P_a", "cohort-a", "POPULATION", ref),
            WorldParty("P_b", "cohort-b", "POPULATION", ref),
            WorldParty("P_c", "cohort-c", "POPULATION", ref),
            WorldParty("P_actor", "coordinator", "PERSON", choice),
        )
        edges = (
            _edge("P_a", "P_b"),
            _edge("P_b", "P_c", "EQUIVALENT_TO"),
        )
        expect_closed = (("P_a", "P_c"), ("P_c", "P_a"), ("P_b", "P_a"))
        effects = (
            WorldEffect(
                "E_a", "A0", "P_a", "cohort registered", "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "PHYSICAL_STATE",
                provenance=ref,
                source_proposition="cohort registered",
                derivation_operation="DIRECT_COPY",
            ),
        )
        actions = (
            WorldAction(
                "A0", "register cohort", "P_actor", ("P_a",), ("E_a",), ref,
            ),
        )
    elif mutation == "harm_cross_branch":
        # Shared label licenses SAME_ENTITY_AS without discourse parsing.
        parties = (
            WorldParty("P2", alias_label, "POPULATION", ref, (qty,)),
            WorldParty("P_alias", alias_label, "POPULATION", ref),
            WorldParty("P_actor", "coordinator", "PERSON", choice),
        )
        edges = ()
        expect_closed = (("P_alias", "P2"), ("P2", "P_alias"))
        effects = (
            WorldEffect(
                "E_surv", "A0", "P_alias", "immediate survival guaranteed",
                "STATE_CHANGE", "BENEFICIAL", "DOWNSTREAM", "CERTAIN",
                "HEALTH_OUTCOME",
                quantities=(qty,),
                provenance=choice,
                source_proposition="immediate survival guaranteed",
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E_death", "A1", "P2", "catastrophic loss of life", "DIES",
                "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME",
                quantities=(qty,),
                provenance=ref,
                source_proposition=f"risks {qty} of lives",
                derivation_operation="DIRECT_COPY",
            ),
        )
        actions = (
            WorldAction(
                "A0", "execute emergency purge",
                "P_actor", ("P_alias",), ("E_surv",), choice + ref,
            ),
            WorldAction(
                "A1", "withhold emergency purge",
                "P_actor", ("P2",), ("E_death",), choice + ref,
            ),
        )
    else:
        # alias_pair / quantity_along_identity — shared label projects identity
        parties = (
            WorldParty("P2", alias_label, "POPULATION", ref, (qty,)),
            WorldParty("P_alias", alias_label, "POPULATION", ref),
            WorldParty("P_actor", "coordinator", "PERSON", choice),
        )
        edges = ()  # label match projects SAME_ENTITY_AS
        expect_closed = (("P2", "P_alias"), ("P_alias", "P2"))
        effects = (
            WorldEffect(
                "E_qty", "A0", "P2", f"records {qty} of lives", "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "PHYSICAL_STATE",
                quantities=(qty,),
                provenance=ref,
                source_proposition=f"records {qty} of lives",
                derivation_operation="DIRECT_COPY",
            ),
        )
        actions = (
            WorldAction(
                "A0", "census populace",
                "P_actor", ("P2", "P_alias"), ("E_qty",), ref,
            ),
        )

    world = ScenarioWorldModel(
        schema_version="1.3",
        parties=parties,
        actions=actions,
        effects=effects,
    )
    return world, edges, expect_closed


@st.composite
def identity_closure_cases(draw) -> IdentityClosureCase:
    """Alias / chain / quantity / cross-branch harm identity oracles."""
    mutation = draw(st.sampled_from(_MUTATIONS))
    qty = draw(st.sampled_from(("thousands", "hundreds", "millions")))
    alias_label = draw(st.sampled_from(("residents", "city populace", "inhabitants")))
    # Avoid colliding with stop tokens used as party labels in other modes.
    if mutation in {"alias_pair", "quantity_along_identity"}:
        alias_label = draw(st.sampled_from(("residents", "inhabitants", "townsfolk")))
    world, edges, expect_closed = _identity_world(
        mutation=mutation,
        qty=qty,
        alias_label=alias_label,
    )
    return IdentityClosureCase(
        mutation=mutation,
        licensed_identity_edges=edges,
        expect_closed_pairs=expect_closed,
        expect_quantity_transfer_ok=mutation != "harm_cross_branch",
        expect_harm_cross_action_ok=False,
        expect_relational_errors=(mutation == "harm_cross_branch"),
        world=world,
        allowed_ops=(
            ("REJECT_CROSS_BRANCH_HARM",)
            if mutation == "harm_cross_branch"
            else ("DERIVE_LICENSED",)
        ),
    )
