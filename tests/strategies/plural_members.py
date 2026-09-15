"""Plural-member distinctness Hypothesis cases with declared oracles.

The oracle is PluralMemberCase.preserves_distinct (conjoined members remain
distinct PERSON parties). Silent merge into one collective party fails.

Mutation modes:
  keep            — three distinct PERSON parties
  merge_group     — one GROUP replaces the members
  drop_one        — only two of three members admitted
  single_person   — one PERSON party labeled with the full conjunction
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


_NAME_TRIPLES = (
    ("Smith", "Jones", "Anderson"),
    ("Lee", "Chen", "Patel"),
    ("Ng", "Kim", "Park"),
)
_MUTATIONS = ("keep", "merge_group", "drop_one", "single_person")


@dataclass(frozen=True, slots=True)
class PluralMemberCase:
    """Conjoined NP members. preserves_distinct is declared."""

    source: str
    member_party_ids: tuple[str, ...]
    preserves_distinct: bool
    mutation: str
    world: ScenarioWorldModel
    issue_code: str = "PLURAL_MEMBER_MERGED"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("RESOLVE_ENTITY",)
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE",)


def _plural_world(
    *,
    source: str,
    names: tuple[str, str, str],
    mutation: str,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    ids = ("P_a", "P_b", "P_c")
    if mutation == "keep":
        parties = tuple(
            WorldParty(party_id, name, "PERSON", ref)
            for party_id, name in zip(ids, names)
        ) + (WorldParty("P_actor", "coordinator", "PERSON", ref),)
        patient_ids = ids
        effect_party = ids[0]
    elif mutation == "merge_group":
        parties = (
            WorldParty("P_group", " and ".join(names), "GROUP", ref),
            WorldParty("P_actor", "coordinator", "PERSON", ref),
        )
        patient_ids = ("P_group",)
        effect_party = "P_group"
    elif mutation == "drop_one":
        parties = (
            WorldParty(ids[0], names[0], "PERSON", ref),
            WorldParty(ids[1], names[1], "PERSON", ref),
            WorldParty("P_actor", "coordinator", "PERSON", ref),
        )
        patient_ids = (ids[0], ids[1])
        effect_party = ids[0]
    else:
        # single_person: conjunction collapsed onto one PERSON label.
        parties = (
            WorldParty("P_one", " and ".join(names), "PERSON", ref),
            WorldParty("P_actor", "coordinator", "PERSON", ref),
        )
        patient_ids = ("P_one",)
        effect_party = "P_one"
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=parties,
        actions=(
            WorldAction(
                "A0", "protect the stranded parties",
                "P_actor", patient_ids,
                ("E0",), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E0", "A0",
                effect_party,
                "are stranded", "STATE_CHANGE",
                "ADVERSE", "DIRECT", "CERTAIN", "PHYSICAL_STATE",
                provenance=ref,
                source_proposition=source,
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def plural_member_cases(draw) -> PluralMemberCase:
    names = draw(st.sampled_from(_NAME_TRIPLES))
    mutation = draw(st.sampled_from(_MUTATIONS))
    preserves = mutation == "keep"
    source = f"{names[0]}, {names[1]} and {names[2]} are stranded."
    return PluralMemberCase(
        source=source,
        member_party_ids=("P_a", "P_b", "P_c"),
        preserves_distinct=preserves,
        mutation=mutation,
        world=_plural_world(
            source=source, names=names, mutation=mutation,
        ),
    )
