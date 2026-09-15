"""Anaphor entity-identity Hypothesis cases with declared oracles.

The oracle is AnaphorIdentityCase.preserves_identity (same party_id on
antecedent and anaphor effects). Production admission is not yet a live
validator for this family; Hypothesis asks the Parliament harness oracle
whether generated keep/split worlds agree with the declared flag.

Mutation modes stress silent identity failure shapes without promoting a
production admit gate.
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


_ANTECEDENT_NAMES = ("Smith", "Jones", "Lee", "Chen", "Patel", "Ng")
_ANAPHOR_FORMS = (
    "pronoun_they",
    "pronoun_he",
    "pronoun_she",
    "name_repeat",
    "demonstrative",
)
_MUTATIONS = (
    "keep",
    "split_intruder",
    "split_group_kind",
    "chain_wrong_antecedent",
)
_TOKEN_STOP = frozenset({
    "and", "he", "owns", "person", "she", "smith", "the", "they", "used",
    "workstation",
})


@dataclass(frozen=True, slots=True)
class AnaphorIdentityCase:
    """Antecedent + anaphor effects. preserves_identity is declared."""

    ante_source: str
    ana_source: str
    antecedent_effect_id: str
    anaphor_effect_id: str
    antecedent_party_id: str
    anaphor_party_id: str
    preserves_identity: bool
    anaphor_form: str
    mutation: str
    world: ScenarioWorldModel
    issue_code: str = "ANAPHOR_ENTITY_SPLIT"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("RESOLVE_ENTITY",)
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "COMPARE_FRAMEWORKS")


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _anaphor_clause(*, form: str, name: str, obj: str) -> str:
    if form == "pronoun_they":
        return f"They used the {obj}."
    if form == "pronoun_he":
        return f"He used the {obj}."
    if form == "pronoun_she":
        return f"She used the {obj}."
    if form == "demonstrative":
        return f"That person used the {obj}."
    return f"{name} used the {obj}."


def _identity_world(
    *,
    ante_source: str,
    ana_source: str,
    name: str,
    obj: str,
    mutation: str,
    intruder: str,
) -> ScenarioWorldModel:
    ante_ref = (SourceRef("C0", ante_source),)
    ana_ref = (SourceRef("C1", ana_source),)
    ante_party = "P_ante"
    parties = [
        WorldParty(ante_party, name, "PERSON", ante_ref),
        WorldParty("P_actor", "coordinator", "PERSON", ante_ref),
    ]
    if mutation == "keep":
        ana_party = ante_party
    elif mutation == "split_group_kind":
        ana_party = "P_other"
        parties.append(WorldParty(ana_party, name, "GROUP", ana_ref))
    elif mutation == "chain_wrong_antecedent":
        ana_party = "P_other"
        parties.append(
            WorldParty("P_decoy", f"{intruder}", "PERSON", ante_ref),
        )
        parties.append(WorldParty(ana_party, intruder, "PERSON", ana_ref))
    else:
        # split_intruder
        ana_party = "P_other"
        parties.append(WorldParty(ana_party, intruder, "PERSON", ana_ref))
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=tuple(parties),
        actions=(
            WorldAction(
                "A0", f"protect {name}",
                "P_actor", (ante_party,), ("E_ante", "E_ana"),
                ante_ref + ana_ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E_ante", "A0", ante_party, f"owns a {obj}", "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "PHYSICAL_STATE",
                provenance=ante_ref,
                source_proposition=ante_source,
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E_ana", "A0", ana_party, f"used the {obj}", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                provenance=ana_ref,
                source_proposition=ana_source,
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def anaphor_identity_cases(draw) -> AnaphorIdentityCase:
    """Keep vs mutated party_id under pronominal / name-repeat anaphora."""
    name = draw(st.sampled_from(_ANTECEDENT_NAMES))
    obj, intruder = _labels(draw, 2)
    form = draw(st.sampled_from(_ANAPHOR_FORMS))
    mutation = draw(st.sampled_from(_MUTATIONS))
    preserves = mutation == "keep"
    ante_source = f"{name} owns a {obj}."
    ana_source = _anaphor_clause(form=form, name=name, obj=obj)
    world = _identity_world(
        ante_source=ante_source,
        ana_source=ana_source,
        name=name,
        obj=obj,
        mutation=mutation,
        intruder=intruder.capitalize(),
    )
    ante_party = "P_ante"
    ana_party = ante_party if preserves else "P_other"
    return AnaphorIdentityCase(
        ante_source=ante_source,
        ana_source=ana_source,
        antecedent_effect_id="E_ante",
        anaphor_effect_id="E_ana",
        antecedent_party_id=ante_party,
        anaphor_party_id=ana_party,
        preserves_identity=preserves,
        anaphor_form=form,
        mutation=mutation,
        world=world,
    )
