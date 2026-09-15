"""Attitude factivity Hypothesis cases with declared oracles.

The oracle is AttitudeFactivityCase.preserves_factivity (factive attitudes
license CERTAIN complements; non-factives must not force CERTAIN).
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


_ATTITUDES = (
    ("knew", True),
    ("realized", True),
    ("believed", False),
    ("suspected", False),
)
_COMPLEMENTS = (
    "won the contract",
    "secured the grant",
    "cleared the audit",
)
_NAMES = ("Smith", "Jones", "Lee", "Chen")
_ORGS = ("ITEL", "ACME", "ORION")


@dataclass(frozen=True, slots=True)
class AttitudeFactivityCase:
    """Attitude + complement. preserves_factivity is declared."""

    source: str
    attitude: str
    factive: bool
    complement_effect_id: str
    preserves_factivity: bool
    world: ScenarioWorldModel
    issue_code: str = "ATTITUDE_FACTIVITY_MISMATCH"
    repair_stage: str = "epistemic_binding"
    allowed_ops: tuple[str, ...] = ("VERIFY_PROVENANCE", "CHALLENGE_PREMISE")
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "COMPARE_FRAMEWORKS")


def _attitude_world(
    *,
    source: str,
    attitude: str,
    agent: str,
    org: str,
    complement: str,
    complement_modality: str,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P_agent", agent, "PERSON", ref),
            WorldParty("P_org", org, "INSTITUTION", ref),
            WorldParty("P_actor", "coordinator", "PERSON", ref),
        ),
        actions=(
            WorldAction(
                "A0", "record the attitude",
                "P_actor", ("P_agent",), ("E_att", "E_comp"), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E_att", "A0", "P_agent",
                f"{attitude} that {org} {complement}", "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "OTHER",
                provenance=ref,
                source_proposition=source,
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E_comp", "A0", "P_org", complement, "STATE_CHANGE",
                "BENEFICIAL", "DOWNSTREAM", complement_modality, "OTHER",
                provenance=ref,
                source_proposition=f"{org} {complement}",
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def attitude_factivity_cases(draw) -> AttitudeFactivityCase:
    attitude, factive = draw(st.sampled_from(_ATTITUDES))
    agent = draw(st.sampled_from(_NAMES))
    org = draw(st.sampled_from(_ORGS))
    complement = draw(st.sampled_from(_COMPLEMENTS))
    preserves = draw(st.booleans())
    if preserves:
        modality = "CERTAIN" if factive else "POSSIBLE"
    else:
        modality = "POSSIBLE" if factive else "CERTAIN"
    source = f"{agent} {attitude} that {org} had {complement}."
    return AttitudeFactivityCase(
        source=source,
        attitude=attitude,
        factive=factive,
        complement_effect_id="E_comp",
        preserves_factivity=preserves,
        world=_attitude_world(
            source=source,
            attitude=attitude,
            agent=agent,
            org=org,
            complement=complement,
            complement_modality=modality,
        ),
    )
