"""Qualifier-preservation Hypothesis cases with declared oracles.

Oracles live on QualifierPreservationCase.records_qualifier. Production is
only asked whether validate_world_model agrees that a source-bound likelihood,
temporal, or scope span either survives on the modified effect or hard-fails
as the matching *_QUALIFIER_MISSING issue.

Placement mutations:
  effect   — span on the bound welfare effect
  omit     — span dropped entirely
  sibling  — span hung on the intervention sibling (misassignment)
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


# (channel, source template with {span}/{outcome}, span, outcome, issue_code, field)
_QUALIFIER_FORMS = (
    (
        "likelihood",
        "A cyberattack risks {span} {outcome} through infrastructure failure.",
        "near-certain",
        "catastrophic loss of life",
        "LIKELIHOOD_QUALIFIER_MISSING",
        "likelihood_qualifiers",
    ),
    (
        "likelihood",
        "A cyberattack creates a {span} of {outcome}.",
        "remote chance",
        "escape",
        "LIKELIHOOD_QUALIFIER_MISSING",
        "likelihood_qualifiers",
    ),
    (
        "temporal",
        "A cyberattack causes {span} {outcome}.",
        "immediate",
        "catastrophic loss of life",
        "TEMPORAL_QUALIFIER_MISSING",
        "temporal_qualifiers",
    ),
    (
        "temporal",
        "A cyberattack causes {span} {outcome} of residents.",
        "immediate",
        "death",
        "TEMPORAL_QUALIFIER_MISSING",
        "temporal_qualifiers",
    ),
    (
        "scope",
        "{span} {outcome} follows the cyberattack.",
        "widespread",
        "catastrophic loss of life",
        "SCOPE_QUALIFIER_MISSING",
        "scope_qualifiers",
    ),
)

_QUALIFIER_PLACEMENTS = ("effect", "omit", "sibling")

_TOKEN_STOP = frozenset({
    "catastrophic", "certain", "chance", "cyberattack", "death", "escape",
    "follows", "immediate", "life", "loss", "near", "remote", "risks",
    "widespread",
})


@dataclass(frozen=True, slots=True)
class QualifierPreservationCase:
    """One source-bound qualifier. records_qualifier is declared."""

    channel: str
    source: str
    span: str
    outcome: str
    records_qualifier: bool
    placement: str
    effect_id: str
    sibling_effect_id: str
    issue_code: str
    field: str
    world: ScenarioWorldModel
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ()
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "COMPARE_FRAMEWORKS")


def _labels(draw, count: int) -> tuple[str, ...]:
    return draw(unique_tokens(count).filter(
        lambda tokens: not any(token in _TOKEN_STOP for token in tokens)
    ))


def _qualifier_world(
    *,
    source: str,
    outcome: str,
    actor: str,
    field: str,
    span: str,
    placement: str,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    on_target = (span,) if placement == "effect" else ()
    on_sibling = (span,) if placement == "sibling" else ()
    modality = "CERTAIN"
    if field == "likelihood_qualifiers" and placement == "effect":
        modality = "PROBABILISTIC"
    target_kwargs: dict[str, object] = {field: on_target}
    sibling_kwargs: dict[str, object] = {field: on_sibling}
    return ScenarioWorldModel(
        schema_version="1.3",
        parties=(
            WorldParty("P0", actor, "HUMAN", ref),
            WorldParty("P2", "populace", "POPULATION", ref),
        ),
        actions=(
            WorldAction(
                "A0", "withhold emergency purge",
                "P0", ("P2",), ("E_loss", "E_direct"), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E_direct", "A0", "P2", "withhold emergency purge",
                "STATE_CHANGE", "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ref,
                source_proposition=source,
                derivation_operation="DIRECT_COPY",
                **sibling_kwargs,
            ),
            WorldEffect(
                "E_loss", "A0", "P2", outcome, "DIES",
                "ADVERSE", "DOWNSTREAM", modality, "HEALTH_OUTCOME",
                provenance=ref,
                source_proposition=source,
                derivation_operation="DIRECT_COPY",
                **target_kwargs,
            ),
        ),
    )


@st.composite
def qualifier_preservation_cases(draw) -> QualifierPreservationCase:
    """Preserve, drop, or mis-hang a source-bound qualifier."""
    actor = _labels(draw, 1)[0]
    channel, template, span, outcome, issue_code, field = draw(
        st.sampled_from(_QUALIFIER_FORMS)
    )
    placement = draw(st.sampled_from(_QUALIFIER_PLACEMENTS))
    records = placement == "effect"
    source = template.format(span=span, outcome=outcome)
    op = {
        "likelihood_qualifiers": "add_likelihood_qualifier",
        "temporal_qualifiers": "add_temporal_qualifier",
        "scope_qualifiers": "add_scope_qualifier",
    }[field]
    return QualifierPreservationCase(
        channel=channel,
        source=source,
        span=span,
        outcome=outcome,
        records_qualifier=records,
        placement=placement,
        effect_id="E_loss",
        sibling_effect_id="E_direct",
        issue_code=issue_code,
        field=field,
        world=_qualifier_world(
            source=source,
            outcome=outcome,
            actor=actor,
            field=field,
            span=span,
            placement=placement,
        ),
        allowed_ops=(op,),
    )
