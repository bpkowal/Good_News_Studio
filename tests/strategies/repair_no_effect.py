"""REPAIR_NO_EFFECT Hypothesis cases with declared oracles.

Oracle: ``expect_unstable``. Production helpers are only asked whether
``repair_no_effect_issues`` / ``annotate_admit_with_repair_no_effect`` /
``classify_world_repair_scope`` / ``should_skip_deterministic_local_patch``
agree. Do not infer instability from production fingerprints here — the
strategy writes the before/after/applied triple and the expected verdict.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st


_ISSUE_CODES = (
    "QUANTITY_BEARING_CONSEQUENCE_MISSING",
    "LIKELIHOOD_QUALIFIER_MISSING",
    "TEMPORAL_QUALIFIER_MISSING",
    "SCOPE_QUALIFIER_MISSING",
)

_EFFECT_IDS = ("E7", "E_loss", "E12", "E_surv")
_FIELDS = ("effects", "quantities", "likelihood_qualifiers", "temporal_qualifiers")

# Modes declare the oracle. Each builds a before/after/applied triple that
# must produce that verdict under the generic unstable-repair property.
_MODES = (
    "unstable_same_target",
    "cleared_after_patch",
    "empty_applied",
    "sibling_residual_only",
    "partial_clear_remaining_target",
)


@dataclass(frozen=True, slots=True)
class RepairNoEffectCase:
    """One repair(op)→validate fingerprint scenario with a declared oracle."""

    mode: str
    before_issues: tuple[dict[str, object], ...]
    after_issues: tuple[dict[str, object], ...]
    applied_patches: tuple[dict[str, object], ...]
    expect_unstable: bool
    unstable_entity_ids: tuple[str, ...]
    issue_code: str = "REPAIR_NO_EFFECT"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("FLAG_UNSTABLE_REPAIR",)
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE",)


def _issue(
    *,
    code: str,
    entity_id: str,
    field: str,
) -> dict[str, object]:
    return {
        "code": code,
        "message": f"{entity_id} fails {code}",
        "entity_kind": "effect" if entity_id.startswith("E") else "world_model",
        "entity_id": entity_id,
        "field": field,
        "related_ids": (),
        "repair_class": "SEMANTIC_PATCH",
        "permits_removal": False,
    }


def _patch(
    *,
    effect_id: str,
    code: str,
    op: str = "add_quantity",
) -> dict[str, object]:
    return {
        "op": op,
        "effect_id": effect_id,
        "field": "quantities",
        "value": "thousands",
        "code": code,
        "patch_kind": "semantic_patch",
        "deterministic": True,
    }


@st.composite
def repair_no_effect_cases(draw) -> RepairNoEffectCase:
    """Generate fingerprint triples. expect_unstable is written by mode."""
    mode = draw(st.sampled_from(_MODES))
    code = draw(st.sampled_from(_ISSUE_CODES))
    field = draw(st.sampled_from(_FIELDS))
    target = draw(st.sampled_from(_EFFECT_IDS))
    sibling = draw(st.sampled_from([e for e in _EFFECT_IDS if e != target]))

    target_issue = _issue(code=code, entity_id=target, field=field)
    sibling_issue = _issue(code=code, entity_id=sibling, field=field)
    target_patch = _patch(effect_id=target, code=code)

    if mode == "unstable_same_target":
        return RepairNoEffectCase(
            mode=mode,
            before_issues=(target_issue,),
            after_issues=(dict(target_issue),),
            applied_patches=(target_patch,),
            expect_unstable=True,
            unstable_entity_ids=(target,),
        )
    if mode == "cleared_after_patch":
        return RepairNoEffectCase(
            mode=mode,
            before_issues=(target_issue,),
            after_issues=(),
            applied_patches=(target_patch,),
            expect_unstable=False,
            unstable_entity_ids=(),
        )
    if mode == "empty_applied":
        return RepairNoEffectCase(
            mode=mode,
            before_issues=(target_issue,),
            after_issues=(dict(target_issue),),
            applied_patches=(),
            expect_unstable=False,
            unstable_entity_ids=(),
        )
    if mode == "sibling_residual_only":
        # Patch cleared the target; sibling still fails — not this repair.
        return RepairNoEffectCase(
            mode=mode,
            before_issues=(target_issue, sibling_issue),
            after_issues=(sibling_issue,),
            applied_patches=(target_patch,),
            expect_unstable=False,
            unstable_entity_ids=(),
        )
    # partial_clear_remaining_target: both failing before; patch hits target
    # only; target still fails after (sibling may clear or linger).
    sibling_clears = draw(st.booleans())
    after = (dict(target_issue),)
    if not sibling_clears:
        after = (dict(target_issue), sibling_issue)
    return RepairNoEffectCase(
        mode=mode,
        before_issues=(target_issue, sibling_issue),
        after_issues=after,
        applied_patches=(target_patch,),
        expect_unstable=True,
        unstable_entity_ids=(target,),
    )
