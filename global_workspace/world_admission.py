"""Parliament-owned orchestration for admitting and restoring world models.

This module is deliberately outside :mod:`relent`.  RelEnt supplies portable
semantic operators and relation checks; Parliament owns its world schema,
completeness policy, repair lifecycle, and admission statuses.  Dependency
direction is therefore always ``global_workspace -> relent``.

The first migration slice keeps behavior identical by delegating to the
existing implementation in ``world_state``.  Callers should use this facade so
normalization, validation, compilation, and admission can later be separated
without duplicating orchestration policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .world_state import (
    ScenarioWorldModel,
    compile_chance_gated_world,
    parse_world_model,
    validate_world_completeness,
    validate_world_model,
    world_model_as_parse_payload,
    world_model_from_dict,
)


@dataclass(frozen=True, slots=True)
class WorldValidationReport:
    """Host-policy validation result, separate from RelEnt kernel findings."""

    errors: tuple[str, ...] = ()
    contradictions: tuple[tuple[str, ...], ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


def admit_world_payload(
    payload: Any,
    *,
    clauses: Sequence[Mapping[str, Any]],
    action_ids: Sequence[str],
    action_texts: Mapping[str, str] | None = None,
    require_completeness: bool = True,
) -> ScenarioWorldModel:
    """Normalize, compile, and validate one Parliament world-model payload."""
    return parse_world_model(
        payload,
        clauses=clauses,
        action_ids=action_ids,
        action_texts=dict(action_texts or {}),
        require_completeness=require_completeness,
    )


def restore_admitted_world(payload: Any) -> ScenarioWorldModel | None:
    """Restore a serialized or already-typed admitted Parliament world."""
    return world_model_from_dict(payload)


def serialize_world_for_admission(model: ScenarioWorldModel) -> dict[str, Any]:
    """Serialize a typed world into the clause-ID payload accepted by admission."""
    return world_model_as_parse_payload(model)


_EXTENSION_ID_FIELDS = {
    "parties": "party_id",
    "effects": "effect_id",
    "conditions": "condition_id",
    "temporal_relations": "relation_id",
}


def _identifier(value: Any) -> str:
    return " ".join(str(value or "").split())[:80]


def admit_world_extension(
    base: ScenarioWorldModel,
    extension: Mapping[str, Any],
    *,
    clauses: Sequence[Mapping[str, Any]],
    action_ids: Sequence[str],
    action_texts: Mapping[str, str] | None = None,
) -> ScenarioWorldModel:
    """Append new world rows and re-admit the complete merged Parliament world.

    Existing identifiers are immutable.  RelEnt remains uninvolved in this
    lifecycle rule: append-only identity and completeness are host policy.
    """
    if not isinstance(extension, Mapping):
        raise ValueError("world-model extension must be an object")
    payload = serialize_world_for_admission(base)
    for key, id_field in _EXTENSION_ID_FIELDS.items():
        extra = extension.get(key) or []
        if extra and not isinstance(extra, list):
            raise ValueError(f"extension {key} must be an array")
        existing = {
            _identifier(row.get(id_field))
            for row in payload.get(key, [])
            if isinstance(row, dict)
        }
        for row in extra if isinstance(extra, list) else []:
            if not isinstance(row, dict):
                continue
            row_id = _identifier(row.get(id_field))
            if not row_id:
                raise ValueError(f"extension {key} row is missing {id_field}")
            if row_id in existing:
                raise ValueError(
                    f"extension reuses {id_field} {row_id}; committed "
                    "world-model ids are append-only"
                )
            payload.setdefault(key, []).append(dict(row))
            existing.add(row_id)
    for key in ("causal_links", "counterfactual_links"):
        extra = extension.get(key) or []
        if extra and not isinstance(extra, list):
            raise ValueError(f"extension {key} must be an array")
        payload.setdefault(key, []).extend(
            dict(row) for row in extra if isinstance(row, dict)
        )
    return admit_world_payload(
        payload,
        clauses=clauses,
        action_ids=action_ids,
        action_texts=action_texts,
        require_completeness=True,
    )


def normalize_admitted_world(model: ScenarioWorldModel) -> ScenarioWorldModel:
    """Apply Parliament's deterministic world compilation/normalization phase."""
    return compile_chance_gated_world(model)


def validate_admitted_world(
    model: ScenarioWorldModel,
    *,
    action_ids: Sequence[str],
    require_completeness: bool = True,
    source_texts: Sequence[str] = (),
) -> WorldValidationReport:
    """Evaluate structural and optional host-completeness admission policy."""
    errors, contradictions = validate_world_model(model, action_ids=action_ids)
    if require_completeness:
        errors.extend(validate_world_completeness(
            model,
            action_ids=action_ids,
            source_texts=source_texts,
        ))
    return WorldValidationReport(
        errors=tuple(errors),
        contradictions=tuple(tuple(group) for group in contradictions),
    )


def admitted_world_contradictions(
    model: ScenarioWorldModel,
    *,
    action_ids: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """Return contradiction groups after admission using the canonical gate."""
    return validate_admitted_world(
        model,
        action_ids=action_ids,
        require_completeness=False,
    ).contradictions


__all__ = (
    "admit_world_payload",
    "admit_world_extension",
    "admitted_world_contradictions",
    "normalize_admitted_world",
    "restore_admitted_world",
    "serialize_world_for_admission",
    "validate_admitted_world",
    "WorldValidationReport",
)
