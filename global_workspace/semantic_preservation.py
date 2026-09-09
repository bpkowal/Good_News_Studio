"""Run-level source → action → world → admission → compact-role trace.

This is an observability artifact. It does not replace specialist JSON traces
or change world-state schema for deliberation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_identity import decompose_action_propositions
from .world_state import (
    explain_compact_role_assignments,
    world_model_from_dict,
)


def build_semantic_preservation_trace(
    *,
    scenario: str,
    clauses: Sequence[Mapping[str, Any]] = (),
    actions: Sequence[str] = (),
    grounding: Mapping[str, Any] | None = None,
    records: Sequence[Mapping[str, Any]] = (),
    user_authored: bool = False,
) -> dict[str, Any]:
    """One chain per world effect, plus repair deltas and compact-role reasons."""
    grounding = dict(grounding or {})
    clauses = [dict(row) for row in clauses or grounding.get("clauses") or []]
    clause_by_id = {
        str(row.get("clause_id") or ""): " ".join(str(row.get("text") or "").split())
        for row in clauses
        if row.get("clause_id")
    }
    world_raw = grounding.get("world_model") or {}
    typed = world_model_from_dict(world_raw) if world_raw else None
    action_by_id = {
        str(record.get("action_id") or ""): record for record in records
    }
    if not action_by_id and actions:
        action_by_id = {
            f"A{index}": {"canonical_semantic_action": action, "action_id": f"A{index}"}
            for index, action in enumerate(actions)
        }
    admitted_ids: set[str] = set()
    if typed is not None:
        admitted_ids = set(typed.admission.admitted_effect_ids) or {
            effect.effect_id for effect in typed.effects
        }
    role_rows: list[dict[str, Any]] = []
    if typed is not None:
        for action in typed.actions:
            for explanation in explain_compact_role_assignments(
                typed, action.action_id,
            ):
                role_rows.append({
                    "action_id": explanation.action_id,
                    "bucket": explanation.bucket,
                    "label": explanation.label,
                    "effect_id": explanation.effect_id,
                    "reason": explanation.reason,
                })
    role_by_effect = {row["effect_id"]: row for row in role_rows}
    chains: list[dict[str, Any]] = []
    if typed is not None:
        party_by_id = {party.party_id: party for party in typed.parties}
        for effect in typed.effects:
            sources = [
                {
                    "clause_id": ref.clause_id,
                    "text": clause_by_id.get(ref.clause_id) or ref.excerpt,
                }
                for ref in effect.provenance
            ]
            record = action_by_id.get(effect.action_id) or {}
            party = party_by_id.get(effect.party_id)
            chains.append({
                "source_proposition": sources,
                "canonical_action": {
                    "action_id": effect.action_id,
                    "text": record.get("canonical_semantic_action")
                    or record.get("intervention")
                    or "",
                },
                "world": {
                    "effect_id": effect.effect_id,
                    "party_id": effect.party_id,
                    "party_label": party.label if party is not None else "",
                    "outcome": effect.outcome,
                    "directness": effect.directness,
                    "effect_kind": effect.effect_kind,
                    "polarity": effect.polarity,
                    "modality": effect.modality,
                    "quantities": list(effect.quantities),
                    "likelihood_qualifiers": list(effect.likelihood_qualifiers),
                    "condition_ids": list(effect.condition_ids),
                    "condition_join": effect.condition_join,
                    "event_gates": [
                        {
                            "condition_id": condition.condition_id,
                            "event_effect_id": condition.event_effect_id,
                        }
                        for condition in typed.conditions
                        if condition.condition_id in effect.condition_ids
                        and condition.event_effect_id
                    ],
                },
                "admitted_status": (
                    "ADMITTED" if effect.effect_id in admitted_ids else "WITHHELD"
                ),
                "compact_role": role_by_effect.get(effect.effect_id),
            })
    action_propositions = []
    source_texts = [scenario, *clause_by_id.values()]
    for action_id, record in action_by_id.items():
        text = str(
            record.get("canonical_semantic_action")
            or record.get("intervention")
            or ""
        )
        action_propositions.append({
            "action_id": action_id,
            "text": text,
            "propositions": [
                {"kind": row.kind, "text": row.text, "supported": row.supported}
                for row in decompose_action_propositions(
                    text, source_texts, user_authored=user_authored,
                )
            ],
        })
    return {
        "status": str(grounding.get("status") or "UNKNOWN").upper(),
        "world_model_status": str(grounding.get("world_model_status") or ""),
        "repair_attempts": int(grounding.get("repair_attempts") or 0),
        "repairs": list(grounding.get("attempts") or []),
        "action_propositions": action_propositions,
        "compact_role_assignments": role_rows,
        "chains": chains,
    }


def render_semantic_preservation_trace(trace: Mapping[str, Any]) -> str:
    """Human checkpoint view of the source → world → role chain."""
    lines = [
        "",
        "--- Semantic preservation trace ---",
        f"status: {trace.get('status') or 'UNKNOWN'}",
        f"repairs: {trace.get('repair_attempts') or 0}",
    ]
    for chain in trace.get("chains") or []:
        sources = chain.get("source_proposition") or []
        source_text = "; ".join(
            f"{row.get('clause_id')}: {row.get('text')}" for row in sources
        ) or "(no source clause)"
        action = chain.get("canonical_action") or {}
        world = chain.get("world") or {}
        role = chain.get("compact_role") or {}
        role_text = (
            f"{role.get('bucket')}={role.get('label')}"
            if role else "none"
        )
        lines.append(
            f"{source_text} → {action.get('action_id')} "
            f"{action.get('text')} → {world.get('effect_id')} "
            f"{world.get('outcome')} [{world.get('party_label')}] → "
            f"{chain.get('admitted_status')} → {role_text}"
        )
        if role.get("reason"):
            lines.append(f"    role reason: {role['reason']}")
    for index, repair in enumerate(trace.get("repairs") or [], start=1):
        delta = repair.get("repair_delta") or {}
        dropped = delta.get("dropped") or {}
        if not dropped and not delta.get("illegal_drops"):
            continue
        lines.append(f"repair {index} delta: {json.dumps(delta, sort_keys=True)}")
    for row in trace.get("compact_role_assignments") or []:
        lines.append(
            f"role {row.get('action_id')} {row.get('bucket')}: "
            f"{row.get('label')} ({row.get('effect_id')}) because {row.get('reason')}"
        )
    return "\n".join(lines)


def write_semantic_preservation_trace(
    path: Path,
    trace: Mapping[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
