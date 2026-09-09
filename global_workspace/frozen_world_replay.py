from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .world_state import world_model_from_dict


class FrozenWorldReplayError(ValueError):
    """A saved trace cannot safely serve as an immutable admitted-world fixture."""


def _required_list(payload: dict[str, Any], name: str) -> list[Any]:
    value = payload.get(name)
    if not isinstance(value, list) or not value:
        raise FrozenWorldReplayError(f"frozen trace requires a non-empty {name} list")
    return value


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenWorldReplay:
    source_path: Path
    scenario: str
    presentation_actions: tuple[str, ...]
    canonical_actions: tuple[str, ...]
    action_source_grounding: dict[str, Any]
    canonical_action_records: tuple[dict[str, Any], ...]
    presentation_action_mapping: tuple[dict[str, Any], ...]
    source_action_legend: dict[str, str]
    fingerprint: str

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "FROZEN_WORLD_TRACE",
            "source_path": str(self.source_path),
            "fingerprint_sha256": self.fingerprint,
            "scenario_match": True,
            "action_identity_match": True,
            "world_generation_calls": 0,
            "validated": True,
        }


def load_frozen_world_trace(
    path: Path,
    *,
    expected_scenario: str,
) -> FrozenWorldReplay:
    """Load a committed trace as a fail-closed world-state replay fixture.

    Unlike the convenience framing cache, this path never repairs, regenerates,
    or accepts a near match. It is intended for repeatable specialist and
    governance development against exactly one admitted world.
    """
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenWorldReplayError(
            f"could not read frozen world trace {resolved}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FrozenWorldReplayError("frozen trace root must be a JSON object")

    scenario = str(payload.get("scenario") or "")
    if scenario != expected_scenario:
        raise FrozenWorldReplayError(
            "frozen trace scenario does not exactly match the requested scenario"
        )

    presentation_actions = _required_list(payload, "presentation_actions")
    canonical_actions = _required_list(payload, "actions")
    if not 2 <= len(canonical_actions) <= 5:
        raise FrozenWorldReplayError("frozen trace must contain 2-5 canonical actions")
    if (
        len(presentation_actions) != len(canonical_actions)
        or not all(isinstance(item, str) and item.strip() for item in presentation_actions)
        or not all(isinstance(item, str) and item.strip() for item in canonical_actions)
    ):
        raise FrozenWorldReplayError("frozen trace action lists are malformed")

    grounding = payload.get("action_source_grounding")
    if not isinstance(grounding, dict):
        raise FrozenWorldReplayError("frozen trace has no action-source grounding object")
    if str(grounding.get("status") or "").upper() != "COMMITTED":
        raise FrozenWorldReplayError("frozen action-source grounding is not COMMITTED")
    if str(grounding.get("world_model_status") or "").upper() != "COMMITTED":
        raise FrozenWorldReplayError("frozen typed world model is not COMMITTED")
    if grounding.get("world_contradictions"):
        raise FrozenWorldReplayError("frozen trace contains unresolved world contradictions")

    world_payload = grounding.get("world_model")
    if not isinstance(world_payload, dict):
        raise FrozenWorldReplayError("frozen trace has no typed world model")
    try:
        world = world_model_from_dict(world_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenWorldReplayError(
            f"frozen typed world could not be restored: {type(exc).__name__}: {exc}"
        ) from exc
    if world is None:
        raise FrozenWorldReplayError("frozen typed world could not be restored")
    admission = world.admission
    if str(admission.status).upper() != "COMMITTED":
        raise FrozenWorldReplayError("frozen typed world admission is not COMMITTED")
    if admission.user_override or admission.quarantined_effects:
        raise FrozenWorldReplayError(
            "frozen typed world must not depend on an override or quarantined effects"
        )

    expected_action_ids = [f"A{index}" for index in range(len(canonical_actions))]
    world_action_ids = [action.action_id for action in world.actions]
    if world_action_ids != expected_action_ids:
        raise FrozenWorldReplayError(
            "frozen typed-world action IDs do not match canonical action order"
        )
    grounding_action_ids = sorted(str(key) for key in (grounding.get("actions") or {}))
    if grounding_action_ids != expected_action_ids:
        raise FrozenWorldReplayError(
            "frozen grounding action IDs do not cover every canonical action"
        )

    effect_ids = [effect.effect_id for effect in world.effects]
    if len(effect_ids) != len(set(effect_ids)):
        raise FrozenWorldReplayError("frozen typed world contains duplicate effect IDs")
    if set(admission.admitted_effect_ids) != set(effect_ids):
        raise FrozenWorldReplayError(
            "frozen admission does not admit exactly the typed-world effects"
        )
    effects_by_action = {
        action.action_id: [effect.effect_id for effect in world.effects_for(action.action_id)]
        for action in world.actions
    }
    for action in world.actions:
        if set(action.effect_ids) != set(effects_by_action[action.action_id]):
            raise FrozenWorldReplayError(
                f"frozen {action.action_id} effect IDs disagree with its typed effects"
            )

    records = _required_list(payload, "canonical_action_records")
    if len(records) != len(canonical_actions) or not all(
        isinstance(record, dict) for record in records
    ):
        raise FrozenWorldReplayError("frozen canonical action records are malformed")
    records_by_id = {str(record.get("action_id") or ""): record for record in records}
    if sorted(records_by_id) != expected_action_ids:
        raise FrozenWorldReplayError(
            "frozen canonical records do not cover every canonical action"
        )
    for index, canonical_action in enumerate(canonical_actions):
        action_id = f"A{index}"
        record = records_by_id[action_id]
        if record.get("canonical_semantic_action") != canonical_action:
            raise FrozenWorldReplayError(
                f"frozen {action_id} record changes its canonical action meaning"
            )
        record_effect_ids = {
            str(effect.get("effect_id") or "")
            for effect in record.get("world_effects", [])
            if isinstance(effect, dict)
        }
        if record_effect_ids != set(effects_by_action[action_id]):
            raise FrozenWorldReplayError(
                f"frozen {action_id} record does not preserve its typed-world effects"
            )

    fingerprint_payload = {
        "scenario": scenario,
        "presentation_actions": presentation_actions,
        "canonical_actions": canonical_actions,
        "grounding_actions": grounding.get("actions"),
        "clauses": grounding.get("clauses"),
        "world_model": world_payload,
        "canonical_action_records": records,
    }
    return FrozenWorldReplay(
        source_path=resolved,
        scenario=scenario,
        presentation_actions=tuple(presentation_actions),
        canonical_actions=tuple(canonical_actions),
        action_source_grounding=copy.deepcopy(grounding),
        canonical_action_records=tuple(copy.deepcopy(records)),
        presentation_action_mapping=tuple(copy.deepcopy(
            payload.get("presentation_action_mapping") or []
        )),
        source_action_legend=copy.deepcopy(payload.get("source_action_legend") or {}),
        fingerprint=_canonical_fingerprint(fingerprint_payload),
    )
