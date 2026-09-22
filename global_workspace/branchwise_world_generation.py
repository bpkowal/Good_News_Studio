"""Deterministically merge independently generated actual-world branches."""
from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence


def _fold(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def merge_binary_branch_worlds(
    branch_worlds: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge two single-action worlds without inventing cross-action semantics."""
    if len(branch_worlds) != 2:
        raise ValueError("binary branch merge requires exactly two worlds")
    merged: dict[str, Any] = {
        "schema_version": "1.3", "parties": [], "actions": [], "effects": [],
        "conditions": [], "temporal_relations": [], "causal_links": [],
        "counterfactual_links": [],
    }
    conflicts: list[dict[str, Any]] = []
    party_by_label: dict[str, dict[str, Any]] = {}

    for branch_index, source_world in enumerate(branch_worlds):
        world = copy.deepcopy(dict(source_world))
        action_id = f"A{branch_index}"
        prefix = f"B{branch_index}_"
        party_map: dict[str, str] = {}
        for party in world.get("parties") or []:
            if not isinstance(party, dict):
                continue
            old_id = str(party.get("party_id") or "")
            key = _fold(party.get("label"))
            existing = party_by_label.get(key) if key else None
            if existing is None:
                new_party = copy.deepcopy(party)
                new_party["party_id"] = f"P{len(merged['parties']) + 1}"
                merged["parties"].append(new_party)
                if key:
                    party_by_label[key] = new_party
                party_map[old_id] = new_party["party_id"]
            else:
                party_map[old_id] = str(existing["party_id"])
                if str(existing.get("kind") or "") != str(party.get("kind") or ""):
                    conflicts.append({
                        "type": "PARTY_KIND_CONFLICT", "label": party.get("label"),
                        "left": existing.get("kind"), "right": party.get("kind"),
                    })
                existing["quantities"] = list(dict.fromkeys([
                    *list(existing.get("quantities") or []),
                    *list(party.get("quantities") or []),
                ]))
                existing["clause_ids"] = list(dict.fromkeys([
                    *list(existing.get("clause_ids") or []),
                    *list(party.get("clause_ids") or []),
                ]))

        effect_map = {
            str(effect.get("effect_id") or ""): prefix + str(effect.get("effect_id") or "")
            for effect in world.get("effects") or [] if isinstance(effect, dict)
        }
        condition_map = {
            str(row.get("condition_id") or ""): prefix + str(row.get("condition_id") or "")
            for row in world.get("conditions") or [] if isinstance(row, dict)
        }
        for action in world.get("actions") or []:
            if not isinstance(action, dict):
                continue
            row = copy.deepcopy(action)
            row["action_id"] = action_id
            row["actor_party_id"] = party_map.get(
                str(row.get("actor_party_id") or ""), row.get("actor_party_id")
            )
            row["recipient_party_ids"] = [
                party_map.get(str(value), str(value))
                for value in row.get("recipient_party_ids") or []
            ]
            row["effect_ids"] = [
                effect_map.get(str(value), prefix + str(value))
                for value in row.get("effect_ids") or []
            ]
            merged["actions"].append(row)
            break
        for effect in world.get("effects") or []:
            if not isinstance(effect, dict):
                continue
            row = copy.deepcopy(effect)
            row["effect_id"] = effect_map[str(effect.get("effect_id") or "")]
            row["action_id"] = action_id
            row["party_id"] = party_map.get(str(row.get("party_id") or ""), row.get("party_id"))
            row["source_effect_ids"] = [
                effect_map.get(str(value), prefix + str(value))
                for value in row.get("source_effect_ids") or []
            ]
            row["condition_ids"] = [
                condition_map.get(str(value), prefix + str(value))
                for value in row.get("condition_ids") or []
            ]
            merged["effects"].append(row)
        for condition in world.get("conditions") or []:
            if not isinstance(condition, dict):
                continue
            row = copy.deepcopy(condition)
            row["condition_id"] = condition_map[str(condition.get("condition_id") or "")]
            event_id = str(row.get("event_effect_id") or "")
            row["event_effect_id"] = effect_map.get(event_id, prefix + event_id) if event_id else ""
            merged["conditions"].append(row)
        for link in world.get("causal_links") or []:
            if not isinstance(link, dict):
                continue
            row = copy.deepcopy(link)
            row["action_id"] = action_id
            row["source_id"] = effect_map.get(str(row.get("source_id") or ""), row.get("source_id"))
            row["target_id"] = effect_map.get(str(row.get("target_id") or ""), row.get("target_id"))
            row["condition_ids"] = [
                condition_map.get(str(value), prefix + str(value))
                for value in row.get("condition_ids") or []
            ]
            merged["causal_links"].append(row)
        for relation in world.get("temporal_relations") or []:
            if not isinstance(relation, dict):
                continue
            row = copy.deepcopy(relation)
            row["relation_id"] = prefix + str(row.get("relation_id") or "")
            for field in ("source_id", "target_id"):
                value = str(row.get(field) or "")
                row[field] = effect_map.get(value, condition_map.get(value, prefix + value))
            merged["temporal_relations"].append(row)
        if world.get("counterfactual_links"):
            conflicts.append({
                "type": "BRANCH_COUNTERFACTUALS_DEFERRED",
                "branch": action_id,
                "count": len(world.get("counterfactual_links") or []),
            })
    return merged, conflicts

