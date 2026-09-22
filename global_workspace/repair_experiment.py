"""Explicit experimental records for world-model repair transitions."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


REPAIR_EVENT_SCHEMA_VERSION = 1
VALIDATOR_VERSION = "world-validator-2026-09-21-v1"
REPAIR_PROMPT_VERSION = "world-repair-prompt-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:length]


def _issues(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def issue_instances(
    rows: Sequence[Mapping[str, Any]], *, event_id: str, phase: str,
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for ordinal, row in enumerate(_issues(rows), 1):
        identity = {
            "code": str(row.get("code") or "WORLD_VALIDATION_ERROR"),
            "entity_kind": str(row.get("entity_kind") or "world_model"),
            "entity_id": str(row.get("entity_id") or ""),
            "field": str(row.get("field") or ""),
            "message": str(row.get("message") or ""),
        }
        instances.append({
            "issue_instance_id": f"{event_id}:{phase}:I{ordinal}:{_digest(identity, 10)}",
            **identity,
            "invariant_version": int(row.get("invariant_version") or 1),
            "repair_class": str(row.get("repair_class") or "UNCLASSIFIED"),
            "related_ids": list(row.get("related_ids") or []),
        })
    return instances


def _world(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        return {}
    world = candidate.get("world_model")
    return dict(world) if isinstance(world, Mapping) else {}


def graph_fingerprint(candidate: Any) -> str | None:
    world = _world(candidate)
    return _digest(world, 24) if world else None


_COLLECTION_IDS = {
    "parties": "party_id",
    "actions": "action_id",
    "effects": "effect_id",
    "conditions": "condition_id",
    "temporal_relations": "relation_id",
}


def _indexed(world: Mapping[str, Any], collection: str, id_field: str) -> dict[str, Any]:
    return {
        str(row.get(id_field) or f"ordinal:{index}"): row
        for index, row in enumerate(world.get(collection) or [])
        if isinstance(row, Mapping)
    }


def _edge_key(row: Mapping[str, Any], collection: str, index: int) -> str:
    if collection == "causal_links":
        fields = ("action_id", "source_id", "link_relation", "target_id")
    else:
        fields = (
            "action_id", "source_effect_id", "relation",
            "alternative_action_id", "alternative_effect_id",
        )
    values = [str(row.get(field) or "") for field in fields]
    return "|".join(values) if any(values) else f"ordinal:{index}"


def graph_diff(before_candidate: Any, after_candidate: Any) -> dict[str, Any]:
    before = _world(before_candidate)
    after = _world(after_candidate)
    diff: dict[str, Any] = {
        "before_fingerprint": graph_fingerprint(before_candidate),
        "after_fingerprint": graph_fingerprint(after_candidate),
        "collections": {},
        "nodes_added": [],
        "nodes_removed": [],
        "nodes_changed": [],
        "edges_added": [],
        "edges_removed": [],
        "certainty_changes": [],
        "ownership_changes": [],
        "provenance_changes": [],
    }
    for collection, id_field in _COLLECTION_IDS.items():
        left = _indexed(before, collection, id_field)
        right = _indexed(after, collection, id_field)
        added = sorted(set(right) - set(left))
        removed = sorted(set(left) - set(right))
        changed: list[dict[str, Any]] = []
        for identifier in sorted(set(left) & set(right)):
            if _canonical(left[identifier]) == _canonical(right[identifier]):
                continue
            fields = sorted({
                key for key in {*left[identifier], *right[identifier]}
                if left[identifier].get(key) != right[identifier].get(key)
            })
            changed.append({"id": identifier, "fields": fields})
            before_row, after_row = left[identifier], right[identifier]
            if any(field in fields for field in ("modality", "likelihood_qualifiers")):
                diff["certainty_changes"].append(identifier)
            if any(field in fields for field in ("action_id", "party_id", "recipient_party_ids")):
                diff["ownership_changes"].append(identifier)
            if any(field in fields for field in ("clause_ids", "provenance", "source_proposition")):
                diff["provenance_changes"].append(identifier)
        diff["collections"][collection] = {
            "added": added, "removed": removed, "changed": changed,
        }
        diff["nodes_added"].extend(f"{collection}:{value}" for value in added)
        diff["nodes_removed"].extend(f"{collection}:{value}" for value in removed)
        diff["nodes_changed"].extend(
            {"node": f"{collection}:{row['id']}", "fields": row["fields"]}
            for row in changed
        )
    for collection in ("causal_links", "counterfactual_links"):
        left = {
            _edge_key(row, collection, index): row
            for index, row in enumerate(before.get(collection) or [])
            if isinstance(row, Mapping)
        }
        right = {
            _edge_key(row, collection, index): row
            for index, row in enumerate(after.get(collection) or [])
            if isinstance(row, Mapping)
        }
        added = sorted(set(right) - set(left))
        removed = sorted(set(left) - set(right))
        diff["collections"][collection] = {"added": added, "removed": removed}
        diff["edges_added"].extend(f"{collection}:{value}" for value in added)
        diff["edges_removed"].extend(f"{collection}:{value}" for value in removed)
    return diff


def _card_records(
    contract: Mapping[str, Any], *, selected: bool,
) -> list[dict[str, Any]]:
    return [{
        "card_id": str(card.get("card_id") or f"{card.get('code', 'ISSUE')}_V1"),
        "card_version": int(card.get("card_version") or 1),
        "issue_code": str(card.get("code") or "WORLD_VALIDATION_ERROR"),
        "entity_id": str(card.get("entity_id") or ""),
        "selected": selected,
    } for card in contract.get("guidance_cards") or [] if isinstance(card, Mapping)]


def build_repair_event(
    *,
    event_index: int,
    execution_mode: str,
    repair_scope: str,
    before_issues: Sequence[Mapping[str, Any]],
    after_issues: Sequence[Mapping[str, Any]],
    before_candidate: Any,
    after_candidate: Any,
    repair_contract: Mapping[str, Any] | None = None,
    terminal_status: str = "REJECTED",
    deterministic_patches: Sequence[Mapping[str, Any]] = (),
    context: Mapping[str, Any] | None = None,
    cards_presented: bool = True,
) -> dict[str, Any]:
    event_id = f"R{event_index:03d}"
    before_instances = issue_instances(before_issues, event_id=event_id, phase="BEFORE")
    after_instances = issue_instances(after_issues, event_id=event_id, phase="AFTER")
    def signature(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("code") or "WORLD_VALIDATION_ERROR"),
            str(row.get("entity_kind") or "world_model"),
            str(row.get("entity_id") or ""),
            str(row.get("field") or ""),
        )

    before_by_signature = {signature(row): row for row in before_instances}
    after_by_signature = {signature(row): row for row in after_instances}
    remaining_signatures = sorted(set(before_by_signature) & set(after_by_signature))
    resolved_signatures = sorted(set(before_by_signature) - set(after_by_signature))
    introduced_signatures = sorted(set(after_by_signature) - set(before_by_signature))
    before_codes = {row["code"] for row in before_instances}
    target_fixed = bool(before_instances) and not remaining_signatures
    introduced = sorted({signature[0] for signature in introduced_signatures})
    contract = dict(repair_contract or {})
    cards = _card_records(contract, selected=cards_presented)
    diff = graph_diff(before_candidate, after_candidate)
    allowed_ids = {str(value) for value in contract.get("allowed_entity_ids") or []}
    changed_ids = {
        str(row.get("node") or "").split(":", 1)[-1]
        for row in diff["nodes_changed"]
    } | {
        value.split(":", 1)[-1] for value in diff["nodes_removed"]
    }
    if "add" not in {str(value).lower() for value in contract.get("allowed_operations") or []}:
        changed_ids.update(value.split(":", 1)[-1] for value in diff["nodes_added"])
    outside = sorted(changed_ids - allowed_ids) if allowed_ids else []
    committed = str(terminal_status).upper() in {
        "COMMITTED", "COMMITTED_WITH_QUARANTINE",
    }
    return {
        "repair_event_schema_version": REPAIR_EVENT_SCHEMA_VERSION,
        "repair_event_id": event_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "execution_mode": str(execution_mode).upper(),
        "repair_scope": str(repair_scope).upper(),
        "context": {
            "validator_version": VALIDATOR_VERSION,
            "repair_prompt_version": REPAIR_PROMPT_VERSION,
            **dict(context or {}),
        },
        "target_issues": before_instances,
        "after_issues": after_instances,
        "selected_cards": [row for row in cards if row["selected"]],
        "suppressed_cards": [row for row in cards if not row["selected"]],
        "deterministic_patches": [dict(row) for row in deterministic_patches],
        "graph_diff": diff,
        "outside_target_entity_changes": outside,
        "outcome": {
            "target_issue_codes": sorted(before_codes),
            "remaining_target_issue_codes": sorted({row[0] for row in remaining_signatures}),
            "resolved_target_issue_codes": sorted({row[0] for row in resolved_signatures}),
            "remaining_target_signatures": [list(row) for row in remaining_signatures],
            "resolved_target_signatures": [list(row) for row in resolved_signatures],
            "introduced_issue_signatures": [list(row) for row in introduced_signatures],
            "introduced_issue_codes": introduced,
            "target_fixed": target_fixed,
            "clean_repair": target_fixed and not introduced and not outside,
            "final_world_committed": committed,
            "repair_yield": target_fixed and not introduced and not outside,
        },
    }


def aggregate_repair_events(
    runs: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build card scorecards, co-occurrence, and observed ordering metrics."""
    card_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    collateral: dict[str, Counter[str]] = defaultdict(Counter)
    cooccurrence: Counter[str] = Counter()
    ordering: Counter[str] = Counter()
    event_count = 0
    for run in runs:
        prior_cards: list[str] = []
        for event in run:
            if not isinstance(event, Mapping):
                continue
            event_count += 1
            outcome = event.get("outcome") or {}
            selected = [
                str(card.get("card_id") or "")
                for card in event.get("selected_cards") or []
                if isinstance(card, Mapping) and card.get("card_id")
            ]
            target_codes = sorted(set(
                str(issue.get("code") or "")
                for issue in event.get("target_issues") or []
                if isinstance(issue, Mapping) and issue.get("code")
            ))
            for left_index, left in enumerate(target_codes):
                for right in target_codes[left_index + 1:]:
                    cooccurrence[f"{left}|{right}"] += 1
            for before in prior_cards:
                for after in selected:
                    ordering[f"{before}->{after}"] += 1
            if selected:
                prior_cards = selected
            graph_diff_row = event.get("graph_diff") or {}
            changed_count = sum(len(graph_diff_row.get(key) or []) for key in (
                "nodes_added", "nodes_removed", "nodes_changed",
                "edges_added", "edges_removed",
            ))
            for card_id in selected:
                metric = card_metrics[card_id]
                metric["selected"] += 1
                metric["target_fixed"] += int(bool(outcome.get("target_fixed")))
                metric["clean_repair"] += int(bool(outcome.get("clean_repair")))
                metric["final_world_committed"] += int(bool(
                    outcome.get("final_world_committed")
                ))
                metric["repeat_fire"] += int(bool(
                    outcome.get("remaining_target_issue_codes")
                ))
                metric["graph_change_total"] += changed_count
                for code in outcome.get("introduced_issue_codes") or []:
                    collateral[card_id][str(code)] += 1
    scorecards: dict[str, Any] = {}
    for card_id, metric in sorted(card_metrics.items()):
        selected = metric["selected"]
        scorecards[card_id] = {
            **dict(metric),
            "target_repair_rate": metric["target_fixed"] / selected if selected else None,
            "clean_repair_rate": metric["clean_repair"] / selected if selected else None,
            "final_world_success_rate": (
                metric["final_world_committed"] / selected if selected else None
            ),
            "repeat_fire_rate": metric["repeat_fire"] / selected if selected else None,
            "mean_graph_changes": metric["graph_change_total"] / selected if selected else None,
            "collateral_issue_codes": dict(collateral[card_id].most_common()),
        }
    return {
        "repair_scorecard_schema_version": 1,
        "run_count": len(runs),
        "repair_event_count": event_count,
        "card_scorecards": scorecards,
        "invariant_cooccurrence": dict(cooccurrence.most_common()),
        "observed_card_ordering": dict(ordering.most_common()),
        "limitations": [
            "When multiple cards are selected in one event, outcomes are associated "
            "with every selected card but are not causally attributable to one card.",
            "Confirmed-error and silent-miss rates require separately labeled trials.",
        ],
    }
