"""Terminal figures for inspecting a typed world before deliberation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _effect_label(effect: Mapping[str, Any]) -> str:
    effect_id = _text(effect.get("effect_id")) or "?"
    directness = _text(effect.get("directness")) or "?"
    polarity = _text(effect.get("polarity")) or "?"
    modality = _text(effect.get("modality")) or "CERTAIN"
    outcome = _text(effect.get("outcome")) or "(missing outcome)"
    party = _text(effect.get("party_id")) or "?"
    quantities = ", ".join(
        _text(item) for item in (effect.get("quantities") or []) if _text(item)
    )
    quantity = f"; q={quantities}" if quantities else ""
    likelihoods = ", ".join(
        _text(item) for item in (effect.get("likelihood_qualifiers") or [])
        if _text(item)
    )
    likelihood = f"; likelihood={likelihoods}" if likelihoods else ""
    epistemic = f"/{modality}" if modality != "CERTAIN" else ""
    return (
        f"[{effect_id}] {directness}/{polarity}{epistemic}: "
        f"{outcome} ({party}{quantity}{likelihood})"
    )


def _action_labels(
    actions: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        _text(action.get("action_id")): (
            _text(action.get("intervention"))
            or _text(action.get("label"))
            or _text(action.get("action_id"))
        )
        for action in actions
        if _text(action.get("action_id"))
    }


def render_causal_topology_figure(world_model: Mapping[str, Any]) -> str:
    """Figure 1: action-local causal edges plus disconnected effect nodes."""
    actions = [row for row in world_model.get("actions", []) if isinstance(row, Mapping)]
    effects = [row for row in world_model.get("effects", []) if isinstance(row, Mapping)]
    links = [row for row in world_model.get("causal_links", []) if isinstance(row, Mapping)]
    action_names = _action_labels(actions)
    effects_by_id = {
        _text(effect.get("effect_id")): effect for effect in effects
        if _text(effect.get("effect_id"))
    }
    action_ids = list(action_names)
    for effect in effects:
        action_id = _text(effect.get("action_id"))
        if action_id and action_id not in action_ids:
            action_ids.append(action_id)

    lines = [
        "",
        "--- Figure 1: Within-action causal topology ---",
        "(actual causal edges only; counterfactual projections are marked as unlinked here)",
    ]
    for action_id in action_ids:
        lines.append(f"\n{action_id}: {action_names.get(action_id, action_id)}")
        local_effects = [
            effect for effect in effects
            if _text(effect.get("action_id")) == action_id
        ]
        local_ids = {_text(effect.get("effect_id")) for effect in local_effects}
        local_links = [
            link for link in links
            if _text(link.get("action_id")) == action_id
            or (
                _text(link.get("source_id")) in local_ids
                and _text(link.get("target_id")) in local_ids
            )
        ]
        touched: set[str] = set()
        for link in local_links:
            source_id = _text(link.get("source_id"))
            target_id = _text(link.get("target_id"))
            relation = _text(link.get("relation") or link.get("link_relation")) or "?"
            touched.update((source_id, target_id))
            source = effects_by_id.get(source_id)
            target = effects_by_id.get(target_id)
            target_derivation = (
                _text(target.get("derivation_operation")) if target else ""
            )
            if target_derivation == "EXCLUSIVE_ALLOCATION_COMPLEMENT":
                relation = f"{relation}/EXCLUSIVE_COMPLEMENT"
            source_label = _effect_label(source) if source else f"[{source_id or '?'}] MISSING"
            target_label = _effect_label(target) if target else f"[{target_id or '?'}] MISSING"
            lines.append(f"  {source_label}")
            lines.append(f"      └──{relation}──▶ {target_label}")
        unlinked = [
            effect for effect in local_effects
            if _text(effect.get("effect_id")) not in touched
        ]
        for effect in unlinked:
            derived = _text(effect.get("derivation_operation"))
            suffix = f"  [unlinked: {derived}]" if derived else "  [unlinked]"
            lines.append(f"  {_effect_label(effect)}{suffix}")
        if not local_effects:
            lines.append("  (no effects)")
    return "\n".join(lines)


def render_counterfactual_figure(world_model: Mapping[str, Any]) -> str:
    """Figure 2: cross-action counterfactual relations and their endpoints."""
    effects = [row for row in world_model.get("effects", []) if isinstance(row, Mapping)]
    links = [
        row for row in world_model.get("counterfactual_links", [])
        if isinstance(row, Mapping)
    ]
    effects_by_id = {
        _text(effect.get("effect_id")): effect for effect in effects
        if _text(effect.get("effect_id"))
    }
    lines = [
        "",
        "--- Figure 2: Cross-action counterfactual topology ---",
        "(each arrow should cross actions and should add a comparison, not duplicate an actual fact)",
    ]
    if not links:
        lines.append("  (no counterfactual links)")
        return "\n".join(lines)
    seen: set[tuple[str, str, str, str, str]] = set()
    for link in links:
        action_id = _text(link.get("action_id")) or "?"
        alternative_action = _text(link.get("alternative_action_id")) or "?"
        source_id = _text(link.get("source_effect_id"))
        target_id = _text(link.get("alternative_effect_id"))
        relation = _text(
            link.get("counterfactual_relation") or link.get("relation")
        ) or "?"
        signature = (action_id, source_id, relation, alternative_action, target_id)
        duplicate = "  ⚠ duplicate" if signature in seen else ""
        seen.add(signature)
        source = effects_by_id.get(source_id)
        target = effects_by_id.get(target_id)
        source_label = _effect_label(source) if source else f"[{source_id or '?'}] MISSING"
        target_label = _effect_label(target) if target else f"[{target_id or '?'}] MISSING"
        same_action = "  ⚠ same action" if action_id == alternative_action else ""
        lines.append(f"  {action_id} {source_label}")
        lines.append(
            f"      └──{relation}──▶ {alternative_action} {target_label}"
            f"{duplicate}{same_action}"
        )
    return "\n".join(lines)


def render_world_graph_figures(grounding: Mapping[str, Any]) -> str:
    """Render both pre-admission diagnostic figures from grounding output."""
    world_model = grounding.get("world_model")
    if not isinstance(world_model, Mapping):
        return "\n--- World graph figures unavailable: no typed world model ---"
    return "\n".join((
        render_causal_topology_figure(world_model),
        render_counterfactual_figure(world_model),
    ))
