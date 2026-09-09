"""Temporary NetworkX views of the canonical world model.

The world-state schema remains the database. These graphs are calculators:
build, query, discard. Validators must not store an nx.DiGraph as Parliament
state, and production ledgers must not import this module just to exist.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx

from .world_state import ScenarioWorldModel

# Same set deontology_ledger uses for INTENDED_AS_MEANS path support.
# Kept here so the adapter does not import the ledger.
MEANS_RELATIONS = frozenset({
    "CAUSES", "ENABLES", "INCREASES", "ACCELERATES",
    "NECESSARY_FOR", "MEANS_TO", "PRODUCES",
})
_COMPILE_ALIASES = {
    "MAY_CAUSE": "CAUSES",
    "MIGHT_CAUSE": "CAUSES",
    "MAY_ENABLE": "ENABLES",
    "ACCELERATES": "INCREASES",
}
_FORECLOSURE = frozenset({
    "FOREGOES", "FOREGOES_ALTERNATIVE_EFFECT",
    "PRECLUDES", "PRECLUDES_ALTERNATIVE_EFFECT",
    "REPLACES", "REPLACES_ALTERNATIVE_EFFECT",
})


def _normalized_relation(relation: str) -> str:
    raw = str(relation or "").upper()
    return _COMPILE_ALIASES.get(raw, raw)


def causal_digraph(
    model: ScenarioWorldModel,
    *,
    relations: Iterable[str] | None = None,
) -> nx.DiGraph:
    """Actual-world causal edges only. Counterfactual overlays stay off."""
    allowed = None if relations is None else frozenset(
        _normalized_relation(item) for item in relations
    )
    graph = nx.DiGraph()
    for effect in model.effects:
        graph.add_node(effect.effect_id, action_id=effect.action_id)
    for link in model.causal_links:
        relation = _normalized_relation(link.relation)
        if relation in _FORECLOSURE:
            continue
        if allowed is not None and relation not in allowed:
            continue
        if link.source_id not in graph or link.target_id not in graph:
            continue
        graph.add_edge(
            link.source_id,
            link.target_id,
            relation=relation,
            action_id=link.action_id,
        )
    return graph


def has_causal_path(
    model: ScenarioWorldModel,
    source_id: str,
    target_id: str,
    *,
    relations: Iterable[str] = MEANS_RELATIONS,
) -> bool:
    """True when target is reachable from source along allowed actual edges."""
    if source_id == target_id:
        return False
    graph = causal_digraph(model, relations=relations)
    if source_id not in graph or target_id not in graph:
        return False
    return nx.has_path(graph, source_id, target_id)


def is_intermediate_means(
    model: ScenarioWorldModel,
    burden_id: str,
    end_id: str,
) -> bool:
    """True when the burden is an ancestor of the end, not a sibling outcome."""
    return has_causal_path(model, burden_id, end_id, relations=MEANS_RELATIONS)
