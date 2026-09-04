"""Transactional, action-indexed Utilitarian consequence ledger."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .graph_transactions import GraphTransactionRecord, SemanticGraphStore
from .scenario_semantics import (
    project_grounded_action_effects,
    query_grounded_action_effects,
)
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode, merge_graphs, validate_graph
from .world_state import counts_as_actual_welfare


class ConsequenceProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    outcome: str = Field(min_length=4, max_length=120)
    scope: str = Field(min_length=1, max_length=80)
    direction: Literal["BENEFIT", "HARM"]
    probability: str = Field(min_length=1, max_length=24)
    magnitude: str = Field(min_length=1, max_length=60)
    duration: str = Field(min_length=1, max_length=40)
    reversibility: Literal["REVERSIBLE", "IRREVERSIBLE", "UNKNOWN"]
    support: Literal["STATED", "INFERRED", "UNKNOWN"]


class ActionConsequenceProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: str = Field(pattern=r"^A\d+$")
    consequences: list[ConsequenceProposal] = Field(min_length=1, max_length=4)


class UtilitarianLedgerProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    actions: list[ActionConsequenceProposal] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def unique_actions(self):
        action_ids = [item.action_id for item in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Utilitarian ledger repeats an action")
        return self


class EffectValuationProposal(BaseModel):
    """Framework valuation of an existing fact; no factual direction is editable."""

    model_config = ConfigDict(extra="forbid", strict=True)
    # World models commonly issue compact stable IDs such as E1. Identity is
    # validated against the graph below; string length carries no authority.
    effect_id: str = Field(min_length=1, max_length=120)
    importance: Literal["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]
    reason: str = Field(min_length=4, max_length=160)


class ActionEffectValuationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: str = Field(pattern=r"^A\d+$")
    valuations: list[EffectValuationProposal] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def unique_effects(self):
        effect_ids = [item.effect_id for item in self.valuations]
        if len(effect_ids) != len(set(effect_ids)):
            raise ValueError("Utilitarian valuation repeats a grounded effect")
        return self


class UtilitarianEffectValuationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    actions: list[ActionEffectValuationProposal] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def unique_actions(self):
        action_ids = [item.action_id for item in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Utilitarian valuation repeats an action")
        return self


def _resolve_action(graph: SemanticGraph, reference: str) -> SemanticNode | None:
    matches = [
        node for node in graph.nodes.values()
        if node.kind == "ACTION" and str(reference).strip() in {
            node.id, node.label,
            str(node.attributes.get("canonical_action_id", "")),
            str(node.attributes.get("semantic_action_key", "")),
        }
    ]
    return matches[0] if len(matches) == 1 else None


_GENERIC = {
    "the", "and", "for", "with", "without", "people", "person", "group",
    "causes", "cause", "results", "result", "risk", "risks", "may", "might",
}


def _words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(word) >= 3 and word not in _GENERIC
    }


_INFLECTIONS = ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s")


def _stem(word: str) -> str:
    """Collapse the inflections that separate a claim from a graph label.

    The graph stores compiled predicate labels ("freeze", "resident") while a
    delegate writes prose ("freezing", "8 residents"). Without this the two
    never compare and every row looks equally unsupported.
    """
    for suffix in _INFLECTIONS:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)].rstrip("e")
    return word.rstrip("e")


def _stems(text: str) -> set[str]:
    return {_stem(word) for word in _words(text)}


def _deterministic_consequences(graph: SemanticGraph, action_id: str) -> list[SemanticNode]:
    return [
        graph.nodes[edge.target]
        for edge in graph.outgoing(action_id, "HAS_CONSEQUENCE")
        if edge.target in graph.nodes
        and graph.nodes[edge.target].kind == "CONSEQUENCE"
        and graph.nodes[edge.target].attributes.get("framework") != "UTILITARIAN"
    ]


def _best_evidence(
    graph: SemanticGraph,
    action_id: str,
    outcome: str,
    scope: str,
    *,
    expected_polarity: str = "",
) -> tuple[SemanticNode | None, int]:
    claim_stems = _stems(f"{outcome} {scope}")
    subject_stems = _stems(scope) or claim_stems
    effects = query_grounded_action_effects(
        graph, action_id, claim=f"{outcome} {scope}",
    )
    # A grounded effect can only confirm or contradict a row when it concerns
    # the same affected subject. The claim query alone returns every effect the
    # action's clause mentions, so without this filter the direction check
    # compared a claim about residents against a fact about audit evidence.
    comparable: list[tuple[SemanticNode, int]] = []
    for effect in effects:
        consequence = graph.nodes.get(effect.consequence_id)
        if consequence is None:
            continue
        if not subject_stems & _stems(effect.affected_subject):
            continue
        score = len(claim_stems & _stems(
            f"{consequence.label} {effect.affected_subject} {effect.dimension}"
        ))
        comparable.append((consequence, max(1, score)))
    if not comparable:
        return None, 0
    # One action routinely carries several comparable effects with opposite
    # polarities: the override both preserves life and leaves residents
    # freezing. A row that agrees with any of them is grounded by it, so the
    # contradiction only stands when every comparable fact runs the other way.
    if expected_polarity:
        agreeing = [
            item for item in comparable
            if item[0].attributes.get("polarity") == expected_polarity
        ]
        if agreeing:
            return max(agreeing, key=lambda item: item[1])
    return comparable[0]


def _stable_id(prefix: str, value: str) -> str:
    normalized = " ".join(str(value).casefold().split()).strip(" ,.;:")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _accounting_role(polarity: str) -> str:
    """Project world polarity without asking a framework to regenerate it."""
    return {
        "BENEFICIAL": "BENEFIT",
        "ADVERSE": "HARM",
        "FOREGONE": "OPPORTUNITY_COST",
        "NEUTRAL": "NEUTRAL",
        "UNRESOLVED": "UNKNOWN",
    }.get(str(polarity).upper(), "UNKNOWN")


def _node_is_foregone(node: SemanticNode | None) -> bool:
    if node is None:
        return False
    polarity = str(node.attributes.get("polarity", "")).upper()
    directness = str(node.attributes.get("directness", "")).upper()
    return polarity == "FOREGONE" or directness == "FOREGONE"


def utilitarian_scored_grounded_effects(graph: SemanticGraph):
    """Grounded effects util should value, without FOREGONE duals of actual rows.

    A FOREGONE row on party P remains an opportunity cost when this action has
    no actual harm or benefit on P. If the action already records that party's
    stipulated death, survival, or equivalent welfare, the FOREGONE overlay is
    the counterfactual dual and must not enter the welfare sum.
    """
    projected = project_grounded_action_effects(graph)
    actual_parties: dict[str, set[str]] = {}
    for effect in projected:
        node = graph.nodes.get(effect.consequence_id)
        if node is None or _node_is_foregone(node):
            continue
        if not counts_as_actual_welfare(
            polarity=str(node.attributes.get("polarity", "")).upper(),
            directness=str(node.attributes.get("directness", "")).upper(),
            effect_kind=str(node.attributes.get("effect_kind", "")).upper(),
            party_kind=str(node.attributes.get("party_kind", "")).upper(),
        ):
            continue
        party_id = str(node.attributes.get("party_id", ""))
        if party_id:
            actual_parties.setdefault(effect.action_id, set()).add(party_id)
    scored = []
    for effect in projected:
        node = graph.nodes.get(effect.consequence_id)
        if node is not None and _node_is_foregone(node):
            party_id = str(node.attributes.get("party_id", ""))
            if party_id and party_id in actual_parties.get(effect.action_id, set()):
                continue
        scored.append(effect)
    return scored


def _apply_effect_valuation_transaction(
    store: SemanticGraphStore,
    proposal: dict[str, Any],
    *,
    cycle: int,
    specialist: str,
    allowed_actions: tuple[str, ...],
) -> GraphTransactionRecord:
    """Commit an effect-ID valuation overlay on immutable grounded facts."""
    raw = dict(proposal) if isinstance(proposal, dict) else {"raw": proposal}
    try:
        validated = UtilitarianEffectValuationProposal.model_validate(proposal)
    except ValidationError as exc:
        errors = [
            f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        ]
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_EFFECT_VALUATION", "REJECTED",
            raw, errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    allowed_ids = {
        str(node.attributes.get("canonical_action_id", node.id))
        for reference in allowed_actions
        if (node := _resolve_action(store.graph, reference)) is not None
    }
    submitted_ids = {item.action_id for item in validated.actions}
    if submitted_ids != allowed_ids:
        errors = ["Utilitarian valuation must cover exactly the canonical action set"]
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_EFFECT_VALUATION", "REJECTED",
            raw, errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    projected_by_action: dict[str, dict[str, Any]] = {}
    for effect in utilitarian_scored_grounded_effects(store.graph):
        evidence = store.graph.nodes.get(effect.consequence_id)
        if evidence is None:
            continue
        projected_by_action.setdefault(effect.action_id, {})[effect.effect_id] = (
            effect, evidence
        )

    errors: list[str] = []
    for action_item in validated.actions:
        expected = set(projected_by_action.get(action_item.action_id, {}))
        submitted = {item.effect_id for item in action_item.valuations}
        if submitted != expected:
            errors.append(
                f"{action_item.action_id} valuations must reference every grounded "
                f"effect exactly once; missing={sorted(expected - submitted)}, "
                f"unknown={sorted(submitted - expected)}"
            )
    if errors:
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_EFFECT_VALUATION", "REJECTED",
            raw, errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    delta = SemanticGraph()
    committed: list[dict[str, Any]] = []
    replace_ids = {
        node.id for node in store.graph.nodes.values()
        if node.kind == "CONSEQUENCE"
        and node.attributes.get("framework") == "UTILITARIAN"
        and node.attributes.get("specialist") == specialist
    }
    for action_item in validated.actions:
        action = _resolve_action(store.graph, action_item.action_id)
        if action is None:
            continue
        for valuation in action_item.valuations:
            effect, evidence = projected_by_action[action_item.action_id][valuation.effect_id]
            polarity = str(evidence.attributes.get("polarity", "UNRESOLVED")).upper()
            direction = _accounting_role(polarity)
            modality = str(evidence.attributes.get("modality", "UNKNOWN")).upper()
            quantities = [
                str(value) for value in evidence.attributes.get("quantities", [])
                if str(value).strip()
            ]
            probability = "CERTAIN" if modality == "CERTAIN" else "UNKNOWN"
            magnitude = ", ".join(quantities)[:60] or "UNKNOWN"
            scope_label = " ".join(effect.affected_subject.casefold().split()).strip(" ,.;:")
            target_id = _stable_id("UTIL_SCOPE", scope_label)
            consequence_id = (
                f"UTIL_CONSEQUENCE:{specialist}:{action.id}:"
                f"{hashlib.sha256(valuation.effect_id.encode('utf-8')).hexdigest()[:12]}"
            )
            provenance = (
                f"delegate:{specialist}", f"cycle:{cycle}",
                f"grounded_effect:{valuation.effect_id}",
            )
            delta.add_node(SemanticNode(
                target_id, "TARGET", scope_label, provenance,
                {"framework": "UTILITARIAN"},
            ))
            attributes = {
                "framework": "UTILITARIAN", "specialist": specialist,
                "cycle": cycle, "canonical_action_id": action_item.action_id,
                "grounded_effect_id": valuation.effect_id,
                "world_effect_id": evidence.attributes.get("world_effect_id", ""),
                "direction": direction,
                "direction_source": "GROUNDED_WORLD_POLARITY",
                "polarity": polarity,
                "probability": probability,
                "modality": modality,
                "magnitude": magnitude,
                "duration": "UNKNOWN", "reversibility": "UNKNOWN",
                "support": "STATED",
                "epistemic_status": effect.epistemic_status,
                "importance": valuation.importance,
                "valuation_reason": valuation.reason,
                "scope_node_id": target_id,
            }
            delta.add_node(SemanticNode(
                consequence_id, "CONSEQUENCE", evidence.label, provenance, attributes,
            ))
            delta.add_edge(SemanticEdge(
                action.id, "HAS_CONSEQUENCE", consequence_id, provenance=provenance,
            ))
            delta.add_edge(SemanticEdge(
                consequence_id, "AFFECTS", target_id, provenance=provenance,
            ))
            delta.add_edge(SemanticEdge(
                consequence_id, "SUPPORTED_BY", evidence.id, provenance=provenance,
            ))
            committed.append({
                "consequence_node_id": consequence_id,
                "canonical_action_id": action_item.action_id,
                "grounded_effect_id": valuation.effect_id,
                "world_effect_id": evidence.attributes.get("world_effect_id", ""),
                "outcome": evidence.label,
                "scope": scope_label,
                "direction": direction,
                "direction_source": "GROUNDED_WORLD_POLARITY",
                "polarity": polarity,
                "probability": probability,
                "modality": modality,
                "magnitude": magnitude,
                "duration": "UNKNOWN",
                "reversibility": "UNKNOWN",
                "support": "STATED",
                "epistemic_status": effect.epistemic_status,
                "importance": valuation.importance,
                "valuation_reason": valuation.reason,
            })

    base = SemanticGraph(
        nodes={key: node for key, node in store.graph.nodes.items() if key not in replace_ids},
        edges=[
            edge for edge in store.graph.edges
            if edge.source not in replace_ids and edge.target not in replace_ids
        ],
    )
    try:
        prospective = merge_graphs([base, delta])
    except ValueError as exc:
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_EFFECT_VALUATION", "REJECTED", raw,
            [f"graph merge rejected: {exc}"], previous_state_preserved=True,
            retryable=True,
        )
        store.transactions.append(record)
        return record
    validation = validate_graph(prospective)
    if not validation.valid:
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_EFFECT_VALUATION", "REJECTED", raw,
            validation.errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record
    store.graph = prospective
    record = GraphTransactionRecord(
        cycle, specialist, "UTILITARIAN_EFFECT_VALUATION", "COMMITTED",
        {"submitted": raw, "committed_consequences": committed}, [],
        previous_state_preserved=False,
    )
    store.transactions.append(record)
    return record


def apply_utilitarian_ledger_transaction(
    store: SemanticGraphStore,
    proposal: dict[str, Any],
    *,
    cycle: int,
    specialist: str,
    allowed_actions: tuple[str, ...],
) -> GraphTransactionRecord:
    actions = proposal.get("actions", []) if isinstance(proposal, dict) else []
    if actions and isinstance(actions[0], dict) and "valuations" in actions[0]:
        return _apply_effect_valuation_transaction(
            store, proposal, cycle=cycle, specialist=specialist,
            allowed_actions=allowed_actions,
        )
    raw = dict(proposal) if isinstance(proposal, dict) else {"raw": proposal}
    try:
        validated = UtilitarianLedgerProposal.model_validate(proposal)
    except ValidationError as exc:
        errors = [
            f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        ]
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_CONSEQUENCE_LEDGER", "REJECTED",
            raw, errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    allowed_ids = {
        str(node.attributes.get("canonical_action_id", node.id))
        for reference in allowed_actions
        if (node := _resolve_action(store.graph, reference)) is not None
    }
    submitted_ids = {item.action_id for item in validated.actions}
    if submitted_ids != allowed_ids:
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_CONSEQUENCE_LEDGER", "REJECTED", raw,
            ["Utilitarian ledger must cover exactly the canonical action set"],
            previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    delta = SemanticGraph()
    warnings: list[str] = []
    committed: list[dict[str, Any]] = []
    replace_ids = {
        node.id for node in store.graph.nodes.values()
        if node.kind == "CONSEQUENCE"
        and node.attributes.get("framework") == "UTILITARIAN"
        and node.attributes.get("specialist") == specialist
    }
    for action_item in validated.actions:
        action = _resolve_action(store.graph, action_item.action_id)
        if action is None:
            warnings.append(f"unknown action {action_item.action_id}")
            continue
        for index, row in enumerate(action_item.consequences):
            expected_polarity = "BENEFICIAL" if row.direction == "BENEFIT" else "ADVERSE"
            evidence, overlap = _best_evidence(
                store.graph, action.id, row.outcome, row.scope,
                expected_polarity=expected_polarity,
            )
            committed_direction = row.direction
            epistemic_status = {
                "STATED": "STATED_UNRESOLVED",
                "INFERRED": "INFERRED",
                "UNKNOWN": "UNKNOWN",
            }[row.support]
            if evidence is not None and overlap:
                if evidence.attributes.get("polarity") == expected_polarity:
                    epistemic_status = "GROUNDED_ACTION_GRAPH"
                else:
                    committed_direction = "UNKNOWN"
                    epistemic_status = "CONTRADICTED_DIRECTION"
                    warnings.append(
                        f"{action_item.action_id} consequence {index} direction "
                        "contradicts its grounded action consequence; committed as UNKNOWN"
                    )
            consequence_id = f"UTIL_CONSEQUENCE:{specialist}:{action.id}:{index}"
            scope_label = " ".join(row.scope.casefold().split()).strip(" ,.;:")
            target_id = _stable_id("UTIL_SCOPE", scope_label)
            provenance = (f"delegate:{specialist}", f"cycle:{cycle}")
            delta.add_node(SemanticNode(
                target_id, "TARGET", scope_label, provenance,
                {"framework": "UTILITARIAN"},
            ))
            delta.add_node(SemanticNode(
                consequence_id, "CONSEQUENCE", row.outcome, provenance,
                {
                    "framework": "UTILITARIAN", "specialist": specialist,
                    "cycle": cycle, "canonical_action_id": action_item.action_id,
                    "direction": committed_direction,
                    "proposed_direction": row.direction,
                    "polarity": {
                        "BENEFIT": "BENEFICIAL", "HARM": "ADVERSE",
                        "UNKNOWN": "UNKNOWN",
                    }[committed_direction],
                    "probability": row.probability, "magnitude": row.magnitude,
                    "duration": row.duration, "reversibility": row.reversibility,
                    "support": row.support, "epistemic_status": epistemic_status,
                    "scope_node_id": target_id,
                },
            ))
            delta.add_edge(SemanticEdge(
                action.id, "HAS_CONSEQUENCE", consequence_id, provenance=provenance,
            ))
            delta.add_edge(SemanticEdge(
                consequence_id, "AFFECTS", target_id, provenance=provenance,
            ))
            if evidence is not None and overlap:
                delta.add_edge(SemanticEdge(
                    consequence_id, "SUPPORTED_BY", evidence.id,
                    provenance=provenance,
                ))
            committed.append({
                "consequence_node_id": consequence_id,
                "canonical_action_id": action_item.action_id,
                "outcome": row.outcome,
                "scope": scope_label,
                "direction": committed_direction,
                "proposed_direction": row.direction,
                "probability": row.probability,
                "magnitude": row.magnitude,
                "duration": row.duration,
                "reversibility": row.reversibility,
                "support": row.support,
                "epistemic_status": epistemic_status,
            })

    if len(committed) != sum(len(item.consequences) for item in validated.actions):
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_CONSEQUENCE_LEDGER", "REJECTED", raw,
            warnings or ["Utilitarian ledger could not bind every action"],
            previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record
    base = SemanticGraph(
        nodes={key: node for key, node in store.graph.nodes.items() if key not in replace_ids},
        edges=[
            edge for edge in store.graph.edges
            if edge.source not in replace_ids and edge.target not in replace_ids
        ],
    )
    try:
        prospective = merge_graphs([base, delta])
    except ValueError as exc:
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_CONSEQUENCE_LEDGER", "REJECTED", raw,
            [f"graph merge rejected: {exc}"], previous_state_preserved=True,
            retryable=True,
        )
        store.transactions.append(record)
        return record
    validation = validate_graph(prospective)
    if not validation.valid:
        record = GraphTransactionRecord(
            cycle, specialist, "UTILITARIAN_CONSEQUENCE_LEDGER", "REJECTED", raw,
            validation.errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record
    store.graph = prospective
    record = GraphTransactionRecord(
        cycle, specialist, "UTILITARIAN_CONSEQUENCE_LEDGER",
        "COMMITTED_WITH_UNCERTAINTY" if warnings else "COMMITTED",
        {"submitted": raw, "committed_consequences": committed}, warnings,
        previous_state_preserved=False,
    )
    store.transactions.append(record)
    return record


def committed_utilitarian_consequences(graph: SemanticGraph) -> list[dict[str, Any]]:
    values = [
        {
            "consequence_node_id": node.id,
            "outcome": node.label,
            **dict(node.attributes),
            "provenance": list(node.provenance),
        }
        for node in graph.nodes.values()
        if node.kind == "CONSEQUENCE"
        and node.attributes.get("framework") == "UTILITARIAN"
    ]
    return sorted(values, key=lambda item: (
        str(item.get("canonical_action_id", "")), str(item.get("consequence_node_id", ""))
    ))
