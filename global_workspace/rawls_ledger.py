"""Transactional Rawlsian comparative-position ledger.

Delegate prose is a proposal, not graph state.  This module binds Rawlsian
position claims to canonical action and target nodes, weakens unsupported
directional claims to UNCERTAIN, and commits the normalized ledger atomically.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .graph_transactions import GraphTransactionRecord, SemanticGraphStore
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode, merge_graphs, validate_graph


RawlsEffect = Literal["IMPROVES", "PRESERVES", "WORSENS", "UNCERTAIN"]
RawlsDimension = Literal[
    "BASIC_LIBERTY", "OPPORTUNITY", "INCOME_WEALTH", "POWERS_OFFICES",
    "SELF_RESPECT", "BASIC_INTEREST_SECURITY", "OTHER_PRIMARY_GOOD", "UNKNOWN",
]


class RawlsPositionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: str = Field(pattern=r"^A\d+$")
    group: str = Field(min_length=2, max_length=100)
    dimension: RawlsDimension
    effect: RawlsEffect
    compared_to_action_id: str = Field(pattern=r"^A\d+$")
    evidence_basis: Literal["ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"]
    reason: str = Field(min_length=4, max_length=180)

    @model_validator(mode="after")
    def distinct_comparison(self):
        if self.action_id == self.compared_to_action_id:
            raise ValueError("Rawlsian position must compare distinct actions")
        return self


class RawlsLedgerProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ranking_basis: Literal[
        "LEXICAL_BASIC_LIBERTY", "FAIR_EQUALITY_OPPORTUNITY",
        "MAXIMIN_PRIMARY_GOODS", "DIFFERENCE_PRINCIPLE",
        "ORIGINAL_POSITION_PUBLIC_RULE", "UNRESOLVED",
    ]
    liberty_status: dict[str, Literal[
        "SATISFIED", "INFRINGED", "CONFLICTED", "UNKNOWN",
    ]]
    positions: list[RawlsPositionProposal] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def unique_actions(self):
        action_ids = [position.action_id for position in self.positions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Rawlsian ledger repeats an action")
        return self


_IGNORED_GROUP_WORDS = {
    "group", "groups", "people", "person", "persons", "affected", "least",
    "advantaged", "disadvantaged", "worse", "off", "stakeholder", "stakeholders",
    "under", "action", "option", "those", "their", "with", "without",
}


def _words(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(token) >= 3 and token not in _IGNORED_GROUP_WORDS
    }


def _resolve_action(graph: SemanticGraph, reference: str) -> SemanticNode | None:
    value = str(reference).strip()
    matches = [
        node for node in graph.nodes.values()
        if node.kind == "ACTION" and value in {
            node.id,
            node.label,
            str(node.attributes.get("canonical_action_id", "")),
            str(node.attributes.get("semantic_action_key", "")),
        }
    ]
    return matches[0] if len(matches) == 1 else None


def _action_consequences(
    graph: SemanticGraph, action_id: str,
) -> list[tuple[SemanticNode, list[SemanticNode]]]:
    values: list[tuple[SemanticNode, list[SemanticNode]]] = []
    for edge in graph.outgoing(action_id, "HAS_CONSEQUENCE"):
        consequence = graph.nodes.get(edge.target)
        if consequence is None or consequence.kind != "CONSEQUENCE":
            continue
        # A framework may inspect shared scenario structure, but another
        # delegate's interpretation is not scenario evidence. This prevents a
        # Utilitarian consequence proposal from silently becoming Rawlsian fact.
        if consequence.attributes.get("framework"):
            continue
        targets = [
            graph.nodes[target_edge.target]
            for target_edge in graph.outgoing(consequence.id, "AFFECTS")
            if target_edge.target in graph.nodes
            and graph.nodes[target_edge.target].kind == "TARGET"
        ]
        values.append((consequence, targets))
    return values


def _action_targets(graph: SemanticGraph, action_id: str) -> list[SemanticNode]:
    targets = [
        graph.nodes[edge.target]
        for edge in graph.outgoing(action_id, "TARGETS")
        if edge.target in graph.nodes and graph.nodes[edge.target].kind == "TARGET"
    ]
    for _, affected in _action_consequences(graph, action_id):
        targets.extend(affected)
    return list({target.id: target for target in targets}.values())


def _group_matches(group: str, targets: list[SemanticNode]) -> bool:
    group_words = _words(group)
    return bool(group_words and any(group_words & _words(target.label) for target in targets))


def _grounded_consequences(
    graph: SemanticGraph, action_id: str, group: str,
) -> tuple[list[SemanticNode], list[SemanticNode]]:
    group_words = _words(group)
    consequences: list[SemanticNode] = []
    targets: list[SemanticNode] = []
    if not group_words:
        return consequences, targets
    for consequence, affected_targets in _action_consequences(graph, action_id):
        matched = [
            target for target in affected_targets
            if group_words & _words(target.label)
        ]
        if matched:
            consequences.append(consequence)
            targets.extend(matched)
    return consequences, list({target.id: target for target in targets}.values())


def _direction_supported(
    effect: str,
    own: list[SemanticNode],
    rival: list[SemanticNode],
) -> bool:
    if effect == "UNCERTAIN":
        return True
    own_polarities = {str(node.attributes.get("polarity", "")) for node in own}
    rival_polarities = {str(node.attributes.get("polarity", "")) for node in rival}
    own_predicates = {node.label.casefold() for node in own}
    if effect == "IMPROVES":
        return bool(
            ("BENEFICIAL" in own_polarities and "BENEFICIAL" not in rival_polarities)
            or ("ADVERSE" in rival_polarities and "ADVERSE" not in own_polarities)
        )
    if effect == "WORSENS":
        return bool(
            ("ADVERSE" in own_polarities and "ADVERSE" not in rival_polarities)
            or ("BENEFICIAL" in rival_polarities and "BENEFICIAL" not in own_polarities)
        )
    if effect == "PRESERVES":
        return bool(
            own_predicates & {"preserve", "preserve_life", "protect"}
            or "ADVERSE" in rival_polarities
        )
    return False


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _canonical_group(group: str) -> str:
    """Give semantically identical casing/spacing one stable graph label."""
    return " ".join(str(group).casefold().split()).strip(" ,.;:")


def apply_rawls_ledger_transaction(
    store: SemanticGraphStore,
    proposal: dict[str, Any],
    *,
    cycle: int,
    specialist: str,
    allowed_actions: tuple[str, ...],
) -> GraphTransactionRecord:
    """Normalize and commit a complete Rawlsian ledger as one transaction."""
    raw = dict(proposal) if isinstance(proposal, dict) else {"raw": proposal}
    try:
        validated = RawlsLedgerProposal.model_validate(proposal)
    except ValidationError as exc:
        errors = [
            f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        ]
        record = GraphTransactionRecord(
            cycle, specialist, "RAWLS_POSITION_LEDGER", "REJECTED", raw,
            errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    allowed_nodes = {
        node.id: node
        for action in allowed_actions
        if (node := _resolve_action(store.graph, action)) is not None
    }
    submitted_ids = {position.action_id for position in validated.positions}
    allowed_ids = {
        str(node.attributes.get("canonical_action_id", node.id))
        for node in allowed_nodes.values()
    }
    errors: list[str] = []
    if submitted_ids != allowed_ids:
        errors.append("Rawlsian ledger must cover exactly the canonical action set")
    if set(validated.liberty_status) != allowed_ids:
        errors.append("Rawlsian liberty status must cover exactly the canonical action set")

    normalized: list[dict[str, Any]] = []
    delta = SemanticGraph()
    assessment_ids: set[str] = set()
    for position in validated.positions:
        action = _resolve_action(store.graph, position.action_id)
        rival = _resolve_action(store.graph, position.compared_to_action_id)
        if action is None or rival is None:
            errors.append(f"Rawlsian position {position.action_id} references an unknown action")
            continue
        own_consequences, own_targets = _grounded_consequences(
            store.graph, action.id, position.group
        )
        rival_consequences, rival_targets = _grounded_consequences(
            store.graph, rival.id, position.group
        )
        grounded_targets = list({
            target.id: target for target in (*own_targets, *rival_targets)
        }.values())
        own_action_targets = _action_targets(store.graph, action.id)
        group_bound_to_action = _group_matches(position.group, own_action_targets)
        supported = bool(
            grounded_targets
            and position.evidence_basis in {"ACTION_GRAPH", "SCENARIO"}
            and (
                position.evidence_basis != "ACTION_GRAPH"
                or group_bound_to_action
            )
            and _direction_supported(position.effect, own_consequences, rival_consequences)
        )
        committed_effect = position.effect
        epistemic_status = "GROUNDED"
        if position.effect != "UNCERTAIN" and not supported:
            committed_effect = "UNCERTAIN"
            epistemic_status = "UNSUPPORTED_DIRECTION"
            errors.append(
                f"{position.action_id} {position.effect} lacked a supporting action-to-group edge; committed as UNCERTAIN"
            )
        elif position.effect == "UNCERTAIN":
            epistemic_status = "EXPLICIT_UNCERTAINTY"

        group_label = _canonical_group(position.group)
        group_id = _stable_id("RAWLS_GROUP", group_label)
        dimension_id = f"RAWLS_VALUE:{position.dimension}"
        assessment_id = f"RAWLS_POSITION:{specialist}:{action.id}"
        assessment_ids.add(assessment_id)
        provenance = (f"delegate:{specialist}", f"cycle:{cycle}")
        delta.add_node(SemanticNode(
            group_id, "TARGET", group_label, provenance,
            {
                "framework": "RAWLSIAN",
                "grounded_target_node_ids": [target.id for target in grounded_targets],
                "epistemic_status": "GROUNDED" if grounded_targets else "UNRESOLVED_TARGET",
            },
        ))
        delta.add_node(SemanticNode(
            dimension_id, "VALUE", position.dimension, ("rawlsian_primary_goods",),
            {"framework": "RAWLSIAN"},
        ))
        delta.add_node(SemanticNode(
            assessment_id,
            "ASSESSMENT",
            f"Rawlsian position for {position.action_id}",
            provenance,
            {
                "framework": "RAWLSIAN",
                "specialist": specialist,
                "cycle": cycle,
                "canonical_action_id": position.action_id,
                "compared_to_action_id": position.compared_to_action_id,
                "effect": committed_effect,
                "proposed_effect": position.effect,
                "group": group_label,
                "dimension": position.dimension,
                "group_node_id": group_id,
                "dimension_node_id": dimension_id,
                "evidence_basis": position.evidence_basis,
                "epistemic_status": epistemic_status,
                "reason": position.reason,
                "ranking_basis": validated.ranking_basis,
                "liberty_status": validated.liberty_status.get(position.action_id, "UNKNOWN"),
                "group_selection_status": (
                    "BOUND_TO_ACTION" if group_bound_to_action else "UNRESOLVED_GROUP_SELECTION"
                ),
            },
        ))
        delta.add_edge(SemanticEdge(
            action.id, "HAS_ASSESSMENT", assessment_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id,
            {
                "IMPROVES": "IMPROVES_POSITION",
                "PRESERVES": "PRESERVES_POSITION",
                "WORSENS": "WORSENS_POSITION",
                "UNCERTAIN": "POSITION_UNCERTAIN",
            }[committed_effect],
            group_id,
            justification=position.reason,
            provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, "ASSESSES", dimension_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, "COMPARES_TO", rival.id, provenance=provenance,
        ))
        for evidence in [*own_consequences, *rival_consequences, *grounded_targets]:
            delta.add_edge(SemanticEdge(
                assessment_id, "SUPPORTED_BY", evidence.id, provenance=provenance,
            ))
        normalized.append({
            **position.model_dump(),
            "effect": committed_effect,
            "epistemic_status": epistemic_status,
            "group_node_id": group_id,
            "evidence_node_ids": [
                node.id for node in [*own_consequences, *rival_consequences, *grounded_targets]
            ],
            "ranking_basis": validated.ranking_basis,
            "liberty_status": validated.liberty_status.get(position.action_id, "UNKNOWN"),
            "group_selection_status": (
                "BOUND_TO_ACTION" if group_bound_to_action else "UNRESOLVED_GROUP_SELECTION"
            ),
        })

    hard_errors = [error for error in errors if "committed as UNCERTAIN" not in error]
    if hard_errors or len(normalized) != len(validated.positions):
        record = GraphTransactionRecord(
            cycle, specialist, "RAWLS_POSITION_LEDGER", "REJECTED", raw,
            list(dict.fromkeys(errors)), previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    # Replace the active ledger nodes for this specialist/action pair while
    # retaining target/value nodes and all unrelated semantic state.
    base = SemanticGraph(
        nodes={
            node_id: node for node_id, node in store.graph.nodes.items()
            if node_id not in assessment_ids
        },
        edges=[
            edge for edge in store.graph.edges
            if edge.source not in assessment_ids and edge.target not in assessment_ids
        ],
    )
    try:
        prospective = merge_graphs([base, delta])
    except ValueError as exc:
        record = GraphTransactionRecord(
            cycle, specialist, "RAWLS_POSITION_LEDGER", "REJECTED", raw,
            [f"graph merge rejected: {exc}"],
            previous_state_preserved=True,
            retryable=True,
        )
        store.transactions.append(record)
        return record
    graph_validation = validate_graph(prospective)
    if not graph_validation.valid:
        record = GraphTransactionRecord(
            cycle, specialist, "RAWLS_POSITION_LEDGER", "REJECTED", raw,
            graph_validation.errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    store.graph = prospective
    status = "COMMITTED_WITH_UNCERTAINTY" if errors else "COMMITTED"
    record = GraphTransactionRecord(
        cycle,
        specialist,
        "RAWLS_POSITION_LEDGER",
        status,
        {"submitted": raw, "committed_positions": normalized},
        list(dict.fromkeys(errors)),
        previous_state_preserved=False,
    )
    store.transactions.append(record)
    return record


def committed_rawls_positions(graph: SemanticGraph) -> list[dict[str, Any]]:
    """Read the active Rawlsian assessments from graph objects only."""
    values = []
    for node in graph.nodes.values():
        if node.kind != "ASSESSMENT" or node.attributes.get("framework") != "RAWLSIAN":
            continue
        values.append({
            "assessment_node_id": node.id,
            **dict(node.attributes),
            "provenance": list(node.provenance),
        })
    return sorted(values, key=lambda value: str(value.get("canonical_action_id", "")))
