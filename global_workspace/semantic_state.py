"""Read-only projections from committed semantic graph state.

The graph remains the mutation boundary.  Projections are deliberately small,
serializable views for downstream consumers that should not inspect raw delegate
text or reconstruct graph topology independently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from .scenario_semantics import semantic_action_key
from .semantic_graph import SemanticGraph, validate_graph


@dataclass(frozen=True, slots=True)
class CommittedReversalBoundary:
    source_action: str
    target_action: str
    predicate: str
    source_specialists: tuple[str, ...] = ()
    source_action_id: str = ""
    target_action_id: str = ""
    source_action_key: str = ""
    target_action_key: str = ""
    typed_predicate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_specialists"] = list(self.source_specialists)
        return data


@dataclass(slots=True)
class AuthoritativeSemanticState:
    """Stable downstream view of decision-critical committed graph state."""

    version: int = 2
    selected_action: str = ""
    selected_action_id: str = ""
    selected_action_key: str = ""
    factual_reversal_boundaries: list[CommittedReversalBoundary] = field(
        default_factory=list
    )
    rawlsian_positions: list[dict[str, Any]] = field(default_factory=list)
    utilitarian_consequences: list[dict[str, Any]] = field(default_factory=list)
    deontological_assessments: list[dict[str, Any]] = field(default_factory=list)
    virtue_assessments: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "selected_action": self.selected_action,
            "selected_action_id": self.selected_action_id,
            "selected_action_key": self.selected_action_key,
            "factual_reversal_boundaries": [
                boundary.to_dict() for boundary in self.factual_reversal_boundaries
            ],
            "rawlsian_positions": [dict(position) for position in self.rawlsian_positions],
            "utilitarian_consequences": [
                dict(consequence) for consequence in self.utilitarian_consequences
            ],
            "deontological_assessments": [
                dict(assessment) for assessment in self.deontological_assessments
            ],
            "virtue_assessments": [
                dict(assessment) for assessment in self.virtue_assessments
            ],
            "validation_errors": list(self.validation_errors),
        }


def _action_identity(node) -> tuple[str, str]:
    action_id = str(node.attributes.get("canonical_action_id", "")) or node.id
    action_key = str(node.attributes.get("semantic_action_key", ""))
    return action_id, action_key or semantic_action_key(node.label)


def _resolve_action_node(graph: SemanticGraph, reference: str):
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


def _typed_predicate(graph: SemanticGraph, node_id: str) -> dict[str, Any]:
    node = graph.nodes.get(node_id)
    if node is None:
        return {}
    if node.kind == "LOGICAL":
        operands = [
            _typed_predicate(graph, edge.target)
            for edge in graph.outgoing(node.id, "HAS_OPERAND")
        ]
        return {
            "type": "COMPOUND",
            "operator": str(node.attributes.get("operator", node.label)).upper(),
            "operands": [operand for operand in operands if operand],
        }
    attributes = node.attributes
    if attributes.get("predicate_type") == "SCALAR_THRESHOLD":
        affected = _resolve_action_node(
            graph, str(attributes.get("affected_action_id", ""))
        )
        affected_id, affected_key = _action_identity(affected) if affected else ("", "")
        return {
            "type": "SCALAR_THRESHOLD",
            "affected_action_id": affected_id,
            "affected_action_key": affected_key,
            "metric": str(attributes.get("metric", "")),
            "metric_valence": str(attributes.get("metric_valence", "")),
            "comparator": str(attributes.get("comparator", "")),
            "threshold": attributes.get("threshold"),
            "unit": str(attributes.get("unit", "")),
        }
    return {"type": "LEGACY_TEXT", "text": node.label}


def project_authoritative_semantic_state(
    graph: SemanticGraph, *, selected_action: str = "",
) -> AuthoritativeSemanticState:
    """Expose only validated, committed switches from the graph.

    Raw proposals and rejected transactions are intentionally unavailable here.
    Consumers therefore cannot accidentally publish or route on failed updates.
    """
    validation = validate_graph(graph)
    selected_node = _resolve_action_node(graph, selected_action) if selected_action else None
    selected_id, selected_key = (
        _action_identity(selected_node) if selected_node else ("", "")
    )
    state = AuthoritativeSemanticState(
        selected_action=(selected_node.label if selected_node else selected_action),
        selected_action_id=selected_id,
        selected_action_key=selected_key,
        validation_errors=list(validation.errors),
    )
    if not validation.valid:
        return state

    state.rawlsian_positions = sorted(
        [
            {
                "assessment_node_id": node.id,
                **dict(node.attributes),
                "provenance": list(node.provenance),
            }
            for node in graph.nodes.values()
            if node.kind == "ASSESSMENT"
            and node.attributes.get("framework") == "RAWLSIAN"
        ],
        key=lambda position: str(position.get("canonical_action_id", "")),
    )
    state.utilitarian_consequences = sorted(
        [
            {
                "consequence_node_id": node.id,
                "outcome": node.label,
                **dict(node.attributes),
                "provenance": list(node.provenance),
            }
            for node in graph.nodes.values()
            if node.kind == "CONSEQUENCE"
            and node.attributes.get("framework") == "UTILITARIAN"
        ],
        key=lambda consequence: (
            str(consequence.get("canonical_action_id", "")),
            str(consequence.get("consequence_node_id", "")),
        ),
    )
    state.deontological_assessments = sorted(
        [
            {
                "assessment_node_id": node.id,
                **dict(node.attributes),
                "provenance": list(node.provenance),
            }
            for node in graph.nodes.values()
            if node.kind == "ASSESSMENT"
            and node.attributes.get("framework") == "DEONTOLOGICAL"
        ],
        key=lambda assessment: str(assessment.get("canonical_action_id", "")),
    )
    state.virtue_assessments = sorted(
        [
            {
                "assessment_node_id": node.id,
                **dict(node.attributes),
                "provenance": list(node.provenance),
            }
            for node in graph.nodes.values()
            if node.kind == "ASSESSMENT"
            and node.attributes.get("framework") == "VIRTUE"
        ],
        key=lambda assessment: str(assessment.get("canonical_action_id", "")),
    )

    seen: set[tuple[str, str, str]] = set()
    for edge in graph.edges:
        if edge.relation != "SWITCHES_TO":
            continue
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        condition = graph.nodes.get(edge.condition)
        if source is None or target is None or condition is None:
            continue
        if source.kind != "ACTION" or target.kind != "ACTION":
            continue
        if condition.kind not in {"CONDITION", "LOGICAL"}:
            continue
        source_id, source_key = _action_identity(source)
        target_id, target_key = _action_identity(target)
        if selected_node is not None and source.id != selected_node.id:
            continue
        typed = _typed_predicate(graph, condition.id)
        signature = (
            source_key,
            target_key,
            json.dumps(typed, sort_keys=True, separators=(",", ":")),
        )
        if signature in seen:
            continue
        seen.add(signature)
        state.factual_reversal_boundaries.append(CommittedReversalBoundary(
            source_action=source.label,
            target_action=target.label,
            predicate=condition.label,
            source_specialists=tuple(
                str(value) for value in condition.attributes.get(
                    "source_specialists", []
                ) if str(value).strip()
            ),
            source_action_id=source_id,
            target_action_id=target_id,
            source_action_key=source_key,
            target_action_key=target_key,
            typed_predicate=typed,
        ))
    return state


def select_committed_reversal_boundary(
    graph: SemanticGraph, *, leading_action: str, competing_action: str,
) -> CommittedReversalBoundary | None:
    """Choose an already-committed transition matching the live action pair."""
    state = project_authoritative_semantic_state(
        graph, selected_action=leading_action
    )
    return next(
        (
            boundary for boundary in state.factual_reversal_boundaries
            if boundary.target_action == competing_action
        ),
        None,
    )
