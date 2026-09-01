"""Small typed semantic graph used at high-risk transformation boundaries.

The graph is deliberately not a knowledge graph of the whole ethical problem. It
records only claims whose direction must survive orchestration: conditions changing
an action, action consequences, preferences, and conditional switches.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


NODE_KINDS = {
    "ACTION", "PROPOSAL", "CONDITION", "CONSEQUENCE", "VALUE", "ACTOR", "METRIC", "THRESHOLD",
    "LOGICAL", "TARGET", "INTERVENTION", "ASSESSMENT", "DECISION", "EVIDENCE",
}
EDGE_RELATIONS = {
    "ACTIVATES", "CAUSES", "PREVENTS", "INCREASES", "DECREASES",
    "CONDITIONAL_ON", "ENABLES",
    "AFFECTS", "MORE_ATTRACTIVE", "LESS_ATTRACTIVE", "PREFERS", "SWITCHES_TO",
    "HAS_METRIC", "COMPARES_TO", "HAS_OPERAND", "BIASES_ESTIMATE",
    "IMPAIRS_EXECUTION",
    "REQUIRES", "NEGATES", "DISABLES", "PRESERVES_AVAILABILITY",
    "HAS_INTERVENTION", "HAS_ACTOR", "TARGETS", "HAS_CONSEQUENCE",
    "HAS_CONSTRAINT",
    "HAS_ASSESSMENT", "ASSESSES", "SUPPORTED_BY", "GROUNDED_IN",
    "HAS_VERDICT", "RESOLVES", "GOVERNED_BY",
    "IMPROVES_POSITION", "PRESERVES_POSITION", "WORSENS_POSITION",
    "MIXED_POSITION", "POSITION_UNCERTAIN",
    "SATISFIES_NORM", "CONSISTENT_WITH_NORM", "VIOLATES_NORM",
    "CONFLICTS_NORM", "NORM_UNCERTAIN",
}


@dataclass(frozen=True, slots=True)
class SemanticNode:
    id: str
    kind: str
    label: str
    provenance: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    def errors(self) -> list[str]:
        errors: list[str] = []
        if not self.id.strip():
            errors.append("node has no id")
        if self.kind not in NODE_KINDS:
            errors.append(f"node {self.id} has unknown kind {self.kind}")
        if not self.label.strip():
            errors.append(f"node {self.id} has no label")
        visibility = self.attributes.get("telemetry_visibility")
        if visibility is not None and (
            isinstance(visibility, bool)
            or not isinstance(visibility, (int, float))
            or not 0.0 <= float(visibility) <= 1.0
        ):
            errors.append(f"node {self.id} has invalid telemetry_visibility")
        return errors


@dataclass(frozen=True, slots=True)
class SemanticEdge:
    source: str
    relation: str
    target: str
    condition: str = ""
    justification: str = ""
    provenance: tuple[str, ...] = ()

    def errors(self) -> list[str]:
        errors: list[str] = []
        if self.relation not in EDGE_RELATIONS:
            errors.append(f"edge has unknown relation {self.relation}")
        if not self.source or not self.target:
            errors.append("edge has a missing endpoint")
        return errors


@dataclass(slots=True)
class SemanticGraph:
    nodes: dict[str, SemanticNode] = field(default_factory=dict)
    edges: list[SemanticEdge] = field(default_factory=list)

    def add_node(self, node: SemanticNode) -> None:
        existing = self.nodes.get(node.id)
        if existing is not None and (existing.kind, existing.label) != (node.kind, node.label):
            raise ValueError(f"semantic node id collision: {node.id}")
        if existing is not None:
            node = SemanticNode(
                node.id,
                node.kind,
                node.label,
                tuple(dict.fromkeys((*existing.provenance, *node.provenance))),
                {**existing.attributes, **node.attributes},
            )
        self.nodes[node.id] = node

    def add_edge(self, edge: SemanticEdge) -> None:
        self.edges.append(edge)

    def outgoing(self, node_id: str, relation: str = "") -> list[SemanticEdge]:
        return [
            edge for edge in self.edges
            if edge.source == node_id and (not relation or edge.relation == relation)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges],
        }


@dataclass(slots=True)
class GraphValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_graph(graph: SemanticGraph) -> GraphValidation:
    """Apply general topology and causal-direction invariants."""
    errors: list[str] = []
    warnings: list[str] = []
    for node in graph.nodes.values():
        errors.extend(node.errors())
    for edge in graph.edges:
        errors.extend(edge.errors())
        if edge.source not in graph.nodes or edge.target not in graph.nodes:
            errors.append(f"{edge.relation} edge references an unknown endpoint")

    logical_nodes = [node for node in graph.nodes.values() if node.kind == "LOGICAL"]
    for node in logical_nodes:
        if node.label not in {"AND", "OR"}:
            errors.append(f"logical node {node.id} has invalid operator {node.label}")
        operands = graph.outgoing(node.id, "HAS_OPERAND")
        if len(operands) < 2:
            errors.append(f"logical node {node.id} has fewer than two operands")
        for edge in operands:
            target = graph.nodes.get(edge.target)
            if target is not None and target.kind not in {"CONDITION", "LOGICAL"}:
                errors.append(f"logical node {node.id} has a non-predicate operand")

    def logical_cycle(node_id: str, path: set[str]) -> bool:
        if node_id in path:
            return True
        next_path = {*path, node_id}
        return any(
            graph.nodes.get(edge.target) is not None
            and graph.nodes[edge.target].kind == "LOGICAL"
            and logical_cycle(edge.target, next_path)
            for edge in graph.outgoing(node_id, "HAS_OPERAND")
        )

    if any(logical_cycle(node.id, set()) for node in logical_nodes):
        errors.append("logical predicate graph contains a cycle")

    # Framework assessments are graph state, not free-floating annotations.
    # Require the complete topology so a partial or misbound ledger cannot be
    # observed by downstream consumers after a transaction commits.
    position_relations = {
        "IMPROVES_POSITION", "PRESERVES_POSITION", "WORSENS_POSITION",
        "MIXED_POSITION", "POSITION_UNCERTAIN",
    }
    assessments = [node for node in graph.nodes.values() if node.kind == "ASSESSMENT"]
    for assessment in assessments:
        framework = str(assessment.attributes.get("framework", ""))
        owners = [
            edge for edge in graph.edges
            if edge.relation == "HAS_ASSESSMENT" and edge.target == assessment.id
        ]
        if len(owners) != 1 or graph.nodes.get(owners[0].source, None) is None \
                or graph.nodes[owners[0].source].kind != "ACTION":
            errors.append(f"assessment {assessment.id} lacks one ACTION owner")
        if framework == "RAWLSIAN":
            positions = [
            edge for edge in graph.outgoing(assessment.id)
            if edge.relation in position_relations
            ]
            dimensions = graph.outgoing(assessment.id, "ASSESSES")
            comparisons = graph.outgoing(assessment.id, "COMPARES_TO")
            if len(positions) != 1 or graph.nodes.get(positions[0].target, None) is None \
                    or graph.nodes[positions[0].target].kind != "TARGET":
                errors.append(f"assessment {assessment.id} lacks one TARGET position")
            if len(dimensions) != 1 or graph.nodes.get(dimensions[0].target, None) is None \
                    or graph.nodes[dimensions[0].target].kind != "VALUE":
                errors.append(f"assessment {assessment.id} lacks one VALUE dimension")
            if len(comparisons) != 1 or graph.nodes.get(comparisons[0].target, None) is None \
                    or graph.nodes[comparisons[0].target].kind != "ACTION":
                errors.append(f"assessment {assessment.id} lacks one ACTION comparison")
            expected_relation = {
                "IMPROVES": "IMPROVES_POSITION",
                "PRESERVES": "PRESERVES_POSITION",
                "WORSENS": "WORSENS_POSITION",
                "MIXED": "MIXED_POSITION",
                "UNCERTAIN": "POSITION_UNCERTAIN",
            }.get(str(assessment.attributes.get("effect", "")))
            if expected_relation and (
                len(positions) != 1 or positions[0].relation != expected_relation
            ):
                errors.append(
                    f"assessment {assessment.id} effect disagrees with its position edge"
                )
        elif framework == "DEONTOLOGICAL":
            norm_relations = {
                "SATISFIES_NORM", "CONSISTENT_WITH_NORM", "VIOLATES_NORM",
                "CONFLICTS_NORM", "NORM_UNCERTAIN",
            }
            norms = [
                edge for edge in graph.outgoing(assessment.id)
                if edge.relation in norm_relations
            ]
            parties = graph.outgoing(assessment.id, "AFFECTS")
            bearers = graph.outgoing(assessment.id, "HAS_ACTOR")
            if len(norms) != 1 or graph.nodes.get(norms[0].target, None) is None \
                    or graph.nodes[norms[0].target].kind != "VALUE":
                errors.append(f"assessment {assessment.id} lacks one normative relation")
            if len(parties) != 1 or graph.nodes.get(parties[0].target, None) is None \
                    or graph.nodes[parties[0].target].kind != "TARGET":
                errors.append(f"assessment {assessment.id} lacks one protected party")
            if len(bearers) != 1 or graph.nodes.get(bearers[0].target, None) is None \
                    or graph.nodes[bearers[0].target].kind != "ACTOR":
                errors.append(f"assessment {assessment.id} lacks one duty bearer")

    util_consequences = [
        node for node in graph.nodes.values()
        if node.kind == "CONSEQUENCE"
        and node.attributes.get("framework") == "UTILITARIAN"
    ]
    for consequence in util_consequences:
        owners = [
            edge for edge in graph.edges
            if edge.relation == "HAS_CONSEQUENCE" and edge.target == consequence.id
        ]
        targets = graph.outgoing(consequence.id, "AFFECTS")
        if len(owners) != 1 or graph.nodes.get(owners[0].source, None) is None \
                or graph.nodes[owners[0].source].kind != "ACTION":
            errors.append(f"utilitarian consequence {consequence.id} lacks one ACTION owner")
        if len(targets) != 1 or graph.nodes.get(targets[0].target, None) is None \
                or graph.nodes[targets[0].target].kind != "TARGET":
            errors.append(f"utilitarian consequence {consequence.id} lacks one affected scope")
        expected_polarity = {
            "BENEFIT": "BENEFICIAL", "HARM": "ADVERSE", "UNKNOWN": "UNKNOWN",
        }.get(str(consequence.attributes.get("direction", "")))
        if expected_polarity and consequence.attributes.get("polarity") != expected_polarity:
            errors.append(
                f"utilitarian consequence {consequence.id} direction disagrees with polarity"
            )

    switches = [edge for edge in graph.edges if edge.relation == "SWITCHES_TO"]
    effects = [
        edge for edge in graph.edges
        if edge.relation in {"MORE_ATTRACTIVE", "LESS_ATTRACTIVE"}
    ]
    for switch in switches:
        if switch.source == switch.target:
            errors.append("conditional switch returns to the same action")
        if not switch.condition:
            errors.append("conditional switch has no condition node")
        elif switch.condition not in graph.nodes:
            errors.append("conditional switch references an unknown condition")
        condition_scope = {switch.condition}
        frontier = [switch.condition]
        while frontier:
            parent = frontier.pop()
            children = [
                edge.target for edge in graph.edges
                if edge.source == parent and edge.relation == "HAS_OPERAND"
            ]
            for child in children:
                if child not in condition_scope:
                    condition_scope.add(child)
                    frontier.append(child)
        relevant = [edge for edge in effects if edge.condition in condition_scope]
        if not relevant:
            warnings.append("conditional switch has no typed attractiveness effect")
        for effect in relevant:
            if effect.target == switch.target and effect.relation == "LESS_ATTRACTIVE":
                errors.append("conditional switch points toward an action made less attractive")
            if effect.target == switch.source and effect.relation == "MORE_ATTRACTIVE":
                errors.append("conditional switch leaves an action made more attractive")

    signatures: dict[tuple[str, str], set[str]] = {}
    for edge in effects:
        signatures.setdefault((edge.condition, edge.target), set()).add(edge.relation)
    for (condition, action), relations in signatures.items():
        if relations == {"MORE_ATTRACTIVE", "LESS_ATTRACTIVE"}:
            errors.append(
                f"condition {condition} gives action {action} contradictory attractiveness effects"
            )
    return GraphValidation(not errors, list(dict.fromkeys(errors)), list(dict.fromkeys(warnings)))


def merge_graphs(graphs: Iterable[SemanticGraph]) -> SemanticGraph:
    combined = SemanticGraph()
    for graph in graphs:
        for node in graph.nodes.values():
            combined.add_node(node)
        combined.edges.extend(graph.edges)
    return combined
