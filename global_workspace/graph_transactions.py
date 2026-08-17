"""Transactional application of delegate-proposed semantic graph updates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .decision_boundaries import (
    CompoundDecisionBoundary,
    DecisionBoundary,
    boundary_satisfied_by_estimates,
)
from .semantic_graph import (
    SemanticEdge, SemanticGraph, merge_graphs, validate_graph,
)


class BoundaryClauseProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    affected_action: str = Field(min_length=1, max_length=240)
    metric: str = Field(min_length=2, max_length=60)
    metric_valence: Literal["ADVERSE", "BENEFICIAL"]
    comparator: Literal["LT", "LE", "GT", "GE"]
    threshold: float
    unit: str = Field(min_length=1, max_length=20)
    source_text: str = Field(min_length=3, max_length=100)


class GraphUpdateProposal(BaseModel):
    """Untrusted delegate payload admitted before any graph objects are created."""
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["NONE", "BOUNDARY", "AND", "OR"]
    from_action: str
    to_action: str
    clauses: list[BoundaryClauseProposal] = Field(max_length=3)

    @model_validator(mode="after")
    def operation_shape(self):
        if self.operation == "NONE":
            if self.from_action != "NONE" or self.to_action != "NONE" or self.clauses:
                raise ValueError("NONE update must have NONE endpoints and no clauses")
        elif self.operation == "BOUNDARY" and len(self.clauses) != 1:
            raise ValueError("BOUNDARY update must contain exactly one clause")
        elif self.operation in {"AND", "OR"} and len(self.clauses) < 2:
            raise ValueError(f"{self.operation} update must contain at least two clauses")
        if self.operation != "NONE" and self.from_action == self.to_action:
            raise ValueError("graph update cannot switch an action to itself")
        return self


@dataclass(slots=True)
class GraphTransactionRecord:
    cycle: int
    specialist: str
    operation: str
    status: str
    proposal: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    previous_state_preserved: bool = True
    vote_disposition: str = "RETAINED"
    retryable: bool = False


class SemanticGraphStore:
    """Commit valid graph deltas atomically; rejected deltas cannot mutate state."""

    def __init__(self, initial_graph: SemanticGraph | None = None) -> None:
        self.graph = initial_graph or SemanticGraph()
        self.transactions: list[GraphTransactionRecord] = []

    def _resolve_action_ref(self, reference: str) -> str:
        """Resolve a run-local ID, action label, or stable key to one node ID."""
        value = str(reference).strip()
        matches = [
            node.id for node in self.graph.nodes.values()
            if node.kind == "ACTION" and value in {
                node.id,
                node.label,
                str(node.attributes.get("canonical_action_id", "")),
                str(node.attributes.get("semantic_action_key", "")),
            }
        ]
        return matches[0] if len(set(matches)) == 1 else ""

    def apply_action_extension(
        self, extension: SemanticGraph, *, cycle: int, source: str = "synthesis",
    ) -> GraphTransactionRecord:
        """Atomically add newly admitted action nodes and their identity subgraph."""
        collisions = [
            node_id for node_id, node in extension.nodes.items()
            if node_id in self.graph.nodes
            and (
                self.graph.nodes[node_id].kind != node.kind
                or self.graph.nodes[node_id].label != node.label
            )
        ]
        new_ids = set(extension.nodes) - set(self.graph.nodes)
        if collisions or not any(
            extension.nodes[node_id].kind == "ACTION" for node_id in new_ids
        ):
            errors = (
                [f"action-extension node collision: {node_id}" for node_id in collisions]
                or ["action extension contains no new canonical action"]
            )
            record = GraphTransactionRecord(
                cycle, source, "ACTION_SET_EXTENSION", "REJECTED",
                {"new_node_ids": sorted(new_ids)}, errors,
                previous_state_preserved=True, retryable=False,
            )
            self.transactions.append(record)
            return record
        delta = SemanticGraph(
            nodes={node_id: extension.nodes[node_id] for node_id in new_ids},
            edges=[
                edge for edge in extension.edges
                if edge.source in new_ids or edge.target in new_ids
            ],
        )
        try:
            prospective = merge_graphs([self.graph, delta])
        except ValueError as exc:
            record = GraphTransactionRecord(
                cycle, source, "ACTION_SET_EXTENSION", "REJECTED",
                {"new_node_ids": sorted(new_ids)}, [str(exc)],
                previous_state_preserved=True, retryable=False,
            )
            self.transactions.append(record)
            return record
        validation = validate_graph(prospective)
        if not validation.valid:
            record = GraphTransactionRecord(
                cycle, source, "ACTION_SET_EXTENSION", "REJECTED",
                {"new_node_ids": sorted(new_ids)}, validation.errors,
                previous_state_preserved=True, retryable=False,
            )
            self.transactions.append(record)
            return record
        self.graph = prospective
        record = GraphTransactionRecord(
            cycle, source, "ACTION_SET_EXTENSION", "COMMITTED",
            {"new_node_ids": sorted(new_ids)}, [],
            previous_state_preserved=False,
        )
        self.transactions.append(record)
        return record

    def _canonicalize_proposal(
        self, proposal: GraphUpdateProposal,
    ) -> GraphUpdateProposal:
        """Translate legacy prose endpoints into authoritative action node IDs."""
        if proposal.operation == "NONE":
            return proposal
        source = self._resolve_action_ref(proposal.from_action)
        target = self._resolve_action_ref(proposal.to_action)
        clauses = []
        for clause in proposal.clauses:
            affected = self._resolve_action_ref(clause.affected_action)
            clauses.append(clause.model_copy(update={
                "affected_action": affected or clause.affected_action,
            }))
        return proposal.model_copy(update={
            "from_action": source or proposal.from_action,
            "to_action": target or proposal.to_action,
            "clauses": clauses,
        })

    def _bind_delta_actions(self, delta: SemanticGraph) -> SemanticGraph:
        """Bind boundary placeholder nodes to existing canonical action nodes."""
        remap: dict[str, str] = {}
        bound = SemanticGraph()
        for node in delta.nodes.values():
            resolved = self._resolve_action_ref(node.label) if node.kind == "ACTION" else ""
            if resolved:
                remap[node.id] = resolved
                bound.add_node(self.graph.nodes[resolved])
            else:
                bound.add_node(node)
        for edge in delta.edges:
            bound.add_edge(SemanticEdge(
                remap.get(edge.source, edge.source),
                edge.relation,
                remap.get(edge.target, edge.target),
                condition=remap.get(edge.condition, edge.condition),
                justification=edge.justification,
                provenance=edge.provenance,
            ))
        return bound

    @staticmethod
    def _boundary(proposal: GraphUpdateProposal, specialist: str):
        operation = proposal.operation
        if operation == "NONE":
            return None
        boundaries = []
        for clause in proposal.clauses:
            boundaries.append(DecisionBoundary(
                from_action=proposal.from_action,
                to_action=proposal.to_action,
                affected_action=clause.affected_action,
                metric=clause.metric,
                comparator=clause.comparator,
                threshold=clause.threshold,
                unit=clause.unit,
                metric_valence=clause.metric_valence,
                source_specialist=specialist,
                source_text=clause.source_text,
            ))
        if operation == "BOUNDARY":
            if len(boundaries) != 1:
                raise ValueError("scalar boundary must contain exactly one clause")
            return boundaries[0]
        if operation in {"AND", "OR"}:
            return CompoundDecisionBoundary(operation, tuple(boundaries))
        raise ValueError(f"unknown graph operation {operation}")

    def apply(
        self, proposal: dict[str, Any], *, cycle: int, specialist: str,
        expected_from_action: str = "", allowed_actions: tuple[str, ...] = (),
        expected_source_text: str = "",
        current_action_values: dict[str, dict[str, Any]] | None = None,
        rejection_policy: Literal["RETAIN_VOTE", "DROP_VOTE"] = "RETAIN_VOTE",
    ) -> GraphTransactionRecord:
        raw_proposal = dict(proposal) if isinstance(proposal, dict) else {"raw": proposal}
        try:
            validated = GraphUpdateProposal.model_validate(proposal)
        except ValidationError as exc:
            errors = [
                f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in exc.errors(include_url=False)
            ]
            record = GraphTransactionRecord(
                cycle, specialist, str(raw_proposal.get("operation", "INVALID")),
                "REJECTED", raw_proposal, errors, True,
                "DROPPED" if rejection_policy == "DROP_VOTE" else "RETAINED",
                True,
            )
            self.transactions.append(record)
            return record
        validated = self._canonicalize_proposal(validated)
        operation = validated.operation
        if operation == "NONE":
            record = GraphTransactionRecord(
                cycle, specialist, operation, "NO_OP", validated.model_dump(),
            )
            self.transactions.append(record)
            return record
        errors: list[str] = []
        delta = None
        expected_source_id = self._resolve_action_ref(expected_from_action)
        if (
            expected_from_action
            and validated.from_action != (expected_source_id or expected_from_action)
        ):
            errors.append("graph proposal source differs from delegate recommendation")
        if allowed_actions:
            allowed_ids = {
                self._resolve_action_ref(action) or str(action)
                for action in allowed_actions
            }
            referenced = {
                validated.from_action, validated.to_action,
                *(clause.affected_action for clause in validated.clauses),
            }
            if referenced - allowed_ids:
                errors.append("graph proposal references an unknown action")
        # Provenance is structural: affected action -> typed metric/comparator ->
        # typed threshold. `source_text` remains audit evidence, never an exact
        # string gate. Canonical node membership is checked above and Pydantic
        # validates the relation/object shape before graph construction.
        try:
            boundary = self._boundary(validated, specialist)
            if not errors:
                validation = boundary.validate()
                errors.extend(validation.errors)
            canonical_values = {
                self._resolve_action_ref(action) or str(action): value
                for action, value in (current_action_values or {}).items()
            }
            if (
                not errors
                and boundary_satisfied_by_estimates(boundary, canonical_values) is True
            ):
                errors.append(
                    "state-relative boundary is already satisfied by grounded current facts "
                    "while the delegate retains its source action"
                )
            if not errors:
                delta = self._bind_delta_actions(boundary.graph())
                prospective = merge_graphs([self.graph, delta])
                errors.extend(validate_graph(prospective).errors)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        if errors or delta is None:
            record = GraphTransactionRecord(
                cycle, specialist, operation, "REJECTED", validated.model_dump(),
                list(dict.fromkeys(errors)), previous_state_preserved=True,
                vote_disposition=("DROPPED" if rejection_policy == "DROP_VOTE" else "RETAINED"),
                retryable=True,
            )
        else:
            self.graph = merge_graphs([self.graph, delta])
            record = GraphTransactionRecord(
                cycle, specialist, operation, "COMMITTED", validated.model_dump(),
                previous_state_preserved=False,
            )
        self.transactions.append(record)
        return record

    def graph_dict(self) -> dict[str, Any]:
        return self.graph.to_dict()

    def transaction_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.transactions]
