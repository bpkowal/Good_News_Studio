"""Transactional virtue-ethics character and practical-wisdom ledger."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .graph_transactions import GraphTransactionRecord, SemanticGraphStore
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode, merge_graphs, validate_graph


class VirtueAssessmentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: str = Field(pattern=r"^A\d+$")
    verdict: Literal["EXEMPLIFIES", "MIXED", "UNDERMINES", "UNCERTAIN"]
    actor_role: str = Field(min_length=2, max_length=100)
    virtues: str = Field(min_length=2, max_length=120)
    vice_risk: str = Field(min_length=2, max_length=120)
    circumstance: str = Field(min_length=2, max_length=140)
    evidence_basis: Literal["ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"]
    reason: str = Field(min_length=4, max_length=180)


class VirtueLedgerProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ranking_basis: Literal[
        "PRACTICAL_WISDOM", "ROLE_FIDELITY", "FLOURISHING",
        "EXEMPLAR_REASONING", "UNRESOLVED",
    ]
    assessments: list[VirtueAssessmentProposal] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def unique_actions(self):
        action_ids = [assessment.action_id for assessment in self.assessments]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("virtue ledger repeats an action")
        return self


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


def apply_virtue_ledger_transaction(
    store: SemanticGraphStore,
    proposal: dict[str, Any],
    *,
    cycle: int,
    specialist: str,
    allowed_actions: tuple[str, ...],
) -> GraphTransactionRecord:
    """Validate the complete action comparison before replacing committed state."""
    raw = dict(proposal) if isinstance(proposal, dict) else {"raw": proposal}
    try:
        validated = VirtueLedgerProposal.model_validate(proposal)
    except ValidationError as exc:
        errors = [
            f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        ]
        record = GraphTransactionRecord(
            cycle, specialist, "VIRTUE_CHARACTER_LEDGER", "REJECTED", raw,
            errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    allowed_ids = {
        str(node.attributes.get("canonical_action_id", node.id))
        for action in allowed_actions
        if (node := _resolve_action(store.graph, action)) is not None
    }
    submitted_ids = {assessment.action_id for assessment in validated.assessments}
    errors: list[str] = []
    if submitted_ids != allowed_ids:
        errors.append("virtue ledger must cover exactly the canonical action set")
    if errors:
        record = GraphTransactionRecord(
            cycle, specialist, "VIRTUE_CHARACTER_LEDGER", "REJECTED", raw,
            errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    delta = SemanticGraph()
    assessment_ids: set[str] = set()
    provenance = (f"delegate:{specialist}", f"cycle:{cycle}")
    for assessment in validated.assessments:
        action = _resolve_action(store.graph, assessment.action_id)
        if action is None:
            errors.append(f"virtue assessment {assessment.action_id} references an unknown action")
            continue
        assessment_id = f"VIRTUE_CHARACTER:{specialist}:{action.id}"
        assessment_ids.add(assessment_id)
        delta.add_node(SemanticNode(
            assessment_id,
            "ASSESSMENT",
            f"Virtue assessment for {assessment.action_id}",
            provenance,
            {
                "framework": "VIRTUE",
                "specialist": specialist,
                "cycle": cycle,
                "canonical_action_id": assessment.action_id,
                "verdict": assessment.verdict,
                "actor_role": assessment.actor_role,
                "virtues": assessment.virtues,
                "vice_risk": assessment.vice_risk,
                "circumstance": assessment.circumstance,
                "evidence_basis": assessment.evidence_basis,
                "reason": assessment.reason,
                "ranking_basis": validated.ranking_basis,
                "epistemic_status": (
                    "EXPLICIT_UNCERTAINTY"
                    if assessment.verdict == "UNCERTAIN"
                    or assessment.evidence_basis == "UNKNOWN"
                    else "FRAMEWORK_GROUNDED"
                ),
            },
        ))
        delta.add_edge(SemanticEdge(
            action.id, "HAS_ASSESSMENT", assessment_id, provenance=provenance,
        ))

    if errors or len(assessment_ids) != len(validated.assessments):
        record = GraphTransactionRecord(
            cycle, specialist, "VIRTUE_CHARACTER_LEDGER", "REJECTED", raw,
            errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

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
            cycle, specialist, "VIRTUE_CHARACTER_LEDGER", "REJECTED", raw,
            [f"graph merge rejected: {exc}"], previous_state_preserved=True,
            retryable=True,
        )
        store.transactions.append(record)
        return record
    validation = validate_graph(prospective)
    if not validation.valid:
        record = GraphTransactionRecord(
            cycle, specialist, "VIRTUE_CHARACTER_LEDGER", "REJECTED", raw,
            validation.errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    store.graph = prospective
    record = GraphTransactionRecord(
        cycle, specialist, "VIRTUE_CHARACTER_LEDGER", "COMMITTED", raw,
        previous_state_preserved=False,
    )
    store.transactions.append(record)
    return record


def committed_virtue_assessments(graph: SemanticGraph) -> list[dict[str, Any]]:
    values = [
        {
            "assessment_node_id": node.id,
            **dict(node.attributes),
            "provenance": list(node.provenance),
        }
        for node in graph.nodes.values()
        if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "VIRTUE"
    ]
    return sorted(values, key=lambda value: str(value.get("canonical_action_id", "")))
