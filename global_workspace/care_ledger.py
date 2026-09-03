"""Transactional Care-ethics relationship and responsiveness ledger."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .graph_transactions import GraphTransactionRecord, SemanticGraphStore
from .scenario_semantics import query_grounded_action_effects
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode, merge_graphs, validate_graph


class CareAssessmentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action_id: str = Field(pattern=r"^A\d+$")
    verdict: Literal["RESPONSIVE", "MIXED", "NEGLECTFUL", "UNCERTAIN"]
    affected_party: str = Field(min_length=2, max_length=100)
    relationship_type: Literal[
        "ENTRUSTED", "DEPENDENCY", "CARE_ROLE", "AGENT_CREATED_VULNERABILITY",
        "COMMUNITY_RELATION", "NONE", "UNRESOLVED",
    ]
    dependency_source: str = Field(min_length=2, max_length=160)
    responsibility_basis: str = Field(min_length=2, max_length=180)
    need_kind: Literal[
        "SURVIVAL_HEALTH", "BASIC_NEED", "ONGOING_DEPENDENCY", "TRUST",
        "RELATIONAL_CONTINUITY", "OTHER", "UNRESOLVED",
    ]
    need_urgency: Literal["IMMEDIATE", "NEAR_TERM", "LONG_TERM", "MIXED", "UNKNOWN"]
    trust_effect: Literal[
        "STRENGTHENS", "PRESERVES", "STRAINS", "BETRAYS", "NOT_APPLICABLE", "UNKNOWN",
    ]
    responsiveness: Literal["DIRECT", "INDIRECT", "DELAYED", "WITHHELD", "MIXED", "UNKNOWN"]
    feasibility: Literal["ESTABLISHED", "CONDITIONAL", "UNKNOWN"]
    competing_care_claim: str = Field(min_length=2, max_length=180)
    resolution_status: Literal["RESOLVED", "CONTESTED", "UNKNOWN"]
    evidence_basis: Literal["ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"]
    reason: str = Field(min_length=4, max_length=180)

    @model_validator(mode="after")
    def resolved_assessment_has_relational_basis(self):
        if self.resolution_status == "RESOLVED":
            if self.relationship_type == "UNRESOLVED":
                raise ValueError("resolved Care assessment requires a relationship classification")
            if self.need_kind == "UNRESOLVED":
                raise ValueError("resolved Care assessment requires a need classification")
            if self.responsiveness == "UNKNOWN" or self.feasibility == "UNKNOWN":
                raise ValueError("resolved Care assessment requires responsiveness and feasibility")
        return self


class CareLedgerProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ranking_basis: Literal[
        "ENTRUSTED_RESPONSIBILITY", "ACUTE_DEPENDENCY", "AGENT_CREATED_VULNERABILITY",
        "RELATIONAL_CONTINUITY", "RESPONSIVE_FEASIBILITY", "UNRESOLVED",
    ]
    assessments: list[CareAssessmentProposal] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def unique_actions(self):
        action_ids = [assessment.action_id for assessment in self.assessments]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Care ledger repeats an action")
        if self.ranking_basis == "UNRESOLVED" and any(
            assessment.resolution_status == "RESOLVED"
            for assessment in self.assessments
        ):
            raise ValueError("unresolved Care ranking cannot contain resolved assessments")
        return self


def _resolve_action(graph: SemanticGraph, reference: str) -> SemanticNode | None:
    value = str(reference).strip()
    matches = [
        node for node in graph.nodes.values()
        if node.kind == "ACTION" and value in {
            node.id, node.label,
            str(node.attributes.get("canonical_action_id", "")),
            str(node.attributes.get("semantic_action_key", "")),
        }
    ]
    return matches[0] if len(matches) == 1 else None


def apply_care_ledger_transaction(
    store: SemanticGraphStore,
    proposal: dict[str, Any],
    *,
    cycle: int,
    specialist: str,
    allowed_actions: tuple[str, ...],
) -> GraphTransactionRecord:
    """Validate and atomically replace one specialist's Care ledger."""
    raw = dict(proposal) if isinstance(proposal, dict) else {"raw": proposal}
    try:
        validated = CareLedgerProposal.model_validate(proposal)
    except ValidationError as exc:
        errors = [
            f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        ]
        record = GraphTransactionRecord(
            cycle, specialist, "CARE_RELATIONSHIP_LEDGER", "REJECTED", raw,
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
    if submitted_ids != allowed_ids:
        record = GraphTransactionRecord(
            cycle, specialist, "CARE_RELATIONSHIP_LEDGER", "REJECTED", raw,
            ["Care ledger must cover exactly the canonical action set"],
            previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    delta = SemanticGraph()
    assessment_ids: set[str] = set()
    uncertainties: list[str] = []
    provenance = (f"delegate:{specialist}", f"cycle:{cycle}")
    for assessment in validated.assessments:
        action = _resolve_action(store.graph, assessment.action_id)
        if action is None:
            continue
        grounded_effects = query_grounded_action_effects(
            store.graph,
            assessment.action_id,
            claim=(
                f"{assessment.affected_party} {assessment.dependency_source} "
                f"{assessment.responsibility_basis} {assessment.reason}"
            ),
        )
        if assessment.evidence_basis in {"ACTION_GRAPH", "SCENARIO"} and not grounded_effects:
            uncertainties.append(
                f"{assessment.action_id} Care relation claims scenario grounding without a matching action effect"
            )
        if assessment.resolution_status != "RESOLVED":
            uncertainties.append(
                f"{assessment.action_id} competing Care claim remains {assessment.resolution_status.lower()}"
            )
        assessment_id = f"CARE_RELATIONSHIP:{specialist}:{action.id}"
        assessment_ids.add(assessment_id)
        delta.add_node(SemanticNode(
            assessment_id,
            "ASSESSMENT",
            f"Care assessment for {assessment.action_id}",
            provenance,
            {
                "framework": "CARE",
                "specialist": specialist,
                "cycle": cycle,
                "canonical_action_id": assessment.action_id,
                "verdict": assessment.verdict,
                "affected_party": assessment.affected_party,
                "relationship_type": assessment.relationship_type,
                "dependency_source": assessment.dependency_source,
                "responsibility_basis": assessment.responsibility_basis,
                "need_kind": assessment.need_kind,
                "need_urgency": assessment.need_urgency,
                "trust_effect": assessment.trust_effect,
                "responsiveness": assessment.responsiveness,
                "feasibility": assessment.feasibility,
                "competing_care_claim": assessment.competing_care_claim,
                "resolution_status": assessment.resolution_status,
                "evidence_basis": assessment.evidence_basis,
                "reason": assessment.reason,
                "ranking_basis": validated.ranking_basis,
                "grounded_effect_ids": [effect.effect_id for effect in grounded_effects],
                "grounded_consequence_ids": [effect.consequence_id for effect in grounded_effects],
                "epistemic_status": (
                    "EXPLICIT_UNCERTAINTY"
                    if assessment.resolution_status != "RESOLVED"
                    or assessment.evidence_basis == "UNKNOWN"
                    else "FRAMEWORK_GROUNDED_WITH_ACTION_EFFECTS"
                    if grounded_effects else "FRAMEWORK_GROUNDED"
                ),
            },
        ))
        delta.add_edge(SemanticEdge(
            action.id, "HAS_ASSESSMENT", assessment_id, provenance=provenance,
        ))
        for effect in grounded_effects:
            delta.add_edge(SemanticEdge(
                assessment_id, "SUPPORTED_BY", effect.consequence_id,
                provenance=provenance,
            ))

    if len(assessment_ids) != len(validated.assessments):
        record = GraphTransactionRecord(
            cycle, specialist, "CARE_RELATIONSHIP_LEDGER", "REJECTED", raw,
            ["Care ledger contains an unknown action"],
            previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    replace_ids = {
        node.id for node in store.graph.nodes.values()
        if node.kind == "ASSESSMENT"
        and node.attributes.get("framework") == "CARE"
        and node.attributes.get("specialist") == specialist
    }
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
            cycle, specialist, "CARE_RELATIONSHIP_LEDGER", "REJECTED", raw,
            [f"graph merge rejected: {exc}"], previous_state_preserved=True,
            retryable=True,
        )
        store.transactions.append(record)
        return record
    validation = validate_graph(prospective)
    if not validation.valid:
        record = GraphTransactionRecord(
            cycle, specialist, "CARE_RELATIONSHIP_LEDGER", "REJECTED", raw,
            validation.errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    store.graph = prospective
    status = "COMMITTED_WITH_UNCERTAINTY" if uncertainties else "COMMITTED"
    record = GraphTransactionRecord(
        cycle, specialist, "CARE_RELATIONSHIP_LEDGER", status, raw,
        uncertainties, previous_state_preserved=False,
    )
    store.transactions.append(record)
    return record


def committed_care_assessments(graph: SemanticGraph) -> list[dict[str, Any]]:
    values = [
        {
            "assessment_node_id": node.id,
            **dict(node.attributes),
            "provenance": list(node.provenance),
        }
        for node in graph.nodes.values()
        if node.kind == "ASSESSMENT" and node.attributes.get("framework") == "CARE"
    ]
    return sorted(values, key=lambda value: str(value.get("canonical_action_id", "")))
