"""Transactional Deontological duty, right, and permission ledger."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .graph_transactions import GraphTransactionRecord, SemanticGraphStore
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode, merge_graphs, validate_graph


class DutyAssessmentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: str = Field(pattern=r"^A\d+$")
    verdict: Literal["REQUIRED", "PERMISSIBLE", "PROHIBITED", "CONFLICTED"]
    norm_kind: Literal[
        "DUTY", "RIGHT", "AUTONOMY", "UNIVERSAL_LAW", "RESPECT_PERSONS",
        "OTHER", "UNKNOWN",
    ]
    norm: str = Field(min_length=3, max_length=100)
    relation: Literal["SATISFIES", "CONSISTENT", "VIOLATES", "CONFLICTS", "UNCERTAIN"]
    duty_bearer: str = Field(min_length=1, max_length=80)
    protected_party: str = Field(min_length=1, max_length=100)
    competing_norm: str = Field(min_length=1, max_length=100)
    evidence_basis: Literal["ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"]
    reason: str = Field(min_length=4, max_length=180)


class DeontologicalLedgerProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    assessments: list[DutyAssessmentProposal] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def unique_actions(self):
        action_ids = [item.action_id for item in self.assessments]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Deontological ledger repeats an action")
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


def _words(text: str) -> set[str]:
    ignored = {"person", "people", "party", "actor", "agent", "the", "and", "for"}
    return {
        value for value in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(value) >= 3 and value not in ignored
    }


def _action_parties(graph: SemanticGraph, action_id: str) -> list[SemanticNode]:
    parties: list[SemanticNode] = []
    for edge in graph.outgoing(action_id):
        target = graph.nodes.get(edge.target)
        if target is not None and target.kind in {"TARGET", "ACTOR"}:
            parties.append(target)
        if target is not None and target.kind == "CONSEQUENCE":
            parties.extend(
                graph.nodes[subedge.target]
                for subedge in graph.outgoing(target.id, "AFFECTS")
                if subedge.target in graph.nodes
                and graph.nodes[subedge.target].kind == "TARGET"
            )
    return list({party.id: party for party in parties}.values())


def _matching_parties(
    graph: SemanticGraph, action_id: str, text: str,
) -> list[SemanticNode]:
    words = _words(text)
    return [party for party in _action_parties(graph, action_id) if words & _words(party.label)]


def _stable_id(prefix: str, value: str) -> str:
    normalized = " ".join(str(value).casefold().split()).strip(" ,.;:")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def apply_deontological_ledger_transaction(
    store: SemanticGraphStore,
    proposal: dict[str, Any],
    *,
    cycle: int,
    specialist: str,
    allowed_actions: tuple[str, ...],
) -> GraphTransactionRecord:
    raw = dict(proposal) if isinstance(proposal, dict) else {"raw": proposal}
    try:
        validated = DeontologicalLedgerProposal.model_validate(proposal)
    except ValidationError as exc:
        errors = [
            f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        ]
        record = GraphTransactionRecord(
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    allowed_ids = {
        str(node.attributes.get("canonical_action_id", node.id))
        for reference in allowed_actions
        if (node := _resolve_action(store.graph, reference)) is not None
    }
    submitted_ids = {item.action_id for item in validated.assessments}
    if submitted_ids != allowed_ids:
        record = GraphTransactionRecord(
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            ["Deontological ledger must cover exactly the canonical action set"],
            previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    allowed_relations = {
        "REQUIRED": {"SATISFIES"},
        # Satisfying a duty may be permissible without being uniquely required.
        "PERMISSIBLE": {"CONSISTENT", "SATISFIES"},
        "PROHIBITED": {"VIOLATES"},
        "CONFLICTED": {"CONFLICTS"},
    }
    relation_edges = {
        "SATISFIES": "SATISFIES_NORM",
        "CONSISTENT": "CONSISTENT_WITH_NORM",
        "VIOLATES": "VIOLATES_NORM",
        "CONFLICTS": "CONFLICTS_NORM",
        "UNCERTAIN": "NORM_UNCERTAIN",
    }
    warnings: list[str] = []
    delta = SemanticGraph()
    committed: list[dict[str, Any]] = []
    replace_ids = {
        node.id for node in store.graph.nodes.values()
        if node.kind == "ASSESSMENT"
        and node.attributes.get("framework") == "DEONTOLOGICAL"
        and node.attributes.get("specialist") == specialist
    }
    for item in validated.assessments:
        action = _resolve_action(store.graph, item.action_id)
        if action is None:
            warnings.append(f"unknown action {item.action_id}")
            continue
        committed_relation = item.relation
        committed_verdict = item.verdict
        epistemic_status = (
            "FRAMEWORK_INTERPRETATION"
            if item.evidence_basis == "FRAMEWORK_ONLY" else "PROPOSED"
        )
        if item.relation not in allowed_relations[item.verdict]:
            committed_relation = "UNCERTAIN"
            committed_verdict = "UNCERTAIN"
            epistemic_status = "INTERNALLY_INCONSISTENT"
            warnings.append(
                f"{item.action_id} {item.verdict} conflicts with relation "
                f"{item.relation}; committed as UNCERTAIN"
            )
        party_label = " ".join(item.protected_party.casefold().split()).strip(" ,.;:")
        bearer_label = " ".join(item.duty_bearer.casefold().split()).strip(" ,.;:")
        norm_label = " ".join(item.norm.casefold().split()).strip(" ,.;:")
        matched = _matching_parties(store.graph, action.id, item.protected_party)
        if matched and epistemic_status == "PROPOSED":
            epistemic_status = "GROUNDED_PARTY"
        elif not matched and item.evidence_basis in {"ACTION_GRAPH", "SCENARIO"}:
            epistemic_status = "UNRESOLVED_PARTY"
        party_id = _stable_id("DEON_PARTY", party_label)
        bearer_id = _stable_id("DEON_BEARER", bearer_label)
        norm_id = _stable_id("DEON_NORM", f"{item.norm_kind}:{norm_label}")
        assessment_id = f"DEON_ASSESSMENT:{specialist}:{action.id}"
        provenance = (f"delegate:{specialist}", f"cycle:{cycle}")
        delta.add_node(SemanticNode(
            party_id, "TARGET", party_label, provenance,
            {"framework": "DEONTOLOGICAL", "grounded_node_ids": [p.id for p in matched]},
        ))
        delta.add_node(SemanticNode(
            bearer_id, "ACTOR", bearer_label, provenance,
            {"framework": "DEONTOLOGICAL"},
        ))
        delta.add_node(SemanticNode(
            norm_id, "VALUE", norm_label, provenance,
            {"framework": "DEONTOLOGICAL", "norm_kind": item.norm_kind},
        ))
        delta.add_node(SemanticNode(
            assessment_id, "ASSESSMENT", f"Deontological status for {item.action_id}",
            provenance,
            {
                "framework": "DEONTOLOGICAL", "specialist": specialist,
                "cycle": cycle, "canonical_action_id": item.action_id,
                "verdict": committed_verdict, "proposed_verdict": item.verdict,
                "relation": committed_relation, "proposed_relation": item.relation,
                "norm_kind": item.norm_kind, "norm_node_id": norm_id,
                "party_node_id": party_id, "bearer_node_id": bearer_id,
                "competing_norm": item.competing_norm,
                "evidence_basis": item.evidence_basis,
                "epistemic_status": epistemic_status, "reason": item.reason,
            },
        ))
        delta.add_edge(SemanticEdge(
            action.id, "HAS_ASSESSMENT", assessment_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, relation_edges[committed_relation], norm_id,
            justification=item.reason, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, "AFFECTS", party_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, "HAS_ACTOR", bearer_id, provenance=provenance,
        ))
        for party in matched:
            delta.add_edge(SemanticEdge(
                assessment_id, "SUPPORTED_BY", party.id, provenance=provenance,
            ))
        committed.append({
            "assessment_node_id": assessment_id,
            "canonical_action_id": item.action_id,
            "verdict": committed_verdict,
            "proposed_verdict": item.verdict,
            "norm_kind": item.norm_kind,
            "norm": norm_label,
            "relation": committed_relation,
            "proposed_relation": item.relation,
            "duty_bearer": bearer_label,
            "protected_party": party_label,
            "competing_norm": item.competing_norm,
            "evidence_basis": item.evidence_basis,
            "epistemic_status": epistemic_status,
            "reason": item.reason,
        })

    if len(committed) != len(validated.assessments):
        record = GraphTransactionRecord(
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            warnings or ["Deontological ledger could not bind every action"],
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
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            [f"graph merge rejected: {exc}"], previous_state_preserved=True,
            retryable=True,
        )
        store.transactions.append(record)
        return record
    validation = validate_graph(prospective)
    if not validation.valid:
        record = GraphTransactionRecord(
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            validation.errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record
    store.graph = prospective
    record = GraphTransactionRecord(
        cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER",
        "COMMITTED_WITH_UNCERTAINTY" if warnings else "COMMITTED",
        {"submitted": raw, "committed_assessments": committed}, warnings,
        previous_state_preserved=False,
    )
    store.transactions.append(record)
    return record


def committed_deontological_assessments(graph: SemanticGraph) -> list[dict[str, Any]]:
    values = [
        {
            "assessment_node_id": node.id,
            **dict(node.attributes),
            "provenance": list(node.provenance),
        }
        for node in graph.nodes.values()
        if node.kind == "ASSESSMENT"
        and node.attributes.get("framework") == "DEONTOLOGICAL"
    ]
    return sorted(values, key=lambda item: str(item.get("canonical_action_id", "")))
