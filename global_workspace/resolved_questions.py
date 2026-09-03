"""Symbolic ledger of the audited questions a run has already settled.

The generative layer is good at proposing which questions matter and bad at
remembering that one of them was answered. Delegates regenerate their open
questions from scratch every cycle, so a question an audit already put to the
whole Parliament reappears under the same stable key and can be re-audited
until the cycle budget is gone. That is not extra rigour: the answers were
already recorded, and re-asking spends the scarce audit slot that a genuinely
live issue needed.

This module keeps the answer as graph state so pruning stays inspectable and
reversible. A resolution stores a signature of the committed evidence its
answer depended on, and the question reopens by itself once that evidence
moves. Nothing here decides a normative question. It records only that the
workspace asked one, that the delegates who tested it agreed on the answer,
and which facts that agreement rested on.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .graph_transactions import GraphTransactionRecord, SemanticGraphStore
from .semantic_graph import (
    SemanticEdge, SemanticGraph, SemanticNode, merge_graphs, validate_graph,
)

# A resolution is workspace bookkeeping, not a framework assessment, so it is
# stored as a DECISION node. ASSESSMENT would require an owning ACTION and
# would be read by the framework ledger projections.
RESOLUTION_KIND = "DECISION"
RESOLUTION_OPERATION = "AUDITED_QUESTION_RESOLUTION"

# A delegate that engaged the audit under any of these participation statuses
# gave a usable answer. CONTESTED and UNRESOLVED are live disagreement, and
# NOT_TESTED means the question never reached that framework.
TESTED_PARTICIPATION = frozenset({
    "RELEVANT", "IRRELEVANT", "TRANSLATED", "REVERSAL_RELEVANT",
})
# CONTESTED is live disagreement about whether the issue matters, which is
# unfinished deliberation and must stay fully open. UNRESOLVED is the weaker
# report that the framework engaged and could not tell, which is a real answer
# about the current evidence even though it settles nothing normative.
BLOCKING_PARTICIPATION = frozenset({"CONTESTED"})
UNRESOLVED_PARTICIPATION = "UNRESOLVED"
SETTLING_EFFECTS = frozenset({"NO_CHANGE", "WEAKENS", "REVERSES"})
UNSETTLED_EFFECTS = frozenset({"UNRESOLVED", ""})
UNRESOLVABLE_RESOLUTION = "UNRESOLVABLE_AT_CURRENT_EVIDENCE"
MINIMUM_RESPONDERS = 2

# The evidence a question was grounded in is only "the same evidence" while
# these committed properties hold. Bookkeeping such as the writing cycle is
# excluded so a re-commit of identical state does not reopen a settled answer.
_SIGNATURE_ATTRIBUTES = frozenset({
    "admitted", "canonical_action_id", "dimension", "direction", "effect",
    "epistemic_status", "evidence_role", "grounding_basis", "grounding_status",
    "magnitude", "polarity", "probability", "proposed_effect",
    "protected_standing", "resolution_status", "support", "targets", "verdict",
})


@dataclass(frozen=True, slots=True)
class QuestionResolution:
    """One audited question, its agreed answer, and the evidence behind it."""

    question_key: str
    proposition: str
    resolution: str
    cycle: int
    grounded_in: tuple[str, ...]
    responders: tuple[str, ...]
    evidence_signature: str
    evidence_requirement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_key": self.question_key,
            "proposition": self.proposition,
            "resolution": self.resolution,
            "cycle": self.cycle,
            "grounded_in": list(self.grounded_in),
            "responders": list(self.responders),
            "evidence_signature": self.evidence_signature,
            "evidence_requirement": self.evidence_requirement,
        }


def resolution_node_id(question_key: str) -> str:
    return f"RESOLVED_QUESTION:{str(question_key).strip()}"


def evidence_signature(
    graph: SemanticGraph | None, grounded_in: Iterable[str],
) -> str:
    """Digest the committed state an answer depended on.

    The digest covers each grounding node's kind, label, salient attributes,
    and outgoing relations. New facts attached to a grounding node therefore
    change the signature, which is what reopens the question.
    """
    parts: list[str] = []
    for node_id in sorted({str(value).strip() for value in grounded_in if str(value).strip()}):
        node = graph.nodes.get(node_id) if graph is not None else None
        if node is None:
            parts.append(f"{node_id}|ABSENT")
            continue
        attributes = "&".join(
            f"{key}={node.attributes[key]!r}"
            for key in sorted(node.attributes)
            if key in _SIGNATURE_ATTRIBUTES
        )
        relations = ",".join(sorted(
            f"{edge.relation}->{edge.target}" for edge in graph.outgoing(node_id)
        ))
        parts.append(f"{node_id}|{node.kind}|{node.label}|{attributes}|{relations}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


def resolve_grounding_ids(
    graph: SemanticGraph | None, references: Iterable[str],
) -> tuple[str, ...]:
    """Map a question's grounding references onto current-run graph nodes.

    ProblemState grounds a delegate question in scenario clause ids while the
    ledger needs node ids, so a clause reference expands to the nodes that
    carry that clause. Anything that resolves to nothing is dropped, which is
    what keeps a resolution from claiming grounding it does not have.
    """
    if graph is None:
        return ()
    nodes_by_clause: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        clause_id = str(
            node.attributes.get("clause_id")
            or node.attributes.get("source_clause_id")
            or ""
        )
        if clause_id:
            nodes_by_clause.setdefault(clause_id, []).append(node.id)
    resolved: list[str] = []
    for reference in references:
        value = str(reference).strip()
        if not value:
            continue
        if value in graph.nodes:
            resolved.append(value)
            continue
        resolved.extend(sorted(nodes_by_clause.get(value, [])))
    return tuple(dict.fromkeys(resolved))


def _named_evidence_requirement(
    proposition: str, responders: Sequence[Any],
) -> str:
    """Name what would have to be known for the question to become answerable.

    The delegates already said this when they explained why they could not
    resolve the audit. Promoting their explanations to a requirement is what
    turns a spent audit into a standing information demand rather than a
    question the next cycle asks again in the same words.
    """
    reasons = [
        " ".join(str(
            getattr(candidate, "audit_framework_explanation", "")
        ).split())
        for candidate in responders
    ]
    named = list(dict.fromkeys(reason for reason in reasons if reason))
    if named:
        return "; ".join(named[:3])[:240]
    return f"evidence bearing on: {' '.join(str(proposition).split())}"[:240]


def resolve_audited_question(
    candidates: Sequence[Any],
    *,
    question_key: str,
    proposition: str,
    cycle: int,
    grounded_in: Sequence[str],
    graph: SemanticGraph | None,
) -> QuestionResolution | None:
    """Report what an audit established about one question, if anything.

    A question settles substantively when at least two delegates tested it and
    every tested delegate reported the same effect on the recommendation. When
    the delegates instead agree that current evidence cannot decide it, that is
    recorded as a typed negative result carrying the evidence it would take to
    reopen the matter. Either way the audit slot is spent once, not each cycle.
    A contested response records nothing: disagreement about whether an issue
    matters is itself unfinished deliberation.
    """
    question_key = str(question_key).strip()
    if not question_key.startswith(("QUESTION:", "CHALLENGE:")):
        return None
    valid = [
        candidate for candidate in candidates
        if getattr(candidate, "schema_valid", False)
    ]

    def participation(candidate: Any) -> str:
        return str(getattr(candidate, "audit_participation", "")).strip().upper()

    if any(participation(candidate) in BLOCKING_PARTICIPATION for candidate in valid):
        return None
    tested = [
        candidate for candidate in valid
        if participation(candidate) in TESTED_PARTICIPATION
    ]
    responded = [
        candidate for candidate in valid
        if participation(candidate) in TESTED_PARTICIPATION
        or participation(candidate) == UNRESOLVED_PARTICIPATION
    ]
    if len(responded) < MINIMUM_RESPONDERS:
        return None
    effects = {
        str(getattr(candidate, "audit_internal_effect", "")).strip().upper()
        for candidate in tested
    }
    if (
        len(tested) >= MINIMUM_RESPONDERS
        and len(effects) == 1
        and effects <= SETTLING_EFFECTS
    ):
        resolution, requirement = next(iter(effects)), ""
    elif effects <= UNSETTLED_EFFECTS:
        resolution = UNRESOLVABLE_RESOLUTION
        requirement = _named_evidence_requirement(proposition, responded)
    else:
        # Delegates reported different substantive effects, so the audit found
        # real divergence rather than an answer or a shared evidence gap.
        return None
    grounding = resolve_grounding_ids(graph, grounded_in)
    if not grounding:
        return None
    return QuestionResolution(
        question_key=question_key,
        proposition=" ".join(str(proposition).split())[:240] or question_key,
        resolution=resolution,
        cycle=max(1, int(cycle)),
        grounded_in=grounding,
        responders=tuple(sorted(
            str(getattr(candidate, "specialist", "")) for candidate in responded
        )),
        evidence_signature=evidence_signature(graph, grounding),
        evidence_requirement=requirement,
    )


def commit_question_resolution(
    store: SemanticGraphStore,
    resolution: QuestionResolution,
    *,
    source: str = "workspace_audit",
) -> GraphTransactionRecord:
    """Record a settled question atomically, or reject it without mutating."""
    proposal = resolution.to_dict()
    missing = [
        node_id for node_id in resolution.grounded_in
        if node_id not in store.graph.nodes
    ]
    if missing:
        record = GraphTransactionRecord(
            resolution.cycle, source, RESOLUTION_OPERATION, "REJECTED", proposal,
            [f"question grounding is not current-run graph state: {', '.join(missing[:3])}"],
            previous_state_preserved=True, retryable=False,
        )
        store.transactions.append(record)
        return record

    node_id = resolution_node_id(resolution.question_key)
    provenance = (f"cycle:{resolution.cycle}", "audited_question_resolution")
    delta = SemanticGraph()
    delta.add_node(SemanticNode(
        node_id, RESOLUTION_KIND, resolution.proposition, provenance,
        {
            "record_kind": "AUDITED_QUESTION_RESOLUTION",
            "question_key": resolution.question_key,
            "resolution": resolution.resolution,
            "resolved_cycle": resolution.cycle,
            "responders": list(resolution.responders),
            "grounded_in": list(resolution.grounded_in),
            "evidence_signature": resolution.evidence_signature,
            "evidence_requirement": resolution.evidence_requirement,
        },
    ))
    for grounding_id in resolution.grounded_in:
        delta.add_edge(SemanticEdge(
            node_id, "RESOLVES", grounding_id,
            justification=resolution.resolution, provenance=provenance,
        ))

    try:
        prospective = merge_graphs([store.graph, delta])
    except ValueError as exc:
        record = GraphTransactionRecord(
            resolution.cycle, source, RESOLUTION_OPERATION, "REJECTED", proposal,
            [str(exc)], previous_state_preserved=True, retryable=False,
        )
        store.transactions.append(record)
        return record
    validation = validate_graph(prospective)
    if not validation.valid:
        record = GraphTransactionRecord(
            resolution.cycle, source, RESOLUTION_OPERATION, "REJECTED", proposal,
            validation.errors, previous_state_preserved=True, retryable=False,
        )
        store.transactions.append(record)
        return record

    store.graph = prospective
    record = GraphTransactionRecord(
        resolution.cycle, source, RESOLUTION_OPERATION, "COMMITTED", proposal,
        [], previous_state_preserved=False, retryable=False,
    )
    store.transactions.append(record)
    return record


def question_resolution_index(
    graph: SemanticGraph | None,
) -> dict[str, dict[str, Any]]:
    """Project every recorded resolution with its live settled/reopened state."""
    if graph is None:
        return {}
    index: dict[str, dict[str, Any]] = {}
    for node in graph.nodes.values():
        if node.attributes.get("record_kind") != "AUDITED_QUESTION_RESOLUTION":
            continue
        question_key = str(node.attributes.get("question_key", ""))
        if not question_key:
            continue
        grounded_in = [str(value) for value in node.attributes.get("grounded_in", [])]
        recorded = str(node.attributes.get("evidence_signature", ""))
        current = evidence_signature(graph, grounded_in)
        index[question_key] = {
            "question_key": question_key,
            "resolution_node_id": node.id,
            "proposition": node.label,
            "resolution": str(node.attributes.get("resolution", "")),
            "resolved_cycle": int(node.attributes.get("resolved_cycle", 0) or 0),
            "responders": list(node.attributes.get("responders", [])),
            "grounded_in": grounded_in,
            "evidence_requirement": str(node.attributes.get("evidence_requirement", "")),
            "status": "SETTLED" if current == recorded else "REOPENED",
            "reopened_by_evidence_change": current != recorded,
        }
    return index


def settled_question_keys(graph: SemanticGraph | None) -> frozenset[str]:
    """Question keys whose answer still rests on unchanged committed evidence."""
    return frozenset(
        key for key, record in question_resolution_index(graph).items()
        if record["status"] == "SETTLED"
    )
