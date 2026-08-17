"""Typed graph validation for synthesis-failure contingency branches."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode, validate_graph


class ContingencyPredicateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    predicate: str = Field(alias="p", min_length=8, max_length=160)
    failure_effect: str = Field(alias="x", min_length=8, max_length=160)


@dataclass(slots=True)
class ContingencyGraphValidation:
    valid: bool
    graph: SemanticGraph = field(default_factory=SemanticGraph)
    errors: list[str] = field(default_factory=list)


def compile_contingency_graph(
    payload: dict[str, Any],
    synthesis_action: str,
    fallback_actions: Sequence[str],
) -> ContingencyGraphValidation:
    """Validate untrusted typed claims, then build a branch-local graph."""
    try:
        proposal = ContingencyPredicateProposal.model_validate(payload)
    except ValidationError as exc:
        return ContingencyGraphValidation(
            False,
            errors=[
                f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in exc.errors(include_url=False)
            ],
        )
    errors: list[str] = []
    if len(fallback_actions) != 2:
        errors.append("contingency requires exactly two canonical fallbacks")
    graph = SemanticGraph()
    graph.add_node(SemanticNode(
        "SYNTHESIS", "ACTION", synthesis_action, ("accepted_synthesis",),
        {"branch_role": "FAILED_SYNTHESIS"},
    ))
    for index, action in enumerate(fallback_actions[:2]):
        graph.add_node(SemanticNode(
            f"A{index}", "ACTION", action, ("original_fallback",),
            {"branch_role": "FALLBACK", "canonical_action_id": f"A{index}"},
        ))
    predicate_key = " ".join(proposal.predicate.casefold().split())
    graph.add_node(SemanticNode(
        "CONTINGENCY_REQUIRED", "CONDITION", proposal.predicate,
        (synthesis_action,),
        {"predicate_key": predicate_key, "truth_state": True},
    ))
    graph.add_node(SemanticNode(
        "CONTINGENCY_FAILURE", "CONDITION", f"not ({proposal.predicate})",
        (proposal.failure_effect,),
        {
            "predicate_key": predicate_key,
            "truth_state": False,
            "failure_effect": proposal.failure_effect,
        },
    ))
    graph.add_edge(SemanticEdge(
        "SYNTHESIS", "REQUIRES", "CONTINGENCY_REQUIRED",
        justification="typed synthesis dependency",
        provenance=(synthesis_action,),
    ))
    graph.add_edge(SemanticEdge(
        "CONTINGENCY_FAILURE", "NEGATES", "CONTINGENCY_REQUIRED",
        justification="same predicate with opposite truth state",
        provenance=(proposal.failure_effect,),
    ))
    graph.add_edge(SemanticEdge(
        "CONTINGENCY_FAILURE", "DISABLES", "SYNTHESIS",
        justification=proposal.failure_effect,
        provenance=(proposal.failure_effect,),
    ))
    errors.extend(validate_graph(graph).errors)
    errors.extend(validate_contingency_graph_dict(
        graph.to_dict(), synthesis_action, fallback_actions,
        require_fallback_availability=False,
    ))
    return ContingencyGraphValidation(
        not errors, graph, list(dict.fromkeys(errors))
    )


def validate_contingency_graph_dict(
    graph: dict[str, Any],
    synthesis_action: str,
    fallback_actions: Sequence[str],
    *,
    require_fallback_availability: bool = True,
) -> list[str]:
    """Revalidate the committed branch topology at the engine boundary."""
    errors: list[str] = []
    nodes = {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    }
    edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    expected_nodes = {"SYNTHESIS", "A0", "A1", "CONTINGENCY_REQUIRED", "CONTINGENCY_FAILURE"}
    if not expected_nodes.issubset(nodes):
        errors.append("contingency graph is missing required branch nodes")
        return errors
    if nodes["SYNTHESIS"].get("label") != synthesis_action:
        errors.append("contingency graph synthesis identity mismatch")
    if len(fallback_actions) != 2 or any(
        nodes[f"A{index}"].get("label") != action
        for index, action in enumerate(fallback_actions[:2])
    ):
        errors.append("contingency graph fallback identity mismatch")
    required_attributes = nodes["CONTINGENCY_REQUIRED"].get("attributes", {})
    failure_attributes = nodes["CONTINGENCY_FAILURE"].get("attributes", {})
    if (
        required_attributes.get("predicate_key")
        != failure_attributes.get("predicate_key")
        or required_attributes.get("truth_state") is not True
        or failure_attributes.get("truth_state") is not False
    ):
        errors.append("failure condition is not the typed negation of the dependency")
    triples = {
        (edge.get("source"), edge.get("relation"), edge.get("target"))
        for edge in edges
    }
    required_triples = {
        ("SYNTHESIS", "REQUIRES", "CONTINGENCY_REQUIRED"),
        ("CONTINGENCY_FAILURE", "NEGATES", "CONTINGENCY_REQUIRED"),
        ("CONTINGENCY_FAILURE", "DISABLES", "SYNTHESIS"),
    }
    if require_fallback_availability:
        required_triples.update({
            ("CONTINGENCY_FAILURE", "PRESERVES_AVAILABILITY", "A0"),
            ("CONTINGENCY_FAILURE", "PRESERVES_AVAILABILITY", "A1"),
        })
    if missing := required_triples - triples:
        errors.append(
            "contingency graph lacks required dependency or availability edges: "
            + ", ".join(f"{source}->{relation}->{target}" for source, relation, target in sorted(missing))
        )
    if any(
        source == "CONTINGENCY_FAILURE" and relation == "DISABLES" and target in {"A0", "A1"}
        for source, relation, target in triples
    ):
        errors.append("failure condition disables an original fallback")
    return list(dict.fromkeys(errors))


def certify_fallback_availability(
    graph: dict[str, Any],
    synthesis_action: str,
    fallback_actions: Sequence[str],
    statuses: dict[str, str],
    reasons: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    """Commit verifier-owned availability edges only when both fallbacks pass."""
    certified = deepcopy(graph)
    errors = validate_contingency_graph_dict(
        certified, synthesis_action, fallback_actions,
        require_fallback_availability=False,
    )
    if set(statuses) != {"A0", "A1"}:
        errors.append("independent verifier did not classify both fallbacks")
    for action_id in ("A0", "A1"):
        status = statuses.get(action_id, "UNKNOWN")
        reason = " ".join(reasons.get(action_id, "").split())
        if status != "AVAILABLE":
            errors.append(
                f"{action_id} is not independently established as available ({status})"
            )
        if len(reason.split()) < 3:
            errors.append(f"{action_id} availability lacks an independent reason")
    if errors:
        return certified, list(dict.fromkeys(errors))
    certified.setdefault("edges", []).extend([
        {
            "source": "CONTINGENCY_FAILURE",
            "relation": "PRESERVES_AVAILABILITY",
            "target": action_id,
            "condition": "CONTINGENCY_FAILURE",
            "justification": reasons[action_id],
            "provenance": ("independent_feasibility_verifier", reasons[action_id]),
        }
        for action_id in ("A0", "A1")
    ])
    errors.extend(validate_contingency_graph_dict(
        certified, synthesis_action, fallback_actions,
        require_fallback_availability=True,
    ))
    return certified, list(dict.fromkeys(errors))
