from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

from .semantic_graph import (
    SemanticEdge, SemanticGraph, SemanticNode, merge_graphs, validate_graph,
)


REVERSAL_TRANSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "from_action": {"type": "string"},
        "to_action": {"type": "string"},
        "condition": {"type": "string"},
        "affected_action": {"type": "string"},
        "effect_on_affected_action": {
            "type": "string",
            "enum": ["MORE_ATTRACTIVE", "LESS_ATTRACTIVE", "UNKNOWN"],
        },
        "justification": {"type": "string"},
    },
    "required": [
        "from_action", "to_action", "condition", "affected_action",
        "effect_on_affected_action", "justification",
    ],
    "additionalProperties": False,
}


def _obvious_effect(condition: str) -> str:
    """Infer only high-precision monotonic changes; leave ambiguous prose unknown."""
    text = " ".join(condition.casefold().split())
    adverse = bool(re.search(
        r"\b(?:risk|harm|mortality|deaths?|casualties|cost|failure|coercion|violation)\b",
        text,
    ))
    if not adverse:
        return "UNKNOWN"
    if re.search(r"\b(?:rise[sd]?|increase[sd]?|higher|worsen[sed]*|exceed[sed]*|above)\b", text):
        return "LESS_ATTRACTIVE"
    if re.search(r"\b(?:fall[sen]*|decrease[sd]?|lower|reduce[sd]?|below)\b", text):
        return "MORE_ATTRACTIVE"
    return "UNKNOWN"


def validate_reversal_transitions(
    transitions: Sequence[dict[str, Any]], actions: Sequence[str],
    selected_action: str = "",
) -> list[str]:
    """Return direction errors for typed action-switch conditions."""
    errors: list[str] = []
    allowed = set(actions)
    graphs: list[SemanticGraph] = []
    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            errors.append(f"reversal {index + 1} is not a typed transition")
            continue
        source = str(transition.get("from_action", ""))
        target = str(transition.get("to_action", ""))
        affected = str(transition.get("affected_action", ""))
        effect = str(transition.get("effect_on_affected_action", ""))
        inferred_effect = _obvious_effect(str(transition.get("condition", "")))
        if inferred_effect != "UNKNOWN" and effect != inferred_effect:
            errors.append(
                f"reversal {index + 1} labels the condition's monotonic effect backwards"
            )
        if source == target:
            errors.append(f"reversal {index + 1} does not switch actions")
        if selected_action and source != selected_action:
            errors.append(
                f"reversal {index + 1} does not start from the selected action"
            )
        if allowed and ({source, target, affected} - allowed):
            errors.append(f"reversal {index + 1} references an unknown action")
        if affected == target and effect == "LESS_ATTRACTIVE":
            errors.append(
                f"reversal {index + 1} switches toward an action made less attractive"
            )
        if affected == source and effect == "MORE_ATTRACTIVE":
            errors.append(
                f"reversal {index + 1} switches away from an action made more attractive"
            )
        graphs.append(reversal_transition_graph(transition, index=index))
    if graphs:
        errors.extend(validate_graph(merge_graphs(graphs)).errors)
    return errors


def reversal_transition_graph(
    transition: dict[str, Any], *, index: int = 0
) -> SemanticGraph:
    """Compile one reversal into a graph without interpreting scenario vocabulary."""
    source = str(transition.get("from_action", ""))
    target = str(transition.get("to_action", ""))
    affected = str(transition.get("affected_action", ""))
    condition = str(transition.get("condition", ""))
    effect = str(transition.get("effect_on_affected_action", "UNKNOWN"))
    justification = str(transition.get("justification", ""))
    prefix = f"reversal:{index}"
    action_ids = {action: f"action:{action}" for action in {source, target, affected}}
    graph = SemanticGraph()
    for action, node_id in action_ids.items():
        graph.add_node(SemanticNode(node_id, "ACTION", action, (prefix,)))
    condition_id = f"condition:{index}:{condition}"
    graph.add_node(SemanticNode(condition_id, "CONDITION", condition, (prefix,)))
    relation = effect if effect in {"MORE_ATTRACTIVE", "LESS_ATTRACTIVE"} else "AFFECTS"
    graph.add_edge(SemanticEdge(
        condition_id, relation, action_ids[affected], condition=condition_id,
        justification=justification, provenance=(prefix,),
    ))
    graph.add_edge(SemanticEdge(
        action_ids[source], "SWITCHES_TO", action_ids[target],
        condition=condition_id, justification=justification, provenance=(prefix,),
    ))
    return graph


def render_reversal_transition(value: Any) -> str:
    """Render typed transitions while retaining compatibility with old artifacts."""
    if not isinstance(value, dict):
        return str(value)
    source = value.get("from_action", "the current action")
    target = value.get("to_action", "the alternative")
    condition = value.get("condition", "material facts change")
    justification = str(value.get("justification", "")).strip()
    text = f"Switch from {source} to {target} if {condition}"
    return text + (f", because {justification}" if justification else "")
