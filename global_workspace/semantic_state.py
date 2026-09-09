"""Read-only projections from committed semantic graph state.

The graph remains the mutation boundary.  Projections are deliberately small,
serializable views for downstream consumers that should not inspect raw delegate
text or reconstruct graph topology independently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from .scenario_semantics import (
    normalize_shared_dimension,
    project_grounded_action_effects,
    semantic_action_key,
)
from .semantic_graph import SemanticGraph, validate_graph
from .world_state import (
    counts_as_obtained_outcome,
    counts_as_settled_adverse,
    is_averted_risk_not_obtained_benefit_consequence,
)


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


@dataclass(frozen=True, slots=True)
class ProblemShapeRelation:
    relation: str
    source_action_ids: tuple[str, ...] = ()
    statement: str = ""
    provenance: str = "DERIVED_STRUCTURE"
    confidence: float = 0.0
    support_node_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_action_ids"] = list(self.source_action_ids)
        data["support_node_ids"] = list(self.support_node_ids)
        return data


@dataclass(frozen=True, slots=True)
class ActionDimensionState:
    action_id: str
    affected_subject: str
    dimension: str
    direction: str
    magnitude_or_qualifier: str = "UNKNOWN"
    provenance: tuple[str, ...] = ()
    confidence: float = 0.0
    epistemic_status: str = "UNKNOWN"
    support_node_ids: tuple[str, ...] = ()
    representation_layer: str = "SHARED_DESCRIPTIVE_FACT"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["provenance"] = list(self.provenance)
        data["support_node_ids"] = list(self.support_node_ids)
        return data


@dataclass(frozen=True, slots=True)
class FrameworkDimensionPriority:
    framework: str
    dimension: str
    relation: str
    compared_to_dimension: str = ""
    basis: str = ""
    provenance: tuple[str, ...] = ()
    confidence: float = 0.0
    epistemic_status: str = "UNKNOWN"
    support_node_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["provenance"] = list(self.provenance)
        data["support_node_ids"] = list(self.support_node_ids)
        return data


@dataclass(slots=True)
class AuthoritativeSemanticState:
    """Stable downstream view of decision-critical committed graph state."""

    version: int = 3
    selected_action: str = ""
    selected_action_id: str = ""
    selected_action_key: str = ""
    factual_reversal_boundaries: list[CommittedReversalBoundary] = field(
        default_factory=list
    )
    problem_shape_relations: list[ProblemShapeRelation] = field(default_factory=list)
    action_dimension_states: list[ActionDimensionState] = field(default_factory=list)
    framework_dimension_priorities: list[FrameworkDimensionPriority] = field(default_factory=list)
    grounded_action_effects: list[dict[str, Any]] = field(default_factory=list)
    rawlsian_positions: list[dict[str, Any]] = field(default_factory=list)
    utilitarian_consequences: list[dict[str, Any]] = field(default_factory=list)
    deontological_assessments: list[dict[str, Any]] = field(default_factory=list)
    virtue_assessments: list[dict[str, Any]] = field(default_factory=list)
    decision_variables: list[dict[str, Any]] = field(default_factory=list)
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
            "problem_shape_relations": [
                relation.to_dict() for relation in self.problem_shape_relations
            ],
            "action_dimension_states": [
                dimension.to_dict() for dimension in self.action_dimension_states
            ],
            "framework_dimension_priorities": [
                priority.to_dict() for priority in self.framework_dimension_priorities
            ],
            "grounded_action_effects": [dict(effect) for effect in self.grounded_action_effects],
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
            "decision_variables": [
                dict(variable) for variable in self.decision_variables
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


def _duration_bucket(text: str) -> str:
    lowered = " ".join(str(text).casefold().split())
    if not lowered:
        return "UNKNOWN"
    present_markers = (
        "immediate", "at once", "now", "instant", "short term", "short-term",
        "hours", "days", "weeks", "today",
    )
    delayed_markers = (
        "future", "later", "long term", "long-term", "multi-year", "months",
        "years", "ongoing",
    )
    if any(marker in lowered for marker in present_markers):
        return "PRESENT"
    if any(marker in lowered for marker in delayed_markers):
        return "DELAYED"
    return "UNKNOWN"


def _probability_bucket(text: str) -> str:
    lowered = " ".join(str(text).casefold().split())
    if not lowered:
        return "UNKNOWN"
    certain_markers = (
        "~1", "1.0", "certain", "guaranteed", "deterministic", "near certain",
        "near-certain",
    )
    uncertain_markers = (
        "unknown", "unverified", "uncertain", "probabilistic", "possible",
        "might", "may",
    )
    if any(marker in lowered for marker in certain_markers):
        return "CERTAIN"
    if any(marker in lowered for marker in uncertain_markers):
        return "UNCERTAIN"
    return "UNKNOWN"


def _utilitarian_profile(consequences: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    profile: dict[str, list[dict[str, Any]]] = {
        "present_benefit": [],
        "present_harm": [],
        "delayed_benefit": [],
        "delayed_harm": [],
        "certain": [],
        "uncertain": [],
    }
    for consequence in consequences:
        direction = str(consequence.get("direction", "")).upper()
        profile_direction = "HARM" if direction == "OPPORTUNITY_COST" else direction
        if profile_direction not in {"BENEFIT", "HARM"}:
            continue
        duration_bucket = _duration_bucket(str(consequence.get("duration", "")))
        if duration_bucket in {"PRESENT", "DELAYED"}:
            profile[f"{duration_bucket.lower()}_{profile_direction.lower()}"].append(consequence)
        certainty_bucket = _probability_bucket(str(consequence.get("probability", "")))
        if certainty_bucket in {"CERTAIN", "UNCERTAIN"}:
            profile[certainty_bucket.lower()].append(consequence)
    return profile


def _support_ids(*groups: list[dict[str, Any]]) -> tuple[str, ...]:
    support: list[str] = []
    for group in groups:
        for item in group:
            for key in ("consequence_node_id", "scope_node_id"):
                value = str(item.get(key, "")).strip()
                if value:
                    support.append(value)
    return tuple(dict.fromkeys(support))


def _consequence_dimension(consequence: dict[str, Any]) -> str:
    text = " ".join((
        str(consequence.get("outcome", "")),
        str(consequence.get("scope", "")),
    )).casefold()
    if re.search(r"\b(?:libert|privacy|autonomy|movement|choice|consent|right)\w*\b", text):
        return "LIBERTY_AUTONOMY"
    if re.search(r"\b(?:income|wealth|material|poverty|poor|aid|redistribut|resource|inequal)\w*\b", text):
        return "MATERIAL_FLOOR"
    return ""


def _derive_action_dimensions(
    rawlsian_positions: list[dict[str, Any]],
    utilitarian_consequences: list[dict[str, Any]],
    action_ids: list[str],
    grounded_action_effects: list[dict[str, Any]] | None = None,
) -> list[ActionDimensionState]:
    util_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    util_by_action: dict[str, list[dict[str, Any]]] = {}
    for consequence in utilitarian_consequences:
        action_id = str(consequence.get("canonical_action_id", ""))
        util_by_action.setdefault(action_id, []).append(consequence)
        dimension = _consequence_dimension(consequence)
        if dimension:
            util_by_key.setdefault((action_id, dimension), []).append(consequence)

    states: list[ActionDimensionState] = []
    represented: set[tuple[str, str]] = set()
    grounded_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    canonical_effects = list(grounded_action_effects or [])
    actions_with_scenario_grounding = {
        str(effect.get("action_id", "")) for effect in canonical_effects
        if str(effect.get("epistemic_status", "")).upper() == "SCENARIO_GROUNDED"
    }
    actions_with_synthesis_grounding = {
        str(effect.get("action_id", "")) for effect in canonical_effects
        if str(effect.get("epistemic_status", "")).upper() == "SYNTHESIS_GROUNDED"
    }
    canonical_effects = [
        effect for effect in canonical_effects
        if str(effect.get("action_id", "")) not in actions_with_scenario_grounding
        or str(effect.get("epistemic_status", "")).upper() == "SCENARIO_GROUNDED"
    ]
    canonical_effects = [
        effect for effect in canonical_effects
        if str(effect.get("action_id", "")) not in actions_with_synthesis_grounding
        or str(effect.get("epistemic_status", "")).upper() == "SYNTHESIS_GROUNDED"
    ]
    for effect in canonical_effects:
        key = (
            str(effect.get("action_id", "")),
            normalize_shared_dimension(effect.get("dimension", "UNKNOWN")),
            str(effect.get("affected_subject", "affected constituency")),
        )
        grounded_groups.setdefault(key, []).append(effect)
    for (action_id, dimension, subject), effects in grounded_groups.items():
        directions = {
            str(effect.get("direction", "UNCERTAIN")).upper() for effect in effects
        }
        direction = next(iter(directions)) if len(directions) == 1 else "MIXED"
        qualifiers = list(dict.fromkeys(
            str(effect.get("magnitude_or_qualifier", "UNKNOWN")).upper()
            for effect in effects
        ))
        epistemic_statuses = {
            str(effect.get("epistemic_status", "UNKNOWN")).upper()
            for effect in effects
        }
        represented.add((action_id, dimension))
        states.append(ActionDimensionState(
            action_id=action_id,
            affected_subject=subject,
            dimension=dimension,
            direction=direction,
            magnitude_or_qualifier=qualifiers[0] if len(qualifiers) == 1 else "MIXED",
            provenance=tuple(dict.fromkeys(
                str(value)
                for effect in effects
                for value in effect.get("provenance", [])
                if str(value)
            )),
            confidence=min(float(effect.get("confidence", 0.0)) for effect in effects),
            epistemic_status=(
                "SCENARIO_GROUNDED"
                if epistemic_statuses == {"SCENARIO_GROUNDED"}
                else "SYNTHESIS_GROUNDED"
                if epistemic_statuses == {"SYNTHESIS_GROUNDED"}
                else "ACTION_TEXT_GROUNDED"
            ),
            support_node_ids=tuple(dict.fromkeys(
                str(effect.get("consequence_id", ""))
                for effect in effects if str(effect.get("consequence_id", ""))
            )),
        ))
    # Rawlsian positions and other framework ledgers remain available to their
    # specialists and to framework_dimension_priorities. They never manufacture
    # shared dimensional facts: only current action/scenario grounding can do so.

    # Feasibility here is epistemic, not a moral score: stated action effects
    # are established enough to compare, while synthesis effects supported only
    # by inference or unknown probability remain conditional.
    for action_id in action_ids:
        evidence = util_by_action.get(action_id, [])
        if not evidence:
            continue
        uncertain = all(
            str(item.get("support", "")).upper() != "STATED"
            for item in evidence
        ) or any(
            str(item.get("probability", "")).upper() == "UNKNOWN"
            and str(item.get("direction", "")).upper() == "BENEFIT"
            for item in evidence
        )
        states.append(ActionDimensionState(
            action_id=action_id,
            affected_subject="implementation",
            dimension="FEASIBILITY",
            direction="UNCERTAIN" if uncertain else "ESTABLISHED",
            magnitude_or_qualifier="CONDITION_DEPENDENT" if uncertain else "HIGH",
            provenance=("committed_consequence_evidence",),
            confidence=0.55 if uncertain else 0.78,
            epistemic_status="INFERRED" if uncertain else "GROUNDED",
            support_node_ids=tuple(
                str(item.get("consequence_node_id", "")) for item in evidence
            ),
        ))
    return sorted(states, key=lambda item: (item.action_id, item.dimension, item.affected_subject))


def _derive_framework_dimension_priorities(
    rawlsian_positions: list[dict[str, Any]],
    utilitarian_consequences: list[dict[str, Any]] | None = None,
    shared_dimension_states: list[ActionDimensionState] | None = None,
) -> list[FrameworkDimensionPriority]:
    """Project framework ordering separately from descriptive action facts."""
    rankings = {
        str(position.get("ranking_basis", "")).upper()
        for position in rawlsian_positions
    }
    dimensions = {
        normalize_shared_dimension(position.get("dimension", ""))
        for position in rawlsian_positions
    }
    priorities: list[FrameworkDimensionPriority] = []
    if "LEXICAL_BASIC_LIBERTY" in rankings and "LIBERTY_AUTONOMY" in dimensions:
        lower_dimensions = sorted(dimension for dimension in dimensions if dimension not in {
            "", "UNKNOWN", "LIBERTY_AUTONOMY",
        })
        support = tuple(dict.fromkeys(
            str(position.get("assessment_node_id", ""))
            for position in rawlsian_positions
            if str(position.get("assessment_node_id", ""))
        ))
        provenance = tuple(dict.fromkeys(
            str(value)
            for position in rawlsian_positions
            for value in position.get("provenance", [])
            if str(value)
        ))
        for lower_dimension in lower_dimensions or [""]:
            priorities.append(FrameworkDimensionPriority(
                framework="RAWLSIAN",
                dimension="LIBERTY_AUTONOMY",
                relation="LEXICAL_PRIORITY_OVER",
                compared_to_dimension=lower_dimension,
                basis="LEXICAL_BASIC_LIBERTY",
                provenance=provenance,
                confidence=0.95,
                epistemic_status="FRAMEWORK_COMMITTED",
                support_node_ids=support,
            ))
    utilitarian_consequences = utilitarian_consequences or []
    util_dimensions = {
        dimension for consequence in utilitarian_consequences
        if (dimension := _consequence_dimension(consequence))
    }
    if {"LIBERTY_AUTONOMY", "MATERIAL_FLOOR"} <= util_dimensions:
        priorities.append(FrameworkDimensionPriority(
            framework="UTILITARIAN",
            dimension="LIBERTY_AUTONOMY",
            relation="AGGREGATES_WITH",
            compared_to_dimension="MATERIAL_FLOOR",
            basis="CONSEQUENCE_AGGREGATION",
            provenance=("committed_utilitarian_consequence_ledger",),
            confidence=0.85,
            epistemic_status="COMPARISON_REQUIRED",
            support_node_ids=tuple(
                str(consequence.get("consequence_node_id", ""))
                for consequence in utilitarian_consequences
                if str(consequence.get("consequence_node_id", ""))
            ),
        ))
    shared_dimensions = {
        state.dimension for state in (shared_dimension_states or [])
        if state.dimension not in {"", "UNKNOWN", "OTHER", "FEASIBILITY"}
    }
    if {"LIBERTY_AUTONOMY", "MATERIAL_FLOOR"} <= shared_dimensions:
        priorities.append(FrameworkDimensionPriority(
            framework="CARE",
            dimension="MATERIAL_FLOOR",
            relation="RELATIONAL_TENSION_WITH",
            compared_to_dimension="LIBERTY_AUTONOMY",
            basis="DEPENDENCY_RELIEF_AND_NONDOMINATION",
            provenance=("care_framework_interpretive_rule",),
            confidence=0.8,
            epistemic_status="CONTEXT_SENSITIVE_COMPARISON",
            support_node_ids=tuple(dict.fromkeys(
                str(position.get("assessment_node_id", ""))
                for position in rawlsian_positions
                if str(position.get("assessment_node_id", ""))
            )),
        ))
    return priorities


def _action_signature(graph: SemanticGraph, action_node_id: str) -> dict[str, Any]:
    action = graph.nodes.get(action_node_id)
    if action is None or action.kind != "ACTION":
        return {}
    targets = [
        graph.nodes[edge.target]
        for edge in graph.outgoing(action.id, "TARGETS")
        if edge.target in graph.nodes and graph.nodes[edge.target].kind == "TARGET"
    ]
    consequences = [
        graph.nodes[edge.target]
        for edge in graph.outgoing(action.id, "HAS_CONSEQUENCE")
        if edge.target in graph.nodes and graph.nodes[edge.target].kind == "CONSEQUENCE"
    ]
    return {
        "action_id": _action_identity(action)[0],
        "action_key": _action_identity(action)[1],
        "target_labels": {target.label.casefold() for target in targets if target.label.strip()},
        "consequence_labels": {
            consequence.label.casefold()
            for consequence in consequences
            if consequence.label.strip()
            and str(consequence.attributes.get("polarity", "")).upper() != "FOREGONE"
            and str(consequence.attributes.get("directness", "")).upper() != "FOREGONE"
        },
        "adverse_count": sum(
            1 for consequence in consequences
            if counts_as_settled_adverse(
                polarity=str(consequence.attributes.get("polarity", "")),
                modality=str(consequence.attributes.get("modality", "")),
                likelihood_qualifiers=consequence.attributes.get(
                    "likelihood_qualifiers", (),
                ),
            )
        ),
        "beneficial_count": sum(
            1 for consequence in consequences
            if str(consequence.attributes.get("polarity", "")).upper() == "BENEFICIAL"
            and counts_as_obtained_outcome(
                polarity=str(consequence.attributes.get("polarity", "")),
                modality=str(consequence.attributes.get("modality", "")),
                likelihood_qualifiers=consequence.attributes.get(
                    "likelihood_qualifiers", (),
                ),
            )
            and not is_averted_risk_not_obtained_benefit_consequence(graph, consequence)
        ),
        "support_nodes": tuple(
            node.id for node in [*targets, *consequences]
        ),
    }


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

    state.grounded_action_effects = [
        effect.to_dict() for effect in project_grounded_action_effects(graph)
    ]

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
            and node.attributes.get("assessment_role") != "COMPETING"
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
    state.decision_variables = sorted(
        [
            {
                "decision_variable_node_id": node.id,
                "kind": node.kind,
                "label": node.label,
                **dict(node.attributes),
                "provenance": list(node.provenance),
            }
            for node in graph.nodes.values()
            if node.kind == "CONDITION"
            and node.attributes.get("probe_kind") == "MISSING_DECISION_VARIABLE"
        ],
        key=lambda variable: (
            str(variable.get("selected_action_id", "")),
            str(variable.get("decision_variable_node_id", "")),
        ),
    )

    action_nodes = [
        node for node in graph.nodes.values()
        if node.kind == "ACTION"
    ]
    action_signatures = {
        node.id: _action_signature(graph, node.id)
        for node in action_nodes
    }
    utilitarian_by_action: dict[str, list[dict[str, Any]]] = {}
    for consequence in state.utilitarian_consequences:
        action_id = str(consequence.get("canonical_action_id", "")).strip()
        if not action_id:
            continue
        utilitarian_by_action.setdefault(action_id, []).append(consequence)
    utilitarian_profiles = {
        action_id: _utilitarian_profile(utilitarian_by_action.get(action_id, []))
        for action_id in action_signatures
    }
    rawls_by_action = {
        str(position.get("canonical_action_id", "")): position
        for position in state.rawlsian_positions
    }
    action_ids = [node.id for node in action_nodes]
    canonical_action_ids = [_action_identity(node)[0] for node in action_nodes]
    state.action_dimension_states = _derive_action_dimensions(
        state.rawlsian_positions,
        state.utilitarian_consequences,
        canonical_action_ids,
        state.grounded_action_effects,
    )
    state.framework_dimension_priorities = _derive_framework_dimension_priorities(
        state.rawlsian_positions,
        state.utilitarian_consequences,
        state.action_dimension_states,
    )
    dimensions_by_action: dict[str, set[str]] = {}
    for dimension_state in state.action_dimension_states:
        dimensions_by_action.setdefault(dimension_state.action_id, set()).add(
            dimension_state.dimension
        )
    substantive_dimensions = {
        item.dimension for item in state.action_dimension_states
        if item.dimension not in {"FEASIBILITY", "UNKNOWN", "OTHER_PRIMARY_GOOD", "OTHER"}
    }
    if (
        len(substantive_dimensions) >= 2
        and sum(bool(values & substantive_dimensions) for values in dimensions_by_action.values()) >= 2
    ):
        state.problem_shape_relations.append(ProblemShapeRelation(
            relation="MULTIDIMENSIONAL_TRADEOFF",
            source_action_ids=tuple(canonical_action_ids),
            statement=(
                "The actions differ across independently grounded dimensions; "
                "no consequence-count summary preserves the resulting tradeoff."
            ),
            confidence=0.9,
            support_node_ids=tuple(dict.fromkeys(
                support_id
                for item in state.action_dimension_states
                for support_id in item.support_node_ids
                if support_id
            )),
        ))
    for i, left_id in enumerate(action_ids):
        left = action_signatures.get(left_id) or {}
        left_profile = utilitarian_profiles.get(left_id, {})
        for right_id in action_ids[i + 1:]:
            right = action_signatures.get(right_id) or {}
            right_profile = utilitarian_profiles.get(right_id, {})
            pair_has_richer_shape = False
            left_canonical = str(left.get("action_id", ""))
            right_canonical = str(right.get("action_id", ""))
            if (
                len(
                    dimensions_by_action.get(left_canonical, set())
                    & dimensions_by_action.get(right_canonical, set())
                    & substantive_dimensions
                ) >= 2
            ):
                pair_has_richer_shape = True
            left_rawls = rawls_by_action.get(left_canonical, {})
            right_rawls = rawls_by_action.get(right_canonical, {})
            left_effect = str(
                left_rawls.get("proposed_effect", left_rawls.get("effect", ""))
            ).upper()
            right_effect = str(
                right_rawls.get("proposed_effect", right_rawls.get("effect", ""))
            ).upper()
            pair_text = " ".join(
                (graph.nodes[left_id].label, graph.nodes[right_id].label)
            ).casefold()
            distributive_case = bool(
                re.search(
                    r"\b(?:aggregate|property value|tax revenue|wealth|profit|economic growth)\b",
                    pair_text,
                )
                and re.search(
                    r"\b(?:least[- ]advantaged|low[- ]income|poorest|baseline|mixed[- ]income)\b",
                    pair_text,
                )
                and {left_effect, right_effect} >= {"IMPROVES", "WORSENS"}
            )
            if distributive_case:
                floor_id, aggregate_id = (
                    (left_canonical, right_canonical)
                    if left_effect == "IMPROVES"
                    else (right_canonical, left_canonical)
                )
                state.problem_shape_relations.append(ProblemShapeRelation(
                    relation="AGGREGATE_VS_DISTRIBUTIVE",
                    source_action_ids=(aggregate_id, floor_id),
                    statement=(
                        f"AGGREGATE-MAXIMUM vs DISTRIBUTIVE-FLOOR: {aggregate_id} pursues "
                        "the higher aggregate maximum while leaving a lower floor for the least "
                        f"advantaged; {floor_id} accepts a somewhat lower aggregate maximum to "
                        "raise that floor. The decision turns on whether aggregate maximization "
                        "can justify a worse position for those already least advantaged."
                    ),
                    confidence=0.9,
                    support_node_ids=tuple(value for value in dict.fromkeys([
                        *left.get("support_nodes", ()), *right.get("support_nodes", ()),
                        str(left_rawls.get("assessment_node_id", "")),
                        str(right_rawls.get("assessment_node_id", "")),
                    ]) if value),
                ))
                pair_has_richer_shape = True
            uninformative_labels = {
                "action", "choice", "decision", "intervention", "option",
                "outcome", "policy", "program", "proposal", "system",
            }
            shared_consequences = sorted(
                label
                for label in (
                    left.get("consequence_labels", set())
                    & right.get("consequence_labels", set())
                )
                if label not in uninformative_labels
            )
            # Shared target nouns such as "policy" or "residents" do not
            # establish either a shared mechanism or outcome equivalence.
            # Emit equivalence only for a concrete consequence common to both.
            if shared_consequences and not pair_has_richer_shape:
                state.problem_shape_relations.append(ProblemShapeRelation(
                    relation="OUTCOME_EQUIVALENCE",
                    source_action_ids=(
                        str(left.get("action_id", "")),
                        str(right.get("action_id", "")),
                    ),
                    statement=(
                        f"{left.get('action_id', left_id)} and {right.get('action_id', right_id)} "
                        f"share a concrete outcome: {', '.join(shared_consequences[:3])}"
                    ),
                    confidence=0.72,
                    support_node_ids=tuple(
                        dict.fromkeys(
                            [
                                *left.get("support_nodes", ()),
                                *right.get("support_nodes", ()),
                            ]
                        )
                    ),
                ))
            temporal_left = bool(
                left_profile.get("present_benefit")
                and left_profile.get("delayed_harm")
                and right_profile.get("present_harm")
                and right_profile.get("delayed_benefit")
            )
            temporal_right = bool(
                right_profile.get("present_benefit")
                and right_profile.get("delayed_harm")
                and left_profile.get("present_harm")
                and left_profile.get("delayed_benefit")
            )
            if temporal_left or temporal_right:
                if temporal_left:
                    primary_present = left_profile.get("present_benefit", [])
                    primary_delayed = left_profile.get("delayed_harm", [])
                    secondary_present = right_profile.get("present_harm", [])
                    secondary_delayed = right_profile.get("delayed_benefit", [])
                else:
                    primary_present = right_profile.get("present_benefit", [])
                    primary_delayed = right_profile.get("delayed_harm", [])
                    secondary_present = left_profile.get("present_harm", [])
                    secondary_delayed = left_profile.get("delayed_benefit", [])
                state.problem_shape_relations.append(ProblemShapeRelation(
                    relation="TEMPORAL_RISK_ASYMMETRY",
                    source_action_ids=(
                        str(left.get("action_id", "")),
                        str(right.get("action_id", "")),
                    ),
                    statement=(
                        "One option protects the present but leaves a larger delayed risk in place; "
                        "the other accepts immediate harm to reduce that future catastrophe."
                    ),
                    confidence=0.78,
                    support_node_ids=_support_ids(
                        primary_present, primary_delayed, secondary_present, secondary_delayed
                    ),
                ))
                certain_present = (
                    left_profile.get("certain", []) if temporal_left else right_profile.get("certain", [])
                )
                uncertain_delayed = (
                    right_profile.get("uncertain", []) if temporal_left else left_profile.get("uncertain", [])
                )
                if certain_present and uncertain_delayed:
                    state.problem_shape_relations.append(ProblemShapeRelation(
                        relation="EPISTEMIC_ASYMMETRY",
                        source_action_ids=(
                            str(left.get("action_id", "")),
                            str(right.get("action_id", "")),
                        ),
                        statement=(
                            "The immediate harm is comparatively certain, while the delayed catastrophe "
                            "remains probabilistic and less fully specified."
                        ),
                        confidence=0.74,
                        support_node_ids=_support_ids(certain_present, uncertain_delayed),
                    ))
                break
            left_adverse = int(left.get("adverse_count", 0))
            right_adverse = int(right.get("adverse_count", 0))
            left_beneficial = int(left.get("beneficial_count", 0))
            right_beneficial = int(right.get("beneficial_count", 0))
            if (
                not pair_has_richer_shape
                and (left_adverse != right_adverse or left_beneficial != right_beneficial)
            ):
                more_costly = left_id if (left_adverse, -left_beneficial) > (right_adverse, -right_beneficial) else right_id
                less_costly = right_id if more_costly == left_id else left_id
                more_costly_id = str(
                    (action_signatures.get(more_costly) or {}).get("action_id", more_costly)
                )
                less_costly_id = str(
                    (action_signatures.get(less_costly) or {}).get("action_id", less_costly)
                )
                state.problem_shape_relations.append(ProblemShapeRelation(
                    relation="ASYMMETRIC_COST",
                    source_action_ids=(more_costly_id, less_costly_id),
                    statement=(
                        f"{more_costly_id} imposes more adverse consequence structure than {less_costly_id}"
                    ),
                    confidence=0.62,
                    support_node_ids=tuple(
                        dict.fromkeys(
                            [
                                *left.get("support_nodes", ()),
                                *right.get("support_nodes", ()),
                            ]
                        )
                    ),
                ))
                break

    if state.decision_variables:
        variable = state.decision_variables[0]
        state.problem_shape_relations.append(ProblemShapeRelation(
            relation="DECISION_CRITICAL_UNKNOWN",
            source_action_ids=tuple(
                str(value) for value in [
                    variable.get("selected_action_id", ""),
                    variable.get("fallback_action_id", ""),
                ] if str(value).strip()
            ),
            statement=(
                f"Decision-critical variable remains unresolved: {variable.get('label', 'unknown')}"
            ),
            confidence=0.58,
            support_node_ids=(str(variable.get("decision_variable_node_id", "")),),
        ))

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
        state.problem_shape_relations.append(ProblemShapeRelation(
            relation="DECISION_BOUNDARY",
            source_action_ids=(source_id, target_id),
            statement=f"{source.label} switches to {target.label} if {condition.label}",
            confidence=0.9 if typed else 0.7,
            support_node_ids=(condition.id,),
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
