"""Typed, graph-backed decision boundaries for publishable reversal rules."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .models import CandidateChunk
from .reversal_semantics import reversal_transition_graph
from .semantic_graph import (
    GraphValidation, SemanticEdge, SemanticGraph, SemanticNode, validate_graph,
)


_COMPARATORS = (
    (re.compile(r"(?:≤|<=|\b(?:falls?|drops?|decreases?|is)\s+(?:to\s+or\s+)?below\b|\bbelow\b)", re.I), "LE"),
    (re.compile(r"(?:≥|>=|\b(?:rises?|increases?|is)\s+(?:to\s+or\s+)?above\b|\bexceeds?\b|\babove\b)", re.I), "GE"),
    (re.compile(r"<"), "LT"),
    (re.compile(r">"), "GT"),
)
_QUANTITY = re.compile(
    r"(?P<currency>[$€£])?\s*(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<scale>million|billion|thousand|m|bn|k)?\s*"
    r"(?P<unit>%|percent|seconds?|minutes?|hours?|days?|weeks?|months?|years?)?",
    re.I,
)
_HARM_WORDS = re.compile(
    r"\b(?:risk|harm|deaths?|mortality|casualties|failure|loss|coercion|violation|cost)\b",
    re.I,
)
_BENEFIT_WORDS = re.compile(
    r"\b(?:benefit|success|survival|lives? saved|containment|effectiveness|"
    r"recovery|coverage|access|welfare|yield|reliability)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class DecisionBoundary:
    from_action: str
    to_action: str
    affected_action: str
    metric: str
    comparator: str
    threshold: float
    unit: str
    metric_valence: str
    source_specialist: str
    source_text: str

    @property
    def predicate(self) -> str:
        symbols = {"LT": "<", "LE": "≤", "GT": ">", "GE": "≥"}
        value = f"{self.threshold:g}"
        suffix = "%" if self.unit == "PERCENT" else f" {self.unit.lower()}"
        return f"{self.metric} {symbols[self.comparator]} {value}{suffix}"

    def transition(self) -> dict[str, str]:
        lower = self.comparator in {"LT", "LE"}
        improves_affected = (
            (self.metric_valence == "ADVERSE" and lower)
            or (self.metric_valence == "BENEFICIAL" and not lower)
        )
        effect = "MORE_ATTRACTIVE" if improves_affected else "LESS_ATTRACTIVE"
        return {
            "from_action": self.from_action,
            "to_action": self.to_action,
            "condition": self.predicate,
            "affected_action": self.affected_action,
            "effect_on_affected_action": effect,
            "justification": (
                f"numeric boundary supplied by {self.source_specialist}'s stated switch threshold"
            ),
        }

    def graph(self, *, include_switch: bool = True, index: int = 0) -> SemanticGraph:
        graph = reversal_transition_graph(self.transition(), index=index)
        condition_id = next(
            node.id for node in graph.nodes.values() if node.kind == "CONDITION"
        )
        condition_node = graph.nodes[condition_id]
        graph.add_node(SemanticNode(
            condition_node.id,
            condition_node.kind,
            condition_node.label,
            condition_node.provenance,
            {
                **condition_node.attributes,
                "predicate_type": "SCALAR_THRESHOLD",
                "affected_action_id": self.affected_action,
                "metric": self.metric,
                "metric_valence": self.metric_valence,
                "comparator": self.comparator,
                "threshold": self.threshold,
                "unit": self.unit,
                "source_specialists": [self.source_specialist],
            },
        ))
        metric_id = f"metric:{self.affected_action}:{self.metric}"
        threshold_id = f"threshold:{self.comparator}:{self.threshold:g}:{self.unit}"
        graph.add_node(SemanticNode(
            metric_id, "METRIC", self.metric, (self.source_specialist, self.source_text),
            {
                "affected_action_id": self.affected_action,
                "metric_valence": self.metric_valence,
            },
        ))
        graph.add_node(SemanticNode(
            threshold_id, "THRESHOLD",
            f"{self.comparator} {self.threshold:g} {self.unit}",
            (self.source_specialist, self.source_text),
            {
                "comparator": self.comparator,
                "threshold": self.threshold,
                "unit": self.unit,
            },
        ))
        graph.add_edge(SemanticEdge(
            condition_id, "HAS_METRIC", metric_id, condition=condition_id,
            provenance=(self.source_specialist,),
        ))
        graph.add_edge(SemanticEdge(
            metric_id, "COMPARES_TO", threshold_id, condition=condition_id,
            justification=self.comparator, provenance=(self.source_specialist,),
        ))
        if not include_switch:
            graph.edges = [edge for edge in graph.edges if edge.relation != "SWITCHES_TO"]
        return graph

    def validate(self) -> GraphValidation:
        result = validate_graph(self.graph())
        errors = list(result.errors)
        if self.comparator not in {"LT", "LE", "GT", "GE"}:
            errors.append("decision boundary has no ordered comparator")
        if self.unit == "PERCENT" and not 0 <= self.threshold <= 100:
            errors.append("percentage boundary lies outside 0..100")
        if self.metric_valence not in {"ADVERSE", "BENEFICIAL"}:
            errors.append("metric has no unambiguous ethical valence")
        if self.affected_action not in {self.from_action, self.to_action}:
            errors.append("boundary affects neither the source nor destination action")
        if not self.metric.strip():
            errors.append("decision boundary has no metric")
        graph = self.graph()
        if not any(edge.relation == "HAS_METRIC" for edge in graph.edges):
            errors.append("condition is not linked to a metric")
        if not any(edge.relation == "COMPARES_TO" for edge in graph.edges):
            errors.append("metric is not linked to a threshold")
        return GraphValidation(not errors, list(dict.fromkeys(errors)), result.warnings)


@dataclass(frozen=True, slots=True)
class CompoundDecisionBoundary:
    operator: str
    clauses: tuple[DecisionBoundary, ...]

    @property
    def from_action(self) -> str:
        return self.clauses[0].from_action

    @property
    def to_action(self) -> str:
        return self.clauses[0].to_action

    @property
    def predicate(self) -> str:
        return f" {self.operator} ".join(f"({clause.predicate})" for clause in self.clauses)

    def transition(self) -> dict[str, str]:
        return {
            "from_action": self.from_action,
            "to_action": self.to_action,
            "condition": self.predicate,
            "affected_action": "MULTIPLE",
            "effect_on_affected_action": "UNKNOWN",
            "justification": f"all {self.operator} boundary clauses support the same switch",
        }

    def graph(self) -> SemanticGraph:
        graph = SemanticGraph()
        condition_id = f"logical:{self.operator}:{self.predicate}"
        graph.add_node(SemanticNode(
            condition_id,
            "LOGICAL",
            self.operator,
            attributes={
                "predicate_type": "COMPOUND",
                "operator": self.operator,
                "source_specialists": list(dict.fromkeys(
                    clause.source_specialist for clause in self.clauses
                ))
            },
        ))
        for index, clause in enumerate(self.clauses):
            atom = clause.graph(include_switch=False, index=index)
            for node in atom.nodes.values():
                graph.add_node(node)
            for edge in atom.edges:
                graph.add_edge(edge)
            atom_condition = next(
                node.id for node in atom.nodes.values() if node.kind == "CONDITION"
            )
            graph.add_edge(SemanticEdge(
                condition_id, "HAS_OPERAND", atom_condition, condition=condition_id,
                provenance=(clause.source_specialist, clause.source_text),
            ))
        source_id = f"action:{self.from_action}"
        target_id = f"action:{self.to_action}"
        graph.add_edge(SemanticEdge(
            source_id, "SWITCHES_TO", target_id, condition=condition_id,
            justification=f"compound {self.operator} boundary",
        ))
        return graph

    def validate(self) -> GraphValidation:
        errors: list[str] = []
        if self.operator not in {"AND", "OR"}:
            errors.append("compound boundary has an unknown logical operator")
        if len(self.clauses) < 2:
            errors.append("compound boundary needs at least two clauses")
        if len({clause.from_action for clause in self.clauses}) != 1:
            errors.append("compound clauses do not share a source action")
        if len({clause.to_action for clause in self.clauses}) != 1:
            errors.append("compound clauses do not share a destination action")
        for clause in self.clauses:
            errors.extend(clause.validate().errors)
        graph_result = validate_graph(self.graph())
        errors.extend(graph_result.errors)
        return GraphValidation(not errors, list(dict.fromkeys(errors)), graph_result.warnings)


def _affected_action(text: str, actions: Sequence[str]) -> str:
    explicit = re.search(r"\bA(\d+)\b", text, re.I)
    if explicit:
        index = int(explicit.group(1))
        if index < len(actions):
            return actions[index]
    words = set(re.findall(r"[a-z0-9]+", text.casefold())) - {
        "if", "risk", "harm", "below", "above", "drops", "rises", "would",
        "flip", "choice", "percent",
    }
    scores = []
    for action in actions:
        action_words = set(re.findall(r"[a-z0-9]+", action.casefold()))
        scores.append((len(words & action_words), action))
    best_score, best = max(scores, default=(0, ""))
    return best if best_score else ""


def _metric_valence(metric: str) -> str:
    adverse = bool(_HARM_WORDS.search(metric))
    beneficial = bool(_BENEFIT_WORDS.search(metric))
    if adverse == beneficial:
        return "UNKNOWN"
    return "ADVERSE" if adverse else "BENEFICIAL"


def _quantity(match: re.Match[str]) -> tuple[float, str]:
    value = float(match.group("value").replace(",", ""))
    scale = (match.group("scale") or "").casefold()
    multiplier = {
        "thousand": 1_000, "k": 1_000, "million": 1_000_000,
        "m": 1_000_000, "billion": 1_000_000_000, "bn": 1_000_000_000,
    }.get(scale, 1)
    raw_unit = (match.group("unit") or "").casefold()
    if raw_unit in {"%", "percent"}:
        return value, "PERCENT"
    if match.group("currency"):
        return value * multiplier, {"$": "USD", "€": "EUR", "£": "GBP"}[match.group("currency")]
    if raw_unit:
        singular = raw_unit[:-1] if raw_unit.endswith("s") else raw_unit
        return value, singular.upper()
    return value * multiplier, "COUNT"


def parse_numeric_boundary(
    text: str, *, from_action: str, to_action: str,
    actions: Sequence[str], source_specialist: str,
) -> DecisionBoundary | None:
    """Parse only explicit scalar comparisons; reject qualitative axis restatements."""
    normalized = " ".join(text.split()).strip(" .")
    comparison = next(
        ((match, code) for pattern, code in _COMPARATORS if (match := pattern.search(normalized))),
        None,
    )
    match, comparator = comparison if comparison is not None else (None, "")
    # Search after the comparator so digits in action IDs (A0/A1) cannot become
    # the threshold value.
    number = _QUANTITY.search(normalized, match.end()) if match is not None else None
    if comparison is None or number is None:
        return None
    # Do not silently reduce conjunctions or disjunctions to their first scalar.
    remainder = normalized[number.end():]
    if re.search(r"\b(?:and|or)\b", remainder, re.I) and _QUANTITY.search(remainder):
        return None
    threshold, unit = _quantity(number)
    metric_end = match.start()
    metric = normalized[:metric_end].strip(" ,;:-")
    metric = re.sub(r"^if\s+", "", metric, flags=re.I)
    if len(metric.split()) < 2:
        return None
    affected = _affected_action(metric, actions)
    if not affected:
        return None
    valence = _metric_valence(metric)
    if valence == "UNKNOWN":
        return None
    boundary = DecisionBoundary(
        from_action, to_action, affected, metric, comparator, threshold, unit, valence,
        source_specialist, normalized,
    )
    return boundary if boundary.validate().valid else None


BoundaryExpression = DecisionBoundary | CompoundDecisionBoundary


def _normalized_unit(unit: str) -> str:
    value = str(unit).strip().upper()
    aliases = {
        "DEATH": "DEATHS", "FATALITY": "DEATHS", "FATALITIES": "DEATHS",
        "CASUALTY": "CASUALTIES", "PERSON": "PEOPLE", "PERSONS": "PEOPLE",
        "%": "PERCENT",
    }
    return aliases.get(value, value)


def _estimate_for_clause(
    clause: DecisionBoundary, estimates: dict[str, dict[str, object]],
) -> float | None:
    """Return a comparable current value, without guessing across dimensions."""
    estimate = estimates.get(clause.affected_action, {})
    if (
        not isinstance(estimate, dict)
        or estimate.get("grounded") is not True
        or estimate.get("validation_status") != "ARITHMETIC_VERIFIED"
    ):
        return None
    try:
        value = float(estimate["value"])
    except (KeyError, TypeError, ValueError):
        return None
    estimate_unit = _normalized_unit(str(estimate.get("unit", "")))
    boundary_unit = _normalized_unit(clause.unit)
    metric = clause.metric.casefold()
    death_metric = any(word in metric for word in ("death", "fatal", "mortal", "casual"))
    count_units = {"COUNT", "DEATHS", "CASUALTIES", "PEOPLE"}
    units_match = estimate_unit == boundary_unit or (
        death_metric and estimate_unit in count_units and boundary_unit in count_units
    )
    direction = str(estimate.get("direction", "")).strip().upper()
    valence_matches = (
        (clause.metric_valence == "ADVERSE" and direction == "HARM")
        or (clause.metric_valence == "BENEFICIAL" and direction == "BENEFIT")
    )
    return value if units_match and valence_matches else None


def boundary_satisfied_by_estimates(
    boundary: BoundaryExpression,
    estimates: dict[str, dict[str, object]] | None,
) -> bool | None:
    """Evaluate a boundary against typed current facts when facts are commensurable."""
    if not estimates:
        return None

    def atom(clause: DecisionBoundary) -> bool | None:
        value = _estimate_for_clause(clause, estimates)
        if value is None:
            return None
        return {
            "LT": value < clause.threshold,
            "LE": value <= clause.threshold,
            "GT": value > clause.threshold,
            "GE": value >= clause.threshold,
        }[clause.comparator]

    if isinstance(boundary, DecisionBoundary):
        return atom(boundary)
    values = [atom(clause) for clause in boundary.clauses]
    if boundary.operator == "AND":
        if False in values:
            return False
        return True if all(value is True for value in values) else None
    if True in values:
        return True
    return False if all(value is False for value in values) else None


def parse_boundary_expression(
    text: str, *, from_action: str, to_action: str,
    actions: Sequence[str], source_specialist: str,
) -> BoundaryExpression | None:
    """Parse one scalar or a homogeneous unparenthesized AND/OR expression."""
    normalized = " ".join(text.split()).strip(" .")
    has_and = bool(re.search(r"\band\b", normalized, re.I))
    has_or = bool(re.search(r"\bor\b", normalized, re.I))
    if has_and and has_or:
        # Precedence would be invented without a structured parenthesized parser.
        return None
    operator = "AND" if has_and else ("OR" if has_or else "")
    if not operator:
        return parse_numeric_boundary(
            normalized, from_action=from_action, to_action=to_action,
            actions=actions, source_specialist=source_specialist,
        )
    parts = [
        part.strip(" ()") for part in re.split(
            rf"\b{operator}\b", normalized, flags=re.I
        ) if part.strip(" ()")
    ]
    if len(parts) < 2:
        return None
    clauses = []
    for part in parts:
        clause = parse_numeric_boundary(
            part, from_action=from_action, to_action=to_action,
            actions=actions, source_specialist=source_specialist,
        )
        if clause is None:
            return None
        clauses.append(clause)
    compound = CompoundDecisionBoundary(operator, tuple(clauses))
    return compound if compound.validate().valid else None


def select_collective_reversal_boundary(
    candidates: Sequence[CandidateChunk], selected_action: str,
    actions: Sequence[str],
) -> BoundaryExpression | None:
    """Choose a numeric threshold from a valid supporter of the current judgment."""
    alternatives = [action for action in actions if action != selected_action]
    if not alternatives:
        return None
    options: list[tuple[float, BoundaryExpression]] = []
    for candidate in candidates:
        if not candidate.schema_valid or candidate.recommended_action != selected_action:
            continue
        text = candidate.factual_reversal_threshold
        if not text or text.casefold() == "none":
            continue
        destination = max(
            alternatives, key=lambda action: candidate.action_scores.get(action, 0.0)
        )
        boundary = parse_boundary_expression(
            text, from_action=selected_action, to_action=destination,
            actions=actions, source_specialist=candidate.specialist,
        )
        # A reversal must describe a state change, not a predicate that already
        # holds while the delegate continues to recommend its source action.
        if (
            boundary is not None
            and boundary_satisfied_by_estimates(
                boundary, candidate.expected_value_estimates
            ) is not True
        ):
            options.append((candidate.epistemic_confidence, boundary))
    return max(options, key=lambda item: item[0])[1] if options else None
