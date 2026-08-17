from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


RELATIONS = {
    "CAUSES", "PREVENTS", "INCREASES", "DECREASES", "ENABLES", "DISABLES",
    "REQUIRES", "VIOLATES", "PRESERVES", "OVERRIDES", "REVERSES_IF",
    "PREFERS", "RESOLVES", "UNKNOWN",
}
CONTEXTS = {
    "BASE", "VISIBILITY_AUDIT", "CONSENSUS_AUDIT", "PLANNING_COUNTERFACTUAL",
    "REFORMULATION_PROBE", "REVERSAL_AUDIT", "SYNTHESIS_REVIEW",
}
EPISTEMIC_STATUSES = {
    "SCENARIO_GROUNDED", "STRONGLY_ENTAILED", "HYPOTHETICAL",
    "CONDITIONAL", "UNSUPPORTED", "NORMATIVE",
}


@dataclass(frozen=True, slots=True)
class SemanticProposition:
    actor: str
    action: str
    relation: str
    consequence: str
    condition: str = "NONE"
    condition_polarity: str = "POSITIVE"
    affected_party: str = "UNSPECIFIED"
    epistemic_status: str = "CONDITIONAL"
    alternative: str = "NONE"
    context: str = "BASE"
    provenance: tuple[str, ...] = ()
    source_text: str = ""

    def errors(self) -> list[str]:
        errors = []
        if len(self.action.split()) < 1:
            errors.append("missing action")
        if self.relation not in RELATIONS:
            errors.append("unknown semantic relation")
        if not self.consequence.strip():
            errors.append("missing consequence")
        if self.condition_polarity not in {"POSITIVE", "NEGATIVE"}:
            errors.append("invalid condition polarity")
        if self.epistemic_status not in EPISTEMIC_STATUSES:
            errors.append("invalid epistemic status")
        if self.context not in CONTEXTS:
            errors.append("invalid proposition context")
        if self.context != "BASE" and not self.provenance:
            errors.append("non-base proposition lacks provenance")
        return errors


@dataclass(slots=True)
class SemanticInvariantRecord:
    boundary: str
    proposition: SemanticProposition
    transformed_text: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    fallback: str = "PRESERVE_SOURCE"

    def __post_init__(self) -> None:
        self.boundary = self.boundary.strip().upper()[:48]
        self.transformed_text = " ".join(self.transformed_text.split())[:600]
        self.errors = [" ".join(error.split())[:180] for error in self.errors][:8]


def validate_transformation(
    boundary: str,
    proposition: SemanticProposition,
    transformed_text: str,
    *,
    required_fragments: Sequence[str] = (),
) -> SemanticInvariantRecord:
    """Validate structure and exact high-risk anchors; preserve source on failure."""
    errors = proposition.errors()
    normalized = " ".join(transformed_text.split()).casefold()
    if not normalized:
        errors.append("empty transformation")
    for fragment in required_fragments:
        anchor = " ".join(str(fragment).split()).casefold()
        if anchor and anchor != "none" and anchor not in normalized:
            errors.append(f"transformation dropped required anchor: {fragment}")
    return SemanticInvariantRecord(
        boundary, proposition, transformed_text, not errors, errors
    )


def compile_preference_rule(
    selected_action: str,
    decision_rule: str,
    qualifier: str,
    reopen_conditions: Sequence[str],
    provenance: Sequence[str],
    *,
    governing_constraint: str = "",
    preserved_objections: Sequence[str] = (),
) -> tuple[str, SemanticInvariantRecord]:
    """Compile a governing rule without flattening committed plural dissent."""
    source_rule = " ".join(decision_rule.split())
    reopen = ", ".join(reopen_conditions) or "material facts change"
    constraint = " ".join(str(governing_constraint).split()).upper()
    rule_label = f"Rule [{constraint}]" if constraint else "Rule"
    objections = [
        " ".join(str(objection).split()).strip(" .")
        for objection in preserved_objections
        if " ".join(str(objection).split()).strip(" .")
    ]
    objections_text = (
        "Preserved objections: " + "; ".join(objections[:3]) + ". "
        if objections else ""
    )
    text = (f"{rule_label}: {source_rule}. " if source_rule else "") + objections_text + (
        f"{qualifier} {selected_action}; reopen if {reopen}."
    )
    proposition = SemanticProposition(
        actor="workspace",
        action=selected_action,
        relation="PREFERS",
        consequence=source_rule or "current policy ranking",
        condition=reopen,
        condition_polarity="NEGATIVE",
        epistemic_status="CONDITIONAL" if qualifier != "prefer" else "NORMATIVE",
        context="BASE",
        provenance=tuple(provenance),
        source_text=source_rule,
    )
    required = [selected_action, source_rule] if source_rule else [selected_action]
    if constraint:
        required.append(constraint)
    required.extend(objections[:3])
    return text, validate_transformation(
        "COMPRESSED_RULE", proposition, text, required_fragments=required
    )
