"""Operator-scope conservation rules (pure metamorphic checks).

Hosts license a Conditional / ExceptionRule. RelEnt answers whether a rewritten
claim preserves antecedent scope and modal attachment. No English parsing.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .operators import (
    AllOf,
    Assignment,
    Conditional,
    ExceptionRule,
    Fact,
    Modal,
    Not,
    Operator,
    AnyOf,
    consequent_licensed,
    evaluate,
    modal_strength_of,
    operator_from_dict,
    operator_to_dict,
)


_MODAL_RANK = {
    "UNKNOWN": 0,
    "POSSIBLE": 1,
    "PROBABLE": 2,
    "CERTAIN": 3,
}


def _facts_equal(left: Fact, right: Fact) -> bool:
    return left.key() == right.key()


def _structural_equal(left: Operator, right: Operator) -> bool:
    return operator_to_dict(left) == operator_to_dict(right)


def assignment_variants_for_and_not(
    positive: Fact,
    negated: Fact,
) -> dict[str, Assignment]:
    """Canonical worlds for ``AllOf(positive, Not(negated)) → C`` metamorphic tests."""
    p, n = positive.key(), negated.key()
    return {
        "A_and_not_B": {p: True, n: False},
        "A_and_B": {p: True, n: True},
        "not_A_and_not_B": {p: False, n: False},
        "not_A_and_B": {p: False, n: True},
    }


def and_not_inheritance_errors(
    rule: Conditional | ExceptionRule | Mapping,
    *,
    positive: Fact | Mapping,
    negated: Fact | Mapping,
) -> list[str]:
    """A ∧ ¬B → C must not license C under A ∧ B or ¬A.

    Returns operator-scope errors; empty when the licensed worlds are correct.
    """
    node = operator_from_dict(rule)  # type: ignore[arg-type]
    pos = operator_from_dict(positive)  # type: ignore[arg-type]
    neg = operator_from_dict(negated)  # type: ignore[arg-type]
    if not isinstance(pos, Fact) or not isinstance(neg, Fact):
        return ["OPERATOR_SCOPE: positive/negated must be Fact leaves"]
    if not isinstance(node, (Conditional, ExceptionRule)):
        return ["OPERATOR_SCOPE: rule must be Conditional or ExceptionRule"]
    worlds = assignment_variants_for_and_not(pos, neg)
    errors: list[str] = []
    if not consequent_licensed(node, worlds["A_and_not_B"]):
        errors.append(
            "OPERATOR_SCOPE: A∧¬B must license the consequent "
            f"({pos.predicate!r} and not {neg.predicate!r})"
        )
    for label in ("A_and_B", "not_A_and_not_B", "not_A_and_B"):
        if consequent_licensed(node, worlds[label]):
            errors.append(
                f"OPERATOR_SCOPE: world {label} must not inherit the consequent "
                f"of A∧¬B → C"
            )
    return errors


def modal_scope_errors(
    source: Conditional | Mapping,
    claim: Conditional | Mapping,
) -> list[str]:
    """Forbid moving or dropping modality across the conditional boundary.

    If source is ``A → probably C``:
    - ``probably A → C`` (modal on antecedent) is illegal
    - bare ``A → C`` (modal dropped / escalated to CERTAIN) is illegal
    """
    src = operator_from_dict(source)  # type: ignore[arg-type]
    clm = operator_from_dict(claim)  # type: ignore[arg-type]
    if not isinstance(src, Conditional) or not isinstance(clm, Conditional):
        return ["OPERATOR_SCOPE: modal_scope_errors requires Conditional nodes"]
    errors: list[str] = []
    source_strength = modal_strength_of(src)
    claim_strength = modal_strength_of(clm)
    if source_strength in {"PROBABLE", "POSSIBLE"}:
        if claim_strength is None or claim_strength == "CERTAIN":
            errors.append(
                "OPERATOR_SCOPE: claim may not drop or escalate source modality "
                f"{source_strength} to bare/CERTAIN consequent"
            )
        # Modal must not migrate onto the antecedent alone.
        if isinstance(clm.if_, Modal) and not isinstance(src.if_, Modal):
            errors.append(
                "OPERATOR_SCOPE: modality may not move from consequent onto "
                "the antecedent (probably A → C)"
            )
        if (
            claim_strength is not None
            and _MODAL_RANK.get(claim_strength, 0) > _MODAL_RANK.get(source_strength, 0)
        ):
            errors.append(
                "OPERATOR_SCOPE: claim modality may not strengthen "
                f"{source_strength} → {claim_strength}"
            )
    if not _structural_equal(src.if_, clm.if_) and source_strength:
        # Allow only modality / then_ edits when checking modal scope; flag
        # antecedent rewrite that embeds a Modal.
        if isinstance(clm.if_, Modal):
            errors.append(
                "OPERATOR_SCOPE: antecedent rewrite attached a Modal without "
                "source license"
            )
    return list(dict.fromkeys(errors))


def exception_scope_errors(
    rule: ExceptionRule | Mapping,
    *,
    base_assignment: Assignment,
    exception_assignment: Assignment,
) -> list[str]:
    """A → C unless B: base world licenses C; exception world must not."""
    node = operator_from_dict(rule)  # type: ignore[arg-type]
    if not isinstance(node, ExceptionRule):
        return ["OPERATOR_SCOPE: exception_scope_errors requires ExceptionRule"]
    errors: list[str] = []
    if not consequent_licensed(node, base_assignment):
        errors.append("OPERATOR_SCOPE: base world (unless false) must license C")
    if consequent_licensed(node, exception_assignment):
        errors.append(
            "OPERATOR_SCOPE: exception world (unless true) must not license C"
        )
    if not evaluate(node.unless, exception_assignment):
        errors.append("OPERATOR_SCOPE: exception_assignment must satisfy unless")
    return errors


def build_and_not_conditional(
    *,
    positive: Fact,
    negated: Fact,
    consequent: Fact,
    modality: Modal | None = None,
) -> Conditional:
    """Sugar for the valve/pump metamorphic shape."""
    return Conditional(
        if_=AllOf(children=(positive, Not(child=negated))),
        then_=consequent,
        modality=modality,
    )


def build_unless_rule(
    *,
    antecedent: Fact,
    consequent: Fact,
    unless: Fact,
    modality: Modal | None = None,
) -> ExceptionRule:
    return ExceptionRule(
        rule=Conditional(if_=antecedent, then_=consequent, modality=modality),
        unless=unless,
    )


def operator_scope_conservation_errors(
    *,
    rule: Conditional | ExceptionRule | Mapping,
    claim: Conditional | ExceptionRule | Mapping | None = None,
    positive: Fact | Mapping | None = None,
    negated: Fact | Mapping | None = None,
    base_assignment: Assignment | None = None,
    exception_assignment: Assignment | None = None,
) -> list[str]:
    """Umbrella: run the applicable scope checks for a rule (± claim rewrite)."""
    errors: list[str] = []
    node = operator_from_dict(rule)  # type: ignore[arg-type]
    if positive is not None and negated is not None and isinstance(
        node, (Conditional, ExceptionRule),
    ):
        errors.extend(and_not_inheritance_errors(
            node, positive=positive, negated=negated,
        ))
    if claim is not None and isinstance(node, Conditional):
        errors.extend(modal_scope_errors(node, claim))
    if (
        isinstance(node, ExceptionRule)
        and base_assignment is not None
        and exception_assignment is not None
    ):
        errors.extend(exception_scope_errors(
            node,
            base_assignment=base_assignment,
            exception_assignment=exception_assignment,
        ))
    return errors
