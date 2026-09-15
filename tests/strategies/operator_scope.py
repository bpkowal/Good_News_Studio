"""OPERATOR_SCOPE_CONSERVATION Hypothesis cases with declared oracles.

Oracle fields: ``expect_errors`` (bool) and optional ``error_substring``.
Synthetic valve/pump worlds only — no live-dilemma nouns as oracles.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from relent.operators import (
    Conditional,
    Fact,
    Modal,
    operator_to_dict,
)
from relent.scope import (
    build_and_not_conditional,
    build_unless_rule,
)


@dataclass(frozen=True, slots=True)
class OperatorScopeCase:
    mode: str
    rule: dict
    claim: dict | None
    positive: dict
    negated: dict
    consequent: dict
    expect_errors: bool
    error_substring: str
    issue_code: str = "OPERATOR_SCOPE_CONSERVATION"
    repair_stage: str = "operator_scope"
    allowed_ops: tuple[str, ...] = ("REJECT_SCOPE_MOVE",)
    forbidden_ops: tuple[str, ...] = ("FREE_REWRITE", "DROP_MODAL")


def _fact(entity: str, predicate: str, *, state: str = "", change: str = "") -> Fact:
    return Fact(entity=entity, predicate=predicate, state=state, change=change)


@st.composite
def operator_scope_cases(draw) -> OperatorScopeCase:
    valve = _fact("valve", "state", state="open")
    pump = _fact("backup_pump", "state", state="active")
    pressure = _fact("pressure", "change", change="increase")

    mode = draw(st.sampled_from((
        "and_not_ok",
        "modal_ok",
        "modal_dropped",
        "modal_on_antecedent",
        "unless_ok",
    )))
    if mode == "and_not_ok":
        rule = build_and_not_conditional(
            positive=valve, negated=pump, consequent=pressure,
        )
        return OperatorScopeCase(
            mode=mode,
            rule=operator_to_dict(rule),
            claim=None,
            positive=operator_to_dict(valve),
            negated=operator_to_dict(pump),
            consequent=operator_to_dict(pressure),
            expect_errors=False,
            error_substring="",
        )
    if mode == "modal_ok":
        rule = build_and_not_conditional(
            positive=valve,
            negated=pump,
            consequent=pressure,
            modality=Modal(strength="PROBABLE", body=pressure),
        )
        claim = Conditional(
            if_=rule.if_,
            then_=pressure,
            modality=Modal(strength="PROBABLE", body=pressure),
        )
        return OperatorScopeCase(
            mode=mode,
            rule=operator_to_dict(rule),
            claim=operator_to_dict(claim),
            positive=operator_to_dict(valve),
            negated=operator_to_dict(pump),
            consequent=operator_to_dict(pressure),
            expect_errors=False,
            error_substring="",
        )
    if mode == "modal_dropped":
        rule = build_and_not_conditional(
            positive=valve,
            negated=pump,
            consequent=pressure,
            modality=Modal(strength="PROBABLE", body=pressure),
        )
        claim = Conditional(if_=rule.if_, then_=pressure, modality=None)
        return OperatorScopeCase(
            mode=mode,
            rule=operator_to_dict(rule),
            claim=operator_to_dict(claim),
            positive=operator_to_dict(valve),
            negated=operator_to_dict(pump),
            consequent=operator_to_dict(pressure),
            expect_errors=True,
            error_substring="drop or escalate",
        )
    if mode == "modal_on_antecedent":
        rule = build_and_not_conditional(
            positive=valve,
            negated=pump,
            consequent=pressure,
            modality=Modal(strength="PROBABLE", body=pressure),
        )
        claim = Conditional(
            if_=Modal(strength="PROBABLE", body=rule.if_),
            then_=pressure,
            modality=None,
        )
        return OperatorScopeCase(
            mode=mode,
            rule=operator_to_dict(rule),
            claim=operator_to_dict(claim),
            positive=operator_to_dict(valve),
            negated=operator_to_dict(pump),
            consequent=operator_to_dict(pressure),
            expect_errors=True,
            error_substring="antecedent",
        )
    # unless_ok
    rule = build_unless_rule(
        antecedent=valve, consequent=pressure, unless=pump,
    )
    return OperatorScopeCase(
        mode=mode,
        rule=operator_to_dict(rule),
        claim=None,
        positive=operator_to_dict(valve),
        negated=operator_to_dict(pump),
        consequent=operator_to_dict(pressure),
        expect_errors=False,
        error_substring="",
        allowed_ops=("KEEP_EXCEPTION_SCOPE",),
    )
