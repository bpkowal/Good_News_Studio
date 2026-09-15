"""Compositional operator-scope AST for RelEnt.

Calculator island only: serializable condition structure that hosts may compile
into their own gates (Parliament: ``condition_ids`` / modality). Not a world
store, not an English parser, not Prolog.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, Mapping, Sequence, Union

ModalStrength = Literal["CERTAIN", "PROBABLE", "POSSIBLE", "UNKNOWN"]
OperatorKind = Literal[
    "Fact", "Not", "AllOf", "AnyOf", "Modal", "Conditional", "ExceptionRule",
]

_OPERATOR_KINDS = frozenset({
    "Fact", "Not", "AllOf", "AnyOf", "Modal", "Conditional", "ExceptionRule",
})


@dataclass(frozen=True, slots=True)
class Fact:
    """Atomic proposition. Hosts bind ``entity`` / ``predicate`` to their IDs."""

    predicate: str
    entity: str = ""
    state: str = ""
    change: str = ""
    quantity: str = ""
    kind: OperatorKind = field(default="Fact", init=False)

    def key(self) -> str:
        parts = [
            str(self.entity or "").strip(),
            str(self.predicate or "").strip(),
            str(self.state or "").strip(),
            str(self.change or "").strip(),
            str(self.quantity or "").strip(),
        ]
        return "|".join(part for part in parts if part).casefold()


@dataclass(frozen=True, slots=True)
class Not:
    child: "Operator"
    kind: OperatorKind = field(default="Not", init=False)


@dataclass(frozen=True, slots=True)
class AllOf:
    children: tuple["Operator", ...]
    kind: OperatorKind = field(default="AllOf", init=False)


@dataclass(frozen=True, slots=True)
class AnyOf:
    children: tuple["Operator", ...]
    kind: OperatorKind = field(default="AnyOf", init=False)


@dataclass(frozen=True, slots=True)
class Modal:
    """Likelihood / certainty scoped over a body (not over the antecedent)."""

    strength: ModalStrength
    body: "Operator"
    kind: OperatorKind = field(default="Modal", init=False)


@dataclass(frozen=True, slots=True)
class Conditional:
    """If antecedent then consequent, optionally with a modal on the consequent."""

    if_: "Operator"
    then_: "Operator"
    modality: Modal | None = None
    kind: OperatorKind = field(default="Conditional", init=False)


@dataclass(frozen=True, slots=True)
class ExceptionRule:
    """``rule`` holds unless ``unless`` is satisfied (A → C unless B)."""

    rule: Conditional
    unless: "Operator"
    kind: OperatorKind = field(default="ExceptionRule", init=False)


Operator = Union[Fact, Not, AllOf, AnyOf, Modal, Conditional, ExceptionRule]

# Assignment: fact key → truth. Missing keys are False.
Assignment = Mapping[str, bool]


def _as_operator(value: Any) -> Operator:
    if isinstance(value, (Fact, Not, AllOf, AnyOf, Modal, Conditional, ExceptionRule)):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"operator payload must be a mapping, got {type(value)!r}")
    kind = str(value.get("kind") or "").strip()
    if kind not in _OPERATOR_KINDS:
        raise ValueError(f"unknown operator kind: {kind!r}")
    if kind == "Fact":
        return Fact(
            predicate=str(value.get("predicate") or ""),
            entity=str(value.get("entity") or ""),
            state=str(value.get("state") or ""),
            change=str(value.get("change") or ""),
            quantity=str(value.get("quantity") or ""),
        )
    if kind == "Not":
        return Not(child=_as_operator(value["child"]))
    if kind == "AllOf":
        return AllOf(children=tuple(_as_operator(item) for item in value.get("children") or ()))
    if kind == "AnyOf":
        return AnyOf(children=tuple(_as_operator(item) for item in value.get("children") or ()))
    if kind == "Modal":
        strength = str(value.get("strength") or "UNKNOWN").upper()
        if strength not in {"CERTAIN", "PROBABLE", "POSSIBLE", "UNKNOWN"}:
            raise ValueError(f"unknown modal strength: {strength!r}")
        return Modal(strength=strength, body=_as_operator(value["body"]))  # type: ignore[arg-type]
    if kind == "Conditional":
        modality = value.get("modality")
        return Conditional(
            if_=_as_operator(value["if_"]),
            then_=_as_operator(value["then_"]),
            modality=_as_operator(modality) if modality else None,  # type: ignore[arg-type]
        )
    return ExceptionRule(
        rule=_as_operator(value["rule"]),  # type: ignore[arg-type]
        unless=_as_operator(value["unless"]),
    )


def operator_to_dict(node: Operator) -> dict[str, Any]:
    """JSON-friendly dict with explicit ``kind`` tags."""
    if isinstance(node, Fact):
        return {
            "kind": "Fact",
            "predicate": node.predicate,
            "entity": node.entity,
            "state": node.state,
            "change": node.change,
            "quantity": node.quantity,
        }
    if isinstance(node, Not):
        return {"kind": "Not", "child": operator_to_dict(node.child)}
    if isinstance(node, AllOf):
        return {
            "kind": "AllOf",
            "children": [operator_to_dict(child) for child in node.children],
        }
    if isinstance(node, AnyOf):
        return {
            "kind": "AnyOf",
            "children": [operator_to_dict(child) for child in node.children],
        }
    if isinstance(node, Modal):
        return {
            "kind": "Modal",
            "strength": node.strength,
            "body": operator_to_dict(node.body),
        }
    if isinstance(node, Conditional):
        payload: dict[str, Any] = {
            "kind": "Conditional",
            "if_": operator_to_dict(node.if_),
            "then_": operator_to_dict(node.then_),
        }
        if node.modality is not None:
            payload["modality"] = operator_to_dict(node.modality)
        return payload
    if isinstance(node, ExceptionRule):
        return {
            "kind": "ExceptionRule",
            "rule": operator_to_dict(node.rule),
            "unless": operator_to_dict(node.unless),
        }
    raise TypeError(f"unsupported operator: {type(node)!r}")


def operator_from_dict(payload: Mapping[str, Any] | Operator) -> Operator:
    """Inverse of ``operator_to_dict``."""
    if is_dataclass(payload) and not isinstance(payload, type):
        return payload  # type: ignore[return-value]
    return _as_operator(dict(payload))


def evaluate(node: Operator, assignment: Assignment) -> bool:
    """Truth of an operator under a boolean assignment of Fact keys."""
    if isinstance(node, Fact):
        return bool(assignment.get(node.key(), False))
    if isinstance(node, Not):
        return not evaluate(node.child, assignment)
    if isinstance(node, AllOf):
        return all(evaluate(child, assignment) for child in node.children)
    if isinstance(node, AnyOf):
        return any(evaluate(child, assignment) for child in node.children)
    if isinstance(node, Modal):
        # Modal does not change truth of the body for antecedent evaluation;
        # hosts treat strength as a separate channel.
        return evaluate(node.body, assignment)
    if isinstance(node, Conditional):
        if not evaluate(node.if_, assignment):
            return True  # material implication: false antecedent ⇒ true
        body = node.modality.body if node.modality is not None else node.then_
        return evaluate(body, assignment)
    if isinstance(node, ExceptionRule):
        if evaluate(node.unless, assignment):
            return True  # exception fires ⇒ base rule not applied / vacuous
        return evaluate(node.rule, assignment)
    raise TypeError(f"unsupported operator: {type(node)!r}")


def consequent_licensed(node: Conditional | ExceptionRule, assignment: Assignment) -> bool:
    """True when the assignment licenses asserting the rule's consequent.

    Stricter than material ``evaluate``: false antecedents do not license C.
    """
    if isinstance(node, ExceptionRule):
        if evaluate(node.unless, assignment):
            return False
        return consequent_licensed(node.rule, assignment)
    if not evaluate(node.if_, assignment):
        return False
    return True


def collect_facts(node: Operator) -> tuple[Fact, ...]:
    """Leaf Fact nodes in preorder."""
    if isinstance(node, Fact):
        return (node,)
    if isinstance(node, Not):
        return collect_facts(node.child)
    if isinstance(node, (AllOf, AnyOf)):
        found: list[Fact] = []
        for child in node.children:
            found.extend(collect_facts(child))
        return tuple(found)
    if isinstance(node, Modal):
        return collect_facts(node.body)
    if isinstance(node, Conditional):
        ordered = (
            *collect_facts(node.if_),
            *collect_facts(node.then_),
            *(collect_facts(node.modality) if node.modality else ()),
        )
        return tuple(dict.fromkeys(ordered))
    if isinstance(node, ExceptionRule):
        ordered = (*collect_facts(node.rule), *collect_facts(node.unless))
        return tuple(dict.fromkeys(ordered))
    return ()


def modal_strength_of(node: Operator) -> ModalStrength | None:
    """Outermost modal strength on a rule or consequent, if any."""
    if isinstance(node, Modal):
        return node.strength
    if isinstance(node, Conditional):
        if node.modality is not None:
            return node.modality.strength
        return modal_strength_of(node.then_)
    if isinstance(node, ExceptionRule):
        return modal_strength_of(node.rule)
    return None
