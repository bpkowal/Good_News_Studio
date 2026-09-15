"""Temporary SymPy arithmetic for Util nets and EV recomputes.

Canonical numbers stay in consequence tables and EV rows. This module builds a
temporary expression, checks identity against a claimed float, and discards the
expression — the same calculator pattern as ``graph_queries.to_networkx``.

Do not parse English here. Callers pass already-parsed probability, magnitude,
and direction signs.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from sympy import Abs, Expr, Integer, Rational, Symbol, cancel, expand


@dataclass(frozen=True, slots=True)
class ArithmeticCheck:
    """Result of comparing a claimed float to a temporary SymPy expression."""

    ok: bool
    claimed: float
    recomputed: float
    identity: str
    errors: tuple[str, ...] = ()
    symbolic_expression: str = ""


def to_sympy_number(value: float | int) -> Expr:
    """Exact Integer/Rational when the float is a simple decimal; else Rational."""
    if isinstance(value, bool):
        raise TypeError("boolean is not a welfare magnitude")
    if isinstance(value, int):
        return Integer(value)
    number = float(value)
    if not number == number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"non-finite arithmetic operand: {value!r}")
    if number.is_integer():
        return Integer(int(number))
    fraction = Fraction(number).limit_denominator(10_000)
    return Rational(fraction.numerator, fraction.denominator)


def signed_welfare_term(
    probability: float | int,
    magnitude: float | int,
    *,
    benefit: bool,
) -> Expr:
    """One admitted row: ± probability * magnitude."""
    sign = Integer(1) if benefit else Integer(-1)
    return sign * to_sympy_number(probability) * to_sympy_number(magnitude)


def sum_welfare_terms(terms: Iterable[Expr]) -> Expr:
    total: Expr = Integer(0)
    for term in terms:
        total += term
    return total


def expression_identity(expression: Expr) -> str:
    """Compact human-readable form for traces and specialist justification."""
    return str(cancel(expand(expression)))


def check_claimed_value(
    claimed: float,
    expression: Expr,
    *,
    rel_tol: float = 0.01,
    abs_tol: float = 0.001,
) -> ArithmeticCheck:
    """True when claimed matches the temporary expression within EV tolerances."""
    canonical = cancel(expand(expression))
    identity = str(canonical)
    try:
        recomputed = float(canonical)
    except Exception as exc:  # noqa: BLE001 — calculator must fail closed
        return ArithmeticCheck(
            False, float(claimed), 0.0, identity,
            (f"SymPy expression did not evaluate: {exc}",),
            symbolic_expression=identity,
        )
    claimed_f = float(claimed)
    delta = abs(claimed_f - recomputed)
    ok = delta <= max(abs_tol, rel_tol * max(abs(claimed_f), abs(recomputed)))
    try:
        claimed_expr = to_sympy_number(claimed_f)
        if cancel(expand(canonical - claimed_expr)) == 0:
            ok = True
    except (TypeError, ValueError):
        pass
    errors: tuple[str, ...] = ()
    if not ok:
        errors = (
            f"claimed {claimed_f:g} does not match SymPy {recomputed:g} ({identity})",
        )
    return ArithmeticCheck(
        ok, claimed_f, recomputed, identity, errors,
        symbolic_expression=identity,
    )


def verify_substitution(
    expression: Expr,
    bindings: Mapping[str, float | int],
    claimed: float,
    *,
    rel_tol: float = 0.01,
    abs_tol: float = 0.001,
) -> ArithmeticCheck:
    """Substitute named symbols into a symbolic EV and compare to claimed value."""
    substituted = cancel(expand(expression))
    free_by_name = {str(symbol): symbol for symbol in substituted.free_symbols}
    for name, value in bindings.items():
        symbol = free_by_name.get(name) or Symbol(name)
        substituted = substituted.subs(symbol, to_sympy_number(value))
    return check_claimed_value(claimed, substituted, rel_tol=rel_tol, abs_tol=abs_tol)


def expected_count_expression(
    terms: Sequence[tuple[float, float, float]],
) -> Expr:
    """Abs of sum(sign * magnitude * probability) for EV DIRECT/EXPECTED rows."""
    expression = sum_welfare_terms(
        to_sympy_number(sign) * to_sympy_number(magnitude) * to_sympy_number(probability)
        for sign, magnitude, probability in terms
    )
    return Abs(cancel(expand(expression)))


def verify_net_from_terms(
    claimed_net: float,
    terms: Sequence[tuple[float, float, bool]],
) -> ArithmeticCheck:
    """Verify one action net against (probability, magnitude, is_benefit) rows."""
    expression = sum_welfare_terms(
        signed_welfare_term(probability, magnitude, benefit=benefit)
        for probability, magnitude, benefit in terms
    )
    return check_claimed_value(claimed_net, expression)


def verify_expected_count(
    claimed: float,
    terms: Sequence[tuple[float, float, float]],
) -> ArithmeticCheck:
    """Verify DIRECT/EXPECTED count: sum of sign * magnitude * probability."""
    return check_claimed_value(claimed, expected_count_expression(terms))
