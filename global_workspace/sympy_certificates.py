"""SymPy ranking certificates and related Util algebraic witnesses.

Derived artifacts only. Callers pass already-parsed operands; this module does
not parse English, resolve parties, or invent ownership.

Prefer expand/cancel for algebraic witnesses. Do not treat heuristic
``simplify()`` as the sole oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from sympy import (
    Expr, Integer, Interval, Symbol, cancel, expand, solveset, S,
)
from sympy.core.relational import Relational

from .sympy_arith import signed_welfare_term, sum_welfare_terms, to_sympy_number


ROBUST_LEADER = "ROBUST_LEADER"
ROBUST_CHALLENGER = "ROBUST_CHALLENGER"
CONDITIONAL = "CONDITIONAL"
INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True, slots=True)
class RankingCertificate:
    """Serializable Util ranking witness built from temporary SymPy exprs."""

    leader: str
    challenger: str
    gap: str
    assumptions: dict[str, str]
    boundary_symbol: str
    boundary_condition: str
    boundary_kind: str
    ranking: str
    robustness_status: str = ""
    robustness_interval: str = ""
    robustness_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        boundary: dict[str, str] = {}
        if self.boundary_condition:
            boundary = {
                "symbol": self.boundary_symbol,
                "condition": self.boundary_condition,
                "kind": self.boundary_kind,
            }
        robustness: dict[str, str] = {}
        if self.robustness_status:
            robustness = {
                "status": self.robustness_status,
                "interval": self.robustness_interval,
                "note": self.robustness_note,
            }
        return {
            "schema_version": 1,
            "ranking": self.ranking,
            "leader": self.leader,
            "challenger": self.challenger,
            "gap": self.gap,
            "assumptions": dict(self.assumptions),
            "boundary": boundary,
            "robustness": robustness,
            "provenance": "sympy_ranking_certificate",
        }


@dataclass(frozen=True, slots=True)
class EquivalenceCheck:
    ok: bool
    left: str
    right: str
    difference: str
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConservationCheck:
    ok: bool
    relation: str
    left: str
    right: str
    errors: tuple[str, ...] = ()


def eu_from_terms(
    terms: Sequence[tuple[float, float, bool]],
    *,
    free_probability_index: int | None = None,
    free_symbol: Symbol | None = None,
) -> Expr:
    """Expected utility: sum ± p * m, optionally leaving one probability symbolic."""
    pieces: list[Expr] = []
    for index, (probability, magnitude, benefit) in enumerate(terms):
        if free_probability_index == index and free_symbol is not None:
            sign = Integer(1) if benefit else Integer(-1)
            pieces.append(sign * free_symbol * to_sympy_number(magnitude))
        else:
            pieces.append(
                signed_welfare_term(probability, magnitude, benefit=benefit)
            )
    return sum_welfare_terms(pieces)


def ranking_gap(eu_leader: Expr, eu_challenger: Expr) -> Expr:
    """EU_leader - EU_challenger after expand/cancel (not heuristic simplify)."""
    return cancel(expand(eu_leader - eu_challenger))


def _format_expr(expression: Expr) -> str:
    return str(cancel(expand(expression)))


def expressions_equivalent(left: Expr, right: Expr) -> EquivalenceCheck:
    """True when expand/cancel(left - right) is identically zero."""
    left_n = cancel(expand(left))
    right_n = cancel(expand(right))
    difference = cancel(expand(left_n - right_n))
    ok = difference == 0
    errors: tuple[str, ...] = ()
    if not ok:
        errors = (f"expressions differ by {_format_expr(difference)}",)
    return EquivalenceCheck(
        ok, _format_expr(left_n), _format_expr(right_n), _format_expr(difference),
        errors,
    )


def verify_derivation_equivalence(
    left_terms: Sequence[tuple[float, float, bool]],
    right_terms: Sequence[tuple[float, float, bool]],
) -> EquivalenceCheck:
    """Two legal term-lists claim the same welfare quantity."""
    return expressions_equivalent(
        eu_from_terms(left_terms),
        eu_from_terms(right_terms),
    )


def check_sum_leq_total(
    parts: Sequence[float | int],
    total: float | int,
) -> ConservationCheck:
    """allocated parts sum ≤ resource total (exact Rational inequality)."""
    left = sum((to_sympy_number(part) for part in parts), Integer(0))
    right = to_sympy_number(total)
    ok = bool(left <= right)
    relation = f"{_format_expr(left)} <= {_format_expr(right)}"
    errors: tuple[str, ...] = ()
    if not ok:
        errors = (f"conservation violated: {relation}",)
    return ConservationCheck(
        ok, relation, _format_expr(left), _format_expr(right), errors,
    )


def check_part_leq_whole(
    part: float | int,
    whole: float | int,
) -> ConservationCheck:
    """subgroup / saved+lost style part ≤ whole."""
    return check_sum_leq_total([part], whole)


def one_variable_boundary(
    gap: Expr,
    symbol: Symbol,
) -> tuple[str, str]:
    """Return (condition_string, kind) for gap > 0 in one free variable."""
    try:
        solution = solveset(gap > 0, symbol, domain=S.Reals)
    except Exception:  # noqa: BLE001 — fail closed to unsupported
        return "", "unsupported"
    text = str(solution)
    if solution == S.EmptySet:
        return "", "empty"
    if solution == S.Reals:
        return f"all real {symbol}", "full"
    if isinstance(solution, Relational):
        return str(solution), "relational"
    try:
        if isinstance(solution, Interval):
            left, right = solution.start, solution.end
            left_open, right_open = solution.left_open, solution.right_open
            if left.is_infinite and not right.is_infinite:
                op = "<" if right_open else "<="
                return f"{symbol} {op} {_format_expr(right)}", (
                    "strict_lt" if right_open else "leq"
                )
            if right.is_infinite and not left.is_infinite:
                op = ">" if left_open else ">="
                return f"{symbol} {op} {_format_expr(left)}", (
                    "strict_gt" if left_open else "geq"
                )
    except Exception:  # noqa: BLE001
        pass
    return f"{symbol} in {text}", "set"


def assess_interval_robustness(
    gap: Expr,
    symbol: Symbol,
    lo: float | int,
    hi: float | int,
) -> tuple[str, str, str]:
    """Classify leader advantage over p ∈ [lo, hi]."""
    low = to_sympy_number(lo)
    high = to_sympy_number(hi)
    if not (low <= high):
        return INDETERMINATE, "", "empty or reversed interval"
    interval = Interval(low, high)
    interval_str = f"[{_format_expr(low)}, {_format_expr(high)}]"
    try:
        win = solveset(gap > 0, symbol, domain=S.Reals)
        lose = solveset(gap < 0, symbol, domain=S.Reals)
        win_here = win.intersect(interval)
        lose_here = lose.intersect(interval)
    except Exception as exc:  # noqa: BLE001
        return INDETERMINATE, interval_str, f"solveset failed: {exc}"

    wins = win_here != S.EmptySet
    loses = lose_here != S.EmptySet
    try:
        win_covers = win_here.measure == interval.measure
        lose_covers = lose_here.measure == interval.measure
    except Exception:  # noqa: BLE001
        win_covers = win_here == interval
        lose_covers = lose_here == interval

    if win_covers and not loses:
        return ROBUST_LEADER, interval_str, "gap > 0 on the whole interval"
    if lose_covers and not wins:
        return ROBUST_CHALLENGER, interval_str, "gap < 0 on the whole interval"
    if wins and loses:
        return CONDITIONAL, interval_str, "ranking flips inside the interval"
    try:
        zero_here = solveset(gap, symbol, domain=S.Reals).intersect(interval)
        if zero_here != S.EmptySet and not wins and not loses:
            return INDETERMINATE, interval_str, "gap is zero on the interval"
    except Exception:  # noqa: BLE001
        pass
    return INDETERMINATE, interval_str, "could not classify interval robustness"


def _term_assumptions(
    action_id: str,
    terms: Sequence[tuple[float, float, bool]],
) -> dict[str, str]:
    assumptions: dict[str, str] = {}
    for index, (probability, magnitude, benefit) in enumerate(terms):
        prefix = f"{action_id}[{index}]"
        assumptions[f"{prefix}.p"] = _format_expr(to_sympy_number(probability))
        assumptions[f"{prefix}.m"] = _format_expr(to_sympy_number(magnitude))
        assumptions[f"{prefix}.direction"] = "BENEFIT" if benefit else "HARM"
    return assumptions


def _uncertain_probability_slots(
    leader_terms: Sequence[tuple[float, float, bool]],
    challenger_terms: Sequence[tuple[float, float, bool]],
) -> list[tuple[str, int, float]]:
    slots: list[tuple[str, int, float]] = []
    for side, terms in (("leader", leader_terms), ("challenger", challenger_terms)):
        for index, (probability, _magnitude, _benefit) in enumerate(terms):
            if abs(float(probability) - 0.0) < 1e-15:
                continue
            if abs(float(probability) - 1.0) < 1e-15:
                continue
            slots.append((side, index, float(probability)))
    return slots


def build_ranking_certificate(
    *,
    leader: str,
    challenger: str,
    leader_terms: Sequence[tuple[float, float, bool]],
    challenger_terms: Sequence[tuple[float, float, bool]],
    probability_interval: tuple[float | int, float | int] | None = None,
) -> RankingCertificate | None:
    """Certificate for leader ≻ challenger from admitted numeric terms."""
    if not leader or not challenger or leader == challenger:
        return None
    if not leader_terms or not challenger_terms:
        return None

    assumptions = {
        **_term_assumptions("leader", leader_terms),
        **_term_assumptions("challenger", challenger_terms),
    }
    eu_l = eu_from_terms(leader_terms)
    eu_c = eu_from_terms(challenger_terms)
    gap = ranking_gap(eu_l, eu_c)
    gap_text = _format_expr(gap)

    boundary_symbol = ""
    boundary_condition = ""
    boundary_kind = ""
    robustness_status = ""
    robustness_interval = ""
    robustness_note = ""
    free_gap: Expr | None = None
    free_symbol: Symbol | None = None
    slots = _uncertain_probability_slots(leader_terms, challenger_terms)
    if len(slots) == 1:
        side, index, observed = slots[0]
        symbol_name = f"p_{side}_{index}"
        symbol = Symbol(symbol_name, real=True, nonnegative=True)
        if side == "leader":
            eu_l_free = eu_from_terms(
                leader_terms, free_probability_index=index, free_symbol=symbol,
            )
            eu_c_free = eu_from_terms(challenger_terms)
        else:
            eu_l_free = eu_from_terms(leader_terms)
            eu_c_free = eu_from_terms(
                challenger_terms, free_probability_index=index, free_symbol=symbol,
            )
        free_gap = ranking_gap(eu_l_free, eu_c_free)
        free_symbol = symbol
        boundary_condition, boundary_kind = one_variable_boundary(free_gap, symbol)
        boundary_symbol = symbol_name
        assumptions[f"{symbol_name}.observed"] = _format_expr(to_sympy_number(observed))

    if (
        probability_interval is not None
        and free_gap is not None
        and free_symbol is not None
    ):
        robustness_status, robustness_interval, robustness_note = (
            assess_interval_robustness(
                free_gap, free_symbol,
                probability_interval[0], probability_interval[1],
            )
        )

    return RankingCertificate(
        leader=leader,
        challenger=challenger,
        gap=gap_text,
        assumptions=assumptions,
        boundary_symbol=boundary_symbol,
        boundary_condition=boundary_condition,
        boundary_kind=boundary_kind,
        ranking=f"{leader} > {challenger}",
        robustness_status=robustness_status,
        robustness_interval=robustness_interval,
        robustness_note=robustness_note,
    )


def certificate_challenger(
    nets: Mapping[str, float],
    leader: str,
) -> str | None:
    """Unique second-best action under higher-is-better nets, if any."""
    if leader not in nets:
        return None
    others = {action: value for action, value in nets.items() if action != leader}
    if not others:
        return None
    best = max(others, key=others.get)
    if list(others.values()).count(others[best]) != 1:
        return None
    return best
