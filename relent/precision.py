"""Quantity precision non-escalation (pure claim/source text rules).

``APPROXIMATES`` does not license sharpening: source ``thousands`` may not
become ``10,000``, life-year products, or invented headcounts unless those
numerals are already licensed by source text.
"""
from __future__ import annotations

import re
from typing import Sequence

_VAGUE_QUANTITY_COLLECTIVE = re.compile(
    r"\b(?:"
    r"dozens|"
    r"hundreds|"
    r"thousands|"
    r"millions|"
    r"several\s+thousands?|"
    r"a\s+few\s+thousands?|"
    r"many\s+thousands?"
    r")\b",
    re.IGNORECASE,
)
_VAGUE_COLLECTIVE_BANDS: tuple[tuple[re.Pattern[str], float, float], ...] = (
    (re.compile(r"\bdozens\b", re.I), 10.0, 99.0),
    (re.compile(r"\bhundreds\b", re.I), 100.0, 999.0),
    (re.compile(r"\b(?:several|a\s+few|many)\s+thousands?\b|\bthousands\b", re.I), 1_000.0, 999_999.0),
    (re.compile(r"\bmillions\b", re.I), 1_000_000.0, 999_999_999_999.0),
)
# Spaced or comma groups: 100 000, 10,000, 8 000 000.
_CLAIM_NUMERIC_LITERAL = re.compile(
    r"(?<![A-Za-z0-9])(?P<approx>~|≈|about\s+|roughly\s+|approximately\s+|around\s+)?"
    r"(?P<n>\d{1,3}(?:[,\s]\d{3})+|\d+)(?!\s*%)",
    re.IGNORECASE,
)
# 5-10,000 / 5–10 000 style ranges (second endpoint is the sharp magnitude).
_CLAIM_NUMERIC_RANGE = re.compile(
    r"(?<![A-Za-z0-9])(?P<lo>\d{1,3}(?:[,\s]\d{3})*|\d+)\s*[–—-]\s*"
    r"(?P<hi>\d{1,3}(?:[,\s]\d{3})+|\d+)(?!\s*%)",
    re.IGNORECASE,
)
_LIFE_YEAR_UNIT = re.compile(
    r"\b(?:life[\s-]?years?|qalys?|dalys?)\b",
    re.IGNORECASE,
)
_POPULATION_UNIT = re.compile(
    r"\b(?:residents?|people|persons?|lives|deaths?|victims?|habitants?)\b",
    re.IGNORECASE,
)
_YEAR_REMAINING = re.compile(
    r"\b\d{1,3}(?:[,\s]\d{3})*\s*[\s-]?years?\b",
    re.IGNORECASE,
)
_PRODUCT_MARK = re.compile(
    r"(?:[×*]|(?<![A-Za-z])x(?![A-Za-z])|\btimes\b)",
    re.IGNORECASE,
)


def _parse_claim_count_literal(raw: str) -> float | None:
    text = str(raw or "").replace(",", "").replace(" ", "")
    if not text or not re.fullmatch(r"\d+(?:\.\d+)?", text):
        return None
    return float(text)


def _iter_claim_magnitudes(claim: str) -> list[tuple[str, float]]:
    """Surface form + numeric value for count-like literals in a claim."""
    found: list[tuple[str, float]] = []
    seen: set[tuple[int, int]] = set()
    for match in _CLAIM_NUMERIC_RANGE.finditer(claim):
        value = _parse_claim_count_literal(match.group("hi"))
        if value is None:
            continue
        seen.add((match.start("hi"), match.end("hi")))
        found.append((match.group(0).strip(), value))
    for match in _CLAIM_NUMERIC_LITERAL.finditer(claim):
        span = (match.start("n"), match.end("n"))
        if span in seen:
            continue
        value = _parse_claim_count_literal(match.group("n"))
        if value is None:
            continue
        found.append((match.group(0).strip(), value))
    return found


def _licensed_numerals(source_blob: str) -> set[float]:
    licensed: set[float] = set()
    for match in _CLAIM_NUMERIC_LITERAL.finditer(source_blob):
        value = _parse_claim_count_literal(match.group("n"))
        if value is not None:
            licensed.add(value)
    for match in _CLAIM_NUMERIC_RANGE.finditer(source_blob):
        for key in ("lo", "hi"):
            value = _parse_claim_count_literal(match.group(key))
            if value is not None:
                licensed.add(value)
    return licensed


def _vague_labels(source_blob: str) -> str:
    return ", ".join(
        sorted({m.group(0) for m in _VAGUE_QUANTITY_COLLECTIVE.finditer(source_blob)})
    )


def _in_vague_band(source_blob: str, value: float) -> bool:
    return any(
        pattern.search(source_blob) and low <= value <= high
        for pattern, low, high in _VAGUE_COLLECTIVE_BANDS
    )


def _claim_invents_life_year_product(claim: str) -> bool:
    """True for unlicensed life-year / N×years welfare arithmetic."""
    if _LIFE_YEAR_UNIT.search(claim) and _CLAIM_NUMERIC_LITERAL.search(claim):
        return True
    if _PRODUCT_MARK.search(claim) and _POPULATION_UNIT.search(claim):
        if _YEAR_REMAINING.search(claim) or len(_iter_claim_magnitudes(claim)) >= 2:
            return True
    return False


def quantity_precision_escalation_errors(
    *,
    source_texts: Sequence[str],
    claim_text: str,
) -> list[str]:
    """Reject claims that sharpen a vague source quantity into an exact numeral.

    ``QUANTITY_PRECISION_NON_ESCALATION``: source ``thousands`` may not become
    ``10,000`` / ``~10,000`` / ``100 000 × 80`` life-years unless those exact
    numerals are already licensed by source text. Digit-free paraphrases of the
    vague span are allowed.
    """
    source_blob = " ".join(
        " ".join(str(text or "").split()) for text in source_texts if str(text or "").strip()
    )
    claim = " ".join(str(claim_text or "").split())
    if not source_blob or not claim:
        return []
    if not _VAGUE_QUANTITY_COLLECTIVE.search(source_blob):
        return []
    licensed = _licensed_numerals(source_blob)
    labels = _vague_labels(source_blob)
    errors: list[str] = []

    if _claim_invents_life_year_product(claim):
        unlicensed = [
            surface for surface, value in _iter_claim_magnitudes(claim)
            if value not in licensed
        ]
        if unlicensed:
            errors.append(
                "quantity precision escalation: claim invents life-year or "
                f"product arithmetic {unlicensed!r} but source only licenses "
                f"a vague collective ({labels}); retain the source span or cite "
                "a derivation that licenses the refinement "
                "(QUANTITY_PRECISION_NON_ESCALATION)"
            )
            return errors

    seen_values: set[float] = set()
    for surface, value in _iter_claim_magnitudes(claim):
        if value in seen_values:
            continue
        seen_values.add(value)
        if value in licensed:
            continue
        if not _in_vague_band(source_blob, value):
            continue
        errors.append(
            f"quantity precision escalation: claim uses {surface!r} but source "
            f"only licenses a vague collective ({labels}); "
            "retain the source span or cite a derivation that licenses the refinement "
            "(QUANTITY_PRECISION_NON_ESCALATION)"
        )
    return errors
