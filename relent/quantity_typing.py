"""Quantity-field typing: magnitude spans vs outcome/severity pseudo-quantities.

Recorded ``quantities`` must stay magnitude-bearing (``thousands of lives``,
``decades``, ``500+``). Outcome phrases such as ``catastrophic loss of life``
are severity/effect text, not math operands. Incomplete tails (``decades of``)
prefer a longer source-licensed form when available, else the bare collective.
"""
from __future__ import annotations

import re
from typing import Literal, Sequence

QuantityKind = Literal["MAGNITUDE", "PSEUDO_OUTCOME", "INCOMPLETE", "EMPTY"]

_MAGNITUDE_CUE = re.compile(
    r"\b(?:"
    r"\d|"
    r"dozen|score|tens|"
    r"hundreds?|thousands?|millions?|billions?|"
    r"decades?|centuries?|"
    r"several|approximately|about|roughly|nearly|almost|"
    r"over|under|at\s+least|at\s+most|more\s+than|fewer\s+than|less\s+than|"
    r"as\s+many\s+as|up\s+to|between"
    r")\b",
    re.IGNORECASE,
)
_OUTCOME_SEVERITY = re.compile(
    r"\b(?:"
    r"catastrophic|"
    r"loss\s+of\s+life|"
    r"(?:immediate\s+)?survival|"
    r"die|dies|died|dying|death|deaths|"
    r"kill|killed|killing|"
    r"fatal|fatality|fatalities|"
    r"casualt(?:y|ies)|"
    r"injur(?:y|ies|ed)|"
    r"harm(?:ed|s)?|"
    r"damage(?:d)?"
    r")\b",
    re.IGNORECASE,
)
_INCOMPLETE_OF = re.compile(
    r"^(?P<head>dozens?|scores?|tens|hundreds|thousands|millions|billions|"
    r"decades|centuries)\s+of$",
    re.IGNORECASE,
)
_COLLECTIVE_HEAD = re.compile(
    r"\b(?P<head>dozens?|scores?|tens|hundreds|thousands|millions|billions|"
    r"decades|centuries)\b",
    re.IGNORECASE,
)
# Unit tails we may attach after ``of`` — not open-ended English.
_QUANTITY_UNIT_TAIL = (
    r"(?:"
    r"lives?|people|persons?|residents?|deaths?|victims?|habitants?|"
    r"patients?|workers?|records?|years?|"
    r"medical\s+research|research|expertise|knowledge"
    r")"
)


def classify_quantity_span(span: str) -> QuantityKind:
    """Classify one recorded quantity candidate."""
    cleaned = " ".join(str(span or "").split()).strip(" ,.;:")
    if not cleaned:
        return "EMPTY"
    if _INCOMPLETE_OF.match(cleaned):
        return "INCOMPLETE"
    has_magnitude = bool(_MAGNITUDE_CUE.search(cleaned))
    has_outcome = bool(_OUTCOME_SEVERITY.search(cleaned))
    if has_outcome and not has_magnitude:
        return "PSEUDO_OUTCOME"
    if has_magnitude:
        return "MAGNITUDE"
    # Multi-word non-magnitude strings are almost always outcome leftovers.
    if len(cleaned.split()) >= 3:
        return "PSEUDO_OUTCOME"
    return "MAGNITUDE"


def _licensed_longer_form(span: str, source_texts: Sequence[str]) -> str:
    """Prefer a longer source phrase headed by the same collective/numeral."""
    cleaned = " ".join(str(span or "").split()).strip(" ,.;:")
    if not cleaned:
        return ""
    incomplete = _INCOMPLETE_OF.match(cleaned)
    head = incomplete.group("head") if incomplete else cleaned
    head_match = _COLLECTIVE_HEAD.search(head)
    if head_match is not None:
        head = head_match.group("head")
    pattern = re.compile(
        rf"(?<![\w.-]){re.escape(head)}"
        rf"(?:\s+of\s+{_QUANTITY_UNIT_TAIL})?"
        rf"(?![\w-])",
        re.IGNORECASE,
    )
    best = head if incomplete else cleaned
    for text in source_texts:
        for match in pattern.finditer(str(text or "")):
            candidate = " ".join(match.group(0).split())
            if len(candidate) > len(best):
                best = candidate
    return best


def sanitize_recorded_quantities(
    quantities: Sequence[str],
    *,
    source_texts: Sequence[str] = (),
) -> tuple[str, ...]:
    """Drop pseudo-outcomes; complete incomplete tails from licensed source."""
    kept: list[str] = []
    seen: set[str] = set()
    for raw in quantities:
        kind = classify_quantity_span(raw)
        if kind in {"EMPTY", "PSEUDO_OUTCOME"}:
            continue
        cleaned = " ".join(str(raw or "").split()).strip(" ,.;:")
        if kind == "INCOMPLETE":
            completed = _licensed_longer_form(cleaned, source_texts)
        else:
            completed = cleaned
        if not completed:
            continue
        if classify_quantity_span(completed) == "PSEUDO_OUTCOME":
            continue
        # Incomplete with no licensed unit becomes the bare collective head.
        if classify_quantity_span(completed) == "INCOMPLETE":
            incomplete = _INCOMPLETE_OF.match(completed)
            completed = incomplete.group("head") if incomplete else completed
        key = completed.casefold()
        if key in seen:
            continue
        dominated = False
        for index, existing in enumerate(list(kept)):
            if existing.casefold() == key:
                dominated = True
                break
            if existing.casefold() in key and len(completed) > len(existing):
                kept[index] = completed
                seen.discard(existing.casefold())
                seen.add(key)
                dominated = True
                break
            if key in existing.casefold() and len(existing) > len(completed):
                dominated = True
                break
        if dominated:
            continue
        seen.add(key)
        kept.append(completed)
    return tuple(kept)


def magnitude_quantity_spans(quantities: Sequence[str]) -> tuple[str, ...]:
    """Filter for math / Util consumers: magnitude-class spans only."""
    return tuple(
        " ".join(str(value).split())
        for value in quantities
        if classify_quantity_span(value) == "MAGNITUDE"
    )


def pseudo_quantity_errors(
    quantities: Sequence[str],
    *,
    prefix: str = "",
) -> list[str]:
    """Admit-time errors when outcome phrases leak into quantities."""
    label = f"{prefix} " if prefix else ""
    errors: list[str] = []
    for raw in quantities:
        kind = classify_quantity_span(raw)
        if kind == "PSEUDO_OUTCOME":
            errors.append(
                f"{label}quantity {raw!r} is an outcome/severity phrase, not a "
                "magnitude span (QUANTITY_FIELD_TYPING)"
            )
        elif kind == "INCOMPLETE":
            errors.append(
                f"{label}quantity {raw!r} is an incomplete magnitude span; "
                "record the source-licensed form (QUANTITY_FIELD_TYPING)"
            )
    return errors
