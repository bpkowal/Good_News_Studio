"""Shared semantic invariants for blueprint party and allocation quantities."""
from __future__ import annotations

import re


_GROUP = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"(?:\s+[A-Za-z]+){0,2}\s+(people|workers|patients|residents|children)\b",
    re.IGNORECASE,
)


def group_quantity(label: str) -> list[str]:
    """Return only a headcount syntactically bound to a plural human group."""
    match = _GROUP.search(label or "")
    return [match.group(1)] if match else []


def group_span(label: str) -> str | None:
    """Return the complete quantified human-group span, if present."""
    match = _GROUP.search(label or "")
    return match.group(0) if match else None


def party_kind(label: str) -> str:
    """Lexicon witness/veto only. Never write ``parties[].kind``.

    A hit may support or block a licensed proposal. Missing evidence is OTHER,
    not PERSON. PERSON is licensed from a construction role.
    """
    folded = label.casefold()
    if _GROUP.search(label):
        return "HUMAN_GROUP"
    if re.search(r"\b(?:dog|animal)\b", folded):
        return "ANIMAL"
    if re.search(r"\b(?:medicine|serum|water|dose|resource|antiviral)\b", folded):
        return "RESOURCE"
    if re.search(r"\b(?:farm|town|city|clinic)\b", folded):
        return "COMMUNITY"
    if re.search(r"\b(?:bot|model|system)\b", folded):
        return "OTHER"
    return "OTHER"


def complement_clause_ids(quantity_clause_ids: list[str]) -> list[str]:
    """Keep local complement magnitude scoped to the resource constraint.

    Exclusivity and parent-transfer provenance remain graph-level evidence and
    are attached by Parliament's complement compiler. Including those mixed
    clauses here lets an unrelated recipient headcount leak into the effect.
    """
    return list(dict.fromkeys(quantity_clause_ids))
