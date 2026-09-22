"""Final-mile factual guards for synthesis and public presentation.

Upstream ledgers remain authoritative.  This module does not infer new facts;
it prevents generated downstream prose from reintroducing unsupported numeric
claims after grounding or framework admission has rejected them.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from .epistemic_ledger import (
    claim_changes_admitted_outcome_type,
    grounded_numeric_literals,
)
from relent.precision import quantity_precision_escalation_errors


_NUMBER_TERM = (
    r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty)"
)
_CARDINAL_VALUES = {
    word: float(value) for value, word in enumerate((
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty",
    ))
}
_RATIO_CLAIM = re.compile(
    rf"(?:\b{_NUMBER_TERM}\s*[x×]\b|\b(?:ratio|odds)\b|"
    rf"\b{_NUMBER_TERM}\s+times\b)",
    re.IGNORECASE,
)
_MORTALITY = re.compile(
    r"\b(?:dead|death|deaths|die|dies|died|fatal|fatalities|kill(?:s|ed|ing)?|mortality)\b",
    re.IGNORECASE,
)
_TRAPPING = re.compile(r"\b(?:trap|traps|trapped|trapping)\b", re.IGNORECASE)
_EXACT_CERTAINTY = re.compile(
    r"\b(?:certainly|guaranteed|guarantees|definitely|inevitably|"
    r"exact(?:ly)? certain|100\s*%\s+(?:chance|probability))\b",
    re.IGNORECASE,
)
_HEDGED_MODALITY = re.compile(
    r"\b(?:may|might|could|possible|possibly|chance|risk|likely|unlikely|"
    r"conditional|if|unless)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:not|never|without|doesn't|does not|do not|did not|cannot|can't|"
    r"will not|won't|no longer)\b",
    re.IGNORECASE,
)
_CONDITIONAL_CUE = re.compile(
    r"\b(?:if|unless|when|provided|conditional(?:ly)?|may|might|could|"
    r"chance|risk|likely|unlikely|possible|possibly)\b",
    re.IGNORECASE,
)
_NEGATED_HARM = re.compile(
    r"\b(?:not|never|without|doesn't|does not|do not|did not|cannot|can't|"
    r"will not|won't)\b.{0,48}\b(?:increase|cause|create|worsen|inflict)?\w*"
    r".{0,20}\b(?:harm|damage|injury|loss|risk)\w*\b",
    re.IGNORECASE,
)
_POSITIVE_IMPROVEMENT = re.compile(
    r"\b(?:benefit|benefits|beneficial|improve|improves|improved|reduces?|"
    r"decreases?|protects?|saves?|prevents?)\b",
    re.IGNORECASE,
)
_CONTENT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for",
    "from", "if", "in", "is", "it", "may", "might", "of", "on", "or",
    "that", "the", "their", "then", "to", "when", "will", "with", "would",
})


def _content_stems(text: str) -> set[str]:
    stems: set[str] = set()
    for word in re.findall(r"[a-z]+", str(text or "").casefold()):
        if word in _CONTENT_STOPWORDS or len(word) < 3:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)]
                break
        stems.add(word)
    return stems


def _predicate_overlap(left: str, right: str) -> bool:
    shared = _content_stems(left) & _content_stems(right)
    return len(shared) >= 2 or bool(shared & {
        "harm", "damage", "recover", "surviv", "die", "death", "trap",
        "restore", "fail", "loss",
    })


def numeric_claim_integrity_errors(
    text: str,
    *,
    source_texts: Sequence[str] = (),
    records: Iterable[Any] = (),
    allow_derived_ratio: bool = False,
) -> list[str]:
    """Return errors when prose mints a number or an uncertified ratio."""
    claim = " ".join(str(text or "").split())
    if not claim:
        return []
    # Claim-side comparison is literal.  ``grounded_numeric_literals`` also
    # recognizes composite quantity phrases for source admission; applying
    # that expansion to generated prose can manufacture a sum (for example,
    # "one ... five" -> six) that the prose never asserted.
    from .local_specialists import _numeric_literals

    admitted = grounded_numeric_literals(*source_texts, records=records)
    claimed = _numeric_literals(claim)
    claimed.update(
        _CARDINAL_VALUES[word.casefold()]
        for word in re.findall(
            r"\b(?:" + "|".join(_CARDINAL_VALUES) + r")\b",
            claim,
            flags=re.IGNORECASE,
        )
    )
    errors: list[str] = []
    novel = sorted(claimed - admitted)
    if novel:
        rendered = ", ".join(f"{value:g}" for value in novel[:6])
        errors.append(f"numeric claim lacks admitted source quantity: {rendered}")
    if _RATIO_CLAIM.search(claim) and not allow_derived_ratio:
        errors.append("ratio claim lacks a verified arithmetic certificate")
    errors.extend(quantity_precision_escalation_errors(
        source_texts=source_texts,
        claim_text=claim,
    ))
    return errors


def semantic_transformation_errors(
    text: str,
    *,
    source_texts: Sequence[str] = (),
    records: Iterable[Any] = (),
    world_effects: Iterable[Any] = (),
) -> list[str]:
    """Reject downstream prose that strengthens an admitted outcome or modality."""
    claim = " ".join(str(text or "").split())
    source = " ".join(str(item or "") for item in source_texts)
    rows = [item for item in records if isinstance(item, dict)]
    effects = [item for item in world_effects if isinstance(item, dict)]
    errors: list[str] = []
    ledger = {
        str(item.get("proposition_id")): item
        for item in rows if item.get("proposition_id")
    }
    if ledger and claim_changes_admitted_outcome_type(claim, ledger):
        errors.append("claim changes an admitted non-mortality outcome into mortality")
    elif (
        _MORTALITY.search(claim)
        and _TRAPPING.search(source)
        and not _MORTALITY.search(source)
    ):
        errors.append("claim converts trapping into mortality without admitted support")
    if (
        _EXACT_CERTAINTY.search(claim)
        and _HEDGED_MODALITY.search(source)
        and not _EXACT_CERTAINTY.search(source)
    ):
        errors.append("claim converts a hedged or conditional outcome into certainty")
    source_atoms = [
        " ".join((
            str(effect.get("outcome") or ""),
            str(effect.get("source_proposition") or ""),
        )).strip()
        for effect in effects
        if _predicate_overlap(
            claim,
            " ".join((
                str(effect.get("outcome") or ""),
                str(effect.get("source_proposition") or ""),
            )),
        )
    ]
    if not source_atoms:
        source_atoms = [
            segment.strip()
            for text in source_texts
            for segment in re.split(r"(?<=[.!?;])\s+", str(text or ""))
            if _predicate_overlap(claim, segment)
        ]
    for atom in source_atoms:
        if _NEGATION.search(atom) and not _NEGATION.search(claim):
            errors.append("claim drops source negation from the matched predicate")
        if _NEGATED_HARM.search(atom) and _POSITIVE_IMPROVEMENT.search(claim):
            errors.append("absence of harm does not establish a positive benefit")
    relevant_effects = [
        effect for effect in effects
        if any(_predicate_overlap(claim, atom) for atom in (
            str(effect.get("outcome") or ""),
            str(effect.get("source_proposition") or ""),
        ))
    ]
    conditional_effect = any(
        effect.get("condition_ids")
        or str(effect.get("modality") or "").upper() in {
            "STIPULATED_CONDITIONAL", "PROBABILISTIC", "POSSIBLE",
        }
        for effect in relevant_effects
    )
    conditional_atom = any(_CONDITIONAL_CUE.search(atom) for atom in source_atoms)
    if (
        (conditional_effect or conditional_atom)
        and not _CONDITIONAL_CUE.search(claim)
    ):
        errors.append("claim drops a condition or modality from the matched outcome")
    return list(dict.fromkeys(errors))


def candidate_public_claim_is_admissible(
    data: dict[str, Any], candidate: dict[str, Any], text: str,
) -> bool:
    """Apply the same final-mile factual rules to every normative framework."""
    specialist = str(candidate.get("specialist") or "").strip().lower()
    ev_status = str(
        candidate.get("expected_value_validation_status") or "NOT_CLAIMED"
    ).upper()
    allow_ratio = specialist == "utilitarian" and ev_status == "ARITHMETIC_VERIFIED"
    source_texts = (
        str(data.get("scenario") or ""),
        *(str(action) for action in data.get("actions") or []),
    )
    records = data.get("proposition_ledger") or ()
    grounding = data.get("action_source_grounding") or {}
    world_model = grounding.get("world_model") or {}
    effects = world_model.get("effects") or ()
    return not (
        numeric_claim_integrity_errors(
            text,
            source_texts=source_texts,
            records=records,
            allow_derived_ratio=allow_ratio,
        )
        + semantic_transformation_errors(
            text,
            source_texts=source_texts,
            records=records,
            world_effects=effects,
        )
    )


__all__ = (
    "candidate_public_claim_is_admissible",
    "numeric_claim_integrity_errors",
    "semantic_transformation_errors",
)
