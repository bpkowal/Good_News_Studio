"""Action-conditioned paraphrase binding for epistemic status transfer.

Licensed ``PARAPHRASE_OF`` / ``EQUIVALENT_TO`` edges only — not free similarity.
If an admitted world row is A1 → CERTAIN catastrophic loss, claims that only
rephrase that branch (no-purge / refrain / purge not executed) may inherit
status. Wrong-action cues must not transfer.
"""
from __future__ import annotations

import re
from typing import Sequence

_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "if",
    "is", "are", "be", "by", "at", "as", "it", "its", "from", "with",
    "that", "this", "then", "than", "into", "over", "under", "when",
    "while", "would", "will", "can", "may", "occurs", "occur", "cause",
    "causes", "caused", "causing", "lead", "leads", "leading", "result",
    "results", "resulting", "follow", "follows", "following",
})
_REFRAIN_FAMILY = re.compile(
    r"\b(?:refrain|withhold|omit|decline|forgo|forego)(?:s|ed|ing)?\b|"
    r"\bno\s+(?:emergency\s+)?purge\b|"
    r"\bwithout\s+(?:the\s+)?(?:emergency\s+)?purge\b|"
    r"\bpurge\s+not\s+(?:executed|triggered|deployed|run)\b|"
    r"\bnot\s+(?:execute|trigger|deploy|run)\b.{0,40}\bpurge\b|"
    r"\bif\s+(?:the\s+)?purge\s+is\s+not\b|"
    r"\bif\s+no\s+purge\b",
    re.IGNORECASE,
)
_EXECUTE_FAMILY = re.compile(
    r"\b(?:execute|executes|executed|executing|deploy|deploys|deployed|deploying|"
    r"trigger|triggers|triggered|triggering|run|runs|running)\b"
    r".{0,40}\b(?:emergency\s+)?purge\b|"
    r"\b(?:emergency\s+)?purge\b.{0,24}\b(?:executed|deployed|triggered|now)\b|"
    r"\bif\s+(?:the\s+)?(?:emergency\s+)?purge\s+(?:is\s+)?(?:executed|deployed|triggered)\b",
    re.IGNORECASE,
)
_LOSS_OUTCOME = re.compile(
    r"\b(?:catastrophic\s+)?loss\s+of\s+life\b|"
    r"\b(?:die|dies|died|dying|death|deaths|kill|killed|fatal)\b|"
    r"\b(?:mass\s+)?casualt",
    re.IGNORECASE,
)
_SURVIVAL_OUTCOME = re.compile(
    r"\b(?:immediate\s+)?survival\b|"
    r"\b(?:survive|survives|survived|saved|spare|spared)\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(str(text or ""))
        if token.casefold() not in _STOP and len(token) > 1
    }


def _stem(token: str) -> str:
    word = token.casefold()
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def _stem_set(text: str) -> set[str]:
    return {_stem(token) for token in _tokens(text)}


def outcome_content_overlaps(claim: str, outcome: str) -> bool:
    """True when claim restates the outcome's content stems."""
    outcome_stems = _stem_set(outcome)
    if not outcome_stems:
        return False
    claim_stems = _stem_set(claim)
    if not claim_stems:
        return False
    # Prefer substantive overlap, not a single stop-ish stem.
    shared = claim_stems & outcome_stems
    if len(shared) >= 2:
        return True
    if outcome.casefold() in claim.casefold():
        return True
    if _LOSS_OUTCOME.search(outcome) and _LOSS_OUTCOME.search(claim):
        return True
    if _SURVIVAL_OUTCOME.search(outcome) and _SURVIVAL_OUTCOME.search(claim):
        return True
    return bool(shared) and len(outcome_stems) <= 2


# Stems shared by both refrain and execute glosses of the same intervention.
_AMBIGUOUS_ACTION_STEMS = frozenset({
    "purge", "emergenc", "execut", "trigger", "deploi", "deploy", "run",
    "action", "intervent",
})


def gloss_cues_claim(
    claim: str,
    glosses: Sequence[str],
    *,
    exclude_stems: set[str] | None = None,
) -> bool:
    """True when claim shares distinctive action stems with an action gloss.

    Outcome / party stems must be excluded so that glosses which mention the
    effect (``option that injured X``) do not license bare outcome restatements.
    """
    claim_stems = _stem_set(claim)
    if not claim_stems:
        return False
    blocked = set(exclude_stems or ()) | _AMBIGUOUS_ACTION_STEMS
    for gloss in glosses:
        folded = str(gloss or "").casefold().strip()
        if folded and len(folded) > 12 and folded in claim.casefold():
            return True
        gloss_stems = _stem_set(gloss)
        distinctive = (claim_stems & gloss_stems) - blocked
        if len(distinctive) >= 1 and (
            distinctive & {
                "refrain", "withhold", "omit", "declin", "forgo", "forego",
            }
            or len(distinctive) >= 2
        ):
            return True
    return False


def _action_family(glosses: Sequence[str]) -> str:
    blob = " ".join(str(item or "") for item in glosses)
    if _REFRAIN_FAMILY.search(blob):
        return "REFRAIN"
    if _EXECUTE_FAMILY.search(blob):
        return "EXECUTE"
    return ""


def _claim_action_family(claim: str) -> str:
    # Prefer explicit refrain/no-purge over bare "purge" mentions.
    if _REFRAIN_FAMILY.search(claim):
        return "REFRAIN"
    if _EXECUTE_FAMILY.search(claim):
        return "EXECUTE"
    return ""


def action_families_conflict(
    claim: str,
    action_glosses: Sequence[str],
) -> bool:
    """True when claim cues the opposed intervention family from the glosses."""
    claim_family = _claim_action_family(claim)
    record_family = _action_family(action_glosses)
    return bool(claim_family and record_family and claim_family != record_family)


def licensed_action_paraphrase(
    claim: str,
    *,
    outcome: str,
    polarity: str,
    action_id: str = "",
    action_glosses: Sequence[str] = (),
) -> bool:
    """True when claim is a licensed paraphrase of an action-conditioned effect.

    Does not use embedding similarity. Requires outcome overlap plus an action
    cue (action id, gloss stems, or refrain/execute family aligned with glosses).
    """
    cleaned = " ".join(str(claim or "").split())
    if not cleaned or not outcome_content_overlaps(cleaned, outcome):
        return False
    polarity_key = str(polarity or "").upper()
    if polarity_key == "ADVERSE" and _SURVIVAL_OUTCOME.search(cleaned) and not _LOSS_OUTCOME.search(cleaned):
        return False
    if polarity_key == "BENEFICIAL" and _LOSS_OUTCOME.search(cleaned) and not re.search(
        r"\b(?:avert|prevent|avoid)\b", cleaned, re.I,
    ):
        return False
    claim_family = _claim_action_family(cleaned)
    record_family = _action_family(action_glosses)
    if claim_family and record_family and claim_family != record_family:
        return False
    if action_id and re.search(rf"\b{re.escape(action_id)}\b", cleaned, re.I):
        return True
    if claim_family and record_family and claim_family == record_family:
        return True
    if gloss_cues_claim(
        cleaned,
        action_glosses,
        exclude_stems=_stem_set(outcome),
    ):
        return True
    return False


def status_conserving_paraphrase_errors(
    *,
    claim: str,
    established_outcome: str,
    established_polarity: str,
    established_action_id: str = "",
    established_action_glosses: Sequence[str] = (),
    expect_bind: bool,
) -> list[str]:
    """Oracle helper: mismatch between expected bind and detector."""
    binds = licensed_action_paraphrase(
        claim,
        outcome=established_outcome,
        polarity=established_polarity,
        action_id=established_action_id,
        action_glosses=established_action_glosses,
    )
    if binds == expect_bind:
        return []
    if expect_bind:
        return [
            "DERIVED_PROPOSITION_STATUS_CONSERVATION: claim should inherit "
            f"established status for {established_action_id or 'action'} / "
            f"{established_outcome!r} but did not bind: {claim!r}"
        ]
    return [
        "DERIVED_PROPOSITION_STATUS_CONSERVATION: claim must not inherit "
        f"status for {established_action_id or 'action'} / "
        f"{established_outcome!r}: {claim!r}"
    ]
