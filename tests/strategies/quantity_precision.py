"""QUANTITY_PRECISION_NON_ESCALATION Hypothesis cases with declared oracles.

Oracle: ``expect_escalation``. Production
``quantity_precision_escalation_errors`` is only asked whether it agrees.
Do not import production collective-band regexes here to invent the oracle —
the strategy writes vague source + claim pairs and the expected verdict.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st


_VAGUE_SOURCES = (
    "A cyberattack risks thousands of lives through infrastructure failure.",
    "The riot predictably kills hundreds of residents.",
    "Flooding endangers dozens of workers overnight.",
    "The breach erases millions of patient records.",
    "Several thousand residents face catastrophic loss of life.",
    "More than 500 residents are at risk; thousands may die.",
)

_ESCALATING_CLAIMS = (
    "the purge would directly save about 10,000 people",
    "roughly 10000 deaths follow from refraining",
    "~10,000 lives are certainly saved",
    "exactly 5,000 residents die",
    "around 250 people are harmed",  # under hundreds band when source has hundreds
    "about 50 workers die",  # under dozens when source has dozens
    "~8 million life-years preserved (100 000 residents × 80-year average)",
    "100,000 residents times 80-year remaining life",
    "approximately 5–10 000 city residents would be saved",
)


@dataclass(frozen=True, slots=True)
class QuantityPrecisionCase:
    source: str
    claim: str
    expect_escalation: bool
    issue_code: str = "QUANTITY_PRECISION_ESCALATION"
    repair_stage: str = "epistemic"
    allowed_ops: tuple[str, ...] = ("REJECT_PRECISION_ESCALATION",)
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "INVENT_EXACT_COUNT")


@st.composite
def quantity_precision_cases(draw) -> QuantityPrecisionCase:
    """Vague source + claim. expect_escalation is written by the mode."""
    escalate = draw(st.booleans())
    source = draw(st.sampled_from(_VAGUE_SOURCES))
    if escalate:
        # Pair claim to a band the source actually covers.
        folded = source.casefold()
        if "thousand" in folded:
            claim = draw(st.sampled_from([
                "the purge would directly save about 10,000 people",
                "roughly 10000 deaths follow from refraining",
                "~10,000 lives are certainly saved",
                "exactly 5,000 residents die",
                "~8 million life-years preserved (100 000 residents × 80-year average)",
                "approximately 5–10 000 city residents would be saved",
            ]))
        elif "hundred" in folded:
            claim = "around 250 people are harmed"
        elif "dozen" in folded:
            claim = "about 50 workers die"
        else:
            claim = "exactly 2,000,000 records are erased"
        return QuantityPrecisionCase(
            source=source, claim=claim, expect_escalation=True,
        )
    folded = source.casefold()
    if "thousand" in folded:
        claim = draw(st.sampled_from((
            "thousands of lives are at stake",
            "the admitted quantity remains thousands",
            "several thousand residents remain at risk",
        )))
    elif "hundred" in folded:
        claim = "hundreds of residents remain endangered"
    elif "dozen" in folded:
        claim = "dozens of workers face the flood"
    else:
        claim = "millions of records would be erased"
    return QuantityPrecisionCase(
        source=source, claim=claim, expect_escalation=False,
    )
