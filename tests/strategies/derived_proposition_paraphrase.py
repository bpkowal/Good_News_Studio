"""DERIVED_PROPOSITION_STATUS_CONSERVATION Hypothesis cases.

Oracle: ``expect_bind``. Licensed action-conditioned paraphrases of an
established effect must inherit status; wrong-action / wrong-polarity wording
must not. Production ``licensed_action_paraphrase`` / ledger resolve only asked
whether they agree — do not invent the oracle from production regexes alone.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st


_REFRAIN_GLOSSES = (
    ("refrain from executing the emergency purge", "refrain"),
    ("withhold the purge", "do not execute the purge"),
    ("omit the emergency purge", "refrain"),
)

_EXECUTE_GLOSSES = (
    ("execute the emergency purge", "deploy the purge"),
    ("trigger the purge now", "execute the purge"),
)

_LOSS_OUTCOMES = (
    "suffers catastrophic loss of life",
    "catastrophic loss of life",
    "thousands die",
)

_SURVIVAL_OUTCOMES = (
    "immediate survival",
    "city residents survive",
)

_REFRAIN_CLAIMS = (
    "Catastrophic loss of life occurs if no purge.",
    "Catastrophic loss of life occurs if purge not executed.",
    "Refraining causes catastrophic loss of life.",
    "If the purge is not executed, catastrophic loss of life occurs.",
    "Without the purge, catastrophic loss of life follows.",
)

_EXECUTE_CLAIMS = (
    "Catastrophic loss of life occurs if the purge is executed.",
    "Executing the purge causes catastrophic loss of life.",
    "If the emergency purge is deployed, catastrophic loss of life occurs.",
)

_WRONG_POLARITY_CLAIMS = (
    "Immediate survival occurs if no purge.",
    "City residents survive if purge not executed.",
)


@dataclass(frozen=True, slots=True)
class DerivedPropositionParaphraseCase:
    claim: str
    outcome: str
    polarity: str
    action_id: str
    action_glosses: tuple[str, ...]
    expect_bind: bool
    issue_code: str = "DERIVED_PROPOSITION_STATUS_CONSERVATION"
    repair_stage: str = "epistemic"
    allowed_ops: tuple[str, ...] = ("BIND_PARAPHRASE",)
    forbidden_ops: tuple[str, ...] = ("MINT_HYPOTHESIS", "FREE_SIMILARITY")


@st.composite
def derived_proposition_paraphrase_cases(draw) -> DerivedPropositionParaphraseCase:
    """Licensed paraphrase nets ± negatives with a declared bind oracle."""
    mode = draw(st.sampled_from((
        "refrain_positive",
        "execute_positive",
        "wrong_action",
        "wrong_polarity",
    )))
    if mode == "refrain_positive":
        glosses = draw(st.sampled_from(_REFRAIN_GLOSSES))
        claim = draw(st.sampled_from(_REFRAIN_CLAIMS))
        outcome = draw(st.sampled_from(_LOSS_OUTCOMES))
        return DerivedPropositionParaphraseCase(
            claim=claim,
            outcome=outcome,
            polarity="ADVERSE",
            action_id="A1",
            action_glosses=glosses,
            expect_bind=True,
        )
    if mode == "execute_positive":
        glosses = draw(st.sampled_from(_EXECUTE_GLOSSES))
        claim = draw(st.sampled_from(_EXECUTE_CLAIMS))
        outcome = draw(st.sampled_from(_LOSS_OUTCOMES))
        return DerivedPropositionParaphraseCase(
            claim=claim,
            outcome=outcome,
            polarity="ADVERSE",
            action_id="A0",
            action_glosses=glosses,
            expect_bind=True,
        )
    if mode == "wrong_action":
        # Established refrain/loss; claim cues execute of the same intervention.
        glosses = draw(st.sampled_from(_REFRAIN_GLOSSES))
        claim = draw(st.sampled_from(_EXECUTE_CLAIMS))
        outcome = draw(st.sampled_from(_LOSS_OUTCOMES))
        return DerivedPropositionParaphraseCase(
            claim=claim,
            outcome=outcome,
            polarity="ADVERSE",
            action_id="A1",
            action_glosses=glosses,
            expect_bind=False,
            allowed_ops=("REJECT_WRONG_ACTION",),
            forbidden_ops=("BIND_PARAPHRASE", "MINT_HYPOTHESIS"),
        )
    glosses = draw(st.sampled_from(_REFRAIN_GLOSSES))
    claim = draw(st.sampled_from(_WRONG_POLARITY_CLAIMS))
    outcome = draw(st.sampled_from(_LOSS_OUTCOMES))
    return DerivedPropositionParaphraseCase(
        claim=claim,
        outcome=outcome,
        polarity="ADVERSE",
        action_id="A1",
        action_glosses=glosses,
        expect_bind=False,
        allowed_ops=("REJECT_WRONG_POLARITY",),
        forbidden_ops=("BIND_PARAPHRASE", "MINT_HYPOTHESIS"),
    )
