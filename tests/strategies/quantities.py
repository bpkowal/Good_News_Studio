"""Util ranking strategies for unadmitted magnitude.

The oracle is ranking_kind, not bind_unadmitted_magnitude_ranking.
MINTED and MORTALITY should revoke. ORDINAL and ADMITTED should stand.
Enforcement is Util-only; these cases are always utilitarian.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from global_workspace.epistemic_ledger import PropositionRecord
from strategies.worlds import unique_tokens


_KINDS = ("MINTED", "ORDINAL", "ADMITTED", "MORTALITY")
_DEATH_TOKENS = frozenset({
    "death", "deaths", "drown", "drowns", "drowned", "lethal",
    "perish", "perished", "die", "died", "dies",
})


@dataclass(frozen=True, slots=True)
class MagnitudeRankingCase:
    ledger: dict[str, PropositionRecord]
    actions: tuple[str, str]
    decision_rule: str
    table: dict[str, list[dict[str, str]]]
    ranking_kind: str

    @property
    def should_revoke(self) -> bool:
        return self.ranking_kind in {"MINTED", "MORTALITY"}


def _labels():
    return unique_tokens(6).filter(
        lambda tokens: not any(token in _DEATH_TOKENS for token in tokens)
    )


@st.composite
def magnitude_ranking_cases(draw) -> MagnitudeRankingCase:
    group, crowd, harm, other, verb0, verb1 = draw(_labels())
    ranking_kind = draw(st.sampled_from(_KINDS))
    actions = (
        f"{verb0} the option that {harm} the {group}",
        f"{verb1} the option that {other} the {crowd}",
    )
    admitted = draw(st.integers(min_value=10, max_value=40))
    minted = draw(st.integers(min_value=2, max_value=9))
    ledger: dict[str, PropositionRecord] = {}
    table: dict[str, list[dict[str, str]]] = {}
    if ranking_kind == "MINTED":
        rule = (
            f"prefer the second action unless lethal probability exceeds "
            f"{minted} percent"
        )
    elif ranking_kind == "ADMITTED":
        ledger["PROP:WORLD:E1"] = PropositionRecord(
            proposition_id="PROP:WORLD:E1",
            claim=f"{harm} the {group}; affected subject: {group}",
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome=f"{harm} the {group}",
            party_labels=[group],
            quantities=[str(admitted)],
            effect_kind="HEALTH_OUTCOME",
            modality="CERTAIN",
            directness="DOWNSTREAM",
        )
        rule = (
            f"prefer the second action unless the admitted count "
            f"{admitted} is exceeded"
        )
    elif ranking_kind == "MORTALITY":
        table = {
            actions[0]: [{
                "outcome": f"{harm} the {group}",
                "valuation_reason": "admitted burden",
                "direction": "HARM",
                "probability": "UNKNOWN",
                "magnitude": "UNKNOWN",
                "support": "STATED",
            }],
            actions[1]: [{
                "outcome": f"{other} the {crowd}",
                "valuation_reason": "potential mass fatality among the crowd",
                "direction": "HARM",
                "probability": "UNKNOWN",
                "magnitude": "UNKNOWN",
                "support": "STATED",
            }],
        }
        rule = "prefer the second action on aggregate welfare"
    else:
        rule = (
            f"prefer the second action because certain harm to the {group} "
            f"outranks certain harm to the {crowd}"
        )
    return MagnitudeRankingCase(
        ledger=ledger,
        actions=actions,
        decision_rule=rule,
        table=table,
        ranking_kind=ranking_kind,
    )


_CLOSED_KINDS = ("NETS", "OPEN")
CLOSED_WORLD_RANKING_NOTE = (
    "Closed-world ranking uses admitted consequences only; unestablished "
    "hypotheses remain reversal boundaries."
)


@dataclass(frozen=True, slots=True)
class ClosedWorldHypothesisCase:
    """Util scores that a hypothesis must not displace when nets exist."""

    ledger: dict[str, PropositionRecord]
    actions: tuple[str, str]
    claim: str
    table: dict[str, list[dict[str, str]]]
    ranking_kind: str

    @property
    def should_restore(self) -> bool:
        return self.ranking_kind == "NETS"


def _closed_row(outcome: str, magnitude: str) -> dict[str, str]:
    return {
        "outcome": outcome,
        "direction": "HARM",
        "probability": "100%",
        "magnitude": magnitude,
        "support": "STATED",
    }


@st.composite
def closed_world_hypothesis_cases(draw) -> ClosedWorldHypothesisCase:
    group, crowd, harm, other, verb0, verb1 = draw(_labels())
    ranking_kind = draw(st.sampled_from(_CLOSED_KINDS))
    actions = (
        f"{verb0} the option that {harm} the {group}",
        f"{verb1} the option that {other} the {crowd}",
    )
    ledger = {
        "PROP:WORLD:E1": PropositionRecord(
            proposition_id="PROP:WORLD:E1",
            claim=f"{harm} the {group}; affected subject: {group}",
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome=f"{harm} the {group}",
            polarity="ADVERSE",
            party_labels=[group],
            quantities=["1"],
            effect_kind="HEALTH_OUTCOME",
            modality="CERTAIN",
            directness="DOWNSTREAM",
        ),
    }
    if ranking_kind == "NETS":
        table = {
            actions[0]: [_closed_row(f"{harm} the {group}", "1")],
            actions[1]: [_closed_row(f"{other} the {crowd}", "40")],
        }
    else:
        table = {
            actions[0]: [_closed_row(f"{harm} the {group}", "UNKNOWN")],
            actions[1]: [_closed_row(f"{other} the {crowd}", "UNKNOWN")],
        }
    return ClosedWorldHypothesisCase(
        ledger=ledger,
        actions=actions,
        claim=f"later unverified {other} for the {crowd} exceeds the admitted margin",
        table=table,
        ranking_kind=ranking_kind,
    )
