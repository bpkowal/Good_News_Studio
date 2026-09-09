"""Specialist premise strategies against a declared ledger atom.

The oracle is claim_kind plus row modality, not
certain_records_contradicted_by. Templates are written so REOPEN uses
the production reopen cues, RESTATE copies the world atom, and AGREE
does neither.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from global_workspace.epistemic_ledger import (
    PropositionRecord,
    _OUTCOME_FAMILIES,
    _STRENGTHENING_WORDS,
)
from strategies.worlds import unique_tokens


_CLAIM_KINDS = ("REOPEN", "RESTATE", "AGREE")
_ROW_MODALITIES = ("CERTAIN", "POSSIBLE")
_SPECIALISTS = ("virtue", "care", "deontological", "rawlsian")


@dataclass(frozen=True, slots=True)
class CertainReopenCase:
    """One admitted world atom and one specialist claim about it."""

    ledger: dict[str, PropositionRecord]
    world_id: str
    actions: tuple[str, str]
    specialist: str
    claim: str
    claim_kind: str
    row_modality: str
    party: str
    outcome: str

    @property
    def should_quarantine(self) -> bool:
        return self.row_modality == "CERTAIN" and self.claim_kind == "REOPEN"


def _world_claim(outcome: str, party: str, modality: str) -> str:
    parts = [outcome, f"affected subject: {party}"]
    if modality != "CERTAIN":
        parts.append(f"modality: {modality}")
    return "; ".join(parts)


def _claim_text(kind: str, *, outcome: str, party: str, world_claim: str) -> str:
    if kind == "REOPEN":
        return f"{outcome} affecting the {party} could be stopped short of happening"
    if kind == "AGREE":
        return f"{outcome} for the {party} remains required by the admitted action"
    return world_claim


@st.composite
def certain_reopen_cases(draw) -> CertainReopenCase:
    party, outcome, other_party, other_outcome, verb0, verb1 = draw(unique_tokens(6))
    claim_kind = draw(st.sampled_from(_CLAIM_KINDS))
    row_modality = draw(st.sampled_from(_ROW_MODALITIES))
    specialist = draw(st.sampled_from(_SPECIALISTS))
    world_id = "PROP:WORLD:E1"
    other_id = "PROP:WORLD:E2"
    world_claim = _world_claim(outcome, party, row_modality)
    ledger = {
        world_id: PropositionRecord(
            proposition_id=world_id,
            claim=world_claim,
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome=outcome,
            polarity="ADVERSE",
            party_labels=[party],
            effect_kind="HEALTH_OUTCOME",
            modality=row_modality,
            directness="DOWNSTREAM",
        ),
        other_id: PropositionRecord(
            proposition_id=other_id,
            claim=_world_claim(other_outcome, other_party, "CERTAIN"),
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome=other_outcome,
            polarity="ADVERSE",
            party_labels=[other_party],
            effect_kind="WELFARE_OUTCOME",
            modality="CERTAIN",
            directness="DOWNSTREAM",
        ),
    }
    actions = (
        f"{verb0} the option that {outcome} the {party}",
        f"{verb1} the option that spares the {party}",
    )
    return CertainReopenCase(
        ledger=ledger,
        world_id=world_id,
        actions=actions,
        specialist=specialist,
        claim=_claim_text(
            claim_kind, outcome=outcome, party=party, world_claim=world_claim,
        ),
        claim_kind=claim_kind,
        row_modality=row_modality,
        party=party,
        outcome=outcome,
    )


_UNSETTLED_MODALITIES = ("POSSIBLE", "UNKNOWN")
_MODALITY_ROWS = ("POSSIBLE", "UNKNOWN", "CERTAIN")
_CLAIM_SHAPES = ("UNHEDGED", "HEDGED", "CANONICAL")
_OUTCOME_POLARITY = (
    ("survive", "BENEFICIAL"),
    ("injured", "ADVERSE"),
)
_HEDGE_TOKENS = frozenset({
    "possible", "possibility", "chance", "might", "could",
    "uncertain", "unknown", "survive", "injured",
})


@dataclass(frozen=True, slots=True)
class UnsettledModalityCase:
    """One world atom and a restatement that is or is not hedged."""

    ledger: dict[str, PropositionRecord]
    world_id: str
    claim: str
    claim_kind: str
    row_modality: str
    party: str
    outcome: str

    @property
    def should_bind(self) -> bool:
        if (
            self.row_modality in _UNSETTLED_MODALITIES
            and self.claim_kind == "UNHEDGED"
        ):
            return False
        return True


def _modality_claim(kind: str, *, outcome: str, party: str, world_claim: str) -> str:
    if kind == "UNHEDGED":
        return outcome
    if kind == "HEDGED":
        return f"{party} might {outcome}"
    return world_claim


@st.composite
def unsettled_modality_cases(draw) -> UnsettledModalityCase:
    party, other = draw(
        unique_tokens(2).filter(
            lambda tokens: not any(token in _HEDGE_TOKENS for token in tokens)
        )
    )
    outcome, polarity = draw(st.sampled_from(_OUTCOME_POLARITY))
    claim_kind = draw(st.sampled_from(_CLAIM_SHAPES))
    row_modality = draw(st.sampled_from(_MODALITY_ROWS))
    world_id = "PROP:WORLD:E1"
    world_claim = _world_claim(outcome, party, row_modality)
    ledger = {
        world_id: PropositionRecord(
            proposition_id=world_id,
            claim=world_claim,
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome=outcome,
            polarity=polarity,
            party_labels=[party],
            effect_kind="HEALTH_OUTCOME",
            modality=row_modality,
            directness="DOWNSTREAM",
        ),
    }
    return UnsettledModalityCase(
        ledger=ledger,
        world_id=world_id,
        claim=_modality_claim(
            claim_kind, outcome=outcome, party=party, world_claim=world_claim,
        ),
        claim_kind=claim_kind,
        row_modality=row_modality,
        party=party,
        outcome=outcome,
    )


_RELATION_TEMPLATES = (
    "{verb} the {facility} is doing harm to the {group}",
    "{verb} the {facility} is allowing harm to the {group}",
    "duty of care to the {group} remains the governing relation",
    "the {group} is least-advantaged under the admitted options",
    "perfect duty not to harm the {group} remains prior",
)
_CLAIM_ROLES = ("NORMATIVE", "EMPIRICAL")


@dataclass(frozen=True, slots=True)
class FrameworkDerivedCase:
    """A specialist premise that is either a framework relation or empirical."""

    ledger: dict[str, PropositionRecord]
    world_id: str
    actions: tuple[str, str]
    specialist: str
    claim: str
    claim_role: str

    @property
    def is_framework_derived(self) -> bool:
        return self.claim_role == "NORMATIVE"


@st.composite
def framework_derived_cases(draw) -> FrameworkDerivedCase:
    group, facility, verb, outcome, verb0, verb1 = draw(unique_tokens(6))
    claim_role = draw(st.sampled_from(_CLAIM_ROLES))
    world_id = "PROP:WORLD:E1"
    world_claim = _world_claim(outcome, group, "CERTAIN")
    ledger = {
        world_id: PropositionRecord(
            proposition_id=world_id,
            claim=world_claim,
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome=outcome,
            polarity="ADVERSE",
            party_labels=[group],
            effect_kind="HEALTH_OUTCOME",
            modality="CERTAIN",
            directness="DOWNSTREAM",
        ),
    }
    if claim_role == "NORMATIVE":
        template = draw(st.sampled_from(_RELATION_TEMPLATES))
        claim = template.format(verb=verb, facility=facility, group=group)
        specialist = "deontological"
    else:
        claim = f"{outcome} for the {group} remains an admitted effect"
        specialist = draw(st.sampled_from(_SPECIALISTS))
    actions = (
        f"{verb0} the {facility}, affecting the {group}",
        f"{verb1} the {facility}, sparing the {group}",
    )
    return FrameworkDerivedCase(
        ledger=ledger,
        world_id=world_id,
        actions=actions,
        specialist=specialist,
        claim=claim,
        claim_role=claim_role,
    )


_IDENTITY_KINDS = ("CANONICAL", "STRONGER", "NOVEL")
_IDENTITY_OUTCOME = "injured"
_IDENTITY_BLOCKED = (
    _STRENGTHENING_WORDS
    | _HEDGE_TOKENS
    | frozenset().union(*_OUTCOME_FAMILIES)
)


@dataclass(frozen=True, slots=True)
class PropositionIdentityCase:
    """One admitted atom and a claim that is or is not that atom."""

    ledger: dict[str, PropositionRecord]
    world_id: str
    actions: tuple[str, str]
    specialist: str
    claim: str
    claim_kind: str
    party: str
    outcome: str

    @property
    def should_bind(self) -> bool:
        return self.claim_kind == "CANONICAL"


def _identity_claim(kind: str, *, outcome: str, party: str, world_claim: str,
                    other_outcome: str, other_party: str) -> str:
    if kind == "STRONGER":
        return f"{outcome} is guaranteed for the {party}"
    if kind == "NOVEL":
        return f"{other_outcome} for the {other_party}"
    return world_claim


@st.composite
def proposition_identity_cases(draw) -> PropositionIdentityCase:
    party, other_party, other_outcome, verb0, verb1 = draw(
        unique_tokens(5).filter(
            lambda tokens: not any(token in _IDENTITY_BLOCKED for token in tokens)
        )
    )
    claim_kind = draw(st.sampled_from(_IDENTITY_KINDS))
    specialist = draw(st.sampled_from(_SPECIALISTS))
    world_id = "PROP:WORLD:E1"
    world_claim = _world_claim(_IDENTITY_OUTCOME, party, "CERTAIN")
    ledger = {
        world_id: PropositionRecord(
            proposition_id=world_id,
            claim=world_claim,
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome=_IDENTITY_OUTCOME,
            polarity="ADVERSE",
            party_labels=[party],
            effect_kind="HEALTH_OUTCOME",
            modality="CERTAIN",
            directness="DOWNSTREAM",
        ),
    }
    actions = (
        f"{verb0} the option that {_IDENTITY_OUTCOME} the {party}",
        f"{verb1} the option that spares the {party}",
    )
    return PropositionIdentityCase(
        ledger=ledger,
        world_id=world_id,
        actions=actions,
        specialist=specialist,
        claim=_identity_claim(
            claim_kind,
            outcome=_IDENTITY_OUTCOME,
            party=party,
            world_claim=world_claim,
            other_outcome=other_outcome,
            other_party=other_party,
        ),
        claim_kind=claim_kind,
        party=party,
        outcome=_IDENTITY_OUTCOME,
    )
