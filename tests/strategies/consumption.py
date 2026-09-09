"""Consumption strategies: chance vs obtained, presentation, cost shape, authority.

Oracles are declared kind fields, not production counters or renderers.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from global_workspace.epistemic_ledger import PropositionRecord
from strategies.worlds import unique_tokens


_CHANCE_KINDS = ("CHANCE", "OBTAINED")
_CHANCE_POLARITIES = ("BENEFICIAL", "ADVERSE")
_CHANCE_MODALITIES = ("POSSIBLE", "PROBABILISTIC", "STIPULATED_CONDITIONAL")
_PRESENTATION_MODALITIES = ("CERTAIN", "POSSIBLE", "PROBABILISTIC")
_FOREGONE_KINDS = ("ACTUAL", "FOREGONE")
_PREFERENCE_KINDS = ("TIED", "PREFERS")
_COST_KINDS = ("BLINDED", "COUNTED")
_UNSETTLED = frozenset({"POSSIBLE", "PROBABILISTIC", "UNKNOWN"})


def _graph(actions: tuple[str, str], rows: list[tuple[str, str, str, str, str]]):
    from global_workspace.semantic_graph import SemanticEdge, SemanticGraph, SemanticNode

    graph = SemanticGraph()
    graph.add_node(SemanticNode(
        "A0", "ACTION", actions[0], attributes={"canonical_action_id": "A0"},
    ))
    graph.add_node(SemanticNode(
        "A1", "ACTION", actions[1], attributes={"canonical_action_id": "A1"},
    ))
    for node_id, action_id, label, polarity, modality in rows:
        graph.add_node(SemanticNode(
            node_id, "CONSEQUENCE", label,
            attributes={
                "polarity": polarity,
                "directness": "DOWNSTREAM",
                "modality": modality,
                "effect_kind": "HEALTH_OUTCOME",
                "scenario_grounded": True,
                "affected_subjects": ["affected constituency"],
            },
        ))
        graph.add_edge(SemanticEdge(action_id, "HAS_CONSEQUENCE", node_id))
    return graph


@dataclass(frozen=True, slots=True)
class ChanceOutcomeCase:
    graph: object
    actions: tuple[str, str]
    label: str
    row_kind: str
    polarity: str
    modality: str

    @property
    def should_count_as_obtained(self) -> bool:
        return self.row_kind == "OBTAINED"

    @property
    def obtained_direction(self) -> str:
        return "IMPROVES" if self.polarity == "BENEFICIAL" else "WORSENS"


@st.composite
def chance_outcome_cases(draw) -> ChanceOutcomeCase:
    label, verb0, verb1 = draw(unique_tokens(3))
    row_kind = draw(st.sampled_from(_CHANCE_KINDS))
    polarity = draw(st.sampled_from(_CHANCE_POLARITIES))
    actions = (f"{verb0} the first option", f"{verb1} the second option")
    modality = (
        "CERTAIN" if row_kind == "OBTAINED"
        else draw(st.sampled_from(_CHANCE_MODALITIES))
    )
    opposite = "ADVERSE" if polarity == "BENEFICIAL" else "BENEFICIAL"
    graph = _graph(actions, [
        ("A0:C", "A0", label, polarity, modality),
        ("A1:C", "A1", label, opposite, "CERTAIN"),
    ])
    return ChanceOutcomeCase(
        graph=graph, actions=actions, label=label, row_kind=row_kind,
        polarity=polarity, modality=modality,
    )


@dataclass(frozen=True, slots=True)
class PresentationModalityCase:
    ledger_rows: tuple[dict, ...]
    claim: str
    modality: str
    world_id: str

    @property
    def should_list_as_established_fact(self) -> bool:
        return self.modality == "CERTAIN"


@st.composite
def presentation_modality_cases(draw) -> PresentationModalityCase:
    party, outcome = draw(unique_tokens(2))
    modality = draw(st.sampled_from(_PRESENTATION_MODALITIES))
    world_id = "PROP:WORLD:E1"
    parts = [outcome, f"affected subject: {party}"]
    if modality != "CERTAIN":
        parts.append(f"modality: {modality}")
    claim = "; ".join(parts)
    row = {
        "proposition_id": world_id,
        "claim": claim,
        "proposition_type": "DESCRIPTIVE",
        "epistemic_status": "ESTABLISHED",
        "epistemic_type": "WORLD_ESTABLISHED",
        "modality": modality,
        "outcome": outcome,
        "party_labels": [party],
    }
    return PresentationModalityCase(
        ledger_rows=(row,),
        claim=claim,
        modality=modality,
        world_id=world_id,
    )


@dataclass(frozen=True, slots=True)
class ForegoneObtainedCase:
    ledger_rows: tuple[dict, ...]
    claim: str
    row_kind: str
    polarity: str
    outcome: str

    @property
    def should_list_as_established_fact(self) -> bool:
        return self.row_kind == "ACTUAL"


@st.composite
def foregone_obtained_cases(draw) -> ForegoneObtainedCase:
    party, outcome = draw(unique_tokens(2))
    row_kind = draw(st.sampled_from(_FOREGONE_KINDS))
    polarity = "FOREGONE" if row_kind == "FOREGONE" else "ADVERSE"
    directness = "FOREGONE" if row_kind == "FOREGONE" else "DOWNSTREAM"
    claim_parts = [outcome, f"affected subject: {party}"]
    if row_kind == "FOREGONE":
        claim_parts.append("foregone opportunity")
    claim = "; ".join(claim_parts)
    row = {
        "proposition_id": "PROP:WORLD:E1",
        "claim": claim,
        "proposition_type": "DESCRIPTIVE",
        "epistemic_status": "ESTABLISHED",
        "epistemic_type": "WORLD_ESTABLISHED",
        "modality": "CERTAIN",
        "outcome": outcome,
        "polarity": polarity,
        "directness": directness,
        "party_labels": [party],
    }
    return ForegoneObtainedCase(
        ledger_rows=(row,),
        claim=claim,
        row_kind=row_kind,
        polarity=polarity,
        outcome=outcome,
    )


@dataclass(frozen=True, slots=True)
class PreferenceSupportCase:
    actions: tuple[str, str]
    ranking_kind: str
    specialist: str

    @property
    def action_scores(self) -> dict[str, float]:
        if self.ranking_kind == "TIED":
            return {self.actions[0]: 0.5, self.actions[1]: 0.5}
        return {self.actions[0]: 0.28, self.actions[1]: 0.72}

    @property
    def recommended_action(self) -> str:
        return "" if self.ranking_kind == "TIED" else self.actions[1]

    @property
    def should_uniquely_support(self) -> bool:
        return self.ranking_kind == "PREFERS"


@st.composite
def preference_support_cases(draw) -> PreferenceSupportCase:
    verb0, verb1 = draw(unique_tokens(2))
    ranking_kind = draw(st.sampled_from(_PREFERENCE_KINDS))
    specialist = draw(st.sampled_from((
        "utilitarian", "care", "rawlsian", "virtue", "deontological",
    )))
    return PreferenceSupportCase(
        actions=(f"{verb0} the first option", f"{verb1} the second option"),
        ranking_kind=ranking_kind,
        specialist=specialist,
    )


@dataclass(frozen=True, slots=True)
class ModalityBlindCostCase:
    graph: object
    actions: tuple[str, str]
    ranking_kind: str

    @property
    def should_treat_as_more_cost(self) -> bool:
        return self.ranking_kind == "COUNTED"


@st.composite
def modality_blind_cost_cases(draw) -> ModalityBlindCostCase:
    harm, extra, verb0, verb1 = draw(unique_tokens(4))
    ranking_kind = draw(st.sampled_from(_COST_KINDS))
    actions = (f"{verb0} the first option", f"{verb1} the second option")
    extra_modality = "CERTAIN" if ranking_kind == "COUNTED" else "PROBABILISTIC"
    graph = _graph(actions, [
        ("A0:C1", "A0", harm, "ADVERSE", extra_modality),
        ("A0:C2", "A0", extra, "ADVERSE", extra_modality),
        ("A1:C", "A1", harm, "ADVERSE", "CERTAIN"),
    ])
    return ModalityBlindCostCase(
        graph=graph, actions=actions, ranking_kind=ranking_kind,
    )


@dataclass(frozen=True, slots=True)
class CompositionalHypothesisCase:
    ledger: dict[str, PropositionRecord]
    actions: tuple[str, str]
    claim: str
    recurrences: int

    @property
    def should_be_governing_eligible(self) -> bool:
        return False


_LIVE_TRACE_LABELS = frozenset({
    "mine", "explosion", "suffocation", "suffocate", "tunnel",
})


@st.composite
def compositional_hypothesis_cases(draw) -> CompositionalHypothesisCase:
    party, outcome, other, verb0, verb1 = draw(
        unique_tokens(5).filter(
            lambda tokens: not any(token in _LIVE_TRACE_LABELS for token in tokens)
        )
    )
    recurrences = draw(st.sampled_from((0, 1, 4)))
    n = draw(st.integers(min_value=2, max_value=40))
    actions = (
        f"{verb0} the option that {outcome} the {party}",
        f"{verb1} the option that spares the {party}",
    )
    world_id = "PROP:WORLD:E1"
    ledger = {
        world_id: PropositionRecord(
            proposition_id=world_id,
            claim=f"{outcome}; affected subject: {party}",
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome=outcome,
            polarity="ADVERSE",
            party_labels=[party],
            quantities=[str(n)],
            effect_kind="HEALTH_OUTCOME",
            modality="CERTAIN",
            directness="DOWNSTREAM",
        ),
    }
    return CompositionalHypothesisCase(
        ledger=ledger,
        actions=actions,
        claim=f"later unverified {other} for the {party} exceeds {n}",
        recurrences=recurrences,
    )


@dataclass(frozen=True, slots=True)
class CompositionalThresholdCase:
    ledger: dict[str, PropositionRecord]
    actions: tuple[str, str]
    claim: str
    table: dict[str, list[dict[str, str]]]
    decision_rule: str


@st.composite
def compositional_threshold_cases(draw) -> CompositionalThresholdCase:
    group, crowd, harm, benefit, verb0, verb1 = draw(unique_tokens(6))
    minted = draw(st.integers(min_value=2, max_value=9))
    actions = (
        f"{verb0} the option that {harm} the {group}",
        f"{verb1} the option that {benefit} the {crowd}",
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
            effect_kind="HEALTH_OUTCOME",
            modality="CERTAIN",
            directness="DOWNSTREAM",
        ),
        "PROP:WORLD:E2": PropositionRecord(
            proposition_id="PROP:WORLD:E2",
            claim=f"{benefit} the {crowd}; polarity: FOREGONE",
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            epistemic_type="WORLD_ESTABLISHED",
            outcome=f"{benefit} the {crowd}",
            polarity="FOREGONE",
            party_labels=[crowd],
            effect_kind="HEALTH_OUTCOME",
            modality="CERTAIN",
            directness="FOREGONE",
        ),
    }
    table = {
        actions[0]: [{
            "outcome": f"{harm} the {group}",
            "direction": "HARM",
            "probability": "UNKNOWN",
            "magnitude": "UNKNOWN",
            "support": "STATED",
        }],
        actions[1]: [{
            "outcome": f"{benefit} the {crowd}",
            "valuation_reason": "foregone dual treated as expected harm",
            "direction": "HARM",
            "probability": "UNKNOWN",
            "magnitude": "UNKNOWN",
            "support": "STATED",
        }],
    }
    return CompositionalThresholdCase(
        ledger=ledger,
        actions=actions,
        claim=f"later unverified {benefit} exceeds {minted} percent",
        table=table,
        decision_rule=(
            f"prefer the second action unless lethal probability exceeds "
            f"{minted} percent"
        ),
    )
