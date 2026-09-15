"""Negation-scope sibling Hypothesis cases with declared oracles.

The oracle is NegationScopeCase.preserves_siblings (sibling polarity and
modality stay put when another conjunct is negated). Hypothesis asks the
Parliament harness oracle whether keep/flip worlds agree with that flag.

Corruption mutations expand beyond polarity/modality flips to neutral and
beneficial soft-leak shapes without promoting a live admit gate.
"""
from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from global_workspace.world_state import (
    ScenarioWorldModel,
    SourceRef,
    WorldAction,
    WorldEffect,
    WorldParty,
)


# Independent phrase bank. expect_role_field is declared only for the
# patient+dies pattern production currently attaches as harmed.
_PATIENT_LABELS = ("patient", "resident", "worker")
_NEGATED_ROWS = (
    ("receives no treatment", "treatment"),
    ("gets no care", "care"),
    ("receives no aid", "aid"),
    ("is given no medicine", "medicine"),
)
_HARM_ROWS = (
    ("dies", "ADVERSE", "CERTAIN"),
    ("perishes", "ADVERSE", "CERTAIN"),
    ("is killed", "ADVERSE", "CERTAIN"),
)
_CORRUPTION_MODES = (
    "flip_polarity",
    "flip_modality",
    "flip_both",
    "neutralize",
    "beneficial_possible",
)


@dataclass(frozen=True, slots=True)
class NegationScopeCase:
    """Negated conjunct + harm sibling. preserves_siblings is declared."""

    source: str
    action_id: str
    negated_effect_id: str
    sibling_effect_id: str
    sibling_polarity: str
    sibling_modality: str
    preserves_siblings: bool
    corruption_mode: str
    patient_label: str
    expect_role_field: str | None
    world: ScenarioWorldModel
    issue_code: str = "NEGATION_SCOPE_LEAK"
    repair_stage: str = "grounding"
    allowed_ops: tuple[str, ...] = ("REPAIR_GROUNDING", "CHALLENGE_PREMISE")
    forbidden_ops: tuple[str, ...] = ("MORE_DEBATE", "VIRTUE_REFRAME")

    @property
    def siblings(self) -> dict[str, dict[str, str]]:
        return {
            self.sibling_effect_id: {
                "polarity": self.sibling_polarity,
                "modality": self.sibling_modality,
            }
        }


def _corrupt_sibling(
    *,
    mode: str,
    polarity: str,
    modality: str,
) -> tuple[str, str]:
    if mode == "flip_polarity":
        return ("BENEFICIAL" if polarity == "ADVERSE" else "ADVERSE", modality)
    if mode == "flip_modality":
        return (polarity, "POSSIBLE" if modality == "CERTAIN" else "CERTAIN")
    if mode == "neutralize":
        return ("NEUTRAL", modality)
    if mode == "beneficial_possible":
        return ("BENEFICIAL", "POSSIBLE")
    # flip_both
    return (
        "BENEFICIAL" if polarity == "ADVERSE" else "ADVERSE",
        "POSSIBLE" if modality == "CERTAIN" else "CERTAIN",
    )


def _negation_world(
    *,
    source: str,
    patient: str,
    treatment_noun: str,
    negated_outcome: str,
    harm_outcome: str,
    sibling_polarity: str,
    sibling_modality: str,
) -> ScenarioWorldModel:
    ref = (SourceRef("C0", source),)
    return ScenarioWorldModel(
        schema_version="1.2",
        parties=(
            WorldParty("P_patient", patient, "PERSON", ref),
            WorldParty("P_actor", "coordinator", "PERSON", ref),
        ),
        actions=(
            WorldAction(
                "A0", f"withhold {treatment_noun}",
                "P_actor", ("P_patient",), ("E_neg", "E_harm"), ref,
            ),
        ),
        effects=(
            WorldEffect(
                "E_neg", "A0", "P_patient", negated_outcome, "STATE_CHANGE",
                "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                provenance=ref,
                source_proposition=f"The {patient} {negated_outcome}",
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E_harm", "A0", "P_patient", harm_outcome, "STATE_CHANGE",
                sibling_polarity, "DOWNSTREAM", sibling_modality, "HEALTH_OUTCOME",
                provenance=ref,
                source_proposition=f"The {patient} {harm_outcome}",
                derivation_operation="DIRECT_COPY",
            ),
        ),
    )


@st.composite
def negation_scope_cases(draw) -> NegationScopeCase:
    """Keep vs mutate sibling polarity/modality under coordinated negation."""
    patient = draw(st.sampled_from(_PATIENT_LABELS))
    negated_outcome, treatment_noun = draw(st.sampled_from(_NEGATED_ROWS))
    harm_outcome, true_polarity, true_modality = draw(st.sampled_from(_HARM_ROWS))
    preserves = draw(st.booleans())
    mode = draw(st.sampled_from(_CORRUPTION_MODES))
    if preserves:
        sibling_polarity, sibling_modality = true_polarity, true_modality
        corruption_mode = "none"
    else:
        sibling_polarity, sibling_modality = _corrupt_sibling(
            mode=mode,
            polarity=true_polarity,
            modality=true_modality,
        )
        corruption_mode = mode
    source = f"The {patient} {negated_outcome} and {harm_outcome}."
    expect_role = None
    if patient == "patient" and harm_outcome == "dies":
        expect_role = "harmed"
    return NegationScopeCase(
        source=source,
        action_id="A0",
        negated_effect_id="E_neg",
        sibling_effect_id="E_harm",
        sibling_polarity=true_polarity,
        sibling_modality=true_modality,
        preserves_siblings=preserves,
        corruption_mode=corruption_mode,
        patient_label=patient,
        expect_role_field=expect_role,
        world=_negation_world(
            source=source,
            patient=patient,
            treatment_noun=treatment_noun,
            negated_outcome=negated_outcome,
            harm_outcome=harm_outcome,
            sibling_polarity=sibling_polarity,
            sibling_modality=sibling_modality,
        ),
    )
