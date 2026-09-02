"""Typed world-model admission must reject the two representation slips.

These tests use a dispatcher/dam fixture so the rules stay domain-neutral:
CERTAIN effects cannot carry conditions, and a foregone opportunity is not
action-relative harm. Quantities may only copy explicit source spans from
provenance, never from the generated outcome alone.
"""
from __future__ import annotations

import unittest

from global_workspace.scenario_semantics import (
    compile_scenario_graph,
    project_grounded_action_effects,
)
from global_workspace.world_state import (
    CausalLink,
    SourceRef,
    WorldAction,
    WorldCondition,
    WorldEffect,
    WorldParty,
    ScenarioWorldModel,
    classify_clause_role,
    explicit_quantity_spans,
    parse_world_model,
    validate_world_model,
)


REF = (SourceRef("C0", "A dispatcher must send one repair crew to Site A or Site B."),)
SCALE_REF = (SourceRef(
    "C1",
    "If the engineer is reached, dam repair would spare tens of thousands of "
    "downstream residents.",
),)
DOZEN_REF = (SourceRef(
    "C2",
    "Site B holds a trapped family of a dozen residents facing acute risk.",
),)
COMPARE_REF = (SourceRef(
    "C3",
    "The system must choose between securing immediate relief for a few or "
    "producing vital future resources for many.",
),)
ACTION_A0_REF = (SourceRef(
    "A0",
    "send the crew to Site A, reach the engineer, and leave the dozen trapped "
    "residents without urgent relief",
),)
CLAUSES = [
    {"clause_id": "C0", "text": REF[0].excerpt},
    {"clause_id": "C1", "text": SCALE_REF[0].excerpt},
    {"clause_id": "C2", "text": DOZEN_REF[0].excerpt},
    {"clause_id": "C3", "text": COMPARE_REF[0].excerpt},
    {"clause_id": "C4", "text": "Option 2 sends the crew to Site B and forgoes the dam repair."},
]


def _effect(
    effect_id: str,
    action_id: str,
    party_id: str,
    *,
    outcome: str = "survives",
    relation: str = "SURVIVES",
    polarity: str = "BENEFICIAL",
    directness: str = "DIRECT",
    modality: str = "CERTAIN",
    condition_ids: tuple[str, ...] = (),
    quantities: tuple[str, ...] = (),
    provenance: tuple[SourceRef, ...] = REF,
) -> WorldEffect:
    return WorldEffect(
        effect_id=effect_id,
        action_id=action_id,
        party_id=party_id,
        outcome=outcome,
        relation=relation,
        polarity=polarity,
        directness=directness,
        modality=modality,
        condition_ids=condition_ids,
        quantities=quantities,
        provenance=provenance,
    )


def _model(
    effects: tuple[WorldEffect, ...],
    *,
    conditions: tuple[WorldCondition, ...] = (),
    links: tuple[CausalLink, ...] = (),
) -> ScenarioWorldModel:
    parties = (
        WorldParty("P0", "dispatcher", "SYSTEM", REF),
        WorldParty("P1", "engineer", "PERSON", REF),
        WorldParty("P2", "trapped family", "GROUP", REF),
        WorldParty("P3", "downstream residents", "POPULATION", SCALE_REF),
    )
    actions = (
        WorldAction(
            "A0", "send crew to Site A", "P0", ("P1",),
            tuple(item.effect_id for item in effects if item.action_id == "A0"),
            REF,
        ),
        WorldAction(
            "A1", "send crew to Site B", "P0", ("P2",),
            tuple(item.effect_id for item in effects if item.action_id == "A1"),
            REF,
        ),
    )
    return ScenarioWorldModel(
        parties=parties, actions=actions, effects=effects,
        conditions=conditions, causal_links=links,
    )


def _valid_effects() -> tuple[WorldEffect, ...]:
    return (
        _effect("E1", "A0", "P1", outcome="engineer is reached and lives"),
        _effect("E2", "A1", "P2", outcome="trapped family lives"),
        _effect(
            "E3", "A1", "P3",
            outcome="dam repair is not undertaken",
            relation="FOREGONE_BENEFIT",
            polarity="FOREGONE",
            directness="FOREGONE",
        ),
        _effect(
            "E4", "A0", "P3",
            outcome="dam repair would spare tens of thousands of residents",
            relation="DOWNSTREAM_BENEFIT",
            polarity="BENEFICIAL",
            directness="DOWNSTREAM",
            modality="STIPULATED_CONDITIONAL",
            condition_ids=("COND1",),
            quantities=("tens of thousands",),
            provenance=SCALE_REF,
        ),
    )


def _valid_model() -> ScenarioWorldModel:
    return _model(
        _valid_effects(),
        conditions=(
            WorldCondition("COND1", "engineer must be reached in time", "UNKNOWN", "MATERIAL", SCALE_REF),
        ),
    )


class ExplicitQuantitySpanTests(unittest.TestCase):
    def test_copies_scale_phrases_and_numerals_not_ages_or_inventions(self):
        text = (
            "An 80-year-old engineer, if reached, can spare tens of thousands "
            "of residents; a family of 3 dies in 40 minutes at a 40% risk."
        )
        self.assertEqual(
            explicit_quantity_spans(text),
            ("tens of thousands", "3", "40 minutes", "40%"),
        )
        self.assertNotIn("80", explicit_quantity_spans(text))
        self.assertEqual(explicit_quantity_spans("later expected value in QALYs"), ())

    def test_dozen_is_a_quantity_but_few_many_and_bare_months_are_not(self):
        self.assertEqual(explicit_quantity_spans("a dozen residents"), ("dozen",))
        self.assertEqual(explicit_quantity_spans("relief for a few or many"), ())
        self.assertEqual(explicit_quantity_spans("sustains residents for months"), ())
        self.assertEqual(
            explicit_quantity_spans("sustains hundreds of residents for 3 months"),
            ("hundreds", "3 months"),
        )


class ClauseRoleTests(unittest.TestCase):
    def test_comparison_and_fact_roles(self):
        self.assertEqual(classify_clause_role(COMPARE_REF[0].excerpt), "COMPARISON")
        self.assertEqual(classify_clause_role(DOZEN_REF[0].excerpt), "FACT")
        self.assertEqual(classify_clause_role("Which pipeline should receive the supply?"), "INTERROGATIVE")


class WorldModelValidationTests(unittest.TestCase):
    def test_valid_model_commits(self):
        errors, contradictions = validate_world_model(
            _valid_model(), action_ids=["A0", "A1"],
        )
        self.assertEqual(errors, [])
        self.assertEqual(contradictions, [])

    def test_certain_effect_cannot_list_conditions(self):
        model = _model((
            _effect("E1", "A0", "P1", condition_ids=("COND1",)),
            _effect("E2", "A1", "P2"),
        ), conditions=(
            WorldCondition("COND1", "engineer must survive", "UNKNOWN", "MATERIAL", REF),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(any("E1" in item and "CERTAIN" in item for item in errors), errors)

    def test_non_certain_effect_still_requires_a_condition(self):
        model = _model((
            _effect("E1", "A0", "P1"),
            _effect(
                "E2", "A1", "P3",
                outcome="residents are spared",
                relation="DOWNSTREAM_BENEFIT",
                polarity="BENEFICIAL",
                directness="DOWNSTREAM",
                modality="STIPULATED_CONDITIONAL",
            ),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(any("E2" in item and "condition" in item for item in errors), errors)

    def test_foregone_effect_cannot_be_scored_adverse(self):
        model = _model((
            _effect("E1", "A0", "P1"),
            _effect(
                "E2", "A1", "P3",
                outcome="dam repair is not undertaken",
                relation="FOREGONE_BENEFIT",
                polarity="ADVERSE",
                directness="FOREGONE",
            ),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(any("E2" in item and "FOREGONE" in item for item in errors), errors)

    def test_foregone_polarity_cannot_attach_to_direct_roles(self):
        model = _model((
            _effect("E1", "A0", "P1", polarity="FOREGONE"),
            _effect("E2", "A1", "P2"),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(any("E1" in item and "directness" in item for item in errors), errors)

    def test_certain_causal_link_cannot_list_conditions(self):
        model = _model(
            (_effect("E1", "A0", "P1"), _effect("E2", "A1", "P2")),
            conditions=(
                WorldCondition("COND1", "engineer must survive", "UNKNOWN", "MATERIAL", REF),
            ),
            links=(
                CausalLink("A0", "ENABLES", "E1", "CERTAIN", ("COND1",), REF),
            ),
        )
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(any("CERTAIN" in item and "condition" in item for item in errors), errors)

    def test_comparison_only_provenance_is_rejected(self):
        model = _model((
            _effect("E1", "A0", "P1"),
            _effect(
                "E2", "A0", "P2",
                outcome="A dozen trapped residents receive no urgent relief",
                relation="DENIED_RELIEF",
                polarity="ADVERSE",
                quantities=("dozen",),
                provenance=COMPARE_REF,
            ),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(any("E2" in item and "comparison" in item.casefold() for item in errors), errors)

    def test_quantity_cannot_self_ground_through_generated_outcome(self):
        """Every explicit quantity must trace to provenance, not outcome text."""
        model = _model((
            _effect("E1", "A0", "P1"),
            _effect(
                "E2", "A0", "P2",
                outcome="A dozen trapped residents receive no urgent relief",
                relation="DENIED_RELIEF",
                polarity="ADVERSE",
                quantities=("dozen",),
                provenance=COMPARE_REF + (
                    SourceRef("C0", "A dispatcher must send one repair crew."),
                ),
            ),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(
            any("dozen" in item and "provenance" in item for item in errors),
            errors,
        )

    def test_outcome_quantity_cannot_self_ground_even_when_quantities_empty(self):
        model = _model((
            _effect("E1", "A0", "P1"),
            _effect(
                "E2", "A0", "P2",
                outcome="A dozen trapped residents receive no urgent relief",
                relation="DENIED_RELIEF",
                polarity="ADVERSE",
                quantities=(),
                provenance=(
                    SourceRef("C0", "A dispatcher must send one repair crew."),
                ),
            ),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(
            any("dozen" in item and "provenance" in item for item in errors),
            errors,
        )

    def test_action_text_and_fact_clause_jointly_support_a_denial(self):
        model = _model((
            _effect(
                "E1", "A0", "P2",
                outcome="A dozen trapped residents receive no urgent relief",
                relation="DENIED_RELIEF",
                polarity="ADVERSE",
                quantities=("dozen",),
                provenance=DOZEN_REF + ACTION_A0_REF + COMPARE_REF,
            ),
            _effect("E2", "A1", "P2", outcome="trapped family lives"),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(errors, [])

    def test_downstream_effect_must_copy_stated_scale_phrase(self):
        model = _model((
            _effect("E1", "A0", "P1"),
            _effect("E2", "A1", "P2"),
            _effect(
                "E3", "A0", "P3",
                outcome="dam repair would spare tens of thousands of residents",
                relation="DOWNSTREAM_BENEFIT",
                polarity="BENEFICIAL",
                directness="DOWNSTREAM",
                modality="STIPULATED_CONDITIONAL",
                condition_ids=("COND1",),
                quantities=(),
                provenance=SCALE_REF,
            ),
        ), conditions=(
            WorldCondition("COND1", "engineer must be reached in time", "UNKNOWN", "MATERIAL", SCALE_REF),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(any("E3" in item and "quantities" in item for item in errors), errors)

    def test_invented_quantities_are_rejected(self):
        model = _model((
            _effect("E1", "A0", "P1"),
            _effect("E2", "A1", "P2"),
            _effect(
                "E3", "A0", "P3",
                outcome="dam repair would spare tens of thousands of residents",
                relation="DOWNSTREAM_BENEFIT",
                polarity="BENEFICIAL",
                directness="DOWNSTREAM",
                modality="STIPULATED_CONDITIONAL",
                condition_ids=("COND1",),
                quantities=("60 QALYs", "0.4"),
                provenance=SCALE_REF,
            ),
        ), conditions=(
            WorldCondition("COND1", "engineer must be reached in time", "UNKNOWN", "MATERIAL", SCALE_REF),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(any("60 QALYs" in item or "0.4" in item for item in errors), errors)

    def test_co_cited_clause_does_not_force_unused_quantities(self):
        """Citing two FACT clauses does not require copying the unused count."""
        model = _model((
            _effect(
                "E1", "A0", "P3",
                outcome="dam repair would spare tens of thousands of residents",
                relation="DOWNSTREAM_BENEFIT",
                polarity="BENEFICIAL",
                directness="DOWNSTREAM",
                modality="STIPULATED_CONDITIONAL",
                condition_ids=("COND1",),
                quantities=("tens of thousands",),
                provenance=SCALE_REF + DOZEN_REF,
            ),
            _effect("E2", "A1", "P2", outcome="trapped family lives"),
        ), conditions=(
            WorldCondition("COND1", "engineer must be reached in time", "UNKNOWN", "MATERIAL", SCALE_REF),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(errors, [])

    def test_direct_effect_need_not_inherit_an_unused_scale_phrase(self):
        model = _model((
            _effect("E1", "A0", "P1", provenance=REF),
            _effect("E2", "A1", "P2"),
        ))
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(errors, [])

    def test_parse_rejects_the_two_slips_without_repairing_them(self):
        raw = {
            "parties": [
                {"party_id": "P0", "label": "dispatcher", "kind": "SYSTEM", "clause_ids": ["C0"]},
                {"party_id": "P1", "label": "engineer", "kind": "PERSON", "clause_ids": ["C0"]},
                {"party_id": "P2", "label": "family", "kind": "GROUP", "clause_ids": ["C2"]},
                {"party_id": "P3", "label": "residents", "kind": "POPULATION", "clause_ids": ["C1"]},
            ],
            "actions": [
                {
                    "action_id": "A0", "intervention": "send crew to Site A",
                    "actor_party_id": "P0", "recipient_party_ids": ["P1"],
                    "effect_ids": ["E1"], "clause_ids": ["C0"],
                },
                {
                    "action_id": "A1", "intervention": "send crew to Site B",
                    "actor_party_id": "P0", "recipient_party_ids": ["P2"],
                    "effect_ids": ["E2"], "clause_ids": ["C4"],
                },
            ],
            "effects": [
                {
                    "effect_id": "E1", "action_id": "A0", "party_id": "P1",
                    "outcome": "engineer lives", "relation": "SURVIVES",
                    "polarity": "BENEFICIAL", "directness": "DIRECT",
                    "modality": "CERTAIN", "condition_ids": ["COND1"],
                    "quantities": [], "clause_ids": ["C0"],
                },
                {
                    "effect_id": "E2", "action_id": "A1", "party_id": "P3",
                    "outcome": "dam repair is not undertaken",
                    "relation": "FOREGONE_BENEFIT", "polarity": "ADVERSE",
                    "directness": "FOREGONE", "modality": "CERTAIN",
                    "condition_ids": [], "quantities": [], "clause_ids": ["C4"],
                },
            ],
            "conditions": [
                {
                    "condition_id": "COND1", "description": "engineer must survive",
                    "value_status": "UNKNOWN", "decision_relevance": "MATERIAL",
                    "clause_ids": ["C1"],
                },
            ],
            "causal_links": [],
        }
        with self.assertRaises(ValueError) as raised:
            parse_world_model(raw, clauses=CLAUSES, action_ids=["A0", "A1"])
        message = str(raised.exception)
        self.assertIn("CERTAIN", message)
        self.assertIn("FOREGONE", message)

    def test_parse_accepts_action_id_as_supporting_provenance(self):
        raw = {
            "parties": [
                {"party_id": "P0", "label": "dispatcher", "kind": "SYSTEM", "clause_ids": ["C0"]},
                {"party_id": "P1", "label": "engineer", "kind": "PERSON", "clause_ids": ["C0"]},
                {"party_id": "P2", "label": "family", "kind": "GROUP", "clause_ids": ["C2"]},
            ],
            "actions": [
                {
                    "action_id": "A0", "intervention": "send crew to Site A",
                    "actor_party_id": "P0", "recipient_party_ids": ["P1"],
                    "effect_ids": ["E1"], "clause_ids": ["C0"],
                },
                {
                    "action_id": "A1", "intervention": "send crew to Site B",
                    "actor_party_id": "P0", "recipient_party_ids": ["P2"],
                    "effect_ids": ["E2"], "clause_ids": ["C4"],
                },
            ],
            "effects": [
                {
                    "effect_id": "E1", "action_id": "A0", "party_id": "P2",
                    "outcome": "dozen residents left without relief",
                    "relation": "DENIED_RELIEF", "polarity": "ADVERSE",
                    "directness": "DIRECT", "modality": "CERTAIN",
                    "condition_ids": [], "quantities": ["dozen"],
                    "clause_ids": ["C2", "A0", "C3"],
                },
                {
                    "effect_id": "E2", "action_id": "A1", "party_id": "P2",
                    "outcome": "family lives", "relation": "SURVIVES",
                    "polarity": "BENEFICIAL", "directness": "DIRECT",
                    "modality": "CERTAIN", "condition_ids": [],
                    "quantities": ["dozen"], "clause_ids": ["C2"],
                },
            ],
            "conditions": [],
            "causal_links": [],
        }
        model = parse_world_model(
            raw,
            clauses=CLAUSES,
            action_ids=["A0", "A1"],
            action_texts={
                "A0": ACTION_A0_REF[0].excerpt,
                "A1": "send crew to Site B and save the family",
            },
        )
        e1 = next(effect for effect in model.effects if effect.effect_id == "E1")
        self.assertEqual(e1.quantities, ("dozen",))
        self.assertEqual(
            [ref.clause_id for ref in e1.provenance],
            ["C2", "A0", "C3"],
        )


class TypedWorldProjectionTests(unittest.TestCase):
    def test_foregone_polarity_projects_as_foregoes_not_worsens(self):
        graph = compile_scenario_graph(
            "A dispatcher must send one repair crew to Site A or Site B.",
            ["send crew to Site A and reach the engineer", "send crew to Site B and save the family"],
            world_model=_valid_model().as_dict(),
        )
        effects = project_grounded_action_effects(graph)
        foregone = [
            item for item in effects
            if item.action_id == "A1"
            and "downstream residents" in item.affected_subject.casefold()
        ]
        self.assertTrue(foregone, effects)
        self.assertEqual(foregone[0].direction, "FOREGOES")
        self.assertNotEqual(foregone[0].direction, "WORSENS")
        downstream = [
            item for item in effects
            if item.action_id == "A0"
            and "downstream residents" in item.affected_subject.casefold()
        ]
        self.assertTrue(downstream, effects)
        self.assertEqual(downstream[0].direction, "IMPROVES")
        self.assertEqual(downstream[0].magnitude_or_qualifier, "tens of thousands")


if __name__ == "__main__":
    unittest.main()
