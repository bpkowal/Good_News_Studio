"""Typed world-model admission must reject the two representation slips.

These tests use a dispatcher/dam fixture so the rules stay domain-neutral:
CERTAIN effects cannot carry conditions, and a foregone opportunity is not
action-relative harm. Quantities may only copy explicit source spans from
provenance, never from the generated outcome alone.
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from global_workspace.scenario_semantics import (
    attach_typed_world_model, compile_scenario_graph,
    project_grounded_action_effects,
)
from global_workspace.world_state import (
    CausalLink,
    CounterfactualLink,
    SourceRef,
    WorldAction,
    WorldCondition,
    WorldEffect,
    WorldParty,
    ScenarioWorldModel,
    classify_clause_role,
    explicit_likelihood_spans,
    explicit_quantity_spans,
    explicit_scope_spans,
    explicit_temporal_spans,
    parse_world_model,
    validate_world_model,
    world_model_from_dict,
)
from global_workspace.action_identity import build_canonical_action_records
from global_workspace.graph_transactions import SemanticGraphStore
from global_workspace.utilitarian_ledger import (
    EffectValuationProposal,
    apply_utilitarian_ledger_transaction,
)
from global_workspace.local_specialists import (
    CompactLocalSpecialist,
    _candidate_from_data,
    ground_actions_in_scenario,
)
from global_workspace.models import CandidateChunk, WorkspaceBroadcast
from global_workspace.engine import WorkspaceConfig, WorkspaceEngine
from global_workspace.epistemic_ledger import (
    apply_side_premise_audit,
    attach_candidate_dependencies,
    ledger_projection,
    seed_proposition_ledger,
)
from global_workspace.semantic_graph import SemanticEdge, SemanticGraph, SemanticNode
from global_workspace.specialist_authority import apply_specialist_authority
from global_workspace.presentation import (
    _candidate_epistemic_qualification,
    render_decision_brief,
)
from global_workspace.premise_audit import audit_side_premises as run_side_premise_audit


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

    def test_written_cardinality_before_population_noun_is_a_quantity(self):
        self.assertEqual(
            explicit_quantity_spans("life support for eight premature infants"),
            ("eight",),
        )
        self.assertEqual(explicit_quantity_spans("one repair crew"), ())

    def test_source_qualifier_extractors_keep_likelihood_scope_and_time_distinct(self):
        text = "an immediate, near-certain fatal failure creates widespread disruption"
        self.assertEqual(explicit_likelihood_spans(text), ("near-certain",))
        self.assertEqual(explicit_scope_spans(text), ("widespread",))
        self.assertEqual(explicit_temporal_spans(text), ("immediate",))


class ClauseRoleTests(unittest.TestCase):
    def test_comparison_and_fact_roles(self):
        self.assertEqual(classify_clause_role(COMPARE_REF[0].excerpt), "COMPARISON")
        self.assertEqual(classify_clause_role(DOZEN_REF[0].excerpt), "FACT")
        self.assertEqual(classify_clause_role("Which pipeline should receive the supply?"), "INTERROGATIVE")


class WorldModelValidationTests(unittest.TestCase):
    def test_action_effect_ids_are_derived_from_owned_effects(self):
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
                    "effect_ids": ["STALE_ID"], "clause_ids": ["C0"],
                },
                {
                    "action_id": "A1", "intervention": "send crew to Site B",
                    "actor_party_id": "P0", "recipient_party_ids": ["P2"],
                    "effect_ids": [], "clause_ids": ["C4"],
                },
            ],
            "effects": [
                {
                    "effect_id": "E1", "action_id": "A0", "party_id": "P1",
                    "outcome": "engineer receives assistance", "relation": "ASSISTED",
                    "polarity": "BENEFICIAL", "directness": "DIRECT",
                    "modality": "CERTAIN", "condition_ids": [], "quantities": [],
                    "clause_ids": ["C0"],
                },
                {
                    "effect_id": "E2", "action_id": "A1", "party_id": "P2",
                    "outcome": "family receives assistance", "relation": "ASSISTED",
                    "polarity": "BENEFICIAL", "directness": "DIRECT",
                    "modality": "CERTAIN", "condition_ids": [], "quantities": [],
                    "clause_ids": ["C2"],
                },
            ],
            "conditions": [],
            "causal_links": [],
        }

        model = parse_world_model(raw, clauses=CLAUSES, action_ids=["A0", "A1"])

        self.assertEqual(model.actions[0].effect_ids, ("E1",))
        self.assertEqual(model.actions[1].effect_ids, ("E2",))

    def test_grounding_repair_receives_the_rejected_candidate(self):
        rejected_candidate = {
            "sentinel": "preserve this candidate",
            "actions": {"A0": {}, "A1": {}},
        }
        committed = {
            "status": "COMMITTED", "actions": {"A0": {}, "A1": {}},
            "errors": [], "clauses": [], "world_contradictions": [],
        }
        rejected = {
            "status": "REJECTED", "actions": {},
            "errors": ["typed world model rejected: E1 omits qualifier 'broader'"],
            "clauses": [], "world_contradictions": [],
        }
        prompts: list[str] = []

        def fake_call(_llm, prompt, **_kwargs):
            prompts.append(prompt)
            return {"choices": [{"text": __import__("json").dumps(rejected_candidate)}]}

        with patch(
            "global_workspace.local_specialists._call_json_llm",
            side_effect=fake_call,
        ), patch(
            "global_workspace.local_specialists._admit_action_source_rows",
            side_effect=[rejected, committed],
        ):
            result = ground_actions_in_scenario(
                object(),
                "Option A: send aid north. Option B: send aid south.",
                ["send aid north", "send aid south"],
                max_attempts=2,
            )

        self.assertEqual(result["status"], "COMMITTED")
        self.assertEqual(result["repair_attempts"], 1)
        self.assertEqual(
            result["attempts"][0]["errors"],
            ["typed world model rejected: E1 omits qualifier 'broader'"],
        )
        self.assertEqual(result["attempts"][1]["errors"], [])
        self.assertEqual(len(prompts), 2)
        self.assertIn('"sentinel": "preserve this candidate"', prompts[1])
        self.assertIn("preserve every field not implicated", prompts[1].casefold())
        self.assertLess(
            prompts[1].index("Rejected candidate JSON"),
            prompts[1].index("[/INST]"),
        )

    @staticmethod
    def _qualified_population_model(*, omit_qualifiers: bool = False) -> ScenarioWorldModel:
        infant_ref = (SourceRef(
            "C1",
            "Without backup power, eight premature infants face immediate, "
            "near-certain fatal equipment failure.",
        ),)
        resident_ref = (SourceRef(
            "C2",
            "The plant protects safe water for tens of thousands of residents "
            "against widespread sanitation failure.",
        ),)
        parties = (
            WorldParty("P0", "allocation system", "AUTOMATED_SYSTEM", REF),
            WorldParty(
                "P1", "premature infants", "POPULATION", infant_ref,
                () if omit_qualifiers else ("eight",),
            ),
            WorldParty(
                "P2", "residents", "POPULATION", resident_ref,
                () if omit_qualifiers else ("tens of thousands",),
            ),
        )
        effects = (
            WorldEffect(
                "E1", "A0", "P1", "immediate near-certain fatal equipment failure",
                "CAUSES", "ADVERSE", "DOWNSTREAM", "PROBABILISTIC",
                "HEALTH_OUTCOME", ("COND1",), (), infant_ref,
                () if omit_qualifiers else ("near-certain",),
                (), () if omit_qualifiers else ("immediate",),
            ),
            WorldEffect(
                "E2", "A1", "P2", "widespread sanitation failure",
                "PREVENTS", "BENEFICIAL", "DOWNSTREAM", "PROBABILISTIC",
                "HEALTH_OUTCOME", ("COND2",), (), resident_ref,
                (), () if omit_qualifiers else ("widespread",), (),
            ),
        )
        return ScenarioWorldModel(
            parties=parties,
            actions=(
                WorldAction("A0", "withhold backup power", "P0", (), ("E1",), REF),
                WorldAction("A1", "power the treatment plant", "P0", (), ("E2",), REF),
            ),
            effects=effects,
            conditions=(
                WorldCondition("COND1", "backup power is absent", "STATED", "MATERIAL", infant_ref),
                WorldCondition("COND2", "plant power is absent", "STATED", "MATERIAL", resident_ref),
            ),
            schema_version="1.2",
        )

    def test_schema_1_2_requires_source_bound_population_and_effect_qualifiers(self):
        errors, _ = validate_world_model(
            self._qualified_population_model(omit_qualifiers=True),
            action_ids=["A0", "A1"],
        )
        self.assertTrue(any("P1 omits" in error and "eight" in error for error in errors))
        self.assertTrue(any(
            "P2 omits" in error and "tens of thousands" in error for error in errors
        ))
        self.assertTrue(any("E1 omits" in error and "likelihood" in error for error in errors))
        self.assertTrue(any("E1 omits" in error and "temporal" in error for error in errors))
        self.assertTrue(any("E2 omits" in error and "scope" in error for error in errors))

    def test_schema_1_2_detects_qualifier_omitted_from_generated_outcome(self):
        model = self._qualified_population_model()
        weakened = replace(
            model.effects[0],
            outcome="fatal equipment failure",
            likelihood_qualifiers=(),
            temporal_qualifiers=(),
        )
        errors, _ = validate_world_model(
            replace(model, effects=(weakened, model.effects[1])),
            action_ids=["A0", "A1"],
        )
        self.assertTrue(any(
            "E1 omits" in error and "near-certain" in error for error in errors
        ))
        self.assertTrue(any(
            "E1 omits" in error and "immediate" in error for error in errors
        ))

    def test_population_quantities_do_not_cross_bind_within_one_clause(self):
        model = self._qualified_population_model()
        shared_ref = (SourceRef(
            "C3",
            "The choice affects eight premature infants while tens of thousands "
            "of residents rely on safe water.",
        ),)
        parties = (
            model.parties[0],
            replace(model.parties[1], provenance=shared_ref),
            replace(model.parties[2], provenance=shared_ref),
        )
        errors, _ = validate_world_model(
            replace(model, parties=parties), action_ids=["A0", "A1"],
        )
        self.assertFalse(any(
            "P1 omits" in error and "tens of thousands" in error for error in errors
        ))
        self.assertFalse(any(
            "P2 omits" in error and "eight" in error for error in errors
        ))

    def test_schema_1_2_qualifiers_project_to_atomic_established_propositions(self):
        model = self._qualified_population_model()
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(errors, [])
        restored = world_model_from_dict(model.as_dict())
        self.assertIsNotNone(restored)
        self.assertEqual(restored.schema_version, "1.2")
        self.assertEqual(restored.parties[1].quantities, ("eight",))
        self.assertEqual(restored.effects[0].likelihood_qualifiers, ("near-certain",))
        graph = SemanticGraph()
        for action_id, action in (("A0", "withhold backup power"), ("A1", "power plant")):
            graph.add_node(SemanticNode(
                action_id, "ACTION", action, ("scenario_action_set",),
                {"canonical_action_id": action_id},
            ))
        attach_typed_world_model(graph, model)
        ledger = seed_proposition_ledger(graph)
        established_claims = {
            row.claim for row in ledger.values()
            if row.epistemic_status == "ESTABLISHED"
        }
        self.assertIn("eight premature infants", established_claims)
        self.assertIn("tens of thousands residents", established_claims)
        self.assertTrue(any(claim.startswith("near-certain —") for claim in established_claims))
        self.assertTrue(any(claim.startswith("widespread —") for claim in established_claims))

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


class ActionScopedCausalityTests(unittest.TestCase):
    """Causal chains stay inside actions; alternative comparisons never become mechanisms."""

    def atomic_model(self) -> ScenarioWorldModel:
        parties = (
            WorldParty("P0", "allocation system", "AUTOMATED_SYSTEM", REF),
            WorldParty("P1", "community facility", "ORGANIZATION", REF),
            WorldParty("P2", "community residents", "POPULATION_GROUP", REF),
            WorldParty("P3", "urgent recipients", "POPULATION_GROUP", REF),
        )
        effects = (
            WorldEffect("A0_RESOURCE", "A0", "P1", "facility receives the resource", "RECEIVES", "BENEFICIAL", "DIRECT", "CERTAIN", "RESOURCE_TRANSFER", (), (), REF),
            WorldEffect("A0_OUTPUT", "A0", "P1", "facility produces supplies", "PRODUCES", "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "CAPABILITY_CHANGE", (), (), REF),
            WorldEffect("A0_PEOPLE", "A0", "P2", "residents receive sustained supplies", "RECEIVE_SUPPLIES", "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME", (), (), REF),
            WorldEffect("A0_FOREGONE", "A0", "P3", "urgent relief is foregone", "FOREGONE_RELIEF", "FOREGONE", "FOREGONE", "CERTAIN", "OPPORTUNITY_LOSS", (), (), REF),
            WorldEffect("A1_RESOURCE", "A1", "P3", "urgent recipients receive the resource", "RECEIVES", "BENEFICIAL", "DIRECT", "CERTAIN", "RESOURCE_TRANSFER", (), (), REF),
            WorldEffect("A1_RELIEF", "A1", "P3", "urgent condition is relieved", "RELIEVES", "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME", (), (), REF),
            WorldEffect("A1_FOREGONE", "A1", "P2", "sustained supplies are foregone", "FOREGONE_SUPPLIES", "FOREGONE", "FOREGONE", "CERTAIN", "OPPORTUNITY_LOSS", (), (), REF),
        )
        actions = (
            WorldAction("A0", "route resource to facility", "P0", ("P1",), tuple(e.effect_id for e in effects if e.action_id == "A0"), REF),
            WorldAction("A1", "route resource to urgent recipients", "P0", ("P3",), tuple(e.effect_id for e in effects if e.action_id == "A1"), REF),
        )
        return ScenarioWorldModel(
            parties=parties, actions=actions, effects=effects,
            causal_links=(
                CausalLink("A0_RESOURCE", "ENABLES", "A0_OUTPUT", "CERTAIN", (), REF, "A0"),
                CausalLink("A0_OUTPUT", "CAUSES", "A0_PEOPLE", "CERTAIN", (), REF, "A0"),
                CausalLink("A1_RESOURCE", "CAUSES", "A1_RELIEF", "CERTAIN", (), REF, "A1"),
            ),
            counterfactual_links=(
                CounterfactualLink("A0", "A0_FOREGONE", "FOREGOES_ALTERNATIVE_EFFECT", "A1", "A1_RELIEF", "CERTAIN", (), REF),
                CounterfactualLink("A1", "A1_FOREGONE", "FOREGOES_ALTERNATIVE_EFFECT", "A0", "A0_PEOPLE", "CERTAIN", (), REF),
            ),
            schema_version="1.1",
        )

    def test_action_mechanisms_do_not_include_counterfactual_links(self):
        model = self.atomic_model()
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(errors, [])
        records = build_canonical_action_records(
            ["route resource to facility", "route resource to urgent recipients"],
            world_model=model.as_dict(),
        )
        a0, a1 = records
        self.assertIn("facility produces supplies", a0.mechanism)
        self.assertNotIn("urgent condition is relieved", a0.mechanism)
        self.assertIn("urgent condition is relieved", a1.mechanism)
        self.assertNotIn("facility produces supplies", a1.mechanism)
        self.assertEqual(len(a0.counterfactual_effects), 1)
        self.assertEqual(len(a1.counterfactual_effects), 1)

    def test_cross_action_causal_link_is_rejected(self):
        model = self.atomic_model()
        invalid = ScenarioWorldModel(
            parties=model.parties, actions=model.actions, effects=model.effects,
            causal_links=(
                CausalLink("A0_RESOURCE", "CAUSES", "A1_RELIEF", "CERTAIN", (), REF, "A0"),
            ),
            schema_version="1.1",
        )
        errors, _ = validate_world_model(invalid, action_ids=["A0", "A1"])
        self.assertTrue(any("crosses action boundaries" in error for error in errors), errors)

    def test_compound_human_outcome_on_organization_is_rejected(self):
        model = self.atomic_model()
        bad_effect = WorldEffect(
            "A0_RESOURCE", "A0", "P1",
            "facility receives water, thereby sustaining residents",
            "RECEIVES", "BENEFICIAL", "DIRECT", "CERTAIN",
            "RESOURCE_TRANSFER", (), (), REF,
        )
        invalid = ScenarioWorldModel(
            parties=model.parties, actions=model.actions,
            effects=tuple(bad_effect if e.effect_id == "A0_RESOURCE" else e for e in model.effects),
            causal_links=model.causal_links,
            counterfactual_links=model.counterfactual_links,
            schema_version="1.1",
        )
        errors, _ = validate_world_model(invalid, action_ids=["A0", "A1"])
        self.assertTrue(any("multiple causal stages" in error for error in errors), errors)
        self.assertTrue(any("human outcome" in error for error in errors), errors)

    def test_counterfactual_edges_are_not_action_consequences(self):
        model = self.atomic_model()
        graph = compile_scenario_graph(
            "A system routes one resource between two recipients.",
            ["route resource to facility", "route resource to urgent recipients"],
            world_model=model.as_dict(),
        )
        self.assertEqual(
            len([edge for edge in graph.edges if edge.relation == "COUNTERFACTUALLY_FOREGOES"]),
            2,
        )
        a0_consequences = {
            graph.nodes[edge.target].attributes.get("world_effect_id")
            for edge in graph.outgoing("A0", "HAS_CONSEQUENCE")
        }
        self.assertNotIn("A1_RELIEF", a0_consequences)

    def test_legacy_cross_action_foregoes_link_migrates_at_ingestion(self):
        serialized = self.atomic_model().as_dict()
        serialized["schema_version"] = "1.0"
        serialized["counterfactual_links"] = []
        serialized["causal_links"] = [*serialized["causal_links"], {
            "source_id": "A0",
            "relation": "FOREGOES",
            "target_id": "A1_RELIEF",
            "modality": "CERTAIN",
            "condition_ids": (),
            "provenance": tuple(ref.as_dict() for ref in REF),
        }]
        restored = world_model_from_dict(serialized)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertFalse(any(
            link.source_id == "A0" and link.target_id == "A1_RELIEF"
            for link in restored.causal_links
        ))
        self.assertTrue(any(
            link.action_id == "A0"
            and link.source_effect_id == "A0_FOREGONE"
            and link.alternative_effect_id == "A1_RELIEF"
            for link in restored.counterfactual_links
        ))

    def test_utilitarian_valuation_inherits_world_polarity_by_effect_id(self):
        model = self.atomic_model()
        graph = compile_scenario_graph(
            "A system routes one resource between two recipients.",
            ["route resource to facility", "route resource to urgent recipients"],
            world_model=model.as_dict(),
        )
        projected = project_grounded_action_effects(graph)
        by_action: dict[str, list[str]] = {"A0": [], "A1": []}
        for effect in projected:
            by_action[effect.action_id].append(effect.effect_id)
        proposal = {"actions": [
            {
                "action_id": action_id,
                "valuations": [
                    {
                        "effect_id": effect_id,
                        "importance": "HIGH",
                        "reason": "material aggregate welfare contribution",
                    }
                    for effect_id in effect_ids
                ],
            }
            for action_id, effect_ids in by_action.items()
        ]}
        store = SemanticGraphStore(graph)
        transaction = apply_utilitarian_ledger_transaction(
            store, proposal, cycle=1, specialist="utilitarian",
            allowed_actions=(
                "route resource to facility", "route resource to urgent recipients",
            ),
        )
        self.assertEqual(transaction.status, "COMMITTED", transaction.errors)
        committed = transaction.proposal["committed_consequences"]
        by_world_id = {row["world_effect_id"]: row for row in committed}
        self.assertEqual(by_world_id["A0_RESOURCE"]["direction"], "BENEFIT")
        self.assertEqual(
            by_world_id["A0_FOREGONE"]["direction"], "OPPORTUNITY_COST",
        )
        self.assertEqual(
            by_world_id["A0_FOREGONE"]["polarity"], "FOREGONE",
        )
        self.assertEqual(
            by_world_id["A0_FOREGONE"]["direction_source"],
            "GROUNDED_WORLD_POLARITY",
        )
        self.assertNotIn("proposed_direction", by_world_id["A0_FOREGONE"])
        self.assertFalse(any("contradicts" in error for error in transaction.errors))

    def test_utilitarian_valuation_cannot_override_direction(self):
        graph = compile_scenario_graph(
            "A system routes one resource between two recipients.",
            ["route resource to facility", "route resource to urgent recipients"],
            world_model=self.atomic_model().as_dict(),
        )
        projected = project_grounded_action_effects(graph)
        proposal = {"actions": []}
        for action_id in ("A0", "A1"):
            valuations = []
            for effect in projected:
                if effect.action_id != action_id:
                    continue
                valuations.append({
                    "effect_id": effect.effect_id,
                    "importance": "HIGH",
                    "reason": "material aggregate welfare contribution",
                    "direction": "BENEFIT",
                })
            proposal["actions"].append({
                "action_id": action_id, "valuations": valuations,
            })
        transaction = apply_utilitarian_ledger_transaction(
            SemanticGraphStore(graph), proposal, cycle=1,
            specialist="utilitarian",
            allowed_actions=(
                "route resource to facility", "route resource to urgent recipients",
            ),
        )
        self.assertEqual(transaction.status, "REJECTED")
        self.assertTrue(any("extra" in error.casefold() for error in transaction.errors))

    def test_specialist_effect_valuations_compile_without_factual_fields(self):
        graph = compile_scenario_graph(
            "A system routes one resource between two recipients.",
            ["route resource to facility", "route resource to urgent recipients"],
            world_model=self.atomic_model().as_dict(),
        )
        grounded: list[dict[str, object]] = []
        valuation_table: dict[str, list[dict[str, str]]] = {"A0": [], "A1": []}
        for effect in project_grounded_action_effects(graph):
            consequence = graph.nodes[effect.consequence_id]
            grounded.append({
                "effect_id": effect.effect_id,
                "action_id": effect.action_id,
                "outcome": consequence.label,
                "subject": effect.affected_subject,
                "direction": effect.direction,
                "polarity": consequence.attributes["polarity"],
                "modality": consequence.attributes["modality"],
                "qualifier": effect.magnitude_or_qualifier,
            })
            valuation_table[effect.action_id].append({
                "eid": effect.effect_id,
                "wi": "HIGH",
                "vr": "material aggregate welfare contribution",
            })
        candidate = _candidate_from_data(
            "utilitarian",
            ["route resource to facility", "route resource to urgent recipients"],
            {
                "scores": {"A0": 0.6, "A1": 0.4}, "r": "A0",
                "c": "IMMINENT_HARM", "u": "NONE", "w": "Higher aggregate welfare",
                "j": "NONE", "e": "STATED_FACTS", "x": "NONE", "z": 0.8,
                "ct": valuation_table, "cd": False, "cm": "NONE",
            },
            WorkspaceBroadcast(), "NONE", {}, grounded_effects=grounded,
        )
        self.assertTrue(candidate.utilitarian_ledger_proposal)
        self.assertEqual(candidate.framework_grounding_penalty, 0.0)
        rows = [
            row for values in candidate.utilitarian_consequence_table.values()
            for row in values
        ]
        foregone = [row for row in rows if row["polarity"] == "FOREGONE"]
        self.assertTrue(foregone)
        self.assertTrue(all(
            row["direction"] == "OPPORTUNITY_COST" for row in foregone
        ))


def _epistemic_grounded_graph() -> SemanticGraph:
    graph = SemanticGraph()
    graph.add_node(SemanticNode(
        "A0", "ACTION", "provide treatment", ("scenario_clause:C0",),
        {"canonical_action_id": "A0"},
    ))
    graph.add_node(SemanticNode(
        "A0:WORLD_EFFECT:E1", "CONSEQUENCE",
        "medical emergencies are prevented", ("scenario_clause:C1",),
        {
            "scenario_grounded": True, "world_effect_id": "E1",
            "polarity": "BENEFICIAL", "relation": "PREVENTS",
            "targets": ["affected residents"],
            "affected_subjects": ["affected residents"],
            "source_clause_id": "C1",
        },
    ))
    graph.add_node(SemanticNode(
        "PARTY:P1", "TARGET", "affected residents", ("scenario_clause:C1",),
        {"semantic_role": "AFFECTED_SUBJECT", "party_id": "P1"},
    ))
    graph.add_edge(SemanticEdge("A0", "HAS_CONSEQUENCE", "A0:WORLD_EFFECT:E1"))
    graph.add_edge(SemanticEdge("A0:WORLD_EFFECT:E1", "AFFECTS", "PARTY:P1"))
    return graph


def _epistemic_candidate(
    *, claim: str = "", tier: str = "NOT_APPLICABLE",
) -> CandidateChunk:
    return CandidateChunk(
        specialist="deontological", constraint="DUTY",
        action_scores={"provide treatment": 0.8, "withhold treatment": 0.2},
        surprise=0.5, friction=0.5, confidence=0.9,
        epistemic_confidence=0.9, recommended_action="provide treatment",
        rationale="A rescue duty favors treatment.",
        decision_rule="Prefer treatment under the rescue duty.",
        adjudication_status="SUPPORTS", governing_eligible=True,
        speculative_claim=claim, evidence_calibration_tier=tier,
    )


class EpistemicPropositionLedgerTests(unittest.TestCase):
    def test_engine_rollback_preserves_same_cycle_side_audit_dependencies(self):
        class RejectingSpecialist:
            name = "duty"

            def __init__(self):
                self.calls = 0

            def evaluate(self, scenario, actions, broadcast):
                self.calls += 1
                rejected = self.calls == 2
                return CandidateChunk(
                    specialist=self.name, constraint="DUTY",
                    action_scores={
                        actions[0]: 0.2 if rejected else 0.8,
                        actions[1]: 0.8 if rejected else 0.2,
                    },
                    surprise=0.5, friction=0.6, confidence=0.8,
                    recommended_action=actions[1] if rejected else actions[0],
                    rationale="A duty-based comparison supports the current position.",
                    decision_rule="Prefer the action supported by the governing duty.",
                    framework_retention_status=(
                        "UPDATE_REJECTED" if rejected else "PRESERVED"
                    ),
                    framework_constraint_retained=True,
                    framework_validation_errors=(
                        ["unjustified duty transition"] if rejected else []
                    ),
                )

        audit_calls = 0

        def audit(_ledger, candidates):
            nonlocal audit_calls
            audit_calls += 1
            if audit_calls == 1:
                return {"status": "PASSED", "findings": [], "error": ""}
            return {
                "status": "FINDINGS", "error": "",
                "findings": [{
                    "specialist": candidates[0].specialist,
                    "claim": "the delayed action creates irreversible harm",
                    "binding": "NEW_HYPOTHESIS", "derived_from": [],
                    "decision_critical": True, "source_field": "decision_rule",
                    "reason": "the revised comparison relies on this outcome",
                }],
            }

        result = WorkspaceEngine(
            [RejectingSpecialist()],
            WorkspaceConfig(
                max_cycles=2, stable_cycles_required=2, min_valid_specialists=1,
                stop_redundant_consensus_cycles=False,
                enable_synthesis=False, enable_planning=False,
                enable_consensus_audit=False, enable_problem_state_audit=False,
                enable_reversal_audit=False,
            ),
        ).run(
            "Choose immediate or delayed action under an uncertain harm claim.",
            ["act immediately", "delay action"],
            audit_side_premises=audit,
        )
        final = result.cycles[-1].candidates[0]
        hypothesis_ids = [
            row["proposition_id"] for row in result.proposition_ledger
            if row["claim"] == "the delayed action creates irreversible harm"
        ]
        self.assertEqual(len(hypothesis_ids), 1)
        self.assertEqual(final.recommended_action, "act immediately")
        self.assertEqual(
            final.framework_retention_status, "PRESERVED_AFTER_REJECTED_UPDATE",
        )
        self.assertIn(hypothesis_ids[0], final.supporting_proposition_ids)
        self.assertIn(hypothesis_ids[0], final.decision_critical_proposition_ids)
        self.assertEqual(final.side_premise_audit_status, "FINDINGS")
        self.assertEqual(final.adjudication_status, "CONDITIONAL_SUPPORTS")
        self.assertTrue(result.shared_unresolved_dependencies)

    def test_presentation_qualifies_domain_general_specific_hypotheses(self):
        data = {"proposition_ledger": [{
            "proposition_id": "PROP:HYPOTHESIS:SUPPLIER",
            "claim": "the supplier will default within a week",
            "proposition_type": "HYPOTHESIS",
            "epistemic_status": "HYPOTHETICAL",
        }]}
        candidate = {
            "supporting_proposition_ids": ["PROP:HYPOTHESIS:SUPPLIER"],
            "decision_critical_proposition_ids": ["PROP:HYPOTHESIS:SUPPLIER"],
        }
        qualification = _candidate_epistemic_qualification(data, candidate)
        self.assertIn("supplier will default within a week", qualification)
        self.assertIn("conditional on the unestablished proposition", qualification)
        self.assertIn("hypothetical", qualification)

    def test_presentation_does_not_qualify_established_specific_claims(self):
        data = {"proposition_ledger": [{
            "proposition_id": "PROP:WORLD:E9",
            "claim": "the supplier has already defaulted",
            "proposition_type": "DESCRIPTIVE",
            "epistemic_status": "ESTABLISHED",
        }]}
        candidate = {
            "supporting_proposition_ids": ["PROP:WORLD:E9"],
            "decision_critical_proposition_ids": ["PROP:WORLD:E9"],
        }
        self.assertEqual(_candidate_epistemic_qualification(data, candidate), "")

    def test_presentation_suppresses_legacy_hypothesis_duplicating_established_atom(self):
        data = {"proposition_ledger": [
            {
                "proposition_id": "PROP:WORLD:E1",
                "claim": "widespread sanitation failure; affected subject: residents",
                "proposition_type": "DESCRIPTIVE",
                "epistemic_status": "ESTABLISHED",
            },
            {
                "proposition_id": "PROP:HYPOTHESIS:DUPLICATE",
                "claim": "widespread sanitation failure",
                "proposition_type": "HYPOTHESIS",
                "epistemic_status": "HYPOTHETICAL",
            },
        ]}
        candidate = {
            "supporting_proposition_ids": ["PROP:HYPOTHESIS:DUPLICATE"],
            "decision_critical_proposition_ids": ["PROP:HYPOTHESIS:DUPLICATE"],
        }
        self.assertEqual(_candidate_epistemic_qualification(data, candidate), "")

    def test_world_effects_seed_established_propositions(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        proposition = ledger["PROP:WORLD:E1"]
        self.assertEqual(proposition.epistemic_status, "ESTABLISHED")
        self.assertEqual(proposition.support_ids, ["E1"])
        self.assertIn("affected subject: affected residents", proposition.claim)

    def test_side_audit_preserves_transparent_composition_as_derived(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        ledger["PROP:WORLD:E2"] = type(ledger["PROP:WORLD:E1"])(
            proposition_id="PROP:WORLD:E2",
            claim="the clinic remains accessible",
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            support_ids=["E2"],
            introduced_by="WORLD_MODEL",
        )
        candidate = _epistemic_candidate()
        apply_side_premise_audit(ledger, [candidate], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": "deontological",
                "claim": "medical emergencies are prevented and the clinic remains accessible",
                "binding": "DERIVED_ESTABLISHED",
                "derived_from": ["PROP:WORLD:E1", "PROP:WORLD:E2"],
                "decision_critical": True,
                "source_field": "decision_rule",
                "reason": "transparent conjunction of established effects",
            }],
            "error": "",
        })

        proposition_id = candidate.decision_critical_proposition_ids[0]
        proposition = ledger[proposition_id]
        self.assertEqual(proposition.epistemic_status, "DERIVED")
        self.assertEqual(
            proposition.derived_from, ["PROP:WORLD:E1", "PROP:WORLD:E2"],
        )
        self.assertEqual(candidate.weakest_decision_critical_status, "DERIVED")
        self.assertEqual(candidate.selection_status, "SELECTED")

    def test_invalid_derived_binding_falls_back_to_hypothesis(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        ledger["PROP:WORLD:E2"] = type(ledger["PROP:WORLD:E1"])(
            proposition_id="PROP:WORLD:E2",
            claim="the clinic remains accessible",
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            support_ids=["E2"],
            introduced_by="WORLD_MODEL",
        )
        candidate = _epistemic_candidate()
        apply_side_premise_audit(ledger, [candidate], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": "deontological",
                "claim": "the prevented emergencies avert death",
                "binding": "DERIVED_ESTABLISHED",
                "derived_from": ["PROP:WORLD:E1", "PROP:WORLD:E2"],
                "decision_critical": True,
                "source_field": "rationale",
                "reason": "unsupported outcome is not transparent composition",
            }],
            "error": "",
        })

        proposition = ledger[candidate.decision_critical_proposition_ids[0]]
        self.assertEqual(proposition.epistemic_status, "HYPOTHETICAL")

    def test_compact_system_effect_id_is_valid_for_utilitarian_valuation(self):
        valuation = EffectValuationProposal.model_validate({
            "effect_id": "E1",
            "importance": "HIGH",
            "reason": "material welfare effect",
        })
        self.assertEqual(valuation.effect_id, "E1")

    def test_deontological_schema_keeps_duty_and_epistemic_dependencies_distinct(self):
        captured: dict[str, object] = {}

        class CaptureLlm:
            def complete_json(self, prompt, *, schema, **kwargs):
                captured.update(schema)
                raise RuntimeError("captured")

        graph = _epistemic_grounded_graph()
        specialist = CompactLocalSpecialist("deontological", CaptureLlm())
        specialist.scenario_graph = graph
        specialist.proposition_ledger = ledger_projection(
            seed_proposition_ledger(graph)
        )
        with self.assertRaisesRegex(RuntimeError, "captured"):
            specialist.evaluate(
                "A treatment prevents an emergency.",
                ["provide treatment", "withhold treatment"],
                WorkspaceBroadcast(),
            )
        properties = captured["properties"]
        self.assertEqual(properties["dp"]["type"], "object")
        self.assertEqual(properties["dcp"]["type"], "array")
        self.assertEqual(properties["ep"]["type"], "array")
        self.assertTrue({"dp", "dcp", "ep", "sps"} <= set(captured["required"]))

    def test_decision_critical_hypothesis_caps_candidate_authority(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        candidate = _epistemic_candidate(
            claim="the medical emergencies carry substantial mortality risk",
            tier="DECISION_CRITICAL",
        )
        candidate.supporting_proposition_ids = ["PROP:WORLD:E1"]
        attach_candidate_dependencies(ledger, candidate)
        profile = apply_specialist_authority(candidate)
        self.assertEqual(candidate.weakest_decision_critical_status, "HYPOTHETICAL")
        self.assertEqual(profile.adjudication_status, "CONDITIONAL_SUPPORTS")
        self.assertEqual(candidate.selection_status, "PROVISIONAL")
        self.assertFalse(candidate.comparison_complete)
        self.assertLessEqual(candidate.epistemic_confidence, 0.5)
        self.assertIn("provided that", candidate.decision_rule)

    def test_recurrence_increases_attention_but_never_status(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        first = _epistemic_candidate(
            claim="the medical emergencies carry substantial mortality risk",
            tier="DECISION_CRITICAL",
        )
        attach_candidate_dependencies(ledger, first)
        hypothesis_id = first.decision_critical_proposition_ids[0]
        second = _epistemic_candidate()
        second.supporting_proposition_ids = [hypothesis_id]
        second.decision_critical_proposition_ids = [hypothesis_id]
        attach_candidate_dependencies(ledger, second)
        self.assertEqual(ledger[hypothesis_id].mention_count, 2)
        self.assertEqual(ledger[hypothesis_id].decision_critical_mentions, 2)
        self.assertEqual(ledger[hypothesis_id].epistemic_status, "HYPOTHETICAL")

    def test_established_critical_dependency_does_not_reduce_authority(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        candidate = _epistemic_candidate()
        candidate.supporting_proposition_ids = ["PROP:WORLD:E1"]
        candidate.decision_critical_proposition_ids = ["PROP:WORLD:E1"]
        attach_candidate_dependencies(ledger, candidate)
        profile = apply_specialist_authority(candidate)
        self.assertEqual(candidate.weakest_decision_critical_status, "ESTABLISHED")
        self.assertEqual(profile.adjudication_status, "SUPPORTS")
        self.assertTrue(profile.governing_eligible)

    def test_stronger_paraphrase_of_world_fact_becomes_hypothesis(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        candidate = _epistemic_candidate()
        candidate.material_empirical_claims = [{
            "claim": "the medical emergencies are fatal",
            "proposition_id": "PROP:WORLD:E1",
            "decision_critical": True,
        }]
        attach_candidate_dependencies(ledger, candidate)
        hypotheses = [
            record for record in ledger.values()
            if record.proposition_type == "HYPOTHESIS"
        ]
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0].epistemic_status, "HYPOTHETICAL")
        self.assertEqual(hypotheses[0].derived_from, ["PROP:WORLD:E1"])
        self.assertIn(
            hypotheses[0].proposition_id,
            candidate.decision_critical_proposition_ids,
        )
        self.assertEqual(candidate.weakest_decision_critical_status, "HYPOTHETICAL")
        self.assertTrue(candidate.epistemic_binding_notes)
        self.assertLessEqual(candidate.epistemic_confidence, 0.5)

    def test_canonical_component_restatement_remains_established(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        candidate = _epistemic_candidate()
        candidate.material_empirical_claims = [{
            "claim": "medical emergencies are prevented",
            "proposition_id": "PROP:WORLD:E1",
            "decision_critical": True,
        }]
        attach_candidate_dependencies(ledger, candidate)
        self.assertEqual(candidate.weakest_decision_critical_status, "ESTABLISHED")
        self.assertEqual(
            candidate.decision_critical_proposition_ids, ["PROP:WORLD:E1"],
        )
        self.assertFalse(any(
            record.proposition_type == "HYPOTHESIS" for record in ledger.values()
        ))

    def test_side_audit_cannot_duplicate_established_component_as_hypothesis(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        candidate = _epistemic_candidate()
        apply_side_premise_audit(ledger, [candidate], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": "deontological",
                "claim": "medical emergencies are prevented",
                "binding": "NEW_HYPOTHESIS",
                "derived_from": ["PROP:WORLD:E1"],
                "decision_critical": True,
                "source_field": "rationale",
                "reason": "auditor incorrectly requested a duplicate",
            }],
            "error": "",
        })
        self.assertEqual(
            candidate.decision_critical_proposition_ids, ["PROP:WORLD:E1"],
        )
        self.assertEqual(candidate.weakest_decision_critical_status, "ESTABLISHED")
        self.assertFalse(any(
            record.proposition_type == "HYPOTHESIS" for record in ledger.values()
        ))

    def test_side_audit_attaches_undeclared_specific_outcome_as_hypothesis(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        candidate = _epistemic_candidate()
        candidate.supporting_proposition_ids = ["PROP:WORLD:E1"]
        attach_candidate_dependencies(ledger, candidate)
        apply_side_premise_audit(ledger, [candidate], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": "deontological",
                "claim": "the prevented emergencies would otherwise cause organ failure",
                "binding": "NEW_HYPOTHESIS",
                "derived_from": ["PROP:WORLD:E1"],
                "decision_critical": True,
                "source_field": "decision_rule",
                "reason": "the rule relies on a more specific medical outcome",
            }],
            "error": "",
        })
        finding = candidate.side_premise_audit_findings[0]
        proposition = ledger[finding["proposition_id"]]
        self.assertEqual(proposition.epistemic_status, "HYPOTHETICAL")
        self.assertEqual(proposition.derived_from, ["PROP:WORLD:E1"])
        self.assertEqual(candidate.side_premise_audit_status, "FINDINGS")
        self.assertEqual(candidate.weakest_decision_critical_status, "HYPOTHETICAL")
        self.assertLessEqual(candidate.epistemic_confidence, 0.5)

    def test_side_audit_reuses_semantically_matching_existing_hypothesis(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        first = _epistemic_candidate(
            claim="the emergencies pose a substantial risk of irreversible injury",
            tier="DECISION_CRITICAL",
        )
        attach_candidate_dependencies(ledger, first)
        hypothesis_id = first.decision_critical_proposition_ids[0]
        second = _epistemic_candidate()
        attach_candidate_dependencies(ledger, second)
        apply_side_premise_audit(ledger, [second], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": "deontological",
                "claim": "the medical danger could cause lasting bodily injury",
                "binding": hypothesis_id,
                "derived_from": ["PROP:WORLD:E1"],
                "decision_critical": True,
                "source_field": "rationale",
                "reason": "this is the existing unresolved injury-risk premise",
            }],
            "error": "",
        })
        self.assertEqual(
            second.decision_critical_proposition_ids, [hypothesis_id],
        )
        self.assertEqual(len([
            record for record in ledger.values()
            if record.proposition_type == "HYPOTHESIS"
        ]), 1)

    def test_unavailable_side_audit_cannot_leave_unconditional_authority(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        candidate = _epistemic_candidate()
        attach_candidate_dependencies(ledger, candidate)
        apply_side_premise_audit(ledger, [candidate], {
            "status": "UNAVAILABLE", "findings": [], "error": "provider timeout",
        })
        profile = apply_specialist_authority(candidate)
        self.assertEqual(candidate.side_premise_audit_status, "UNAVAILABLE")
        self.assertEqual(profile.adjudication_status, "CONDITIONAL_SUPPORTS")
        self.assertLessEqual(candidate.epistemic_confidence, 0.5)

    def test_batched_auditor_receives_decision_fields_and_typed_ledger(self):
        observed: dict[str, str] = {}

        class AuditLlm:
            def complete_json(self, prompt, *, schema, **kwargs):
                observed["prompt"] = prompt
                observed["schema"] = schema
                return {"choices": [{"text": '{"findings":[]}'}]}

        candidate = _epistemic_candidate()
        candidate.rationale = "A stringent rescue duty applies to the emergency."
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        result = run_side_premise_audit(
            AuditLlm(), ledger_projection(ledger), [candidate],
        )
        self.assertEqual(result["status"], "PASSED")
        self.assertIn("medical emergencies are prevented", observed["prompt"])
        self.assertIn("stringent rescue duty", observed["prompt"])
        self.assertIn("semantic content,\nnot keywords", observed["prompt"])

    def test_unknown_dependency_ids_invalidate_candidate(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        candidate = _epistemic_candidate()
        candidate.supporting_proposition_ids = ["PROP:UNKNOWN"]
        attach_candidate_dependencies(ledger, candidate)
        self.assertFalse(candidate.schema_valid)
        self.assertTrue(any(
            "unknown proposition IDs" in error for error in candidate.validation_errors
        ))

    def test_engine_correlates_support_without_promoting_hypothesis(self):
        class HypothesisSpecialist:
            def __init__(self, name: str):
                self.name = name

            def evaluate(self, scenario, actions, broadcast):
                return CandidateChunk(
                    specialist=self.name, constraint="IMMINENT_HARM",
                    action_scores={actions[0]: 0.8, actions[1]: 0.2},
                    surprise=0.5, friction=0.5, confidence=0.9,
                    epistemic_confidence=0.9, recommended_action=actions[0],
                    rationale="Possible mortality makes intervention urgent.",
                    decision_rule="Prefer intervention under urgent risk.",
                    speculative_claim="the emergency carries substantial mortality risk",
                    evidence_calibration_tier="DECISION_CRITICAL",
                    adjudication_status="SUPPORTS", governing_eligible=True,
                )

        engine = WorkspaceEngine(
            [HypothesisSpecialist("duty"), HypothesisSpecialist("care")],
            WorkspaceConfig(
                max_cycles=1, stable_cycles_required=1, enable_synthesis=False,
                enable_planning=False, enable_consensus_audit=False,
                enable_problem_state_audit=False, enable_reversal_audit=False,
            ),
        )
        result = engine.run(
            "A treatment prevents an emergency, but mortality is not specified.",
            ["provide the treatment", "withhold the treatment"],
        )
        hypotheses = [
            row for row in result.proposition_ledger
            if row["proposition_type"] == "HYPOTHESIS"
        ]
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0]["mention_count"], 2)
        self.assertEqual(hypotheses[0]["epistemic_status"], "HYPOTHETICAL")
        hypothesis_id = hypotheses[0]["proposition_id"]
        self.assertIn(
            hypothesis_id, result.cycles[0].broadcast.focus_proposition_ids,
        )
        self.assertEqual(
            result.shared_unresolved_dependencies[0]["dependent_specialist_count"],
            2,
        )
        self.assertTrue(all(
            candidate.adjudication_status == "CONDITIONAL_SUPPORTS"
            for candidate in result.cycles[0].candidates
        ))
        self.assertIn(
            "recurrence does not provide independent factual support",
            render_decision_brief(result),
        )

    def test_engine_side_audit_caps_multiple_undeclared_dependencies(self):
        class OmissiveSpecialist:
            def __init__(self, name: str):
                self.name = name

            def evaluate(self, scenario, actions, broadcast):
                return CandidateChunk(
                    specialist=self.name, constraint="IMMINENT_HARM",
                    action_scores={actions[0]: 0.8, actions[1]: 0.2},
                    surprise=0.5, friction=0.5, confidence=0.9,
                    epistemic_confidence=0.9, recommended_action=actions[0],
                    rationale="The intervention prevents irreversible organ failure.",
                    decision_rule="Prefer intervention to prevent organ failure.",
                    adjudication_status="SUPPORTS", governing_eligible=True,
                )

        def audit(_ledger, candidates):
            return {
                "status": "FINDINGS", "error": "",
                "findings": [{
                    "specialist": candidate.specialist,
                    "claim": "the emergency would otherwise cause organ failure",
                    "binding": "NEW_HYPOTHESIS",
                    "derived_from": [],
                    "decision_critical": True,
                    "source_field": "decision_rule",
                    "reason": "the specific outcome exceeds the admitted emergency claim",
                } for candidate in candidates],
            }

        result = WorkspaceEngine(
            [OmissiveSpecialist("duty"), OmissiveSpecialist("care")],
            WorkspaceConfig(
                max_cycles=1, stable_cycles_required=1, enable_synthesis=False,
                enable_planning=False, enable_consensus_audit=False,
                enable_problem_state_audit=False, enable_reversal_audit=False,
            ),
        ).run(
            "A treatment prevents an emergency, but its severity is unspecified.",
            ["provide the treatment", "withhold the treatment"],
            audit_side_premises=audit,
        )
        candidates = result.cycles[0].candidates
        self.assertEqual(result.active_specialists, ["duty", "care"])
        self.assertTrue(all(
            candidate.adjudication_status == "CONDITIONAL_SUPPORTS"
            for candidate in candidates
        ))
        self.assertTrue(all(
            candidate.side_premise_audit_status == "FINDINGS"
            for candidate in candidates
        ))
        self.assertEqual(
            result.shared_unresolved_dependencies[0]["dependent_specialist_count"],
            2,
        )
        self.assertEqual(result.side_premise_audits[0]["status"], "FINDINGS")
        brief = render_decision_brief(result)
        self.assertIn("organ failure", brief)
        self.assertIn("unestablished proposition", brief)
        self.assertIn("Status: hypothetical", brief)
        self.assertIn("Conditional on [hypothetical:", brief)


if __name__ == "__main__":
    unittest.main()
