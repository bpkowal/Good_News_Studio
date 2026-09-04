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
from global_workspace.utilitarian_ledger import (
    EffectValuationProposal,
    apply_utilitarian_ledger_transaction,
    utilitarian_scored_grounded_effects,
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
    admit_world_model_extension,
    classify_clause_role,
    compact_committed_world,
    explicit_likelihood_spans,
    explicit_quantity_spans,
    explicit_scope_spans,
    explicit_temporal_spans,
    parse_world_model,
    project_world_action_roles,
    utilitarian_omits_foregone_dual,
    validate_world_completeness,
    validate_world_model,
    world_model_from_dict,
)
from global_workspace.action_identity import build_canonical_action_records
from global_workspace.graph_transactions import SemanticGraphStore
from global_workspace.local_specialists import (
    CompactLocalSpecialist,
    _candidate_from_data,
    ground_actions_in_scenario,
)
from global_workspace.models import CandidateChunk, WorkspaceBroadcast
from global_workspace.engine import WorkspaceConfig, WorkspaceEngine
from global_workspace.deliberative_state import (
    build_deliberative_problem_state,
    opening_problem_state,
)
from global_workspace.epistemic_ledger import (
    apply_side_premise_audit,
    attach_candidate_dependencies,
    ledger_projection,
    PropositionRecord,
    register_framework_derived_proposition,
    register_hypothesis,
    resolve_proposition,
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
    effect_kind: str = "OTHER",
    likelihood_qualifiers: tuple[str, ...] = (),
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
        effect_kind=effect_kind,
        condition_ids=condition_ids,
        quantities=quantities,
        provenance=provenance,
        likelihood_qualifiers=likelihood_qualifiers,
    )


def _model(
    effects: tuple[WorldEffect, ...],
    *,
    conditions: tuple[WorldCondition, ...] = (),
    links: tuple[CausalLink, ...] = (),
    extra_parties: tuple[WorldParty, ...] = (),
) -> ScenarioWorldModel:
    parties = (
        WorldParty("P0", "dispatcher", "SYSTEM", REF),
        WorldParty("P1", "engineer", "PERSON", REF),
        WorldParty("P2", "trapped family", "GROUP", REF),
        WorldParty("P3", "downstream residents", "POPULATION", SCALE_REF),
        *extra_parties,
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

    def test_longest_compound_span_wins_over_nested_cardinals(self):
        self.assertEqual(
            explicit_quantity_spans("over five hundred residents"),
            ("over five hundred",),
        )
        self.assertEqual(
            explicit_quantity_spans("five hundred residents"),
            ("five hundred",),
        )
        self.assertEqual(
            explicit_quantity_spans("five residents"),
            ("five",),
        )
        self.assertEqual(explicit_quantity_spans("4 patients"), ("4",))
        self.assertEqual(
            explicit_quantity_spans("a hundred residents"),
            ("a hundred",),
        )

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
        self.assertEqual(
            validate_world_completeness(model, action_ids=["A0", "A1"]),
            [],
        )

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

    def test_schema_1_2_cross_action_foregoes_migrates_instead_of_entering_the_causal_graph(self):
        serialized = self.atomic_model().as_dict()
        serialized["schema_version"] = "1.2"
        serialized["counterfactual_links"] = []
        serialized["causal_links"] = [*serialized["causal_links"], {
            "action_id": "A0",
            "source_id": "A0_FOREGONE",
            "relation": "FOREGOES_ALTERNATIVE_EFFECT",
            "target_id": "A1_RELIEF",
            "modality": "CERTAIN",
            "condition_ids": (),
            "provenance": tuple(ref.as_dict() for ref in REF),
        }]
        restored = world_model_from_dict(serialized)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertFalse(any(
            {link.source_id, link.target_id} == {"A0_FOREGONE", "A1_RELIEF"}
            for link in restored.causal_links
        ))
        self.assertTrue(any(
            link.action_id == "A0"
            and link.source_effect_id == "A0_FOREGONE"
            and link.alternative_effect_id == "A1_RELIEF"
            for link in restored.counterfactual_links
        ))

    def test_foregone_effect_cannot_be_a_causal_endpoint(self):
        model = self.atomic_model()
        invalid = ScenarioWorldModel(
            parties=model.parties, actions=model.actions, effects=model.effects,
            causal_links=model.causal_links + (
                CausalLink(
                    "A0_RESOURCE", "CAUSES", "A0_FOREGONE", "CERTAIN", (), REF, "A0",
                ),
            ),
            counterfactual_links=model.counterfactual_links,
            schema_version="1.2",
        )
        errors, _ = validate_world_model(invalid, action_ids=["A0", "A1"])
        self.assertTrue(any("FOREGONE effect" in error for error in errors), errors)
        self.assertEqual(
            validate_world_completeness(invalid, action_ids=["A0", "A1"]),
            [],
        )

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

    def test_utilitarian_keeps_foregone_when_the_party_has_no_actual_outcome(self):
        model = self.atomic_model()
        self.assertFalse(utilitarian_omits_foregone_dual(
            next(effect for effect in model.effects if effect.effect_id == "A0_FOREGONE"),
            model,
        ))
        graph = compile_scenario_graph(
            "A system routes one resource between two recipients.",
            ["route resource to facility", "route resource to urgent recipients"],
            world_model=model.as_dict(),
        )
        scored_ids = {
            effect.effect_id for effect in utilitarian_scored_grounded_effects(graph)
        }
        self.assertIn("A0_FOREGONE", scored_ids)
        self.assertIn("A1_FOREGONE", scored_ids)

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
        self.assertEqual(final.adjudication_status, "SUPPORTS")
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
        self.assertIn("Admitted ranking stands", qualification)
        self.assertIn("Reversal boundary", qualification)
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
        self.assertEqual(proposition.epistemic_type, "WORLD_ESTABLISHED")
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

    def test_decision_critical_hypothesis_does_not_unsettle_closed_world_ranking(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        candidate = _epistemic_candidate(
            claim="the medical emergencies carry substantial mortality risk",
            tier="DECISION_CRITICAL",
        )
        candidate.supporting_proposition_ids = ["PROP:WORLD:E1"]
        attach_candidate_dependencies(ledger, candidate)
        profile = apply_specialist_authority(candidate)
        self.assertEqual(candidate.weakest_decision_critical_status, "HYPOTHETICAL")
        self.assertEqual(profile.adjudication_status, "SUPPORTS")
        self.assertEqual(candidate.selection_status, "SELECTED")
        self.assertTrue(candidate.comparison_complete)
        self.assertLessEqual(candidate.epistemic_confidence, 0.70)
        self.assertGreater(candidate.action_scores["provide treatment"], 0.5)
        self.assertEqual(candidate.unresolved, "DECISION_BOUNDARY")
        self.assertNotEqual(candidate.factual_reversal_threshold.casefold(), "none")
        self.assertTrue(candidate.governing_eligible)
        self.assertNotIn("provided that", candidate.decision_rule.casefold())

    def test_unverified_downstream_hypothesis_does_not_move_closed_world_scores(self):
        actions = ["open the spillway", "keep the spillway closed"]
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        hypothesis_id = register_hypothesis(
            ledger,
            "opening the spillway later causes more downstream deaths than the admitted margin",
            specialist="utilitarian",
            decision_critical=True,
        )
        candidate = CandidateChunk(
            specialist="utilitarian", constraint="WELFARE",
            action_scores={actions[0]: 0.25, actions[1]: 0.75},
            surprise=0.4, friction=0.5, confidence=0.9,
            epistemic_confidence=0.9, recommended_action=actions[1],
            rationale="Unverified later deaths appear to outweigh the admitted flood.",
            decision_rule="Prefer keeping the spillway closed if later deaths dominate.",
            adjudication_status="SUPPORTS", governing_eligible=True,
            comparison_complete=True, selection_status="SELECTED",
            utilitarian_decision_depends_on_unknown=True,
            utilitarian_missing_comparison="later downstream deaths versus the admitted flood margin",
            utilitarian_consequence_table={
                actions[0]: [{
                    "outcome": "one operator is injured", "scope": "operator",
                    "direction": "HARM", "probability": "100%", "magnitude": "1",
                    "duration": "immediate", "reversibility": "IRREVERSIBLE",
                    "support": "STATED",
                }],
                actions[1]: [{
                    "outcome": "five hundred residents drown", "scope": "residents",
                    "direction": "HARM", "probability": "100%", "magnitude": "500",
                    "duration": "immediate", "reversibility": "IRREVERSIBLE",
                    "support": "STATED",
                }],
            },
            supporting_proposition_ids=["PROP:WORLD:E1", hypothesis_id],
            decision_critical_proposition_ids=[hypothesis_id],
            material_empirical_claims=[{
                "claim": "opening the spillway later causes more downstream deaths than the admitted margin",
                "proposition_id": hypothesis_id,
                "decision_critical": True,
            }],
        )
        attach_candidate_dependencies(ledger, candidate)
        profile = apply_specialist_authority(candidate)
        self.assertEqual(candidate.recommended_action, actions[0])
        self.assertGreater(
            candidate.action_scores[actions[0]], candidate.action_scores[actions[1]],
        )
        self.assertFalse(candidate.utilitarian_decision_depends_on_unknown)
        self.assertTrue(candidate.comparison_complete)
        self.assertNotEqual(candidate.assumption_status, "UNDERDETERMINED")
        self.assertEqual(candidate.unresolved, "DECISION_BOUNDARY")
        self.assertEqual(profile.adjudication_status, "SUPPORTS")
        self.assertTrue(candidate.governing_eligible)
        self.assertIn("unestablished", candidate.factual_reversal_threshold.casefold())
        qualification = _candidate_epistemic_qualification(
            {
                "proposition_ledger": ledger_projection(ledger),
            },
            {
                "specialist": "utilitarian",
                "comparison_complete": candidate.comparison_complete,
                "decision_critical_proposition_ids": candidate.decision_critical_proposition_ids,
                "supporting_proposition_ids": candidate.supporting_proposition_ids,
            },
        )
        self.assertIn("Admitted ranking stands", qualification)
        self.assertIn("Reversal boundary", qualification)

    def test_unknown_parameter_still_blocks_closed_world_comparison(self):
        ledger = seed_proposition_ledger(_epistemic_grounded_graph())
        ledger["PROP:WORLD:CONDITION:failure-odds"] = PropositionRecord(
            proposition_id="PROP:WORLD:CONDITION:failure-odds",
            claim="the probability of pump failure remains unknown",
            proposition_type="DESCRIPTIVE",
            epistemic_status="UNRESOLVED",
            support_ids=["failure-odds"],
            introduced_by="WORLD_MODEL",
            epistemic_type="UNKNOWN_PARAMETER",
        )
        candidate = _epistemic_candidate()
        candidate.supporting_proposition_ids = ["PROP:WORLD:CONDITION:failure-odds"]
        candidate.decision_critical_proposition_ids = ["PROP:WORLD:CONDITION:failure-odds"]
        attach_candidate_dependencies(ledger, candidate)
        profile = apply_specialist_authority(candidate)
        self.assertEqual(candidate.weakest_decision_critical_status, "UNRESOLVED")
        self.assertEqual(candidate.assumption_status, "UNDERDETERMINED")
        self.assertFalse(candidate.comparison_complete)
        self.assertEqual(candidate.selection_status, "PROVISIONAL")
        self.assertEqual(profile.adjudication_status, "CONDITIONAL_SUPPORTS")

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
        self.assertLessEqual(candidate.epistemic_confidence, 0.70)
        self.assertTrue(candidate.comparison_complete)

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
        self.assertLessEqual(candidate.epistemic_confidence, 0.70)
        self.assertTrue(candidate.comparison_complete)

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
            candidate.adjudication_status == "SUPPORTS"
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
            candidate.adjudication_status == "SUPPORTS"
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
        self.assertIn("Reversal boundary [hypothetical:", brief)


MAGISTRATE_CLAUSES = [
    {
        "clause_id": "C0",
        "text": (
            "A city magistrate must decide whether to frame and execute an "
            "innocent person to quell an escalating riot during a severe "
            "municipal crisis."
        ),
    },
    {
        "clause_id": "C1",
        "text": (
            "A0 refuses to execute the innocent individual, strictly upholding "
            "the inviolable duty against framing and killing an innocent citizen, "
            "but allowing the riot to spread and predictably result in the deaths "
            "of over five hundred residents."
        ),
    },
    {
        "clause_id": "C2",
        "text": (
            "A1 frames and executes the innocent individual, satisfying the mob's "
            "demand for retribution, immediately halting the violence, and "
            "preserving the lives of over five hundred citizens at the cost of one."
        ),
    },
]
A0_TEXT = (
    "refuse to frame and execute the innocent individual, allowing the riot "
    "to spread and predictably kill over five hundred residents"
)
A1_TEXT = (
    "frame and execute the innocent individual to satisfy the mob, immediately "
    "halting the riot and preserving the lives of over five hundred residents"
)


def _raw_effect(**fields: object) -> dict[str, object]:
    row: dict[str, object] = {
        "condition_ids": [],
        "quantities": [],
        "likelihood_qualifiers": [],
        "scope_qualifiers": [],
        "temporal_qualifiers": [],
        "clause_ids": ["C1"],
        "relation": "EXPERIENCES",
        "modality": "CERTAIN",
    }
    row.update(fields)
    return row


def _raw_link(
    action_id: str, source: str, target: str, clause: str = "C1",
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "source_id": source,
        "relation": "CAUSES",
        "target_id": target,
        "modality": "CERTAIN",
        "condition_ids": [],
        "clause_ids": [clause],
    }


def _complete_magistrate_raw() -> dict[str, object]:
    return {
        "schema_version": "1.2",
        "parties": [
            {
                "party_id": "P0", "label": "city magistrate", "kind": "HUMAN",
                "quantities": [], "clause_ids": ["C0"],
            },
            {
                "party_id": "P1", "label": "innocent individual", "kind": "HUMAN",
                "quantities": ["one"], "clause_ids": ["C0", "C1", "C2"],
            },
            {
                "party_id": "P2", "label": "mob", "kind": "GROUP",
                "quantities": [], "clause_ids": ["C0", "C2"],
            },
            {
                "party_id": "P3", "label": "city public order", "kind": "INSTITUTION",
                "quantities": [], "clause_ids": ["C0", "C1", "C2"],
            },
            {
                "party_id": "P4", "label": "city residents", "kind": "GROUP",
                "quantities": ["over five hundred"], "clause_ids": ["C1", "C2"],
            },
        ],
        "actions": [
            {
                "action_id": "A0",
                "intervention": "refuse to frame and execute",
                "actor_party_id": "P0",
                "recipient_party_ids": ["P1"],
                "effect_ids": ["E0", "E1", "E1f", "E2", "E3", "E4", "EF0"],
                "clause_ids": ["C1"],
            },
            {
                "action_id": "A1",
                "intervention": "frame and execute innocent individual",
                "actor_party_id": "P0",
                "recipient_party_ids": ["P1"],
                "effect_ids": ["E5", "E6", "E7", "E8", "E9", "E10", "EF1"],
                "clause_ids": ["C2"],
            },
        ],
        "effects": [
            _raw_effect(
                effect_id="E0", action_id="A0", party_id="P0",
                outcome="refusal carried out", relation="PERFORMS",
                polarity="NEUTRAL", directness="DIRECT",
                effect_kind="INTERVENTION",
            ),
            _raw_effect(
                effect_id="E1", action_id="A0", party_id="P1",
                outcome="not executed", relation="IS",
                polarity="BENEFICIAL", directness="DIRECT",
                effect_kind="INTERVENTION",
            ),
            _raw_effect(
                effect_id="E1f", action_id="A0", party_id="P1",
                outcome="not framed", relation="IS",
                polarity="BENEFICIAL", directness="DIRECT",
                effect_kind="INSTITUTIONAL_OUTCOME",
            ),
            _raw_effect(
                effect_id="E2", action_id="A0", party_id="P1",
                outcome="survival", polarity="BENEFICIAL",
                directness="DOWNSTREAM", effect_kind="HEALTH_OUTCOME",
            ),
            _raw_effect(
                effect_id="E3", action_id="A0", party_id="P3",
                outcome="riot continues", relation="IS",
                polarity="ADVERSE", directness="DOWNSTREAM",
                effect_kind="PHYSICAL_STATE",
            ),
            _raw_effect(
                effect_id="E4", action_id="A0", party_id="P4",
                outcome="killed", polarity="ADVERSE",
                directness="DOWNSTREAM", effect_kind="HEALTH_OUTCOME",
                quantities=["over five hundred"],
            ),
            _raw_effect(
                effect_id="EF0", action_id="A0", party_id="P4",
                outcome="lives preserved", relation="DOES_NOT_EXPERIENCE",
                polarity="FOREGONE", directness="FOREGONE",
                effect_kind="OPPORTUNITY_LOSS",
                quantities=["over five hundred"], clause_ids=["C1", "A0"],
            ),
            _raw_effect(
                effect_id="E5", action_id="A1", party_id="P0",
                outcome="execution order carried out", relation="PERFORMS",
                polarity="NEUTRAL", directness="DIRECT",
                effect_kind="INTERVENTION", clause_ids=["C2"],
            ),
            _raw_effect(
                effect_id="E6", action_id="A1", party_id="P1",
                outcome="falsely framed", relation="IS",
                polarity="ADVERSE", directness="DIRECT",
                effect_kind="INSTITUTIONAL_OUTCOME", clause_ids=["C2"],
            ),
            _raw_effect(
                effect_id="E7", action_id="A1", party_id="P1",
                outcome="subjected to execution", relation="SUBJECT_TO",
                polarity="NEUTRAL", directness="DIRECT",
                effect_kind="INTERVENTION", clause_ids=["C2"],
                quantities=["one"],
            ),
            _raw_effect(
                effect_id="E8", action_id="A1", party_id="P1",
                outcome="executed", relation="SUBJECT_TO",
                polarity="ADVERSE", directness="DOWNSTREAM",
                effect_kind="HEALTH_OUTCOME", clause_ids=["C2"],
                quantities=["one"],
            ),
            _raw_effect(
                effect_id="E9", action_id="A1", party_id="P3",
                outcome="riot halted", relation="STATE_CHANGE",
                polarity="BENEFICIAL", directness="DOWNSTREAM",
                effect_kind="PHYSICAL_STATE", clause_ids=["C2"],
                temporal_qualifiers=["immediately"],
            ),
            _raw_effect(
                effect_id="E10", action_id="A1", party_id="P4",
                outcome="lives preserved", relation="BENEFITS",
                polarity="BENEFICIAL", directness="DOWNSTREAM",
                effect_kind="HEALTH_OUTCOME", clause_ids=["C2"],
                quantities=["over five hundred"],
            ),
            _raw_effect(
                effect_id="EF1", action_id="A1", party_id="P4",
                outcome="killed", relation="FOREGONE",
                polarity="FOREGONE", directness="FOREGONE",
                effect_kind="OPPORTUNITY_LOSS", clause_ids=["C2", "A1"],
                quantities=["over five hundred"],
            ),
        ],
        "conditions": [],
        "causal_links": [
            _raw_link("A0", "E0", "E3"),
            _raw_link("A0", "E3", "E4"),
            _raw_link("A0", "E0", "E1"),
            _raw_link("A0", "E0", "E1f"),
            _raw_link("A0", "E1", "E2"),
            _raw_link("A1", "E5", "E6", "C2"),
            _raw_link("A1", "E6", "E7", "C2"),
            _raw_link("A1", "E7", "E8", "C2"),
            _raw_link("A1", "E8", "E9", "C2"),
            _raw_link("A1", "E9", "E10", "C2"),
        ],
        "counterfactual_links": [
            {
                "action_id": "A0", "source_effect_id": "EF0",
                "relation": "FOREGOES_ALTERNATIVE_EFFECT",
                "alternative_action_id": "A1", "alternative_effect_id": "E10",
                "modality": "CERTAIN", "condition_ids": [],
                "clause_ids": ["A0", "C2"],
            },
            {
                "action_id": "A1", "source_effect_id": "EF1",
                "relation": "FOREGOES_ALTERNATIVE_EFFECT",
                "alternative_action_id": "A0", "alternative_effect_id": "E4",
                "modality": "CERTAIN", "condition_ids": [],
                "clause_ids": ["A1", "C1"],
            },
        ],
    }


def _parse_complete_magistrate():
    return parse_world_model(
        _complete_magistrate_raw(),
        clauses=MAGISTRATE_CLAUSES,
        action_ids=["A0", "A1"],
        action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
    )


class WorldModelCompletenessTests(unittest.TestCase):
    def test_complete_magistrate_model_admits(self):
        model = _parse_complete_magistrate()
        errors, contradictions = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(errors, [])
        self.assertEqual(contradictions, [])
        self.assertEqual(validate_world_completeness(model, action_ids=["A0", "A1"]), [])

    def test_roles_count_downstream_health_not_the_mob(self):
        model = _parse_complete_magistrate()
        a0 = project_world_action_roles(model, "A0")
        a1 = project_world_action_roles(model, "A1")
        self.assertEqual(a0.beneficiaries, ("innocent individual",))
        self.assertEqual(a0.harmed, ("city residents",))
        self.assertEqual(a1.harmed, ("innocent individual",))
        self.assertEqual(a1.beneficiaries, ("city residents",))

    def test_canonical_records_project_roles_and_framing(self):
        records = build_canonical_action_records(
            [A0_TEXT, A1_TEXT],
            actor="city magistrate",
            world_model=_parse_complete_magistrate().as_dict(),
        )
        by_id = {record.action_id: record for record in records}
        self.assertEqual(by_id["A0"].beneficiaries, ("innocent individual",))
        self.assertEqual(by_id["A0"].harmed, ("city residents",))
        self.assertEqual(by_id["A1"].harmed, ("innocent individual",))
        self.assertEqual(by_id["A1"].beneficiaries, ("city residents",))
        self.assertIn("falsely framed", by_id["A1"].institutional_effect)

    def test_refusal_without_patient_or_intermediate_is_incomplete(self):
        raw = _complete_magistrate_raw()
        raw["actions"][0]["recipient_party_ids"] = []
        raw["effects"] = [
            effect for effect in raw["effects"]
            if not (
                effect["action_id"] == "A0"
                and effect["party_id"] in {"P1", "P3"}
            )
        ]
        raw["causal_links"] = [
            _raw_link("A0", "E0", "E4"),
            _raw_link("A1", "E5", "E6", "C2"),
            _raw_link("A1", "E6", "E7", "C2"),
            _raw_link("A1", "E7", "E8", "C2"),
            _raw_link("A1", "E8", "E9", "C2"),
            _raw_link("A1", "E9", "E10", "C2"),
        ]
        with self.assertRaises(ValueError) as raised:
            parse_world_model(
                raw,
                clauses=MAGISTRATE_CLAUSES,
                action_ids=["A0", "A1"],
                action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
            )
        message = str(raised.exception)
        self.assertIn("patient-status effect", message)
        self.assertIn("intermediate", message)

    def test_framing_predicate_requires_juridical_effect(self):
        raw = _complete_magistrate_raw()
        raw["effects"] = [
            effect for effect in raw["effects"] if effect["effect_id"] != "E6"
        ]
        raw["causal_links"] = [
            link for link in raw["causal_links"]
            if link["source_id"] != "E5" and link["source_id"] != "E6"
        ]
        raw["causal_links"].insert(3, _raw_link("A1", "E5", "E7", "C2"))
        with self.assertRaises(ValueError) as raised:
            parse_world_model(
                raw,
                clauses=MAGISTRATE_CLAUSES,
                action_ids=["A0", "A1"],
                action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
            )
        self.assertIn("INSTITUTIONAL_OUTCOME", str(raised.exception))

    def test_refusal_requires_spared_juridical_effect(self):
        raw = _complete_magistrate_raw()
        raw["effects"] = [
            effect for effect in raw["effects"] if effect["effect_id"] != "E1f"
        ]
        raw["causal_links"] = [
            link for link in raw["causal_links"]
            if link["source_id"] != "E1f" and link["target_id"] != "E1f"
        ]
        with self.assertRaises(ValueError) as raised:
            parse_world_model(
                raw,
                clauses=MAGISTRATE_CLAUSES,
                action_ids=["A0", "A1"],
                action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
            )
        self.assertIn("INSTITUTIONAL_OUTCOME", str(raised.exception))

    def test_nested_five_is_rejected_on_admit(self):
        raw = _complete_magistrate_raw()
        raw["parties"][-1]["quantities"] = ["over five hundred", "five"]
        with self.assertRaises(ValueError) as raised:
            parse_world_model(
                raw,
                clauses=MAGISTRATE_CLAUSES,
                action_ids=["A0", "A1"],
                action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
            )
        self.assertIn("nested quantity", str(raised.exception))

    def test_neutral_intervention_does_not_contradict_survival(self):
        model = _parse_complete_magistrate()
        _errors, contradictions = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(contradictions, [])

    def test_restore_skips_completeness(self):
        raw = _complete_magistrate_raw()
        raw["parties"][-1]["quantities"] = ["over five hundred", "five"]
        model = parse_world_model(
            raw,
            clauses=MAGISTRATE_CLAUSES,
            action_ids=["A0", "A1"],
            action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
            require_completeness=False,
        )
        self.assertIn("five", model.parties[-1].quantities)

    def test_restore_projects_downstream_roles_on_incomplete_a0(self):
        raw = _complete_magistrate_raw()
        raw["actions"][0]["recipient_party_ids"] = []
        raw["effects"] = [
            effect for effect in raw["effects"]
            if not (
                effect["action_id"] == "A0"
                and effect["party_id"] in {"P1", "P3"}
            )
        ]
        raw["causal_links"] = [
            _raw_link("A0", "E0", "E4"),
            _raw_link("A1", "E5", "E6", "C2"),
            _raw_link("A1", "E6", "E7", "C2"),
            _raw_link("A1", "E7", "E8", "C2"),
            _raw_link("A1", "E8", "E9", "C2"),
            _raw_link("A1", "E9", "E10", "C2"),
        ]
        model = parse_world_model(
            raw,
            clauses=MAGISTRATE_CLAUSES,
            action_ids=["A0", "A1"],
            action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
            require_completeness=False,
        )
        a0 = project_world_action_roles(model, "A0")
        a1 = project_world_action_roles(model, "A1")
        self.assertEqual(a0.beneficiaries, ())
        self.assertEqual(a0.harmed, ("city residents",))
        self.assertEqual(a1.harmed, ("innocent individual",))
        self.assertEqual(a1.beneficiaries, ("city residents",))


class CompactActionRoleTests(unittest.TestCase):
    def test_certain_physical_state_on_population_is_compact_benefit(self):
        model = _model((
            _effect(
                "E_w", "A0", "P3",
                outcome="safe water preserved",
                polarity="BENEFICIAL",
                directness="DOWNSTREAM",
                effect_kind="PHYSICAL_STATE",
            ),
        ))
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.beneficiaries, ("downstream residents",))
        self.assertEqual(roles.harmed, ())
        self.assertEqual(roles.unresolved, ())

    def test_facility_physical_state_is_not_a_compact_role(self):
        model = _model(
            (
                _effect(
                    "E_f", "A0", "P4",
                    outcome="left without backup power",
                    polarity="ADVERSE",
                    directness="DIRECT",
                    effect_kind="PHYSICAL_STATE",
                ),
            ),
            extra_parties=(WorldParty("P4", "field clinic", "FACILITY", REF),),
        )
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.beneficiaries, ())
        self.assertEqual(roles.harmed, ())
        self.assertEqual(roles.unresolved, ())

    def test_near_certain_probabilistic_health_is_compact_harm(self):
        model = _model(
            (
                _effect(
                    "E_h", "A0", "P2",
                    outcome="fatal equipment failure causes death",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    modality="PROBABILISTIC",
                    condition_ids=("COND1",),
                    effect_kind="HEALTH_OUTCOME",
                    likelihood_qualifiers=("near-certain",),
                ),
            ),
            conditions=(WorldCondition("COND1", "equipment fails", provenance=REF),),
        )
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.harmed, ("trapped family",))
        self.assertEqual(roles.unresolved, ())

    def test_unqualified_probabilistic_institutional_stays_unresolved(self):
        model = _model(
            (
                _effect(
                    "E_i", "A0", "P3",
                    outcome="public-health crisis risk increased",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    modality="PROBABILISTIC",
                    condition_ids=("COND2",),
                    effect_kind="INSTITUTIONAL_OUTCOME",
                ),
            ),
            conditions=(WorldCondition("COND2", "sanitation fails", provenance=REF),),
        )
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.harmed, ())
        self.assertEqual(roles.beneficiaries, ())
        self.assertEqual(roles.unresolved, ("downstream residents",))

    def test_settled_harm_outranks_weaker_unresolved_on_same_party(self):
        model = _model(
            (
                _effect(
                    "E_e", "A0", "P2",
                    outcome="exposed to equipment failure",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    effect_kind="PHYSICAL_STATE",
                ),
                _effect(
                    "E_h", "A0", "P2",
                    outcome="later deaths remain possible",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    modality="PROBABILISTIC",
                    condition_ids=("COND1",),
                    effect_kind="HEALTH_OUTCOME",
                    likelihood_qualifiers=("likely",),
                ),
            ),
            conditions=(WorldCondition("COND1", "later deaths occur", provenance=REF),),
        )
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.harmed, ("trapped family",))
        self.assertEqual(roles.unresolved, ())

    def test_foregone_physical_state_stays_out_of_compact_roles(self):
        model = _model((
            _effect(
                "E_f", "A0", "P3",
                outcome="safe water foregone",
                relation="FOREGOES",
                polarity="FOREGONE",
                directness="FOREGONE",
                effect_kind="OPPORTUNITY_LOSS",
            ),
        ))
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.beneficiaries, ())
        self.assertEqual(roles.harmed, ())
        self.assertEqual(roles.unresolved, ())


ALLOCATOR_A0 = (
    "send the crew to Site A, reach the engineer, and hold the spillway"
)
ALLOCATOR_A1 = (
    "send the crew to Site B and reach the trapped family"
)


def _connected_allocator_raw() -> dict[str, object]:
    return {
        "schema_version": "1.2",
        "parties": [
            {
                "party_id": "P0", "label": "dispatcher", "kind": "SYSTEM",
                "quantities": [], "clause_ids": ["C0"],
            },
            {
                "party_id": "P1", "label": "engineer", "kind": "PERSON",
                "quantities": [], "clause_ids": ["C0"],
            },
            {
                "party_id": "P2", "label": "trapped family", "kind": "GROUP",
                "quantities": [], "clause_ids": ["C2"],
            },
            {
                "party_id": "P3", "label": "downstream residents",
                "kind": "POPULATION", "quantities": ["tens of thousands"],
                "clause_ids": ["C1"],
            },
            {
                "party_id": "P4", "label": "spillway", "kind": "FACILITY",
                "quantities": [], "clause_ids": ["C1"],
            },
        ],
        "actions": [
            {
                "action_id": "A0", "intervention": "send crew to Site A",
                "actor_party_id": "P0", "recipient_party_ids": ["P1"],
                "effect_ids": ["E0", "E1", "E2"], "clause_ids": ["C0"],
            },
            {
                "action_id": "A1", "intervention": "send crew to Site B",
                "actor_party_id": "P0", "recipient_party_ids": ["P2"],
                "effect_ids": ["E3", "E4"], "clause_ids": ["C4"],
            },
        ],
        "effects": [
            _raw_effect(
                effect_id="E0", action_id="A0", party_id="P1",
                outcome="crew reaches engineer", relation="REACHES",
                polarity="NEUTRAL", directness="DIRECT",
                effect_kind="INTERVENTION", clause_ids=["C0", "A0"],
            ),
            _raw_effect(
                effect_id="E1", action_id="A0", party_id="P4",
                outcome="spillway held", relation="STATE_CHANGE",
                polarity="BENEFICIAL", directness="DOWNSTREAM",
                effect_kind="PHYSICAL_STATE", clause_ids=["C1"],
            ),
            _raw_effect(
                effect_id="E2", action_id="A0", party_id="P3",
                outcome="residents spared", relation="SURVIVES",
                polarity="BENEFICIAL", directness="DOWNSTREAM",
                effect_kind="HEALTH_OUTCOME",
                modality="STIPULATED_CONDITIONAL",
                condition_ids=["COND1"],
                quantities=["tens of thousands"],
                clause_ids=["C1"],
            ),
            _raw_effect(
                effect_id="E3", action_id="A1", party_id="P2",
                outcome="crew reaches family", relation="REACHES",
                polarity="NEUTRAL", directness="DIRECT",
                effect_kind="INTERVENTION", clause_ids=["C4", "A1"],
            ),
            _raw_effect(
                effect_id="E4", action_id="A1", party_id="P2",
                outcome="family lives", relation="SURVIVES",
                polarity="BENEFICIAL", directness="DOWNSTREAM",
                effect_kind="HEALTH_OUTCOME", clause_ids=["C2", "A1"],
            ),
        ],
        "conditions": [
            {
                "condition_id": "COND1",
                "description": "engineer is reached",
                "value_status": "UNKNOWN",
                "decision_relevance": "MATERIAL",
                "clause_ids": ["C1"],
            },
        ],
        "causal_links": [
            _raw_link("A0", "E0", "E1", "C1"),
            _raw_link("A0", "E1", "E2", "C1"),
            _raw_link("A1", "E3", "E4", "C2"),
        ],
        "counterfactual_links": [],
    }


def _parse_allocator(raw: dict[str, object] | None = None):
    return parse_world_model(
        raw or _connected_allocator_raw(),
        clauses=CLAUSES,
        action_ids=["A0", "A1"],
        action_texts={"A0": ALLOCATOR_A0, "A1": ALLOCATOR_A1},
    )


class CausalChainCompletenessTests(unittest.TestCase):
    def test_connected_intervention_process_health_admits(self):
        model = _parse_allocator()
        self.assertEqual(validate_world_completeness(model, action_ids=["A0", "A1"]), [])
        self.assertEqual(model.counterfactual_links, ())
        records = {
            record.action_id: record
            for record in build_canonical_action_records(
                [ALLOCATOR_A0, ALLOCATOR_A1],
                world_model=model.as_dict(),
            )
        }
        a0_links = {
            (link["source_id"], link["target_id"])
            for link in records["A0"].causal_links
        }
        self.assertEqual(a0_links, {("E0", "E1"), ("E1", "E2")})

    def test_unconnected_process_health_is_incomplete(self):
        raw = _connected_allocator_raw()
        raw["causal_links"] = [
            _raw_link("A0", "E1", "E2", "C1"),
            _raw_link("A1", "E3", "E4", "C2"),
        ]
        with self.assertRaises(ValueError) as raised:
            _parse_allocator(raw)
        self.assertIn("ancestry never reaches a DIRECT act", str(raised.exception))

    def test_direct_effect_cannot_attach_to_non_recipient_facility(self):
        raw = _connected_allocator_raw()
        for effect in raw["effects"]:
            if effect["effect_id"] == "E1":
                effect["directness"] = "DIRECT"
        with self.assertRaises(ValueError) as raised:
            _parse_allocator(raw)
        self.assertIn("neither the actor nor a named recipient", str(raised.exception))

    def test_other_party_health_cannot_skip_the_process(self):
        raw = _connected_allocator_raw()
        raw["effects"] = [
            effect for effect in raw["effects"] if effect["effect_id"] != "E1"
        ]
        raw["effects"].append(_raw_effect(
            effect_id="E_h", action_id="A0", party_id="P1",
            outcome="engineer lives", relation="SURVIVES",
            polarity="BENEFICIAL", directness="DOWNSTREAM",
            effect_kind="HEALTH_OUTCOME", clause_ids=["C0", "A0"],
        ))
        raw["causal_links"] = [
            _raw_link("A0", "E0", "E_h", "C0"),
            _raw_link("A0", "E_h", "E2", "C1"),
            _raw_link("A1", "E3", "E4", "C2"),
        ]
        with self.assertRaises(ValueError) as raised:
            _parse_allocator(raw)
        self.assertIn("intermediate", str(raised.exception))
        self.assertIn("E_h", str(raised.exception))

    def test_institutional_process_state_can_parent_other_party_health(self):
        raw = _connected_allocator_raw()
        for effect in raw["effects"]:
            if effect["effect_id"] == "E1":
                effect["effect_kind"] = "INSTITUTIONAL_OUTCOME"
                effect["outcome"] = "spillway held under repair order"
        model = _parse_allocator(raw)
        self.assertEqual(validate_world_completeness(model, action_ids=["A0", "A1"]), [])

    def test_person_institutional_act_cannot_parent_other_party_health(self):
        raw = _complete_magistrate_raw()
        raw["causal_links"] = [
            link for link in raw["causal_links"]
            if not (link["source_id"] == "E9" and link["target_id"] == "E10")
        ]
        raw["causal_links"].append(_raw_link("A1", "E6", "E10", "C2"))
        with self.assertRaises(ValueError) as raised:
            parse_world_model(
                raw,
                clauses=MAGISTRATE_CLAUSES,
                action_ids=["A0", "A1"],
                action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
            )
        message = str(raised.exception)
        self.assertIn("E10", message)
        self.assertIn("E6", message)
        self.assertIn("intermediate", message)

    def test_person_kind_satisfies_juridical_patient_gate(self):
        raw = _complete_magistrate_raw()
        for party in raw["parties"]:
            if party["party_id"] == "P1":
                party["kind"] = "PERSON"
        model = parse_world_model(
            raw,
            clauses=MAGISTRATE_CLAUSES,
            action_ids=["A0", "A1"],
            action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
        )
        self.assertEqual(validate_world_completeness(model, action_ids=["A0", "A1"]), [])

    def test_institutional_direct_on_recipient_is_an_atomic_act(self):
        raw = _connected_allocator_raw()
        for effect in raw["effects"]:
            if effect["effect_id"] == "E0":
                effect["effect_kind"] = "INSTITUTIONAL_OUTCOME"
                effect["outcome"] = "engineer ordered to the spillway"
        model = _parse_allocator(raw)
        self.assertEqual(validate_world_completeness(model, action_ids=["A0", "A1"]), [])

    def test_opposed_nonrecipient_welfare_requires_foregone_counterfactual_overlay(self):
        raw = _complete_magistrate_raw()
        raw["effects"] = [
            effect for effect in raw["effects"]
            if effect["effect_id"] not in {"EF0", "EF1"}
        ]
        raw["counterfactual_links"] = []
        with self.assertRaises(ValueError) as raised:
            parse_world_model(
                raw,
                clauses=MAGISTRATE_CLAUSES,
                action_ids=["A0", "A1"],
                action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
            )
        message = str(raised.exception)
        self.assertIn("opposed welfare", message)
        self.assertIn("counterfactual_link", message)
        self.assertIn("causal_links", message)

    def test_foregone_kind_normalizes_to_opportunity_loss(self):
        raw = _connected_allocator_raw()
        raw["effects"].append(_raw_effect(
            effect_id="EF", action_id="A0", party_id="P3",
            outcome="lives not preserved", relation="FOREGONE",
            polarity="FOREGONE", directness="FOREGONE",
            effect_kind="HEALTH_OUTCOME",
            quantities=["tens of thousands"],
            clause_ids=["C1"],
        ))
        model = _parse_allocator(raw)
        lost = next(effect for effect in model.effects if effect.effect_id == "EF")
        self.assertEqual(lost.effect_kind, "OPPORTUNITY_LOSS")

    def test_empty_predicate_defaults_when_outcome_is_present(self):
        raw = _connected_allocator_raw()
        for effect in raw["effects"]:
            if effect["effect_id"] == "E0":
                effect["relation"] = ""
        model = _parse_allocator(raw)
        e0 = next(effect for effect in model.effects if effect.effect_id == "E0")
        self.assertEqual(e0.relation, "EXPERIENCES")
        self.assertTrue(e0.outcome)

    def test_parse_copies_local_qualifiers_and_strips_invented_ones(self):
        clauses = [dict(row) for row in CLAUSES]
        clauses[1] = {
            "clause_id": "C1",
            "text": (
                "If the engineer is reached, immediately holding the spillway, "
                "and sparing tens of thousands of downstream residents."
            ),
        }
        raw = _connected_allocator_raw()
        for effect in raw["effects"]:
            if effect["effect_id"] == "E1":
                effect["temporal_qualifiers"] = []
                effect["clause_ids"] = ["C1"]
            if effect["effect_id"] == "E2":
                effect["scope_qualifiers"] = ["widespread"]
                effect["clause_ids"] = ["C1"]
        model = parse_world_model(
            raw,
            clauses=clauses,
            action_ids=["A0", "A1"],
            action_texts={"A0": ALLOCATOR_A0, "A1": ALLOCATOR_A1},
        )
        e1 = next(effect for effect in model.effects if effect.effect_id == "E1")
        e2 = next(effect for effect in model.effects if effect.effect_id == "E2")
        self.assertEqual(e1.temporal_qualifiers, ("immediately",))
        self.assertEqual(e2.scope_qualifiers, ())

    def test_temporal_qualifier_does_not_cross_list_conjuncts(self):
        ref = (SourceRef(
            "C1",
            "Send the crew, immediately holding the spillway, and sparing residents.",
        ),)
        model = replace(
            _model((
                _effect(
                    "E1", "A0", "P4",
                    outcome="spillway held",
                    polarity="BENEFICIAL",
                    directness="DOWNSTREAM",
                    effect_kind="PHYSICAL_STATE",
                    provenance=ref,
                ),
                _effect(
                    "E2", "A0", "P3",
                    outcome="residents spared",
                    polarity="BENEFICIAL",
                    directness="DOWNSTREAM",
                    effect_kind="HEALTH_OUTCOME",
                    provenance=ref,
                ),
            ), extra_parties=(WorldParty("P4", "spillway", "FACILITY", ref),)),
            schema_version="1.2",
        )
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(any(
            "E1 omits" in error and "immediately" in error for error in errors
        ))
        self.assertFalse(any(
            "E2 omits" in error and "immediately" in error for error in errors
        ))


class WorldModelExtensionTests(unittest.TestCase):
    def test_approved_extension_connects_downstream_health(self):
        raw = _connected_allocator_raw()
        raw["effects"] = [
            effect for effect in raw["effects"] if effect["effect_id"] != "E2"
        ]
        raw["causal_links"] = [
            _raw_link("A0", "E0", "E1", "C1"),
            _raw_link("A1", "E3", "E4", "C2"),
        ]
        base = _parse_allocator(raw)
        self.assertEqual(validate_world_completeness(base, action_ids=["A0", "A1"]), [])
        updated = admit_world_model_extension(
            base,
            {
                "effects": [
                    _raw_effect(
                        effect_id="E2", action_id="A0", party_id="P3",
                        outcome="residents spared", relation="SURVIVES",
                        polarity="BENEFICIAL", directness="DOWNSTREAM",
                        effect_kind="HEALTH_OUTCOME",
                        modality="STIPULATED_CONDITIONAL",
                        condition_ids=["COND1"],
                        quantities=["tens of thousands"],
                        clause_ids=["C1"],
                    ),
                ],
                "causal_links": [_raw_link("A0", "E1", "E2", "C1")],
            },
            clauses=CLAUSES,
            action_ids=["A0", "A1"],
            action_texts={"A0": ALLOCATOR_A0, "A1": ALLOCATOR_A1},
        )
        self.assertEqual(
            validate_world_completeness(updated, action_ids=["A0", "A1"]), [],
        )
        self.assertIn("E2", {effect.effect_id for effect in updated.effects})
        compact = compact_committed_world(updated)
        self.assertIn(
            ("E1", "E2"),
            {(link["source_id"], link["target_id"]) for link in compact["causal_links"]},
        )

    def test_unconnected_health_extension_is_rejected(self):
        raw = _connected_allocator_raw()
        raw["effects"] = [
            effect for effect in raw["effects"] if effect["effect_id"] != "E2"
        ]
        raw["causal_links"] = [
            _raw_link("A0", "E0", "E1", "C1"),
            _raw_link("A1", "E3", "E4", "C2"),
        ]
        base = _parse_allocator(raw)
        with self.assertRaises(ValueError) as raised:
            admit_world_model_extension(
                base,
                {
                    "effects": [
                        _raw_effect(
                            effect_id="E2", action_id="A0", party_id="P3",
                            outcome="residents spared", relation="SURVIVES",
                            polarity="BENEFICIAL", directness="DOWNSTREAM",
                            effect_kind="HEALTH_OUTCOME",
                            modality="STIPULATED_CONDITIONAL",
                            condition_ids=["COND1"],
                            quantities=["tens of thousands"],
                            clause_ids=["C1"],
                        ),
                    ],
                },
                clauses=CLAUSES,
                action_ids=["A0", "A1"],
                action_texts={"A0": ALLOCATOR_A0, "A1": ALLOCATOR_A1},
            )
        self.assertIn("downstream human outcome", str(raised.exception))

    def test_extension_cannot_reuse_an_effect_id(self):
        base = _parse_allocator()
        with self.assertRaises(ValueError) as raised:
            admit_world_model_extension(
                base,
                {
                    "effects": [
                        _raw_effect(
                            effect_id="E0", action_id="A0", party_id="P1",
                            outcome="duplicate", relation="IS",
                            polarity="NEUTRAL", directness="DIRECT",
                            effect_kind="INTERVENTION", clause_ids=["C0"],
                        ),
                    ],
                },
                clauses=CLAUSES,
                action_ids=["A0", "A1"],
                action_texts={"A0": ALLOCATOR_A0, "A1": ALLOCATOR_A1},
            )
        self.assertIn("append-only", str(raised.exception))

    def test_reattach_is_idempotent_and_extension_adds_one_edge(self):
        base = _parse_allocator()
        graph = compile_scenario_graph(
            " ".join(clause["text"] for clause in CLAUSES),
            [ALLOCATOR_A0, ALLOCATOR_A1],
            world_model=base.as_dict(),
        )
        before = len(graph.edges)
        attach_typed_world_model(graph, base)
        self.assertEqual(len(graph.edges), before)
        raw = _connected_allocator_raw()
        raw["effects"] = [
            effect for effect in raw["effects"] if effect["effect_id"] != "E2"
        ]
        raw["causal_links"] = [
            _raw_link("A0", "E0", "E1", "C1"),
            _raw_link("A1", "E3", "E4", "C2"),
        ]
        incomplete = _parse_allocator(raw)
        graph = compile_scenario_graph(
            " ".join(clause["text"] for clause in CLAUSES),
            [ALLOCATOR_A0, ALLOCATOR_A1],
            world_model=incomplete.as_dict(),
        )
        before = len(graph.edges)
        updated = admit_world_model_extension(
            incomplete,
            {
                "effects": [
                    _raw_effect(
                        effect_id="E2", action_id="A0", party_id="P3",
                        outcome="residents spared", relation="SURVIVES",
                        polarity="BENEFICIAL", directness="DOWNSTREAM",
                        effect_kind="HEALTH_OUTCOME",
                        modality="STIPULATED_CONDITIONAL",
                        condition_ids=["COND1"],
                        quantities=["tens of thousands"],
                        clause_ids=["C1"],
                    ),
                ],
                "causal_links": [_raw_link("A0", "E1", "E2", "C1")],
            },
            clauses=CLAUSES,
            action_ids=["A0", "A1"],
            action_texts={"A0": ALLOCATOR_A0, "A1": ALLOCATOR_A1},
        )
        attach_typed_world_model(graph, updated)
        self.assertGreater(len(graph.edges), before)
        self.assertTrue(any(
            "WORLD_EFFECT:E2" in edge.target and edge.relation == "CAUSES"
            for edge in graph.edges
        ))


class PromulgatedWorldTests(unittest.TestCase):
    def test_opening_and_later_cycles_copy_committed_causal_links(self):
        model = _parse_allocator()
        opening = opening_problem_state(
            [ALLOCATOR_A0, ALLOCATOR_A1],
            " ".join(clause["text"] for clause in CLAUSES),
            world_model=model,
        )
        links = {
            (link["source_id"], link["target_id"])
            for link in opening["committed_world"]["causal_links"]
            if link["action_id"] == "A0"
        }
        self.assertEqual(links, {("E0", "E1"), ("E1", "E2")})
        winner = CandidateChunk(
            specialist="duty", constraint="DUTY",
            action_scores={ALLOCATOR_A0: 0.7, ALLOCATOR_A1: 0.3},
            surprise=0.1, friction=0.1, confidence=0.8,
            recommended_action=ALLOCATOR_A0, schema_valid=True,
        )
        rival = CandidateChunk(
            specialist="care", constraint="CARE",
            action_scores={ALLOCATOR_A0: 0.4, ALLOCATOR_A1: 0.6},
            surprise=0.1, friction=0.1, confidence=0.7,
            recommended_action=ALLOCATOR_A1, schema_valid=True,
        )
        later = build_deliberative_problem_state(
            1, [ALLOCATOR_A0, ALLOCATOR_A1], [winner, rival],
            ALLOCATOR_A0, winner, previous_state=opening,
        )
        self.assertEqual(
            later.committed_world["causal_links"],
            opening["committed_world"]["causal_links"],
        )

    def test_committed_world_promulgates_counterfactual_links(self):
        model = _parse_complete_magistrate()
        opening = opening_problem_state(
            [A0_TEXT, A1_TEXT],
            " ".join(clause["text"] for clause in MAGISTRATE_CLAUSES),
            world_model=model,
        )
        links = {
            (
                row["action_id"],
                row["source_effect_id"],
                row["alternative_effect_id"],
            )
            for row in opening["committed_world"]["counterfactual_links"]
        }
        self.assertEqual(links, {("A0", "EF0", "E10"), ("A1", "EF1", "E4")})
        causal_endpoints = {
            endpoint
            for row in opening["committed_world"]["causal_links"]
            for endpoint in (row["source_id"], row["target_id"])
        }
        self.assertNotIn("EF0", causal_endpoints)
        self.assertNotIn("EF1", causal_endpoints)

    def test_utilitarian_drops_foregone_duals_of_actual_party_welfare(self):
        model = _parse_complete_magistrate()
        omitted = {
            effect.effect_id
            for effect in model.effects
            if utilitarian_omits_foregone_dual(effect, model)
        }
        self.assertEqual(omitted, {"EF0", "EF1"})
        innocent_foregone = replace(
            next(effect for effect in model.effects if effect.effect_id == "E2"),
            effect_id="EF_INNOCENT",
            polarity="FOREGONE",
            directness="FOREGONE",
            effect_kind="OPPORTUNITY_LOSS",
        )
        self.assertTrue(utilitarian_omits_foregone_dual(innocent_foregone, model))
        graph = compile_scenario_graph(
            " ".join(clause["text"] for clause in MAGISTRATE_CLAUSES),
            [A0_TEXT, A1_TEXT],
            world_model=model.as_dict(),
        )
        projected = {effect.effect_id for effect in project_grounded_action_effects(graph)}
        scored = {
            effect.effect_id for effect in utilitarian_scored_grounded_effects(graph)
        }
        self.assertTrue({"EF0", "EF1"} <= projected)
        self.assertNotIn("EF0", scored)
        self.assertNotIn("EF1", scored)
        self.assertIn("E4", scored)
        self.assertIn("E10", scored)
        proposal = {"actions": [
            {
                "action_id": action_id,
                "valuations": [
                    {
                        "effect_id": effect_id,
                        "importance": "HIGH",
                        "reason": "material aggregate welfare contribution",
                    }
                    for effect_id in (
                        item.effect_id
                        for item in utilitarian_scored_grounded_effects(graph)
                        if item.action_id == action_id
                    )
                ],
            }
            for action_id in ("A0", "A1")
        ]}
        transaction = apply_utilitarian_ledger_transaction(
            SemanticGraphStore(graph), proposal, cycle=1, specialist="utilitarian",
            allowed_actions=(A0_TEXT, A1_TEXT),
        )
        self.assertEqual(transaction.status, "COMMITTED", transaction.errors)
        committed_ids = {
            row["world_effect_id"]
            for row in transaction.proposal["committed_consequences"]
        }
        self.assertNotIn("EF0", committed_ids)
        self.assertNotIn("EF1", committed_ids)


class PropositionIdentityTests(unittest.TestCase):
    def _ledger(self):
        model = _parse_complete_magistrate()
        graph = compile_scenario_graph(
            " ".join(clause["text"] for clause in MAGISTRATE_CLAUSES),
            [A0_TEXT, A1_TEXT],
            world_model=model.as_dict(),
        )
        return seed_proposition_ledger(graph)

    def test_world_paraphrases_bind_to_canonical_effects(self):
        ledger = self._ledger()
        killed = resolve_proposition(ledger, "refusing will kill 500+")
        executed = resolve_proposition(ledger, "the innocent is executed")
        halted = resolve_proposition(
            ledger, "execution immediately halts the riot",
        )
        self.assertEqual(killed, "PROP:WORLD:E4")
        self.assertEqual(executed, "PROP:WORLD:E8")
        self.assertEqual(halted, "PROP:WORLD:E9")
        self.assertEqual(ledger[killed].epistemic_status, "ESTABLISHED")
        self.assertEqual(ledger[killed].epistemic_type, "WORLD_ESTABLISHED")

    def test_hypothesis_registration_aliases_world_facts(self):
        ledger = self._ledger()
        bound = register_hypothesis(
            ledger, "refusing will kill 500+",
            specialist="utilitarian", decision_critical=True,
        )
        self.assertEqual(bound, "PROP:WORLD:E4")
        self.assertIn(
            "refusing will kill 500+",
            ledger["PROP:WORLD:E4"].aliases,
        )
        self.assertFalse(any(
            record.proposition_type == "HYPOTHESIS" for record in ledger.values()
        ))

    def test_candidate_premise_rebinds_instead_of_duplicating(self):
        ledger = self._ledger()
        candidate = _epistemic_candidate()
        candidate.material_empirical_claims = [{
            "claim": "refusing will kill 500+",
            "proposition_id": "HYPOTHESIS",
            "decision_critical": True,
        }]
        attach_candidate_dependencies(ledger, candidate)
        self.assertEqual(
            candidate.decision_critical_proposition_ids, ["PROP:WORLD:E4"],
        )
        self.assertEqual(candidate.weakest_decision_critical_status, "ESTABLISHED")
        self.assertEqual(
            candidate.material_empirical_claims[0]["canonical_proposition"],
            "PROP:WORLD:E4",
        )
        self.assertFalse(any(
            record.proposition_type == "HYPOTHESIS" for record in ledger.values()
        ))

    def test_side_audit_cannot_mint_duplicate_world_paraphrase(self):
        ledger = self._ledger()
        candidate = _epistemic_candidate()
        apply_side_premise_audit(ledger, [candidate], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": "deontological",
                "claim": "the innocent is executed",
                "binding": "NEW_HYPOTHESIS",
                "derived_from": ["PROP:WORLD:E8"],
                "decision_critical": True,
                "source_field": "rationale",
                "reason": "auditor restated an admitted world effect",
            }],
            "error": "",
        })
        self.assertEqual(
            candidate.decision_critical_proposition_ids, ["PROP:WORLD:E8"],
        )
        self.assertEqual(candidate.weakest_decision_critical_status, "ESTABLISHED")
        self.assertFalse(any(
            record.proposition_type == "HYPOTHESIS" for record in ledger.values()
        ))

    def test_framework_derived_does_not_duplicate_world_facts(self):
        ledger = self._ledger()
        bound = register_framework_derived_proposition(
            ledger, "execution immediately halts the riot",
            specialist="care",
            derived_from=["PROP:WORLD:E9"],
        )
        self.assertEqual(bound, "PROP:WORLD:E9")
        relation = register_framework_derived_proposition(
            ledger,
            "magistrate has a special relation to a citizen under judicial authority",
            specialist="care",
            derived_from=["PROP:WORLD:E1"],
        )
        self.assertEqual(ledger[relation].epistemic_type, "FRAMEWORK_DERIVED")
        self.assertEqual(ledger[relation].epistemic_status, "DERIVED")
        self.assertNotEqual(relation, bound)

    def test_presentation_does_not_qualify_rebound_world_paraphrase(self):
        data = {
            "proposition_ledger": [{
                "proposition_id": "PROP:WORLD:E4",
                "claim": "killed; affected subject: city residents; magnitude or qualifier: over five hundred",
                "proposition_type": "DESCRIPTIVE",
                "epistemic_status": "ESTABLISHED",
                "epistemic_type": "WORLD_ESTABLISHED",
                "outcome": "killed",
                "polarity": "ADVERSE",
                "party_labels": ["city residents"],
                "quantities": ["over five hundred"],
                "aliases": ["refusing will kill 500+"],
            }]
        }
        candidate = {
            "supporting_proposition_ids": ["PROP:WORLD:E4"],
            "decision_critical_proposition_ids": ["PROP:WORLD:E4"],
        }
        self.assertEqual(_candidate_epistemic_qualification(data, candidate), "")


if __name__ == "__main__":
    unittest.main()
