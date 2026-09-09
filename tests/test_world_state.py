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
    compile_event_condition_bindings,
    compact_committed_world,
    explicit_likelihood_spans,
    explicit_likelihood_span_records,
    explicit_quantity_spans,
    explicit_scope_spans,
    explicit_temporal_spans,
    normalize_redundant_link_gates,
    normalize_event_probability_ownership,
    assigned_party_quantities,
    parse_world_model,
    project_world_action_roles,
    utilitarian_omits_foregone_dual,
    validate_world_completeness,
    validate_effect_source_bindings,
    validate_world_model,
    world_model_from_dict,
    _closed_class_qualifiers,
    _condition_restates_outcome,
    _effect_expected_qualifiers,
    _event_referenced_condition_errors,
    _unique_likelihood_spans,
)
from global_workspace.action_identity import build_canonical_action_records
from global_workspace.world_validation import (
    WorldModelValidationError,
    repair_patch_contract,
    validation_issues_from_messages,
)
from global_workspace.graph_transactions import SemanticGraphStore
from global_workspace.local_specialists import (
    CompactLocalSpecialist,
    _admit_action_source_rows,
    _candidate_from_data,
    ground_actions_in_scenario,
    _preserve_stable_world_bookkeeping,
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
    claim_changes_admitted_outcome_type,
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
    _factual_status_lines,
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
            ("tens of thousands", "3", "40 minutes"),
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
        self.assertEqual(
            explicit_quantity_spans("as many as three hundred residents"),
            ("as many as three hundred",),
        )
        self.assertEqual(
            explicit_quantity_spans("up to three hundred residents"),
            ("up to three hundred",),
        )
        self.assertEqual(
            explicit_quantity_spans("at most three hundred residents"),
            ("at most three hundred",),
        )
        self.assertEqual(
            explicit_quantity_spans("no more than three hundred residents"),
            ("no more than three hundred",),
        )

    def test_as_many_as_is_the_source_grounded_party_quantity(self):
        text = (
            "as many as three hundred residents could be killed"
        )
        party = WorldParty(
            "P0", "district residents", "POPULATION",
            (SourceRef("C1", text),),
            quantities=("as many as three hundred",),
        )
        self.assertEqual(
            assigned_party_quantities((party,))["P0"],
            ("as many as three hundred",),
        )
        errors, _contradictions = validate_world_model(
            ScenarioWorldModel(
                parties=(party,),
                actions=(),
                effects=(),
                schema_version="1.2",
            ),
            action_ids=(),
        )
        self.assertFalse(any("omits source-grounded" in item for item in errors))
        self.assertFalse(any("nested quantity" in item for item in errors))

    def test_source_qualifier_extractors_keep_likelihood_scope_and_time_distinct(self):
        text = "an immediate, near-certain fatal failure creates widespread disruption"
        self.assertEqual(explicit_likelihood_spans(text), ("near-certain",))
        self.assertEqual(explicit_scope_spans(text), ("widespread",))
        self.assertEqual(explicit_temporal_spans(text), ("immediate",))

    def test_chance_phrases_are_likelihood_spans_and_do_not_nest(self):
        self.assertEqual(
            explicit_likelihood_spans("subjects have a chance to escape"),
            ("a chance",),
        )
        self.assertEqual(
            explicit_likelihood_spans(
                "subjects drown with almost no chance of survival"
            ),
            ("almost no chance",),
        )
        self.assertEqual(
            explicit_likelihood_spans("the leak could contaminate the supply"),
            ("could",),
        )
        self.assertEqual(
            explicit_likelihood_spans("subjects have a remote chance of escape"),
            ("remote chance",),
        )
        self.assertEqual(
            explicit_likelihood_spans("subjects face nearly certain harm"),
            ("nearly certain",),
        )
        self.assertEqual(
            explicit_quantity_spans("a 30% chance the residents escape"),
            (),
        )
        self.assertEqual(
            explicit_likelihood_spans("a 30% chance the residents escape"),
            ("30% chance",),
        )
        self.assertEqual(
            explicit_quantity_spans("30% of the residents escape"),
            ("30%",),
        )


class QualifierBindingTests(unittest.TestCase):
    """Likelihood/scope/time attach to the modified outcome, not sibling rows."""

    SHARED = (
        "Disconnecting the device causes it to lose power and face "
        "near-certain death of twelve subjects."
    )

    def _effect(
        self,
        effect_id: str,
        outcome: str,
        kind: str,
        *,
        excerpt: str | None = None,
        modality: str = "PROBABILISTIC",
        polarity: str = "ADVERSE",
        likelihood: tuple[str, ...] = (),
        scope: tuple[str, ...] = (),
        temporal: tuple[str, ...] = (),
        condition_ids: tuple[str, ...] = ("COND1",),
    ) -> WorldEffect:
        ref = (SourceRef("C2", excerpt or self.SHARED),)
        return WorldEffect(
            effect_id, "A0", "P1", outcome, "EXPERIENCES", polarity,
            "DOWNSTREAM", modality, kind, condition_ids, (), ref,
            likelihood, scope, temporal,
        )

    def test_near_certain_binds_to_death_not_to_same_clause_power_loss(self):
        death = self._effect("E11", "death", "HEALTH_OUTCOME")
        power = self._effect(
            "E10", "power lost", "PHYSICAL_STATE",
            modality="CERTAIN", condition_ids=(),
        )
        self.assertEqual(
            _effect_expected_qualifiers(death, explicit_likelihood_spans),
            ("near-certain",),
        )
        self.assertEqual(
            _effect_expected_qualifiers(power, explicit_likelihood_spans),
            (),
        )

    def test_closed_class_fill_does_not_copy_death_hedge_onto_power_loss(self):
        power = self._effect(
            "E10", "power lost", "PHYSICAL_STATE",
            modality="CERTAIN", condition_ids=(),
        )
        filled = _closed_class_qualifiers(power)
        self.assertEqual(filled.likelihood_qualifiers, ())

    def test_closed_class_fill_still_copies_near_certain_onto_death(self):
        death = self._effect("E11", "death", "HEALTH_OUTCOME")
        filled = _closed_class_qualifiers(death)
        self.assertEqual(filled.likelihood_qualifiers, ("near-certain",))

    def test_closed_class_untypes_certain_when_source_is_almost_certain(self):
        death = self._effect(
            "E11", "death", "HEALTH_OUTCOME",
            excerpt="Subjects face almost certain death.",
            modality="CERTAIN", condition_ids=(),
        )
        filled = _closed_class_qualifiers(death)
        self.assertEqual(filled.modality, "PROBABILISTIC")
        self.assertEqual(filled.likelihood_qualifiers, ("almost certain",))
        self.assertEqual(filled.condition_ids, ())

    def test_certain_plus_chance_hedge_is_rejected(self):
        ref = (SourceRef("C2", "Subjects face almost certain death."),)
        death = self._effect(
            "E11", "death", "HEALTH_OUTCOME", excerpt=ref[0].excerpt,
            modality="CERTAIN", condition_ids=(),
            likelihood=("almost certain",),
        )
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "subjects", "POPULATION", ref),
            ),
            actions=(
                WorldAction("A0", "disconnect the device", "P0", (), ("E11",), ref),
            ),
            effects=(death,),
            schema_version="1.2",
        )
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertTrue(any(
            "E11" in error and "CERTAIN" in error and "chance hedge" in error
            for error in errors
        ), errors)

    def test_duplicate_chance_identity_collapses(self):
        self.assertEqual(
            _unique_likelihood_spans(["20% chance", "20% chance", "20%chance"]),
            ("20% chance",),
        )

    def test_validate_omits_near_certain_on_death_not_on_power_loss(self):
        ref = (SourceRef("C2", self.SHARED),)
        parties = (
            WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
            WorldParty("P1", "device", "FACILITY", ref),
            WorldParty(
                "P2", "subjects", "POPULATION", ref, ("twelve",),
            ),
        )
        death = self._effect("E11", "death", "HEALTH_OUTCOME", likelihood=())
        power = self._effect(
            "E10", "power lost", "PHYSICAL_STATE",
            modality="CERTAIN", condition_ids=(), likelihood=(),
        )
        model = ScenarioWorldModel(
            parties=parties,
            actions=(
                WorldAction("A0", "disconnect the device", "P0", ("P1",), ("E10", "E11"), ref),
            ),
            effects=(power, death),
            conditions=(
                WorldCondition("COND1", "device power is lost", "STATED", "MATERIAL", ref),
            ),
            schema_version="1.2",
        )
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertTrue(any(
            "E11 omits" in error and "near-certain" in error for error in errors
        ))
        self.assertFalse(any(
            "E10 omits" in error and "near-certain" in error for error in errors
        ))

    def test_human_head_does_not_bind_via_shared_crowd_noun(self):
        excerpt = (
            "Twelve subjects lose device power and face near-certain death."
        )
        power = self._effect(
            "E10", "subjects lose device power", "PHYSICAL_STATE",
            excerpt=excerpt, modality="CERTAIN", condition_ids=(),
        )
        death = self._effect(
            "E11", "subjects die", "HEALTH_OUTCOME", excerpt=excerpt,
        )
        self.assertEqual(
            _effect_expected_qualifiers(power, explicit_likelihood_spans),
            (),
        )
        self.assertEqual(
            _effect_expected_qualifiers(death, explicit_likelihood_spans),
            ("near-certain",),
        )

    def test_scope_still_binds_to_the_process_row_it_modifies(self):
        excerpt = "The choice triggers widespread rolling blackouts across the grid."
        blackouts = self._effect(
            "E4", "rolling blackouts", "PHYSICAL_STATE",
            excerpt=excerpt, modality="CERTAIN", condition_ids=(),
            polarity="ADVERSE",
        )
        self.assertEqual(
            _effect_expected_qualifiers(blackouts, explicit_scope_spans),
            ("widespread",),
        )

    def test_stacked_adjectives_both_attach_to_the_modified_harm(self):
        excerpt = "Subjects suffer immediate and prolonged harm."
        harm = self._effect(
            "E6", "harm", "WELFARE_OUTCOME", excerpt=excerpt,
        )
        self.assertEqual(
            _effect_expected_qualifiers(harm, explicit_temporal_spans),
            ("immediate", "prolonged"),
        )

    def test_clause_final_likely_attaches_to_the_preceding_noun(self):
        excerpt = "The crop fails; famine is likely."
        famine = self._effect(
            "E2", "famine", "WELFARE_OUTCOME", excerpt=excerpt,
        )
        crop = self._effect(
            "E1", "crop fails", "PHYSICAL_STATE",
            excerpt=excerpt, modality="CERTAIN", condition_ids=(),
        )
        self.assertEqual(
            _effect_expected_qualifiers(famine, explicit_likelihood_spans),
            ("likely",),
        )
        self.assertEqual(
            _effect_expected_qualifiers(crop, explicit_likelihood_spans),
            (),
        )

    def test_chance_binds_to_escape_not_to_same_clause_spread(self):
        excerpt = (
            "Keeping the device connected allows twelve subjects a chance to "
            "escape, but leaves an open route for the leak to spread."
        )
        escape = self._effect(
            "E2", "subjects escape", "HEALTH_OUTCOME",
            excerpt=excerpt, polarity="BENEFICIAL",
        )
        spread = self._effect(
            "E1", "leak spreads", "PHYSICAL_STATE",
            excerpt=excerpt, modality="CERTAIN", condition_ids=(),
            polarity="NEUTRAL",
        )
        self.assertEqual(
            _effect_expected_qualifiers(escape, explicit_likelihood_spans),
            ("a chance",),
        )
        self.assertEqual(
            _effect_expected_qualifiers(spread, explicit_likelihood_spans),
            (),
        )

    def test_survive_is_the_same_head_family_as_escape(self):
        excerpt = (
            "Keeping the device connected allows twelve subjects a chance to "
            "escape."
        )
        survive = self._effect(
            "E2", "subjects survive", "HEALTH_OUTCOME",
            excerpt=excerpt, polarity="BENEFICIAL",
        )
        self.assertEqual(
            _effect_expected_qualifiers(survive, explicit_likelihood_spans),
            ("a chance",),
        )

    def test_could_binds_to_contaminate_not_to_spread(self):
        excerpt = "The leak could contaminate the supply used by residents."
        contaminate = self._effect(
            "E4", "supply contaminated", "WELFARE_OUTCOME", excerpt=excerpt,
        )
        spread = self._effect(
            "E2", "leak spreads", "PHYSICAL_STATE",
            excerpt=excerpt, modality="CERTAIN", condition_ids=(),
            polarity="NEUTRAL",
        )
        self.assertEqual(
            _effect_expected_qualifiers(contaminate, explicit_likelihood_spans),
            ("could",),
        )
        self.assertEqual(
            _effect_expected_qualifiers(spread, explicit_likelihood_spans),
            (),
        )

    def test_almost_no_chance_binds_to_drowning_not_to_the_filled_chamber(self):
        excerpt = (
            "Sealing the chamber fills it, drowning twelve subjects with "
            "almost no chance of survival."
        )
        death = self._effect(
            "E11", "subjects drown", "HEALTH_OUTCOME", excerpt=excerpt,
        )
        filled = self._effect(
            "E10", "chamber filled", "PHYSICAL_STATE",
            excerpt=excerpt, modality="CERTAIN", condition_ids=(),
            polarity="NEUTRAL",
        )
        self.assertEqual(
            _effect_expected_qualifiers(death, explicit_likelihood_spans),
            ("almost no chance",),
        )
        self.assertEqual(
            _effect_expected_qualifiers(filled, explicit_likelihood_spans),
            (),
        )
        filled_closed = _closed_class_qualifiers(filled)
        death_closed = _closed_class_qualifiers(death)
        self.assertEqual(filled_closed.likelihood_qualifiers, ())
        self.assertEqual(death_closed.likelihood_qualifiers, ("almost no chance",))

    def test_percent_chance_the_clause_binds_to_the_predicate(self):
        excerpt = (
            "There is a 20% chance the device fails, and the subjects face death."
        )
        fails = self._effect(
            "E2", "FAILS", "PHYSICAL_STATE", excerpt=excerpt,
            condition_ids=(), likelihood=("a 20% chance",),
        )
        intervention = self._effect(
            "E0", "holding the device", "INTERVENTION",
            excerpt=excerpt, modality="CERTAIN", condition_ids=(),
            polarity="NEUTRAL",
        )
        death = self._effect(
            "E3", "subjects die", "HEALTH_OUTCOME",
            excerpt=excerpt, modality="CERTAIN", condition_ids=(),
        )
        self.assertEqual(
            _effect_expected_qualifiers(fails, explicit_likelihood_spans),
            ("20% chance",),
        )
        self.assertEqual(
            _effect_expected_qualifiers(intervention, explicit_likelihood_spans),
            (),
        )
        self.assertEqual(
            _effect_expected_qualifiers(death, explicit_likelihood_spans),
            (),
        )
        self.assertEqual(
            _closed_class_qualifiers(fails).likelihood_qualifiers,
            ("20% chance",),
        )

    def test_percent_chance_of_being_binds_to_the_participle(self):
        excerpt = (
            "There is a 10% chance of being blocked, and the subjects face death."
        )
        blocked = self._effect(
            "E12", "BLOCKED", "PHYSICAL_STATE", excerpt=excerpt,
            condition_ids=(), likelihood=("a 10% chance",),
        )
        intervention = self._effect(
            "E0", "holding the device", "INTERVENTION",
            excerpt=excerpt, modality="CERTAIN", condition_ids=(),
            polarity="NEUTRAL",
        )
        self.assertEqual(
            _effect_expected_qualifiers(blocked, explicit_likelihood_spans),
            ("10% chance",),
        )
        self.assertEqual(
            _effect_expected_qualifiers(intervention, explicit_likelihood_spans),
            (),
        )
        self.assertEqual(
            _closed_class_qualifiers(blocked).likelihood_qualifiers,
            ("10% chance",),
        )

    def test_glued_percent_chance_keeps_literal_offsets_and_canonical(self):
        excerpt = "There is a 20%chance the device fails."
        records = explicit_likelihood_span_records(excerpt)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].literal, "20%chance")
        self.assertEqual(records[0].canonical, "20% chance")
        self.assertEqual(excerpt[records[0].start:records[0].end], "20%chance")
        self.assertEqual(explicit_likelihood_spans(excerpt), ("20% chance",))
        fails = self._effect(
            "E2", "FAILS", "PHYSICAL_STATE", excerpt=excerpt,
            condition_ids=(), likelihood=(),
        )
        filled = _closed_class_qualifiers(fails)
        self.assertEqual(filled.likelihood_qualifiers, ("20%chance",))

    def test_at_moderate_risk_on_exposure_is_certain_not_chance(self):
        excerpt = "Five workers would stay to assist them at moderate risk."
        exposed = self._effect(
            "E9", "AT_MODERATE_RISK", "HEALTH_OUTCOME", excerpt=excerpt,
            condition_ids=(), likelihood=("moderate risk",),
        )
        stay = self._effect(
            "E10", "STAYS_TO_ASSIST", "INTERVENTION", excerpt=excerpt,
            modality="CERTAIN", condition_ids=(), polarity="NEUTRAL",
        )
        self.assertEqual(explicit_likelihood_spans(excerpt), ("at moderate risk",))
        self.assertEqual(
            _effect_expected_qualifiers(exposed, explicit_likelihood_spans),
            ("at moderate risk",),
        )
        self.assertEqual(
            _effect_expected_qualifiers(stay, explicit_likelihood_spans),
            (),
        )
        filled = _closed_class_qualifiers(exposed)
        self.assertEqual(filled.likelihood_qualifiers, ("at moderate risk",))
        self.assertEqual(filled.modality, "CERTAIN")
        self.assertEqual(filled.condition_ids, ())

    def test_at_high_risk_of_dying_is_a_chance_hedge_on_death(self):
        excerpt = "The subjects are at high risk of dying."
        death = self._effect(
            "E8", "DIES", "HEALTH_OUTCOME", excerpt=excerpt,
            condition_ids=(), likelihood=(),
        )
        self.assertEqual(explicit_likelihood_spans(excerpt), ("at high risk",))
        self.assertEqual(
            _effect_expected_qualifiers(death, explicit_likelihood_spans),
            ("at high risk",),
        )
        filled = _closed_class_qualifiers(death)
        self.assertEqual(filled.likelihood_qualifiers, ("at high risk",))
        self.assertEqual(filled.modality, "PROBABILISTIC")


class IndependentConditionTests(unittest.TestCase):
    """Conditions must add an unknown, not restate a CERTAIN parent."""

    def test_redundant_certain_link_gate_is_owned_by_conditional_target(self):
        effects = (
            WorldEffect(
                "E0", "A0", "P1", "facility exposed", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                provenance=REF,
            ),
            WorldEffect(
                "E1", "A0", "P2", "water contaminated", "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "STIPULATED_CONDITIONAL",
                "WELFARE_OUTCOME", ("COND0",), provenance=REF,
            ),
        )
        normalized = normalize_redundant_link_gates((
            CausalLink(
                "E0", "CAUSES", "E1", "CERTAIN", ("COND0",), REF, "A0",
            ),
        ), effects)
        self.assertEqual(normalized[0].condition_ids, ())
        self.assertEqual(normalized[0].modality, "CERTAIN")
        self.assertEqual(effects[1].condition_ids, ("COND0",))

    def test_unique_link_gate_is_not_silently_rewritten(self):
        effects = (
            WorldEffect(
                "E0", "A0", "P1", "facility exposed", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                provenance=REF,
            ),
            WorldEffect(
                "E1", "A0", "P2", "water contaminated", "EXPERIENCES",
                "ADVERSE", "DOWNSTREAM", "STIPULATED_CONDITIONAL",
                "WELFARE_OUTCOME", ("COND0",), provenance=REF,
            ),
        )
        normalized = normalize_redundant_link_gates((
            CausalLink(
                "E0", "CAUSES", "E1", "CERTAIN", ("COND1",), REF, "A0",
            ),
        ), effects)
        self.assertEqual(normalized[0].condition_ids, ("COND1",))

    def test_unique_branch_local_probability_event_is_bound_by_compiler(self):
        ref = (SourceRef(
            "C1", "There is a 20% chance the backup process fails.",
        ),)
        effects = (
            WorldEffect(
                "E2", "A0", "P3", "backup process fails", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
                likelihood_qualifiers=("20% chance",), provenance=ref,
            ),
            WorldEffect(
                "E3", "A0", "P2", "subjects die", "DIES", "ADVERSE",
                "DOWNSTREAM", "STIPULATED_CONDITIONAL", "HEALTH_OUTCOME",
                ("COND0",), provenance=ref,
            ),
        )
        compiled = compile_event_condition_bindings((
            WorldCondition("COND0", "if the backup process fails", provenance=ref),
        ), effects)
        self.assertEqual(compiled[0].event_effect_id, "E2")

    def test_event_binding_never_reaches_across_action_branches(self):
        ref = (SourceRef("C1", "The backup process may fail."),)
        effects = (
            WorldEffect(
                "E2", "A1", "P3", "backup process fails", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "POSSIBLE", "PHYSICAL_STATE",
                likelihood_qualifiers=("may",), provenance=ref,
            ),
            WorldEffect(
                "E3", "A0", "P2", "subjects die", "DIES", "ADVERSE",
                "DOWNSTREAM", "STIPULATED_CONDITIONAL", "HEALTH_OUTCOME",
                ("COND0",), provenance=ref,
            ),
        )
        compiled = compile_event_condition_bindings((
            WorldCondition("COND0", "if the backup process fails", provenance=ref),
        ), effects)
        self.assertEqual(compiled[0].event_effect_id, "")

    def test_event_probability_is_removed_from_gated_descendant(self):
        ref = (SourceRef(
            "C1", "There is a 20% chance the backup fails and subjects die.",
        ),)
        effects = (
            WorldEffect(
                "E2", "A0", "P3", "backup fails", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
                likelihood_qualifiers=("20% chance",), provenance=ref,
            ),
            WorldEffect(
                "E3", "A0", "P2", "subjects die", "DIES", "ADVERSE",
                "DOWNSTREAM", "PROBABILISTIC", "HEALTH_OUTCOME",
                ("COND0",), provenance=ref,
                likelihood_qualifiers=("20% chance",),
            ),
        )
        conditions = (
            WorldCondition(
                "COND0", "backup fails", provenance=ref, event_effect_id="E2",
            ),
        )
        normalized = normalize_event_probability_ownership(effects, conditions)
        self.assertEqual(normalized[0].likelihood_qualifiers, ("20% chance",))
        self.assertEqual(normalized[1].likelihood_qualifiers, ())
        self.assertEqual(normalized[1].modality, "STIPULATED_CONDITIONAL")

    EXCERPT = (
        "The allocator keeps the device connected, preserving support for "
        "twelve subjects."
    )

    def _chain(
        self,
        *,
        child_modality: str = "POSSIBLE",
        condition_text: str = "if the device remains powered",
        child_outcome: str = "subjects survive",
        child_likelihood: tuple[str, ...] = (),
        excerpt: str | None = None,
        include_condition: bool = True,
        child_kind: str = "HEALTH_OUTCOME",
    ) -> ScenarioWorldModel:
        ref = (SourceRef("C1", excerpt or self.EXCERPT),)
        cond = ()
        conditions: tuple[WorldCondition, ...] = ()
        if include_condition:
            cond = ("COND0",)
            conditions = (
                WorldCondition("COND0", condition_text, "UNKNOWN", "MATERIAL", ref),
            )
        parties = (
            WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
            WorldParty("P1", "device", "FACILITY", ref),
            WorldParty("P2", "subjects", "POPULATION", ref, ("twelve",)),
        )
        effects = (
            WorldEffect(
                "E0", "A0", "P1", "device connected", "IS", "NEUTRAL",
                "DIRECT", "CERTAIN", "INTERVENTION", (), (), ref,
            ),
            WorldEffect(
                "E1", "A0", "P1", "powered", "STATE_CHANGE", "BENEFICIAL",
                "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE", (), (), ref,
            ),
            WorldEffect(
                "E2", "A0", "P2", child_outcome, "SURVIVES", "BENEFICIAL",
                "DOWNSTREAM", child_modality, child_kind, cond, (), ref,
                child_likelihood,
            ),
        )
        links = (
            CausalLink("E0", "ENABLES", "E1", "CERTAIN", (), ref, "A0"),
            CausalLink("E1", "CAUSES", "E2", "CERTAIN", (), ref, "A0"),
        )
        return ScenarioWorldModel(
            parties=parties,
            actions=(
                WorldAction("A0", "keep the device connected", "P0", ("P1",), ("E0", "E1", "E2"), ref),
            ),
            effects=effects,
            conditions=conditions,
            causal_links=links,
            schema_version="1.2",
        )

    def test_parent_restatement_is_detected_on_stems(self):
        self.assertTrue(
            _condition_restates_outcome("if the device remains powered", "powered")
        )
        self.assertTrue(
            _condition_restates_outcome("if device power is lost", "power lost")
        )
        self.assertFalse(
            _condition_restates_outcome("if the attempt works", "attempt is made")
        )
        self.assertFalse(
            _condition_restates_outcome("engineer is reached", "spillway held")
        )

    def test_survive_gated_on_certain_powered_is_rejected(self):
        errors = validate_world_completeness(self._chain(), action_ids=["A0"])
        self.assertTrue(any(
            "E2" in error and "restates immediate parent E1" in error
            for error in errors
        ), errors)

    def test_unhedged_possible_survival_is_rejected_even_with_a_novel_if(self):
        errors = validate_world_completeness(
            self._chain(condition_text="if the subjects remain clinically stable"),
            action_ids=["A0"],
        )
        self.assertTrue(any(
            "E2" in error and "unhedged indicative" in error for error in errors
        ), errors)

    def test_source_if_clause_keeps_a_real_stipulated_conditional(self):
        excerpt = (
            "If the attempt works, twelve subjects survive after the device "
            "is powered."
        )
        model = self._chain(
            child_modality="STIPULATED_CONDITIONAL",
            condition_text="if the attempt works",
            excerpt=excerpt,
        )
        self.assertEqual(validate_world_completeness(model, action_ids=["A0"]), [])

    def test_attempt_made_then_attempt_works_is_not_a_restatement(self):
        ref = (SourceRef(
            "C1",
            "The allocator funds the attempt. If the attempt works, twelve "
            "subjects survive.",
        ),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "program", "PROCESS", ref),
                WorldParty("P2", "subjects", "POPULATION", ref, ("twelve",)),
            ),
            actions=(
                WorldAction("A0", "fund the attempt", "P0", ("P1",), ("E0", "E1", "E2"), ref),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "attempt is funded", "IS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", (), (), ref,
                ),
                WorldEffect(
                    "E1", "A0", "P1", "attempt is made", "STATE_CHANGE", "NEUTRAL",
                    "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE", (), (), ref,
                ),
                WorldEffect(
                    "E2", "A0", "P2", "subjects survive", "SURVIVES", "BENEFICIAL",
                    "DOWNSTREAM", "STIPULATED_CONDITIONAL", "HEALTH_OUTCOME",
                    ("COND0",), (), ref,
                ),
            ),
            conditions=(
                WorldCondition("COND0", "if the attempt works", "UNKNOWN", "MATERIAL", ref),
            ),
            causal_links=(
                CausalLink("E0", "ENABLES", "E1", "CERTAIN", (), ref, "A0"),
                CausalLink("E1", "CAUSES", "E2", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.2",
        )
        self.assertEqual(validate_world_completeness(model, action_ids=["A0"]), [])

    def test_probabilistic_near_certain_need_not_invent_an_if(self):
        excerpt = (
            "Disconnecting the device causes twelve subjects to face "
            "near-certain death."
        )
        model = self._chain(
            child_modality="PROBABILISTIC",
            child_outcome="death",
            child_likelihood=("near-certain",),
            excerpt=excerpt,
            include_condition=False,
            child_kind="HEALTH_OUTCOME",
        )
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertFalse(any("E2" in error and "no condition" in error for error in errors), errors)
        self.assertEqual(validate_world_completeness(model, action_ids=["A0"]), [])

    def test_probabilistic_without_condition_or_likelihood_still_fails(self):
        model = self._chain(
            child_modality="PROBABILISTIC",
            include_condition=False,
        )
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertTrue(any("E2" in error and "no condition" in error for error in errors), errors)

    def test_allocator_engineer_condition_is_not_a_spillway_restatement(self):
        model = _parse_allocator()
        self.assertEqual(validate_world_completeness(model, action_ids=["A0", "A1"]), [])

    def test_chance_to_escape_allows_possible_without_a_parent_if(self):
        excerpt = (
            "Keeping the device connected allows twelve subjects a chance to "
            "escape, but leaves an open route for the leak to spread."
        )
        model = self._chain(
            child_modality="POSSIBLE",
            child_outcome="subjects escape",
            excerpt=excerpt,
            include_condition=False,
            child_likelihood=("a chance",),
        )
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertFalse(
            any("E2" in error and "no condition" in error for error in errors),
            errors,
        )
        self.assertEqual(validate_world_completeness(model, action_ids=["A0"]), [])

    def test_same_clause_unhedged_spread_possible_is_still_rejected(self):
        excerpt = (
            "Keeping the device connected allows twelve subjects a chance to "
            "escape, but leaves an open route for the leak to spread."
        )
        ref = (SourceRef("C1", excerpt),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "device", "FACILITY", ref),
                WorldParty("P2", "subjects", "POPULATION", ref, ("twelve",)),
                WorldParty("P3", "leak", "PROCESS", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "keep the device connected", "P0", ("P1",),
                    ("E0", "E1", "E2"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "device connected", "IS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", (), (), ref,
                ),
                WorldEffect(
                    "E1", "A0", "P3", "leak spreads", "STATE_CHANGE", "ADVERSE",
                    "DOWNSTREAM", "POSSIBLE", "PHYSICAL_STATE", ("COND0",), (), ref,
                ),
                WorldEffect(
                    "E2", "A0", "P2", "subjects escape", "SURVIVES", "BENEFICIAL",
                    "DOWNSTREAM", "POSSIBLE", "HEALTH_OUTCOME", (), (), ref,
                    ("a chance",),
                ),
            ),
            conditions=(
                WorldCondition(
                    "COND0", "if the leak remains uncontained", "UNKNOWN",
                    "MATERIAL", ref,
                ),
            ),
            causal_links=(
                CausalLink("E0", "ENABLES", "E1", "CERTAIN", (), ref, "A0"),
                CausalLink("E0", "CAUSES", "E2", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.2",
        )
        errors = validate_world_completeness(model, action_ids=["A0"])
        self.assertTrue(
            any("E1" in error and "unhedged indicative" in error for error in errors),
            errors,
        )
        self.assertFalse(
            any("E2" in error and "unhedged indicative" in error for error in errors),
            errors,
        )

    def test_almost_no_chance_drowning_need_not_invent_an_if(self):
        excerpt = (
            "Sealing the chamber fills it, drowning twelve subjects with "
            "almost no chance of survival."
        )
        model = self._chain(
            child_modality="PROBABILISTIC",
            child_outcome="subjects drown",
            child_likelihood=("almost no chance",),
            excerpt=excerpt,
            include_condition=False,
            child_kind="HEALTH_OUTCOME",
        )
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertFalse(
            any("E2" in error and "no condition" in error for error in errors),
            errors,
        )
        self.assertEqual(validate_world_completeness(model, action_ids=["A0"]), [])

    def test_drowning_gated_on_filled_parent_is_rejected(self):
        excerpt = (
            "Sealing the chamber fills it, drowning twelve subjects with "
            "almost no chance of survival."
        )
        ref = (SourceRef("C1", excerpt),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "chamber", "FACILITY", ref),
                WorldParty("P2", "subjects", "POPULATION", ref, ("twelve",)),
            ),
            actions=(
                WorldAction(
                    "A0", "seal the chamber", "P0", ("P1",),
                    ("E0", "E1", "E2"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "chamber sealed", "IS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", (), (), ref,
                ),
                WorldEffect(
                    "E1", "A0", "P1", "chamber filled", "STATE_CHANGE", "NEUTRAL",
                    "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE", (), (), ref,
                ),
                WorldEffect(
                    "E2", "A0", "P2", "subjects drown", "DIES", "ADVERSE",
                    "DOWNSTREAM", "PROBABILISTIC", "HEALTH_OUTCOME",
                    ("COND0",), (), ref, ("almost no chance",),
                ),
            ),
            conditions=(
                WorldCondition(
                    "COND0", "if the chamber remains filled", "UNKNOWN",
                    "MATERIAL", ref,
                ),
            ),
            causal_links=(
                CausalLink("E0", "ENABLES", "E1", "CERTAIN", (), ref, "A0"),
                CausalLink("E1", "CAUSES", "E2", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.2",
        )
        errors = validate_world_completeness(model, action_ids=["A0"])
        self.assertTrue(
            any(
                "E2" in error and "restates immediate parent E1" in error
                for error in errors
            ),
            errors,
        )

    def test_event_referenced_gate_admits(self):
        excerpt = (
            "The allocator keeps the device connected. There is a 20% chance "
            "the backup process fails. If it fails, twelve subjects face "
            "near-certain death."
        )
        ref = (SourceRef("C1", excerpt),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "device", "FACILITY", ref),
                WorldParty("P2", "subjects", "POPULATION", ref, ("twelve",)),
                WorldParty("P3", "backup", "PROCESS", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "keep the device connected", "P0", ("P1",),
                    ("E0", "E1", "E2", "E3"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "device connected", "IS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P1", "device held", "STATE_CHANGE", "NEUTRAL",
                    "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE", provenance=ref,
                ),
                WorldEffect(
                    "E2", "A0", "P3", "fails", "STATE_CHANGE", "NEUTRAL",
                    "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
                    likelihood_qualifiers=("20% chance",), provenance=ref,
                ),
                WorldEffect(
                    "E3", "A0", "P2", "subjects die", "DIES", "ADVERSE",
                    "DOWNSTREAM", "STIPULATED_CONDITIONAL", "HEALTH_OUTCOME",
                    ("COND0",), (), ref, ("near-certain",),
                ),
            ),
            conditions=(
                WorldCondition(
                    "COND0", "the backup process fails", provenance=ref,
                    event_effect_id="E2",
                ),
            ),
            causal_links=(
                CausalLink("E0", "ENABLES", "E1", "CERTAIN", (), ref, "A0"),
                CausalLink("E1", "CAUSES", "E3", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.2",
        )
        self.assertEqual(validate_world_completeness(model, action_ids=["A0"]), [])
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertEqual(errors, [])

    def test_event_reference_ignores_other_action_foregone_mirror(self):
        ref = (SourceRef(
            "C1",
            "There is a 25% chance the facility floods and residents lose water.",
        ),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "facility", "FACILITY", ref),
                WorldParty("P2", "residents", "POPULATION", ref),
            ),
            actions=(
                WorldAction("A0", "protect facility", "P0", ("P1",), ("E8",), REF),
                WorldAction(
                    "A1", "leave facility exposed", "P0", ("P1",),
                    ("E3B", "E4B"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E8", "A0", "P1", "facility flooded", "STATE_CHANGE",
                    "FOREGONE", "FOREGONE", "CERTAIN", "OPPORTUNITY_LOSS",
                    provenance=ref,
                ),
                WorldEffect(
                    "E3B", "A1", "P1", "facility flooded", "STATE_CHANGE",
                    "ADVERSE", "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
                    likelihood_qualifiers=("25% chance",), provenance=ref,
                ),
                WorldEffect(
                    "E4B", "A1", "P2", "residents lose water", "EXPERIENCES",
                    "ADVERSE", "DOWNSTREAM", "STIPULATED_CONDITIONAL",
                    "WELFARE_OUTCOME", ("CT2",), provenance=ref,
                ),
            ),
            conditions=(
                WorldCondition(
                    "CT2", "facility flooded", provenance=ref,
                    event_effect_id="E3B",
                ),
            ),
            schema_version="1.2",
        )
        errors = _event_referenced_condition_errors(model)
        self.assertFalse(
            any("description restates E8" in error for error in errors),
            errors,
        )

    def test_free_text_restating_independent_event_is_rejected(self):
        excerpt = (
            "The allocator keeps the device connected. There is a 20% chance "
            "the backup process fails. If it fails, twelve subjects die."
        )
        ref = (SourceRef("C1", excerpt),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "device", "FACILITY", ref),
                WorldParty("P2", "subjects", "POPULATION", ref, ("twelve",)),
                WorldParty("P3", "backup", "PROCESS", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "keep the device connected", "P0", ("P1",),
                    ("E0", "E1", "E2", "E3"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "device connected", "IS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P1", "device held", "STATE_CHANGE", "NEUTRAL",
                    "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE", provenance=ref,
                ),
                WorldEffect(
                    "E2", "A0", "P3", "fails", "STATE_CHANGE", "NEUTRAL",
                    "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
                    likelihood_qualifiers=("20% chance",), provenance=ref,
                ),
                WorldEffect(
                    "E3", "A0", "P2", "subjects die", "DIES", "ADVERSE",
                    "DOWNSTREAM", "STIPULATED_CONDITIONAL", "HEALTH_OUTCOME",
                    ("COND0",), (), ref, ("near-certain",),
                ),
            ),
            conditions=(
                WorldCondition("COND0", "the backup process fails", provenance=ref),
            ),
            causal_links=(
                CausalLink("E0", "ENABLES", "E1", "CERTAIN", (), ref, "A0"),
                CausalLink("E1", "CAUSES", "E3", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.2",
        )
        errors = validate_world_completeness(model, action_ids=["A0"])
        self.assertTrue(any("event_effect_id" in error for error in errors), errors)

    def test_independent_event_as_causal_parent_is_rejected(self):
        excerpt = (
            "The allocator keeps the device connected. There is a 20% chance "
            "the backup process fails. If it fails, twelve subjects die."
        )
        ref = (SourceRef("C1", excerpt),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "device", "FACILITY", ref),
                WorldParty("P2", "subjects", "POPULATION", ref, ("twelve",)),
                WorldParty("P3", "backup", "PROCESS", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "keep the device connected", "P0", ("P1",),
                    ("E0", "E1", "E2", "E3"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "device connected", "IS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P1", "device held", "STATE_CHANGE", "NEUTRAL",
                    "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE", provenance=ref,
                ),
                WorldEffect(
                    "E2", "A0", "P3", "fails", "STATE_CHANGE", "NEUTRAL",
                    "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
                    likelihood_qualifiers=("20% chance",), provenance=ref,
                ),
                WorldEffect(
                    "E3", "A0", "P2", "subjects die", "DIES", "ADVERSE",
                    "DOWNSTREAM", "POSSIBLE", "HEALTH_OUTCOME",
                    provenance=ref, likelihood_qualifiers=("near-certain",),
                ),
            ),
            causal_links=(
                CausalLink("E0", "ENABLES", "E1", "CERTAIN", (), ref, "A0"),
                CausalLink("E1", "CAUSES", "E3", "CERTAIN", (), ref, "A0"),
                CausalLink("E2", "CAUSES", "E3", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.2",
        )
        errors = validate_world_completeness(model, action_ids=["A0"])
        self.assertTrue(any(
            "independent stochastic" in error and "E2" in error
            for error in errors
        ), errors)


class NeutralDirectInterventionTests(unittest.TestCase):
    def _facility_act(self, *, polarity: str) -> ScenarioWorldModel:
        ref = (SourceRef("C1", "The allocator connects the device."),)
        return ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "device", "FACILITY", ref),
            ),
            actions=(
                WorldAction("A0", "connect the device", "P0", ("P1",), ("E0",), ref),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "connected", "IS", polarity,
                    "DIRECT", "CERTAIN", "INTERVENTION", (), (), ref,
                ),
            ),
            schema_version="1.2",
        )

    def test_beneficial_connect_on_a_facility_is_rejected(self):
        errors, _ = validate_world_model(
            self._facility_act(polarity="BENEFICIAL"), action_ids=["A0"],
        )
        self.assertTrue(any(
            "E0" in error and "NEUTRAL" in error and "P1" in error for error in errors
        ), errors)

    def test_downstream_intervention_tells_the_repair_to_change_kind(self):
        ref = (SourceRef("C1", "The allocator connects the device."),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "device", "FACILITY", ref),
                WorldParty("P2", "workers", "POPULATION", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "connect the device", "P0", ("P1",), ("E0", "E1"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "connected", "IS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P2", "workers assist", "PERFORMS", "NEUTRAL",
                    "DOWNSTREAM", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
            ),
            schema_version="1.2",
        )
        errors, _ = validate_world_model(model, action_ids=["A0"])
        joined = " ".join(errors)
        self.assertIn("E1", joined)
        self.assertIn("directness must be DIRECT", joined)
        self.assertIn("change effect_kind", joined)
        self.assertIn("do not add a later crowd as a recipient", joined)

    def test_direct_intervention_on_a_non_recipient_crowd_does_not_add_them(self):
        ref = (SourceRef("C1", "The allocator connects the device. Workers assist."),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "device", "FACILITY", ref),
                WorldParty("P2", "workers", "POPULATION", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "connect the device", "P0", ("P1",), ("E0", "E1"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "connected", "IS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P2", "workers assist", "PERFORMS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
            ),
            schema_version="1.2",
        )
        errors = validate_world_completeness(model, action_ids=["A0"])
        joined = " ".join(errors)
        self.assertIn("E1", joined)
        self.assertIn("neither the actor nor a named recipient", joined)
        self.assertIn("do not add that party as a recipient", joined)
        self.assertIn("DOWNSTREAM", joined)
        self.assertIn("assigning or allocating", joined)

    def test_assignment_recipient_then_downstream_conduct_admits(self):
        ref = (SourceRef(
            "C1",
            "The allocator assigns workers to the site. Assigned workers assist. "
            "The workers are at moderate risk.",
        ),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "workers", "POPULATION", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "assign workers to the site", "P0", ("P1",),
                    ("E0", "E1", "E2"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "workers assigned", "PERFORMS", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P1", "workers assist", "PERFORMS", "NEUTRAL",
                    "DOWNSTREAM", "CERTAIN", "OTHER", provenance=ref,
                ),
                WorldEffect(
                    "E2", "A0", "P1", "workers are at moderate risk", "AT_RISK",
                    "ADVERSE", "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME",
                    provenance=ref, likelihood_qualifiers=("at moderate risk",),
                ),
            ),
            causal_links=(
                CausalLink("E0", "CAUSES", "E1", "CERTAIN", (), ref, "A0"),
                CausalLink("E1", "CAUSES", "E2", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.2",
        )
        errors, _ = validate_world_model(model, action_ids=["A0"])
        complete = validate_world_completeness(model, action_ids=["A0"])
        self.assertEqual(errors, [])
        self.assertEqual(complete, [])

    def test_neutral_connect_on_a_facility_is_allowed(self):
        errors, _ = validate_world_model(
            self._facility_act(polarity="NEUTRAL"), action_ids=["A0"],
        )
        self.assertFalse(any("E0" in error and "NEUTRAL" in error for error in errors), errors)

    def test_beneficial_refusal_on_a_human_recipient_is_allowed(self):
        ref = (SourceRef("C1", "The magistrate refuses to execute the prisoner."),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "magistrate", "HUMAN", REF),
                WorldParty("P1", "prisoner", "PERSON", ref),
            ),
            actions=(
                WorldAction("A0", "refuse to execute", "P0", ("P1",), ("E0",), ref),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "not executed", "IS", "BENEFICIAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", (), (), ref,
                ),
            ),
            schema_version="1.2",
        )
        errors, _ = validate_world_model(model, action_ids=["A0"])
        self.assertFalse(any("E0" in error and "NEUTRAL" in error for error in errors), errors)


class ClauseRoleTests(unittest.TestCase):
    def test_comparison_and_fact_roles(self):
        self.assertEqual(classify_clause_role(COMPARE_REF[0].excerpt), "COMPARISON")
        self.assertEqual(classify_clause_role(DOZEN_REF[0].excerpt), "FACT")
        self.assertEqual(classify_clause_role("Which pipeline should receive the supply?"), "INTERROGATIVE")


class WorldModelValidationTests(unittest.TestCase):
    def test_schema_13_action_source_candidate_commits_end_to_end(self):
        clauses = [
            {"clause_id": "C0", "text": "An automated system controls two gates."},
            {"clause_id": "C1", "text": "A0 opens north gate."},
            {"clause_id": "C2", "text": "A1 opens south gate."},
        ]
        candidate = {
            "actions": {
                "A0": {"clause_ids": ["C1"], "reason": "C1 states A0."},
                "A1": {"clause_ids": ["C2"], "reason": "C2 states A1."},
            },
            "world_model": {
                "schema_version": "1.3",
                "parties": [
                    {"party_id": "P0", "label": "automated system", "kind": "AUTOMATED_SYSTEM", "quantities": [], "clause_ids": ["C0"]},
                    {"party_id": "PN", "label": "north gate", "kind": "INFRASTRUCTURE", "quantities": [], "clause_ids": ["C1"]},
                    {"party_id": "PS", "label": "south gate", "kind": "INFRASTRUCTURE", "quantities": [], "clause_ids": ["C2"]},
                ],
                "actions": [
                    {"action_id": "A0", "intervention": "opens north gate", "actor_party_id": "P0", "recipient_party_ids": ["PN"], "effect_ids": ["E0"], "clause_ids": ["C1"]},
                    {"action_id": "A1", "intervention": "opens south gate", "actor_party_id": "P0", "recipient_party_ids": ["PS"], "effect_ids": ["E1"], "clause_ids": ["C2"]},
                ],
                "effects": [
                    {
                        "effect_id": "E0", "action_id": "A0", "party_id": "PN",
                        "outcome": "opens north gate", "predicate": "ACTS",
                        "polarity": "NEUTRAL", "directness": "DIRECT",
                        "modality": "CERTAIN", "effect_kind": "INTERVENTION",
                        "condition_ids": [], "quantities": [],
                        "likelihood_qualifiers": [], "overall_likelihood_qualifiers": [],
                        "scope_qualifiers": [], "temporal_qualifiers": [],
                        "condition_join": "AND", "source_proposition": "opens north gate",
                        "source_effect_ids": [], "derivation_operation": "DIRECT_COPY",
                        "derivation_explanation": "C1 states the action.",
                        "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
                        "clause_ids": ["C1"],
                    },
                    {
                        "effect_id": "E1", "action_id": "A1", "party_id": "PS",
                        "outcome": "opens south gate", "predicate": "ACTS",
                        "polarity": "NEUTRAL", "directness": "DIRECT",
                        "modality": "CERTAIN", "effect_kind": "INTERVENTION",
                        "condition_ids": [], "quantities": [],
                        "likelihood_qualifiers": [], "overall_likelihood_qualifiers": [],
                        "scope_qualifiers": [], "temporal_qualifiers": [],
                        "condition_join": "AND", "source_proposition": "opens south gate",
                        "source_effect_ids": [], "derivation_operation": "DIRECT_COPY",
                        "derivation_explanation": "C2 states the action.",
                        "derivation_assumptions": [], "outcome_type_transformation": "PRESERVED",
                        "clause_ids": ["C2"],
                    },
                ],
                "conditions": [], "causal_links": [], "counterfactual_links": [],
            },
        }
        result = _admit_action_source_rows(
            candidate,
            ["opens north gate", "opens south gate"],
            ["A0", "A1"],
            clauses,
        )
        self.assertEqual(result["status"], "COMMITTED", result["errors"])
        self.assertEqual(result["world_model"]["schema_version"], "1.3")
        binding = result["world_model"]["effects"][0]
        self.assertEqual(binding["source_proposition"], "opens north gate")
        restored = world_model_from_dict(result["world_model"])
        self.assertIsNotNone(restored)
        self.assertEqual(restored.effects[0].source_proposition, "opens north gate")
        compact = compact_committed_world(restored)
        self.assertEqual(
            compact["effects"][0]["source_binding"]["derivation_operation"],
            "DIRECT_COPY",
        )

    def test_schema_13_explicit_source_proposition_binding_admits(self):
        ref = (SourceRef("C1", "The action guarantees safe water."),)
        effect = WorldEffect(
            "E1", "A0", "P1", "safe water", "HAS", "BENEFICIAL",
            "DIRECT", "CERTAIN", "RESOURCE_TRANSFER", provenance=ref,
            source_proposition="guarantees safe water",
            derivation_operation="DIRECT_COPY",
            derivation_explanation="The source states the atomic outcome.",
        )
        model = ScenarioWorldModel(
            parties=(WorldParty("P1", "residents", "POPULATION", ref),),
            actions=(WorldAction("A0", "act", "P1", ("P1",), ("E1",), ref),),
            effects=(effect,),
            schema_version="1.3",
        )
        self.assertEqual(validate_effect_source_bindings(model), [])

    def test_schema_13_rejects_outcome_leap_hidden_as_direct_copy(self):
        ref = (SourceRef("C1", "Residents become trapped in the facility."),)
        effect = WorldEffect(
            "E1", "A0", "P1", "residents die", "DIES", "ADVERSE",
            "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME", provenance=ref,
            source_proposition="Residents become trapped in the facility",
            derivation_operation="DIRECT_COPY",
            derivation_explanation="Claims trapping means death.",
            outcome_type_transformation="MORTALITY",
        )
        model = ScenarioWorldModel(
            parties=(WorldParty("P1", "residents", "POPULATION", ref),),
            actions=(WorldAction("A0", "act", "P1", ("P1",), ("E1",), ref),),
            effects=(effect,),
            schema_version="1.3",
        )
        errors = validate_effect_source_bindings(model)
        self.assertTrue(any("does not state its normalized outcome" in row for row in errors))
        self.assertTrue(any("changes source outcome type" in row for row in errors))

    def test_schema_13_causal_binding_requires_named_immediate_parent(self):
        ref = (SourceRef(
            "C1", "Opening the valve supplies safe water to residents.",
        ),)
        effects = (
            WorldEffect(
                "E0", "A0", "P0", "opening the valve", "ACTS", "NEUTRAL",
                "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                source_proposition="Opening the valve",
                derivation_operation="DIRECT_COPY",
                derivation_explanation="The source states the intervention.",
            ),
            WorldEffect(
                "E1", "A0", "P1", "safe water", "HAS", "BENEFICIAL",
                "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME", provenance=ref,
                source_proposition="supplies safe water to residents",
                source_effect_ids=("E0",),
                derivation_operation="SOURCE_STIPULATED_CAUSAL",
                derivation_explanation="The source says opening supplies the water.",
            ),
        )
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "valve", "INFRASTRUCTURE", ref),
                WorldParty("P1", "residents", "POPULATION", ref),
            ),
            actions=(WorldAction("A0", "open", "P0", ("P0",), ("E0", "E1"), ref),),
            effects=effects,
            causal_links=(
                CausalLink("E0", "CAUSES", "E1", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.3",
        )
        self.assertEqual(validate_effect_source_bindings(model), [])
        broken = replace(model, causal_links=())
        errors = validate_effect_source_bindings(broken)
        self.assertTrue(any("not same-action immediate parents" in row for row in errors))

    def test_structural_abstraction_cannot_smuggle_in_human_harm(self):
        ref = (SourceRef("C1", "Residents are trapped after the gate closes."),)
        effects = (
            WorldEffect(
                "E0", "A0", "P0", "gate closes", "ACTS", "NEUTRAL",
                "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                source_proposition="the gate closes",
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E1", "A0", "P1", "trapped residents die", "DIES", "ADVERSE",
                "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME", provenance=ref,
                source_proposition="Residents are trapped",
                source_effect_ids=("E0",),
                derivation_operation="STRUCTURAL_ABSTRACTION",
                derivation_explanation="Attempts to turn trapping into death.",
                outcome_type_transformation="MORTALITY",
            ),
        )
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "gate", "INFRASTRUCTURE", ref),
                WorldParty("P1", "residents", "POPULATION", ref),
            ),
            actions=(WorldAction("A0", "close", "P0", ("P0",), ("E0", "E1"), ref),),
            effects=effects,
            causal_links=(
                CausalLink("E0", "CAUSES", "E1", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.3",
        )
        errors = validate_effect_source_bindings(model)
        self.assertTrue(any("must be an actual NEUTRAL" in row for row in errors))
        self.assertTrue(any("changes source outcome type" in row for row in errors))

    def test_neutral_source_anchored_structural_abstraction_admits(self):
        ref = (SourceRef("C1", "Opening the gate sends the surge along this path."),)
        effects = (
            WorldEffect(
                "E0", "A0", "P0", "opening the gate", "ACTS", "NEUTRAL",
                "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                source_proposition="Opening the gate",
                derivation_operation="DIRECT_COPY",
            ),
            WorldEffect(
                "E1", "A0", "P1", "surge path exposure", "STATE_CHANGE",
                "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE",
                provenance=ref,
                source_proposition="sends the surge along this path",
                source_effect_ids=("E0",),
                derivation_operation="STRUCTURAL_ABSTRACTION",
                derivation_explanation="Represents the stated surge path.",
            ),
        )
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "gate", "INFRASTRUCTURE", ref),
                WorldParty("P1", "path", "PROCESS", ref),
            ),
            actions=(WorldAction("A0", "open", "P0", ("P0",), ("E0", "E1"), ref),),
            effects=effects,
            causal_links=(
                CausalLink("E0", "CAUSES", "E1", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.3",
        )
        self.assertEqual(validate_effect_source_bindings(model), [])

    def test_validation_messages_have_stable_typed_repair_metadata(self):
        issues = validation_issues_from_messages((
            "E2 is a downstream human outcome with no causal parent",
            "causal_link[7] is CERTAIN but lists conditions",
        ))
        self.assertEqual(issues[0].code, "MISSING_CAUSAL_PARENT")
        self.assertEqual(issues[0].entity_id, "E2")
        self.assertEqual(issues[0].field, "causal_links")
        self.assertEqual(issues[1].code, "CERTAIN_ROW_HAS_CONDITIONS")
        self.assertEqual(issues[1].entity_kind, "causal_link")
        contract = repair_patch_contract(issues)
        self.assertEqual(contract["allowed_operations"], ["add", "replace"])
        self.assertIn("E2", contract["allowed_entity_ids"])

    def test_world_model_validation_error_retains_typed_issues(self):
        issue = validation_issues_from_messages(("E2 lacks source provenance",))[0]
        error = WorldModelValidationError((issue.message,), (issue,))
        self.assertIsInstance(error, ValueError)
        self.assertEqual(error.issues[0].entity_id, "E2")
        self.assertEqual(str(error), "E2 lacks source provenance")

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
            "validation_issues": [
                validation_issues_from_messages((
                    "E1 omits source-grounded scope qualifier 'broader'",
                ))[0].as_dict(),
            ],
            "clauses": [], "world_contradictions": [],
        }
        prompts: list[str] = []
        schemas: list[dict] = []

        def fake_call(_llm, prompt, **kwargs):
            prompts.append(prompt)
            schemas.append(kwargs["schema"])
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
        self.assertIn("do not delete the downstream human row", prompts[1].casefold())
        self.assertIn("keep counterfactual_links", prompts[1].casefold())
        self.assertIn("transferred resource", prompts[1].casefold())
        self.assertIn("Typed validation issues", prompts[1])
        self.assertIn("Transactional repair boundary", prompts[1])
        self.assertIn('"allowed_entity_ids": ["E1"]', prompts[1])
        world_schema = schemas[0]["properties"]["world_model"]
        self.assertEqual(
            world_schema["properties"]["schema_version"]["enum"], ["1.3"],
        )
        effect_required = world_schema["properties"]["effects"]["items"]["required"]
        self.assertIn("source_proposition", effect_required)
        self.assertIn("derivation_operation", effect_required)

    def test_repair_restores_dropped_counterfactual_overlays(self):
        previous = {
            "actions": {"A0": {}, "A1": {}},
            "world_model": {
                "effects": [
                    {
                        "effect_id": "E1", "action_id": "A0", "party_id": "P2",
                        "directness": "DOWNSTREAM",
                    },
                    {
                        "effect_id": "EF0", "action_id": "A0", "party_id": "P2",
                        "directness": "FOREGONE", "effect_kind": "OPPORTUNITY_LOSS",
                    },
                ],
                "actions": [
                    {"action_id": "A0", "effect_ids": ["E1", "EF0"]},
                ],
                "counterfactual_links": [
                    {
                        "action_id": "A0",
                        "source_effect_id": "EF0",
                        "alternative_action_id": "A1",
                        "alternative_effect_id": "E1",
                    },
                ],
            },
        }
        candidate = {
            "actions": {"A0": {}, "A1": {}},
            "world_model": {
                "effects": [
                    {
                        "effect_id": "E1", "action_id": "A0", "party_id": "P2",
                        "directness": "DOWNSTREAM",
                    },
                ],
                "actions": [
                    {"action_id": "A0", "effect_ids": ["E1"]},
                ],
                "counterfactual_links": [],
            },
        }
        merged = _preserve_stable_world_bookkeeping(previous, candidate)
        world = merged["world_model"]
        self.assertEqual(len(world["counterfactual_links"]), 1)
        self.assertEqual(world["counterfactual_links"][0]["source_effect_id"], "EF0")
        self.assertIn("EF0", {row["effect_id"] for row in world["effects"]})
        self.assertIn("EF0", world["actions"][0]["effect_ids"])

    def test_rejected_grounding_keeps_the_candidate(self):
        rejected_candidate = {
            "sentinel": "inspect this draft",
            "actions": {"A0": {}, "A1": {}},
            "world_model": {"schema_version": "1.2"},
        }
        rejected = {
            "status": "REJECTED", "actions": {},
            "errors": ["typed world model rejected: E2 has no causal parent"],
            "clauses": [], "world_contradictions": [],
            "world_model": {},
        }

        def fake_call(_llm, prompt, **_kwargs):
            return {"choices": [{"text": __import__("json").dumps(rejected_candidate)}]}

        with patch(
            "global_workspace.local_specialists._call_json_llm",
            side_effect=fake_call,
        ), patch(
            "global_workspace.local_specialists._admit_action_source_rows",
            return_value=rejected,
        ):
            result = ground_actions_in_scenario(
                object(),
                "Option A: send aid north. Option B: send aid south.",
                ["send aid north", "send aid south"],
                max_attempts=1,
            )

        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["world_model"], {})
        self.assertEqual(result["rejected_candidate"]["sentinel"], "inspect this draft")

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

    def test_two_count_clause_assigns_each_span_to_one_party(self):
        shared_ref = (SourceRef(
            "C5",
            "forty workers at the plant lose pay while six hundred households "
            "downstream lose water.",
        ),)
        parties = (
            WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
            WorldParty(
                "P1", "plant workers", "GROUP", shared_ref, ("forty",),
            ),
            WorldParty(
                "P2", "downstream households", "HOUSEHOLD", shared_ref,
                ("six hundred",),
            ),
            WorldParty("P3", "people", "POPULATION", shared_ref, ()),
        )
        assigned = assigned_party_quantities(parties)
        self.assertEqual(assigned["P1"], ("forty",))
        self.assertEqual(assigned["P2"], ("six hundred",))
        self.assertEqual(assigned["P3"], ())
        effects = (
            WorldEffect(
                "E1", "A0", "P1", "workers unpaid", "CAUSES", "ADVERSE",
                "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME", (), (), shared_ref,
            ),
            WorldEffect(
                "E2", "A1", "P2", "households lose water", "CAUSES", "ADVERSE",
                "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME", (), (), shared_ref,
            ),
        )
        model = ScenarioWorldModel(
            parties=parties,
            actions=(
                WorldAction("A0", "idle the plant", "P0", (), ("E1",), REF),
                WorldAction("A1", "cut the main", "P0", (), ("E2",), REF),
            ),
            effects=effects,
            schema_version="1.2",
        )
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertFalse(any("P1 omits" in error for error in errors))
        self.assertFalse(any("P2 omits" in error for error in errors))
        self.assertFalse(any("P3 omits" in error for error in errors))

        both_on_workers = (
            parties[0],
            replace(parties[1], quantities=("forty", "six hundred")),
            parties[2],
            parties[3],
        )
        assigned_wrong = assigned_party_quantities(both_on_workers)
        self.assertEqual(assigned_wrong["P1"], ("forty",))
        omitted_errors, _ = validate_world_model(
            replace(model, parties=both_on_workers), action_ids=["A0", "A1"],
        )
        self.assertFalse(any(
            "P1 omits" in error and "six hundred" in error for error in omitted_errors
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
            if row.epistemic_status in {"ESTABLISHED", "STIPULATED"}
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

    def test_party_recorded_quantity_grounds_anaphoric_effect_row(self):
        count_ref = (SourceRef("C5", "Twelve subjects remain on the device."),)
        anaphor_ref = (SourceRef("C6", "The subjects lose support."),)
        model = _model(
            (
                _effect("E1", "A0", "P1"),
                _effect(
                    "E2", "A0", "P5",
                    outcome="the subjects lose support",
                    relation="EXPERIENCES",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    quantities=("twelve",),
                    provenance=anaphor_ref,
                    effect_kind="HEALTH_OUTCOME",
                ),
                _effect("E3", "A1", "P2"),
            ),
            extra_parties=(
                WorldParty("P5", "subjects", "POPULATION", count_ref, ("twelve",)),
            ),
        )
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(errors, [])

    def test_party_quantity_does_not_license_a_different_party(self):
        count_ref = (SourceRef("C5", "Twelve subjects remain on the device."),)
        anaphor_ref = (SourceRef("C6", "The residents lose supply."),)
        model = _model(
            (
                _effect("E1", "A0", "P1"),
                _effect(
                    "E2", "A0", "P3",
                    outcome="the residents lose supply",
                    relation="EXPERIENCES",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    quantities=("twelve",),
                    provenance=anaphor_ref,
                    effect_kind="WELFARE_OUTCOME",
                ),
                _effect("E3", "A1", "P2"),
            ),
            extra_parties=(
                WorldParty("P5", "subjects", "POPULATION", count_ref, ("twelve",)),
            ),
        )
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertTrue(
            any("E2" in item and "twelve" in item and "provenance" in item for item in errors),
            errors,
        )

    def test_hyphenated_scale_matches_spaced_party_quantity(self):
        count_ref = (SourceRef(
            "C5",
            "The plant serves more than fifty thousand residents.",
        ),)
        anaphor_ref = (SourceRef("C6", "The residents lose supply."),)
        model = _model(
            (
                _effect("E1", "A0", "P1"),
                _effect(
                    "E2", "A0", "P5",
                    outcome="the residents lose supply",
                    relation="EXPERIENCES",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    quantities=("fifty-thousand",),
                    provenance=anaphor_ref,
                    effect_kind="WELFARE_OUTCOME",
                ),
                _effect("E3", "A1", "P2"),
            ),
            extra_parties=(
                WorldParty(
                    "P5", "served residents", "POPULATION",
                    count_ref, ("more than fifty thousand",),
                ),
            ),
        )
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(errors, [])

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
        self.assertEqual(downstream[0].direction, "UNCERTAIN")
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

    def test_same_sign_prevents_rejected_on_enables_chain(self):
        model = self.atomic_model()
        inverted = replace(model, causal_links=tuple(
            replace(link, relation="PREVENTS")
            if link.source_id == "A0_RESOURCE" and link.target_id == "A0_OUTPUT"
            else link
            for link in model.causal_links
        ))
        errors, _ = validate_world_model(inverted, action_ids=["A0", "A1"])
        matching = [error for error in errors if "PREVENTS" in error]
        self.assertEqual(len(matching), 1, errors)
        self.assertIn("A0_RESOURCE", matching[0])
        self.assertIn("A0_OUTPUT", matching[0])
        self.assertIn("keep the same endpoints", matching[0])
        admitted, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(admitted, [])

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

    def test_transferred_resource_is_not_a_recipient(self):
        ref = (SourceRef("C1", "The dispatcher sends the supply to the families."),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "dispatcher", "SYSTEM", REF),
                WorldParty("P1", "families", "GROUP", ref),
                WorldParty("P2", "supply", "RESOURCE", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "send the supply to the families", "P0", ("P2",),
                    ("E0", "E1"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P2", "supply sent", "TRANSFERRED", "NEUTRAL",
                    "DIRECT", "CERTAIN", "RESOURCE_TRANSFER", provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P1", "families receive the supply", "RECEIVES",
                    "BENEFICIAL", "DOWNSTREAM", "CERTAIN", "WELFARE_OUTCOME",
                    provenance=ref,
                ),
            ),
            causal_links=(
                CausalLink("E0", "CAUSES", "E1", "CERTAIN", (), ref, "A0"),
            ),
            schema_version="1.2",
        )
        errors = validate_world_completeness(model, action_ids=["A0"])
        joined = " ".join(errors)
        self.assertIn("transferred resource P2", joined)
        self.assertIn("P1 is the receiving group", joined)

    def test_receiving_group_is_the_transfer_recipient(self):
        ref = (SourceRef("C1", "The dispatcher sends the supply to the families."),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "dispatcher", "SYSTEM", REF),
                WorldParty("P1", "families", "GROUP", ref),
                WorldParty("P2", "supply", "RESOURCE", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "send the supply to the families", "P0", ("P1",),
                    ("E0",), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "families receive the supply", "RECEIVES",
                    "BENEFICIAL", "DIRECT", "CERTAIN", "RESOURCE_TRANSFER",
                    provenance=ref,
                ),
            ),
            schema_version="1.2",
        )
        errors = validate_world_completeness(model, action_ids=["A0"])
        self.assertFalse(any("transferred resource" in error for error in errors), errors)
        structural, _ = validate_world_model(model, action_ids=["A0"])
        self.assertEqual(structural, [])

    def test_demolished_resource_may_remain_a_recipient(self):
        ref = (SourceRef("C1", "The allocator demolishes the depot."),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "allocator", "AUTOMATED_SYSTEM", REF),
                WorldParty("P1", "depot", "RESOURCE", ref),
            ),
            actions=(
                WorldAction("A0", "demolish the depot", "P0", ("P1",), ("E0",), ref),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "depot demolished", "DESTROYED", "NEUTRAL",
                    "DIRECT", "CERTAIN", "INTERVENTION", provenance=ref,
                ),
            ),
            schema_version="1.2",
        )
        errors = validate_world_completeness(model, action_ids=["A0"])
        self.assertFalse(any("transferred resource" in error for error in errors), errors)

    def test_same_sign_prevents_is_a_verb_error(self):
        model = _parse_complete_magistrate()
        inverted = replace(model, causal_links=tuple(
            replace(link, relation="PREVENTS")
            if link.source_id == "E3" and link.target_id == "E4"
            else link
            for link in model.causal_links
        ))
        errors, _ = validate_world_model(inverted, action_ids=["A0", "A1"])
        matching = [error for error in errors if "PREVENTS" in error]
        self.assertEqual(len(matching), 1, errors)
        self.assertIn("E3", matching[0])
        self.assertIn("E4", matching[0])
        self.assertIn("keep the same endpoints", matching[0])
        self.assertEqual(
            validate_world_completeness(inverted, action_ids=["A0", "A1"]),
            [],
        )

    def test_neutral_prevents_is_not_same_sign_error(self):
        model = _parse_complete_magistrate()
        inverted = replace(model, causal_links=tuple(
            replace(link, relation="PREVENTS")
            if link.source_id == "E0" and link.target_id == "E1"
            else link
            for link in model.causal_links
        ))
        errors, _ = validate_world_model(inverted, action_ids=["A0", "A1"])
        self.assertFalse(any("PREVENTS" in error for error in errors), errors)

    def test_roles_count_downstream_health_not_the_mob(self):
        model = _parse_complete_magistrate()
        a0 = project_world_action_roles(model, "A0")
        a1 = project_world_action_roles(model, "A1")
        self.assertEqual(a0.beneficiaries, ("innocent individual",))
        self.assertEqual(a0.harmed, ("city residents",))
        self.assertEqual(a0.at_risk, ())
        self.assertEqual(a0.conditionally_benefited, ())
        self.assertEqual(a1.harmed, ("innocent individual",))
        self.assertEqual(a1.beneficiaries, ("city residents",))
        self.assertEqual(a1.at_risk, ())
        self.assertEqual(a1.conditionally_benefited, ())

    def test_canonical_records_project_roles_and_framing(self):
        records = build_canonical_action_records(
            [A0_TEXT, A1_TEXT],
            actor="city magistrate",
            world_model=_parse_complete_magistrate().as_dict(),
        )
        by_id = {record.action_id: record for record in records}
        self.assertEqual(by_id["A0"].beneficiaries, ("innocent individual",))
        self.assertEqual(by_id["A0"].harmed, ("city residents",))
        self.assertEqual(by_id["A0"].at_risk, ())
        self.assertEqual(by_id["A0"].conditionally_benefited, ())
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

    def test_nested_five_is_stripped_to_the_longest_span_on_parse(self):
        raw = _complete_magistrate_raw()
        raw["parties"][-1]["quantities"] = ["over five hundred", "five"]
        model = parse_world_model(
            raw,
            clauses=MAGISTRATE_CLAUSES,
            action_ids=["A0", "A1"],
            action_texts={"A0": A0_TEXT, "A1": A1_TEXT},
        )
        self.assertEqual(model.parties[-1].quantities, ("over five hundred",))

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
        self.assertEqual(model.parties[-1].quantities, ("over five hundred",))
        self.assertNotIn("five", model.parties[-1].quantities)

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
        self.assertEqual(roles.at_risk, ())
        self.assertEqual(roles.unresolved, ())

    def test_almost_no_chance_probabilistic_health_is_compact_harm(self):
        model = _model(
            (
                _effect(
                    "E_h", "A0", "P2",
                    outcome="subjects drown",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    modality="PROBABILISTIC",
                    condition_ids=("COND1",),
                    effect_kind="HEALTH_OUTCOME",
                    likelihood_qualifiers=("almost no chance",),
                ),
            ),
            conditions=(WorldCondition("COND1", "chamber fills", provenance=REF),),
        )
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.harmed, ("trapped family",))
        self.assertEqual(roles.at_risk, ())

    def test_a_chance_possible_health_is_conditionally_benefited(self):
        model = _model(
            (
                _effect(
                    "E_h", "A0", "P2",
                    outcome="subjects escape",
                    polarity="BENEFICIAL",
                    directness="DOWNSTREAM",
                    modality="POSSIBLE",
                    condition_ids=("COND1",),
                    effect_kind="HEALTH_OUTCOME",
                    likelihood_qualifiers=("a chance",),
                ),
            ),
            conditions=(WorldCondition("COND1", "escape succeeds", provenance=REF),),
        )
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.beneficiaries, ())
        self.assertEqual(roles.conditionally_benefited, ("trapped family",))
        self.assertEqual(roles.harmed, ())

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
        self.assertEqual(roles.at_risk, ("downstream residents",))
        self.assertEqual(roles.unresolved, ("downstream residents",))

    def test_settled_harm_outranks_weaker_unresolved_on_same_party(self):
        model = _model(
            (
                _effect(
                    "E_e", "A0", "P2",
                    outcome="subjects die",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    effect_kind="HEALTH_OUTCOME",
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
        self.assertEqual(roles.at_risk, ())
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

    def test_expected_probabilistic_health_is_at_risk_not_harmed(self):
        model = _model(
            (
                _effect(
                    "E_h", "A0", "P3",
                    outcome="severe long-term harm",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    modality="PROBABILISTIC",
                    condition_ids=("COND2",),
                    effect_kind="HEALTH_OUTCOME",
                    likelihood_qualifiers=("expected",),
                ),
            ),
            conditions=(WorldCondition("COND2", "exposure continues", provenance=REF),),
        )
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.harmed, ())
        self.assertEqual(roles.at_risk, ("downstream residents",))
        self.assertEqual(roles.conditionally_benefited, ())
        self.assertEqual(roles.unresolved, ("downstream residents",))

    def test_stipulated_crowd_protection_is_conditionally_benefited(self):
        model = _model(
            (
                _effect(
                    "E_p", "A1", "P3",
                    outcome="protected from harm",
                    polarity="BENEFICIAL",
                    directness="DOWNSTREAM",
                    modality="STIPULATED_CONDITIONAL",
                    condition_ids=("COND2",),
                    effect_kind="HEALTH_OUTCOME",
                ),
            ),
            conditions=(WorldCondition("COND2", "release ends", provenance=REF),),
        )
        roles = project_world_action_roles(model, "A1")
        self.assertEqual(roles.beneficiaries, ())
        self.assertEqual(roles.conditionally_benefited, ("downstream residents",))
        self.assertEqual(roles.at_risk, ())
        self.assertEqual(roles.unresolved, ("downstream residents",))

    def test_near_certain_stipulated_conditional_is_at_risk(self):
        gated = _model(
            (
                _effect(
                    "E_g", "A0", "P5",
                    outcome="fails",
                    polarity="NEUTRAL",
                    directness="DOWNSTREAM",
                    modality="PROBABILISTIC",
                    effect_kind="PHYSICAL_STATE",
                    likelihood_qualifiers=("20% chance",),
                ),
                _effect(
                    "E_h", "A0", "P2",
                    outcome="subjects die",
                    polarity="ADVERSE",
                    directness="DOWNSTREAM",
                    modality="STIPULATED_CONDITIONAL",
                    condition_ids=("COND1",),
                    effect_kind="HEALTH_OUTCOME",
                    likelihood_qualifiers=("near-certain",),
                ),
            ),
            conditions=(
                WorldCondition(
                    "COND1", "backup fails", provenance=REF, event_effect_id="E_g",
                ),
            ),
            extra_parties=(WorldParty("P5", "backup", "PROCESS", REF),),
        )
        roles = project_world_action_roles(gated, "A0")
        self.assertEqual(roles.harmed, ())
        self.assertEqual(roles.at_risk, ("trapped family",))
        self.assertEqual(roles.beneficiaries, ())

    def test_certain_at_moderate_risk_is_at_risk_not_harmed(self):
        model = _model((
            _effect(
                "E_r", "A0", "P2",
                outcome="at moderate risk",
                polarity="ADVERSE",
                directness="DOWNSTREAM",
                effect_kind="PHYSICAL_STATE",
                likelihood_qualifiers=("at moderate risk",),
            ),
        ))
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.harmed, ())
        self.assertEqual(roles.at_risk, ("trapped family",))
        self.assertEqual(roles.unresolved, ("trapped family",))

    def test_crowd_use_process_is_not_a_compact_role(self):
        model = _model((
            _effect(
                "E_u", "A0", "P3",
                outcome="residents use the access route",
                polarity="NEUTRAL",
                directness="DOWNSTREAM",
                effect_kind="PHYSICAL_STATE",
            ),
        ))
        roles = project_world_action_roles(model, "A0")
        self.assertEqual(roles.beneficiaries, ())
        self.assertEqual(roles.harmed, ())
        self.assertEqual(roles.at_risk, ())


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

    def test_crowd_use_process_may_hang_off_another_party_transfer(self):
        clauses = [
            {
                "clause_id": "C0",
                "text": (
                    "A dispatcher must send the supply to the families or "
                    "keep it in reserve."
                ),
            },
            {
                "clause_id": "C1",
                "text": (
                    "Sending the supply to the families means the workers "
                    "use the access route."
                ),
            },
            {
                "clause_id": "C2",
                "text": "The workers are trapped.",
            },
            {
                "clause_id": "C3",
                "text": "Keeping the supply in reserve leaves the families without it.",
            },
        ]
        raw = {
            "schema_version": "1.2",
            "parties": [
                {
                    "party_id": "P0", "label": "dispatcher", "kind": "SYSTEM",
                    "quantities": [], "clause_ids": ["C0"],
                },
                {
                    "party_id": "P1", "label": "families", "kind": "GROUP",
                    "quantities": [], "clause_ids": ["C0"],
                },
                {
                    "party_id": "P2", "label": "workers", "kind": "POPULATION",
                    "quantities": [], "clause_ids": ["C1", "C2"],
                },
                {
                    "party_id": "P3", "label": "supply", "kind": "RESOURCE",
                    "quantities": [], "clause_ids": ["C0"],
                },
                {
                    "party_id": "P4", "label": "access route",
                    "kind": "INFRASTRUCTURE", "quantities": [],
                    "clause_ids": ["C1"],
                },
            ],
            "actions": [
                {
                    "action_id": "A0",
                    "intervention": "send the supply to the families",
                    "actor_party_id": "P0", "recipient_party_ids": ["P1"],
                    "effect_ids": ["E0", "E1", "E2"], "clause_ids": ["C1"],
                },
                {
                    "action_id": "A1",
                    "intervention": "keep the supply in reserve",
                    "actor_party_id": "P0", "recipient_party_ids": [],
                    "effect_ids": ["E3"], "clause_ids": ["C3"],
                },
            ],
            "effects": [
                _raw_effect(
                    effect_id="E0", action_id="A0", party_id="P1",
                    outcome="families receive the supply", relation="RECEIVES",
                    polarity="BENEFICIAL", directness="DIRECT",
                    effect_kind="RESOURCE_TRANSFER",
                    clause_ids=["C1", "A0"],
                ),
                _raw_effect(
                    effect_id="E1", action_id="A0", party_id="P2",
                    outcome="workers use the access route", relation="USES",
                    polarity="NEUTRAL", directness="DOWNSTREAM",
                    effect_kind="PHYSICAL_STATE", clause_ids=["C1"],
                ),
                _raw_effect(
                    effect_id="E2", action_id="A0", party_id="P2",
                    outcome="workers trapped", relation="EXPERIENCES",
                    polarity="ADVERSE", directness="DOWNSTREAM",
                    effect_kind="WELFARE_OUTCOME", clause_ids=["C2"],
                ),
                _raw_effect(
                    effect_id="E3", action_id="A1", party_id="P0",
                    outcome="supply kept in reserve", relation="PERFORMS",
                    polarity="NEUTRAL", directness="DIRECT",
                    effect_kind="INTERVENTION", clause_ids=["C3", "A1"],
                ),
            ],
            "conditions": [],
            "causal_links": [
                _raw_link("A0", "E0", "E1", "C1"),
                _raw_link("A0", "E1", "E2", "C2"),
            ],
            "counterfactual_links": [],
        }
        model = parse_world_model(
            raw,
            clauses=clauses,
            action_ids=["A0", "A1"],
            action_texts={
                "A0": "send the supply to the families",
                "A1": "keep the supply in reserve",
            },
        )
        self.assertEqual(validate_world_completeness(model, action_ids=["A0", "A1"]), [])
        errors, _ = validate_world_model(model, action_ids=["A0", "A1"])
        self.assertEqual(errors, [])

    def test_adverse_crowd_use_process_is_rejected(self):
        clauses = [
            {
                "clause_id": "C0",
                "text": "A dispatcher must send the supply to the families.",
            },
            {
                "clause_id": "C1",
                "text": "Sending the supply means the workers use the access route.",
            },
        ]
        raw = {
            "schema_version": "1.2",
            "parties": [
                {
                    "party_id": "P0", "label": "dispatcher", "kind": "SYSTEM",
                    "quantities": [], "clause_ids": ["C0"],
                },
                {
                    "party_id": "P1", "label": "families", "kind": "GROUP",
                    "quantities": [], "clause_ids": ["C0"],
                },
                {
                    "party_id": "P2", "label": "workers", "kind": "POPULATION",
                    "quantities": [], "clause_ids": ["C1"],
                },
                {
                    "party_id": "P3", "label": "supply", "kind": "RESOURCE",
                    "quantities": [], "clause_ids": ["C0"],
                },
            ],
            "actions": [
                {
                    "action_id": "A0",
                    "intervention": "send the supply to the families",
                    "actor_party_id": "P0", "recipient_party_ids": ["P1"],
                    "effect_ids": ["E0", "E1"], "clause_ids": ["C0", "C1"],
                },
                {
                    "action_id": "A1",
                    "intervention": "keep the supply in reserve",
                    "actor_party_id": "P0", "recipient_party_ids": [],
                    "effect_ids": ["E2"], "clause_ids": ["C0"],
                },
            ],
            "effects": [
                _raw_effect(
                    effect_id="E0", action_id="A0", party_id="P1",
                    outcome="families receive the supply", relation="RECEIVES",
                    polarity="BENEFICIAL", directness="DIRECT",
                    effect_kind="RESOURCE_TRANSFER",
                    clause_ids=["C0", "C1", "A0"],
                ),
                _raw_effect(
                    effect_id="E1", action_id="A0", party_id="P2",
                    outcome="workers use the access route", relation="USES",
                    polarity="ADVERSE", directness="DOWNSTREAM",
                    effect_kind="PHYSICAL_STATE", clause_ids=["C1"],
                ),
                _raw_effect(
                    effect_id="E2", action_id="A1", party_id="P0",
                    outcome="supply kept in reserve", relation="PERFORMS",
                    polarity="NEUTRAL", directness="DIRECT",
                    effect_kind="INTERVENTION", clause_ids=["C0", "A1"],
                ),
            ],
            "conditions": [],
            "causal_links": [_raw_link("A0", "E0", "E1", "C1")],
            "counterfactual_links": [],
        }
        with self.assertRaisesRegex(ValueError, "crowd process state"):
            parse_world_model(
                raw, clauses=clauses, action_ids=["A0", "A1"],
                action_texts={
                    "A0": "send the supply to the families",
                    "A1": "keep the supply in reserve",
                },
            )

    def test_does_not_increase_is_not_a_causal_parent(self):
        excerpt = (
            "The dispatcher sends the supply to the families. Using the supply "
            "does not increase the chance the backup process fails. There is a "
            "20% chance the backup process fails."
        )
        ref = (SourceRef("C1", excerpt),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "dispatcher", "SYSTEM", REF),
                WorldParty("P1", "families", "GROUP", ref),
                WorldParty("P2", "supply", "RESOURCE", ref),
                WorldParty("P3", "backup", "PROCESS", ref),
            ),
            actions=(
                WorldAction(
                    "A0", "send the supply to the families", "P0", ("P1",),
                    ("E0", "E1"), ref,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "families receive the supply", "RECEIVES",
                    "BENEFICIAL", "DIRECT", "CERTAIN", "RESOURCE_TRANSFER",
                    provenance=ref,
                ),
                WorldEffect(
                    "E1", "A0", "P3", "fails", "STATE_CHANGE", "NEUTRAL",
                    "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
                    likelihood_qualifiers=("20% chance",), provenance=ref,
                ),
            ),
            causal_links=(
                CausalLink(
                    "E0", "DOES_NOT_INCREASE", "E1", "CERTAIN", (), ref, "A0",
                ),
            ),
            schema_version="1.2",
        )
        self.assertEqual(validate_world_completeness(model, action_ids=["A0"]), [])

    def test_direct_effect_clause_must_be_cited_by_the_action(self):
        excerpt = "Five workers stay to assist at moderate risk."
        background = "The dispatcher sends the supply to the families."
        ref_a = (SourceRef("C0", background),)
        ref_b = (SourceRef("C1", excerpt),)
        model = ScenarioWorldModel(
            parties=(
                WorldParty("P0", "dispatcher", "SYSTEM", ref_a),
                WorldParty("P1", "families", "GROUP", ref_a),
                WorldParty("P2", "workers", "POPULATION", ref_b, ("five",)),
            ),
            actions=(
                WorldAction(
                    "A0", "send the supply to the families", "P0", ("P1", "P2"),
                    ("E0", "E1"), ref_a,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "families receive the supply", "RECEIVES",
                    "BENEFICIAL", "DIRECT", "CERTAIN", "RESOURCE_TRANSFER",
                    provenance=ref_a,
                ),
                WorldEffect(
                    "E1", "A0", "P2", "workers assigned to assist", "ASSIGNED",
                    "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
                    provenance=ref_b, quantities=("five",),
                ),
            ),
            schema_version="1.2",
        )
        errors = validate_world_completeness(model, action_ids=["A0"])
        self.assertTrue(any(
            "omits source clauses" in error and "C1" in error for error in errors
        ), errors)

    def test_party_keeps_total_count_and_assignment_keeps_subgroup(self):
        c0 = "A crew of fifty workers is available."
        c1 = "Five workers stay to assist."
        parties = (
            WorldParty("P0", "dispatcher", "HUMAN", (SourceRef("C0", c0),)),
            WorldParty(
                "P1", "workers", "POPULATION",
                (SourceRef("C0", c0), SourceRef("C1", c1)),
            ),
        )
        assigned = assigned_party_quantities(parties)
        self.assertIn("fifty", assigned["P1"])
        effect = WorldEffect(
            "E1", "A0", "P1", "five workers assigned", "ASSIGNED", "NEUTRAL",
            "DIRECT", "CERTAIN", "INTERVENTION",
            quantities=("five",),
            provenance=(SourceRef("C1", c1),),
        )
        self.assertEqual(effect.quantities, ("five",))

    def test_direct_intervention_on_recipient_facility_admits(self):
        raw = _connected_allocator_raw()
        raw["actions"][0]["recipient_party_ids"] = ["P1", "P4"]
        raw["actions"][0]["clause_ids"] = ["C0", "C1"]
        for effect in raw["effects"]:
            if effect["effect_id"] == "E1":
                effect["directness"] = "DIRECT"
                effect["effect_kind"] = "INTERVENTION"
                effect["outcome"] = "spillway repaired"
                effect["polarity"] = "NEUTRAL"
        model = _parse_allocator(raw)
        self.assertEqual(validate_world_completeness(model, action_ids=["A0", "A1"]), [])

    def test_direct_intervention_on_recipient_infrastructure_admits(self):
        raw = _connected_allocator_raw()
        for party in raw["parties"]:
            if party["party_id"] == "P4":
                party["kind"] = "INFRASTRUCTURE"
        raw["actions"][0]["recipient_party_ids"] = ["P1", "P4"]
        raw["actions"][0]["clause_ids"] = ["C0", "C1"]
        for effect in raw["effects"]:
            if effect["effect_id"] == "E1":
                effect["directness"] = "DIRECT"
                effect["effect_kind"] = "INTERVENTION"
                effect["outcome"] = "channel diverted"
                effect["polarity"] = "NEUTRAL"
        model = _parse_allocator(raw)
        self.assertEqual(validate_world_completeness(model, action_ids=["A0", "A1"]), [])

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
        self.assertIn("Do not delete E2", str(raised.exception))

    def test_foreign_parent_error_names_existing_process_row(self):
        raw = _connected_allocator_raw()
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
        message = str(raised.exception)
        self.assertIn("E2", message)
        self.assertIn("E_h", message)
        self.assertIn("E1", message)
        self.assertNotIn("reparent onto existing", message)
        self.assertIn("Do not delete E2", message)

    def test_foreign_parent_error_names_unused_source_facility(self):
        raw = _connected_allocator_raw()
        raw["parties"].append({
            "party_id": "P5", "label": "reservoir", "kind": "FACILITY",
            "quantities": [], "clause_ids": ["C1"],
        })
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
        clauses = [dict(row) for row in CLAUSES]
        clauses[1] = {
            "clause_id": "C1",
            "text": (
                "If the engineer is reached, reservoir holding would spare "
                "tens of thousands of downstream residents."
            ),
        }
        with self.assertRaises(ValueError) as raised:
            parse_world_model(
                raw,
                clauses=clauses,
                action_ids=["A0", "A1"],
                action_texts={"A0": ALLOCATOR_A0, "A1": ALLOCATOR_A1},
            )
        message = str(raised.exception)
        self.assertIn("E2", message)
        self.assertIn("P5", message)
        self.assertIn("reservoir", message)
        self.assertIn("plausible", message)
        self.assertNotIn("reparent onto existing", message)

    def test_dropping_the_illegal_parent_still_leaves_an_orphan(self):
        raw = _connected_allocator_raw()
        raw["causal_links"] = [
            _raw_link("A0", "E0", "E1", "C1"),
            _raw_link("A1", "E3", "E4", "C2"),
        ]
        with self.assertRaises(ValueError) as raised:
            _parse_allocator(raw)
        message = str(raised.exception)
        self.assertIn("E2", message)
        self.assertIn("no causal parent", message)
        self.assertIn("Do not drop E2", message)

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


def _admitted_comparison_graph() -> SemanticGraph:
    """Two-action world: gated trapping vs gated death, plus protective walls."""
    wall_ref = (SourceRef(
        "C4",
        "Fire-resistant walls make deaths unlikely. Crews remaining on site "
        "face moderate risk.",
    ),)
    trap_ref = (SourceRef(
        "C5",
        "The narrow road has a 10% chance of blockage. If blocked, about sixty "
        "residents could be trapped. Remaining patients escape almost certainly.",
    ),)
    parties = (
        WorldParty("P0", "coordinator", "INSTITUTION", REF),
        WorldParty("P1", "eastern residents", "POPULATION", trap_ref, ("about sixty",)),
        WorldParty("P2", "immobile patients", "POPULATION", trap_ref),
        WorldParty("P3", "site crews", "POPULATION", wall_ref, ("five",)),
        WorldParty("P4", "clinic", "FACILITY", wall_ref),
        WorldParty("P5", "narrow road", "FACILITY", trap_ref),
    )
    effects = (
        WorldEffect(
            "E0_WALLS", "A0", "P4", "fire-resistant walls", "STATE_CHANGE",
            "NEUTRAL", "DOWNSTREAM", "CERTAIN", "PHYSICAL_STATE", (), (), wall_ref,
        ),
        WorldEffect(
            "E0_DEATH", "A0", "P2", "NEAR_CERTAIN_DEATH", "CAUSES",
            "ADVERSE", "DOWNSTREAM", "STIPULATED_CONDITIONAL", "HEALTH_OUTCOME",
            ("COND0",), (), wall_ref, ("near-certain",), (), (), "",
            ("deaths unlikely",),
        ),
        WorldEffect(
            "E0_RISK", "A0", "P3", "AT_MODERATE_RISK", "CAUSES",
            "ADVERSE", "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME", (), (), wall_ref,
        ),
        WorldEffect(
            "E1_ESCAPE", "A1", "P2", "ESCAPE_SAFELY", "CAUSES",
            "BENEFICIAL", "DOWNSTREAM", "PROBABILISTIC", "HEALTH_OUTCOME",
            (), (), trap_ref, ("almost certain",),
        ),
        WorldEffect(
            "E1_BLOCK", "A1", "P5", "BLOCKAGE", "CAUSES",
            "ADVERSE", "DOWNSTREAM", "PROBABILISTIC", "PHYSICAL_STATE",
            (), (), trap_ref, ("10% chance",),
        ),
        WorldEffect(
            "E1_TRAP", "A1", "P1", "TRAPPED", "CAUSES",
            "ADVERSE", "DOWNSTREAM", "STIPULATED_CONDITIONAL", "WELFARE_OUTCOME",
            ("COND1",), ("about sixty",), trap_ref, ("could",),
        ),
    )
    model = ScenarioWorldModel(
        parties=parties,
        actions=(
            WorldAction("A0", "keep the crew at the clinic", "P0", (), ("E0_WALLS", "E0_DEATH", "E0_RISK"), wall_ref),
            WorldAction("A1", "send the crew to the road", "P0", (), ("E1_ESCAPE", "E1_BLOCK", "E1_TRAP"), trap_ref),
        ),
        effects=effects,
        conditions=(
            WorldCondition("COND0", "clinic ventilation fails", "STATED", "MATERIAL", wall_ref),
            WorldCondition("COND1", "the road is blocked", "STATED", "MATERIAL", trap_ref, "E1_BLOCK"),
        ),
        schema_version="1.2",
    )
    graph = SemanticGraph()
    graph.add_node(SemanticNode(
        "A0", "ACTION", "keep the crew at the clinic", ("scenario_action_set",),
        {"canonical_action_id": "A0"},
    ))
    graph.add_node(SemanticNode(
        "A1", "ACTION", "send the crew to the road", ("scenario_action_set",),
        {"canonical_action_id": "A1"},
    ))
    attach_typed_world_model(graph, model)
    return graph


class AdmittedWorldGroundingTests(unittest.TestCase):
    def test_admitted_alternative_action_facts_rebind_as_stipulated(self):
        ledger = seed_proposition_ledger(_admitted_comparison_graph())
        trapped = ledger["PROP:WORLD:E1_TRAP"]
        self.assertEqual(trapped.epistemic_status, "STIPULATED")
        self.assertEqual(
            resolve_proposition(
                ledger, "TRAPPED; affected subject: eastern residents; about sixty if road blocked",
            ),
            "PROP:WORLD:E1_TRAP",
        )
        bound = resolve_proposition(ledger, "10 % chance road blocked")
        self.assertTrue(bound.startswith("PROP:WORLD:E1_BLOCK"))
        self.assertEqual(ledger[bound].epistemic_status, "STIPULATED")
        escape = resolve_proposition(
            ledger, "ESCAPE_SAFELY; affected subject: immobile patients",
        )
        self.assertEqual(escape, "PROP:WORLD:E1_ESCAPE")
        self.assertEqual(ledger[escape].epistemic_status, "STIPULATED")
        candidate = _epistemic_candidate()
        candidate.material_empirical_claims = [{
            "claim": "TRAPPED; affected subject: eastern residents; about sixty if road blocked",
            "proposition_id": "HYPOTHESIS",
            "decision_critical": True,
        }]
        attach_candidate_dependencies(ledger, candidate)
        self.assertEqual(
            candidate.decision_critical_proposition_ids, ["PROP:WORLD:E1_TRAP"],
        )
        self.assertEqual(candidate.weakest_decision_critical_status, "STIPULATED")
        self.assertFalse(any(
            record.proposition_type == "HYPOTHESIS" for record in ledger.values()
        ))

    def test_expected_trapped_is_not_expected_deaths(self):
        ledger = seed_proposition_ledger(_admitted_comparison_graph())
        self.assertTrue(claim_changes_admitted_outcome_type(
            "A1 entails roughly 6 expected deaths among the eastern residents",
            ledger,
            derived_from=["PROP:WORLD:E1_TRAP"],
        ))
        self.assertTrue(claim_changes_admitted_outcome_type(
            "A0 entails roughly 4 expected deaths among the immobile patients",
            ledger,
        ))
        self.assertFalse(claim_changes_admitted_outcome_type(
            "TRAPPED; affected subject: eastern residents; about sixty",
            ledger,
            derived_from=["PROP:WORLD:E1_TRAP"],
        ))
        hypothesis_id = register_hypothesis(
            ledger,
            "A1 entails roughly 6 expected deaths among the eastern residents",
            specialist="utilitarian",
            derived_from=["PROP:WORLD:E1_TRAP"],
            decision_critical=True,
        )
        data = {
            "proposition_ledger": [record.to_dict() for record in ledger.values()],
        }
        candidate = {
            "supporting_proposition_ids": [hypothesis_id],
            "decision_critical_proposition_ids": [hypothesis_id],
        }
        self.assertEqual(_candidate_epistemic_qualification(data, candidate), "")
        lines = _factual_status_lines(data, [candidate])
        joined = "\n".join(lines)
        self.assertNotIn("6 expected deaths", joined)
        self.assertIn("Stipulated in the admitted world", joined)
        self.assertIn("TRAPPED", joined)

    def test_protective_walls_seed_a_relation_not_an_orphan_row(self):
        ledger = seed_proposition_ledger(_admitted_comparison_graph())
        relations = [
            record for record in ledger.values()
            if ":PROTECTS:" in record.proposition_id
        ]
        self.assertEqual(len(relations), 1)
        self.assertIn("fire-resistant walls", relations[0].claim.casefold())
        self.assertIn("unlikely", relations[0].claim.casefold())
        lines = _factual_status_lines(
            {"proposition_ledger": [record.to_dict() for record in ledger.values()]},
            [],
        )
        established = "\n".join(lines)
        self.assertIn("makes", established.casefold())
        walls_lines = [
            line for line in lines
            if "fire-resistant walls" in line.casefold() and "makes" not in line.casefold()
        ]
        self.assertFalse(walls_lines)

    def test_public_brief_uses_short_action_labels(self):
        long_a0 = (
            "Send the only buses east so nearly all 300 residents evacuate safely "
            "while the 20 immobile patients remain behind protected by fire-resistant "
            "walls but face a 20% chance of ventilation failure"
        )
        long_a1 = (
            "Send the only buses to the hospital so the 20 immobile patients escape "
            "almost certainly while the 300 eastern residents must flee via a narrow "
            "road carrying a 10% blockage risk"
        )
        brief = render_decision_brief({
            "judgment_status": "CONTESTED_RECOMMENDATION",
            "selected_action": long_a0,
            "current_plurality": long_a0,
            "canonical_action_records": [
                {
                    "canonical_action_id": "A0",
                    "canonical_semantic_action": long_a0,
                    "short_label": "Send buses east to evacuate residents",
                },
                {
                    "canonical_action_id": "A1",
                    "canonical_semantic_action": long_a1,
                    "short_label": "Send buses to hospital to evacuate patients",
                },
            ],
            "cycles": [{
                "candidates": [{
                    "specialist": "utilitarian",
                    "schema_valid": True,
                    "recommended_action": long_a0,
                    "adjudication_status": "SUPPORTS",
                    "decision_rule": "prefer the larger rescue",
                    "rationale": "fewer expected harms",
                }],
                "policy": {long_a0: 0.7, long_a1: 0.3},
            }],
        })
        self.assertIn("Send buses east to evacuate residents", brief)
        self.assertNotIn("Send the only buses east so nearly all 300", brief)

    def test_malformed_empty_position_is_not_supports(self):
        chunk = CandidateChunk(
            specialist="utilitarian", constraint="NONE",
            action_scores={"keep the crew at the clinic": 0.5, "send the crew to the road": 0.5},
            surprise=0.0, friction=0.0, confidence=0.0, epistemic_confidence=0.0,
            schema_valid=False, recommended_action="",
            adjudication_status="SUPPORTS", governing_eligible=True,
            policy_weight_factor=1.0,
        )
        profile = apply_specialist_authority(chunk)
        self.assertEqual(profile.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertFalse(profile.governing_eligible)
        self.assertEqual(profile.policy_weight_factor, 0.0)
        self.assertEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        abstention = CandidateChunk(
            specialist="utilitarian", constraint="NONE",
            action_scores={"keep the crew at the clinic": 0.5, "send the crew to the road": 0.5},
            surprise=0.0, friction=0.0, confidence=0.0, epistemic_confidence=0.0,
            schema_valid=True, recommended_action="?",
            adjudication_status="SUPPORTS", governing_eligible=True,
            policy_weight_factor=1.0,
        )
        abstention_profile = apply_specialist_authority(abstention)
        self.assertEqual(abstention_profile.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(abstention_profile.policy_weight_factor, 0.0)


if __name__ == "__main__":
    unittest.main()
