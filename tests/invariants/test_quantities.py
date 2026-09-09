"""Unadmitted-magnitude ranking, bound-quantity normalization, and unique
party assignment.

Magnitude oracles are MagnitudeRankingCase.should_revoke. Bound-quantity
oracles are BoundQuantityCase.canonical. Party-uniqueness oracles are
QuantityPartyCase.expected. Production is only asked whether Util lost
its unique ranking, whether the declared span is the one admission
records, or whether assigned_party_quantities matches the declared map.
"""
from __future__ import annotations

import unittest

from dataclasses import replace

from hypothesis import given, settings

from global_workspace.scenario_semantics import attach_typed_world_model
from global_workspace.semantic_graph import SemanticGraph, SemanticNode
from global_workspace.epistemic_ledger import (
    HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
    UNADMITTED_MAGNITUDE_NOTE,
    attach_candidate_dependencies,
)
from global_workspace.models import CandidateChunk
from global_workspace.specialist_authority import apply_specialist_authority
from global_workspace.world_state import (
    _nested_recorded_quantities,
    _quantity_core,
    assigned_party_quantities,
    canonical_quantity_span,
    closed_class_party_quantities,
    explicit_quantity_spans,
    parse_world_model,
    validate_world_completeness,
    validate_world_model,
    world_model_as_parse_payload,
)
from strategies.bound_quantities import (
    BoundQuantityCase,
    PlainQuantityCase,
    QuantityPartyCase,
    bound_quantity_cases,
    plain_quantity_cases,
    quantity_party_cases,
)
from strategies.quantities import (
    CLOSED_WORLD_RANKING_NOTE,
    ClosedWorldHypothesisCase,
    MagnitudeRankingCase,
    closed_world_hypothesis_cases,
    magnitude_ranking_cases,
)


def _chunk(case: MagnitudeRankingCase) -> CandidateChunk:
    return CandidateChunk(
        specialist="utilitarian",
        constraint="IMMINENT_HARM",
        action_scores={case.actions[0]: 0.38, case.actions[1]: 0.62},
        surprise=0.1,
        friction=0.24,
        confidence=0.8,
        recommended_action=case.actions[1],
        preference_strength=0.24,
        epistemic_confidence=0.8,
        schema_valid=True,
        decision_rule=case.decision_rule,
        utilitarian_consequence_table=case.table,
    )


class QuantityInvariantTests(unittest.TestCase):
    @given(magnitude_ranking_cases())
    @settings(max_examples=40, deadline=None)
    def test_unadmitted_magnitude_cannot_uniquely_rank(
        self, case: MagnitudeRankingCase,
    ):
        chunk = _chunk(case)
        attach_candidate_dependencies(case.ledger, chunk)
        revoked = chunk.adjudication_status == "CONTESTED_NO_LEANING"
        noted = any(
            UNADMITTED_MAGNITUDE_NOTE in note
            for note in chunk.epistemic_binding_notes
        )
        self.assertEqual(revoked, case.should_revoke)
        self.assertEqual(noted, case.should_revoke)
        if case.should_revoke:
            self.assertEqual(chunk.recommended_action, "")
            self.assertAlmostEqual(chunk.action_scores[case.actions[0]], 0.5)
            self.assertAlmostEqual(chunk.action_scores[case.actions[1]], 0.5)
            profile = apply_specialist_authority(chunk)
            self.assertEqual(profile.adjudication_status, "CONTESTED_NO_LEANING")
            self.assertEqual(profile.policy_weight_factor, 0.0)
        else:
            self.assertEqual(chunk.recommended_action, case.actions[1])
            self.assertGreater(
                chunk.action_scores[case.actions[1]],
                chunk.action_scores[case.actions[0]],
            )


def _closed_world_chunk(case: ClosedWorldHypothesisCase) -> CandidateChunk:
    return CandidateChunk(
        specialist="utilitarian",
        constraint="IMMINENT_HARM",
        action_scores={case.actions[0]: 0.25, case.actions[1]: 0.75},
        surprise=0.1,
        friction=0.5,
        confidence=0.9,
        recommended_action=case.actions[1],
        preference_strength=0.5,
        epistemic_confidence=0.9,
        schema_valid=True,
        decision_rule="prefer the second action if later unverified harm dominates",
        utilitarian_consequence_table=case.table,
        utilitarian_decision_depends_on_unknown=True,
        comparison_complete=True,
        adjudication_status="SUPPORTS",
        governing_eligible=True,
        selection_status="SELECTED",
        material_empirical_claims=[{
            "claim": case.claim,
            "proposition_id": "HYPOTHESIS",
            "decision_critical": True,
        }],
    )


class ClosedWorldHypothesisTests(unittest.TestCase):
    @given(closed_world_hypothesis_cases())
    @settings(max_examples=40, deadline=None)
    def test_hypothesis_does_not_move_closed_world_scores(
        self, case: ClosedWorldHypothesisCase,
    ):
        chunk = _closed_world_chunk(case)
        attach_candidate_dependencies(case.ledger, chunk)
        restored = chunk.recommended_action == case.actions[0]
        noted = any(
            CLOSED_WORLD_RANKING_NOTE in note
            for note in chunk.epistemic_binding_notes
        )
        self.assertEqual(restored, case.should_restore)
        self.assertEqual(noted, case.should_restore)
        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertLessEqual(
            chunk.epistemic_confidence, HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
        )
        self.assertEqual(chunk.unresolved, "DECISION_BOUNDARY")
        if case.should_restore:
            self.assertGreater(
                chunk.action_scores[case.actions[0]],
                chunk.action_scores[case.actions[1]],
            )
            self.assertFalse(chunk.utilitarian_decision_depends_on_unknown)
            self.assertTrue(chunk.comparison_complete)
            profile = apply_specialist_authority(chunk)
            self.assertEqual(profile.adjudication_status, "SUPPORTS")
        else:
            self.assertEqual(chunk.recommended_action, case.actions[1])
            self.assertGreater(
                chunk.action_scores[case.actions[1]],
                chunk.action_scores[case.actions[0]],
            )


def _admission_errors(world) -> list[str]:
    structural, _contradictions = validate_world_model(world, action_ids=["A0"])
    complete = validate_world_completeness(world, action_ids=["A0"])
    return [*structural, *complete]


class BoundQuantityNormalizationTests(unittest.TestCase):
    @given(bound_quantity_cases())
    @settings(max_examples=50, deadline=None)
    def test_bound_phrase_is_one_canonical_span(self, case: BoundQuantityCase):
        spans = explicit_quantity_spans(case.source)
        self.assertEqual(spans, (case.canonical,))
        self.assertTrue(
            spans[0].casefold().startswith(case.prefix.casefold()),
            (spans, case.prefix),
        )
        self.assertEqual(_quantity_core(spans[0]), _quantity_core(case.core))
        self.assertEqual(
            assigned_party_quantities(case.world.parties)["P2"],
            (case.canonical,),
        )
        self.assertEqual(
            case.world.effects[-1].quantities, (case.canonical,),
        )
        self.assertEqual(_admission_errors(case.world), [])
        self.assertEqual(
            canonical_quantity_span(case.canonical), case.canonical,
        )
        self.assertEqual(
            canonical_quantity_span(canonical_quantity_span(case.canonical)),
            canonical_quantity_span(case.canonical),
        )
        self.assertEqual(
            explicit_quantity_spans(f"{case.canonical} {case.noun}"),
            (case.canonical,),
        )
        self.assertEqual(_nested_recorded_quantities((case.canonical,)), ())
        self.assertIn(
            case.core.casefold(),
            {item.casefold() for item in _nested_recorded_quantities(
                (case.canonical, case.core),
            )},
        )


class OrdinaryQuantitySpanTests(unittest.TestCase):
    @given(plain_quantity_cases())
    @settings(max_examples=40, deadline=None)
    def test_plain_quantity_is_one_canonical_span(self, case: PlainQuantityCase):
        spans = explicit_quantity_spans(case.source)
        self.assertIn(case.canonical, spans)
        self.assertEqual(
            canonical_quantity_span(case.canonical), case.canonical,
        )
        self.assertEqual(
            canonical_quantity_span(canonical_quantity_span(case.canonical)),
            canonical_quantity_span(case.canonical),
        )
        self.assertEqual(_nested_recorded_quantities((case.canonical,)), ())
        if case.binds_to_party:
            self.assertEqual(spans, (case.canonical,))
            self.assertEqual(
                assigned_party_quantities(case.world.parties)["P2"],
                (case.canonical,),
            )
            self.assertEqual(_admission_errors(case.world), [])
        else:
            assigned = assigned_party_quantities(case.world.parties)
            self.assertNotIn(
                case.canonical.casefold(),
                {item.casefold() for values in assigned.values() for item in values},
            )
            self.assertEqual(_admission_errors(case.world), [])


class QuantityPartyUniquenessTests(unittest.TestCase):
    @given(quantity_party_cases())
    @settings(max_examples=50, deadline=None)
    def test_quantity_attaches_to_the_unique_party(self, case: QuantityPartyCase):
        spans = explicit_quantity_spans(case.source)
        self.assertEqual(set(spans), {case.left_span, case.right_span})
        for inner in case.nested_inners:
            self.assertNotIn(inner, spans)
        assigned = assigned_party_quantities(case.world.parties)
        for party_id, expected in case.expected.items():
            self.assertEqual(assigned[party_id], expected)
        self.assertEqual(assigned[case.world.parties[0].party_id], ())
        self.assertEqual(_admission_errors(case.world), [])

        leaked = tuple(
            replace(party, quantities=(case.left_span, case.right_span))
            if party.party_id == case.left_id
            else replace(party, quantities=())
            if party.party_id in {case.right_id, case.generic_id}
            else party
            for party in case.world.parties
        )
        stripped = closed_class_party_quantities(leaked)
        stripped_by_id = {party.party_id: party.quantities for party in stripped}
        self.assertEqual(stripped_by_id[case.left_id], (case.left_span,))
        self.assertEqual(stripped_by_id[case.right_id], (case.right_span,))
        self.assertEqual(stripped_by_id[case.generic_id], ())
        leaked_errors, _ = validate_world_model(
            replace(case.world, parties=leaked), action_ids=["A0"],
        )
        self.assertFalse(any(
            case.left_id in error
            and "omits" in error
            and case.right_span in error
            for error in leaked_errors
        ))

        payload = world_model_as_parse_payload(replace(case.world, parties=leaked))
        parsed = parse_world_model(
            payload,
            clauses=[{"clause_id": "C0", "text": case.source}],
            action_ids=["A0"],
        )
        parsed_by_id = {party.party_id: party.quantities for party in parsed.parties}
        self.assertEqual(parsed_by_id[case.left_id], (case.left_span,))
        self.assertEqual(parsed_by_id[case.right_id], (case.right_span,))
        self.assertEqual(parsed_by_id[case.generic_id], ())

        graph = SemanticGraph()
        graph.add_node(SemanticNode(
            "A0", "ACTION", parsed.actions[0].intervention, ("test",), {},
        ))
        attach_typed_world_model(graph, parsed)
        self.assertEqual(
            tuple(graph.nodes[f"PARTY:{case.left_id}"].attributes.get("quantities", ())),
            (case.left_span,),
        )
        self.assertEqual(
            tuple(graph.nodes[f"PARTY:{case.right_id}"].attributes.get("quantities", ())),
            (case.right_span,),
        )

