"""Compositional invariants: hypothetical necessity and counterfactual-plus-quantity.

Single-invariant wires already exist. These cases compose them. Oracles are
declared: a decision-critical HYPOTHETICAL necessary for a unique ranking
must not govern, regardless of mention count; a minted threshold over a
FOREGONE dual must not uniquely rank.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.epistemic_ledger import (
    UNADMITTED_MAGNITUDE_NOTE,
    _apply_candidate_authority_cap,
    attach_candidate_dependencies,
    register_hypothesis,
)
from global_workspace.models import CandidateChunk
from global_workspace.specialist_authority import apply_specialist_authority
from strategies.consumption import (
    CompositionalHypothesisCase,
    CompositionalThresholdCase,
    compositional_hypothesis_cases,
    compositional_threshold_cases,
)


class CompositionalHypothesisTests(unittest.TestCase):
    @given(compositional_hypothesis_cases())
    @settings(max_examples=40, deadline=None)
    def test_hypothetical_necessary_for_unique_ranking_cannot_govern(
        self, case: CompositionalHypothesisCase,
    ):
        ledger = {
            proposition_id: type(record)(**record.to_dict())
            for proposition_id, record in case.ledger.items()
        }
        world_id = "PROP:WORLD:E1"
        hyp_id = register_hypothesis(
            ledger, case.claim,
            specialist="utilitarian",
            derived_from=[world_id],
            decision_critical=True,
        )
        if case.recurrences == 0:
            ledger[hyp_id].mention_count = 0
        chunk = CandidateChunk(
            specialist="utilitarian",
            constraint="IMMINENT_HARM",
            action_scores={case.actions[0]: 0.28, case.actions[1]: 0.72},
            surprise=0.6,
            friction=0.44,
            confidence=0.86,
            recommended_action=case.actions[1],
            preference_strength=0.44,
            epistemic_confidence=0.86,
            schema_valid=True,
            salience=0.9,
            decision_rule="prefer the second action on the unverified premise",
            adjudication_status="SUPPORTS",
            governing_eligible=True,
            supporting_proposition_ids=[world_id, hyp_id],
            decision_critical_proposition_ids=[hyp_id],
            utilitarian_consequence_table={
                case.actions[0]: [{
                    "outcome": case.ledger[world_id].outcome,
                    "direction": "HARM",
                    "probability": "UNKNOWN",
                    "magnitude": "UNKNOWN",
                    "support": "STATED",
                }],
                case.actions[1]: [{
                    "outcome": case.claim,
                    "direction": "BENEFIT",
                    "probability": "UNKNOWN",
                    "magnitude": "UNKNOWN",
                    "support": "STATED",
                }],
            },
            material_empirical_claims=(
                []
                if case.recurrences == 0
                else [{
                    "claim": case.claim,
                    "proposition_id": hyp_id,
                    "decision_critical": True,
                }]
            ),
        )
        if case.recurrences == 0:
            _apply_candidate_authority_cap(ledger, chunk)
        else:
            for _ in range(case.recurrences):
                attach_candidate_dependencies(ledger, chunk)
        profile = apply_specialist_authority(chunk)
        governs = (
            profile.governing_eligible
            and profile.adjudication_status == "SUPPORTS"
            and bool(chunk.recommended_action)
            and chunk.adjudication_status != "CONTESTED_NO_LEANING"
        )
        self.assertEqual(governs, case.should_be_governing_eligible)


class CompositionalThresholdTests(unittest.TestCase):
    @given(compositional_threshold_cases())
    @settings(max_examples=40, deadline=None)
    def test_foregone_plus_minted_threshold_cannot_uniquely_rank(
        self, case: CompositionalThresholdCase,
    ):
        chunk = CandidateChunk(
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
            material_empirical_claims=[{
                "claim": case.claim,
                "proposition_id": "HYPOTHESIS",
                "decision_critical": True,
            }],
        )
        attach_candidate_dependencies(case.ledger, chunk)
        self.assertEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertTrue(any(
            UNADMITTED_MAGNITUDE_NOTE in note
            for note in chunk.epistemic_binding_notes
        ))
        profile = apply_specialist_authority(chunk)
        self.assertEqual(profile.adjudication_status, "CONTESTED_NO_LEANING")
        self.assertEqual(profile.policy_weight_factor, 0.0)
