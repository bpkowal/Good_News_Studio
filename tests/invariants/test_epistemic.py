"""CERTAIN-quarantine over Hypothesis-generated labels.

The oracle is CertainReopenCase.should_quarantine (CERTAIN row plus a
declared REOPEN claim). Production is only asked whether the candidate
lost its unique ranking.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.epistemic_ledger import (
    CERTAIN_CONTRADICTION_NOTE,
    HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
    apply_side_premise_audit,
    attach_candidate_dependencies,
    resolve_proposition,
)
from global_workspace.models import CandidateChunk
from global_workspace.specialist_authority import apply_specialist_authority
from strategies.specialist_claims import (
    CertainReopenCase,
    FrameworkDerivedCase,
    PropositionIdentityCase,
    UnsettledModalityCase,
    certain_reopen_cases,
    framework_derived_cases,
    proposition_identity_cases,
    unsettled_modality_cases,
)


def _chunk(case: CertainReopenCase) -> CandidateChunk:
    return CandidateChunk(
        specialist=case.specialist,
        constraint="DUTY",
        action_scores={case.actions[0]: 0.72, case.actions[1]: 0.28},
        surprise=0.1,
        friction=0.44,
        confidence=0.86,
        recommended_action=case.actions[0],
        preference_strength=0.44,
        epistemic_confidence=0.86,
        schema_valid=True,
        decision_rule="prefer the first action on the stated ranking",
        supporting_proposition_ids=[case.world_id],
        material_empirical_claims=[{
            "claim": case.claim,
            "proposition_id": "HYPOTHESIS",
            "decision_critical": True,
        }],
    )


class EpistemicInvariantTests(unittest.TestCase):
    @given(certain_reopen_cases())
    @settings(max_examples=40, deadline=None)
    def test_certain_reopen_cannot_uniquely_rank(
        self, case: CertainReopenCase,
    ):
        chunk = _chunk(case)
        attach_candidate_dependencies(case.ledger, chunk)
        quarantined = chunk.adjudication_status == "CONTESTED_NO_LEANING"
        noted = any(
            CERTAIN_CONTRADICTION_NOTE in note
            for note in chunk.epistemic_binding_notes
        )
        self.assertEqual(quarantined, case.should_quarantine)
        self.assertEqual(noted, case.should_quarantine)
        if case.should_quarantine:
            self.assertTrue(chunk.schema_valid)
            self.assertEqual(chunk.recommended_action, "")
            self.assertAlmostEqual(chunk.action_scores[case.actions[0]], 0.5)
            self.assertAlmostEqual(chunk.action_scores[case.actions[1]], 0.5)
            profile = apply_specialist_authority(chunk)
            self.assertEqual(profile.adjudication_status, "CONTESTED_NO_LEANING")
            self.assertEqual(profile.policy_weight_factor, 0.0)
        else:
            self.assertEqual(chunk.recommended_action, case.actions[0])
            self.assertNotAlmostEqual(chunk.action_scores[case.actions[0]], 0.5)

    @given(unsettled_modality_cases())
    @settings(max_examples=40, deadline=None)
    def test_unsettled_row_is_not_rebound_as_settled(
        self, case: UnsettledModalityCase,
    ):
        bound = resolve_proposition(case.ledger, case.claim)
        if case.should_bind:
            self.assertEqual(bound, case.world_id)
        else:
            self.assertEqual(bound, "")

    def _framework_chunk(
        self, case: FrameworkDerivedCase, *, basis: str,
    ) -> CandidateChunk:
        return CandidateChunk(
            specialist=case.specialist,
            constraint="DUTY",
            action_scores={case.actions[0]: 0.72, case.actions[1]: 0.28},
            surprise=0.1,
            friction=0.44,
            confidence=0.86,
            recommended_action=case.actions[0],
            preference_strength=0.44,
            epistemic_confidence=0.86,
            schema_valid=True,
            decision_rule="prefer the first action on the stated ranking",
            supporting_proposition_ids=[case.world_id],
            material_empirical_claims=[{
                "claim": case.claim,
                "proposition_id": basis,
                "decision_critical": True,
            }],
        )

    @given(framework_derived_cases())
    @settings(max_examples=40, deadline=None)
    def test_framework_derived_is_not_a_descriptive_hypothesis(
        self, case: FrameworkDerivedCase,
    ):
        basis = "FRAMEWORK_DERIVED" if case.is_framework_derived else "HYPOTHESIS"
        chunk = self._framework_chunk(case, basis=basis)
        attach_candidate_dependencies(case.ledger, chunk)
        bound = chunk.material_empirical_claims[0]["proposition_id"]
        record = case.ledger[bound]
        if case.is_framework_derived:
            self.assertEqual(record.epistemic_type, "FRAMEWORK_DERIVED")
            self.assertEqual(record.proposition_type, "NORMATIVE")
            self.assertNotEqual(record.epistemic_status, "HYPOTHETICAL")
            self.assertGreater(
                chunk.epistemic_confidence, HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
            )
        else:
            self.assertEqual(record.epistemic_type, "HYPOTHESIS")
            self.assertLessEqual(
                chunk.epistemic_confidence, HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
            )
        self.assertEqual(chunk.recommended_action, case.actions[0])
        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")

    @given(framework_derived_cases())
    @settings(max_examples=40, deadline=None)
    def test_audit_reclassifies_normative_new_hypothesis(
        self, case: FrameworkDerivedCase,
    ):
        chunk = CandidateChunk(
            specialist=case.specialist,
            constraint="DUTY",
            action_scores={case.actions[0]: 0.72, case.actions[1]: 0.28},
            surprise=0.1,
            friction=0.44,
            confidence=0.86,
            recommended_action=case.actions[0],
            preference_strength=0.44,
            epistemic_confidence=0.86,
            schema_valid=True,
            decision_rule="prefer the first action on the stated ranking",
            supporting_proposition_ids=[case.world_id],
        )
        apply_side_premise_audit(case.ledger, [chunk], {
            "status": "FINDINGS",
            "findings": [{
                "specialist": case.specialist,
                "claim": case.claim,
                "binding": "NEW_HYPOTHESIS",
                "derived_from": [case.world_id],
                "decision_critical": True,
                "source_field": "rationale",
                "reason": "audit finding",
            }],
        })
        bound = chunk.side_premise_audit_findings[0]["proposition_id"]
        record = case.ledger[bound]
        if case.is_framework_derived:
            self.assertEqual(record.epistemic_type, "FRAMEWORK_DERIVED")
            self.assertGreater(
                chunk.epistemic_confidence, HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
            )
        else:
            self.assertEqual(record.epistemic_type, "HYPOTHESIS")
            self.assertLessEqual(
                chunk.epistemic_confidence, HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
            )
        self.assertNotEqual(chunk.adjudication_status, "CONTESTED_NO_LEANING")

    def _identity_chunk(self, case: PropositionIdentityCase) -> CandidateChunk:
        return CandidateChunk(
            specialist=case.specialist,
            constraint="DUTY",
            action_scores={case.actions[0]: 0.72, case.actions[1]: 0.28},
            surprise=0.1,
            friction=0.44,
            confidence=0.86,
            recommended_action=case.actions[0],
            preference_strength=0.44,
            epistemic_confidence=0.86,
            schema_valid=True,
            decision_rule="prefer the first action on the stated ranking",
            material_empirical_claims=[{
                "claim": case.claim,
                "proposition_id": "HYPOTHESIS",
                "decision_critical": True,
            }],
        )

    @given(proposition_identity_cases())
    @settings(max_examples=40, deadline=None)
    def test_restatement_rebinds_without_raising_status(
        self, case: PropositionIdentityCase,
    ):
        world = case.ledger[case.world_id]
        status_before = world.epistemic_status
        mentions_before = world.mention_count
        chunk = self._identity_chunk(case)
        attach_candidate_dependencies(case.ledger, chunk)
        bound = chunk.material_empirical_claims[0]["proposition_id"]
        if case.should_bind:
            self.assertEqual(bound, case.world_id)
            self.assertEqual(world.epistemic_status, status_before)
            self.assertGreater(world.mention_count, mentions_before)
            second = self._identity_chunk(case)
            attach_candidate_dependencies(case.ledger, second)
            self.assertEqual(
                second.material_empirical_claims[0]["proposition_id"],
                case.world_id,
            )
            self.assertEqual(world.epistemic_status, "ESTABLISHED")
            self.assertEqual(world.epistemic_type, "WORLD_ESTABLISHED")
        else:
            self.assertNotEqual(bound, case.world_id)
            self.assertEqual(world.epistemic_status, "ESTABLISHED")
            minted = case.ledger[bound]
            self.assertEqual(minted.epistemic_status, "HYPOTHETICAL")
            self.assertEqual(minted.epistemic_type, "HYPOTHESIS")
