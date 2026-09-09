"""Exact semantic comparison for sequential and concurrent workspace runs.

Execution metadata and elapsed time are intentionally excluded. Everything
that can change deliberation, admission, governance, or the public judgment is
compared without fuzzy matching or prose normalization.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

from .models import CandidateChunk, WorkspaceResult
from .presentation import render_public_judgment


_LEDGER_FIELDS = (
    "proposition_ledger",
    "utilitarian_consequence_ledger",
    "deontological_duty_ledger",
    "virtue_character_ledger",
    "care_relationship_ledger",
    "rawlsian_position_ledger",
)


def _candidate_payload(candidate: CandidateChunk | None) -> dict[str, Any] | None:
    return asdict(candidate) if candidate is not None else None


@dataclass(frozen=True, slots=True)
class SemanticRunSnapshot:
    """The decision-relevant surfaces that concurrency must not alter."""

    candidate_payloads: tuple[tuple[dict[str, Any], ...], ...]
    transactions_and_ledgers: dict[str, Any]
    vote_admission_decisions: tuple[tuple[dict[str, Any], ...], ...]
    policy: dict[str, Any]
    challenge_agenda: tuple[dict[str, Any], ...]
    governing_authority: dict[str, Any]
    final_report: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticEquivalenceComparison:
    equivalent: bool
    mismatched_sections: tuple[str, ...]
    sequential: SemanticRunSnapshot
    concurrent: SemanticRunSnapshot


def semantic_run_snapshot(result: WorkspaceResult) -> SemanticRunSnapshot:
    """Project one run without timing, telemetry, or completion order."""
    candidates = tuple(
        tuple(asdict(candidate) for candidate in cycle.candidates)
        for cycle in result.cycles
    )
    transactions_and_ledgers = {
        "graph_transactions": copy.deepcopy(result.graph_transactions),
        "semantic_graphs": copy.deepcopy(result.semantic_graphs),
        **{
            field: copy.deepcopy(getattr(result, field))
            for field in _LEDGER_FIELDS
        },
    }
    votes = tuple(
        tuple({
            "specialist": candidate.specialist,
            "framework_vote_integrity_required": (
                candidate.framework_vote_integrity_required
            ),
            "framework_vote_status": candidate.framework_vote_status,
            "framework_vote_reason": candidate.framework_vote_reason,
            "framework_ledger_kind": candidate.framework_ledger_kind,
            "framework_ledger_status": candidate.framework_ledger_status,
            "derived_claim_validation_status": (
                candidate.derived_claim_validation_status
            ),
            "derived_claim_validation_errors": list(
                candidate.derived_claim_validation_errors
            ),
            "policy_weight_factor": candidate.policy_weight_factor,
            "governing_eligible": candidate.governing_eligible,
        } for candidate in cycle.candidates)
        for cycle in result.cycles
    )
    policy = {
        "cycles": [copy.deepcopy(cycle.policy) for cycle in result.cycles],
        "policy_leaders": [cycle.policy_leader for cycle in result.cycles],
        "selected_action": result.selected_action,
        "current_plurality": result.current_plurality,
        "confidence": result.confidence,
        "epistemic_confidence": result.epistemic_confidence,
        "judgment_status": result.judgment_status,
    }
    agenda = tuple({
        "cycle": cycle.cycle,
        "received": copy.deepcopy(
            list(cycle.received_broadcast.challenge_agenda)
            if cycle.received_broadcast is not None else []
        ),
        "next": copy.deepcopy(list(cycle.broadcast.challenge_agenda)),
    } for cycle in result.cycles)
    governing = {
        "cycles": [{
            "cycle": cycle.cycle,
            "governing_claim": _candidate_payload(cycle.governing_claim),
            "broadcast_focus": _candidate_payload(cycle.broadcast_focus),
            "broadcast_authority": cycle.broadcast.broadcast_authority,
            "salient_specialist": cycle.broadcast.salient_specialist,
        } for cycle in result.cycles],
        "transitions": copy.deepcopy(result.governing_authority_transitions),
        "governing_justification_status": result.governing_justification_status,
        "governing_attack_reason": result.governing_attack_reason,
    }
    return SemanticRunSnapshot(
        candidate_payloads=candidates,
        transactions_and_ledgers=transactions_and_ledgers,
        vote_admission_decisions=votes,
        policy=policy,
        challenge_agenda=agenda,
        governing_authority=governing,
        final_report=render_public_judgment(result),
    )


def compare_semantic_results(
    sequential: WorkspaceResult,
    concurrent: WorkspaceResult,
) -> SemanticEquivalenceComparison:
    """Require exact equality section-by-section; no similarity tolerance."""
    left = semantic_run_snapshot(sequential)
    right = semantic_run_snapshot(concurrent)
    left_dict = left.to_dict()
    right_dict = right.to_dict()
    mismatches = tuple(
        section for section in left_dict
        if left_dict[section] != right_dict[section]
    )
    return SemanticEquivalenceComparison(
        equivalent=not mismatches,
        mismatched_sections=mismatches,
        sequential=left,
        concurrent=right,
    )
