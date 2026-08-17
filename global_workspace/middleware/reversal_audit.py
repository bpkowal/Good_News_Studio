from __future__ import annotations

from dataclasses import dataclass
import re

from ..models import CandidateChunk
from ..semantic_state import CommittedReversalBoundary


@dataclass(frozen=True, slots=True)
class ReversalAuditRequest:
    critic: str
    leading_action: str
    competing_action: str
    decision_rule: str
    proposed_condition: str
    challenge: str
    boundary_sources: tuple[str, ...] = ()


def build_reversal_audit_request(
    dissent: CandidateChunk | None,
    selected_action: str,
    committed_boundary: CommittedReversalBoundary | None,
    *,
    min_preference_strength: float = 0.50,
    min_epistemic_confidence: float = 0.65,
) -> ReversalAuditRequest | None:
    """Construct an audit only from a committed graph transition.

    Dissent determines whether an alternative deserves workspace attention. The
    graph exclusively determines the action direction and factual condition.
    """
    if dissent is None or not dissent.schema_valid or committed_boundary is None:
        return None
    if (
        dissent.preference_strength < min_preference_strength
        or dissent.epistemic_confidence < min_epistemic_confidence
    ):
        return None
    alternative = dissent.recommended_action or (
        max(dissent.action_scores, key=dissent.action_scores.get)
        if dissent.action_scores else ""
    )
    if not alternative or alternative == selected_action:
        return None
    if (
        committed_boundary.source_action != selected_action
        or committed_boundary.target_action != alternative
        or not committed_boundary.predicate.strip()
    ):
        return None
    proposed = committed_boundary.predicate
    typed_rule = (
        f"switch from {selected_action} to {alternative} if {proposed}"
    )
    explanation = " ".join(
        (dissent.decision_rule or dissent.rationale).split()
    ).strip()
    challenge = (
        f"Committed boundary P tests '{selected_action}' against '{alternative}': "
        f"switch if {proposed}. Dissent source: {dissent.specialist}."
    )
    if explanation:
        challenge += f" Explanatory framework claim: {explanation}."
    if len(challenge.split()) < 8:
        return None
    return ReversalAuditRequest(
        dissent.specialist,
        selected_action,
        alternative,
        typed_rule,
        proposed,
        challenge,
        committed_boundary.source_specialists,
    )


def dissent_reversal_condition(
    dissent: CandidateChunk | None,
    selected_action: str,
) -> str:
    """Turn preserved substantive dissent into an operational reopen rule."""
    if dissent is None or not dissent.schema_valid or not dissent.rationale:
        return ""
    alternative = dissent.recommended_action or (
        max(dissent.action_scores, key=dissent.action_scores.get)
        if dissent.action_scores else ""
    )
    if not alternative or alternative == selected_action:
        return ""
    if re.search(r"\b(?:random|coin|lottery)\b", dissent.landscape_tiebreaker, re.I):
        return ""
    typed_factual = (
        dissent.factual_reversal_threshold
        if dissent.factual_reversal_threshold.casefold() != "none"
        else ""
    )
    # A typed threshold describes when the dissenter abandons its own action. It
    # supports the current leader and therefore is not a condition for reopening
    # the collective judgment toward the dissenter. Only legacy challenge text
    # (or the dissent's normative axis below) can supply that direction here.
    legacy_challenge = "" if (dissent.revised_reversal_condition or typed_factual) else dissent.reversal_condition
    factual = " ".join(legacy_challenge.split()).strip(" .")
    if factual:
        return f"If {factual[0].lower() + factual[1:]}, prefer {alternative}"
    normative = " ".join(dissent.normative_reversal_threshold.split()).strip(" .")
    if normative and normative.casefold() != "none" and not typed_factual:
        return f"If {normative[0].lower() + normative[1:]}, prefer {alternative}"
    # An axis or rationale identifies disagreement but not a switch boundary.
    # Preserve it as moral residue instead of fabricating a tautological rule.
    return ""
