"""Read-only projection of the live configuration of deliberation.

This state never commits facts or framework conclusions. It attributes every
preference, constraint, uncertainty, and justification to the specialist that
supplied it, leaving the semantic graph as the sole factual authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import re
from typing import Any, Sequence

from .models import CandidateChunk, SynthesisProposal
from .framework_native_projection import committed_native_reasoning
from .resolved_questions import question_resolution_index
from .scenario_semantics import segment_scenario_clauses


@dataclass(frozen=True, slots=True)
class AgentDeliberativePosition:
    specialist: str
    preferred_action: str
    action_scores: dict[str, float]
    preference_strength: float
    epistemic_confidence: float
    active_constraint: str
    unresolved: str
    assumption_status: str
    choice_status: str
    choice_condition: str
    preferred_extension: str
    supporting_proposition_ids: tuple[str, ...]
    decision_critical_proposition_ids: tuple[str, ...]
    weakest_decision_critical_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorkspaceContribution:
    """Loss-minimizing contract between one specialist and the workspace.

    This is attributed deliberative state, never shared world fact. It carries
    enough structure for recurrence without granting any field an automatic
    effect on scores, salience, or another framework's priority ordering.
    """

    agent: str
    tendency: str
    core_ground: tuple[str, ...]
    unresolved: tuple[str, ...]
    defeat_conditions: tuple[str, ...]
    new_considerations: tuple[str, ...]
    update_type: str
    constraint: str
    choice_status: str
    # Framework-attributed cases for every action. These are deliberative
    # counterpositions, never facts or cross-framework priority rules.
    action_cases: dict[str, str] = field(default_factory=dict)
    native_reasoning: dict[str, Any] = field(default_factory=dict)
    preservation_transitions: tuple[dict[str, str], ...] = ()
    retained_issue_visibility: str = "NOT_APPLICABLE"
    visible_retained_issue: str = ""
    voting_effect: str = "NONE"
    source_type: str = "FRAMEWORK_ATTRIBUTED_CONTRIBUTION"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProblemDelta:
    from_cycle: int
    to_cycle: int
    preference_changes: tuple[dict[str, Any], ...] = ()
    confidence_changes: tuple[dict[str, Any], ...] = ()
    constraint_changes: tuple[dict[str, Any], ...] = ()
    new_constraints: tuple[dict[str, Any], ...] = ()
    removed_constraints: tuple[dict[str, Any], ...] = ()
    new_conflicts: tuple[dict[str, Any], ...] = ()
    resolved_conflicts: tuple[dict[str, Any], ...] = ()
    new_questions: tuple[dict[str, Any], ...] = ()
    resolved_questions: tuple[dict[str, Any], ...] = ()
    new_internal_conflicts: tuple[dict[str, Any], ...] = ()
    resolved_internal_conflicts: tuple[dict[str, Any], ...] = ()
    reframed_internal_conflicts: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(frozen=True, slots=True)
class BroadcastInfluenceRecord:
    """Observed, non-causal uptake of one broadcast consideration.

    These records are deliberately diagnostic.  They do not change salience:
    exposure to a broadcast is not yet evidence that it caused the subsequent
    update, and framework-safe influence must be learned before it is rewarded.
    """

    broadcast_id: str
    consideration_id: str
    source_agent: str
    receiving_agent: str
    source_cycle: int
    observed_cycle: int
    observation_lag: int
    consideration: dict[str, Any]
    preference_changes: tuple[dict[str, Any], ...] = ()
    confidence_changes: tuple[dict[str, Any], ...] = ()
    constraint_changes: tuple[dict[str, Any], ...] = ()
    conflict_changes: tuple[dict[str, Any], ...] = ()
    question_changes: tuple[dict[str, Any], ...] = ()
    uncertainty_changes: tuple[dict[str, Any], ...] = ()
    framework_effect: str = "RETAINED"
    framework_retained: bool = True
    grounding_retained: bool = True
    persistence: str = "NOT_YET_OBSERVED"
    influence_class: str = "NONE"
    attribution_status: str = "OBSERVED_AFTER_EXPOSURE_NOT_CAUSALLY_ESTABLISHED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeliberativeProblemState:
    cycle: int
    live_actions: tuple[dict[str, str], ...]
    agent_positions: tuple[AgentDeliberativePosition, ...]
    active_constraints: tuple[dict[str, Any], ...]
    active_conflicts: tuple[dict[str, Any], ...]
    unresolved_questions: tuple[dict[str, Any], ...]
    framework_internal_conflicts: tuple[dict[str, Any], ...]
    framework_specific_open_questions: tuple[dict[str, Any], ...]
    framework_insights: tuple[dict[str, Any], ...]
    workspace_contributions: tuple[WorkspaceContribution, ...]
    dissenting_positions: tuple[AgentDeliberativePosition, ...]
    current_plurality: str
    support_composition: dict[str, Any] = field(default_factory=dict)
    surface_consensus: str = "INSUFFICIENT"
    deliberative_consensus: str = "INSUFFICIENT"
    framework_warnings: tuple[dict[str, Any], ...] = ()
    audit_candidates: tuple[dict[str, Any], ...] = ()
    resolved_questions: tuple[dict[str, Any], ...] = ()
    unresolved_categories: tuple[str, ...] = ()
    primary_unresolved: str = "NONE"
    proposals: tuple[dict[str, Any], ...] = ()
    salient_position: dict[str, Any] = field(default_factory=dict)
    problem_delta: ProblemDelta | None = None
    state_role: str = "DESCRIPTIVE_DELIBERATIVE_PROJECTION"
    committed_world: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "state_role": self.state_role,
            "live_actions": [dict(item) for item in self.live_actions],
            "agent_positions": [item.to_dict() for item in self.agent_positions],
            "active_constraints": [dict(item) for item in self.active_constraints],
            "active_conflicts": [dict(item) for item in self.active_conflicts],
            "unresolved_questions": [dict(item) for item in self.unresolved_questions],
            "framework_internal_conflicts": [
                dict(item) for item in self.framework_internal_conflicts
            ],
            "framework_specific_open_questions": [
                dict(item) for item in self.framework_specific_open_questions
            ],
            "framework_insights": [dict(item) for item in self.framework_insights],
            "workspace_contributions": [
                item.to_dict() for item in self.workspace_contributions
            ],
            "dissenting_positions": [item.to_dict() for item in self.dissenting_positions],
            "current_plurality": self.current_plurality,
            "support_composition": dict(self.support_composition),
            "surface_consensus": self.surface_consensus,
            "deliberative_consensus": self.deliberative_consensus,
            "framework_warnings": [dict(item) for item in self.framework_warnings],
            "audit_candidates": [dict(item) for item in self.audit_candidates],
            "resolved_questions": [dict(item) for item in self.resolved_questions],
            "unresolved_categories": list(self.unresolved_categories),
            "primary_unresolved": self.primary_unresolved,
            "proposals": [dict(item) for item in self.proposals],
            "salient_position": dict(self.salient_position),
            "problem_delta": (
                self.problem_delta.to_dict() if self.problem_delta is not None else {}
            ),
            "committed_world": dict(self.committed_world),
        }


def _stable_key(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _workspace_contribution(candidate: CandidateChunk) -> WorkspaceContribution:
    action_ground = str(
        candidate.framework_action_map.get(candidate.recommended_action, "")
    ).partition(":")[2].strip()
    grounds = list(dict.fromkeys(
        " ".join(str(value).split())[:240]
        for value in (action_ground, candidate.decision_rule, candidate.rationale)
        if " ".join(str(value).split())
    ))[:3]
    unresolved = list(dict.fromkeys(
        " ".join(str(value).split())[:240]
        for value in (
            *candidate.framework_internal_conflicts,
            *candidate.framework_specific_open_questions,
            candidate.utilitarian_missing_comparison,
            candidate.unsupported_assumption,
        )
        if " ".join(str(value).split())
        and str(value).strip().upper() != "NONE"
    ))[:6]
    defeat_conditions = list(dict.fromkeys(
        " ".join(str(value).split())[:240]
        for value in (
            candidate.reversal_condition,
            candidate.factual_reversal_threshold,
            candidate.normative_reversal_threshold,
        )
        if " ".join(str(value).split())
        and str(value).strip().upper() != "NONE"
    ))[:4]
    considerations = list(dict.fromkeys(
        " ".join(str(item.get("proposition", "")).split())[:240]
        for item in candidate.framework_insights
        if " ".join(str(item.get("proposition", "")).split())
    ))[:4]
    update_type = (
        candidate.assumption_status
        if candidate.assumption_status not in {"", "NOT_AUDITED", "SUPPORTED"}
        else candidate.framework_retention_status
        if candidate.framework_retention_status not in {"", "NOT_MEASURED"}
        else candidate.selection_status
    )
    return WorkspaceContribution(
        agent=candidate.specialist,
        tendency=candidate.recommended_action,
        core_ground=tuple(grounds),
        unresolved=tuple(unresolved),
        defeat_conditions=tuple(defeat_conditions),
        new_considerations=tuple(considerations),
        update_type=update_type,
        constraint=candidate.constraint,
        choice_status=_position(candidate).choice_status,
        action_cases={
            str(action): " ".join(str(case).split())[:240]
            for action, case in candidate.framework_action_map.items()
            if " ".join(str(case).split())
        },
        native_reasoning=committed_native_reasoning(candidate),
    )


def _normalized_structure(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _structure_overlap(left: str, right: str) -> float:
    left_tokens = set(_normalized_structure(left).split())
    right_tokens = set(_normalized_structure(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


_CHALLENGE_SUPERSESSION_MARKERS = {
    "DOING_ALLOWING_CLASSIFICATION": frozenset({"omission", "right", "correlative"}),
    "DUTY_PERFECTION_BASIS": frozenset({"perfect", "imperfect"}),
    "MEANS_CAUSAL_PATH": frozenset({"causal", "path"}),
}
_ACTIVE_FRAMEWORK_ISSUE_STATUSES = {"UNRESOLVED", "OPEN", "SUSPENDED"}


def _verified_challenge_targets(challenge: dict[str, Any]) -> set[str]:
    targets = {
        str(challenge.get("about_specialist", "")).casefold(),
        *(
            str(value).casefold()
            for value in challenge.get("target_specialists", []) or []
        ),
    }
    targets.discard("")
    return targets


def _challenge_supersedes_issue(challenge: dict[str, Any], specialist: str, text: str) -> bool:
    if specialist.casefold() not in _verified_challenge_targets(challenge):
        return False
    structure = set(_normalized_structure(text).split())
    if not structure:
        return False
    kind = str(challenge.get("challenge_kind", "")).upper()
    markers = _CHALLENGE_SUPERSESSION_MARKERS.get(kind)
    if markers and markers <= structure:
        return True
    question = " ".join(str(challenge.get("question", challenge.get("proposition", ""))).split())
    return bool(question) and _structure_overlap(text, question) >= 0.35


def apply_verified_challenge_supersession(
    problem_state: dict[str, Any],
    challenges: Sequence[dict[str, Any]],
) -> None:
    """Retire live framework issues whose matching challenge is VERIFIED_RESOLVED.

    A verified repair must mutate the operative issue, not sit beside it:
    old_issue.status becomes SUPERSEDED rather than remaining UNRESOLVED or
    SUSPENDED while the repair is also recorded.
    """
    verified = [
        dict(item)
        for item in challenges
        if isinstance(item, dict)
        and str(
            (item.get("last_response") or {}).get("verification_status", "")
        ).upper() == "VERIFIED_RESOLVED"
    ]
    if not verified:
        return

    for collection, text_field in (
        ("framework_internal_conflicts", "conflict"),
        ("framework_specific_open_questions", "question"),
    ):
        for issue in problem_state.get(collection, []) or []:
            if not isinstance(issue, dict):
                continue
            if str(issue.get("status", "")).upper() not in _ACTIVE_FRAMEWORK_ISSUE_STATUSES:
                continue
            text = " ".join(str(issue.get(text_field, "")).split())
            specialist = str(issue.get("source_specialist", ""))
            if any(
                _challenge_supersedes_issue(challenge, specialist, text)
                for challenge in verified
            ):
                issue["status"] = "SUPERSEDED"

    for contribution in problem_state.get("workspace_contributions", []) or []:
        if not isinstance(contribution, dict):
            continue
        agent = str(contribution.get("agent", ""))
        unresolved = [
            str(item) for item in contribution.get("unresolved", []) or []
            if str(item).strip()
        ]
        transitions = [
            dict(item)
            for item in contribution.get("preservation_transitions", []) or []
            if isinstance(item, dict)
        ]
        kept: list[str] = []
        superseded_visible = False
        visible = " ".join(str(contribution.get("visible_retained_issue", "")).split())
        for item in unresolved:
            if any(
                _challenge_supersedes_issue(challenge, agent, item)
                for challenge in verified
            ):
                transitions.append({
                    "category": "unresolved",
                    "prior_item": item,
                    "status": "SUPERSEDED",
                    "replacement": "",
                })
                if visible and _normalized_structure(item) == _normalized_structure(visible):
                    superseded_visible = True
            else:
                kept.append(item)
        contribution["unresolved"] = kept
        contribution["preservation_transitions"] = transitions
        if superseded_visible or (
            visible
            and any(
                _challenge_supersedes_issue(challenge, agent, visible)
                for challenge in verified
            )
        ):
            contribution["visible_retained_issue"] = ""
            contribution["retained_issue_visibility"] = "NOT_APPLICABLE"


def _reconcile_workspace_contribution(
    current: WorkspaceContribution,
    previous: dict[str, Any] | None,
    candidate: CandidateChunk,
) -> WorkspaceContribution:
    """Preserve framework-local structure unless its disposition is explicit."""
    if not previous or previous.get("agent") != current.agent:
        return current

    transitions: list[dict[str, str]] = []
    reconciled: dict[str, tuple[str, ...]] = {}
    update_type = current.update_type.upper()
    explicitly_resolved = update_type == "RESOLUTION"
    explicitly_reversed = update_type == "REVERSAL"

    for field_name in (
        "core_ground", "unresolved", "defeat_conditions", "new_considerations",
    ):
        prior_items = [
            str(item) for item in previous.get(field_name, []) if str(item).strip()
        ]
        current_items = list(getattr(current, field_name))
        matched_current: set[int] = set()
        retained_prior: list[str] = []
        for prior_item in prior_items:
            exact_index = next((
                index for index, item in enumerate(current_items)
                if _normalized_structure(item) == _normalized_structure(prior_item)
            ), None)
            if exact_index is not None:
                matched_current.add(exact_index)
                status = "RETAINED"
                replacement = current_items[exact_index]
            else:
                best = max(
                    ((_structure_overlap(prior_item, item), index, item)
                     for index, item in enumerate(current_items)),
                    default=(0.0, -1, ""),
                )
                if best[0] >= 0.55:
                    matched_current.add(best[1])
                    status = "REFINED"
                    replacement = best[2]
                elif explicitly_resolved and field_name == "unresolved":
                    status = "RESOLVED"
                    replacement = ""
                elif explicitly_reversed and field_name in {
                    "core_ground", "defeat_conditions",
                }:
                    status = "DEFEATED"
                    replacement = ""
                else:
                    status = "SUSPENDED"
                    replacement = prior_item
                    retained_prior.append(prior_item)
            transitions.append({
                "category": field_name,
                "prior_item": prior_item,
                "status": status,
                "replacement": replacement,
            })
        for index, item in enumerate(current_items):
            if index not in matched_current:
                transitions.append({
                    "category": field_name,
                    "prior_item": "",
                    "status": "NEW",
                    "replacement": item,
                })
        reconciled[field_name] = tuple(dict.fromkeys([
            *current_items, *retained_prior,
        ]))

    prior_unresolved = [
        str(item) for item in previous.get("unresolved", []) if str(item).strip()
    ]
    visible_prose = " ".join((
        candidate.rationale,
        candidate.decision_rule,
        candidate.framework_application,
    ))
    structured_text = " ".join((
        *candidate.framework_internal_conflicts,
        *candidate.framework_specific_open_questions,
        json.dumps(candidate.framework_action_map, sort_keys=True),
        json.dumps(candidate.framework_insights, sort_keys=True),
    ))
    visible_issue = ""
    visibility = "NOT_APPLICABLE"
    if prior_unresolved:
        for issue in prior_unresolved:
            if _normalized_structure(issue) in _normalized_structure(visible_prose):
                visibility, visible_issue = "EXPLICIT", issue
                break
        if not visible_issue:
            best_prose = max(
                ((_structure_overlap(issue, visible_prose), issue)
                 for issue in prior_unresolved),
                default=(0.0, ""),
            )
            if best_prose[0] >= 0.35:
                visibility, visible_issue = "PARAPHRASED", best_prose[1]
        if not visible_issue:
            structured_issue = next((
                issue for issue in prior_unresolved
                if _structure_overlap(issue, structured_text) >= 0.35
            ), "")
            if structured_issue:
                visibility, visible_issue = "STRUCTURED_ONLY", structured_issue
            else:
                visibility, visible_issue = "OMITTED", prior_unresolved[0]

    return replace(
        current,
        core_ground=reconciled["core_ground"],
        unresolved=reconciled["unresolved"],
        defeat_conditions=reconciled["defeat_conditions"],
        new_considerations=reconciled["new_considerations"],
        action_cases=(
            dict(current.action_cases)
            if current.action_cases
            else dict(previous.get("action_cases", {}) or {})
        ),
        native_reasoning=(
            dict(current.native_reasoning)
            if current.native_reasoning
            else dict(previous.get("native_reasoning", {}) or {})
        ),
        preservation_transitions=tuple(transitions),
        retained_issue_visibility=visibility,
        visible_retained_issue=visible_issue,
    )


def consideration_key(salient_position: dict[str, Any]) -> str:
    """Identify the consideration, not the agent who happened to voice it."""
    return _stable_key("CONSIDERATION", {
        "constraint": str(salient_position.get("constraint", "")).upper(),
        "action": str(salient_position.get("preferred_action", "")),
        "justification": " ".join(
            str(salient_position.get("conditional_justification", "")).casefold().split()
        ),
        "unresolved": str(salient_position.get("unresolved", "")).upper(),
        "evidence_basis": str(salient_position.get("evidence_basis", "")).upper(),
    })


def observe_broadcast_influence(
    received_problem_state: dict[str, Any] | None,
    current_state: "DeliberativeProblemState",
    candidates: Sequence[CandidateChunk],
) -> tuple[BroadcastInfluenceRecord, ...]:
    """Describe next-cycle problem-state changes following a broadcast.

    The classification rewards neither convergence nor reversal.  It records
    conflict discovery/localization and question refinement as useful changes,
    while framework loss gates the result as capture or drift.
    """
    prior = dict(received_problem_state or {})
    salient = dict(prior.get("salient_position", {}) or {})
    delta = current_state.problem_delta
    if not salient or delta is None or not prior.get("cycle"):
        return ()
    source_cycle = int(prior.get("cycle", 0))
    observed_cycle = int(current_state.cycle)
    source_agent = str(salient.get("source_specialist", ""))
    key = consideration_key(salient)
    broadcast_id = _stable_key("BROADCAST", {
        "cycle": source_cycle, "consideration_id": key,
    })

    def for_agent(items: Sequence[dict[str, Any]], agent: str) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in items if item.get("specialist") == agent)

    records: list[BroadcastInfluenceRecord] = []
    for candidate in sorted(candidates, key=lambda item: item.specialist):
        agent = candidate.specialist
        preferences = for_agent(delta.preference_changes, agent)
        confidences = for_agent(delta.confidence_changes, agent)
        constraints = for_agent(delta.constraint_changes, agent)
        questions = tuple(
            dict(item) for item in (*delta.new_questions, *delta.resolved_questions)
            if item.get("source_specialist") == agent
        )
        conflicts = tuple(
            dict(item) for item in (*delta.new_conflicts, *delta.resolved_conflicts)
            if agent in item.get("specialists", [])
        )
        uncertainty = tuple(
            item for item in questions
            if item.get("category") or item.get("resolution_type")
        )
        meaningful = bool(
            preferences or confidences or constraints or questions or conflicts
        )
        grounding_retained = bool(
            str(salient.get("evidence_basis", "")).upper() != "UNSTATED_FACTS"
            and
            candidate.evidence_basis != "UNSTATED_FACTS"
            and candidate.landscape_semantic_valid
        )
        framework_retained = bool(candidate.framework_constraint_retained)
        retention_status = candidate.framework_retention_status.upper()
        uptake_signal = bool(
            candidate.workspace_reasoning_effect in {"FACTUAL", "NORMATIVE", "BOTH"}
            or candidate.self_reported_broadcast_dependence in {"LOW", "MEDIUM", "HIGH"}
        )
        if not framework_retained:
            framework_effect = (
                "CAPTURED" if candidate.self_reported_broadcast_dependence != "NONE"
                else "DRIFTED"
            )
            influence_class = "FRAMEWORK_CAPTURE" if framework_effect == "CAPTURED" else "DRIFT"
        elif retention_status in {"UNCLEAR", "COMMITTED_WITH_UNCERTAINTY", "UPDATE_REJECTED"}:
            framework_effect = "STRETCHED"
            influence_class = "NONE"
        elif meaningful and agent != source_agent and uptake_signal:
            framework_effect = "TRANSLATED"
            if questions:
                influence_class = "REFRAMING"
            elif any(item.get("conflict_type") for item in delta.new_conflicts):
                influence_class = "CONFLICT_DISCOVERY"
            elif any(item.get("resolution_type") for item in delta.resolved_conflicts):
                influence_class = "CONFLICT_RESOLUTION"
            elif preferences:
                influence_class = "REVERSAL" if any(
                    item.get("previous_action") != item.get("current_action")
                    for item in preferences
                ) else "REFINEMENT"
            else:
                influence_class = "REFINEMENT"
        elif meaningful and agent == source_agent:
            framework_effect = "RETAINED"
            influence_class = "REFINEMENT"
        else:
            framework_effect = "RETAINED"
            influence_class = "NONE"
        if not grounding_retained and influence_class not in {"FRAMEWORK_CAPTURE", "DRIFT"}:
            influence_class = "DRIFT"

        records.append(BroadcastInfluenceRecord(
            broadcast_id=broadcast_id,
            consideration_id=key,
            source_agent=source_agent,
            receiving_agent=agent,
            source_cycle=source_cycle,
            observed_cycle=observed_cycle,
            observation_lag=max(1, observed_cycle - source_cycle),
            consideration=salient,
            preference_changes=preferences,
            confidence_changes=confidences,
            constraint_changes=constraints,
            conflict_changes=conflicts,
            question_changes=questions,
            uncertainty_changes=uncertainty,
            framework_effect=framework_effect,
            framework_retained=framework_retained,
            grounding_retained=grounding_retained,
            influence_class=influence_class,
        ))
    return tuple(records)


def update_broadcast_influence_persistence(
    records: list[dict[str, Any]],
    current_state: "DeliberativeProblemState",
) -> None:
    """Check whether the prior cycle's observed update survives one more cycle.

    Persistence is structural: it asks whether the changed preference,
    constraint, question, or conflict remains in the problem state.  It does
    not infer that the original broadcast caused that persistence.
    """
    positions = {
        item.specialist: item.to_dict() for item in current_state.agent_positions
    }
    current_questions = {
        item.get("question_key") for item in current_state.unresolved_questions
    }
    current_conflicts = {
        item.get("conflict_key") for item in current_state.active_conflicts
    }
    for record in records:
        if (
            record.get("persistence") != "NOT_YET_OBSERVED"
            or int(record.get("observed_cycle", 0)) != current_state.cycle - 1
        ):
            continue
        if record.get("influence_class") in {
            "NONE", "FRAMEWORK_CAPTURE", "DRIFT",
        }:
            record["persistence"] = "NOT_APPLICABLE"
            continue
        position = positions.get(str(record.get("receiving_agent", "")))
        checks: list[bool] = []
        if position is not None:
            for change in record.get("preference_changes", []):
                checks.append(
                    position.get("preferred_action") == change.get("current_action")
                )
            for change in record.get("constraint_changes", []):
                checks.append(
                    position.get("active_constraint") == change.get("current_constraint")
                )
            for change in record.get("confidence_changes", []):
                old = float(change.get("previous_epistemic_confidence", 0.0))
                new = float(change.get("current_epistemic_confidence", 0.0))
                now = float(position.get("epistemic_confidence", 0.0))
                checks.append(now >= new if new >= old else now <= new)
        for change in record.get("question_changes", []):
            key = change.get("question_key")
            checks.append(
                key not in current_questions
                if change.get("resolution_type") else key in current_questions
            )
        for change in record.get("conflict_changes", []):
            key = change.get("conflict_key")
            checks.append(
                key not in current_conflicts
                if change.get("resolution_type") else key in current_conflicts
            )
        if not checks:
            record["persistence"] = "NOT_OBSERVABLE"
        elif any(checks):
            record["persistence"] = "PERSISTENT"
        else:
            record["persistence"] = "REVERSED_OR_DROPPED"


def _question_key(specialist: str, category: str, question: str) -> str:
    words = sorted(set(re.findall(r"[a-z0-9]+", question.casefold())) - {
        "a", "an", "and", "are", "is", "of", "or", "the", "to", "whether",
    })
    return _stable_key("QUESTION", {
        "specialist": specialist.casefold(),
        "category": category.upper(),
        "concepts": words,
    })


def _framework_local_key(prefix: str, specialist: str, text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return _stable_key(prefix, {
        "specialist": specialist.casefold(), "text": normalized,
    })


def _framework_issue_type(text: str, declared_type: str) -> str:
    """Validate issue form without interpreting its normative substance."""
    normalized = " ".join(str(text).split())
    lowered = normalized.casefold()
    if (
        normalized.endswith("?")
        or re.match(
            r"^(?:whether|how|what|which|who|when|where|why|does|do|did|"
            r"can|could|should|would|will|is|are|was|were)\b",
            lowered,
        )
    ):
        return "OPEN_QUESTION"
    if re.search(r"(?:↔|\bversus\b|\bvs\.?\b|\bconflict\w*\b|\btension\b|\btrade[- ]off\b)", lowered):
        return "INTERNAL_CONFLICT"
    return declared_type


def _prior_framework_issues(previous_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for item in (previous_state or {}).get("framework_internal_conflicts", []):
        if isinstance(item, dict):
            issues.append({
                **dict(item),
                "issue_type": "INTERNAL_CONFLICT",
                "text": str(item.get("conflict", "")),
                "issue_key": str(item.get("issue_key") or item.get("conflict_key", "")),
            })
    for item in (previous_state or {}).get("framework_specific_open_questions", []):
        if isinstance(item, dict):
            issues.append({
                **dict(item),
                "issue_type": "OPEN_QUESTION",
                "text": str(item.get("question", "")),
                "issue_key": str(item.get("issue_key") or item.get("question_key", "")),
            })
    return issues


def _match_prior_framework_issue(
    prior_issues: Sequence[dict[str, Any]],
    specialist: str,
    text: str,
) -> dict[str, Any] | None:
    matches = [
        (_structure_overlap(text, str(item.get("text", ""))), item)
        for item in prior_issues
        if item.get("source_specialist") == specialist
    ]
    score, match = max(matches, default=(0.0, None), key=lambda pair: pair[0])
    return match if score >= 0.35 else None


def _problem_delta(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    candidates: Sequence[CandidateChunk],
) -> ProblemDelta:
    previous = previous or {}
    previous_positions = {
        str(item.get("specialist", "")): item
        for item in previous.get("agent_positions", [])
    }
    current_positions = {
        str(item.get("specialist", "")): item
        for item in current.get("agent_positions", [])
    }
    preference_changes: list[dict[str, Any]] = []
    confidence_changes: list[dict[str, Any]] = []
    constraint_changes: list[dict[str, Any]] = []
    for specialist in sorted(set(previous_positions) & set(current_positions)):
        old = previous_positions[specialist]
        new = current_positions[specialist]
        if (
            old.get("preferred_action") != new.get("preferred_action")
            or old.get("action_scores") != new.get("action_scores")
        ):
            preference_changes.append({
                "specialist": specialist,
                "previous_action": old.get("preferred_action", ""),
                "current_action": new.get("preferred_action", ""),
                "previous_scores": dict(old.get("action_scores", {})),
                "current_scores": dict(new.get("action_scores", {})),
            })
        old_epistemic = float(old.get("epistemic_confidence", 0.0))
        new_epistemic = float(new.get("epistemic_confidence", 0.0))
        old_preference = float(old.get("preference_strength", 0.0))
        new_preference = float(new.get("preference_strength", 0.0))
        if old_epistemic != new_epistemic or old_preference != new_preference:
            confidence_changes.append({
                "specialist": specialist,
                "previous_epistemic_confidence": old_epistemic,
                "current_epistemic_confidence": new_epistemic,
                "previous_preference_strength": old_preference,
                "current_preference_strength": new_preference,
            })
        if old.get("active_constraint") != new.get("active_constraint"):
            constraint_changes.append({
                "specialist": specialist,
                "previous_constraint": old.get("active_constraint", ""),
                "current_constraint": new.get("active_constraint", ""),
            })

    previous_constraints = {
        str(item.get("constraint_key", "")): item
        for item in previous.get("active_constraints", [])
    }
    current_constraints = {
        str(item.get("constraint_key", "")): item
        for item in current.get("active_constraints", [])
    }
    previous_conflicts = {
        str(item.get("conflict_key", "")): item
        for item in previous.get("active_conflicts", [])
    }
    current_conflicts = {
        str(item.get("conflict_key", "")): item
        for item in current.get("active_conflicts", [])
    }
    previous_questions = {
        str(item.get("question_key", "")): item
        for item in previous.get("unresolved_questions", [])
    }
    current_questions = {
        str(item.get("question_key", "")): item
        for item in current.get("unresolved_questions", [])
    }
    previous_internal_conflicts = {
        str(item.get("conflict_key", "")): item
        for item in previous.get("framework_internal_conflicts", [])
    }
    current_internal_conflicts = {
        str(item.get("conflict_key", "")): item
        for item in current.get("framework_internal_conflicts", [])
    }
    previous_actions = {
        str(item.get("action", "")) for item in previous.get("live_actions", [])
    }
    current_actions = {
        str(item.get("action", "")) for item in current.get("live_actions", [])
    }
    synthesis_added = bool(current_actions - previous_actions) and bool(previous)
    candidate_by_specialist = {candidate.specialist: candidate for candidate in candidates}

    resolved_questions: list[dict[str, Any]] = []
    for key in sorted(set(previous_questions) - set(current_questions)):
        item = previous_questions[key]
        specialist = str(item.get("source_specialist", ""))
        candidate = candidate_by_specialist.get(specialist)
        same_source_new = any(
            question.get("source_specialist") == specialist
            for question in current_questions.values()
        )
        if same_source_new:
            resolution = "RECLASSIFIED"
        elif candidate is None:
            resolution = "WITHDRAWN"
        elif candidate.framework_retention_status == "NEW_ACTION_GROUNDING":
            resolution = "RESOLVED_BY_ACTION_GROUNDING"
        elif synthesis_added:
            resolution = "RESOLVED_BY_SYNTHESIS"
        elif candidate.workspace_reasoning_effect in {"FACTUAL", "BOTH"}:
            resolution = "RESOLVED_BY_SCENARIO_FACT"
        elif candidate.workspace_reasoning_effect == "NORMATIVE":
            resolution = "RESOLVED_BY_FRAMEWORK_CLARIFICATION"
        else:
            resolution = "WITHDRAWN"
        resolved_questions.append({
            **dict(item),
            "resolution_type": resolution,
            "resolved_in_cycle": current.get("cycle", 0),
        })

    resolved_conflicts: list[dict[str, Any]] = []
    current_conflict_types = {
        str(item.get("conflict_type", "")) for item in current_conflicts.values()
    }
    for key in sorted(set(previous_conflicts) - set(current_conflicts)):
        item = previous_conflicts[key]
        conflict_type = str(item.get("conflict_type", ""))
        if conflict_type in current_conflict_types:
            resolution = "RECLASSIFIED"
        elif synthesis_added:
            resolution = "RESOLVED_BY_SYNTHESIS"
        elif conflict_type == "ACTION_PREFERENCE":
            resolution = "RESOLVED_BY_DELIBERATIVE_CONVERGENCE"
        elif conflict_type == "EPISTEMIC_UNCERTAINTY" and resolved_questions:
            resolution = resolved_questions[0]["resolution_type"]
        else:
            resolution = "WITHDRAWN"
        resolved_conflicts.append({
            **dict(item),
            "resolution_type": resolution,
            "resolved_in_cycle": current.get("cycle", 0),
        })

    reframed_internal_conflicts: list[dict[str, Any]] = []
    for key in sorted(set(previous_internal_conflicts) & set(current_internal_conflicts)):
        old_item = previous_internal_conflicts[key]
        new_item = current_internal_conflicts[key]
        old_text = str(old_item.get("conflict", ""))
        new_text = str(new_item.get("conflict", ""))
        if _normalized_structure(old_text) != _normalized_structure(new_text):
            reframed_internal_conflicts.append({
                "source_specialist": str(new_item.get("source_specialist", "")),
                "previous_conflict_key": key,
                "current_conflict_key": key,
                "previous_conflict": old_text,
                "current_conflict": new_text,
                "reframed_in_cycle": current.get("cycle", 0),
            })

    old_unmatched = {
        key: item for key, item in previous_internal_conflicts.items()
        if key not in current_internal_conflicts
    }
    new_unmatched = {
        key: item for key, item in current_internal_conflicts.items()
        if key not in previous_internal_conflicts
    }
    reframed_old_keys: set[str] = set()
    reframed_new_keys: set[str] = set()
    specialists = sorted({
        str(item.get("source_specialist", ""))
        for item in (*old_unmatched.values(), *new_unmatched.values())
    })
    for specialist in specialists:
        old_items = [
            (key, item) for key, item in old_unmatched.items()
            if item.get("source_specialist") == specialist
        ]
        new_items = [
            (key, item) for key, item in new_unmatched.items()
            if item.get("source_specialist") == specialist
        ]
        # A one-for-one replacement is safely interpretable as a reframe. With
        # multiple unmatched tensions, avoid guessing semantic correspondence.
        if len(old_items) == 1 and len(new_items) == 1:
            old_key, old_item = old_items[0]
            new_key, new_item = new_items[0]
            reframed_old_keys.add(old_key)
            reframed_new_keys.add(new_key)
            reframed_internal_conflicts.append({
                "source_specialist": specialist,
                "previous_conflict_key": old_key,
                "current_conflict_key": new_key,
                "previous_conflict": old_item.get("conflict", ""),
                "current_conflict": new_item.get("conflict", ""),
                "reframed_in_cycle": current.get("cycle", 0),
            })

    return ProblemDelta(
        from_cycle=int(previous.get("cycle", 0)),
        to_cycle=int(current.get("cycle", 0)),
        preference_changes=tuple(preference_changes),
        confidence_changes=tuple(confidence_changes),
        constraint_changes=tuple(constraint_changes),
        new_constraints=tuple(
            dict(current_constraints[key])
            for key in sorted(set(current_constraints) - set(previous_constraints))
        ),
        removed_constraints=tuple(
            dict(previous_constraints[key])
            for key in sorted(set(previous_constraints) - set(current_constraints))
        ),
        new_conflicts=tuple(
            dict(current_conflicts[key])
            for key in sorted(set(current_conflicts) - set(previous_conflicts))
        ),
        resolved_conflicts=tuple(resolved_conflicts),
        new_questions=tuple(
            dict(current_questions[key])
            for key in sorted(set(current_questions) - set(previous_questions))
        ),
        resolved_questions=tuple(resolved_questions),
        new_internal_conflicts=tuple(
            dict(item) for key, item in sorted(new_unmatched.items())
            if key not in reframed_new_keys
        ),
        resolved_internal_conflicts=tuple({
            **dict(item),
            "resolution_type": "RESOLVED_OR_WITHDRAWN",
            "resolved_in_cycle": current.get("cycle", 0),
        } for key, item in sorted(old_unmatched.items()) if key not in reframed_old_keys),
        reframed_internal_conflicts=tuple(reframed_internal_conflicts),
    )


def _position(candidate: CandidateChunk) -> AgentDeliberativePosition:
    choice_status = {
        "DIRECT": "DIRECT",
        "CONDITIONAL": "CONDITIONAL",
        "UNDERDETERMINED": "UNDERDETERMINED",
        "NORMATIVELY_CONTESTED": "PROVISIONAL",
        "OUTSIDE_ACTION_SET_WITH_FALLBACK": "FALLBACK",
        "OUTSIDE_ACTION_SET": "FORCED",
        "NO_POSITION": "FORCED",
        "PARSE_FAILURE": "INVALID",
        "UNAVAILABLE": "PROVISIONAL",
    }.get(candidate.baseline_status, "PROVISIONAL")
    # Choice status describes the live judgment, while retaining its source
    # baseline shape. A parser failure must not erase independently structured
    # underdetermination reported by the current delegate.
    if choice_status not in {"FALLBACK", "FORCED"}:
        if candidate.assumption_status == "UNDERDETERMINED":
            choice_status = "UNDERDETERMINED"
        elif candidate.assumption_status == "CONDITIONAL":
            choice_status = "CONDITIONAL"
        elif candidate.assumption_status == "NORMATIVELY_CONTESTED":
            choice_status = "PROVISIONAL"
    return AgentDeliberativePosition(
        specialist=candidate.specialist,
        preferred_action=candidate.recommended_action,
        action_scores=dict(candidate.action_scores),
        preference_strength=candidate.preference_strength,
        epistemic_confidence=candidate.epistemic_confidence,
        active_constraint=candidate.constraint,
        unresolved=candidate.unresolved,
        assumption_status=candidate.assumption_status,
        choice_status=choice_status,
        choice_condition=candidate.baseline_condition,
        preferred_extension=candidate.baseline_preferred_extension,
        supporting_proposition_ids=tuple(candidate.supporting_proposition_ids),
        decision_critical_proposition_ids=tuple(
            candidate.decision_critical_proposition_ids
        ),
        weakest_decision_critical_status=candidate.weakest_decision_critical_status,
    )


def _question_grounding_ids(
    question: str, graph: Any | None, scenario_text: str = "",
) -> list[str]:
    """Trace a question to current scenario clauses using graph vocabulary."""
    stop = {
        "and", "are", "for", "from", "how", "the", "their", "this", "versus",
        "what", "whether", "with", "unknown", "relative", "weights", "value",
    }
    concepts = set(re.findall(r"[a-z]{4,}", question.casefold())) - stop
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for node in (graph.nodes.values() if graph is not None else []):
        clause_id = str(node.attributes.get("clause_id") or node.attributes.get("source_clause_id") or "")
        if not clause_id or clause_id in seen:
            continue
        vocabulary = set(re.findall(r"[a-z]{4,}", node.label.casefold()))
        overlap = len(concepts & vocabulary)
        if overlap:
            scored.append((overlap, clause_id))
            seen.add(clause_id)
    grounded = [clause_id for _score, clause_id in sorted(scored, reverse=True)[:4]]
    if grounded or not scenario_text:
        return grounded
    for clause in segment_scenario_clauses(scenario_text):
        vocabulary = set(re.findall(r"[a-z]{4,}", clause["text"].casefold()))
        overlap = len(concepts & vocabulary)
        if overlap:
            scored.append((overlap, clause["clause_id"]))
    return [clause_id for _score, clause_id in sorted(scored, reverse=True)[:4]]


def _uncertainty_kind(category: str, question: str) -> str:
    """Classify the uncertainty independently of its requested next operation."""
    from .uncertainty_types import uncertainty_kind_for
    return uncertainty_kind_for(category, question)


def opening_problem_state(
    actions: Sequence[str],
    scenario_text: str = "",
    graph: Any | None = None,
    world_model: Any | None = None,
) -> dict[str, Any]:
    """Frame the shared problem before any delegate has spoken.

    The opening cycle used to receive an empty state, so the first responses
    were written without the canonical clause and action vocabulary the rest of
    the run is validated against. Delegates then grounded their questions in
    words the graph does not use, and those questions were unauditable for the
    remainder of the run.

    This frame is deliberately positionless. It carries the action set, the
    scenario clauses, and the already-compiled graph identities, and it carries
    no preference, no salient claim, and no framework verdict, so the opening
    cycle stays an independent read of a shared world rather than a reaction to
    a preselected answer. ``committed_world`` is the admitted typed model when
    one exists; it is scenario evidence and is copied forward every cycle.
    """
    from .world_state import compact_committed_world
    state = DeliberativeProblemState(
        cycle=0,
        live_actions=tuple(
            {"action_id": f"A{index}", "action": str(action)}
            for index, action in enumerate(actions)
        ),
        agent_positions=(),
        active_constraints=(),
        active_conflicts=(),
        unresolved_questions=(),
        framework_internal_conflicts=(),
        framework_specific_open_questions=(),
        framework_insights=(),
        workspace_contributions=(),
        dissenting_positions=(),
        current_plurality="",
        primary_unresolved="ASSESS_FACTS",
        unresolved_categories=("ASSESS_FACTS",),
        state_role="OPENING_PROBLEM_FRAME",
        committed_world=compact_committed_world(world_model),
    )
    data = state.to_dict()
    data["scenario_clauses"] = [
        {"clause_id": clause["clause_id"], "text": " ".join(str(clause["text"]).split())[:240]}
        for clause in segment_scenario_clauses(scenario_text)
    ]
    grounded_identities: list[dict[str, str]] = []
    for node in (graph.nodes.values() if graph is not None else []):
        clause_id = str(
            node.attributes.get("clause_id")
            or node.attributes.get("source_clause_id")
            or ""
        )
        grounded_identities.append({
            "node_id": node.id,
            "kind": node.kind,
            "label": " ".join(str(node.label).split())[:120],
            "clause_id": clause_id,
        })
    data["grounded_identities"] = sorted(
        grounded_identities, key=lambda item: item["node_id"],
    )[:24]
    data["grounding_vocabulary_note"] = (
        "Cite these clause_id and node_id values when reporting unresolved "
        "questions; a question the graph cannot trace to one of them cannot be "
        "audited."
    )
    return data


def build_deliberative_problem_state(
    cycle: int,
    actions: Sequence[str],
    candidates: Sequence[CandidateChunk],
    current_plurality: str,
    winner: CandidateChunk,
    previous_state: dict[str, Any] | None = None,
    graph: Any | None = None,
    scenario_text: str = "",
    proposals: Sequence[SynthesisProposal] = (),
) -> DeliberativeProblemState:
    """Project one completed cycle without creating new moral or factual claims."""
    valid = sorted(
        (candidate for candidate in candidates if candidate.schema_valid),
        key=lambda candidate: candidate.specialist,
    )
    positions = tuple(_position(candidate) for candidate in valid)
    previous_contributions = {
        str(item.get("agent", "")): dict(item)
        for item in (previous_state or {}).get("workspace_contributions", [])
        if isinstance(item, dict) and str(item.get("agent", ""))
    }
    contributions = tuple(
        _reconcile_workspace_contribution(
            _workspace_contribution(candidate),
            previous_contributions.get(candidate.specialist),
            candidate,
        )
        for candidate in valid
    )
    live_actions = tuple(
        {"action_id": f"A{index}", "action": str(action)}
        for index, action in enumerate(actions)
    )

    constraints: list[dict[str, Any]] = []
    for constraint in sorted({candidate.constraint for candidate in valid}):
        holders = [candidate for candidate in valid if candidate.constraint == constraint]
        constraint_entry = {
            "constraint": constraint,
            "specialists": [candidate.specialist for candidate in holders],
            "preferred_actions": sorted({candidate.recommended_action for candidate in holders}),
            "source_type": "FRAMEWORK_ATTRIBUTED",
        }
        constraint_entry["constraint_key"] = _stable_key(
            "CONSTRAINT", {"constraint": constraint}
        )
        constraints.append(constraint_entry)

    conflicts: list[dict[str, Any]] = []
    preferred_actions = sorted({candidate.recommended_action for candidate in valid})
    if len(preferred_actions) > 1:
        conflict = {
            "conflict_type": "ACTION_PREFERENCE",
            "actions": preferred_actions,
            "specialists": [candidate.specialist for candidate in valid],
        }
        conflict["conflict_key"] = _stable_key("CONFLICT", {
            "type": conflict["conflict_type"], "actions": preferred_actions,
        })
        conflicts.append(conflict)
    active_constraint_names = sorted({candidate.constraint for candidate in valid})
    if len(active_constraint_names) > 1:
        conflict = {
            "conflict_type": "NORMATIVE_CONSTRAINT_TENSION",
            "constraints": active_constraint_names,
            "specialists": [candidate.specialist for candidate in valid],
        }
        conflict["conflict_key"] = _stable_key("CONFLICT", {
            "type": conflict["conflict_type"], "constraints": active_constraint_names,
        })
        conflicts.append(conflict)
    uncertain = [candidate for candidate in valid if candidate.unresolved != "NONE"]
    if uncertain:
        conflict = {
            "conflict_type": "EPISTEMIC_UNCERTAINTY",
            "specialists": [candidate.specialist for candidate in uncertain],
            "categories": sorted({candidate.unresolved for candidate in uncertain}),
        }
        conflict["conflict_key"] = _stable_key("CONFLICT", {
            "type": conflict["conflict_type"],
            "categories": conflict["categories"],
        })
        conflicts.append(conflict)

    constrained_choices = [
        position for position in positions
        if position.choice_status in {"FALLBACK", "FORCED"}
    ]
    if constrained_choices:
        conflict = {
            "conflict_type": "ACTION_SET_ADEQUACY",
            "actions": list(actions),
            "specialists": [position.specialist for position in constrained_choices],
            "choice_statuses": sorted({position.choice_status for position in constrained_choices}),
            "source_type": "BASELINE_CHOICE_STRUCTURE",
        }
        conflict["conflict_key"] = _stable_key("CONFLICT", {
            "type": conflict["conflict_type"],
            "specialists": conflict["specialists"],
        })
        conflicts.append(conflict)

    framework_internal_conflicts: list[dict[str, Any]] = []
    framework_specific_open_questions: list[dict[str, Any]] = []
    framework_insights: list[dict[str, Any]] = []
    prior_framework_issues = _prior_framework_issues(previous_state)

    def admit_framework_issue(
        specialist: str, text: str, declared_type: str,
    ) -> None:
        text = " ".join(str(text).split())[:240]
        if not text:
            return
        issue_type = _framework_issue_type(text, declared_type)
        prior = _match_prior_framework_issue(
            prior_framework_issues, specialist, text,
        )
        issue_key = (
            str(prior.get("issue_key", ""))
            if prior else ""
        ) or _framework_local_key("FRAMEWORK_ISSUE", specialist, text)
        history = [
            " ".join(str(item).split())[:240]
            for item in ((prior or {}).get("wording_history", []) or [])
            if " ".join(str(item).split())
        ]
        prior_text = " ".join(str((prior or {}).get("text", "")).split())
        if prior_text and _normalized_structure(prior_text) != _normalized_structure(text):
            history.append(prior_text)
        history = list(dict.fromkeys(history))[:8]
        reclassified_from = (
            str(prior.get("issue_type", ""))
            if prior and prior.get("issue_type") != issue_type else ""
        )
        if issue_type == "INTERNAL_CONFLICT":
            if any(item.get("issue_key") == issue_key for item in framework_internal_conflicts):
                return
            framework_internal_conflicts.append({
                "issue_key": issue_key,
                "issue_type": issue_type,
                "conflict_key": issue_key,
                "source_specialist": specialist,
                "conflict": text,
                "wording_history": history,
                "reclassified_from": reclassified_from,
                "status": "UNRESOLVED",
                "source_type": "FRAMEWORK_ATTRIBUTED_INTERNAL_CONFLICT",
            })
        else:
            if any(item.get("issue_key") == issue_key for item in framework_specific_open_questions):
                return
            framework_specific_open_questions.append({
                "issue_key": issue_key,
                "issue_type": issue_type,
                "question_key": issue_key,
                "source_specialist": specialist,
                "question": text,
                "wording_history": history,
                "reclassified_from": reclassified_from,
                "status": "OPEN",
                "source_type": "FRAMEWORK_ATTRIBUTED_OPEN_QUESTION",
            })

    for candidate in valid:
        for conflict_text in candidate.framework_internal_conflicts:
            admit_framework_issue(
                candidate.specialist, conflict_text, "INTERNAL_CONFLICT",
            )
        for question_text in candidate.framework_specific_open_questions:
            admit_framework_issue(
                candidate.specialist, question_text, "OPEN_QUESTION",
            )
        for insight in candidate.framework_insights:
            proposition = " ".join(str(insight.get("proposition", "")).split())[:240]
            if not proposition:
                continue
            framework_insights.append({
                "insight_id": str(insight.get("insight_id")) or _framework_local_key(
                    "FRAMEWORK_INSIGHT", candidate.specialist, proposition,
                ),
                "source_specialist": candidate.specialist,
                "action_id": str(insight.get("action_id", "")),
                "insight_kind": str(insight.get("insight_kind", "REFINEMENT")),
                "proposition": proposition,
                "normative_construct": str(insight.get("normative_construct", "")),
                "relation_to_committed_state": str(
                    insight.get("relation_to_committed_state", "ELABORATES")
                ),
                "grounded_in": list(insight.get("grounded_in", [])),
                "admission_status": "ADMITTED_INSIGHT_ONLY",
                "voting_effect": "NONE",
                "qualification": str(insight.get("qualification", "")),
            })
    # Keep framework-local structures visible in ProblemState when a recurrent
    # response merely omits them. They remain attributed to their originating
    # framework and are marked suspended, so they cannot become shared facts or
    # silently masquerade as resolved questions.
    contribution_by_agent = {item.agent: item for item in contributions}
    current_conflict_keys = {
        (item["source_specialist"], _normalized_structure(item["conflict"]))
        for item in framework_internal_conflicts
    }
    current_question_keys = {
        (item["source_specialist"], _normalized_structure(item["question"]))
        for item in framework_specific_open_questions
    }
    for prior in (previous_state or {}).get("framework_internal_conflicts", []):
        if not isinstance(prior, dict):
            continue
        if str(prior.get("status", "")).upper() in {
            "SUPERSEDED", "RESOLVED", "WITHDRAWN", "REFINED",
        }:
            continue
        agent = str(prior.get("source_specialist", ""))
        text = " ".join(str(prior.get("conflict", "")).split())
        contribution = contribution_by_agent.get(agent)
        key = (agent, _normalized_structure(text))
        has_reframed_successor = any(
            item["source_specialist"] == agent
            and _structure_overlap(text, item["conflict"]) >= 0.35
            for item in framework_internal_conflicts
        )
        if (
            text and contribution is not None
            and contribution.update_type.upper() != "RESOLUTION"
            and key not in current_conflict_keys
            and not has_reframed_successor
        ):
            framework_internal_conflicts.append({
                "issue_key": str(prior.get("issue_key") or prior.get("conflict_key")),
                "issue_type": "INTERNAL_CONFLICT",
                "conflict_key": str(prior.get("issue_key") or prior.get("conflict_key")) or _framework_local_key(
                    "FRAMEWORK_INTERNAL_CONFLICT", agent, text,
                ),
                "source_specialist": agent,
                "conflict": text,
                "wording_history": list(prior.get("wording_history", []) or []),
                "reclassified_from": "",
                "status": "SUSPENDED",
                "source_type": "FRAMEWORK_ATTRIBUTED_INTERNAL_CONFLICT",
            })
            current_conflict_keys.add(key)
    for prior in (previous_state or {}).get("framework_specific_open_questions", []):
        if not isinstance(prior, dict):
            continue
        if str(prior.get("status", "")).upper() in {
            "SUPERSEDED", "RESOLVED", "WITHDRAWN", "REFINED",
        }:
            continue
        agent = str(prior.get("source_specialist", ""))
        text = " ".join(str(prior.get("question", "")).split())
        contribution = contribution_by_agent.get(agent)
        key = (agent, _normalized_structure(text))
        has_refined_successor = any(
            item["source_specialist"] == agent
            and _structure_overlap(text, item["question"]) >= 0.55
            for item in framework_specific_open_questions
        )
        if (
            text and contribution is not None
            and contribution.update_type.upper() != "RESOLUTION"
            and key not in current_question_keys
            and not has_refined_successor
        ):
            framework_specific_open_questions.append({
                "issue_key": str(prior.get("issue_key") or prior.get("question_key")),
                "issue_type": "OPEN_QUESTION",
                "question_key": str(prior.get("issue_key") or prior.get("question_key")) or _framework_local_key(
                    "FRAMEWORK_SPECIFIC_QUESTION", agent, text,
                ),
                "source_specialist": agent,
                "question": text,
                "wording_history": list(prior.get("wording_history", []) or []),
                "reclassified_from": "",
                "status": "SUSPENDED",
                "source_type": "FRAMEWORK_ATTRIBUTED_OPEN_QUESTION",
            })
            current_question_keys.add(key)
    framework_internal_conflicts.sort(
        key=lambda item: (item["source_specialist"], item["conflict_key"])
    )
    framework_specific_open_questions.sort(
        key=lambda item: (item["source_specialist"], item["question_key"])
    )
    framework_insights.sort(
        key=lambda item: (item["source_specialist"], item["insight_id"])
    )

    unresolved: list[dict[str, str]] = []
    for candidate in uncertain:
        question = next((value for value in (
            candidate.utilitarian_missing_comparison,
            candidate.unsupported_assumption,
            candidate.reversal_condition,
            candidate.landscape_tiebreaker_failure,
        ) if str(value).strip() and str(value).strip().upper() != "NONE"), candidate.unresolved)
        question_text = " ".join(str(question).split())[:240]
        from .uncertainty_types import normalize_unresolved_marker
        category = normalize_unresolved_marker(candidate.unresolved)
        unresolved.append({
            "question_key": _question_key(
                candidate.specialist, category, question_text,
            ),
            "source_specialist": candidate.specialist,
            "category": category,
            "uncertainty_kind": _uncertainty_kind(category, question_text),
            "question": question_text,
            "grounded_in": _question_grounding_ids(
                question_text, graph, scenario_text,
            ),
            "source_type": "FRAMEWORK_ATTRIBUTED_UNRESOLVED",
        })

    projected_proposals: list[dict[str, Any]] = []
    expected_reviewers = {candidate.specialist for candidate in valid}
    for proposal in proposals:
        if not proposal.proposal_id:
            continue
        reviews = dict(proposal.framework_reviews)
        valid_reviews = [review for review in reviews.values() if review.get("valid")]
        substantive_reviews = [
            review for review in valid_reviews
            if str(review.get("framework_status", "")).upper()
            != "UNDERDETERMINED"
        ]
        status_counts: dict[str, int] = {}
        for review in valid_reviews:
            status = str(review.get("framework_status", "UNDERDETERMINED"))
            status_counts[status] = status_counts.get(status, 0) + 1
        feasibility_concerns = list(dict.fromkeys(
            str(concern)
            for review in valid_reviews
            for concern in review.get("feasibility_concerns", [])
            if str(concern).strip()
        ))
        required_conditions = list(dict.fromkeys(
            str(condition)
            for review in valid_reviews
            for condition in review.get("required_conditions", [])
            if str(condition).strip()
        ))
        projected_proposals.append({
            "proposal_id": proposal.proposal_id,
            "text": proposal.action,
            "source_constraints": list(proposal.addressed_constraints),
            "source_agents": list(proposal.source_agents),
            "predicted_consequences": [dict(item) for item in proposal.predicted_consequences],
            "feasibility": proposal.feasibility,
            "feasibility_status": proposal.feasibility_status,
            "grounding_status": proposal.grounding_status,
            "framework_reviews": reviews,
            "review_summary": {
                "expected_reviewers": sorted(expected_reviewers),
                "received_reviewers": sorted(reviews),
                "missing_reviewers": sorted(expected_reviewers - set(reviews)),
                "valid_review_count": len(valid_reviews),
                "substantive_review_count": len(substantive_reviews),
                "invalid_review_count": len(reviews) - len(valid_reviews),
                "framework_status_counts": status_counts,
                "feasibility_concerns": feasibility_concerns,
                "required_conditions": required_conditions,
            },
            "promotion_status": proposal.promotion_status,
            "source_type": "SYNTHESIS_PROPOSAL",
        })
    proposal_projection = tuple(projected_proposals)

    for proposal in proposals:
        if not proposal.proposal_id or proposal.promotion_status == "REJECTED":
            continue
        unresolved_claims = [
            *proposal.introduced_requirements,
            *( ["feasibility"] if proposal.feasibility_status in {"UNKNOWN", "UNCERTAIN", "PLAUSIBLE"} else [] ),
        ]
        unresolved_claims = [
            " ".join(str(claim).split()) for claim in unresolved_claims
            if " ".join(str(claim).split())
        ]
        if unresolved_claims:
            question_text = (
                f"Whether synthesis proposal {proposal.proposal_id} is feasible: "
                + "; ".join(unresolved_claims)
            )[:240]
            unresolved.append({
                "question_key": _question_key(
                    "workspace_synthesis", "CHECK_FEASIBILITY", question_text,
                ),
                "source_specialist": "workspace_synthesis",
                "raised_by": list(proposal.source_agents) or ["workspace_synthesis"],
                "category": "CHECK_FEASIBILITY",
                "uncertainty_kind": "EMPIRICAL_UNCERTAINTY",
                "question": question_text,
                "grounded_in": [proposal.proposal_id],
                "source_type": "SYNTHESIS_PROPOSAL",
            })

    if constrained_choices:
        specialists = [position.specialist for position in constrained_choices]
        question_text = (
            "Is apparent agreement partly an artifact of forcing framework-preferred "
            "responses into the current action set?"
        )
        unresolved.append({
            "question_key": _question_key(
                "+".join(specialists), "ACTION_SET_ADEQUACY", question_text,
            ),
            "source_specialist": "+".join(specialists),
            "raised_by": specialists,
            "category": "ACTION_SET_ADEQUACY",
            "uncertainty_kind": "NORMATIVE_UNCERTAINTY",
            "question": question_text,
            "grounded_in": [f"A{index}" for index in range(len(actions))],
            "source_type": "BASELINE_CHOICE_STRUCTURE",
        })

    dissent = tuple(
        position for position in positions
        if position.preferred_action != current_plurality
    )
    salient_justification = " ".join(
        value for value in (winner.decision_rule, winner.rationale) if value
    ).strip()
    salient = {
        "source_specialist": winner.specialist,
        "preferred_action": winner.recommended_action,
        "constraint": winner.constraint,
        "conditional_justification": salient_justification[:480],
        "unresolved": winner.unresolved,
        "reversal_condition": winner.reversal_condition,
        "evidence_basis": winner.evidence_basis,
        "tension_engagement": winner.tension_engagement,
        "tension_target_keys": list(winner.tension_target_keys),
        "source_type": "FRAMEWORK_ATTRIBUTED_POSITION",
        "workspace_contribution": next((
            item.to_dict() for item in contributions
            if item.agent == winner.specialist
        ), {}),
    }
    composition_counts: dict[str, int] = {}
    by_action: dict[str, dict[str, int]] = {}
    for position in positions:
        composition_counts[position.choice_status] = (
            composition_counts.get(position.choice_status, 0) + 1
        )
        action_counts = by_action.setdefault(position.preferred_action, {})
        action_counts[position.choice_status] = action_counts.get(position.choice_status, 0) + 1
    preferred = {position.preferred_action for position in positions if position.preferred_action}
    surface_consensus = "UNANIMOUS" if len(preferred) == 1 and positions else "SPLIT"
    if not positions:
        deliberative_consensus = "INSUFFICIENT"
    elif surface_consensus != "UNANIMOUS":
        deliberative_consensus = "CONTESTED"
    elif set(composition_counts) == {"DIRECT"}:
        deliberative_consensus = "UNANIMOUS_DIRECT"
    else:
        deliberative_consensus = "SURFACE_AGREEMENT_WITH_QUALIFICATIONS"
    warnings = tuple({
        "specialist": candidate.specialist,
        "warning": "FRAMEWORK_RETENTION",
        "status": candidate.framework_retention_status,
    } for candidate in valid if (
        not candidate.framework_constraint_retained
        or candidate.framework_retention_status in {
            "LOST", "UNCLEAR", "COMMITTED_WITH_UNCERTAINTY", "UPDATE_REJECTED",
        }
    ))
    previous_questions = list((previous_state or {}).get("unresolved_questions", []))
    previous_question_types = {
        (str(item.get("source_specialist", "")), str(item.get("category", "")))
        for item in previous_questions
    }
    # Delegates rebuild their open questions every cycle, so a question the
    # workspace already audited to a stable answer reappears here verbatim.
    # Annotate it from the graph ledger and withhold it from the audit slot
    # while the evidence its answer rested on is unchanged.
    resolution_index = question_resolution_index(graph)
    for item in unresolved:
        record = resolution_index.get(str(item.get("question_key", "")))
        if record is None:
            item["resolution_status"] = "OPEN"
            continue
        item["resolution_status"] = record["status"]
        item["recorded_resolution"] = record["resolution"]
        item["resolved_cycle"] = record["resolved_cycle"]
        item["evidence_requirement"] = record.get("evidence_requirement", "")
    settled_keys = {
        key for key, record in resolution_index.items()
        if record["status"] == "SETTLED"
    }

    action_node_ids = [f"A{index}" for index in range(len(actions))]

    def audit_candidate(item: dict[str, Any], grounding_status: str) -> dict[str, Any]:
        return {
            "issue_id": item["question_key"],
            "source": "problem_state.unresolved_questions",
            # An ungrounded question still concerns this action set, so the
            # audit is anchored to the canonical action nodes. That keeps its
            # answer recordable and reopenable while grounding_status stays
            # honest that no scenario clause supports the question itself.
            "proposition": item["question"],
            "grounded_in": (
                list(item.get("grounded_in", []))
                if grounding_status == "CLAUSE_GROUNDED"
                else list(action_node_ids)
            ),
            "grounding_status": grounding_status,
            "raised_by": list(item.get("raised_by", [])) or [item["source_specialist"]],
            "category": item["category"],
            "uncertainty_kind": item.get("uncertainty_kind", "UNCLASSIFIED_UNCERTAINTY"),
            "status": (
                "PERSISTENT_UNRESOLVED"
                if (item["source_specialist"], item["category"]) in previous_question_types
                else "UNRESOLVED"
            ),
        }

    auditable = [
        item for item in unresolved
        if item["question_key"] not in settled_keys
    ]
    # A question the graph cannot tie to a clause is weaker evidence that the
    # issue is live, not proof that it is idle. Offer it only when nothing
    # grounded is competing for the slot, so ungrounded inquiry stays testable
    # without ever displacing a question the scenario actually supports.
    grounded_candidates = tuple(
        audit_candidate(item, "CLAUSE_GROUNDED")
        for item in auditable if item.get("grounded_in")
    )
    audit_candidates = grounded_candidates or tuple(
        audit_candidate(item, "UNGROUNDED")
        for item in auditable if not item.get("grounded_in")
    )
    resolved_questions = tuple(
        dict(record) for _key, record in sorted(resolution_index.items())
    )
    # A settled question keeps its category. "Resolving this would not change
    # the recommendation" is an answer about decision relevance, not a claim
    # that the underlying uncertainty is now known, so it must still be
    # reported as unresolved even though it no longer earns an audit.
    from .uncertainty_types import normalize_uncertainty_category
    unresolved_categories = list(dict.fromkeys(
        normalize_uncertainty_category(str(item.get("category", "")).upper())
        for item in unresolved if str(item.get("category", "")).strip()
    ))
    unresolved_categories = [c for c in unresolved_categories if c and c != "NONE"]
    if warnings:
        unresolved_categories.append("FRAMEWORK_GROUNDING_UNCERTAINTY")
    unresolved_categories = list(dict.fromkeys(unresolved_categories))
    unresolved_priority = (
        "VERIFY_FACTS", "DECISION_BOUNDARY", "CHECK_FEASIBILITY",
        "NORMATIVE_ADJUDICATION", "ACTION_SET_ADEQUACY",
        "FRAMEWORK_GROUNDING_UNCERTAINTY",
    )
    primary_unresolved = next(
        (category for category in unresolved_priority if category in unresolved_categories),
        unresolved_categories[0] if unresolved_categories else "NONE",
    )
    state = DeliberativeProblemState(
        cycle=max(1, int(cycle)),
        live_actions=live_actions,
        agent_positions=positions,
        active_constraints=tuple(constraints),
        active_conflicts=tuple(conflicts),
        unresolved_questions=tuple(unresolved),
        framework_internal_conflicts=tuple(framework_internal_conflicts),
        framework_specific_open_questions=tuple(framework_specific_open_questions),
        framework_insights=tuple(framework_insights),
        workspace_contributions=contributions,
        dissenting_positions=dissent,
        current_plurality=current_plurality,
        support_composition={
            "counts": composition_counts,
            "by_action": by_action,
            "total_positions": len(positions),
        },
        surface_consensus=surface_consensus,
        deliberative_consensus=deliberative_consensus,
        framework_warnings=warnings,
        audit_candidates=audit_candidates,
        resolved_questions=resolved_questions,
        unresolved_categories=tuple(unresolved_categories),
        primary_unresolved=primary_unresolved,
        proposals=proposal_projection,
        salient_position=salient,
        committed_world=dict((previous_state or {}).get("committed_world") or {}),
    )
    return replace(
        state,
        problem_delta=_problem_delta(previous_state, state.to_dict(), valid),
    )
