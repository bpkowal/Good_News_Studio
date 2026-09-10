from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class TestimonyBaseline:
    """Typed source stance preserved before workspace deliberation begins.

    ``action_id`` is authoritative only for a DIRECT stance.  A conditional
    testimony may retain a ``provisional_action_id`` without turning that
    provisional ranking into a frozen commitment.  ``NO_POSITION`` means the
    testimony does not offer a recommendation; ``PARSE_FAILURE`` means the
    parser could not reliably extract one.
    """

    status: str = "UNAVAILABLE"
    action_id: str = "NONE"
    provisional_action_id: str = "NONE"
    preferred_extension: str = ""
    condition: str = ""
    reason: str = ""
    rejected_action_ids: list[str] = field(default_factory=list)
    framework_commitments: dict[str, str] = field(default_factory=dict)
    numerical_role: str = "UNASSESSED"

    def __post_init__(self) -> None:
        allowed = {
            "DIRECT", "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
            "OUTSIDE_ACTION_SET", "OUTSIDE_ACTION_SET_WITH_FALLBACK",
            "NO_POSITION", "PARSE_FAILURE", "UNAVAILABLE",
        }
        normalized = str(self.status).strip().upper()
        self.status = normalized if normalized in allowed else "UNAVAILABLE"
        self.action_id = str(self.action_id).strip().upper() or "NONE"
        self.provisional_action_id = (
            str(self.provisional_action_id).strip().upper() or "NONE"
        )
        self.preferred_extension = " ".join(str(self.preferred_extension).split())[:240]
        self.condition = " ".join(str(self.condition).split())[:240]
        self.reason = " ".join(str(self.reason).split())[:240]
        self.rejected_action_ids = list(dict.fromkeys(
            str(action_id).strip().upper()
            for action_id in self.rejected_action_ids
            if str(action_id).strip()
        ))
        self.framework_commitments = {
            str(action_id).strip().upper(): " ".join(str(reason).split())[:180]
            for action_id, reason in self.framework_commitments.items()
            if str(action_id).strip() and " ".join(str(reason).split())
        }
        role = str(self.numerical_role).strip().upper()
        self.numerical_role = (
            role if role in {"DECISIVE", "SECONDARY", "IRRELEVANT", "UNASSESSED"}
            else "UNASSESSED"
        )
        if self.status == "DIRECT":
            self.provisional_action_id = self.action_id
            self.condition = ""
            self.preferred_extension = ""
        else:
            self.action_id = "NONE"
        if self.status not in {
            "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
        }:
            self.condition = ""
        if self.status == "OUTSIDE_ACTION_SET":
            self.preferred_extension = ""
        if self.status != "OUTSIDE_ACTION_SET_WITH_FALLBACK":
            self.preferred_extension = ""
        if self.status in {
            "OUTSIDE_ACTION_SET", "NO_POSITION", "PARSE_FAILURE", "UNAVAILABLE",
        }:
            self.provisional_action_id = "NONE"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArgumentChallenge:
    """Typed, non-evidentiary request to audit an inferential bridge.

    ``generated_by`` identifies the actual challenge author. ``about_specialist``
    identifies whose reasoning triggered the audit. They must not be conflated:
    a workspace-generated objection is not self-criticism by the target agent.
    """

    challenge_kind: str
    question: str
    about_specialist: str
    target_specialists: tuple[str, ...]
    grounded_in: tuple[str, ...] = ()
    trigger_fields: tuple[str, ...] = ()
    focus_action: str = ""
    priority: float = 0.5
    generated_by: str = "WORKSPACE_ARGUMENT_AUDITOR"
    raised_by: tuple[str, ...] = ()
    grounding_status: str = "UNGROUNDED"
    status: str = "UNTESTED"
    issue_id: str = ""

    def __post_init__(self) -> None:
        self.challenge_kind = self.challenge_kind.strip().upper()[:64] or "UNSPECIFIED"
        self.question = " ".join(self.question.split())[:240]
        self.about_specialist = self.about_specialist.strip().casefold()[:32]
        self.target_specialists = tuple(dict.fromkeys(
            str(value).strip().casefold()[:32]
            for value in self.target_specialists if str(value).strip()
        ))[:5]
        self.grounded_in = tuple(dict.fromkeys(
            str(value).strip()[:160]
            for value in self.grounded_in if str(value).strip()
        ))[:8]
        self.trigger_fields = tuple(dict.fromkeys(
            str(value).strip()[:120]
            for value in self.trigger_fields if str(value).strip()
        ))[:8]
        self.focus_action = " ".join(self.focus_action.split())[:180]
        self.priority = clamp(self.priority)
        self.generated_by = self.generated_by.strip().upper()[:64]
        self.raised_by = tuple(dict.fromkeys(
            str(value).strip().casefold()[:32]
            for value in self.raised_by if str(value).strip()
        ))[:5]
        grounding = self.grounding_status.strip().upper()
        self.grounding_status = grounding if grounding in {
            "PROPOSITION_GROUNDED", "CLAUSE_GROUNDED", "UNGROUNDED",
        } else "UNGROUNDED"
        self.status = self.status.strip().upper()[:32] or "UNTESTED"
        if not self.issue_id:
            identity = "|".join((
                self.generated_by,
                self.about_specialist,
                self.challenge_kind,
                self.question.casefold(),
                *self.trigger_fields,
            ))
            self.issue_id = "CHALLENGE:" + hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:16]
        elif not self.issue_id.startswith("CHALLENGE:"):
            raise ValueError("argument challenge issue_id must start with CHALLENGE:")
        if not self.question:
            raise ValueError("argument challenge requires a question")
        if not self.about_specialist or not self.target_specialists:
            raise ValueError("argument challenge requires an audited and target specialist")

    def as_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "source": "framework_argument_audit",
            "generated_by": self.generated_by,
            "about_specialist": self.about_specialist,
            "raised_by": list(self.raised_by),
            "target_specialists": list(self.target_specialists),
            "target_framework": self.target_specialists[0],
            "category": "ARGUMENT_CHALLENGE",
            "challenge_kind": self.challenge_kind,
            # Compatibility alias retained for existing trace readers.
            "uncertainty_kind": self.challenge_kind,
            "proposition": self.question,
            "question": self.question,
            "trigger_fields": list(self.trigger_fields),
            "grounded_in": list(self.grounded_in),
            "grounding_status": self.grounding_status,
            "status": self.status,
            "priority": self.priority,
            "focus_action": self.focus_action,
        }


def _broadcast_text(value: Any, limit: int) -> str:
    """Bound one semantic field before JSON serialization, never after it."""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip(" ,;:-") + "…"


def _broadcast_action_ids(problem_state: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("action", "")): str(item.get("action_id", ""))
        for item in problem_state.get("live_actions", []) or []
        if str(item.get("action", "")) and str(item.get("action_id", ""))
    }


def _broadcast_action_ref(value: Any, action_ids: dict[str, str]) -> str:
    text = str(value or "")
    return action_ids.get(text, _broadcast_text(text, 72) or "NONE")


def _attributed_items(
    values: Any, *, key: str, specialist: str, field: str, limit: int, count: int,
) -> list[str]:
    rows = [
        row for row in (values or [])
        if isinstance(row, dict) and str(row.get(key, "")) == specialist
    ]
    return list(dict.fromkeys(
        _broadcast_text(row.get(field, ""), limit)
        for row in rows
        if _broadcast_text(row.get(field, ""), limit)
    ))[:count]


_NATIVE_REASONING_CHAR_BUDGET = 6500
_FRAMEWORK_CAPSULE_CHAR_BUDGET = 8300


def _shrink_native_value(value: Any, text_limit: int) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _shrink_native_value(item, text_limit)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_shrink_native_value(item, text_limit) for item in value]
    if isinstance(value, str):
        return _broadcast_text(value, text_limit)
    return value


def _fit_native_reasoning(payload: Any) -> dict[str, Any]:
    """Keep native structure intact while bounding its explanatory prose."""
    native = dict(payload or {})
    if len(json.dumps(native, sort_keys=True)) <= _NATIVE_REASONING_CHAR_BUDGET:
        return native
    for limit in (64, 40, 24):
        fitted = _shrink_native_value(native, limit)
        if len(json.dumps(fitted, sort_keys=True)) <= _NATIVE_REASONING_CHAR_BUDGET:
            return fitted
    # Schema keys and action rows remain complete. If an unusually large
    # ledger still exceeds the ordinary cap, remove explanatory prose fields
    # before considering any structural row for omission.
    fitted = _shrink_native_value(native, 24)
    explanatory = {
        "valuation_reason", "special_obligation_basis", "priority_rule",
        "public_justification", "practical_judgment", "dependency_source",
        "responsibility_basis", "competing_care_claim",
        "classification_justification", "lexical_priority_justification",
        "public_reason",
    }

    def remove_explanations(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: remove_explanations(item)
                for key, item in value.items()
                if key not in explanatory
            }
        if isinstance(value, list):
            return [remove_explanations(item) for item in value]
        return value

    return remove_explanations(fitted)


def _fit_framework_capsule(capsule: dict[str, Any]) -> dict[str, Any]:
    """Enforce a per-framework cap by shrinking fields, never serialized JSON."""
    if len(json.dumps(capsule, sort_keys=True)) <= _FRAMEWORK_CAPSULE_CHAR_BUDGET:
        return capsule
    fitted = dict(capsule)
    fitted["qualifiers"] = dict(capsule.get("qualifiers", {}))
    fitted["counterclaims"] = [
        dict(item) for item in capsule.get("counterclaims", [])
    ]
    for text_limit, item_limit, id_limit in ((90, 2, 4), (64, 1, 3)):
        fitted["reason_for_lean"] = _broadcast_text(
            fitted.get("reason_for_lean", ""), text_limit
        )
        for field_name in (
            "supporting_premises", "conditional_dependencies", "defeaters",
            "decision_boundaries", "open_questions",
        ):
            fitted[field_name] = [
                _broadcast_text(item, text_limit)
                for item in fitted.get(field_name, [])
            ][:item_limit]
        fitted["supporting_proposition_ids"] = [
            _broadcast_text(item, 64)
            for item in fitted.get("supporting_proposition_ids", [])
        ][:id_limit]
        fitted["qualifiers"]["choice_condition"] = _broadcast_text(
            fitted["qualifiers"].get("choice_condition", "NONE"), text_limit
        ) or "NONE"
        fitted["counterclaims"] = [
            {
                "action_id": str(item.get("action_id", "UNRESOLVED")),
                "case": _broadcast_text(item.get("case", ""), text_limit),
            }
            for item in fitted.get("counterclaims", [])
        ][:item_limit]
        if len(json.dumps(fitted, sort_keys=True)) <= _FRAMEWORK_CAPSULE_CHAR_BUDGET:
            break
    return fitted


def _balanced_problem_state_projection(
    problem_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Project equal, attributed semantic capsules for recurrent broadcasts.

    This is a transport-only view. It does not mutate the complete ProblemState
    used by audits, salience, scoring, or trace serialization.
    """
    state = dict(problem_state or {})
    if not state:
        return {}
    action_ids = _broadcast_action_ids(state)
    live_actions = [
        {
            "action_id": str(item.get("action_id", "")),
            "label": _broadcast_text(item.get("action", ""), 120),
        }
        for item in state.get("live_actions", []) or []
        if isinstance(item, dict)
    ]
    positions = {
        str(item.get("specialist", "")): dict(item)
        for item in state.get("agent_positions", []) or []
        if isinstance(item, dict) and str(item.get("specialist", ""))
    }
    contributions = {
        str(item.get("agent", "")): dict(item)
        for item in state.get("workspace_contributions", []) or []
        if isinstance(item, dict) and str(item.get("agent", ""))
    }
    specialists = sorted(set(positions) | set(contributions))
    capsules: list[dict[str, Any]] = []
    for specialist in specialists:
        position = positions.get(specialist, {})
        contribution = contributions.get(specialist, {})
        native_reasoning = _fit_native_reasoning(
            contribution.get("native_reasoning", {})
        )
        grounds = [
            _broadcast_text(item, 120)
            for item in contribution.get("core_ground", []) or []
            if _broadcast_text(item, 120)
        ][:3]
        unresolved = [
            _broadcast_text(item, 120)
            for item in contribution.get("unresolved", []) or []
            if _broadcast_text(item, 120)
        ][:3]
        defeaters = [
            _broadcast_text(item, 120)
            for item in contribution.get("defeat_conditions", []) or []
            if _broadcast_text(item, 120)
        ][:3]
        action_cases = dict(contribution.get("action_cases", {}) or {})
        current_lean = _broadcast_action_ref(
            position.get("preferred_action", contribution.get("tendency", "")),
            action_ids,
        )
        counterclaims = [
            {
                "action_id": _broadcast_action_ref(action, action_ids),
                "case": _broadcast_text(case, 140),
            }
            for action, case in sorted(
                action_cases.items(), key=lambda item: _broadcast_action_ref(
                    item[0], action_ids
                )
            )
            if _broadcast_action_ref(action, action_ids) != current_lean
            and _broadcast_text(case, 140)
        ][:2]
        if not counterclaims:
            counterclaims = [
                {"action_id": "UNRESOLVED", "case": item}
                for item in _attributed_items(
                    state.get("framework_internal_conflicts", []),
                    key="source_specialist", specialist=specialist, field="conflict",
                    limit=140, count=2,
                )
            ]
        choice_condition = _broadcast_text(position.get("choice_condition", ""), 140)
        qualifiers = {
            "assumption_status": str(position.get("assumption_status", "NOT_AUDITED")),
            "choice_condition": choice_condition or "NONE",
            "weakest_dependency_status": str(
                position.get("weakest_decision_critical_status", "ESTABLISHED")
            ),
        }
        open_questions = _attributed_items(
            state.get("framework_specific_open_questions", []),
            key="source_specialist", specialist=specialist, field="question",
            limit=120, count=2,
        )
        capsules.append(_fit_framework_capsule({
            "agent": specialist,
            "source_type": "FRAMEWORK_ATTRIBUTED_PROJECTION",
            "current_lean": current_lean,
            "constraint": str(position.get(
                "active_constraint", contribution.get("constraint", "")
            )),
            "choice_status": str(position.get(
                "choice_status", contribution.get("choice_status", "")
            )),
            "unresolved_classification": str(position.get("unresolved", "NONE")),
            "reason_for_lean": grounds[0] if grounds else "NONE",
            "supporting_premises": grounds[1:] if len(grounds) > 1 else grounds,
            "supporting_proposition_ids": list(
                position.get("supporting_proposition_ids", []) or []
            )[:6],
            "qualifiers": qualifiers,
            "conditional_dependencies": unresolved,
            "defeaters": defeaters,
            "counterclaims": counterclaims,
            "decision_boundaries": list(dict.fromkeys([
                *defeaters, *([choice_condition] if choice_condition else []),
            ]))[:3],
            "open_questions": open_questions,
            "framework_native_reasoning": native_reasoning,
        }))

    # Opening frames contain no specialists. Preserve their grounding frame in
    # a bounded, field-aware form instead of truncating a serialized dictionary.
    opening = not specialists
    projection: dict[str, Any] = {
        "state_role": str(state.get("state_role", "")),
        "cycle": int(state.get("cycle", 0) or 0),
        "live_actions": live_actions,
        "current_plurality": _broadcast_action_ref(
            state.get("current_plurality", ""), action_ids
        ),
        "primary_unresolved": str(state.get("primary_unresolved", "NONE")),
        "unresolved_categories": list(state.get("unresolved_categories", []) or [])[:6],
        "surface_consensus": str(state.get("surface_consensus", "INSUFFICIENT")),
        "deliberative_consensus": str(
            state.get("deliberative_consensus", "INSUFFICIENT")
        ),
        "framework_capsules": capsules,
    }
    if opening:
        projection["scenario_clauses"] = [
            {
                "clause_id": str(item.get("clause_id", "")),
                "text": _broadcast_text(item.get("text", ""), 180),
            }
            for item in state.get("scenario_clauses", []) or []
            if isinstance(item, dict)
        ][:12]
        projection["grounded_identities"] = [
            {
                "node_id": str(item.get("node_id", "")),
                "kind": str(item.get("kind", "")),
                "label": _broadcast_text(item.get("label", ""), 90),
                "clause_id": str(item.get("clause_id", "")),
            }
            for item in state.get("grounded_identities", []) or []
            if isinstance(item, dict)
        ][:24]
    return projection


def _balanced_challenge_projection(challenges: Any) -> list[dict[str, Any]]:
    """Keep every target visible without slicing a serialized challenge list."""
    return [
        {
            "issue_id": str(item.get("issue_id", "")),
            "generated_by": str(item.get("generated_by", "")),
            "about_specialist": str(item.get("about_specialist", "")),
            "raised_by": list(item.get("raised_by", []) or [])[:3],
            "target_specialists": list(item.get("target_specialists", []) or [])[:3],
            "challenge_kind": str(item.get("challenge_kind", "")),
            "question": _broadcast_text(item.get("question", ""), 180),
            "status": str(item.get("status", "UNTESTED")),
            "grounded_in": list(item.get("grounded_in", []) or [])[:6],
        }
        for item in challenges or []
        if isinstance(item, dict)
    ][:8]


def _problem_delta_projection(delta: Any) -> dict[str, Any]:
    value = dict(delta or {})
    if not value:
        return {}
    categories = (
        "preference_changes", "confidence_changes", "constraint_changes",
        "new_constraints", "removed_constraints", "new_conflicts",
        "resolved_conflicts", "new_questions", "resolved_questions",
        "new_internal_conflicts", "resolved_internal_conflicts",
        "reframed_internal_conflicts",
    )
    return {
        "from_cycle": int(value.get("from_cycle", 0) or 0),
        "to_cycle": int(value.get("to_cycle", 0) or 0),
        "change_counts": {
            category: len(value.get(category, []) or [])
            for category in categories
            if value.get(category, [])
        },
    }


@dataclass(slots=True)
class WorkspaceBroadcast:
    constraint: str = "OPEN_DELIBERATION"
    intent: str = "identify_a_defensible_action"
    salient_specialist: str = ""
    salient_action: str = ""
    salient_claim: str = ""
    # INVESTIGATIVE vs GOVERNING_CANDIDATE: attention authority is not decision
    # authority. A provisional claim may win the broadcast without becoming the
    # final justificatory rule.
    broadcast_authority: str = ""
    adjudication_status: str = ""
    urgency: float = 0.5
    danger_probability: float = 0.5
    unresolved: str = "NONE"
    contingency_question: str = ""
    contingency_synthesis_action: str = ""
    contingency_failure_condition: str = ""
    contingency_predicate: str = ""
    contingency_failure_truth: bool = False
    contingency_fallback_actions: tuple[str, ...] = ()
    audit_variable: dict[str, Any] = field(default_factory=dict)
    reformulation_context: str = ""
    branch_kind: str = "BASE"
    branch_origin_action: str = ""
    branch_condition: str = ""
    branch_fallback: str = ""
    reversal_challenge: str = ""
    # Parallel, non-evidentiary questions for the next ordinary cycle.
    challenge_agenda: tuple[dict[str, Any], ...] = ()
    # Exact ledger propositions receiving attention in the next cycle. This is
    # an address, not evidence and never changes proposition authority.
    focus_proposition_ids: tuple[str, ...] = ()
    problem_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.constraint = self.constraint.strip().upper()[:48] or "OPEN_DELIBERATION"
        self.intent = self.intent.strip().lower()[:80] or "identify_a_defensible_action"
        self.salient_specialist = self.salient_specialist.strip()[:48]
        self.salient_action = " ".join(self.salient_action.split())[:180]
        self.salient_claim = " ".join(self.salient_claim.split())[:240]
        authority = self.broadcast_authority.strip().upper()
        self.broadcast_authority = (
            authority if authority in {"", "GOVERNING_CANDIDATE", "INVESTIGATIVE", "NONE"}
            else ""
        )
        adjudication = self.adjudication_status.strip().upper()
        from .specialist_authority import normalize_specialist_status
        if adjudication in {"", "NOT_APPLICABLE"}:
            self.adjudication_status = ""
        elif adjudication in {
            "ADJUDICATED_SUPPORTS",
            "PROVISIONAL_LEANING",
            "CONFLICTED_NO_LEANING",
            "CONTESTED_NO_LEANING",
            "ADJUDICATION_INCOMPLETE",
            "SUPPORTS",
            "CONDITIONAL_SUPPORTS",
        }:
            self.adjudication_status = normalize_specialist_status(adjudication)
        else:
            self.adjudication_status = ""
        from .uncertainty_types import normalize_unresolved_marker
        self.unresolved = normalize_unresolved_marker(self.unresolved)[:48] or "NONE"
        self.contingency_question = " ".join(self.contingency_question.split())[:240]
        self.contingency_synthesis_action = " ".join(
            self.contingency_synthesis_action.split()
        )[:120]
        self.contingency_failure_condition = " ".join(
            self.contingency_failure_condition.split()
        )[:180]
        self.contingency_predicate = " ".join(
            self.contingency_predicate.split()
        )[:160]
        self.contingency_fallback_actions = tuple(
            " ".join(str(action).split())[:120]
            for action in self.contingency_fallback_actions[:2]
            if " ".join(str(action).split())
        )
        self.audit_variable = dict(self.audit_variable or {})
        self.reformulation_context = " ".join(self.reformulation_context.split())[:600]
        self.branch_kind = self.branch_kind.strip().upper()[:32] or "BASE"
        self.branch_origin_action = " ".join(self.branch_origin_action.split())[:120]
        self.branch_condition = " ".join(self.branch_condition.split())[:240]
        self.branch_fallback = " ".join(self.branch_fallback.split())[:180]
        self.reversal_challenge = " ".join(self.reversal_challenge.split())[:360]
        self.challenge_agenda = tuple(
            dict(item) for item in self.challenge_agenda
            if isinstance(item, dict)
            and str(item.get("issue_id", "")).startswith("CHALLENGE:")
        )[:8]
        self.focus_proposition_ids = tuple(dict.fromkeys(
            str(value).strip() for value in self.focus_proposition_ids
            if str(value).strip()
        ))[:6]
        self.problem_state = dict(self.problem_state or {})
        self.urgency = clamp(self.urgency)
        self.danger_probability = clamp(self.danger_probability)

    def compact(self) -> str:
        challenge_projection = _balanced_challenge_projection(
            self.challenge_agenda
        )
        state_projection = _balanced_problem_state_projection(
            self.problem_state
        )
        delta_projection = _problem_delta_projection(
            self.problem_state.get("problem_delta", {})
            if self.problem_state else {}
        )
        return (
            f"constraint={self.constraint}; intent={self.intent}; "
            f"salient={self.salient_specialist or 'NONE'}:"
            f"{self.salient_action or 'NONE'}; "
            f"claim={self.salient_claim or 'NONE'}; "
            f"urgency={self.urgency:.2f}; danger={self.danger_probability:.2f}; "
            f"unresolved={self.unresolved}; "
            f"contingency={self.contingency_question or 'NONE'}; "
            f"failed_synthesis={self.contingency_synthesis_action or 'NONE'}; "
            f"failure_condition={self.contingency_failure_condition or 'NONE'}; "
            f"typed_failure=NOT({self.contingency_predicate or 'NONE'}); "
            f"typed_fallbacks={list(self.contingency_fallback_actions) or 'NONE'}; "
            f"audit_variable={json.dumps(self.audit_variable, sort_keys=True)[:1200] if self.audit_variable else 'NONE'}; "
            f"reformulation={self.reformulation_context or 'NONE'}; "
            f"branch={self.branch_kind}; origin={self.branch_origin_action or 'NONE'}; "
            f"condition={self.branch_condition or 'NONE'}; "
            f"fallback={self.branch_fallback or 'NONE'}"
            f"; reversal_challenge={self.reversal_challenge or 'NONE'}; "
            f"challenge_agenda={json.dumps(challenge_projection, sort_keys=True) if challenge_projection else 'NONE'}; "
            f"focus_propositions={list(self.focus_proposition_ids) or 'NONE'}; "
            f"problem_delta={json.dumps(delta_projection, sort_keys=True) if delta_projection else 'NONE'}; "
            f"problem_state={json.dumps(state_projection, sort_keys=True) if state_projection else 'NONE'}"
        )

    def trace_summary(self) -> str:
        claim = " ".join(self.salient_claim.split()).strip()
        if len(claim) > 120:
            claim = claim[:117].rstrip(" ,;:-") + "..."
        if not claim:
            claim = "NONE"
        salient = f"{self.salient_specialist or 'NONE'}:{self.salient_action or 'NONE'}"
        # The opening cycle has no salient position by design. Say so, rather
        # than rendering identically to a cycle that lost one.
        frame = ""
        if not self.salient_specialist and str(
            self.problem_state.get("state_role", "")
        ) == "OPENING_PROBLEM_FRAME":
            frame = (
                f" | frame=OPENING_PROBLEM_FRAME"
                f"(clauses={len(self.problem_state.get('scenario_clauses', []))},"
                f"identities={len(self.problem_state.get('grounded_identities', []))})"
            )
        return (
            f"{self.constraint} | salient={salient} | claim={claim} | "
            f"authority={self.broadcast_authority or 'NONE'} | "
            f"unresolved={self.unresolved} | "
            f"focus_propositions={list(self.focus_proposition_ids) or 'NONE'}{frame}"
        )


@dataclass(slots=True)
class ProposalFrameworkReview:
    proposal_id: str
    specialist: str
    framework_status: str
    framework_reason: str
    predicted_consequences: list[dict[str, Any]] = field(default_factory=list)
    feasibility_concerns: list[str] = field(default_factory=list)
    required_conditions: list[str] = field(default_factory=list)
    framework_retained: bool = True
    valid: bool = True
    validation_errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.proposal_id = self.proposal_id.strip().upper()[:24]
        self.specialist = self.specialist.strip()[:32]
        status = self.framework_status.strip().upper()
        self.framework_status = status if status in {
            "SUPPORTS", "QUALIFIES", "OPPOSES", "UNDERDETERMINED",
        } else "UNDERDETERMINED"
        self.framework_reason = " ".join(self.framework_reason.split())[:240]
        self.predicted_consequences = [
            dict(item) for item in self.predicted_consequences
            if isinstance(item, dict)
        ][:8]
        self.feasibility_concerns = list(dict.fromkeys(
            " ".join(str(item).split())[:160]
            for item in self.feasibility_concerns if " ".join(str(item).split())
        ))[:6]
        self.required_conditions = list(dict.fromkeys(
            " ".join(str(item).split())[:160]
            for item in self.required_conditions if " ".join(str(item).split())
        ))[:6]
        self.validation_errors = list(dict.fromkeys(
            " ".join(str(item).split())[:180]
            for item in self.validation_errors if " ".join(str(item).split())
        ))[:8]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateChunk:
    specialist: str
    constraint: str
    action_scores: dict[str, float]
    surprise: float
    friction: float
    confidence: float
    unresolved: str = "NONE"
    rationale: str = ""
    salience: float = 0.0
    tension_engagement: float = 0.0
    tension_target_keys: list[str] = field(default_factory=list)
    schema_valid: bool = True
    delegate_status: str = "VALID"
    error_type: str = "NONE"
    # Unexpected implementation failures are kept distinct from malformed
    # delegate judgments.  `error_type` remains the broad pipeline category;
    # these fields identify the concrete exception and evaluation stage.
    exception_type: str = "NONE"
    failure_stage: str = "NONE"
    validation_errors: list[str] = field(default_factory=list)
    recommended_action: str = ""
    baseline_action: str = ""
    baseline_status: str = "UNAVAILABLE"
    baseline_condition: str = ""
    baseline_preferred_extension: str = ""
    testimony_alignment: str = "UNCLEAR"
    previous_action: str = ""
    position_changed: bool = False
    change_justification: str = ""
    conformity_penalty: float = 0.0
    previous_confidence: float = 0.0
    confidence_drift: float = 0.0
    confidence_drift_penalty: float = 0.0
    assumption_status: str = "NOT_AUDITED"
    unsupported_assumption: str = ""
    reversal_condition: str = ""
    boundary_position: str = "NOT_TESTED"
    decisive_axis: str = ""
    boundary_switch_condition: str = ""
    evidence_basis: str = "STATED_FACTS"
    speculative_claim: str = ""
    evidence_calibration_tier: str = "NOT_APPLICABLE"
    evidence_calibration_reason: str = ""
    evidence_direction_retention: float = 1.0
    # Proposition-level factual dependencies. IDs are system-owned ledger keys;
    # agents may cite them but cannot assign or promote their epistemic status.
    supporting_proposition_ids: list[str] = field(default_factory=list)
    decision_critical_proposition_ids: list[str] = field(default_factory=list)
    weakest_decision_critical_status: str = "ESTABLISHED"
    decision_critical_dependency_claims: list[str] = field(default_factory=list)
    # Agent-declared inventory of every material empirical premise. A premise
    # either copies an authoritative proposition exactly or is reclassified by
    # the system as a hypothesis.
    material_empirical_claims: list[dict[str, Any]] = field(default_factory=list)
    epistemic_binding_notes: list[str] = field(default_factory=list)
    side_premise_audit_status: str = "NOT_RUN"
    side_premise_audit_findings: list[dict[str, Any]] = field(default_factory=list)
    coercion_tag: str = "NONE"
    coercion_surcharge: float = 0.0
    visibility_response: str = "NOT_TESTED"
    visibility_justification: str = ""
    visibility_harm_revision: str = "NONE"
    visibility_magnitude_status: str = "NOT_APPLICABLE"
    visibility_magnitude_overreach: bool = False
    landscape_cases: dict[str, str] = field(default_factory=dict)
    landscape_decisive_axis: str = ""
    landscape_tiebreaker: str = ""
    landscape_tiebreaker_failure: str = ""
    landscape_search_complete: bool = False
    landscape_search_attempted: bool = False
    landscape_semantic_valid: bool = True
    landscape_validation_errors: list[str] = field(default_factory=list)
    independence_bonus: float = 0.0
    # Preference is how decisively this framework ranks the actions. Epistemic
    # confidence is how likely that ranking is to survive new facts/scrutiny.
    # `confidence` remains a compatibility alias for epistemic_confidence.
    preference_strength: float = -1.0
    # The gap the delegate actually reported, before system-owned damping.
    # Drift detection compares this value across cycles so recovery from an
    # audit or claim-damping intervention is not mistaken for identity drift.
    reported_preference_strength: float = -1.0
    epistemic_confidence: float = -1.0
    previous_preference_strength: float = 0.0
    preference_drift: float = 0.0
    preference_drift_penalty: float = 0.0
    preference_shift_reason_strength: float = 0.0
    decision_rule: str = ""
    # Authority typing: policy weight, investigative attention, and governing
    # eligibility are independent. A provisional leaning may interrupt the
    # workspace without supplying a final justificatory rule.
    adjudication_status: str = "SUPPORTS"
    broadcast_authority: str = "GOVERNING_CANDIDATE"
    governing_eligible: bool = True
    policy_weight_factor: float = 1.0
    # System-owned admission result for the framework's directional vote.
    # Specialists cannot self-certify these fields.
    framework_vote_integrity_required: bool = False
    framework_vote_status: str = "NOT_APPLICABLE"
    framework_vote_reason: str = ""
    framework_ledger_kind: str = ""
    framework_ledger_status: str = ""
    derived_claim_validation_status: str = "NOT_RUN"
    derived_claim_validation_errors: list[str] = field(default_factory=list)
    investigative_claim: str = ""
    investigative_priority: float = 0.0
    reopen_eligible: bool = False
    reopen_reason: str = ""
    reopen_question_key: str = ""
    factual_reversal_threshold: str = "NONE"
    normative_reversal_threshold: str = "NONE"
    reversal_review_response: str = "NOT_TESTED"
    reversal_review_justification: str = ""
    revised_reversal_condition: str = ""
    reversal_review_valid: bool = True
    reversal_review_error: str = ""
    contingency_choice: str = ""
    contingency_justification: str = ""
    contingency_response_valid: bool = True
    contingency_response_error: str = ""
    audit_variable: dict[str, Any] = field(default_factory=dict)
    audit_internal_effect: str = "UNRESOLVED"
    audit_participation: str = "NOT_TESTED"
    audit_framework_explanation: str = ""
    # Response to this framework's assigned item in the shared challenge agenda.
    # The embedded issue ID preserves who authored and who was targeted by it.
    challenge_response: dict[str, Any] = field(default_factory=dict)
    graph_update_proposal: dict[str, Any] = field(default_factory=dict)
    expected_value_estimates: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Passive construct-validity measurements. These fields are serialized for
    # evaluation but are intentionally absent from policy, salience, drift, and
    # stopping calculations.
    selection_status: str = "SELECTED"
    action_admissibility: dict[str, str] = field(default_factory=dict)
    comparison_complete: bool = True
    evidence_sufficient_for_action: bool = True
    interim_action: str = ""
    workspace_proposition_response: str = "NOT_APPLICABLE"
    workspace_reasoning_effect: str = "NONE"
    framework_application: str = ""
    framework_constraint_retained: bool = True
    self_reported_broadcast_dependence: str = "NONE"
    framework_retention_status: str = "NOT_MEASURED"
    # Transaction visibility: a rejected recurrent proposal does not disappear.
    # The operative candidate exposes both the attempted framework state and the
    # last committed state, plus which current-cycle components survived rollback.
    proposed_framework_state: dict[str, Any] = field(default_factory=dict)
    committed_framework_state: dict[str, Any] = field(default_factory=dict)
    # System-owned snapshot of graph-committed native ledger records. Unlike
    # committed_framework_state, this cannot contain a merely submitted or
    # first-state-admitted proposal.
    committed_native_ledger: dict[str, Any] = field(default_factory=dict)
    preserved_current_cycle_components: list[str] = field(default_factory=list)
    # Framework-local tensions are observations about a specialist's own
    # reasoning, not shared world facts or cross-framework priority rules.
    framework_internal_conflicts: list[str] = field(default_factory=list)
    framework_specific_open_questions: list[str] = field(default_factory=list)
    # Non-voting, framework-attributed refinements recovered from an update
    # that was not allowed to replace the operative framework state.
    framework_insights: list[dict[str, Any]] = field(default_factory=list)
    # Framework-specific, action-indexed construct state.  Unlike the short
    # rationale, this map records how the assigned framework evaluates every
    # option and survives presentation-label changes.
    framework_action_map: dict[str, str] = field(default_factory=dict)
    framework_numerical_role: str = "NOT_APPLICABLE"
    framework_numerical_justification: str = ""
    framework_grounding_penalty: float = 0.0
    framework_validation_errors: list[str] = field(default_factory=list)
    utilitarian_consequence_table: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    utilitarian_decision_depends_on_unknown: bool = False
    utilitarian_missing_comparison: str = ""
    utilitarian_incommensurable_remainder: list[str] = field(default_factory=list)
    utilitarian_ledger_proposal: dict[str, Any] = field(default_factory=dict)
    rawls_position_proposal: dict[str, Any] = field(default_factory=dict)
    deontological_ledger_proposal: dict[str, Any] = field(default_factory=dict)
    virtue_character_proposal: dict[str, Any] = field(default_factory=dict)
    care_ledger_proposal: dict[str, Any] = field(default_factory=dict)
    care_relational_map: dict[str, str] = field(default_factory=dict)
    care_numerical_role: str = "NOT_APPLICABLE"
    care_numerical_justification: str = ""
    care_grounding_penalty: float = 0.0
    proposal_review: ProposalFrameworkReview | None = None

    def __post_init__(self) -> None:
        self.specialist = self.specialist.strip()[:32]
        self.constraint = self.constraint.strip().upper()[:48] or "UNSPECIFIED"
        self.exception_type = (
            " ".join(str(self.exception_type).split())[:80] or "NONE"
        )
        self.failure_stage = (
            " ".join(str(self.failure_stage).split()).upper()[:80] or "NONE"
        )
        self.framework_internal_conflicts = list(dict.fromkeys(
            " ".join(str(item).split())[:180]
            for item in self.framework_internal_conflicts
            if " ".join(str(item).split())
        ))[:3]
        self.framework_specific_open_questions = list(dict.fromkeys(
            " ".join(str(item).split())[:180]
            for item in self.framework_specific_open_questions
            if " ".join(str(item).split())
        ))[:3]
        self.framework_insights = [
            dict(item) for item in self.framework_insights
            if isinstance(item, dict)
            and str(item.get("proposition", "")).strip()
        ][:6]
        self.challenge_response = dict(self.challenge_response or {})
        self.proposed_framework_state = dict(self.proposed_framework_state or {})
        self.committed_framework_state = dict(self.committed_framework_state or {})
        self.committed_native_ledger = dict(self.committed_native_ledger or {})
        self.preserved_current_cycle_components = list(dict.fromkeys(
            str(value).strip()[:64]
            for value in self.preserved_current_cycle_components
            if str(value).strip()
        ))[:24]
        self.tension_engagement = clamp(self.tension_engagement)
        self.tension_target_keys = list(dict.fromkeys(
            str(key).strip()[:64] for key in self.tension_target_keys
            if str(key).strip()
        ))[:8]
        from .uncertainty_types import normalize_unresolved_marker
        self.unresolved = normalize_unresolved_marker(self.unresolved)[:48] or "NONE"
        status = self.delegate_status.strip().upper()
        self.delegate_status = status if status in {
            "VALID", "MODEL_ERROR", "SCHEMA_ERROR", "SEMANTIC_VALIDATION_ERROR",
            "SPECIALIST_INTERNAL_ERROR",
        } else ("VALID" if self.schema_valid else "SEMANTIC_VALIDATION_ERROR")
        error_type = self.error_type.strip().upper()
        self.error_type = error_type if error_type in {
            "NONE", "MODEL_ERROR", "SCHEMA_REJECTION", "SEMANTIC_VALIDATION_ERROR",
            "SPECIALIST_INTERNAL_ERROR",
        } else "NONE"
        if not self.schema_valid and self.delegate_status == "VALID":
            self.delegate_status = "SEMANTIC_VALIDATION_ERROR"
        if not self.schema_valid:
            # Execution failure is not a moral or epistemic constraint.
            self.constraint = "NONE"
        if self.schema_valid:
            self.delegate_status = "VALID"
            self.error_type = "NONE"
        self.rationale = " ".join(self.rationale.split())[:180]
        baseline_status = self.baseline_status.strip().upper()
        self.baseline_status = (
            baseline_status
            if baseline_status in {
                "DIRECT", "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
                "OUTSIDE_ACTION_SET", "OUTSIDE_ACTION_SET_WITH_FALLBACK",
                "NO_POSITION", "PARSE_FAILURE", "UNAVAILABLE",
            }
            else "PARSE_FAILURE"
        )
        self.baseline_condition = " ".join(self.baseline_condition.split())[:240]
        self.baseline_preferred_extension = " ".join(
            self.baseline_preferred_extension.split()
        )[:240]
        self.surprise = clamp(self.surprise)
        self.friction = clamp(self.friction)
        if self.preference_strength < 0:
            ordered = sorted(self.action_scores.values(), reverse=True)
            self.preference_strength = (
                ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
            ) if ordered else 0.0
        if self.epistemic_confidence < 0:
            self.epistemic_confidence = self.confidence
        self.preference_strength = clamp(self.preference_strength)
        if self.reported_preference_strength < 0:
            self.reported_preference_strength = self.preference_strength
        self.reported_preference_strength = clamp(self.reported_preference_strength)
        self.epistemic_confidence = clamp(self.epistemic_confidence)
        self.previous_preference_strength = clamp(self.previous_preference_strength)
        self.preference_drift = max(-1.0, min(1.0, float(self.preference_drift)))
        self.preference_drift_penalty = clamp(self.preference_drift_penalty)
        self.preference_shift_reason_strength = clamp(self.preference_shift_reason_strength)
        self.confidence = self.epistemic_confidence
        self.conformity_penalty = clamp(self.conformity_penalty)
        self.previous_confidence = clamp(self.previous_confidence)
        self.confidence_drift = max(-1.0, min(1.0, float(self.confidence_drift)))
        self.confidence_drift_penalty = clamp(self.confidence_drift_penalty)
        self.change_justification = " ".join(self.change_justification.split())[:180]
        self.assumption_status = self.assumption_status.strip().upper()[:24] or "NOT_AUDITED"
        self.unsupported_assumption = " ".join(self.unsupported_assumption.split())[:180]
        self.reversal_condition = " ".join(self.reversal_condition.split())[:180]
        participation = self.audit_participation.strip().upper()
        self.audit_participation = participation if participation in {
            "NOT_TESTED", "RELEVANT", "IRRELEVANT", "TRANSLATED", "CONTESTED",
            "REVERSAL_RELEVANT", "UNRESOLVED",
        } else "NOT_TESTED"
        self.audit_framework_explanation = " ".join(
            self.audit_framework_explanation.split()
        )[:240]
        self.boundary_position = self.boundary_position.strip().upper()[:16] or "NOT_TESTED"
        self.decisive_axis = " ".join(self.decisive_axis.split())[:100]
        self.boundary_switch_condition = " ".join(self.boundary_switch_condition.split())[:180]
        self.evidence_basis = self.evidence_basis.strip().upper()[:24] or "STATED_FACTS"
        self.speculative_claim = " ".join(self.speculative_claim.split())[:180]
        self.evidence_calibration_tier = (
            self.evidence_calibration_tier.strip().upper()[:32] or "NOT_APPLICABLE"
        )
        self.evidence_calibration_reason = " ".join(
            self.evidence_calibration_reason.split()
        )[:180]
        self.evidence_direction_retention = clamp(self.evidence_direction_retention)
        self.supporting_proposition_ids = list(dict.fromkeys(
            str(value).strip() for value in self.supporting_proposition_ids
            if str(value).strip()
        ))[:24]
        self.decision_critical_proposition_ids = list(dict.fromkeys(
            str(value).strip() for value in self.decision_critical_proposition_ids
            if str(value).strip()
        ))[:12]
        status = str(self.weakest_decision_critical_status).strip().upper()
        self.weakest_decision_critical_status = (
            status if status in {
                "ESTABLISHED", "DERIVED", "UNRESOLVED", "HYPOTHETICAL", "REJECTED",
            } else "ESTABLISHED"
        )
        self.decision_critical_dependency_claims = list(dict.fromkeys(
            " ".join(str(value).split())[:240]
            for value in self.decision_critical_dependency_claims
            if " ".join(str(value).split())
        ))[:4]
        self.material_empirical_claims = [
            dict(value) for value in self.material_empirical_claims
            if isinstance(value, dict)
        ][:12]
        self.epistemic_binding_notes = list(dict.fromkeys(
            " ".join(str(value).split())[:240]
            for value in self.epistemic_binding_notes
            if " ".join(str(value).split())
        ))[:12]
        audit_status = str(self.side_premise_audit_status).strip().upper()
        self.side_premise_audit_status = (
            audit_status if audit_status in {
                "NOT_RUN", "PASSED", "FINDINGS", "UNAVAILABLE",
            } else "NOT_RUN"
        )
        self.side_premise_audit_findings = [
            dict(value) for value in self.side_premise_audit_findings
            if isinstance(value, dict)
        ][:12]
        self.coercion_tag = self.coercion_tag.strip().upper()[:32] or "NONE"
        self.coercion_surcharge = clamp(self.coercion_surcharge)
        response = self.visibility_response.strip().upper()
        self.visibility_response = (
            response if response in {"NOT_TESTED", "ACCEPT", "QUALIFY", "REJECT"}
            else "NOT_TESTED"
        )
        self.visibility_justification = " ".join(
            self.visibility_justification.split()
        )[:180]
        revision = self.visibility_harm_revision.strip().upper()
        self.visibility_harm_revision = (
            revision if revision in {"NONE", "UPWARD", "DOWNWARD", "UNCHANGED"}
            else "NONE"
        )
        magnitude = self.visibility_magnitude_status.strip().upper()
        self.visibility_magnitude_status = (
            magnitude
            if magnitude in {"NOT_APPLICABLE", "UNKNOWN", "GROUNDED_BOUNDED"}
            else "NOT_APPLICABLE"
        )
        self.landscape_cases = {
            str(action): " ".join(str(reason).split())[:180]
            for action, reason in self.landscape_cases.items()
            if str(action).strip() and " ".join(str(reason).split())
        }
        self.landscape_decisive_axis = " ".join(
            self.landscape_decisive_axis.split()
        )[:120]
        self.landscape_tiebreaker = " ".join(self.landscape_tiebreaker.split())[:180]
        self.landscape_tiebreaker_failure = " ".join(
            self.landscape_tiebreaker_failure.split()
        )[:180]
        if (
            self.landscape_cases
            or self.landscape_decisive_axis
            or self.landscape_tiebreaker
            or self.landscape_tiebreaker_failure
        ):
            self.landscape_search_attempted = True
        self.landscape_validation_errors = [
            " ".join(str(error).split())[:180]
            for error in self.landscape_validation_errors
            if " ".join(str(error).split())
        ][:8]
        self.independence_bonus = clamp(self.independence_bonus)
        self.decision_rule = " ".join(self.decision_rule.split())[:180]
        adjudication = self.adjudication_status.strip().upper()
        from .specialist_authority import normalize_specialist_status
        if adjudication in {
            "NOT_APPLICABLE",
            "ADJUDICATED_SUPPORTS",
            "PROVISIONAL_LEANING",
            "CONFLICTED_NO_LEANING",
            "CONTESTED_NO_LEANING",
            "ADJUDICATION_INCOMPLETE",
            "SUPPORTS",
            "CONDITIONAL_SUPPORTS",
            "",
        }:
            # Empty / NOT_APPLICABLE stay untyped until apply_specialist_authority.
            self.adjudication_status = (
                "SUPPORTS"
                if adjudication in {"", "NOT_APPLICABLE"}
                else normalize_specialist_status(adjudication)
            )
        else:
            self.adjudication_status = "SUPPORTS"
        authority = self.broadcast_authority.strip().upper()
        self.broadcast_authority = (
            authority if authority in {
                "GOVERNING_CANDIDATE", "INVESTIGATIVE", "NONE",
            }
            else "GOVERNING_CANDIDATE"
        )
        self.governing_eligible = bool(self.governing_eligible)
        if self.adjudication_status in {
            "PROVISIONAL_LEANING", "CONTESTED_NO_LEANING",
        }:
            self.governing_eligible = False
            if self.broadcast_authority == "GOVERNING_CANDIDATE":
                self.broadcast_authority = "INVESTIGATIVE"
        self.policy_weight_factor = clamp(float(self.policy_weight_factor))
        self.framework_vote_integrity_required = bool(
            self.framework_vote_integrity_required
        )
        vote_status = str(self.framework_vote_status).strip().upper()
        self.framework_vote_status = (
            vote_status
            if vote_status in {"NOT_APPLICABLE", "FULL", "ATTENUATED", "ABSTAIN"}
            else "NOT_APPLICABLE"
        )
        self.framework_vote_reason = " ".join(
            str(self.framework_vote_reason).split()
        )[:240]
        self.framework_ledger_kind = str(
            self.framework_ledger_kind
        ).strip().upper()[:64]
        self.framework_ledger_status = str(
            self.framework_ledger_status
        ).strip().upper()[:64]
        derivation_status = str(self.derived_claim_validation_status).strip().upper()
        self.derived_claim_validation_status = (
            derivation_status
            if derivation_status in {"NOT_RUN", "PASSED", "QUARANTINED"}
            else "NOT_RUN"
        )
        self.derived_claim_validation_errors = list(dict.fromkeys(
            " ".join(str(value).split())[:240]
            for value in self.derived_claim_validation_errors
            if " ".join(str(value).split())
        ))[:12]
        self.investigative_claim = " ".join(self.investigative_claim.split())[:240]
        self.investigative_priority = clamp(float(self.investigative_priority))
        self.reopen_eligible = bool(self.reopen_eligible)
        self.reopen_reason = " ".join(str(self.reopen_reason).split())[:240]
        self.reopen_question_key = " ".join(str(self.reopen_question_key).split())[:120]
        self.factual_reversal_threshold = (
            " ".join(self.factual_reversal_threshold.split())[:180] or "NONE"
        )
        self.normative_reversal_threshold = (
            " ".join(self.normative_reversal_threshold.split())[:180] or "NONE"
        )
        response = self.reversal_review_response.strip().upper()
        self.reversal_review_response = (
            response if response in {"NOT_TESTED", "ACCEPT", "REVISE", "REJECT"}
            else "NOT_TESTED"
        )
        self.reversal_review_justification = " ".join(
            self.reversal_review_justification.split()
        )[:180]
        self.revised_reversal_condition = " ".join(
            self.revised_reversal_condition.split()
        )[:180]
        self.reversal_review_error = " ".join(
            self.reversal_review_error.split()
        )[:180]
        self.contingency_choice = " ".join(self.contingency_choice.split())[:120]
        self.contingency_justification = " ".join(
            self.contingency_justification.split()
        )[:180]
        self.contingency_response_error = " ".join(
            self.contingency_response_error.split()
        )[:180]
        self.audit_variable = dict(self.audit_variable or {})
        effect = self.audit_internal_effect.strip().upper()[:16]
        self.audit_internal_effect = (
            effect if effect in {"NO_CHANGE", "WEAKENS", "REVERSES", "UNRESOLVED"}
            else "UNRESOLVED"
        )
        self.graph_update_proposal = dict(self.graph_update_proposal or {})
        self.expected_value_estimates = dict(self.expected_value_estimates or {})
        selection = self.selection_status.strip().upper()
        self.selection_status = (
            selection if selection in {"SELECTED", "PROVISIONAL", "UNSELECTED"}
            else "SELECTED"
        )
        allowed_admissibility = {"REQUIRED", "PERMISSIBLE", "REJECTED", "UNASSESSED"}
        self.action_admissibility = {
            str(action): str(status).strip().upper()
            for action, status in self.action_admissibility.items()
            if str(status).strip().upper() in allowed_admissibility
        }
        self.interim_action = " ".join(self.interim_action.split())[:120]
        response = self.workspace_proposition_response.strip().upper()
        self.workspace_proposition_response = (
            response if response in {"NOT_APPLICABLE", "ACCEPT", "QUALIFY", "REJECT"}
            else "NOT_APPLICABLE"
        )
        effect = self.workspace_reasoning_effect.strip().upper()
        self.workspace_reasoning_effect = (
            effect if effect in {"NONE", "FACTUAL", "NORMATIVE", "BOTH"} else "NONE"
        )
        self.framework_application = " ".join(self.framework_application.split())[:180]
        dependence = self.self_reported_broadcast_dependence.strip().upper()
        self.self_reported_broadcast_dependence = (
            dependence if dependence in {"NONE", "LOW", "MEDIUM", "HIGH"} else "NONE"
        )
        retention = self.framework_retention_status.strip().upper()
        self.framework_retention_status = (
            retention if retention in {
                "NOT_MEASURED", "PRESERVED", "UNCLEAR", "LOST",
                # The delegate proposed an unexplained change, but the
                # transactional ledger kept the prior authoritative state.
                "UPDATE_REJECTED",
                # The graph accepted only the uncertainty-preserving form of
                # the proposal. This is not equivalent to identity loss.
                "COMMITTED_WITH_UNCERTAINTY",
                "REVERSAL", "REFINEMENT", "SPECIALIZATION", "RESOLUTION",
                "NEW_ACTION_GROUNDING",
            }
            else "NOT_MEASURED"
        )
        self.framework_action_map = {
            str(action): " ".join(str(reason).split())[:180]
            for action, reason in self.framework_action_map.items()
            if str(action).strip() and " ".join(str(reason).split())
        }
        framework_role = str(self.framework_numerical_role).strip().upper()
        self.framework_numerical_role = (
            framework_role
            if framework_role in {
                "NOT_APPLICABLE", "DECISIVE", "SECONDARY", "IRRELEVANT",
            }
            else "NOT_APPLICABLE"
        )
        self.framework_numerical_justification = " ".join(
            str(self.framework_numerical_justification).split()
        )[:180]
        self.framework_grounding_penalty = clamp(self.framework_grounding_penalty)
        self.framework_validation_errors = [
            " ".join(str(error).split())[:180]
            for error in self.framework_validation_errors
            if " ".join(str(error).split())
        ][:8]
        self.utilitarian_consequence_table = {
            str(action): [dict(row) for row in rows if isinstance(row, dict)][:12]
            for action, rows in self.utilitarian_consequence_table.items()
            if str(action).strip() and isinstance(rows, list)
        }
        self.utilitarian_decision_depends_on_unknown = bool(
            self.utilitarian_decision_depends_on_unknown
        )
        self.utilitarian_missing_comparison = " ".join(
            str(self.utilitarian_missing_comparison).split()
        )[:180]
        self.utilitarian_incommensurable_remainder = [
            " ".join(str(item).split())[:160]
            for item in self.utilitarian_incommensurable_remainder
            if " ".join(str(item).split())
        ][:8]
        self.utilitarian_ledger_proposal = dict(self.utilitarian_ledger_proposal or {})
        self.rawls_position_proposal = dict(self.rawls_position_proposal or {})
        self.deontological_ledger_proposal = dict(
            self.deontological_ledger_proposal or {}
        )
        self.virtue_character_proposal = dict(self.virtue_character_proposal or {})
        self.care_ledger_proposal = dict(self.care_ledger_proposal or {})
        self.care_relational_map = {
            str(action): " ".join(str(reason).split())[:160]
            for action, reason in self.care_relational_map.items()
            if str(action).strip() and " ".join(str(reason).split())
        }
        care_role = str(self.care_numerical_role).strip().upper()
        self.care_numerical_role = (
            care_role
            if care_role in {"NOT_APPLICABLE", "DECISIVE", "SECONDARY", "IRRELEVANT"}
            else "NOT_APPLICABLE"
        )
        self.care_numerical_justification = " ".join(
            str(self.care_numerical_justification).split()
        )[:180]
        self.care_grounding_penalty = clamp(self.care_grounding_penalty)
        self.action_scores = {str(k): clamp(v) for k, v in self.action_scores.items()}


@dataclass(slots=True)
class CycleRecord:
    cycle: int
    broadcast: WorkspaceBroadcast
    candidates: list[CandidateChunk]
    # Deprecated positional alias of broadcast_focus (migration cycle).
    winner: CandidateChunk | None
    dissent: CandidateChunk | None
    policy: dict[str, float]
    entropy: float
    stable_cycles: int
    elapsed_seconds: float
    received_broadcast: WorkspaceBroadcast | None = None
    is_hypothetical: bool = False
    execution_status: str = "VALID"
    system_error: str = "NONE"
    # Explicit workspace roles — prefer these over winner.
    policy_leader: str = ""
    governing_claim: CandidateChunk | None = None
    broadcast_focus: CandidateChunk | None = None

    def __post_init__(self) -> None:
        self.policy_leader = " ".join(str(self.policy_leader or "").split())[:240]
        if self.broadcast_focus is None:
            self.broadcast_focus = self.winner
        else:
            # New traces serialize broadcast_focus; keep winner as its alias.
            self.winner = self.broadcast_focus


@dataclass(slots=True)
class PlanningBranchEvaluation:
    cycle: int
    origin_action: str
    condition: str
    fallback: str
    selected_action: str
    confidence: float
    policy: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cycle = max(1, int(self.cycle))
        self.origin_action = " ".join(self.origin_action.split())[:240]
        self.condition = " ".join(self.condition.split())[:240]
        self.fallback = " ".join(self.fallback.split())[:240]
        self.selected_action = " ".join(self.selected_action.split())[:240]
        self.confidence = clamp(self.confidence)
        self.policy = {str(action): clamp(score) for action, score in self.policy.items()}


@dataclass(slots=True)
class SynthesisProposal:
    action: str
    grounded_in: list[str]
    addressed_constraints: list[str]
    feasibility: float
    rationale: str
    executable: bool = True
    non_evasive: bool = True
    accepted: bool = False
    rejection_reason: str = ""
    admission_status: str = "PENDING"
    admission_annotations: list[str] = field(default_factory=list)
    introduced_requirements: list[str] = field(default_factory=list)
    proposal_id: str = ""
    source_agents: list[str] = field(default_factory=list)
    predicted_consequences: list[dict[str, Any]] = field(default_factory=list)
    feasibility_status: str = "UNCERTAIN"
    grounding_status: str = "PENDING"
    framework_reviews: dict[str, dict[str, Any]] = field(default_factory=dict)
    promotion_status: str = "PROPOSED"

    def __post_init__(self) -> None:
        self.action = " ".join(self.action.split())[:240]
        self.grounded_in = list(dict.fromkeys(str(item).strip().lower() for item in self.grounded_in if str(item).strip()))[:5]
        self.addressed_constraints = list(dict.fromkeys(str(item).strip().upper() for item in self.addressed_constraints if str(item).strip()))[:5]
        self.feasibility = clamp(self.feasibility)
        self.rationale = " ".join(self.rationale.split())[:240]
        self.rejection_reason = " ".join(self.rejection_reason.split())[:200]
        self.admission_status = self.admission_status.strip().upper()[:40] or "PENDING"
        self.admission_annotations = [
            " ".join(str(item).split())[:160]
            for item in dict.fromkeys(self.admission_annotations)
            if " ".join(str(item).split())
        ][:8]
        self.introduced_requirements = list(dict.fromkeys(
            " ".join(str(item).split())[:100]
            for item in self.introduced_requirements
            if " ".join(str(item).split())
        ))[:8]
        self.proposal_id = self.proposal_id.strip().upper()[:24]
        self.source_agents = list(dict.fromkeys(
            str(agent).strip()[:48] for agent in self.source_agents
            if str(agent).strip()
        ))[:8]
        self.predicted_consequences = [
            dict(item) for item in self.predicted_consequences
            if isinstance(item, dict)
        ][:12]
        feasibility_status = self.feasibility_status.strip().upper()
        self.feasibility_status = (
            feasibility_status if feasibility_status in {
                "UNKNOWN", "UNCERTAIN", "PLAUSIBLE", "ESTABLISHED", "FAILED",
            } else "UNCERTAIN"
        )
        grounding_status = self.grounding_status.strip().upper()
        self.grounding_status = (
            grounding_status if grounding_status in {
                "PENDING", "PARTIALLY_GROUNDED", "GROUNDED", "REJECTED",
            } else "PENDING"
        )
        self.framework_reviews = {
            str(agent): dict(review) for agent, review in self.framework_reviews.items()
            if str(agent).strip() and isinstance(review, dict)
        }
        promotion_status = self.promotion_status.strip().upper()
        self.promotion_status = (
            promotion_status if promotion_status in {
                "PROPOSED", "UNDER_REVIEW", "ADMISSIBLE", "REJECTED", "PROMOTED",
            } else "PROPOSED"
        )


@dataclass(slots=True)
class FailureCondition:
    synthesis_action: str
    necessary_condition: str
    failure_condition: str
    contingency_question: str
    fallback_actions: list[str] = field(default_factory=list)
    predicate_label: str = ""
    required_truth: bool = True
    failure_truth: bool = False
    fallback_availability: dict[str, str] = field(default_factory=dict)
    fallback_availability_reasons: dict[str, str] = field(default_factory=dict)
    semantic_graph: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        self.synthesis_action = " ".join(self.synthesis_action.split())[:240]
        self.necessary_condition = " ".join(self.necessary_condition.split())[:180]
        self.failure_condition = " ".join(self.failure_condition.split())[:180]
        self.contingency_question = " ".join(self.contingency_question.split())[:240]
        self.fallback_actions = [
            " ".join(str(action).split())[:240]
            for action in self.fallback_actions[:2]
            if " ".join(str(action).split())
        ]
        self.predicate_label = " ".join(self.predicate_label.split())[:160]
        self.fallback_availability = {
            str(action_id): str(status).strip().upper()
            for action_id, status in self.fallback_availability.items()
        }
        self.fallback_availability_reasons = {
            str(action_id): " ".join(str(reason).split())[:120]
            for action_id, reason in self.fallback_availability_reasons.items()
        }
        self.semantic_graph = dict(self.semantic_graph or {})
        self.error = " ".join(self.error.split())[:200]


@dataclass(slots=True)
class SynthesisViabilityAssessment:
    """Post-review evidence that a synthesis remains worth branching on."""

    synthesis_action: str
    review_cycle: int
    valid_delegates: int
    recommendation_count: int
    admissible_count: int
    rejection_count: int
    mean_score: float
    policy_support: float
    leader_support: float
    viable: bool
    reason: str

    def __post_init__(self) -> None:
        self.synthesis_action = " ".join(self.synthesis_action.split())[:240]
        self.review_cycle = max(0, int(self.review_cycle))
        self.valid_delegates = max(0, int(self.valid_delegates))
        self.recommendation_count = max(0, int(self.recommendation_count))
        self.admissible_count = max(0, int(self.admissible_count))
        self.rejection_count = max(0, int(self.rejection_count))
        self.mean_score = clamp(self.mean_score)
        self.policy_support = clamp(self.policy_support)
        self.leader_support = clamp(self.leader_support)
        self.reason = " ".join(self.reason.split())[:240]


@dataclass(slots=True)
class ContingencyFeasibilityAssessment:
    """Independent, non-normative audit of branch fallback executability."""

    synthesis_action: str
    predicate_label: str
    fallback_statuses: dict[str, str]
    fallback_reasons: dict[str, str]
    evidence_bases: dict[str, str]
    shared_failure: bool
    valid: bool
    approved: bool
    error: str = ""

    def __post_init__(self) -> None:
        self.synthesis_action = " ".join(self.synthesis_action.split())[:240]
        self.predicate_label = " ".join(self.predicate_label.split())[:160]
        self.fallback_statuses = {
            str(key): str(value).strip().upper()
            for key, value in self.fallback_statuses.items()
        }
        self.fallback_reasons = {
            str(key): " ".join(str(value).split())[:120]
            for key, value in self.fallback_reasons.items()
        }
        self.evidence_bases = {
            str(key): str(value).strip().upper()
            for key, value in self.evidence_bases.items()
        }
        self.error = " ".join(self.error.split())[:240]


@dataclass(slots=True)
class PlanningAssessment:
    """Non-normative implementation analysis for a currently leading action."""

    target_action: str
    activation_reason: str
    feasibility: float
    necessary_condition: str
    failure_condition: str
    fallback: str
    actor_constraints: list[str] = field(default_factory=list)
    resource_constraints: list[str] = field(default_factory=list)
    strategic_forces: list[str] = field(default_factory=list)
    broadcast_worthy: bool = False
    valid: bool = True
    error: str = ""
    grounded_evidence: str = ""
    fallback_available: bool = False
    fallback_availability_reason: str = ""
    target_action_node_id: str = ""
    failure_grounding_status: str = "UNASSESSED"

    def __post_init__(self) -> None:
        self.target_action = " ".join(self.target_action.split())[:240]
        self.activation_reason = " ".join(self.activation_reason.split())[:100]
        self.feasibility = clamp(self.feasibility)
        self.necessary_condition = " ".join(self.necessary_condition.split())[:180]
        self.failure_condition = " ".join(self.failure_condition.split())[:180]
        self.fallback = " ".join(self.fallback.split())[:240]
        self.actor_constraints = self._clean_list(self.actor_constraints)
        self.resource_constraints = self._clean_list(self.resource_constraints)
        self.strategic_forces = self._clean_list(self.strategic_forces)
        self.error = " ".join(self.error.split())[:200]
        self.grounded_evidence = " ".join(self.grounded_evidence.split())[:240]
        self.fallback_availability_reason = " ".join(
            self.fallback_availability_reason.split()
        )[:180]
        self.target_action_node_id = self.target_action_node_id.strip().upper()[:20]
        grounding = self.failure_grounding_status.strip().upper()
        self.failure_grounding_status = (
            grounding if grounding in {
                "UNASSESSED",
                "GROUNDED_FAILURE_CONDITION",
                "MECHANISM_DERIVED_FAILURE_CONDITION",
                "PLAUSIBLE_HYPOTHETICAL_FAILURE_CONDITION",
                "REJECTED_UNGROUNDED_FAILURE_CONDITION",
            } else "UNASSESSED"
        )

    @staticmethod
    def _clean_list(values: list[str]) -> list[str]:
        return list(dict.fromkeys(
            " ".join(str(value).split())[:120]
            for value in values
            if " ".join(str(value).split())
        ))[:5]


@dataclass(slots=True)
class WorkspaceAccessDecision:
    """Traceable decision by the capacity-limited workspace access gate."""

    cycle: int
    content_type: str
    admitted: bool
    signals: list[str]
    question: str = ""
    rationale: str = ""
    semantic_node_id: str = ""
    semantic_node_kind: str = ""
    semantic_node_label: str = ""
    audit_variable: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cycle = max(1, int(self.cycle))
        self.content_type = self.content_type.strip().upper()[:40]
        self.signals = list(dict.fromkeys(
            str(signal).strip().lower().replace(" ", "_")[:64]
            for signal in self.signals
            if str(signal).strip()
        ))[:10]
        self.question = " ".join(self.question.split())[:300]
        self.rationale = " ".join(self.rationale.split())[:240]
        self.semantic_node_id = " ".join(self.semantic_node_id.split())[:120]
        self.semantic_node_kind = self.semantic_node_kind.strip().upper()[:24]
        self.semantic_node_label = " ".join(self.semantic_node_label.split())[:240]
        self.audit_variable = dict(self.audit_variable or {})


@dataclass(slots=True)
class VisibilityAssessment:
    """Non-voting audit of action advantages created by unequal observability."""

    low_observability: bool
    endogenous: bool
    affected_group: str
    mechanism: str
    evidence_quote: str
    action_multipliers: dict[str, float] = field(default_factory=dict)
    activated: bool = False
    valid: bool = True
    error: str = ""
    proposition: str = ""
    typed_facts: list[dict] = field(default_factory=list)
    mechanism_provenance: str = "SCENARIO_GROUNDED"

    def __post_init__(self) -> None:
        self.affected_group = " ".join(self.affected_group.split())[:120]
        self.mechanism = " ".join(self.mechanism.split())[:220]
        self.evidence_quote = " ".join(self.evidence_quote.split())[:240]
        # Visibility reduces confidence in an apparent evidential advantage; it
        # cannot veto an action or manufacture support for its alternative.
        self.action_multipliers = {
            " ".join(str(action).split())[:120]: max(0.65, min(1.0, float(value)))
            for action, value in self.action_multipliers.items()
        }
        self.error = " ".join(self.error.split())[:240]
        provenance = self.mechanism_provenance.strip().upper()
        self.mechanism_provenance = (
            provenance
            if provenance in {"SCENARIO_GROUNDED", "HYPOTHETICAL", "EXTERNAL_GENERALIZATION"}
            else "HYPOTHETICAL"
        )
        if not self.proposition and self.activated:
            penalized = [
                action for action, value in self.action_multipliers.items() if value < 0.999
            ]
            target = penalized[0] if penalized else "the affected action"
            self.proposition = (
                f"The estimated harm of {target} is downward-biased because "
                f"{self.affected_group or 'a disadvantaged population'} is less observable "
                f"through {self.mechanism}."
            )
        self.proposition = " ".join(self.proposition.split())[:320]


@dataclass(slots=True)
class AutonomyAssessment:
    """Non-voting audit of coercion and its required evidentiary basis."""

    action_tags: dict[str, str]
    catastrophic_harm_threshold: dict[str, bool]
    evidence: dict[str, str]
    voluntary_alternative: str = "NONE"
    surcharge_multiplier: float = 0.70
    activated: bool = True
    valid: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        allowed = {"NONE", "COERCIVE", "RIGHTS_INTRUSION", "COVENANT_BREACH"}
        self.action_tags = {
            " ".join(str(action).split())[:120]: (
                str(tag).strip().upper() if str(tag).strip().upper() in allowed else "NONE"
            )
            for action, tag in self.action_tags.items()
        }
        self.catastrophic_harm_threshold = {
            " ".join(str(action).split())[:120]: bool(value)
            for action, value in self.catastrophic_harm_threshold.items()
        }
        self.evidence = {
            " ".join(str(action).split())[:120]: " ".join(str(value).split())[:220]
            for action, value in self.evidence.items()
        }
        self.voluntary_alternative = " ".join(self.voluntary_alternative.split())[:220] or "NONE"
        self.surcharge_multiplier = max(0.5, min(1.0, float(self.surcharge_multiplier)))
        self.activated = self.valid and any(tag != "NONE" for tag in self.action_tags.values())
        self.error = " ".join(self.error.split())[:240]


@dataclass(slots=True)
class CalibrationOutcome:
    action: str
    dimension: str
    direction: str
    description: str
    probability: float
    magnitude: float
    unit: str
    horizon: str
    measurement_family: str = "OTHER"
    population_basis: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        self.action = " ".join(self.action.split())[:120]
        self.dimension = " ".join(self.dimension.split())[:80]
        self.direction = self.direction.strip().upper()[:12]
        self.description = " ".join(self.description.split())[:140]
        self.probability = clamp(self.probability)
        self.magnitude = max(0.0, float(self.magnitude))
        unit = " ".join(self.unit.split())
        # Direction belongs in ``direction``. Keeping it in the unit prevents
        # legitimate within-unit comparisons (for example lives saved/lost).
        unit = re.sub(
            r"\s+(?:gained|lost|added|avoided|saved|prevented|protected|foregone)$",
            "",
            unit,
            flags=re.IGNORECASE,
        )
        self.unit = unit[:48]
        self.horizon = " ".join(self.horizon.split())[:64]
        allowed_families = {
            "MORTALITY", "HEALTH_DURATION", "ECONOMIC", "RESOURCE",
            "RIGHTS", "WELLBEING", "OTHER",
        }
        family = self.measurement_family.strip().upper()
        self.measurement_family = family if family in allowed_families else "OTHER"
        self.population_basis = (
            " ".join(self.population_basis.split())[:80] or "UNSPECIFIED"
        )


@dataclass(slots=True)
class CategoricalAxis:
    name: str
    action_values: dict[str, str]
    ethical_relevance: str
    fixed_by_scenario: bool = True

    def __post_init__(self) -> None:
        self.name = " ".join(self.name.split())[:80]
        self.action_values = {
            " ".join(str(action).split())[:120]: " ".join(str(value).split())[:140]
            for action, value in self.action_values.items()
            if " ".join(str(action).split()) and " ".join(str(value).split())
        }
        self.ethical_relevance = " ".join(self.ethical_relevance.split())[:180]


@dataclass(slots=True)
class NumericComparison:
    dimension: str
    unit: str
    action_values: dict[str, float]
    absolute_gap: float
    relative_gap: float
    measurement_family: str = "OTHER"
    population_basis: str = "UNSPECIFIED"
    time_basis: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        self.dimension = " ".join(self.dimension.split())[:80]
        self.unit = " ".join(self.unit.split())[:48]
        self.action_values = {
            " ".join(str(action).split())[:120]: float(value)
            for action, value in self.action_values.items()
        }
        self.absolute_gap = max(0.0, float(self.absolute_gap))
        self.relative_gap = max(0.0, float(self.relative_gap))
        self.measurement_family = self.measurement_family.strip().upper()[:32] or "OTHER"
        self.population_basis = " ".join(self.population_basis.split())[:80] or "UNSPECIFIED"
        self.time_basis = " ".join(self.time_basis.split())[:64] or "UNSPECIFIED"


@dataclass(slots=True)
class ProblemReformulation:
    unknowns: list[str]
    outcomes: list[CalibrationOutcome]
    switch_condition: str
    residual_tension: str
    question: str
    grounded_in: list[str]
    fixed_facts: list[str] = field(default_factory=list)
    categorical_axes: list[CategoricalAxis] = field(default_factory=list)
    changed_fixed_facts: list[str] = field(default_factory=list)
    numeric_comparisons: list[NumericComparison] = field(default_factory=list)
    hypothetical: bool = True
    accepted: bool = False
    rejection_reason: str = ""
    probe_result: str = "UNTESTED"
    switch_claim_valid: bool = False
    unresolved_numeric_tradeoffs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.unknowns = list(dict.fromkeys(
            " ".join(str(value).split())[:140]
            for value in self.unknowns
            if " ".join(str(value).split())
        ))[:5]
        self.outcomes = self.outcomes[:6]
        self.switch_condition = " ".join(self.switch_condition.split())[:220]
        self.residual_tension = " ".join(self.residual_tension.split())[:220]
        self.question = " ".join(self.question.split())[:300]
        self.grounded_in = list(dict.fromkeys(
            str(value).strip().lower()[:32]
            for value in self.grounded_in
            if str(value).strip()
        ))[:8]
        self.fixed_facts = self._clean(self.fixed_facts, 140, 8)
        self.categorical_axes = self.categorical_axes[:6]
        self.changed_fixed_facts = self._clean(self.changed_fixed_facts, 140, 8)
        self.numeric_comparisons = self.numeric_comparisons[:8]
        self.rejection_reason = " ".join(self.rejection_reason.split())[:240]
        self.probe_result = self.probe_result.strip().upper()[:32] or "UNTESTED"
        self.unresolved_numeric_tradeoffs = self._clean(
            self.unresolved_numeric_tradeoffs, 180, 8
        )

    @staticmethod
    def _clean(values: list[str], limit: int, count: int) -> list[str]:
        return list(dict.fromkeys(
            " ".join(str(value).split())[:limit]
            for value in values
            if " ".join(str(value).split())
        ))[:count]

    def compact(self) -> str:
        stakes = "; ".join(
            f"{outcome.dimension}/{outcome.action}: {outcome.probability:.0%} chance of "
            f"{outcome.magnitude:g} {outcome.unit} {outcome.direction.lower()} "
            f"over {outcome.horizon}"
            for outcome in self.outcomes
        )
        axes = "; ".join(
            f"{axis.name}: " + " / ".join(axis.action_values.values())
            for axis in self.categorical_axes
        )
        tradeoffs = "; ".join(self.unresolved_numeric_tradeoffs) or "none"
        return " ".join(
            f"HYPOTHETICAL CALIBRATION — {stakes}. Categorical axes: {axes or 'none'}. "
            f"Non-commensurable tradeoffs: {tradeoffs}. Remaining tension: "
            f"{self.residual_tension}. {self.question}".split()
        )[:600]


@dataclass(slots=True)
class WorkspaceResult:
    scenario: str
    actions: list[str]
    presentation_actions: list[str] = field(default_factory=list)
    presentation_action_mapping: list[dict[str, Any]] = field(default_factory=list)
    source_action_legend: dict[str, str] = field(default_factory=dict)
    action_source_grounding: dict[str, Any] = field(default_factory=dict)
    canonical_action_records: list[dict[str, Any]] = field(default_factory=list)
    source_testimonies: dict[str, str] = field(default_factory=dict)
    source_errors: dict[str, str] = field(default_factory=dict)
    source_baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_retrievals: dict[str, dict[str, Any]] = field(default_factory=dict)
    core_quote_pack: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_specialists: list[str] = field(default_factory=list)
    framing_cache: dict[str, Any] = field(default_factory=dict)
    frozen_world_replay: dict[str, Any] = field(default_factory=dict)
    performance_trace: dict[str, Any] = field(default_factory=dict)
    scenario_facts: dict[str, Any] = field(default_factory=dict)
    cycles: list[CycleRecord] = field(default_factory=list)
    selected_action: str = ""
    confidence: float = 0.0
    halted_by: str = ""
    moral_residue: list[str] = field(default_factory=list)
    compressed_rule: str = ""
    reopen_conditions: list[str] = field(default_factory=list)
    synthesis_proposals: list[SynthesisProposal] = field(default_factory=list)
    synthesis_viability_assessments: list[SynthesisViabilityAssessment] = field(
        default_factory=list
    )
    contingency_feasibility_assessments: list[ContingencyFeasibilityAssessment] = field(
        default_factory=list
    )
    failure_conditions: list[FailureCondition] = field(default_factory=list)
    planning_assessments: list[PlanningAssessment] = field(default_factory=list)
    planning_branches: list[PlanningBranchEvaluation] = field(default_factory=list)
    access_decisions: list[WorkspaceAccessDecision] = field(default_factory=list)
    visibility_assessments: list[VisibilityAssessment] = field(default_factory=list)
    autonomy_assessments: list[AutonomyAssessment] = field(default_factory=list)
    problem_reformulations: list[ProblemReformulation] = field(default_factory=list)
    judgment_status: str = "GOVERNED_RECOMMENDATION"
    current_plurality: str = ""
    epistemic_confidence: float = 0.0
    governing_justification_status: str = "NONE"
    governing_attack_reason: str = ""
    governing_authority_transitions: list[dict[str, Any]] = field(default_factory=list)
    semantic_invariants: list[Any] = field(default_factory=list)
    semantic_graphs: list[dict[str, Any]] = field(default_factory=list)
    graph_transactions: list[dict[str, Any]] = field(default_factory=list)
    authoritative_semantic_state: dict[str, Any] = field(default_factory=dict)
    deliberative_problem_state: dict[str, Any] = field(default_factory=dict)
    broadcast_influence_records: list[dict[str, Any]] = field(default_factory=list)
    rawlsian_position_ledger: list[dict[str, Any]] = field(default_factory=list)
    utilitarian_consequence_ledger: list[dict[str, Any]] = field(default_factory=list)
    deontological_duty_ledger: list[dict[str, Any]] = field(default_factory=list)
    virtue_character_ledger: list[dict[str, Any]] = field(default_factory=list)
    care_relationship_ledger: list[dict[str, Any]] = field(default_factory=list)
    proposition_ledger: list[dict[str, Any]] = field(default_factory=list)
    shared_unresolved_dependencies: list[dict[str, Any]] = field(default_factory=list)
    side_premise_audits: list[dict[str, Any]] = field(default_factory=list)
    ev_dominance_assessments: list[dict[str, Any]] = field(default_factory=list)
    termination_assessment: Any = None
    moral_residue_records: list[Any] = field(default_factory=list)
    access_construct_records: list[Any] = field(default_factory=list)
    further_deliberation_estimate: Any = None
    trace_health: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
