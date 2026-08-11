from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from .models import CalibrationOutcome, CandidateChunk, CategoricalAxis, FailureCondition, NumericComparison, PlanningAssessment, ProblemReformulation, SynthesisProposal, VisibilityAssessment, WorkspaceBroadcast


FRAMEWORK_ROLES = {
    "utilitarian": "Score expected harms, benefits, urgency, and reversibility.",
    "deontological": "Score duties, rights, entitlement, coercion, and universal rules.",
    "virtue": "Score practical wisdom, character, honesty, courage, and habituation.",
    "care": "Score vulnerability, dependency, trust, relationship, and responsiveness.",
    "rawlsian": "Score fairness, equal liberty, public rules, and the least advantaged.",
}

ALLOWED_CONSTRAINTS = {
    "IMMINENT_HARM",
    "RIGHTS",
    "DUTY",
    "FAIRNESS",
    "CARE",
    "CHARACTER",
    "FEASIBILITY",
    "UNCERTAINTY",
    "PUBLIC_RULE",
}

FRAMEWORK_CONSTRAINTS = {
    "utilitarian": {"IMMINENT_HARM", "FEASIBILITY", "UNCERTAINTY"},
    "deontological": {"DUTY", "RIGHTS", "PUBLIC_RULE", "UNCERTAINTY"},
    "virtue": {"CHARACTER", "FEASIBILITY", "UNCERTAINTY"},
    "care": {"CARE", "FEASIBILITY", "UNCERTAINTY"},
    "rawlsian": {"FAIRNESS", "RIGHTS", "PUBLIC_RULE", "UNCERTAINTY"},
}

ALLOWED_UNRESOLVED = {
    "NONE",
    "VERIFY_FACTS",
    "CHECK_FEASIBILITY",
    "CLARIFY_SCENARIO",
}

# Explicitly disclosed speculation is useful for discovering contingencies. It
# remains discounted, but not so heavily that a productive hypothesis becomes
# indistinguishable from an uninformative tie.
SPECULATIVE_DIRECTION_RETENTION = 0.55
SPECULATIVE_EPISTEMIC_CAP = 0.50


@lru_cache(maxsize=8)
def _json_grammar(schema_text: str) -> Any | None:
    """Build a llama.cpp grammar when supported; tests and older builds may omit it."""
    try:
        from llama_cpp import LlamaGrammar

        return LlamaGrammar.from_json_schema(schema_text)
    except (ImportError, AttributeError, ValueError):
        return None


def _call_json_llm(llm: Any, prompt: str, *, max_tokens: int, temperature: float, schema: dict[str, Any]):
    if hasattr(llm, "complete_json"):
        return llm.complete_json(
            prompt,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    grammar = _json_grammar(json.dumps(schema, sort_keys=True))
    if grammar is not None:
        kwargs["grammar"] = grammar
    return llm(prompt, **kwargs)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Local model returned no JSON object: {text[:160]!r}")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Local model response must be a JSON object")
    return value


def _number(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strict_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def _semantic_word_count(text: str) -> int:
    """Count identifier-style labels as ordinary multiword phrases."""
    return len(re.findall(r"[A-Za-z0-9]+", str(text).replace("_", " ")))


_LANDSCAPE_STOPWORDS = {
    "about", "action", "against", "into", "must", "their", "there", "these",
    "those", "through", "with", "without", "would", "should", "could", "from",
    "that", "this", "then", "than", "when", "where", "while", "have", "has",
    "strike", "choose", "select", "option", "instead",
    # Quantities, generic recipients, and decision verbs are not entity anchors.
    # A case for an action normally mentions who the alternative would save or
    # harm; treating that comparison as an inversion produced false positives.
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "hundred", "thousand", "million",
    "person", "people", "individual", "individuals", "life", "lives",
    "save", "allow", "continue", "pull", "push", "redirect", "protect",
    "give", "receive", "publish", "disclose", "report", "remain", "keep",
    "tell", "swerve", "evacuate", "leave", "mandate", "transfer",
}
_HARM_ACTION = re.compile(
    r"^\s*(?:strike|kill|sacrifice|harm|injure|run\s+over|withhold|deny|abandon)\b",
    re.IGNORECASE,
)
_HARM_ACKNOWLEDGMENT = re.compile(
    r"\b(?:strike|kill|killed|harm|harmed|injur|sacrific|death|dies|die|loss|lost|"
    r"withhold|deny|abandon)\w*\b",
    re.IGNORECASE,
)


def _landscape_semantic_errors(
    actions: Sequence[str],
    landscape_cases: dict[str, str],
    unresolved: str,
    assumption_status: str,
    tiebreaker: str,
    tiebreaker_failure: str,
) -> list[str]:
    """Catch lightweight causal inversions without constructing a full graph."""
    errors: list[str] = []
    def stems(text: str) -> set[str]:
        values = set()
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            if len(token) < 4 or token in _LANDSCAPE_STOPWORDS:
                continue
            for suffix in ("ing", "ied", "ed", "es", "s"):
                if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                    token = token[:-len(suffix)]
                    break
            values.add(token[:8])
        return values

    token_sets = [
        {
            token for token in stems(action)
        }
        for action in actions
    ]
    for index, action in enumerate(actions):
        case = landscape_cases.get(action, "")
        case_tokens = stems(case)
        leading_clause = re.split(
            r"\s*[;—–]\s*|\b(?:but|while|although|however|yet|whereas|risks?)\b",
            case,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        leading_tokens = stems(leading_clause)
        other_tokens = set().union(*(tokens for offset, tokens in enumerate(token_sets) if offset != index))
        distinctive = token_sets[index] - other_tokens
        if _HARM_ACTION.search(action):
            # Harm acknowledgment is a stronger and more paraphrase-tolerant
            # causal anchor than literal repetition of a victim label.
            if not _HARM_ACKNOWLEDGMENT.search(case):
                errors.append(f"case for A{index} omits the harm caused by choosing it")
        elif distinctive and not (distinctive & case_tokens):
            opposing_distinctive = other_tokens - token_sets[index]
            if opposing_distinctive & leading_tokens:
                errors.append(
                    f"case for A{index} tracks the opposing action but not its own action"
                )

    if len(actions) >= 2:
        normalized_cases = [
            set(re.findall(r"[a-z0-9]+", landscape_cases.get(action, "").casefold()))
            for action in actions
        ]
        for first in range(len(normalized_cases)):
            for second in range(first + 1, len(normalized_cases)):
                union = normalized_cases[first] | normalized_cases[second]
                if union and len(normalized_cases[first] & normalized_cases[second]) / len(union) >= 0.85:
                    errors.append("action cases are substantively redundant")
                    break

    unresolved_judgment = (
        unresolved != "NONE"
        or assumption_status in {"CONDITIONAL", "UNDERDETERMINED"}
    )
    if unresolved_judgment and tiebreaker_failure.casefold() == "none":
        errors.append("unresolved judgment claims its tiebreaker fully succeeded")
    if re.search(r"\b(?:random|coin|lottery)\b", tiebreaker, re.IGNORECASE) and not re.search(
        r"\b(?:random|coin|lottery)\b", tiebreaker_failure, re.IGNORECASE
    ):
        errors.append("procedural randomization is represented as a substantive action ranking")
    return list(dict.fromkeys(errors))


_CLOSED_CHOICE_PATTERNS = (
    re.compile(r"\bno\s+(?:other|third|alternative)\s+(?:way|option|choice|action)\b", re.I),
    re.compile(r"\b(?:only|sole)\s+(?:available\s+)?(?:options|choices|actions)\b", re.I),
    re.compile(r"\bmust\s+(?:choose|decide)\s+between\b", re.I),
    re.compile(r"\bmust\s+either\b", re.I),
    re.compile(r"\bcannot\s+(?:do|act|choose)\s+otherwise\b", re.I),
    re.compile(r"\bcannot\b[^.!?]{0,80}\b(?:another|any\s+other)\s+way\b", re.I),
    re.compile(r"\bno\s+(?:compromise|middle|additional)\s+(?:option|choice|action)\b", re.I),
)


def _scenario_closes_action_set(scenario: str) -> bool:
    """Recognize explicit closed-world dilemmas without relying on topic words."""
    normalized = " ".join(scenario.split())
    return any(pattern.search(normalized) for pattern in _CLOSED_CHOICE_PATTERNS)


def _candidate_from_data(
    specialist: str,
    actions: Sequence[str],
    data: dict[str, Any],
    broadcast: WorkspaceBroadcast,
    baseline_id: str,
    scenario_facts: dict[str, Any],
    previous_recommendation_id: str = "",
    previous_confidence: float | None = None,
    prior_assumption_status: str = "NOT_AUDITED",
    prior_unsupported_assumption: str = "",
    prior_reversal_condition: str = "",
) -> CandidateChunk:
    action_ids = [f"A{index}" for index in range(len(actions))]
    raw_scores = data.get("scores")
    if not isinstance(raw_scores, dict) or set(raw_scores) != set(action_ids):
        raise ValueError(f"scores must contain exactly {action_ids}")
    id_scores = {
        action_id: _strict_number(raw_scores[action_id], f"scores.{action_id}")
        for action_id in action_ids
    }
    scores = {
        action: id_scores[action_id]
        for action_id, action in zip(action_ids, actions)
    }

    recommended_id = str(data.get("r", data.get("recommended", ""))).strip().upper()
    if recommended_id not in action_ids:
        raise ValueError("recommended must be a valid action ID")
    if id_scores[recommended_id] != max(id_scores.values()):
        raise ValueError("recommended must have the highest score")

    if baseline_id not in {*action_ids, "NONE"}:
        raise ValueError("baseline must be a valid action ID or NONE")
    if baseline_id == "NONE":
        alignment = "UNCLEAR"
    elif recommended_id == baseline_id:
        alignment = "SUPPORTS"
    else:
        alignment = "RECONSIDERS"
    if broadcast.constraint == "OPEN_DELIBERATION" and baseline_id != "NONE" and recommended_id != baseline_id:
        raise ValueError("the initial recommendation must match the source-testimony baseline")

    rationale = " ".join(str(data.get("w", data.get("why", ""))).split())
    if len(rationale.split()) < 2:
        raise ValueError("why must contain at least two words")

    recipient_facts = scenario_facts.get("survival_chance", {})
    if isinstance(recipient_facts, dict) and recipient_facts:
        selected_action = actions[action_ids.index(recommended_id)].lower()
        selected_recipients = [name for name in recipient_facts if name in selected_action]
        mentioned_recipients = [name for name in recipient_facts if name in rationale.lower()]
        if mentioned_recipients and selected_recipients and not set(selected_recipients) & set(mentioned_recipients):
            raise ValueError("why names a different recipient than the recommended action")
        if "higher survival" in rationale.lower() and selected_recipients:
            highest = max(recipient_facts, key=recipient_facts.get)
            if highest not in selected_recipients:
                raise ValueError("why contradicts the scenario's survival probabilities")

    constraint = str(data.get("c", data.get("constraint", ""))).strip().upper().replace(" ", "_")
    allowed_constraints = FRAMEWORK_CONSTRAINTS[specialist]
    if constraint not in allowed_constraints:
        raise ValueError(f"constraint for {specialist} must be one of {sorted(allowed_constraints)}")
    unresolved = str(data.get("u", data.get("unresolved", "NONE"))).strip().upper().replace(" ", "_")
    if unresolved not in ALLOWED_UNRESOLVED:
        raise ValueError(f"unresolved must be one of {sorted(ALLOWED_UNRESOLVED)}")

    evidence_basis = str(data.get("e", "STATED_FACTS")).strip().upper()
    if evidence_basis not in {"STATED_FACTS", "FRAMEWORK_ONLY", "UNSTATED_FACTS"}:
        raise ValueError("evidence basis must be STATED_FACTS, FRAMEWORK_ONLY, or UNSTATED_FACTS")
    speculative_claim = " ".join(str(data.get("x", "")).split())
    if evidence_basis == "UNSTATED_FACTS":
        if len(speculative_claim.split()) < 3:
            raise ValueError("unstated factual reasoning must identify the speculative claim")
        if unresolved == "NONE":
            unresolved = "VERIFY_FACTS"
        # Speculation may identify a useful branch, but cannot manufacture a
        # confident base-case vote. Preserve more of its direction than before
        # while still pulling it toward a tie and requiring factual review.
        id_scores = {
            action_id: 0.5 + (score - 0.5) * SPECULATIVE_DIRECTION_RETENTION
            for action_id, score in id_scores.items()
        }
        scores = {
            action: id_scores[action_id]
            for action_id, action in zip(action_ids, actions)
        }

    raw_landscape = data.get("l", {})
    landscape_search_attempted = any(key in data for key in ("l", "da", "t", "tf"))
    landscape_cases: dict[str, str] = {}
    if isinstance(raw_landscape, dict):
        landscape_cases = {
            actions[action_ids.index(action_id)]: " ".join(str(raw_landscape.get(action_id, "")).split())
            for action_id in action_ids
            if " ".join(str(raw_landscape.get(action_id, "")).split())
        }
    landscape_axis = " ".join(str(data.get("da", "")).split())
    landscape_tiebreaker = " ".join(str(data.get("t", "")).split())
    landscape_failure = " ".join(str(data.get("tf", "")).split())
    # `t` is a compatibility fallback for older compact responses. New
    # structured responses carry an explicit decision rule and keep empirical
    # and moral switch conditions separate.
    decision_rule = " ".join(
        str(data.get("dr") or landscape_tiebreaker or rationale).split()
    )
    factual_threshold = " ".join(str(data.get("ft", "NONE")).split()) or "NONE"
    normative_threshold = " ".join(str(data.get("nt", "NONE")).split()) or "NONE"
    if _semantic_word_count(decision_rule) < 2:
        raise ValueError("decision rule must state how the actions are ranked")
    if factual_threshold.casefold() != "none" and _semantic_word_count(factual_threshold) < 3:
        raise ValueError("factual reversal threshold must be specific or NONE")
    if normative_threshold.casefold() != "none" and _semantic_word_count(normative_threshold) < 3:
        raise ValueError("normative reversal threshold must be specific or NONE")
    if factual_threshold.casefold() == normative_threshold.casefold() != "none":
        raise ValueError("factual and normative reversal thresholds must be distinct")

    review_response = "NOT_TESTED"
    review_justification = ""
    revised_reversal = ""
    if broadcast.constraint == "REVERSAL_AUDIT":
        review_response = str(data.get("rr", "")).strip().upper()
        review_justification = " ".join(str(data.get("rj", "")).split())
        revised_reversal = " ".join(str(data.get("rv", "")).split())
        if review_response not in {"ACCEPT", "REVISE", "REJECT"}:
            raise ValueError("reversal review must ACCEPT, REVISE, or REJECT the challenge")
        if len(review_justification.split()) < 3:
            raise ValueError("reversal review must justify its response")
        if review_response == "REVISE" and len(revised_reversal.split()) < 3:
            raise ValueError("a revised reversal condition is required when revising")
    landscape_search_complete = bool(
        len(landscape_cases) == len(actions)
        and all(_semantic_word_count(reason) >= 2 for reason in landscape_cases.values())
        and _semantic_word_count(landscape_axis) >= 2
        and _semantic_word_count(landscape_tiebreaker) >= 2
        and (
            landscape_failure.casefold() == "none"
            or _semantic_word_count(landscape_failure) >= 2
        )
    )

    assumption_status = prior_assumption_status
    unsupported_assumption = prior_unsupported_assumption
    reversal_condition = prior_reversal_condition
    if broadcast.constraint == "CONSENSUS_AUDIT":
        assumption_status = str(data.get("d", "")).strip().upper()
        unsupported_assumption = " ".join(str(data.get("a", "")).split())
        reversal_condition = " ".join(str(data.get("v", "")).split())
        if assumption_status not in {"SUPPORTED", "CONDITIONAL", "UNDERDETERMINED"}:
            raise ValueError("audit status must be SUPPORTED, CONDITIONAL, or UNDERDETERMINED")
        if len(unsupported_assumption.split()) < 3:
            raise ValueError("audit must identify the central assumption")
        if len(reversal_condition.split()) < 3:
            raise ValueError("audit must identify a plausible reversal condition")
        if assumption_status != "SUPPORTED" and unresolved == "NONE":
            raise ValueError("conditional or underdetermined audits must preserve uncertainty")
    elif assumption_status in {"CONDITIONAL", "UNDERDETERMINED"} and unresolved == "NONE":
        # A later ordinary or hypothetical cycle cannot erase missing real-world
        # facts merely by omitting the marker.
        unresolved = "VERIFY_FACTS"

    boundary_position = "NOT_TESTED"
    decisive_axis = ""
    boundary_switch_condition = ""
    if broadcast.constraint == "PROBLEM_REFORMULATION":
        boundary_position = str(data.get("bp", "")).strip().upper()
        decisive_axis = " ".join(str(data.get("dx", "")).split())
        boundary_switch_condition = " ".join(str(data.get("sv", "")).split())
        if boundary_position not in {*action_ids, "SPLIT"}:
            raise ValueError("boundary position must be an action ID or SPLIT")
        if boundary_position in action_ids and boundary_position != recommended_id:
            raise ValueError("boundary position action must match recommended")
        if len(decisive_axis.split()) < 2:
            raise ValueError("boundary response must name a decisive axis")
        if len(boundary_switch_condition.split()) < 4:
            raise ValueError("boundary response must state a switch condition")

    landscape_validation_errors = (
        _landscape_semantic_errors(
            actions,
            landscape_cases,
            unresolved,
            assumption_status,
            landscape_tiebreaker,
            landscape_failure,
        )
        if landscape_search_complete else ["landscape search is structurally incomplete"]
    )
    if (
        ("ft" in data or "nt" in data)
        and factual_threshold.casefold() == "none"
        and normative_threshold.casefold() == "none"
    ):
        landscape_validation_errors.append(
            "decision rule supplies no factual or normative reversal threshold"
        )
    landscape_semantic_valid = not landscape_validation_errors

    audit_certainty = {
        "NOT_AUDITED": 1.0,
        "SUPPORTED": 1.0,
        "CONDITIONAL": 0.60,
        "UNDERDETERMINED": 0.35,
    }[assumption_status]
    if audit_certainty < 1.0:
        id_scores = {
            action_id: 0.5 + (score - 0.5) * audit_certainty
            for action_id, score in id_scores.items()
        }
        scores = {
            action: id_scores[action_id]
            for action_id, action in zip(action_ids, actions)
        }

    ordered_scores = sorted(id_scores.values(), reverse=True)
    score_gap = ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else ordered_scores[0]
    preference_strength = max(0.0, min(1.0, score_gap))
    friction = preference_strength
    default_epistemic = {
        "STATED_FACTS": 0.85,
        "FRAMEWORK_ONLY": 0.70,
        "UNSTATED_FACTS": SPECULATIVE_EPISTEMIC_CAP,
    }[evidence_basis]
    epistemic_confidence = _strict_number(
        data.get("z", default_epistemic), "epistemic confidence"
    )
    if evidence_basis == "UNSTATED_FACTS":
        epistemic_confidence = min(epistemic_confidence, SPECULATIVE_EPISTEMIC_CAP)
    if unresolved != "NONE":
        epistemic_confidence = min(epistemic_confidence, 0.55)
    if assumption_status == "CONDITIONAL":
        epistemic_confidence = min(epistemic_confidence, 0.60)
    elif assumption_status == "UNDERDETERMINED":
        epistemic_confidence = min(epistemic_confidence, 0.35)
    if landscape_search_attempted and not landscape_semantic_valid:
        epistemic_confidence = min(epistemic_confidence, 0.60)
    surprise = 0.7 if alignment == "RECONSIDERS" else (0.3 if alignment == "UNCLEAR" else 0.0)
    previous_id = (
        previous_recommendation_id
        if previous_recommendation_id in action_ids
        else ""
    )
    position_changed = bool(previous_id and recommended_id != previous_id)
    change_justification = " ".join(str(data.get("j", "NONE")).split())
    recommended_action = actions[action_ids.index(recommended_id)]
    favored_fragment = (
        broadcast.intent.removeprefix("evaluate_").casefold().strip()
        if broadcast.intent.startswith("evaluate_")
        else ""
    )
    moved_to_favored = bool(
        position_changed
        and favored_fragment
        and recommended_action.casefold().startswith(favored_fragment)
    )
    conformity_penalty = 0.0
    review_constraints = {
        "SYNTHESIS_REVIEW", "CONTINGENCY_REVIEW", "PLANNING_REVIEW",
        "CONSENSUS_AUDIT", "PROBLEM_REFORMULATION",
    }
    if moved_to_favored and broadcast.constraint not in review_constraints:
        conformity_penalty = 0.65
        epistemic_confidence *= 1.0 - conformity_penalty
        surprise = max(surprise, 0.8)
    confidence_drift = (
        preference_strength - previous_confidence
        if previous_confidence is not None and not position_changed
        else 0.0
    )
    recommended_is_favored = bool(
        favored_fragment and recommended_action.casefold().startswith(favored_fragment)
    )
    drift_favors_salient_action = bool(
        previous_confidence is not None
        and not position_changed
        and (
            (recommended_is_favored and confidence_drift > 0.10)
            or (not recommended_is_favored and confidence_drift < -0.10)
        )
    )
    justification_is_specific = (
        change_justification.casefold() != "none"
        and len(change_justification.split()) >= 3
        and broadcast.constraint in allowed_constraints
    )
    confidence_drift_penalty = 0.0
    if (
        drift_favors_salient_action
        and broadcast.constraint not in review_constraints
        and not justification_is_specific
    ):
        confidence_drift_penalty = 0.75
        # Preserve the observed preference, but reduce how much an unjustified
        # shift is trusted in aggregation.
        epistemic_confidence *= 1.0 - confidence_drift_penalty

    return CandidateChunk(
        specialist=specialist,
        constraint=constraint,
        action_scores=scores,
        surprise=surprise,
        friction=friction,
        confidence=epistemic_confidence,
        unresolved=unresolved,
        rationale=rationale,
        recommended_action=recommended_action,
        baseline_action=(actions[action_ids.index(baseline_id)] if baseline_id != "NONE" else ""),
        testimony_alignment=alignment,
        previous_action=(actions[action_ids.index(previous_id)] if previous_id else ""),
        position_changed=position_changed,
        change_justification=change_justification,
        conformity_penalty=conformity_penalty,
        previous_confidence=(previous_confidence or 0.0),
        confidence_drift=confidence_drift,
        confidence_drift_penalty=confidence_drift_penalty,
        assumption_status=assumption_status,
        unsupported_assumption=unsupported_assumption,
        reversal_condition=reversal_condition,
        boundary_position=boundary_position,
        decisive_axis=decisive_axis,
        boundary_switch_condition=boundary_switch_condition,
        evidence_basis=evidence_basis,
        speculative_claim=speculative_claim,
        landscape_cases=landscape_cases,
        landscape_decisive_axis=landscape_axis,
        landscape_tiebreaker=landscape_tiebreaker,
        landscape_tiebreaker_failure=landscape_failure,
        landscape_search_complete=landscape_search_complete,
        landscape_search_attempted=landscape_search_attempted,
        landscape_semantic_valid=landscape_semantic_valid,
        landscape_validation_errors=landscape_validation_errors,
        preference_strength=preference_strength,
        epistemic_confidence=epistemic_confidence,
        previous_preference_strength=(previous_confidence or 0.0),
        preference_drift=confidence_drift,
        preference_drift_penalty=confidence_drift_penalty,
        decision_rule=decision_rule,
        factual_reversal_threshold=factual_threshold,
        normative_reversal_threshold=normative_threshold,
        reversal_review_response=review_response,
        reversal_review_justification=review_justification,
        revised_reversal_condition=revised_reversal,
    )


def _invalid_candidate(specialist: str, actions: Sequence[str], error: str) -> CandidateChunk:
    return CandidateChunk(
        specialist=specialist,
        constraint="MALFORMED_RESPONSE",
        action_scores={action: 0.5 for action in actions},
        surprise=0.0,
        friction=0.0,
        confidence=0.0,
        unresolved="REVIEW_MODEL_OUTPUT",
        rationale="Delegate output failed semantic validation.",
        schema_valid=False,
        validation_errors=[error[:300]],
    )


@dataclass(slots=True)
class CompactLocalSpecialist:
    name: str
    llm: Any
    testimony: str = ""
    baseline_action_id: str = "NONE"
    scenario_facts: dict[str, Any] | None = None
    max_tokens: int = 128
    previous_recommendation_id: str = ""
    previous_confidence: float | None = None
    assumption_status: str = "NOT_AUDITED"
    unsupported_assumption: str = ""
    reversal_condition: str = ""
    memory_profile: dict[str, Any] | None = None

    def evaluate(
        self,
        scenario: str,
        actions: Sequence[str],
        broadcast: WorkspaceBroadcast,
    ) -> CandidateChunk:
        role = FRAMEWORK_ROLES[self.name]
        testimony = " ".join(self.testimony.split())[:900]
        action_ids = [f"A{index}" for index in range(len(actions))]
        action_legend = {action_id: action for action_id, action in zip(action_ids, actions)}
        allowed_constraints = sorted(FRAMEWORK_CONSTRAINTS[self.name])
        fixed_baseline = self.baseline_action_id if self.baseline_action_id in {*action_ids, "NONE"} else "NONE"
        schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "properties": {
                        action_id: {"type": "number", "minimum": 0, "maximum": 1}
                        for action_id in action_ids
                    },
                    "required": action_ids,
                    "additionalProperties": False,
                },
                "r": {"type": "string", "enum": action_ids},
                "c": {"type": "string", "enum": allowed_constraints},
                "u": {"type": "string", "enum": sorted(ALLOWED_UNRESOLVED)},
                "w": {"type": "string"},
                "j": {"type": "string"},
                "e": {
                    "type": "string",
                    "enum": ["STATED_FACTS", "FRAMEWORK_ONLY", "UNSTATED_FACTS"],
                },
                "x": {"type": "string"},
                "l": {
                    "type": "object",
                    "properties": {
                        action_id: {"type": "string", "maxLength": 100}
                        for action_id in action_ids
                    },
                    "required": action_ids,
                    "additionalProperties": False,
                },
                "da": {"type": "string", "maxLength": 80},
                "t": {"type": "string", "maxLength": 100},
                "tf": {"type": "string", "maxLength": 100},
                "dr": {"type": "string", "maxLength": 120},
                "ft": {"type": "string", "maxLength": 120},
                "nt": {"type": "string", "maxLength": 120},
                "z": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "scores", "r", "c", "u", "w", "j", "e", "x", "l", "da", "t", "tf",
                "dr", "ft", "nt", "z",
            ],
            "additionalProperties": False,
        }
        if broadcast.constraint == "CONSENSUS_AUDIT":
            schema["properties"].update({
                "d": {
                    "type": "string",
                    "enum": ["SUPPORTED", "CONDITIONAL", "UNDERDETERMINED"],
                },
                "a": {"type": "string"},
                "v": {"type": "string"},
            })
            schema["required"].extend(["d", "a", "v"])
        if broadcast.constraint == "PROBLEM_REFORMULATION":
            schema["properties"].update({
                "bp": {"type": "string", "enum": [*action_ids, "SPLIT"]},
                "dx": {"type": "string"},
                "sv": {"type": "string"},
            })
            schema["required"].extend(["bp", "dx", "sv"])
        if broadcast.constraint == "REVERSAL_AUDIT":
            schema["properties"].update({
                "rr": {"type": "string", "enum": ["ACCEPT", "REVISE", "REJECT"]},
                "rj": {"type": "string", "maxLength": 120},
                "rv": {"type": "string", "maxLength": 120},
            })
            schema["required"].extend(["rr", "rj", "rv"])
        prompt = f"""[INST]
You are the {self.name} specialist in a bandwidth-limited ethical workspace.
Task: {role}
Scenario: {' '.join(scenario.split())[:700]}
Your original corpus-grounded testimony: {testimony}
Frozen testimony baseline: {fixed_baseline}
Previous cycle recommendation: {self.previous_recommendation_id or 'NONE'}
Workspace: {broadcast.compact()}
Scenario facts: {json.dumps(self.scenario_facts or {}, sort_keys=True)}
Prior contribution profile: {json.dumps(self.memory_profile or {}, sort_keys=True)}
Action IDs: {json.dumps(action_legend)}

Return ONLY compact JSON like:
{{"scores":{{"A0":0.8,"A1":0.2}},"r":"A0","c":"{allowed_constraints[0]}","u":"NONE","w":"short reason","j":"NONE","e":"STATED_FACTS","x":"NONE","l":{{"A0":"best case A0","A1":"best case A1"}},"da":"decisive ethical axis","t":"attempted comparison rule","tf":"NONE","dr":"prefer A0 when its reason outweighs A1","ft":"NONE","nt":"prefer A1 if its value is overriding","z":0.8}}
Return scores for every action ID. recommended must have the highest score.
r=recommended and must have the highest score. The frozen baseline was extracted
separately from your testimony. Python derives whether the result supports or
reconsiders it. During OPEN_DELIBERATION, recommended
must equal a known baseline. Each score means recommendation strength:
1 strongly recommends; 0 strongly rejects. w must be 2-8 words.
z is epistemic confidence: how likely the ranking is to survive further factual
inquiry and critical scrutiny. z is NOT preference strength. A strong preference
may have low z when it depends on uncertain facts; a close moral tradeoff may have
high z when its facts and framework interpretation are stable.
c must be one of: {', '.join(allowed_constraints)}.
Choose u only from: {', '.join(sorted(ALLOWED_UNRESOLVED))}.
The listed actions are exhaustive for this deliberation. Do not propose recording
testimony first, delaying, combining actions, obtaining different treatment, or any
other third option unless it is already one of the action IDs. Evaluate the dilemma
as stated even when a real-world alternative would be preferable.
Return e=STATED_FACTS when factual claims come directly from the scenario;
e=FRAMEWORK_ONLY when the difference is purely a moral principle; or
e=UNSTATED_FACTS when your preference relies on a prediction not stated in the
scenario. For UNSTATED_FACTS, x must name that prediction and u must preserve
uncertainty. Otherwise x="NONE". Never treat possible repeat offenses, institutional
effects, hidden alternatives, or affected populations as facts unless stated.
Before recommending, search the ethical landscape. In l, give the strongest
framework-specific case for EVERY action, including the action you reject. da names
the ethical axis that actually separates them. t names the comparison or tie-breaking
rule you attempted. tf="NONE" if it resolves the comparison; otherwise state exactly
why the rule fails. These fields are mandatory even when you have a clear preference.
dr is your explicit decision rule: "prefer [action] when [comparative condition]."
It must say what makes the recommended action outrank its strongest rival.
ft is the smallest change in an empirical fact, probability, magnitude, duration,
or feasibility that would reverse the ranking; use NONE when facts cannot reverse it.
nt is the smallest change in moral priority, right, duty, or acceptable principle
that would reverse it; use NONE when normative priority cannot reverse it. Never put
a moral priority in ft or an empirical prediction in nt. At least one threshold
should expose how the recommendation could change unless reversal truly requires a
different scenario.
Each case must describe what happens IF ITS OWN ACTION is chosen. Name the affected
person or value from that action, and for harmful actions explicitly acknowledge the
harm before giving the countervailing reason. Do not place the benefit of sparing a
person under the action that kills or harms that same person. If u is not NONE, or
the judgment remains conditional or underdetermined, tf cannot be NONE. A coin flip
or randomizer is a procedural resolution, not evidence that one listed action is
substantively better; say this limitation in tf.
The prior contribution profile is diagnostic: repeated shallow searches require a
more discriminating axis now, but history never overrides the present scenario.
If it records causal_mapping_errors, explicitly verify that each case describes its
own action rather than the opposite action. If it records unresolved_tiebreaker_errors,
name the unresolved limitation in tf. Repeated consensus is not evidence: a grounded,
framework-relevant minority position receives access credit, while unsupported
contrarianism, random switching, and unexplained reversals do not.
Keep your previous recommendation unless the broadcast supplies a framework-relevant
reason to reconsider it. If you change, j must name that new reason; agreement with
the currently favored action is not itself a reason. Also use j when changing your
preference strength by more than 0.10: name the new framework-relevant information that
justifies the change. Repetition or popularity is not new information. Otherwise j="NONE".
When Workspace contingency is not NONE, directly analyze that failure condition.
Score the original fallback actions for that branch and make w state the preferred
fallback; do not merely repeat your general position on the original scenario.
During CONSENSUS_AUDIT, test whether the leading action depends on facts not stated
in the scenario. Add d=SUPPORTED, CONDITIONAL, or UNDERDETERMINED; a=the central
assumption; v=a plausible factual condition that would reverse the recommendation.
If d is not SUPPORTED, u must be VERIFY_FACTS or CLARIFY_SCENARIO. Do not treat
agreement, vividness, temporal proximity, or an abstract group as evidence.
An explicit scenario fact is not an unsupported assumption. If reversal would
require changing the actor, actions, consent, causal structure, or another stated
fact, use d=SUPPORTED, make a identify the fact supporting the judgment, and make v
say that reversal requires a different scenario. Do not manufacture uncertainty.
During PROBLEM_REFORMULATION, treat every supplied number as a hypothetical
switch-point probe, not as a newly discovered scenario fact. Score the original
actions at that boundary and use w and j to identify the framework-specific value
tension that remains. Do not clear prior factual uncertainty or invent a third action.
Also return bp=A0, A1, or SPLIT; dx=the decisive numeric dimension or categorical
axis from the broadcast; and sv=the smallest directional change that would switch
your recommendation. bp must match r unless bp=SPLIT. Generic references to harm,
fairness, or uncertainty are insufficient unless tied to a named supplied axis.
During REVERSAL_AUDIT, another specialist has challenged the leading rule. Return
rr=ACCEPT if that condition really reverses your ranking, REVISE if a nearby but
different condition does, or REJECT only if it changes the scenario or conflicts
with your framework. Explain in rj. For REVISE, put the smallest corrected condition
in rv; otherwise rv="NONE". Do not reject merely because the challenge favors the
opposing action.
[/INST]"""
        output = _call_json_llm(
            self.llm,
            prompt,
            max_tokens=self.max_tokens,
            temperature=0.2,
            schema=schema,
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        try:
            data = _extract_json(raw)
            candidate = _candidate_from_data(
                self.name, actions, data, broadcast, fixed_baseline, self.scenario_facts or {},
                self.previous_recommendation_id,
                self.previous_confidence,
                self.assumption_status,
                self.unsupported_assumption,
                self.reversal_condition,
            )
            self.previous_recommendation_id = action_ids[actions.index(candidate.recommended_action)]
            self.previous_confidence = candidate.preference_strength
            self.assumption_status = candidate.assumption_status
            self.unsupported_assumption = candidate.unsupported_assumption
            self.reversal_condition = candidate.reversal_condition
            return candidate
        except (ValueError, json.JSONDecodeError) as first_error:
            repair_prompt = f"""[INST]
Repair this invalid answer as JSON only: {raw[:400]}
Required fields: scores object for {', '.join(action_ids)}, r, c, u, w, j. Also return
e=STATED_FACTS, FRAMEWORK_ONLY, or UNSTATED_FACTS and x naming any unstated claim.
Return l with a case for every action ID, da, t, tf, explicit decision rule dr,
separate factual/normative thresholds ft and nt, and epistemic confidence z.
r must have the highest score and respect frozen baseline {fixed_baseline} initially.
c must be one of: {', '.join(allowed_constraints)}. No prose.
{('Also include d, a, v for the CONSENSUS_AUDIT; if d is not SUPPORTED, u cannot be NONE.' if broadcast.constraint == 'CONSENSUS_AUDIT' else '')}
{('Also include bp, dx, sv for the PROBLEM_REFORMULATION.' if broadcast.constraint == 'PROBLEM_REFORMULATION' else '')}
{('Also include rr, rj, rv for REVERSAL_AUDIT.' if broadcast.constraint == 'REVERSAL_AUDIT' else '')}
[/INST]"""
            repaired = _call_json_llm(
                self.llm,
                repair_prompt,
                max_tokens=max(128, self.max_tokens),
                temperature=0.0,
                schema=schema,
            )
            repaired_raw = repaired["choices"][0]["text"] if isinstance(repaired, dict) else str(repaired)
            try:
                data = _extract_json(repaired_raw)
                candidate = _candidate_from_data(
                    self.name, actions, data, broadcast, fixed_baseline, self.scenario_facts or {},
                    self.previous_recommendation_id,
                    self.previous_confidence,
                    self.assumption_status,
                    self.unsupported_assumption,
                    self.reversal_condition,
                )
                self.previous_recommendation_id = action_ids[actions.index(candidate.recommended_action)]
                self.previous_confidence = candidate.preference_strength
                self.assumption_status = candidate.assumption_status
                self.unsupported_assumption = candidate.unsupported_assumption
                self.reversal_condition = candidate.reversal_condition
                return candidate
            except (ValueError, json.JSONDecodeError) as repair_error:
                return _invalid_candidate(
                    self.name,
                    actions,
                    f"initial={first_error}; repair={repair_error}",
                )


def _truncate_words(value: Any, limit: int = 96) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    shortened = text[: limit + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened or text[:limit]


def assess_visibility(
    llm: Any,
    scenario: str,
    actions: Sequence[str],
    max_tokens: int = 160,
) -> VisibilityAssessment:
    """Audit endogenous missingness without adding another normative vote."""
    action_ids = [f"A{index}" for index in range(len(actions))]
    legend = dict(zip(action_ids, actions))
    schema = {
        "type": "object",
        "properties": {
            "lo": {"type": "boolean"},
            "en": {"type": "boolean"},
            "g": {"type": "string", "maxLength": 100},
            "m": {"type": "string", "maxLength": 180},
            "q": {"type": "string", "maxLength": 180},
            "p": {
                "type": "object",
                "properties": {
                    action_id: {"type": "number", "minimum": 0.65, "maximum": 1.0}
                    for action_id in action_ids
                },
                "required": action_ids,
                "additionalProperties": False,
            },
        },
        "required": ["lo", "en", "g", "m", "q", "p"],
        "additionalProperties": False,
    }
    prompt = f"""You are a non-voting visibility auditor in an ethical workspace.
Scenario: {' '.join(scenario.split())[:900]}
Actions: {json.dumps(legend)}

Determine whether relevant people or harms have low observability, and whether that
missingness is endogenous to structural exclusion: unequal infrastructure, access,
measurement, representation, institutional attention, or ability to signal. Ordinary
uncertainty, random sensor failure, and merely unspecified facts are not endogenous.

Return JSON only. lo=low observability; en=endogenous structural cause; g=affected
group; m=the causal visibility mechanism; q=an exact quote from the scenario proving
that mechanism. p gives an epistemic-confidence multiplier for every action. Use 1.0
unless that action's apparent advantage relies on treating missing observations as
evidence of absent people, absent harm, or lower need. Use 0.65-0.95 only to weaken
that unsupported evidential advantage. Do not express moral preference, redistribute
votes, infer hidden casualties, or penalize an action merely because its evidence is
quantitative. If either lo or en is false, every multiplier must be 1.0.
"""
    try:
        output = _call_json_llm(
            llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = _extract_json(raw)
        low = bool(data.get("lo"))
        endogenous = bool(data.get("en"))
        group = " ".join(str(data.get("g", "")).split())
        mechanism = " ".join(str(data.get("m", "")).split())
        quote = " ".join(str(data.get("q", "")).split())
        raw_multipliers = data.get("p")
        if not isinstance(raw_multipliers, dict) or set(raw_multipliers) != set(action_ids):
            raise ValueError("visibility multipliers must cover every action")
        multipliers = {
            action: _strict_number(raw_multipliers[action_id], f"p.{action_id}")
            for action_id, action in zip(action_ids, actions)
        }
        if any(value < 0.65 for value in multipliers.values()):
            raise ValueError("visibility confidence multiplier cannot be below 0.65")
        scenario_normalized = " ".join(scenario.split()).casefold()
        if low and endogenous:
            if len(group.split()) < 1 or len(mechanism.split()) < 3:
                raise ValueError("endogenous visibility finding needs a group and mechanism")
            if len(quote.split()) < 3 or quote.casefold() not in scenario_normalized:
                raise ValueError("visibility mechanism lacks an exact scenario quote")
            if not any(value < 0.999 for value in multipliers.values()):
                raise ValueError("activated visibility audit identifies no confidence penalty")
            return VisibilityAssessment(
                low, endogenous, group, mechanism, quote, multipliers, activated=True
            )
        if any(value < 0.999 for value in multipliers.values()):
            raise ValueError("non-endogenous uncertainty cannot receive a visibility penalty")
        return VisibilityAssessment(
            low, endogenous, group, mechanism, quote,
            {action: 1.0 for action in actions}, activated=False,
        )
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
        return VisibilityAssessment(
            False, False, "", "", "",
            {action: 1.0 for action in actions}, activated=False,
            valid=False, error=str(error),
        )


def _action_similarity(first: str, second: str) -> float:
    ignored = {"a", "an", "and", "in", "of", "the", "to", "with"}
    first_words = {word for word in re.findall(r"[a-z0-9]+", first.casefold()) if word not in ignored}
    second_words = {word for word in re.findall(r"[a-z0-9]+", second.casefold()) if word not in ignored}
    union = first_words | second_words
    return len(first_words & second_words) / len(union) if union else 1.0


def _feasible_actions(data: dict[str, Any], scenario: str = "") -> list[str]:
    actor = _truncate_words(data.get("actor"), 60)
    if not actor:
        raise ValueError("action set must identify one grounded decision-maker")
    sides = data.get("sides")
    if not isinstance(sides, dict) or not _truncate_words(sides.get("A")) or not _truncate_words(sides.get("B")):
        raise ValueError("ethical conflict must identify two materially different sides")
    raw_actions = data.get("actions", [])
    actions = []
    represented_sides = set()
    if isinstance(raw_actions, list):
        for item in raw_actions:
            if not isinstance(item, dict) or item.get("e") is not True:
                continue
            try:
                feasibility = _strict_number(item.get("f"), "action feasibility")
            except ValueError:
                continue
            position = str(item.get("p", "")).strip().upper()
            action = _truncate_words(item.get("a", ""))
            if (
                action
                and feasibility >= 0.65
                and position in {"SIDE_A", "SIDE_B"}
                and not re.search(
                    r"\b(?:compromise|balance[sd]?|hybrid|middle[- ]ground)\b",
                    action,
                    flags=re.IGNORECASE,
                )
                and action not in actions
                and all(_action_similarity(action, existing) < 0.8 for existing in actions)
            ):
                actions.append(action)
                represented_sides.add(position)
    actions = actions[:2]
    if len(actions) < 2:
        raise ValueError("fewer than two feasible, materially distinct actions")
    if not {"SIDE_A", "SIDE_B"}.issubset(represented_sides):
        raise ValueError("actions do not represent both sides of the ethical conflict")
    institutional_context = re.search(
        r"\b(?:government|authority|employer|school|hospital|legislature|policymaker|"
        r"official|institution|organization|company|community leader|public health agency)\b",
        scenario,
        flags=re.IGNORECASE,
    )
    institutional_action = re.search(
        r"\b(?:organize|provide education|set up|deploy|mandate|fine|fines|launch|fund|"
        r"clinic|program|incentive|exemption)\b",
        " ".join(actions),
        flags=re.IGNORECASE,
    )
    if institutional_action and not institutional_context:
        raise ValueError("actions invent an institutional policymaker or implementation program")
    return actions


def extract_explicit_actions(scenario: str) -> list[str]:
    """Extract a closed natural-language either/or choice without model generation."""
    cleaned = " ".join(scenario.split())
    match = re.search(
        r"\beither\s+(.+?)\s+or\s+(.+?)(?=[?.]|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        # Forced choices are often written as ``must <decision>: <action>, or
        # <target>``. The second alternative may elide the first action's verb,
        # as in ``must swerve: strike A, or B``.
        match = re.search(
            r"\bmust\s+[a-z][a-z -]{0,40}:\s*(.+?)\s*,\s*or\s+(.+?)(?=[?]|$)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if not match:
            return []
        first, second = (value.strip(" ,;:") for value in match.groups())
        leading_verb = re.match(r"^([a-z][a-z-]*)\s+", first, flags=re.IGNORECASE)
        second_starts_with_target = re.match(
            r"^(?:a|an|the|this|that|these|those|his|her|their)\b",
            second,
            flags=re.IGNORECASE,
        )
        if leading_verb and second_starts_with_target:
            second = f"{leading_verb.group(1)} {second}"
        match_groups = (first, second)
    else:
        match_groups = match.groups()
    actions = []
    for value in match_groups:
        action = value.strip(" ,;:")
        if action and action.lower() not in {item.lower() for item in actions}:
            actions.append(action[0].upper() + action[1:])
    return actions if len(actions) == 2 else []


def extract_allocation_actions(scenario: str) -> list[str]:
    """Recognize simple one-resource/two-recipient allocation questions."""
    cleaned = " ".join(scenario.split())
    if not re.search(r"\bwho should receive (?:it|the\s+\w+)\b", cleaned, re.IGNORECASE):
        return []
    resource_match = re.search(
        r"\b(?:has|have)\s+(?:only\s+)?one\s+([a-z][a-z -]{0,30}?)(?=\s+and\s+|\s+for\s+|\s+to\s+|[.,])",
        cleaned,
        re.IGNORECASE,
    )
    if not resource_match:
        return []
    resource = " ".join(resource_match.group(1).split())
    recipients = []
    for match in re.finditer(
        r"\b(?:a|an|the)\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})\s+(?=with\b|who\b)",
        cleaned,
        re.IGNORECASE,
    ):
        recipient = " ".join(match.group(1).lower().split())
        if recipient not in recipients and recipient not in {"hospital", "patient"}:
            recipients.append(recipient)
    if len(recipients) != 2:
        return []
    return [f"Give the {resource} to the {recipient}" for recipient in recipients]


def extract_acceptability_actions(scenario: str) -> list[str]:
    """Turn an abstract 'is it acceptable to X?' question into a policy choice."""
    cleaned = " ".join(scenario.split())
    match = re.search(
        r"\bis it (?:ever )?(?:(?:morally|ethically)\s+)?"
        r"(?:acceptable|okay|ok|right|permissible) to\s+(.+?)(?=[?]|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    proposed = match.group(1).strip(" ,;:.")[:100]
    if not proposed:
        return []
    affirmative = proposed[0].upper() + proposed[1:]
    if re.match(r"tell\s+(?:a\s+)?(?:small\s+|white\s+)?lie\b", proposed, re.IGNORECASE):
        alternative = "Tell the truth instead"
    else:
        alternative = f"Do not {proposed}"
    return [affirmative, alternative]


def extract_scenario_facts(scenario: str) -> dict[str, Any]:
    """Extract a deliberately small fact table used for contradiction checks."""
    cleaned = " ".join(scenario.lower().split())
    survival: dict[str, float] = {}
    for match in re.finditer(
        r"\b(?:a|an|the)\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})\s+with\s+(?:an?\s+)?(\d{1,3})%\s+(?:survival\s+)?chance",
        cleaned,
    ):
        survival[match.group(1)] = int(match.group(2)) / 100.0
    return {"survival_chance": survival} if survival else {}


def infer_testimony_baseline(
    llm: Any,
    specialist: str,
    testimony: str,
    actions: Sequence[str],
    max_tokens: int = 64,
) -> tuple[str, str]:
    action_ids = [f"A{index}" for index in range(len(actions))]
    legend = {action_id: action for action_id, action in zip(action_ids, actions)}
    schema = {
        "type": "object",
        "properties": {
            "b": {"type": "string", "enum": [*action_ids, "NONE"]},
            "w": {"type": "string"},
            "q": {
                "type": "string",
                "enum": ["DIRECT", "CONDITIONAL", "OUTSIDE_ACTION_SET"],
            },
        },
        "required": ["b", "w", "q"],
        "additionalProperties": False,
    }
    prompt = f"""[INST]
Framework: {specialist}
Original testimony: {' '.join(testimony.split())[:1000]}
Actions: {json.dumps(legend)}
Which listed action does the testimony directly recommend as the answer to the
stated dilemma? q=DIRECT only when it clearly selects one listed action without
requiring an unstated factual condition. q=CONDITIONAL when the selection depends
on missing facts. q=OUTSIDE_ACTION_SET when it evades the dilemma by recommending
a sequence, compromise, delay, different treatment, or third option. For either
non-DIRECT label, b must be NONE. Do not force the nearest action onto an evasive
or conditional conclusion.
Return JSON only: {{"b":"A0","w":"short evidence","q":"DIRECT"}}
[/INST]"""
    output = _call_json_llm(llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema)
    raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
    try:
        data = _extract_json(raw)
        baseline = str(data.get("b", "NONE")).strip().upper()
        reason = " ".join(str(data.get("w", "")).split())[:160]
        quality = str(data.get("q", "DIRECT")).strip().upper()
        if quality not in {"DIRECT", "CONDITIONAL", "OUTSIDE_ACTION_SET"}:
            return "NONE", "invalid baseline quality"
        if quality != "DIRECT":
            return "NONE", f"{quality.lower()}: {reason}"[:160]
        if baseline not in {*action_ids, "NONE"}:
            return "NONE", "invalid baseline ID"
        return baseline, reason
    except (ValueError, json.JSONDecodeError) as exc:
        return "NONE", f"baseline extraction failed: {exc}"[:160]


def propose_actions(llm: Any, scenario: str, max_tokens: int = 128) -> list[str]:
    explicit_actions = extract_explicit_actions(scenario)
    if explicit_actions:
        return explicit_actions
    allocation_actions = extract_allocation_actions(scenario)
    if allocation_actions:
        return allocation_actions
    acceptability_actions = extract_acceptability_actions(scenario)
    if acceptability_actions:
        return acceptability_actions

    action_schema = {
        "type": "object",
        "properties": {
            "actor": {"type": "string"},
            "sides": {
                "type": "object",
                "properties": {"A": {"type": "string"}, "B": {"type": "string"}},
                "required": ["A", "B"],
                "additionalProperties": False,
            },
            "actions": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "f": {"type": "number", "minimum": 0, "maximum": 1},
                        "e": {"type": "boolean"},
                        "p": {"type": "string", "enum": ["SIDE_A", "SIDE_B"]},
                    },
                    "required": ["a", "f", "e", "p"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actor", "sides", "actions"],
        "additionalProperties": False,
    }
    prompt = f"""[INST]
Scenario: {' '.join(scenario.split())[:1200]}
Identify the ONE decision-maker who directly faces the ethical conflict and the
two competing sides, even when the input is a statement rather than a question.
Do not silently replace an individual's choice with a government, institutional,
or community program. Use policies only when an institution is the stated actor.
All actions must be mutually exclusive choices available to the same actor and
at the same level of abstraction. Do not bundle implementation tactics.
Return ONLY JSON:
{{"actor":"grounded decision-maker","sides":{{"A":"first value or interest","B":"competing value or interest"}},
"actions":[{{"a":"short action","f":0.9,"e":true,"p":"SIDE_A"}}]}}
Give exactly 2 materially different actions: one SIDE_A and one SIDE_B. Do not
include a compromise, hybrid, balanced, conditional middle-ground, or third option.
Do not give several implementation methods pursuing the same side. f is feasibility (0 to 1). e is
true when the action is genuinely available without inventing facts. Do not invent
waiting, authorities, escape, rescue, or resources. Preserve genuinely closed choices.
[/INST]"""
    output = _call_json_llm(
        llm,
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,
        schema=action_schema,
    )
    raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
    try:
        return _feasible_actions(_extract_json(raw), scenario)
    except (ValueError, json.JSONDecodeError) as first_error:
        repair_prompt = f"""[INST]
Repair this action plan: {raw[:400]}
Return JSON only: {{"actor":"one decision-maker from the scenario","sides":{{"A":"first ethical side","B":"competing ethical side"}},
"actions":[{{"a":"short action","f":0.9,"e":true,"p":"SIDE_A"}},
{{"a":"opposing action","f":0.9,"e":true,"p":"SIDE_B"}}]}}
Keep feasible actions available without invented facts. Represent both sides.
Reject multiple tactics that all advance the same ethical position.
Do not turn an individual's dilemma into an institutional policy menu.
Return exactly the two opposing actions. Remove compromises and middle options;
the recurrent workspace may generate one later if disagreement warrants synthesis.
[/INST]"""
        repaired = _call_json_llm(
            llm,
            repair_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            schema=action_schema,
        )
        repaired_raw = repaired["choices"][0]["text"] if isinstance(repaired, dict) else str(repaired)
        try:
            return _feasible_actions(_extract_json(repaired_raw), scenario)
        except (ValueError, json.JSONDecodeError) as repair_error:
            raise ValueError(
                f"Action plan failed feasibility validation: initial={first_error}; "
                f"repair={repair_error}"
            ) from repair_error


def propose_synthesis(
    llm: Any,
    scenario: str,
    actions: Sequence[str],
    candidates: Sequence[CandidateChunk],
    broadcast: WorkspaceBroadcast,
    testimonies: dict[str, str],
    max_tokens: int = 128,
) -> SynthesisProposal:
    """Propose one testimony-grounded action and apply permissive safety checks."""
    source_names = sorted({
        candidate.specialist.lower()
        for candidate in candidates
        if candidate.schema_valid
    } & set(testimonies))
    constraints = sorted({candidate.constraint for candidate in candidates if candidate.schema_valid})
    source_recommendations = {
        candidate.recommended_action
        for candidate in candidates
        if candidate.schema_valid
        and candidate.specialist.lower() in source_names
        and candidate.recommended_action
    }
    if len(source_names) < 2 or len(constraints) < 2 or len(source_recommendations) < 2:
        return SynthesisProposal(
            "", [], [], 0.0, "", accepted=False,
            rejection_reason="insufficient disagreeing globally available evidence",
        )
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "maxLength": 120},
            "g": {"type": "array", "minItems": 2, "maxItems": 5, "items": {"type": "string", "enum": source_names}},
            "k": {"type": "array", "minItems": 2, "maxItems": 5, "items": {"type": "string", "enum": constraints}},
            "q": {"type": "array", "maxItems": 1, "items": {"type": "string"}},
            "f": {"type": "number", "minimum": 0, "maximum": 1},
            "x": {"type": "boolean"},
            "n": {"type": "boolean"},
            "w": {"type": "string"},
        },
        "required": ["a", "g", "k", "q", "f", "x", "n", "w"],
        "additionalProperties": False,
    }
    evidence = {
        candidate.specialist: {
            "constraint": candidate.constraint,
            "recommendation": candidate.recommended_action,
            "reason": candidate.rationale,
            "testimony": " ".join(testimonies.get(candidate.specialist, "").split())[:600],
        }
        for candidate in candidates
        if candidate.schema_valid and candidate.specialist in testimonies
    }
    prompt = f"""[INST]
Scenario: {' '.join(scenario.split())[:1200]}
Current actions: {json.dumps(list(actions))}
Globally available evidence: {json.dumps(evidence)}
Broadcast: {broadcast.compact()}
Propose ONE materially useful action only if it combines or resolves competing
constraints. Cite at least two specialists who currently recommend different
actions and at least two of their competing constraints. It must be executable
in the scenario, grounded in named evidence, and not invent time, helpers,
authorities, escape, personnel, facilities, tools, processes, or resources. List
the single indispensable new requirement in q, or use [] when none. A closed choice
may be refined but not evaded. Do not merely rename an existing action.
Propose the SMALLEST policy or action distinction that could bridge the conflict:
one grammatical clause of 3-18 words. Do not give examples, parentheses, named
administrative procedures, schedules, benchmarks, surveys, enforcement systems,
review mechanisms, implementation steps, or a package of several measures.
Stay at the scenario's level of abstraction and reuse its concepts. Do not turn a
broad concern such as harm, safety, welfare, liberty, fairness, or quality of life
into a new concrete subtype such as noise, traffic, surveillance, staffing, or a
numerical limit unless that subtype already appears in the scenario or testimony.
Prefer q=[]; declare a requirement only when its exact substance is already supported.
Return JSON only: {{"a":"action","g":["care","deontological"],"k":["CARE","DUTY"],"q":[],"f":0.8,"x":true,"n":true,"w":"why"}}
x means executable; n means non-evasive.
[/INST]"""
    output = _call_json_llm(llm, prompt, max_tokens=max_tokens, temperature=0.1, schema=schema)
    raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
    try:
        data = _extract_json(raw)
        proposal = SynthesisProposal(
            action=str(data.get("a", "")),
            grounded_in=list(data.get("g", [])) if isinstance(data.get("g"), list) else [],
            addressed_constraints=list(data.get("k", [])) if isinstance(data.get("k"), list) else [],
            feasibility=_strict_number(data.get("f"), "synthesis feasibility"),
            executable=data.get("x") is True,
            non_evasive=data.get("n") is True,
            rationale=str(data.get("w", "")),
            introduced_requirements=(list(data.get("q", [])) if isinstance(data.get("q"), list) else []),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        return SynthesisProposal("", [], [], 0.0, "", accepted=False, rejection_reason=f"malformed proposal: {exc}")

    reasons = []
    normalized = proposal.action.casefold().strip(" .")
    if not normalized or normalized in {action.casefold().strip(" .") for action in actions}:
        reasons.append("not a distinct action")
    if normalized and _scenario_closes_action_set(scenario):
        reasons.append("scenario explicitly closes the action set; a third action is unavailable")
    if len(proposal.grounded_in) < 2 or not set(proposal.grounded_in).issubset(set(source_names)):
        reasons.append("not grounded in available testimony")
    candidate_by_name = {
        candidate.specialist.lower(): candidate
        for candidate in candidates
        if candidate.schema_valid
    }
    grounded_recommendations = {
        candidate_by_name[name].recommended_action
        for name in proposal.grounded_in
        if name in candidate_by_name and candidate_by_name[name].recommended_action
    }
    if len(grounded_recommendations) < 2:
        reasons.append("grounding sources do not represent disagreeing recommendations")
    if len(proposal.addressed_constraints) < 2 or not set(proposal.addressed_constraints).issubset(set(constraints)):
        reasons.append("does not address at least two competing constraints")
    if proposal.feasibility < 0.55 or not proposal.executable:
        reasons.append("insufficiently feasible")
    if not proposal.non_evasive:
        reasons.append("evades rather than resolves the dilemma")
    action_word_count = len(re.findall(r"\b[\w'-]+\b", proposal.action))
    procedural_terms = re.findall(
        r"\b(?:benchmark(?:s)?|survey(?:s)?|referendum|enforcement|monitor(?:ing)?|"
        r"certification|implementation|schedule|timeline|review mechanism|"
        r"administrative process|license suspension|fact sheet(?:s)?)\b",
        normalized,
    )
    if (
        action_word_count < 3
        or action_word_count > 18
        or "(" in proposal.action
        or ")" in proposal.action
        or len(proposal.introduced_requirements) > 1
        or len(procedural_terms) >= 1
    ):
        reasons.append("over-engineered synthesis; use one minimal action clause")
    unsupported_escape = re.search(r"\b(?:call (?:the )?(?:police|authorities)|wait|flee|escape|obtain help|summon help)\b", normalized)
    support_text = f"{scenario} {' '.join(testimonies.values())}".casefold()
    if unsupported_escape and unsupported_escape.group(0) not in support_text:
        reasons.append("introduces an unsupported escape or helper")
    unsupported_requirements = [
        requirement
        for requirement in proposal.introduced_requirements
        if requirement.casefold() not in support_text
    ]
    resource_terms = re.findall(
        r"\b(?:mobile|van(?:s)?|nurse(?:s)?|staff(?:ed|ing)?|clinic(?:s)?|facility|facilities|"
        r"funding|equipment|technology|committee|counselor(?:s)?|doctor(?:s)?|expert(?:s)?|"
        r"mediator(?:s)?|police|authorities)\b",
        normalized,
    )
    unsupported_terms = sorted({term for term in resource_terms if term not in support_text})
    introduced_numbers = sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", normalized)))
    unsupported_numbers = [number for number in introduced_numbers if number not in support_text]
    if unsupported_requirements or unsupported_terms or unsupported_numbers:
        details = unsupported_requirements + unsupported_terms + unsupported_numbers
        reasons.append(f"introduces unsupported concrete requirements: {', '.join(details[:5])}")
    proposal.accepted = not reasons
    proposal.rejection_reason = "; ".join(reasons)
    return proposal


def generate_failure_condition(
    llm: Any,
    scenario: str,
    synthesis_action: str,
    original_actions: Sequence[str],
    max_tokens: int = 128,
) -> FailureCondition:
    """Turn an unresolved synthesis dependency into an explicit fallback question."""
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "string", "maxLength": 180},
            "f": {"type": "string", "maxLength": 180},
            "q": {"type": "string", "maxLength": 240},
        },
        "required": ["n", "f", "q"],
        "additionalProperties": False,
    }
    prompt = f"""[INST]
Scenario: {' '.join(scenario.split())[:1000]}
Admitted synthesis: {synthesis_action}
Original fallback actions: {json.dumps(list(original_actions[:2]))}
Identify the single most important condition the synthesis needs in order to work.
State its direct failure case, then ask one explicit ethical fallback question using
the original actions. Stay at the scenario's abstraction level. Do not invent a
deadline, probability, institution, resource, or new option.
Return JSON only: {{"n":"necessary condition","f":"that condition fails",
"q":"If the failure occurs, should the actor choose fallback A or fallback B?"}}
[/INST]"""
    try:
        output = _call_json_llm(
            llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = _extract_json(raw)
        necessary = " ".join(str(data.get("n", "")).split())
        failure = " ".join(str(data.get("f", "")).split())
        question = " ".join(str(data.get("q", "")).split())
        errors = []
        if len(necessary.split()) < 3:
            errors.append("necessary condition is too vague")
        if len(failure.split()) < 3 or failure.casefold() == necessary.casefold():
            errors.append("failure condition is too vague")
        if len(question.split()) < 6 or not question.endswith("?"):
            errors.append("contingency question is not explicit")
        mentioned_fallbacks = sum(
            1 for action in original_actions[:2]
            if any(word in question.casefold() for word in re.findall(r"[a-z]{4,}", action.casefold())[:3])
        )
        if mentioned_fallbacks < 2:
            errors.append("question does not connect both original fallback actions")
        return FailureCondition(
            synthesis_action,
            necessary,
            failure,
            question,
            valid=not errors,
            error="; ".join(errors),
        )
    except Exception as exc:
        return FailureCondition(
            synthesis_action, "", "", "", valid=False,
            error=f"contingency analysis unavailable: {exc}",
        )


def analyze_action_plan(
    llm: Any,
    scenario: str,
    actions: Sequence[str],
    target_action: str,
    broadcast: WorkspaceBroadcast,
    candidates: Sequence[CandidateChunk],
    activation_reason: str,
    max_tokens: int = 160,
) -> PlanningAssessment:
    """Assess execution without contributing a moral preference or policy score."""
    action_ids = [f"A{index}" for index in range(len(actions))]
    legend = {action_id: action for action_id, action in zip(action_ids, actions)}
    try:
        target_id = action_ids[list(actions).index(target_action)]
    except ValueError:
        return PlanningAssessment(
            target_action, activation_reason, 0.0, "", "", "",
            valid=False, error="target action is not in the workspace action set",
        )
    evidence = [
        {
            "specialist": candidate.specialist,
            "constraint": candidate.constraint,
            "unresolved": candidate.unresolved,
            "reason": candidate.rationale,
        }
        for candidate in candidates
        if candidate.schema_valid
    ]
    force_types = [
        "RIVALRY", "SUBSTITUTE_PATHS", "RESOURCE_SUPPLIERS",
        "BENEFICIARIES_ALLIES", "ENTRY_EXIT_PRESSURES",
        "INSTITUTIONAL_POWER", "TIME_ENERGY_BUDGET",
    ]
    schema = {
        "type": "object",
        "properties": {
            "f": {"type": "number", "minimum": 0, "maximum": 1},
            "n": {"type": "string", "maxLength": 180},
            "x": {"type": "string", "maxLength": 180},
            "b": {"type": "string", "enum": action_ids},
            "a": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
            "r": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
            "s": {"type": "array", "maxItems": 4, "items": {"type": "string", "enum": force_types}},
            "m": {"type": "boolean"},
            "g": {"type": "string", "maxLength": 240},
            "va": {"type": "boolean"},
            "vr": {"type": "string", "maxLength": 180},
        },
        "required": ["f", "n", "x", "b", "a", "r", "s", "m", "g", "va", "vr"],
        "additionalProperties": False,
    }
    prompt = f"""[INST]
You are a non-normative planning system. Do NOT decide which action is ethical.
Scenario: {' '.join(scenario.split())[:900]}
Actions: {json.dumps(legend)}
Current policy leader: {target_id}
Activation: {activation_reason}
Workspace: {broadcast.compact()}
Ethical constraints already raised: {json.dumps(evidence)}

Assess only whether and how {target_id} can be carried out by the stated actor.
Use an abstract strategic ecology: rivalry, substitute paths, resource suppliers,
beneficiaries or allies, entry/exit pressures, institutional power, and short- versus
long-term time/energy budgets. Include only force types actually relevant here.
Do not invent institutions, helpers, resources, deadlines, probabilities, or facts.
Treat every action explicitly offered by the scenario as available unless the scenario
itself makes access uncertain. Do not manufacture failures involving clinics, physicians,
authorization, cost, transport, eligibility, facilities, or appointments.
n is the most important necessary condition; x is its direct failure condition.
b is the best fallback among the EXISTING actions, not a new moral recommendation.
a lists actor/power constraints; r lists resource/time constraints. m is true only
when x is a material unresolved risk worth broadcasting to the ethical specialists.
g must be an exact, contiguous quote from the scenario or Workspace text that states
the implementation obstacle. If no such quote exists, m must be false. va states
whether fallback b remains physically executable after x has occurred; vr briefly
explains why. va must be false when x removes a capability, actor, authority, control,
or resource required by both the target and fallback. Ethical desirability is not
physical availability.
Return compact JSON only.
[/INST]"""
    try:
        output = _call_json_llm(llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema)
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = _extract_json(raw)
        feasibility = _strict_number(data.get("f"), "planning feasibility")
        necessary = " ".join(str(data.get("n", "")).split())
        failure = " ".join(str(data.get("x", "")).split())
        fallback_id = str(data.get("b", "")).strip().upper()
        if fallback_id not in action_ids:
            raise ValueError("planning fallback must be an existing action ID")
        if len(necessary.split()) < 3:
            raise ValueError("planning necessary condition is too vague")
        if len(failure.split()) < 3 or failure.casefold() == necessary.casefold():
            raise ValueError("planning failure condition is too vague")
        actor_constraints = data.get("a", [])
        resource_constraints = data.get("r", [])
        forces = data.get("s", [])
        if not all(isinstance(value, list) for value in (actor_constraints, resource_constraints, forces)):
            raise ValueError("planning constraint fields must be lists")
        material = data.get("m") is True
        grounded_evidence = " ".join(str(data.get("g", "")).split())
        fallback_available = data.get("va") is True
        fallback_availability_reason = " ".join(str(data.get("vr", "")).split())
        grounding_context = " ".join(
            (scenario, broadcast.contingency_question, broadcast.reformulation_context)
        ).casefold()
        if len(grounded_evidence.split()) < 2 or grounded_evidence.casefold() not in grounding_context:
            raise ValueError("planning obstacle must quote the scenario or workspace exactly")
        if material and not fallback_available:
            raise ValueError("material planning branch leaves no physically available fallback")
        if len(fallback_availability_reason.split()) < 3:
            raise ValueError("planning must explain fallback physical availability")
        grounded_text = " ".join((scenario, *actions)).casefold()
        assessment_text = " ".join(
            (necessary, failure, *(str(value) for value in actor_constraints),
             *(str(value) for value in resource_constraints))
        ).casefold()
        access_terms = (
            "clinic", "physician", "doctor", "appointment", "transport", "cost",
            "coverage", "eligibility", "facility", "authorization", "authorisation",
        )
        invented_access = sorted({
            term for term in access_terms
            if term in assessment_text and term not in grounded_text
        })
        if invented_access:
            raise ValueError(
                "planning introduced unsupported access conditions: "
                + ", ".join(invented_access)
            )
        broadcast_worthy = bool(
            material and fallback_available and (feasibility < 0.75 or fallback_id != target_id)
        )
        return PlanningAssessment(
            target_action=target_action,
            activation_reason=activation_reason,
            feasibility=feasibility,
            necessary_condition=necessary,
            failure_condition=failure,
            fallback=legend[fallback_id],
            actor_constraints=list(actor_constraints),
            resource_constraints=list(resource_constraints),
            strategic_forces=list(forces),
            broadcast_worthy=broadcast_worthy,
            grounded_evidence=grounded_evidence,
            fallback_available=fallback_available,
            fallback_availability_reason=fallback_availability_reason,
        )
    except Exception as exc:
        return PlanningAssessment(
            target_action, activation_reason, 0.0, "", "", "",
            valid=False, error=f"planning analysis unavailable: {exc}",
        )


def propose_problem_reformulation(
    llm: Any,
    scenario: str,
    actions: Sequence[str],
    candidates: Sequence[CandidateChunk],
    max_tokens: int = 256,
) -> ProblemReformulation:
    """Create a hypothetical boundary case without pretending its numbers are facts."""
    action_ids = [f"A{index}" for index in range(len(actions))]
    legend = {action_id: action for action_id, action in zip(action_ids, actions)}
    audit_evidence = {
        candidate.specialist: {
            "status": candidate.assumption_status,
            "assumption": candidate.unsupported_assumption,
            "reversal": candidate.reversal_condition,
            "framework_constraint": candidate.constraint,
        }
        for candidate in candidates
        if candidate.schema_valid
        and candidate.assumption_status in {"CONDITIONAL", "UNDERDETERMINED"}
    }
    if len(audit_evidence) < 2:
        return ProblemReformulation(
            [], [], "", "", "", [], accepted=False,
            rejection_reason="fewer than two audited specialists identified underdetermination",
        )
    fixed_facts = [
        " ".join(clause.split())[:180]
        for clause in re.split(r"(?<=[.!?;])\s+|\s+but\s+", scenario, flags=re.IGNORECASE)
        if re.search(
            r"\b(?:unknown|intact|must|only|cannot|without|will|transfers?|costs?|either)\b",
            clause,
            flags=re.IGNORECASE,
        )
    ][:8]
    schema = {
        "type": "object",
        "properties": {
            "u": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}},
            "o": {
                "type": "array", "minItems": 2, "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string", "enum": action_ids},
                        "k": {"type": "string"},
                        "d": {"type": "string", "enum": ["BENEFIT", "HARM"]},
                        "x": {"type": "string"},
                        "p": {"type": "number", "minimum": 0, "maximum": 1},
                        "m": {"type": "number", "exclusiveMinimum": 0},
                        "unit": {"type": "string"},
                        "h": {"type": "string"},
                    },
                    "required": ["a", "k", "d", "x", "p", "m", "unit", "h"],
                    "additionalProperties": False,
                },
            },
            "s": {"type": "string"},
            "t": {"type": "string"},
            "q": {"type": "string"},
            "g": {
                "type": "array", "minItems": 2, "maxItems": 8,
                "items": {"type": "string", "enum": sorted(audit_evidence)},
            },
            "fixed": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "string"}},
            "c": {
                "type": "array", "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "string"},
                        "v": {
                            "type": "object",
                            "properties": {action_id: {"type": "string"} for action_id in action_ids},
                            "required": action_ids,
                            "additionalProperties": False,
                        },
                        "e": {"type": "string"},
                        "fixed": {"type": "boolean"},
                    },
                    "required": ["n", "v", "e", "fixed"],
                    "additionalProperties": False,
                },
            },
            "changed": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
            "hyp": {"type": "boolean", "const": True},
        },
        "required": ["u", "o", "s", "t", "q", "g", "fixed", "c", "changed", "hyp"],
        "additionalProperties": False,
    }
    prompt = f"""[INST]
Scenario: {' '.join(scenario.split())[:900]}
Original actions: {json.dumps(legend)}
Consensus-audit evidence: {json.dumps(audit_evidence)}
Explicit scenario invariants detected by Python: {json.dumps(fixed_facts)}

Reformulate the UNDERDETERMINED question as one explicitly HYPOTHETICAL boundary
case where consensus should be difficult, not easy. Do not choose an action and do
not add a third action. Specify probabilities, positive magnitudes, units, and time
horizons for 2-6 consequences. Values are calibration probes, NEVER scenario facts.
Ground every consequence type in the scenario or audit evidence, while numbers may
be hypothetical. Preserve distinct units when the ethical tension is genuinely
non-commensurable (for example lives versus livelihoods); do not silently convert
everything into money or expected utility. Include consequences associated with
both A0 and A1. Probabilities must be explicit and may be 1.0 when the
thought experiment specifies a deterministic outcome.
For each outcome, k is a shared comparison dimension such as mortality,
employment, fair access, bodily harm, or resource loss. Mirrored outcomes must use
the same k and normalized unit for both actions; direction belongs only in d.

u=the missing variables; s=the condition at which consensus should switch or split;
t=the normative tension that remains even with the hypothetical numbers specified;
q=one question naming both actions and asking the specialists to reason at that
boundary; g=at least two audited specialists grounding the reformulation; hyp=true.
fixed=the scenario facts that may NOT change. c=categorical moral axes such as
consent, causal agency, intention, rights, acting/allowing, role obligation, or
fairness; each axis maps every action ID to its fixed value and explains its ethical
relevance. changed must be []; if a proposed probe needs consent, a volunteer,
compensation, therapy, helpers, or any other fact absent from the scenario, it is a
DIFFERENT dilemma and must not be proposed. Never treat a fixed fact as an unknown.
Return compact JSON only.
[/INST]"""
    try:
        output = _call_json_llm(llm, prompt, max_tokens=max_tokens, temperature=0.1, schema=schema)
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = _extract_json(raw)
        raw_outcomes = data.get("o", [])
        if not isinstance(raw_outcomes, list):
            raise ValueError("calibration outcomes must be a list")
        outcomes = []
        for item in raw_outcomes:
            if not isinstance(item, dict):
                raise ValueError("each calibration outcome must be an object")
            action_id = str(item.get("a", "")).strip().upper()
            if action_id not in action_ids:
                raise ValueError("calibration outcome references an unknown action")
            direction = str(item.get("d", "")).strip().upper()
            if direction not in {"BENEFIT", "HARM"}:
                raise ValueError("calibration direction must be BENEFIT or HARM")
            probability = _strict_number(item.get("p"), "calibration probability")
            magnitude = float(item.get("m", 0))
            if magnitude <= 0:
                raise ValueError("calibration magnitude must be positive")
            outcomes.append(CalibrationOutcome(
                legend[action_id], str(item.get("k", "")), direction,
                str(item.get("x", "")), probability,
                magnitude, str(item.get("unit", "")), str(item.get("h", "")),
            ))
        represented = {outcome.action for outcome in outcomes}
        if not set(actions).issubset(represented):
            raise ValueError("calibration must represent every original action")
        if any(
            not outcome.dimension or not outcome.description or not outcome.unit or not outcome.horizon
            for outcome in outcomes
        ):
            raise ValueError("calibration outcomes need descriptions, units, and horizons")
        grounded_in = list(data.get("g", [])) if isinstance(data.get("g"), list) else []
        unknowns = list(data.get("u", [])) if isinstance(data.get("u"), list) else []
        raw_axes = data.get("c", [])
        if not isinstance(raw_axes, list):
            raise ValueError("categorical axes must be a list")
        categorical_axes = []
        for item in raw_axes:
            if not isinstance(item, dict) or not isinstance(item.get("v"), dict):
                raise ValueError("categorical axis must map every action")
            values = item["v"]
            if set(values) != set(action_ids):
                raise ValueError("categorical axis must map exactly the original actions")
            categorical_axes.append(CategoricalAxis(
                name=str(item.get("n", "")),
                action_values={legend[action_id]: str(values[action_id]) for action_id in action_ids},
                ethical_relevance=str(item.get("e", "")),
                fixed_by_scenario=item.get("fixed") is True,
            ))
        declared_fixed = list(data.get("fixed", [])) if isinstance(data.get("fixed"), list) else []
        changed_fixed = list(data.get("changed", [])) if isinstance(data.get("changed"), list) else []
        proposal = ProblemReformulation(
            unknowns=unknowns,
            outcomes=outcomes,
            switch_condition=str(data.get("s", "")),
            residual_tension=str(data.get("t", "")),
            question=str(data.get("q", "")),
            grounded_in=grounded_in,
            fixed_facts=list(dict.fromkeys([*fixed_facts, *declared_fixed])),
            categorical_axes=categorical_axes,
            changed_fixed_facts=changed_fixed,
            hypothetical=data.get("hyp") is True,
        )
        dimension_units: dict[str, set[str]] = {}
        for outcome in outcomes:
            dimension_units.setdefault(outcome.dimension.casefold(), set()).add(outcome.unit.casefold())
        inconsistent_dimensions = [
            dimension for dimension, units_for_dimension in dimension_units.items()
            if len(units_for_dimension) > 1
        ]
        numeric_comparisons = []
        for dimension, units_for_dimension in dimension_units.items():
            if len(units_for_dimension) != 1:
                continue
            action_values = {
                action: sum(
                    (1.0 if outcome.direction == "BENEFIT" else -1.0)
                    * outcome.probability * outcome.magnitude
                    for outcome in outcomes
                    if outcome.dimension.casefold() == dimension and outcome.action == action
                )
                for action in actions
            }
            values = list(action_values.values())
            absolute_gap = max(values) - min(values)
            gross = sum(
                outcome.probability * outcome.magnitude
                for outcome in outcomes
                if outcome.dimension.casefold() == dimension
            )
            numeric_comparisons.append(NumericComparison(
                dimension=dimension,
                unit=next(iter(units_for_dimension)),
                action_values=action_values,
                absolute_gap=absolute_gap,
                relative_gap=(absolute_gap / gross if gross else 0.0),
            ))
        proposal.numeric_comparisons = numeric_comparisons
        reasons = []
        if not proposal.hypothetical:
            reasons.append("calibration was not explicitly hypothetical")
        if not proposal.unknowns:
            reasons.append("missing variables were not identified")
        if len(proposal.switch_condition.split()) < 5:
            reasons.append("switch condition is too vague")
        if len(proposal.residual_tension.split()) < 5:
            reasons.append("residual ethical tension is too vague")
        if len(proposal.question.split()) < 8 or not proposal.question.endswith("?"):
            reasons.append("boundary question is not explicit")
        if len(set(grounded_in) & set(audit_evidence)) < 2:
            reasons.append("reformulation lacks two audited grounding sources")
        if proposal.changed_fixed_facts:
            reasons.append("probe changes fixed scenario facts")
        unknown_text = " ".join(proposal.unknowns).casefold()
        for fact in fixed_facts:
            significant = {
                word for word in re.findall(r"[a-z]{5,}", fact.casefold())
                if word not in {"which", "their", "doing"}
            }
            if significant and len(significant & set(re.findall(r"[a-z]{5,}", unknown_text))) >= 2:
                reasons.append("probe treats an explicit scenario invariant as unknown")
                break
        variation_text = " ".join([
            *proposal.unknowns,
            *(outcome.description for outcome in proposal.outcomes),
            proposal.switch_condition,
            proposal.question,
        ]).casefold()
        support_text = scenario.casefold()
        unsupported_alternatives = [
            term for term in (
                "consent", "volunteer", "compensation", "therapy", "counseling",
                "helper", "authority", "support network",
            )
            if term in variation_text and term not in support_text
        ]
        if unsupported_alternatives:
            reasons.append(
                "probe introduces a different scenario: " + ", ".join(unsupported_alternatives)
            )
        normative_text = f"{scenario} {json.dumps(audit_evidence)}".casefold()
        needs_categorical_axis = re.search(
            r"\b(?:autonomy|consent|stranger|transfer|push|pull|redirect|means|"
            r"duty|rights?|fairness|intend|allowing|agency)\b",
            normative_text,
        )
        if needs_categorical_axis and not proposal.categorical_axes:
            reasons.append("normative conflict lacks a categorical moral axis")
        if any(not axis.name or not axis.ethical_relevance for axis in proposal.categorical_axes):
            reasons.append("categorical axis is incomplete")
        if inconsistent_dimensions:
            reasons.append(
                "shared dimensions use inconsistent units: " + ", ".join(inconsistent_dimensions)
            )
        near_numeric_boundary = any(
            comparison.relative_gap <= 0.25
            for comparison in proposal.numeric_comparisons
        )
        if not near_numeric_boundary and not proposal.categorical_axes:
            reasons.append("calibration has neither numeric parity nor categorical tension")
        comparison_text = "; ".join(
            f"{comparison.dimension}: "
            + ", ".join(
                f"{action_id}={comparison.action_values[action]:g} {comparison.unit}"
                for action_id, action in legend.items()
            )
            + f" (relative gap={comparison.relative_gap:.2f})"
            for comparison in proposal.numeric_comparisons
        )
        proposal.switch_condition = (
            f"Python-computed calibration: {comparison_text}. "
            + (
                "At least one numeric dimension is near parity; categorical axes may determine the split."
                if near_numeric_boundary
                else "No numeric dimension is near parity; any split must be explained by categorical axes."
            )
        )[:220]
        proposal.accepted = not reasons
        proposal.rejection_reason = "; ".join(reasons)
        return proposal
    except Exception as exc:
        return ProblemReformulation(
            [], [], "", "", "", [], accepted=False,
            rejection_reason=f"reformulation unavailable: {exc}",
        )
