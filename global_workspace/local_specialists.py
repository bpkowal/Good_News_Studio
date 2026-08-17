from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .action_identity import compile_action_identity
from .evidence_calibration import EvidenceCalibration
from .contingency_graph import compile_contingency_graph
from .middleware.claim_damping import (
    SPECULATIVE_EPISTEMIC_CAP,
    apply_symmetric_claim_damping,
)
from .models import CalibrationOutcome, CandidateChunk, CategoricalAxis, FailureCondition, NumericComparison, PlanningAssessment, ProblemReformulation, SynthesisProposal, TestimonyBaseline, WorkspaceBroadcast
from .structured_io import (
    call_json_llm as _call_json_llm,
    extract_json as _extract_json,
    number as _number,
    strict_number as _strict_number,
    structured_text_error,
)
from .visibility import assess_visibility
from .scenario_semantics import normalize_action_labels


FRAMEWORK_ROLES = {
    "utilitarian": "Compare typed consequences, expected harms, benefits, urgency, and reversibility.",
    "deontological": "Score duties, rights, entitlement, coercion, and universal rules.",
    "virtue": "Compare role, practical wisdom, character, virtues, vices, and habituation.",
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

_FRAMEWORK_CONSTRUCT_MARKERS = {
    "care": re.compile(
        r"\b(?:entrust\w*|depend\w*|trust\w*|vulnerab\w*|responsib\w*|"
        r"attent\w*|responsive\w*|relationship\w*|obligation\w*|"
        r"abandon\w*|caregiv\w*|interdepend\w*|need\w*)\b",
        re.IGNORECASE,
    ),
    "deontological": re.compile(
        r"\b(?:duty|duties|right\w*|autonom\w*|coerc\w*|universa\w*|"
        r"maxim\w*|categorical|person\w*\s+as\s+ends?|instrumentali[sz]\w*|"
        r"respect\w*|prohibit\w*|permissib\w*|required)\b",
        re.IGNORECASE,
    ),
    "rawlsian": re.compile(
        r"\b(?:least[- ]advantaged|worse[- ]off|worst[- ]off|basic\s+libert\w*|primary\s+goods?|"
        r"difference\s+principle|fair\s+equality|original\s+position|"
        r"veil\s+of\s+ignorance|justice\s+as\s+fairness|disadvantaged)\b",
        re.IGNORECASE,
    ),
    "virtue": re.compile(
        r"\b(?:virtue\w*|vice\w*|character\w*|practical\s+wisdom|phronesis|"
        r"flourish\w*|courage\w*|honest\w*|just\w*|temperan\w*|compassion\w*|"
        r"integrity|habitu\w*|exemplar\w*|role\w*|pruden\w*|humil\w*|"
        r"solidarit\w*|friendship|benevolen\w*|generos\w*|loyal\w*|"
        r"foresight|steward\w*|callous\w*|reckless\w*|hubris)\b",
        re.IGNORECASE,
    ),
}


def _framework_grounded_action_sections(
    testimony: str, action_ids: Sequence[str], marker: re.Pattern[str],
) -> set[str]:
    """Find action sections whose source testimony contains framework constructs.

    This checks the corpus-grounded testimony rather than requiring the baseline
    classifier to repeat an exact preferred synonym in its compact map.
    """
    normalized = normalize_action_labels(testimony)
    headings = list(re.finditer(
        r"(?im)^\s*(?:action|option)\s+(A\d+)\b[^\n]*",
        normalized,
    ))
    grounded: set[str] = set()
    allowed = set(action_ids)
    for index, heading in enumerate(headings):
        action_id = heading.group(1).upper()
        if action_id not in allowed:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(normalized)
        if marker.search(normalized[heading.start():end]):
            grounded.add(action_id)
    return grounded


def _construct_map_errors(
    specialist: str,
    actions: Sequence[str],
    action_map: dict[str, str],
    numerical_role: str,
    numerical_justification: str,
    *,
    recommended_action: str = "",
    rationale: str = "",
) -> list[str]:
    """Validate framework identity without deciding the substantive outcome."""
    marker = _FRAMEWORK_CONSTRUCT_MARKERS.get(specialist)
    errors: list[str] = []
    if marker is None:
        return errors
    if set(action_map) != set(actions):
        errors.append("framework map must assess every action")
    else:
        for action, assessment in action_map.items():
            if not marker.search(assessment):
                errors.append(f"framework map for {action} lacks framework-specific grounds")
    if numerical_role not in {"DECISIVE", "SECONDARY", "IRRELEVANT"}:
        errors.append("numerical role is missing")
    if _semantic_word_count(numerical_justification) < 3:
        errors.append("numerical role lacks justification")

    if specialist == "deontological":
        allowed_prefixes = ("REQUIRED:", "PERMISSIBLE:", "PROHIBITED:", "CONFLICTED:")
        if any(not value.upper().startswith(allowed_prefixes) for value in action_map.values()):
            errors.append("deontological map lacks an action-level duty verdict")
        if (
            recommended_action in action_map
            and action_map[recommended_action].upper().startswith("PROHIBITED:")
        ):
            errors.append("deontological recommendation selects its own prohibited action")
        if numerical_role == "DECISIVE" and not re.search(
            r"\b(?:scope|right\w*|duty|duties|universal\w*|person\w*|"
            r"violation\w*|imminent)\b",
            numerical_justification,
            re.IGNORECASE,
        ):
            errors.append("decisive numbers are not connected to a duty or right")
    elif specialist == "rawlsian":
        allowed_prefixes = ("IMPROVES:", "PRESERVES:", "WORSENS:", "UNCERTAIN:")
        if any(not value.upper().startswith(allowed_prefixes) for value in action_map.values()):
            errors.append("Rawlsian map lacks a comparative position verdict")
        rawls_numerical_ground = re.search(
            r"\b(?:least[- ]advantaged|worse[- ]off|worst[- ]off|position|primary\s+good|"
            r"basic\s+libert|comparab\w*|same\s+libert)\b",
            numerical_justification,
            re.IGNORECASE,
        )
        if numerical_role == "DECISIVE" and not rawls_numerical_ground:
            errors.append("decisive numbers are not tied to the least-advantaged position")
        if recommended_action in action_map and len(action_map) > 1:
            effect_rank = {
                "IMPROVES": 3, "PRESERVES": 2, "UNCERTAIN": 1, "WORSENS": 0,
            }
            effects = {
                action: assessment.partition(":")[0].strip().upper()
                for action, assessment in action_map.items()
            }
            selected_rank = effect_rank.get(effects.get(recommended_action, ""), -1)
            rival_ranks = [
                effect_rank.get(effect, -1)
                for action, effect in effects.items()
                if action != recommended_action
            ]
            discriminating_priority = re.search(
                r"\b(?:lexical\s+priorit\w*|basic\s+libert\w*\s+(?:over|before)|"
                r"fair\s+equality|difference\s+principle|more\s+sever\w*|"
                r"greater\s+impair\w*|primary\s+good\w*\s+(?:over|before))\b",
                rationale,
                re.IGNORECASE,
            )
            numerically_discriminating = bool(
                numerical_role == "DECISIVE" and rawls_numerical_ground
            )
            if rival_ranks and (
                selected_rank < max(rival_ranks)
                or selected_rank == max(rival_ranks)
            ) and not discriminating_priority and not numerically_discriminating:
                errors.append(
                    "Rawlsian preference lacks a stated difference in position or principle priority"
                )
    elif specialist == "care" and numerical_role == "DECISIVE" and not re.search(
        r"\b(?:comparab\w*|equivalent|equally|same relational|"
        r"similar depend\w*|no stronger relational)\b",
        numerical_justification,
        re.IGNORECASE,
    ):
        errors.append("decisive counts lack relational comparability")
    elif specialist == "virtue":
        allowed_prefixes = ("EXEMPLIFIES:", "MIXED:", "UNDERMINES:", "UNCERTAIN:")
        if any(not value.upper().startswith(allowed_prefixes) for value in action_map.values()):
            errors.append("virtue map lacks an action-level character verdict")
        if numerical_role == "DECISIVE" and not re.search(
            r"\b(?:practical\s+wisdom|phronesis|stakes|circumstance\w*|"
            r"flourish\w*|virtue\w*|vice\w*)\b",
            numerical_justification,
            re.IGNORECASE,
        ):
            errors.append("decisive numbers are not connected to practical wisdom")
    return list(dict.fromkeys(errors))


def _utilitarian_table_errors(
    actions: Sequence[str],
    table: dict[str, list[dict[str, Any]]],
    depends_on_unknown: bool,
    missing_comparison: str,
) -> list[str]:
    """Check consequence-accounting structure without inventing utilities."""
    errors: list[str] = []
    if set(table) != set(actions):
        errors.append("consequence table must assess every action")
    for action in actions:
        rows = table.get(action, [])
        identity = compile_action_identity(action)
        stated_polarities = {
            "BENEFIT" if consequence.polarity == "BENEFICIAL" else "HARM"
            for consequence in identity.consequences
        }
        if not rows:
            errors.append(f"consequence table for {action} is empty")
            continue
        for row in rows:
            if _semantic_word_count(row.get("outcome", "")) < 2:
                errors.append(f"consequence for {action} lacks an outcome")
            if _semantic_word_count(row.get("scope", "")) < 1:
                errors.append(f"consequence for {action} lacks affected scope")
            if row.get("direction") not in {"BENEFIT", "HARM"}:
                errors.append(f"consequence for {action} lacks benefit/harm direction")
            if row.get("support") not in {"STATED", "INFERRED", "UNKNOWN"}:
                errors.append(f"consequence for {action} lacks evidence status")
            elif (
                row.get("support") == "STATED"
                and stated_polarities
                and row.get("direction") not in stated_polarities
            ):
                errors.append(
                    f"stated consequence for {action} reverses its committed polarity"
                )
            if row.get("reversibility") not in {
                "REVERSIBLE", "IRREVERSIBLE", "UNKNOWN",
            }:
                errors.append(f"consequence for {action} lacks reversibility status")
            if not str(row.get("probability", "")).strip():
                errors.append(f"consequence for {action} lacks probability status")
            if not str(row.get("magnitude", "")).strip():
                errors.append(f"consequence for {action} lacks magnitude")
            if not str(row.get("duration", "")).strip():
                errors.append(f"consequence for {action} lacks duration")
    if depends_on_unknown and _semantic_word_count(missing_comparison) < 3:
        errors.append("underdetermined utility ranking omits the missing comparison")
    return list(dict.fromkeys(errors))

ALLOWED_UNRESOLVED = {
    "NONE",
    "VERIFY_FACTS",
    "CHECK_FEASIBILITY",
    "CLARIFY_SCENARIO",
    "RESOLVE_NORMATIVE_TENSION",
}

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
            # A short lexical root makes common derivations comparable without
            # a heavyweight NLP dependency: containment/contains, mutation/
            # mutating, and expropriation/expropriating should share anchors.
            values.add(token[:6])
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
        or assumption_status in {
            "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
        }
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
    re.compile(
        r"(?:\bfirst\s+action\s*\(\s*A0\s*\)|\b(?:action|option)\s+A0\b)"
        r".+?\bor\b.+?"
        r"(?:second\s+action\s*\(\s*A1\s*\)|(?:action|option)\s+A1)",
        re.I,
    ),
    re.compile(
        r"\b(?:action|option)\s+A0\b.+?\b(?:action|option)\s+A1\b",
        re.I,
    ),
)


def _scenario_closes_action_set(scenario: str) -> bool:
    """Recognize explicit closed-world dilemmas without relying on topic words."""
    from .scenario_semantics import normalize_action_labels

    normalized = " ".join(normalize_action_labels(scenario).split())
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
    previous_context: str = "",
    prior_assumption_status: str = "NOT_AUDITED",
    prior_unsupported_assumption: str = "",
    prior_reversal_condition: str = "",
    evidence_calibration: EvidenceCalibration | None = None,
    landscape_verifier: Callable[[str, Sequence[str], dict[str, str], Sequence[str]], list[str]] | None = None,
    baseline_status: str = "UNAVAILABLE",
    baseline_provisional_id: str = "NONE",
    baseline_condition: str = "",
    scenario_text: str = "",
) -> CandidateChunk:
    action_ids = [f"A{index}" for index in range(len(actions))]
    raw_scores = data.get("scores")
    if not isinstance(raw_scores, dict) or set(raw_scores) != set(action_ids):
        raise ValueError(f"scores must contain exactly {action_ids}")
    id_scores = {
        action_id: _strict_number(raw_scores[action_id], f"scores.{action_id}")
        for action_id in action_ids
    }
    reported_ordered_scores = sorted(id_scores.values(), reverse=True)
    reported_preference_strength = max(
        0.0,
        min(
            1.0,
            reported_ordered_scores[0] - reported_ordered_scores[1]
            if len(reported_ordered_scores) > 1
            else reported_ordered_scores[0],
        ),
    )
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
    allowed_baseline_statuses = {
        "DIRECT", "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
        "OUTSIDE_ACTION_SET", "UNAVAILABLE",
    }
    baseline_status = str(baseline_status).strip().upper()
    if baseline_status not in allowed_baseline_statuses:
        baseline_status = "DIRECT" if baseline_id != "NONE" else "UNAVAILABLE"
    if baseline_status == "UNAVAILABLE" and baseline_id != "NONE":
        # Backward compatibility for callers created before typed baselines.
        baseline_status = "DIRECT"
    if baseline_provisional_id not in {*action_ids, "NONE"}:
        baseline_provisional_id = "NONE"
    baseline_reference_id = (
        baseline_id if baseline_status == "DIRECT" else baseline_provisional_id
    )
    if baseline_status == "NORMATIVELY_CONTESTED":
        alignment = (
            "CONTESTED_SUPPORTS_PROVISIONAL"
            if baseline_reference_id != "NONE" and recommended_id == baseline_reference_id
            else "CONTESTED_RECONSIDERS_PROVISIONAL"
            if baseline_reference_id != "NONE"
            else "NORMATIVELY_CONTESTED"
        )
    elif baseline_status == "UNDERDETERMINED":
        alignment = "UNRESOLVED"
    elif baseline_status == "CONDITIONAL" and baseline_reference_id == "NONE":
        alignment = "UNRESOLVED"
    elif baseline_status == "CONDITIONAL":
        alignment = (
            "CONDITIONAL_SUPPORTS"
            if recommended_id == baseline_reference_id
            else "CONDITIONAL_RECONSIDERS"
        )
    elif baseline_id == "NONE":
        alignment = "UNCLEAR"
    elif recommended_id == baseline_id:
        alignment = "SUPPORTS"
    else:
        alignment = "RECONSIDERS"
    if (
        broadcast.constraint == "OPEN_DELIBERATION"
        and baseline_status == "DIRECT"
        and baseline_id != "NONE"
        and recommended_id != baseline_id
    ):
        raise ValueError("the initial recommendation must match the source-testimony baseline")

    contingency_choice = ""
    contingency_justification = ""
    contingency_valid = True
    contingency_error = ""
    if broadcast.constraint == "CONTINGENCY_REVIEW":
        fallback_actions = list(broadcast.contingency_fallback_actions)
        fallback_ids = [
            action_ids[actions.index(action)]
            for action in fallback_actions
            if action in actions
        ]
        choice_id = str(data.get("cr", "")).strip().upper()
        contingency_justification = " ".join(str(data.get("cj", "")).split())
        if len(fallback_actions) != 2 or len(fallback_ids) != 2:
            contingency_error = "contingency broadcast lacks two canonical fallback actions"
        elif choice_id not in fallback_ids:
            contingency_error = "contingency response did not choose a fallback action"
        elif recommended_id != choice_id:
            contingency_error = "recommendation does not match the contingency choice"
        elif len(contingency_justification.split()) < 3:
            contingency_error = "contingency response did not explain the conditional choice"
        if contingency_error:
            raise ValueError(contingency_error)
        contingency_choice = actions[action_ids.index(choice_id)]

    rationale = " ".join(str(data.get("w", data.get("why", ""))).split())
    if len(rationale.split()) < 2:
        raise ValueError("why must contain at least two words")
    framework_action_map: dict[str, str] = {}
    framework_numerical_role = "NOT_APPLICABLE"
    framework_numerical_justification = ""
    framework_grounding_penalty = 0.0
    framework_validation_errors: list[str] = []
    utilitarian_consequence_table: dict[str, list[dict[str, Any]]] = {}
    utilitarian_depends_on_unknown = False
    utilitarian_missing_comparison = ""
    utilitarian_ledger_proposal: dict[str, Any] = {}
    deontological_ledger_proposal: dict[str, Any] = {}
    utilitarian_fields_present = specialist == "utilitarian" and any(
        key in data for key in ("ct", "cd", "cm")
    )
    if utilitarian_fields_present:
        raw_table = data.get("ct", {})
        if isinstance(raw_table, dict):
            for action_id, action in zip(action_ids, actions):
                raw_rows = raw_table.get(action_id, [])
                rows: list[dict[str, Any]] = []
                if isinstance(raw_rows, list):
                    for raw_row in raw_rows[:4]:
                        if not isinstance(raw_row, dict):
                            continue
                        rows.append({
                            "outcome": " ".join(str(raw_row.get("o", "")).split())[:120],
                            "scope": " ".join(str(raw_row.get("s", "")).split())[:80],
                            "direction": str(raw_row.get("d", "")).strip().upper(),
                            "probability": " ".join(str(raw_row.get("p", "UNKNOWN")).split())[:24],
                            "magnitude": " ".join(str(raw_row.get("m", "UNKNOWN")).split())[:60],
                            "duration": " ".join(str(raw_row.get("h", "UNKNOWN")).split())[:40],
                            "reversibility": str(raw_row.get("rv", "UNKNOWN")).strip().upper(),
                            "support": str(raw_row.get("g", "UNKNOWN")).strip().upper(),
                        })
                utilitarian_consequence_table[action] = rows
        utilitarian_depends_on_unknown = data.get("cd") is True
        utilitarian_missing_comparison = " ".join(str(data.get("cm", "")).split())[:180]
        table_errors = _utilitarian_table_errors(
            actions,
            utilitarian_consequence_table,
            utilitarian_depends_on_unknown,
            utilitarian_missing_comparison,
        )
        framework_validation_errors.extend(table_errors)
        if table_errors:
            framework_grounding_penalty = 0.35
        else:
            utilitarian_ledger_proposal = {
                "actions": [
                    {
                        "action_id": action_id,
                        "consequences": [dict(row) for row in utilitarian_consequence_table[action]],
                    }
                    for action_id, action in zip(action_ids, actions)
                ]
            }

    construct_key = "rm" if specialist == "care" else "fm"
    construct_fields_present = specialist in _FRAMEWORK_CONSTRUCT_MARKERS and any(
        key in data for key in (construct_key, "nr", "np")
    )
    if construct_fields_present:
        raw_action_map = data.get(construct_key, {})
        if isinstance(raw_action_map, dict):
            framework_action_map = {
                action: " ".join(str(raw_action_map.get(action_id, "")).split())[:180]
                for action_id, action in zip(action_ids, actions)
            }
        framework_numerical_role = str(data.get("nr", "")).strip().upper()
        framework_numerical_justification = " ".join(
            str(data.get("np", "")).split()
        )[:180]
        construct_errors = _construct_map_errors(
            specialist,
            actions,
            framework_action_map,
            framework_numerical_role,
            framework_numerical_justification,
            recommended_action=actions[action_ids.index(recommended_id)],
            rationale=rationale,
        )
        framework_validation_errors.extend(construct_errors)
        if construct_errors:
            framework_grounding_penalty = 0.35

        framework_marker = _FRAMEWORK_CONSTRUCT_MARKERS[specialist]
        aggregate_shorthand = re.search(
            r"\b(?:more lives|fewer deaths|greatest number|most people|aggregate|"
            r"larger total|higher total)\b",
            rationale,
            re.IGNORECASE,
        )
        if aggregate_shorthand and not framework_marker.search(rationale):
            replacement = framework_action_map.get(
                actions[action_ids.index(recommended_id)], ""
            )
            if replacement:
                rationale = replacement
            if specialist in {"deontological", "rawlsian", "virtue"}:
                framework_grounding_penalty = max(framework_grounding_penalty, 0.35)

    # Preserve the old Care-specific trace fields while exposing the common
    # construct state for all framework delegates.
    care_relational_map = framework_action_map if specialist == "care" else {}
    care_numerical_role = (
        framework_numerical_role if specialist == "care" else "NOT_APPLICABLE"
    )
    care_numerical_justification = (
        framework_numerical_justification if specialist == "care" else ""
    )
    care_grounding_penalty = (
        framework_grounding_penalty if specialist == "care" else 0.0
    )
    rawls_position_proposal: dict[str, Any] = {}
    virtue_character_proposal: dict[str, Any] = {}
    if specialist == "rawlsian" and "rp" in data:
        raw_positions = data.get("rp", {})
        ranking_basis = str(data.get("rb", "")).strip().upper()
        raw_liberty_status = data.get("lc", {})
        position_errors: list[str] = []
        positions: list[dict[str, Any]] = []
        if ranking_basis not in {
            "LEXICAL_BASIC_LIBERTY", "FAIR_EQUALITY_OPPORTUNITY",
            "MAXIMIN_PRIMARY_GOODS", "DIFFERENCE_PRINCIPLE",
            "ORIGINAL_POSITION_PUBLIC_RULE", "UNRESOLVED",
        }:
            position_errors.append("Rawlsian ledger lacks a valid principle-ranking basis")
        liberty_status = {
            action_id: str(raw_liberty_status.get(action_id, "")).strip().upper()
            for action_id in action_ids
        } if isinstance(raw_liberty_status, dict) else {}
        if (
            not isinstance(raw_liberty_status, dict)
            or set(raw_liberty_status) != set(action_ids)
        ):
            position_errors.append("Rawlsian liberty comparison must cover every action")
        elif any(status not in {
            "SATISFIED", "INFRINGED", "CONFLICTED", "UNKNOWN",
        } for status in liberty_status.values()):
            position_errors.append("Rawlsian liberty comparison has an invalid status")
        selected_liberty = liberty_status.get(recommended_id, "UNKNOWN")
        rival_liberties = [
            status for action_id, status in liberty_status.items()
            if action_id != recommended_id
        ]
        if (
            selected_liberty == "INFRINGED"
            and "SATISFIED" in rival_liberties
        ):
            position_errors.append(
                "Rawlsian recommendation crosses lexical liberty priority without a conflict"
            )
        if (
            ranking_basis in {
                "FAIR_EQUALITY_OPPORTUNITY", "MAXIMIN_PRIMARY_GOODS",
                "DIFFERENCE_PRINCIPLE",
            }
            and any(status in {"INFRINGED", "CONFLICTED", "UNKNOWN"}
                    for status in liberty_status.values())
        ):
            position_errors.append(
                "Rawlsian lower-order ranking was applied before basic-liberty status was resolved"
            )
        if ranking_basis == "UNRESOLVED" and (
            str(data.get("ss", "SELECTED")).strip().upper() != "PROVISIONAL"
            or data.get("cc") is not False
            or data.get("esa") is not False
        ):
            position_errors.append(
                "unresolved Rawlsian principle ranking must remain provisional"
            )
        if not isinstance(raw_positions, dict) or set(raw_positions) != set(action_ids):
            position_errors.append("Rawlsian position ledger must cover every action")
        else:
            for action_id, action in zip(action_ids, actions):
                raw_position = raw_positions.get(action_id, {})
                if not isinstance(raw_position, dict):
                    position_errors.append(f"Rawlsian position for {action_id} is not an object")
                    continue
                effect = str(raw_position.get("e", "")).strip().upper()
                compared_to = str(raw_position.get("ca", "")).strip().upper()
                dimension = str(raw_position.get("d", "")).strip().upper()
                basis = str(raw_position.get("b", "")).strip().upper()
                group = " ".join(str(raw_position.get("g", "")).split())[:100]
                reason = " ".join(str(raw_position.get("rs", "")).split())[:180]
                allowed_dimensions = {
                    "BASIC_LIBERTY", "OPPORTUNITY", "INCOME_WEALTH",
                    "POWERS_OFFICES", "SELF_RESPECT", "BASIC_INTEREST_SECURITY",
                    "OTHER_PRIMARY_GOOD", "UNKNOWN",
                }
                if effect not in {"IMPROVES", "PRESERVES", "WORSENS", "UNCERTAIN"}:
                    position_errors.append(f"Rawlsian position for {action_id} has invalid effect")
                if compared_to not in set(action_ids) - {action_id}:
                    position_errors.append(f"Rawlsian position for {action_id} lacks its rival")
                if dimension not in allowed_dimensions:
                    position_errors.append(f"Rawlsian position for {action_id} has invalid dimension")
                if basis not in {"ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"}:
                    position_errors.append(f"Rawlsian position for {action_id} has invalid evidence basis")
                if _semantic_word_count(group) < 1 or _semantic_word_count(reason) < 2:
                    position_errors.append(f"Rawlsian position for {action_id} lacks group or reason")
                map_effect = framework_action_map.get(action, "").partition(":")[0].upper()
                if map_effect and effect and map_effect != effect:
                    position_errors.append(
                        f"Rawlsian position for {action_id} conflicts with its framework map"
                    )
                positions.append({
                    "action_id": action_id,
                    "group": group,
                    "dimension": dimension,
                    "effect": effect,
                    "compared_to_action_id": compared_to,
                    "evidence_basis": basis,
                    "reason": reason,
                })
        if ranking_basis == "DIFFERENCE_PRINCIPLE":
            selected_position = next(
                (item for item in positions if item.get("action_id") == recommended_id),
                {},
            )
            if selected_position.get("dimension") not in {
                "INCOME_WEALTH", "POWERS_OFFICES",
            }:
                position_errors.append(
                    "difference principle was applied outside social or economic inequality"
                )
        framework_validation_errors.extend(position_errors)
        if position_errors:
            framework_grounding_penalty = max(framework_grounding_penalty, 0.35)
        else:
            rawls_position_proposal = {
                "ranking_basis": ranking_basis,
                "liberty_status": liberty_status,
                "positions": positions,
            }

    if specialist == "virtue" and "vl" in data:
        raw_assessments = data.get("vl", {})
        ranking_basis = str(data.get("vb", "")).strip().upper()
        virtue_errors: list[str] = []
        assessments: list[dict[str, Any]] = []
        if ranking_basis not in {
            "PRACTICAL_WISDOM", "ROLE_FIDELITY", "FLOURISHING",
            "EXEMPLAR_REASONING", "UNRESOLVED",
        }:
            virtue_errors.append("virtue ledger lacks a valid practical ranking basis")
        if not isinstance(raw_assessments, dict) or set(raw_assessments) != set(action_ids):
            virtue_errors.append("virtue ledger must cover every action")
        else:
            for action_id, action in zip(action_ids, actions):
                raw_assessment = raw_assessments.get(action_id, {})
                if not isinstance(raw_assessment, dict):
                    virtue_errors.append(f"virtue assessment for {action_id} is not an object")
                    continue
                verdict = str(raw_assessment.get("v", "")).strip().upper()
                role = " ".join(str(raw_assessment.get("r", "")).split())[:100]
                virtues = " ".join(str(raw_assessment.get("vs", "")).split())[:120]
                vice = " ".join(str(raw_assessment.get("x", "")).split())[:120]
                circumstance = " ".join(str(raw_assessment.get("c", "")).split())[:140]
                basis = str(raw_assessment.get("g", "")).strip().upper()
                reason = " ".join(str(raw_assessment.get("rs", "")).split())[:180]
                if verdict not in {"EXEMPLIFIES", "MIXED", "UNDERMINES", "UNCERTAIN"}:
                    virtue_errors.append(f"virtue assessment for {action_id} has invalid verdict")
                if basis not in {"ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"}:
                    virtue_errors.append(f"virtue assessment for {action_id} has invalid evidence basis")
                if min(map(_semantic_word_count, (role, virtues, vice, circumstance, reason))) < 1:
                    virtue_errors.append(f"virtue assessment for {action_id} is incomplete")
                map_verdict = framework_action_map.get(action, "").partition(":")[0].upper()
                if map_verdict and verdict and map_verdict != verdict:
                    virtue_errors.append(
                        f"virtue assessment for {action_id} conflicts with its framework map"
                    )
                assessments.append({
                    "action_id": action_id,
                    "verdict": verdict,
                    "actor_role": role,
                    "virtues": virtues,
                    "vice_risk": vice,
                    "circumstance": circumstance,
                    "evidence_basis": basis,
                    "reason": reason,
                })
        if ranking_basis == "UNRESOLVED" and (
            str(data.get("ss", "SELECTED")).strip().upper() != "PROVISIONAL"
            or data.get("cc") is not False
        ):
            virtue_errors.append("unresolved virtue ranking must remain provisional")
        framework_validation_errors.extend(virtue_errors)
        if virtue_errors:
            framework_grounding_penalty = max(framework_grounding_penalty, 0.35)
        else:
            virtue_character_proposal = {
                "ranking_basis": ranking_basis,
                "assessments": assessments,
            }

    if specialist == "deontological" and "dp" in data:
        raw_assessments = data.get("dp", {})
        duty_errors: list[str] = []
        assessments: list[dict[str, Any]] = []
        if not isinstance(raw_assessments, dict) or set(raw_assessments) != set(action_ids):
            duty_errors.append("Deontological duty ledger must cover every action")
        else:
            for action_id, action in zip(action_ids, actions):
                raw_assessment = raw_assessments.get(action_id, {})
                if not isinstance(raw_assessment, dict):
                    duty_errors.append(f"Deontological assessment for {action_id} is not an object")
                    continue
                verdict = str(raw_assessment.get("v", "")).strip().upper()
                relation = str(raw_assessment.get("rel", "")).strip().upper()
                norm_kind = str(raw_assessment.get("k", "")).strip().upper()
                basis = str(raw_assessment.get("g", "")).strip().upper()
                norm = " ".join(str(raw_assessment.get("n", "")).split())[:100]
                bearer = " ".join(str(raw_assessment.get("b", "")).split())[:80]
                party = " ".join(str(raw_assessment.get("p", "")).split())[:100]
                competing = " ".join(str(raw_assessment.get("cn", "NONE")).split())[:100]
                reason = " ".join(str(raw_assessment.get("rs", "")).split())[:180]
                if verdict not in {"REQUIRED", "PERMISSIBLE", "PROHIBITED", "CONFLICTED"}:
                    duty_errors.append(f"Deontological assessment for {action_id} has invalid verdict")
                if relation not in {"SATISFIES", "CONSISTENT", "VIOLATES", "CONFLICTS", "UNCERTAIN"}:
                    duty_errors.append(f"Deontological assessment for {action_id} has invalid relation")
                if norm_kind not in {
                    "DUTY", "RIGHT", "AUTONOMY", "UNIVERSAL_LAW",
                    "RESPECT_PERSONS", "OTHER", "UNKNOWN",
                }:
                    duty_errors.append(f"Deontological assessment for {action_id} has invalid norm kind")
                if basis not in {"ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"}:
                    duty_errors.append(f"Deontological assessment for {action_id} has invalid evidence basis")
                if any(_semantic_word_count(value) < 1 for value in (norm, bearer, party, reason)):
                    duty_errors.append(f"Deontological assessment for {action_id} lacks required terms")
                map_verdict = framework_action_map.get(action, "").partition(":")[0].upper()
                if map_verdict and verdict and map_verdict != verdict:
                    duty_errors.append(
                        f"Deontological assessment for {action_id} conflicts with its framework map"
                    )
                assessments.append({
                    "action_id": action_id,
                    "verdict": verdict,
                    "norm_kind": norm_kind,
                    "norm": norm,
                    "relation": relation,
                    "duty_bearer": bearer,
                    "protected_party": party,
                    "competing_norm": competing or "NONE",
                    "evidence_basis": basis,
                    "reason": reason,
                })
        framework_validation_errors.extend(duty_errors)
        if duty_errors:
            framework_grounding_penalty = max(framework_grounding_penalty, 0.35)
        else:
            deontological_ledger_proposal = {"assessments": assessments}

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
    claim_direction_damped = False
    calibration_tier = "NOT_APPLICABLE"
    calibration_reason = ""
    calibration_retention = 1.0
    calibration_epistemic_cap = SPECULATIVE_EPISTEMIC_CAP
    if evidence_basis == "UNSTATED_FACTS":
        if len(speculative_claim.split()) < 3:
            raise ValueError("unstated factual reasoning must identify the speculative claim")
        if evidence_calibration is not None:
            calibration_tier = evidence_calibration.tier
            calibration_reason = evidence_calibration.reason
            calibration_retention = evidence_calibration.direction_retention
            calibration_epistemic_cap = evidence_calibration.epistemic_cap
        else:
            calibration_tier = "DECISION_CRITICAL"
            calibration_retention = 0.55
        if calibration_tier == "ENTAILED":
            # The separate audit found that the purported speculation is part
            # of the stated causal setup. Do not retain a verification marker
            # that arose solely from that misclassification.
            evidence_basis = "STATED_FACTS"
            if unresolved == "VERIFY_FACTS":
                unresolved = "NONE"
        damping = apply_symmetric_claim_damping(
            id_scores,
            evidence_basis,
            unresolved,
            retention=calibration_retention,
        )
        id_scores = damping.scores
        unresolved = damping.unresolved
        claim_direction_damped = damping.applied
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
    raw_graph_update = data.get(
        "gu", {"operation": "NONE", "from_action": "NONE", "to_action": "NONE", "clauses": []}
    )
    graph_update: dict[str, Any] = (
        dict(raw_graph_update) if isinstance(raw_graph_update, dict)
        else {"operation": "INVALID", "clauses": []}
    )
    # Keep graph references as run-local canonical IDs until the transaction
    # layer commits them. Human-readable labels are presentation metadata and
    # must not become graph identity or contaminate permutation tests.
    for field_name in ("from_action", "to_action"):
        graph_update[field_name] = str(
            graph_update.get(field_name, "NONE")
        ).strip().upper()
    graph_clauses = graph_update.get("clauses", [])
    if isinstance(graph_clauses, list):
        for clause in graph_clauses:
            if isinstance(clause, dict):
                clause["affected_action"] = str(
                    clause.get("affected_action", "NONE")
                ).strip().upper()
    raw_ev = data.get("ev", {})
    expected_values = {
        actions[action_ids.index(action_id)]: {
            "value": float(value.get("value", 0.0)),
            "unit": str(value.get("unit", "")).upper(),
            "direction": str(value.get("direction", "")).upper(),
            "grounded": bool(value.get("grounded", False)),
        }
        for action_id, value in raw_ev.items()
        if action_id in action_ids and isinstance(value, dict)
    } if isinstance(raw_ev, dict) else {}
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
    review_valid = True
    review_error = ""
    if broadcast.constraint == "REVERSAL_AUDIT":
        review_response = str(data.get("rr", "")).strip().upper()
        review_justification = " ".join(str(data.get("rj", "")).split())
        revised_reversal = " ".join(str(data.get("rv", "")).split())
        if review_response not in {"ACCEPT", "REVISE", "REJECT"}:
            review_error = "reversal review did not classify the challenge"
        elif text_error := structured_text_error(review_justification):
            review_error = f"reversal review justification {text_error}"
        elif review_response == "REVISE" and (
            text_error := structured_text_error(revised_reversal)
        ):
            review_error = f"revised reversal condition {text_error}"
        if review_error:
            # A malformed audit sub-answer must not erase an otherwise valid
            # ethical evaluation. Preserve the vote, but exclude this delegate
            # from the audit tally and expose the bounded failure in the trace.
            review_valid = False
            review_response = "NOT_TESTED"
            revised_reversal = ""
    visibility_response = "NOT_TESTED"
    visibility_justification = ""
    visibility_harm_revision = "NONE"
    visibility_magnitude_status = "NOT_APPLICABLE"
    visibility_magnitude_overreach = False
    if broadcast.constraint == "VISIBILITY_AUDIT":
        visibility_response = str(data.get("vp", "")).strip().upper()
        visibility_justification = " ".join(str(data.get("vj", "")).split())
        visibility_harm_revision = str(data.get("vh", "")).strip().upper()
        visibility_magnitude_status = str(data.get("vm", "UNKNOWN")).strip().upper()
        if visibility_response not in {"ACCEPT", "QUALIFY", "REJECT"}:
            raise ValueError("visibility response must ACCEPT, QUALIFY, or REJECT")
        if len(visibility_justification.split()) < 3:
            raise ValueError("visibility response must explain its reasoning")
        if visibility_harm_revision not in {"UPWARD", "DOWNWARD", "UNCHANGED"}:
            raise ValueError("visibility response must state the harm revision direction")
        if visibility_magnitude_status not in {"UNKNOWN", "GROUNDED_BOUNDED"}:
            visibility_magnitude_status = "UNKNOWN"
        magnitude_language = re.search(
            r"\b(?:far below|unlikely to|cannot (?:reach|approach|exceed)|"
            r"at most|no more than|plausible (?:range|maximum)|orders? of magnitude)\b",
            visibility_justification,
            re.IGNORECASE,
        )
        visibility_magnitude_overreach = bool(
            visibility_magnitude_status == "UNKNOWN" and magnitude_language
        )
        if visibility_magnitude_overreach:
            visibility_justification = (
                "The stated observability gap revises harm direction "
                f"{visibility_harm_revision.casefold()}; its magnitude remains unknown."
            )
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
    if (
        broadcast.constraint != "CONSENSUS_AUDIT"
        and assumption_status == "NOT_AUDITED"
        and baseline_status in {
            "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
        }
    ):
        # Source testimony uncertainty is part of the specialist's initial
        # semantic state. It cannot disappear merely because the compact
        # delegate schema requires a provisional ranking.
        assumption_status = baseline_status
        unsupported_assumption = baseline_condition or "source testimony left a decisive comparison unresolved"
        reversal_condition = baseline_condition
    if (
        specialist == "utilitarian"
        and utilitarian_depends_on_unknown
        and broadcast.constraint != "CONSENSUS_AUDIT"
    ):
        assumption_status = "UNDERDETERMINED"
        unsupported_assumption = utilitarian_missing_comparison
        reversal_condition = utilitarian_missing_comparison
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
    elif assumption_status in {
        "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
    } and unresolved == "NONE":
        # A later ordinary or hypothetical cycle cannot erase missing real-world
        # facts merely by omitting the marker.
        unresolved = (
            "RESOLVE_NORMATIVE_TENSION"
            if assumption_status == "NORMATIVELY_CONTESTED"
            else "VERIFY_FACTS"
        )

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
    if landscape_verifier is not None and landscape_search_complete:
        grounding_cases = dict(landscape_cases)
        grounding_action = actions[action_ids.index(recommended_id)]
        if grounding_action:
            grounding_cases[grounding_action] = " ".join(
                part for part in (
                    grounding_cases.get(grounding_action, ""),
                    f"Rationale: {rationale}",
                ) if part
            )
        landscape_validation_errors = landscape_verifier(
            scenario_text, actions, grounding_cases, landscape_validation_errors
        )
    comparative_grounding_errors = [
        error for error in landscape_validation_errors
        if "unverified comparative beneficiary claim" in error
    ]
    if comparative_grounding_errors and evidence_basis != "UNSTATED_FACTS":
        # Preserve the normative vote while removing confidence supplied by an
        # unsupported action-to-beneficiary edge. This is the same symmetric
        # uncertainty treatment used for delegate-disclosed speculation.
        evidence_basis = "UNSTATED_FACTS"
        speculative_claim = comparative_grounding_errors[0][:180]
        calibration_tier = "DECISION_CRITICAL"
        calibration_reason = (
            "No committed scenario edge establishes the claimed comparative treatment."
        )
        calibration_retention = 0.55
        calibration_epistemic_cap = SPECULATIVE_EPISTEMIC_CAP
        damping = apply_symmetric_claim_damping(
            id_scores,
            evidence_basis,
            unresolved,
            retention=calibration_retention,
        )
        id_scores = damping.scores
        unresolved = damping.unresolved
        claim_direction_damped = damping.applied
        scores = {
            action: id_scores[action_id]
            for action_id, action in zip(action_ids, actions)
        }
    landscape_semantic_valid = not landscape_validation_errors

    audit_certainty = {
        "NOT_AUDITED": 1.0,
        "SUPPORTED": 1.0,
        "CONDITIONAL": 0.60,
        "UNDERDETERMINED": 0.35,
        "NORMATIVELY_CONTESTED": 0.65,
    }[assumption_status]
    if audit_certainty < 1.0 and not claim_direction_damped:
        id_scores = {
            action_id: 0.5 + (score - 0.5) * audit_certainty
            for action_id, score in id_scores.items()
        }
        scores = {
            action: id_scores[action_id]
            for action_id, action in zip(action_ids, actions)
        }
    if framework_grounding_penalty > 0.0:
        retention = 1.0 - framework_grounding_penalty
        id_scores = {
            action_id: 0.5 + (score - 0.5) * retention
            for action_id, score in id_scores.items()
        }
        scores = {
            action: id_scores[action_id]
            for action_id, action in zip(action_ids, actions)
        }
    # Claim damping and consensus-audit damping are alternative calibrations of
    # the same missing evidential support. Multiplying them would charge the
    # same uncertainty twice and can erase a delegate's substantive ranking.

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
        epistemic_confidence = min(epistemic_confidence, calibration_epistemic_cap)
    if unresolved != "NONE":
        epistemic_confidence = min(epistemic_confidence, 0.55)
    if assumption_status == "CONDITIONAL":
        epistemic_confidence = min(epistemic_confidence, 0.60)
    elif assumption_status == "UNDERDETERMINED":
        epistemic_confidence = min(epistemic_confidence, 0.35)
    elif assumption_status == "NORMATIVELY_CONTESTED":
        epistemic_confidence = min(epistemic_confidence, 0.60)
    if landscape_search_attempted and not landscape_semantic_valid:
        epistemic_confidence = min(epistemic_confidence, 0.60)
    if framework_grounding_penalty > 0.0:
        epistemic_confidence = min(epistemic_confidence, 0.55)
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
        "CONSENSUS_AUDIT", "PROBLEM_REFORMULATION", "VISIBILITY_AUDIT",
    }
    if moved_to_favored and broadcast.constraint not in review_constraints:
        conformity_penalty = 0.65
        epistemic_confidence *= 1.0 - conformity_penalty
        surprise = max(surprise, 0.8)
    # Compare delegate-authored score gaps, not policy-effective gaps after
    # claim damping or an uncertainty audit. Otherwise lifting a temporary
    # system contraction looks like an unexplained change of conviction.
    current_context = (
        "BASE"
        if broadcast.constraint not in {
            "VISIBILITY_AUDIT", "CONSENSUS_AUDIT", "REVERSAL_AUDIT",
            "PROBLEM_REFORMULATION", "PLANNING_REVIEW", "CONTINGENCY_REVIEW",
            "SYNTHESIS_REVIEW",
        }
        else broadcast.constraint
    )
    comparable_context = bool(
        not previous_context
        or previous_context == current_context
        or (previous_context == "BASE" and current_context == "BASE")
    )
    confidence_drift = (
        reported_preference_strength - previous_confidence
        if previous_confidence is not None and not position_changed and comparable_context
        else 0.0
    )
    recommended_is_favored = bool(
        favored_fragment and recommended_action.casefold().startswith(favored_fragment)
    )
    drift_favors_salient_action = bool(
        previous_confidence is not None
        and not position_changed
        and comparable_context
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

    # Passive measurement layer: malformed or omitted measurements fall back to
    # transparent derivations and never invalidate the delegate's ethical vote.
    selection_status = str(data.get("ss", "SELECTED")).strip().upper()
    if selection_status not in {"SELECTED", "PROVISIONAL", "UNSELECTED"}:
        selection_status = "SELECTED"
    raw_admissibility = data.get("am", {})
    action_admissibility = {}
    for action_id, action in zip(action_ids, actions):
        emitted = (
            str(raw_admissibility.get(action_id, "")).strip().upper()
            if isinstance(raw_admissibility, dict) else ""
        )
        if emitted not in {"REQUIRED", "PERMISSIBLE", "REJECTED", "UNASSESSED"}:
            emitted = "PERMISSIBLE" if id_scores[action_id] >= 0.5 else "REJECTED"
        action_admissibility[action] = emitted
    comparison_complete = bool(data.get("cc", True))
    evidence_sufficient = bool(data.get("esa", unresolved == "NONE"))
    if (
        broadcast.constraint == "OPEN_DELIBERATION"
        and baseline_status in {
            "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
        }
    ):
        selection_status = "PROVISIONAL"
        if baseline_status != "NORMATIVELY_CONTESTED":
            evidence_sufficient = False
        comparison_complete = False
    if (
        broadcast.constraint == "OPEN_DELIBERATION"
        and specialist == "utilitarian"
        and utilitarian_depends_on_unknown
    ):
        selection_status = "PROVISIONAL"
        evidence_sufficient = False
        comparison_complete = False
    interim_id = str(data.get("ia", recommended_id)).strip().upper()
    interim_action = actions[action_ids.index(interim_id)] if interim_id in action_ids else ""
    proposition_response = str(data.get("wp", "NOT_APPLICABLE")).strip().upper()
    if proposition_response not in {"NOT_APPLICABLE", "ACCEPT", "QUALIFY", "REJECT"}:
        proposition_response = "NOT_APPLICABLE"
    reasoning_effect = str(data.get("we", "NONE")).strip().upper()
    if reasoning_effect not in {"NONE", "FACTUAL", "NORMATIVE", "BOTH"}:
        reasoning_effect = "NONE"
    framework_application = " ".join(str(data.get("fa", "")).split())
    framework_retained = bool(data.get("fr", True))
    if framework_grounding_penalty > 0.0:
        framework_retained = False
    broadcast_dependence = str(data.get("bd", "NONE")).strip().upper()
    if broadcast_dependence not in {"NONE", "LOW", "MEDIUM", "HIGH"}:
        broadcast_dependence = "NONE"
    proposition_present = broadcast.constraint != "OPEN_DELIBERATION"
    if framework_grounding_penalty > 0.0:
        retention_status = "LOST"
    elif not proposition_present:
        retention_status = "NOT_MEASURED"
    elif not framework_retained:
        retention_status = "LOST"
    elif _semantic_word_count(framework_application) >= 2:
        retention_status = "PRESERVED"
    else:
        retention_status = "UNCLEAR"

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
        baseline_action=(
            actions[action_ids.index(baseline_reference_id)]
            if baseline_reference_id != "NONE" else ""
        ),
        baseline_status=baseline_status,
        baseline_condition=baseline_condition,
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
        evidence_calibration_tier=calibration_tier,
        evidence_calibration_reason=calibration_reason,
        evidence_direction_retention=calibration_retention,
        landscape_cases=landscape_cases,
        landscape_decisive_axis=landscape_axis,
        landscape_tiebreaker=landscape_tiebreaker,
        landscape_tiebreaker_failure=landscape_failure,
        landscape_search_complete=landscape_search_complete,
        landscape_search_attempted=landscape_search_attempted,
        landscape_semantic_valid=landscape_semantic_valid,
        landscape_validation_errors=landscape_validation_errors,
        preference_strength=preference_strength,
        reported_preference_strength=reported_preference_strength,
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
        reversal_review_valid=review_valid,
        reversal_review_error=review_error,
        contingency_choice=contingency_choice,
        contingency_justification=contingency_justification,
        contingency_response_valid=contingency_valid,
        contingency_response_error=contingency_error,
        graph_update_proposal=graph_update,
        expected_value_estimates=expected_values,
        visibility_response=visibility_response,
        visibility_justification=visibility_justification,
        visibility_harm_revision=visibility_harm_revision,
        visibility_magnitude_status=visibility_magnitude_status,
        visibility_magnitude_overreach=visibility_magnitude_overreach,
        selection_status=selection_status,
        action_admissibility=action_admissibility,
        comparison_complete=comparison_complete,
        evidence_sufficient_for_action=evidence_sufficient,
        interim_action=interim_action,
        workspace_proposition_response=proposition_response,
        workspace_reasoning_effect=reasoning_effect,
        framework_application=framework_application,
        framework_constraint_retained=framework_retained,
        self_reported_broadcast_dependence=broadcast_dependence,
        framework_retention_status=retention_status,
        framework_action_map=framework_action_map,
        framework_numerical_role=framework_numerical_role,
        framework_numerical_justification=framework_numerical_justification,
        framework_grounding_penalty=framework_grounding_penalty,
        framework_validation_errors=framework_validation_errors,
        utilitarian_consequence_table=utilitarian_consequence_table,
        utilitarian_decision_depends_on_unknown=utilitarian_depends_on_unknown,
        utilitarian_missing_comparison=utilitarian_missing_comparison,
        utilitarian_ledger_proposal=utilitarian_ledger_proposal,
        rawls_position_proposal=rawls_position_proposal,
        deontological_ledger_proposal=deontological_ledger_proposal,
        virtue_character_proposal=virtue_character_proposal,
        care_relational_map=care_relational_map,
        care_numerical_role=care_numerical_role,
        care_numerical_justification=care_numerical_justification,
        care_grounding_penalty=care_grounding_penalty,
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
    baseline_status: str = "UNAVAILABLE"
    baseline_provisional_action_id: str = "NONE"
    baseline_condition: str = ""
    baseline_framework_commitments: dict[str, str] = field(default_factory=dict)
    baseline_numerical_role: str = "UNASSESSED"
    source_action_legend: dict[str, str] | None = None
    scenario_facts: dict[str, Any] | None = None
    max_tokens: int = 128
    previous_recommendation_id: str = ""
    previous_confidence: float | None = None
    previous_context: str = ""
    assumption_status: str = "NOT_AUDITED"
    unsupported_assumption: str = ""
    reversal_condition: str = ""
    memory_profile: dict[str, Any] | None = None
    evidence_calibrator: Any | None = None
    landscape_verifier: Any | None = None
    epistemic_commitments: list[str] = field(default_factory=list)
    # Populated only after a Rawls ledger transaction commits. Generated prose
    # never becomes the delegate's recurrent principle state by itself.
    previous_framework_state: dict[str, Any] = field(default_factory=dict)

    def _audit_framework_state_change(
        self, candidate: CandidateChunk, broadcast: WorkspaceBroadcast,
    ) -> None:
        """Keep recurrent framework state stable unless a typed review warrants change."""
        if (
            self.name not in {"rawlsian", "virtue"}
            or not self.previous_framework_state
        ):
            return
        previous = self.previous_framework_state
        current = (
            candidate.rawls_position_proposal
            if self.name == "rawlsian" else candidate.virtue_character_proposal
        )
        if not current:
            return
        if self.name == "rawlsian":
            previous_positions = {
                str(item.get("canonical_action_id", item.get("action_id", ""))): (
                    str(item.get("proposed_effect", item.get("effect", ""))),
                    str(item.get("dimension", "")),
                )
                for item in previous.get("positions", [])
                if isinstance(item, dict)
            }
            current_positions = {
                str(item.get("action_id", "")): (
                    str(item.get("effect", "")), str(item.get("dimension", "")),
                )
                for item in current.get("positions", [])
                if isinstance(item, dict)
            }
            auxiliary_changed = (
                dict(previous.get("liberty_status", {}))
                != dict(current.get("liberty_status", {}))
            )
        else:
            previous_positions = {
                str(item.get("action_id", "")): (
                    str(item.get("verdict", "")), str(item.get("actor_role", "")),
                )
                for item in previous.get("assessments", [])
                if isinstance(item, dict)
            }
            current_positions = {
                str(item.get("action_id", "")): (
                    str(item.get("verdict", "")), str(item.get("actor_role", "")),
                )
                for item in current.get("assessments", [])
                if isinstance(item, dict)
            }
            auxiliary_changed = False
        same_action_set = bool(
            previous_positions and set(previous_positions) == set(current_positions)
        )
        changed = same_action_set and (
            str(previous.get("ranking_basis", ""))
            != str(current.get("ranking_basis", ""))
            or auxiliary_changed
            or previous_positions != current_positions
        )
        if not changed:
            return

        change_details: list[str] = []
        previous_basis = str(previous.get("ranking_basis", ""))
        current_basis = str(current.get("ranking_basis", ""))
        if previous_basis != current_basis:
            change_details.append(
                f"ranking basis {previous_basis or 'NONE'} -> {current_basis or 'NONE'}"
            )
        for action_id in sorted(set(previous_positions) & set(current_positions)):
            if previous_positions[action_id] != current_positions[action_id]:
                old = "/".join(previous_positions[action_id])
                new = "/".join(current_positions[action_id])
                change_details.append(f"{action_id} {old} -> {new}")
        if self.name == "rawlsian" and auxiliary_changed:
            change_details.append("basic-liberty comparison changed")
        change_summary = "; ".join(change_details[:3]) or "typed principle state changed"

        explanation = " ".join(
            (candidate.change_justification, candidate.framework_application)
        ).strip()
        review_supplied_reason = bool(
            broadcast.constraint != "OPEN_DELIBERATION"
            and candidate.workspace_reasoning_effect in {"FACTUAL", "NORMATIVE", "BOTH"}
            and _semantic_word_count(explanation) >= 4
            and _FRAMEWORK_CONSTRUCT_MARKERS[self.name].search(
                " ".join((explanation, candidate.rationale))
            )
        )
        if review_supplied_reason:
            return

        framework_name = "Rawlsian" if self.name == "rawlsian" else "Virtue"
        error = (
            f"{framework_name} principle state changed without a framework-relevant "
            f"workspace reason ({change_summary}); previous committed ledger preserved"
        )
        if error not in candidate.framework_validation_errors:
            candidate.framework_validation_errors.append(error)
        if candidate.framework_grounding_penalty < 0.35:
            candidate.action_scores = {
                action: 0.5 + (score - 0.5) * 0.65
                for action, score in candidate.action_scores.items()
            }
            ordered = sorted(candidate.action_scores.values(), reverse=True)
            candidate.preference_strength = (
                ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
            )
            candidate.friction = candidate.preference_strength
        candidate.framework_grounding_penalty = max(
            candidate.framework_grounding_penalty, 0.35
        )
        candidate.epistemic_confidence = min(candidate.epistemic_confidence, 0.55)
        candidate.confidence = candidate.epistemic_confidence
        # The proposed update failed, but the prior committed constraint remains
        # authoritative. Keep those two facts separate so trace health does not
        # mistake a successful transactional rejection for framework loss.
        candidate.framework_constraint_retained = True
        candidate.framework_retention_status = "UPDATE_REJECTED"
        # Do not let an unexplained principle transition overwrite authoritative
        # graph state. The vote remains visible but is epistemically damped.
        if self.name == "rawlsian":
            candidate.rawls_position_proposal = {}
        else:
            candidate.virtue_character_proposal = {}

    def _evaluate_contingency(
        self,
        scenario: str,
        actions: Sequence[str],
        broadcast: WorkspaceBroadcast,
    ) -> CandidateChunk:
        """Answer a synthesis-failure branch with a deliberately small schema."""
        action_ids = [f"A{index}" for index in range(len(actions))]
        if len(actions) != 2 or tuple(actions) != broadcast.contingency_fallback_actions:
            return _invalid_candidate(
                self.name, actions,
                "contingency evaluator requires exactly the two typed fallback actions",
            )
        allowed_constraints = sorted(FRAMEWORK_CONSTRAINTS[self.name])
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
                "cr": {"type": "string", "enum": action_ids},
                "c": {"type": "string", "enum": allowed_constraints},
                "u": {"type": "string", "enum": sorted(ALLOWED_UNRESOLVED)},
                "cj": {"type": "string", "minLength": 12, "maxLength": 160},
                "z": {"type": "number", "minimum": 0, "maximum": 1},
                "fr": {"type": "boolean"},
            },
            "required": ["scores", "cr", "c", "u", "cj", "z", "fr"],
            "additionalProperties": False,
        }
        legend = {action_id: action for action_id, action in zip(action_ids, actions)}
        prompt = f"""[INST]
You are the {self.name} ethical specialist. Framework: {FRAMEWORK_ROLES[self.name]}
Original testimony: {_compact_testimony(self.testimony, 400) or 'NONE'}
Scenario: {' '.join(scenario.split())[:600]}
Failed synthesis: {broadcast.contingency_synthesis_action}
Required predicate: {broadcast.contingency_predicate}
Assume the required predicate is FALSE: {broadcast.contingency_failure_condition}
Fallbacks: {json.dumps(legend)}
The synthesis is unavailable. Compare ONLY the two fallbacks under your framework.
Return compact JSON: scores for both IDs; cr=the higher-scored fallback; c=your
framework constraint; u=remaining uncertainty or NONE; cj=one short reason that
explicitly applies the assumed failure; z=epistemic confidence; fr=true only if your
framework remains operative. Do not restate the base case without applying failure.
[/INST]"""

        def parse(raw_text: str) -> CandidateChunk:
            data = _extract_json(raw_text)
            raw_scores = data.get("scores")
            if not isinstance(raw_scores, dict) or set(raw_scores) != set(action_ids):
                raise ValueError("fallback scores must contain exactly A0 and A1")
            id_scores = {
                action_id: _strict_number(raw_scores[action_id], f"scores.{action_id}")
                for action_id in action_ids
            }
            choice_id = str(data.get("cr", "")).strip().upper()
            if choice_id not in action_ids:
                raise ValueError("contingency choice must be a fallback ID")
            if id_scores[choice_id] != max(id_scores.values()):
                raise ValueError("contingency choice must have the highest fallback score")
            constraint = str(data.get("c", "")).strip().upper().replace(" ", "_")
            if constraint not in FRAMEWORK_CONSTRAINTS[self.name]:
                raise ValueError("contingency constraint is outside the assigned framework")
            unresolved = str(data.get("u", "NONE")).strip().upper().replace(" ", "_")
            if unresolved not in ALLOWED_UNRESOLVED:
                raise ValueError("contingency unresolved state is invalid")
            justification = " ".join(str(data.get("cj", "")).split())
            if len(justification.split()) < 3:
                raise ValueError("contingency response did not explain the conditional choice")
            epistemic = _strict_number(data.get("z"), "contingency epistemic confidence")
            framework_retained = data.get("fr") is True
            scores = {
                action: id_scores[action_id]
                for action_id, action in zip(action_ids, actions)
            }
            ordered = sorted(id_scores.values(), reverse=True)
            preference = ordered[0] - ordered[1]
            choice = actions[action_ids.index(choice_id)]
            baseline_id = self.baseline_action_id if self.baseline_action_id in action_ids else "NONE"
            baseline_status = str(self.baseline_status).strip().upper()
            if baseline_status == "UNAVAILABLE" and baseline_id != "NONE":
                baseline_status = "DIRECT"
            provisional_id = (
                self.baseline_provisional_action_id
                if self.baseline_provisional_action_id in action_ids else "NONE"
            )
            baseline_reference_id = (
                baseline_id if baseline_status == "DIRECT" else provisional_id
            )
            previous_id = (
                self.previous_recommendation_id
                if self.previous_recommendation_id in action_ids else ""
            )
            alignment = (
                "UNRESOLVED" if baseline_status in {
                    "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
                }
                else "UNCLEAR" if baseline_reference_id == "NONE"
                else "SUPPORTS" if choice_id == baseline_reference_id else "RECONSIDERS"
            )
            admissibility = {
                action: (
                    "PERMISSIBLE"
                    if action_id == choice_id or id_scores[action_id] >= 0.5
                    else "REJECTED"
                )
                for action_id, action in zip(action_ids, actions)
            }
            return CandidateChunk(
                specialist=self.name,
                constraint=constraint,
                action_scores=scores,
                surprise=0.7 if alignment == "RECONSIDERS" else 0.0,
                friction=preference,
                confidence=epistemic,
                unresolved=unresolved,
                rationale=justification,
                recommended_action=choice,
                baseline_action=(
                    actions[action_ids.index(baseline_reference_id)]
                    if baseline_reference_id != "NONE" else ""
                ),
                baseline_status=baseline_status,
                baseline_condition=self.baseline_condition,
                testimony_alignment=alignment,
                previous_action=(actions[action_ids.index(previous_id)] if previous_id else ""),
                position_changed=bool(previous_id and previous_id != choice_id),
                change_justification=justification,
                preference_strength=preference,
                reported_preference_strength=preference,
                epistemic_confidence=epistemic,
                decision_rule=(
                    f"If {broadcast.contingency_failure_condition}, prefer {choice}"
                ),
                contingency_choice=choice,
                contingency_justification=justification,
                contingency_response_valid=True,
                selection_status="PROVISIONAL" if unresolved != "NONE" else "SELECTED",
                action_admissibility=admissibility,
                comparison_complete=True,
                evidence_sufficient_for_action=(unresolved == "NONE"),
                interim_action=choice,
                workspace_proposition_response="ACCEPT",
                workspace_reasoning_effect="FACTUAL",
                framework_application=justification,
                framework_constraint_retained=framework_retained,
                framework_retention_status=("PRESERVED" if framework_retained else "LOST"),
            )

        try:
            output = _call_json_llm(
                self.llm, prompt, max_tokens=max(80, min(self.max_tokens, 112)),
                temperature=0.0, schema=schema,
            )
            raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
            return parse(raw)
        except (ValueError, json.JSONDecodeError) as first_error:
            repair_prompt = f"""[INST]
Repair this contingency answer as JSON only: {raw[:300] if 'raw' in locals() else ''}
Use only fallback IDs A0 and A1. cr must have the higher score. cj must explain
the choice assuming this failure is true: {broadcast.contingency_failure_condition}
Required: scores, cr, c, u, cj, z, fr. No other fields.
[/INST]"""
            try:
                repaired = _call_json_llm(
                    self.llm, repair_prompt,
                    max_tokens=max(80, min(self.max_tokens, 112)),
                    temperature=0.0, schema=schema,
                )
                repaired_raw = (
                    repaired["choices"][0]["text"]
                    if isinstance(repaired, dict) else str(repaired)
                )
                return parse(repaired_raw)
            except (ValueError, json.JSONDecodeError) as repair_error:
                return _invalid_candidate(
                    self.name, actions,
                    f"contingency initial={first_error}; repair={repair_error}",
                )

    def evaluate(
        self,
        scenario: str,
        actions: Sequence[str],
        broadcast: WorkspaceBroadcast,
    ) -> CandidateChunk:
        if broadcast.constraint == "CONTINGENCY_REVIEW":
            return self._evaluate_contingency(scenario, actions, broadcast)
        role = FRAMEWORK_ROLES[self.name]
        testimony = _compact_testimony(self.testimony, 900)
        action_ids = [f"A{index}" for index in range(len(actions))]
        action_legend = {action_id: action for action_id, action in zip(action_ids, actions)}
        allowed_constraints = sorted(FRAMEWORK_CONSTRAINTS[self.name])
        fixed_baseline = self.baseline_action_id if self.baseline_action_id in {*action_ids, "NONE"} else "NONE"
        baseline_status = str(self.baseline_status).strip().upper()
        if baseline_status == "UNAVAILABLE" and fixed_baseline != "NONE":
            baseline_status = "DIRECT"
        if baseline_status not in {
            "DIRECT", "CONDITIONAL", "UNDERDETERMINED", "NORMATIVELY_CONTESTED",
            "OUTSIDE_ACTION_SET", "UNAVAILABLE",
        }:
            baseline_status = "DIRECT" if fixed_baseline != "NONE" else "UNAVAILABLE"
        baseline_provisional = (
            self.baseline_provisional_action_id
            if self.baseline_provisional_action_id in {*action_ids, "NONE"}
            else "NONE"
        )
        if baseline_status == "DIRECT":
            baseline_provisional = fixed_baseline
        baseline_state = {
            "status": baseline_status,
            "committed_action": fixed_baseline,
            "provisional_action": baseline_provisional,
            "condition": self.baseline_condition or "NONE",
            "framework_commitments": dict(self.baseline_framework_commitments or {}),
            "numerical_role": self.baseline_numerical_role,
        }
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
                "gu": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["NONE", "BOUNDARY", "AND", "OR"]},
                        "from_action": {"type": "string", "enum": [*action_ids, "NONE"]},
                        "to_action": {"type": "string", "enum": [*action_ids, "NONE"]},
                        "clauses": {
                            "type": "array", "minItems": 0, "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "affected_action": {"type": "string", "enum": action_ids},
                                    "metric": {"type": "string", "maxLength": 60},
                                    "metric_valence": {"type": "string", "enum": ["ADVERSE", "BENEFICIAL"]},
                                    "comparator": {"type": "string", "enum": ["LT", "LE", "GT", "GE"]},
                                    "threshold": {"type": "number"},
                                    "unit": {"type": "string", "maxLength": 20},
                                    "source_text": {"type": "string", "maxLength": 100},
                                },
                                "required": ["affected_action", "metric", "metric_valence", "comparator", "threshold", "unit", "source_text"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["operation", "from_action", "to_action", "clauses"],
                    "additionalProperties": False,
                },
                "ev": {
                    "type": "object",
                    "properties": {
                        action_id: {
                            "type": "object",
                            "properties": {
                                "value": {"type": "number", "minimum": 0},
                                "unit": {"type": "string", "maxLength": 24},
                                "direction": {"type": "string", "enum": ["BENEFIT", "HARM"]},
                                "grounded": {"type": "boolean"},
                            },
                            "required": ["value", "unit", "direction", "grounded"],
                            "additionalProperties": False,
                        } for action_id in action_ids
                    },
                    "required": action_ids,
                    "additionalProperties": False,
                },
                "ss": {"type": "string", "enum": ["SELECTED", "PROVISIONAL", "UNSELECTED"]},
                "am": {
                    "type": "object",
                    "properties": {
                        action_id: {
                            "type": "string",
                            "enum": ["REQUIRED", "PERMISSIBLE", "REJECTED", "UNASSESSED"],
                        } for action_id in action_ids
                    },
                    "required": action_ids,
                    "additionalProperties": False,
                },
                "cc": {"type": "boolean"},
                "esa": {"type": "boolean"},
                "ia": {"type": "string", "enum": [*action_ids, "NONE"]},
                "wp": {
                    "type": "string",
                    "enum": ["NOT_APPLICABLE", "ACCEPT", "QUALIFY", "REJECT"],
                },
                "we": {"type": "string", "enum": ["NONE", "FACTUAL", "NORMATIVE", "BOTH"]},
                "fa": {"type": "string", "maxLength": 140},
                "fr": {"type": "boolean"},
                "bd": {"type": "string", "enum": ["NONE", "LOW", "MEDIUM", "HIGH"]},
            },
            "required": [
                "scores", "r", "c", "u", "w", "j", "e", "x", "l", "da", "t", "tf",
                "dr", "ft", "nt", "z", "gu", "ev",
                "ss", "am", "cc", "esa", "ia", "wp", "we", "fa", "fr", "bd",
            ],
            "additionalProperties": False,
        }
        is_care = self.name == "care"
        is_construct_specialist = self.name in _FRAMEWORK_CONSTRUCT_MARKERS
        construct_map_key = "rm" if is_care else "fm"
        if is_construct_specialist:
            schema["properties"].update({
                construct_map_key: {
                    "type": "object",
                    "properties": {
                        action_id: {"type": "string", "minLength": 8, "maxLength": 180}
                        for action_id in action_ids
                    },
                    "required": action_ids,
                    "additionalProperties": False,
                },
                "nr": {
                    "type": "string",
                    "enum": ["DECISIVE", "SECONDARY", "IRRELEVANT"],
                },
                "np": {"type": "string", "minLength": 8, "maxLength": 140},
            })
            schema["required"].extend([construct_map_key, "nr", "np"])
        if self.name == "utilitarian":
            consequence_row_schema = {
                "type": "object",
                "properties": {
                    "o": {"type": "string", "minLength": 4, "maxLength": 120},
                    "s": {"type": "string", "minLength": 1, "maxLength": 80},
                    "d": {"type": "string", "enum": ["BENEFIT", "HARM"]},
                    "p": {"type": "string", "minLength": 1, "maxLength": 24},
                    "m": {"type": "string", "minLength": 1, "maxLength": 60},
                    "h": {"type": "string", "minLength": 1, "maxLength": 40},
                    "rv": {
                        "type": "string",
                        "enum": ["REVERSIBLE", "IRREVERSIBLE", "UNKNOWN"],
                    },
                    "g": {
                        "type": "string",
                        "enum": ["STATED", "INFERRED", "UNKNOWN"],
                    },
                },
                "required": ["o", "s", "d", "p", "m", "h", "rv", "g"],
                "additionalProperties": False,
            }
            schema["properties"].update({
                "ct": {
                    "type": "object",
                    "properties": {
                        action_id: {
                            "type": "array", "minItems": 1, "maxItems": 3,
                            "items": consequence_row_schema,
                        }
                        for action_id in action_ids
                    },
                    "required": action_ids,
                    "additionalProperties": False,
                },
                "cd": {"type": "boolean"},
                "cm": {"type": "string", "maxLength": 160},
            })
            schema["required"].extend(["ct", "cd", "cm"])
        if self.name == "deontological":
            duty_assessment_schema = {
                "type": "object",
                "properties": {
                    "v": {
                        "type": "string",
                        "enum": ["REQUIRED", "PERMISSIBLE", "PROHIBITED", "CONFLICTED"],
                    },
                    "k": {
                        "type": "string",
                        "enum": [
                            "DUTY", "RIGHT", "AUTONOMY", "UNIVERSAL_LAW",
                            "RESPECT_PERSONS", "OTHER", "UNKNOWN",
                        ],
                    },
                    "n": {"type": "string", "minLength": 3, "maxLength": 100},
                    "rel": {
                        "type": "string",
                        "enum": ["SATISFIES", "CONSISTENT", "VIOLATES", "CONFLICTS", "UNCERTAIN"],
                    },
                    "b": {"type": "string", "minLength": 1, "maxLength": 80},
                    "p": {"type": "string", "minLength": 1, "maxLength": 100},
                    "cn": {"type": "string", "minLength": 1, "maxLength": 100},
                    "g": {
                        "type": "string",
                        "enum": ["ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"],
                    },
                    "rs": {"type": "string", "minLength": 4, "maxLength": 180},
                },
                "required": ["v", "k", "n", "rel", "b", "p", "cn", "g", "rs"],
                "additionalProperties": False,
            }
            schema["properties"]["dp"] = {
                "type": "object",
                "properties": {
                    action_id: duty_assessment_schema for action_id in action_ids
                },
                "required": action_ids,
                "additionalProperties": False,
            }
            schema["required"].append("dp")
        if self.name == "virtue":
            virtue_assessment_schema = {
                "type": "object",
                "properties": {
                    "v": {
                        "type": "string",
                        "enum": ["EXEMPLIFIES", "MIXED", "UNDERMINES", "UNCERTAIN"],
                    },
                    "r": {"type": "string", "minLength": 2, "maxLength": 100},
                    "vs": {"type": "string", "minLength": 2, "maxLength": 120},
                    "x": {"type": "string", "minLength": 2, "maxLength": 120},
                    "c": {"type": "string", "minLength": 2, "maxLength": 140},
                    "g": {
                        "type": "string",
                        "enum": ["ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"],
                    },
                    "rs": {"type": "string", "minLength": 4, "maxLength": 180},
                },
                "required": ["v", "r", "vs", "x", "c", "g", "rs"],
                "additionalProperties": False,
            }
            schema["properties"]["vl"] = {
                "type": "object",
                "properties": {
                    action_id: virtue_assessment_schema for action_id in action_ids
                },
                "required": action_ids,
                "additionalProperties": False,
            }
            schema["properties"]["vb"] = {
                "type": "string",
                "enum": [
                    "PRACTICAL_WISDOM", "ROLE_FIDELITY", "FLOURISHING",
                    "EXEMPLAR_REASONING", "UNRESOLVED",
                ],
            }
            schema["required"].extend(["vl", "vb"])
        if self.name == "rawlsian":
            rawls_position_schema = {
                "type": "object",
                "properties": {
                    "g": {"type": "string", "minLength": 2, "maxLength": 100},
                    "d": {
                        "type": "string",
                        "enum": [
                            "BASIC_LIBERTY", "OPPORTUNITY", "INCOME_WEALTH",
                            "POWERS_OFFICES", "SELF_RESPECT", "BASIC_INTEREST_SECURITY",
                            "OTHER_PRIMARY_GOOD", "UNKNOWN",
                        ],
                    },
                    "e": {
                        "type": "string",
                        "enum": ["IMPROVES", "PRESERVES", "WORSENS", "UNCERTAIN"],
                    },
                    "ca": {"type": "string", "enum": action_ids},
                    "b": {
                        "type": "string",
                        "enum": ["ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"],
                    },
                    "rs": {"type": "string", "minLength": 4, "maxLength": 180},
                },
                "required": ["g", "d", "e", "ca", "b", "rs"],
                "additionalProperties": False,
            }
            schema["properties"]["rp"] = {
                "type": "object",
                "properties": {
                    action_id: rawls_position_schema for action_id in action_ids
                },
                "required": action_ids,
                "additionalProperties": False,
            }
            schema["properties"]["rb"] = {
                "type": "string",
                "enum": [
                    "LEXICAL_BASIC_LIBERTY", "FAIR_EQUALITY_OPPORTUNITY",
                    "MAXIMIN_PRIMARY_GOODS", "DIFFERENCE_PRINCIPLE",
                    "ORIGINAL_POSITION_PUBLIC_RULE", "UNRESOLVED",
                ],
            }
            schema["properties"]["lc"] = {
                "type": "object",
                "properties": {
                    action_id: {
                        "type": "string",
                        "enum": ["SATISFIED", "INFRINGED", "CONFLICTED", "UNKNOWN"],
                    }
                    for action_id in action_ids
                },
                "required": action_ids,
                "additionalProperties": False,
            }
            schema["properties"]["fa"] = {
                "type": "string", "minLength": 8, "maxLength": 140,
            }
            schema["required"].extend(["rp", "rb", "lc"])
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
                "rj": {"type": "string", "minLength": 12, "maxLength": 120},
                "rv": {"type": "string", "maxLength": 120},
            })
            schema["required"].extend(["rr", "rj", "rv"])
        if broadcast.constraint == "VISIBILITY_AUDIT":
            schema["properties"].update({
                "vp": {"type": "string", "enum": ["ACCEPT", "QUALIFY", "REJECT"]},
                "vj": {"type": "string", "maxLength": 160},
                "vh": {"type": "string", "enum": ["UPWARD", "DOWNWARD", "UNCHANGED"]},
                "vm": {"type": "string", "enum": ["UNKNOWN", "GROUNDED_BOUNDED"]},
            })
            schema["required"].extend(["vp", "vj", "vh", "vm"])
        if broadcast.constraint == "CONTINGENCY_REVIEW":
            fallback_ids = [
                action_ids[actions.index(action)]
                for action in broadcast.contingency_fallback_actions
                if action in actions
            ]
            schema["properties"].update({
                "cr": {"type": "string", "enum": fallback_ids},
                "cj": {"type": "string", "minLength": 12, "maxLength": 160},
            })
            # The ordinary recommendation is the branch decision in this
            # context; make non-fallback positional voting unrepresentable.
            schema["properties"]["r"] = {"type": "string", "enum": fallback_ids}
            schema["required"].extend(["cr", "cj"])
        framework_prompt = ""
        framework_example = ""
        if is_care:
            framework_example = (
                ',"rm":{"A0":"entrusted dependency under A0",'
                '"A1":"agent-created vulnerability under A1"},'
                '"nr":"SECONDARY","np":"counts inform responsiveness after relational comparison"'
            )
            framework_prompt = """
CARE-SPECIFIC COMPARISON: rm must give the strongest relational basis for EVERY
action using direct entrustment, dependency, trust, agent-created vulnerability,
responsibility, attentiveness, or responsiveness. nr says whether numerical
magnitude is DECISIVE, SECONDARY, or IRRELEVANT; np explains why. Counts may inform
competence and responsiveness. They are DECISIVE only when the competing relational
claims are otherwise comparable; "more lives" alone is not a care-ethical rule.
When the frozen baseline is NORMATIVELY_CONTESTED, preserve its relational
commitments rather than its provisional action: choose an interim r, but set
ss=PROVISIONAL, cc=false, u=RESOLVE_NORMATIVE_TENSION, and name in tf/nt what
priority between the care commitments would settle the judgment.
"""
        elif self.name == "deontological":
            framework_example = (
                ',"fm":{"A0":"PERMISSIBLE: respects autonomy; no conflicting duty",'
                '"A1":"PROHIBITED: instrumentalizes a person; rescue duty conflicts"},'
                '"nr":"SECONDARY","np":"numbers establish the scope of duties, not aggregate value",'
                '"dp":{"A0":{"v":"PERMISSIBLE","k":"AUTONOMY",'
                '"n":"respect competent choice","rel":"CONSISTENT","b":"decision maker",'
                '"p":"affected person","cn":"NONE","g":"FRAMEWORK_ONLY",'
                '"rs":"A0 remains consistent with respect for autonomy"},'
                '"A1":{"v":"PROHIBITED","k":"RESPECT_PERSONS",'
                '"n":"do not instrumentalize persons","rel":"VIOLATES","b":"decision maker",'
                '"p":"affected person","cn":"duty to rescue","g":"FRAMEWORK_ONLY",'
                '"rs":"A1 uses the affected person merely as a means"}}'
            )
            framework_prompt = """
DEONTOLOGICAL COMPARISON: fm must assess EVERY action. Start each value with
REQUIRED:, PERMISSIBLE:, PROHIBITED:, or CONFLICTED:, then name the universal-law,
right, autonomy, respect-for-persons, or duty ground and any competing duty. nr
states whether numerical magnitude is DECISIVE, SECONDARY, or IRRELEVANT; np must
connect any decisive quantity to the scope or category of a duty or rights
violation, never merely to aggregate welfare. If the frozen baseline is
NORMATIVELY_CONTESTED, preserve the conflicting duties rather than its provisional
action: use ss=PROVISIONAL, cc=false, u=RESOLVE_NORMATIVE_TENSION, and identify the
non-consequential priority rule needed to settle the conflict in tf/nt.
dp is the typed duty ledger for every action. v must match the fm prefix. k and n
identify the norm; rel states whether the action SATISFIES, is CONSISTENT with,
VIOLATES, or CONFLICTS with that norm; b is the duty bearer; p is the protected
party; cn names a competing norm or NONE; g separates action/scenario grounding
from framework interpretation; rs gives the shortest justification. A REQUIRED
verdict normally SATISFIES, PERMISSIBLE is CONSISTENT, PROHIBITED VIOLATES, and
CONFLICTED CONFLICTS. Use rel=UNCERTAIN rather than forcing an inconsistent pair.
"""
        elif self.name == "rawlsian":
            framework_example = (
                ',"fm":{"A0":"PRESERVES: least-advantaged group keeps equal liberty",'
                '"A1":"IMPROVES: worse-off group gains the relevant primary good"},'
                '"nr":"DECISIVE","np":"same basic liberties make the least-advantaged position comparable",'
                '"rp":{"A0":{"g":"least-advantaged group","d":"BASIC_LIBERTY",'
                '"e":"PRESERVES","ca":"A1","b":"ACTION_GRAPH",'
                '"rs":"A0 preserves its stated basic liberty"},'
                '"A1":{"g":"worse-off group","d":"INCOME_WEALTH",'
                '"e":"IMPROVES","ca":"A0","b":"ACTION_GRAPH",'
                '"rs":"A1 supplies the stated primary good"}},'
                '"rb":"MAXIMIN_PRIMARY_GOODS",'
                '"lc":{"A0":"SATISFIED","A1":"SATISFIED"}'
            )
            framework_prompt = """
RAWLSIAN COMPARISON: fm must assess EVERY action. Start each value with IMPROVES:,
PRESERVES:, WORSENS:, or UNCERTAIN:, then identify the least-advantaged affected
group, its position relative to the rival action, and the relevant basic liberty,
opportunity, or primary good. Do not say an action protects that group when no
stated comparative fact supports it. nr states whether numerical magnitude is
DECISIVE, SECONDARY, or IRRELEVANT; np must tie decisive quantities to the position
of the least advantaged under compatible basic liberties, not aggregate welfare.
If the frozen baseline is NORMATIVELY_CONTESTED, preserve the conflicting Rawlsian
principles rather than its provisional action: use ss=PROVISIONAL, cc=false,
u=RESOLVE_NORMATIVE_TENSION, and identify the priority question in tf/nt.
rp is the proposed graph ledger. For each action, first identify who bears its
gravest stated burden. Then identify a socially or institutionally least-advantaged
group only when the scenario supports that relation. Do not equate “most harmed in
this event” with Rawls's least advantaged, and do not choose the largest group, the
action's beneficiaries, or the group favored by the current broadcast. When the
scenario has no distributive structure, use ORIGINAL_POSITION_PUBLIC_RULE to test
the public rule parties would accept under label uncertainty; do not fabricate a
difference-principle beneficiary. d names the relevant basic liberty, opportunity,
primary good, or basic interest in security; e must match the fm
prefix; ca is the rival action ID; b says whether the relation comes from the action
graph, the shared scenario, framework interpretation only, or remains unknown; rs
states the shortest supporting reason. Use e=UNCERTAIN and b=UNKNOWN rather than
inventing an action-to-group edge.
rb identifies the governing Rawlsian stage. Equal basic liberties have lexical
priority; use the difference principle or maximin primary goods only after recording
each action's liberty status in lc. The difference principle is not a synonym for
maximizing lives or total welfare, and applies specifically to social/economic
inequality. Counts of lives may inform a public-rule choice behind the veil, but
must not be relabeled as aggregate utility. If the relevant positions, liberty priority, or
primary-good comparison cannot be established, use rb=UNRESOLVED, UNKNOWN/CONFLICTED
in lc, ss=PROVISIONAL, cc=false, esa=false, and preserve that uncertainty. Never
recommend an INFRINGED-liberty action over a SATISFIED-liberty rival by appealing
to aggregate benefits. A later workspace proposition may change the factual burden
assigned to a group; it does not change Rawls's ordering of principles. If rb, lc,
or a position effect changes between cycles, j must identify the new fact or genuine
Rawlsian priority conflict that warrants the revision. When Previous graph-committed
framework state is nonempty and no such reason exists, copy its ranking basis,
liberty statuses, and action effects exactly; do not silently resolve an UNCERTAIN
effect or UNRESOLVED ranking merely because the current broadcast is salient.
"""
        elif self.name == "virtue":
            framework_example = (
                ',"fm":{"A0":"EXEMPLIFIES: responsible role expresses courage with prudence",'
                '"A1":"MIXED: compassion risks imprudence in these circumstances"},'
                '"nr":"SECONDARY","np":"stakes inform practical wisdom without defining virtue",'
                '"vl":{"A0":{"v":"EXEMPLIFIES","r":"public steward",'
                '"vs":"courage and practical wisdom","x":"callousness","c":"stated emergency",'
                '"g":"FRAMEWORK_ONLY","rs":"A0 fits the role with proportionate judgment"},'
                '"A1":{"v":"MIXED","r":"public steward","vs":"compassion",'
                '"x":"imprudence","c":"stated emergency","g":"FRAMEWORK_ONLY",'
                '"rs":"A1 expresses care but risks excess"}},"vb":"PRACTICAL_WISDOM"'
            )
            framework_prompt = """
VIRTUE-ETHICS COMPARISON: fm must assess EVERY action. Start each value with
EXEMPLIFIES:, MIXED:, UNDERMINES:, or UNCERTAIN:, then identify the actor's role,
virtues expressed, vices risked, and circumstances relevant to practical wisdom.
nr states whether numerical magnitude is DECISIVE, SECONDARY, or IRRELEVANT; np
must explain how the stakes inform phronesis rather than substituting a numerical
maximization rule for character. If the frozen baseline is NORMATIVELY_CONTESTED,
preserve the conflicting virtues rather than its provisional action: use
ss=PROVISIONAL, cc=false, u=RESOLVE_NORMATIVE_TENSION, and identify the practical-
wisdom comparison needed to settle the conflict in tf/nt.
vl is the proposed character ledger for every action. v must match the fm prefix;
r names the actor's role; vs names the virtues expressed; x names the vice or
excess risk; c names the circumstance that phronesis must interpret; g distinguishes
scenario/action grounding from framework interpretation; rs gives the shortest
integrated judgment. vb names the governing mode: PRACTICAL_WISDOM, ROLE_FIDELITY,
FLOURISHING, EXEMPLAR_REASONING, or UNRESOLVED. If independent virtues favor
different actions without a practical-wisdom resolution, use vb=UNRESOLVED,
ss=PROVISIONAL, and cc=false rather than converting the conflict into aggregate
welfare. In later cycles, changing vb, role, or verdict requires j to identify new
framework-relevant information. When Previous graph-committed framework state is
nonempty and no such information exists, copy its ranking basis, action verdicts,
and actor roles exactly. Virtue-like prose does not justify changing a MIXED or
UNRESOLVED committed assessment into a resolved ranking.
"""
        elif self.name == "utilitarian":
            framework_example = (
                ',"ct":{"A0":[{"o":"prevents immediate deaths","s":"residents",'
                '"d":"BENEFIT","p":"99%","m":"large","h":"immediate",'
                '"rv":"IRREVERSIBLE","g":"STATED"}],'
                '"A1":[{"o":"risks system failure","s":"region",'
                '"d":"HARM","p":"UNKNOWN","m":"unknown","h":"unknown",'
                '"rv":"UNKNOWN","g":"UNKNOWN"}]},'
                '"cd":true,"cm":"probability and magnitude of the regional failure"'
            )
            framework_prompt = """
UTILITARIAN CONSEQUENCE ACCOUNTING: ct must contain 1-3 material consequence rows
for EVERY action. o=outcome; s=affected scope; d=BENEFIT/HARM; p=stated probability
or UNKNOWN; m=magnitude; h=duration; rv=reversibility; g=STATED, INFERRED, or
UNKNOWN support. Do not place assumptions in STATED rows and do not omit the losing
action. cd=true exactly when the ranking depends on an unknown consequence or
comparison; then cm must name it, use ss=PROVISIONAL, cc=false, esa=false, and
preserve uncertainty. A compact table is more important than listing remote effects.
"""
        prompt = f"""[INST]
You are the {self.name} specialist in a bandwidth-limited ethical workspace.
Task: {role}
Scenario: {' '.join(scenario.split())[:700]}
Your original corpus-grounded testimony: {testimony}
Frozen testimony baseline state: {json.dumps(baseline_state)}
Previous cycle recommendation: {self.previous_recommendation_id or 'NONE'}
Workspace: {broadcast.compact()}
Scenario facts: {json.dumps(self.scenario_facts or {}, sort_keys=True)}
Prior contribution profile: {json.dumps(self.memory_profile or {}, sort_keys=True)}
Active audited propositions: {json.dumps(self.epistemic_commitments[-3:])}
Previous graph-committed framework state: {json.dumps(self.previous_framework_state)}
Action IDs: {json.dumps(action_legend)}
Original testimony source labels: {json.dumps(self.source_action_legend or action_legend)}
CRITICAL STATE MAPPING: the Action IDs above are immutable for this run. Every
score, recommendation, admissibility judgment, rationale, and graph update must
refer to the physical action attached to that exact ID. Do not reuse a label from
the original testimony unless it denotes the same action in this mapping.

Return ONLY compact JSON like:
{{"scores":{{"A0":0.8,"A1":0.2}},"r":"A0","c":"{allowed_constraints[0]}","u":"NONE","w":"short reason","j":"NONE","e":"STATED_FACTS","x":"NONE","l":{{"A0":"best case A0","A1":"best case A1"}},"da":"decisive ethical axis","t":"attempted comparison rule","tf":"NONE","dr":"prefer A0 when its reason outweighs A1","ft":"NONE","nt":"prefer A1 if its value is overriding","z":0.8,"gu":{{"operation":"NONE","from_action":"NONE","to_action":"NONE","clauses":[]}},"ev":{{"A0":{{"value":0,"unit":"NONE","direction":"HARM","grounded":false}},"A1":{{"value":0,"unit":"NONE","direction":"HARM","grounded":false}}}},"ss":"SELECTED","am":{{"A0":"PERMISSIBLE","A1":"REJECTED"}},"cc":true,"esa":true,"ia":"A0","wp":"NOT_APPLICABLE","we":"NONE","fa":"NONE","fr":true,"bd":"NONE"{framework_example}}}
Return scores for every action ID. recommended must have the highest score.
r=recommended and must have the highest score. The frozen baseline was extracted
separately from your testimony. Python derives whether the result supports or
reconsiders it. During OPEN_DELIBERATION, a DIRECT committed baseline must be
preserved. A CONDITIONAL baseline is not a settled commitment: preserve its missing
comparison in u/tf, use ss=PROVISIONAL and esa=false, and rank either action only as
an interim judgment. UNDERDETERMINED likewise requires explicit uncertainty rather
than silently becoming a direct choice. Each score means recommendation strength:
1 strongly recommends; 0 strongly rejects. w must be 2-8 words.
z is epistemic confidence: how likely the ranking is to survive further factual
inquiry and critical scrutiny. z is NOT preference strength. A strong preference
may have low z when it depends on uncertain facts; a close moral tradeoff may have
high z when its facts and framework interpretation are stable.
Also emit passive measurement fields that MUST NOT change scores or r merely to make
the measurements look consistent. ss is SELECTED, PROVISIONAL, or UNSELECTED. am
classifies every action as REQUIRED, PERMISSIBLE, REJECTED, or UNASSESSED. cc says
whether comparison of the listed actions is complete; esa says whether evidence is
sufficient to act now; ia is the action to take while reasoning remains incomplete,
or NONE. These are general decision-state dimensions, not named moral psychologies.
{framework_prompt}
For OPEN_DELIBERATION return wp=NOT_APPLICABLE, we=NONE, fa="NONE", fr=true,
and bd=NONE. In later workspace cycles, wp states whether you accept the anonymous
workspace proposition; we says whether it changed factual reasoning, normative
reasoning, both, or neither. fa briefly states how YOUR assigned framework applies
the proposition, fr says whether your prior framework constraint remains operative,
and bd reports NONE/LOW/MEDIUM/HIGH dependence of this recommendation on the current
broadcast. Do not infer authorship and do not reward or punish agreement.
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
gu is a proposed semantic-graph update. When ft contains a measurable threshold,
encode it directly instead of asking middleware to recover it from prose. Set
operation=BOUNDARY for one clause, AND/OR for 2-3 clauses, from_action=r, and
to_action to the action preferred when the predicate holds. Each clause names the
affected action ID, metric, ADVERSE or BENEFICIAL valence, LT/LE/GT/GE comparator,
numeric threshold, unit (PERCENT, COUNT, DAY, USD, etc.), and exact source_text.
The predicate must cross away from from_action: increasing an ADVERSE property of
from_action or decreasing its BENEFICIAL property may support a switch; increasing
an ADVERSE property of to_action cannot support switching to it. If you cannot encode
that direction confidently, return operation=NONE rather than a decorative boundary.
Use operation=NONE with NONE action IDs and [] clauses when no measurable factual
boundary exists. Do not encode moral priorities or vague axes as numeric clauses.
ev reports expected value only when the scenario itself supplies enough quantities
to compute comparable values for EVERY action. Use one shared unit and one shared
direction (BENEFIT or HARM); grounded=true only for arithmetic from stated facts.
Otherwise return value=0, unit=NONE, a shared direction, and grounded=false for all.
Each case must describe what happens IF ITS OWN ACTION is chosen. Name the affected
person or value from that action, and for harmful actions explicitly acknowledge the
harm before giving the countervailing reason. Do not place the benefit of sparing a
person under the action that kills or harms that same person.
Begin each case with that action's own mechanism or direct consequence, using a clear
anchor from its action description; put comparisons with the rival action afterward.
For symmetric trade-offs, it is valid for both cases to mention the same shared value,
but each must first say how its own action affects that value. If u is not NONE, or
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
During CONTINGENCY_REVIEW, reason only inside the stated hypothetical branch. Assume
the failure condition is TRUE and the admitted synthesis is unavailable. Choose one
of the two typed fallback actions: set cr to its action ID and set r=cr. In cj, state
why that fallback follows under YOUR framework given the failure. A baseline argument
that does not apply the failure condition is not an answer. Do not select or repair
the failed synthesis, and do not invent a third fallback.
During VISIBILITY_AUDIT, evaluate the explicit visibility proposition in the workspace.
Return vp=ACCEPT when it is grounded and materially relevant, QUALIFY when its direction
is plausible but magnitude or action relevance remains uncertain, or REJECT when the
scenario does not support it. Return vh=UPWARD, DOWNWARD, or UNCHANGED for your estimate
of the affected action's harm, and vj explaining why. Then rescore every action. You may
keep the same recommendation, but say whether it survives the revised harm estimate;
agreement with the proposition does not require changing actions.
Return vm=GROUNDED_BOUNDED only when the scenario states a bound on the missing
magnitude. Otherwise vm=UNKNOWN: revise direction without inventing a plausible ceiling,
range, population size, or claim that the hidden harm cannot approach a threshold.
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
            calibration = self._calibrate_evidence(scenario, actions, data)
            candidate = _candidate_from_data(
                self.name, actions, data, broadcast, fixed_baseline, self.scenario_facts or {},
                self.previous_recommendation_id,
                self.previous_confidence,
                self.previous_context,
                self.assumption_status,
                self.unsupported_assumption,
                self.reversal_condition,
                calibration,
                self._verify_landscape,
                baseline_status,
                baseline_provisional,
                self.baseline_condition,
                scenario,
            )
            self._audit_framework_state_change(candidate, broadcast)
            self.previous_recommendation_id = action_ids[actions.index(candidate.recommended_action)]
            self.previous_confidence = candidate.reported_preference_strength
            self.previous_context = self._context_class(broadcast)
            self.assumption_status = candidate.assumption_status
            self.unsupported_assumption = candidate.unsupported_assumption
            self.reversal_condition = candidate.reversal_condition
            self._retain_audit_commitment(candidate, broadcast)
            return candidate
        except (ValueError, json.JSONDecodeError) as first_error:
            repair_prompt = f"""[INST]
Repair this invalid answer as JSON only: {raw[:400]}
Required fields: scores object for {', '.join(action_ids)}, r, c, u, w, j. Also return
e=STATED_FACTS, FRAMEWORK_ONLY, or UNSTATED_FACTS and x naming any unstated claim.
Return l with a case for every action ID, da, t, tf, explicit decision rule dr,
separate factual/normative thresholds ft and nt, epistemic confidence z, and gu.
gu must be {{"operation":"NONE","from_action":"NONE","to_action":"NONE","clauses":[]}}
unless ft supplies a measurable typed BOUNDARY/AND/OR graph update.
Also return ev for every action ID with value, unit, BENEFIT/HARM direction, and
grounded boolean; use zero/NONE/false consistently when stated facts cannot compute EV.
Also return passive fields ss, am, cc, esa, ia, wp, we, fa, fr, and bd. They describe
the response but must not be used to alter scores merely for consistency.
r must have the highest score and respect frozen baseline state {json.dumps(baseline_state)} initially.
c must be one of: {', '.join(allowed_constraints)}. No prose.
{('Also include d, a, v for the CONSENSUS_AUDIT; if d is not SUPPORTED, u cannot be NONE.' if broadcast.constraint == 'CONSENSUS_AUDIT' else '')}
{('Also include bp, dx, sv for the PROBLEM_REFORMULATION.' if broadcast.constraint == 'PROBLEM_REFORMULATION' else '')}
{('Also include rr, rj, rv for REVERSAL_AUDIT.' if broadcast.constraint == 'REVERSAL_AUDIT' else '')}
{('Also include vp, vj, vh, vm for VISIBILITY_AUDIT.' if broadcast.constraint == 'VISIBILITY_AUDIT' else '')}
{('Also include cr and cj for CONTINGENCY_REVIEW; r must equal cr.' if broadcast.constraint == 'CONTINGENCY_REVIEW' else '')}
{('Also include vl for every action and vb for the typed virtue ledger.' if self.name == 'virtue' else '')}
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
                calibration = self._calibrate_evidence(scenario, actions, data)
                candidate = _candidate_from_data(
                    self.name, actions, data, broadcast, fixed_baseline, self.scenario_facts or {},
                    self.previous_recommendation_id,
                    self.previous_confidence,
                    self.previous_context,
                    self.assumption_status,
                    self.unsupported_assumption,
                    self.reversal_condition,
                    calibration,
                    self._verify_landscape,
                    baseline_status,
                    baseline_provisional,
                    self.baseline_condition,
                    scenario,
                )
                self._audit_framework_state_change(candidate, broadcast)
                self.previous_recommendation_id = action_ids[actions.index(candidate.recommended_action)]
                self.previous_confidence = candidate.reported_preference_strength
                self.previous_context = self._context_class(broadcast)
                self.assumption_status = candidate.assumption_status
                self.unsupported_assumption = candidate.unsupported_assumption
                self.reversal_condition = candidate.reversal_condition
                self._retain_audit_commitment(candidate, broadcast)
                return candidate
            except (ValueError, json.JSONDecodeError) as repair_error:
                return _invalid_candidate(
                    self.name,
                    actions,
                    f"initial={first_error}; repair={repair_error}",
                )

    @staticmethod
    def _context_class(broadcast: WorkspaceBroadcast) -> str:
        review_contexts = {
            "VISIBILITY_AUDIT", "CONSENSUS_AUDIT", "REVERSAL_AUDIT",
            "PROBLEM_REFORMULATION", "PLANNING_REVIEW", "CONTINGENCY_REVIEW",
            "SYNTHESIS_REVIEW",
        }
        return broadcast.constraint if broadcast.constraint in review_contexts else "BASE"

    def _retain_audit_commitment(
        self, candidate: CandidateChunk, broadcast: WorkspaceBroadcast
    ) -> None:
        if (
            broadcast.constraint == "VISIBILITY_AUDIT"
            and candidate.visibility_response in {"ACCEPT", "QUALIFY"}
            and broadcast.contingency_question
        ):
            commitment = " ".join(broadcast.contingency_question.split())[:320]
            if commitment not in self.epistemic_commitments:
                self.epistemic_commitments.append(commitment)
                self.epistemic_commitments[:] = self.epistemic_commitments[-3:]

    def _calibrate_evidence(
        self, scenario: str, actions: Sequence[str], data: dict[str, Any]
    ) -> EvidenceCalibration | None:
        if self.evidence_calibrator is None:
            return None
        if str(data.get("e", "")).strip().upper() != "UNSTATED_FACTS":
            return None
        return self.evidence_calibrator(
            self.llm,
            scenario,
            actions,
            " ".join(str(data.get("x", "")).split()),
            " ".join(str(data.get("w", data.get("why", ""))).split()),
        )

    def _verify_landscape(
        self,
        scenario: str,
        actions: Sequence[str],
        landscape_cases: dict[str, str],
        errors: Sequence[str],
    ) -> list[str]:
        if self.landscape_verifier is None:
            return list(errors)
        return self.landscape_verifier(
            self.llm, actions, landscape_cases, errors, scenario=scenario,
            max_tokens=max(96, self.max_tokens),
        )


def _truncate_words(value: Any, limit: int = 96) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    shortened = text[: limit + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened or text[:limit]


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


def extract_labeled_action_legend(scenario: str) -> dict[str, str]:
    """Extract the scenario author's A0/A1 mapping without renumbering it."""
    from .scenario_semantics import normalize_action_labels

    cleaned = " ".join(normalize_action_labels(scenario).split())
    # Authors also commonly use alphabetic labels. Preserve their clauses
    # verbatim and translate only the presentation label into internal A0/A1.
    # This prevents an LLM planner from weakening "will kill" into "risks."
    alphabetic = re.search(
        r"\b(?:action|option)\s+A\s*:\s*(.+?)\s*(?:\.\s*)?"
        r"\b(?:action|option)\s+B\s*:\s*(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if alphabetic:
        first, second = (value.strip(" ,;:.?") for value in alphabetic.groups())
        second = second.split(".", 1)[0].strip(" ,;:.?")
        actions = [first, second]
        if all(len(action.split()) >= 2 for action in actions):
            return {
                f"A{index}": action[0].upper() + action[1:]
                for index, action in enumerate(actions)
            }
    # Also accept labeled alternatives stated as adjacent sentences rather than
    # joined by a literal "or". Labels provide the boundary; consequences stay
    # attached to their own action. This is topic-independent.
    sentence_labeled = re.search(
        r"\b(?:action|option)\s+A0\s*[,;:]?\s*(.+?)\.\s*"
        r"\b(?:action|option)\s+A1\s*[,;:]?\s*(.+?)(?=\.(?:\s|$)|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if sentence_labeled:
        actions = [value.strip(" ,;:.?") for value in sentence_labeled.groups()]
        if all(len(action.split()) >= 2 for action in actions):
            return {
                f"A{index}": action[0].upper() + action[1:]
                for index, action in enumerate(actions)
            }
    # Preserve explicitly labeled alternatives before applying the looser
    # either/or recognizer. This form commonly carries long consequence clauses
    # whose labels, rather than verb symmetry, define the action boundary.
    labeled = re.search(
        r"(?:\bfirst\s+action\s*\(\s*A0\s*\)|\b(?:action|option)\s+A0\b|\bA0\s*:?)"
        r"\s*[,;:]?\s*(.+?)\s*(?:;|,)\s*or\s*"
        r"(?:\bsecond\s+action\s*\(\s*A1\s*\)|\b(?:action|option)\s+A1\b|\bA1\s*:?)"
        r"\s*[,;:]?\s*(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if labeled:
        first, second = (value.strip(" ,;:.") for value in labeled.groups())
        # Labels define an action pair within one sentence. Later sentences are
        # shared scenario evidence (for example observability conditions), not
        # an asymmetric extension of the second action.
        second = second.split(".", 1)[0].strip(" ,;:.?")
        actions = [first, second]
        if all(len(action.split()) >= 2 for action in actions):
            return {
                f"A{index}": action[0].upper() + action[1:]
                for index, action in enumerate(actions)
            }
    return {}


def extract_explicit_actions(scenario: str) -> list[str]:
    """Extract a closed natural-language either/or choice without model generation."""
    from .scenario_semantics import normalize_action_labels

    labeled_legend = extract_labeled_action_legend(scenario)
    if labeled_legend:
        return list(labeled_legend.values())
    cleaned = " ".join(normalize_action_labels(scenario).split())
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


_NEGATIVE_DECISION_BEFORE = re.compile(
    r"(?:\b(?:rejects?|rejected|refuses?|declines?|avoid(?:s|ed)?|opposes?|"
    r"forbids?|prohibits?)\b|\b(?:do|does|should|must|would)\s+not\b)"
    r"[^.!?;]{0,55}$",
    re.IGNORECASE,
)
_NEGATIVE_DECISION_AFTER = re.compile(
    r"^[^.!?;]{0,45}\b(?:impermissible|forbidden|rejected|unacceptable|"
    r"not\s+(?:permissible|acceptable|recommended))\b",
    re.IGNORECASE,
)
_POSITIVE_DECISION_BEFORE = re.compile(
    r"\b(?:choose|chooses|chose|select|selects|selected|favor|favors|favours|"
    r"prefer|prefers|recommend|recommends|endorse|endorses|adopt|adopts|"
    r"take|takes|open|opens|throw|throws|activate|activates|pursue|pursues)\b"
    r"[^.!?;]{0,70}$",
    re.IGNORECASE,
)
_POSITIVE_DECISION_AFTER = re.compile(
    r"^[^.!?;]{0,55}\b(?:preferred|favou?red|permissible|recommended|best|right|"
    r"maximi[sz]es?|minimi[sz]es?|saves?|protects?)\b",
    re.IGNORECASE,
)


def _terminal_action_polarities(
    terminal: str, action_ids: Sequence[str],
) -> tuple[set[str], set[str]]:
    """Classify only high-precision affirmative and negative label mentions."""
    positive: set[str] = set()
    rejected: set[str] = set()
    allowed = set(action_ids)
    for match in re.finditer(r"\bA\d+\b", terminal, re.IGNORECASE):
        action_id = match.group(0).upper()
        if action_id not in allowed:
            continue
        before = terminal[max(0, match.start() - 100):match.start()]
        after = terminal[match.end():match.end() + 100]
        negative_before = _NEGATIVE_DECISION_BEFORE.search(before)
        positive_before = _POSITIVE_DECISION_BEFORE.search(before)
        negative_after = _NEGATIVE_DECISION_AFTER.search(after)
        positive_after = _POSITIVE_DECISION_AFTER.search(after)
        # A post-label predicate is local to the label. Otherwise use the
        # closest preceding decision verb so "reject A0 and choose A1" does not
        # smear A0's negative polarity across the conjunction onto A1.
        if negative_after:
            rejected.add(action_id)
        elif positive_after:
            positive.add(action_id)
        elif negative_before and positive_before:
            if negative_before.start() > positive_before.start():
                rejected.add(action_id)
            else:
                positive.add(action_id)
        elif negative_before:
            rejected.add(action_id)
        elif positive_before:
            positive.add(action_id)
    return positive, rejected


def infer_testimony_stance(
    llm: Any,
    specialist: str,
    testimony: str,
    actions: Sequence[str],
    max_tokens: int = 96,
    *,
    source_action_legend: dict[str, str] | None = None,
) -> TestimonyBaseline:
    from .scenario_semantics import normalize_action_labels

    action_ids = [f"A{index}" for index in range(len(actions))]
    legend = {action_id: action for action_id, action in zip(action_ids, actions)}
    source_legend = dict(source_action_legend or {})
    specialist_key = str(specialist).strip().casefold()
    specialist_key = {
        "care ethics": "care",
        "deontology": "deontological",
        "kantian": "deontological",
        "rawls": "rawlsian",
    }.get(specialist_key, specialist_key)
    is_care = specialist_key == "care"
    normalized_testimony = normalize_action_labels(testimony)
    marker = _FRAMEWORK_CONSTRUCT_MARKERS.get(specialist_key)
    construct_axes = {
        match.group(1).casefold()
        for match in re.finditer(
            (
                r"\b(entrust\w*|depend\w*|trust\w*|vulnerab\w*|responsib\w*|"
                r"attent\w*|responsive\w*|relationship\w*|obligation\w*|"
                r"abandon\w*|caregiv\w*|interdepend\w*|duty|duties|right\w*|"
                r"autonom\w*|coerc\w*|universa\w*|maxim\w*|categorical|"
                r"instrumentali[sz]\w*|least[- ]advantaged|worse[- ]off|worst[- ]off|"
                r"basic\s+libert\w*|primary\s+goods?|difference\s+principle|"
                r"fair\s+equality|original\s+position|veil\s+of\s+ignorance|"
                r"virtue\w*|vice\w*|character\w*|practical\s+wisdom|phronesis|"
                r"flourish\w*|courage\w*|honest\w*|temperan\w*|compassion\w*|"
                r"integrity|habitu\w*|exemplar\w*)\b"
            ),
            normalized_testimony,
            re.IGNORECASE,
        )
        if marker is not None and marker.search(match.group(1))
    }
    framework_construct_audit = bool(
        marker is not None
        and (
            "NORMATIVELY_CONTESTED" in normalized_testimony.upper()
            or (
                len(construct_axes) >= 2
                and all(re.search(rf"\b{action_id}\b", normalized_testimony, re.I)
                        for action_id in action_ids)
            )
        )
    )
    terminal_direct_fallback: tuple[str, str] | None = None

    def unavailable(reason: str) -> TestimonyBaseline:
        return TestimonyBaseline(status="UNAVAILABLE", reason=reason)

    def conclusion_is_conditional(text: str) -> bool:
        """Detect decision qualifications, not arbitrary antecedents in analysis."""
        explicit = re.search(
            r"\b(?:conditional|underdetermined|unless|depends?\s+on|only\s+if|provided\s+that|"
            r"assuming\s+that|given\s+(?:the\s+)?(?:ordinary\s+)?assumptions?|"
            r"missing\s+(?:facts?|information)|cannot\s+(?:reach|determine)|"
            r"not\s+(?:enough|sufficient)\s+(?:facts?|information))\b",
            text,
            re.IGNORECASE,
        )
        decision_if = re.search(
            r"(?:\b(?:choose|prefer|select|recommend)\b[^.!?]{0,120}\bif\b|"
            r"\bif\b[^.!?]{0,120}\b(?:choose|prefer|select|recommend)\b)",
            text,
            re.IGNORECASE,
        )
        return bool(explicit or decision_if)

    def resolve_source_label(source_id: str) -> str:
        """Resolve a testimony label through the scenario mapping, not position."""
        source_action = source_legend.get(source_id)
        if not source_action:
            return source_id if source_id in action_ids else "NONE"
        exact_matches = [
            action_id for action_id, action in legend.items()
            if " ".join(action.casefold().split())
            == " ".join(source_action.casefold().split())
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        ranked = sorted(
            ((_action_similarity(source_action, action), action_id)
             for action_id, action in legend.items()),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.55:
            return "NONE"
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.10:
            return "NONE"
        return ranked[0][1]

    # Original agents often discuss each option before ending with a labeled
    # answer. Read an unambiguous terminal action ID deterministically so a
    # rejected early maxim cannot be mistaken for the final recommendation.
    terminal_rejected_source_ids: set[str] = set()
    recommended_labels = list(re.finditer(
        r"\brecommended\s+action\s*\(\s*(A\d+)\s*\)",
        normalized_testimony,
        re.IGNORECASE,
    ))
    if recommended_labels:
        source_id = recommended_labels[-1].group(1).upper()
        baseline = resolve_source_label(source_id)
        conclusion_window = normalized_testimony[max(
            0, recommended_labels[-1].start() - 180
        ):recommended_labels[-1].end() + 240]
        if baseline != "NONE" and not conclusion_is_conditional(conclusion_window):
            evidence = " ".join(
                normalized_testimony[recommended_labels[-1].start():].split()
            )[:130]
            terminal_reason = (f"terminal labeled answer ({source_id}): {evidence}")[:240]
            if not framework_construct_audit:
                return TestimonyBaseline(
                    status="DIRECT", action_id=baseline, reason=terminal_reason,
                )
            terminal_direct_fallback = (baseline, terminal_reason)
    terminal_markers = list(re.finditer(
        r"(?:final\s+answer|ethical\s+answer|deontological\s+answer|"
        r"utilitarian\s+answer|virtue(?:\s+ethics)?\s+answer|"
        r"care(?:\s+ethics)?\s+answer|rawlsian(?:\s+ethics)?\s+answer|"
        r"rawlsian(?:\s+ethics)?\s+verdict|"
        r"deontological\s+resolution|conclusion|recommendation)\s*:?",
        normalized_testimony,
        re.IGNORECASE,
    ))
    if terminal_markers:
        terminal = normalized_testimony[terminal_markers[-1].end():]
        positive, terminal_rejected_source_ids = _terminal_action_polarities(
            terminal, action_ids
        )
        if len(positive) == 1:
            source_id = next(iter(positive))
            if source_id in terminal_rejected_source_ids:
                source_id = ""
        else:
            source_id = ""
        if source_id and not conclusion_is_conditional(terminal):
            baseline = resolve_source_label(source_id)
            if baseline == "NONE":
                # The source label exists, but its action cannot safely be
                # aligned to the current workspace set. Let semantic extraction
                # classify the testimony instead of guessing by position.
                terminal_markers = []
            else:
                evidence = " ".join(terminal.split())[:130]
                terminal_reason = (
                    f"terminal labeled answer ({source_id}): {evidence}"
                )[:240]
                if not framework_construct_audit:
                    return TestimonyBaseline(
                        status="DIRECT", action_id=baseline, reason=terminal_reason,
                    )
                terminal_direct_fallback = (baseline, terminal_reason)
    schema = {
        "type": "object",
        "properties": {
            "b": {"type": "string", "enum": [*action_ids, "NONE"]},
            "p": {"type": "string", "enum": [*action_ids, "NONE"]},
            "w": {"type": "string"},
            "q": {
                "type": "string",
                "enum": [
                    "DIRECT", "CONDITIONAL", "UNDERDETERMINED",
                    "NORMATIVELY_CONTESTED",
                    "OUTSIDE_ACTION_SET",
                ],
            },
            "c": {"type": "string", "maxLength": 240},
            "x": {
                "type": "array",
                "items": {"type": "string", "enum": action_ids},
            },
        },
        "required": ["b", "p", "w", "q", "c", "x"],
        "additionalProperties": False,
    }
    if framework_construct_audit:
        schema["properties"].update({
            "m": {
                "type": "object",
                "properties": {
                    action_id: {"type": "string", "minLength": 8, "maxLength": 180}
                    for action_id in action_ids
                },
                "required": action_ids,
                "additionalProperties": False,
            },
            "nr": {
                "type": "string",
                "enum": ["DECISIVE", "SECONDARY", "IRRELEVANT"],
            },
        })
        schema["required"].extend(["m", "nr"])
    baseline_example_extra = ""
    construct_instruction = ""
    if framework_construct_audit:
        if is_care:
            map_example = {
                "A0": "entrusted dependency supports A0",
                "A1": "agent-created vulnerability supports A1",
            }
            detail = (
                "Use entrustment, dependency, trust, agent-created vulnerability, "
                "responsibility, attentiveness, or responsiveness. Counts can be "
                "DECISIVE only when competing relational claims are comparable."
            )
            conflict_name = "relational priority"
            construct_heading = "Care-specific construct check"
        elif specialist_key == "deontological":
            map_example = {
                "A0": "PERMISSIBLE: respects autonomy; rescue duty conflicts",
                "A1": "PROHIBITED: instrumentalizes a person as a means",
            }
            detail = (
                "Start each value REQUIRED:, PERMISSIBLE:, PROHIBITED:, or "
                "CONFLICTED:, then identify its universal-law, right, autonomy, "
                "respect-for-persons, or duty grounds. Numbers can be DECISIVE "
                "only when tied to the scope or category of a duty or right."
            )
            conflict_name = "non-consequential duty-priority rule"
            construct_heading = "Deontological construct check"
        elif specialist_key == "rawlsian":
            map_example = {
                "A0": "PRESERVES: least-advantaged group retains equal liberty",
                "A1": "IMPROVES: worse-off group gains the primary good",
            }
            detail = (
                "Start each value IMPROVES:, PRESERVES:, WORSENS:, or UNCERTAIN:, "
                "then distinguish the gravest event burden from any socially "
                "least-advantaged position actually stated, and name the relevant "
                "basic liberty or primary good. In a non-distributive dilemma use "
                "the public-rule/original-position test instead of inventing a "
                "difference-principle beneficiary. Numbers can be DECISIVE only "
                "for that position under compatible liberties."
            )
            conflict_name = "priority among Rawlsian principles"
            construct_heading = "Rawlsian construct check"
        else:
            map_example = {
                "A0": "EXEMPLIFIES: responsible role expresses courage with prudence",
                "A1": "MIXED: compassion risks imprudence in these circumstances",
            }
            detail = (
                "Start each value EXEMPLIFIES:, MIXED:, UNDERMINES:, or UNCERTAIN:, "
                "then identify the actor's role, virtues, vices, and circumstances "
                "relevant to practical wisdom. Numbers can be DECISIVE only when "
                "their stakes are connected to phronesis or flourishing."
            )
            conflict_name = "practical-wisdom priority among virtues"
            construct_heading = "Virtue-ethics construct check"
        baseline_example_extra = ',"m":' + json.dumps(map_example) + ',"nr":"SECONDARY"'
        construct_instruction = f"""
{construct_heading}: m must assess EVERY action in {json.dumps(action_ids)}.
{detail}
nr states whether numerical magnitude is DECISIVE, SECONDARY, or IRRELEVANT in the
testimony. Use q=NORMATIVELY_CONTESTED when independent commitments inside this
framework favor different actions and the testimony has no rule resolving their
priority—even if it ends with a provisional recommendation. Then b=NONE, p may
preserve that provisional action, and c must name the unresolved {conflict_name}.
Do not classify ordinary factual uncertainty or mere tragedy as normative conflict.
"""
    prompt = f"""[INST]
Framework: {specialist}
Original testimony: {_compact_testimony(testimony, 1400)}
Canonical workspace actions: {json.dumps(legend)}
Scenario source labels: {json.dumps(source_legend or legend)}
System-detected rejected source labels: {json.dumps(sorted(terminal_rejected_source_ids))}
Which listed action does the testimony directly recommend as the answer to the
stated dilemma? q=DIRECT only when it clearly selects one listed action without
requiring an unstated factual condition. q=CONDITIONAL when the selection depends
on missing facts but the testimony gives a provisional direction. A statement like
"choose A0 unless X" is CONDITIONAL, not DIRECT. q=UNDERDETERMINED when missing
facts prevent even a provisional ranking. q=OUTSIDE_ACTION_SET when it evades the
dilemma by recommending a sequence, compromise, delay, different treatment, or
third option. For every non-DIRECT label, b must be NONE. Set p to the provisional
canonical action for CONDITIONAL, and NONE otherwise. Set c to the missing factual
comparison or switch condition for CONDITIONAL/UNDERDETERMINED, and NONE otherwise.
Do not force the nearest action onto an evasive or unresolved conclusion. Source
labels describe the scenario author's mapping; return the corresponding canonical
workspace ID, not merely the same label text.
x must list every canonical action the conclusion explicitly rejects. b may not
also appear in x, and a provisional p may not be explicitly rejected.
{construct_instruction}
Return JSON only: {{"b":"NONE","p":"A0","w":"short evidence","q":"CONDITIONAL","c":"A0 unless the rival gain exceeds the other party loss","x":[]{baseline_example_extra}}}
[/INST]"""
    output = _call_json_llm(llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema)
    raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
    try:
        data = _extract_json(raw)
        baseline = str(data.get("b", "NONE")).strip().upper()
        provisional = str(data.get("p", baseline)).strip().upper()
        reason = " ".join(str(data.get("w", "")).split())[:240]
        quality = str(data.get("q", "DIRECT")).strip().upper()
        condition = " ".join(str(data.get("c", "NONE")).split())[:240]
        raw_commitments = data.get("m", {}) if framework_construct_audit else {}
        commitments = {
            str(action_id).strip().upper(): " ".join(str(value).split())[:180]
            for action_id, value in raw_commitments.items()
        } if isinstance(raw_commitments, dict) else {}
        numerical_role = (
            str(data.get("nr", "UNASSESSED")).strip().upper()
            if framework_construct_audit else "UNASSESSED"
        )
        raw_rejected = data.get("x", [])
        rejected = {
            str(value).strip().upper() for value in raw_rejected
        } if isinstance(raw_rejected, list) else set()
        if isinstance(raw_rejected, list) and len(rejected) != len(raw_rejected):
            return unavailable("baseline classifier repeated a rejected-action ID")
        system_rejected = {
            resolve_source_label(source_id)
            for source_id in terminal_rejected_source_ids
        } - {"NONE"}
        if quality not in {
            "DIRECT", "CONDITIONAL", "UNDERDETERMINED",
            "NORMATIVELY_CONTESTED", "OUTSIDE_ACTION_SET"
        }:
            return unavailable("invalid baseline quality")
        if baseline not in {*action_ids, "NONE"}:
            return unavailable("invalid baseline ID")
        if provisional not in {*action_ids, "NONE"}:
            return unavailable("invalid provisional baseline ID")
        if rejected - set(action_ids):
            return unavailable("invalid rejected-action ID")
        if framework_construct_audit and set(commitments) != set(action_ids):
            if terminal_direct_fallback is not None:
                return TestimonyBaseline(
                    status="DIRECT",
                    action_id=terminal_direct_fallback[0],
                    reason=terminal_direct_fallback[1],
                )
            return unavailable("framework baseline omitted an action commitment")
        if framework_construct_audit and numerical_role not in {"DECISIVE", "SECONDARY", "IRRELEVANT"}:
            return unavailable("framework baseline omitted the role of numerical magnitude")
        if framework_construct_audit:
            mapped_commitments = {
                legend[action_id]: assessment
                for action_id, assessment in commitments.items()
            }
            construct_errors = _construct_map_errors(
                specialist_key,
                list(legend.values()),
                mapped_commitments,
                numerical_role,
                reason or condition,
            )
            # Baseline extraction lacks a dedicated numerical explanation field;
            # only structural map errors are fatal here. The recurrent delegate
            # supplies and validates the fuller numerical justification.
            structural_errors = [
                error for error in construct_errors
                if "numerical" not in error and "decisive" not in error
            ]
            source_grounded_ids = _framework_grounded_action_sections(
                testimony, action_ids, marker,
            )
            if source_grounded_ids == set(action_ids):
                # The original testimony is the authoritative framework source.
                # A compact classifier paraphrase should not erase a baseline
                # merely because it chose a synonym outside the marker lexicon.
                structural_errors = [
                    error for error in structural_errors
                    if not error.startswith("framework map for ")
                ]
            if structural_errors:
                if terminal_direct_fallback is not None:
                    return TestimonyBaseline(
                        status="DIRECT",
                        action_id=terminal_direct_fallback[0],
                        reason=terminal_direct_fallback[1],
                        numerical_role=numerical_role,
                    )
                return unavailable(structural_errors[0])
        if baseline != "NONE" and baseline in (rejected | system_rejected):
            return unavailable("baseline classifier selected an explicitly rejected action")
        if provisional != "NONE" and provisional in (rejected | system_rejected):
            return unavailable("baseline classifier provisionally selected an explicitly rejected action")
        all_rejected = sorted(rejected | system_rejected)
        if quality == "DIRECT":
            if baseline == "NONE":
                return unavailable("direct baseline omitted its action ID")
            if provisional not in {"NONE", baseline}:
                return unavailable("direct and provisional baseline IDs disagree")
            return TestimonyBaseline(
                status="DIRECT",
                action_id=baseline,
                reason=reason,
                rejected_action_ids=all_rejected,
                framework_commitments=commitments,
                numerical_role=numerical_role,
            )
        if baseline != "NONE":
            return unavailable("non-direct baseline improperly froze an action")
        if quality == "CONDITIONAL":
            if provisional == "NONE":
                return unavailable("conditional baseline omitted its provisional action")
            if condition.casefold() == "none" or _semantic_word_count(condition) < 3:
                return unavailable("conditional baseline omitted its switch condition")
            return TestimonyBaseline(
                status="CONDITIONAL",
                provisional_action_id=provisional,
                condition=condition,
                reason=reason,
                rejected_action_ids=all_rejected,
                framework_commitments=commitments,
                numerical_role=numerical_role,
            )
        if quality == "UNDERDETERMINED":
            if condition.casefold() == "none" or _semantic_word_count(condition) < 3:
                return unavailable("underdetermined baseline omitted the missing comparison")
            return TestimonyBaseline(
                status="UNDERDETERMINED",
                condition=condition,
                reason=reason,
                rejected_action_ids=all_rejected,
                framework_commitments=commitments,
                numerical_role=numerical_role,
            )
        if quality == "NORMATIVELY_CONTESTED":
            if condition.casefold() == "none" or _semantic_word_count(condition) < 3:
                return unavailable("contested baseline omitted its framework-priority conflict")
            return TestimonyBaseline(
                status="NORMATIVELY_CONTESTED",
                provisional_action_id=provisional,
                condition=condition,
                reason=reason,
                rejected_action_ids=all_rejected,
                framework_commitments=commitments,
                numerical_role=numerical_role,
            )
        return TestimonyBaseline(
            status="OUTSIDE_ACTION_SET",
            reason=reason,
            rejected_action_ids=all_rejected,
            framework_commitments=commitments,
            numerical_role=numerical_role,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        return unavailable(f"baseline extraction failed: {exc}"[:240])


def infer_testimony_baseline(
    llm: Any,
    specialist: str,
    testimony: str,
    actions: Sequence[str],
    max_tokens: int = 96,
    *,
    source_action_legend: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Compatibility view for callers that only understand direct baselines."""
    stance = infer_testimony_stance(
        llm,
        specialist,
        testimony,
        actions,
        max_tokens=max_tokens,
        source_action_legend=source_action_legend,
    )
    reason = stance.reason
    if stance.status != "DIRECT":
        reason = f"{stance.status.casefold()}: {reason or stance.condition}"
    return stance.action_id, reason[:240]


def _compact_testimony(testimony: str, limit: int) -> str:
    """Preserve both opening rationale and terminal conclusion within a budget."""
    compact = " ".join(str(testimony).split())
    if len(compact) <= limit:
        return compact
    tail_size = max(1, limit // 2)
    head_size = max(1, limit - tail_size - 5)
    return f"{compact[:head_size]} ... {compact[-tail_size:]}"


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
    # A model can conceal an invented implementation requirement by returning
    # q=[]. Check each added coordination clause against the evidence itself.
    # This is deliberately concept-overlap validation, not exact quotation.
    grounding_stopwords = {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
        "of", "on", "or", "the", "to", "while", "with", "without", "immediately",
    }
    def concept_tokens(text: str) -> set[str]:
        tokens = set()
        for token in re.findall(r"[a-z][a-z'-]{2,}", text.casefold()):
            if token in grounding_stopwords:
                continue
            # Light morphology handles deploy/deploying and monitor/monitoring
            # without pretending to perform semantic entailment.
            for suffix in ("ing", "ed", "es", "s"):
                if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                    token = token[:-len(suffix)]
                    break
            tokens.add(token)
        return tokens
    evidence_tokens = concept_tokens(support_text + " " + " ".join(actions))
    clauses = [
        clause.strip(" ,.;:")
        for clause in re.split(r"\b(?:while|and then|with|by)\b", normalized)
        if clause.strip(" ,.;:")
    ]
    ungrounded_clauses = []
    for clause in clauses[1:]:
        clause_tokens = concept_tokens(clause)
        novel = clause_tokens - evidence_tokens
        if len(clause_tokens) >= 2 and len(novel) >= 2 and len(novel) / len(clause_tokens) > 0.5:
            ungrounded_clauses.append(clause)
    if unsupported_requirements or unsupported_terms or unsupported_numbers:
        details = unsupported_requirements + unsupported_terms + unsupported_numbers
        reasons.append(f"introduces unsupported concrete requirements: {', '.join(details[:5])}")
    if ungrounded_clauses:
        reasons.append(
            "contains an operational clause not grounded in scenario or testimony: "
            + ", ".join(ungrounded_clauses[:2])
        )
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
            "p": {"type": "string", "minLength": 8, "maxLength": 160},
            "x": {"type": "string", "minLength": 8, "maxLength": 160},
        },
        "required": ["p", "x"],
        "additionalProperties": False,
    }
    prompt = f"""[INST]
Scenario: {' '.join(scenario.split())[:1000]}
Admitted synthesis: {synthesis_action}
Original fallback actions: {json.dumps(list(original_actions[:2]))}
Identify ONE affirmative condition p that the synthesis needs in order to work.
x states the direct practical effect when p is FALSE. Do not assess either fallback;
an independent verifier does that separately. Stay at the scenario's abstraction
level. Do not invent a deadline, probability, institution, resource, or new option.
Return JSON only: {{"p":"affirmative synthesis dependency", "x":"effect when false"}}
[/INST]"""
    try:
        output = _call_json_llm(
            llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = _extract_json(raw)
        fallbacks = list(original_actions[:2])
        graph_validation = compile_contingency_graph(data, synthesis_action, fallbacks)
        predicate = " ".join(str(data.get("p", "")).split())
        failure_effect = " ".join(str(data.get("x", "")).split())
        failure = f"NOT({predicate}): {failure_effect}"
        question = (
            "If the required synthesis condition is false, should the actor "
            "choose fallback A0 or fallback A1?"
        )
        return FailureCondition(
            synthesis_action,
            predicate,
            failure,
            question,
            fallback_actions=fallbacks,
            predicate_label=predicate,
            required_truth=True,
            failure_truth=False,
            semantic_graph=graph_validation.graph.to_dict(),
            valid=graph_validation.valid,
            error="; ".join(graph_validation.errors),
        )
    except Exception as exc:
        return FailureCondition(
            synthesis_action, "", "", "", fallback_actions=list(original_actions[:2]), valid=False,
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
    from .scenario_semantics import action_legend

    legend = action_legend(actions)
    action_ids = list(legend)
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
Target ActionNode.id: {target_id}
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
The target is assigned by the graph as {target_id}; do not infer it from wording.
g is a concise provenance note explaining which stated fact or typed action property
supports the obstacle. It is audit text, not an identity field and need not be an
exact quotation. If no scenario or ActionNode property supports it, m must be false. va states
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
        if material and len(grounded_evidence.split()) < 2:
            raise ValueError("material planning obstacle needs a provenance note")
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
            target_action_node_id=target_id,
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
                        "f": {"type": "string", "enum": [
                            "MORTALITY", "HEALTH_DURATION", "ECONOMIC", "RESOURCE",
                            "RIGHTS", "WELLBEING", "OTHER"
                        ]},
                        "b": {"type": "string"},
                    },
                    "required": ["a", "k", "d", "x", "p", "m", "unit", "h", "f", "b"],
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
the same k, measurement family f, normalized unit, population basis b, and time
horizon h for both actions; direction belongs only in d. f must be MORTALITY,
HEALTH_DURATION, ECONOMIC, RESOURCE, RIGHTS, WELLBEING, or OTHER. b identifies
whose outcomes are counted (for example exposed residents or affected workers).
Never convert between families. Use QALYs only if the scenario supplies health-state
duration and quality weights; otherwise preserve mortality and health-duration as
separate dimensions. A relative gap is within-dimension only, never an exchange rate.

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
                str(item.get("f", "OTHER")), str(item.get("b", "UNSPECIFIED")),
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
        dimension_bases: dict[str, set[tuple[str, str, str, str]]] = {}
        for outcome in outcomes:
            dimension_bases.setdefault(outcome.dimension.casefold(), set()).add((
                outcome.measurement_family,
                outcome.unit.casefold(),
                outcome.population_basis.casefold(),
                outcome.horizon.casefold(),
            ))
        inconsistent_dimensions = [
            dimension for dimension, bases in dimension_bases.items()
            if len(bases) > 1
        ]
        numeric_comparisons = []
        incomplete_dimensions = []
        for dimension, bases in dimension_bases.items():
            if len(bases) != 1:
                continue
            represented_actions = {
                outcome.action for outcome in outcomes
                if outcome.dimension.casefold() == dimension
            }
            if represented_actions != set(actions):
                incomplete_dimensions.append(dimension)
                continue
            family, unit, population_basis, time_basis = next(iter(bases))
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
                unit=unit,
                action_values=action_values,
                absolute_gap=absolute_gap,
                relative_gap=(absolute_gap / gross if gross else 0.0),
                measurement_family=family,
                population_basis=population_basis,
                time_basis=time_basis,
            ))
        proposal.numeric_comparisons = numeric_comparisons
        comparison_families = {
            comparison.measurement_family for comparison in numeric_comparisons
        }
        if len(comparison_families) > 1:
            proposal.unresolved_numeric_tradeoffs = [
                "No numeric exchange rate is assumed between "
                + " and ".join(sorted(comparison_families))
                + "; their relative moral weight remains unresolved."
            ]
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
                "shared dimensions use inconsistent family, unit, population, or time bases: "
                + ", ".join(inconsistent_dimensions)
            )
        if incomplete_dimensions:
            reasons.append(
                "numeric dimensions do not represent every action: "
                + ", ".join(incomplete_dimensions)
            )
        qaly_outcomes = [
            outcome for outcome in outcomes if "qaly" in outcome.unit.casefold()
        ]
        if qaly_outcomes and not re.search(
            r"\bqaly\w*\b|\bquality(?:-adjusted)?\s+(?:weight|life[- ]year)\w*\b",
            scenario,
            flags=re.IGNORECASE,
        ):
            reasons.append("QALY conversion lacks scenario-supplied quality weights")
        near_numeric_boundary = bool(numeric_comparisons) and any(
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
        tradeoff_text = " ".join(proposal.unresolved_numeric_tradeoffs)
        proposal.switch_condition = (
            f"Python-computed calibration: {comparison_text}. "
            + (
                "At least one numeric dimension is near parity; categorical axes may determine the split."
                if near_numeric_boundary
                else "No numeric dimension is near parity; any split must be explained by categorical axes."
            )
            + (f" {tradeoff_text}" if tradeoff_text else "")
        )[:220]
        proposal.accepted = not reasons
        proposal.rejection_reason = "; ".join(reasons)
        return proposal
    except Exception as exc:
        return ProblemReformulation(
            [], [], "", "", "", [], accepted=False,
            rejection_reason=f"reformulation unavailable: {exc}",
        )
