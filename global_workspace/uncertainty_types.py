"""Typed uncertainty categories for audit and graph machinery.

VERIFY_FACTS — unresolved empirical / world-state proposition.
DECISION_BOUNDARY — already-derived threshold/counterfactual for ranking sensitivity.
NORMATIVE_ADJUDICATION — unresolved rule/priority/scope conflict inside a framework.

Old names may parse; only canonical forms serialize. Deterministically identifiable
VERIFY_FACTS mislabels that are structural boundaries are normalized on admission.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

VERIFY_FACTS = "VERIFY_FACTS"
DECISION_BOUNDARY = "DECISION_BOUNDARY"
NORMATIVE_ADJUDICATION = "NORMATIVE_ADJUDICATION"

CANONICAL_UNCERTAINTY_CATEGORIES = frozenset({
    VERIFY_FACTS,
    DECISION_BOUNDARY,
    NORMATIVE_ADJUDICATION,
})

# One-migration parse aliases. Only canonical names serialize.
_CATEGORY_PARSE_ALIASES = {
    "RESOLVE_NORMATIVE_TENSION": NORMATIVE_ADJUDICATION,
}

BOUNDARY_STATUS_VALUES = ("NOT_CROSSED", "CROSSED", "UNRESOLVED")
EXPECTED_EFFECT_VALUES = (
    "NO_CHANGE",
    "WEAKENS",
    "REVERSES_FRAMEWORK_PREFERENCE",
)

# Normative adjudication relation subtypes for the graph / audit payload.
NORMATIVE_RELATION_TYPES = frozenset({
    "PRIORITY_CONFLICT",
    "SCOPE_CONFLICT",
    "STRICT_DUTY_CONFLICT",
    "NORMATIVE_ADJUDICATION",
})

_CONDITIONAL_FORM = re.compile(
    r"\b(?:if|unless|provided\s+that|when|whenever|should|were)\b",
    re.IGNORECASE,
)
_RANKING_EFFECT = re.compile(
    r"\b(?:revers(?:e|es|ed|al)?|weaken(?:s|ed)?|switch(?:es|ed)?|"
    r"chang(?:e|es|ed)|flip(?:s|ped)?|prefer(?:s|ence)?|ranking|"
    r"threshold|support\s+for|utilitarian\s+support)\b",
    re.IGNORECASE,
)
# Conservative: measurable threshold vocabulary or numeric comparison —
# not bare "over"/"under" or action IDs like A0.
_THRESHOLD_ANTECEDENT = re.compile(
    r"(?:"
    r"[<>]=?\s*\d|"
    r"(?<![A-Za-z])\d+(?:\.\d+)?\s*"
    r"(?:%|percent|months?|weeks?|days?|years?|hours?|minutes?|"
    r"lives?|people|persons?|patients?)\b|"
    r"\b(?:less|more|fewer|greater|under|over|below|above|shorter|longer|"
    r"at\s+least|at\s+most)\s+(?:than\s+)?\d|"
    r"\b(?:threshold|duration|magnitude|probability|frequency|"
    r"uptake|survival|permanen\w*|temporary|short[- ]term|long[- ]term)\b"
    r")",
    re.IGNORECASE,
)
_LEGACY_EFFECT = {
    "NO_CHANGE": "NO_CHANGE",
    "WEAKENS": "WEAKENS",
    "REVERSES": "REVERSES_FRAMEWORK_PREFERENCE",
    "REVERSES_FRAMEWORK_PREFERENCE": "REVERSES_FRAMEWORK_PREFERENCE",
    "UNRESOLVED": "UNRESOLVED",
}


@dataclass(frozen=True)
class BoundaryPayload:
    condition: str
    boundary_status: str = "UNRESOLVED"
    expected_effect: str = "REVERSES_FRAMEWORK_PREFERENCE"
    target_framework: str = ""
    target_claim_key: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "category": DECISION_BOUNDARY,
            "relation": DECISION_BOUNDARY,
            "condition": self.condition,
            "boundary_status": self.boundary_status,
            "expected_effect": self.expected_effect,
            "possible_values": list(BOUNDARY_STATUS_VALUES),
        }
        if self.target_framework:
            payload["target_framework"] = self.target_framework
        if self.target_claim_key:
            payload["target_claim_key"] = self.target_claim_key
        return payload


def normalize_uncertainty_category(category: str | None) -> str:
    raw = str(category or "").strip().upper()
    if not raw or raw == "NONE":
        return raw or "NONE"
    if raw in CANONICAL_UNCERTAINTY_CATEGORIES:
        return raw
    return _CATEGORY_PARSE_ALIASES.get(raw, raw)


def normalize_unresolved_marker(unresolved: str | None) -> str:
    """Normalize specialist.unresolved markers; preserve non-category sentinels."""
    raw = str(unresolved or "NONE").strip().upper() or "NONE"
    if raw == "NONE":
        return "NONE"
    return normalize_uncertainty_category(raw)


def looks_like_decision_boundary(
    proposition: str,
    *,
    relation: str = "",
    expected_effect: str = "",
) -> bool:
    """Conservative structural test: conditional + threshold + ranking effect."""
    relation_name = str(relation or "").strip().upper()
    if relation_name == DECISION_BOUNDARY:
        return True
    text = " ".join(str(proposition or "").split())
    if not text:
        return False
    if not _CONDITIONAL_FORM.search(text):
        return False
    if not _THRESHOLD_ANTECEDENT.search(text):
        return False
    effect = str(expected_effect or "").strip().upper()
    if effect in {"WEAKENS", "REVERSES", "REVERSES_FRAMEWORK_PREFERENCE"}:
        return True
    return bool(_RANKING_EFFECT.search(text))


def looks_like_normative_adjudication(proposition: str, *, relation: str = "") -> bool:
    relation_name = str(relation or "").strip().upper()
    if relation_name in NORMATIVE_RELATION_TYPES or relation_name in {
        "RESOLVE_NORMATIVE_TENSION", NORMATIVE_ADJUDICATION,
    }:
        return True
    text = " ".join(str(proposition or "").split()).casefold()
    if not text:
        return False
    return bool(re.search(
        r"\b(?:strict\s+(?:duty|right)|perfect\s+dut|priority|which\s+horn|"
        r"normative|adjudicat|duty\s+conflict|rights?\s+conflict|"
        r"entrusted|scope\s+of\s+(?:the\s+)?(?:duty|right)|"
        r"non-?maleficence|should\s+govern|competing\s+dut)\b",
        text,
    ))


def infer_expected_effect(proposition: str, default: str = "REVERSES_FRAMEWORK_PREFERENCE") -> str:
    text = " ".join(str(proposition or "").split()).casefold()
    if re.search(r"\bweaken", text):
        return "WEAKENS"
    if re.search(r"\b(?:no\s+change|unchanged|does\s+not\s+change)\b", text):
        return "NO_CHANGE"
    if re.search(r"\b(?:revers|switch|flip|chang)", text):
        return "REVERSES_FRAMEWORK_PREFERENCE"
    return default if default in EXPECTED_EFFECT_VALUES else "REVERSES_FRAMEWORK_PREFERENCE"


def extract_boundary_condition(proposition: str) -> str:
    text = " ".join(str(proposition or "").split())
    match = re.search(
        r"\b(?:if|unless|provided\s+that|when|whenever)\b(.+?)(?:"
        r"\b(?:then|,|:)\b|\b(?:revers|weaken|switch|chang|prefer|ranking)\b|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return " ".join(match.group(0).split())[:240]
    return text[:240]


def build_boundary_audit_fields(
    *,
    condition: str,
    expected_effect: str = "REVERSES_FRAMEWORK_PREFERENCE",
    target_framework: str = "",
    target_claim_key: str = "",
    boundary_status: str = "UNRESOLVED",
) -> dict[str, Any]:
    effect = _LEGACY_EFFECT.get(
        str(expected_effect or "").strip().upper(),
        "REVERSES_FRAMEWORK_PREFERENCE",
    )
    if effect not in EXPECTED_EFFECT_VALUES:
        effect = "REVERSES_FRAMEWORK_PREFERENCE"
    status = str(boundary_status or "UNRESOLVED").strip().upper()
    if status not in BOUNDARY_STATUS_VALUES:
        status = "UNRESOLVED"
    return BoundaryPayload(
        condition=" ".join(str(condition or "").split())[:240],
        boundary_status=status,
        expected_effect=effect,
        target_framework=" ".join(str(target_framework or "").split())[:48],
        target_claim_key=" ".join(str(target_claim_key or "").split())[:120],
    ).as_dict()


def normalize_audit_variable(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Serialize canonical uncertainty typing onto an admitted audit variable."""
    if not isinstance(raw, dict) or not raw:
        return {}
    payload = dict(raw)
    proposition = " ".join(str(
        payload.get("proposition")
        or payload.get("question")
        or payload.get("condition")
        or ""
    ).split())
    relation = str(payload.get("relation", "") or "").strip().upper()
    raw_category = str(payload.get("category") or "").strip().upper()
    # Prefer explicit category; fall back to relation only for known uncertainty types.
    seed = raw_category
    if not seed and relation in {
        VERIFY_FACTS, DECISION_BOUNDARY, NORMATIVE_ADJUDICATION,
        "RESOLVE_NORMATIVE_TENSION",
        *NORMATIVE_RELATION_TYPES,
    }:
        seed = relation
    category = normalize_uncertainty_category(seed) if seed else ""

    reclassified_from = ""
    review_flag = ""
    if not category or category == "NONE":
        category = VERIFY_FACTS

    if category == VERIFY_FACTS and looks_like_decision_boundary(
        proposition,
        relation=relation,
        expected_effect=str(payload.get("expected_effect", "")),
    ):
        reclassified_from = VERIFY_FACTS
        category = DECISION_BOUNDARY
    elif category == VERIFY_FACTS and (
        relation in NORMATIVE_RELATION_TYPES
        or relation in {"RESOLVE_NORMATIVE_TENSION", NORMATIVE_ADJUDICATION}
    ):
        # Relation already names a normative conflict: normalize category.
        reclassified_from = VERIFY_FACTS
        category = NORMATIVE_ADJUDICATION
    elif category == VERIFY_FACTS and looks_like_normative_adjudication(
        proposition, relation=relation,
    ):
        # Ambiguous prose stays VERIFY_FACTS; flag for review rather than rewrite.
        review_flag = "AMBIGUOUS_UNCERTAINTY_TYPING"

    payload["category"] = category
    if category == DECISION_BOUNDARY:
        payload["relation"] = DECISION_BOUNDARY
        boundary = build_boundary_audit_fields(
            condition=str(payload.get("condition") or extract_boundary_condition(proposition)),
            expected_effect=str(
                payload.get("expected_effect")
                or infer_expected_effect(proposition)
            ),
            target_framework=str(payload.get("target_framework", "") or ""),
            target_claim_key=str(
                payload.get("target_claim_key")
                or payload.get("claim_key")
                or payload.get("question_key")
                or ""
            ),
            boundary_status=str(payload.get("boundary_status", "UNRESOLVED")),
        )
        payload.update(boundary)
        # Preserve entity/question/focus_action from the caller.
        if not payload.get("entity"):
            payload["entity"] = boundary["condition"] or "decision boundary"
        if not payload.get("possible_values"):
            payload["possible_values"] = list(BOUNDARY_STATUS_VALUES)
        else:
            payload["possible_values"] = list(BOUNDARY_STATUS_VALUES)
    elif category == NORMATIVE_ADJUDICATION:
        if relation not in NORMATIVE_RELATION_TYPES:
            if re.search(r"\bstrict\b", proposition, re.I):
                payload["relation"] = "STRICT_DUTY_CONFLICT"
            elif re.search(r"\bscope\b", proposition, re.I):
                payload["relation"] = "SCOPE_CONFLICT"
            else:
                payload["relation"] = "PRIORITY_CONFLICT"
        elif relation == NORMATIVE_ADJUDICATION:
            payload["relation"] = "PRIORITY_CONFLICT"
    else:
        # Leave non-uncertainty relations (COMPARATIVE_MAGNITUDE, etc.) intact.
        if relation in {
            "RESOLVE_NORMATIVE_TENSION", NORMATIVE_ADJUDICATION, DECISION_BOUNDARY,
        }:
            payload["relation"] = "EMPIRICAL_UNKNOWN"
        if category not in CANONICAL_UNCERTAINTY_CATEGORIES:
            payload["category"] = VERIFY_FACTS

    if reclassified_from:
        payload["reclassified_from"] = reclassified_from
        payload["reclassification"] = f"ADMISSION_NORMALIZED:{category}"
    if review_flag:
        payload["typing_review"] = review_flag
    return payload


def uncertainty_kind_for(category: str, question: str = "") -> str:
    """Classify uncertainty independently of the next audit operation."""
    normalized = normalize_uncertainty_category(category)
    if normalized == NORMATIVE_ADJUDICATION:
        return "NORMATIVE_UNCERTAINTY"
    if normalized == DECISION_BOUNDARY:
        return "BOUNDARY_SENSITIVITY"
    text = str(question).casefold()
    empirical = bool(re.search(
        r"\b(?:amount|duration|effect|feasib|frequency|likelihood|magnitude|number|"
        r"probability|rate|survive|size|sufficient|uptake)\b", text,
    ))
    normative = bool(re.search(
        r"\b(?:compar|justify|lexical|moral|outweigh|overrid|priority|relative value|"
        r"versus|vs\.?|weight)\b", text,
    ))
    # Category seeds: VERIFY_FACTS is empirical terrain; feasibility likewise.
    if normalized == VERIFY_FACTS or normalized in {
        "CHECK_FEASIBILITY", "CLARIFY_SCENARIO",
    }:
        empirical = True
    if empirical and normative:
        return "MIXED_UNCERTAINTY"
    if normative:
        return "NORMATIVE_UNCERTAINTY"
    if empirical:
        return "EMPIRICAL_UNCERTAINTY"
    return "UNCLASSIFIED_UNCERTAINTY"
