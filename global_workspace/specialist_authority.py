"""Framework-general specialist adjudicative authority.

Resolved claims can govern. Conditional claims can govern conditionally.
Provisional claims can influence. Contested claims can destabilize confidence.
Any sufficiently important unresolved claim can interrupt.

Invariant: a governing claim may never have stronger epistemic status than the
specialist state that produced it. Compression may remove detail; it may never
increase authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from .uncertainty_types import (
    DECISION_BOUNDARY,
    NORMATIVE_ADJUDICATION,
    normalize_unresolved_marker,
)

# Canonical statuses — only these may serialize.
SUPPORTS = "SUPPORTS"
CONDITIONAL_SUPPORTS = "CONDITIONAL_SUPPORTS"
PROVISIONAL_LEANING = "PROVISIONAL_LEANING"
CONTESTED_NO_LEANING = "CONTESTED_NO_LEANING"

CANONICAL_SPECIALIST_STATUSES = frozenset({
    SUPPORTS,
    CONDITIONAL_SUPPORTS,
    PROVISIONAL_LEANING,
    CONTESTED_NO_LEANING,
})

# Old names may parse; only new names may serialize.
_STATUS_PARSE_ALIASES = {
    "CONFLICTED_NO_LEANING": CONTESTED_NO_LEANING,
    "ADJUDICATED_SUPPORTS": SUPPORTS,
    "ADJUDICATION_INCOMPLETE": CONTESTED_NO_LEANING,
    # Opening / untyped candidates were historically treated as governing-ready.
    "NOT_APPLICABLE": SUPPORTS,
    "": SUPPORTS,
}

PROVISIONAL_LEANING_POLICY_FACTOR = 0.45
CONTESTED_NO_LEANING_POLICY_FACTOR = 0.0

# Investigative interrupt gate. Named and tested at the boundary.
REOPEN_PRIORITY_THRESHOLD = 0.70


def _is_normative_unresolved(unresolved: str) -> bool:
    return normalize_unresolved_marker(unresolved) == NORMATIVE_ADJUDICATION


def _is_decision_boundary_unresolved(unresolved: str) -> bool:
    return normalize_unresolved_marker(unresolved) == DECISION_BOUNDARY


# Deprecated broadcast_authority projection: attention without reopen privilege.
INVESTIGATIVE_ATTENTION_THRESHOLD = 0.40

# Canonical terminal judgment statuses — only these may serialize.
GOVERNED_RECOMMENDATION = "GOVERNED_RECOMMENDATION"
CONTESTED_RECOMMENDATION = "CONTESTED_RECOMMENDATION"
UNRESOLVED = "UNRESOLVED"

CANONICAL_JUDGMENT_STATUSES = frozenset({
    GOVERNED_RECOMMENDATION,
    CONTESTED_RECOMMENDATION,
    UNRESOLVED,
})

_JUDGMENT_PARSE_ALIASES = {
    "ACTION_RECOMMENDATION": GOVERNED_RECOMMENDATION,
    "INCONCLUSIVE": UNRESOLVED,
    "UNDERDETERMINED": UNRESOLVED,
    "CONDITIONAL": CONTESTED_RECOMMENDATION,
}

# Decision-criticality of an open condition for CONDITIONAL_SUPPORTS.
# Reversal potential matters more than a fixed discount.
_CONDITION_FACTORS = {
    "NO_CHANGE": 1.0,
    "WEAKENS": 0.70,
    "REVERSES": 0.45,
    "UNRESOLVED": 0.50,
}

_MIN_LEANING_PREFERENCE = 0.10

_CATEGORICAL_AUTHORITY = re.compile(
    r"\b(?:perfect\s+dut(?:y|ies)\s+(?:require|override|forbid|prohibit)|"
    r"(?:is|are)\s+required\b|"
    r"(?:is|are)\s+prohibited\b|"
    r"\bREQUIRED\b|\bPROHIBITED\b|"
    r"categorically\s+(?:required|forbidden|prohibited)|"
    r"non[- ]negotiable)\b",
    re.IGNORECASE,
)

_CONDITIONAL_MARKERS = re.compile(
    r"\b(?:provided\s+that|unless|only\s+if|if\s+and\s+only\s+if|"
    r"conditional(?:ly)?|depends?\s+on|so\s+long\s+as|assuming)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpecialistAuthorityProfile:
    adjudication_status: str
    broadcast_authority: str
    governing_eligible: bool
    policy_weight_factor: float
    investigative_claim: str
    condition_factor: float = 1.0
    investigative_priority: float = 0.0
    reopen_eligible: bool = False
    reopen_reason: str = ""
    reopen_question_key: str = ""


@dataclass(frozen=True)
class TerminalJudgment:
    status: str
    policy_direction: str
    governing_rule: str
    governing_justification_status: str
    governing_attack_reason: str = ""


def normalize_judgment_status(status: str | None) -> str:
    raw = str(status or "").strip().upper()
    if raw in CANONICAL_JUDGMENT_STATUSES:
        return raw
    return _JUDGMENT_PARSE_ALIASES.get(raw, UNRESOLVED)

def normalize_specialist_status(status: str | None) -> str:
    """Parse legacy or canonical status into a serializable canonical value."""
    raw = str(status or "").strip().upper()
    if raw in CANONICAL_SPECIALIST_STATUSES:
        return raw
    return _STATUS_PARSE_ALIASES.get(raw, SUPPORTS)


def condition_factor_from_effect(effect: str | None) -> float:
    key = str(effect or "UNRESOLVED").strip().upper()
    return _CONDITION_FACTORS.get(key, _CONDITION_FACTORS["UNRESOLVED"])


def _open_condition_text(candidate: Any) -> str:
    parts = [
        getattr(candidate, "baseline_condition", "") or "",
        getattr(candidate, "utilitarian_missing_comparison", "") or "",
        getattr(candidate, "factual_reversal_threshold", "") or "",
        getattr(candidate, "normative_reversal_threshold", "") or "",
        getattr(candidate, "reversal_condition", "") or "",
    ]
    return " ".join(" ".join(str(part).split()) for part in parts if part and part != "NONE")


def _infer_condition_effect(candidate: Any) -> str:
    """Map live candidate signals onto NO_CHANGE / WEAKENS / REVERSES / UNRESOLVED."""
    effect = str(getattr(candidate, "audit_internal_effect", "") or "").strip().upper()
    if effect in _CONDITION_FACTORS:
        if effect != "UNRESOLVED":
            return effect
        # Explicit UNRESOLVED from an audit still counts when a condition is open.
        return "UNRESOLVED"
    if bool(getattr(candidate, "utilitarian_decision_depends_on_unknown", False)):
        return "UNRESOLVED"
    condition = _open_condition_text(candidate)
    if not condition:
        return "NO_CHANGE"
    if re.search(r"\b(?:revers|flip|switch|invert|would\s+prefer)\b", condition, re.I):
        return "REVERSES"
    if re.search(r"\b(?:weaken|undercut|reduce|less\s+decisive)\b", condition, re.I):
        return "WEAKENS"
    return "UNRESOLVED"


def _has_normative_contestation(candidate: Any) -> bool:
    assumption = str(getattr(candidate, "assumption_status", "") or "").upper()
    baseline = str(getattr(candidate, "baseline_status", "") or "").upper()
    unresolved = str(getattr(candidate, "unresolved", "") or "").upper()
    conflicts = list(getattr(candidate, "framework_internal_conflicts", []) or [])
    if assumption == "NORMATIVELY_CONTESTED" or baseline == "NORMATIVELY_CONTESTED":
        return True
    if _is_normative_unresolved(unresolved):
        return True
    if conflicts:
        return True
    return False


def _is_conditional_state(candidate: Any) -> bool:
    assumption = str(getattr(candidate, "assumption_status", "") or "").upper()
    baseline = str(getattr(candidate, "baseline_status", "") or "").upper()
    if assumption == "CONDITIONAL" or baseline == "CONDITIONAL":
        return True
    if bool(getattr(candidate, "utilitarian_decision_depends_on_unknown", False)):
        return True
    if _open_condition_text(candidate) and assumption in {
        "CONDITIONAL", "UNDERDETERMINED",
    }:
        return True
    return False


def _has_directional_lean(candidate: Any) -> bool:
    preference = float(getattr(candidate, "preference_strength", 0.0) or 0.0)
    if preference >= _MIN_LEANING_PREFERENCE:
        return True
    recommended = str(getattr(candidate, "recommended_action", "") or "").strip()
    return bool(recommended)


def derive_specialist_status(candidate: Any) -> str:
    """Derive framework-general status from candidate signals.

    Prefer an already-classified canonical status when present (e.g. Kantian
    ledger output). Otherwise map assumption / baseline / conflict signals.
    """
    existing = str(getattr(candidate, "adjudication_status", "") or "").strip().upper()
    if existing in {
        SUPPORTS, CONDITIONAL_SUPPORTS, PROVISIONAL_LEANING, CONTESTED_NO_LEANING,
        "ADJUDICATED_SUPPORTS", "CONFLICTED_NO_LEANING", "ADJUDICATION_INCOMPLETE",
    }:
        # Kantian path already typed the claim; only normalize aliases.
        # Re-check conditional overlay when the ledger said SUPPORTS but the
        # candidate still carries an open decision-critical condition.
        normalized = normalize_specialist_status(existing)
        if normalized == SUPPORTS and _is_conditional_state(candidate):
            return CONDITIONAL_SUPPORTS
        if normalized == SUPPORTS and _has_normative_contestation(candidate):
            return (
                PROVISIONAL_LEANING
                if _has_directional_lean(candidate)
                else CONTESTED_NO_LEANING
            )
        return normalized

    if _has_normative_contestation(candidate):
        return (
            PROVISIONAL_LEANING
            if _has_directional_lean(candidate)
            else CONTESTED_NO_LEANING
        )
    if _is_conditional_state(candidate):
        return CONDITIONAL_SUPPORTS
    assumption = str(getattr(candidate, "assumption_status", "") or "").upper()
    if assumption == "UNDERDETERMINED":
        return (
            PROVISIONAL_LEANING
            if _has_directional_lean(candidate)
            else CONTESTED_NO_LEANING
        )
    return SUPPORTS


def condition_retained_in_rule(rule: str, condition: str = "") -> bool:
    text = " ".join(str(rule or "").split())
    if not text:
        return False
    if _CONDITIONAL_MARKERS.search(text):
        return True
    condition_text = " ".join(str(condition or "").split())
    if not condition_text or condition_text.upper() == "NONE":
        return False
    # Require at least one overlapping content word so a bare "prefer A0"
    # cannot pass merely because a condition exists elsewhere on the candidate.
    ignored = {"a", "an", "the", "to", "of", "and", "or", "if", "is", "be"}
    words = {
        word for word in re.findall(r"[a-z0-9]+", condition_text.casefold())
        if word not in ignored and len(word) > 2
    }
    rule_words = set(re.findall(r"[a-z0-9]+", text.casefold()))
    return bool(words and len(words & rule_words) >= min(2, len(words)))


def ensure_conditional_governing_rule(candidate: Any) -> str:
    """Return a governing rule that retains the open condition, or empty."""
    rule = " ".join(str(getattr(candidate, "decision_rule", "") or "").split())
    condition = _open_condition_text(candidate)
    if condition_retained_in_rule(rule, condition):
        return rule[:180]
    if not condition:
        return ""
    action = " ".join(
        str(getattr(candidate, "recommended_action", "") or "the preferred action").split()
    )
    specialist = str(getattr(candidate, "specialist", "the framework") or "the framework")
    return (
        f"{action} is {specialist}ly preferred provided that {condition}"
    )[:180]


def downgrade_claim_authority(text: str, status: str) -> str:
    """Compression may remove detail; it may never increase authority."""
    normalized = normalize_specialist_status(status)
    claim = " ".join(str(text or "").split())
    if not claim:
        return ""
    if normalized in {SUPPORTS, CONDITIONAL_SUPPORTS}:
        return claim[:240]
    # Strip categorical requirement/prohibition language from incomplete states.
    softened = _CATEGORICAL_AUTHORITY.sub("provisionally favored under incomplete adjudication", claim)
    if normalized == PROVISIONAL_LEANING and not softened.lower().startswith("provisional"):
        if not re.search(r"\bprovisionally\b|\bunresolved\b|\bleans?\b", softened, re.I):
            softened = f"Provisionally: {softened}"
    if normalized == CONTESTED_NO_LEANING and "unresolved" not in softened.casefold():
        softened = f"Unresolved contestation: {softened}"
    return softened[:240]


def policy_weight_for(
    status: str,
    *,
    epistemic_confidence: float = 1.0,
    condition_factor: float = 1.0,
    base_weight: float = 1.0,
) -> float:
    """Directional policy contribution by specialist state."""
    normalized = normalize_specialist_status(status)
    confidence = max(0.0, min(1.0, float(epistemic_confidence)))
    base = max(0.0, float(base_weight))
    if normalized == SUPPORTS:
        return max(0.0, min(1.0, base * confidence))
    if normalized == CONDITIONAL_SUPPORTS:
        factor = max(0.0, min(1.0, float(condition_factor)))
        return max(0.0, min(1.0, base * confidence * factor))
    if normalized == PROVISIONAL_LEANING:
        return max(0.0, min(1.0, base * PROVISIONAL_LEANING_POLICY_FACTOR))
    if normalized == CONTESTED_NO_LEANING:
        return CONTESTED_NO_LEANING_POLICY_FACTOR
    return 0.0


def governing_eligible_for(status: str, *, conditional_rule_retained: bool = False) -> bool:
    normalized = normalize_specialist_status(status)
    if normalized == SUPPORTS:
        return True
    if normalized == CONDITIONAL_SUPPORTS:
        return bool(conditional_rule_retained)
    return False


def classify_specialist_authority(candidate: Any) -> SpecialistAuthorityProfile:
    """Type policy, attention, and governing authority for any specialist."""
    status = derive_specialist_status(candidate)
    condition_effect = _infer_condition_effect(candidate)
    condition_factor = (
        condition_factor_from_effect(condition_effect)
        if status == CONDITIONAL_SUPPORTS
        else 1.0
    )
    confidence = float(
        getattr(candidate, "epistemic_confidence", None)
        if getattr(candidate, "epistemic_confidence", None) not in (None, -1)
        else getattr(candidate, "confidence", 1.0)
    )
    if confidence < 0:
        confidence = 1.0
    weight = policy_weight_for(
        status,
        epistemic_confidence=confidence,
        condition_factor=condition_factor,
    )
    conditional_rule = ""
    if status == CONDITIONAL_SUPPORTS:
        conditional_rule = ensure_conditional_governing_rule(candidate)
    eligible = governing_eligible_for(
        status,
        conditional_rule_retained=(
            True if status == SUPPORTS else bool(conditional_rule)
        ),
    )
    investigative = ""
    if status == PROVISIONAL_LEANING:
        action = " ".join(
            str(getattr(candidate, "recommended_action", "") or "").split()
        )
        conflict = "; ".join(
            str(item) for item in (
                getattr(candidate, "framework_internal_conflicts", []) or []
            )[:2]
        ) or str(getattr(candidate, "unresolved", "NORMATIVE_ADJUDICATION"))
        investigative = downgrade_claim_authority(
            f"UNRESOLVED FRAMEWORK CONFLICT: {conflict}. Current reasoning leans "
            f"{action or 'the present recommendation'}, but the competing internal "
            "claim has not been vindicated or defeated.",
            status,
        )
    elif status == CONTESTED_NO_LEANING:
        conflict = "; ".join(
            str(item) for item in (
                getattr(candidate, "framework_internal_conflicts", []) or []
            )[:2]
        ) or "live competing derivations"
        investigative = downgrade_claim_authority(
            f"UNRESOLVED FRAMEWORK CONFLICT: {conflict}. No comparative leaning "
            "is yet grounded.",
            status,
        )
    broadcast = (
        "INVESTIGATIVE"
        if status in {PROVISIONAL_LEANING, CONTESTED_NO_LEANING}
        else "GOVERNING_CANDIDATE"
    )
    return SpecialistAuthorityProfile(
        adjudication_status=status,
        broadcast_authority=broadcast,
        governing_eligible=eligible,
        policy_weight_factor=weight,
        investigative_claim=investigative,
        condition_factor=condition_factor,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_unresolvedness(candidate: Any) -> float:
    status = normalize_specialist_status(
        getattr(candidate, "adjudication_status", SUPPORTS)
    )
    unresolved = str(getattr(candidate, "unresolved", "NONE") or "NONE").upper()
    if status == CONTESTED_NO_LEANING:
        return 1.0
    if status == PROVISIONAL_LEANING:
        if _is_normative_unresolved(unresolved):
            return 0.95
        return 0.85
    if status == CONDITIONAL_SUPPORTS:
        return 0.55
    if _is_normative_unresolved(unresolved):
        return 0.85
    if _is_decision_boundary_unresolved(unresolved):
        return 0.70
    if unresolved not in {"", "NONE"}:
        return 0.55
    return 0.12


def score_reversal_potential(
    candidate: Any,
    *,
    plurality: str = "",
    policy: dict[str, float] | None = None,
) -> float:
    """Score the issue's ability to change the current result — not present vote weight."""
    recommended = str(getattr(candidate, "recommended_action", "") or "").strip()
    effect = str(getattr(candidate, "audit_internal_effect", "") or "").upper()
    unresolved = str(getattr(candidate, "unresolved", "") or "").upper()
    score = 0.20
    if plurality and recommended and recommended != plurality:
        score = max(score, 0.90)
    elif plurality and recommended == plurality:
        # A same-side provisional can still reverse if its open conflict is decided
        # against the plurality.
        status = normalize_specialist_status(
            getattr(candidate, "adjudication_status", SUPPORTS)
        )
        if _is_normative_unresolved(unresolved):
            score = max(score, 0.90)
        elif status in {PROVISIONAL_LEANING, CONTESTED_NO_LEANING, CONDITIONAL_SUPPORTS}:
            score = max(score, 0.75)
        else:
            score = max(score, 0.25)
    if effect in {"REVERSES", "REVERSES_FRAMEWORK_PREFERENCE"}:
        score = max(score, 0.95)
    elif effect == "WEAKENS":
        score = max(score, 0.65)
    elif effect == "UNRESOLVED" and _open_condition_text(candidate):
        score = max(score, 0.70)
    if _is_normative_unresolved(unresolved):
        score = max(score, 0.90)
    if _is_decision_boundary_unresolved(unresolved):
        score = max(score, 0.85)
    if policy and plurality and recommended and recommended != plurality:
        lead = float(policy.get(plurality, 0.0) or 0.0)
        rival = float(policy.get(recommended, 0.0) or 0.0)
        # Narrow leads are easier to overturn.
        if lead - rival < 0.20:
            score = max(score, 0.85)
    return _clamp01(score)


def score_grounding(candidate: Any, problem_state: dict[str, Any] | None = None) -> float:
    evidence = str(getattr(candidate, "evidence_basis", "") or "").upper()
    if evidence == "UNSTATED_FACTS":
        return 0.15
    targets = list(getattr(candidate, "tension_target_keys", []) or [])
    state = dict(problem_state or {})
    audit_candidates = list(state.get("audit_candidates", []) or [])
    grounded_keys = {
        str(item.get("question_key") or item.get("issue_id") or "")
        for item in audit_candidates
        if item.get("grounding_status") == "CLAUSE_GROUNDED"
        or list(item.get("grounded_in", []) or [])
    }
    if targets and any(key in grounded_keys for key in targets):
        return 0.95
    if evidence in {"STATED_FACTS", "SCENARIO_GROUNDED", "FRAMEWORK_AND_FACTS"}:
        return 0.80
    if list(getattr(candidate, "framework_internal_conflicts", []) or []):
        return 0.55
    if str(getattr(candidate, "unresolved", "NONE")).upper() not in {"", "NONE"}:
        return 0.40
    return 0.25


def score_severity(candidate: Any) -> float:
    unresolved = str(getattr(candidate, "unresolved", "") or "").upper()
    constraint = str(getattr(candidate, "constraint", "") or "").upper()
    conflicts = " ".join(
        str(item) for item in (getattr(candidate, "framework_internal_conflicts", []) or [])
    )
    claim = " ".join([
        str(getattr(candidate, "investigative_claim", "") or ""),
        conflicts,
        unresolved,
        constraint,
    ])
    score = 0.30
    if _is_normative_unresolved(unresolved):
        score = max(score, 0.85)
    if constraint in {"DUTY", "RIGHTS", "FAIRNESS"}:
        score = max(score, 0.75)
    if re.search(
        r"\b(?:strict\s+(?:duty|right)|perfect\s+dut|entrusted|basic\s+libert|"
        r"coerc|prohibit|required|rights?\s+violation)\b",
        claim,
        re.IGNORECASE,
    ):
        score = max(score, 0.90)
    if normalize_specialist_status(
        getattr(candidate, "adjudication_status", SUPPORTS)
    ) in {PROVISIONAL_LEANING, CONTESTED_NO_LEANING}:
        score = max(score, 0.70)
    return _clamp01(score)


def compute_investigative_priority(
    candidate: Any,
    *,
    plurality: str = "",
    policy: dict[str, float] | None = None,
    problem_state: dict[str, Any] | None = None,
) -> tuple[float, float, float, float, float]:
    """Return (priority, unresolvedness, reversal, grounding, severity)."""
    unresolvedness = score_unresolvedness(candidate)
    reversal = score_reversal_potential(
        candidate, plurality=plurality, policy=policy,
    )
    grounding = score_grounding(candidate, problem_state)
    severity = score_severity(candidate)
    priority = _clamp01(unresolvedness * reversal * grounding * severity)
    return priority, unresolvedness, reversal, grounding, severity


def reopen_question_key_for(candidate: Any) -> str:
    targets = [
        str(key).strip()
        for key in (getattr(candidate, "tension_target_keys", []) or [])
        if str(key).strip()
    ]
    for key in targets:
        if key.startswith("QUESTION:"):
            return key
    if targets:
        return targets[0]
    specialist = str(getattr(candidate, "specialist", "") or "specialist").strip()
    unresolved = str(getattr(candidate, "unresolved", "NONE") or "NONE").strip()
    conflict = " ".join(
        str(item) for item in (getattr(candidate, "framework_internal_conflicts", []) or [])[:1]
    )
    digest_source = f"{specialist}|{unresolved}|{conflict}".casefold()
    digest = abs(hash(digest_source)) % (16**10)
    return f"QUESTION:authority:{specialist}:{digest:010x}"


def evidence_fingerprint_for(candidate: Any, problem_state: dict[str, Any] | None = None) -> str:
    """Detect materially new grounded evidence for a previously fired reopen key."""
    parts = [
        str(getattr(candidate, "audit_internal_effect", "") or ""),
        str(getattr(candidate, "evidence_basis", "") or ""),
        " ".join(str(item) for item in (getattr(candidate, "framework_internal_conflicts", []) or [])),
        " ".join(str(item) for item in (getattr(candidate, "framework_specific_open_questions", []) or [])),
        str(getattr(candidate, "investigative_claim", "") or ""),
    ]
    state = dict(problem_state or {})
    for item in state.get("audit_candidates", []) or []:
        key = str(item.get("question_key") or item.get("issue_id") or "")
        if key and key in set(getattr(candidate, "tension_target_keys", []) or []):
            parts.append(str(item.get("grounding_status", "")))
            parts.append(",".join(str(v) for v in (item.get("grounded_in") or [])))
            parts.append(str(item.get("proposition") or item.get("question") or ""))
    return "|".join(" ".join(part.split()) for part in parts if part)


def is_claim_grounded(candidate: Any, problem_state: dict[str, Any] | None = None) -> bool:
    return score_grounding(candidate, problem_state) >= 0.50


def is_issue_novel(
    question_key: str,
    *,
    fired_keys: dict[str, str] | None,
    evidence_fingerprint: str,
    settled_keys: Sequence[str] | None = None,
) -> bool:
    settled = {str(key) for key in (settled_keys or []) if str(key)}
    if question_key in settled:
        return False
    prior = dict(fired_keys or {})
    if question_key not in prior:
        return True
    # Escape hatch: same key may reopen once more if premises materially changed.
    return bool(evidence_fingerprint) and prior.get(question_key) != evidence_fingerprint


def evaluate_reopen_eligibility(
    candidate: Any,
    *,
    investigative_priority: float,
    plurality: str = "",
    policy: dict[str, float] | None = None,
    problem_state: dict[str, Any] | None = None,
    fired_keys: dict[str, str] | None = None,
    settled_keys: Sequence[str] | None = None,
    threshold: float = REOPEN_PRIORITY_THRESHOLD,
) -> tuple[bool, str, str]:
    question_key = reopen_question_key_for(candidate)
    fingerprint = evidence_fingerprint_for(candidate, problem_state)
    grounded = is_claim_grounded(candidate, problem_state)
    novel = is_issue_novel(
        question_key,
        fired_keys=fired_keys,
        evidence_fingerprint=fingerprint,
        settled_keys=settled_keys,
    )
    eligible = bool(
        grounded
        and novel
        and float(investigative_priority) >= float(threshold)
    )
    reason = ""
    if eligible:
        claim = str(getattr(candidate, "investigative_claim", "") or "").strip()
        unresolved = str(getattr(candidate, "unresolved", "") or "").strip()
        reason = claim or (
            f"Reopen required: {unresolved or 'unresolved framework conflict'} "
            f"(priority={investigative_priority:.2f})"
        )
    return eligible, reason[:240], question_key


def derive_broadcast_authority(
    *,
    governing_eligible: bool,
    is_governing_focus: bool,
    reopen_eligible: bool,
    investigative_priority: float,
    attention_threshold: float = INVESTIGATIVE_ATTENTION_THRESHOLD,
) -> str:
    """Deprecated projection for one migration cycle — not authoritative state."""
    if is_governing_focus and governing_eligible:
        return "GOVERNING_CANDIDATE"
    if reopen_eligible or float(investigative_priority) >= float(attention_threshold):
        return "INVESTIGATIVE"
    return "NONE"


def apply_investigative_authority(
    candidate: Any,
    *,
    plurality: str = "",
    policy: dict[str, float] | None = None,
    problem_state: dict[str, Any] | None = None,
    fired_keys: dict[str, str] | None = None,
    settled_keys: Sequence[str] | None = None,
    is_governing_focus: bool = False,
) -> SpecialistAuthorityProfile:
    """Attach orthogonal investigative fields and refresh deprecated projection."""
    priority, *_parts = compute_investigative_priority(
        candidate,
        plurality=plurality,
        policy=policy,
        problem_state=problem_state,
    )
    reopen_eligible, reopen_reason, question_key = evaluate_reopen_eligibility(
        candidate,
        investigative_priority=priority,
        plurality=plurality,
        policy=policy,
        problem_state=problem_state,
        fired_keys=fired_keys,
        settled_keys=settled_keys,
    )
    governing_eligible = bool(getattr(candidate, "governing_eligible", False))
    broadcast = derive_broadcast_authority(
        governing_eligible=governing_eligible,
        is_governing_focus=is_governing_focus,
        reopen_eligible=reopen_eligible,
        investigative_priority=priority,
    )
    candidate.investigative_priority = priority
    candidate.reopen_eligible = reopen_eligible
    candidate.reopen_reason = reopen_reason
    candidate.reopen_question_key = question_key if reopen_eligible else question_key
    candidate.broadcast_authority = broadcast
    return SpecialistAuthorityProfile(
        adjudication_status=normalize_specialist_status(
            getattr(candidate, "adjudication_status", SUPPORTS)
        ),
        broadcast_authority=broadcast,
        governing_eligible=governing_eligible,
        policy_weight_factor=float(getattr(candidate, "policy_weight_factor", 1.0) or 0.0),
        investigative_claim=str(getattr(candidate, "investigative_claim", "") or ""),
        investigative_priority=priority,
        reopen_eligible=reopen_eligible,
        reopen_reason=reopen_reason,
        reopen_question_key=question_key,
    )


def governing_claim_under_attack(
    governing: Any | None,
    candidates: Sequence[Any],
) -> tuple[bool, str]:
    """True when a reopen-eligible issue attacks a necessary premise of the governing claim."""
    if governing is None:
        return False, ""
    governing_rule = " ".join(str(getattr(governing, "decision_rule", "") or "").split())
    governing_rationale = " ".join(str(getattr(governing, "rationale", "") or "").split())
    governing_text = f"{governing_rule} {governing_rationale}".casefold()
    attacks = []
    for candidate in candidates:
        if not getattr(candidate, "reopen_eligible", False):
            continue
        if candidate is governing:
            continue
        reason = " ".join(str(getattr(candidate, "reopen_reason", "") or "").split())
        claim = " ".join(str(getattr(candidate, "investigative_claim", "") or "").split())
        text = f"{reason} {claim}"
        # Direct attack: unresolved strict conflict about the recommended action
        # or lexical overlap with the governing justification.
        recommended = str(getattr(candidate, "recommended_action", "") or "")
        gov_action = str(getattr(governing, "recommended_action", "") or "")
        severity = score_severity(candidate)
        if severity >= 0.80 and (
            (recommended and gov_action and recommended != gov_action)
            or _is_normative_unresolved(str(getattr(candidate, "unresolved", "")))
        ):
            attacks.append(reason or claim or "high-severity unresolved conflict")
            continue
        words = {
            word for word in re.findall(r"[a-z]{4,}", text.casefold())
            if word not in {"that", "this", "with", "from", "have", "been"}
        }
        if words and governing_text and len(words & set(re.findall(r"[a-z]{4,}", governing_text))) >= 2:
            attacks.append(reason or claim)
    if not attacks:
        return False, ""
    return True, attacks[0][:240]


def classify_terminal_judgment(
    *,
    plurality: str,
    governing: Any | None,
    candidates: Sequence[Any],
    halted_by: str = "",
    has_stable_plurality: bool = True,
) -> TerminalJudgment:
    """Map policy plurality + governing eligibility onto canonical terminal status."""
    direction = " ".join(str(plurality or "").split())
    if not has_stable_plurality or not direction or direction in {
        "INCONCLUSIVE", "UNDERDETERMINED", "CONDITIONAL", "NONE",
    }:
        return TerminalJudgment(
            status=UNRESOLVED,
            policy_direction="",
            governing_rule="",
            governing_justification_status="NONE",
        )
    under_attack, attack_reason = governing_claim_under_attack(governing, candidates)
    if governing is not None and getattr(governing, "governing_eligible", False):
        rule = " ".join(str(getattr(governing, "decision_rule", "") or "").split())
        if under_attack:
            return TerminalJudgment(
                status=CONTESTED_RECOMMENDATION,
                policy_direction=direction,
                governing_rule=rule,
                governing_justification_status="UNDER_ATTACK",
                governing_attack_reason=attack_reason,
            )
        return TerminalJudgment(
            status=GOVERNED_RECOMMENDATION,
            policy_direction=direction,
            governing_rule=rule,
            governing_justification_status="ADMISSIBLE",
        )
    return TerminalJudgment(
        status=CONTESTED_RECOMMENDATION,
        policy_direction=direction,
        governing_rule="",
        governing_justification_status="NONE",
    )


def apply_specialist_authority(candidate: Any) -> SpecialistAuthorityProfile:
    """Write typed authority onto a candidate and enforce claim-status discipline."""
    profile = classify_specialist_authority(candidate)
    candidate.adjudication_status = profile.adjudication_status
    candidate.governing_eligible = profile.governing_eligible
    candidate.policy_weight_factor = profile.policy_weight_factor
    # Provisional projection until investigative fields are attached with plurality.
    candidate.broadcast_authority = profile.broadcast_authority
    if profile.investigative_claim and not str(
        getattr(candidate, "investigative_claim", "") or ""
    ).strip():
        candidate.investigative_claim = profile.investigative_claim
    if profile.adjudication_status == CONDITIONAL_SUPPORTS:
        retained = ensure_conditional_governing_rule(candidate)
        if retained:
            candidate.decision_rule = retained
        else:
            candidate.governing_eligible = False
    elif profile.adjudication_status in {PROVISIONAL_LEANING, CONTESTED_NO_LEANING}:
        candidate.decision_rule = downgrade_claim_authority(
            candidate.decision_rule, profile.adjudication_status,
        )
        candidate.rationale = downgrade_claim_authority(
            getattr(candidate, "rationale", ""), profile.adjudication_status,
        )
        candidate.governing_eligible = False
    return profile


def select_governing_claim(
    candidates: Sequence[Any],
    plurality: str,
    preferred: Any | None = None,
) -> Any | None:
    """Pick a justificatory rule source — never an investigative interrupt alone."""
    eligible = []
    for candidate in candidates:
        if not getattr(candidate, "schema_valid", True):
            continue
        status = normalize_specialist_status(
            getattr(candidate, "adjudication_status", SUPPORTS)
        )
        if status not in {SUPPORTS, CONDITIONAL_SUPPORTS}:
            continue
        if not getattr(candidate, "governing_eligible", False):
            continue
        if getattr(candidate, "recommended_action", "") != plurality:
            continue
        rule = str(getattr(candidate, "decision_rule", "") or "").strip()
        if not rule:
            continue
        if status == CONDITIONAL_SUPPORTS and not condition_retained_in_rule(
            rule, _open_condition_text(candidate),
        ):
            continue
        eligible.append(candidate)
    if not eligible:
        return None
    if preferred is not None and preferred in eligible:
        return preferred

    def sort_key(item: Any) -> tuple:
        status = normalize_specialist_status(
            getattr(item, "adjudication_status", SUPPORTS)
        )
        # Prefer fully resolved governing claims over conditional ones.
        resolved_rank = 1 if status == SUPPORTS else 0
        confidence = float(
            getattr(item, "epistemic_confidence", None)
            if getattr(item, "epistemic_confidence", None) not in (None, -1)
            else getattr(item, "confidence", 0.0) or 0.0
        )
        return (resolved_rank, confidence, str(getattr(item, "specialist", "")))

    return max(eligible, key=sort_key)


def claim_ref(candidate: Any | None) -> str:
    if candidate is None:
        return ""
    specialist = str(getattr(candidate, "specialist", "") or "").strip()
    constraint = str(getattr(candidate, "constraint", "") or "").strip()
    if specialist and constraint:
        return f"{specialist}:{constraint}"
    return specialist or constraint
