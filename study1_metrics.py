"""Deterministic response-state extraction for Study 1.

These metrics deliberately do not judge whether a response is good, correct, or
usable.  An unresolved, abstaining, or malformed final judgment is still a
measurable response state for perturbation-sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ResponseVector:
    """Comparable state emitted by a system for one scenario variant."""

    choice: str
    judgment_status: str
    confidence: float
    framework_state: tuple[tuple[str, str, str], ...]
    constraints: tuple[str, ...]
    propositions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _canonical_choice(value: Any, actions: Sequence[str]) -> str:
    text = _clean(value)
    if not text:
        return "UNRESOLVED"
    for index, action in enumerate(actions):
        if text == _clean(action):
            return f"A{index}"
    return text.casefold()


def _confidence(result: Mapping[str, Any]) -> float:
    raw = result.get("epistemic_confidence", result.get("confidence", 0.0))
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def response_vector_from_parliament(
    result: Mapping[str, Any], actions: Sequence[str],
) -> ResponseVector:
    """Extract state from a Parliament trace, including unresolved outcomes."""
    raw_choice = _canonical_choice(result.get("selected_action"), actions)
    if raw_choice in {"unresolved", "inconclusive", "underdetermined", "conditional"}:
        raw_choice = _canonical_choice(result.get("current_plurality"), actions)
    choice = raw_choice
    status = _clean(result.get("judgment_status")).upper() or "UNRESOLVED"
    candidates: list[Mapping[str, Any]] = []
    for cycle in result.get("cycles", []) or []:
        if cycle.get("is_hypothetical"):
            continue
        candidates.extend(
            item for item in cycle.get("candidates", []) or []
            if isinstance(item, Mapping) and item.get("schema_valid")
        )
    latest: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        name = _clean(candidate.get("specialist")).lower()
        if name:
            latest[name] = candidate
    framework_state = tuple(sorted((
        name,
        _canonical_choice(candidate.get("recommended_action"), actions),
        _clean(candidate.get("framework_vote_status") or "NOT_APPLICABLE").upper(),
    ) for name, candidate in latest.items()))
    constraints = tuple(sorted({
        _clean(candidate.get("constraint")).upper()
        for candidate in latest.values()
        if _clean(candidate.get("constraint"))
        and _clean(candidate.get("constraint")).upper() != "NONE"
    }))
    proposition_ids: set[str] = set()
    for candidate in latest.values():
        for key in ("supporting_proposition_ids", "supporting_propositions"):
            values = candidate.get(key) or []
            if isinstance(values, (list, tuple)):
                proposition_ids.update(
                    _clean(value) for value in values if _clean(value)
                )
        for claim in candidate.get("material_empirical_claims", []) or []:
            if not isinstance(claim, Mapping):
                continue
            values = claim.get("source_proposition_ids") or claim.get("source_effect_ids") or []
            if isinstance(values, (list, tuple)):
                proposition_ids.update(
                    _clean(value) for value in values if _clean(value)
                )
    return ResponseVector(
        choice=choice,
        judgment_status=status,
        confidence=_confidence(result),
        framework_state=framework_state,
        constraints=constraints,
        propositions=tuple(sorted(proposition_ids)),
    )


def response_vector_from_solo(
    result: Mapping[str, Any] | str, actions: Sequence[str],
) -> ResponseVector:
    """Extract the same state from a structured solo response.

    Plain-text legacy outputs are accepted as a fallback, but expose only an
    action mention and an explicit ``PLAIN_TEXT`` status; they do not receive
    inferred confidence or framework fields.
    """
    payload: Mapping[str, Any]
    if isinstance(result, Mapping):
        payload = result
    else:
        text = _clean(result)
        lowered = text.casefold()
        mentions = [
            index for index, action in enumerate(actions)
            if _clean(action).casefold() in lowered
        ]
        choice = f"A{mentions[-1]}" if len(mentions) == 1 else "UNRESOLVED"
        return ResponseVector(choice, "PLAIN_TEXT", 0.0, (), (), ())
    raw_choice = payload.get("choice", payload.get("selected_action"))
    if isinstance(raw_choice, int) and 0 <= raw_choice < len(actions):
        choice = f"A{raw_choice}"
    else:
        choice = _canonical_choice(raw_choice, actions)
    framework_rows = payload.get("framework_state") or payload.get("frameworks") or []
    framework_state = tuple(sorted(
        (
            _clean(item.get("name") or item.get("framework")).lower(),
            _canonical_choice(item.get("choice") or item.get("recommendation"), actions),
            _clean(item.get("status") or "NOT_APPLICABLE").upper(),
        )
        for item in framework_rows
        if isinstance(item, Mapping) and _clean(item.get("name") or item.get("framework"))
    ))
    constraints = tuple(sorted({
        _clean(value).upper() for value in payload.get("constraints", []) or []
        if _clean(value)
    }))
    propositions = tuple(sorted({
        _clean(value) for value in payload.get("propositions", []) or []
        if _clean(value)
    }))
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return ResponseVector(
        choice=choice,
        judgment_status=_clean(payload.get("status") or "STRUCTURED").upper(),
        confidence=confidence,
        framework_state=framework_state,
        constraints=constraints,
        propositions=propositions,
    )


def response_distance(left: ResponseVector, right: ResponseVector) -> float:
    """Simple equal-weight distance; component distances remain inspectable."""
    choice = float(left.choice != right.choice)
    status = float(left.judgment_status != right.judgment_status)
    confidence = abs(left.confidence - right.confidence)
    framework = 1.0 - _mapping_similarity(left.framework_state, right.framework_state)
    constraints = _set_distance(left.constraints, right.constraints)
    propositions = _set_distance(left.propositions, right.propositions)
    return (choice + status + confidence + framework + constraints + propositions) / 6.0


def _mapping_similarity(
    left: Sequence[tuple[str, str, str]], right: Sequence[tuple[str, str, str]],
) -> float:
    a = {key: value for key, *value in left}
    b = {key: value for key, *value in right}
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    return sum(a.get(key) == b.get(key) for key in keys) / len(keys)


def _set_distance(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


__all__ = (
    "ResponseVector",
    "response_distance",
    "response_vector_from_parliament",
)
