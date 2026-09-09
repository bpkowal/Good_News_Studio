"""Admission control for framework votes derived from an admitted world.

Specialist prose may propose a ranking, but only the framework's operative,
graph-committed ledger may authorize directional policy influence. This module
checks typed state already produced by framework transactions; it parses no new
scenario facts.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence


EXPECTED_LEDGER_KINDS = {
    "utilitarian": "UTILITARIAN_CONSEQUENCE_LEDGER",
    "deontological": "DEONTOLOGICAL_DUTY_LEDGER",
    "virtue": "VIRTUE_CHARACTER_LEDGER",
    "care": "CARE_RELATIONSHIP_LEDGER",
    "rawlsian": "RAWLSIAN_POSITION_LEDGER",
}


_MORTALITY_CLAIM = re.compile(
    r"\b(?:dead|death|deaths|die|dies|died|fatal|fatality|fatalities|"
    r"kill|kills|killed|mortality)\b",
    re.IGNORECASE,
)
_TRAPPING_OUTCOME = re.compile(r"\b(?:trap|trapped|trapping)\b", re.IGNORECASE)
_EXACT_CERTAINTY_CLAIM = re.compile(
    r"(?:\b(?:probability|chance|likelihood)\s*(?:is|=|of)\s*"
    r"(?:100\s*%(?!\w)|1(?:\.0+)?\b)|"
    r"\b(?:treat\w*|set|round\w*|convert\w*|assum\w*)\b.{0,32}"
    r"(?:100\s*%(?!\w)|probability\s*(?:of|=|is)\s*1(?:\.0+)?\b)|"
    r"\bexact(?:ly)?\s+certain(?:ty)?\b)",
    re.IGNORECASE,
)


def _record_effect_ids(record: dict[str, Any]) -> set[str]:
    values = {
        str(value) for value in record.get("grounded_effect_ids", []) or []
        if str(value)
    }
    singular = str(record.get("grounded_effect_id", "") or "")
    if singular:
        values.add(singular)
    return values


def _record_supports_exact_certainty(record: dict[str, Any]) -> bool:
    probability = str(record.get("probability", "") or "").strip().upper()
    if probability in {"1", "1.0", "100", "100%", "CERTAIN"}:
        return True
    return str(record.get("modality", "") or "").strip().upper() == "CERTAIN"


@dataclass(frozen=True, slots=True)
class FrameworkVoteDecision:
    status: str
    weight_ceiling: float
    governing_eligible: bool
    reason: str
    ledger_kind: str = ""
    ledger_status: str = ""


def _decision(
    status: str,
    reason: str,
    *,
    ledger_kind: str = "",
    ledger_status: str = "",
) -> FrameworkVoteDecision:
    return FrameworkVoteDecision(
        status=status,
        weight_ceiling={
            "FULL": 1.0,
            "ATTENUATED": 0.5,
            "ABSTAIN": 0.0,
            "NOT_APPLICABLE": 1.0,
        }[status],
        governing_eligible=status in {"FULL", "NOT_APPLICABLE"},
        reason=reason,
        ledger_kind=ledger_kind,
        ledger_status=ledger_status,
    )


def _derived_claim_errors(candidate: Any, records: list[dict[str, Any]]) -> list[str]:
    """Validate declared transformations without reparsing specialist prose."""
    claims = [
        dict(item)
        for item in getattr(candidate, "material_empirical_claims", []) or []
        if isinstance(item, dict)
    ]
    derived = [
        item for item in claims
        if (
            str(item.get("declared_basis", item.get("proposition_id", ""))).upper()
            == "FRAMEWORK_DERIVED"
            or str(item.get("derivation_operation", "DIRECT_COPY")).upper()
            != "DIRECT_COPY"
            or str(item.get("outcome_type_transformation", "PRESERVED")).upper()
            != "PRESERVED"
        )
    ]
    if not derived:
        return ["framework ranking exposes no derivation from admitted effects"]
    grounded_by_action: dict[str, set[str]] = {}
    for record in records:
        action_id = str(record.get("canonical_action_id", ""))
        grounded_by_action.setdefault(action_id, set()).update(
            str(value) for value in record.get("grounded_effect_ids", []) or []
            if str(value)
        )
    known_effects = (
        set().union(*grounded_by_action.values()) if grounded_by_action else set()
    )
    effect_actions = {
        effect_id: action_id
        for action_id, effect_ids in grounded_by_action.items()
        for effect_id in effect_ids
    }
    records_by_effect: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for effect_id in _record_effect_ids(record):
            records_by_effect.setdefault(effect_id, []).append(record)
    errors: list[str] = []
    for index, claim in enumerate(derived):
        label = f"derived claim {index + 1}"
        source_ids = {
            str(value) for value in claim.get("source_effect_ids", []) or []
            if str(value)
        }
        operation = str(claim.get("derivation_operation", "")).upper()
        calculation = " ".join(str(claim.get("calculation", "")).split())
        outcome_transform = str(
            claim.get("outcome_type_transformation", "")
        ).upper()
        scope = str(claim.get("scope_action_id", "")).upper()
        if not source_ids:
            errors.append(f"{label} omits source effect IDs")
        elif known_effects and not source_ids.issubset(known_effects):
            errors.append(f"{label} cites effects outside its committed framework ledger")
        if operation in {"", "DIRECT_COPY"}:
            errors.append(f"{label} omits a framework derivation operation")
        if len(calculation.split()) < 2:
            errors.append(f"{label} omits its calculation or inference")
        if "assumptions" not in claim or not isinstance(claim.get("assumptions"), list):
            errors.append(f"{label} omits its explicit assumptions list")
        if outcome_transform not in {"PRESERVED", "NORMATIVE_CLASSIFICATION"}:
            errors.append(
                f"{label} changes admitted outcome type to "
                f"{outcome_transform or 'UNKNOWN'}"
            )
        if scope == "COMPARISON" and operation not in {
            "QUALITATIVE_COMPARISON", "ARITHMETIC"
        }:
            errors.append(f"{label} has an invalid cross-action operation")
        source_actions = {
            effect_actions[value] for value in source_ids if value in effect_actions
        }
        source_records = [
            record
            for effect_id in source_ids
            for record in records_by_effect.get(effect_id, [])
        ]
        claim_text = " ".join((
            str(claim.get("claim", "") or ""), calculation,
        ))
        source_outcomes = " ".join(
            str(record.get("outcome", "") or "") for record in source_records
        ).replace("_", " ")
        if (
            source_records
            and _MORTALITY_CLAIM.search(claim_text)
            and _TRAPPING_OUTCOME.search(source_outcomes)
            and not _MORTALITY_CLAIM.search(source_outcomes)
        ):
            errors.append(
                f"{label} converts a trapping outcome into mortality"
            )
        if (
            source_records
            and _EXACT_CERTAINTY_CLAIM.search(claim_text)
            and not all(
                _record_supports_exact_certainty(record)
                for record in source_records
            )
        ):
            errors.append(
                f"{label} converts hedged or conditional likelihood to exact certainty"
            )
        if scope.startswith("A") and source_actions - {scope}:
            errors.append(f"{label} attaches a cross-action effect to {scope}")
        if scope.startswith("A") and operation in {
            "QUALITATIVE_COMPARISON", "ARITHMETIC"
        }:
            errors.append(f"{label} performs a comparison under one action's scope")
        if scope == "COMPARISON" and known_effects and len(source_actions) < 2:
            errors.append(f"{label} comparison does not expose effects from both actions")
    return list(dict.fromkeys(errors))[:12]


def evaluate_framework_vote(
    candidate: Any,
    actions: Sequence[str],
) -> FrameworkVoteDecision:
    """Classify whether a candidate's directional score may enter policy."""
    if not bool(getattr(candidate, "framework_vote_integrity_required", False)):
        return _decision("NOT_APPLICABLE", "framework vote gate was not requested")

    specialist = str(getattr(candidate, "specialist", "")).casefold()
    expected_kind = EXPECTED_LEDGER_KINDS.get(specialist)
    if expected_kind is None:
        return _decision("NOT_APPLICABLE", "candidate is not a ledger-backed framework")
    if not bool(getattr(candidate, "schema_valid", True)):
        return _decision("ABSTAIN", "specialist response is not schema-valid")

    native = dict(getattr(candidate, "committed_native_ledger", {}) or {})
    ledger_kind = str(native.get("ledger_kind", ""))
    ledger_status = str(native.get("transaction_status", "")).upper()
    records = [
        dict(item) for item in native.get("records", []) or []
        if isinstance(item, dict)
    ]
    context = {"ledger_kind": ledger_kind, "ledger_status": ledger_status}
    if ledger_kind != expected_kind:
        return _decision(
            "ABSTAIN",
            f"expected {expected_kind}, but no matching committed ledger is operative",
            **context,
        )
    if not ledger_status.startswith("COMMITTED") or not records:
        return _decision("ABSTAIN", "framework ledger is absent or rejected", **context)
    if any(
        str(item.get("specialist", specialist)).casefold() != specialist
        for item in records
    ):
        return _decision(
            "ABSTAIN", "committed ledger contains cross-specialist records", **context,
        )

    recommended = str(getattr(candidate, "recommended_action", "") or "")
    if recommended not in actions:
        return _decision("ABSTAIN", "framework did not select a live action", **context)
    recommended_id = f"A{list(actions).index(recommended)}"
    by_action: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        by_action.setdefault(str(item.get("canonical_action_id", "")), []).append(item)
    live_ids = {f"A{index}" for index in range(len(actions))}
    if not live_ids.issubset(by_action):
        return _decision(
            "ABSTAIN", "committed ledger does not compare every live action", **context,
        )
    if not bool(getattr(candidate, "comparison_complete", True)):
        return _decision(
            "ABSTAIN", "framework explicitly reports an incomplete comparison", **context,
        )

    derivation_errors = _derived_claim_errors(candidate, records)
    candidate.derived_claim_validation_status = (
        "QUARANTINED" if derivation_errors else "PASSED"
    )
    candidate.derived_claim_validation_errors = derivation_errors
    if derivation_errors:
        return _decision(
            "ABSTAIN",
            "derived claim validation failed: " + "; ".join(derivation_errors[:2]),
            **context,
        )

    if specialist == "utilitarian":
        if bool(getattr(candidate, "utilitarian_decision_depends_on_unknown", False)):
            return _decision(
                "ABSTAIN", "utilitarian ranking depends on an unresolved comparison", **context,
            )
        if not bool(getattr(candidate, "evidence_sufficient_for_action", True)):
            return _decision(
                "ABSTAIN", "utilitarian evidence is insufficient to rank actions", **context,
            )

    elif specialist == "deontological":
        chosen = by_action[recommended_id][-1]
        chosen_verdict = str(chosen.get("verdict", "")).upper()
        chosen_duty = str(chosen.get("duty_type", "")).upper()
        if chosen_verdict in {"", "UNCERTAIN", "CONFLICTED", "PROHIBITED", "REJECTED"}:
            return _decision(
                "ABSTAIN",
                f"selected action has non-ranking duty verdict {chosen_verdict or 'UNKNOWN'}",
                **context,
            )
        if chosen_duty in {"", "UNKNOWN", "UNRESOLVED"}:
            return _decision(
                "ABSTAIN", "selected action lacks a classified duty type", **context,
            )
        rivals = [
            rows[-1]
            for action_id, rows in by_action.items()
            if action_id in live_ids and action_id != recommended_id
        ]
        if any(str(row.get("verdict", "")).upper() == "REQUIRED" for row in rivals):
            return _decision(
                "ABSTAIN", "a rival action retains an unresolved required duty", **context,
            )
        if (
            chosen_verdict == "PERMISSIBLE"
            and not any(
                str(row.get("verdict", "")).upper() == "PROHIBITED"
                for row in rivals
            )
        ):
            return _decision(
                "ABSTAIN",
                "the duty ledger permits the selected action but does not rank it above its rivals",
                **context,
            )

    elif specialist == "rawlsian":
        bases = {str(item.get("ranking_basis", "")).upper() for item in records}
        if not bases or bases & {"", "UNKNOWN", "UNRESOLVED"}:
            return _decision("ABSTAIN", "Rawlsian ranking basis remains unresolved", **context)
        if any(
            str(item.get("dimension", "")).upper() in {"", "UNKNOWN", "UNRESOLVED"}
            or str(item.get("institutional_relation", "")).upper()
            in {"", "UNKNOWN", "UNRESOLVED"}
            for item in records
        ):
            return _decision(
                "ABSTAIN", "Rawlsian positions lack a typed distributive relation", **context,
            )
        if (
            str(getattr(candidate, "framework_numerical_role", "")).upper()
            == "DECISIVE"
            and not bases.issubset({"MAXIMIN_PRIMARY_GOODS", "DIFFERENCE_PRINCIPLE"})
        ):
            return _decision(
                "ABSTAIN",
                "decisive aggregation is not licensed by the committed Rawlsian ranking basis",
                **context,
            )

    elif specialist == "virtue":
        if any(
            str(item.get("ranking_basis", "")).upper() in {"", "UNKNOWN", "UNRESOLVED"}
            for item in records
        ):
            return _decision(
                "ABSTAIN", "virtue ranking lacks a practical-wisdom basis", **context,
            )
        chosen_verdict = str(by_action[recommended_id][-1].get("verdict", "")).upper()
        if chosen_verdict in {"", "UNCERTAIN"}:
            return _decision(
                "ABSTAIN", "virtue assessment does not rank the selected action", **context,
            )

    elif specialist == "care":
        if any(
            str(item.get("ranking_basis", "")).upper() in {"", "UNKNOWN", "UNRESOLVED"}
            for item in records
        ):
            return _decision(
                "ABSTAIN", "Care ranking lacks a relational priority basis", **context,
            )
        chosen_verdict = str(by_action[recommended_id][-1].get("verdict", "")).upper()
        if chosen_verdict in {"", "UNCERTAIN"}:
            return _decision(
                "ABSTAIN", "Care assessment does not rank the selected action", **context,
            )

    uncertain = ledger_status == "COMMITTED_WITH_UNCERTAINTY" or bool(
        getattr(candidate, "framework_grounding_penalty", 0.0)
    )
    if uncertain:
        return _decision(
            "ATTENUATED",
            "framework ledger committed with unresolved validation residue",
            **context,
        )
    return _decision(
        "FULL", "framework ranking is supported by its committed typed ledger", **context,
    )


def apply_framework_vote_integrity(
    candidate: Any,
    actions: Sequence[str],
) -> FrameworkVoteDecision:
    """Apply the gate after ordinary authority typing and before aggregation."""
    decision = evaluate_framework_vote(candidate, actions)
    candidate.framework_vote_status = decision.status
    candidate.framework_vote_reason = decision.reason
    candidate.framework_ledger_kind = decision.ledger_kind
    candidate.framework_ledger_status = decision.ledger_status
    candidate.policy_weight_factor = min(
        float(getattr(candidate, "policy_weight_factor", 1.0) or 0.0),
        decision.weight_ceiling,
    )
    if not decision.governing_eligible:
        candidate.governing_eligible = False
        if str(getattr(candidate, "broadcast_authority", "")).upper() == "GOVERNING_CANDIDATE":
            candidate.broadcast_authority = "INVESTIGATIVE"
    return decision
