"""Framework-native, non-evidentiary summaries for shared workspace transport.

The originating ledger remains authoritative. These adapters only expose a
bounded reasoning kernel from the operative committed state so other frameworks
can see its native structure without treating it as fact or importing its norm.
"""

from __future__ import annotations

from typing import Any


def _text(value: Any, limit: int = 100) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip(" ,;:-") + "…"


def _base(
    schema_kind: str, candidate: Any, ledger: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_kind": schema_kind,
        "schema_version": 1,
        "source_type": "GRAPH_COMMITTED_FRAMEWORK_LEDGER",
        "source_specialist": str(candidate.specialist),
        "ledger_kind": str(ledger.get("ledger_kind", "")),
        "admission_status": str(ledger.get("transaction_status", "COMMITTED")),
    }


def _utilitarian(candidate: Any, state: dict[str, Any]) -> dict[str, Any]:
    records = list(state.get("records", []) or [])
    if not records:
        return {}
    action_ids = sorted({str(row.get("canonical_action_id", "")) for row in records})
    actions = [{
        "action_id": action_id,
        "effects": [{
            "effect_id": str(row.get(
                "world_effect_id", row.get("grounded_effect_id", "")
            )),
            "outcome": _text(row.get("outcome", ""), 90),
            # Direction remains graph-derived world polarity.  The adapter
            # exposes it; it never asks the utilitarian specialist to recreate it.
            "direction": str(row.get("direction", "UNKNOWN")),
            "polarity": str(row.get("polarity", "UNKNOWN")),
            "probability": str(row.get("probability", "UNKNOWN")),
            "modality": str(row.get("modality", "UNKNOWN")),
            "magnitude": str(row.get("magnitude", "UNKNOWN")),
            "scope": _text(row.get("scope", ""), 60),
            "importance": str(row.get("importance", "UNKNOWN")),
            "valuation_reason": _text(row.get("valuation_reason", ""), 80),
            "epistemic_status": str(row.get("epistemic_status", "UNKNOWN")),
        } for row in records if str(row.get("canonical_action_id", "")) == action_id],
    } for action_id in action_ids]
    return {
        **_base("UTILITARIAN_EFFECT_COMPARISON", candidate, state),
        "actions": actions,
    }


def _deontological(candidate: Any, state: dict[str, Any]) -> dict[str, Any]:
    records = list(state.get("records", []) or [])
    if not records:
        return {}
    assessments = []
    for item in records:
        assessment = {
            "action_id": str(item.get("canonical_action_id", "")),
            "verdict": str(item.get("verdict", "")),
            "norm_kind": str(item.get("norm_kind", "UNKNOWN")),
            "norm": _text(item.get("norm", ""), 90),
            "relation": str(item.get("relation", "UNCERTAIN")),
            "duty_type": str(item.get("duty_type", "UNRESOLVED")),
            "duty_bearer": _text(item.get("duty_bearer", ""), 60),
            "protected_party": _text(item.get("protected_party", ""), 70),
            "harm_relation": str(item.get("harm_relation", "UNRESOLVED")),
            "special_obligation_status": str(
                item.get("special_obligation_status", "UNKNOWN")
            ),
            "special_obligation_basis": _text(
                item.get("special_obligation_basis", ""), 90
            ),
            "means_relation": str(item.get("means_relation", "UNRESOLVED")),
            "competing_norm": _text(item.get("competing_norm", ""), 90),
            "competing_relation": str(item.get("competing_relation", "UNCERTAIN")),
            "competing_protected_party": _text(
                item.get("competing_protected_party", ""), 70
            ),
            "priority_basis": str(item.get("priority_basis", "UNRESOLVED")),
            "priority_rule": _text(item.get("priority_rule", ""), 100),
            "derivation": str(item.get("derivation", "UNRESOLVED")),
            "universalization_status": (
                str(item.get("relation", "UNCERTAIN"))
                if str(item.get("derivation", "UNRESOLVED")) == "UNIVERSAL_LAW"
                or str(item.get("norm_kind", "UNKNOWN")) == "UNIVERSAL_LAW"
                else "NOT_APPLICABLE"
            ),
            "resolution_status": str(item.get("resolution_status", "UNKNOWN")),
            "calibration_errors": [
                _text(error, 90) for error in item.get("calibration_errors", []) or []
            ][:3],
        }
        coercion_kind = str(item.get("coercion_kind", "UNKNOWN"))
        if coercion_kind not in {"", "NONE", "UNKNOWN"}:
            assessment["coercion"] = {
                "kind": coercion_kind,
                "actor": _text(item.get("coercive_actor", ""), 60),
                "party": _text(item.get("coerced_party", ""), 60),
                "authorization": str(item.get("authorization_status", "UNKNOWN")),
                "public_justification": _text(
                    item.get("public_justification", ""), 90
                ),
            }
        assessments.append(assessment)
    return {
        **_base("DEONTOLOGICAL_DUTY_ANALYSIS", candidate, state),
        "assessments": assessments,
    }


def _virtue(candidate: Any, state: dict[str, Any]) -> dict[str, Any]:
    records = list(state.get("records", []) or [])
    if not records:
        return {}
    return {
        **_base("VIRTUE_PHRONESIS_ANALYSIS", candidate, state),
        "ranking_basis": str(records[0].get("ranking_basis", "UNRESOLVED")),
        "assessments": [{
            "action_id": str(item.get("canonical_action_id", "")),
            "verdict": str(item.get("verdict", "UNCERTAIN")),
            "actor_role": _text(item.get("actor_role", ""), 70),
            "virtues": _text(item.get("virtues", ""), 90),
            "vice_risk": _text(item.get("vice_risk", ""), 90),
            "salient_circumstance": _text(item.get("circumstance", ""), 100),
            "practical_judgment": _text(item.get("reason", ""), 100),
        } for item in records],
    }


def _care(candidate: Any, state: dict[str, Any]) -> dict[str, Any]:
    records = list(state.get("records", []) or [])
    if not records:
        return {}
    return {
        **_base("CARE_RELATIONAL_ANALYSIS", candidate, state),
        "ranking_basis": str(records[0].get("ranking_basis", "UNRESOLVED")),
        "assessments": [{
            "action_id": str(item.get("canonical_action_id", "")),
            "verdict": str(item.get("verdict", "UNCERTAIN")),
            "affected_party": _text(item.get("affected_party", ""), 70),
            "relationship_type": str(item.get("relationship_type", "UNRESOLVED")),
            "dependency_source": _text(item.get("dependency_source", ""), 90),
            "responsibility_basis": _text(item.get("responsibility_basis", ""), 100),
            "need_kind": str(item.get("need_kind", "UNRESOLVED")),
            "need_urgency": str(item.get("need_urgency", "UNKNOWN")),
            "trust_effect": str(item.get("trust_effect", "UNKNOWN")),
            "responsiveness": str(item.get("responsiveness", "UNKNOWN")),
            "feasibility": str(item.get("feasibility", "UNKNOWN")),
            "competing_care_claim": _text(item.get("competing_care_claim", ""), 100),
            "resolution_status": str(item.get("resolution_status", "UNKNOWN")),
        } for item in records],
    }


def _rawlsian(candidate: Any, state: dict[str, Any]) -> dict[str, Any]:
    records = list(state.get("records", []) or [])
    if not records:
        return {}
    return {
        **_base("RAWLSIAN_POSITION_ANALYSIS", candidate, state),
        "ranking_basis": str(records[0].get("ranking_basis", "UNRESOLVED")),
        "classification_justification": _text(
            records[0].get("ranking_classification_justification", ""), 120
        ),
        "lexical_priority_justification": _text(
            records[0].get("lexical_priority_justification", ""), 120
        ),
        "liberty_status": {
            str(item.get("canonical_action_id", "")): str(
                item.get("liberty_status", "UNKNOWN")
            ) for item in records
        },
        "positions": [{
            "action_id": str(item.get("canonical_action_id", "")),
            "representative_subject": _text(item.get("subject", ""), 70),
            "subject_kind": str(item.get("subject_kind", "UNKNOWN")),
            "dimension": str(item.get("dimension", "UNKNOWN")),
            "additional_dimensions": list(item.get("additional_dimensions", []) or [])[:5],
            "basic_liberty_kind": str(item.get("basic_liberty_kind", "UNRESOLVED")),
            "institutional_relation": str(item.get("institutional_relation", "UNRESOLVED")),
            "comparative_effect": str(item.get("effect", "UNCERTAIN")),
            "compared_to_action_id": str(item.get("compared_to_action_id", "")),
            "principle_basis": _text(item.get("principle_basis", ""), 80),
            "public_reason": _text(item.get("reason", ""), 100),
        } for item in records],
    }


_ADAPTERS = {
    "utilitarian": _utilitarian,
    "deontological": _deontological,
    "virtue": _virtue,
    "care": _care,
    "rawlsian": _rawlsian,
}
_LEDGER_KINDS = {
    "utilitarian": "UTILITARIAN_CONSEQUENCE_LEDGER",
    "deontological": "DEONTOLOGICAL_DUTY_LEDGER",
    "virtue": "VIRTUE_CHARACTER_LEDGER",
    "care": "CARE_RELATIONSHIP_LEDGER",
    "rawlsian": "RAWLSIAN_POSITION_LEDGER",
}


def committed_native_reasoning(candidate: Any) -> dict[str, Any]:
    """Return a tagged native kernel only from operative committed state."""
    state = dict(getattr(candidate, "committed_native_ledger", {}) or {})
    specialist = str(getattr(candidate, "specialist", "")).casefold()
    adapter = _ADAPTERS.get(specialist)
    if not state or adapter is None:
        return {}
    if str(state.get("ledger_kind", "")) != _LEDGER_KINDS[specialist]:
        return {}
    records = list(state.get("records", []) or [])
    if any(
        str(record.get("specialist", specialist)).casefold() != specialist
        for record in records if isinstance(record, dict)
    ):
        return {}
    return adapter(candidate, state)
