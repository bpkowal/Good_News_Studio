"""System validation for specialist-authored expected-value calculations.

An EV row is not authoritative merely because a specialist labels it grounded.
This module preserves the specialist's calculation while checking that its
operands point to admitted, action-local effects.  Exact numeric authority is
granted only when the submitted result can be recomputed from those effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from .world_state import quantity_magnitude


EV_NOT_CLAIMED = "NOT_CLAIMED"
EV_SOURCE_CORRESPONDENCE = "SOURCE_CORRESPONDENCE"
EV_ARITHMETIC_VERIFIED = "ARITHMETIC_VERIFIED"
EV_INVALID = "INVALID"

_METHODS = {
    "NOT_COMPUTED", "DIRECT_COUNT", "EXPECTED_COUNT", "UTILITY_INDEX",
}
_DIRECTION = {
    "BENEFIT": 1.0,
    "BENEFICIAL": 1.0,
    "HARM": -1.0,
    "ADVERSE": -1.0,
    "OPPORTUNITY_COST": -1.0,
    "FOREGONE": -1.0,
    "NEUTRAL": 0.0,
}
_MORTALITY_LANGUAGE = re.compile(
    r"\b(?:dead|death|deaths|die|dies|fatal|fatality|fatalities|"
    r"kill|killed|lives?\s+(?:lost|saved)|mortality)\b",
    re.IGNORECASE,
)
_TRAPPING_LANGUAGE = re.compile(r"\b(?:trap|trapped|trapping)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ExpectedValueValidation:
    status: str
    estimates: dict[str, dict[str, Any]]
    errors: tuple[str, ...] = ()


def _record_effect_ids(record: Mapping[str, Any]) -> set[str]:
    values = {
        str(value).strip()
        for value in record.get("grounded_effect_ids", []) or []
        if str(value).strip()
    }
    for key in ("grounded_effect_id", "effect_id"):
        value = str(record.get(key, "") or "").strip()
        if value:
            values.add(value)
    return values


def _probability(record: Mapping[str, Any]) -> float | None:
    raw = " ".join(str(record.get("probability", "") or "").split())
    modality = str(record.get("modality", "") or "").strip().upper()
    if raw.upper() in {"CERTAIN", "CERTAINTY", "DEFINITE", "DEFINITELY"}:
        return 1.0
    if modality == "CERTAIN":
        return 1.0
    for value in (
        raw,
        str(record.get("qualifier", "") or ""),
        " ".join(str(value) for value in record.get("likelihood_qualifiers", []) or []),
    ):
        match = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|percent\b)?", value, re.I)
        if not match:
            continue
        number = float(match.group(1))
        if match.group(2):
            return max(0.0, min(1.0, number / 100.0))
        if 0.0 <= number <= 1.0:
            return number
    return None


def _magnitude(record: Mapping[str, Any]) -> float | None:
    spans = [str(record.get("magnitude", "") or "")]
    for key in ("affected_subject_quantities", "party_quantities", "quantities"):
        spans.extend(str(value) for value in record.get(key, []) or [])
    # A general qualifier is a last resort because it often contains
    # likelihood rather than population.  Percentages are never magnitudes.
    spans.append(str(record.get("qualifier", "") or ""))
    for span in spans:
        if not span.strip() or re.search(r"%|\bpercent\b", span, re.I):
            continue
        parsed = quantity_magnitude(span)
        if parsed is not None:
            return float(parsed)
    return None


def _signed_direction(record: Mapping[str, Any]) -> float | None:
    for key in ("direction", "polarity"):
        value = str(record.get(key, "") or "").strip().upper()
        if value in _DIRECTION:
            return _DIRECTION[value]
    return None


def _recomputed_value(
    source_records: Sequence[Mapping[str, Any]], method: str,
) -> tuple[float, str] | None:
    if method not in {"DIRECT_COUNT", "EXPECTED_COUNT"}:
        return None
    total = 0.0
    for record in source_records:
        direction = _signed_direction(record)
        magnitude = _magnitude(record)
        probability = _probability(record)
        if direction is None or magnitude is None:
            return None
        if method == "DIRECT_COUNT":
            if probability != 1.0:
                return None
            probability = 1.0
        elif probability is None:
            return None
        total += direction * magnitude * probability
    direction = "BENEFIT" if total >= 0 else "HARM"
    return abs(total), direction


def validate_expected_value_estimates(
    estimates: Mapping[str, Mapping[str, Any]] | None,
    *,
    action_id_by_key: Mapping[str, str],
    effect_records: Sequence[Mapping[str, Any]],
) -> ExpectedValueValidation:
    """Validate specialist EV rows against admitted action-effect records.

    ``action_id_by_key`` allows callers to use canonical IDs or action text as
    estimate keys.  Missing EV is valid and remains distinct from a failed EV.
    """
    submitted = dict(estimates or {})
    normalized: dict[str, dict[str, Any]] = {}
    records_by_effect: dict[str, list[Mapping[str, Any]]] = {}
    for record in effect_records:
        for effect_id in _record_effect_ids(record):
            records_by_effect.setdefault(effect_id, []).append(record)

    claimed_keys: list[str] = []
    errors: list[str] = []
    arithmetic_rows = 0
    correspondence_rows = 0
    for key, action_id in action_id_by_key.items():
        raw = dict(submitted.get(key) or {})
        claimed = raw.get("grounded") is True
        if claimed:
            claimed_keys.append(key)
        method = str(raw.get("method", "NOT_COMPUTED") or "NOT_COMPUTED").strip().upper()
        source_ids_raw = raw.get("source_effect_ids", [])
        source_ids = list(dict.fromkeys(
            str(value).strip()
            for value in source_ids_raw
            if str(value).strip()
        )) if isinstance(source_ids_raw, list) else []
        calculation = " ".join(str(raw.get("calculation", "") or "").split())[:240]
        assumptions_raw = raw.get("assumptions", [])
        assumptions = [
            " ".join(str(value).split())[:160]
            for value in assumptions_raw
            if " ".join(str(value).split())
        ] if isinstance(assumptions_raw, list) else []
        try:
            value = float(raw.get("value", 0.0))
        except (TypeError, ValueError):
            value = 0.0
            if claimed:
                errors.append(f"{action_id} EV value is not numeric")
        unit = str(raw.get("unit", "NONE") or "NONE").strip().upper()
        direction = str(raw.get("direction", "HARM") or "HARM").strip().upper()
        row_errors: list[str] = []
        if claimed:
            if method not in _METHODS or method == "NOT_COMPUTED":
                row_errors.append("grounded EV omits a calculation method")
            if not source_ids:
                row_errors.append("grounded EV omits source effect IDs")
            if not isinstance(source_ids_raw, list):
                row_errors.append("grounded EV source effect IDs must be a list")
            if len(calculation.split()) < 2:
                row_errors.append("grounded EV omits the specialist calculation")
            if not isinstance(assumptions_raw, list):
                row_errors.append("grounded EV assumptions must be a list")
            if unit in {"", "NONE"}:
                row_errors.append("grounded EV omits a comparable unit")
            if direction not in {"BENEFIT", "HARM"}:
                row_errors.append("grounded EV has an invalid direction")
            if not math.isfinite(value) or value < 0:
                row_errors.append("grounded EV must be a finite non-negative value")

            source_records: list[Mapping[str, Any]] = []
            for effect_id in source_ids:
                matches = records_by_effect.get(effect_id, [])
                if not matches:
                    row_errors.append(f"EV cites unknown effect {effect_id}")
                    continue
                local = [
                    record for record in matches
                    if str(record.get("action_id", record.get("canonical_action_id", ""))).upper()
                    == action_id.upper()
                ]
                if not local:
                    row_errors.append(f"EV cites cross-action effect {effect_id}")
                    continue
                source_records.append(local[0])

            source_outcomes = " ".join(
                str(record.get("outcome", "") or "").replace("_", " ")
                for record in source_records
            )
            ev_claim = f"{unit} {calculation}"
            if (
                source_outcomes
                and _MORTALITY_LANGUAGE.search(ev_claim)
                and _TRAPPING_LANGUAGE.search(source_outcomes)
                and not _MORTALITY_LANGUAGE.search(source_outcomes)
            ):
                row_errors.append(
                    "EV changes a trapping outcome into a mortality quantity"
                )

            recomputed = _recomputed_value(source_records, method)
            if method in {"DIRECT_COUNT", "EXPECTED_COUNT"}:
                if recomputed is None:
                    row_errors.append(
                        "numeric EV cannot be recomputed from its cited effects"
                    )
                else:
                    expected_value, expected_direction = recomputed
                    if not math.isclose(value, expected_value, rel_tol=0.01, abs_tol=0.001):
                        row_errors.append(
                            f"EV value {value:g} does not match recomputed {expected_value:g}"
                        )
                    if value > 0 and direction != expected_direction:
                        row_errors.append(
                            f"EV direction {direction} contradicts recomputed {expected_direction}"
                        )
                    if not row_errors:
                        arithmetic_rows += 1
            elif method == "UTILITY_INDEX" and not row_errors:
                correspondence_rows += 1

        row_status = (
            EV_INVALID if row_errors
            else EV_ARITHMETIC_VERIFIED if claimed and method in {"DIRECT_COUNT", "EXPECTED_COUNT"}
            else EV_SOURCE_CORRESPONDENCE if claimed
            else EV_NOT_CLAIMED
        )
        errors.extend(f"{action_id}: {error}" for error in row_errors)
        normalized[key] = {
            "value": value,
            "unit": unit,
            "direction": direction,
            "grounded": claimed and not row_errors,
            "claimed_grounded": claimed,
            "method": method,
            "source_effect_ids": source_ids,
            "calculation": calculation,
            "assumptions": assumptions,
            "validation_status": row_status,
            "validation_errors": row_errors,
        }

    if not claimed_keys:
        return ExpectedValueValidation(EV_NOT_CLAIMED, normalized)
    if set(claimed_keys) != set(action_id_by_key):
        errors.append("grounded EV comparison must cover every live action")
    units = {normalized[key]["unit"] for key in claimed_keys}
    directions = {normalized[key]["direction"] for key in claimed_keys}
    if len(units) != 1:
        errors.append("grounded EV rows do not share one unit")
    if len(directions) != 1:
        errors.append("grounded EV rows do not share one direction")
    if errors:
        for row in normalized.values():
            row["grounded"] = False
            if row.get("claimed_grounded"):
                row["validation_status"] = EV_INVALID
        return ExpectedValueValidation(
            EV_INVALID, normalized, tuple(dict.fromkeys(errors))[:12],
        )
    if arithmetic_rows == len(action_id_by_key):
        return ExpectedValueValidation(EV_ARITHMETIC_VERIFIED, normalized)
    if correspondence_rows or arithmetic_rows:
        return ExpectedValueValidation(EV_SOURCE_CORRESPONDENCE, normalized)
    return ExpectedValueValidation(EV_INVALID, normalized, ("EV validation produced no usable rows",))


def expected_value_leader(
    estimates: Mapping[str, Mapping[str, Any]], actions: Sequence[str],
) -> str | None:
    """Return a unique winner only from system-verified arithmetic rows."""
    rows = [dict(estimates.get(action) or {}) for action in actions]
    if not rows or not all(
        row.get("validation_status") == EV_ARITHMETIC_VERIFIED for row in rows
    ):
        return None
    units = {str(row.get("unit", "")).upper() for row in rows}
    directions = {str(row.get("direction", "")).upper() for row in rows}
    if len(units) != 1 or units & {"", "NONE"} or directions not in ({"BENEFIT"}, {"HARM"}):
        return None
    values = {action: float(estimates[action]["value"]) for action in actions}
    best = max(values, key=values.get) if directions == {"BENEFIT"} else min(values, key=values.get)
    return best if list(values.values()).count(values[best]) == 1 else None
