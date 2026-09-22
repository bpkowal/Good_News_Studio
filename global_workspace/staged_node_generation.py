"""Action-neutral proposition extraction followed by explicit branch ownership."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import re
from typing import Any


ADMIT = "ADMIT"
QUARANTINE = "QUARANTINE"
REJECT = "REJECT"


_STRUCTURAL_REJECTION_MARKERS = (
    "unknown action_id",
    "unknown party_id",
    "cites unknown clauses",
    "unique non-empty proposition_id",
    "proposition rows must be objects",
)


def _identifier_mentioned(message: str, identifier: str) -> bool:
    if not identifier:
        return False
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])", message) is not None


def triage_node_admission(
    skeleton: Mapping[str, Any], validation_errors: Sequence[str],
) -> dict[str, Any]:
    """Assign ADMIT/QUARANTINE/REJECT without globally erasing valid nodes.

    Quarantine is for plausible semantic evidence whose trust conditions are
    incomplete (for example imperfect provenance or a missing quantity).
    Reject is reserved for structurally unsafe graph projections. Errors that
    cannot be attributed to a node remain explicit stage-level issues.
    """
    propositions = [
        row for row in skeleton.get("propositions") or []
        if isinstance(row, Mapping)
    ]
    errors = list(dict.fromkeys(str(error) for error in validation_errors))
    decisions: dict[str, dict[str, Any]] = {}
    attributed: set[int] = set()

    for proposition in propositions:
        proposition_id = str(proposition.get("proposition_id") or "")
        neutral_id = str(proposition.get("neutral_proposition_id") or "")
        identifiers = [value for value in (proposition_id, neutral_id) if value]
        node_errors: list[str] = []
        for index, error in enumerate(errors):
            if any(_identifier_mentioned(error, identifier) for identifier in identifiers):
                node_errors.append(error)
                attributed.add(index)
        status = ADMIT
        if node_errors:
            status = (
                REJECT
                if any(
                    marker in error.casefold()
                    for error in node_errors
                    for marker in _STRUCTURAL_REJECTION_MARKERS
                )
                else QUARANTINE
            )
        decisions[proposition_id] = {
            "status": status,
            "reasons": node_errors,
            "neutral_proposition_id": neutral_id or None,
        }

    # Some validators report source obligations at clause or party scope rather
    # than naming a proposition. Attribute those conservatively to projections
    # that cite the same evidence instead of poisoning the whole skeleton.
    for index, error in enumerate(errors):
        if index in attributed:
            continue
        folded = error.casefold()
        matched_ids: list[str] = []
        for proposition in propositions:
            proposition_id = str(proposition.get("proposition_id") or "")
            clause_ids = [str(value) for value in proposition.get("clause_ids") or []]
            party_id = str(proposition.get("party_id") or "")
            if any(_identifier_mentioned(error, clause_id) for clause_id in clause_ids):
                matched_ids.append(proposition_id)
            elif party_id and _identifier_mentioned(error, party_id):
                matched_ids.append(proposition_id)
        if not matched_ids and (
            "ellipsis" in folded or "required source quantity" in folded
        ):
            # An unscoped evidence-completeness defect is epistemic, not proof
            # that every node is false. Keep the evidence, but admit none of it
            # as fully trusted until the obligation is resolved.
            matched_ids = list(decisions)
        if not matched_ids and any(
            marker in folded for marker in (
                "parties require unique", "party rows must be objects",
                "skeleton requires non-empty", "skeleton must be an object",
            )
        ):
            matched_ids = list(decisions)
            target_status = REJECT
        else:
            target_status = QUARANTINE
        if matched_ids:
            attributed.add(index)
            for proposition_id in matched_ids:
                decision = decisions[proposition_id]
                if target_status == REJECT or decision["status"] == ADMIT:
                    decision["status"] = target_status
                if error not in decision["reasons"]:
                    decision["reasons"].append(error)

    global_errors = [error for index, error in enumerate(errors) if index not in attributed]
    admitted_ids = {
        proposition_id for proposition_id, decision in decisions.items()
        if decision["status"] == ADMIT
    }
    admitted_skeleton = copy.deepcopy(dict(skeleton))
    admitted_skeleton["propositions"] = [
        copy.deepcopy(dict(row)) for row in propositions
        if str(row.get("proposition_id") or "") in admitted_ids
    ]
    admitted_count = len(admitted_ids)
    non_admitted_count = len(decisions) - admitted_count
    if admitted_count and not non_admitted_count and not global_errors:
        stage_status = "ADMITTED"
    elif admitted_count:
        stage_status = "PARTIALLY_ADMITTED"
    elif any(row["status"] == QUARANTINE for row in decisions.values()):
        stage_status = "QUARANTINED"
    else:
        stage_status = "REJECTED"
    return {
        "stage_status": stage_status,
        "node_decisions": decisions,
        "global_errors": global_errors,
        "admitted_skeleton": admitted_skeleton,
        "counts": {
            "admitted": sum(row["status"] == ADMIT for row in decisions.values()),
            "quarantined": sum(row["status"] == QUARANTINE for row in decisions.values()),
            "rejected": sum(row["status"] == REJECT for row in decisions.values()),
        },
    }


def neutral_skeleton_schema(clause_ids: Sequence[str]) -> dict[str, Any]:
    strings = {"type": "array", "items": {"type": "string"}}
    sources = {
        "type": "array", "minItems": 1,
        "items": {"type": "string", "enum": list(clause_ids)},
    }
    return {
        "type": "object",
        "properties": {
            "parties": {"type": "array", "minItems": 1, "items": {
                "type": "object", "properties": {
                    "party_id": {"type": "string"}, "label": {"type": "string"},
                    "kind": {"type": "string"}, "quantities": strings,
                    "clause_ids": sources,
                },
                "required": ["party_id", "label", "kind", "quantities", "clause_ids"],
                "additionalProperties": False,
            }},
            "propositions": {"type": "array", "minItems": 1, "items": {
                "type": "object", "properties": {
                    "proposition_id": {"type": "string"},
                    "party_id": {"type": "string"}, "outcome": {"type": "string"},
                    "polarity": {"type": "string", "enum": [
                        "BENEFICIAL", "ADVERSE", "NEUTRAL", "UNRESOLVED",
                    ]},
                    "directness": {"type": "string", "enum": ["DIRECT", "DOWNSTREAM"]},
                    "modality": {"type": "string", "enum": [
                        "CERTAIN", "STIPULATED_CONDITIONAL", "PROBABILISTIC",
                        "POSSIBLE", "UNKNOWN",
                    ]},
                    "effect_kind": {"type": "string"}, "quantities": strings,
                    "source_proposition": {"type": "string", "minLength": 1},
                    "clause_ids": sources,
                },
                "required": [
                    "proposition_id", "party_id", "outcome", "polarity",
                    "directness", "modality", "effect_kind", "quantities",
                    "source_proposition", "clause_ids",
                ],
                "additionalProperties": False,
            }},
            "unresolved_source_spans": strings,
        },
        "required": ["parties", "propositions", "unresolved_source_spans"],
        "additionalProperties": False,
    }


def ownership_schema(
    proposition_ids: Sequence[str], action_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "bindings": {"type": "array", "minItems": 1, "items": {
                "type": "object", "properties": {
                    "proposition_id": {"type": "string", "enum": list(proposition_ids)},
                    "action_ids": {"type": "array", "items": {
                        "type": "string", "enum": list(action_ids),
                    }},
                    "status": {"type": "string", "enum": [
                        "OWNED", "SHARED", "UNRESOLVED",
                    ]},
                    "reason": {"type": "string"},
                    "evidence_spans": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "proposition_id", "action_ids", "status", "reason",
                    "evidence_spans",
                ],
                "additionalProperties": False,
            }},
        },
        "required": ["bindings"],
        "additionalProperties": False,
    }


def materialize_owned_skeleton(
    neutral: Mapping[str, Any], ownership: Mapping[str, Any],
    action_ids: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    """Duplicate shared atoms by action while preserving their neutral origin."""
    allowed_actions = set(action_ids)
    propositions = {
        str(row.get("proposition_id") or ""): dict(row)
        for row in neutral.get("propositions") or []
        if isinstance(row, Mapping) and row.get("proposition_id")
    }
    bindings: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for row in ownership.get("bindings") or []:
        if not isinstance(row, Mapping):
            errors.append("ownership binding must be an object")
            continue
        proposition_id = str(row.get("proposition_id") or "")
        if proposition_id not in propositions:
            errors.append(f"ownership cites unknown proposition {proposition_id!r}")
            continue
        if proposition_id in bindings:
            errors.append(f"ownership duplicates proposition {proposition_id!r}")
            continue
        bindings[proposition_id] = row
    owned: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for proposition_id, proposition in propositions.items():
        binding = bindings.get(proposition_id)
        ledger_row = {
            "neutral_proposition_id": proposition_id,
            "proposition": copy.deepcopy(proposition),
            "extracted": True,
            "ownership_status": "MISSING",
            "assigned_action_ids": [],
            "materialized_proposition_ids": [],
            "ownership_reasons": [],
            "normalization_status": "NOT_RUN",
            "normalization_annotations": [],
            "provenance_status": "NOT_RUN",
            "graph_admission_status": "NOT_EVALUATED",
        }
        ledger.append(ledger_row)
        if binding is None:
            errors.append(f"proposition {proposition_id} has no ownership binding")
            ledger_row["ownership_reasons"].append("MISSING_BINDING")
            continue
        assigned = list(dict.fromkeys(
            str(value) for value in binding.get("action_ids") or []
        ))
        invalid = [value for value in assigned if value not in allowed_actions]
        if invalid:
            errors.append(f"proposition {proposition_id} has invalid actions {invalid}")
            ledger_row["ownership_status"] = "INVALID"
            ledger_row["ownership_reasons"].append("INVALID_ACTION_IDS")
            continue
        status = str(binding.get("status") or "").upper()
        if status == "SHARED" and len(assigned) == 1:
            status = "OWNED"
            ledger_row["ownership_reasons"].append(
                "STATUS_REPAIRED_SHARED_TO_OWNED_FROM_ACTION_CARDINALITY"
            )
        elif status == "OWNED" and len(assigned) > 1:
            status = "SHARED"
            ledger_row["ownership_reasons"].append(
                "STATUS_REPAIRED_OWNED_TO_SHARED_FROM_ACTION_CARDINALITY"
            )
        ledger_row["ownership_status"] = status or "MISSING"
        ledger_row["assigned_action_ids"] = assigned
        if status == "UNRESOLVED" or not assigned:
            errors.append(f"proposition {proposition_id} ownership is unresolved")
            ledger_row["ownership_reasons"].append("UNRESOLVED")
            continue
        if status == "OWNED" and len(assigned) != 1:
            errors.append(f"OWNED proposition {proposition_id} requires one action")
            ledger_row["ownership_reasons"].append("OWNED_CARDINALITY_INVALID")
            continue
        if status == "SHARED" and len(assigned) < 2:
            errors.append(f"SHARED proposition {proposition_id} requires multiple actions")
            ledger_row["ownership_reasons"].append("SHARED_CARDINALITY_INVALID")
            continue
        for action_id in assigned:
            row = copy.deepcopy(proposition)
            row["neutral_proposition_id"] = proposition_id
            row["proposition_id"] = (
                proposition_id if len(assigned) == 1 else f"{proposition_id}_{action_id}"
            )
            row["action_id"] = action_id
            owned.append(row)
            ledger_row["materialized_proposition_ids"].append(row["proposition_id"])
    return {
        "parties": copy.deepcopy(list(neutral.get("parties") or [])),
        "propositions": owned,
        "unresolved_source_spans": copy.deepcopy(
            list(neutral.get("unresolved_source_spans") or [])
        ),
        "ellipsis_resolutions": copy.deepcopy(
            list(neutral.get("ellipsis_resolutions") or [])
        ),
        "ownership_bindings": copy.deepcopy(list(ownership.get("bindings") or [])),
        "node_evidence_ledger": ledger,
    }, errors


def finalize_node_evidence_ledger(
    skeleton: dict[str, Any], validation_errors: Sequence[str], *,
    admitted: bool | None = None,
    node_decisions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, int | float | None]:
    """Annotate, never erase, each neutral node's transformation lifecycle."""
    ledger = [
        row for row in skeleton.get("node_evidence_ledger") or []
        if isinstance(row, dict)
    ]
    surviving_ids = {
        str(row.get("proposition_id") or "")
        for row in skeleton.get("propositions") or []
        if isinstance(row, Mapping)
    }
    for row in ledger:
        materialized = [str(value) for value in row.get("materialized_proposition_ids") or []]
        survived = [value for value in materialized if value in surviving_ids]
        if not materialized:
            row["normalization_status"] = "NOT_MATERIALIZED"
        elif len(survived) == len(materialized):
            row["normalization_status"] = "SURVIVED"
        elif survived:
            row["normalization_status"] = "PARTIALLY_SURVIVED"
        else:
            row["normalization_status"] = "DROPPED_FROM_GRAPH_VIEW"
            if not row.get("normalization_annotations"):
                row["normalization_annotations"] = [{
                    "reason": "UNCLASSIFIED_NORMALIZATION_DROP",
                    "detail": "Materialized projection disappeared without a typed annotation.",
                }]
        implicated = [
            error for error in validation_errors
            if any(identifier and identifier in str(error) for identifier in materialized)
        ]
        for identifier in survived:
            implicated.extend(
                str(error)
                for error in (node_decisions or {}).get(identifier, {}).get("reasons") or []
            )
        implicated = list(dict.fromkeys(implicated))
        provenance_errors = [
            error for error in implicated
            if "source_proposition" in error.casefold()
            or "provenance" in error.casefold()
            or "source span" in error.casefold()
        ]
        row["validation_errors"] = implicated
        row["provenance_status"] = (
            "IMPERFECT" if provenance_errors
            else "VALID" if survived
            else "NOT_EVALUATED"
        )
        materialized_decisions = [
            str((node_decisions or {}).get(identifier, {}).get("status") or "")
            for identifier in survived
        ]
        row["materialized_admission_decisions"] = {
            identifier: copy.deepcopy(dict((node_decisions or {}).get(identifier) or {}))
            for identifier in survived
        }
        if node_decisions is not None and materialized_decisions:
            if REJECT in materialized_decisions:
                row["graph_admission_status"] = "REJECTED"
            elif QUARANTINE in materialized_decisions:
                row["graph_admission_status"] = "QUARANTINED"
            else:
                row["graph_admission_status"] = "ADMITTED"
        else:
            row["graph_admission_status"] = (
                "ADMITTED" if admitted and survived and not implicated
                else "QUARANTINED" if materialized or implicated
                else "UNRESOLVED"
            )
    extracted = len(ledger)
    ownership = sum(bool(row.get("materialized_proposition_ids")) for row in ledger)
    normalized = sum(
        row.get("normalization_status") in {"SURVIVED", "PARTIALLY_SURVIVED"}
        for row in ledger
    )
    provenance = sum(row.get("provenance_status") == "VALID" for row in ledger)
    admitted_count = sum(row.get("graph_admission_status") == "ADMITTED" for row in ledger)
    quarantined_count = sum(
        row.get("graph_admission_status") == "QUARANTINED" for row in ledger
    )
    rejected_count = sum(
        row.get("graph_admission_status") == "REJECTED" for row in ledger
    )
    waterfall: dict[str, int | float | None] = {
        "neutral_nodes_extracted": extracted,
        "nodes_materialized_after_ownership": ownership,
        "nodes_surviving_normalization": normalized,
        "nodes_with_valid_provenance": provenance,
        "nodes_admitted_to_graph": admitted_count,
        "nodes_quarantined_from_graph": quarantined_count,
        "nodes_rejected_from_graph": rejected_count,
        "ownership_retention_rate": ownership / extracted if extracted else None,
        "normalization_retention_rate": normalized / ownership if ownership else None,
        "provenance_retention_rate": provenance / normalized if normalized else None,
        "graph_admission_rate": admitted_count / extracted if extracted else None,
    }
    skeleton["node_evidence_ledger"] = ledger
    skeleton["node_preservation_waterfall"] = waterfall
    return waterfall
