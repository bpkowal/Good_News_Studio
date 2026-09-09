"""Typed validation and repair metadata for world-model admission.

World validators historically returned human-readable strings.  Those strings
remain the public compatibility surface, while this module gives repair and
audit code stable issue codes, entity scopes, fields, and permitted mutation
classes.  A later patch-only model protocol can consume the same contract
without changing validation semantics again.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence


_ENTITY_ID = re.compile(
    r"(?<![A-Za-z0-9_])(?:A\d+(?:_[A-Za-z0-9_]+)?|E[A-Za-z0-9_]*|CT[A-Za-z0-9_]*|COND[A-Za-z0-9_]*)(?![A-Za-z0-9_])"
)
_LINK_INDEX = re.compile(r"causal_link\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    entity_kind: str = "world_model"
    entity_id: str = ""
    field: str = ""
    related_ids: tuple[str, ...] = ()
    repair_class: str = "SEMANTIC_PATCH"
    permits_removal: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class WorldModelValidationError(ValueError):
    """A backwards-compatible ValueError that retains typed issue metadata."""

    def __init__(self, messages: Sequence[str], issues: Sequence[ValidationIssue]):
        self.messages = tuple(str(message) for message in messages)
        self.issues = tuple(issues)
        super().__init__("; ".join(self.messages))


def _issue_code(message: str) -> tuple[str, str, str, bool]:
    lowered = message.casefold()
    if "is certain but lists conditions" in lowered:
        return "CERTAIN_ROW_HAS_CONDITIONS", "condition_ids", "DETERMINISTIC", False
    if "description restates" in lowered and "event_effect_id" in lowered:
        return "EVENT_REFERENCE_CONFLICT", "event_effect_id", "SEMANTIC_PATCH", False
    if "description does not identify referenced event" in lowered:
        return "EVENT_REFERENCE_MISMATCH", "event_effect_id", "SEMANTIC_PATCH", False
    if "restates existing event" in lowered:
        return "FREE_TEXT_EVENT_ALIAS", "event_effect_id", "DETERMINISTIC", False
    if "downstream human outcome with no causal parent" in lowered:
        return "MISSING_CAUSAL_PARENT", "causal_links", "SEMANTIC_PATCH", False
    if "only path to a direct act passes through" in lowered:
        return "INVALID_CAUSAL_ANCESTRY", "causal_links", "SEMANTIC_PATCH", False
    if "independent background event" in lowered or "independent stochastic event" in lowered:
        return "INDEPENDENT_EVENT_AS_CAUSAL_PARENT", "causal_links", "SEMANTIC_PATCH", False
    if "no foregone effect" in lowered:
        return "MISSING_COUNTERFACTUAL_PROJECTION", "counterfactual_links", "COMPILER_PATCH", False
    if "effect_ids do not exactly match" in lowered:
        return "STALE_EFFECT_INDEX", "effect_ids", "DETERMINISTIC", False
    if "foregone so its effect_kind" in lowered:
        return "FOREGONE_KIND_MISMATCH", "effect_kind", "DETERMINISTIC", False
    if "qualifier" in lowered and ("provenance" in lowered or "source-grounded" in lowered):
        return "SOURCE_QUALIFIER_MISMATCH", "provenance", "SOURCE_PATCH", False
    if "quantity" in lowered and "provenance" in lowered:
        return "SOURCE_QUANTITY_MISMATCH", "provenance", "SOURCE_PATCH", False
    if "source_proposition" in lowered:
        return "SOURCE_PROPOSITION_BINDING", "source_proposition", "SOURCE_PATCH", False
    if "derivation assumptions" in lowered or "changes source outcome type" in lowered:
        return "UNSUPPORTED_WORLD_DERIVATION", "derivation_operation", "QUARANTINE", False
    if "derivation_operation" in lowered or "source_effect_ids" in lowered:
        return "DERIVATION_BINDING_MISMATCH", "source_effect_ids", "SEMANTIC_PATCH", False
    if "incorrectly assigned" in lowered or "contradictory direct" in lowered:
        return "MISASSIGNED_DIRECT_EFFECT", "effects", "SEMANTIC_PATCH", True
    return "WORLD_VALIDATION_ERROR", "", "SEMANTIC_PATCH", False


def validation_issues_from_messages(
    messages: Sequence[str],
) -> tuple[ValidationIssue, ...]:
    """Attach stable repair metadata while string validators are migrated."""
    issues: list[ValidationIssue] = []
    for raw in messages:
        message = str(raw)
        code, field, repair_class, permits_removal = _issue_code(message)
        identifiers = list(dict.fromkeys(_ENTITY_ID.findall(message)))
        link = _LINK_INDEX.search(message)
        if link:
            entity_kind = "causal_link"
            entity_id = link.group(1)
        elif identifiers:
            entity_id = identifiers[0]
            if entity_id.startswith(("CT", "COND")):
                entity_kind = "condition"
            elif entity_id.startswith("E"):
                entity_kind = "effect"
            elif entity_id.startswith("A"):
                entity_kind = "action"
            else:
                entity_kind = "party"
        else:
            entity_kind = "world_model"
            entity_id = ""
        issues.append(ValidationIssue(
            code=code,
            message=message,
            entity_kind=entity_kind,
            entity_id=entity_id,
            field=field,
            related_ids=tuple(
                identifier for identifier in identifiers if identifier != entity_id
            ),
            repair_class=repair_class,
            permits_removal=permits_removal,
        ))
    return tuple(issues)


def repair_patch_contract(
    issues: Sequence[ValidationIssue | dict[str, object]],
) -> dict[str, object]:
    """Return the mutation boundary for a transactional repair response."""
    issue_rows = [
        issue.as_dict() if isinstance(issue, ValidationIssue) else dict(issue)
        for issue in issues
    ]
    entity_ids: list[str] = []
    fields: list[str] = []
    codes: list[str] = []
    removal_allowed = False
    for row in issue_rows:
        codes.append(str(row.get("code") or "WORLD_VALIDATION_ERROR"))
        entity_ids.extend([
            str(row.get("entity_id") or ""),
            *[str(value) for value in row.get("related_ids", ())],
        ])
        fields.append(str(row.get("field") or ""))
        removal_allowed = removal_allowed or bool(row.get("permits_removal"))
    operations = ["add", "replace"]
    if removal_allowed:
        operations.append("remove")
    return {
        "schema_version": 1,
        "issue_codes": list(dict.fromkeys(code for code in codes if code)),
        "allowed_entity_ids": list(dict.fromkeys(value for value in entity_ids if value)),
        "allowed_fields": list(dict.fromkeys(value for value in fields if value)),
        "allowed_operations": operations,
        "full_candidate_response_compatibility": True,
    }


def issue_mentions_identifier(issue: ValidationIssue | dict[str, object], identifier: str) -> bool:
    row = issue.as_dict() if isinstance(issue, ValidationIssue) else issue
    values = {
        str(row.get("entity_id") or ""),
        *[str(value) for value in row.get("related_ids", ())],
    }
    return str(identifier) in values
