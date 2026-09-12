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
    r"(?<![A-Za-z0-9_])(?:A\d+(?:_[A-Za-z0-9_]+)?|E[A-Za-z0-9_]*|P\d+(?:_[A-Za-z0-9_]+)?|CT[A-Za-z0-9_]*|COND[A-Za-z0-9_]*)(?![A-Za-z0-9_])"
)
_LINK_INDEX = re.compile(r"causal_link\[(\d+)\]")

LOCAL_PATCH = "LOCAL_PATCH"
SUBGRAPH_REBUILD = "SUBGRAPH_REBUILD"
FULL_REBUILD = "FULL_REBUILD"
REPAIR_SCOPES = frozenset({LOCAL_PATCH, SUBGRAPH_REBUILD, FULL_REBUILD})

_SUBGRAPH_ISSUE_CODES = frozenset({
    "MISSING_DIRECT_INTERVENTION",
    "MISSING_CAUSAL_PARENT",
    "DIRECT_EFFECT_TARGET_MISMATCH",
    "INVALID_CAUSAL_ANCESTRY",
    "INDEPENDENT_EVENT_AS_CAUSAL_PARENT",
    "MISSING_PROCESS_INTERMEDIATE",
    "RESOURCE_TRANSFER_TARGET_MISMATCH",
})
_GLOBAL_FAILURE_PATTERNS = (
    "action-source mapping must cover every canonical action",
    "typed world actions must cover every canonical action exactly once",
    "effects require unique non-empty effect_id values",
    "world_model is required",
)
_SUBGRAPH_FAILURE_PATTERNS = (
    "lacks an atomic direct intervention",
    "downstream human outcome with no causal parent",
    "only path to a direct act passes through",
    "independent background event",
    "independent stochastic event",
    "caused directly by another party's act",
    "no path to this action's direct",
    "insert a process, facility, institution, infrastructure, or resource state",
    "names transferred resource",
)


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
    if "recipient" in lowered and "lacks an atomic direct intervention" in lowered:
        return "MISSING_DIRECT_INTERVENTION", "effects", "SEMANTIC_PATCH", False
    if "names transferred resource" in lowered and "as a recipient" in lowered:
        return "RESOURCE_TRANSFER_TARGET_MISMATCH", "effects", "SEMANTIC_PATCH", False
    if "caused directly by another party's act" in lowered:
        return "MISSING_PROCESS_INTERMEDIATE", "causal_links", "SEMANTIC_PATCH", False
    if (
        "is direct on" in lowered
        and "neither the actor nor a named recipient" in lowered
    ):
        return "DIRECT_EFFECT_TARGET_MISMATCH", "effects", "SEMANTIC_PATCH", False
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
    if (
        "downstream human outcome whose causal ancestry" in lowered
        and "never reaches a direct act" in lowered
    ):
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


def _issue_rows(
    issues: Sequence[ValidationIssue | dict[str, object]],
) -> list[dict[str, object]]:
    return [
        issue.as_dict() if isinstance(issue, ValidationIssue) else dict(issue)
        for issue in issues
    ]


def implicated_action_ids(
    issues: Sequence[ValidationIssue | dict[str, object]],
    candidate: object | None,
) -> tuple[str, ...]:
    """Resolve typed issue entities to their owning canonical action."""
    world = (
        candidate.get("world_model")
        if isinstance(candidate, dict) else None
    )
    if not isinstance(world, dict):
        return ()
    effect_actions = {
        str(effect.get("effect_id") or ""): str(effect.get("action_id") or "")
        for effect in world.get("effects") or []
        if isinstance(effect, dict) and effect.get("effect_id")
    }
    condition_events = {
        str(condition.get("condition_id") or ""): str(
            condition.get("event_effect_id") or ""
        )
        for condition in world.get("conditions") or []
        if isinstance(condition, dict) and condition.get("condition_id")
    }
    links = [
        link for link in world.get("causal_links") or []
        if isinstance(link, dict)
    ]
    found: set[str] = set()
    for issue in _issue_rows(issues):
        identifiers = [
            str(issue.get("entity_id") or ""),
            *[str(value) for value in issue.get("related_ids", ())],
        ]
        kind = str(issue.get("entity_kind") or "")
        if kind == "causal_link" and identifiers[0].isdigit():
            index = int(identifiers[0])
            if 0 <= index < len(links):
                action_id = str(links[index].get("action_id") or "")
                if action_id:
                    found.add(action_id)
        for identifier in identifiers:
            if re.fullmatch(r"A\d+", identifier):
                found.add(identifier)
                continue
            effect_id = condition_events.get(identifier, identifier)
            action_id = effect_actions.get(effect_id, "")
            if action_id:
                found.add(action_id)
    return tuple(sorted(found))


def classify_world_repair_scope(
    issues: Sequence[ValidationIssue | dict[str, object]],
    *,
    errors: Sequence[str] = (),
    candidate: object | None = None,
) -> str:
    """Choose how much rejected state a repair call may safely inherit.

    Scope is intentionally separate from repair_class. The latter describes the
    kind of mutation; this classification governs the size of the graph region
    that may be replaced.
    """
    if not isinstance(candidate, dict):
        return FULL_REBUILD
    rows = _issue_rows(issues)
    messages = [
        str(row.get("message") or "") for row in rows
    ] + [str(error) for error in errors]
    folded = " ".join(messages).casefold()
    if not rows and "action-source mapping failed:" in folded:
        return FULL_REBUILD
    if any(pattern in folded for pattern in _GLOBAL_FAILURE_PATTERNS):
        return FULL_REBUILD
    codes = {str(row.get("code") or "") for row in rows}
    if codes & _SUBGRAPH_ISSUE_CODES or any(
        pattern in folded for pattern in _SUBGRAPH_FAILURE_PATTERNS
    ):
        return SUBGRAPH_REBUILD
    return LOCAL_PATCH


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
            elif entity_id.startswith("P"):
                entity_kind = "party"
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
    *,
    errors: Sequence[str] = (),
    candidate: object | None = None,
) -> dict[str, object]:
    """Return the mutation boundary for a transactional repair response."""
    issue_rows = _issue_rows(issues)
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
    repair_scope = classify_world_repair_scope(
        issue_rows, errors=errors, candidate=candidate,
    )
    return {
        "schema_version": 1,
        "repair_scope": repair_scope,
        "implicated_action_ids": list(
            implicated_action_ids(issue_rows, candidate)
            if repair_scope == SUBGRAPH_REBUILD else ()
        ),
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
