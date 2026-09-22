"""Typed validation and repair metadata for world-model admission.

World validators historically returned human-readable strings.  Those strings
remain the public compatibility surface, while this module gives repair and
audit code stable issue codes, entity scopes, fields, and permitted mutation
classes.  A later patch-only model protocol can consume the same contract
without changing validation semantics again.
"""
from __future__ import annotations

import ast
import copy
import re
from dataclasses import asdict, dataclass
from typing import Sequence


_ENTITY_ID = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"A\d+_e\d+"
    r"|E[A-Za-z0-9_]*"
    r"|(?:AV|F)[A-Za-z0-9_]*"
    r"|S[A-Za-z0-9_]*"
    r"|A\d+(?:_[A-Za-z0-9_]+)?"
    r"|P\d+(?:_[A-Za-z0-9_]+)?"
    r"|CT[A-Za-z0-9_]*"
    r"|COND[A-Za-z0-9_]*"
    r")(?![A-Za-z0-9_])"
)
_LINK_INDEX = re.compile(r"causal_link\[(\d+)\]")

LOCAL_PATCH = "LOCAL_PATCH"
DETERMINISTIC_LOCAL_PATCH = "DETERMINISTIC_LOCAL_PATCH"
SUBGRAPH_REBUILD = "SUBGRAPH_REBUILD"
FULL_REBUILD = "FULL_REBUILD"
REPAIR_SCOPES = frozenset({LOCAL_PATCH, SUBGRAPH_REBUILD, FULL_REBUILD})
# DETERMINISTIC_LOCAL_PATCH is an execution mode of LOCAL_PATCH, not a
# separate inheritance scope for classify_world_repair_scope.

_SUBGRAPH_ISSUE_CODES = frozenset({
    "MISSING_DIRECT_INTERVENTION",
    "MISSING_CAUSAL_PARENT",
    "DIRECT_EFFECT_TARGET_MISMATCH",
    "INVALID_CAUSAL_ANCESTRY",
    "INDEPENDENT_EVENT_AS_CAUSAL_PARENT",
    "MISSING_PROCESS_INTERMEDIATE",
    "RESOURCE_TRANSFER_TARGET_MISMATCH",
    "STAGE_SKELETON_DRIFT",
    "GLOBAL_CONSTRAINT_AS_PROCESS",
    "EXCLUSIVE_ALLOCATION_BRANCH_MISSING",
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

SOURCE_REPAIR_OWNER = "SOURCE_SEMANTIC_REPAIR"
QUANTITY_REPAIR_OWNER = "QUANTITY_NORMALIZER"
TOPOLOGY_REPAIR_OWNER = "TOPOLOGY_COMPILER"
COUNTERFACTUAL_REPAIR_OWNER = "COUNTERFACTUAL_COMPILER"
SYSTEM_REPAIR_OWNER = "SYSTEM_INVARIANT"

VALIDATION_GATE_ORDER = (
    "SOURCE_BINDING",
    "ACTUAL_TOPOLOGY",
    "COUNTERFACTUAL_DERIVATION",
    "GLOBAL_INVARIANTS",
)


def repair_owner_for_code(code: str) -> str:
    """Assign one subsystem responsibility for a validation failure."""
    normalized = str(code or "").upper()
    if normalized == "DERIVATION_CONTRACT_OPERATION_MISMATCH":
        return SOURCE_REPAIR_OWNER
    if "QUANTITY" in normalized or normalized in {
        "LIKELIHOOD_QUALIFIER_MISSING", "SCOPE_QUALIFIER_MISSING",
        "TEMPORAL_QUALIFIER_MISSING",
    }:
        return QUANTITY_REPAIR_OWNER
    if any(token in normalized for token in (
        "COUNTERFACTUAL", "ALTERNATIVE", "FOREGONE", "AVERTED", "DERIVATION",
    )):
        return COUNTERFACTUAL_REPAIR_OWNER
    if normalized in _SUBGRAPH_ISSUE_CODES or any(
        token in normalized for token in ("CAUSAL", "PROCESS", "PARENT", "ANCESTRY")
    ):
        return TOPOLOGY_REPAIR_OWNER
    if normalized in {"COMPILER_SEMANTIC_REGRESSION", "REPAIR_NO_EFFECT"}:
        return SYSTEM_REPAIR_OWNER
    return SOURCE_REPAIR_OWNER


def validation_gate_for_code(code: str) -> str:
    """Locate a typed failure in the ordered admission pipeline."""
    owner = repair_owner_for_code(code)
    if owner in {SOURCE_REPAIR_OWNER, QUANTITY_REPAIR_OWNER}:
        return "SOURCE_BINDING"
    if owner == TOPOLOGY_REPAIR_OWNER:
        return "ACTUAL_TOPOLOGY"
    if owner == COUNTERFACTUAL_REPAIR_OWNER:
        return "COUNTERFACTUAL_DERIVATION"
    return "GLOBAL_INVARIANTS"


def validation_issues_by_gate(
    issues: Sequence[ValidationIssue | dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Group issues in deterministic gate order for diagnostics and repair."""
    grouped = {gate: [] for gate in VALIDATION_GATE_ORDER}
    for row in _issue_rows(issues):
        grouped[validation_gate_for_code(str(row.get("code") or ""))].append(row)
    return grouped


def repair_issue_signature(
    issues: Sequence[ValidationIssue | dict[str, object]],
) -> tuple[tuple[str, str, str], ...]:
    """Stable signature used to detect non-converging repair attempts."""
    return tuple(sorted({
        (
            str(row.get("code") or ""),
            str(row.get("entity_id") or ""),
            str(row.get("field") or ""),
        )
        for row in _issue_rows(issues)
    }))


def repair_made_progress(
    before: Sequence[ValidationIssue | dict[str, object]],
    after: Sequence[ValidationIssue | dict[str, object]],
) -> bool:
    """True only if a repair removes at least one prior typed violation."""
    before_signature = set(repair_issue_signature(before))
    after_signature = set(repair_issue_signature(after))
    return bool(before_signature - after_signature)


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

    def __init__(
        self,
        messages: Sequence[str],
        issues: Sequence[ValidationIssue],
        *,
        compiler_loss_telemetry: dict[str, object] | None = None,
    ):
        self.messages = tuple(str(message) for message in messages)
        self.issues = tuple(issues)
        self.compiler_loss_telemetry = dict(compiler_loss_telemetry or {})
        super().__init__("; ".join(self.messages))


def _issue_code(message: str) -> tuple[str, str, str, bool]:
    lowered = message.casefold()
    if "derivation contract operation mismatch" in lowered:
        return (
            "DERIVATION_CONTRACT_OPERATION_MISMATCH",
            "derivation_operation,source_proposition,clause_ids,quantities",
            "SOURCE_PATCH",
            False,
        )
    if "stage-one proposition drift" in lowered:
        return "STAGE_SKELETON_DRIFT", "effects", "SEMANTIC_PATCH", False
    if "omits exclusive-allocation branch" in lowered:
        return (
            "EXCLUSIVE_ALLOCATION_BRANCH_MISSING", "effects",
            "SEMANTIC_PATCH", False,
        )
    if "uses global resource constraint" in lowered and "causal process parent" in lowered:
        return "GLOBAL_CONSTRAINT_AS_PROCESS", "causal_links", "SEMANTIC_PATCH", False
    if "condition polarity" in lowered or "uses unless but is not negated" in lowered:
        return "NEGATION_GATE_SCOPE", "conditions", "SEMANTIC_PATCH", False
    if "condition operator" in lowered:
        return "EXCEPTION_GATE_SCOPE", "conditions", "SEMANTIC_PATCH", False
    if "temporal" in lowered and any(
        token in lowered for token in (
            "cycle", "ordering", "reflexive", "endpoints", "relation",
        )
    ):
        return "TEMPORAL_RELATION_SCOPE", "temporal_relations", "SEMANTIC_PATCH", False
    if (
        "omits target effect conditions" in lowered
        or "unconditional but target effect" in lowered
    ):
        return "CONDITIONAL_LINK_MISMATCH", "causal_links", "SYSTEM_REPAIR", False
    if "compiler semantic regression" in lowered:
        return "COMPILER_SEMANTIC_REGRESSION", "effects", "SYSTEM_REPAIR", False
    if "recipient" in lowered and "lacks an atomic direct intervention" in lowered:
        return "MISSING_DIRECT_INTERVENTION", "effects", "SEMANTIC_PATCH", False
    if "names transferred resource" in lowered and "as a recipient" in lowered:
        return "RESOURCE_TRANSFER_TARGET_MISMATCH", "effects", "SEMANTIC_PATCH", False
    if (
        "caused directly by another party's act" in lowered
        or "caused directly by act" in lowered
        or "lacks an action-specific nonreceipt or untreated state" in lowered
    ):
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
    if "no foregone effect" in lowered or "no counterfactual foregoes/precludes link" in lowered:
        return "MISSING_COUNTERFACTUAL_PROJECTION", "counterfactual_links", "COMPILER_PATCH", False
    if "duplicates an existing counterfactual link" in lowered:
        return "DUPLICATE_COUNTERFACTUAL_LINK", "counterfactual_links", "DETERMINISTIC", False
    if "duplicates an averted-alternative benefit" in lowered:
        return "DUPLICATE_AVERTED_BENEFIT", "counterfactual_links", "DETERMINISTIC", False
    if "effect_ids do not exactly match" in lowered:
        return "STALE_EFFECT_INDEX", "effect_ids", "DETERMINISTIC", False
    if "omits source clauses of its direct effects" in lowered:
        return "ACTION_DIRECT_PROVENANCE_MISSING", "clause_ids", "SOURCE_PATCH", False
    if "foregone so its effect_kind" in lowered:
        return "FOREGONE_KIND_MISMATCH", "effect_kind", "DETERMINISTIC", False
    if "omits source-grounded likelihood qualifiers" in lowered:
        return "LIKELIHOOD_QUALIFIER_MISSING", "likelihood_qualifiers", "SOURCE_PATCH", False
    if "omits source-bound likelihood" in lowered:
        return "LIKELIHOOD_BINDING_MISSING", "likelihood_qualifiers", "SOURCE_PATCH", False
    if "has no unique source-licensed antecedent" in lowered or (
        "resolution confidence" in lowered and "quarantined" in lowered
    ):
        return "ELLIPSIS_ANTECEDENT_AMBIGUOUS", "ellipsis_resolutions", "QUARANTINE", False
    if "uses ethical context as provenance" in lowered:
        return "ELLIPSIS_CONTEXT_OVERREACH", "ellipsis_resolutions", "SOURCE_PATCH", False
    if (
        "missing_constituent_type must be" in lowered
        or "ethical_context_roles do not match" in lowered
    ):
        return "ELLIPSIS_TEMPLATE_MISMATCH", "ellipsis_resolutions", "DETERMINISTIC", False
    if "ellipsis resolution record" in lowered or (
        "predicate ellipsis" in lowered and "must be resolved" in lowered
    ) or "antecedent_span is not an exact span" in lowered or (
        "selected antecedent_span was not retained" in lowered
    ) or "empty reconstructed_span" in lowered:
        return "ELLIPSIS_RESOLUTION_REQUIRED", "ellipsis_resolutions", "SOURCE_PATCH", False
    if "must remain unresolved" in lowered and "reconstructed_span" in lowered:
        return "SLUICING_MUST_REMAIN_UNRESOLVED", "ellipsis_resolutions", "DETERMINISTIC", False
    if "omits source-grounded temporal qualifiers" in lowered:
        return "TEMPORAL_QUALIFIER_MISSING", "temporal_qualifiers", "SOURCE_PATCH", False
    if "omits source-grounded scope qualifiers" in lowered:
        return "SCOPE_QUALIFIER_MISSING", "scope_qualifiers", "SOURCE_PATCH", False
    if "qualifier" in lowered and ("provenance" in lowered or "source-grounded" in lowered):
        return "SOURCE_QUALIFIER_MISMATCH", "provenance", "SOURCE_PATCH", False
    if "omits quantities stated in its outcome and provenance" in lowered:
        return "EFFECT_QUANTITY_MISSING", "quantities", "SOURCE_PATCH", False
    if "not bound to singular affected party" in lowered:
        return "EFFECT_QUANTITY_LEAK", "quantities", "DETERMINISTIC", False
    if "cross-effect quantity leak" in lowered:
        return "EFFECT_QUANTITY_LEAK", "quantities", "DETERMINISTIC", False
    if ("quantity" in lowered or "quantities" in lowered) and "provenance" in lowered:
        return "SOURCE_QUANTITY_MISMATCH", "provenance", "SOURCE_PATCH", False
    if "outcome lemma does not bind" in lowered:
        return "VERB_LEMMA_MISMATCH", "outcome", "SOURCE_PATCH", True
    if "source_proposition" in lowered:
        # Message offers bind-or-omit; removal must be contractually allowed.
        return "SOURCE_PROPOSITION_BINDING", "source_proposition", "SOURCE_PATCH", True
    if "incomplete predicate" in lowered:
        return "OUTCOME_PREDICATE_INCOMPLETE", "outcome", "SEMANTIC_PATCH", True
    if "contradicts polarity" in lowered:
        return "OUTCOME_POLARITY_CONTRADICTION", "polarity", "SEMANTIC_PATCH", True
    if "source-stipulated outcome" in lowered:
        return "SOURCE_STIPULATED_OUTCOME_MISSING", "effects", "SEMANTIC_PATCH", False
    if "source quantity" in lowered and "quantity-bearing consequence" in lowered:
        return "QUANTITY_BEARING_CONSEQUENCE_MISSING", "quantities", "SOURCE_PATCH", False
    if "quantity-bearing consequence" in lowered:
        return "QUANTITY_BEARING_CONSEQUENCE_MISSING", "effects", "SEMANTIC_PATCH", False
    if "repair had no effect" in lowered or "unstable repair" in lowered:
        return "REPAIR_NO_EFFECT", "repair", "QUARANTINE", False
    if "records quantity" in lowered and "binds to" in lowered:
        return "QUANTIFIER_PARTY_LEAK", "quantities", "SOURCE_PATCH", False
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
    if "REPAIR_NO_EFFECT" in codes:
        # Deterministic patch already proved non-effective; escalate away from
        # LOCAL_PATCH / DET retries.
        return SUBGRAPH_REBUILD
    if codes & _SUBGRAPH_ISSUE_CODES or any(
        pattern in folded for pattern in _SUBGRAPH_FAILURE_PATTERNS
    ):
        return SUBGRAPH_REBUILD
    return LOCAL_PATCH


def widen_stuck_source_binding_repair_scope(
    repair_scope: str,
    *,
    issue_codes: Sequence[str],
    prior_attempt_scopes: Sequence[str] = (),
    prior_attempt_issue_codes: Sequence[Sequence[str]] = (),
) -> str:
    """Widen LOCAL_PATCH after a prior LOCAL_PATCH still left binding residuals.

    Keeps the first source-binding repair local (rewrite or omit). Repeated
    identical residuals under LOCAL_PATCH escalate so model escalation does not
    inherit a contract that cannot clear the error.
    """
    if repair_scope != LOCAL_PATCH:
        return repair_scope
    codes = {str(code) for code in issue_codes if code}
    if "SOURCE_PROPOSITION_BINDING" not in codes:
        return repair_scope
    for scope, prior_codes in zip(prior_attempt_scopes, prior_attempt_issue_codes):
        if str(scope) != LOCAL_PATCH:
            continue
        if "SOURCE_PROPOSITION_BINDING" in {str(code) for code in prior_codes}:
            return SUBGRAPH_REBUILD
    return repair_scope


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
            elif entity_id.startswith(("E", "S")) or re.fullmatch(
                r"A\d+_[A-Za-z0-9_]+", entity_id,
            ):
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
        related = [
            identifier for identifier in identifiers if identifier != entity_id
        ]
        lic_match = re.search(r"licensing_clause_id=([A-Za-z0-9_]+)", message)
        if lic_match:
            clause_id = lic_match.group(1)
            if clause_id not in related and clause_id != entity_id:
                related.insert(0, clause_id)
        issues.append(ValidationIssue(
            code=code,
            message=message,
            entity_kind=entity_kind,
            entity_id=entity_id,
            field=field,
            related_ids=tuple(related),
            repair_class=repair_class,
            permits_removal=permits_removal,
        ))
    return tuple(issues)


def repair_patch_contract(
    issues: Sequence[ValidationIssue | dict[str, object]],
    *,
    errors: Sequence[str] = (),
    candidate: object | None = None,
    clauses: Sequence[dict[str, object]] | None = None,
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
    unique_codes = list(dict.fromkeys(code for code in codes if code))
    cards = repair_guidance_cards(
        issue_rows, candidate, clauses=clauses,
    )
    return {
        "schema_version": 1,
        "repair_scope": repair_scope,
        "implicated_action_ids": list(
            implicated_action_ids(issue_rows, candidate)
            if repair_scope == SUBGRAPH_REBUILD else ()
        ),
        "issue_codes": unique_codes,
        "allowed_entity_ids": list(dict.fromkeys(value for value in entity_ids if value)),
        "allowed_fields": list(dict.fromkeys(value for value in fields if value)),
        "allowed_operations": operations,
        "full_candidate_response_compatibility": True,
        "guidance_cards": cards,
        "guidance_prompt": format_repair_guidance_for_prompt(cards),
    }


def _clause_text_by_id(
    clauses: Sequence[dict[str, object]] | None,
) -> dict[str, str]:
    by_id: dict[str, str] = {}
    for row in clauses or ():
        if not isinstance(row, dict):
            continue
        clause_id = str(row.get("clause_id") or "").strip()
        text = " ".join(str(row.get("text") or "").split())
        if clause_id and text:
            by_id[clause_id] = text
    return by_id


def _licensing_clause_id_for_quantity(
    *,
    quantity: str,
    consequence: str,
    clause_by_id: dict[str, str],
) -> str:
    """Resolve the source clause that must license a quantity span on an effect.

    Completeness requires the span; compile_grounded_quantities keeps it only
    when provenance (or a uniquely assigned party) licenses it. Prefer the
    clause that states the quantity-bearing consequence; fall back to the
    unique clause containing the span.
    """
    if not clause_by_id:
        return ""
    qty = str(quantity or "").casefold().strip()
    cons = " ".join(str(consequence or "").split()).casefold()
    if cons:
        ranked: list[tuple[int, int, str]] = []
        for clause_id, text in clause_by_id.items():
            folded = text.casefold()
            if not (
                folded == cons
                or cons in folded
                or folded in cons
            ):
                continue
            score = 0
            if folded == cons:
                score += 3
            if qty and qty in folded:
                score += 2
            ranked.append((score, -len(text), clause_id))
        if ranked:
            ranked.sort(reverse=True)
            return ranked[0][2]
    if not qty:
        return ""
    hits = [
        clause_id for clause_id, text in clause_by_id.items()
        if qty in text.casefold()
    ]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return min(hits, key=lambda clause_id: len(clause_by_id[clause_id]))
    return ""


def _source_binding_outcome_suggestions(
    *,
    outcome: str,
    proposition: str,
    clause_spans: Sequence[str],
) -> list[dict[str, object]]:
    """Concrete outcome rewrites that reuse source wording.

    Lazy-imports world_state so this module stays import-safe under the
    world_state → world_validation dependency.
    """
    from .world_state import (
        WorldEffect,
        _source_proposition_supports_outcome,
        outcome_predicate_is_incomplete,
    )

    candidates: list[str] = []
    prop = " ".join(str(proposition or "").split())
    if prop:
        candidates.append(prop)
        for chunk in re.split(r"\s*(?:,|;|\bor\b)\s*", prop, flags=re.IGNORECASE):
            cleaned = " ".join(chunk.split()).strip(" .")
            if cleaned and cleaned.casefold() != prop.casefold():
                candidates.append(cleaned)
    for span in clause_spans:
        cleaned = " ".join(str(span or "").split()).strip(" .")
        if cleaned:
            candidates.append(cleaned)

    seen: set[str] = set()
    verified: list[dict[str, object]] = []
    for value in candidates:
        key = value.casefold()
        if key in seen or key == str(outcome or "").casefold():
            continue
        seen.add(key)
        if outcome_predicate_is_incomplete(value):
            continue
        probe = WorldEffect(
            "E_probe",
            "A0",
            "P0",
            value,
            "STATE_CHANGE",
            "NEUTRAL",
            "DIRECT",
            "CERTAIN",
            "INTERVENTION",
            provenance=(),
            source_proposition=prop or value,
            derivation_operation="DIRECT_COPY",
        )
        if not _source_proposition_supports_outcome(probe):
            continue
        verified.append({
            "op": "replace_outcome",
            "field": "outcome",
            "value": value,
            "why": (
                "Outcome wording must share event identity with "
                "source_proposition; this rewrite reuses source text."
            ),
        })
    # Prefer short local rewrites (e.g. "refrain") over whole-clause copies,
    # especially when the rejected outcome is a negation paraphrase.
    verified.sort(key=lambda row: len(str(row.get("value") or "")))
    return verified[:3]


def _incomplete_outcome_suggestions(
    *,
    outcome: str,
    proposition: str,
    clause_spans: Sequence[str],
) -> list[dict[str, object]]:
    """Complete rewrites for dangling-copula / dangling-preposition fragments."""
    from .world_state import outcome_predicate_is_incomplete

    fragment = " ".join(str(outcome or "").split()).strip(" ,.;:")
    head = re.sub(
        r"\b(?:is|are|was|were|be|been|being|"
        r"to|from|of|for|against|onto|into|over|under|before|after)\s*$",
        "",
        fragment,
        flags=re.IGNORECASE,
    ).strip(" ,.;:")
    head_tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z]+", head)
        if len(token) > 2
    }
    candidates: list[str] = []
    prop = " ".join(str(proposition or "").split())
    if prop:
        candidates.append(prop)
        for chunk in re.split(r"\s*(?:,|;|\bor\b)\s*", prop, flags=re.IGNORECASE):
            cleaned = " ".join(chunk.split()).strip(" .")
            if cleaned:
                candidates.append(cleaned)
    for span in clause_spans:
        cleaned = " ".join(str(span or "").split()).strip(" .")
        if cleaned:
            candidates.append(cleaned)
    if head:
        # Prefer a finished passive only when the head is already participial.
        folded = head.casefold()
        if folded.endswith(("ed", "en", "ing")):
            candidates.insert(0, f"is {head}")
        candidates.insert(0, f"{head} executed")
        candidates.insert(0, f"{head} completed")

    seen: set[str] = set()
    verified: list[dict[str, object]] = []
    for value in candidates:
        cleaned = " ".join(str(value or "").split()).strip(" .")
        key = cleaned.casefold()
        if not cleaned or key in seen or key == fragment.casefold():
            continue
        seen.add(key)
        if outcome_predicate_is_incomplete(cleaned):
            continue
        if head_tokens:
            value_tokens = {
                token.casefold()
                for token in re.findall(r"[A-Za-z]+", cleaned)
            }
            if not (head_tokens & value_tokens):
                continue
        verified.append({
            "op": "replace_outcome",
            "field": "outcome",
            "value": cleaned,
            "why": (
                "Replace the truncated predicate with a finished state or "
                "event that keeps the fragment's event head."
            ),
        })
    verified.sort(key=lambda row: len(str(row.get("value") or "")))
    return verified[:3]


def repair_guidance_cards(
    issues: Sequence[ValidationIssue | dict[str, object]],
    candidate: object | None = None,
    *,
    clauses: Sequence[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Concrete per-issue repair cards for model feedback.

    Typed codes alone are not enough: SOURCE_PROPOSITION_BINDING needs the
    current outcome / source_proposition pair, available clause spans, and
    bind-or-omit patches the model can apply without guessing.
    """
    world = (
        candidate.get("world_model")
        if isinstance(candidate, dict) else None
    )
    effects_by_id: dict[str, dict[str, object]] = {}
    actions_by_id: dict[str, dict[str, object]] = {}
    parties_by_id: dict[str, dict[str, object]] = {}
    causal_links: list[dict[str, object]] = []
    if isinstance(world, dict):
        for party in world.get("parties") or []:
            if isinstance(party, dict) and party.get("party_id"):
                parties_by_id[str(party["party_id"])] = party
        for action in world.get("actions") or []:
            if isinstance(action, dict) and action.get("action_id"):
                actions_by_id[str(action["action_id"])] = action
        for effect in world.get("effects") or []:
            if isinstance(effect, dict) and effect.get("effect_id"):
                effects_by_id[str(effect["effect_id"])] = effect
        causal_links = [
            link for link in (world.get("causal_links") or [])
            if isinstance(link, dict)
        ]
    clause_by_id = _clause_text_by_id(clauses)
    cards: list[dict[str, object]] = []
    for row in _issue_rows(issues):
        code = str(row.get("code") or "")
        entity_id = str(row.get("entity_id") or "")
        card: dict[str, object] = {
            "code": code,
            "card_id": f"{code or 'WORLD_VALIDATION_ERROR'}_V1",
            "card_version": 1,
            "entity_id": entity_id,
            "entity_kind": str(row.get("entity_kind") or ""),
            "field": str(row.get("field") or ""),
            "repair_class": str(row.get("repair_class") or ""),
            "permits_removal": bool(row.get("permits_removal")),
            "message": str(row.get("message") or ""),
            "repair_owner": repair_owner_for_code(code),
        }
        if code == "GLOBAL_CONSTRAINT_AS_PROCESS":
            patch = {
                "op": "replace_constraint_process_with_allocation_state",
                "effect_id": entity_id,
                "field": "causal_links",
                "why": (
                    "Keep the indivisibility/scarcity statement as a global fact; "
                    "replace it in the action chain with a source-licensed receipt, "
                    "nonreceipt, allocation, or deprivation state on the affected branch."
                ),
            }
            card.update({
                "allowed_operations": [
                    "replace_constraint_process_with_allocation_state",
                    "add_effect", "replace_effect", "replace_causal_link",
                ],
                "concrete_patches": [patch],
                "fix_examples": [patch["why"]],
            })
        elif code == "DERIVATION_CONTRACT_OPERATION_MISMATCH":
            effect = effects_by_id.get(entity_id, {})
            scarcity_candidates: list[tuple[int, str, str]] = []
            for clause_id, span in clause_by_id.items():
                folded = span.casefold()
                if re.search(
                    r"\b(?:can be divided|divisible|several|multiple|two|three|"
                    r"additional|another (?:dose|seat|organ|tank|scholarship|"
                    r"source)|more than one|not exhaustive)\b",
                    folded,
                ):
                    continue
                score = sum(
                    weight for pattern, weight in (
                        (r"\b(?:one|single|sole|only one|exactly one)\b", 4),
                        (r"\b(?:cannot|can't|can not)\s+be\s+divid", 4),
                        (r"\b(?:either|or|but not both|one recipient)\b", 2),
                        (r"\b(?:dose|seat|organ|tank|scholarship|resource|unit)\b", 1),
                    )
                    if re.search(pattern, folded)
                )
                if score:
                    scarcity_candidates.append((score, clause_id, span))
            scarcity_candidates.sort(key=lambda row: (-row[0], len(row[2]), row[1]))
            scarcity_clause_id = scarcity_candidates[0][1] if scarcity_candidates else ""
            scarcity_span = scarcity_candidates[0][2] if scarcity_candidates else ""
            quantities: list[str] = []
            if scarcity_span:
                from .world_state import explicit_quantity_spans
                quantities = list(explicit_quantity_spans(scarcity_span))
                if not quantities:
                    lexical_quantity = re.search(
                        r"\b(exactly\s+one|only\s+one|one|single|sole)\b",
                        scarcity_span,
                        re.IGNORECASE,
                    )
                    if lexical_quantity:
                        quantities = [
                            "one" if "one" in lexical_quantity.group(1).casefold()
                            else lexical_quantity.group(1).casefold()
                        ]
            patch = {
                "op": (
                    "repair_derivation_contract_metadata" if scarcity_span
                    else "quarantine_unlicensed_allocation_complement"
                ),
                "effect_id": entity_id,
                "preserve": [
                    "effect_id", "action_id", "party_id", "outcome",
                    "directness", "polarity", "modality", "effect_kind",
                    "source_effect_ids", "causal_links",
                ],
                "set": ({
                    "derivation_operation": "EXCLUSIVE_ALLOCATION_COMPLEMENT",
                    "source_proposition": scarcity_span,
                    "clause_ids": [scarcity_clause_id],
                    "quantities": quantities,
                } if scarcity_span else {}),
                "why": ((
                    "The nonreceipt proposition and its topology already exist. "
                    "Repair only the fields that justify the derived node; do not "
                    "delete, recreate, rename, or reconnect any effect."
                ) if scarcity_span else (
                    "No exact source span licenses an exclusive complement. Do not "
                    "relabel or reconstruct the graph; quarantine this derived node "
                    "pending source clarification."
                )),
            }
            card.update({
                "current_derivation_operation": str(
                    effect.get("derivation_operation") or ""
                ),
                "allowed_operations": [patch["op"]],
                "concrete_patches": [patch],
                "fix_examples": [patch["why"]],
                "permits_removal": False,
            })
        elif code == "EXCLUSIVE_ALLOCATION_BRANCH_MISSING":
            message = str(row.get("message") or "")
            match = re.search(
                r"\b(A\d+)\b.*?nonrecipient\s+([A-Za-z0-9_]+)",
                message,
                re.IGNORECASE,
            )
            action_id = match.group(1) if match else entity_id
            party_id = match.group(2) if match else ""
            action = actions_by_id.get(action_id, {})
            direct_transfer_ids = [
                effect_id for effect_id, effect in effects_by_id.items()
                if str(effect.get("action_id") or "") == action_id
                and str(effect.get("effect_kind") or "").upper()
                == "RESOURCE_TRANSFER"
                and str(effect.get("directness") or "").upper() == "DIRECT"
            ]
            patch = {
                "op": "restore_exclusive_allocation_branch",
                "action_id": action_id,
                "nonrecipient_party_id": party_id,
                "nonrecipient_label": str(
                    parties_by_id.get(party_id, {}).get("label") or ""
                ),
                "allocation_parent_effect_ids": direct_transfer_ids,
                "required_effect_sequence": [
                    {
                        "outcome": (
                            f"{str(parties_by_id.get(party_id, {}).get('label') or party_id)} "
                            "does not receive the allocated resource"
                            if party_id else
                            "the nonrecipient does not receive the allocated resource"
                        ),
                        "directness": "DOWNSTREAM",
                        "polarity": "ADVERSE",
                        "effect_kind": "OTHER",
                        "derivation_operation": "EXCLUSIVE_ALLOCATION_COMPLEMENT",
                        "source_effect_ids": direct_transfer_ids,
                        "source_proposition": (
                            "<copy the source span establishing exclusivity, "
                            "indivisibility, or one-recipient capacity>"
                        ),
                    },
                    {
                        "outcome": "<copy the source-stipulated consequence>",
                        "directness": "DOWNSTREAM",
                        "polarity": "ADVERSE",
                        "effect_kind": "HEALTH_OUTCOME",
                        "derivation_operation": "SOURCE_STIPULATED_CAUSAL",
                        "source_effect_ids": ["<new nonreceipt effect id>"],
                        "source_proposition": (
                            "<copy the source span stating the consequence of "
                            "nonreceipt>"
                        ),
                    },
                ],
                "required_links": [
                    {
                        "source_id": direct_transfer_ids[0]
                        if direct_transfer_ids else "<direct transfer effect id>",
                        "relation": "CAUSES",
                        "target_id": "<new nonreceipt effect id>",
                        "derivation_operation": (
                            "EXCLUSIVE_ALLOCATION_COMPLEMENT"
                        ),
                    },
                    {
                        "source_id": "<new nonreceipt effect id>",
                        "relation": "CAUSES",
                        "target_id": "<source-stipulated consequence effect id>",
                    },
                ],
                "source_spans": list(clause_by_id.values()),
                "why": (
                    "A complete exclusive allocation must include the unselected "
                    "eligible recipient's nonreceipt branch and every consequence "
                    "the source stipulates for going without the resource. Link "
                    "transfer -> nonreceipt -> consequence; do not use the global "
                    "scarcity statement as the causal parent."
                ),
            }
            card.update({
                "action_id": action_id,
                "nonrecipient_party_id": party_id,
                "allowed_operations": [
                    "restore_exclusive_allocation_branch", "add_effect",
                    "add_causal_link",
                ],
                "concrete_patches": [patch],
                "fix_examples": [patch["why"]],
            })
        elif code in {
            "ELLIPSIS_RESOLUTION_REQUIRED", "SLUICING_MUST_REMAIN_UNRESOLVED",
            "ELLIPSIS_ANTECEDENT_AMBIGUOUS", "ELLIPSIS_CONTEXT_OVERREACH",
            "ELLIPSIS_TEMPLATE_MISMATCH",
        }:
            message = str(row.get("message") or "")
            match = re.search(r"\b(EL\d+)\s+([A-Z_]+)", message)
            obligation_id = match.group(1) if match else entity_id
            ellipsis_kind = match.group(2) if match else "ELLIPSIS"
            unresolved = code in {
                "SLUICING_MUST_REMAIN_UNRESOLVED",
                "ELLIPSIS_ANTECEDENT_AMBIGUOUS",
            }
            operation = {
                "SLUICING_MUST_REMAIN_UNRESOLVED": "preserve_ellipsis_unresolved",
                "ELLIPSIS_ANTECEDENT_AMBIGUOUS": "quarantine_ambiguous_ellipsis",
                "ELLIPSIS_CONTEXT_OVERREACH": "restore_source_licensed_antecedent",
                "ELLIPSIS_TEMPLATE_MISMATCH": "copy_ellipsis_template_fields",
            }.get(code, "add_ellipsis_resolution")
            patch = {
                "op": operation,
                "obligation_id": obligation_id,
                "ellipsis_kind": ellipsis_kind,
                "field": "ellipsis_resolutions",
                "status": "UNRESOLVED" if unresolved else "RESOLVED",
                "requirements": (
                    "Set reconstructed_span empty, retain the unresolved source "
                    "span, and do not admit effects that depend on guessing it."
                    if unresolved else
                    "Quote an exact antecedent_span, reconstruct only material "
                    "copied from compatible source clauses, copy the obligation's "
                    "template fields, and use ethical context only to rank candidates."
                ),
                "why": message,
            }
            card.update({
                "obligation_id": obligation_id,
                "ellipsis_kind": ellipsis_kind,
                "allowed_operations": [patch["op"]],
                "concrete_patches": [patch],
                "fix_examples": [patch["requirements"]],
            })
        elif code == "LIKELIHOOD_BINDING_MISSING":
            message = str(row.get("message") or "")
            match = re.search(
                r"(?P<actions>A\d+(?:,A\d+)*) omits source-bound likelihood "
                r"'(?P<qualifier>[^']+)' for '(?P<subject>[^']+)' from "
                r"(?P<clause>[A-Za-z0-9_]+)",
                message,
                re.IGNORECASE,
            )
            action_ids = match.group("actions").split(",") if match else []
            qualifier = match.group("qualifier") if match else ""
            subject = match.group("subject") if match else ""
            clause_id = match.group("clause") if match else ""
            patch = {
                "op": "restore_likelihood_binding",
                "action_ids": action_ids,
                "subject": subject,
                "qualifier": qualifier,
                "clause_id": clause_id,
                "source_text": clause_by_id.get(clause_id, ""),
                "field": "effects",
                "why": (
                    "Add or repair the effect for the named subject on its owning "
                    "action, preserving the exact source likelihood and proposition."
                ),
            }
            card.update({
                "action_ids": action_ids,
                "allowed_operations": [
                    "restore_likelihood_binding", "add_effect", "replace_effect",
                ],
                "concrete_patches": [patch],
                "fix_examples": [patch["why"]],
            })
        elif code == "STAGE_SKELETON_DRIFT":
            message = str(row.get("message") or "")
            action_match = re.search(r"\b(A\d+)\b", message)
            action_id = action_match.group(1) if action_match else ""
            patch = {
                "op": "restore_stage_skeleton_proposition",
                "action_id": action_id,
                "skeleton_proposition_id": entity_id,
                "field": "effects",
                "requirements": message,
                "why": (
                    "Restore the admitted Stage 1 factual proposition in Stage 2 "
                    "before changing or adding causal and counterfactual topology."
                ),
            }
            card.update({
                "action_id": action_id,
                "allowed_operations": [
                    "restore_stage_skeleton_proposition", "add_effect", "replace_effect",
                ],
                "concrete_patches": [patch],
                "fix_examples": [patch["why"]],
            })
        elif code == "ACTION_DIRECT_PROVENANCE_MISSING" and entity_id in actions_by_id:
            message = str(row.get("message") or "")
            list_match = re.search(
                r"omits source clauses of its DIRECT effects:\s*\[([^]]*)\]",
                message,
                re.IGNORECASE,
            )
            clause_ids: list[str] = []
            if list_match:
                try:
                    parsed = ast.literal_eval(f"[{list_match.group(1)}]")
                except (SyntaxError, ValueError):
                    parsed = []
                if isinstance(parsed, (list, tuple)):
                    clause_ids = [
                        str(item).strip() for item in parsed if str(item).strip()
                    ]
            concrete_patches = [{
                "op": "add_action_provenance",
                "action_id": entity_id,
                "field": "clause_ids",
                "value": clause_id,
                "patch_kind": "licensing_patch",
                "why": (
                    "Synchronize action provenance with a supporting source "
                    "clause already cited by its own DIRECT effect."
                ),
            } for clause_id in clause_ids]
            card.update({
                "action_id": entity_id,
                "missing_clause_ids": clause_ids,
                "allowed_operations": ["add_action_provenance"],
                "concrete_patches": concrete_patches,
                "fix_examples": [patch["why"] for patch in concrete_patches],
            })
        elif code == "MISSING_PROCESS_INTERMEDIATE" and entity_id in effects_by_id:
            child = effects_by_id[entity_id]
            action_id = str(child.get("action_id") or "")
            incoming = [
                link for link in causal_links
                if str(link.get("target_id") or "") == entity_id
                and str(link.get("action_id") or action_id) == action_id
            ]
            parent_ids = list(dict.fromkeys(
                str(link.get("source_id") or "") for link in incoming
                if str(link.get("source_id") or "")
            ))
            message = str(row.get("message") or "")
            named_party_ids = [
                party_id for party_id in re.findall(
                    r"\b(P[A-Za-z0-9_]+)\s*\(", message,
                )
                if party_id in parties_by_id
                and party_id != str(child.get("party_id") or "")
            ]
            cited = [
                str(item) for item in (child.get("clause_ids") or [])
                if str(item).strip()
            ]
            available_spans = [
                clause_by_id[cid] for cid in cited if cid in clause_by_id
            ]
            candidates = named_party_ids or [""]
            concrete_patches = []
            for party_id in candidates:
                party = parties_by_id.get(party_id, {})
                process_id = f"{action_id}_PROCESS_FOR_{entity_id}"
                concrete_patches.append({
                    "op": "insert_process_intermediate",
                    "action_id": action_id,
                    "child_effect_id": entity_id,
                    "replace_parent_effect_ids": parent_ids,
                    "new_effect_id": process_id,
                    "value": {
                        "party_id": party_id or "<source-named process party>",
                        "party_label": str(party.get("label") or ""),
                        "outcome": "<copy the source-stated process event>",
                        "polarity": "NEUTRAL",
                        "directness": "DOWNSTREAM",
                        "effect_kind": "PHYSICAL_STATE",
                        "modality": str(child.get("modality") or "CERTAIN"),
                        "source_proposition": "<copy the licensing source span>",
                        "clause_ids": cited,
                    },
                    "remove_links": [
                        {"source_id": parent_id, "target_id": entity_id}
                        for parent_id in parent_ids
                    ],
                    "add_links": [
                        {"source_id": parent_id, "target_id": process_id}
                        for parent_id in parent_ids
                    ] + [{"source_id": process_id, "target_id": entity_id}],
                    "why": (
                        "Insert the source-named physical process between the "
                        "direct intervention and the preserved bodily outcome."
                    ),
                })
            card.update({
                "action_id": action_id,
                "child_effect_id": entity_id,
                "current_parent_effect_ids": parent_ids,
                "candidate_process_party_ids": named_party_ids,
                "available_clause_spans": available_spans,
                "allowed_operations": ["insert_process_intermediate"],
                "concrete_patches": concrete_patches,
                "fix_examples": [patch["why"] for patch in concrete_patches],
            })
        elif code == "SOURCE_PROPOSITION_BINDING" and entity_id in effects_by_id:
            effect = effects_by_id[entity_id]
            outcome = str(effect.get("outcome") or "")
            proposition = str(effect.get("source_proposition") or "")
            action_id = str(effect.get("action_id") or "")
            cited = [
                str(item) for item in (effect.get("clause_ids") or [])
                if str(item).strip()
            ]
            available_spans = [
                clause_by_id[cid] for cid in cited if cid in clause_by_id
            ]
            concrete_patches: list[dict[str, object]] = list(
                _source_binding_outcome_suggestions(
                    outcome=outcome,
                    proposition=proposition,
                    clause_spans=available_spans,
                )
            )
            if available_spans and outcome:
                # Prefer an exact cited-clause span that already contains the
                # outcome wording when the current source_proposition does not.
                outcome_fold = outcome.casefold()
                for span in available_spans:
                    if outcome_fold in span.casefold() and (
                        span.casefold() != proposition.casefold()
                    ):
                        concrete_patches.append({
                            "op": "replace_source_proposition",
                            "field": "source_proposition",
                            "value": span,
                            "why": (
                                "Bind source_proposition to the cited clause "
                                "span that already states the outcome."
                            ),
                        })
                        break
            if row.get("permits_removal"):
                concrete_patches.append({
                    "op": "remove_effect",
                    "remove_effect_id": entity_id,
                    "also_update": [
                        f"{action_id}.effect_ids" if action_id else "action.effect_ids",
                        "causal_links",
                        "counterfactual_links",
                    ],
                    "why": (
                        "Omit this derived world effect when no cited clause "
                        "states that outcome."
                    ),
                })
            card.update({
                "action_id": action_id,
                "current_outcome": outcome,
                "current_source_proposition": proposition,
                "cited_clause_ids": cited,
                "available_clause_spans": available_spans,
                "allowed_operations": (
                    ["replace_outcome", "replace_source_proposition", "remove_effect"]
                    if row.get("permits_removal")
                    else ["replace_outcome", "replace_source_proposition"]
                ),
                "concrete_patches": concrete_patches,
                "fix_examples": [
                    patch.get("why") or patch.get("op")
                    for patch in concrete_patches
                ],
            })
        elif code == "OUTCOME_PREDICATE_INCOMPLETE" and entity_id in effects_by_id:
            effect = effects_by_id[entity_id]
            outcome = str(effect.get("outcome") or "")
            proposition = str(effect.get("source_proposition") or "")
            action_id = str(effect.get("action_id") or "")
            cited = [
                str(item) for item in (effect.get("clause_ids") or [])
                if str(item).strip()
            ]
            available_spans = [
                clause_by_id[cid] for cid in cited if cid in clause_by_id
            ]
            concrete_patches: list[dict[str, object]] = list(
                _incomplete_outcome_suggestions(
                    outcome=outcome,
                    proposition=proposition,
                    clause_spans=available_spans,
                )
            )
            if str(effect.get("effect_kind") or "").upper() == "RESOURCE_TRANSFER":
                transfer_candidates: list[str] = []
                for span in [proposition, *available_spans]:
                    for match in re.finditer(
                        r"\b(?:give|gives|gave|given|giving|receive|receives|"
                        r"received|receiving|allocate|allocates|allocated|"
                        r"assign|assigned|send|sent|deliver|delivered|transfer|"
                        r"transferred|administer|administered|provide|provided)\b"
                        r"[^,;.?]{0,120}",
                        str(span or ""),
                        re.IGNORECASE,
                    ):
                        candidate = " ".join(match.group(0).split()).strip(" ,.;:")
                        if candidate:
                            transfer_candidates.append(candidate)
                existing_values = {
                    str(patch.get("value") or "").casefold()
                    for patch in concrete_patches
                }
                transfer_patches = [{
                    "op": "replace_outcome",
                    "field": "outcome",
                    "value": candidate,
                    "why": (
                        "RESOURCE_TRANSFER must state the source-licensed transfer "
                        "event, not merely name the transferred object."
                    ),
                } for candidate in dict.fromkeys(transfer_candidates)
                  if candidate.casefold() not in existing_values]
                concrete_patches = [*transfer_patches[:3], *concrete_patches]
            if row.get("permits_removal"):
                concrete_patches.append({
                    "op": "remove_effect",
                    "remove_effect_id": entity_id,
                    "also_update": [
                        f"{action_id}.effect_ids" if action_id else "action.effect_ids",
                        "causal_links",
                        "counterfactual_links",
                    ],
                    "why": (
                        "Omit this fragment when no finished predicate can be "
                        "sourced for the event."
                    ),
                })
            card.update({
                "action_id": action_id,
                "current_outcome": outcome,
                "current_source_proposition": proposition,
                "cited_clause_ids": cited,
                "available_clause_spans": available_spans,
                "allowed_operations": (
                    ["replace_outcome", "remove_effect"]
                    if row.get("permits_removal")
                    else ["replace_outcome"]
                ),
                "concrete_patches": concrete_patches,
                "fix_examples": [
                    patch.get("why") or patch.get("op")
                    for patch in concrete_patches
                ],
            })
        elif code == "OUTCOME_POLARITY_CONTRADICTION" and entity_id in effects_by_id:
            effect = effects_by_id[entity_id]
            outcome = str(effect.get("outcome") or "")
            proposition = str(effect.get("source_proposition") or "")
            card.update({
                "action_id": str(effect.get("action_id") or ""),
                "current_outcome": outcome,
                "current_polarity": str(effect.get("polarity") or ""),
                "current_source_proposition": proposition,
                "allowed_operations": ["replace_outcome", "replace_polarity", "remove_effect"],
                "concrete_patches": [],
                "fix_examples": [
                    "Preserve the source event's explicit negation and assign the matching polarity.",
                    "If the source states the opposite branch's event, remove this effect instead of relabeling it.",
                ],
            })
        elif code == "SOURCE_STIPULATED_OUTCOME_MISSING":
            message = str(row.get("message") or "")
            match = re.search(
                r"source-stipulated outcome '([^']+)'", message, re.IGNORECASE,
            )
            consequence = match.group(1) if match else ""
            polarity = "ADVERSE" if re.search(
                r"\b(?:loss\s+of\s+lif|death|die|kill|catastroph)",
                consequence,
                re.IGNORECASE,
            ) else "BENEFICIAL"
            action_id = entity_id if re.fullmatch(r"A\d+", entity_id) else ""
            available_spans = [
                text for text in clause_by_id.values() if text
            ]
            concrete_patches: list[dict[str, object]] = []
            if consequence:
                concrete_patches.append({
                    "op": "add_effect",
                    "action_id": action_id,
                    "field": "effects",
                    "value": {
                        "outcome": consequence,
                        "polarity": polarity,
                        "effect_kind": "HEALTH_OUTCOME",
                        "directness": "DOWNSTREAM",
                        "modality": "CERTAIN",
                        "source_proposition": consequence,
                    },
                    "why": (
                        "Admit the binary-contrast consequence as a downstream "
                        "effect on this action and bind it to a source-licensed "
                        "parent, or mark polarity UNRESOLVED to quarantine it "
                        "explicitly."
                    ),
                })
                concrete_patches.append({
                    "op": "add_effect",
                    "action_id": action_id,
                    "field": "effects",
                    "value": {
                        "outcome": consequence,
                        "polarity": "UNRESOLVED",
                        "effect_kind": "HEALTH_OUTCOME",
                        "directness": "DOWNSTREAM",
                        "modality": "CERTAIN",
                        "source_proposition": consequence,
                    },
                    "why": (
                        "Explicit UNRESOLVED quarantine when the stake is "
                        "acknowledged but not yet settled."
                    ),
                })
            card.update({
                "action_id": action_id,
                "stipulated_outcome": consequence,
                "available_clause_spans": available_spans[:4],
                "allowed_operations": ["add_effect"],
                "concrete_patches": concrete_patches,
                "fix_examples": [
                    patch.get("why") or patch.get("op")
                    for patch in concrete_patches
                ],
            })
        elif code == "EFFECT_QUANTITY_LEAK":
            message = str(row.get("message") or "")
            quantity_match = re.search(
                r"quantity\s+(['\"])(.*?)\1\s+is not bound",
                message,
                re.IGNORECASE,
            )
            quantity = quantity_match.group(2).strip() if quantity_match else ""
            concrete_patches = []
            if entity_id in effects_by_id and quantity:
                concrete_patches.append({
                    "op": "remove_quantity",
                    "effect_id": entity_id,
                    "field": "quantities",
                    "value": quantity,
                    "patch_kind": "semantic_patch",
                    "why": (
                        "Remove the quantity that belongs only to another "
                        "effect or population in shared provenance."
                    ),
                })
            card.update({
                "entity_id": entity_id,
                "leaked_quantity": quantity,
                "allowed_operations": ["remove_quantity"],
                "concrete_patches": concrete_patches,
                "fix_examples": [
                    patch.get("why") or patch.get("op")
                    for patch in concrete_patches
                ],
            })
        elif code == "EFFECT_QUANTITY_MISSING":
            message = str(row.get("message") or "")
            list_match = re.search(
                r"omits quantities stated in its outcome and provenance:\s*\[([^]]*)\]",
                message,
                re.IGNORECASE,
            )
            quantities = []
            if list_match:
                payload = f"[{list_match.group(1)}]"
                try:
                    parsed = ast.literal_eval(payload)
                except (SyntaxError, ValueError):
                    parsed = []
                if isinstance(parsed, (list, tuple)):
                    quantities = [
                        str(item).strip() for item in parsed if str(item).strip()
                    ]
            concrete_patches = [
                {
                    "op": "add_quantity",
                    "effect_id": entity_id,
                    "field": "quantities",
                    "value": quantity,
                    "patch_kind": "semantic_patch",
                    "why": (
                        "Synchronize the typed quantity field with the quantity "
                        "already stated by this effect and its provenance."
                    ),
                }
                for quantity in quantities
                if entity_id in effects_by_id
            ]
            card.update({
                "entity_id": entity_id,
                "missing_quantities": quantities,
                "allowed_operations": ["add_quantity"],
                "concrete_patches": concrete_patches,
                "fix_examples": [
                    patch.get("why") or patch.get("op")
                    for patch in concrete_patches
                ],
            })
        elif code == "QUANTITY_BEARING_CONSEQUENCE_MISSING":
            message = str(row.get("message") or "")
            qty_match = re.search(
                r"source quantity '([^']+)'", message, re.IGNORECASE,
            )
            cons_match = re.search(
                r"quantity-bearing consequence '([^']+)'|"
                r'quantity-bearing consequence "([^"]+)"',
                message,
                re.IGNORECASE,
            )
            quantity = qty_match.group(1) if qty_match else ""
            consequence = ""
            if cons_match:
                consequence = cons_match.group(1) or cons_match.group(2) or ""
            # Prefer membership in the candidate effect index so action-scoped
            # ids like A1_e2 still get quantity DET, not a blind add_effect.
            effect_id = entity_id if entity_id in effects_by_id else (
                entity_id if entity_id.startswith("E") else ""
            )
            action_id = entity_id if re.fullmatch(r"A\d+", entity_id) else ""
            if effect_id and effect_id in effects_by_id:
                action_id = str(effects_by_id[effect_id].get("action_id") or action_id)
            # Prefer detector-carried licensing clause; text search is fallback only.
            licensing_clause_id = ""
            lic_match = re.search(
                r"licensing_clause_id=([A-Za-z0-9_]+)", message,
            )
            if lic_match:
                licensing_clause_id = lic_match.group(1)
            else:
                for related in row.get("related_ids") or ():
                    related_id = str(related or "").strip()
                    if re.fullmatch(r"C\d+(?:_[A-Za-z0-9]+)?", related_id):
                        licensing_clause_id = related_id
                        break
            if not licensing_clause_id:
                licensing_clause_id = _licensing_clause_id_for_quantity(
                    quantity=quantity,
                    consequence=consequence,
                    clause_by_id=clause_by_id,
                )
            concrete_patches: list[dict[str, object]] = []
            if quantity and effect_id:
                concrete_patches.append({
                    "op": "add_quantity",
                    "effect_id": effect_id,
                    "field": "quantities",
                    "value": quantity,
                    "patch_kind": "semantic_patch",
                    "why": (
                        "Copy the source quantity span onto the matched "
                        "consequence effect (or its population party)."
                    ),
                })
                if licensing_clause_id:
                    concrete_patches.append({
                        "op": "add_provenance",
                        "effect_id": effect_id,
                        "field": "clause_ids",
                        "value": licensing_clause_id,
                        "patch_kind": "licensing_patch",
                        "why": (
                            "Add only the detector-named provenance edge that "
                            "licenses this quantity under compile; do not cite "
                            "sibling clauses."
                        ),
                    })
            elif quantity and action_id:
                value: dict[str, object] = {
                    "outcome": consequence or quantity,
                    "polarity": "ADVERSE",
                    "effect_kind": "HEALTH_OUTCOME",
                    "directness": "DIRECT",
                    "modality": "CERTAIN",
                    "quantities": [quantity],
                    "source_proposition": consequence or quantity,
                }
                if licensing_clause_id:
                    value["clause_ids"] = [licensing_clause_id]
                concrete_patches.append({
                    "op": "add_effect",
                    "action_id": action_id,
                    "field": "effects",
                    "value": value,
                    "patch_kind": "semantic_patch",
                    "why": (
                        "Admit the quantity-bearing consequence and record "
                        "the source quantity span on that effect."
                    ),
                })
            allowed_ops: list[str] = []
            if effect_id:
                allowed_ops.append("add_quantity")
                if licensing_clause_id:
                    allowed_ops.append("add_provenance")
            else:
                allowed_ops.append("add_effect")
            card.update({
                "action_id": action_id,
                "entity_id": entity_id,
                "missing_quantity": quantity,
                "stipulated_outcome": consequence,
                "licensing_clause_id": licensing_clause_id,
                "allowed_operations": allowed_ops,
                "concrete_patches": concrete_patches,
                "fix_examples": [
                    patch.get("why") or patch.get("op")
                    for patch in concrete_patches
                ],
            })
        elif code in {
            "LIKELIHOOD_QUALIFIER_MISSING",
            "TEMPORAL_QUALIFIER_MISSING",
            "SCOPE_QUALIFIER_MISSING",
        } and entity_id in effects_by_id:
            message = str(row.get("message") or "")
            field = {
                "LIKELIHOOD_QUALIFIER_MISSING": "likelihood_qualifiers",
                "TEMPORAL_QUALIFIER_MISSING": "temporal_qualifiers",
                "SCOPE_QUALIFIER_MISSING": "scope_qualifiers",
            }[code]
            op = {
                "LIKELIHOOD_QUALIFIER_MISSING": "add_likelihood_qualifier",
                "TEMPORAL_QUALIFIER_MISSING": "add_temporal_qualifier",
                "SCOPE_QUALIFIER_MISSING": "add_scope_qualifier",
            }[code]
            missing_match = re.search(
                r"qualifiers:\s*\[([^\]]*)\]", message, re.IGNORECASE,
            )
            missing: list[str] = []
            if missing_match:
                missing = [
                    item.strip().strip("'\"")
                    for item in missing_match.group(1).split(",")
                    if item.strip().strip("'\"")
                ]
            concrete_patches = []
            for span in missing[:3]:
                concrete_patches.append({
                    "op": op,
                    "effect_id": entity_id,
                    "field": field,
                    "value": span,
                    "why": (
                        f"Copy the source {field.replace('_', ' ')} span "
                        "onto the effect whose outcome it modifies."
                    ),
                })
            effect = effects_by_id[entity_id]
            card.update({
                "action_id": str(effect.get("action_id") or ""),
                "missing_qualifiers": missing,
                "allowed_operations": [op],
                "concrete_patches": concrete_patches,
                "fix_examples": [
                    patch.get("why") or patch.get("op")
                    for patch in concrete_patches
                ],
            })
        elif code == "VERB_LEMMA_MISMATCH" and entity_id in effects_by_id:
            effect = effects_by_id[entity_id]
            outcome = str(effect.get("outcome") or "")
            proposition = str(effect.get("source_proposition") or "")
            action_id = str(effect.get("action_id") or "")
            cited = [
                str(item) for item in (effect.get("clause_ids") or [])
                if str(item).strip()
            ]
            available_spans = [
                clause_by_id[cid] for cid in cited if cid in clause_by_id
            ]
            concrete_patches: list[dict[str, object]] = list(
                _source_binding_outcome_suggestions(
                    outcome=outcome,
                    proposition=proposition,
                    clause_spans=available_spans,
                )
            )
            if proposition and outcome:
                concrete_patches.insert(0, {
                    "op": "replace_outcome",
                    "effect_id": entity_id,
                    "field": "outcome",
                    "value": proposition,
                    "why": (
                        "Rewrite the outcome to a morphological variant of "
                        "the bound source event lemma."
                    ),
                })
            if row.get("permits_removal"):
                concrete_patches.append({
                    "op": "remove_effect",
                    "remove_effect_id": entity_id,
                    "also_update": [
                        f"{action_id}.effect_ids" if action_id else "action.effect_ids",
                        "causal_links",
                        "counterfactual_links",
                    ],
                    "why": (
                        "Omit this derived world effect when no lemma-compatible "
                        "outcome can be sourced."
                    ),
                })
            card.update({
                "action_id": action_id,
                "current_outcome": outcome,
                "current_source_proposition": proposition,
                "cited_clause_ids": cited,
                "available_clause_spans": available_spans,
                "allowed_operations": (
                    ["replace_outcome", "replace_source_proposition", "remove_effect"]
                    if row.get("permits_removal")
                    else ["replace_outcome", "replace_source_proposition"]
                ),
                "concrete_patches": concrete_patches,
                "fix_examples": [
                    patch.get("why") or patch.get("op")
                    for patch in concrete_patches
                ],
            })
        elif code == "QUANTIFIER_PARTY_LEAK":
            message = str(row.get("message") or "")
            qty_match = re.search(
                r"(\w+) records quantity '([^']+)' that source binds to (\w+)",
                message,
                re.IGNORECASE,
            )
            leak_id = qty_match.group(1) if qty_match else (
                entity_id if entity_id.startswith("P") else ""
            )
            quantity = qty_match.group(2) if qty_match else ""
            owner_id = qty_match.group(3) if qty_match else ""
            concrete_patches: list[dict[str, object]] = []
            if quantity and owner_id:
                concrete_patches.append({
                    "op": "move_quantity",
                    "from_party_id": leak_id,
                    "to_party_id": owner_id,
                    "field": "quantities",
                    "value": quantity,
                    "why": (
                        "Move the source quantity span onto the uniquely "
                        "owning population party."
                    ),
                })
            if quantity and leak_id:
                concrete_patches.append({
                    "op": "clear_quantity",
                    "party_id": leak_id,
                    "field": "quantities",
                    "value": quantity,
                    "why": (
                        "Clear the leaked span from the non-owner party "
                        "(owner must still record it)."
                    ),
                })
            card.update({
                "party_id": leak_id,
                "owner_party_id": owner_id,
                "leaked_quantity": quantity,
                "allowed_operations": ["move_quantity", "clear_quantity"],
                "concrete_patches": concrete_patches,
                "fix_examples": [
                    patch.get("why") or patch.get("op")
                    for patch in concrete_patches
                ],
            })
        cards.append(card)
    return cards


def issue_repair_fingerprint(
    issue: ValidationIssue | dict[str, object],
) -> tuple[str, str, str]:
    """Stable (code, entity_id, field) key for unstable-repair detection."""
    row = issue.as_dict() if isinstance(issue, ValidationIssue) else dict(issue)
    return (
        str(row.get("code") or "").strip(),
        str(row.get("entity_id") or "").strip(),
        str(row.get("field") or "").strip(),
    )


def repair_no_effect_issues(
    before_issues: Sequence[ValidationIssue | dict[str, object]],
    after_issues: Sequence[ValidationIssue | dict[str, object]],
    applied_patches: Sequence[dict[str, object]] | None = None,
) -> tuple[ValidationIssue, ...]:
    """Return REPAIR_NO_EFFECT issues when a patch left the same target failing.

    Generic property: repair(op) -> validate -> same violation on same target
    means the repair was non-effective. Requires that patches were actually
    applied; an empty applied list is a different failure mode.
    """
    applied = [dict(row) for row in (applied_patches or ()) if isinstance(row, dict)]
    if not applied:
        return ()
    before_fps = {
        issue_repair_fingerprint(row)
        for row in before_issues
        if issue_repair_fingerprint(row)[0]
        and issue_repair_fingerprint(row)[0] != "REPAIR_NO_EFFECT"
    }
    after_fps = {
        issue_repair_fingerprint(row)
        for row in after_issues
        if issue_repair_fingerprint(row)[0]
        and issue_repair_fingerprint(row)[0] != "REPAIR_NO_EFFECT"
    }
    lingering = before_fps & after_fps
    if not lingering:
        return ()
    targeted_entities = {
        str(row.get("effect_id") or row.get("entity_id") or "").strip()
        for row in applied
    }
    targeted_entities.discard("")
    patch_targets = {
        (
            str(row.get("code") or "").strip(),
            str(row.get("effect_id") or row.get("entity_id") or "").strip(),
            str(row.get("field") or "").strip(),
        )
        for row in applied
    }
    unstable: list[ValidationIssue] = []
    for code, entity_id, field in sorted(lingering):
        if entity_id:
            # Blame only patches that named this entity. Same issue codes on
            # siblings are a different failure, not this repair's no-effect.
            if entity_id not in targeted_entities:
                continue
            # A quantity patch on E6 did not attempt to repair E6's source
            # proposition. No-effect attribution must follow the patch's
            # intended issue/field, not merely its entity ID.
            if not any(
                patch_entity == entity_id
                and (
                    patch_code == code
                    or (patch_field and patch_field == field)
                )
                for patch_code, patch_entity, patch_field in patch_targets
            ):
                continue
        elif targeted_entities:
            # World-level residual while patches named concrete entities:
            # do not treat as this DET's unstable fingerprint.
            continue
        target_label = entity_id or "world"
        unstable.append(ValidationIssue(
            code="REPAIR_NO_EFFECT",
            message=(
                f"repair had no effect on {code} for {target_label}; "
                "same violation remains after deterministic patch "
                "(unstable repair)"
            ),
            entity_kind="effect" if entity_id.startswith("E") else (
                "action" if entity_id.startswith("A") else "world_model"
            ),
            entity_id=entity_id,
            field=field or "repair",
            related_ids=(code,) if code else (),
            repair_class="QUARANTINE",
            permits_removal=False,
        ))
    return tuple(unstable)


def should_skip_deterministic_local_patch(
    issues: Sequence[ValidationIssue | dict[str, object]],
) -> bool:
    """Once REPAIR_NO_EFFECT is flagged, do not re-run the same DET path."""
    return any(
        str(row.get("code") or "") == "REPAIR_NO_EFFECT"
        for row in _issue_rows(issues)
    )


def annotate_admit_with_repair_no_effect(
    admit_result: dict[str, object],
    *,
    before_issues: Sequence[ValidationIssue | dict[str, object]],
    applied_patches: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Attach REPAIR_NO_EFFECT issues when DET left the same failures intact."""
    if admit_result.get("status") == "COMMITTED":
        return admit_result
    after_issues = list(admit_result.get("validation_issues") or [])
    unstable = repair_no_effect_issues(
        before_issues, after_issues, applied_patches,
    )
    if not unstable:
        return admit_result
    out = copy.deepcopy(admit_result)
    messages = [issue.message for issue in unstable]
    out["errors"] = list(dict.fromkeys([
        *[str(item) for item in (out.get("errors") or [])],
        *messages,
    ]))
    out["validation_issues"] = [
        *after_issues,
        *[issue.as_dict() for issue in unstable],
    ]
    out["repair_no_effect"] = [issue.as_dict() for issue in unstable]
    if out.get("status") == "COMMITTED":
        out["status"] = "REJECTED"
    return out


def apply_deterministic_local_patches(
    candidate: object | None,
    issues: Sequence[ValidationIssue | dict[str, object]],
    *,
    clauses: Sequence[dict[str, object]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Apply unambiguous LOCAL_PATCH ops that need no reinterpretation.

    Mechanical defects where the repair card already names the target entity
    and missing payload are corrected here before any model re-grounding.

    Supports:
    - ``QUANTITY_BEARING_CONSEQUENCE_MISSING`` → ``semantic_patch``
      ``add_quantity`` and, when the detector names a clause,
      ``licensing_patch`` ``add_provenance`` for that single clause only
      (``REPAIR_PROVENANCE_MINIMALITY``; legacy op ``add_clause_id`` accepted)
    - ``LIKELIHOOD_QUALIFIER_MISSING`` → ``add_likelihood_qualifier``
    - ``TEMPORAL_QUALIFIER_MISSING`` → ``add_temporal_qualifier``
    - ``SCOPE_QUALIFIER_MISSING`` → ``add_scope_qualifier``

    Ambiguous repairs (``add_effect``, coreference, lemma rewrite) are left
    untouched.
    """
    if not isinstance(candidate, dict):
        return {}, []
    cards = repair_guidance_cards(issues, candidate, clauses=clauses)
    working = copy.deepcopy(candidate)
    world = working.get("world_model")
    if not isinstance(world, dict):
        return working, []
    effects = world.get("effects")
    if not isinstance(effects, list):
        return working, []
    effects_by_id: dict[str, dict[str, object]] = {
        str(effect["effect_id"]): effect
        for effect in effects
        if isinstance(effect, dict) and effect.get("effect_id")
    }
    actions_by_id: dict[str, dict[str, object]] = {
        str(action["action_id"]): action
        for action in (world.get("actions") or [])
        if isinstance(action, dict) and action.get("action_id")
    }
    applied: list[dict[str, object]] = []
    qualifier_ops = {
        "add_likelihood_qualifier": "likelihood_qualifiers",
        "add_temporal_qualifier": "temporal_qualifiers",
        "add_scope_qualifier": "scope_qualifiers",
    }
    for card in cards:
        code = str(card.get("code") or "")
        if code in {
            "QUANTITY_BEARING_CONSEQUENCE_MISSING",
            "EFFECT_QUANTITY_MISSING",
            "EFFECT_QUANTITY_LEAK",
        }:
            for patch in card.get("concrete_patches") or []:
                if not isinstance(patch, dict):
                    continue
                op = str(patch.get("op") or "")
                effect_id = str(patch.get("effect_id") or "").strip()
                value = str(patch.get("value") or "").strip()
                effect = effects_by_id.get(effect_id)
                if effect is None or not value:
                    continue
                if op == "add_quantity":
                    from .world_state import merge_recorded_quantity_spans
                    from relent.quantity_typing import sanitize_recorded_quantities

                    existing = [
                        str(item).strip()
                        for item in (effect.get("quantities") or [])
                        if str(item).strip()
                    ]
                    source_blob = " ".join(
                        str(item)
                        for item in (
                            effect.get("outcome"),
                            effect.get("source_proposition"),
                            value,
                        )
                        if item
                    )
                    merged = list(sanitize_recorded_quantities(
                        merge_recorded_quantity_spans(existing, value),
                        source_texts=(source_blob,),
                    ))
                    if merged == existing:
                        continue
                    effect["quantities"] = merged
                    applied.append({
                        "op": op,
                        "effect_id": effect_id,
                        "field": "quantities",
                        "value": value,
                        "code": code,
                        "patch_kind": "semantic_patch",
                        "deterministic": True,
                    })
                elif op == "remove_quantity":
                    existing = [
                        str(item).strip()
                        for item in (effect.get("quantities") or [])
                        if str(item).strip()
                    ]
                    kept = [
                        item for item in existing
                        if item.casefold() != value.casefold()
                    ]
                    if kept == existing:
                        continue
                    effect["quantities"] = kept
                    applied.append({
                        "op": op,
                        "effect_id": effect_id,
                        "field": "quantities",
                        "value": value,
                        "code": code,
                        "patch_kind": "semantic_patch",
                        "deterministic": True,
                    })
                elif op in {"add_provenance", "add_clause_id"}:
                    existing_ids = [
                        str(item).strip()
                        for item in (effect.get("clause_ids") or [])
                        if str(item).strip()
                    ]
                    if any(item.casefold() == value.casefold() for item in existing_ids):
                        continue
                    effect["clause_ids"] = [*existing_ids, value]
                    applied.append({
                        "op": "add_provenance",
                        "effect_id": effect_id,
                        "field": "clause_ids",
                        "value": value,
                        "code": code,
                        "patch_kind": "licensing_patch",
                        "deterministic": True,
                    })
            continue
        for patch in card.get("concrete_patches") or []:
            if not isinstance(patch, dict):
                continue
            op = str(patch.get("op") or "")
            if op == "add_action_provenance" and code == "ACTION_DIRECT_PROVENANCE_MISSING":
                action_id = str(patch.get("action_id") or "").strip()
                value = str(patch.get("value") or "").strip()
                action = actions_by_id.get(action_id)
                if action is None or not value:
                    continue
                existing = [
                    str(item).strip()
                    for item in (action.get("clause_ids") or [])
                    if str(item).strip()
                ]
                if value in existing:
                    continue
                action["clause_ids"] = [*existing, value]
                applied.append({
                    "op": op,
                    "action_id": action_id,
                    "field": "clause_ids",
                    "value": value,
                    "code": code,
                    "patch_kind": "licensing_patch",
                    "deterministic": True,
                })
                continue
            effect_id = str(patch.get("effect_id") or "").strip()
            value = str(patch.get("value") or "").strip()
            effect = effects_by_id.get(effect_id)
            if effect is None or not value:
                continue
            if op in qualifier_ops and code in {
                "LIKELIHOOD_QUALIFIER_MISSING",
                "TEMPORAL_QUALIFIER_MISSING",
                "SCOPE_QUALIFIER_MISSING",
            }:
                field = qualifier_ops[op]
            elif op == "replace_outcome" and code == "VERB_LEMMA_MISMATCH":
                current = " ".join(str(effect.get("outcome") or "").split())
                replacement = " ".join(value.split())
                bound_source = " ".join(
                    str(effect.get("source_proposition") or "").split()
                )
                if (
                    not replacement
                    or replacement != bound_source
                    or current == replacement
                ):
                    continue
                effect["outcome"] = replacement
                applied.append({
                    "op": op,
                    "effect_id": effect_id,
                    "field": "outcome",
                    "value": replacement,
                    "code": code,
                    "patch_kind": "source_patch",
                    "deterministic": True,
                })
                continue
            else:
                continue
            existing = [
                str(item).strip()
                for item in (effect.get(field) or [])
                if str(item).strip()
            ]
            if any(item.casefold() == value.casefold() for item in existing):
                break
            effect[field] = [*existing, value]
            # Chance hedges on CERTAIN rows must untype when likelihood is attached.
            if field == "likelihood_qualifiers":
                modality = str(effect.get("modality") or "").upper()
                if modality == "CERTAIN":
                    effect["modality"] = "PROBABILISTIC"
            applied.append({
                "op": op,
                "effect_id": effect_id,
                "field": field,
                "value": value,
                "code": code,
                "deterministic": True,
            })
            break
    return working, applied


def format_repair_guidance_for_prompt(
    cards: Sequence[dict[str, object]],
) -> str:
    """Render guidance cards as compact, model-readable repair instructions."""
    if not cards:
        return ""
    blocks: list[str] = []
    for index, card in enumerate(cards, start=1):
        code = str(card.get("code") or "ISSUE")
        entity = str(card.get("entity_id") or "world")
        lines = [f"{index}. {entity} [{code}]"]
        message = str(card.get("message") or "").strip()
        if message:
            lines.append(f"   error: {message}")
        outcome = str(card.get("current_outcome") or "").strip()
        proposition = str(card.get("current_source_proposition") or "").strip()
        if outcome or proposition:
            lines.append(f"   current outcome: {outcome!r}")
            lines.append(f"   current source_proposition: {proposition!r}")
        spans = [
            str(item) for item in (card.get("available_clause_spans") or [])
            if str(item).strip()
        ]
        if spans:
            lines.append("   available clause spans:")
            for span in spans[:4]:
                lines.append(f"     - {span}")
        patches = list(card.get("concrete_patches") or [])
        if patches:
            quantity_pair = (
                str(card.get("code") or "") == "QUANTITY_BEARING_CONSEQUENCE_MISSING"
                and any(
                    str(patch.get("op") or "") in {
                        "add_clause_id", "add_provenance",
                    }
                    for patch in patches
                    if isinstance(patch, dict)
                )
            )
            lines.append(
                "   apply these patches together:"
                if quantity_pair else
                "   apply exactly one patch:"
            )
            for patch in patches[:4]:
                op = str(patch.get("op") or "")
                if op == "replace_outcome":
                    lines.append(
                        f"     • set outcome={patch.get('value')!r}"
                    )
                elif op == "replace_source_proposition":
                    lines.append(
                        f"     • set source_proposition={patch.get('value')!r}"
                    )
                elif op == "add_effect":
                    value = patch.get("value") or {}
                    lines.append(
                        f"     • add effect on {patch.get('action_id') or entity}: "
                        f"outcome={value.get('outcome')!r}, "
                        f"polarity={value.get('polarity')!r}"
                    )
                elif op == "insert_process_intermediate":
                    value = patch.get("value") or {}
                    lines.append(
                        f"     • insert {patch.get('new_effect_id')} on "
                        f"{value.get('party_id')} between "
                        f"{patch.get('replace_parent_effect_ids')} and "
                        f"{patch.get('child_effect_id')}; copy the process "
                        "outcome/source_proposition from the cited source span "
                        "and replace the direct parent→child edge"
                    )
                elif op == "restore_exclusive_allocation_branch":
                    sequence = list(patch.get("required_effect_sequence") or [])
                    links = list(patch.get("required_links") or [])
                    lines.append(
                        f"     • on {patch.get('action_id')}, restore the branch "
                        f"for {patch.get('nonrecipient_label') or patch.get('nonrecipient_party_id')}:"
                    )
                    for step in sequence:
                        lines.append(
                            "       - add "
                            f"{step.get('derivation_operation')} effect "
                            f"outcome={step.get('outcome')!r}, "
                            f"source_effect_ids={step.get('source_effect_ids')}; "
                            f"source_proposition={step.get('source_proposition')!r}"
                        )
                    for link in links:
                        lines.append(
                            "       - link "
                            f"{link.get('source_id')} --{link.get('relation')}--> "
                            f"{link.get('target_id')}"
                        )
                elif op == "repair_derivation_contract_metadata":
                    values = patch.get("set") or {}
                    lines.append(
                        f"     • preserve node {patch.get('effect_id')} and every "
                        "causal link; change metadata only: "
                        f"derivation_operation={values.get('derivation_operation')!r}, "
                        f"source_proposition={values.get('source_proposition')!r}, "
                        f"clause_ids={values.get('clause_ids')!r}, "
                        f"quantities={values.get('quantities')!r}"
                    )
                elif op == "quarantine_unlicensed_allocation_complement":
                    lines.append(
                        f"     • quarantine node {patch.get('effect_id')}; no exact "
                        "source span licenses an exclusive-allocation complement, "
                        "so do not relabel it or reconstruct surrounding topology"
                    )
                elif op == "restore_stage_skeleton_proposition":
                    lines.append(
                        f"     • restore Stage 1 proposition "
                        f"{patch.get('skeleton_proposition_id')} on "
                        f"{patch.get('action_id')}; preserve its party, source "
                        "span, polarity, modality, directness, and quantities"
                    )
                elif op == "add_quantity":
                    lines.append(
                        f"     • semantic_patch add quantities+={patch.get('value')!r} "
                        f"on {patch.get('effect_id') or entity}"
                    )
                elif op in {"add_clause_id", "add_provenance"}:
                    lines.append(
                        f"     • licensing_patch add provenance "
                        f"clause_ids+={patch.get('value')!r} "
                        f"on {patch.get('effect_id') or entity}"
                    )
                elif op in {
                    "add_likelihood_qualifier",
                    "add_temporal_qualifier",
                    "add_scope_qualifier",
                }:
                    lines.append(
                        f"     • add {patch.get('field')}+={patch.get('value')!r} "
                        f"on {patch.get('effect_id') or entity}"
                    )
                elif op == "move_quantity":
                    lines.append(
                        f"     • move quantity={patch.get('value')!r} "
                        f"from {patch.get('from_party_id')} to "
                        f"{patch.get('to_party_id')}"
                    )
                elif op == "clear_quantity":
                    lines.append(
                        f"     • clear quantity={patch.get('value')!r} "
                        f"from {patch.get('party_id') or entity}"
                    )
                elif op == "remove_effect":
                    lines.append(
                        f"     • remove effect {patch.get('remove_effect_id')}; "
                        f"also update {', '.join(patch.get('also_update') or [])}"
                    )
                else:
                    lines.append(f"     • {op}: {patch}")
                why = str(patch.get("why") or "").strip()
                if why:
                    lines.append(f"       ({why})")
        elif card.get("fix_examples"):
            lines.append("   fix options:")
            for example in card.get("fix_examples") or []:
                lines.append(f"     • {example}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def apply_stuck_source_binding_scope(
    contract: dict[str, object],
    *,
    issues: Sequence[ValidationIssue | dict[str, object]],
    candidate: object | None = None,
    prior_attempt_scopes: Sequence[str] = (),
    prior_attempt_issue_codes: Sequence[Sequence[str]] = (),
) -> dict[str, object]:
    """Return a repair contract with stuck SOURCE_PROPOSITION_BINDING widened."""
    widened = widen_stuck_source_binding_repair_scope(
        str(contract.get("repair_scope") or LOCAL_PATCH),
        issue_codes=[str(code) for code in contract.get("issue_codes") or ()],
        prior_attempt_scopes=prior_attempt_scopes,
        prior_attempt_issue_codes=prior_attempt_issue_codes,
    )
    if widened == contract.get("repair_scope"):
        return contract
    issue_rows = _issue_rows(issues)
    updated = dict(contract)
    updated["repair_scope"] = widened
    updated["implicated_action_ids"] = list(
        implicated_action_ids(issue_rows, candidate)
        if widened == SUBGRAPH_REBUILD else ()
    )
    return updated


def issue_mentions_identifier(issue: ValidationIssue | dict[str, object], identifier: str) -> bool:
    row = issue.as_dict() if isinstance(issue, ValidationIssue) else issue
    values = {
        str(row.get("entity_id") or ""),
        *[str(value) for value in row.get("related_ids", ())],
    }
    return str(identifier) in values
