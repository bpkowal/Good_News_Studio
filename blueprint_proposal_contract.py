"""Shared proposal-envelope contract for candidate graph blueprints.

The envelope is diagnostic until Parliament admits its candidate.  A family
whose semantics cannot be represented in world schema 1.3 still emits the same
evidence/provenance envelope, with ``candidate`` set to ``None`` and a named
construction problem.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from blueprint_derivation_license import apply_derivation_license, derivation_errors
from blueprint_kind_license import apply_kind_license


CONTRACT_VERSION = "blueprint-proposal-contract/0.4"
ASSIGNMENT_KINDS = {"z10_proposition_ids", "intervention_text", "none"}
PROVENANCE_ORIGINS = {
    "SOURCE_ASSERTED", "STRUCTURALLY_DERIVED",
    "WORLD_KNOWLEDGE_HYPOTHESIS", "UNRESOLVED",
}
EXCLUSIVITY_STATUSES = {"EXPLICIT", "DERIVED", "HYPOTHESIZED", "UNKNOWN"}
PROPOSAL_KEYS = {
    "proposal_id", "blueprint_id", "status", "assignment_kind", "assignment",
    "slot_bindings", "selection", "selection_validation", "candidate", "clauses",
    "unfilled_required_slots", "unresolved_readings", "construction_problems",
    "admission_authorized", "notes",
    "pre_world_assessment", "accepted_evidence", "world_withheld",
    "construction_provenance", "relation_alternatives", "exclusivity_proof",
}
WORLD_KEYS = {
    "schema_version", "parties", "actions", "effects", "conditions",
    "temporal_relations", "causal_links", "counterfactual_links",
}
ACTION_KEYS = {
    "action_id", "intervention", "actor_party_id", "recipient_party_ids",
    "effect_ids", "clause_ids",
}
EFFECT_KEYS = {
    "effect_id", "action_id", "party_id", "outcome", "predicate", "polarity",
    "directness", "modality", "effect_kind", "condition_ids", "quantities",
    "likelihood_qualifiers", "overall_likelihood_qualifiers", "scope_qualifiers",
    "temporal_qualifiers", "condition_join", "source_proposition",
    "source_effect_ids", "derivation_operation", "derivation_explanation",
    "derivation_assumptions", "outcome_type_transformation", "clause_ids",
}


def world_model(parties: Sequence[dict[str, Any]],
                actions: Sequence[dict[str, Any]],
                effects: Sequence[dict[str, Any]],
                causal_links: Sequence[dict[str, Any]],
                conditions: Sequence[dict[str, Any]] = (),
                temporal_relations: Sequence[dict[str, Any]] = (),
                counterfactual_links: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    return {
        "schema_version": "1.3",
        "parties": list(parties),
        "actions": list(actions),
        "effects": list(effects),
        "conditions": list(conditions),
        "temporal_relations": list(temporal_relations),
        "causal_links": list(causal_links),
        "counterfactual_links": list(counterfactual_links),
    }


def candidate(actions: dict[str, dict[str, Any]], world: dict[str, Any],
              ellipsis_resolutions: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    return {
        "actions": actions,
        "world_model": world,
        "ellipsis_resolutions": list(ellipsis_resolutions),
    }


def proposal(*, proposal_id: str, blueprint_id: str, status: str,
             assignment_kind: str, assignment: Sequence[str],
             slot_bindings: dict[str, Any], selection: dict[str, Any] | None,
             selection_validation: dict[str, Any], candidate_value: dict[str, Any] | None,
             clauses: Sequence[dict[str, str]],
             unfilled_required_slots: Sequence[str] = (),
             unresolved_readings: Sequence[dict[str, Any] | str] = (),
             construction_problems: Sequence[dict[str, Any] | str] = (),
             admission_authorized: bool = True,
             notes: Sequence[Any] = (),
             pre_world_assessment: dict[str, Any] | None = None,
             accepted_evidence: dict[str, Any] | None = None,
             construction_provenance: Sequence[dict[str, Any]] = (),
             relation_alternatives: Sequence[dict[str, Any]] = (),
             exclusivity_proof: dict[str, Any] | None = None,
             world_withheld: Sequence[str] = ()) -> dict[str, Any]:
    row = {
        "proposal_contract_version": CONTRACT_VERSION,
        "proposal_id": proposal_id,
        "blueprint_id": blueprint_id,
        "status": status,
        "assignment_kind": assignment_kind,
        "assignment": list(assignment),
        "slot_bindings": slot_bindings,
        "selection": selection,
        "selection_validation": selection_validation,
        "candidate": candidate_value,
        "clauses": list(clauses),
        "unfilled_required_slots": list(unfilled_required_slots),
        "unresolved_readings": list(unresolved_readings),
        "construction_problems": list(construction_problems),
        "admission_authorized": admission_authorized,
        "notes": list(notes),
        "pre_world_assessment": pre_world_assessment or {
            "status": "not_assessed",
            "eligible_for_world_state": admission_authorized,
        },
        "accepted_evidence": accepted_evidence or {},
        "construction_provenance": list(construction_provenance),
        "relation_alternatives": list(relation_alternatives),
        "exclusivity_proof": exclusivity_proof or {
            "status": "UNKNOWN", "evidence": [], "assumptions": [],
            "explanation": "No exclusivity claim is needed or established.",
        },
        "world_withheld": list(world_withheld),
    }
    if isinstance(candidate_value, dict) and isinstance(candidate_value.get("world_model"), dict):
        apply_derivation_license(
            candidate_value["world_model"],
            list(clauses),
            exclusivity_proof or row.get("exclusivity_proof") or {},
        )
        apply_kind_license(
            candidate_value["world_model"],
            list(clauses),
            blueprint_id,
        )
    errors = validate_proposal(row)
    if errors:
        raise ValueError("Invalid blueprint proposal envelope: " + "; ".join(errors))
    return row


def withheld_proposal(*, proposal_id: str, blueprint_id: str,
                      assignment: Sequence[str], slot_bindings: dict[str, Any],
                      clauses: Sequence[dict[str, str]],
                      unfilled_required_slots: Sequence[str],
                      unresolved_readings: Sequence[dict[str, Any] | str],
                      construction_problems: Sequence[dict[str, Any] | str],
                      selection: dict[str, Any] | None = None,
                      selection_validation: dict[str, Any] | None = None,
                      notes: Sequence[Any] = (),
                      pre_world_assessment: dict[str, Any] | None = None,
                      accepted_evidence: dict[str, Any] | None = None,
                      construction_provenance: Sequence[dict[str, Any]] = (),
                      relation_alternatives: Sequence[dict[str, Any]] = (),
                      exclusivity_proof: dict[str, Any] | None = None) -> dict[str, Any]:
    validation = selection_validation or {
        "contract_valid": None,
        "status": "not_assessed",
        "reason": "No Z10 candidate selection was constructed.",
    }
    return proposal(
        proposal_id=proposal_id,
        blueprint_id=blueprint_id,
        status="WITHHELD",
        assignment_kind="intervention_text" if assignment else "none",
        assignment=assignment,
        slot_bindings=slot_bindings,
        selection=selection,
        selection_validation=validation,
        candidate_value=None,
        clauses=clauses,
        unfilled_required_slots=unfilled_required_slots,
        unresolved_readings=unresolved_readings,
        construction_problems=construction_problems,
        admission_authorized=False,
        notes=notes,
        pre_world_assessment=pre_world_assessment or {
            "status": "WITHHELD",
            "eligible_for_world_state": False,
        },
        accepted_evidence=accepted_evidence or slot_bindings,
        construction_provenance=construction_provenance,
        relation_alternatives=relation_alternatives,
        exclusivity_proof=exclusivity_proof,
        world_withheld=[
            problem.get("message", str(problem)) if isinstance(problem, dict) else str(problem)
            for problem in construction_problems
        ],
    )


def validate_proposal(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(PROPOSAL_KEYS - set(row))
    if missing:
        errors.append("missing proposal fields: " + ", ".join(missing))
    if row.get("assignment_kind") not in ASSIGNMENT_KINDS:
        errors.append("assignment_kind is invalid")
    proof = row.get("exclusivity_proof") or {}
    if proof.get("status") not in EXCLUSIVITY_STATUSES:
        errors.append("exclusivity_proof.status is invalid")
    for index, provenance in enumerate(row.get("construction_provenance") or []):
        if provenance.get("origin") not in PROVENANCE_ORIGINS:
            errors.append(f"construction_provenance[{index}] has invalid origin")
        if not provenance.get("atom_id"):
            errors.append(f"construction_provenance[{index}] has no atom_id")
    candidate_value = row.get("candidate")
    authorized = row.get("admission_authorized")
    problems = row.get("construction_problems") or []
    if candidate_value is None:
        if authorized is not False:
            errors.append("candidate null requires admission_authorized false")
        if not problems:
            errors.append("candidate null requires a construction problem")
        if not row.get("world_withheld"):
            errors.append("candidate null requires a withholding reason")
        return errors
    if authorized is not True:
        errors.append("a candidate world requires admission_authorized true")
    assessment = row.get("pre_world_assessment", {})
    if assessment.get("eligible_for_world_state") is False:
        errors.append("an authorized candidate cannot fail the pre-world assessment")
    if (
        row.get("blueprint_id") == "exclusive_allocation"
        and proof.get("status") not in {"EXPLICIT", "DERIVED"}
        and assessment.get("exclusivity") != "evidenced"
    ):
        errors.append(
            "an exclusive-allocation candidate requires evidenced exclusivity")
    world = candidate_value.get("world_model") if isinstance(candidate_value, dict) else None
    if not isinstance(world, dict):
        errors.append("candidate.world_model is missing")
        return errors
    if world.get("schema_version") != "1.3":
        errors.append("candidate world schema must be 1.3")
    absent_world = sorted(WORLD_KEYS - set(world))
    if absent_world:
        errors.append("missing world fields: " + ", ".join(absent_world))
    _validate_unique_ids(world.get("parties") or [], "party_id", errors)
    _validate_unique_ids(world.get("actions") or [], "action_id", errors)
    _validate_unique_ids(world.get("effects") or [], "effect_id", errors)
    for action in world.get("actions") or []:
        extra = sorted(set(action) - ACTION_KEYS)
        if extra:
            errors.append(
                f"{action.get('action_id', 'action')} has unknown fields: " + ", ".join(extra))
    for effect in world.get("effects") or []:
        missing_effect = sorted(EFFECT_KEYS - set(effect))
        if missing_effect:
            errors.append(
                f"{effect.get('effect_id', 'effect')} missing fields: "
                + ", ".join(missing_effect))
        if not effect.get("clause_ids"):
            errors.append(f"{effect.get('effect_id', 'effect')} has no clause evidence")
        if not effect.get("source_proposition"):
            errors.append(f"{effect.get('effect_id', 'effect')} has no source proposition")
    errors.extend(derivation_errors(
        world, row.get("clauses") or [], row.get("exclusivity_proof") or {}))
    return errors


def coverage_metrics(required: dict[str, Any], optional: dict[str, Any],
                     rejected: int = 0, unsupported_atoms: int = 0,
                     unresolved_readings: int = 0) -> dict[str, int]:
    return {
        "required_filled": sum(bool(value) for value in required.values()),
        "required_total": len(required),
        "optional_filled": sum(bool(value) for value in optional.values()),
        "optional_total": len(optional),
        "rejected_evidence": rejected,
        "unsupported_atoms": unsupported_atoms,
        "unresolved_readings": unresolved_readings,
    }


def _validate_unique_ids(rows: Iterable[dict[str, Any]], key: str,
                         errors: list[str]) -> None:
    seen: set[str] = set()
    for row in rows:
        ident = row.get(key)
        if not isinstance(ident, str) or not ident:
            errors.append(f"{key} is missing")
        elif ident in seen:
            errors.append(f"duplicate {key}: {ident}")
        else:
            seen.add(ident)
