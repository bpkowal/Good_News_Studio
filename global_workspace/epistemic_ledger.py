"""Authoritative proposition identities and monotone epistemic status.

World facts seed the ledger. Delegates may cite them or introduce hypotheses,
but repetition and cross-framework reuse only increase attention, never status.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any, Iterable

from .scenario_semantics import project_grounded_action_effects
from .semantic_graph import SemanticGraph


EPISTEMIC_STATUS_RANK = {
    "REJECTED": 0,
    "HYPOTHETICAL": 1,
    "UNRESOLVED": 2,
    "DERIVED": 3,
    "ESTABLISHED": 4,
}
DECISION_CRITICAL_CAP_STATUSES = {"REJECTED", "HYPOTHETICAL", "UNRESOLVED"}
_COMPOSITION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
    "is", "of", "on", "or", "that", "the", "to", "with",
}


@dataclass(slots=True)
class PropositionRecord:
    proposition_id: str
    claim: str
    proposition_type: str
    epistemic_status: str
    support_ids: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    introduced_by: str = "SYSTEM"
    mention_count: int = 1
    decision_critical_mentions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(prefix: str, value: str) -> str:
    normalized = " ".join(str(value).casefold().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"PROP:{prefix}:{digest}"


def seed_proposition_ledger(graph: SemanticGraph) -> dict[str, PropositionRecord]:
    """Create established descriptive propositions from admitted world effects."""
    ledger: dict[str, PropositionRecord] = {}
    seen_consequences: set[str] = set()
    seen_party_quantities: set[tuple[str, str]] = set()
    for effect in project_grounded_action_effects(graph):
        if effect.consequence_id in seen_consequences:
            continue
        seen_consequences.add(effect.consequence_id)
        consequence = graph.nodes.get(effect.consequence_id)
        if consequence is None:
            continue
        world_effect_id = str(consequence.attributes.get("world_effect_id", "")).strip()
        proposition_id = (
            f"PROP:WORLD:{world_effect_id}"
            if world_effect_id else _stable_id("GROUND", effect.consequence_id)
        )
        claim_parts = [consequence.label]
        affected_subject = " ".join(effect.affected_subject.split())
        if affected_subject and affected_subject.casefold() != "affected constituency":
            claim_parts.append(f"affected subject: {affected_subject}")
        qualifier = " ".join(effect.magnitude_or_qualifier.split())
        if qualifier and qualifier.upper() != "STATED":
            qualifier_words = set(qualifier.casefold().split())
            subject_words = set(affected_subject.casefold().split())
            if not qualifier_words <= subject_words:
                claim_parts.append(f"magnitude or qualifier: {qualifier}")
        ledger[proposition_id] = PropositionRecord(
            proposition_id=proposition_id,
            claim="; ".join(claim_parts),
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            support_ids=[world_effect_id or effect.consequence_id],
            introduced_by="WORLD_MODEL",
        )
        party_id = (
            effect.affected_subject_node_ids[0].removeprefix("PARTY:")
            if effect.affected_subject_node_ids else effect.affected_subject
        )
        for quantity in effect.affected_subject_quantities:
            quantity_key = (party_id, quantity.casefold())
            if quantity_key in seen_party_quantities:
                continue
            seen_party_quantities.add(quantity_key)
            subject = effect.affected_subject
            atomic_claim = (
                subject if quantity.casefold() in subject.casefold()
                else f"{quantity} {subject}"
            )
            quantity_digest = hashlib.sha256(
                f"{party_id}|{quantity.casefold()}".encode("utf-8")
            ).hexdigest()[:12]
            atomic_id = f"PROP:WORLD:PARTY:{party_id}:QUANTITY:{quantity_digest}"
            ledger[atomic_id] = PropositionRecord(
                proposition_id=atomic_id,
                claim=atomic_claim,
                proposition_type="DESCRIPTIVE",
                epistemic_status="ESTABLISHED",
                support_ids=[party_id, quantity],
                introduced_by="WORLD_MODEL",
            )
        qualifier_groups = (
            ("LIKELIHOOD", effect.likelihood_qualifiers),
            ("SCOPE", effect.scope_qualifiers),
            ("TEMPORAL", effect.temporal_qualifiers),
        )
        for qualifier_kind, qualifiers in qualifier_groups:
            for index, qualifier in enumerate(qualifiers):
                atomic_id = (
                    f"PROP:WORLD:{world_effect_id or effect.effect_id}:"
                    f"{qualifier_kind}:{index}"
                )
                ledger[atomic_id] = PropositionRecord(
                    proposition_id=atomic_id,
                    claim=f"{qualifier} — {consequence.label}",
                    proposition_type="DESCRIPTIVE",
                    epistemic_status="ESTABLISHED",
                    support_ids=[world_effect_id or effect.effect_id, qualifier],
                    introduced_by="WORLD_MODEL",
                )
    return ledger


def ledger_projection(
    ledger: dict[str, PropositionRecord],
) -> list[dict[str, Any]]:
    return [ledger[key].to_dict() for key in sorted(ledger)]


def register_hypothesis(
    ledger: dict[str, PropositionRecord],
    claim: str,
    *,
    specialist: str,
    derived_from: Iterable[str] = (),
    decision_critical: bool = False,
) -> str:
    """Register or mention a hypothesis without ever promoting its status."""
    cleaned = " ".join(str(claim).split())[:240]
    if not cleaned or cleaned.upper() == "NONE":
        return ""
    proposition_id = _stable_id("HYPOTHESIS", cleaned)
    known_dependencies = [
        value for value in dict.fromkeys(str(item) for item in derived_from)
        if value in ledger
    ]
    existing = ledger.get(proposition_id)
    if existing is None:
        ledger[proposition_id] = PropositionRecord(
            proposition_id=proposition_id,
            claim=cleaned,
            proposition_type="HYPOTHESIS",
            epistemic_status="HYPOTHETICAL",
            derived_from=known_dependencies,
            introduced_by=specialist,
            decision_critical_mentions=1 if decision_critical else 0,
        )
    else:
        existing.mention_count += 1
        existing.decision_critical_mentions += 1 if decision_critical else 0
        existing.derived_from = list(dict.fromkeys([
            *existing.derived_from, *known_dependencies,
        ]))
        # Deliberately no status mutation: recurrence is attention, not evidence.
    return proposition_id


def register_derived_proposition(
    ledger: dict[str, PropositionRecord],
    claim: str,
    *,
    specialist: str,
    derived_from: Iterable[str],
    decision_critical: bool = False,
) -> str:
    """Register a transparent composition without promoting any dependency."""
    cleaned = " ".join(str(claim).split())[:240]
    dependencies = [
        value for value in dict.fromkeys(str(item) for item in derived_from)
        if value in ledger
    ]
    if not cleaned:
        return ""
    dependency_statuses_are_authoritative = all(
        ledger[value].epistemic_status in {"ESTABLISHED", "DERIVED"}
        for value in dependencies
    )
    dependency_words = {
        word
        for value in dependencies
        for word in re.findall(r"[a-z0-9]+", ledger[value].claim.casefold())
        if word not in _COMPOSITION_STOPWORDS
    }
    claim_words = {
        word for word in re.findall(r"[a-z0-9]+", cleaned.casefold())
        if word not in _COMPOSITION_STOPWORDS
    }
    # A model may propose a transparent composition, but only this deterministic
    # coverage check can admit it as DERIVED. A single proposition should be
    # cited directly; new vocabulary remains a hypothesis.
    transparent_composition = (
        len(dependencies) >= 2
        and dependency_statuses_are_authoritative
        and claim_words <= dependency_words
    )
    if not transparent_composition:
        return register_hypothesis(
            ledger, cleaned, specialist=specialist,
            derived_from=dependencies,
            decision_critical=decision_critical,
        )
    proposition_id = _stable_id("DERIVED", cleaned)
    existing = ledger.get(proposition_id)
    if existing is None:
        ledger[proposition_id] = PropositionRecord(
            proposition_id=proposition_id,
            claim=cleaned,
            proposition_type="DESCRIPTIVE",
            epistemic_status="DERIVED",
            derived_from=dependencies,
            introduced_by=specialist,
            decision_critical_mentions=1 if decision_critical else 0,
        )
    else:
        existing.mention_count += 1
        existing.decision_critical_mentions += 1 if decision_critical else 0
        existing.derived_from = list(dict.fromkeys([
            *existing.derived_from, *dependencies,
        ]))
    return proposition_id


def weakest_status(
    ledger: dict[str, PropositionRecord], proposition_ids: Iterable[str],
) -> str:
    records = [ledger.get(str(value)) for value in proposition_ids]
    records = [record for record in records if record is not None]
    if not records:
        return "ESTABLISHED"
    return min(
        (record.epistemic_status for record in records),
        key=lambda status: EPISTEMIC_STATUS_RANK.get(status, 0),
    )


def _normalized_claim(value: str) -> str:
    return " ".join(str(value).casefold().split()).strip(" .")


def _claim_is_covered(claim: str, authoritative_claim: str) -> bool:
    """Accept exact canonical atoms and lossless projections of bundled display text."""
    normalized = _normalized_claim(claim)
    if normalized == _normalized_claim(authoritative_claim):
        return True
    components = [
        _normalized_claim(value) for value in str(authoritative_claim).split(";")
        if _normalized_claim(value)
    ]
    return normalized in components


def _apply_candidate_authority_cap(
    ledger: dict[str, PropositionRecord], candidate: Any,
) -> None:
    status = weakest_status(ledger, candidate.decision_critical_proposition_ids)
    candidate.weakest_decision_critical_status = status
    candidate.decision_critical_dependency_claims = [
        ledger[value].claim for value in candidate.decision_critical_proposition_ids
        if value in ledger
    ][:4]
    if status not in DECISION_CRITICAL_CAP_STATUSES:
        return
    if str(getattr(candidate, "assumption_status", "")).upper() != "NORMATIVELY_CONTESTED":
        candidate.assumption_status = "UNDERDETERMINED"
    if str(getattr(candidate, "unresolved", "NONE")).upper() == "NONE":
        candidate.unresolved = "VERIFY_FACTS"
    candidate.selection_status = "PROVISIONAL"
    candidate.comparison_complete = False
    candidate.evidence_sufficient_for_action = False
    candidate.epistemic_confidence = min(
        float(getattr(candidate, "epistemic_confidence", 1.0)), 0.50,
    )
    candidate.confidence = candidate.epistemic_confidence


def attach_candidate_dependencies(
    ledger: dict[str, PropositionRecord], candidate: Any,
) -> None:
    """Resolve candidate references and impose the decision-critical status cap."""
    submitted_cited = list(dict.fromkeys(
        str(item) for item in getattr(candidate, "supporting_proposition_ids", [])
    ))
    submitted_critical = list(dict.fromkeys(
        str(item) for item in getattr(candidate, "decision_critical_proposition_ids", [])
    ))
    unknown = (set(submitted_cited) | set(submitted_critical)) - set(ledger)
    if unknown:
        candidate.schema_valid = False
        candidate.validation_errors = list(dict.fromkeys([
            *list(getattr(candidate, "validation_errors", []) or []),
            f"candidate cites unknown proposition IDs: {sorted(unknown)}",
        ]))
        return
    cited = [
        value for value in dict.fromkeys(
            submitted_cited
        ) if value in ledger
    ]
    critical = [
        value for value in dict.fromkeys(
            submitted_critical
        ) if value in ledger
    ]
    registered_hypotheses: set[str] = set()
    registered_claims: dict[str, str] = {}
    processed_premises: set[tuple[str, str, bool]] = set()
    binding_notes = list(getattr(candidate, "epistemic_binding_notes", []) or [])
    for premise in list(getattr(candidate, "material_empirical_claims", []) or []):
        if not isinstance(premise, dict):
            continue
        claim = " ".join(str(premise.get("claim", "")).split())[:240]
        basis = str(premise.get("proposition_id", "HYPOTHESIS")).strip()
        decision_critical = premise.get("decision_critical") is True
        premise_key = (_normalized_claim(claim), basis, decision_critical)
        if premise_key in processed_premises:
            continue
        processed_premises.add(premise_key)
        authoritative = ledger.get(basis)
        if (
            authoritative is not None
            and _claim_is_covered(claim, authoritative.claim)
        ):
            cited.append(basis)
            if decision_critical:
                critical.append(basis)
            continue
        derived_from = [basis] if authoritative is not None else cited
        hypothesis_id = register_hypothesis(
            ledger, claim,
            specialist=str(getattr(candidate, "specialist", "unknown")),
            derived_from=derived_from,
            decision_critical=decision_critical,
        )
        if not hypothesis_id:
            continue
        registered_hypotheses.add(hypothesis_id)
        registered_claims[_normalized_claim(claim)] = hypothesis_id
        cited.append(hypothesis_id)
        if decision_critical:
            critical.append(hypothesis_id)
        if authoritative is not None:
            binding_notes.append(
                f"Premise strengthened {basis}; reclassified as {hypothesis_id}."
            )
    speculative = str(getattr(candidate, "speculative_claim", "") or "").strip()
    tier = str(getattr(candidate, "evidence_calibration_tier", "") or "").upper()
    if speculative and speculative.upper() != "NONE":
        hypothesis_id = registered_claims.get(_normalized_claim(speculative), "")
        critical_speculation = tier in {"DECISION_CRITICAL", "REMOTE"}
        if not hypothesis_id:
            hypothesis_id = register_hypothesis(
                ledger, speculative,
                specialist=str(getattr(candidate, "specialist", "unknown")),
                derived_from=cited,
                decision_critical=critical_speculation,
            )
            registered_hypotheses.add(hypothesis_id)
        elif critical_speculation and hypothesis_id not in critical:
            ledger[hypothesis_id].decision_critical_mentions += 1
        cited.append(hypothesis_id)
        if critical_speculation:
            critical.append(hypothesis_id)
    candidate.supporting_proposition_ids = list(dict.fromkeys(cited))
    candidate.decision_critical_proposition_ids = list(dict.fromkeys(critical))
    candidate.epistemic_binding_notes = list(dict.fromkeys(binding_notes))[:12]
    for proposition_id in candidate.supporting_proposition_ids:
        if proposition_id in registered_hypotheses:
            continue
        ledger[proposition_id].mention_count += 1
        if proposition_id in candidate.decision_critical_proposition_ids:
            ledger[proposition_id].decision_critical_mentions += 1
    _apply_candidate_authority_cap(ledger, candidate)


def apply_side_premise_audit(
    ledger: dict[str, PropositionRecord],
    candidates: Iterable[Any],
    audit: dict[str, Any],
) -> None:
    """Attach independent audit findings and conservatively handle audit failure."""
    candidate_list = [
        candidate for candidate in candidates
        if bool(getattr(candidate, "schema_valid", True))
    ]
    by_specialist = {
        str(getattr(candidate, "specialist", "")): candidate
        for candidate in candidate_list
    }
    status = str((audit or {}).get("status", "UNAVAILABLE")).strip().upper()
    if status not in {"PASSED", "FINDINGS", "UNAVAILABLE"}:
        status = "UNAVAILABLE"
    if status == "UNAVAILABLE":
        error = " ".join(str((audit or {}).get("error", "")).split())[:220]
        for candidate in candidate_list:
            candidate.side_premise_audit_status = "UNAVAILABLE"
            candidate.weakest_decision_critical_status = "UNRESOLVED"
            candidate.decision_critical_dependency_claims = list(dict.fromkeys([
                *candidate.decision_critical_dependency_claims,
                "independent empirical-premise coverage remains unverified",
            ]))[:4]
            if str(getattr(candidate, "assumption_status", "")).upper() != "NORMATIVELY_CONTESTED":
                candidate.assumption_status = "UNDERDETERMINED"
            if str(getattr(candidate, "unresolved", "NONE")).upper() == "NONE":
                candidate.unresolved = "VERIFY_FACTS"
            candidate.selection_status = "PROVISIONAL"
            candidate.comparison_complete = False
            candidate.evidence_sufficient_for_action = False
            candidate.epistemic_confidence = min(candidate.epistemic_confidence, 0.50)
            candidate.confidence = candidate.epistemic_confidence
            candidate.epistemic_binding_notes = list(dict.fromkeys([
                *candidate.epistemic_binding_notes,
                f"Independent side-premise audit unavailable: {error or 'unknown error'}",
            ]))[:12]
        return

    findings_by_specialist: dict[str, list[dict[str, Any]]] = {}
    for raw in list((audit or {}).get("findings", []) or []):
        if not isinstance(raw, dict):
            continue
        specialist = str(raw.get("specialist", ""))
        candidate = by_specialist.get(specialist)
        if candidate is None:
            continue
        claim = " ".join(str(raw.get("claim", "")).split())[:240]
        if not claim:
            continue
        critical = raw.get("decision_critical") is True
        binding = str(raw.get("binding", "NEW_HYPOTHESIS"))
        derived_from = [
            str(value) for value in raw.get("derived_from", [])
            if str(value) in ledger
        ] if isinstance(raw.get("derived_from", []), list) else []
        already_cited = set(candidate.supporting_proposition_ids)
        already_critical = set(candidate.decision_critical_proposition_ids)
        covered_id = next((
            proposition_id for proposition_id, record in ledger.items()
            if record.epistemic_status in {"ESTABLISHED", "DERIVED"}
            and _claim_is_covered(claim, record.claim)
        ), "")
        if covered_id:
            binding = covered_id
        if binding in ledger:
            proposition_id = binding
            if proposition_id not in already_cited:
                ledger[proposition_id].mention_count += 1
            if critical and proposition_id not in already_critical:
                ledger[proposition_id].decision_critical_mentions += 1
        elif binding == "DERIVED_ESTABLISHED":
            proposition_id = register_derived_proposition(
                ledger, claim, specialist=specialist,
                derived_from=derived_from,
                decision_critical=critical,
            )
        else:
            proposition_id = register_hypothesis(
                ledger, claim, specialist=specialist,
                derived_from=derived_from or candidate.supporting_proposition_ids,
                decision_critical=critical,
            )
        candidate.supporting_proposition_ids = list(dict.fromkeys([
            *candidate.supporting_proposition_ids, proposition_id,
        ]))
        if critical:
            candidate.decision_critical_proposition_ids = list(dict.fromkeys([
                *candidate.decision_critical_proposition_ids, proposition_id,
            ]))
        normalized = {
            "claim": claim,
            "proposition_id": proposition_id,
            "binding": binding,
            "derived_from": derived_from,
            "decision_critical": critical,
            "source_field": " ".join(str(raw.get("source_field", "")).split())[:80],
            "reason": " ".join(str(raw.get("reason", "")).split())[:180],
        }
        findings_by_specialist.setdefault(specialist, []).append(normalized)

    for candidate in candidate_list:
        findings = findings_by_specialist.get(candidate.specialist, [])
        candidate.side_premise_audit_status = "FINDINGS" if findings else "PASSED"
        candidate.side_premise_audit_findings = findings[:12]
        _apply_candidate_authority_cap(ledger, candidate)


def focus_proposition_ids(
    ledger: dict[str, PropositionRecord], candidates: Iterable[Any], *, limit: int = 4,
) -> tuple[str, ...]:
    """Rank unresolved propositions for attention without promoting them."""
    agents_by_id: dict[str, set[str]] = {}
    for candidate in candidates:
        if not bool(getattr(candidate, "schema_valid", True)):
            continue
        agent = str(getattr(candidate, "specialist", "unknown"))
        for proposition_id in getattr(candidate, "decision_critical_proposition_ids", []):
            record = ledger.get(str(proposition_id))
            if record is None or record.epistemic_status not in {
                "REJECTED", "HYPOTHETICAL", "UNRESOLVED",
            }:
                continue
            agents_by_id.setdefault(record.proposition_id, set()).add(agent)
    ranked = sorted(
        agents_by_id,
        key=lambda proposition_id: (
            len(agents_by_id[proposition_id]),
            ledger[proposition_id].decision_critical_mentions,
            ledger[proposition_id].mention_count,
            proposition_id,
        ),
        reverse=True,
    )
    return tuple(ranked[:max(0, limit)])


def shared_unresolved_dependency_projection(
    ledger: dict[str, PropositionRecord], candidates: Iterable[Any],
) -> list[dict[str, Any]]:
    """Expose correlated support resting on the same unresolved proposition."""
    agents_by_id: dict[str, set[str]] = {}
    for candidate in candidates:
        if not bool(getattr(candidate, "schema_valid", True)):
            continue
        for proposition_id in getattr(candidate, "decision_critical_proposition_ids", []):
            record = ledger.get(str(proposition_id))
            if record is None or record.epistemic_status not in {
                "REJECTED", "HYPOTHETICAL", "UNRESOLVED",
            }:
                continue
            agents_by_id.setdefault(record.proposition_id, set()).add(
                str(getattr(candidate, "specialist", "unknown"))
            )
    rows = [{
        "proposition_id": proposition_id,
        "claim": ledger[proposition_id].claim,
        "epistemic_status": ledger[proposition_id].epistemic_status,
        "dependent_specialists": sorted(agents),
        "dependent_specialist_count": len(agents),
        "mention_count": ledger[proposition_id].mention_count,
    } for proposition_id, agents in agents_by_id.items()]
    return sorted(
        rows,
        key=lambda row: (
            int(row["dependent_specialist_count"]), int(row["mention_count"]),
            str(row["proposition_id"]),
        ),
        reverse=True,
    )
