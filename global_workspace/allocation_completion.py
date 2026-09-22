"""Source-licensed completion of exclusive allocation action worlds."""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any


_WORD = re.compile(r"[a-z][a-z0-9-]*", re.I)
_TRANSFER = re.compile(
    r"\b(?:give|gives|given|receive|receives|allocate|assign|send|deliver|"
    r"administer|provide|supply)\w*\b", re.I,
)
_NONRECEIPT = re.compile(
    r"\b(?:no|not|without|does\s+not|do\s+not|is\s+not)\b[^.;]{0,50}"
    r"\b(?:receive|receives|given|antidote|medicine|dose|resource)\b|"
    r"\b(?:receive|receives)\s+no\b", re.I,
)
_STOP = {
    "a", "an", "and", "be", "give", "given", "gives", "giving", "it",
    "to", "the", "receive", "receives", "received", "patient", "person",
}
_WELFARE_KINDS = {"PERSON", "HUMAN", "GROUP", "POPULATION", "HOUSEHOLD"}


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold().rstrip("s")
        for token in _WORD.findall(str(value or ""))
        if token.casefold() not in _STOP
    }


def _party_recipients(
    parties: Sequence[Mapping[str, Any]], actions: Mapping[str, str],
) -> dict[str, str]:
    recipients: dict[str, str] = {}
    for action_id, action_text in actions.items():
        matches = []
        folded = str(action_text).casefold()
        for party in parties:
            label = str(party.get("label") or "").strip()
            if (
                label
                and str(party.get("kind") or "").upper() in _WELFARE_KINDS
                and label.casefold() in folded
            ):
                matches.append((len(label), str(party.get("party_id") or "")))
        if matches:
            recipients[action_id] = max(matches)[1]
    return recipients


def _shared_resource_tokens(
    actions: Mapping[str, str], parties: Sequence[Mapping[str, Any]],
    recipients: Mapping[str, str],
) -> set[str]:
    party_by_id = {str(row.get("party_id") or ""): row for row in parties}
    signatures = []
    for action_id, text in actions.items():
        values = _tokens(text)
        recipient = party_by_id.get(recipients.get(action_id, ""), {})
        values -= _tokens(recipient.get("label"))
        signatures.append(values)
    return set.intersection(*signatures) if len(signatures) >= 2 else set()


def _constraint_provenance(
    clauses: Sequence[Mapping[str, Any]], profile: Mapping[str, Any],
) -> tuple[str, str]:
    patterns = []
    if profile.get("explicit_capacity_or_indivisibility"):
        patterns.append(re.compile(
            r"\b(?:one|single|sole|only|last|cannot|can't|indivisible|not\s+enough)\b",
            re.I,
        ))
    if profile.get("explicit_recipient_exclusion"):
        patterns.append(re.compile(r"\b(?:not\s+both|mutually\s+exclusive|the\s+other)\b", re.I))
    if profile.get("explicit_alternative_choice"):
        patterns.append(re.compile(r"\b(?:either|or|choose|must)\b", re.I))
    for pattern in patterns:
        for clause in clauses:
            text = str(clause.get("text") or "")
            if pattern.search(text):
                return str(clause.get("clause_id") or ""), text
    return "", ""


def complete_exclusive_allocations(
    skeleton: Mapping[str, Any], *, actions: Mapping[str, str],
    clauses: Sequence[Mapping[str, Any]], generation_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Add only closed allocation complements licensed by source constraints."""
    result = copy.deepcopy(dict(skeleton))
    parties = [row for row in result.get("parties") or [] if isinstance(row, dict)]
    propositions = [
        row for row in result.get("propositions") or [] if isinstance(row, dict)
    ]
    profile = dict(generation_contract.get("allocation_constraint_profile") or {})
    recipients = _party_recipients(parties, actions)
    shared_resource = _shared_resource_tokens(actions, parties, recipients)
    licensed = bool(
        len(recipients) >= 2
        and len(set(recipients.values())) >= 2
        and not profile.get("explicit_nonexclusive_allocation")
        and (
            profile.get("explicit_capacity_or_indivisibility")
            or profile.get("explicit_recipient_exclusion")
            or (
                profile.get("explicit_alternative_choice")
                and bool(shared_resource)
            )
        )
    )
    audit: list[dict[str, Any]] = [{
        "status": "LICENSED" if licensed else "NOT_LICENSED",
        "recipients": dict(recipients),
        "shared_resource_tokens": sorted(shared_resource),
        "constraint_profile": profile,
    }]
    if not licensed:
        result["allocation_completions"] = audit
        return result, audit

    constraint_clause_id, constraint_span = _constraint_provenance(
        clauses, profile,
    )
    existing_ids = {str(row.get("proposition_id") or "") for row in propositions}
    party_by_id = {str(row.get("party_id") or ""): row for row in parties}
    for action_id, selected_id in recipients.items():
        action_text = str(actions[action_id])
        selected_label = str((party_by_id.get(selected_id) or {}).get("label") or selected_id)
        own = [row for row in propositions if str(row.get("action_id") or "") == action_id]
        transfer_candidates = [
            row for row in own
            if str(row.get("party_id") or "") == selected_id
            and _TRANSFER.search(" ".join((
                str(row.get("outcome") or ""),
                str(row.get("source_proposition") or ""),
            )))
            and not _NONRECEIPT.search(" ".join((
                str(row.get("outcome") or ""),
                str(row.get("source_proposition") or ""),
            )))
        ]
        transfer = max(
            transfer_candidates,
            key=lambda row: (
                int(action_id in {str(value) for value in row.get("clause_ids") or []}),
                int(str((row.get("provenance_binding") or {}).get("status") or "") == "BOUND"),
                int(str(row.get("effect_kind") or "") == "INTERVENTION"),
            ),
            default=None,
        )
        if transfer is None:
            proposition_id = f"ALLOC_{action_id}_{selected_id}"
            transfer = {
                "proposition_id": proposition_id,
                "neutral_proposition_id": None,
                "action_id": action_id,
                "party_id": selected_id,
                "outcome": action_text,
                "polarity": "BENEFICIAL",
                "directness": "DIRECT",
                "modality": "CERTAIN",
                "effect_kind": "RESOURCE_TRANSFER",
                "quantities": [],
                "source_proposition": action_text,
                "clause_ids": [action_id],
                "derivation_operation": "DIRECT_COPY",
            }
            propositions.append(transfer)
            existing_ids.add(proposition_id)
        else:
            transfer["party_id"] = selected_id
            transfer["directness"] = "DIRECT"
            transfer["modality"] = "CERTAIN"
            transfer["effect_kind"] = "RESOURCE_TRANSFER"
        transfer_id = str(transfer.get("proposition_id") or "")

        for nonrecipient_id in sorted(set(recipients.values()) - {selected_id}):
            nonrecipient_label = str(
                (party_by_id.get(nonrecipient_id) or {}).get("label") or nonrecipient_id
            )
            existing = next((
                row for row in own
                if str(row.get("party_id") or "") == nonrecipient_id
                and _NONRECEIPT.search(" ".join((
                    str(row.get("outcome") or ""),
                    str(row.get("source_proposition") or ""),
                )))
            ), None)
            if existing is not None:
                existing["directness"] = "DOWNSTREAM"
                existing["modality"] = "CERTAIN"
                existing["effect_kind"] = "OTHER"
                existing["source_effect_ids"] = [transfer_id]
                existing["derivation_operation"] = "SOURCE_STIPULATED_CAUSAL"
                proposition_id = str(existing.get("proposition_id") or "")
                source_kind = "EXPLICIT_NONRECEIPT"
            else:
                proposition_id = f"ALLOC_NONE_{action_id}_{nonrecipient_id}"
                suffix = 2
                base_id = proposition_id
                while proposition_id in existing_ids:
                    proposition_id = f"{base_id}_{suffix}"
                    suffix += 1
                existing_ids.add(proposition_id)
                existing = {
                    "proposition_id": proposition_id,
                    "neutral_proposition_id": None,
                    "action_id": action_id,
                    "party_id": nonrecipient_id,
                    "outcome": f"{nonrecipient_label} does not receive the allocated resource",
                    "polarity": "ADVERSE",
                    "directness": "DOWNSTREAM",
                    "modality": "CERTAIN",
                    "effect_kind": "OTHER",
                    "quantities": [],
                    "source_proposition": constraint_span,
                    "clause_ids": [constraint_clause_id],
                    "source_effect_ids": [transfer_id],
                    "derivation_operation": "EXCLUSIVE_ALLOCATION_COMPLEMENT",
                }
                propositions.append(existing)
                source_kind = "STRUCTURAL_COMPLEMENT"
            audit.append({
                "status": "COMPLETED",
                "action_id": action_id,
                "selected_recipient_id": selected_id,
                "selected_recipient_label": selected_label,
                "transfer_proposition_id": transfer_id,
                "nonrecipient_id": nonrecipient_id,
                "nonreceipt_proposition_id": proposition_id,
                "source_kind": source_kind,
                "constraint_clause_id": constraint_clause_id,
                "shared_resource_tokens": sorted(shared_resource),
            })
    result["propositions"] = propositions
    result["allocation_completions"] = audit
    return result, audit
