"""Advisory repair deltas derived from the Stage-1 evidence packet."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


_STOP = {"a", "an", "and", "are", "be", "by", "is", "of", "the", "to", "was", "will"}


def _tokens(value: Any) -> set[str]:
    return {
        _stem(token) for token in re.findall(r"[a-z0-9%]+", str(value or "").casefold())
        if token not in _STOP
    }


def _stem(token: str) -> str:
    irregular = {"closed": "close", "closes": "close", "opened": "open", "opens": "open"}
    if token in irregular:
        return irregular[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _matches(left: Any, right: Any) -> bool:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    return overlap / len(left_tokens) >= 0.67 or overlap / len(right_tokens) >= 0.67


def build_evidence_repair_delta(
    candidate: Mapping[str, Any] | None,
    evidence_packet: Mapping[str, Any] | None,
    validation_issues: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Identify evidence/candidate disagreements without prescribing graph edits."""
    candidate = candidate or {}
    packet = evidence_packet or {}
    world = candidate.get("world_model") if isinstance(candidate, Mapping) else {}
    world = world if isinstance(world, Mapping) else {}
    effects = [row for row in world.get("effects") or [] if isinstance(row, Mapping)]

    ownership: dict[str, tuple[str, list[str]]] = {}
    for row in packet.get("ownership_annotations") or []:
        if not isinstance(row, Mapping):
            continue
        proposition_id = str(row.get("proposition_id") or "")
        ownership[proposition_id] = (
            str(row.get("status") or "").upper(),
            [str(value) for value in row.get("action_ids") or []],
        )

    omissions: list[dict[str, Any]] = []
    for observation in packet.get("source_observations") or []:
        if not isinstance(observation, Mapping):
            continue
        proposition_id = str(observation.get("proposition_id") or "")
        ownership_status, action_ids = ownership.get(proposition_id, ("", []))
        modality = str(observation.get("modality") or "").upper()
        effect_kind = str(observation.get("effect_kind") or "").upper()
        provenance = observation.get("provenance_binding") or {}
        provenance_status = str(
            provenance.get("status") if isinstance(provenance, Mapping) else ""
        ).upper()
        if (
            ownership_status != "OWNED"
            or len(action_ids) != 1
            or modality not in {"CERTAIN", "PROBABILISTIC"}
            or effect_kind in {"OBLIGATION", "DECISION", "CHOICE", "QUESTION"}
        ):
            continue
        if provenance_status and provenance_status not in {"BOUND", "EXACT"}:
            continue
        outcome = str(observation.get("outcome") or "")
        if re.search(r"\b(?:must|should)\s+(?:choose|decide|select)\b", outcome, re.I):
            continue
        missing_actions = [
            action_id for action_id in action_ids
            if not any(
                str(effect.get("action_id") or "") == action_id
                and _matches(outcome, " ".join(filter(None, [
                    str(effect.get("outcome") or ""),
                    str(effect.get("source_proposition") or ""),
                ])))
                for effect in effects
            )
        ]
        if missing_actions:
            omissions.append({
                "kind": "REQUIRED_OBSERVATION_OMISSION",
                "evidence_proposition_id": proposition_id,
                "action_ids": missing_actions,
                "outcome": outcome,
                "source_proposition": observation.get("source_proposition"),
                "clause_ids": list(observation.get("clause_ids") or []),
                "instruction": (
                    "Re-read the cited source. Add the observation only if it is "
                    "actually licensed for the named action; otherwise explain the conflict."
                ),
            })

    conditional_conflicts: list[dict[str, Any]] = []
    for rule in (packet.get("unresolved") or {}).get("conditional_rules") or []:
        if not isinstance(rule, Mapping):
            continue
        consequent = str(rule.get("consequent_span") or "")
        for effect in effects:
            if not _matches(consequent, effect.get("outcome")):
                continue
            if (
                str(effect.get("modality") or "").upper() == "CERTAIN"
                and not effect.get("condition_ids")
            ):
                conditional_conflicts.append({
                    "kind": "UNRESOLVED_CONDITION_FLATTENED",
                    "effect_id": effect.get("effect_id"),
                    "action_id": effect.get("action_id"),
                    "consequent": consequent,
                    "rule_source_span": rule.get("source_span"),
                    "instruction": (
                        "Do not make this consequence unconditional. Restore an "
                        "explicit condition or quarantine the consequence."
                    ),
                })

    issue_codes = list(dict.fromkeys(
        str(row.get("code") or "") for row in validation_issues
        if isinstance(row, Mapping) and row.get("code")
    ))
    return {
        "delta_version": "1.0",
        "authority": "ADVISORY_REPAIR_EVIDENCE",
        "triggering_validation_issue_codes": issue_codes,
        "required_observation_omissions": omissions,
        "conditionality_conflicts": conditional_conflicts,
        "has_findings": bool(omissions or conditional_conflicts),
        "constraints": [
            "This delta supplements validator repair cards; it does not replace them.",
            "Do not change an already-valid field merely to imitate Stage 1.",
            "Do not copy a derived hypothesis as a fact without source support.",
            "Keep unresolved conditions explicit; never flatten them to certainty.",
        ],
    }
