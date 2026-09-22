"""Conservative topology hypotheses over an admitted factual skeleton."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

_EXPLICIT_RELATIONS = (
    (re.compile(r"\b(?:prevent|prevents|prevented|preventing)\b", re.I), "PREVENTS"),
    (re.compile(r"\b(?:enable|enables|enabled|enabling|allow|allows|allowed)\b", re.I), "ENABLES"),
    (re.compile(r"\b(?:cause|causes|caused|causing|results?\s+in|leads?\s+to)\b", re.I), "CAUSES"),
)
_SEQUENCE_CUES = re.compile(
    r"\b(?:thereby|so\s+that|and\s+then|which|where|leaving|moving|diverting|"
    r"resulting\s+in|causing|enabling|preventing)\b",
    re.IGNORECASE,
)
_PROCESS_KINDS = {"PHYSICAL_STATE", "INSTITUTIONAL_OUTCOME", "OTHER"}
_TERMINAL_KINDS = {"HEALTH_OUTCOME", "WELFARE_OUTCOME", "CAPABILITY_CHANGE"}


def _fold(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _relation_between(text: str, left: str, right: str) -> tuple[str, str]:
    blob = _fold(text)
    left_pos = blob.find(_fold(left))
    right_pos = blob.find(_fold(right))
    if left_pos < 0 or right_pos < 0 or left_pos == right_pos:
        return "", ""
    if left_pos > right_pos:
        return "", ""
    bridge = blob[left_pos + len(_fold(left)):right_pos]
    for pattern, relation in _EXPLICIT_RELATIONS:
        match = pattern.search(bridge)
        if match:
            return relation, "EXPLICIT_CAUSAL"
    if _SEQUENCE_CUES.search(bridge) or re.search(r"[,;:]", bridge):
        return "CAUSES", "SYNTACTIC_EVENT_SEQUENCE"
    return "", ""


def build_topology_scaffold(
    skeleton: Mapping[str, Any],
    clauses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return licensed candidates and minimal hypotheses without choosing one."""
    clause_by_id = {
        str(row.get("clause_id") or ""): str(row.get("text") or "")
        for row in clauses
    }
    propositions = [
        row for row in skeleton.get("propositions") or [] if isinstance(row, Mapping)
    ]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(source: Mapping[str, Any], target: Mapping[str, Any], *, relation: str,
            support_type: str, clause_id: str = "", span: str = "",
            commitment: str = "CANDIDATE") -> None:
        key = (
            str(source.get("proposition_id") or ""),
            str(target.get("proposition_id") or ""), relation,
        )
        if not all(key):
            return
        if key in seen:
            if commitment == "SOURCE_LICENSED":
                existing = next(
                    row for row in candidates
                    if (
                        row["source_proposition_id"],
                        row["target_proposition_id"],
                        row["relation"],
                    ) == key
                )
                existing.update({
                    "support_type": support_type,
                    "clause_id": clause_id,
                    "support_span": span,
                    "commitment": commitment,
                })
            return
        seen.add(key)
        candidates.append({
            "source_proposition_id": key[0],
            "target_proposition_id": key[1],
            "relation": relation,
            "support_type": support_type,
            "clause_id": clause_id,
            "support_span": span,
            "commitment": commitment,
        })

    for left in propositions:
        for right in propositions:
            if left is right or left.get("action_id") != right.get("action_id"):
                continue
            shared = [
                clause_id for clause_id in left.get("clause_ids") or []
                if clause_id in set(right.get("clause_ids") or [])
            ]
            for clause_id in shared:
                clause = clause_by_id.get(str(clause_id), "")
                relation, support_type = _relation_between(
                    clause,
                    str(left.get("source_proposition") or ""),
                    str(right.get("source_proposition") or ""),
                )
                if relation:
                    add(
                        left, right, relation=relation,
                        support_type=support_type, clause_id=str(clause_id),
                        span=clause,
                        commitment=(
                            "SOURCE_LICENSED" if support_type == "EXPLICIT_CAUSAL"
                            else "CANDIDATE"
                        ),
                    )
            if (
                left.get("directness") == "DIRECT"
                and right.get("directness") == "DOWNSTREAM"
            ):
                add(
                    left, right, relation="CAUSES",
                    support_type="SHARED_ACTION_CONSEQUENCE",
                    commitment="REQUIRES_STAGE2_JUSTIFICATION",
                )
            if (
                left.get("effect_kind") in _PROCESS_KINDS
                and left.get("directness") == "DOWNSTREAM"
                and right.get("effect_kind") in _TERMINAL_KINDS
                and right.get("directness") == "DOWNSTREAM"
            ):
                add(
                    left, right, relation="CAUSES",
                    support_type="STRUCTURALLY_PLAUSIBLE_INTERMEDIATE",
                    commitment="REQUIRES_STAGE2_JUSTIFICATION",
                )

    proposition_by_id = {
        str(row.get("proposition_id") or ""): row for row in propositions
    }
    for rule in skeleton.get("conditional_rules") or []:
        if not isinstance(rule, Mapping) or rule.get("status") != "INSTANTIATED":
            continue
        source = proposition_by_id.get(str(rule.get("antecedent_proposition_id") or ""))
        target = proposition_by_id.get(str(rule.get("consequent_proposition_id") or ""))
        if source and target and source.get("action_id") == target.get("action_id"):
            support = proposition_by_id.get(
                str(rule.get("antecedent_support_proposition_id") or "")
            )
            if support and support.get("action_id") == source.get("action_id"):
                add(
                    support, source, relation="CAUSES",
                    support_type="CESSATION_TO_NEGATED_STATE",
                    clause_id=str(rule.get("clause_id") or ""),
                    span=str(rule.get("source_span") or ""),
                    commitment="SOURCE_LICENSED",
                )
            add(
                source, target, relation=str(rule.get("relation") or "CAUSES"),
                support_type="INSTANTIATED_CONDITIONAL_RULE",
                clause_id=str(rule.get("clause_id") or ""),
                span=str(rule.get("source_span") or ""),
                commitment="SOURCE_LICENSED",
            )
    for completion in skeleton.get("allocation_completions") or []:
        if not isinstance(completion, Mapping) or completion.get("status") != "COMPLETED":
            continue
        source = proposition_by_id.get(str(completion.get("transfer_proposition_id") or ""))
        target = proposition_by_id.get(str(completion.get("nonreceipt_proposition_id") or ""))
        if source and target and source.get("action_id") == target.get("action_id"):
            add(
                source, target, relation="CAUSES",
                support_type="EXCLUSIVE_ALLOCATION_COMPLEMENT",
                clause_id=str(completion.get("constraint_clause_id") or ""),
                commitment="SOURCE_LICENSED",
            )
    by_action: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        source = next(
            (row for row in propositions if row.get("proposition_id") == candidate["source_proposition_id"]),
            {},
        )
        by_action.setdefault(str(source.get("action_id") or ""), []).append(candidate)
    hypotheses: list[dict[str, Any]] = []
    for action_id, edges in sorted(by_action.items()):
        licensed = [edge for edge in edges if edge["commitment"] == "SOURCE_LICENSED"]
        hypotheses.append({
            "hypothesis_id": f"{action_id}_MINIMAL_SOURCE_LICENSED",
            "action_id": action_id,
            "edge_keys": [
                [edge["source_proposition_id"], edge["relation"], edge["target_proposition_id"]]
                for edge in licensed
            ],
            "status": "ADMISSIBLE_PARTIAL" if licensed else "UNRESOLVED",
            "note": (
                "This is a partial topology. Stage 2 may add a candidate edge only "
                "with exact source support and may leave ambiguous ancestry unresolved."
            ),
        })
    return {
        "scaffold_version": "1.0",
        "policy": (
            "Candidates are permissions to evaluate, not mandatory edges. Only "
            "SOURCE_LICENSED edges may be committed without further justification."
        ),
        "edge_candidates": candidates,
        "minimal_hypotheses": hypotheses,
    }
