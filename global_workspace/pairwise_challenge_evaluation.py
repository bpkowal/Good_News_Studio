"""Gold-to-Stage-1 alignment and denominator-safe topology metrics.

The evaluator is deliberately independent of grounding admission.  It never
forces an alignment: exact matching is followed by conservative token/lemma
overlap, then optional semantic and adjudication callbacks supplied by a
caller.  Unresolved gold nodes remain scientific evidence of node loss or
alignment uncertainty rather than being silently attached to the nearest row.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import re
from typing import Any


POSITIVE_RELATIONS = {"CAUSES", "PREVENTS", "ENABLES"}
_STOP = {
    "a", "an", "and", "are", "be", "by", "does", "is", "of", "the", "to",
    "was", "were", "will",
}


def _normalize(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9%]+", str(value or "").casefold()))


def _lemma(token: str) -> str:
    irregular = {
        "dies": "die", "died": "die", "dying": "die",
        "causes": "cause", "caused": "cause", "causing": "cause",
        "receives": "receive", "received": "receive", "receiving": "receive",
        "stops": "stop", "stopped": "stop", "stopping": "stop",
        "hits": "hit", "hitting": "hit", "strikes": "strike", "struck": "strike",
        "pulling": "pull", "opening": "open", "closing": "close",
        "activating": "activate", "issuing": "issue", "diverting": "divert",
        "sending": "send", "leaving": "leave",
        "sent": "send", "kept": "keep", "powered": "power",
        "rises": "rise", "rose": "rise", "fallen": "fall", "falls": "fall",
    }
    if token in irregular:
        return irregular[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        root = token[:-3]
        return root[:-1] if len(root) > 2 and root[-1:] == root[-2:-1] else root
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(value: Any) -> set[str]:
    return {
        _lemma(token) for token in _normalize(value).split()
        if token not in _STOP
    }


def _candidate_text(row: Mapping[str, Any]) -> str:
    return " ".join(filter(None, [
        str(row.get("outcome") or ""),
        str(row.get("source_proposition") or ""),
    ]))


def align_gold_events(
    skeleton: Mapping[str, Any],
    gold_pairs: Sequence[Mapping[str, Any]],
    *,
    semantic_scorer: Callable[[str, Mapping[str, Any]], float] | None = None,
    adjudicator: Callable[[str, str, Sequence[Mapping[str, Any]]], str | None] | None = None,
    token_threshold: float = 0.67,
    token_margin: float = 0.15,
    containment_threshold: float = 0.75,
    containment_margin: float = 0.15,
    semantic_threshold: float = 0.82,
    semantic_margin: float = 0.05,
) -> list[dict[str, Any]]:
    """Align every unique gold endpoint within its action, without forcing ties."""
    propositions = [
        dict(row) for row in skeleton.get("propositions") or []
        if isinstance(row, Mapping) and row.get("proposition_id")
    ]
    endpoints: list[tuple[str, str]] = []
    for pair in gold_pairs:
        action_id = str(pair.get("action_id") or "")
        for field in ("source_event", "target_event"):
            endpoint = (action_id, str(pair.get(field) or ""))
            if endpoint[1] and endpoint not in endpoints:
                endpoints.append(endpoint)

    alignments: list[dict[str, Any]] = []
    for action_id, gold_event in endpoints:
        candidates = [
            row for row in propositions
            if str(row.get("action_id") or "") == action_id
        ]
        gold_normalized = _normalize(gold_event)
        exact = [
            row for row in candidates
            if gold_normalized
            and (
                gold_normalized == _normalize(row.get("outcome"))
                or gold_normalized == _normalize(row.get("source_proposition"))
                or gold_normalized in _normalize(_candidate_text(row))
            )
        ]
        matched: Mapping[str, Any] | None = exact[0] if len(exact) == 1 else None
        method = "EXACT" if matched else ""
        score: float | None = 1.0 if matched else None

        if matched is None and candidates:
            gold_tokens = _tokens(gold_event)
            ranked: list[tuple[float, Mapping[str, Any]]] = []
            for row in candidates:
                candidate_tokens = _tokens(_candidate_text(row))
                union = gold_tokens | candidate_tokens
                ranked.append((
                    len(gold_tokens & candidate_tokens) / len(union) if union else 0.0,
                    row,
                ))
            ranked.sort(key=lambda item: item[0], reverse=True)
            best = ranked[0][0]
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            if best >= token_threshold and best - runner_up >= token_margin:
                score, matched = ranked[0]
                method = "TOKEN_LEMMA"

        if matched is None and candidates:
            gold_tokens = _tokens(gold_event)
            ranked = []
            for row in candidates:
                candidate_tokens = _tokens(_candidate_text(row))
                ranked.append((
                    len(gold_tokens & candidate_tokens) / len(gold_tokens)
                    if gold_tokens else 0.0,
                    row,
                ))
            ranked.sort(key=lambda item: item[0], reverse=True)
            best = ranked[0][0]
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            if best >= containment_threshold and best - runner_up >= containment_margin:
                score, matched = ranked[0]
                method = "GOLD_LEMMA_CONTAINMENT"

        if matched is None and semantic_scorer is not None and candidates:
            ranked = sorted(
                ((float(semantic_scorer(gold_event, row)), row) for row in candidates),
                key=lambda item: item[0], reverse=True,
            )
            best = ranked[0][0]
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            if best >= semantic_threshold and best - runner_up >= semantic_margin:
                score, matched = ranked[0]
                method = "SEMANTIC_SIMILARITY"

        if matched is None and adjudicator is not None and candidates:
            selected_id = adjudicator(gold_event, action_id, candidates)
            selected = [
                row for row in candidates
                if str(row.get("proposition_id") or "") == str(selected_id or "")
            ]
            if len(selected) == 1:
                matched = selected[0]
                method = "ADJUDICATED"
                score = None

        alignments.append({
            "action_id": action_id,
            "gold_event": gold_event,
            "matched_proposition_id": (
                str(matched.get("proposition_id") or "") if matched else None
            ),
            "method": method or None,
            "score": score,
            "status": "MATCHED" if matched else "UNRESOLVED",
            "candidate_count": len(candidates),
            "unresolved_reason": (
                None if matched else
                "NO_ACTION_NODES" if not candidates else
                "NODE_ABSENT_OR_ALIGNMENT_UNCERTAIN"
            ),
        })
    return alignments


def score_aligned_challenge(
    audit: Mapping[str, Any],
    gold_pairs: Sequence[Mapping[str, Any]],
    alignments: Sequence[Mapping[str, Any]],
    *,
    gold_edge_set_complete: bool = False,
) -> dict[str, Any]:
    """Compare scaffold and pair audit only where both gold nodes are aligned."""
    aligned = {
        (str(row.get("action_id") or ""), str(row.get("gold_event") or "")):
            str(row.get("matched_proposition_id") or "")
        for row in alignments
        if row.get("status") == "MATCHED" and row.get("matched_proposition_id")
    }
    audit_rows = {
        (
            str(row.get("action_id") or ""),
            str(row.get("source_proposition_id") or ""),
            str(row.get("target_proposition_id") or ""),
        ): row
        for row in audit.get("records") or []
        if isinstance(row, Mapping) and row.get("status") == "JUDGED"
    }
    gold_node_keys = {
        (str(pair.get("action_id") or ""), str(pair.get(field) or ""))
        for pair in gold_pairs for field in ("source_event", "target_event")
        if pair.get(field)
    }
    aligned_gold_edges: dict[tuple[str, str, str], str] = {}
    details: list[dict[str, Any]] = []
    scaffold_correct = audit_correct = 0
    novelty_denominator = novelty_recovered = 0
    for pair in gold_pairs:
        action_id = str(pair.get("action_id") or "")
        source_id = aligned.get((action_id, str(pair.get("source_event") or "")))
        target_id = aligned.get((action_id, str(pair.get("target_event") or "")))
        if not source_id or not target_id:
            continue
        expected = str(pair.get("relation") or "").upper()
        key = (action_id, source_id, target_id)
        aligned_gold_edges[key] = expected
        row = audit_rows.get(key, {})
        audit_relation = str(row.get("relation") or "")
        scaffold_relations = set(row.get("scaffold_relations") or [])
        scaffold_match = (
            expected in scaffold_relations
            or (expected == "NONE" and not scaffold_relations)
        )
        audit_match = audit_relation == expected
        scaffold_correct += int(scaffold_match)
        audit_correct += int(audit_match)
        scaffold_missed_positive = (
            expected in POSITIVE_RELATIONS and expected not in scaffold_relations
        )
        novelty_denominator += int(scaffold_missed_positive)
        novelty_recovered += int(scaffold_missed_positive and audit_match)
        details.append({
            "action_id": action_id,
            "source_proposition_id": source_id,
            "target_proposition_id": target_id,
            "gold_relation": expected,
            "scaffold_relations": sorted(scaffold_relations),
            "audit_relation": audit_relation or None,
            "scaffold_correct": scaffold_match,
            "audit_correct": audit_match,
        })

    audit_only = [
        (key, row) for key, row in audit_rows.items()
        if not row.get("scaffold_relations")
        and str(row.get("relation") or "") in POSITIVE_RELATIONS
    ]
    correct_audit_only = sum(
        aligned_gold_edges.get(key) == str(row.get("relation") or "")
        for key, row in audit_only
    )
    aligned_edge_count = len(details)
    return {
        "gold_node_count": len(gold_node_keys),
        "aligned_gold_node_count": sum(key in aligned for key in gold_node_keys),
        "node_recall": (
            sum(key in aligned for key in gold_node_keys) / len(gold_node_keys)
            if gold_node_keys else None
        ),
        "gold_edge_count": len(gold_pairs),
        "aligned_gold_edge_count": aligned_edge_count,
        "relation_recall_conditional_on_node_availability": (
            audit_correct / aligned_edge_count if aligned_edge_count else None
        ),
        "pairwise_gold_accuracy_given_aligned_nodes": (
            audit_correct / aligned_edge_count if aligned_edge_count else None
        ),
        "scaffold_gold_accuracy_given_aligned_nodes": (
            scaffold_correct / aligned_edge_count if aligned_edge_count else None
        ),
        "gold_relations_absent_from_scaffold": novelty_denominator,
        "correct_audit_only_relations": novelty_recovered,
        "audit_novelty_rate": (
            novelty_recovered / novelty_denominator if novelty_denominator else None
        ),
        "audit_only_relation_count": len(audit_only),
        "correct_audit_only_relation_count": correct_audit_only,
        "audit_only_precision": (
            correct_audit_only / len(audit_only)
            if gold_edge_set_complete and audit_only else None
        ),
        "audit_only_precision_status": (
            "MEASURED" if gold_edge_set_complete
            else "NOT_IDENTIFIED_SPARSE_GOLD"
        ),
        "details": details,
    }


def diagnose_node_inventory(
    skeleton: Mapping[str, Any],
    gold_pairs: Sequence[Mapping[str, Any]],
    alignments: Sequence[Mapping[str, Any]],
    *,
    token_threshold: float = 0.67,
    token_margin: float = 0.15,
) -> dict[str, Any]:
    """Diagnose presence, ownership, atomicity, branches, and unresolved text.

    Wrong-owner detection reuses only exact and conservative token matching over
    propositions outside the expected action. It does not promote those nodes
    into relation scoring.
    """
    propositions = [
        dict(row) for row in skeleton.get("propositions") or []
        if isinstance(row, Mapping) and row.get("proposition_id")
    ]
    proposition_by_id = {
        str(row.get("proposition_id") or ""): row for row in propositions
    }
    expected = {
        (str(pair.get("action_id") or ""), str(pair.get(field) or ""))
        for pair in gold_pairs for field in ("source_event", "target_event")
        if pair.get(field)
    }
    initial = {
        (str(row.get("action_id") or ""), str(row.get("gold_event") or "")): row
        for row in alignments
    }
    rows: list[dict[str, Any]] = []
    for action_id, gold_event in sorted(expected):
        alignment = initial.get((action_id, gold_event), {})
        matched_id = str(alignment.get("matched_proposition_id") or "")
        matched = proposition_by_id.get(matched_id)
        status = "PRESENT_CORRECT_OWNER" if matched else "ABSENT_OR_UNALIGNED"
        method = alignment.get("method")
        score = alignment.get("score")
        if matched is None:
            outside = [
                row for row in propositions
                if str(row.get("action_id") or "") != action_id
            ]
            gold_normalized = _normalize(gold_event)
            exact = [
                row for row in outside
                if gold_normalized
                and (
                    gold_normalized == _normalize(row.get("outcome"))
                    or gold_normalized == _normalize(row.get("source_proposition"))
                    or gold_normalized in _normalize(_candidate_text(row))
                )
            ]
            if len(exact) == 1:
                matched = exact[0]
                status, method, score = "PRESENT_WRONG_OWNER", "EXACT_GLOBAL", 1.0
            elif outside:
                gold_tokens = _tokens(gold_event)
                ranked = []
                for row in outside:
                    candidate_tokens = _tokens(_candidate_text(row))
                    union = gold_tokens | candidate_tokens
                    ranked.append((
                        len(gold_tokens & candidate_tokens) / len(union)
                        if union else 0.0,
                        row,
                    ))
                ranked.sort(key=lambda item: item[0], reverse=True)
                best = ranked[0][0]
                runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
                if best >= token_threshold and best - runner_up >= token_margin:
                    score, matched = ranked[0]
                    status, method = "PRESENT_WRONG_OWNER", "TOKEN_LEMMA_GLOBAL"
        rows.append({
            "expected_action_id": action_id,
            "gold_event": gold_event,
            "status": status,
            "matched_proposition_id": (
                str(matched.get("proposition_id") or "") if matched else None
            ),
            "actual_action_id": (
                str(matched.get("action_id") or "") if matched else None
            ),
            "method": method,
            "score": score,
        })

    node_to_gold: dict[str, list[str]] = {}
    for row in rows:
        proposition_id = str(row.get("matched_proposition_id") or "")
        if proposition_id:
            node_to_gold.setdefault(proposition_id, []).append(str(row["gold_event"]))
    for row in rows:
        proposition_id = str(row.get("matched_proposition_id") or "")
        row["atomicity"] = (
            "UNAVAILABLE" if not proposition_id
            else "FUSED" if len(set(node_to_gold[proposition_id])) > 1
            else "ATOMIC"
        )

    expected_actions = sorted({action_id for action_id, _ in expected})
    branches = []
    for action_id in expected_actions:
        branch_rows = [row for row in rows if row["expected_action_id"] == action_id]
        correct = sum(row["status"] == "PRESENT_CORRECT_OWNER" for row in branch_rows)
        branches.append({
            "action_id": action_id,
            "required_node_count": len(branch_rows),
            "correctly_owned_node_count": correct,
            "status": "REPRESENTED" if correct == len(branch_rows) else "OMITTED_OR_INCOMPLETE",
        })
    correct_count = sum(row["status"] == "PRESENT_CORRECT_OWNER" for row in rows)
    wrong_count = sum(row["status"] == "PRESENT_WRONG_OWNER" for row in rows)
    fused_count = sum(row["atomicity"] == "FUSED" for row in rows)
    unresolved_spans = list(skeleton.get("unresolved_source_spans") or [])
    unresolved_ellipsis = [
        row for row in skeleton.get("ellipsis_resolutions") or []
        if isinstance(row, Mapping)
        and str(row.get("status") or "").upper() != "RESOLVED"
    ]
    return {
        "gold_node_count": len(rows),
        "present_correct_owner_count": correct_count,
        "present_wrong_owner_count": wrong_count,
        "absent_or_unaligned_count": len(rows) - correct_count - wrong_count,
        "node_recall_correct_owner": correct_count / len(rows) if rows else None,
        "node_presence_recall_any_owner": (
            (correct_count + wrong_count) / len(rows) if rows else None
        ),
        "fused_gold_node_count": fused_count,
        "atomic_present_gold_node_count": sum(
            row["atomicity"] == "ATOMIC" for row in rows
        ),
        "unresolved_source_span_count": len(unresolved_spans),
        "unresolved_ellipsis_count": len(unresolved_ellipsis),
        "action_branches": branches,
        "nodes": rows,
    }


def diagnose_neutral_node_inventory(
    neutral_skeleton: Mapping[str, Any],
    gold_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure semantic node presence before action ownership is attempted."""
    actions = sorted({str(row.get("action_id") or "") for row in gold_pairs})
    projected = {"propositions": []}
    for proposition in neutral_skeleton.get("propositions") or []:
        if not isinstance(proposition, Mapping):
            continue
        for action_id in actions:
            row = dict(proposition)
            row["action_id"] = action_id
            row["proposition_id"] = (
                f"{proposition.get('proposition_id')}__NEUTRAL__{action_id}"
            )
            projected["propositions"].append(row)
    alignments = align_gold_events(projected, gold_pairs)
    matched = [row for row in alignments if row.get("status") == "MATCHED"]
    node_to_gold: dict[str, list[str]] = {}
    for row in matched:
        neutral_id = str(row.get("matched_proposition_id") or "").split(
            "__NEUTRAL__", 1,
        )[0]
        node_to_gold.setdefault(neutral_id, []).append(str(row.get("gold_event") or ""))
    gold_count = len(alignments)
    return {
        "gold_node_count": gold_count,
        "present_neutral_node_count": len(matched),
        "neutral_node_recall": len(matched) / gold_count if gold_count else None,
        "fused_gold_node_count": sum(
            len(set(events)) for events in node_to_gold.values()
            if len(set(events)) > 1
        ),
        "alignments": alignments,
    }
