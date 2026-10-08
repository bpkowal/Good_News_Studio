"""Relative interpretive support over competing cloze world graphs.

Cloze remains constrained abductive probing: a blank asks which source span
makes a template proposition true. This module scores a cap-3 beam of distinct
compiled graphs and collapses to one frozen world only when support separates
or occurrence conclusions are invariant. Weights are relative support, not
calibrated probabilities.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from blueprint_evidence_graph import (
    clause_ids_for_span,
    hypothesis,
)
from z10_world_model_adapter import segment_source_clauses
from blueprint_selection_review import semantic_signature


BEAM_CAP = 3
FALLBACK_VARIANTS = {"flexible", "composition"}
COLLAPSE_SEPARATED = "SEPARATED"
COLLAPSE_INVARIANT = "INVARIANT"
COLLAPSE_UNRESOLVED = "UNRESOLVED_BEAM"
PIPELINE_INTERPRETIVE_BEAM = "INTERPRETIVE_BEAM"


def slot_hypothesis(*, text: str, slot: str, check: str, span: str,
                    producer: str, relent_compatibility: str,
                    z10_candidate_ids: Sequence[str] = (),
                    index: int = 0) -> dict[str, Any]:
    """A typed fill for one cloze blank. Losing hyps do not mint world effects."""
    clauses = segment_source_clauses(text) if text and span else []
    clause_ids = clause_ids_for_span(clauses, span) if span else []
    if check == "implied" and producer == "llm_note":
        provenance = "WORLD_KNOWLEDGE_HYPOTHESIS"
    elif relent_compatibility == "copy_ok" and producer == "z10":
        provenance = "STRUCTURALLY_DERIVED" if z10_candidate_ids else "SOURCE_ASSERTED"
    elif relent_compatibility == "copy_ok":
        provenance = "SOURCE_ASSERTED"
    else:
        provenance = "UNRESOLVED"
    ident = f"slot:{slot}:{producer}:{index}"
    row = hypothesis(
        hypothesis_id=ident,
        producer=producer,
        span=span,
        clause_ids=clause_ids,
        z10_candidate_ids=z10_candidate_ids,
        hypothesis={
            "kind": "slot_fill",
            "slot": slot,
            "check": check,
            "source_span": span,
            "semantic_type": slot,
            "parser_support": list(z10_candidate_ids),
            "relent_compatibility": relent_compatibility,
        },
        scope={"contexts": [{"kind": "clause", "clause_ids": clause_ids}]}
        if clause_ids else None,
        provenance=provenance,
    )
    row.update(
        source_span=span,
        semantic_type=slot,
        parser_support=list(z10_candidate_ids),
        relent_compatibility=relent_compatibility,
        competing_ids=[],
    )
    return row


def mark_competitors(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ids = [row.get("hypothesis_id") for row in rows if row.get("hypothesis_id")]
    marked = []
    for row in rows:
        item = dict(row)
        ident = item.get("hypothesis_id")
        item["competing_ids"] = [other for other in ids if other != ident]
        hypothesis = dict(item.get("hypothesis") or {})
        hypothesis["competing_ids"] = list(item["competing_ids"])
        item["hypothesis"] = hypothesis
        marked.append(item)
    return marked


def graph_identity(blueprint_id: str, assignment: Sequence[str] | None,
                   withheld: bool) -> tuple:
    return (blueprint_id or "", tuple(assignment or ()), bool(withheld))


def occurrence_signature(proposal: Mapping[str, Any] | None) -> tuple:
    world = ((proposal or {}).get("candidate") or {}).get("world_model") or {}
    actions = tuple(
        str(row.get("intervention") or "")
        for row in world.get("actions") or []
    )
    if not any(actions):
        actions = tuple(str(name) for name in (proposal or {}).get("assignment") or [])
    parties = {row.get("party_id"): row for row in world.get("parties") or []}
    effects = []
    for row in world.get("effects") or []:
        party = parties.get(row.get("party_id")) or {}
        effects.append((
            str(party.get("kind") or ""),
            str(row.get("relation") or ""),
            str(row.get("polarity") or ""),
            str(row.get("directness") or ""),
        ))
    return (actions, tuple(sorted(effects)))


def raw_support(attempt: Mapping[str, Any], admission_status: str = "",
                overlay: Mapping[str, Any] | None = None) -> float:
    status = str(attempt.get("template_status") or "")
    score = 8.0 if status == "FILLED" else 2.0 if status == "PARTIAL" else 0.0
    score += 2.0 * float(attempt.get("core_filled") or 0)
    score += float(attempt.get("optional_filled") or 0)
    rank = float(attempt.get("rank") or 0)
    score += 8.0 * max(0.0, 3.0 - rank)
    if admission_status == "ADMITTED":
        score += 8.0
    proposal = attempt.get("proposal") or {}
    if proposal.get("world_withheld") or admission_status in {"NOT_RUN", ""}:
        if admission_status != "ADMITTED":
            score -= 3.0
    variant = str(attempt.get("variant") or "baseline")
    if variant in FALLBACK_VARIANTS:
        score -= 16.0
    score += _copy_bind_bonus(attempt)
    for row in (overlay or {}).get("hypotheses") or []:
        if row.get("provenance") == "WORLD_KNOWLEDGE_HYPOTHESIS":
            score -= 1.0
    rejected = float(attempt.get("rejected") or 0)
    score -= 2.0 * rejected
    return max(0.0, score)


def _copy_bind_bonus(attempt: Mapping[str, Any]) -> float:
    hyps = attempt.get("slot_hypotheses") or (
        (attempt.get("proposal") or {}).get("slot_bindings") or {}
    ).get("slot_hypotheses") or {}
    copy_ok = 0
    z10 = 0
    for rows in hyps.values() if isinstance(hyps, dict) else []:
        for row in rows or []:
            if row.get("relent_compatibility") == "copy_ok":
                copy_ok += 1
            if row.get("producer") == "z10" or row.get("z10_candidate_ids"):
                z10 += 1
    bonus = 2.0 if copy_ok else 0.0
    if z10:
        bonus += 1.0
    return bonus


def normalize_weights(scores: Sequence[float]) -> list[float]:
    positive = [max(0.0, float(score)) for score in scores]
    total = sum(positive)
    if not total:
        n = len(positive)
        return [1.0 / n for _ in positive] if n else []
    return [score / total for score in positive]


def collapse_admitted(admitted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose at most one RelEnt world after native admission.

    Family support is the max member score for a blueprint_id. Collapse when
    one family has at least twice the next family, or when occurrence
    conclusions are invariant across admitted worlds.
    """
    if not admitted:
        return {
            "collapse_status": COLLAPSE_UNRESOLVED,
            "chosen": None,
            "members": [],
            "label": "relative_interpretive_support",
        }
    scored = []
    for item in admitted:
        attempt = item.get("attempt") or item
        record = item.get("record") or {}
        overlay = item.get("overlay") or {}
        status = record.get("status") or item.get("admission_status") or "ADMITTED"
        score = raw_support(attempt, status, overlay)
        proposal = item.get("proposal") or attempt.get("proposal") or {}
        scored.append({
            "item": item,
            "attempt": attempt,
            "raw_support": score,
            "blueprint_id": attempt.get("blueprint_id") or record.get("blueprint_id"),
            "rank": float(attempt.get("rank") if attempt.get("rank") is not None
                          else record.get("rank") or 0),
            "assignment": list(proposal.get("assignment") or []),
            "signature": semantic_signature(proposal),
        })
    competing = [row for row in scored if not _is_fallback(row)]
    real = competing or list(scored)
    weights = normalize_weights([row["raw_support"] for row in real])
    weight_by_identity = {id(row): weight for row, weight in zip(real, weights)}
    members = []
    for row in scored:
        weight = weight_by_identity.get(id(row))
        attempt = row["attempt"]
        record = row["item"].get("record") or {}
        members.append({
            "blueprint_id": row["blueprint_id"],
            "rank": row["rank"],
            "variant": attempt.get("variant") or record.get("variant") or "baseline",
            "template_status": attempt.get("template_status"),
            "admission_status": record.get("status") or "ADMITTED",
            "relative_interpretive_support": weight,
            "raw_support": row["raw_support"],
            "assignment": row["assignment"],
            "trace_path": record.get("trace_path"),
            "overlay_path": record.get("overlay_path"),
            "selected_for_relent": False,
            "competing": row in real,
        })
    ranked_members = sorted(
        zip(scored, members),
        key=lambda pair: (-pair[0]["raw_support"], pair[0]["rank"]),
    )
    display = [member for row, member in ranked_members if row in real]
    noncompeting = [member for row, member in ranked_members if row not in real]

    def finish(status, chosen, reason):
        result = _collapse_result(status, chosen, display)
        result.update(collapse_reason=reason, noncompeting_members=noncompeting)
        return result

    if len(scored) == 1:
        return finish(COLLAPSE_SEPARATED, scored[0]["item"], "SOLE_NONFALLBACK" if competing else "SOLE_FALLBACK")
    if len(real) == 1:
        return finish(COLLAPSE_SEPARATED, real[0]["item"], "SOLE_NONFALLBACK" if competing else "SOLE_FALLBACK")
    signatures = {row["signature"] for row in real}
    if len(signatures) == 1:
        chosen = _preferred_member(real)
        return finish(COLLAPSE_INVARIANT, chosen, "INVARIANT")

    families: dict[str, float] = defaultdict(float)
    for row in real:
        ident = str(row["blueprint_id"] or "")
        families[ident] = max(families[ident], row["raw_support"])
    ordered = sorted(families.items(), key=lambda item: (-item[1], item[0]))
    top_id, top_score = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    family = [row for row in real if str(row["blueprint_id"] or "") == top_id]
    if len(ordered) == 1:
        family_scores = sorted((row["raw_support"] for row in family), reverse=True)
        next_member = family_scores[1] if len(family_scores) > 1 else 0.0
        if next_member <= 0 or family_scores[0] >= 2.0 * next_member:
            chosen = _preferred_member(family)
            return finish(COLLAPSE_SEPARATED, chosen, "MEMBER_TWICE_SUPPORT")
        return finish(COLLAPSE_UNRESOLVED, None, "UNRESOLVED_BEAM")
    if second <= 0 and top_score > 0 or top_score >= 2.0 * second:
        chosen = _preferred_member(family)
        return finish(COLLAPSE_SEPARATED, chosen, "FAMILY_TWICE_SUPPORT")
    return finish(COLLAPSE_UNRESOLVED, None, "UNRESOLVED_BEAM")


def beam_payload(collapse: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "world_beam/0.1",
        "collapse_status": collapse.get("collapse_status"),
        "label": "relative_interpretive_support",
        "members": list(collapse.get("members") or []),
        "noncompeting_members": list(collapse.get("noncompeting_members") or []),
        "collapse_reason": collapse.get("collapse_reason"),
        "chosen_blueprint_id": (
            ((collapse.get("chosen") or {}).get("attempt") or {}).get("blueprint_id")
            if collapse.get("chosen") else None
        ),
    }


def _is_fallback(row: Mapping[str, Any]) -> bool:
    attempt = row.get("attempt") or {}
    variant = str(attempt.get("variant") or row.get("variant") or "baseline")
    return variant in FALLBACK_VARIANTS


def _preferred_member(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(rows, key=lambda row: (row["rank"], -row["raw_support"]))["item"]


def _collapse_result(status: str, chosen: Mapping[str, Any] | None,
                     members: list[dict[str, Any]]) -> dict[str, Any]:
    if chosen is not None:
        attempt = chosen.get("attempt") or {}
        record = chosen.get("record") or {}
        ident = attempt.get("blueprint_id") or record.get("blueprint_id")
        rank = float(attempt.get("rank") if attempt.get("rank") is not None
                     else record.get("rank") or 0)
        variant = attempt.get("variant") or record.get("variant") or "baseline"
        for member in members:
            member["selected_for_relent"] = (
                member.get("blueprint_id") == ident
                and float(member.get("rank") or 0) == rank
                and (member.get("variant") or "baseline") == variant
            )
    return {
        "collapse_status": status,
        "chosen": chosen,
        "members": members,
        "label": "relative_interpretive_support",
    }
