"""Read-only spaCy-informed ranking of already extracted proposition pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import random
import re
from typing import Any

from .syntactic_annotation import predicate_lemmas


STRUCTURAL_LINKS = {"advcl", "ccomp", "xcomp", "conj", "csubj", "relcl"}
DISCOURSE_BRIDGE = re.compile(
    r"^\s*(?:consequently|therefore|then|thus|as a result|without|because)\b",
    re.IGNORECASE,
)


def _clause_indexes(proposition: Mapping[str, Any]) -> set[int]:
    indexes = set()
    for value in proposition.get("clause_ids") or []:
        text = str(value)
        if text.startswith("C") and text[1:].isdigit():
            indexes.add(int(text[1:]))
    return indexes


def _anchors(
    proposition: Mapping[str, Any], annotation: Mapping[str, Any], *, nlp: Any,
) -> list[dict[str, Any]]:
    text = str(proposition.get("source_proposition") or proposition.get("outcome") or "")
    lemmas = set(predicate_lemmas(text, nlp=nlp))
    candidates = [
        dict(row) for row in annotation.get("predicates") or []
        if str(row.get("lemma") or "") in lemmas
    ]
    clause_indexes = _clause_indexes(proposition)
    local = [row for row in candidates if row.get("sentence_index") in clause_indexes]
    return local or candidates


def _ancestor_distances(token_index: int, tokens: dict[int, dict[str, Any]]) -> dict[int, int]:
    distances = {token_index: 0}
    current = token_index
    while current in tokens:
        parent = int(tokens[current].get("head_token_index", current))
        if parent == current or parent in distances:
            break
        distances[parent] = distances[current] + 1
        current = parent
    return distances


def _dependency_distance(left: int, right: int, tokens: dict[int, dict[str, Any]]) -> int | None:
    left_path = _ancestor_distances(left, tokens)
    right_path = _ancestor_distances(right, tokens)
    shared = set(left_path) & set(right_path)
    if not shared:
        return None
    return min(left_path[node] + right_path[node] for node in shared)


def score_job(job: Mapping[str, Any], annotation: Mapping[str, Any], *, nlp: Any) -> dict[str, Any]:
    """Score structural proximity without deciding a semantic relation."""
    source = job.get("source") or {}
    target = job.get("target") or {}
    source_anchors = _anchors(source, annotation, nlp=nlp)
    target_anchors = _anchors(target, annotation, nlp=nlp)
    tokens = {
        int(row["token_index"]): dict(row) for row in annotation.get("tokens") or []
    }
    same_clause = bool(_clause_indexes(source) & _clause_indexes(target))
    best: dict[str, Any] = {
        "same_sentence": False,
        "dependency_distance": None,
        "direct_dependency": False,
        "structural_link": False,
        "shared_argument": False,
        "adjacent_sentence": False,
        "discourse_bridge": False,
    }
    sentences = {
        int(row["sentence_index"]): str(row.get("text") or "")
        for row in annotation.get("sentences") or []
    }
    for left in source_anchors:
        for right in target_anchors:
            sentence_delta = int(right.get("sentence_index", -99)) - int(
                left.get("sentence_index", 99)
            )
            if sentence_delta == 1:
                best["adjacent_sentence"] = True
                best["discourse_bridge"] = bool(DISCOURSE_BRIDGE.search(
                    sentences.get(int(right.get("sentence_index", -1)), "")
                ))
            if sentence_delta != 0:
                continue
            distance = _dependency_distance(
                int(left["token_index"]), int(right["token_index"]), tokens,
            )
            direct = distance == 1
            structural = (
                str(left.get("dependency") or "") in STRUCTURAL_LINKS
                or str(right.get("dependency") or "") in STRUCTURAL_LINKS
            )
            left_args = set(left.get("subjects") or []) | set(left.get("objects") or [])
            right_args = set(right.get("subjects") or []) | set(right.get("objects") or [])
            shared_argument = bool(left_args & right_args)
            candidate = (
                direct, structural, shared_argument,
                -(distance if distance is not None else 10_000),
            )
            incumbent = (
                best["direct_dependency"], best["structural_link"],
                best["shared_argument"],
                -(best["dependency_distance"] if best["dependency_distance"] is not None else 10_000),
            )
            if not best["same_sentence"] or candidate > incumbent:
                best = {
                    "same_sentence": True,
                    "dependency_distance": distance,
                    "direct_dependency": direct,
                    "structural_link": structural,
                    "shared_argument": shared_argument,
                    "adjacent_sentence": best["adjacent_sentence"],
                    "discourse_bridge": best["discourse_bridge"],
                    "source_anchor_id": left.get("predicate_id"),
                    "target_anchor_id": right.get("predicate_id"),
                }
    structured = job.get("structured_evidence") or {}
    source_licensed = any(
        row.get("commitment") == "SOURCE_LICENSED"
        for row in structured.get("scaffold_evidence_without_label") or []
    )
    distance_bonus = (
        max(0, 10 - int(best["dependency_distance"]))
        if best["dependency_distance"] is not None else 0
    )
    score = (
        100 * source_licensed
        + 40 * best["same_sentence"]
        + 30 * best["direct_dependency"]
        + 25 * best["structural_link"]
        + 15 * best["shared_argument"]
        + 20 * same_clause
        + 30 * best["adjacent_sentence"]
        + 20 * best["discourse_bridge"]
        + distance_bonus
    )
    return {
        "score": score,
        "same_clause": same_clause,
        "source_licensed_cue": source_licensed,
        "source_anchor_count": len(source_anchors),
        "target_anchor_count": len(target_anchors),
        **best,
    }


def rank_jobs(
    jobs: Sequence[Mapping[str, Any]],
    annotation: Mapping[str, Any],
    *,
    nlp: Any,
    budget: int,
    exploration_count: int = 3,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select high-scoring jobs plus a deterministic low-rank exploration sample."""
    scored = [
        {**dict(job), "syntactic_rank_evidence": score_job(job, annotation, nlp=nlp)}
        for job in jobs
    ]
    scored.sort(key=lambda row: (
        -int(row["syntactic_rank_evidence"]["score"]),
        str((row.get("source") or {}).get("proposition_id") or ""),
        str((row.get("target") or {}).get("proposition_id") or ""),
    ))
    capped_budget = min(max(0, budget), len(scored))
    reserve = min(max(0, exploration_count), capped_budget)
    exploitation_count = capped_budget - reserve
    selected = list(scored[:exploitation_count])
    remainder = scored[exploitation_count:]
    randomizer = random.Random(seed)
    explored = randomizer.sample(remainder, min(reserve, len(remainder)))
    selected.extend(explored)
    selected_keys = {
        (
            str((row.get("source") or {}).get("proposition_id") or ""),
            str((row.get("target") or {}).get("proposition_id") or ""),
        ) for row in selected
    }
    for row in selected:
        row["selection_route"] = (
            "EXPLOITATION" if row in scored[:exploitation_count] else "EXPLORATION"
        )
    return selected, {
        "candidate_pair_count": len(scored),
        "budget": capped_budget,
        "exploitation_count": exploitation_count,
        "exploration_count": len(explored),
        "selected_pair_keys": sorted([list(key) for key in selected_keys]),
    }
