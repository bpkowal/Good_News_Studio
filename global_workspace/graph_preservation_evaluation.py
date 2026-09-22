"""Score source facts, topology, and projection as separate graph boundaries."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _fact_score(fact: Mapping[str, Any], chain: Mapping[str, Any]) -> float:
    world = chain.get("world") or {}
    text = " ".join((
        str(world.get("party_label") or ""), str(world.get("outcome") or ""),
        " ".join(str(value) for value in world.get("quantities") or []),
        " ".join(str(value) for value in world.get("likelihood_qualifiers") or []),
    ))
    haystack = _tokens(text)
    groups = fact.get("required_term_groups") or []
    if groups and not all(haystack & _tokens(group) for group in groups):
        return 0.0
    gold = _tokens(fact.get("event")) | _tokens(fact.get("party"))
    return len(gold & haystack) / len(gold) if gold else 0.0


def align_gold_facts(
    facts: Sequence[Mapping[str, Any]], chains: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for fact in facts:
        expected_action = str(fact.get("action_id") or "")
        candidates: list[tuple[float, Mapping[str, Any]]] = []
        for chain in chains:
            world = chain.get("world") or {}
            effect_id = str(world.get("effect_id") or "")
            if not effect_id or effect_id in used:
                continue
            score = _fact_score(fact, chain)
            if score:
                candidates.append((score, chain))
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0] if candidates else None
        chain = best[1] if best else {}
        world = chain.get("world") or {}
        actual_action = str((chain.get("canonical_action") or {}).get("action_id") or "")
        effect_id = str(world.get("effect_id") or "")
        if effect_id:
            used.add(effect_id)
        attribute_checks = {}
        for field in ("polarity", "modality", "directness", "effect_kind"):
            expected = str(fact.get(field) or "")
            if expected:
                attribute_checks[field] = expected.upper() == str(world.get(field) or "").upper()
        status = "MISSING"
        if effect_id:
            status = "PRESENT" if not expected_action or expected_action == actual_action else "WRONG_OWNER"
        rows.append({
            "fact_id": str(fact.get("fact_id") or ""),
            "event": str(fact.get("event") or ""),
            "expected_action_id": expected_action,
            "matched_effect_id": effect_id or None,
            "actual_action_id": actual_action or None,
            "status": status,
            "alignment_score": best[0] if best else 0.0,
            "attribute_checks": attribute_checks,
            "attributes_correct": bool(attribute_checks) and all(attribute_checks.values()),
        })
    return rows


def score_trace(case: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    facts = list(case.get("gold_facts") or [])
    chains = list(trace.get("chains") or [])
    alignments = align_gold_facts(facts, chains)
    aligned_by_fact = {row["fact_id"]: row for row in alignments}
    role_by_effect = {
        (str(row.get("action_id") or ""), str(row.get("effect_id") or "")):
        str(row.get("bucket") or "")
        for row in trace.get("compact_role_assignments") or []
    }
    topology_available = "topology_edges" in trace
    edges = {
        (
            str(row.get("action_id") or ""), str(row.get("source_id") or ""),
            str(row.get("target_id") or ""), str(row.get("relation") or "").upper(),
        )
        for row in trace.get("topology_edges") or []
    }
    adjacency: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for action_id, source_id, target_id, relation in edges:
        adjacency.setdefault((action_id, source_id), []).append((target_id, relation))

    def valid_path(action_id: str, source_id: str, target_id: str, relation: str) -> bool:
        # A chain of CAUSES edges preserves causal reachability even when the
        # generated graph makes a source-licensed intermediate explicit.
        if relation != "CAUSES":
            return False
        seen: set[str] = set()
        stack = [source_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for child, edge_relation in adjacency.get((action_id, current), []):
                if edge_relation != "CAUSES":
                    continue
                if child == target_id:
                    return True
                stack.append(child)
        return False
    edge_rows = []
    for edge in case.get("gold_edges") or []:
        source = aligned_by_fact.get(str(edge.get("source_fact_id") or ""), {})
        target = aligned_by_fact.get(str(edge.get("target_fact_id") or ""), {})
        endpoints = bool(source.get("matched_effect_id") and target.get("matched_effect_id"))
        key = (
            str(edge.get("action_id") or ""), str(source.get("matched_effect_id") or ""),
            str(target.get("matched_effect_id") or ""), str(edge.get("relation") or "").upper(),
        )
        direct_correct = endpoints and key in edges
        path_correct = endpoints and not direct_correct and valid_path(
            key[0], key[1], key[2], key[3],
        )
        edge_rows.append({
            **dict(edge), "endpoints_available": endpoints,
            "match_type": (
                "DIRECT_EDGE_CORRECT" if topology_available and direct_correct else
                "VALID_CAUSAL_PATH_CORRECT" if topology_available and path_correct else
                "NO_VALID_CONNECTION" if topology_available and endpoints else
                "ENDPOINT_MISSING" if topology_available else
                "NOT_MEASURED"
            ),
            "correct": (
                direct_correct or path_correct if topology_available else None
            ),
        })
    projection_rows = []
    for expected in case.get("gold_projection") or []:
        fact = aligned_by_fact.get(str(expected.get("fact_id") or ""), {})
        effect_id = str(fact.get("matched_effect_id") or "")
        actual = role_by_effect.get((str(expected.get("action_id") or ""), effect_id))
        expected_bucket = expected.get("bucket")
        projection_rows.append({
            **dict(expected), "effect_id": effect_id or None, "actual_bucket": actual,
            "correct": bool(effect_id) and actual == expected_bucket,
        })
    present = sum(row["status"] == "PRESENT" for row in alignments)
    available_edges = sum(row["endpoints_available"] for row in edge_rows)
    scored_attrs = [row for row in alignments if row["attribute_checks"] and row["status"] == "PRESENT"]
    aligned_effects = {row["matched_effect_id"] for row in alignments if row["matched_effect_id"]}
    unsupported = sorted(
        str((chain.get("world") or {}).get("effect_id") or "")
        for chain in chains
        if str((chain.get("world") or {}).get("effect_id") or "") not in aligned_effects
    )
    exhaustive = bool(case.get("gold_inventory_exhaustive"))
    projection_available = bool(trace.get("projection_available", True))
    return {
        "case_id": case.get("id"),
        "source_to_graph": {
            "gold_fact_count": len(facts), "present_count": present,
            "recall": present / len(facts) if facts else None,
            "attribute_accuracy_given_present": (
                sum(row["attributes_correct"] for row in scored_attrs) / len(scored_attrs)
                if scored_attrs else None
            ),
            "unsupported_effect_ids": unsupported if exhaustive else [],
            "unsupported_precision_status": "MEASURED" if exhaustive else "NOT_MEASURED_SPARSE_GOLD",
            "alignments": alignments,
        },
        "topology": {
            "measurement_status": (
                "MEASURED" if topology_available else "NOT_MEASURED_ARTIFACT_LACKS_EDGES"
            ),
            "gold_edge_count": len(edge_rows), "aligned_edge_count": available_edges,
            "accuracy_given_aligned_endpoints": (
                sum(bool(row["correct"]) for row in edge_rows) / available_edges
                if topology_available and available_edges else None
            ),
            "edges": edge_rows,
        },
        "projection": {
            "measurement_status": (
                "MEASURED" if projection_available else "NOT_MEASURED_UNADMITTED_CANDIDATE"
            ),
            "gold_role_count": len(projection_rows),
            "accuracy_given_gold_graph": (
                sum(row["correct"] for row in projection_rows) / len(projection_rows)
                if projection_available and projection_rows else None
            ),
            "roles": projection_rows,
        },
    }


def trace_from_unadmitted_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Create a benchmark-only evidence view without accepting the candidate."""
    world = candidate.get("world_model") if isinstance(candidate, Mapping) else {}
    world = world if isinstance(world, Mapping) else {}
    parties = {
        str(row.get("party_id") or ""): row
        for row in world.get("parties") or [] if isinstance(row, Mapping)
    }
    actions = {
        str(row.get("action_id") or ""): row
        for row in world.get("actions") or [] if isinstance(row, Mapping)
    }
    chains = []
    for effect in world.get("effects") or []:
        if not isinstance(effect, Mapping):
            continue
        action_id = str(effect.get("action_id") or "")
        party = parties.get(str(effect.get("party_id") or ""), {})
        chains.append({
            "canonical_action": {
                "action_id": action_id,
                "text": str(actions.get(action_id, {}).get("intervention") or ""),
            },
            "world": {
                "effect_id": str(effect.get("effect_id") or ""),
                "party_id": str(effect.get("party_id") or ""),
                "party_label": str(party.get("label") or ""),
                "party_kind": str(party.get("kind") or ""),
                "outcome": str(effect.get("outcome") or ""),
                "directness": str(effect.get("directness") or ""),
                "effect_kind": str(effect.get("effect_kind") or ""),
                "polarity": str(effect.get("polarity") or ""),
                "modality": str(effect.get("modality") or ""),
                "quantities": list(effect.get("quantities") or []),
                "likelihood_qualifiers": list(effect.get("likelihood_qualifiers") or []),
            },
            "compact_role": None,
            "admitted_status": "REJECTED_CANDIDATE_ONLY",
        })
    topology_edges = [{
        "layer": "ACTUAL",
        "action_id": str(row.get("action_id") or ""),
        "source_id": str(row.get("source_id") or ""),
        "target_id": str(row.get("target_id") or ""),
        "relation": str(row.get("link_relation") or row.get("relation") or ""),
    } for row in world.get("causal_links") or [] if isinstance(row, Mapping)]
    return {
        "status": "REJECTED_CANDIDATE_ONLY",
        "chains": chains,
        "topology_edges": topology_edges,
        "compact_role_assignments": [],
        "projection_available": False,
    }
