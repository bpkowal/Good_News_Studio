"""Build non-authoritative semantic evidence for the world-graph generator.

Stage 1 is allowed to observe, derive, and flag.  This module deliberately does
not turn those findings into graph-admission requirements; the final world
generator remains responsible for synthesizing a candidate from the source.
"""
from __future__ import annotations

import copy
from typing import Any


_DERIVED_OPERATIONS = {
    "EXCLUSIVE_ALLOCATION_COMPLEMENT",
    "UNIVERSAL_RULE_INSTANTIATION",
    "CONDITIONAL_RULE_INSTANTIATION",
    "CESSATION_ENTAILMENT",
    "SOURCE_STIPULATED_CAUSAL",
}


def build_stage_one_evidence_packet(
    stage_one: dict[str, Any],
) -> dict[str, Any]:
    """Return a compact packet of observations, hypotheses, and cautions.

    The categories are epistemic rather than graph-authoritative.  In
    particular, a derived proposition or suggested edge is never described as
    an admitted fact merely because deterministic machinery produced it.
    """
    neutral = stage_one.get("neutral_skeleton") or {}
    skeleton = stage_one.get("skeleton") or {}
    neutral_rows = [
        copy.deepcopy(row) for row in neutral.get("propositions") or []
        if isinstance(row, dict)
    ]
    materialized_rows = [
        copy.deepcopy(row) for row in skeleton.get("propositions") or []
        if isinstance(row, dict)
    ]

    observations: list[dict[str, Any]] = []
    for row in neutral_rows:
        observations.append({
            "proposition_id": row.get("proposition_id"),
            "outcome": row.get("outcome"),
            "party_id": row.get("party_id"),
            "modality": row.get("modality"),
            "polarity": row.get("polarity"),
            "effect_kind": row.get("effect_kind"),
            "source_proposition": row.get("source_proposition"),
            "clause_ids": list(row.get("clause_ids") or []),
            "provenance_binding": copy.deepcopy(row.get("provenance_binding")),
        })

    hypotheses: list[dict[str, Any]] = []
    for row in materialized_rows:
        operation = str(row.get("derivation_operation") or "").upper()
        if operation not in _DERIVED_OPERATIONS:
            continue
        hypotheses.append({
            "proposition_id": row.get("proposition_id"),
            "action_id": row.get("action_id"),
            "party_id": row.get("party_id"),
            "outcome": row.get("outcome"),
            "modality": row.get("modality"),
            "polarity": row.get("polarity"),
            "derivation_operation": operation,
            "source_effect_ids": list(row.get("source_effect_ids") or []),
            "source_proposition": row.get("source_proposition"),
            "clause_ids": list(row.get("clause_ids") or []),
        })

    scaffold = stage_one.get("topology_scaffold") or {}
    audit = stage_one.get("pairwise_relation_audit") or {}
    relation_evidence = {
        "deterministic_candidates": copy.deepcopy(
            scaffold.get("edge_candidates") or []
        ),
        "pairwise_judgments": copy.deepcopy(audit.get("records") or []),
    }
    unresolved = {
        "source_spans": copy.deepcopy(
            neutral.get("unresolved_source_spans")
            or skeleton.get("unresolved_source_spans")
            or []
        ),
        "validation_findings": [
            str(value) for value in stage_one.get("errors") or []
            if str(value).strip()
        ],
        "conditional_rules": [
            copy.deepcopy(row)
            for row in stage_one.get("conditional_rule_instantiations") or []
            if isinstance(row, dict)
            and str(row.get("status") or "").upper() != "INSTANTIATED"
        ],
    }
    return {
        "packet_version": "1.0",
        "authority": "ADVISORY_EVIDENCE_ONLY",
        "source_observations": observations,
        "ownership_annotations": copy.deepcopy(
            skeleton.get("ownership_bindings")
            or stage_one.get("ownership", {}).get("bindings")
            or []
        ),
        "derived_hypotheses": hypotheses,
        "relation_evidence": relation_evidence,
        "unresolved": unresolved,
        "generator_contract": [
            "Construct the world graph from the original source, using this packet as advisory evidence.",
            "Account for well-supported observations, but independently verify ownership, modality, and topology.",
            "Treat derived hypotheses and relation candidates as suggestions, never as mandatory nodes or edges.",
            "Prefer the original source when packet entries conflict, and preserve unresolved ambiguity rather than guessing.",
        ],
    }
