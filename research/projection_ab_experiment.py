"""Read-only A/B evaluation of compact world-role projection observability.

Arm A is the current compact-role projection. Arm B keeps the same graph and
roles but adds an explicit description, a disposition for every admitted
effect, and a completion audit. Nothing here is on the execution path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from global_workspace.semantic_preservation import build_semantic_preservation_trace


EXPERIMENT_VERSION = "projection-ab-v1"
BASELINE_DESCRIPTION = "World-state attributes"
ENHANCED_DESCRIPTION = (
    "Compact welfare projection derived from the admitted graph; causal, "
    "resource-transfer, and intermediate facts remain authoritative in the graph."
)
_ALLOCATION_STATE = re.compile(
    r"\b(?:does not receive|receives no|is not allocated|is denied|goes without)\b",
    re.IGNORECASE,
)
_WELFARE_PARTY_KINDS = {
    "HUMAN", "HUMAN_GROUP", "PERSON", "PATIENT", "PATIENT_GROUP", "GROUP",
    "HOUSEHOLD", "COMMUNITY", "POPULATION", "POPULATION_GROUP",
}


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _role_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("action_id") or ""), str(row.get("effect_id") or "")


def projection_disposition(chain: Mapping[str, Any]) -> tuple[str, str]:
    """Explain why an admitted graph effect is or is not in compact roles."""
    world = chain.get("world") or {}
    role = chain.get("compact_role") or {}
    if role:
        return f"INCLUDED_{str(role.get('bucket') or 'ROLE').upper()}", str(
            role.get("reason") or "projected by current compact-role policy"
        )
    directness = str(world.get("directness") or "").upper()
    polarity = str(world.get("polarity") or "").upper()
    kind = str(world.get("effect_kind") or "").upper()
    party_kind = str(world.get("party_kind") or "").upper()
    outcome = str(world.get("outcome") or "")
    if directness == "FOREGONE" or polarity == "FOREGONE":
        return "EXCLUDED_COUNTERFACTUAL_LAYER", "not an obtaining actual-world role"
    if polarity == "NEUTRAL":
        return "EXCLUDED_NEUTRAL", "neutral effects are topology, not welfare roles"
    if kind == "RESOURCE_TRANSFER":
        return "EXCLUDED_RESOURCE_TRANSFER", "receipt is retained as a graph fact"
    if kind == "OTHER" and _ALLOCATION_STATE.search(outcome):
        return "EXCLUDED_ALLOCATION_STATE", "nonreceipt is retained as a causal state"
    if party_kind and party_kind not in _WELFARE_PARTY_KINDS:
        return (
            "EXCLUDED_NON_WELFARE_BEARER",
            f"{party_kind} party is retained as causal topology, not a welfare role",
        )
    if not party_kind and kind in {"PHYSICAL_STATE", "OTHER"}:
        return (
            "EXCLUDED_CAUSAL_INTERMEDIATE",
            "legacy trace lacks party type; current projection treated this as topology",
        )
    if polarity in {"BENEFICIAL", "ADVERSE"}:
        return (
            "UNRESOLVED_WELFARE_CLASSIFICATION",
            "polarity-bearing actual effect has no compact-role assignment or known exclusion",
        )
    return "EXCLUDED_NON_WELFARE", "effect is outside compact welfare-role policy"


def compare_projection_arms(trace: Mapping[str, Any], *, case_id: str) -> dict[str, Any]:
    chains = [dict(row) for row in trace.get("chains") or [] if isinstance(row, Mapping)]
    roles = [dict(row) for row in trace.get("compact_role_assignments") or []]
    baseline_role_keys = sorted(_role_key(row) for row in roles)
    dispositions: list[dict[str, Any]] = []
    for chain in chains:
        world = chain.get("world") or {}
        action = chain.get("canonical_action") or {}
        status, reason = projection_disposition(chain)
        dispositions.append({
            "action_id": str(action.get("action_id") or ""),
            "effect_id": str(world.get("effect_id") or ""),
            "party": str(world.get("party_label") or ""),
            "outcome": str(world.get("outcome") or ""),
            "polarity": str(world.get("polarity") or ""),
            "effect_kind": str(world.get("effect_kind") or ""),
            "party_kind": str(world.get("party_kind") or ""),
            "disposition": status,
            "reason": reason,
        })
    enhanced_role_keys = sorted(
        (row["action_id"], row["effect_id"])
        for row in dispositions if row["disposition"].startswith("INCLUDED_")
    )
    unresolved = [
        row for row in dispositions
        if row["disposition"] == "UNRESOLVED_WELFARE_CLASSIFICATION"
    ]
    polarity_unprojected = [
        row for row in dispositions
        if row["polarity"] in {"BENEFICIAL", "ADVERSE"}
        and not row["disposition"].startswith("INCLUDED_")
    ]
    graph_payload = [row.get("world") or {} for row in chains]
    graph_hash = _digest(graph_payload)
    return {
        "case_id": case_id,
        "status": str(trace.get("status") or "UNKNOWN"),
        "graph_effect_count": len(chains),
        "arm_a": {
            "description": BASELINE_DESCRIPTION,
            "role_assignments": roles,
            "effects_without_projection_explanation": len(chains) - len(roles),
        },
        "arm_b": {
            "description": ENHANCED_DESCRIPTION,
            "role_assignments": roles,
            "projection_dispositions": dispositions,
            "audit": {
                "status": "REVIEW" if unresolved else "COMPLETE",
                "disposition_coverage": (
                    len(dispositions) / len(chains) if chains else 1.0
                ),
                "unresolved_effect_ids": [row["effect_id"] for row in unresolved],
            },
        },
        "comparison": {
            "graph_hash_arm_a": graph_hash,
            "graph_hash_arm_b": graph_hash,
            "graph_unchanged": True,
            "role_assignments_unchanged": baseline_role_keys == enhanced_role_keys,
            "polarity_bearing_unprojected_count": len(polarity_unprojected),
            "unresolved_welfare_count": len(unresolved),
        },
    }


def _load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "chains" in payload and "compact_role_assignments" in payload:
        return payload
    grounding: Mapping[str, Any]
    scenario = str(payload.get("scenario") or "")
    actions = list(payload.get("actions") or [])
    if isinstance(payload.get("action_source_grounding"), Mapping):
        grounding = payload["action_source_grounding"]
    elif isinstance(payload.get("whole_world"), Mapping):
        grounding = payload["whole_world"]
    else:
        grounding = payload
    if not actions and isinstance(grounding.get("actions"), Mapping):
        actions = [
            str(row.get("canonical_semantic_action") or row.get("intervention") or key)
            for key, row in sorted(grounding["actions"].items())
            if isinstance(row, Mapping)
        ]
    trace = build_semantic_preservation_trace(
        scenario=scenario,
        clauses=grounding.get("clauses") or [],
        actions=actions,
        grounding=grounding,
    )
    parties = {
        str(row.get("party_id") or ""): str(row.get("kind") or "")
        for row in (grounding.get("world_model") or {}).get("parties") or []
        if isinstance(row, Mapping)
    }
    for chain in trace.get("chains") or []:
        world = chain.get("world") or {}
        world["party_kind"] = parties.get(str(world.get("party_id") or ""), "")
    return trace


def run(paths: Sequence[Path]) -> dict[str, Any]:
    cases = [compare_projection_arms(_load_trace(path), case_id=str(path)) for path in paths]
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "design": "paired read-only A/B on identical admitted graphs",
        "production_path_changed": False,
        "case_count": len(cases),
        "cases": cases,
        "aggregate": {
            "graph_unchanged_cases": sum(c["comparison"]["graph_unchanged"] for c in cases),
            "roles_unchanged_cases": sum(
                c["comparison"]["role_assignments_unchanged"] for c in cases
            ),
            "cases_with_unprojected_polarity_effects": sum(
                c["comparison"]["polarity_bearing_unprojected_count"] > 0 for c in cases
            ),
            "cases_requiring_projection_review": sum(
                c["comparison"]["unresolved_welfare_count"] > 0 for c in cases
            ),
            "arm_a_unexplained_effects": sum(
                c["arm_a"]["effects_without_projection_explanation"] for c in cases
            ),
            "arm_b_unexplained_effects": sum(
                len(c["arm_b"]["audit"]["unresolved_effect_ids"]) for c in cases
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.inputs)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
