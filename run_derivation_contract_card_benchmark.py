#!/usr/bin/env python3
"""Frozen preflight comparison of broad branch repair and metadata-only repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from global_workspace.world_validation import (
    repair_guidance_cards,
    validation_issues_from_messages,
)


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    effect_id = "E_NORECV"
    candidate = {"world_model": {
        "actions": [{"action_id": "A0", "recipient_party_ids": ["P_SELECTED"]}],
        "parties": [
            {"party_id": "P_SELECTED", "label": "selected party"},
            {"party_id": "P_OTHER", "label": "unselected party"},
        ],
        "effects": [{
            "effect_id": effect_id,
            "action_id": "A0",
            "party_id": "P_OTHER",
            "outcome": "unselected party does not receive the allocated resource",
            "derivation_operation": "SOURCE_STIPULATED_CAUSAL",
            "source_effect_ids": ["E_GIVE"],
        }],
        "causal_links": [{
            "action_id": "A0", "source_id": "E_GIVE",
            "link_relation": "CAUSES", "target_id": effect_id,
        }],
    }}
    clauses = [{"clause_id": case["clause_id"], "text": case["source"]}]
    broad_message = (
        "A0 omits exclusive-allocation branch for nonrecipient P_OTHER: "
        "missing nonreceipt state, source-stipulated adverse outcome"
    )
    narrow_message = (
        f"{effect_id} derivation contract operation mismatch: the semantically "
        "valid nonreceipt effect existed in the raw candidate but was removed "
        "during SOURCE_BINDING because derivation_operation="
        "SOURCE_STIPULATED_CAUSAL"
    )
    broad = repair_guidance_cards(
        validation_issues_from_messages([broad_message]), candidate, clauses=clauses,
    )[0]["concrete_patches"][0]
    narrow = repair_guidance_cards(
        validation_issues_from_messages([narrow_message]), candidate, clauses=clauses,
    )[0]["concrete_patches"][0]
    values = narrow["set"]
    bound = bool(values.get("clause_ids"))
    expected = bool(case["exclusive"])
    return {
        "id": case["id"],
        "exclusive": expected,
        "legacy_operation": broad["op"],
        "metadata_operation": narrow["op"],
        "metadata_bound_source": bound,
        "source_binding_correct": bound == expected,
        "false_metadata_application": (
            not expected and narrow["op"] == "repair_derivation_contract_metadata"
        ),
        "metadata_preserves_topology": "causal_links" in narrow.get("preserve", []),
        "legacy_requests_graph_reconstruction": broad["op"] == "restore_exclusive_allocation_branch",
        "selected_source_proposition": values.get("source_proposition"),
        "selected_quantities": values.get("quantities"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("evals/exclusive_allocation_derivation_contract.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [evaluate_case(dict(case)) for case in payload["cases"]]
    report = {
        "schema_version": 1,
        "case_count": len(rows),
        "source_binding_accuracy": sum(row["source_binding_correct"] for row in rows) / len(rows),
        "metadata_topology_preservation_rate": sum(row["metadata_preserves_topology"] for row in rows) / len(rows),
        "false_application_rate": sum(row["false_metadata_application"] for row in rows) / len(rows),
        "legacy_graph_reconstruction_request_rate": sum(row["legacy_requests_graph_reconstruction"] for row in rows) / len(rows),
        "cases": rows,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["source_binding_accuracy"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
