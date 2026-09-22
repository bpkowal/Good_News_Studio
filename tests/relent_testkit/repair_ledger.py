"""Repair-policy rows derived from enforced catalog coverage."""
from __future__ import annotations

from invariants.catalog import invariant_by_id

from .coverage import load_map


def load_repair_ledger() -> dict:
    return {
        "entries": [
            {
                "failure_class": row["phenomenon_id"],
                "invariant_id": row["invariant_id"],
                "promote": "enforced",
                "issue_code": row["invariant_id"],
                "allowed_ops": ["REPAIR_GROUNDING"],
                "repair_stage": "semantic_validation",
            }
            for row in load_map()["entries"]
            if row["enforcement"] == "enforced"
            and invariant_by_id()[row["invariant_id"]]
            .integrity_layers.production_validator
        ]
    }
