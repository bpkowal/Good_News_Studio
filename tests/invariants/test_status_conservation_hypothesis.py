"""Status-conservation invariants over Hypothesis-generated worlds.

Oracles are declared case fields. Production is only asked whether admit /
completeness agree with expect_incomplete, preserves_stakes, and
records_quantities.
"""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from dataclasses import replace

from global_workspace.world_state import (
    ScenarioWorldModel,
    SourceRef,
    extract_binary_contrast_stipulations,
    extract_quantity_bearing_consequences,
    outcome_predicate_is_incomplete,
    validate_world_completeness,
    validate_world_model,
)
from global_workspace.world_validation import (
    apply_deterministic_local_patches,
    validation_issues_from_messages,
)
from strategies.status_conservation import (
    BinaryStipulationCase,
    OutcomePredicateCase,
    QuantityConsequenceCase,
    binary_stipulation_cases,
    outcome_predicate_cases,
    quantity_consequence_cases,
)


def _candidate_and_clauses(
    world: ScenarioWorldModel,
    *,
    extra_clauses: dict[str, str] | None = None,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """DET/repair path consumes clause_ids dicts, not nested SourceRef rows."""
    raw = world.as_dict()
    clause_text: dict[str, str] = dict(extra_clauses or {})

    def with_clause_ids(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            ids: list[str] = []
            for ref in item.get("provenance") or []:
                if not isinstance(ref, dict):
                    continue
                clause_id = str(ref.get("clause_id") or "").strip()
                if not clause_id:
                    continue
                ids.append(clause_id)
                excerpt = str(ref.get("excerpt") or "").strip()
                if excerpt and clause_id not in clause_text:
                    clause_text[clause_id] = excerpt
            item["clause_ids"] = ids
            out.append(item)
        return out

    candidate = {
        "world_model": {
            "schema_version": str(raw.get("schema_version") or "1.2"),
            "parties": with_clause_ids(list(raw.get("parties") or [])),
            "actions": with_clause_ids(list(raw.get("actions") or [])),
            "effects": with_clause_ids(list(raw.get("effects") or [])),
            "conditions": with_clause_ids(list(raw.get("conditions") or [])),
            "causal_links": with_clause_ids(list(raw.get("causal_links") or [])),
            "counterfactual_links": with_clause_ids(
                list(raw.get("counterfactual_links") or []),
            ),
        },
    }
    clauses = [
        {"clause_id": clause_id, "text": text}
        for clause_id, text in sorted(clause_text.items())
    ]
    return candidate, clauses


def _world_with_patched_effects(
    world: ScenarioWorldModel,
    patched: dict[str, object],
    clause_lookup: dict[str, str],
) -> ScenarioWorldModel:
    """Apply DET dict patches onto the typed world without re-admission."""
    world_model = patched.get("world_model") or {}
    rows = {
        str(row.get("effect_id") or ""): row
        for row in (world_model.get("effects") or [])
        if isinstance(row, dict) and row.get("effect_id")
    }
    updated = []
    for effect in world.effects:
        row = rows.get(effect.effect_id)
        if row is None:
            updated.append(effect)
            continue
        quantities = tuple(
            str(item).strip()
            for item in (row.get("quantities") or [])
            if str(item).strip()
        )
        existing = {
            ref.clause_id: ref for ref in effect.provenance if ref.clause_id
        }
        provenance = []
        for clause_id in row.get("clause_ids") or []:
            cid = str(clause_id).strip()
            if not cid:
                continue
            if cid in existing:
                provenance.append(existing[cid])
            else:
                provenance.append(SourceRef(cid, clause_lookup.get(cid, "")))
        updated.append(replace(
            effect,
            quantities=quantities,
            provenance=tuple(provenance),
        ))
    return replace(world, effects=tuple(updated))


class OutcomePredicateHypothesisTests(unittest.TestCase):
    @given(outcome_predicate_cases())
    @settings(max_examples=40, deadline=None)
    def test_production_matches_declared_predicate_oracle(
        self, case: OutcomePredicateCase,
    ):
        self.assertEqual(
            outcome_predicate_is_incomplete(case.outcome),
            case.expect_incomplete,
            case.outcome,
        )
        errors, _ = validate_world_model(case.world, action_ids=["A0"])
        incomplete_errors = [
            error for error in errors if "incomplete predicate" in error
        ]
        if case.expect_incomplete:
            self.assertTrue(incomplete_errors, errors)
            self.assertEqual(case.repair_stage, "grounding")
            self.assertIn("replace_outcome", case.allowed_ops)
        else:
            self.assertFalse(incomplete_errors, errors)


class BinaryStipulationHypothesisTests(unittest.TestCase):
    @given(binary_stipulation_cases())
    @settings(max_examples=40, deadline=None)
    def test_production_matches_declared_stipulation_oracle(
        self, case: BinaryStipulationCase,
    ):
        stipulations = extract_binary_contrast_stipulations([case.source])
        spans = {
            item.consequence_span.casefold() for item in stipulations
        }
        self.assertIn(case.survival_span.casefold(), spans, case.source)
        self.assertIn(case.loss_span.casefold(), spans, case.source)

        errors = validate_world_completeness(
            case.world, action_ids=["A0", "A1"],
        )
        stip_errors = [
            error for error in errors if "source-stipulated outcome" in error
        ]
        by_id = {effect.effect_id: effect for effect in case.world.effects}
        has_surv = case.survival_effect_id in by_id
        has_loss = case.loss_effect_id in by_id
        self.assertEqual(has_surv, not case.omit_survival)
        self.assertEqual(has_loss, not case.omit_loss)
        if case.preserves_stakes:
            self.assertFalse(stip_errors, errors)
        else:
            self.assertTrue(stip_errors, errors)
            blob = " ".join(stip_errors).casefold()
            if case.omit_survival:
                self.assertIn(case.survival_span.casefold(), blob)
            if case.omit_loss:
                self.assertIn(case.loss_span.casefold(), blob)
            self.assertEqual(case.repair_stage, "grounding")
            self.assertIn("add_effect", case.allowed_ops)
            self.assertNotIn("MORE_DEBATE", case.allowed_ops)


class QuantityConsequenceHypothesisTests(unittest.TestCase):
    @given(quantity_consequence_cases())
    @settings(max_examples=60, deadline=None)
    def test_production_matches_declared_quantity_oracle(
        self, case: QuantityConsequenceCase,
    ):
        extracted = extract_quantity_bearing_consequences([
            case.life_source,
            case.research_source,
        ])
        spans = {
            span.casefold()
            for item in extracted
            for span in item.quantity_spans
        }
        self.assertIn(case.life_quantity.casefold(), spans, case.life_source)
        self.assertIn(
            case.research_quantity.casefold(), spans, case.research_source,
        )

        errors = validate_world_completeness(
            case.world, action_ids=["A0", "A1"],
        )
        qty_errors = [
            error for error in errors if "quantity-bearing consequence" in error
        ]
        by_id = {effect.effect_id: effect for effect in case.world.effects}
        parties = {party.party_id: party for party in case.world.parties}
        life = by_id[case.life_effect_id]
        research = by_id[case.research_effect_id]
        if case.placement == "effect":
            self.assertIn(case.life_quantity, life.quantities)
            self.assertIn(case.research_quantity, research.quantities)
            self.assertFalse(qty_errors, errors)
        elif case.placement == "party":
            self.assertIn(case.life_quantity, parties["P2"].quantities)
            self.assertIn(case.research_quantity, parties["P3"].quantities)
            self.assertFalse(qty_errors, errors)
        elif case.placement == "omit":
            self.assertFalse(life.quantities)
            self.assertFalse(research.quantities)
            self.assertTrue(qty_errors, errors)
            blob = " ".join(qty_errors).casefold()
            self.assertIn(case.life_quantity.casefold(), blob)
            self.assertIn(case.research_quantity.casefold(), blob)
        else:
            # misassign: span exists somewhere else; matched effects still miss it
            self.assertTrue(qty_errors, errors)
            blob = " ".join(qty_errors).casefold()
            self.assertIn(case.life_quantity.casefold(), blob)
            self.assertIn(case.research_quantity.casefold(), blob)
        if not case.records_quantities:
            self.assertEqual(case.repair_stage, "grounding")
            self.assertTrue(
                set(case.allowed_ops) & {"add_quantity", "add_effect"}
            )
            self.assertIn("add_provenance", case.allowed_ops)
            blob = " ".join(qty_errors)
            if case.expected_licensing_clause_id:
                self.assertIn(
                    f"licensing_clause_id={case.expected_licensing_clause_id}",
                    blob,
                    (
                        case.licensing_topology,
                        case.placement,
                        qty_errors,
                    ),
                )

    @given(
        quantity_consequence_cases().filter(
            lambda case: not case.records_quantities,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_det_provenance_complete_repair_is_minimal_across_topologies(
        self, case: QuantityConsequenceCase,
    ):
        """Omit/misassign + licensing topology: DET restores span minimally."""
        errors = validate_world_completeness(
            case.world, action_ids=["A0", "A1"],
        )
        qty_errors = [
            error for error in errors if "quantity-bearing consequence" in error
        ]
        self.assertTrue(qty_errors, errors)

        candidate, clauses = _candidate_and_clauses(
            case.world,
            extra_clauses={
                "C0": case.life_source,
                "C1": case.research_source,
            },
        )
        before_ids = {
            str(row["effect_id"]): set(row.get("clause_ids") or [])
            for row in candidate["world_model"]["effects"]
        }
        patched, applied = apply_deterministic_local_patches(
            candidate,
            validation_issues_from_messages(qty_errors),
            clauses=clauses,
        )
        qty_ops = [
            row for row in applied if row.get("op") == "add_quantity"
        ]
        lic_ops = [
            row for row in applied if row.get("op") == "add_provenance"
        ]
        self.assertEqual(
            {row.get("effect_id") for row in qty_ops},
            {case.life_effect_id, case.research_effect_id},
            (case.placement, case.licensing_topology, applied),
        )
        self.assertEqual(
            {row.get("patch_kind") for row in qty_ops},
            {"semantic_patch"},
            qty_ops,
        )
        for row in lic_ops:
            self.assertEqual(row.get("patch_kind"), "licensing_patch", row)
            effect_id = str(row.get("effect_id") or "")
            value = str(row.get("value") or "")
            if effect_id == case.life_effect_id:
                self.assertEqual(value, "C0", row)
            elif effect_id == case.research_effect_id:
                self.assertEqual(value, "C1", row)
            else:
                self.fail(f"provenance spray onto {effect_id}: {row}")

        after_by_id = {
            str(row["effect_id"]): row
            for row in patched["world_model"]["effects"]
        }
        life = after_by_id[case.life_effect_id]
        research = after_by_id[case.research_effect_id]
        self.assertIn(case.life_quantity, life["quantities"])
        self.assertIn(case.research_quantity, research["quantities"])
        self.assertIn("C0", life.get("clause_ids") or [])
        self.assertIn("C1", research.get("clause_ids") or [])
        self.assertNotIn("C0", research.get("clause_ids") or [])
        # Misassign may leave the wrong span on the sibling; DET must not
        # clear it — only license the matched consequence.
        if case.placement == "misassign":
            self.assertIn(case.life_quantity, research["quantities"])
        self.assertTrue(
            before_ids[case.life_effect_id].issubset(
                set(life.get("clause_ids") or []),
            ),
        )
        self.assertTrue(
            before_ids[case.research_effect_id].issubset(
                set(research.get("clause_ids") or []),
            ),
        )

        clause_lookup = {
            str(row["clause_id"]): str(row.get("text") or "")
            for row in clauses
        }
        repaired = _world_with_patched_effects(
            case.world, patched, clause_lookup,
        )
        remaining = [
            error for error in validate_world_completeness(
                repaired, action_ids=["A0", "A1"],
            )
            if "quantity-bearing consequence" in error
        ]
        self.assertFalse(
            remaining,
            (case.placement, case.licensing_topology, remaining, applied),
        )


if __name__ == "__main__":
    unittest.main()
