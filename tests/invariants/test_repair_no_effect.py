"""REPAIR_NO_EFFECT / unstable-repair guard beside DETERMINISTIC_LOCAL_PATCH."""
from __future__ import annotations

import unittest
from dataclasses import replace

from hypothesis import given, settings

from global_workspace.world_state import (
    ScenarioWorldModel,
    SourceRef,
    validate_world_completeness,
)
from global_workspace.world_validation import (
    SUBGRAPH_REBUILD,
    ValidationIssue,
    annotate_admit_with_repair_no_effect,
    apply_deterministic_local_patches,
    classify_world_repair_scope,
    repair_no_effect_issues,
    should_skip_deterministic_local_patch,
    validation_issues_from_messages,
)
from strategies.repair_no_effect import RepairNoEffectCase, repair_no_effect_cases
from strategies.status_conservation import (
    QuantityConsequenceCase,
    quantity_consequence_cases,
)


def _qty_issue(effect_id: str = "E7") -> ValidationIssue:
    return ValidationIssue(
        code="QUANTITY_BEARING_CONSEQUENCE_MISSING",
        message=(
            f"{effect_id} omits quantity-bearing consequence "
            "'catastrophic loss of life' (licensing_clause_id=C0)"
        ),
        entity_kind="effect",
        entity_id=effect_id,
        field="effects",
        related_ids=("C0",),
        repair_class="SEMANTIC_PATCH",
        permits_removal=False,
    )


def _applied(effect_id: str = "E7", *, with_provenance: bool = False) -> list[dict]:
    rows = [{
        "op": "add_quantity",
        "effect_id": effect_id,
        "field": "quantities",
        "value": "thousands",
        "code": "QUANTITY_BEARING_CONSEQUENCE_MISSING",
        "patch_kind": "semantic_patch",
        "deterministic": True,
    }]
    if with_provenance:
        rows.append({
            "op": "add_provenance",
            "effect_id": effect_id,
            "field": "clause_ids",
            "value": "C0",
            "code": "QUANTITY_BEARING_CONSEQUENCE_MISSING",
            "patch_kind": "licensing_patch",
            "deterministic": True,
        })
    return rows


def _candidate_and_clauses(
    world: ScenarioWorldModel,
    *,
    extra_clauses: dict[str, str] | None = None,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """DET path consumes clause_ids dicts, not nested SourceRef rows."""
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


class RepairNoEffectGuardTests(unittest.TestCase):
    def test_same_target_after_applied_patch_is_unstable(self):
        before = [_qty_issue("E7")]
        after = [_qty_issue("E7")]
        unstable = repair_no_effect_issues(before, after, _applied("E7"))
        self.assertEqual(len(unstable), 1)
        self.assertEqual(unstable[0].code, "REPAIR_NO_EFFECT")
        self.assertEqual(unstable[0].entity_id, "E7")
        self.assertIn("QUANTITY_BEARING_CONSEQUENCE_MISSING", unstable[0].message)

    def test_empty_applied_is_not_repair_no_effect(self):
        before = [_qty_issue()]
        after = [_qty_issue()]
        self.assertEqual(repair_no_effect_issues(before, after, []), ())

    def test_cleared_issue_is_not_unstable(self):
        before = [_qty_issue("E7")]
        after: list[ValidationIssue] = []
        self.assertEqual(
            repair_no_effect_issues(before, after, _applied("E7", with_provenance=True)),
            (),
        )

    def test_unrelated_remaining_issue_is_not_blamed(self):
        before = [_qty_issue("E7"), _qty_issue("E1")]
        after = [_qty_issue("E1")]
        unstable = repair_no_effect_issues(before, after, _applied("E7"))
        self.assertEqual(unstable, ())

    def test_annotate_admit_merges_and_escalates_scope(self):
        before = [_qty_issue("E7")]
        after = [_qty_issue("E7")]
        admit = {
            "status": "REJECTED",
            "errors": [before[0].message],
            "validation_issues": [issue.as_dict() for issue in after],
        }
        annotated = annotate_admit_with_repair_no_effect(
            admit,
            before_issues=before,
            applied_patches=_applied("E7"),
        )
        codes = {
            str(row.get("code") or "")
            for row in annotated.get("validation_issues") or []
        }
        self.assertIn("REPAIR_NO_EFFECT", codes)
        self.assertTrue(annotated.get("repair_no_effect"))
        self.assertTrue(should_skip_deterministic_local_patch(
            annotated.get("validation_issues") or [],
        ))
        self.assertEqual(
            classify_world_repair_scope(
                annotated.get("validation_issues") or [],
                errors=annotated.get("errors") or [],
                candidate={"world_model": {}},
            ),
            SUBGRAPH_REBUILD,
        )

    def test_committed_admit_is_untouched(self):
        admit = {"status": "COMMITTED", "validation_issues": []}
        out = annotate_admit_with_repair_no_effect(
            admit,
            before_issues=[_qty_issue()],
            applied_patches=_applied(),
        )
        self.assertEqual(out, admit)


class RepairNoEffectHypothesisTests(unittest.TestCase):
    @given(repair_no_effect_cases())
    @settings(max_examples=80, deadline=None)
    def test_oracle_agrees_with_unstable_detection(self, case: RepairNoEffectCase):
        unstable = repair_no_effect_issues(
            case.before_issues,
            case.after_issues,
            case.applied_patches,
        )
        if case.expect_unstable:
            self.assertTrue(unstable, case.mode)
            self.assertEqual(
                {issue.entity_id for issue in unstable},
                set(case.unstable_entity_ids),
                case.mode,
            )
            self.assertTrue(all(issue.code == "REPAIR_NO_EFFECT" for issue in unstable))
        else:
            self.assertEqual(unstable, (), case.mode)

    @given(repair_no_effect_cases().filter(lambda case: case.expect_unstable))
    @settings(max_examples=40, deadline=None)
    def test_unstable_annotates_escalates_and_skips_det(
        self, case: RepairNoEffectCase,
    ):
        admit = {
            "status": "REJECTED",
            "errors": [str(row.get("message") or "") for row in case.after_issues],
            "validation_issues": [dict(row) for row in case.after_issues],
        }
        annotated = annotate_admit_with_repair_no_effect(
            admit,
            before_issues=case.before_issues,
            applied_patches=case.applied_patches,
        )
        codes = {
            str(row.get("code") or "")
            for row in annotated.get("validation_issues") or []
        }
        self.assertIn("REPAIR_NO_EFFECT", codes, case.mode)
        self.assertTrue(annotated.get("repair_no_effect"), case.mode)
        self.assertTrue(
            should_skip_deterministic_local_patch(
                annotated.get("validation_issues") or [],
            ),
            case.mode,
        )
        self.assertEqual(
            classify_world_repair_scope(
                annotated.get("validation_issues") or [],
                errors=annotated.get("errors") or [],
                candidate={"world_model": {}},
            ),
            SUBGRAPH_REBUILD,
            case.mode,
        )

    @given(repair_no_effect_cases().filter(lambda case: not case.expect_unstable))
    @settings(max_examples=40, deadline=None)
    def test_stable_cases_do_not_escalate_or_skip(self, case: RepairNoEffectCase):
        admit = {
            "status": "REJECTED",
            "errors": [str(row.get("message") or "") for row in case.after_issues],
            "validation_issues": [dict(row) for row in case.after_issues],
        }
        annotated = annotate_admit_with_repair_no_effect(
            admit,
            before_issues=case.before_issues,
            applied_patches=case.applied_patches,
        )
        codes = {
            str(row.get("code") or "")
            for row in annotated.get("validation_issues") or []
        }
        self.assertNotIn("REPAIR_NO_EFFECT", codes, case.mode)
        self.assertFalse(annotated.get("repair_no_effect"), case.mode)
        self.assertFalse(
            should_skip_deterministic_local_patch(
                annotated.get("validation_issues") or [],
            ),
            case.mode,
        )

    @given(
        quantity_consequence_cases().filter(
            lambda case: not case.records_quantities,
        ),
    )
    @settings(max_examples=40, deadline=None)
    def test_provenance_complete_det_is_not_unstable(
        self, case: QuantityConsequenceCase,
    ):
        """Full DET that clears quantity omissions must not raise REPAIR_NO_EFFECT."""
        errors = validate_world_completeness(
            case.world, action_ids=["A0", "A1"],
        )
        qty_errors = [
            error for error in errors if "quantity-bearing consequence" in error
        ]
        self.assertTrue(qty_errors, errors)
        before = validation_issues_from_messages(qty_errors)
        candidate, clauses = _candidate_and_clauses(
            case.world,
            extra_clauses={
                "C0": case.life_source,
                "C1": case.research_source,
            },
        )
        patched, applied = apply_deterministic_local_patches(
            candidate, before, clauses=clauses,
        )
        self.assertTrue(applied, (case.placement, case.licensing_topology))

        clause_lookup = {
            str(row["clause_id"]): str(row.get("text") or "")
            for row in clauses
        }
        repaired = _world_with_patched_effects(
            case.world, patched, clause_lookup,
        )
        after_errors = [
            error for error in validate_world_completeness(
                repaired, action_ids=["A0", "A1"],
            )
            if "quantity-bearing consequence" in error
        ]
        after = validation_issues_from_messages(after_errors)
        unstable = repair_no_effect_issues(before, after, applied)
        self.assertEqual(
            unstable,
            (),
            (case.placement, case.licensing_topology, after_errors, applied),
        )

    @given(
        quantity_consequence_cases().filter(
            lambda case: not case.records_quantities,
        ),
    )
    @settings(max_examples=40, deadline=None)
    def test_quantity_only_det_without_clearance_is_unstable(
        self, case: QuantityConsequenceCase,
    ):
        """Motivating witness: DET applied on the target but same qty fingerprint remains."""
        errors = validate_world_completeness(
            case.world, action_ids=["A0", "A1"],
        )
        qty_errors = [
            error for error in errors if "quantity-bearing consequence" in error
        ]
        self.assertTrue(qty_errors, errors)
        before = validation_issues_from_messages(qty_errors)
        candidate, clauses = _candidate_and_clauses(
            case.world,
            extra_clauses={
                "C0": case.life_source,
                "C1": case.research_source,
            },
        )
        _, applied = apply_deterministic_local_patches(
            candidate, before, clauses=clauses,
        )
        # Simulate a non-effective repair: patches ran, but admit still reports
        # the same quantity fingerprints (e.g. provenance-stripped compile).
        qty_only = [
            row for row in applied if row.get("op") == "add_quantity"
        ]
        self.assertTrue(qty_only, applied)
        unstable = repair_no_effect_issues(before, before, qty_only)
        targeted = {str(row.get("effect_id") or "") for row in qty_only}
        expected_entities = {
            issue.entity_id for issue in before
            if issue.entity_id in targeted
        }
        self.assertTrue(unstable, (case.placement, case.licensing_topology))
        self.assertEqual(
            {issue.entity_id for issue in unstable},
            expected_entities,
            (case.placement, case.licensing_topology, unstable, qty_only),
        )
        annotated = annotate_admit_with_repair_no_effect(
            {
                "status": "REJECTED",
                "errors": list(qty_errors),
                "validation_issues": [issue.as_dict() for issue in before],
            },
            before_issues=before,
            applied_patches=qty_only,
        )
        self.assertTrue(should_skip_deterministic_local_patch(
            annotated.get("validation_issues") or [],
        ))
        self.assertEqual(
            classify_world_repair_scope(
                annotated.get("validation_issues") or [],
                errors=annotated.get("errors") or [],
                candidate=candidate,
            ),
            SUBGRAPH_REBUILD,
        )


if __name__ == "__main__":
    unittest.main()
