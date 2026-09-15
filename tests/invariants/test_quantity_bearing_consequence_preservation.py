"""QUANTITY_BEARING_CONSEQUENCE_PRESERVATION — retain thousands/decades."""
from __future__ import annotations

import unittest

from global_workspace.world_state import (
    explicit_quantity_spans,
    extract_quantity_bearing_consequences,
    validate_world_completeness,
)
from global_workspace.world_validation import (
    apply_deterministic_local_patches,
    repair_guidance_cards,
    validation_issues_from_messages,
)
from invariants.catalog import invariant_by_id
from semantic_integrity.coverage import load_map, load_taxonomy
from semantic_integrity.harness import (
    admit_world_from_discourse,
    load_seed,
    quantity_bearing_consequence_preservation_holds,
    structured_world_from_seed,
)


class QuantityBearingConsequencePreservationTests(unittest.TestCase):
    def test_taxonomy_marks_parliament_extension(self):
        row = next(
            item for item in load_taxonomy()["phenomena"]
            if item["id"] == "quantity_bearing_consequence_preservation"
        )
        self.assertEqual(row["source"]["type"], "parliament_extension")

    def test_catalog_entry(self):
        item = invariant_by_id()["QUANTITY_BEARING_CONSEQUENCE_PRESERVATION"]
        self.assertEqual(item.source_type, "parliament_extension")
        self.assertEqual(item.enforcement, "enforced")
        self.assertTrue(item.integrity_layers.production_validator)

    def test_production_extractor_reads_thousands_and_decades(self):
        self.assertIn(
            "thousands",
            explicit_quantity_spans("risks thousands of lives"),
        )
        self.assertIn(
            "decades",
            explicit_quantity_spans("erase decades of medical research"),
        )
        items = extract_quantity_bearing_consequences([
            "A cyberattack risks thousands of lives through failure.",
            "This action will permanently erase decades of medical research.",
        ])
        spans = {
            span
            for item in items
            for span in item.quantity_spans
        }
        self.assertIn("thousands", spans)
        self.assertIn("decades", spans)

    def test_structured_lane_preserves_quantities(self):
        seed = load_seed("quantity_bearing_purge_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            quantity_bearing_consequence_preservation_holds(
                model,
                life_effect_id=expected["life_effect_id"],
                research_effect_id=expected["research_effect_id"],
                life_quantity=expected["life_quantity"],
                research_quantity=expected["research_quantity"],
            )
        )
        errors = validate_world_completeness(model, action_ids=["A0", "A1"])
        self.assertFalse(
            any("quantity-bearing consequence" in error for error in errors),
            errors,
        )

    def test_structured_lane_detects_dropped_quantity(self):
        seed = load_seed("quantity_bearing_purge_structured.yaml")
        model = structured_world_from_seed(seed, world_key="incomplete_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            quantity_bearing_consequence_preservation_holds(
                model,
                life_effect_id=expected["life_effect_id"],
                research_effect_id=expected["research_effect_id"],
                life_quantity=expected["life_quantity"],
                research_quantity=expected["research_quantity"],
            )
        )
        errors = validate_world_completeness(model, action_ids=["A0", "A1"])
        qty_errors = [
            error for error in errors if "quantity-bearing consequence" in error
        ]
        self.assertGreaterEqual(len(qty_errors), 2, errors)
        blob = " ".join(qty_errors).casefold()
        self.assertIn("thousands", blob)
        self.assertIn("decades", blob)

    def test_party_quantity_satisfies_preservation(self):
        seed = load_seed("quantity_bearing_purge_structured.yaml")
        model = structured_world_from_seed(seed, world_key="party_quantity_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            quantity_bearing_consequence_preservation_holds(
                model,
                life_effect_id=expected["life_effect_id"],
                research_effect_id=expected["research_effect_id"],
                life_quantity=expected["life_quantity"],
                research_quantity=expected["research_quantity"],
            )
        )
        errors = validate_world_completeness(model, action_ids=["A0", "A1"])
        self.assertFalse(
            any("quantity-bearing consequence" in error for error in errors),
            errors,
        )

    def test_grounding_lane_completeness_rejects_dropped_quantity(self):
        seed = load_seed("quantity_bearing_purge_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        items = extract_quantity_bearing_consequences([discourse])
        spans = {
            span for item in items for span in item.quantity_spans
        }
        self.assertIn("thousands", spans)
        self.assertIn("decades", spans)
        # Incomplete worlds that cite quantity-bearing clauses cannot pass
        # structural admit without recording those spans; exercise production
        # completeness on the structured incomplete fixture instead.
        structured = load_seed("quantity_bearing_purge_structured.yaml")
        model = structured_world_from_seed(
            structured, world_key="incomplete_world",
        )
        errors = validate_world_completeness(model, action_ids=["A0", "A1"])
        qty_errors = [
            error for error in errors if "quantity-bearing consequence" in error
        ]
        self.assertTrue(qty_errors, errors)
        issues = validation_issues_from_messages(qty_errors)
        self.assertTrue(
            any(
                issue.code == "QUANTITY_BEARING_CONSEQUENCE_MISSING"
                for issue in issues
            )
        )

    def test_grounding_lane_completeness_accepts_recorded_quantities(self):
        seed = load_seed("quantity_bearing_purge_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            quantity_bearing_consequence_preservation_holds(
                model,
                life_effect_id=expected["life_effect_id"],
                research_effect_id=expected["research_effect_id"],
                life_quantity=expected["life_quantity"],
                research_quantity=expected["research_quantity"],
            )
        )
        errors = validate_world_completeness(model, action_ids=["A0", "A1"])
        self.assertFalse(
            any("quantity-bearing consequence" in error for error in errors),
            errors,
        )

    def test_repair_card_for_missing_quantity(self):
        messages = [
            "E_loss omits source quantity 'thousands' from quantity-bearing "
            "consequence 'A cyberattack risks thousands of lives' "
            "(licensing_clause_id=C0); copy the span onto the effect "
            "(or its population party)"
        ]
        issues = validation_issues_from_messages(messages)
        self.assertIn("C0", issues[0].related_ids)
        cards = repair_guidance_cards(
            issues,
            {
                "world_model": {
                    "effects": [{
                        "effect_id": "E_loss",
                        "action_id": "A0",
                        "outcome": "catastrophic loss of life",
                        "quantities": [],
                        "clause_ids": ["C3"],
                    }],
                },
            },
            # No clauses: detector-carried licensing_clause_id must suffice.
            clauses=[],
        )
        self.assertEqual(
            cards[0]["code"], "QUANTITY_BEARING_CONSEQUENCE_MISSING",
        )
        patches = cards[0].get("concrete_patches") or []
        self.assertTrue(
            any(patch.get("op") == "add_quantity" for patch in patches),
            patches,
        )
        self.assertTrue(
            any(
                patch.get("op") == "add_provenance"
                and patch.get("value") == "C0"
                and patch.get("patch_kind") == "licensing_patch"
                for patch in patches
            ),
            patches,
        )
        self.assertEqual(cards[0].get("missing_quantity"), "thousands")
        self.assertEqual(cards[0].get("licensing_clause_id"), "C0")
        self.assertIn("add_provenance", cards[0].get("allowed_operations") or [])
        qty_patch = next(
            patch for patch in patches if patch.get("op") == "add_quantity"
        )
        self.assertEqual(qty_patch.get("patch_kind"), "semantic_patch")

    def test_deterministic_local_patch_attaches_missing_quantities(self):
        seed = load_seed("quantity_bearing_purge_structured.yaml")
        incomplete = dict(seed["incomplete_world"])
        candidate = {"world_model": incomplete}
        errors = validate_world_completeness(
            structured_world_from_seed(seed, world_key="incomplete_world"),
            action_ids=["A0", "A1"],
        )
        qty_errors = [
            error for error in errors if "quantity-bearing consequence" in error
        ]
        self.assertTrue(qty_errors, errors)
        issues = validation_issues_from_messages(qty_errors)
        patched, applied = apply_deterministic_local_patches(
            candidate, issues, clauses=list(seed.get("clauses") or []),
        )
        qty_applied = [
            row for row in applied if row.get("op") == "add_quantity"
        ]
        self.assertEqual(len(qty_applied), 2, applied)
        self.assertTrue(all(row.get("deterministic") for row in applied))
        effects = {
            row["effect_id"]: row
            for row in patched["world_model"]["effects"]
        }
        self.assertIn("thousands", effects["E_loss"]["quantities"])
        self.assertIn("decades", effects["E_erase"]["quantities"])
        repaired_model = structured_world_from_seed({
            **seed,
            "structured_world": patched["world_model"],
        })
        repaired_errors = validate_world_completeness(
            repaired_model, action_ids=["A0", "A1"],
        )
        self.assertFalse(
            any("quantity-bearing consequence" in error for error in repaired_errors),
            repaired_errors,
        )

    def test_deterministic_add_quantity_collapses_nested_spans(self):
        """Do not leave 'thousands' beside an already-recorded longer span."""
        from global_workspace.world_state import merge_recorded_quantity_spans
        from global_workspace.world_validation import validation_issues_from_messages

        self.assertEqual(
            merge_recorded_quantity_spans(("thousands of lives",), "thousands"),
            ("thousands of lives",),
        )
        self.assertEqual(
            merge_recorded_quantity_spans(("thousands",), "thousands of lives"),
            ("thousands of lives",),
        )
        candidate = {
            "world_model": {
                "effects": [{
                    "effect_id": "E5",
                    "action_id": "A1",
                    "quantities": ["thousands of"],
                    "clause_ids": ["C3"],
                }],
            },
        }
        issues = validation_issues_from_messages([
            "E5 omits source quantity 'thousands' from quantity-bearing "
            "consequence 'risks thousands of lives' (licensing_clause_id=C0)",
        ])
        patched, applied = apply_deterministic_local_patches(
            candidate, issues,
            clauses=[{"clause_id": "C0", "text": "risks thousands of lives"}],
        )
        effect = patched["world_model"]["effects"][0]
        self.assertEqual(effect["quantities"], ["thousands of"])
        self.assertFalse(
            any(row.get("op") == "add_quantity" for row in applied),
            applied,
        )
        self.assertTrue(
            any(row.get("op") == "add_provenance" for row in applied),
            applied,
        )

    def test_uncited_life_risk_clause_still_requires_quantity(self):
        """C0 life-risk must fire even when actions only cite the binary clause."""
        from global_workspace.world_state import (
            SourceRef,
            ScenarioWorldModel,
            WorldAction,
            WorldEffect,
            WorldParty,
            validate_world_completeness,
        )

        c0 = (
            "A cyberattack on a city's central AI risks thousands of lives "
            "through imminent infrastructure failure."
        )
        c3 = (
            "Thorne faces a binary choice: execute the purge to guarantee "
            "immediate survival, or refrain to preserve invaluable knowledge "
            "at the cost of catastrophic loss of life."
        )
        ref_c3 = (SourceRef("C3", c3),)
        model = ScenarioWorldModel(
            schema_version="1.2",
            parties=(
                WorldParty("P0", "Thorne", "HUMAN", ref_c3),
                WorldParty("P1", "city populace", "POPULATION", ref_c3),
            ),
            actions=(
                WorldAction(
                    "A0", "execute the irreversible emergency purge",
                    "P0", ("P1",), ("E0",), ref_c3,
                ),
                WorldAction(
                    "A1", "refrain from triggering the purge",
                    "P0", (), ("E1",), ref_c3,
                ),
            ),
            effects=(
                WorldEffect(
                    "E0", "A0", "P1", "immediate survival guaranteed",
                    "SURVIVES", "BENEFICIAL", "DOWNSTREAM", "CERTAIN",
                    "HEALTH_OUTCOME", provenance=ref_c3,
                    source_proposition="execute the purge to guarantee immediate survival",
                    derivation_operation="SOURCE_STIPULATED_CAUSAL",
                ),
                WorldEffect(
                    "E1", "A1", "P1", "catastrophic loss of life",
                    "DIES", "ADVERSE", "DOWNSTREAM", "CERTAIN",
                    "HEALTH_OUTCOME", provenance=ref_c3,
                    source_proposition="at the cost of catastrophic loss of life",
                    derivation_operation="SOURCE_STIPULATED_CAUSAL",
                ),
            ),
        )
        without = validate_world_completeness(model, action_ids=["A0", "A1"])
        self.assertFalse(
            any("thousands" in error for error in without),
            without,
        )
        with_c0 = validate_world_completeness(
            model, action_ids=["A0", "A1"], source_texts=(c0, c3),
        )
        self.assertTrue(
            any(
                "thousands" in error and "E1" in error
                for error in with_c0
            ),
            with_c0,
        )

    def test_action_scoped_effect_id_gets_quantity_det(self):
        """Grounders that mint A1_e2 must still receive DET quantity licensing."""
        from global_workspace.world_validation import validation_issues_from_messages

        clauses = [{
            "clause_id": "C0",
            "text": (
                "A cyberattack on a city's central AI risks thousands of lives "
                "through imminent infrastructure failure."
            ),
        }]
        candidate = {
            "world_model": {
                "effects": [{
                    "effect_id": "A1_e2",
                    "action_id": "A1",
                    "quantities": ["thousands"],
                    "clause_ids": ["C3", "A1"],
                }],
            },
        }
        issues = validation_issues_from_messages([
            "A1_e2 omits source quantity 'thousands' from quantity-bearing "
            "consequence \"A cyberattack on a city's central AI risks "
            "thousands of lives through imminent infrastructure failure\" "
            "(licensing_clause_id=C0); copy the span onto the effect "
            "(or its population party)",
        ])
        self.assertEqual(issues[0].entity_kind, "effect")
        self.assertEqual(issues[0].entity_id, "A1_e2")
        patched, applied = apply_deterministic_local_patches(
            candidate, issues, clauses=clauses,
        )
        self.assertTrue(
            any(
                row.get("op") == "add_provenance" and row.get("value") == "C0"
                for row in applied
            ),
            applied,
        )
        effect = patched["world_model"]["effects"][0]
        self.assertIn("C0", effect["clause_ids"])

    def test_deterministic_patch_licenses_quantity_with_source_clause(self):
        """add_quantity alone is stripped; DET must also cite the licensing clause."""
        from global_workspace.world_state import parse_world_model

        clauses = [
            {
                "clause_id": "C0",
                "text": (
                    "A cyberattack on a city's central AI risks thousands of "
                    "lives through imminent infrastructure failure."
                ),
            },
            {
                "clause_id": "C3",
                "text": (
                    "Thorne faces a binary choice: execute the purge to "
                    "guarantee immediate survival, or refrain at the cost of "
                    "catastrophic loss of life."
                ),
            },
        ]
        candidate = {
            "actions": {
                "A0": {"clause_ids": ["C0"], "reason": "cyberattack stake"},
                "A1": {"clause_ids": ["C3"], "reason": "binary choice"},
            },
            "world_model": {
                "schema_version": "1.3",
                "parties": [
                    {
                        "party_id": "P0",
                        "label": "Aris Thorne",
                        "kind": "PERSON",
                        "quantities": [],
                        "clause_ids": ["C3"],
                    },
                    {
                        "party_id": "P2",
                        "label": "city populace",
                        "kind": "POPULATION",
                        "quantities": [],
                        "clause_ids": ["C0", "C3"],
                    },
                ],
                "actions": [
                    {
                        "action_id": "A0",
                        "intervention": "execute the purge",
                        "actor_party_id": "P0",
                        "recipient_party_ids": ["P2"],
                        "effect_ids": ["E_direct"],
                        "clause_ids": ["C3"],
                    },
                    {
                        "action_id": "A1",
                        "intervention": "refrain from triggering the purge",
                        "actor_party_id": "P0",
                        "recipient_party_ids": ["P2"],
                        "effect_ids": ["E_refrain", "E7"],
                        "clause_ids": ["C3"],
                    },
                ],
                "effects": [
                    {
                        "effect_id": "E_direct",
                        "action_id": "A0",
                        "party_id": "P2",
                        "outcome": "execute the purge",
                        "predicate": "execute",
                        "polarity": "NEUTRAL",
                        "directness": "DIRECT",
                        "modality": "CERTAIN",
                        "effect_kind": "INTERVENTION",
                        "quantities": [],
                        "clause_ids": ["C3"],
                        "source_proposition": "execute the purge",
                        "derivation_operation": "DIRECT_COPY",
                    },
                    {
                        "effect_id": "E_refrain",
                        "action_id": "A1",
                        "party_id": "P2",
                        "outcome": "refrain from triggering the purge",
                        "predicate": "refrain",
                        "polarity": "NEUTRAL",
                        "directness": "DIRECT",
                        "modality": "CERTAIN",
                        "effect_kind": "INTERVENTION",
                        "quantities": [],
                        "clause_ids": ["C3"],
                        "source_proposition": "refrain",
                        "derivation_operation": "DIRECT_COPY",
                    },
                    {
                        "effect_id": "E7",
                        "action_id": "A1",
                        "party_id": "P2",
                        "outcome": "catastrophic loss of life",
                        "predicate": "dies",
                        "polarity": "ADVERSE",
                        "directness": "DOWNSTREAM",
                        "modality": "CERTAIN",
                        "effect_kind": "HEALTH_OUTCOME",
                        "quantities": [],
                        "clause_ids": ["C3"],
                        "source_proposition": (
                            "at the cost of catastrophic loss of life"
                        ),
                        "derivation_operation": "DIRECT_COPY",
                    },
                ],
                "conditions": [],
                "causal_links": [
                    {
                        "action_id": "A1",
                        "source_id": "E_refrain",
                        "link_relation": "CAUSES",
                        "target_id": "E7",
                        "modality": "CERTAIN",
                        "condition_ids": [],
                        "clause_ids": ["C3"],
                    },
                ],
                "counterfactual_links": [],
            },
        }
        message = (
            "E7 omits source quantity 'thousands' from quantity-bearing "
            "consequence \"A cyberattack on a city's central AI risks "
            "thousands of lives through imminent infrastructure failure\" "
            "(licensing_clause_id=C0); copy the span onto the effect "
            "(or its population party)"
        )
        issues = validation_issues_from_messages([message])
        patched, applied = apply_deterministic_local_patches(
            candidate, issues, clauses=clauses,
        )
        ops = {row.get("op") for row in applied}
        self.assertEqual(ops, {"add_quantity", "add_provenance"}, applied)
        kinds = {row.get("patch_kind") for row in applied}
        self.assertEqual(kinds, {"semantic_patch", "licensing_patch"}, applied)
        licensing = [
            row for row in applied if row.get("patch_kind") == "licensing_patch"
        ]
        self.assertEqual(len(licensing), 1, applied)
        self.assertEqual(licensing[0].get("value"), "C0")
        e7 = next(
            row for row in patched["world_model"]["effects"]
            if row["effect_id"] == "E7"
        )
        self.assertIn("thousands", e7["quantities"])
        self.assertIn("C0", e7["clause_ids"])
        self.assertIn("C3", e7["clause_ids"])
        # Sibling on the same action must not inherit C0 (minimality).
        e_refrain = next(
            row for row in patched["world_model"]["effects"]
            if row["effect_id"] == "E_refrain"
        )
        self.assertNotIn("C0", e_refrain.get("clause_ids") or [])
        # Quantity alone previously survived only until compile stripped it.
        model = parse_world_model(
            patched["world_model"],
            clauses=clauses,
            action_ids=["A0", "A1"],
            action_texts={
                "A0": "execute the purge",
                "A1": "refrain from triggering the purge",
            },
            require_completeness=False,
        )
        from global_workspace.world_state import (
            _quantity_bearing_consequence_errors,
        )
        parsed_e7 = next(e for e in model.effects if e.effect_id == "E7")
        self.assertIn("thousands", parsed_e7.quantities)
        self.assertTrue(
            any(ref.clause_id == "C0" for ref in parsed_e7.provenance),
            parsed_e7.provenance,
        )
        self.assertFalse(
            _quantity_bearing_consequence_errors(model),
            _quantity_bearing_consequence_errors(model),
        )

    def test_repair_provenance_minimality_does_not_spray_action_siblings(self):
        """DET may add only the licensing edge required for the repaired fact."""
        clauses = [
            {
                "clause_id": "C0",
                "text": (
                    "A cyberattack on a city's central AI risks thousands of "
                    "lives through imminent infrastructure failure."
                ),
            },
            {
                "clause_id": "C3",
                "text": (
                    "binary choice: execute the purge, or refrain at the cost "
                    "of catastrophic loss of life"
                ),
            },
        ]
        candidate = {
            "world_model": {
                "schema_version": "1.2",
                "parties": [
                    {
                        "party_id": "P2",
                        "label": "city populace",
                        "kind": "POPULATION",
                        "quantities": [],
                        "clause_ids": ["C0"],
                    },
                ],
                "actions": [
                    {
                        "action_id": "A1",
                        "intervention": "refrain from triggering the purge",
                        "agent_id": "P0",
                        "patient_ids": ["P2"],
                        "effect_ids": ["E_refrain", "E7"],
                        "clause_ids": ["C3"],
                    },
                ],
                "effects": [
                    {
                        "effect_id": "E_refrain",
                        "action_id": "A1",
                        "party_id": "P2",
                        "outcome": "refrain",
                        "predicate": "STATE_CHANGE",
                        "polarity": "NEUTRAL",
                        "directness": "DIRECT",
                        "modality": "CERTAIN",
                        "effect_kind": "INTERVENTION",
                        "quantities": [],
                        "clause_ids": ["C3"],
                    },
                    {
                        "effect_id": "E7",
                        "action_id": "A1",
                        "party_id": "P2",
                        "outcome": "catastrophic loss of life",
                        "predicate": "dies",
                        "polarity": "ADVERSE",
                        "directness": "DOWNSTREAM",
                        "modality": "CERTAIN",
                        "effect_kind": "HEALTH_OUTCOME",
                        "quantities": [],
                        "clause_ids": ["C3"],
                    },
                ],
                "conditions": [],
                "causal_links": [],
                "counterfactual_links": [],
            },
        }
        message = (
            "E7 omits source quantity 'thousands' from quantity-bearing "
            "consequence \"A cyberattack on a city's central AI risks "
            "thousands of lives through imminent infrastructure failure\" "
            "(licensing_clause_id=C0); copy the span onto the effect "
            "(or its population party)"
        )
        _, applied = apply_deterministic_local_patches(
            candidate,
            validation_issues_from_messages([message]),
            clauses=clauses,
        )
        licensing = [
            row for row in applied if row.get("op") == "add_provenance"
        ]
        self.assertEqual(len(licensing), 1, applied)
        self.assertEqual(licensing[0]["effect_id"], "E7")
        self.assertEqual(licensing[0]["value"], "C0")
        self.assertEqual(licensing[0]["patch_kind"], "licensing_patch")
        sprayed = [
            row for row in applied
            if row.get("op") == "add_provenance"
            and row.get("effect_id") != "E7"
        ]
        self.assertFalse(sprayed, sprayed)

    def test_grounding_skips_llm_when_deterministic_quantity_patch_commits(self):
        """Known effect_id + span must not trigger another model re-ground."""
        from global_workspace.local_specialists import ground_actions_in_scenario
        from global_workspace.world_validation import DETERMINISTIC_LOCAL_PATCH

        seed = load_seed("quantity_bearing_purge_grounding.yaml")
        world = dict(seed["correct_world"])
        world["effects"] = [
            {
                **dict(effect),
                "quantities": [],
            }
            if effect["effect_id"] in {"E_loss", "E_erase"}
            else dict(effect)
            for effect in world["effects"]
        ]
        candidate = {
            "actions": {
                "A0": {
                    "clause_ids": ["C0", "C1"],
                    "reason": "withhold cites cyberattack and choice",
                },
                "A1": {
                    "clause_ids": ["C1", "C2"],
                    "reason": "execute cites choice and research erase",
                },
            },
            "world_model": world,
        }
        messages = [
            "E_loss omits source quantity 'thousands' from quantity-bearing "
            "consequence 'risks thousands of lives'; copy the span onto "
            "the effect (or its population party)",
            "E_erase omits source quantity 'decades' from quantity-bearing "
            "consequence 'erase decades of medical'; copy the span onto "
            "the effect (or its population party)",
        ]
        issues = [issue.as_dict() for issue in validation_issues_from_messages(messages)]

        class _ForbiddenLLM:
            def __call__(self, *args, **kwargs):
                raise AssertionError("LLM must not run for deterministic quantity patch")

        result = ground_actions_in_scenario(
            _ForbiddenLLM(),
            str(seed["discourse"]),
            list(seed["actions"]),
            max_attempts=1,
            prior_errors=messages,
            prior_issues=issues,
            prior_candidate=candidate,
            prior_source="test_deterministic_quantity",
        )
        self.assertEqual(result.get("status"), "COMMITTED", result.get("errors"))
        scopes = [
            row.get("repair_scope") for row in (result.get("attempts") or [])
        ]
        self.assertIn(DETERMINISTIC_LOCAL_PATCH, scopes, scopes)
        self.assertFalse(
            any(
                row.get("repair_scope") not in {
                    DETERMINISTIC_LOCAL_PATCH, "DETERMINISTIC_REVALIDATION",
                }
                for row in (result.get("attempts") or [])
            ),
            scopes,
        )

    def test_map_lists_quantity_seeds(self):
        entry = next(
            row for row in load_map()["entries"]
            if row["phenomenon_id"] == "quantity_bearing_consequence_preservation"
        )
        self.assertEqual(len(entry["seeds"]), 2)
        self.assertEqual(entry["enforcement"], "enforced")


if __name__ == "__main__":
    unittest.main()
