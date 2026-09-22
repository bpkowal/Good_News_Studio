"""Averted-alternative-harm invariants over Hypothesis-generated worlds.

Oracles are declared case fields. Silent DIRECT_COPY of the adverse quantity
onto survival is forbidden. Licensed inheritance uses AVERTED_ALTERNATIVE_HARM
plus a counterfactual edge.

Grounding lane: discourse → admit → compile must mint the derived claim and
leave survival without a silent source copy; epistemic marks that claim
ESTABLISHED (not hypothetical).
"""
from __future__ import annotations

import unittest
from dataclasses import replace

from hypothesis import given, settings

from global_workspace.epistemic_ledger import seed_proposition_ledger
from global_workspace.relent_adapt import averted_alternative_relational_errors
from global_workspace.scenario_semantics import compile_scenario_graph
from global_workspace.world_state import (
    compile_averted_alternative_harm_overlays,
    compile_chance_gated_world,
    compile_foregone_overlays,
    compile_grounded_quantities,
    is_averted_alternative_harm_effect,
    validate_effect_source_bindings,
    validate_world_model,
)
from invariants.catalog import invariant_by_id
from relent_testkit.coverage import load_map, load_taxonomy
from relent_testkit.harness import admit_world_from_discourse, load_seed
from strategies.averted_alternative_harm import (
    AvertedAlternativeHarmCase,
    averted_alternative_harm_cases,
)


def _survival_has_silent_source_copy(case: AvertedAlternativeHarmCase) -> bool:
    by_id = {effect.effect_id: effect for effect in case.world.effects}
    survival = by_id[case.survival_effect_id]
    return (
        case.life_quantity in survival.quantities
        and survival.derivation_operation == "DIRECT_COPY"
    )


def _licensed_averted_quantity(case: AvertedAlternativeHarmCase) -> bool:
    return any(
        is_averted_alternative_harm_effect(effect)
        and case.life_quantity in effect.quantities
        and case.death_effect_id in effect.source_effect_ids
        for effect in case.world.effects
    )


class AvertedAlternativeHarmHypothesisTests(unittest.TestCase):
    @given(
        averted_alternative_harm_cases().filter(
            lambda case: case.mutation == "adverse_only",
        ),
    )
    @settings(max_examples=25, deadline=None)
    def test_alternative_harm_projects_benefit_when_own_branch_omits_survival(
        self, case: AvertedAlternativeHarmCase,
    ):
        harm_only = replace(
            case.world,
            effects=tuple(
                effect for effect in case.world.effects
                if effect.effect_id != case.survival_effect_id
            ),
        )
        compiled = compile_averted_alternative_harm_overlays(harm_only)
        derived = [
            effect for effect in compiled.effects
            if is_averted_alternative_harm_effect(effect)
        ]
        self.assertTrue(derived, compiled.effects)
        self.assertTrue(
            any(case.life_quantity in effect.quantities for effect in derived),
            derived,
        )
        compact = compile_foregone_overlays(compiled)
        self.assertFalse(
            any(effect.directness == "FOREGONE" for effect in compact.effects),
            "an averted projection must not generate a projection-of-projection",
        )

    @given(
        averted_alternative_harm_cases().filter(
            lambda case: case.mutation == "adverse_only",
        ),
    )
    @settings(max_examples=25, deadline=None)
    def test_compile_does_not_duplicate_quantity_complete_actual_benefit(
        self, case: AvertedAlternativeHarmCase,
    ):
        effects = tuple(
            replace(effect, quantities=(case.life_quantity,))
            if effect.effect_id == case.survival_effect_id else effect
            for effect in case.world.effects
        )
        compiled = compile_averted_alternative_harm_overlays(
            replace(case.world, effects=effects),
        )
        self.assertFalse(
            any(is_averted_alternative_harm_effect(effect)
                for effect in compiled.effects),
            compiled.effects,
        )
        self.assertTrue(
            any(
                link.relation == "PRECLUDES_ALTERNATIVE_EFFECT"
                and link.alternative_effect_id == case.death_effect_id
                for link in compiled.counterfactual_links
            ),
            compiled.counterfactual_links,
        )
        compact = compile_foregone_overlays(compiled)
        self.assertFalse(
            any(
                effect.action_id == "A0" and effect.directness == "FOREGONE"
                for effect in compact.effects
            ),
            "a direct PRECLUDES relation must not trigger a same-action "
            "FOREGONE duplicate",
        )

    @given(averted_alternative_harm_cases())
    @settings(max_examples=40, deadline=None)
    def test_oracle_rejects_silent_source_copy(
        self, case: AvertedAlternativeHarmCase,
    ):
        silent = _survival_has_silent_source_copy(case)
        if case.mutation == "silent_copy_survival":
            self.assertTrue(silent)
            self.assertFalse(case.permits_silent_source_copy)
            # RelEnt non-transfer before compile strip.
            self.assertTrue(averted_alternative_relational_errors(case.world))
            # Compile must strip the unlicensed survival copy.
            stripped = compile_grounded_quantities(case.world)
            surv = next(
                effect for effect in stripped.effects
                if effect.effect_id == case.survival_effect_id
            )
            self.assertNotIn(case.life_quantity, surv.quantities)
        else:
            self.assertFalse(silent)

    @given(averted_alternative_harm_cases())
    @settings(max_examples=40, deadline=None)
    def test_derived_row_licenses_quantity_via_counterfactual(
        self, case: AvertedAlternativeHarmCase,
    ):
        licensed = _licensed_averted_quantity(case)
        self.assertEqual(
            licensed,
            case.expects_licensed_averted_quantity,
            case.mutation,
        )
        if case.mutation == "derived_averted":
            errors = validate_effect_source_bindings(case.world)
            self.assertFalse(
                [error for error in errors if "E_avert" in error],
                errors,
            )
            self.assertTrue(
                any(
                    link.source_effect_id == "E_avert"
                    and link.alternative_effect_id == case.death_effect_id
                    for link in case.world.counterfactual_links
                )
            )

    @given(
        averted_alternative_harm_cases().filter(
            lambda case: case.mutation == "adverse_only",
        ),
    )
    @settings(max_examples=25, deadline=None)
    def test_compile_mints_averted_alternative_harm(
        self, case: AvertedAlternativeHarmCase,
    ):
        compiled = compile_averted_alternative_harm_overlays(case.world)
        derived = [
            effect for effect in compiled.effects
            if is_averted_alternative_harm_effect(effect)
        ]
        self.assertTrue(derived, compiled.effects)
        self.assertTrue(
            any(case.life_quantity in effect.quantities for effect in derived),
            derived,
        )
        self.assertTrue(
            any(
                case.death_effect_id in effect.source_effect_ids
                for effect in derived
            ),
            derived,
        )
        surv = next(
            effect for effect in compiled.effects
            if effect.effect_id == case.survival_effect_id
        )
        self.assertNotIn(case.life_quantity, surv.quantities)
        binding_errors = validate_effect_source_bindings(compiled)
        self.assertFalse(
            [
                error for error in binding_errors
                if "AVERTED_ALTERNATIVE_HARM" in error
                or error.startswith("AV")
            ],
            binding_errors,
        )


class AvertedAlternativeHarmCompileTests(unittest.TestCase):
    def test_chance_gated_world_mints_and_keeps_inherited_quantity(self):
        from strategies.averted_alternative_harm import _averted_world
        from global_workspace.world_state import validate_world_completeness

        world = _averted_world(
            life_source=(
                "A cyberattack on the archive risks thousands of lives "
                "through imminent infrastructure failure."
            ),
            choice_source=(
                "binary choice: execute the purge for immediate survival, "
                "or withhold at the cost of catastrophic loss of life"
            ),
            actor="operator",
            facility="archive",
            life_quantity="thousands",
            mutation="adverse_only",
        )
        compiled = compile_chance_gated_world(world)
        derived = [
            effect for effect in compiled.effects
            if is_averted_alternative_harm_effect(effect)
        ]
        self.assertTrue(derived)
        self.assertIn("thousands", derived[0].quantities)
        self.assertTrue(derived[0].provenance)
        self.assertIn(derived[0].effect_id, compiled.admission.admitted_effect_ids)
        errors, _ = validate_world_model(
            compiled, action_ids=["A0", "A1"],
        )
        self.assertFalse(
            [
                error for error in errors
                if "AVERTED_ALTERNATIVE_HARM" in error
                or error.startswith("AV")
            ],
            errors,
        )
        completeness = validate_world_completeness(
            compiled, action_ids=["A0", "A1"],
        )
        self.assertFalse(
            [error for error in completeness if error.startswith("AV")],
            completeness,
        )

    def test_world_model_from_dict_admits_recompiled_averted_overlays(self):
        """Stale admission lists must not leave AV* WITHHELD after reload."""
        from pathlib import Path
        import json

        from global_workspace.world_state import world_model_from_dict
        from global_workspace.semantic_preservation import (
            build_semantic_preservation_trace,
        )

        trace = Path("workspace_outputs/workspace_workspace_20260914_233315_304849_20260914_233406.json")
        if not trace.exists():
            self.skipTest("latest purge workspace trace not present")
        data = json.loads(trace.read_text(encoding="utf-8"))
        grounding = data.get("action_source_grounding") or {}
        world = grounding.get("world_model") or {}
        # Force a stale admission that omits any AV ids.
        stale = json.loads(json.dumps(world))
        stale["admission"] = {
            "status": "COMMITTED",
            "admitted_effect_ids": [
                effect["effect_id"]
                for effect in stale.get("effects") or []
                if not str(effect.get("effect_id") or "").startswith("AV")
            ],
            "quarantined_effects": [],
            "user_override": False,
            "override_reason": "",
        }
        restored = world_model_from_dict(stale)
        self.assertIsNotNone(restored)
        av_ids = [
            effect.effect_id for effect in restored.effects
            if is_averted_alternative_harm_effect(effect)
        ]
        self.assertTrue(av_ids, restored.effects)
        self.assertTrue(
            set(av_ids) <= set(restored.admission.admitted_effect_ids),
            restored.admission.admitted_effect_ids,
        )
        spt = build_semantic_preservation_trace(
            scenario=str(data.get("scenario") or ""),
            clauses=list(grounding.get("clauses") or []),
            actions=list(data.get("actions") or []),
            grounding={
                **grounding,
                "world_model": restored.as_dict(),
            },
        )
        for chain in spt.get("chains") or []:
            wid = str((chain.get("world") or {}).get("effect_id") or "")
            if wid.startswith("AV"):
                self.assertEqual(chain.get("admitted_status"), "ADMITTED", chain)


class AvertedAlternativeHarmGroundingTests(unittest.TestCase):
    """Live-shaped E1/E7 world through discourse admit + compile + epistemic."""

    def test_taxonomy_and_map_list_grounding_seed(self):
        taxonomy = load_taxonomy()
        row = next(
            item for item in taxonomy["phenomena"]
            if item["id"] == "averted_alternative_harm"
        )
        self.assertEqual(row["source"]["type"], "parliament_extension")
        self.assertIn("grounding", row["lanes"])
        mapping = load_map()
        entry = next(
            item for item in mapping["entries"]
            if item["phenomenon_id"] == "averted_alternative_harm"
        )
        self.assertIn(
            "averted_alternative_harm_grounding.yaml",
            entry["seeds"],
        )
        item = invariant_by_id()["AVERTED_ALTERNATIVE_HARM"]
        self.assertEqual(item.enforcement, "enforced")

    def test_grounding_lane_mints_averted_quantity_without_silent_survival_copy(self):
        """Admit the bare E1/E7 shape; compile must invent AV*, not copy onto E1."""
        seed = load_seed("averted_alternative_harm_grounding.yaml")
        expected = seed["parliament_expectation"]["expected"]
        discourse = " ".join(str(seed["discourse"]).split())
        admitted = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["adverse_only_world"]),
        )
        surv = next(
            effect for effect in admitted.effects
            if effect.effect_id == expected["survival_effect_id"]
        )
        death = next(
            effect for effect in admitted.effects
            if effect.effect_id == expected["death_effect_id"]
        )
        self.assertNotIn(expected["life_quantity"], surv.quantities)
        self.assertIn(expected["life_quantity"], death.quantities)

        compiled = compile_chance_gated_world(admitted)
        derived = [
            effect for effect in compiled.effects
            if is_averted_alternative_harm_effect(effect)
        ]
        self.assertTrue(derived, "compiler should mint AVERTED_ALTERNATIVE_HARM")
        self.assertTrue(
            any(
                expected["life_quantity"] in effect.quantities
                and expected["death_effect_id"] in effect.source_effect_ids
                for effect in derived
            ),
            derived,
        )
        # Derived rows carry inherited opposed-harm provenance (licensed DERIVED).
        self.assertTrue(
            all(effect.provenance for effect in derived),
            derived,
        )
        surv_after = next(
            effect for effect in compiled.effects
            if effect.effect_id == expected["survival_effect_id"]
        )
        self.assertNotIn(expected["life_quantity"], surv_after.quantities)
        self.assertEqual(surv_after.derivation_operation, "DIRECT_COPY")
        # Operative admission must include the derived overlays.
        self.assertTrue(
            set(effect.effect_id for effect in derived)
            <= set(compiled.admission.admitted_effect_ids),
            compiled.admission.admitted_effect_ids,
        )

    def test_grounding_lane_strips_silent_survival_copy(self):
        seed = load_seed("averted_alternative_harm_grounding.yaml")
        expected = seed["parliament_expectation"]["expected"]
        discourse = " ".join(str(seed["discourse"]).split())
        admitted = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["silent_copy_world"]),
        )
        compiled = compile_chance_gated_world(admitted)
        surv = next(
            effect for effect in compiled.effects
            if effect.effect_id == expected["survival_effect_id"]
        )
        self.assertNotIn(expected["life_quantity"], surv.quantities)
        derived = [
            effect for effect in compiled.effects
            if is_averted_alternative_harm_effect(effect)
        ]
        self.assertTrue(derived)
        self.assertTrue(
            any(expected["life_quantity"] in effect.quantities for effect in derived),
            derived,
        )

    def test_epistemic_marks_averted_claim_established(self):
        """Util-style 'saves thousands' must bind to an ESTABLISHED world fact."""
        seed = load_seed("averted_alternative_harm_grounding.yaml")
        expected = seed["parliament_expectation"]["expected"]
        discourse = " ".join(str(seed["discourse"]).split())
        admitted = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["adverse_only_world"]),
        )
        compiled = compile_chance_gated_world(admitted)
        derived = [
            effect for effect in compiled.effects
            if is_averted_alternative_harm_effect(effect)
        ]
        self.assertTrue(derived)
        averted = derived[0]

        graph = compile_scenario_graph(
            discourse,
            list(seed["actions"]),
            world_model=compiled,
        )
        ledger = seed_proposition_ledger(graph)
        prop_id = f"PROP:WORLD:{averted.effect_id}"
        self.assertIn(prop_id, ledger, sorted(ledger))
        record = ledger[prop_id]
        self.assertEqual(
            record.epistemic_status,
            expected["expects_epistemic_status"],
            record,
        )
        self.assertEqual(record.epistemic_type, "WORLD_ESTABLISHED")
        self.assertIn(expected["life_quantity"], record.quantities)
        self.assertTrue(
            any("avert" in part.casefold() for part in record.claim.split(";"))
            or "avert" in record.claim.casefold(),
            record.claim,
        )
        # Survival itself still has no magnitude — the derived row carries it.
        surv_prop = ledger.get(f"PROP:WORLD:{expected['survival_effect_id']}")
        self.assertIsNotNone(surv_prop)
        self.assertNotIn(expected["life_quantity"], surv_prop.quantities)


if __name__ == "__main__":
    unittest.main()
