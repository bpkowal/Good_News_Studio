"""VERB_LEMMA_OUTCOME_BINDING — FraCaS §8 morphological lemma binding."""
from __future__ import annotations

import unittest

from relent_testkit.coverage import load_map, load_taxonomy
from relent_testkit.harness import (
    admit_world_from_discourse,
    load_seed,
    structured_world_from_seed,
    verb_lemma_outcome_binding_holds,
)
from invariants.catalog import invariant_by_id


class VerbLemmaOutcomeBindingTests(unittest.TestCase):
    def test_taxonomy_marks_fracas_section_eight(self):
        row = next(
            item for item in load_taxonomy()["phenomena"]
            if item["id"] == "verb_lemma_outcome_binding"
        )
        self.assertEqual(row["source"]["type"], "fracas")
        self.assertEqual(row["source"]["section"], 8)

    def test_catalog_entry(self):
        item = invariant_by_id()["VERB_LEMMA_OUTCOME_BINDING"]
        self.assertEqual(item.source_type, "fracas")
        self.assertTrue(item.integrity_layers.grounding_property)

    def test_structured_lane_binds_purge_lemma(self):
        seed = load_seed("verb_lemma_purge_structured.yaml")
        model = structured_world_from_seed(seed)
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            verb_lemma_outcome_binding_holds(
                model, effect_id=expected["effect_id"],
            )
        )

    def test_structured_lane_detects_wrong_lemma(self):
        seed = load_seed("verb_lemma_purge_structured.yaml")
        model = structured_world_from_seed(seed, world_key="wrong_lemma_world")
        expected = seed["parliament_expectation"]["expected"]
        self.assertFalse(
            verb_lemma_outcome_binding_holds(
                model, effect_id=expected["effect_id"],
            )
        )

    def test_grounding_lane_admits_purge_lemma(self):
        seed = load_seed("verb_lemma_purge_grounding.yaml")
        discourse = " ".join(str(seed["discourse"]).split())
        model = admit_world_from_discourse(
            discourse,
            actions=list(seed["actions"]),
            world_candidate=dict(seed["correct_world"]),
        )
        expected = seed["parliament_expectation"]["expected"]
        self.assertTrue(
            verb_lemma_outcome_binding_holds(
                model, effect_id=expected["effect_id"],
            )
        )

    def test_map_lists_lemma_seeds(self):
        entry = next(
            row for row in load_map()["entries"]
            if row["phenomenon_id"] == "verb_lemma_outcome_binding"
        )
        self.assertEqual(len(entry["seeds"]), 2)

    def test_effect_kind_predicate_does_not_break_lemma_binding(self):
        """Grounders often copy effect_kind into predicate; that is not a lemma."""
        from global_workspace.world_state import (
            SourceRef,
            WorldEffect,
            WorldParty,
            _effect_outcome_and_predicate,
            _identity_head_text,
            _source_proposition_supports_outcome,
            _verb_lemma_binding_errors,
            ScenarioWorldModel,
            WorldAction,
        )

        outcome, predicate = _effect_outcome_and_predicate({
            "outcome": "refrain",
            "predicate": "INTERVENTION",
            "effect_kind": "INTERVENTION",
        })
        self.assertEqual(outcome, "refrain")
        self.assertEqual(predicate, "EXPERIENCES")

        ref = (SourceRef("C0", "refrain"),)
        effect = WorldEffect(
            "E0", "A0", "P0", "refrain", "INTERVENTION",
            "NEUTRAL", "DIRECT", "CERTAIN", "INTERVENTION",
            provenance=ref,
            source_proposition="refrain",
            derivation_operation="DIRECT_COPY",
        )
        party = WorldParty("P0", "operator", "HUMAN", ref)
        self.assertEqual(_identity_head_text(effect), "refrain")
        self.assertTrue(_source_proposition_supports_outcome(effect, party))
        world = ScenarioWorldModel(
            schema_version="1.2",
            parties=(party,),
            actions=(WorldAction("A0", "refrain", "P0", (), ("E0",), ref),),
            effects=(effect,),
        )
        self.assertEqual(_verb_lemma_binding_errors(world), [])

    def test_typed_allocation_predicate_does_not_break_active_passive_binding(self):
        """Ontology predicates must not dilute given/give event identity."""
        from global_workspace.world_state import (
            ScenarioWorldModel,
            SourceRef,
            WorldAction,
            WorldEffect,
            WorldParty,
            _source_proposition_supports_outcome,
            _verb_lemma_binding_errors,
        )

        ref = (SourceRef("C4", "give it to Patient B"),)
        patient = WorldParty("P2", "Patient B", "PATIENT", ref)
        effect = WorldEffect(
            "E9", "A1", "P2", "is given the medicine", "RECEIVES_MEDICINE",
            "BENEFICIAL", "DIRECT", "CERTAIN", "RESOURCE_TRANSFER",
            provenance=ref,
            source_proposition="give it to Patient B",
            derivation_operation="DIRECT_COPY",
        )
        world = ScenarioWorldModel(
            schema_version="1.3",
            parties=(patient,),
            actions=(WorldAction(
                "A1", "Give it to Patient B", "", ("P2",), ("E9",), ref,
            ),),
            effects=(effect,),
        )
        self.assertTrue(_source_proposition_supports_outcome(effect, patient))
        self.assertEqual([], _verb_lemma_binding_errors(world))

    def test_short_dies_form_binds_to_source_die_lemma(self):
        from global_workspace.world_state import (
            ScenarioWorldModel,
            SourceRef,
            WorldAction,
            WorldEffect,
            WorldParty,
            _source_proposition_supports_outcome,
            _verb_lemma_binding_errors,
        )

        ref = (SourceRef("C0", "two patients who will die without it"),)
        patient = WorldParty("PB", "Patient B", "PATIENT", ref)
        effect = WorldEffect(
            "E_DEATH", "A0", "PB", "dies", "DIES", "ADVERSE",
            "DOWNSTREAM", "CERTAIN", "HEALTH_OUTCOME", provenance=ref,
            source_proposition="will die without it",
            source_effect_ids=("E_NORECV",),
            derivation_operation="SOURCE_STIPULATED_CAUSAL",
            derivation_explanation="The patient without the dose dies.",
        )
        world = ScenarioWorldModel(
            schema_version="1.3", parties=(patient,),
            actions=(WorldAction("A0", "allocate the dose"),),
            effects=(effect,),
        )
        self.assertTrue(_source_proposition_supports_outcome(effect, patient))
        self.assertEqual([], _verb_lemma_binding_errors(world))

    def test_elliptical_likelihood_expands_to_containing_source_clause(self):
        from global_workspace.world_state import (
            ScenarioWorldModel,
            SourceRef,
            WorldAction,
            WorldEffect,
            WorldParty,
            compile_source_proposition_spans,
        )

        clause = (
            "Patient A has a 90% chance of surviving if given the medicine, "
            "while Patient B has a 40% chance."
        )
        ref = (SourceRef("C1", clause),)
        patient = WorldParty("PB", "Patient B", "PATIENT", ref)
        effect = WorldEffect(
            "EB", "A1", "PB", "survives", "SURVIVES", "BENEFICIAL",
            "DOWNSTREAM", "PROBABILISTIC", "HEALTH_OUTCOME",
            provenance=ref, likelihood_qualifiers=("40% chance",),
            source_proposition="Patient B has a 40% chance",
            derivation_operation="SOURCE_STIPULATED_CAUSAL",
            source_effect_ids=("EA",), derivation_explanation="Treatment chance.",
        )
        world = ScenarioWorldModel(
            schema_version="1.3", parties=(patient,),
            actions=(WorldAction("A1", "give to Patient B"),),
            effects=(effect,),
        )
        compiled = compile_source_proposition_spans(world)
        self.assertEqual(clause, compiled.effects[0].source_proposition)


if __name__ == "__main__":
    unittest.main()
