import unittest

from blueprint_allocation_invariants import party_kind
from blueprint_admission_core import supported_core
from blueprint_cloze_chooser import _predicate, assess_question, choose_by_cloze
from blueprint_kind_license import (
    REGEX_CATALOG,
    apply_kind_license,
    catalog_roles,
    license_effect_kind,
    license_kind,
    licensed_source_predicate,
    locative_medium,
)
from test_blueprint_cloze_chooser import (
    MEDICINE,
    RESCUE,
    TROLLEY,
    WATER,
    Script,
    _none_sheet,
)


class RegexCatalogTests(unittest.TestCase):
    def test_inventory_has_no_ontology_induction(self):
        roles = catalog_roles()
        self.assertEqual(roles["ontology_induction"], [])
        self.assertIn("_TRANSFER_EVENT", roles["witness"])
        self.assertIn("_NONRECEIPT", roles["witness"])
        self.assertIn("_EXPLICIT_EXCLUSIVITY", roles["witness"])
        self.assertIn("_GROUP", roles["witness"])
        self.assertIn("party_kind", roles["negative_control"])
        self.assertIn("_INDIVISIBLE_ONE", roles["negative_control"])
        self.assertIn("_PROCESS_PREDICATES", roles["negative_control"])
        self.assertIn("_PROCESS_BEARER", roles["negative_control"])
        names = {row["name"] for row in REGEX_CATALOG}
        self.assertNotIn("train", " ".join(names).casefold())


class KindMeaningTests(unittest.TestCase):
    def test_lexicon_does_not_write_person_or_resource(self):
        self.assertEqual(party_kind("Maria"), "OTHER")
        self.assertEqual(party_kind("the lever"), "OTHER")
        self.assertEqual(license_kind("the lever")["kind"], "OTHER")
        self.assertEqual(license_kind("the lever")["origin"], "UNRESOLVED")
        self.assertEqual(license_kind("one dose")["kind"], "OTHER")
        self.assertEqual(license_kind("one dose")["origin"], "UNRESOLVED")

    def test_lever_is_not_a_welfare_bearer(self):
        licensed = license_kind(
            "the lever", role="process", construction="omission_harm",
            action_span="pulls the lever")
        self.assertEqual(licensed["kind"], "PROCESS")
        self.assertEqual(licensed["origin"], "STRUCTURALLY_DERIVED")
        self.assertNotEqual(licensed["kind"], "PERSON")
        welfare = license_kind("one worker", role="bearer", construction="omission_harm")
        self.assertEqual(welfare["kind"], "PERSON")
        self.assertNotEqual(license_kind("the lever")["kind"], "PERSON")

    def test_water_as_medium_is_not_water_as_resource(self):
        medium = "A child and a dog are in the water. Maria can save either."
        self.assertTrue(locative_medium(medium, "water"))
        blocked = license_kind(
            "water", role="resource", text=medium, construction="exclusive_allocation")
        self.assertEqual(blocked["kind"], "OTHER")
        self.assertIn("locative_medium", blocked["witness"]["vetoes"])
        allocated = license_kind(
            "water", role="resource", text=WATER,
            construction="exclusive_allocation")
        self.assertEqual(allocated["kind"], "RESOURCE")
        self.assertFalse(locative_medium(WATER, "water"))

    def test_dose_is_not_sufficient_for_resource(self):
        self.assertEqual(party_kind("one dose"), "RESOURCE")
        self.assertEqual(license_kind("one dose")["kind"], "OTHER")
        licensed = license_kind(
            "one dose of medicine", role="resource",
            construction="exclusive_allocation", quantities=["one"])
        self.assertEqual(licensed["kind"], "RESOURCE")
        recorded = license_effect_kind(
            "RESOURCE_TRANSFER", span="Ada records one dose",
            construction="exclusive_allocation",
            parties=[{"kind": "OTHER", "kind_origin": "UNRESOLVED"}],
        )
        self.assertEqual(recorded["kind"], "INTERVENTION")
        self.assertIn("no_give_or_receive_span", recorded["vetoes"])
        given = license_effect_kind(
            "RESOURCE_TRANSFER", span="Ada gives one dose",
            construction="exclusive_allocation",
            parties=[{"kind": "OTHER", "kind_origin": "UNRESOLVED"}],
        )
        self.assertEqual(given["kind"], "INTERVENTION")
        self.assertIn("no_licensed_resource", given["vetoes"])

    def test_bare_or_is_not_exclusivity(self):
        hypothesized = assess_question("Ada can give one dose to Ben or Cara.")
        explicit = assess_question(
            "Ada can give one dose to Ben or Cara, but not both.")
        self.assertEqual(hypothesized["exclusivity_proof"]["status"], "HYPOTHESIZED")
        self.assertEqual(explicit["exclusivity_proof"]["status"], "EXPLICIT")
        self.assertNotEqual(hypothesized["exclusivity"], "evidenced")


class LicensedPredicateTests(unittest.TestCase):
    def test_copied_receive_wins_over_first_word_fallback(self):
        self.assertEqual(
            licensed_source_predicate("Malik receives the serum"), "receive")
        self.assertEqual(_predicate("Malik receives the serum"), "receive")
        self.assertNotEqual(_predicate("Malik receives the serum"), "malik")

    def test_diverts_the_train_does_not_mint_a_train_kind(self):
        self.assertEqual(licensed_source_predicate("diverts the train"), "divert")
        self.assertEqual(license_kind("the train")["kind"], "OTHER")
        self.assertEqual(license_kind("the train")["origin"], "UNRESOLVED")


class SemanticFreezeTests(unittest.TestCase):
    def test_ada_medicine_kinds(self):
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": {
                    "decider": "Ada", "resource": "the medicine",
                    "assignment": "give the medicine", "quantity": "one dose",
                    "first_recipient": "Ben", "second_recipient": "Cara",
                    "exclusivity": "but not both",
                    "first_outcome": "Ben has a 95% chance of survival",
                    "second_outcome": "Cara has a 5% chance of survival",
                    "nonreceipt": "The patient who does not get the medicine will die",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        )
        world = choose_by_cloze(MEDICINE, script)["graph"]["candidate"]["world_model"]
        kinds = {row["label"]: (row["kind"], row.get("kind_origin"))
                 for row in world["parties"]}
        self.assertEqual(kinds["Ada"][0], "PERSON")
        self.assertEqual(kinds["the medicine"][0], "RESOURCE")
        self.assertEqual(kinds["Ben"][0], "PERSON")
        self.assertEqual(kinds["Cara"][0], "PERSON")
        self.assertTrue(all(origin != "UNRESOLVED" for _, origin in kinds.values()))
        self.assertTrue(any(row["effect_kind"] == "RESOURCE_TRANSFER"
                            for row in world["effects"]))

    def test_child_dog_rescue_kinds(self):
        script = Script(
            ["rescue_contrast", "exclusive_allocation", "conditional_outcome"],
            {
                "rescue_contrast": {
                    "rescuer": "Maria", "first_saved": "the child",
                    "second_saved": "the dog", "rescue_exclusivity": "but not both",
                    "first_rescue_action": "Maria saves the child",
                    "second_rescue_action": "Maria saves the dog",
                    "first_benefit": "the child will live",
                    "first_harm": "the dog will drown",
                    "second_benefit": "the dog will live",
                    "second_harm": "the child will drown", "scene": "NONE",
                },
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
                "conditional_outcome": _none_sheet("conditional_outcome"),
            },
        )
        world = choose_by_cloze(RESCUE, script)["graph"]["candidate"]["world_model"]
        kinds = {row["label"]: row["kind"] for row in world["parties"]}
        self.assertEqual(kinds["Maria"], "PERSON")
        self.assertEqual(kinds["the child"], "PERSON")
        self.assertEqual(kinds["the dog"], "ANIMAL")
        self.assertNotIn("RESOURCE", kinds.values())

    def test_maria_lever_is_process_not_person(self):
        script = Script(
            ["omission_harm", "conditional_outcome", "exclusive_allocation"],
            {
                "omission_harm": {
                    "actor": "Maria", "done": "pulls the lever",
                    "omitted": "does not pull the lever",
                    "harm_done": "one worker will die",
                    "harm_omitted": "five workers will die",
                    "done_hedge": "NONE", "omitted_hedge": "NONE",
                    "instrument": "the brake", "group_counts": "NONE",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        )
        world = choose_by_cloze(TROLLEY, script)["graph"]["candidate"]["world_model"]
        lever = next(row for row in world["parties"] if row["label"] == "the lever")
        self.assertEqual(lever["kind"], "PROCESS")
        self.assertEqual(lever["kind_origin"], "STRUCTURALLY_DERIVED")
        self.assertNotEqual(lever["kind"], "PERSON")
        self.assertTrue(any(row["effect_kind"] == "PHYSICAL_STATE"
                            for row in world["effects"]))
        self.assertFalse(any(row["label"] == "the train" for row in world["parties"]))

    def test_water_farm_keeps_medium_and_or_out_of_the_world(self):
        script = Script(
            ["exclusive_allocation", "conditional_outcome", "rescue_contrast"],
            {
                "exclusive_allocation": {
                    "decider": "An AI bot", "resource": "water",
                    "assignment": "devote water",
                    "first_recipient": "a farm that can sustain a city's dietary needs",
                    "second_recipient": "a small town that will lose 100 people to drowning",
                    "exclusivity": "NONE", "quantity": "NONE",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "rescue_contrast": _none_sheet("rescue_contrast"),
            },
        )
        result = choose_by_cloze(WATER, script)
        self.assertEqual(result["question"]["exclusivity"], "unspecified")
        self.assertIsNone(result["graph"])
        self.assertEqual(
            license_kind("water", role="resource", text=WATER,
                         construction="exclusive_allocation")["kind"],
            "RESOURCE",
        )


class OverlayAndApplyTests(unittest.TestCase):
    def test_unlicensed_transfer_is_overlaid(self):
        proposal = {"candidate": {"world_model": {
            "parties": [{"party_id": "P1", "label": "Ada", "kind": "PERSON",
                         "kind_origin": "STRUCTURALLY_DERIVED"}],
            "actions": [{"action_id": "A0", "effect_ids": ["E1"]}],
            "effects": [{
                "effect_id": "E1", "effect_kind": "RESOURCE_TRANSFER",
                "outcome": "waves", "source_effect_ids": [],
            }],
            "causal_links": [], "temporal_relations": [],
            "counterfactual_links": [],
        }}}
        core, overlay = supported_core(proposal)
        self.assertEqual(core["candidate"]["world_model"]["effects"], [])
        self.assertIn("unlicensed_resource_transfer", overlay["reasons"]["E1"])

    def test_apply_kind_license_demotes_transfer_without_resource(self):
        world = {
            "parties": [{"party_id": "P1", "label": "Ada", "kind": "PERSON",
                         "kind_origin": "STRUCTURALLY_DERIVED"}],
            "effects": [{
                "effect_id": "E1", "effect_kind": "RESOURCE_TRANSFER",
                "predicate": "give", "outcome": "Ada gives a lecture",
                "source_proposition": "Ada gives a lecture",
                "scope_qualifiers": [],
            }],
        }
        apply_kind_license(world, [{"clause_id": "C0", "text": "Ada gives a lecture"}],
                           "conditional_outcome")
        self.assertEqual(world["effects"][0]["effect_kind"], "INTERVENTION")


if __name__ == "__main__":
    unittest.main()
