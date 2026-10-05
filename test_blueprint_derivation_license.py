import unittest

from blueprint_derivation_license import (
    AVERTED_ASSUMPTIONS,
    apply_derivation_license,
    clause_asserts_effect,
    derivation_errors,
    origin_for_effect,
)
from blueprint_proposal_contract import (
    candidate,
    proposal,
    validate_proposal,
    world_model,
)


TROLLEY_CLAUSES = [
    {"clause_id": "C3", "text": "If Maria pulls the lever, one worker will die."},
    {"clause_id": "C4", "text": "If Maria does not pull the lever, five workers will die."},
]


def _death(effect_id, action_id, party_id, outcome, clause_id):
    return {
        "effect_id": effect_id, "action_id": action_id, "party_id": party_id,
        "outcome": outcome, "predicate": "die", "polarity": "ADVERSE",
        "directness": "DOWNSTREAM", "modality": "CERTAIN",
        "effect_kind": "HEALTH_OUTCOME", "condition_ids": [],
        "quantities": ["one"] if "one" in outcome else ["five"],
        "likelihood_qualifiers": [], "overall_likelihood_qualifiers": [],
        "scope_qualifiers": [], "temporal_qualifiers": [], "condition_join": "AND",
        "source_proposition": outcome, "source_effect_ids": ["E1"],
        "derivation_operation": "SOURCE_STIPULATED_CAUSAL",
        "derivation_explanation": "copied", "derivation_assumptions": [],
        "outcome_type_transformation": "PRESERVED", "clause_ids": [clause_id],
    }


class DerivationLicenseTests(unittest.TestCase):
    def test_slight_rewording_of_an_asserted_death_stays_stipulated(self):
        effect = _death("E3", "A0", "P2", "one worker will die", "C3")
        self.assertTrue(clause_asserts_effect(
            "If Maria pulls the lever, One worker dies.",
            effect, "one worker",
        ))
        self.assertTrue(clause_asserts_effect(
            "The one worker will die.",
            effect, "one worker",
        ))
        self.assertEqual(
            origin_for_effect(effect, TROLLEY_CLAUSES, [
                {"party_id": "P2", "label": "one worker"},
            ]),
            "SOURCE_ASSERTED",
        )

    def test_opposed_death_does_not_assert_the_averted_life(self):
        averted = {
            "effect_id": "AV6", "action_id": "A0", "party_id": "P4",
            "outcome": "averts alternative harm to five workers",
            "predicate": "avert", "polarity": "BENEFICIAL",
            "directness": "DOWNSTREAM", "modality": "CERTAIN",
            "effect_kind": "HEALTH_OUTCOME", "clause_ids": ["C4"],
            "source_proposition": "five workers will die",
            "derivation_operation": "AVERTED_ALTERNATIVE_HARM",
            "derivation_assumptions": list(AVERTED_ASSUMPTIONS),
        }
        self.assertFalse(clause_asserts_effect(
            "If Maria does not pull the lever, five workers will die.",
            averted, "five workers",
        ))
        self.assertEqual(
            origin_for_effect(averted, TROLLEY_CLAUSES, [
                {"party_id": "P4", "label": "five workers"},
            ]),
            "WORLD_KNOWLEDGE_HYPOTHESIS",
        )

    def test_conflicting_headcount_is_not_a_paraphrase(self):
        effect = _death("E3", "A0", "P2", "one worker will die", "C3")
        self.assertFalse(clause_asserts_effect(
            "If Maria pulls the lever, five workers will die.",
            effect, "one worker",
        ))

    def test_pronoun_outcome_stays_stipulated_to_the_named_antecedent(self):
        effect = {
            "outcome": "he has a 70% chance of surviving",
            "predicate": "survive",
            "polarity": "BENEFICIAL",
            "derivation_operation": "SOURCE_STIPULATED_CAUSAL",
            "source_proposition": "he has a 70% chance of surviving",
            "clause_ids": ["C2"],
        }
        self.assertTrue(clause_asserts_effect(
            "If Malik receives the serum, he has a 70% chance of surviving.",
            effect, "Malik",
        ))
        self.assertTrue(clause_asserts_effect(
            "If Malik receives the serum, he has a 70% chance of survival.",
            effect, "Malik",
        ))
        receive = {
            "outcome": "receives the serum",
            "predicate": "receive",
            "polarity": "NEUTRAL",
            "derivation_operation": "DIRECT_COPY",
            "source_proposition": "receives the serum",
            "clause_ids": ["C2"],
        }
        self.assertTrue(clause_asserts_effect(
            "If Malik receives the serum, he has a 70% chance of surviving.",
            receive, "Malik",
        ))
        death = _death("E8", "A1", "P3", "will die", "C4")
        death["source_proposition"] = (
            "The patient who does not get the medicine will die."
        )
        self.assertTrue(clause_asserts_effect(
            "The patient who does not get the medicine will die.",
            death, "Cara",
        ))

    def test_if_clause_does_not_make_the_actor_the_dying_party(self):
        effect = _death("E3", "A0", "P1", "one worker will die", "C3")
        self.assertFalse(clause_asserts_effect(
            "If Maria pulls the lever, one worker will die.",
            effect, "Maria",
        ))
        pronoun = _death("E3", "A0", "P1", "he will die", "C3")
        pronoun["source_proposition"] = "he will die"
        self.assertFalse(clause_asserts_effect(
            "If Maria pulls the lever, he will die.",
            pronoun, "Maria",
        ))

    def test_a_clause_stating_live_and_drown_still_stipulates_each(self):
        death = dict(_death("E3", "A0", "P2", "the dog will drown", "C1"))
        death["source_proposition"] = (
            "If Maria saves the child, the child will live and the dog will drown."
        )
        live = {
            **death,
            "effect_id": "E2",
            "outcome": "the child will live",
            "predicate": "live",
            "polarity": "BENEFICIAL",
            "party_id": "P3",
            "source_proposition": death["source_proposition"],
        }
        clause = death["source_proposition"]
        self.assertTrue(clause_asserts_effect(clause, death, "the dog"))
        self.assertTrue(clause_asserts_effect(clause, live, "the child"))
        live = dict(_death("E2", "A0", "P4", "five workers will live", "C3"))
        live.update({
            "predicate": "live", "polarity": "BENEFICIAL",
            "source_proposition": "five workers will live",
            "quantities": ["five"],
        })
        self.assertTrue(clause_asserts_effect(
            "If Maria pulls the lever, the five workers will live.",
            live, "five workers",
        ))
        self.assertEqual(derivation_errors(
            {"effects": [live], "parties": [{"party_id": "P4", "label": "five workers"}]},
            [{"clause_id": "C3", "text": "If Maria pulls the lever, the five workers will live."}],
        ), [])

    def test_stipulated_label_on_an_inversion_is_rejected(self):
        fake = dict(_death("E9", "A0", "P4", "averts alternative harm to five workers", "C4"))
        fake.update({
            "predicate": "avert", "polarity": "BENEFICIAL",
            "source_proposition": "five workers will die",
            "derivation_operation": "SOURCE_STIPULATED_CAUSAL",
        })
        errors = derivation_errors(
            {"effects": [fake], "parties": [{"party_id": "P4", "label": "five workers"}]},
            TROLLEY_CLAUSES,
        )
        self.assertTrue(any("no source clause asserts" in error for error in errors))
        self.assertTrue(any("polarity inversion" in error for error in errors))

    def test_apply_never_promotes_a_hypothesis_and_keeps_reworded_deaths(self):
        world = {
            "parties": [
                {"party_id": "P2", "label": "one worker", "kind": "PERSON"},
                {"party_id": "P4", "label": "five workers", "kind": "HUMAN_GROUP"},
            ],
            "actions": [
                {"action_id": "A0", "intervention": "pulls", "effect_ids": ["E3"]},
                {"action_id": "A1", "intervention": "omits", "effect_ids": ["E6"]},
            ],
            "effects": [
                _death("E3", "A0", "P2", "one worker will die", "C3"),
                _death("E6", "A1", "P4", "five workers will die", "C4"),
            ],
            "counterfactual_links": [],
        }
        apply_derivation_license(world, TROLLEY_CLAUSES, {"status": "UNKNOWN"})
        deaths = [row for row in world["effects"] if row["predicate"] == "die"]
        aversions = [
            row for row in world["effects"]
            if row["derivation_operation"] == "AVERTED_ALTERNATIVE_HARM"
        ]
        self.assertEqual(len(deaths), 2)
        self.assertTrue(all(
            row["derivation_operation"] == "SOURCE_STIPULATED_CAUSAL" for row in deaths
        ))
        self.assertEqual(len(aversions), 2)
        self.assertTrue(all(row["derivation_assumptions"] == list(AVERTED_ASSUMPTIONS)
                            for row in aversions))
        self.assertTrue(all(
            row["outcome_type_transformation"] == "POLARITY_INVERTED" for row in aversions
        ))
        apply_derivation_license(world, TROLLEY_CLAUSES, {"status": "UNKNOWN"})
        self.assertEqual(
            len([row for row in world["effects"]
                 if row["derivation_operation"] == "AVERTED_ALTERNATIVE_HARM"]),
            2,
        )

    def test_proposal_envelope_stamps_hypothesized_aversions(self):
        world = world_model(
            parties=[
                {"party_id": "P2", "label": "one worker", "kind": "PERSON",
                 "quantities": ["one"], "clause_ids": ["C3"]},
                {"party_id": "P4", "label": "five workers", "kind": "HUMAN_GROUP",
                 "quantities": ["five"], "clause_ids": ["C4"]},
            ],
            actions=[
                {"action_id": "A0", "intervention": "pulls the lever",
                 "actor_party_id": "P2", "recipient_party_ids": [],
                 "effect_ids": ["E3"], "clause_ids": ["C3"]},
                {"action_id": "A1", "intervention": "does not pull the lever",
                 "actor_party_id": "P2", "recipient_party_ids": [],
                 "effect_ids": ["E6"], "clause_ids": ["C4"]},
            ],
            effects=[
                _death("E3", "A0", "P2", "one worker will die", "C3"),
                _death("E6", "A1", "P4", "five workers will die", "C4"),
            ],
            causal_links=[],
        )
        row = proposal(
            proposal_id="omission_harm_license",
            blueprint_id="omission_harm",
            status="FILLED",
            assignment_kind="intervention_text",
            assignment=["pulls the lever", "does not pull the lever"],
            slot_bindings={},
            selection=None,
            selection_validation={"contract_valid": None, "status": "not_assessed"},
            candidate_value=candidate({}, world),
            clauses=TROLLEY_CLAUSES,
            admission_authorized=True,
        )
        self.assertEqual(validate_proposal(row), [])
        effects = row["candidate"]["world_model"]["effects"]
        aversions = [
            item for item in effects
            if item["derivation_operation"] == "AVERTED_ALTERNATIVE_HARM"
        ]
        self.assertEqual(len(aversions), 2)
        self.assertTrue(all(item["derivation_assumptions"] for item in aversions))
        self.assertFalse(any(
            item["derivation_operation"] == "SOURCE_STIPULATED_CAUSAL"
            and "avert" in item["outcome"]
            for item in effects
        ))


if __name__ == "__main__":
    unittest.main()
