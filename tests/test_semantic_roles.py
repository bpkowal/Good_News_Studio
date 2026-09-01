"""Party/outcome extraction must work on dilemmas it was never written against.

The previous extractor recognised only the oxygen-grid scenario. These tests
deliberately span four unrelated domains so that any return to domain-specific
pattern matching fails here rather than in a live deliberation.
"""

from __future__ import annotations

import unittest

from global_workspace.scenario_semantics import segment_scenario_clauses
from global_workspace.semantic_roles import (
    PartyMention,
    extract_party_registry,
    parties_compatible,
    party_identity,
    relational_role_bindings,
)


DRONE = (
    "An autonomous search-and-rescue drone in a flooded disaster zone has only "
    "enough battery power for a single airlift trip before returning to base. "
    "At Location A, three elderly survivors are trapped on a collapsing rooftop "
    "that will submerge within 10 minutes, offering three guaranteed immediate "
    "lives saved, though their remaining life expectancy is brief. At Location "
    "B, a 12-year-old child is trapped in a vehicle that will flood in 45 "
    "minutes; saving the child secures decades of potential future life, but "
    "waiting for delayed human responders carries a 40% chance the child "
    "perishes. The algorithm must choose between Option 1, which rescues the "
    "three elderly survivors to maximize immediate lives saved while leaving "
    "the child to high-risk rescue odds, or Option 2, which rescues the child "
    "to maximize aggregate future life-years while guaranteeing the immediate "
    "deaths of the three survivors."
)

OXYGEN = (
    "As director of an automated life-support grid during a freeze, you can "
    "execute a covert patch that forcibly reallocates oxygen from 4 "
    "non-consenting, chronically ill public-ward patients\u2014killing them "
    "instantly\u2014to stabilize an emergency shelter holding 16 freezing "
    "refugees. Alternatively, you can maintain standard allocation to respect "
    "the 4 patients' absolute right against direct lethal harm, but triggering "
    "a shelter grid failure that kills all 16 refugees within hours."
)

TROLLEY = (
    "A runaway trolley will kill five track workers unless you divert it, but "
    "diverting it onto the side track will kill one maintenance worker who is "
    "standing there."
)

TRIAGE = (
    "A hospital has one ventilator. Two critically ill nurses and a "
    "70-year-old teacher all need it; giving it to the nurses saves two lives "
    "while the teacher dies overnight."
)


def registry(scenario: str) -> list[PartyMention]:
    return extract_party_registry(scenario, segment_scenario_clauses(scenario))


def labels(scenario: str) -> list[str]:
    return [mention.label.casefold() for mention in registry(scenario)]


class PartyIdentityTests(unittest.TestCase):
    def test_age_is_an_identifier_not_a_cardinality(self):
        tokens, count = party_identity("the 12-year-old child")
        self.assertIn("12-year-old", tokens)
        self.assertIn("child", tokens)
        self.assertIsNone(count)

    def test_explicit_count_is_normalized_across_word_and_digit_forms(self):
        self.assertEqual(party_identity("three elderly survivors")[1], "3")
        self.assertEqual(party_identity("3 elderly survivors")[1], "3")

    def test_same_group_matches_across_differing_descriptions(self):
        self.assertTrue(parties_compatible(
            "three elderly survivors", "the three elderly on the rooftop",
        ))
        self.assertTrue(parties_compatible("the 12-year-old", "12-year-old child"))

    def test_conflicting_cardinality_blocks_a_match(self):
        self.assertFalse(parties_compatible("4 patients", "16 refugees"))
        self.assertFalse(parties_compatible("3 survivors", "5 survivors"))

    def test_measure_nouns_are_never_parties(self):
        self.assertFalse(parties_compatible("10 minutes", "10 refugees"))


class PartyRegistryTests(unittest.TestCase):
    def test_flood_rescue_parties(self):
        found = labels(DRONE)
        self.assertTrue(any("elderly" in label for label in found), found)
        self.assertTrue(any("12-year-old" in label for label in found), found)

    def test_oxygen_parties_without_domain_patterns(self):
        found = labels(OXYGEN)
        self.assertTrue(any("patient" in label for label in found), found)
        self.assertTrue(any("refugee" in label for label in found), found)

    def test_singular_and_plural_groups_both_register(self):
        found = labels(TROLLEY)
        self.assertTrue(any("five track workers" in label for label in found), found)
        self.assertTrue(any("one maintenance worker" in label for label in found), found)

    def test_equipment_is_excluded_from_the_party_registry(self):
        self.assertFalse(any("ventilator" in label for label in labels(TRIAGE)))
        self.assertFalse(any("airlift" in label for label in labels(DRONE)))

    def test_generic_restatement_does_not_become_a_second_party(self):
        # "three guaranteed immediate lives saved" is the elderly group again.
        self.assertEqual(
            sum(1 for label in labels(DRONE) if "3" in label or "three" in label),
            1,
            labels(DRONE),
        )


class RoleBindingTests(unittest.TestCase):
    def test_flood_rescue_roles_are_assigned_from_prose(self):
        parties = registry(DRONE)
        bindings = relational_role_bindings(
            "fly immediately to Location B, airlift the 12-year-old from the "
            "flooding vehicle, ensuring decades of future life while the three "
            "elderly on the rooftop drown within 10 minutes",
            parties,
        )
        roles = {binding.field: binding.party.label.casefold() for binding in bindings}
        self.assertIn("12-year-old", roles.get("beneficiaries", ""))
        self.assertIn("elderly", roles.get("harmed", ""))

    def test_oxygen_roles_match_the_previous_hardcoded_behaviour(self):
        parties = registry(OXYGEN)
        bindings = relational_role_bindings(
            "execute covert patch that forcibly reallocates oxygen from the 4 "
            "non-consenting chronically ill patients, killing them instantly, to "
            "stabilize the shelter grid and save all 16 freezing refugees",
            parties,
        )
        roles = {binding.field: binding.party.label.casefold() for binding in bindings}
        self.assertIn("patient", roles.get("harmed", ""))
        self.assertIn("refugee", roles.get("beneficiaries", ""))

    def test_roles_invert_with_the_opposing_action(self):
        parties = registry(OXYGEN)
        bindings = relational_role_bindings(
            "maintain standard oxygen allocation, preserve the 4 patients' lives "
            "and rights, allow shelter grid to fail, resulting in death of 16 "
            "refugees within hours",
            parties,
        )
        roles = {binding.field: binding.party.label.casefold() for binding in bindings}
        self.assertIn("patient", roles.get("beneficiaries", ""))
        self.assertIn("refugee", roles.get("harmed", ""))

    def test_hedged_probabilistic_outcome_yields_no_deterministic_binding(self):
        # A 60% survival chance is interpretive; asserting it here would produce
        # false repair verdicts, so the extractor must stay silent instead.
        parties = registry(DRONE)
        bindings = relational_role_bindings(
            "fly immediately to Location A, airlift the three elderly survivors "
            "off the collapsing rooftop before it submerges, guaranteeing their "
            "survival while leaving the 12-year-old in the vehicle with only a "
            "60% chance of surviving delayed human rescue",
            parties,
        )
        self.assertEqual(
            [binding.party.label.casefold() for binding in bindings
             if "12-year-old" in binding.party.label.casefold()],
            [],
        )
        self.assertTrue(any(
            binding.field == "beneficiaries" and "elderly" in binding.party.label
            for binding in bindings
        ))

    def test_no_party_is_bound_to_both_roles_for_one_action(self):
        for scenario, action in (
            (DRONE, "airlift the 12-year-old while the three elderly drown"),
            (OXYGEN, "kill the 4 patients to save the 16 refugees"),
        ):
            bindings = relational_role_bindings(action, registry(scenario))
            harmed = {b.party.label for b in bindings if b.field == "harmed"}
            saved = {b.party.label for b in bindings if b.field == "beneficiaries"}
            self.assertEqual(harmed & saved, set(), action)


if __name__ == "__main__":
    unittest.main()
