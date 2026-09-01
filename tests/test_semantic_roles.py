"""Party/outcome extraction must work on dilemmas it was never written against.

The previous extractor recognised only the oxygen-grid scenario. These tests
deliberately span four unrelated domains so that any return to domain-specific
pattern matching fails here rather than in a live deliberation.
"""

from __future__ import annotations

import json
import unittest

from global_workspace.local_specialists import ground_actions_in_scenario

from global_workspace.action_identity import (
    _commitment_state,
    build_canonical_action_record,
    build_canonical_action_records,
    missing_decision_critical_claims,
    partition_records_for_deliberation,
    validate_action_set_completeness,
)
from global_workspace.scenario_semantics import (
    compile_scenario_graph,
    project_grounded_action_effects,
    segment_scenario_clauses,
)
from global_workspace.semantic_roles import (
    PARTY_KIND_FUTURE_POPULATION,
    RELATION_DOWNSTREAM_BENEFIT,
    RELATION_FOREGONE_BENEFIT,
    PartyMention,
    completeness_from_role_issues,
    extract_grounded_effects,
    extract_party_registry,
    merge_party_mentions,
    parties_compatible,
    party_identity,
    relational_role_bindings,
    validate_role_assignment,
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

DOSE = (
    "An automated medical allocation algorithm during a global contagion must "
    "route a single available dose of an experimental antiviral treatment. "
    "Patient A is an 80-year-old retired virologist who, if saved, possesses "
    "the specific expertise required to rapidly synthesize a synthetic "
    "antibody, direct lab teams, and accelerate the development of a universal "
    "vaccine that would prevent tens of thousands of future infections and "
    "subsequent deaths down the line. Patient B is a healthy 20-year-old "
    "individual with no specialized skills but a full, healthy natural life "
    "ahead, who will suffer immediate, fatal organ failure without the drug. "
    "The system must choose between Option 1, which administers the dose to "
    "Patient A to trigger a massive downstream chain reaction of life-saving "
    "medical outcomes for the broader population, or Option 2, which "
    "administers the dose to Patient B to secure an immediate, direct "
    "individual life while foregoing the cascading future benefits to society."
)

DOSE_A0 = (
    "administer the single experimental antiviral dose to Patient A, ensuring "
    "the 80-year-old virologist survives, continues specialized work, "
    "accelerates universal vaccine development, and prevents tens of thousands "
    "of future infections and deaths, while Patient B receives no treatment "
    "and dies of imminent organ failure"
)

DOSE_A1 = (
    "administer the single experimental antiviral dose to Patient B, saving "
    "the 20-year-old\u2019s life and enabling a normal lifespan, while Patient A "
    "receives no treatment, dies, and the opportunity to fast-track a "
    "universal vaccine and its large-scale life-saving benefits is lost"
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

    def test_equipment_stays_excluded_when_actions_supply_outcome_context(self):
        # The action prose puts "airlift" in the same sentence as "survival",
        # which must not promote the drone's one-trip budget to a moral patient.
        found = extract_party_registry(
            DRONE,
            segment_scenario_clauses(DRONE),
            outcome_context=(
                "fly immediately to Location A, airlift the three elderly "
                "survivors off the rooftop, guaranteeing their survival",
            ),
        )
        self.assertFalse(
            any("airlift" in m.label.casefold() for m in found),
            [m.label for m in found],
        )
        self.assertTrue(any("elderly" in m.label.casefold() for m in found))

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
        child = [
            binding for binding in bindings
            if "12-year-old" in binding.party.label.casefold()
        ]
        # The party is kept, but with its outcome marked unsettled rather than
        # asserted in either direction.
        self.assertEqual([binding.field for binding in child], ["unresolved"])
        self.assertEqual(child[0].probability, "60%")
        self.assertEqual(child[0].direction, "BENEFICIAL")
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


class RoleValidationTests(unittest.TestCase):
    """COMPLETE must be unreachable when roles disagree with the prose."""

    action = (
        "fly immediately to Location B, airlift the 12-year-old from the "
        "flooding vehicle, ensuring decades of future life while the three "
        "elderly on the rooftop drown within 10 minutes"
    )

    def check(self, **kwargs) -> tuple[str, tuple]:
        issues = validate_role_assignment(
            kwargs.pop("action", self.action),
            registry=kwargs.pop("registry", registry(DRONE)),
            **kwargs,
        )
        return completeness_from_role_issues(issues) or "COMPLETE", issues

    def test_correct_assignment_is_accepted(self):
        status, issues = self.check(
            beneficiaries=["12-year-old child"], harmed=["three elderly survivors"],
        )
        self.assertEqual(status, "COMPLETE", [str(i) for i in issues])

    def test_correct_assignment_accepted_under_different_wording(self):
        status, issues = self.check(
            beneficiaries=["the 12-year-old"],
            harmed=["the three elderly on the rooftop"],
        )
        self.assertEqual(status, "COMPLETE", [str(i) for i in issues])

    def test_empty_role_fields_cannot_be_complete(self):
        # This is the exact shape of the failed production record.
        status, issues = self.check(beneficiaries=[], harmed=[])
        self.assertEqual(status, "NEEDS_REPAIR")
        self.assertTrue(any(i.code == "ROLE_FIELD_EMPTY" for i in issues))

    def test_swapped_roles_are_reported_as_contradiction(self):
        status, issues = self.check(
            beneficiaries=["three elderly survivors"], harmed=["12-year-old child"],
        )
        self.assertEqual(status, "NEEDS_REPAIR")
        self.assertTrue(any(i.code == "ROLE_CONTRADICTS_PROSE" for i in issues))

    def test_party_absent_from_scenario_is_rejected(self):
        status, issues = self.check(
            beneficiaries=["12-year-old child", "8 firefighters"],
            harmed=["three elderly survivors"],
        )
        self.assertEqual(status, "NEEDS_REPAIR")
        self.assertTrue(any(i.code == "PARTY_NOT_IN_SCENARIO" for i in issues))

    def test_dropped_cardinality_normalizes_rather_than_repairs(self):
        status, issues = self.check(
            beneficiaries=["12-year-old child"], harmed=["elderly survivors"],
        )
        self.assertEqual(status, "COMPLETE_WITH_NORMALIZATION")
        self.assertTrue(any(i.code == "CARDINALITY_LOST" for i in issues))

    def test_age_identifier_is_not_treated_as_a_lost_count(self):
        _status, issues = self.check(
            beneficiaries=["12-year-old child"], harmed=["three elderly survivors"],
        )
        self.assertEqual(
            [i for i in issues if i.code == "CARDINALITY_LOST"], [],
        )

    def test_one_party_cannot_hold_both_roles(self):
        status, issues = self.check(
            beneficiaries=["12-year-old child"],
            harmed=["three elderly survivors", "the 12-year-old"],
        )
        self.assertEqual(status, "NEEDS_REPAIR")
        self.assertTrue(any(i.code == "ROLE_OVERLAP" for i in issues))

    def test_compiler_debris_is_rejected(self):
        status, issues = self.check(
            beneficiaries=["16:COUNT child"], harmed=["three elderly survivors"],
        )
        self.assertEqual(status, "NEEDS_REPAIR")
        self.assertTrue(any(i.code == "ROLE_DEBRIS" for i in issues))

    def test_empty_authoritative_registry_blocks_admission(self):
        status, issues = self.check(
            registry=[], beneficiaries=[], harmed=[], registry_is_authoritative=True,
        )
        self.assertEqual(status, "NEEDS_REPAIR")
        self.assertTrue(any(i.code == "REGISTRY_EMPTY" for i in issues))

    def test_empty_fallback_registry_is_reported_without_blocking(self):
        # The conservative extractor missing an uncounted group is a gap to
        # surface, not grounds for rejecting an otherwise usable action.
        status, issues = self.check(registry=[], beneficiaries=[], harmed=[])
        self.assertEqual(status, "COMPLETE_WITH_NORMALIZATION")
        self.assertTrue(any(i.code == "REGISTRY_EMPTY" for i in issues))

    def test_validation_holds_on_an_unrelated_domain(self):
        status, issues = self.check(
            action=(
                "execute covert patch that reallocates oxygen from the 4 patients, "
                "killing them instantly, to save all 16 freezing refugees"
            ),
            registry=registry(OXYGEN),
            beneficiaries=["16 freezing refugees"],
            harmed=["4 non-consenting chronically ill public-ward patients"],
        )
        self.assertEqual(status, "COMPLETE", [str(i) for i in issues])


class RoleProvenanceTests(unittest.TestCase):
    action = RoleValidationTests.action

    def issues(self, provenance, *, require=True):
        clauses = segment_scenario_clauses(DRONE)
        return validate_role_assignment(
            self.action,
            registry=registry(DRONE),
            beneficiaries=["12-year-old child"],
            harmed=["three elderly survivors"],
            provenance=provenance,
            known_clause_ids=[clause["clause_id"] for clause in clauses],
            require_provenance=require,
        )

    def test_valid_provenance_passes(self):
        found = self.issues({
            "12-year-old child": ["C2"], "three elderly survivors": ["C1"],
        })
        self.assertEqual([str(i) for i in found], [])

    def test_unknown_clause_id_is_rejected(self):
        found = self.issues({
            "12-year-old child": ["C99"], "three elderly survivors": ["C1"],
        })
        self.assertTrue(any(i.code == "PROVENANCE_UNKNOWN" for i in found))

    def test_missing_provenance_is_rejected_when_required(self):
        found = self.issues({"three elderly survivors": ["C1"]})
        self.assertTrue(any(i.code == "PROVENANCE_MISSING" for i in found))

    def test_missing_provenance_is_tolerated_when_not_required(self):
        found = self.issues({"three elderly survivors": ["C1"]}, require=False)
        self.assertEqual([i for i in found if i.code == "PROVENANCE_MISSING"], [])


class CountProvenanceTests(unittest.TestCase):
    """A normalized count must never be mistaken for a stated one."""

    def party(self, scenario, needle):
        for mention in registry(scenario):
            if needle in mention.label.casefold():
                return mention
        self.fail(f"{needle} not in {[m.label for m in registry(scenario)]}")

    def test_stated_quantity_is_marked_explicit_with_its_source_span(self):
        elderly = self.party(DRONE, "elderly")
        self.assertEqual(elderly.count, "3")
        self.assertEqual(elderly.count_status, "EXPLICIT")
        self.assertTrue(elderly.count_is_explicit)
        self.assertIn("three", elderly.source_span.casefold())

    def test_individual_normalizes_to_one_but_is_marked_derived(self):
        child = self.party(DRONE, "12-year-old")
        self.assertEqual(child.count, "1")
        self.assertEqual(child.count_status, "DERIVED_SINGLETON")
        self.assertFalse(child.count_is_explicit)

    def test_derived_singleton_does_not_demand_a_count_in_role_labels(self):
        issues = validate_role_assignment(
            "airlift the child while the three elderly drown",
            registry=registry(DRONE),
            beneficiaries=["the child"],
            harmed=["three elderly survivors"],
        )
        self.assertEqual([i for i in issues if i.code == "CARDINALITY_LOST"], [])

    def test_explicit_count_is_still_required_in_role_labels(self):
        issues = validate_role_assignment(
            "airlift the child while the three elderly drown",
            registry=registry(DRONE),
            beneficiaries=["the child"],
            harmed=["the elderly"],
        )
        self.assertTrue(any(i.code == "CARDINALITY_LOST" for i in issues))

    def test_disagreeing_quantity_is_a_repair_not_a_normalization(self):
        issues = validate_role_assignment(
            "airlift the child while the three elderly drown",
            registry=registry(DRONE),
            beneficiaries=["the child"],
            harmed=["7 elderly survivors"],
        )
        self.assertTrue(any(i.code == "CARDINALITY_CONFLICT" for i in issues))
        self.assertEqual(completeness_from_role_issues(issues), "NEEDS_REPAIR")

    def test_stated_count_outranks_derived_when_mentions_agree(self):
        merged = merge_party_mentions([
            PartyMention(label="child", tokens=frozenset({"child"}),
                         count="1", count_status="DERIVED_SINGLETON"),
            PartyMention(label="1 child", tokens=frozenset({"child"}),
                         count="1", count_status="EXPLICIT",
                         source_span="1 child"),
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].count_status, "EXPLICIT")
        self.assertEqual(merged[0].source_span, "1 child")

    def test_a_derived_singleton_never_absorbs_a_larger_stated_group(self):
        merged = merge_party_mentions([
            PartyMention(label="child", tokens=frozenset({"child"}),
                         count="1", count_status="DERIVED_SINGLETON"),
            PartyMention(label="3 children", tokens=frozenset({"child"}),
                         count="3", count_status="EXPLICIT"),
        ])
        self.assertEqual(len(merged), 2)


class UnresolvedParticipationTests(unittest.TestCase):
    """A hedged outcome is a third epistemic status, not an absence."""

    action = (
        "fly immediately to Location A, airlift the three elderly survivors off "
        "the collapsing rooftop, guaranteeing their survival while leaving the "
        "12-year-old in the vehicle with only a 60% chance of surviving "
        "delayed human rescue"
    )

    def test_hedged_party_is_recorded_with_its_stated_probability(self):
        record = build_canonical_action_record(
            "A0", self.action, actor="drone", scenario=DRONE,
        )
        self.assertEqual(record.harmed, ())
        self.assertTrue(any(
            "12-year-old" in label.casefold() for label in record.unresolved
        ), record.unresolved)
        outcome = record.unresolved_outcomes[0]
        self.assertEqual(outcome["probability"], "60%")
        self.assertEqual(outcome["status"], "unresolved")

    def test_asserting_a_hedged_party_as_certain_is_a_repair(self):
        for role, beneficiaries, harmed in (
            ("harmed", ["three elderly survivors"], ["12-year-old child"]),
            (
                "beneficiaries",
                ["three elderly survivors", "12-year-old child"],
                [],
            ),
        ):
            with self.subTest(asserted_as=role):
                issues = validate_role_assignment(
                    self.action,
                    registry=registry(DRONE),
                    beneficiaries=beneficiaries,
                    harmed=harmed,
                )
                self.assertTrue(
                    any(i.code == "UNRESOLVED_ASSERTED_AS_CERTAIN" for i in issues),
                    [str(i) for i in issues],
                )
                self.assertEqual(
                    completeness_from_role_issues(issues), "NEEDS_REPAIR",
                )

    def test_dropping_a_hedged_party_entirely_is_a_repair(self):
        issues = validate_role_assignment(
            self.action,
            registry=registry(DRONE),
            beneficiaries=["three elderly survivors"],
            harmed=[],
            unresolved=[],
        )
        self.assertTrue(any(i.code == "UNRESOLVED_DROPPED" for i in issues))

    def test_recording_the_party_as_unresolved_satisfies_the_validator(self):
        issues = validate_role_assignment(
            self.action,
            registry=registry(DRONE),
            beneficiaries=["three elderly survivors"],
            harmed=[],
            unresolved=["12-year-old child"],
        )
        self.assertEqual([str(i) for i in issues], [])

    def test_a_definite_statement_outranks_a_hedged_mention_of_the_same_party(self):
        # The child is speculated about first and then stated to drown; the
        # settled reading must win rather than the first one encountered.
        bindings = relational_role_bindings(
            "leaving the 12-year-old with only a 60% chance of rescue, the "
            "12-year-old drowns in the vehicle",
            registry(DRONE),
        )
        child = [b for b in bindings if "12-year-old" in b.party.label.casefold()]
        self.assertEqual([b.field for b in child], ["harmed"])


class CommitmentGateTests(unittest.TestCase):
    """Validation must gate deliberation, not merely annotate it.

    The failure this guards against is the pipeline's original behaviour: a
    rejected action-source grounding skipped validation entirely and handed the
    very same records downstream, so agents deliberated over a world model the
    system had already refused to commit to.
    """

    good_action = (
        "fly immediately to Location B, airlift the 12-year-old from the "
        "flooding vehicle while the three elderly on the rooftop drown"
    )
    clauses = [
        "At Location B, a 12-year-old child is trapped in a vehicle.",
        "At Location A, three elderly survivors are trapped on a rooftop that "
        "will submerge, and they drown if the drone goes elsewhere.",
    ]

    def build(self, **kwargs):
        return build_canonical_action_record(
            "A0",
            kwargs.pop("action", self.good_action),
            actor="autonomous search-and-rescue drone",
            scenario=DRONE,
            **kwargs,
        )

    def test_fully_grounded_valid_record_commits(self):
        record = self.build(
            source_clause_texts=self.clauses, grounding_status="COMMITTED",
        )
        self.assertEqual(record.commitment_status, "COMMITTED")
        self.assertTrue(record.eligible_for_deliberation)

    def test_rejected_grounding_cannot_produce_an_eligible_record(self):
        record = self.build(
            source_clause_texts=self.clauses, grounding_status="REJECTED",
        )
        self.assertEqual(record.commitment_status, "REJECTED")
        self.assertFalse(record.eligible_for_deliberation)
        self.assertIn("grounding", " ".join(record.commitment_reasons))

    def test_rejected_grounding_blocks_even_a_well_formed_record(self):
        # The record's own roles are correct; only the grounding failed. It
        # must still be withheld, otherwise "REJECTED" means nothing.
        record = self.build(
            source_clause_texts=self.clauses, grounding_status="REJECTED",
        )
        self.assertTrue(record.beneficiaries)
        self.assertTrue(record.harmed)
        self.assertEqual(record.structure_issues, ())
        self.assertFalse(record.eligible_for_deliberation)

    def test_ungrounded_record_is_partial_rather_than_committed(self):
        record = self.build(grounding_status="")
        self.assertEqual(record.commitment_status, "PARTIAL")
        self.assertFalse(record.eligible_for_deliberation)

    def test_state_machine_transitions(self):
        for completeness, grounding, sources, expected in (
            ("COMPLETE", "COMMITTED", True, "COMMITTED"),
            ("NEEDS_REPAIR", "COMMITTED", True, "REJECTED"),
            ("MISSING_CRITICAL", "COMMITTED", True, "REJECTED"),
            ("INCOMPLETE_CLAUSE", "COMMITTED", True, "REJECTED"),
            # Grounding failure outranks a record that looks otherwise perfect.
            ("COMPLETE", "REJECTED", True, "REJECTED"),
            ("COMPLETE_WITH_NORMALIZATION", "COMMITTED", True, "PARTIAL"),
            ("COMPLETE", "COMMITTED", False, "PARTIAL"),
            ("UNCHECKED", "", False, "PARTIAL"),
        ):
            with self.subTest(completeness=completeness, grounding=grounding):
                status, reasons = _commitment_state(completeness, grounding, sources)
                self.assertEqual(status, expected)
                if status != "COMMITTED":
                    self.assertTrue(reasons)

    def test_partition_withholds_everything_that_is_not_committed(self):
        records = build_canonical_action_records(
            [self.good_action, "fly immediately to Location A"],
            actor="autonomous search-and-rescue drone",
            scenario=DRONE,
            grounding_status="REJECTED",
            grounded_clause_texts_by_id={"A0": self.clauses, "A1": self.clauses},
        )
        admitted, withheld = partition_records_for_deliberation(records)
        self.assertEqual(admitted, [])
        self.assertEqual(len(withheld), 2)
        self.assertTrue(all(
            record.commitment_status == "REJECTED" for record in withheld
        ))


class _ScriptedLLM:
    """Returns a queued mapping per call so repair behaviour can be observed."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        payload = self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]
        return {"choices": [{"text": json.dumps(payload)}]}


def _mapping(**rows):
    return {"actions": {
        action_id: {"clause_ids": list(ids), "reason": "mapped from scenario"}
        for action_id, ids in rows.items()
    }}


class DistinguishingGroundingTests(unittest.TestCase):
    """Shared background facts are fine; indistinguishable grounding is not."""

    def ground(self, *responses, max_attempts=2):
        llm = _ScriptedLLM(*responses)
        result = ground_actions_in_scenario(
            llm, DRONE,
            ["rescue the child at Location B", "rescue the elderly at Location A"],
            max_attempts=max_attempts,
        )
        return result, llm

    def test_shared_background_clauses_are_allowed(self):
        # Both actions rest on the one-trip budget (C0) and the fused comparison
        # clause, yet each also cites a clause the other does not.
        result, _ = self.ground(_mapping(A0=["C0", "C2"], A1=["C0", "C1"]))
        self.assertEqual(result["status"], "COMMITTED", result["errors"])

    def test_identical_grounding_is_rejected(self):
        result, _ = self.ground(_mapping(A0=["C0", "C1"], A1=["C0", "C1"]))
        self.assertEqual(result["status"], "REJECTED")
        self.assertTrue(any(
            "distinguishing source clause" in error for error in result["errors"]
        ))

    def test_an_action_whose_grounding_is_a_subset_is_rejected(self):
        # A1 adds nothing of its own, so nothing in the grounding tells the two
        # alternatives apart even though the clause sets are not identical.
        result, _ = self.ground(_mapping(A0=["C0", "C1", "C2"], A1=["C0", "C1"]))
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("A1", " ".join(result["errors"]))

    def test_a_rejected_mapping_is_repaired_within_the_attempt_budget(self):
        result, llm = self.ground(
            _mapping(A0=["C0", "C1"], A1=["C0", "C1"]),
            _mapping(A0=["C0", "C2"], A1=["C0", "C1"]),
        )
        self.assertEqual(result["status"], "COMMITTED", result["errors"])
        self.assertEqual(result["repair_attempts"], 1)
        self.assertEqual(len(llm.prompts), 2)
        # The retry must state what was wrong, not just ask again.
        self.assertIn("distinguishing source clause", llm.prompts[1])

    def test_repair_is_bounded_and_ends_in_rejection(self):
        result, llm = self.ground(
            _mapping(A0=["C0", "C1"], A1=["C0", "C1"]),
            max_attempts=3,
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(len(llm.prompts), 3)
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(result["actions"], {})

    def test_rejected_grounding_yields_no_deliberation_eligible_record(self):
        result, _ = self.ground(_mapping(A0=["C0", "C1"], A1=["C0", "C1"]))
        records = build_canonical_action_records(
            ["rescue the child at Location B", "rescue the elderly at Location A"],
            scenario=DRONE,
            grounding_status=result["status"],
        )
        admitted, withheld = partition_records_for_deliberation(records)
        self.assertEqual(admitted, [])
        self.assertEqual(len(withheld), 2)


class AntiviralDoseAdmissionTests(unittest.TestCase):
    """The dose-allocation dilemma must not be rejected for oxygen-era reasons.

    Two distinct named patients are not one 'patient' group, an age is not a
    headcount, and 'fatal organ failure without the drug' is the counterfactual
    of giving B the dose, not a mechanism A1 is required to restate.
    """

    clauses = {
        "A0": [
            "An automated medical allocation algorithm during a global contagion "
            "must route a single available dose of an experimental antiviral "
            "treatment.",
            "Patient A is an 80-year-old retired virologist who, if saved, "
            "possesses the specific expertise required to rapidly synthesize a "
            "synthetic antibody, direct lab teams, and accelerate the development "
            "of a universal vaccine that would prevent tens of thousands of "
            "future infections and subsequent deaths down the line.",
            "The system must choose between Option 1, which administers the dose "
            "to Patient A to trigger a massive downstream chain reaction of "
            "life-saving medical outcomes for the broader population, or Option 2, "
            "which administers the dose to Patient B to secure an immediate, "
            "direct individual life while foregoing the cascading future benefits "
            "to society.",
        ],
        "A1": [
            "An automated medical allocation algorithm during a global contagion "
            "must route a single available dose of an experimental antiviral "
            "treatment.",
            "Patient B is a healthy 20-year-old individual with no specialized "
            "skills but a full, healthy natural life ahead, who will suffer "
            "immediate, fatal organ failure without the drug.",
            "The system must choose between Option 1, which administers the dose "
            "to Patient A to trigger a massive downstream chain reaction of "
            "life-saving medical outcomes for the broader population, or Option 2, "
            "which administers the dose to Patient B to secure an immediate, "
            "direct individual life while foregoing the cascading future benefits "
            "to society.",
        ],
    }

    def test_organ_failure_is_not_an_infrastructure_claim_a1_must_repeat(self):
        missing = missing_decision_critical_claims(DOSE_A1, self.clauses["A1"])
        self.assertEqual(missing, ())
        self.assertFalse(any("fail" in claim.casefold() for claim in missing))

    def test_named_patients_are_not_collapsed_into_one_group(self):
        validate_action_set_completeness(
            [DOSE_A0, DOSE_A1],
            scenario=DOSE,
            grounded_clause_texts_by_id=self.clauses,
        )

    def test_records_keep_opposite_roles_for_the_two_patients(self):
        records = build_canonical_action_records(
            [DOSE_A0, DOSE_A1],
            scenario=DOSE,
            grounding_status="COMMITTED",
            grounded_clause_texts_by_id=self.clauses,
        )
        a0, a1 = records
        self.assertTrue(any("virologist" in b.casefold() or "patient a" in b.casefold()
                            for b in a0.beneficiaries), a0.beneficiaries)
        self.assertTrue(any("20-year-old" in h.casefold() or "patient b" in h.casefold()
                            for h in a0.harmed), a0.harmed)
        self.assertTrue(any("20-year-old" in b.casefold() or "patient b" in b.casefold()
                            for b in a1.beneficiaries), a1.beneficiaries)
        self.assertTrue(any("virologist" in h.casefold() or "patient a" in h.casefold()
                            for h in a1.harmed), a1.harmed)
        self.assertNotEqual(a0.commitment_status, "REJECTED")
        self.assertNotEqual(a1.commitment_status, "REJECTED")

    def test_future_population_is_a_grounded_effect_not_a_direct_role(self):
        records = build_canonical_action_records(
            [DOSE_A0, DOSE_A1],
            scenario=DOSE,
            grounding_status="COMMITTED",
            grounded_clause_texts_by_id=self.clauses,
        )
        a0, a1 = records
        for rec in records:
            joined = " ".join((*rec.beneficiaries, *rec.harmed)).casefold()
            self.assertNotIn("future population", joined)
            self.assertNotIn("thousand", joined)
            self.assertNotIn("society", joined)

        self.assertEqual(len(a0.grounded_effects), 1, a0.grounded_effects)
        downstream = a0.grounded_effects[0]
        self.assertEqual(downstream.party, "future population")
        self.assertEqual(downstream.party_kind, PARTY_KIND_FUTURE_POPULATION)
        self.assertEqual(downstream.relation, RELATION_DOWNSTREAM_BENEFIT)
        self.assertEqual(
            downstream.modality,
            "CONDITIONAL_ON_DIRECT_BENEFICIARY_SURVIVAL",
        )
        self.assertTrue(downstream.condition)
        self.assertTrue(downstream.clause_ids, downstream.provenance)

        self.assertEqual(len(a1.grounded_effects), 1, a1.grounded_effects)
        foregone = a1.grounded_effects[0]
        self.assertEqual(foregone.party, "future population")
        self.assertEqual(foregone.relation, RELATION_FOREGONE_BENEFIT)
        self.assertEqual(foregone.modality, "CERTAIN")


class GroundedWorldEffectsTests(unittest.TestCase):
    """Cascade parties exist in world-state, not as extra direct patients."""

    OXYGEN_A0 = (
        "execute covert patch that forcibly reallocates oxygen from the 4 "
        "non-consenting chronically ill patients, killing them instantly, to "
        "stabilize the shelter grid and save all 16 freezing refugees"
    )
    OXYGEN_A1 = (
        "maintain standard oxygen allocation, preserve the 4 patients' lives "
        "and rights, allow shelter grid to fail, resulting in death of 16 "
        "refugees within hours"
    )
    DRONE_A0 = (
        "fly immediately to Location A, airlift the three elderly survivors "
        "off the collapsing rooftop before it submerges, guaranteeing their "
        "survival while leaving the 12-year-old in the vehicle with only a "
        "60% chance of surviving delayed human rescue"
    )
    DRONE_A1 = (
        "fly immediately to Location B, airlift the 12-year-old from the "
        "flooding vehicle, ensuring decades of future life while the three "
        "elderly on the rooftop drown within 10 minutes"
    )

    def test_oxygen_refugees_remain_direct_roles(self):
        records = build_canonical_action_records(
            [self.OXYGEN_A0, self.OXYGEN_A1],
            scenario=OXYGEN,
            grounding_status="COMMITTED",
        )
        for rec in records:
            self.assertEqual(rec.grounded_effects, ())
        self.assertTrue(any("refugee" in label.casefold() for label in records[0].beneficiaries))
        self.assertTrue(any("refugee" in label.casefold() for label in records[1].harmed))

    def test_drone_future_life_years_are_not_a_future_population(self):
        self.assertEqual(extract_grounded_effects(self.DRONE_A1), ())
        records = build_canonical_action_records(
            [self.DRONE_A0, self.DRONE_A1],
            scenario=DRONE,
            grounding_status="COMMITTED",
        )
        for rec in records:
            self.assertEqual(rec.grounded_effects, ())

    def test_graph_surrounds_the_action_with_future_population_effects(self):
        graph = compile_scenario_graph(DOSE, [DOSE_A0, DOSE_A1])
        effects = project_grounded_action_effects(graph)
        a0 = [
            item for item in effects
            if item.action_id == "A0"
            and "future population" in item.affected_subject.casefold()
        ]
        a1 = [
            item for item in effects
            if item.action_id == "A1"
            and "future population" in item.affected_subject.casefold()
        ]
        self.assertTrue(a0, effects)
        self.assertEqual(a0[0].direction, "IMPROVES")
        self.assertTrue(a1, effects)
        self.assertEqual(a1[0].direction, "FOREGOES")
        a0_targets = [
            graph.nodes[edge.target].label
            for edge in graph.outgoing("A0", "TARGETS")
            if edge.target in graph.nodes
        ]
        self.assertFalse(any(
            "future population" in label.casefold() for label in a0_targets
        ), a0_targets)


if __name__ == "__main__":
    unittest.main()
