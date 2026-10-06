import json
import unittest

from blueprint_cloze_chooser import choose_by_cloze
from blueprint_discourse import (
    attach_discourse_collections,
    attach_source_discourse,
    build_ability_permission,
    build_deontic_rule,
    build_disputed_report,
    build_promise_reliance,
    content_as_effect_errors,
    ensure_schema,
    occurrence_subgraph,
    promise_slots_from_source,
    relent_workspace_split,
    world_for_relent_compile,
)
from blueprint_ensemble import choose_blueprint
from blueprint_proposal_contract import validate_proposal
from test_blueprint_cloze_chooser import (
    MEDICINE,
    RESCUE,
    TROLLEY,
    Script,
    _none_sheet,
)
import parsing_game_Z10 as z10


RIVERA = (
    "A clinic has one dose of antiviral. Dr. Rivera must give the antiviral "
    "to either one child or three adult patients, but not both. If Dr. Rivera "
    "gives the antiviral to the child, the child has an 80% chance of survival. "
    "If Dr. Rivera gives the antiviral to the three adult patients, each of "
    "the three adult patients has a 60% chance of survival. The patients who "
    "do not receive the antiviral will die."
)


def _empty_discourse(world):
    for key in ("propositions", "reports", "commitments",
                "modal_operators", "normative_propositions"):
        if world.get(key):
            return False
    return True


class DiscourseAdmissionTests(unittest.TestCase):
    def test_disputed_report_attributes_safe_without_making_it_an_effect(self):
        world, _notes = build_disputed_report(
            "Ada claims the medicine is safe.",
            {"source": "Ada", "report_words": "claims",
             "reported_content": "the medicine is safe"},
        )
        self.assertEqual(world["schema_version"], "1.4")
        self.assertEqual(world["effects"], [])
        self.assertEqual(world["reports"][0]["speech_act"], "claims")
        self.assertEqual(world["propositions"][0]["status"], "ATTRIBUTED")
        self.assertEqual(world["propositions"][0]["predication"], "the medicine is safe")
        self.assertEqual(content_as_effect_errors(world), [])
        leak = ensure_schema({
            **world,
            "effects": [{
                "effect_id": "E1", "action_id": "A0", "party_id": "P1",
                "outcome": "the medicine is safe", "predicate": "safe",
                "source_proposition": "the medicine is safe",
            }],
        })
        self.assertTrue(content_as_effect_errors(leak))

    def test_promise_reliance_commits_without_delivered(self):
        world, _notes = build_promise_reliance(
            "Ada promised Ben she would deliver it.",
            {"promisor": "Ada", "promisee": "Ben",
             "commitment_event": "promised",
             "commitment_content": "she would deliver it"},
        )
        self.assertEqual(world["effects"], [])
        self.assertEqual(world["commitments"][0]["content_proposition_id"], "PR1")
        self.assertEqual(world["propositions"][0]["status"], "COMMITTED_CONTENT")
        self.assertFalse(any("deliver" in (row.get("predicate") or "")
                             for row in world["effects"]))

    def test_ability_and_deontic_govern_without_minting_actions(self):
        ability, _ = build_ability_permission(
            "Ada can give medicine to Ben.",
            {"actor": "Ada", "modal_action": "give medicine to Ben",
             "target": "Ben", "modal_words": "can"},
        )
        self.assertEqual(ability["actions"], [])
        self.assertEqual(ability["effects"], [])
        self.assertEqual(ability["modal_operators"][0]["governed_proposition_id"], "PR1")
        self.assertTrue(any(row["label"] == "Ben" for row in ability["parties"]))
        deontic, _ = build_deontic_rule(
            "Ada must deliver medicine to Ben.",
            {"deontic_words": "must", "governed_action": "deliver medicine to Ben",
             "bearer": "Ada"},
        )
        self.assertEqual(deontic["actions"], [])
        self.assertEqual(deontic["effects"], [])
        self.assertEqual(deontic["normative_propositions"][0]["force"], "obligation")
        self.assertTrue(any(row["label"] == "Ben" for row in deontic["parties"]))

    def test_authorized_cloze_discourse_families_are_contract_valid(self):
        cases = (
            ("disputed_report", "Ada claims the medicine is safe.", {
                "source": "Ada", "report_words": "claims",
                "reported_content": "the medicine is safe",
                "competing_report": "NONE", "reliability": "NONE",
                "confirmation": "NONE",
            }),
            ("promise_reliance", "Ada promised Ben she would deliver it.", {
                "promisor": "Ada", "commitment_event": "promised",
                "commitment_content": "she would deliver it", "promisee": "Ben",
                "reliance": "NONE", "breach": "NONE",
            }),
            ("ability_permission", "Ada can give medicine to Ben.", {
                "actor": "Ada", "modal_action": "give medicine to Ben",
                "target": "Ben", "modal_words": "can", "outcome": "NONE",
                "duty_or_prohibition": "NONE",
            }),
            ("deontic_rule", "Ada must deliver medicine to Ben.", {
                "deontic_words": "must",
                "governed_action": "deliver medicine to Ben",
                "bearer": "Ada", "authority": "NONE", "exception": "NONE",
                "sanction": "NONE",
            }),
        )
        for blueprint_id, text, sheet in cases:
            with self.subTest(blueprint_id=blueprint_id):
                script = Script(
                    [blueprint_id, "conditional_outcome", "exclusive_allocation"],
                    {
                        blueprint_id: sheet,
                        "conditional_outcome": _none_sheet("conditional_outcome"),
                        "exclusive_allocation": _none_sheet("exclusive_allocation"),
                    },
                )
                result = choose_by_cloze(text, script)
                proposal = result["graph"]
                self.assertEqual(validate_proposal(proposal), [])
                world = proposal["candidate"]["world_model"]
                self.assertEqual(world["schema_version"], "1.4")
                self.assertEqual(world["effects"], [])
                occurrence = occurrence_subgraph(world)
                self.assertEqual(occurrence["schema_version"], "1.3")
                self.assertEqual(occurrence["effects"], [])

    def test_ada_rivera_child_dog_maria_occurrence_graphs_keep_empty_discourse(self):
        medicine = choose_by_cloze(MEDICINE, Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": {
                    **_none_sheet("exclusive_allocation"),
                    "decider": "Ada", "resource": "the medicine",
                    "assignment": "give the medicine", "quantity": "one dose",
                    "first_recipient": "Ben", "second_recipient": "Cara",
                    "exclusivity": "but not both",
                    "first_outcome": "Ben has a 95% chance of survival",
                    "second_outcome": "Cara has a 5% chance of survival",
                    "survival_chance": "95% chance",
                    "nonreceipt": "The patient who does not get the medicine will die",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        ))["graph"]["candidate"]["world_model"]
        rivera = choose_by_cloze(RIVERA, Script(
            ["exclusive_allocation", "conditional_outcome", "omission_harm"],
            {
                "exclusive_allocation": {
                    **_none_sheet("exclusive_allocation"),
                    "decider": "Dr. Rivera", "resource": "antiviral",
                    "assignment": "give the antiviral", "quantity": "one dose",
                    "first_recipient": "one child",
                    "second_recipient": "three adult patients",
                    "exclusivity": "but not both",
                    "first_outcome": "the child has an 80% chance of survival",
                    "second_outcome": (
                        "each of the three adult patients has a 60% chance of survival"
                    ),
                    "nonreceipt": "The patients who do not receive the antiviral will die",
                },
                "conditional_outcome": _none_sheet("conditional_outcome"),
                "omission_harm": _none_sheet("omission_harm"),
            },
        ))["graph"]["candidate"]["world_model"]
        child_dog = choose_by_cloze(RESCUE, Script(
            ["rescue_contrast", "exclusive_allocation", "conditional_outcome"],
            {
                "rescue_contrast": {
                    **_none_sheet("rescue_contrast"),
                    "rescuer": "Maria",
                    "first_saved": "the child",
                    "second_saved": "the dog",
                    "rescue_exclusivity": "but not both",
                    "first_rescue_action": "Maria saves the child",
                    "second_rescue_action": "Maria saves the dog",
                    "first_benefit": "the child will live",
                    "first_harm": "the dog will drown",
                    "second_benefit": "the dog will live",
                    "second_harm": "the child will drown",
                },
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
                "conditional_outcome": _none_sheet("conditional_outcome"),
            },
        ))["graph"]["candidate"]["world_model"]
        maria = choose_by_cloze(TROLLEY, Script(
            ["omission_harm", "rescue_contrast", "exclusive_allocation"],
            {
                "omission_harm": {
                    **_none_sheet("omission_harm"),
                    "actor": "Maria",
                    "done": "pull the lever",
                    "omitted": "NONE",
                    "harm_done": "one worker will die",
                    "harm_omitted": "five workers will die",
                    "done_hedge": "If",
                    "omitted_hedge": "If Maria does not pull the lever",
                    "group_counts": "Five workers",
                    "instrument": "the brake",
                },
                "rescue_contrast": _none_sheet("rescue_contrast"),
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
            },
        ))["graph"]["candidate"]["world_model"]
        for name, world in (
                ("ada", medicine), ("rivera", rivera),
                ("child_dog", child_dog), ("maria", maria)):
            with self.subTest(name=name):
                self.assertEqual(world["schema_version"], "1.4")
                self.assertTrue(_empty_discourse(world))
                self.assertEqual(occurrence_subgraph(world)["schema_version"], "1.3")
                self.assertTrue(world["effects"])
                self.assertTrue(world["actions"])


class RelEntWorkspaceSplitTests(unittest.TestCase):
    def test_established_is_occurrence_effects_only(self):
        world, _ = build_disputed_report(
            "Ada claims the medicine is safe.",
            {"source": "Ada", "report_words": "claims",
             "reported_content": "the medicine is safe"},
        )
        split = relent_workspace_split(world)
        self.assertEqual(split["established"], [])
        self.assertEqual(split["factual_status_sources"], ["occurrence_effects"])
        attributed = [row for row in split["not_established"]
                      if row["kind"] == "proposition"]
        self.assertEqual(attributed[0]["status"], "ATTRIBUTED")
        self.assertIn(world["propositions"][0]["proposition_id"],
                      split["ev_forbidden_proposition_ids"])
        self.assertFalse(any("safe" in json.dumps(split["established"]).casefold()
                             for _ in [0]))

    def test_duty_ledger_may_cite_norms_and_commitments(self):
        promise, _ = build_promise_reliance(
            "Ada promised Ben she would deliver it.",
            {"promisor": "Ada", "promisee": "Ben", "commitment_event": "promised",
             "commitment_content": "she would deliver it"},
        )
        deontic, _ = build_deontic_rule(
            "Ada must deliver medicine to Ben.",
            {"deontic_words": "must", "governed_action": "deliver medicine to Ben",
             "bearer": "Ada"},
        )
        promise_split = relent_workspace_split(promise)
        deontic_split = relent_workspace_split(deontic)
        self.assertEqual(promise_split["duty_ledger_citations"][0]["kind"], "commitment")
        self.assertEqual(deontic_split["duty_ledger_citations"][0]["kind"], "norm")
        self.assertEqual(promise_split["established"], [])
        self.assertEqual(deontic_split["established"], [])

    def test_ensemble_ability_does_not_establish_give(self):
        package = z10.export_candidate_graph(
            "Ada can give medicine to Ben.", package_id="discourse_ability")
        choice = choose_blueprint(package)
        row = next(item for item in choice["considered"]
                   if item["blueprint_id"] == "ability_permission")
        if row["status"] != "FILLED":
            self.skipTest("ability slots were not filled from Z10")
        world = row["proposals"][0]["candidate"]["world_model"]
        self.assertEqual(world["actions"], [])
        self.assertEqual(world["effects"], [])
        split = relent_workspace_split(world)
        self.assertEqual(split["established"], [])


    def test_mixed_allocation_keeps_occurrence_actions_and_promise_collections(self):
        text = (
            "An AI bot promised a child the only water tanker. It must decide whether "
            "to devote the tanker to the child or five elderly patients, but not both. "
            "If the child gets the water, the child will live. If the five elderly "
            "patients get the water, the five elderly patients will live. The people "
            "who do not get the water will die."
        )
        allocation, _notes = build_promise_reliance(
            "Ada promised Ben she would deliver it.",
            {"promisor": "Ada", "promisee": "Ben", "commitment_event": "promised",
             "commitment_content": "she would deliver it"},
        )
        occurrence = ensure_schema({
            "parties": [
                {"party_id": "P1", "label": "An AI bot", "kind": "OTHER"},
                {"party_id": "P2", "label": "the only water tanker", "kind": "RESOURCE"},
                {"party_id": "P3", "label": "a child", "kind": "PERSON"},
                {"party_id": "P4", "label": "five elderly patients", "kind": "HUMAN_GROUP"},
            ],
            "actions": [
                {"action_id": "A0", "intervention": "devote the tanker to the child",
                 "actor_party_id": "P1", "recipient_party_ids": ["P3"],
                 "effect_ids": ["E1"], "clause_ids": ["C0"]},
                {"action_id": "A1", "intervention": "devote the tanker to five elderly patients",
                 "actor_party_id": "P1", "recipient_party_ids": ["P4"],
                 "effect_ids": ["E2"], "clause_ids": ["C1"]},
            ],
            "effects": [
                {"effect_id": "E1", "action_id": "A0", "party_id": "P3",
                 "outcome": "the child will live", "predicate": "live"},
                {"effect_id": "E2", "action_id": "A1", "party_id": "P4",
                 "outcome": "the five elderly patients will live", "predicate": "live"},
            ],
        })
        mixed = attach_discourse_collections(occurrence, allocation)
        self.assertEqual([row["action_id"] for row in mixed["actions"]], ["A0", "A1"])
        self.assertTrue(mixed["commitments"])
        self.assertFalse(mixed["commitments"][0].get("action_id"))
        self.assertFalse(any(
            "deliver" in (row.get("predicate") or "").casefold()
            for row in mixed["effects"]
        ))
        split = relent_workspace_split(mixed)
        self.assertEqual({row["predicate"] for row in split["established"]}, {"live"})
        self.assertEqual(split["duty_ledger_citations"][0]["kind"], "commitment")
        self.assertIn(mixed["propositions"][0]["proposition_id"],
                      split["ev_forbidden_proposition_ids"])
        package = z10.export_candidate_graph(text, package_id="mixed_promise_attach")
        attached = attach_source_discourse(text, occurrence, package)
        self.assertTrue(attached.get("commitments"), attached)
        self.assertEqual(len(attached["actions"]), 2)
        self.assertIn("tanker", attached["propositions"][0]["predication"].casefold())
        self.assertIn("child", attached["propositions"][0]["predication"].casefold())
        self.assertEqual(content_as_effect_errors(attached), [])
        compile_world = world_for_relent_compile(occurrence_subgraph(attached), attached)
        self.assertEqual(compile_world["schema_version"], "1.3")
        self.assertTrue(compile_world["commitments"])
        self.assertEqual(len(compile_world["actions"]), 2)

    def test_promise_slots_copy_the_promised_object(self):
        text = "An AI bot promised a child the only water tanker."
        package = z10.export_candidate_graph(text, package_id="promise_theme")
        slots = promise_slots_from_source(text, package)
        self.assertIsNotNone(slots)
        self.assertEqual(slots["commitment_event"], "promised")
        self.assertIn("tanker", slots["commitment_content"].casefold())
        self.assertIn("child", slots["commitment_content"].casefold())


if __name__ == "__main__":
    unittest.main()
