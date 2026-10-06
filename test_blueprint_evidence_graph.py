import unittest
from pathlib import Path
from unittest import mock

import parsing_game_Z10 as z10
from blueprint_admission_core import supported_core
from blueprint_cloze_chooser import choose_by_cloze
from blueprint_evidence_graph import (
    ATTACH_TYPES,
    HYPOTHESIS_KEYS,
    IR_HOPS,
    assemble_evidence_graph,
    citation_errors,
    exclusive_allocation_signals,
    hypotheses_from_implied_notes,
    hypotheses_from_z10,
    is_licensed,
)
from blueprint_proposal_contract import (
    candidate,
    proposal,
    validate_proposal,
    world_model,
)
from candidate_graph_blueprints import (
    instantiate_exclusive_allocation,
    match_exclusive_allocation,
)
from run_blueprint_parliament import _parliament_deliberation_command
from test_blueprint_cloze_chooser import (
    MEDICINE,
    RESCUE,
    TROLLEY,
    Script,
    _none_sheet,
)
from test_candidate_graph_blueprints import GROUPED_ACTIONS, GROUPED_MEDICINE


ADA_ACTIONS = ["give the medicine to Ben", "give the medicine to Cara"]


def _allocation_cloze(text, answers):
    sheet = _none_sheet("exclusive_allocation")
    sheet.update(answers)
    script = Script(
        ["exclusive_allocation", "conditional_outcome", "omission_harm"],
        {
            "exclusive_allocation": sheet,
            "conditional_outcome": _none_sheet("conditional_outcome"),
            "omission_harm": _none_sheet("omission_harm"),
        },
    )
    return choose_by_cloze(text, script)


def _medicine_answers():
    return {
        "decider": "Ada",
        "resource": "the medicine",
        "assignment": "give the medicine",
        "quantity": "one dose",
        "first_recipient": "Ben",
        "second_recipient": "Cara",
        "exclusivity": "but not both",
        "first_outcome": "Ben has a 95% chance of survival",
        "second_outcome": "Cara has a 5% chance of survival",
        "survival_chance": "95% chance",
        "nonreceipt": "The patient who does not get the medicine will die",
    }


class EvidenceGraphHopTests(unittest.TestCase):
    def test_hops_are_named_without_a_sixth_language(self):
        names = [row["name"] for row in IR_HOPS]
        self.assertEqual(names, [
            "z10_package", "cloze_slots", "blueprint_slot_bindings",
            "proposal_construction_provenance", "schema_1_3_world",
            "relent_workspace",
        ])
        self.assertNotIn("semantic_evidence_graph", names)

    def test_authorized_cloze_must_cite_source_copy_or_z10(self):
        result = _allocation_cloze(MEDICINE, _medicine_answers())
        row = result["graph"]
        self.assertTrue(row["admission_authorized"])
        self.assertEqual(row["selection_validation"]["status"], "source_copy")
        self.assertNotEqual(row["selection_validation"]["status"], "not_assessed")
        self.assertTrue(row["evidence_graph"])
        self.assertEqual(validate_proposal(row), [])
        world = row["candidate"]["world_model"]
        self.assertEqual(citation_errors(row["evidence_graph"], world), [])
        recoveries = result["considered"][0].get("semantic_recoveries") or {}
        self.assertTrue(any(
            item.get("producer") == "parsing_game_Z10" and item.get("z10_candidate_ids")
            for item in recoveries.values()
        ))

    def test_cloze_and_ensemble_share_hypothesis_record_shape(self):
        cloze = _allocation_cloze(MEDICINE, _medicine_answers())["graph"]
        package = z10.export_candidate_graph(MEDICINE, package_id="evidence_ensemble")
        ensemble = instantiate_exclusive_allocation(package, ADA_ACTIONS)["proposals"][0]
        cloze_keys = {key for row in cloze["evidence_graph"] for key in row}
        ensemble_keys = {key for row in ensemble["evidence_graph"] for key in row}
        self.assertTrue(HYPOTHESIS_KEYS <= cloze_keys)
        self.assertTrue(HYPOTHESIS_KEYS <= ensemble_keys)
        self.assertNotEqual(
            type(cloze["accepted_evidence"]).__name__,
            "NoneType",
        )
        # Slot evidence may still differ; the envelope that licenses 1.3 does not.
        self.assertEqual(
            {row["producer"] for row in cloze["evidence_graph"]} & {"z10", "cloze_copy"},
            {"z10", "cloze_copy"},
        )
        self.assertIn("z10", {row["producer"] for row in ensemble["evidence_graph"]})

    def test_cloze_worlds_account_for_attachable_z10_candidates(self):
        package = z10.export_candidate_graph(MEDICINE, package_id="evidence_unconsumed")
        cloze = _allocation_cloze(MEDICINE, _medicine_answers())["graph"]
        attached = {
            candidate_id
            for row in cloze["evidence_graph"]
            for candidate_id in row.get("z10_candidate_ids") or []
        }
        attachable = {
            row["id"] for row in package["candidates"] if row["type"] in ATTACH_TYPES
        }
        self.assertTrue(attachable)
        self.assertEqual(attachable, attached & attachable)

    def test_frozen_scripts_still_cite_z10_or_copied_spans(self):
        rivera = _allocation_cloze(GROUPED_MEDICINE, {
            "decider": "Dr. Rivera", "resource": "antiviral",
            "assignment": "give the antiviral", "quantity": "one dose",
            "first_recipient": "one child",
            "second_recipient": "three adult patients",
            "exclusivity": "but not both",
            "first_outcome": "the child has an 80% chance of survival",
            "second_outcome": (
                "each of the three adult patients has a 60% chance of survival"
            ),
            "first_hedge": "80% chance",
            "second_hedge": "60% chance",
            "nonreceipt": "The patients who do not receive the antiviral will die",
        })["graph"]
        child_dog_script = Script(
            ["rescue_contrast", "exclusive_allocation", "conditional_outcome"],
            {
                "rescue_contrast": {
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
                    "scene": "NONE",
                },
                "exclusive_allocation": _none_sheet("exclusive_allocation"),
                "conditional_outcome": _none_sheet("conditional_outcome"),
            },
        )
        child_dog = choose_by_cloze(RESCUE, child_dog_script)["graph"]
        lever_script = Script(
            ["omission_harm", "rescue_contrast", "exclusive_allocation"],
            {
                "omission_harm": {
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
        )
        question = {
            "ethical_question": "pulls / not pull",
            "scenario_options": [],
            "participants": [],
            "exclusivity": "unspecified",
        }
        with mock.patch("blueprint_cloze_chooser.assess_question",
                        return_value=question):
            lever = choose_by_cloze(TROLLEY, lever_script)["graph"]
        package = z10.export_candidate_graph(GROUPED_MEDICINE, package_id="evidence_rivera")
        ensemble_rivera = instantiate_exclusive_allocation(
            package, GROUPED_ACTIONS)["proposals"][0]
        for row in (rivera, child_dog, lever, ensemble_rivera):
            self.assertTrue(row["admission_authorized"])
            self.assertNotEqual(
                (row.get("selection_validation") or {}).get("status"),
                "not_assessed",
            )
            self.assertEqual(validate_proposal(row), [])
            for record in row["evidence_graph"]:
                if not is_licensed(record):
                    continue
                self.assertTrue(
                    record.get("z10_candidate_ids")
                    or (record.get("span") and record.get("clause_ids"))
                )


class EvidenceGraphRecordTests(unittest.TestCase):
    def test_z10_attach_does_not_mutate_the_package(self):
        package = z10.export_candidate_graph(MEDICINE, package_id="evidence_wrap")
        before = [dict(row) for row in package["candidates"]]
        rows = hypotheses_from_z10(package)
        self.assertEqual(package["candidates"], before)
        kinds = {row["hypothesis"]["kind"] for row in rows}
        self.assertTrue({"PREDICATION", "QUANTITY", "OPTION_OF", "CONDITIONAL_ON"} <= kinds)
        self.assertTrue(all(row["producer"] == "z10" for row in rows
                            if row["hypothesis"]["kind"] in ATTACH_TYPES))

    def test_llm_notes_stay_off_the_1_3_candidate(self):
        notes = hypotheses_from_implied_notes(
            {"first_implied_process": "drowning"},
            [{"clause_id": "C0", "text": "The town will lose 100 people to drowning."}],
        )
        self.assertEqual(notes[0]["producer"], "llm_note")
        self.assertEqual(notes[0]["provenance"], "WORLD_KNOWLEDGE_HYPOTHESIS")
        self.assertFalse(is_licensed(notes[0]))
        self.assertEqual(notes[0]["atom_ids"], [])

    def test_implied_process_cloze_slot_is_not_source_asserted(self):
        graph = assemble_evidence_graph(
            accepted={"first_implied_process": "distribution", "resource": "water"},
            slots={"first_implied_process": "distribution"},
            clauses=[{"clause_id": "C0", "text": "devote the water"}],
            text="devote the water",
        )
        self.assertFalse(any(
            row.get("hypothesis_id") == "cloze:first_implied_process" for row in graph
        ))
        notes = [row for row in graph if row.get("hypothesis_id") == "llm:first_implied_process"]
        self.assertEqual(notes[0]["provenance"], "WORLD_KNOWLEDGE_HYPOTHESIS")
        self.assertFalse(is_licensed(notes[0]))

    def test_validate_rejects_authorized_not_assessed(self):
        world = world_model(
            parties=[{"party_id": "P1", "label": "Ada", "kind": "PERSON",
                      "quantities": [], "clause_ids": ["C0"]}],
            actions=[{"action_id": "A0", "intervention": "gives medicine",
                      "actor_party_id": "P1", "recipient_party_ids": [],
                      "effect_ids": ["E1"], "clause_ids": ["C0"]}],
            effects=[{
                "effect_id": "E1", "action_id": "A0", "party_id": "P1",
                "outcome": "Ben recovers", "predicate": "recover",
                "polarity": "BENEFICIAL", "directness": "DIRECT",
                "modality": "CERTAIN", "effect_kind": "HEALTH_OUTCOME",
                "condition_ids": [], "quantities": [],
                "likelihood_qualifiers": [], "overall_likelihood_qualifiers": [],
                "scope_qualifiers": [], "temporal_qualifiers": [],
                "condition_join": "AND", "source_proposition": "Ben recovers",
                "source_effect_ids": [], "derivation_operation": "DIRECT_COPY",
                "derivation_explanation": "Copied.", "derivation_assumptions": [],
                "outcome_type_transformation": "PRESERVED", "clause_ids": ["C0"],
            }],
            causal_links=[],
        )
        row = {
            "proposal_id": "bad_0", "blueprint_id": "conditional_outcome",
            "status": "FILLED", "assignment_kind": "intervention_text",
            "assignment": ["gives medicine"], "slot_bindings": {},
            "selection": None,
            "selection_validation": {"status": "not_assessed", "contract_valid": None},
            "candidate": candidate({"A0": {"clause_ids": ["C0"], "reason": "test"}}, world),
            "clauses": [{"clause_id": "C0", "text": "Ada gives medicine. Ben recovers."}],
            "unfilled_required_slots": [], "unresolved_readings": [],
            "construction_problems": [], "admission_authorized": True,
            "notes": [], "pre_world_assessment": {
                "status": "ASSESSED", "eligible_for_world_state": True,
            },
            "accepted_evidence": {}, "world_withheld": [],
            "construction_provenance": [], "relation_alternatives": [],
            "exclusivity_proof": {
                "status": "UNKNOWN", "evidence": [], "assumptions": [],
                "explanation": "none",
            },
            "evidence_graph": [],
        }
        errors = validate_proposal(row)
        self.assertTrue(any("not_assessed" in error for error in errors))
        self.assertTrue(any("licensed hypothesis" in error for error in errors))

    def test_proposal_helper_stamps_source_copy_and_citations(self):
        world = world_model(
            parties=[{"party_id": "P1", "label": "Ada", "kind": "PERSON",
                      "quantities": [], "clause_ids": ["C0"]}],
            actions=[{"action_id": "A0", "intervention": "gives medicine",
                      "actor_party_id": "P1", "recipient_party_ids": [],
                      "effect_ids": [], "clause_ids": ["C0"]}],
            effects=[],
            causal_links=[],
        )
        row = proposal(
            proposal_id="mentions_0",
            blueprint_id="conditional_outcome",
            status="FILLED",
            assignment_kind="intervention_text",
            assignment=["gives medicine"],
            slot_bindings={},
            selection=None,
            selection_validation={"contract_valid": None, "status": "not_assessed"},
            candidate_value=candidate(
                {"A0": {"clause_ids": ["C0"], "reason": "test"}}, world),
            clauses=[{"clause_id": "C0", "text": "Ada gives medicine."}],
            admission_authorized=True,
        )
        self.assertEqual(row["selection_validation"]["status"], "source_copy")
        self.assertEqual(validate_proposal(row), [])
        self.assertTrue(row["evidence_graph"])

    def test_unsupported_notes_stay_in_the_overlay(self):
        world = world_model(
            parties=[{"party_id": "P1", "label": "Ada", "kind": "PERSON",
                      "quantities": [], "clause_ids": ["C0"]}],
            actions=[{"action_id": "A0", "intervention": "gives medicine",
                      "actor_party_id": "P1", "recipient_party_ids": [],
                      "effect_ids": [], "clause_ids": ["C0"]}],
            effects=[],
            causal_links=[],
        )
        graph = assemble_evidence_graph(
            clauses=[{"clause_id": "C0", "text": "Ada gives medicine."}],
            slots={"first_implied_process": "an unstated drowning"},
            world=world,
        )
        row = {
            "candidate": {"world_model": world},
            "assignment": ["gives medicine"],
            "admission_authorized": True,
            "evidence_graph": graph,
        }
        projected, overlay = supported_core(row)
        self.assertEqual(projected["candidate"]["world_model"]["effects"], [])
        self.assertTrue(any(
            item["producer"] == "llm_note" for item in overlay["hypotheses"]))
        self.assertFalse(any(
            item["effect_kind"] == "PROCESS"
            for item in projected["candidate"]["world_model"]["effects"]
        ))


class SharedMatcherTests(unittest.TestCase):
    def test_cloze_and_ensemble_share_exclusive_allocation_matcher(self):
        package = z10.export_candidate_graph(MEDICINE, package_id="evidence_match")
        ensemble_match = match_exclusive_allocation(package, ADA_ACTIONS)
        cloze = _allocation_cloze(MEDICINE, _medicine_answers())["graph"]
        cloze_match = match_exclusive_allocation(
            package, ADA_ACTIONS, evidence_graph=cloze["evidence_graph"])
        self.assertTrue(ensemble_match["matched"])
        self.assertTrue(cloze_match["matched"])
        self.assertEqual(
            set(ensemble_match["required_slots"]),
            set(cloze_match["required_slots"]),
        )
        signals = exclusive_allocation_signals(cloze["evidence_graph"], package)
        self.assertTrue(signals["quantity_ids"] or signals["quantity_spans"])
        self.assertTrue(signals["exclusivity_clause_ids"])
        self.assertEqual(
            cloze["slot_bindings"]["exclusive_allocation_match"]["matched"],
            True,
        )


class RelEntHandoffEnvelopeTests(unittest.TestCase):
    def test_relent_command_uses_frozen_world_only(self):
        command = _parliament_deliberation_command(
            parliament_python=Path("/tmp/parliament-smoke-env/bin/python"),
            scenario_path=Path("/tmp/scenario.json"),
            trace_path=Path("/tmp/frozen_world_trace.json"),
            output_dir=Path("/tmp/out"),
            openai_model="o3",
            agents=["utilitarian", "deontological"],
            max_cycles=1,
            time_budget=60.0,
            agent_timeout=300.0,
            use_rag=False,
            prototype=True,
        )
        joined = " ".join(command)
        self.assertIn("--frozen-world-trace", command)
        self.assertNotIn("evidence_graph", joined)
        self.assertNotIn("--evidence-graph", command)


if __name__ == "__main__":
    unittest.main()
