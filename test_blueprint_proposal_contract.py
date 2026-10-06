import unittest

import parsing_game_Z10 as z10
from blueprint_ensemble import choose_blueprint
from blueprint_proposal_contract import (
    candidate,
    proposal,
    validate_proposal,
    withheld_proposal,
    world_model,
)
from candidate_graph_blueprints import instantiate_exclusive_allocation
from run_blueprint_parliament import _authorized_proposal


MEDICINE = (
    "A clinic has one dose of medicine. "
    "Ada can give the medicine to Ben or Cara, but not both. "
    "If Ada gives the medicine to Ben, Ben has a 95% chance of survival. "
    "If Ada gives the medicine to Cara, Cara has a 5% chance of survival. "
    "The patient who does not get the medicine will die."
)
RESCUE = (
    "Maria can save either the child or the dog, but not both. "
    "If Maria saves the child, the child will live. "
    "If Maria saves the dog, the dog will live."
)


class ProposalContractTests(unittest.TestCase):
    def test_medical_allocation_uses_the_normalized_water_first_envelope(self):
        package = z10.export_candidate_graph(MEDICINE, package_id="contract_medicine")
        result = instantiate_exclusive_allocation(
            package, ["give the medicine to Ben", "give the medicine to Cara"])
        row = result["proposals"][0]
        self.assertEqual(validate_proposal(row), [])
        self.assertTrue(row["pre_world_assessment"]["eligible_for_world_state"])
        self.assertEqual(row["pre_world_assessment"]["exclusivity"], "evidenced")
        self.assertTrue(row["accepted_evidence"])
        self.assertTrue(row["admission_authorized"])
        without_exclusivity = {
            **row,
            "pre_world_assessment": {
                **row["pre_world_assessment"],
                "exclusivity": "unspecified",
            },
        }
        self.assertIn(
            "an exclusive-allocation candidate requires evidenced exclusivity",
            validate_proposal(without_exclusivity),
        )

    def test_every_ensemble_family_emits_a_contract_valid_envelope(self):
        package = z10.export_candidate_graph(RESCUE, package_id="contract_rescue")
        result = choose_blueprint(package)
        proposals = [
            proposal_row
            for considered in result["considered"]
            for proposal_row in considered["proposals"]
        ]
        self.assertTrue(proposals)
        self.assertTrue(all(validate_proposal(row) == [] for row in proposals))
        withheld = [row for row in proposals if row["candidate"] is None]
        self.assertTrue(withheld)
        self.assertTrue(all(row["world_withheld"] for row in withheld))

    def test_withheld_candidate_requires_a_named_problem(self):
        row = {
            "assignment_kind": "none",
            "candidate": None,
            "admission_authorized": False,
            "construction_problems": [],
            "world_withheld": [],
        }
        errors = validate_proposal(row)
        self.assertTrue(any("construction problem" in error for error in errors))

    def test_equal_surface_mentions_may_remain_distinct_parties(self):
        world = world_model(
            parties=[
                {"party_id": "P1", "label": "the patient", "kind": "PERSON",
                 "quantities": [], "clause_ids": ["C0"]},
                {"party_id": "P2", "label": "the patient", "kind": "PERSON",
                 "quantities": [], "clause_ids": ["C1"]},
            ],
            actions=[],
            effects=[],
            causal_links=[],
        )
        row = proposal(
            proposal_id="mentions_0",
            blueprint_id="conditional_outcome",
            status="FILLED",
            assignment_kind="intervention_text",
            assignment=[],
            slot_bindings={},
            selection=None,
            selection_validation={"contract_valid": None, "status": "not_assessed"},
            candidate_value=candidate({}, world),
            clauses=[{"clause_id": "C0", "text": "A patient waits."},
                     {"clause_id": "C1", "text": "A patient leaves."}],
            admission_authorized=True,
        )
        self.assertEqual(validate_proposal(row), [])

    def test_withheld_helper_keeps_evidence_without_a_world(self):
        row = withheld_proposal(
            proposal_id="report_0",
            blueprint_id="disputed_report",
            assignment=[],
            slot_bindings={"source": "Ada", "reported_content": "the medicine is safe"},
            clauses=[{"clause_id": "C0", "text": "Ada claims the medicine is safe."}],
            unfilled_required_slots=[],
            unresolved_readings=[{"kind": "reported_truth", "status": "unresolved"}],
            construction_problems=[{
                "code": "attribution_not_world_fact",
                "message": "Reported content is not admitted as fact.",
            }],
        )
        self.assertEqual(validate_proposal(row), [])
        self.assertIsNone(row["candidate"])
        self.assertFalse(row["admission_authorized"])
        with self.assertRaisesRegex(ValueError, "withheld"):
            _authorized_proposal({"proposals": [row]})


if __name__ == "__main__":
    unittest.main()
