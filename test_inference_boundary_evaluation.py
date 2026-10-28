from copy import deepcopy
import json
from pathlib import Path
import unittest

from audit_inference_boundaries import audit
from run_inference_boundary_pilot import prepare,assess
import test_inference_review as protocol_tests

ROOT=Path(__file__).resolve().parent


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((ROOT/'fixtures/inference_boundary_contrasts.json').read_text())
        cls.manifest=json.loads((ROOT/'fixtures/inference_boundary_pilot_manifest.json').read_text())

    def test_rubric_excluded_and_prompt_deterministic(self):
        for case,spec in zip(self.fixture['cases'],self.manifest['cases']):
            _,prompt=prepare(case,spec)
            self.assertNotIn(case['reviewer_expected_distinction'],prompt)
            self.assertNotIn('held_out_content_criteria',prompt)
            self.assertEqual(prompt,prepare(case,spec)[1])
        self.assertEqual(len(self.fixture['cases'])*len(self.manifest['arms']),16)

    def test_adversarial_audit_exposes_known_gaps_instead_of_counting_contract_passes(self):
        result=audit(self.fixture)
        self.assertEqual(len(result['cases']),10)
        self.assertIn('kind_laundering',result['gaps'])
        self.assertNotIn('indirect_conditional_promotion',result['gaps'])
        self.assertNotIn('unknown_is_not_asserted_promotion',result['gaps'])
        self.assertNotIn('malformed_null_text',result['gaps'])
        self.assertIn('matching_quote_overclaim',result['gaps'])
        self.assertNotIn('valid_mistaken_belief_control',result['gaps'])

    def test_assessment_separates_usable_prose_and_invalid_typed_map(self):
        case=self.fixture['cases'][0];challenge,_=prepare(case,self.manifest['cases'][0])
        reply=protocol_tests.InferenceReviewTests().response(challenge)
        reply['reasoning_revision']={'claims':[],'bridges':[]}
        result=assess(case,challenge,reply,'typed')
        self.assertTrue(result['ordinary_contract']['contract_valid'])
        self.assertIn('typed_error',result)
        self.assertEqual(result['native_admission'],'NOT_ASSESSED_ARGUMENT_ONLY')


if __name__=='__main__':unittest.main()
