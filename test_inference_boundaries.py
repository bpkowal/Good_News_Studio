from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from inference_boundaries import boundary_card,check_reasoning
from inference_review import make_challenge,check_challenge,request_review,assess_response
from inference_history import start_history,record_response,followup_state
import test_inference_review as protocol_tests

ROOT=Path(__file__).resolve().parent


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((ROOT/'fixtures/inference_boundary_contrasts.json').read_text())
        cls.cases={c['id']:c for c in cls.fixture['cases']}

    def challenge(self,case):
        return make_challenge(case['prior_reply'],claim_path='explanation',quote=case['prior_reply']['explanation'],
            defect_kind='MISSING_BRIDGE',missing_premise=case['reviewer_expected_distinction'],
            evidence=case['evidence'],affected_paths=['classification'],reviewer='FIXTURE_REVIEWER',
            inference_map=case['inference_map'])

    def test_eight_contrasts_retain_review_questions_without_world_edits(self):
        self.assertEqual(len(self.cases),8)
        original=deepcopy(self.fixture)
        for case in self.cases.values():
            with self.subTest(case=case['id']):
                card=boundary_card(case['inference_map'],case['evidence'])
                self.assertEqual([f['code'] for f in card['findings']],case['expected_findings'])
                self.assertFalse(card['operative'])
                self.assertEqual(card['semantic_support'],'NOT_INDEPENDENTLY_ASSESSED')
        self.assertEqual(self.fixture,original)

    def test_mistaken_belief_and_explicit_source_are_not_forced_unknown(self):
        for name in ('mistaken_means_belief','mistaken_reviewer_positive_support'):
            case=self.cases[name]
            card=boundary_card(case['inference_map'],case['evidence'])
            self.assertEqual(card['findings'],[])
            self.assertFalse(card['physical_path_required_for_belief'])
            self.assertEqual(card['reasoning']['claims'][-1]['interpretation_mode'],'SOURCE_STATED')

    def test_conditional_hypothesis_survives_but_promotion_is_flagged(self):
        case=self.cases['conditional_rescue_maxim']; mapping=deepcopy(case['inference_map'])
        self.assertEqual(boundary_card(mapping,case['evidence'])['findings'],[])
        mapping['claims'][-1]['interpretation_mode']='ATTRIBUTED_INFERENCE'
        self.assertIn('CONDITIONAL_SCOPE_NOT_PRESERVED',
                      [f['code'] for f in boundary_card(mapping,case['evidence'])['findings']])

    def test_explicit_denial_does_not_count_as_affirmative_practical_support(self):
        case=self.cases['mistaken_means_belief'];mapping=deepcopy(case['inference_map'])
        next(c for c in mapping['claims'] if c['id']=='belief')['polarity']='DENIED'
        self.assertIn('PRACTICAL_REASONING_BRIDGE_INCOMPLETE',
            [f['code'] for f in boundary_card(mapping,case['evidence'])['findings']])

    def test_exact_provenance_quote_not_id_only_and_not_entailment_oracle(self):
        case=self.cases['provenance_purpose_not_exclusivity']
        self.assertTrue(check_reasoning(case['inference_map'],case['evidence']))
        # A matching provenance citation can support purpose but does not verify
        # 'sole'. No keyword-based semantic verdict is smuggled into this check.
        self.assertEqual(boundary_card(case['inference_map'],case['evidence'])['semantic_support'],
                         'NOT_INDEPENDENTLY_ASSESSED')
        mapping=deepcopy(case['inference_map'])
        mapping['claims'][-1]['supports'][0]['value_path']='/outcome'
        with self.assertRaises(ValueError):check_reasoning(mapping,case['evidence'])
        mapping=deepcopy(case['inference_map']);mapping['claims'][-1]['supports'][0]['quote']='fabricated text'
        with self.assertRaises(ValueError):check_reasoning(mapping,case['evidence'])

    def test_dangling_dependency_cycles_and_unknown_modes_are_rejected(self):
        case=self.cases['mistaken_means_belief']
        mapping=deepcopy(case['inference_map']);mapping['bridges'][0]['premise_ids']=['missing']
        with self.assertRaises(ValueError):check_reasoning(mapping,case['evidence'])
        mapping=deepcopy(case['inference_map']);mapping['bridges'][0]['premise_ids']=['intended']
        with self.assertRaises(ValueError):check_reasoning(mapping,case['evidence'])
        mapping=deepcopy(case['inference_map']);mapping['claims'][-1]['interpretation_mode']='ACTUAL_CERTIFIED'
        with self.assertRaises(ValueError):check_reasoning(mapping,case['evidence'])

    def test_typed_protocol_requests_revision_and_preserves_history_budget(self):
        case=self.cases['contribution_without_awareness']; challenge=self.challenge(case)
        expected=protocol_tests.InferenceReviewTests().response(challenge)
        expected['reasoning_revision']=deepcopy(case['inference_map'])
        class Backend:
            def complete_json(inner,prompt,**kwargs):
                self.assertIn('reasoning_revision',kwargs['schema']['required'])
                self.assertIn('Actual efficacy',prompt)
                self.assertIn('value_path',json.dumps(kwargs['schema']))
                return {'choices':[{'text':json.dumps(expected)}]}
        response=request_review(challenge,case['prior_reply'],{'type':'object'},Backend())
        result=assess_response(challenge,case['prior_reply'],response)
        self.assertTrue(result['contract_valid'])
        self.assertEqual(result['effective_status'],'OPEN_PENDING_CONTENT_REVIEW')
        self.assertTrue(result['boundary_review']['findings'])
        history=start_history('WORLD',challenge,case['prior_reply'])
        history=record_response(history,challenge,case['prior_reply'],response,native_audit={'status':'NOT_RUN'})
        self.assertEqual(history['events'][0]['payload']['response']['reasoning_revision'],case['inference_map'])
        self.assertEqual(followup_state(history)['rounds_remaining'],1)

    def test_typed_context_and_revision_are_hash_bound_and_required(self):
        case=self.cases['mistaken_means_belief'];challenge=self.challenge(case)
        bad=deepcopy(challenge);bad['inference_map']['claims'][-1]['text']='changed'
        with self.assertRaises(ValueError):check_challenge(bad,case['prior_reply'])
        response=protocol_tests.InferenceReviewTests().response(challenge)
        self.assertFalse(assess_response(challenge,case['prior_reply'],response)['contract_valid'])
        response['reasoning_revision']=deepcopy(case['inference_map'])
        response['reasoning_revision']['claims'][0]['id']='new-physical'
        response['reasoning_revision']['bridges'][0]['premise_ids'][0]='new-physical'
        result=assess_response(challenge,case['prior_reply'],response)
        self.assertIn('MISSING_PRIOR_CLAIM_IDENTITIES',result['errors'])

    def test_all_preexisting_saved_challenges_still_validate(self):
        challenge=json.loads((ROOT/'fixtures/inference_review_lab_20261020.json').read_text())
        prior=json.loads((ROOT/'diagnostics/deontology_repair_cards_live_20261020_network/live_probe.json').read_text())
        check_challenge(challenge,prior['local_response'])
        challenge=json.loads((ROOT/'diagnostics/inference_history_20261020/next_challenge.json').read_text())
        prior=json.loads((ROOT/'diagnostics/deontology_inference_review_live_20261020/live_probe.json').read_text())
        check_challenge(challenge,prior['local_response'])

    def test_native_separates_candidate_instrumental_account_from_actual_path(self):
        sys.path.insert(0,str(ROOT/'.runtime/parliament'))
        from deontology_repair_cards import _native_proposal
        from global_workspace.deontology_ledger import DutyAssessmentProposal,calibrate_deontological_adjudication
        from global_workspace.scenario_semantics import compile_scenario_graph
        generation=json.loads((ROOT/'diagnostics/lab_all_agents_20261019/frameworks/framework_generation.json').read_text())
        ledger=next(c for c in generation['cycles'][-1]['candidates'] if c['specialist']=='deontological')['committed_native_ledger']
        row=_native_proposal(ledger)['assessments'][0]
        row['means_relation']='INTENDED_AS_MEANS'
        row['candidate_maxim'].update(person_role='INTENDED_MEANS',instrumental_belief='RELIES_ON_PERSON',
            instrumental_use='USES_PERSON',practical_reasoning='Under this candidate policy, the operator believes employing the staff will further rescue, even if mistaken.')
        grounding=generation['action_source_grounding']
        graph=compile_scenario_graph(generation['scenario'],generation['actions'],grounding['actions'],world_model=grounding['world_model'])
        action=next(n for n in graph.nodes.values() if n.kind=='ACTION' and n.attributes.get('canonical_action_id')=='A0')
        errors=calibrate_deontological_adjudication(graph,action,DutyAssessmentProposal.model_validate(row)).errors
        self.assertNotIn('intended-as-means classification lacks an in-action causal path',errors)
        row['candidate_maxim']=None
        legacy=calibrate_deontological_adjudication(graph,action,DutyAssessmentProposal.model_validate(row)).errors
        self.assertIn('intended-as-means classification lacks an in-action causal path',legacy)


if __name__=='__main__':unittest.main()
