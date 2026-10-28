from copy import deepcopy
import json
import unittest
from pathlib import Path
import tempfile

from deontology_repair_cards import fingerprint
from inference_review import make_challenge
from inference_history import (start_history,validate_history,record_response,review_response,
    followup_state,revision_context,review_context_for,dependency_projection,current_claims)
import test_inference_review as protocol_tests


class HistoryTests(unittest.TestCase):
    def setup_history(self):
        helper=protocol_tests.InferenceReviewTests()
        prior,challenge=helper.packet()
        state=start_history('WORLD',challenge,prior)
        target=state['root']['target_claim_id']
        state=start_history('WORLD',challenge,prior,dependent_claims=[
            {'id':'classification','text':'Classification based on that inference','status':'SUPPORTED',
             'depends_on':[target],'evidence_refs':[],'origin':'PRIOR_ARGUMENT'},
            {'id':'priority','text':'Priority relying on that classification','status':'SUPPORTED',
             'depends_on':['classification'],'evidence_refs':[],'origin':'PRIOR_ARGUMENT'},
            {'id':'other','text':'Independent supported claim','status':'SUPPORTED',
             'depends_on':[],'evidence_refs':['E1'],'origin':'PRIOR_ARGUMENT'}])
        response=helper.response(challenge)
        return state,prior,challenge,response

    def reviewed(self,state,response,**kwargs):
        params=dict(reviewer='CONTENT_REVIEWER',outcome='PARTIAL',rationale='Specific inference remains open.',
                    remaining_gap='Defend the remaining bridge.',corrected_points=['An overclaim was removed.'],
                    new_unsupported_claims=[])
        params.update(kwargs)
        return review_response(state,fingerprint(response),**params)

    def first(self):
        state,prior,challenge,response=self.setup_history()
        state=record_response(state,challenge,prior,response,native_audit={'status':'STAGED_NATIVE_REVALIDATED'})
        return state,prior,challenge,response

    def test_persistent_roundtrip_preserves_events_and_inputs(self):
        state,prior,challenge,response=self.setup_history(); original=deepcopy(state)
        updated=record_response(state,challenge,prior,response,native_audit={'status':'REJECTED'})
        self.assertEqual(state,original)
        self.assertTrue(validate_history(json.loads(json.dumps(updated))))
        self.assertEqual(updated['events'][0]['payload']['native_status'],'REJECTED')
        self.assertEqual(followup_state(updated)['status'],'AWAITING_CONTENT_REVIEW')

    def test_withdrawal_reopens_transitive_dependents_only(self):
        state,_,_,response=self.first(); target=state['root']['target_claim_id']
        state=self.reviewed(state,response,claim_updates=[{'id':target,'status':'WITHDRAWN'}])
        claims={c['id']:c for c in dependency_projection(state)}
        self.assertEqual(claims['classification']['effective_status'],'REQUIRES_REASSESSMENT')
        self.assertEqual(claims['priority']['effective_status'],'REQUIRES_REASSESSMENT')
        self.assertEqual(claims['other']['effective_status'],'SUPPORTED')

    def test_conditional_support_propagates_without_becoming_fact(self):
        state,_,_,response=self.first(); target=state['root']['target_claim_id']
        state=self.reviewed(state,response,claim_updates=[{'id':target,'status':'CONDITIONAL'}])
        projected={c['id']:c['effective_status'] for c in dependency_projection(state)}
        self.assertEqual(projected['classification'],'CONDITIONAL')
        self.assertEqual(projected['priority'],'CONDITIONAL')
        self.assertFalse(state['root']['operative'])

    def test_reviewer_can_be_wrong_and_defended_claim_can_stop(self):
        state,prior,challenge,response=self.setup_history()
        response['disposition']='DEFENDED'
        state=record_response(state,challenge,prior,response,native_audit={'status':'STAGED_NATIVE_REVALIDATED'})
        self.assertEqual(followup_state(state)['status'],'AWAITING_CONTENT_REVIEW')
        state=self.reviewed(state,response,outcome='DEFENDED',remaining_gap='',corrected_points=[],
            rationale='Reviewer overlooked supplied support; the original claim is defensible.',
            claim_updates=[{'id':state['root']['target_claim_id'],'status':'SUPPORTED'}])
        self.assertEqual(followup_state(state)['status'],'STOP_REVIEWED')
        self.assertEqual(dependency_projection(state)[0]['effective_status'],'SUPPORTED')

    def test_two_rounds_latest_reply_and_remaining_gap(self):
        state,prior,challenge,response=self.first()
        state=self.reviewed(state,response,new_unsupported_claims=['A fresh unsupported premise.'])
        context=revision_context(state)
        self.assertEqual(context['round'],2)
        self.assertEqual(context['remaining_gap'],'Defend the remaining bridge.')
        self.assertIn('A fresh unsupported premise.',str(context))
        # The helper's local response lacks explanation; anchor to classification.
        next_prior=deepcopy(state['events'][0]['payload']['response']['local_response'])
        next_card=make_challenge(next_prior,claim_path='classification',quote='UNRESOLVED',
            defect_kind='MISSING_BRIDGE',missing_premise=context['remaining_gap'],
            evidence=challenge['evidence'],affected_paths=challenge['affected_paths'],reviewer='CONTENT_REVIEWER')
        second=protocol_tests.InferenceReviewTests().response(next_card)
        state=record_response(state,next_card,next_prior,second,native_audit={'status':'STAGED_NATIVE_REVALIDATED'})
        state=self.reviewed(state,second)
        self.assertEqual(followup_state(state)['status'],'STOP_ROUND_LIMIT')
        with self.assertRaises(ValueError):revision_context(state)

    def test_no_progress_stops_even_with_budget_left(self):
        state,_,_,response=self.first()
        state=self.reviewed(state,response,outcome='NO_PROGRESS',corrected_points=[])
        self.assertEqual(followup_state(state)['status'],'STOP_NO_PROGRESS')

    def test_stale_reply_changed_world_and_expanded_scope_fail_before_call(self):
        state,prior,challenge,response=self.first();state=self.reviewed(state,response)
        with self.assertRaises(ValueError):review_context_for(state,challenge,prior,'WORLD')
        state,prior,challenge,response=self.setup_history()
        with self.assertRaises(ValueError):review_context_for(state,challenge,prior,'CHANGED')
        wider=make_challenge(prior,claim_path='explanation',quote=prior['explanation'],
            defect_kind='MISSING_BRIDGE',missing_premise='Extra scope',evidence=challenge['evidence'],
            affected_paths=['verdict'],reviewer='REVIEWER')
        with self.assertRaises(ValueError):review_context_for(state,wider,prior,'WORLD')

    def test_edited_event_duplicate_review_and_bad_dependencies_rejected(self):
        state,_,_,response=self.first(); altered=deepcopy(state)
        altered['events'][0]['payload']['response']['disposition']='DEFENDED'
        with self.assertRaises(ValueError):validate_history(altered)
        reviewed=self.reviewed(state,response)
        with self.assertRaises(ValueError):self.reviewed(reviewed,response)
        target=state['root']['target_claim_id']
        with self.assertRaises(ValueError):self.reviewed(state,response,
            claim_updates=[{'id':target,'depends_on':['priority']}])
        with self.assertRaises(ValueError):self.reviewed(state,response,
            claim_updates=[{'id':target,'depends_on':['missing']}])

    def test_new_claims_are_separate_from_old_claim_identity(self):
        state,_,_,response=self.first()
        reviewed=self.reviewed(state,response,new_unsupported_claims=['Unstated risk bound'],
            new_claims=[{'id':'new-bound','text':'Risk is negligible','status':'UNKNOWN',
                         'depends_on':[],'evidence_refs':[],'origin':'REVIEWED_MODEL_ARGUMENT'}])
        self.assertEqual(len(current_claims(reviewed)),5)
        self.assertEqual(len(current_claims(state)),4)

    def test_saved_lab_replay_preserves_sources_and_next_question(self):
        from replay_inference_history import replay
        root=Path(__file__).resolve().parent
        probe=root/'diagnostics/deontology_inference_review_live_20261020/live_probe.json'
        review=root/'fixtures/inference_history_lab_content_review_20261020.json'
        before=(probe.read_bytes(),review.read_bytes())
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)/'history'
            result=replay(probe,review,destination)
            self.assertEqual(result['api_calls'],0)
            self.assertEqual(result['followup']['rounds_remaining'],1)
            projected={c['id']:c['effective_status'] for c in result['argument_dependencies']}
            self.assertEqual(projected['permissibility'],'REQUIRES_REASSESSMENT')
            self.assertEqual(projected['staff-risk'],'SUPPORTED')
            self.assertEqual(projected['side-effect'],'CONDITIONAL')
            self.assertTrue((destination/'next_challenge.json').exists())
            with self.assertRaises(FileExistsError):replay(probe,review,destination)
        self.assertEqual((probe.read_bytes(),review.read_bytes()),before)


if __name__=='__main__':unittest.main()
