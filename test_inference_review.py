from copy import deepcopy
import json
import unittest

from inference_review import make_challenge, check_challenge, request_review, assess_response


class InferenceReviewTests(unittest.TestCase):
    def packet(self,claim='The graph proves that nobody relies on the person.',kind='MISSING_BRIDGE'):
        prior={'explanation':claim}
        card=make_challenge(prior,claim_path='explanation',quote=claim,
            defect_kind=kind,missing_premise='The asserted bridge is not established.',
            evidence=[{'ref':'E1','kind':'SOURCE','value':'a stated premise'}],
            affected_paths=['classification'],reviewer='TEST_REVIEWER')
        return prior,card

    def response(self,card,disposition='WITHDRAWN'):
        return dict(challenge_id=card['challenge_id'],disposition=disposition,
                    premises=[{'claim':'A source premise','basis':'SOURCE_EVIDENCE','evidence_refs':['E1']}],
                    bridge='The broader inference remains an assumption.',
                    withdrawn_claims=[card['quote']],qualifications=[],strongest_conclusion='A qualified reading only.',
                    remaining_questions=['Is the missing premise true?'],local_response={'classification':'UNRESOLVED'})

    def test_same_protocol_accepts_distinct_inference_families(self):
        for claim,kind in [('No causal edge proves no intention.','ABSENCE_AS_EVIDENCE'),
                           ('Trust costs cannot outweigh a life.','UNSUPPORTED_FACT'),
                           ('The posted rule defeats emergency rescue.','NORMATIVE_PRIORITY')]:
            prior,card=self.packet(claim,kind)
            result=assess_response(card,prior,self.response(card))
            self.assertTrue(result['contract_valid'])
            self.assertEqual(result['effective_status'],'OPEN_PENDING_CONTENT_REVIEW')

    def test_self_reported_defense_cannot_close_objection(self):
        prior,card=self.packet()
        result=assess_response(card,prior,self.response(card,'DEFENDED'))
        self.assertTrue(result['contract_valid'])
        self.assertEqual(result['semantic_support'],'NOT_INDEPENDENTLY_VERIFIED')
        self.assertEqual(result['effective_status'],'OPEN_PENDING_CONTENT_REVIEW')

    def test_exact_claim_and_prior_hash_are_required(self):
        prior,card=self.packet()
        with self.assertRaises(ValueError):check_challenge(card,{'explanation':'changed'})
        card['affected_paths'].append('verdict')
        with self.assertRaises(ValueError):check_challenge(card,prior)

    def test_citation_ids_do_not_verify_semantic_support(self):
        prior,card=self.packet(); response=self.response(card)
        response['premises'][0]['claim']='Unsupported assertion with a real citation'
        result=assess_response(card,prior,response)
        self.assertTrue(result['contract_valid'])
        self.assertEqual(result['semantic_support'],'NOT_INDEPENDENTLY_VERIFIED')
        response['premises'][0]['evidence_refs']=['invented']
        self.assertFalse(assess_response(card,prior,response)['contract_valid'])

    def test_disposition_requires_corresponding_account(self):
        prior,card=self.packet()
        for disposition,field in [('WITHDRAWN','withdrawn_claims'),('QUALIFIED','qualifications'),
                                  ('UNRESOLVED','remaining_questions')]:
            response=self.response(card,disposition); response[field]=[]
            self.assertFalse(assess_response(card,prior,response)['contract_valid'])

    def test_request_exposes_prior_claim_and_separates_assumptions(self):
        prior,card=self.packet(); expected=self.response(card)
        class Backend:
            def complete_json(inner,prompt,**kwargs):
                self.assertIn(card['quote'],prompt)
                self.assertIn('NORMATIVE_ASSUMPTION',json.dumps(kwargs['schema']))
                self.assertNotIn('Mira',prompt)
                return {'choices':[{'text':json.dumps(expected)}]}
        self.assertEqual(request_review(card,prior,{'type':'object'},Backend()),expected)


if __name__=='__main__':unittest.main()
