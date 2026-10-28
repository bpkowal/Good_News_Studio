"""Offline scope, continuity and native-boundary tests for local repairs."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'.runtime/parliament'))
from deontology_repair_cards import build_cards, apply_repair, request_repair, PERSON_PATHS, DEPENDENTS, fingerprint
from replay_deontology_repairs import replay


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT/'diagnostics/lab_all_agents_20261019/frameworks/framework_generation.json'
        cls.reply_path = ROOT/'diagnostics/lab_all_agents_20261019/review/deontological/model_calls.json'
        cls.generation = json.loads(cls.source.read_text())
        cls.ledger = next(c for c in cls.generation['cycles'][-1]['candidates']
                          if c['specialist']=='deontological')['committed_native_ledger']
        cls.world = cls.generation['action_source_grounding']['world_model']

    def card(self, family='HARM_CLASSIFICATION', ledger=None):
        return next(c for c in build_cards(self.world, ledger or self.ledger) if c.family==family)

    def response(self, card):
        changes = deepcopy(card.current_values)
        if card.family=='HARM_CLASSIFICATION':
            changes['harm_relation']='MIXED'
        else:
            changes['candidate_maxim.instrumental_belief']='DOES_NOT_RELY'
            changes['candidate_maxim.instrumental_use']='DOES_NOT_USE'
            changes['candidate_maxim.practical_reasoning']='The door, rather than the staff, is used to reach the technician.'
        return dict(card_id=card.card_id, changes=changes, evidence_refs=['E2'],
                    explanation='A proposed local classification, not a moral verdict.', unresolved_questions=[])

    def test_cards_are_trusted_local_scopes_with_exact_world_copies(self):
        cards=build_cards(self.world,self.ledger)
        self.assertEqual({c.family for c in cards},{'HARM_CLASSIFICATION','PERSON_ROLE_CONSISTENCY'})
        for card in cards:
            self.assertEqual(card.action_id,'A0')
            self.assertEqual(card.world_sha256,fingerprint(self.world))
            self.assertEqual(card.record_sha256,fingerprint(self.ledger['records'][0]))
            for e in card.evidence:
                if e['kind']=='FROZEN_WORLD_COPY':
                    collection,index=e['path'].strip('/').split('/')
                    self.assertEqual(e['value'],self.world[collection][int(index)])
                    self.assertEqual(e['value']['action_id'],'A0')

    def test_repairs_reopen_closure_preserve_world_and_other_action(self):
        original_world,original_ledger=deepcopy(self.world),deepcopy(self.ledger)
        for family in ('HARM_CLASSIFICATION','PERSON_ROLE_CONSISTENCY'):
            card=self.card(family)
            result,audit=apply_repair(self.world,self.ledger,card,self.response(card))
            self.assertEqual(audit['status'],'STAGED_NATIVE_REVALIDATED',audit['errors'])
            self.assertFalse(audit['operative'])
            self.assertEqual(result['transaction_status'],'STAGED_NONVOTING')
            self.assertEqual(result['records'][1],self.ledger['records'][1])
            for path,value in DEPENDENTS.items():
                from deontology_repair_cards import get_path
                self.assertEqual(get_path(result['records'][0],path),value)
                self.assertEqual(audit['superseded_closure'][path],get_path(self.ledger['records'][0],path))
            self.assertEqual(result['records'][0]['norm'],self.ledger['records'][0]['norm'])
        self.assertEqual(self.world,original_world)
        self.assertEqual(self.ledger,original_ledger)

    def test_scope_escalation_and_model_self_authorization_rejected(self):
        card=self.card()
        for key,value in [('norm','new principle'),('verdict','PERMITTED'),('priority_basis','PERFECT_DUTY'),
                          ('resolution_status','RESOLVED'),('world_model',{}),('scores',{'A0':1}),
                          ('A1.harm_relation','MIXED'),('revision_category','EVIDENCE_REPAIR')]:
            response=self.response(card); response['changes'][key]=value
            result,audit=apply_repair(self.world,self.ledger,card,response)
            self.assertEqual(audit['status'],'REJECTED',key)
            self.assertEqual(result,self.ledger)
        response=self.response(card); response['operative']=True
        self.assertEqual(apply_repair(self.world,self.ledger,card,response)[1]['status'],'REJECTED')

    def test_stale_world_record_and_tampered_permission_rejected(self):
        card=self.card(); response=self.response(card)
        world=deepcopy(self.world); world['schema_version']='changed'
        ledger=deepcopy(self.ledger); ledger['records'][0]['norm']='changed'
        for w,l,c in [(world,self.ledger,card),(self.world,ledger,card),
                      (self.world,self.ledger,replace(card,target_paths=('norm',)))]:
            self.assertIn('STALE_OR_ALTERED_CARD',apply_repair(w,l,c,response)[1]['errors'])

    def test_atomic_person_bundle_and_native_contradiction_rejected(self):
        card=self.card('PERSON_ROLE_CONSISTENCY'); response=self.response(card)
        del response['changes']['means_relation']
        self.assertEqual(apply_repair(self.world,self.ledger,card,response)[1]['status'],'REJECTED')
        response=self.response(card)
        response['changes']['candidate_maxim.instrumental_belief']='RELIES_ON_PERSON'
        response['changes']['candidate_maxim.instrumental_use']='USES_PERSON'
        self.assertEqual(apply_repair(self.world,self.ledger,card,response)[1]['status'],'REJECTED')
        response=self.response(card); response['changes']['candidate_maxim.means_effect_ids']=['E3']
        self.assertEqual(apply_repair(self.world,self.ledger,card,response)[1]['status'],'REJECTED')

    def test_wrong_evidence_invalid_values_and_noop_rejected(self):
        card=self.card()
        for refs in ([],['E3'],['invented']):
            response=self.response(card); response['evidence_refs']=refs
            self.assertEqual(apply_repair(self.world,self.ledger,card,response)[1]['status'],'REJECTED')
        for value in ('DOING_HARM','INVENTED',{},None):
            response=self.response(card); response['changes']['harm_relation']=value
            self.assertEqual(apply_repair(self.world,self.ledger,card,response)[1]['status'],'REJECTED')

    def test_rejected_sibling_does_not_erase_successful_repair(self):
        card=self.card(); result,audit=apply_repair(self.world,self.ledger,card,self.response(card))
        self.assertEqual(audit['status'],'STAGED_NATIVE_REVALIDATED')
        stale=self.card('PERSON_ROLE_CONSISTENCY')
        retained,failed=apply_repair(self.world,result,stale,self.response(stale))
        self.assertEqual(failed['status'],'REJECTED')
        self.assertEqual(retained,result)
        fresh=self.card('PERSON_ROLE_CONSISTENCY',result)
        final,passed=apply_repair(self.world,result,fresh,self.response(fresh))
        self.assertEqual(passed['status'],'STAGED_NATIVE_REVALIDATED',passed['errors'])
        self.assertEqual(final['records'][0]['harm_relation'],'MIXED')

    def test_person_repair_does_not_hide_unrepaired_harm_issue(self):
        card=self.card('PERSON_ROLE_CONSISTENCY')
        result,audit=apply_repair(self.world,self.ledger,card,self.response(card))
        self.assertEqual(audit['status'],'STAGED_NATIVE_REVALIDATED',audit['errors'])
        self.assertEqual([c.family for c in build_cards(self.world,result)],['HARM_CLASSIFICATION'])
        harm=self.card('HARM_CLASSIFICATION',result)
        final,passed=apply_repair(self.world,result,harm,self.response(harm))
        self.assertEqual(passed['status'],'STAGED_NATIVE_REVALIDATED',passed['errors'])
        self.assertEqual(build_cards(self.world,final),[])

    def test_compact_backend_request_has_only_local_schema(self):
        card=self.card('PERSON_ROLE_CONSISTENCY'); expected=self.response(card)
        class Backend:
            def complete_json(inner,prompt,**kwargs):
                self.assertLess(len(prompt),14000)
                self.assertEqual(set(kwargs['schema']['properties']['changes']['properties']),set(PERSON_PATHS))
                self.assertEqual(kwargs['max_tokens'],1200)
                return {'choices':[{'text':json.dumps(expected)}]}
        self.assertEqual(request_repair(card,Backend()),expected)

    def test_saved_reply_replay_is_nonvoting_and_two_repairs_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            result=replay(self.source,self.reply_path,folder,ROOT/'.runtime/parliament')
            self.assertEqual(result['api_calls'],0)
            self.assertFalse(result['operative'])
            self.assertTrue(result['unrelated_action_unchanged'])
            self.assertTrue(all(a['status']=='STAGED_NATIVE_REVALIDATED' for a in result['audits']))
            self.assertEqual(len(result['audits']),2)
            self.assertEqual(result['staged_ledger']['records'][0]['norm'],self.ledger['records'][0]['norm'])
            self.assertEqual(build_cards(self.world,result['staged_ledger']),[])


if __name__=='__main__':
    unittest.main()
