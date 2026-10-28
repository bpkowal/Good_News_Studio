from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from behavioral_policy import ContextualPolicy
import deliberation_game as game


class DeliberationGameTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)/'game'
        self.case={'scenario':'Ellis promised Sasha the blanket. Rowan also needs it.',
                   'world':{'actions':[{'action_id':'A0','intervention':'give Rowan'},
                                       {'action_id':'A1','intervention':'give Sasha'}]},
                   'frameworks':[{'framework':'care','reason':'Consider responsiveness.'},
                                 {'framework':'contractualist','reason':'Compare individual complaints.'}]}
        game.create(self.folder,self.case)
        self.requests=[]

    def backend(self,request,folder):
        self.requests.append(request)
        return {'status':'RETURNED','provider_calls':1,'response':{
            'summary':'The effect of reliance is still unresolved.',
            'source_quotes':['Ellis promised Sasha the blanket.'],
            'remaining_questions':['Did Sasha rely on the promise?'],
            'counterfactuals':['If reliance caused an additional burden, the comparison might change.'],
            'suggested_action_id':None}}

    def test_preparation_and_no_feedback_make_no_learning_claim(self):
        state,_,moves,ratings=game.read(self.folder)
        self.assertEqual(state['policy']['revision'],0)
        self.assertFalse(moves or ratings)
        self.assertFalse(state['world_mutation'] or state['native_dispatch'])
        self.assertFalse(ContextualPolicy.restore(state['policy']).interactions)

    def test_unassessed_cues_are_not_confused_with_assessed_zero(self):
        state,*_=game.read(self.folder)
        unknown=game.context(state)
        state['question_cues']['normative_conflict']=0
        assessed=game.context(state)
        self.assertEqual(unknown['normative_conflict'],assessed['normative_conflict'])
        self.assertNotEqual(unknown['normative_conflict_assessed'],assessed['normative_conflict_assessed'])

    def test_round_limit_and_stop_keep_world_unchanged(self):
        before=(self.folder/'case.json').read_bytes()
        for _ in range(2):game.move(self.folder,'investigate',framework='care',backend=self.backend)
        with self.assertRaises(ValueError):game.move(self.folder,'investigate',framework='care',backend=self.backend)
        record=game.move(self.folder,'stop',action_id=None,rationale='The priority is unresolved.')
        self.assertEqual(record['provider_calls'],0)
        self.assertEqual(len(self.requests),2)
        self.assertEqual((self.folder/'case.json').read_bytes(),before)

    def test_only_explicit_human_rating_updates_weights_and_duplicate_rolls_back(self):
        selected=game.move(self.folder,'investigate',framework='contractualist',backend=self.backend)
        self.assertEqual(selected['status'],'ARGUMENT_RECORDED_NOT_NATIVE_ADMITTED')
        state,*_=game.read(self.folder)
        self.assertEqual(state['policy']['revision'],0)
        game.rate(self.folder,selected['id'],0.9,rater='tester',reason='The question exposed a missing premise.')
        state,*_=game.read(self.folder);saved=deepcopy(state['policy'])
        self.assertEqual(saved['revision'],1)
        with self.assertRaises(sqlite3.IntegrityError):
            game.rate(self.folder,selected['id'],1,rater='tester',reason='Duplicate.')
        self.assertEqual(game.read(self.folder)[0]['policy'],saved)

    def test_weight_checkpoint_continues_in_another_episode(self):
        m=game.move(self.folder,'investigate',framework='care',backend=self.backend)
        game.rate(self.folder,m['id'],1,rater='tester',reason='Useful question.')
        saved=json.loads((self.folder/'policy_checkpoint.json').read_text())
        other=Path(self.temp.name)/'next'
        state=game.create(other,self.case,policy=saved)
        self.assertEqual(state['policy'],saved)

    def test_scenario_action_weights_cannot_be_relabeled_as_control_weights(self):
        bad=ContextualPolicy(('disclose','protect'),('coercion',)).snapshot()
        with self.assertRaises(ValueError):game.create(Path(self.temp.name)/'invalid',self.case,policy=bad)

    def test_unsupported_quotes_and_actions_stay_flagged_for_peers(self):
        def unsupported(request,folder):
            r=self.backend(request,folder)
            r['response']['source_quotes']=['Nobody was promised anything.']
            r['response']['suggested_action_id']='A99'
            return r
        m=game.move(self.folder,'self_counterfactual',framework='care',backend=unsupported)
        self.assertEqual(m['status'],'ARGUMENT_RECORDED_NOT_NATIVE_ADMITTED')
        self.assertEqual({f['kind']for f in m['findings']},{'UNSUPPORTED_SOURCE_QUOTE','UNKNOWN_ACTION_PROPOSAL'})
        game.move(self.folder,'deliberate',framework='contractualist',peer='care',backend=self.backend)
        packet=json.loads(self.requests[-1]['messages'][1]['content'])
        self.assertEqual(packet['game_history'][0]['findings'],m['findings'])
        self.assertEqual(packet['world'],self.case['world'])

    def test_counterfactual_attention_zero_removes_that_message_from_peer_packet(self):
        game.move(self.folder,'self_counterfactual',framework='care',backend=self.backend)
        game.configure(self.folder,counterfactual_attention=0,author='tester')
        game.move(self.folder,'deliberate',framework='contractualist',peer='care',backend=self.backend)
        packet=json.loads(self.requests[-1]['messages'][1]['content'])
        self.assertEqual(packet['game_history'],[])
        self.assertEqual(len(game.read(self.folder)[2]),2)

    def test_new_information_is_attributed_and_kept_outside_frozen_world(self):
        before=(self.folder/'case.json').read_bytes()
        game.observe(self.folder,'Sasha reports declining other help because of the assurance.',author='player')
        game.move(self.folder,'investigate',framework='contractualist',backend=self.backend)
        packet=json.loads(self.requests[-1]['messages'][1]['content'])
        self.assertEqual(packet['user_provided_game_information'][0]['author'],'player')
        self.assertEqual((self.folder/'case.json').read_bytes(),before)

    def test_failed_provider_attempt_is_retained_and_not_rewarded(self):
        def failure(request,folder):return {'status':'FAILED','provider_calls':1}
        m=game.move(self.folder,'investigate',framework='care',backend=failure)
        state,_,moves,ratings=game.read(self.folder)
        self.assertEqual(m['status'],'FAILED_OR_INCOMPLETE')
        self.assertEqual(state['remaining_rounds'],1)
        self.assertEqual(state['policy']['revision'],0)
        self.assertEqual(len(moves),1)
        self.assertEqual(ratings,[])


if __name__=='__main__':unittest.main()
