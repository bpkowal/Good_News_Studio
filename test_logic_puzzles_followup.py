from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess

from logic_puzzles_followup import significant_questions,followup_offer,deepen,stage_historical_review,fingerprint
from run_parliament_demo import approve_followup

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'diagnostics/causal_means_reasoning_20261015/live_review/targeted_review.json'


class FollowupTests(unittest.TestCase):
    def native(self, body):
        from parliament_runtime import DEFAULT_PARLIAMENT_PYTHON, DEFAULT_PARLIAMENT_ROOT
        setup=r'''
import sys,json,tempfile
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,sys.argv[1])
from logic_puzzles_followup import deepen,stage_historical_review,fingerprint
source=Path('diagnostics/causal_means_reasoning_20261015/live_review/targeted_review.json')
before=source.read_bytes()
saved=json.loads((source.parent/'deontological/model_calls.json').read_text())[0]['response']
class Fake:
    calls=0
    change=True
    def complete_json(self,prompt,**kwargs):
        self.calls+=1
        assert 'Your assigned challenges: []' not in prompt
        assert 'intending harm as an end' in prompt
        if self.calls==2:assert 'Additional defense attempt 1' in prompt
        data=json.loads(saved['choices'][0]['text'])
        if self.change:data['dp']['A0']['mx']['universalization']+=' Additional defense attempt '+str(self.calls)+'.'
        data['qa']={'issue_id':kwargs['schema']['properties']['qa']['properties']['issue_id']['enum'][0],
            'disposition':'RESOLVED','effect':'NO_CHANGE','current_position_effect':'NO_CHANGE','boundary_effect':'NO_SWITCH',
            'answer':'This is an additional normative defense, not a new physical fact about Dana.',
            'follow_up_question':'Does the proposed prohibition adequately address rescue?'}
        return {'choices':[{'text':json.dumps(data)}]}
'''
        result=subprocess.run([str(DEFAULT_PARLIAMENT_PYTHON),'-c',setup+body,str(DEFAULT_PARLIAMENT_ROOT)],
                              cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_significant_application_is_a_question_not_a_world_fact(self):
        data=json.loads(SOURCE.read_text());before=deepcopy(data)
        offer=followup_offer(data)
        self.assertEqual(data,before)
        self.assertEqual(offer['status'],'AVAILABLE')
        self.assertTrue(offer['requires_user_opt_in'])
        q=offer['questions'][0]
        self.assertEqual(q['kind'],'NORM_APPLICATION')
        self.assertEqual(q['authority'],'INVESTIGATIVE_ONLY')
        self.assertIn('intending harm as an end',q['question'])
        self.assertIn('does not establish',q['question'])
        self.assertIn('another prohibition',q['question'])
        self.assertIn('retain or revise',q['question'])

    def test_no_optin_performs_no_model_calls_and_preserves_source(self):
        before=SOURCE.read_bytes()
        def forbidden(name):raise AssertionError('No backend may be constructed')
        with tempfile.TemporaryDirectory() as tmp:
            result=deepen(SOURCE,Path(tmp),forbidden)
            self.assertEqual(result['reasoning_followup']['total_adapter_calls'],0)
            self.assertEqual(result['reasoning_followup']['stop_reason'],'USER_OPT_IN_REQUIRED')
            self.assertTrue(result['reasoning_followup']['remaining_questions'])
        self.assertEqual(SOURCE.read_bytes(),before)

    def test_report_shows_offer_and_old_outputs_cannot_be_overwritten(self):
        from run_parliament_demo import summarize,write_report
        data=json.loads(SOURCE.read_text());data['follow_up_offer']=followup_offer(data)
        report=summarize(data['scenario'],{'pipeline_status':'ADMITTED'},data,None)
        with tempfile.TemporaryDirectory() as tmp:
            write_report(report,Path(tmp)/'report')
            text=(Path(tmp)/'report/DEMO_REPORT.md').read_text()
            self.assertIn('Optional deeper review',text)
            self.assertIn('--review-depth deep',text)
            self.assertIn('intending harm as an end',text)
            output=Path(tmp)/'offer'
            deepen(SOURCE,output,lambda n:None)
            before=(output/'reasoning_followup.json').read_bytes()
            with self.assertRaises(ValueError):deepen(SOURCE,output,lambda n:None)
            self.assertEqual((output/'reasoning_followup.json').read_bytes(),before)

    def test_native_second_followup_uses_first_admitted_revision_and_honors_cap(self):
        self.native(r'''
fake=Fake()
with tempfile.TemporaryDirectory() as tmp:
    result=deepen(source,Path(tmp),lambda n:fake,opt_in=True,max_rounds=2)
    deep=result['reasoning_followup']
    assert fake.calls==2 and deep['total_adapter_calls']==2
    assert len(deep['history'])==2 and deep['stop_reason']=='ROUND_LIMIT'
    assert all(h['native_review']['review_status']=='RECONCILED' for h in deep['history'])
    assert all(h['native_review']['semantic_resolution']=='NOT_INDEPENDENTLY_VERIFIED' for h in deep['history'])
    assert all(h['native_review']['objection_check']=='OPEN' for h in deep['history'])
    assert 'Additional defense attempt 1' in json.dumps(deep['history'][1]['question']['claim'])
    assert result['action_source_grounding']==json.loads(before)['action_source_grounding']
    assert source.read_bytes()==before
''')

    def test_unchanged_answer_stops_and_self_report_does_not_resolve_question(self):
        self.native(r'''
fake=Fake();fake.change=False
with tempfile.TemporaryDirectory() as tmp:
    result=deepen(source,Path(tmp),lambda n:fake,opt_in=True,max_rounds=2)
    deep=result['reasoning_followup']
    assert fake.calls==1 and deep['stop_reason']=='NO_SUBSTANTIVE_CHANGE'
    assert deep['remaining_questions']
    assert deep['history'][0]['native_review']['objection_check']=='OPEN'
    assert deep['semantic_resolution']=='NOT_INDEPENDENTLY_VERIFIED'
''')

    def test_semantically_wrong_prior_is_blocked_before_followup_call(self):
        self.native(r'''
fake=Fake()
with tempfile.TemporaryDirectory() as tmp:
    staged=stage_historical_review(source,Path(tmp)/'stage')
    path=staged.parent/'review_resume.json';resume=json.loads(path.read_text())
    row=resume['responses']['deontological']
    data=json.loads(row['response']['choices'][0]['text'])
    data['dp']['A0']['mr']='INTENDED_AS_MEANS'
    row['response']['choices'][0]['text']=json.dumps(data)
    row['response_sha256']=fingerprint(row['response'])
    path.write_text(json.dumps(resume))
    result=deepen(staged,Path(tmp)/'out',lambda n:fake,opt_in=True)
    assert fake.calls==0
    assert result['reasoning_followup']['stop_reason']=='PRIOR_STATE_REPLAY_MISMATCH'
    assert result['action_source_grounding']==json.loads(before)['action_source_grounding']
''')

    def test_rejected_followup_retains_prior_and_stops_without_extra_calls(self):
        self.native(r'''
class Bad:
    calls=0
    def complete_json(self,prompt,**kwargs):
        self.calls+=1
        return {'choices':[{'text':'{}'}]}
fake=Bad()
with tempfile.TemporaryDirectory() as tmp:
    result=deepen(source,Path(tmp),lambda n:fake,opt_in=True,max_rounds=2)
    assert fake.calls==1
    assert result['reasoning_followup']['stop_reason']=='NATIVE_REVISION_NOT_ACCEPTED'
    assert result['cycles'][-1]['candidates']==json.loads(before)['cycles'][-1]['candidates']
    assert result['action_source_grounding']==json.loads(before)['action_source_grounding']
''')
    def test_prompt_requires_explicit_yes_and_noninteractive_does_not_approve(self):
        offer=followup_offer(json.loads(SOURCE.read_text()))
        def forbidden(prompt):raise AssertionError('No hidden prompt')
        for depth in ('standard','ask'):
            self.assertFalse(approve_followup(offer,depth,interactive=False,ask=forbidden,emit=lambda x:None))
        self.assertTrue(approve_followup(offer,'deep',interactive=False,ask=forbidden))
        for answer in ('','n','no','later'):
            self.assertFalse(approve_followup(offer,'ask',interactive=True,ask=lambda p:answer,emit=lambda x:None))
        self.assertTrue(approve_followup(offer,'ask',interactive=True,ask=lambda p:'yes',emit=lambda x:None))
        def end_of_input(prompt):raise EOFError()
        self.assertFalse(approve_followup(offer,'ask',interactive=True,ask=end_of_input,emit=lambda x:None))

    def test_round_limit_and_resume_hash_are_checked_before_calls(self):
        for limit in (0,3,True):
            with self.assertRaises(ValueError):deepen(SOURCE,'unused',lambda n:None,opt_in=True,max_rounds=limit)
        with tempfile.TemporaryDirectory() as tmp:
            staged=stage_historical_review(SOURCE,Path(tmp)/'stage')
            p=staged.parent/'review_resume.json';data=json.loads(p.read_text())
            data['responses']['deontological']['response_sha256']='tampered';p.write_text(json.dumps(data))
            def forbidden(name):raise AssertionError('No model call with mismatched evidence')
            result=deepen(staged,Path(tmp)/'out',forbidden,opt_in=True)
            self.assertEqual(result['reasoning_followup']['stop_reason'],'RESUME_EVIDENCE_MISMATCH')
            self.assertEqual(result['reasoning_followup']['total_adapter_calls'],0)


if __name__=='__main__':unittest.main()
