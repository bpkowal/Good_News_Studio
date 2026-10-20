from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from deontology_rule_attention import source_rule_context, rule_attention_instruction, authored_duty_proposal
from logic_puzzles_review import candidate_maxim_review, review_jobs, reconcile
from run_logic_puzzles import project_conflicts
from parliament_runtime import DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON

BASE = Path('diagnostics/deontic_choice_composition_20261016')
TRACE = BASE/'deliberation/frameworks/framework_generation.json'


class RuleAttentionTests(unittest.TestCase):
    def source(self):
        return json.loads(TRACE.read_text())

    def test_unusable_model_payload_does_not_break_diagnostic_preservation(self):
        calls = [{'response': {'choices': [{'text': raw}]}} for raw in ('[]', 'null', '42', '{')]
        self.assertIsNone(authored_duty_proposal(calls))

    def test_source_binding_preserves_unknowns_and_does_not_infer_authority(self):
        world = self.source()['action_source_grounding']['world_model']; before = deepcopy(world)
        context = source_rule_context(world)
        self.assertEqual(world, before)
        self.assertEqual(context['moral_authority'], 'NOT_ASSESSED')
        self.assertEqual(context['rules'][0]['governed_proposition_id'], 'PR1')
        self.assertIn('while', context['rules'][0]['content'])
        self.assertIn('only after', context['rules'][1]['content'])
        self.assertIsNone(context['rules'][1]['bearer'])
        self.assertNotEqual(context['rules'][0]['authority'], context['rules'][1]['authority'])
        self.assertEqual(rule_attention_instruction(source_rule_context({})), '')

    def test_rejected_authored_maxims_remain_available_but_nonoperative(self):
        trace = self.source()
        candidate = next(c for c in trace['cycles'][-1]['candidates'] if c['specialist']=='deontological')
        calls = json.loads((TRACE.parent/'deontological/model_calls.json').read_text())
        proposal = authored_duty_proposal(calls)
        self.assertFalse(proposal['operative'])
        self.assertEqual(proposal['duties'], json.loads(calls[-1]['response']['choices'][0]['text'])['dp'])
        candidate['authored_duty_proposal'] = proposal
        packet = candidate_maxim_review(candidate, trace['action_source_grounding']['world_model'])
        self.assertEqual(len(packet['options']), 2)
        self.assertTrue(all(o['candidate_maxim'] for o in packet['options']))
        self.assertFalse(packet['world_fact'])

    def test_unadmitted_warnings_route_to_rule_and_maxim_review_without_support(self):
        trace = self.source(); before = deepcopy(trace)
        graph = project_conflicts(trace)
        ids = [n['id'] for n in graph['nodes'] if n.get('origin')=='deontological' and n['kind']=='UNRESOLVED_CONFLICT']
        job = review_jobs(trace, ids)[0]
        self.assertEqual(job['framework'], 'deontological')
        self.assertEqual(job['claims'][0]['kind'], 'PROPOSED_ARGUMENT')
        self.assertIn('emergency', job['question'])
        self.assertIn('POSSIBLE staff exposure', job['question'])
        self.assertEqual(trace, before)
        self.assertFalse(any(e['kind']=='SUPPORT' and e['target']==job['claims'][0]['id'] for e in graph['edges']))
        result = reconcile(trace, [job], {}, {})
        self.assertEqual(result['cycles'][-1]['candidates'], trace['cycles'][-1]['candidates'])
        self.assertTrue(all(i['objection_check']=='OPEN' for i in result['targeted_review']['issues']))

    def test_native_recovery_is_one_fresh_assessment_not_admitted_prior_replay(self):
        if not DEFAULT_PARLIAMENT_PYTHON.exists():
            self.skipTest('Native runtime unavailable')
        code = r'''
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from run_logic_puzzles import project_conflicts
from logic_puzzles_review import review
p=Path(sys.argv[2]); source=json.loads(p.read_text()); before=p.read_bytes()
ids=[n['id'] for n in project_conflicts(source)['nodes'] if n.get('origin')=='deontological' and n['kind']=='UNRESOLVED_CONFLICT'][:1]
class Fake:
    def complete_json(self,prompt,**kwargs):
        assert 'SOURCE-RULE ASSESSMENT' in prompt
        assert 'SAVED_MODEL_RESPONSE' in prompt
        assert 'Your assigned challenges: []' not in prompt
        return {'choices':[{'text':'{}'}]}
r=review(p,Path(sys.argv[3]),lambda name:Fake(),max_calls=1,allow_partial=False,allow_residual=False,issue_ids=ids)
assert p.read_bytes()==before
assert r['action_source_grounding']==source['action_source_grounding']
assert r['targeted_review']['adapter_calls']=={'deontological':1},r['targeted_review']
assert r['targeted_review']['nonoperative_recovery_frameworks']==['deontological']
assert r['targeted_review']['prior_replay_calls']=={'deontological':0}
assert all(i['objection_check']=='OPEN' for i in r['targeted_review']['issues'])
assert len(json.loads((Path(sys.argv[3])/'deontological/native_trace.json').read_text())['cycles'])<=1
assert not json.loads((Path(sys.argv[3])/'deontological/prior_replay.json').read_text())['seeded_as_operative']
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(DEFAULT_PARLIAMENT_PYTHON), '-c', code,
                                     str(DEFAULT_PARLIAMENT_ROOT), str(TRACE.resolve()), folder],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
