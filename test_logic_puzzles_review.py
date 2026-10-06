from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from logic_puzzles_review import review_jobs, reconcile, close_deontological_map
from run_logic_puzzles import project_conflicts

FIXTURE = Path('diagnostics/logic_puzzles_independent_live/framework_generation.json')


class TargetedReviewTests(unittest.TestCase):
    def source(self):
        return json.loads(FIXTURE.read_text())

    def test_localized_packet_preserves_claims_and_limits_exposure(self):
        source = self.source()
        before = deepcopy(source)
        jobs = review_jobs(source)
        self.assertEqual(source, before)
        self.assertLessEqual(sum(len(j['issues']) for j in jobs), 2)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(len(jobs[0]['claims']), 1)
        self.assertEqual(jobs[0]['framework'], 'deontological')
        self.assertTrue(jobs[0]['issue_id'].startswith('CHALLENGE:'))
        self.assertFalse(any(n['kind'] == 'UNRESOLVED_CONFLICT' for n in jobs[0]['support_and_dependencies']))
        original = {n['id']: n for n in project_conflicts(source)['nodes']}
        self.assertTrue(all(c == original[c['id']] for c in jobs[0]['claims']))

    def proposal(self, source, job):
        candidate = deepcopy(next(c for c in source['cycles'][-1]['candidates'] if c['specialist'] == job['framework']))
        candidate['challenge_response'] = {'issue_id': job['issue_id'], 'disposition': 'RESOLVED',
                                           'answer': 'I reviewed the inferential classifications.'}
        return candidate

    def test_model_claiming_resolved_does_not_remove_native_objections(self):
        source = self.source(); job = review_jobs(source)[0]
        proposal = self.proposal(source, job)
        result = reconcile(source, [job], {job['framework']: proposal}, source['framework_proposition_ledgers'])
        self.assertTrue(all(i['objection_check'] == 'OPEN' for i in result['targeted_review']['issues']))
        self.assertTrue(all(i['semantic_resolution'] == 'NOT_INDEPENDENTLY_VERIFIED' for i in result['targeted_review']['issues']))
        self.assertEqual(result['action_source_grounding'], source['action_source_grounding'])

    def test_admitted_revision_is_attributed_and_other_frameworks_survive(self):
        source = self.source(); before = deepcopy(source); job = review_jobs(source)[0]
        proposal = self.proposal(source, job)
        for record in proposal['committed_native_ledger']['records']:
            record['calibration_errors'] = []
        result = reconcile(source, [job], {job['framework']: proposal}, source['framework_proposition_ledgers'])
        self.assertEqual(source, before)
        self.assertEqual(result['cycles'][0], source['cycles'][0])
        for candidate in source['cycles'][-1]['candidates']:
            if candidate['specialist'] != job['framework']:
                self.assertIn(candidate, result['cycles'][-1]['candidates'])
        self.assertTrue(all(i['objection_check'] == 'NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR' for i in result['targeted_review']['issues']))
        self.assertEqual(result['judgment_status'], 'NOT_ADJUDICATED')

    def test_failed_or_misaddressed_revision_cannot_replace_original(self):
        source = self.source(); job = review_jobs(source)[0]
        for failure in ('REJECTED', 'wrong_issue'):
            proposed = self.proposal(source, job)
            if failure == 'REJECTED':
                proposed['committed_native_ledger']['transaction_status'] = 'REJECTED'
            else:
                proposed['challenge_response']['issue_id'] = 'wrong_issue'
            result = reconcile(source, [job], {job['framework']: proposed}, {})
            self.assertEqual(result['cycles'][-1]['candidates'], source['cycles'][-1]['candidates'])
            self.assertTrue(all(i['review_status'] == 'REJECTED_OR_UNAVAILABLE' for i in result['targeted_review']['issues']))
            self.assertEqual(result['targeted_review']['proposal_attempts'][job['framework']], proposed)

    def test_native_review_delivers_only_local_packet_and_preserves_world(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        root = Path('/tmp/parliament-smoke-614af0c')
        if not native.exists() or not root.exists():
            self.skipTest('Pinned native Parliament unavailable')
        code = r'''
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from logic_puzzles_review import review
class Fake:
    def complete_json(self,prompt,**kwargs):
        assert 'TARGETED FRAMEWORK REVIEW' in prompt
        assert 'Localized review packet' in prompt
        assert 'Your assigned challenges:' in prompt
        assert 'Your assigned challenges: []' not in prompt
        assert 'qa' in kwargs['schema']['properties']
        assert 'INDEPENDENT FRAMEWORK PASS' not in prompt
        return {'choices':[{'text':'{}'}]}
path=Path(sys.argv[2]); before=path.read_bytes()
result=review(path,Path(sys.argv[3]),lambda name:Fake())
assert path.read_bytes()==before
assert result['targeted_review']['adapter_calls']=={'deontological':2}
assert len(result['targeted_review']['issues'])==2
assert all(i['review_status']=='REJECTED_OR_UNAVAILABLE' for i in result['targeted_review']['issues'])
assert result['cycles'][-1]['candidates']==result['cycles'][0]['candidates']
assert (Path(sys.argv[3])/'targeted_review.md').exists()
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(native), '-c', code, str(root), str(FIXTURE.resolve()), folder],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_display_closure_preserves_every_typed_semantic_field(self):
        data = {'dp': {'A0': {'v': 'CONFLICTED', 'n': 'respect persons', 'dt': 'PERFECT_NEGATIVE',
                              'rel': 'UNRESOLVED', 'rs': 'harm remains unresolved'}},
                'fm': {'A0': 'PROHIBITED/PERMISSIBLE'}, 'ep': [{'p': 'HYPOTHESIS', 'c': 'uncertain link'}]}
        response = {'choices': [{'text': json.dumps(data)}]}
        before = deepcopy(response)
        revised, provenance = close_deontological_map(response)
        self.assertEqual(response, before)
        after = json.loads(revised['choices'][0]['text'])
        self.assertEqual(after['dp'], data['dp'])
        self.assertEqual(after['ep'], data['ep'])
        self.assertTrue(after['fm']['A0'].startswith('CONFLICTED:'))
        self.assertEqual(provenance['changed_fields'], ['fm'])
        self.assertFalse(provenance['semantic_fields_changed'])

    def test_native_review_preserves_prior_after_conflicting_saved_revision(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        root = Path('/tmp/parliament-smoke-614af0c')
        if not native.exists() or not root.exists():
            self.skipTest('Pinned native Parliament unavailable')
        saved = Path('diagnostics/logic_puzzles_targeted_review_live/agenda_corrected/deontological/model_calls.json')
        code = r'''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from logic_puzzles_review import review
response=json.loads(Path(sys.argv[3]).read_text())[0]['response']
class Saved:
    model='o3-saved-response-replay'
    execution_mode='SAVED_RESPONSE_REPLAY'
    def complete_json(self,prompt,**kwargs):return response
source=json.loads(Path(sys.argv[2]).read_text())
result=review(Path(sys.argv[2]),Path(sys.argv[4]),lambda name:Saved(),allow_partial=False)
assert result['targeted_review']['execution_modes']=={'deontological':'SAVED_RESPONSE_REPLAY'}
assert all(i['review_status']=='RETAINED_AFTER_NATIVE_REJECTION' for i in result['targeted_review']['issues'])
assert all(i['objection_check']=='OPEN' for i in result['targeted_review']['issues'])
assert result['action_source_grounding']==source['action_source_grounding']
candidate=next(c for c in result['cycles'][-1]['candidates'] if c['specialist']=='deontological')
record=next(r for r in candidate['committed_native_ledger']['records'] if r['canonical_action_id']=='A0')
assert record['harm_relation']=='DOING_HARM'
assert record['means_relation']=='INTENDED_AS_MEANS'
assert result['targeted_review']['acceptance_path']=='NATIVE_RECURRENT_PROPOSAL_REVIEW'
native=json.loads((Path(sys.argv[4])/'deontological/native_trace.json').read_text())
assert len(native['cycles'])==2
assert native['cycles'][0]['candidates'][0]['committed_native_ledger']==candidate['committed_native_ledger']
assert json.loads((Path(sys.argv[4])/'deontological/prior_framework_state.json').read_text())['prior_framework_state']
assert candidate['committed_native_ledger']['transaction_status']=='COMMITTED_WITH_UNCERTAINTY'
assert candidate['framework_validation_errors']
for c in source['cycles'][0]['candidates']:
    if c['specialist']!='deontological':assert c in result['cycles'][-1]['candidates']
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(native), '-c', code, str(root), str(FIXTURE.resolve()),
                                     str(saved.resolve()), folder], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_review_accepts_saved_means_correction_and_exposes_other_changes(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        root = Path('/tmp/parliament-smoke-614af0c')
        if not native.exists() or not root.exists():
            self.skipTest('Pinned native Parliament unavailable')
        code = r'''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from logic_puzzles_review import review
from logic_puzzles_acceptance import SavedResponses
source=Path('diagnostics/logic_puzzles_matched_live/independent/framework_generation.json')
response=json.loads(Path('diagnostics/logic_puzzles_matched_live/targeted/deontological/model_calls.json').read_text())[0]['response']
before=json.loads(source.read_text())
result=review(source,Path(sys.argv[2]),lambda name:SavedResponses([response]),max_calls=1)
assert result['targeted_review']['adapter_calls']=={'deontological':1}
assert result['targeted_review']['prior_replay_calls']=={'deontological':1}
assert all(i['review_status']=='RECONCILED' for i in result['targeted_review']['issues'])
assert [i['objection_check'] for i in result['targeted_review']['issues']]==['OPEN','NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR']
native=json.loads((Path(sys.argv[2])/'deontological/native_trace.json').read_text())
prior=next(c for c in before['cycles'][-1]['candidates'] if c['specialist']=='deontological')
assert native['cycles'][0]['candidates'][0]['committed_native_ledger']==prior['committed_native_ledger']
assert result['action_source_grounding']==before['action_source_grounding']
assert any('competing_protected_party' in row for issue in result['targeted_review']['issues'] for row in issue['classification_changes'])
assert 'competing_protected_party' in (Path(sys.argv[2])/'targeted_review.md').read_text()
for c in before['cycles'][-1]['candidates']:
 if c['specialist']!='deontological':assert c in result['cycles'][-1]['candidates']
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(native), '-c', code, str(root), folder], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
