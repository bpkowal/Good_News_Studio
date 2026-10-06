from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from logic_puzzles_partial_review import isolate_classifications
from logic_puzzles_review import review_jobs


class PartialRevisionTests(unittest.TestCase):
    def fixtures(self):
        root = Path('diagnostics/logic_puzzles_independent_live')
        assessment = json.loads((root/'framework_generation.json').read_text())
        prior = json.loads((root/'deontological/model_calls.json').read_text())[0]['response']
        proposed = json.loads(Path('diagnostics/logic_puzzles_targeted_review_live/agenda_corrected/deontological/model_calls.json').read_text())[0]['response']
        return assessment, prior, proposed, review_jobs(assessment)[0]

    def test_isolation_retains_all_prior_substantive_fields_except_addressed_means(self):
        assessment, prior, proposed, job = self.fixtures()
        proposed_text = json.loads(proposed['choices'][0]['text'])
        proposed_text['dp']['A0']['res'] = 'RESOLVED'
        proposed['choices'][0]['text'] = json.dumps(proposed_text)
        before = deepcopy((prior, proposed, job))
        isolated, patches = isolate_classifications(prior, proposed, job)
        self.assertEqual((prior, proposed, job), before)
        self.assertEqual([(p['action_id'],p['field']) for p in patches], [('A0','mr')])
        old = json.loads(prior['choices'][0]['text'])
        new = json.loads(isolated['choices'][0]['text'])
        new['dp']['A0']['mr'] = old['dp']['A0']['mr']
        for key in ('qa','j'):
            new.pop(key,None); old.pop(key,None)
        self.assertEqual(new,old)

    def test_unrelated_or_invalid_proposals_are_not_guessed(self):
        assessment, prior, proposed, job = self.fixtures()
        job['framework'] = 'care'
        self.assertEqual(isolate_classifications(prior, proposed, job), (None,[]))
        job['framework'] = 'deontological'
        self.assertEqual(isolate_classifications(prior, {}, job), (None,[]))
        self.assertEqual(isolate_classifications(prior, prior, job), (None,[]))

    def test_harm_uncertainty_preserves_committed_not_raw_duty_fields(self):
        _, prior, proposed, job = self.fixtures()
        isolated, patches = isolate_classifications(prior, proposed, job)
        self.assertEqual({p['field'] for p in patches}, {'hr', 'mr'})
        row = json.loads(isolated['choices'][0]['text'])['dp']['A0']
        self.assertEqual(row['v'], 'CONFLICTED')
        self.assertEqual(row['gv'], 'UNRESOLVED')
        self.assertEqual(row['res'], 'CONTESTED')
        self.assertEqual(row['hr'], 'UNRESOLVED')
        self.assertEqual(patches[0]['prior_native_rendering']['v'],
                         {'response_value': 'PROHIBITED', 'committed_value': 'CONFLICTED'})
        incomplete = deepcopy(job)
        for claim in incomplete['claims']:
            claim['record'].pop('governing_norm', None)
        self.assertEqual(isolate_classifications(prior, proposed, incomplete), (None, []))

    def test_harm_requires_explicit_compatible_uncertainty(self):
        _, prior, proposed, job = self.fixtures()
        for harm, resolution in [('ALLOWING_HARM', 'CONTESTED'), ('UNRESOLVED', 'RESOLVED'),
                                  ('UNRESOLVED', None)]:
            altered = deepcopy(proposed)
            payload = json.loads(altered['choices'][0]['text'])
            payload['dp']['A0'].update(hr=harm, res=resolution)
            altered['choices'][0]['text'] = json.dumps(payload)
            _, patches = isolate_classifications(prior, altered, job)
            self.assertEqual([p['field'] for p in patches], ['mr'])

    def test_explicit_unknown_resolution_is_a_companion_not_a_third_issue(self):
        _, prior, proposed, job = self.fixtures()
        payload = json.loads(proposed['choices'][0]['text'])
        payload['dp']['A0']['res'] = 'UNKNOWN'
        proposed['choices'][0]['text'] = json.dumps(payload)
        isolated, patches = isolate_classifications(prior, proposed, job)
        self.assertEqual({p['field'] for p in patches}, {'hr', 'res', 'mr'})
        self.assertEqual(len({p['issue_id'] for p in patches}), 2)
        companion = next(p for p in patches if p['field'] == 'res')
        self.assertEqual(companion['requires_field'], 'hr')
        self.assertEqual(json.loads(isolated['choices'][0]['text'])['dp']['A0']['res'], 'UNKNOWN')

    def test_native_partial_acceptance_preserves_duties_and_disputed_bundle(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        if not native.exists(): self.skipTest('Native interpreter unavailable')
        code = r'''
import sys,json
from pathlib import Path
sys.path.insert(0,'/tmp/parliament-smoke-614af0c')
from logic_puzzles_review import review
from logic_puzzles_acceptance import SavedResponses
from run_logic_puzzles import project_conflicts
source=Path('diagnostics/logic_puzzles_independent_live/framework_generation.json')
before=json.loads(source.read_text())
saved=json.loads(Path('diagnostics/logic_puzzles_targeted_review_live/agenda_corrected/deontological/model_calls.json').read_text())[0]['response']
payload=json.loads(saved['choices'][0]['text'])
payload['dp']['A0']['res']='RESOLVED'
saved['choices'][0]['text']=json.dumps(payload)
result=review(source,Path(sys.argv[1]),lambda name:SavedResponses([saved]),max_calls=1)
reviewed=result['targeted_review']
assert reviewed['adapter_calls']=={'deontological':1}
assert len(reviewed['retained_proposals'])==1
assert reviewed['retained_proposals'][0]['status']=='DISPUTED_NONOPERATIVE'
assert reviewed['isolated_classification_reviews'][0]['native_accepted']
assert reviewed['isolated_classification_reviews'][0]['new_api_calls']==0
assert not reviewed['isolated_classification_reviews'][0]['full_bundle_accepted']
old=next(c for c in before['cycles'][-1]['candidates'] if c['specialist']=='deontological')
new=next(c for c in result['cycles'][-1]['candidates'] if c['specialist']=='deontological')
for oldrow,newrow in zip(old['committed_native_ledger']['records'],new['committed_native_ledger']['records']):
 for field in ('verdict','relation','norm','competing_norm','protected_party','competing_protected_party','governing_norm','priority_basis'):
  assert oldrow[field]==newrow[field],(field,oldrow[field],newrow[field])
assert new['committed_native_ledger']['records'][0]['means_relation']=='FORESEEN_SIDE_EFFECT'
assert new['committed_native_ledger']['records'][0]['harm_relation']=='DOING_HARM'
assert reviewed['issues'][0]['objection_check']=='OPEN'
assert reviewed['issues'][1]['objection_check']=='NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR'
assert reviewed['issues'][0]['review_status']=='ORIGINAL_CLASSIFICATION_RETAINED'
assert reviewed['issues'][1]['review_status']=='ISOLATED_CLASSIFICATION_ACCEPTED'
assert result['action_source_grounding']==before['action_source_grounding']
for c in before['cycles'][-1]['candidates']:
 if c['specialist']!='deontological':assert c in result['cycles'][-1]['candidates']
graph=project_conflicts(result)
assert any(n['kind']=='PROPOSED_REVISION' and n['world_state_authority']=='NONE' for n in graph['nodes'])
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(native), '-c', code, folder], capture_output=True, text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


    def test_native_harm_uncertainty_does_not_resurrect_original_prohibition(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        if not native.exists(): self.skipTest('Native interpreter unavailable')
        code = r'''
import sys,json
from pathlib import Path
sys.path.insert(0,'/tmp/parliament-smoke-614af0c')
from logic_puzzles_review import review
from logic_puzzles_acceptance import SavedResponses
source=Path('diagnostics/logic_puzzles_independent_live/framework_generation.json')
before=json.loads(source.read_text())
saved=json.loads(Path('diagnostics/logic_puzzles_targeted_review_live/agenda_corrected/deontological/model_calls.json').read_text())[0]['response']
result=review(source,Path(sys.argv[1]),lambda name:SavedResponses([saved]),max_calls=1)
audit=result['targeted_review']
partial=audit['isolated_classification_reviews'][0]
assert partial['native_accepted'] and partial['new_api_calls']==0
assert not partial['full_bundle_accepted']
assert {p['field'] for p in partial['patches']}=={'hr','mr'}
assert audit['retained_proposals'][0]['status']=='DISPUTED_NONOPERATIVE'
old=next(c for c in before['cycles'][-1]['candidates'] if c['specialist']=='deontological')
new=next(c for c in result['cycles'][-1]['candidates'] if c['specialist']=='deontological')
assert not new['framework_validation_errors']
for a,b in zip(old['committed_native_ledger']['records'],new['committed_native_ledger']['records']):
 for field in ('verdict','relation','norm','competing_norm','protected_party','competing_protected_party','governing_norm','priority_basis','resolution_status'):
  assert a[field]==b[field],(field,a[field],b[field])
row=new['committed_native_ledger']['records'][0]
assert row['verdict']=='CONFLICTED' and row['governing_norm']=='UNRESOLVED'
assert row['harm_relation']=='UNRESOLVED' and row['resolution_status']=='CONTESTED'
assert row['means_relation']=='FORESEEN_SIDE_EFFECT'
for issue in audit['issues']:
 assert issue['review_status']=='ISOLATED_CLASSIFICATION_ACCEPTED'
 assert issue['semantic_resolution']=='NOT_INDEPENDENTLY_VERIFIED'
 assert any(r.get('harm_relation')=='UNRESOLVED' for r in issue['remaining_unresolved_fields'])
assert result['action_source_grounding']==before['action_source_grounding']
for c in before['cycles'][-1]['candidates']:
 if c['specialist']!='deontological':assert c in result['cycles'][-1]['candidates']
assert json.loads(source.read_text())==before
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(native), '-c', code, folder], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__': unittest.main()
