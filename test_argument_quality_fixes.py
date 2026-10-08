from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from parliament_recognition import survivor_count_projection, supported_records
from logic_puzzles_partial_review import isolate_classifications
from logic_puzzles_review import reconcile, review_jobs, residual_issue_ids
from run_logic_puzzles import project_conflicts


class ArgumentQualityFixes(unittest.TestCase):
    def test_residual_round_serves_unreviewed_bundle_and_preserves_cap(self):
        graph = {'nodes': [{'id': f'{name}{i}', 'kind': 'UNRESOLVED_CONFLICT', 'status': 'OPEN', 'origin': name}
                 for name, count in [('care', 3), ('utilitarian', 1), ('rawlsian', 3)] for i in range(count)]}
        self.assertEqual(residual_issue_ids(graph, {'care'}), ['rawlsian0', 'rawlsian1'])
        self.assertEqual(residual_issue_ids(graph, {'care', 'rawlsian', 'utilitarian'}), [])
    def test_survivor_endpoint_does_not_create_welfare_or_group_counts(self):
        attrs = {'party_id': 'P3', 'party_kind': 'PERSON', 'party_label': 'Ben',
                 'modality': 'PROBABILISTIC', 'likelihood_qualifiers': ['95% chance']}
        count = survivor_count_projection(attrs, 'Ben has a 95% chance of survival')
        self.assertEqual(count['amount'], 1)
        self.assertEqual(count['welfare_magnitude'], 'UNKNOWN')
        for changed, text in [({'party_kind': 'POPULATION'}, 'Ben has a 95% chance of survival'),
                              ({'party_label': 'workers'}, 'workers have a 95% chance of survival'),
                              ({}, 'Ben may survive'),
                              ({}, 'Ben does not have a 95% chance of survival')]:
            self.assertEqual(survivor_count_projection({**attrs, **changed}, text), {})

    def test_promise_occurrence_is_exact_scoped_source_not_promised_performance(self):
        frozen = json.loads(Path('fixtures/recognition_gap_medicine_frozen.json').read_text())
        world, source = frozen['action_source_grounding']['world_model'], frozen['scenario']
        before = deepcopy(world)
        sentence = 'Ada promised Cara that she would give Cara the medicine.'
        cards = supported_records(world, source+' '+sentence)
        promise = next(c for c in cards if c['kind'] == 'COMMITMENT_OCCURRENCE')
        self.assertEqual(promise['content_occurrence'], 'NOT_ESTABLISHED')
        self.assertEqual(promise['normative_force'], 'NOT_ASSESSED')
        span = promise['source_evidence'][0]
        self.assertEqual((source+' '+sentence)[span['start']:span['end']], sentence)
        self.assertEqual(world, before)
        for altered in ['If Ada promised Cara that she would give Cara the medicine.',
                        'Ada did not promise Cara that she would give Cara the medicine.',
                        'Ben says Ada promised Cara that she would give Cara the medicine.',
                        '"Ada promised Cara that she would give Cara the medicine."']:
            self.assertFalse(any(c['kind'] == 'COMMITMENT_OCCURRENCE' for c in supported_records(world, source+' '+altered)))

    def test_care_status_only_and_competence_restatement_are_not_isolated_resolutions(self):
        prior = {'cl': {'A0': {'res': 'CONTESTED', 'rs': 'competence', 'cc': 'Cara needs help'}}, 'nr': 'DECISIVE', 'np': 'counts'}
        job = {'framework': 'care', 'issue_id': 'job', 'issues': [{'id': 'q', 'label': 'A0 competing Care claim remains contested', 'claim_ids': ['c']}],
               'claims': [{'id': 'c', 'record': {'canonical_action_id': 'A0'}}]}
        wrap = lambda x: {'choices': [{'text': json.dumps(x)}]}
        proposed = deepcopy(prior); proposed['cl']['A0']['res'] = 'RESOLVED'
        proposed['qa'] = {'answer': 'Competence is decisive.'}
        self.assertEqual(isolate_classifications(wrap(prior), wrap(proposed), job), (None, []))
        numerical = deepcopy(job); numerical['issues'][0]['label'] = 'decisive counts lack relational comparability'
        proposed['nr'] = 'SECONDARY'; proposed['np'] = 'Numbers inform feasibility after responsibility is considered.'
        isolated, patches = isolate_classifications(wrap(prior), wrap(proposed), numerical)
        self.assertEqual({p['field'] for p in patches}, {'nr', 'np'})
        self.assertEqual(json.loads(isolated['choices'][0]['text'])['cl'], prior['cl'])

    def test_warnings_cannot_retire_by_dropping_records_or_unchanged_ledger(self):
        record = {'assessment_node_id': 'c', 'canonical_action_id': 'A0', 'resolution_status': 'CONTESTED',
                  'reason': 'competence', 'competing_care_claim': 'Cara needs help'}
        source = {'cycles': [{'candidates': [{'specialist': 'care', 'schema_valid': True,
                  'framework_validation_errors': ['A0 competing Care claim remains contested'],
                  'committed_native_ledger': {'ledger_kind': 'CARE_RELATIONSHIP_LEDGER', 'transaction_status': 'COMMITTED_WITH_UNCERTAINTY', 'records': [record]}}]}],
                  'framework_proposition_ledgers': {'care': []}}
        source['action_source_grounding'] = json.loads(Path('fixtures/recognition_gap_medicine_frozen.json').read_text())['action_source_grounding']
        job = review_jobs(source)[0]
        for mode in ('unchanged', 'drop', 'change', 'reasoned_change'):
            proposed = deepcopy(source['cycles'][0]['candidates'][0])
            proposed.update(framework_validation_errors=[], challenge_response={'issue_id': job['issue_id'], 'disposition': 'RESOLVED'})
            if mode == 'drop': proposed['committed_native_ledger']['records'] = []
            if mode in ('change', 'reasoned_change'): proposed['committed_native_ledger']['records'][0]['resolution_status'] = 'RESOLVED'
            if mode == 'reasoned_change':
                proposed['committed_native_ledger']['records'][0]['reason'] = 'explicit authored response to competing claim'
                proposed['challenge_response']['answer'] = 'Cara needs help. Here is an explicit attributed explanation addressing that competing claim while preserving the source evidence and its limits.'
            result = reconcile(source, [job], {'care': proposed}, {'care': []})
            self.assertEqual(result['targeted_review']['issues'][0]['objection_check'],
                             'NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR' if mode == 'reasoned_change' else 'OPEN')
        graph = project_conflicts(source)
        self.assertEqual(graph['open_conflicts'], graph['review_queue'])

    def test_native_saved_responses_verify_arithmetic_promise_and_bounded_repair(self):
        code = r'''
import sys,json
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,'/tmp/parliament-smoke-614af0c')
from logic_puzzles_frameworks import generate
from global_workspace.scenario_semantics import compile_scenario_graph
from global_workspace.epistemic_ledger import seed_proposition_ledger
from global_workspace.expected_value import validate_expected_value_estimates
root=Path('diagnostics/argument_quality_pilot_live_20261006')
source=Path('fixtures/recognition_gap_medicine_frozen.json')
frozen=json.loads(source.read_text())
scenario=frozen['scenario']+' Ada promised Cara that she would give Cara the medicine.'
graph=compile_scenario_graph(scenario, frozen['actions'], world_model=frozen['action_source_grounding']['world_model'])
ledger=seed_proposition_ledger(graph)
assert any(r.claim=='Ada promised Cara that she would give Cara the medicine.' and r.epistemic_status=='STIPULATED' for r in ledger.values())
class Saved:
 model='saved-test'
 def __init__(self,rows):self.rows=rows;self.prompts=[]
 def complete_json(self,prompt,**kwargs):self.prompts.append(prompt);return deepcopy(self.rows[min(len(self.prompts)-1,len(self.rows)-1)])
u=json.loads((root/'baseline/standard/utilitarian/model_calls.json').read_text())[0]['response']
backend=Saved([u])
run=generate(source,Path(sys.argv[1])/'u',['utilitarian'],lambda _:backend,core_quotes={},call_budget=1)
c=run['cycles'][0]['candidates'][0]
assert not c['expected_value_validation_errors'],c['expected_value_validation_errors']
assert all(v['validation_status']=='ARITHMETIC_VERIFIED' for v in c['expected_value_estimates'].values())
d=json.loads((root/'baseline/argument_checklist/deontological/model_calls.json').read_text())[0]['response']
data=json.loads(d['choices'][0]['text'])
for row in data['dp'].values():row['pb']='PERFECT_DUTY'
bad=deepcopy(d);bad['choices'][0]['text']=json.dumps(data)
backend=Saved([bad,d])
run=generate(source,Path(sys.argv[1])/'d',['deontological'],lambda _:backend,core_quotes={},call_budget=2)
assert len(backend.prompts)==2
assert 'DUTY_CONSISTENCY_REPAIR' in backend.prompts[1]
assert all(r['dt']=='IMPERFECT' for r in json.loads(d['choices'][0]['text'])['dp'].values())
c=run['cycles'][0]['candidates'][0]
assert c['deontological_ledger_proposal']
backend=Saved([bad,bad])
run=generate(source,Path(sys.argv[1])/'failed',['deontological'],lambda _:backend,core_quotes={},call_budget=2)
assert len(backend.prompts)==2
assert not run['cycles'][0]['candidates'][0]['deontological_ledger_proposal']
# Count annotations cannot license welfare units or duplicate patients.
record={'effect_id':'E2','action_id':'A0','direction':'BENEFIT','modality':'PROBABILISTIC','likelihood_qualifiers':['95% chance'],
 'count_projection':{'endpoint':'SURVIVOR_COUNT','basis':'ADMITTED_SINGLE_PERSON_IDENTITY','amount':1,'party_id':'P3'}}
estimate={'A0':{'value':.95,'unit':'UTILITY','direction':'BENEFIT','grounded':True,'method':'EXPECTED_COUNT','source_effect_ids':['E2'],'calculation':'1 * .95','assumptions':[]}}
assert validate_expected_value_estimates(estimate,action_id_by_key={'A0':'A0'},effect_records=[record]).errors
'''
        with tempfile.TemporaryDirectory() as folder:
            run = subprocess.run(['/tmp/parliament-smoke-env/bin/python', '-c', code, folder], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

    def test_native_residual_round_reaches_rawlsian_without_raising_first_cap(self):
        code = r'''
import sys,json
from pathlib import Path
sys.path.insert(0,'/tmp/parliament-smoke-614af0c')
from logic_puzzles_review import review
source=Path('diagnostics/occurrence_path_beam/medicine/logic_puzzles/independent/framework_generation.json')
before=source.read_bytes()
class Invalid:
 model='offline-test'
 def complete_json(self,prompt,**kwargs):return {'choices':[{'text':'{}'}]}
result=review(source,Path(sys.argv[1]),lambda _:Invalid(),allow_partial=False)
assert result['targeted_review']['rounds']==2
assert set(result['targeted_review']['adapter_calls'])=={'care','rawlsian'}
assert result['targeted_review']['coverage']['total_adapter_calls']==4
assert len(result['targeted_review']['issues'])==4
assert len(result['targeted_review']['coverage']['unattempted_issue_ids'])==2
assert all(i['objection_check']=='OPEN' for i in result['targeted_review']['issues'])
assert before==source.read_bytes()
assert result['action_source_grounding']==json.loads(before)['action_source_grounding']
'''
        with tempfile.TemporaryDirectory() as folder:
            run = subprocess.run(['/tmp/parliament-smoke-env/bin/python', '-c', code, folder], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)


if __name__ == '__main__': unittest.main()
