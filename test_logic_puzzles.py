from copy import deepcopy
import json
from pathlib import Path
import unittest
import subprocess
import tempfile
from run_logic_puzzles import project_conflicts


class ConflictProjectionTests(unittest.TestCase):
    def trace(self):
        return json.loads(Path('diagnostics/promise_specialist_handoff/live/workspace_scenario_20261004_234129.json').read_text())

    def test_projection_preserves_world_full_records_and_uncertainty(self):
        trace = self.trace(); before = deepcopy(trace)
        graph = project_conflicts(trace)
        self.assertEqual(trace, before)
        self.assertEqual(graph, project_conflicts(trace))
        claims = [n for n in graph['nodes'] if n['kind']=='CLAIM']
        native = [r for c in trace['cycles'][-1]['candidates'] for r in c['committed_native_ledger']['records']]
        self.assertEqual([n['record'] for n in claims], native)
        self.assertLessEqual(len(graph['review_queue']), 2)
        self.assertTrue(any(n.get('conflict_type')=='UNSUPPORTED_INFERENCE' for n in graph['nodes']))
        ids = {n['id'] for n in graph['nodes']}
        self.assertTrue(all(e['source'] in ids and e['target'] in ids for e in graph['edges']))
        self.assertFalse(graph['world_mutation'])

    def test_rejected_records_and_missing_evidence_never_gain_support(self):
        trace = self.trace()
        candidate = trace['cycles'][-1]['candidates'][0]
        candidate['committed_native_ledger']['transaction_status']='REJECTED'
        graph = project_conflicts(trace)
        self.assertFalse(any(n.get('origin')==candidate['specialist'] for n in graph['nodes']))
        self.assertTrue(any(d.get('status')=='NO_COMMITTED_LEDGER' for d in graph['diagnostics']))
        trace = self.trace()
        trace['cycles'][-1]['candidates'][0]['committed_native_ledger']['records'][0]['grounded_effect_ids']=['NONEXISTENT']
        graph=project_conflicts(trace)
        self.assertFalse(any(e['source']=='WORLD:NONEXISTENT' for e in graph['edges']))
        self.assertTrue(any(d.get('unresolved_effect_id')=='NONEXISTENT' for d in graph['diagnostics']))

    def test_preferences_do_not_fabricate_conflicts(self):
        trace=self.trace()
        for candidate in trace['cycles'][-1]['candidates']:
            candidate['framework_internal_conflicts']=[]
            for record in candidate['committed_native_ledger']['records']:
                record['calibration_errors']=[]
        graph=project_conflicts(trace)
        self.assertFalse(any(n['kind']=='UNRESOLVED_CONFLICT' for n in graph['nodes']))

    def test_isolated_proposition_ids_do_not_merge_framework_hypotheses(self):
        trace = self.trace()
        names = [c['specialist'] for c in trace['cycles'][-1]['candidates']]
        trace['framework_proposition_ledgers'] = {
            name: [{'proposition_id': 'LOCAL_P1', 'claim': name + ' hypothesis',
                    'epistemic_status': 'HYPOTHESIZED'}] for name in names}
        for candidate in trace['cycles'][-1]['candidates']:
            candidate['decision_critical_proposition_ids'] = ['LOCAL_P1']
        graph = project_conflicts(trace)
        by_id = {n['id']: n for n in graph['nodes']}
        for edge in graph['edges']:
            if edge['kind'] == 'DEPENDENCY':
                origin = by_id[edge['source']]['origin']
                self.assertEqual(edge['target'], origin + '::LOCAL_P1')
                self.assertEqual(by_id[edge['target']]['epistemic_status'], 'HYPOTHESIZED')

    def test_native_independent_pass_has_identical_inputs_and_no_peer_state(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        root = Path('/tmp/parliament-smoke-614af0c')
        if not native.exists() or not root.exists():
            self.skipTest('Pinned native Parliament checkout is unavailable')
        code = r'''
import json, sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from logic_puzzles_frameworks import generate, FRAMEWORKS, fingerprint
from global_workspace.local_specialists import CompactLocalSpecialist
observed=[]
original=CompactLocalSpecialist.evaluate
def capture(self,scenario,actions,broadcast):
    observed.append({'name':self.name,'world':deepcopy(self.scenario_graph.to_dict()),
                     'ledger':deepcopy(self.proposition_ledger),
                     'advisory':deepcopy(self.source_construction_advisory),
                     'broadcast':asdict(broadcast),
                     'prior':deepcopy(self.previous_framework_state)})
    return original(self,scenario,actions,broadcast)
CompactLocalSpecialist.evaluate=capture
class Fake:
    def complete_json(self,prompt,**kwargs):return {'choices':[{'text':'{}'}]}
source=Path(sys.argv[2]);before=source.read_bytes()
result=generate(source,Path(sys.argv[3]),FRAMEWORKS,lambda name:Fake())
assert source.read_bytes()==before
assert len(observed)==5
assert len({fingerprint(o['world']) for o in observed})==1
assert len({fingerprint(o['ledger']) for o in observed})==1
assert len({fingerprint(o['advisory']) for o in observed})==1
for o in observed:
    b=o['broadcast']
    assert not b['salient_specialist'] and not b['salient_action']
    assert not b['challenge_agenda'] and not o['prior']
    assert not b['problem_state'].get('agent_positions')
assert len(result['framework_runs'])==5
assert all(r['adapter_calls']<=2 for r in result['framework_runs'])
assert all(r['candidate_statuses']==['SEMANTIC_VALIDATION_ERROR'] for r in result['framework_runs']),result['framework_runs']
assert not result['world_mutation'] and result['judgment_status']=='NOT_ADJUDICATED'
assert not any(n['kind']=='CLAIM' for n in json.loads((Path(sys.argv[3])/'conflict_graph.json').read_text())['nodes'])
for name in FRAMEWORKS:
    calls=json.loads((Path(sys.argv[3])/name/'model_calls.json').read_text())
    assert calls and all(json.dumps(result['scenario']) in c['prompt'] for c in calls)
print('Five independent native workspaces verified; invalid fake responses remain diagnostic.')
'''
        with tempfile.TemporaryDirectory() as folder:
            proc = subprocess.run([str(native), '-c', code, str(root),
                                   str(Path('diagnostics/promise_specialist_handoff/live/workspace_scenario_20261004_234129.json').resolve()),
                                   folder], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__=='__main__':unittest.main()
