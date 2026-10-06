from copy import deepcopy
import json
from pathlib import Path
import unittest
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


if __name__=='__main__':unittest.main()
