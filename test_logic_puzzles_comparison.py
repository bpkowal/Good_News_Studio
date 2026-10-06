import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from logic_puzzles_comparison import OpeningThenLive, metrics


class ComparisonTests(unittest.TestCase):
    def test_saved_opening_is_immutable_and_followup_is_distinct(self):
        class Backend:
            model = 'test'
            def complete_json(self, prompt, **kwargs): return {'followup': True}
        opening = {'opening': []}
        model = OpeningThenLive(opening, Backend())
        model.complete_json('first')['opening'].append('mutation')
        self.assertEqual(opening, {'opening': []})
        self.assertEqual(model.last_call_origin, 'SAVED_OPENING_RESPONSE')
        self.assertEqual(model.complete_json('second'), {'followup': True})
        self.assertEqual(model.last_call_origin, 'LIVE_FOLLOWUP')

    def test_missing_ledgers_remain_visible_even_without_errors(self):
        trace = json.loads(Path('diagnostics/logic_puzzles_independent_live/framework_generation.json').read_text())
        trace['cycles'][-1]['candidates'] = []
        result = metrics(trace)
        self.assertEqual(result['committed_frameworks'], 0)
        self.assertEqual(len(result['missing_frameworks']), 5)
        self.assertEqual(result['semantic_support'], 'NOT_INDEPENDENTLY_ASSESSED')

    def test_native_offline_comparison_preserves_controls_and_failures(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        root = Path('/tmp/parliament-smoke-614af0c')
        if not native.exists() or not root.exists(): self.skipTest('Native checkout unavailable')
        code = '''
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from logic_puzzles_comparison import compare
class Fake:
 model='o3'
 def complete_json(self,prompt,**kwargs):return {}
out=Path(sys.argv[2])/'comparison'
result=compare(Path('diagnostics/logic_puzzles_independent_live'),out,lambda name:Fake())
assert all(a['world_unchanged'] for a in result['arms'].values())
assert result['arms']['independent']['saved_opening_calls']==5
assert result['arms']['legacy']['saved_opening_calls']==5
assert result['arms']['legacy']['logical_calls_including_openings']<=10
assert result['arms']['targeted']['followup_attempts']<=2
assert result['controls']['opening_models']==['o3']
for arm in result['arms']:assert (out/arm/'conflict_graph.md').exists()
for name in result['controls']['opening_response_sha256']:
 a=json.loads((out/'independent'/name/'model_calls.json').read_text())[0]['response']
 b=json.loads((out/'legacy'/name/'model_calls.json').read_text())[0]['response']
 assert a==b
calls=json.loads((out/'legacy/deontological/model_calls.json').read_text())
assert calls[0]['normalization']['semantic_fields_changed']==False
assert len(calls)==2
assert 'NATIVE PARLIAMENT PASS' in calls[1]['prompt']
assert result['arms']['targeted']['calibration_errors']==3
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(native), '-c', code, str(root), folder], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
