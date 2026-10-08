import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from run_argument_quality_pilot import CHECKLIST, ChecklistBackend, transform


class ArgumentQualityPilotTests(unittest.TestCase):
    def test_treatment_changes_prompt_only(self):
        class Backend:
            model = 'test'
            def complete_json(self, prompt, **kwargs):
                self.prompt, self.kwargs = prompt, kwargs
                return {'unchanged': True}
        backend = Backend()
        wrapper = ChecklistBackend(backend)
        schema = {'type': 'object'}
        self.assertEqual(wrapper.complete_json('native', schema=schema, max_tokens=3000), {'unchanged': True})
        self.assertEqual(backend.kwargs, {'schema': schema, 'max_tokens': 3000})
        self.assertEqual(backend.prompt, 'native\n'+CHECKLIST)

    def test_swap_preserves_identifiers_and_does_not_mutate_source(self):
        source = {'A0': ['95%', '5%'], 'A1': 1}
        changed = transform(source, lambda s: s.replace('95%', 'x').replace('5%', '95%').replace('x', '5%'))
        self.assertEqual(changed, {'A0': ['5%', '95%'], 'A1': 1})
        self.assertEqual(source['A0'], ['95%', '5%'])

    def test_native_freezes_counterfactuals_without_changing_medical_topology(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        if not native.exists(): self.skipTest('Native interpreter unavailable')
        with tempfile.TemporaryDirectory() as folder:
            run = subprocess.run([str(native), 'run_argument_quality_pilot.py', '--output-dir', folder], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            manifest = json.loads((Path(folder)/'manifest.json').read_text())
            baseline = json.loads((Path(folder)/'baseline/frozen_world_trace.json').read_text())
            promise = json.loads((Path(folder)/'competing_promise/frozen_world_trace.json').read_text())
            self.assertEqual(baseline['action_source_grounding'], promise['action_source_grounding'])
            self.assertIn('promised Cara', promise['scenario'])
            for case in manifest['cases'].values():
                self.assertIn(case['admission'], ('COMMITTED', 'COMMITTED_WITH_UNCERTAINTY'))
                self.assertTrue(all(not s.startswith('If ') for s in case['action_labels']))
            swapped = manifest['cases']['probability_swap']['expected_survivor_endpoint']
            self.assertEqual(swapped, {'A0': .05, 'A1': .95})


if __name__ == '__main__': unittest.main()
