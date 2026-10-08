import json
from pathlib import Path
import tempfile
import unittest

from run_specific_inference_probe import CHALLENGE, prepare
from run_objection_response_pilot import decode


class SpecificInferenceProbeTests(unittest.TestCase):
    def test_freezes_exact_previous_reply_and_one_call_allowance(self):
        source = Path('diagnostics/objection_response_pilot_live_20261007')
        if not source.exists(): self.skipTest('Saved response unavailable')
        path = source/'competing_promise/objection_focused/utilitarian/revision_call.json'
        before = path.read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            manifest, packet, quotes = prepare(source, Path(folder))
            raw = decode(json.loads(before)['response'])
            self.assertEqual(packet['initial_argument'], raw['native_submission'])
            self.assertEqual(packet['previous_argument_response'], raw['argument_response'])
            self.assertEqual(manifest['maximum_provider_adapter_calls'], 1)
            self.assertEqual(manifest['max_tokens'], 4096)
            self.assertEqual(manifest['model'], 'o3')
            self.assertIn('utilitarian', quotes)
            self.assertEqual(path.read_bytes(), before)

    def test_challenge_does_not_require_reversal_or_fabricated_support(self):
        self.assertIn('need not reverse', CHALLENGE)
        self.assertIn('unsupported empirical generalization', CHALLENGE)
        self.assertIn('total-welfare ranking remains unresolved', CHALLENGE)
        self.assertIn('life-years', CHALLENGE)


if __name__ == '__main__': unittest.main()
