import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import subprocess

from run_objection_response_pilot import RevisionBackend, SavedNativeBackend, decode, prepare, summarize, argument_schema


class ObjectionResponseTests(unittest.TestCase):
    def test_envelope_preserves_native_submission_and_bounds_calls(self):
        class Backend:
            model = 'test'
            def complete_json(self, prompt, **kwargs):
                self.prompt, self.kwargs = prompt, kwargs
                return {'argument_response': {'disposition': 'partly_answered'}, 'native_submission': {'r': 'A0'}}
        packet = {'initial_argument': {'r': 'A1'}, 'objection': 'competing need'}
        native = {'type': 'object', 'properties': {'r': {'type': 'string'}}}
        original = deepcopy(native)
        with tempfile.TemporaryDirectory() as folder:
            backend = Backend()
            wrapper = RevisionBackend(backend, packet, 'objection_focused', folder)
            result = wrapper.complete_json('native prompt', schema=native, max_tokens=4096)
            self.assertEqual(decode(result), {'r': 'A0'})
            self.assertEqual(native, original)
            self.assertEqual(backend.kwargs['schema']['properties']['native_submission'], native)
            self.assertEqual(backend.kwargs['max_tokens'], 4096)
            with self.assertRaises(RuntimeError): wrapper.complete_json('again', schema=native, max_tokens=4096)
            saved = json.loads((Path(folder)/'argument_response.json').read_text())
            self.assertEqual(saved['authority'], 'ATTRIBUTED_NOT_WORLD_STATE')
            self.assertEqual(saved['assessment'], 'REQUIRES_CONTENT_REVIEW')
            self.assertEqual(packet['initial_argument']['r'], 'A1')

    def test_both_arms_receive_same_packet_schema_and_budget(self):
        class Backend:
            model = 'test'
            def complete_json(self, prompt, **kwargs):
                self.prompt, self.kwargs = prompt, kwargs
                return {'argument_response': {}, 'native_submission': {}}
        with tempfile.TemporaryDirectory() as folder:
            backends = [Backend(), Backend()]
            packet = {'objection': 'exact same objection', 'initial_argument': {'dp': 'original'}}
            for arm, backend in zip(('ordinary_revision', 'objection_focused'), backends):
                RevisionBackend(backend, packet, arm, Path(folder)/arm).complete_json('same native', schema={'type': 'object'}, max_tokens=4096)
            self.assertEqual(backends[0].kwargs, backends[1].kwargs)
            for b in backends:
                self.assertIn(json.dumps(packet), b.prompt)
            self.assertNotEqual(backends[0].prompt, backends[1].prompt)

    def test_failed_response_retained_without_argument_or_world_authority(self):
        class Backend:
            model = 'test'
            def complete_json(self, prompt, **kwargs): return {'bad': 'response'}
        with tempfile.TemporaryDirectory() as folder:
            wrapper = RevisionBackend(Backend(), {}, 'ordinary_revision', folder)
            with self.assertRaises(ValueError): wrapper.complete_json('p', schema={}, max_tokens=100)
            call = json.loads((Path(folder)/'revision_call.json').read_text())
            self.assertEqual(call['response'], {'bad': 'response'})
            self.assertFalse((Path(folder)/'argument_response.json').exists())

    def test_preparation_freezes_complete_openings_without_source_mutation(self):
        source = Path('diagnostics/argument_quality_pilot_fixes_20261006')
        if not source.exists(): self.skipTest('Frozen opening unavailable')
        original = (source/'competing_promise/frozen_world_trace.json').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            manifest = prepare(source, Path(folder))
            self.assertEqual(manifest['total_revision_calls'], 12)
            self.assertEqual(set(manifest['cases']), {'competing_promise', 'probability_swap'})
            packet = json.loads((Path(folder)/'competing_promise/care_revision_input.json').read_text())
            self.assertIn('priority', packet['objection'])
            self.assertIn('cl', packet['initial_argument'])
            summarize(folder, manifest)
            self.assertEqual(json.loads((Path(folder)/'results.json').read_text()), [])
        self.assertEqual((source/'competing_promise/frozen_world_trace.json').read_bytes(), original)

    def test_author_labels_do_not_assert_resolution(self):
        self.assertEqual(argument_schema()['properties']['disposition']['enum'], ['answered', 'partly_answered', 'unanswered'])

    def test_replay_preserves_native_fields_without_provider_call(self):
        raw = {'r': 'A1', 'ss': 'PROVISIONAL', 'dp': None}
        backend = SavedNativeBackend({'model': 'o3', 'response': {'argument_response': {}, 'native_submission': raw}})
        self.assertEqual(decode(backend.complete_json('ignored', schema={}, max_tokens=4096)), raw)
        self.assertEqual(backend.last_call_origin, 'SAVED_NATIVE_SUBMISSION_NO_API')
        with self.assertRaises(RuntimeError): backend.complete_json('again')

    def test_saved_native_submission_reaches_native_assessment_without_transport_failure(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        if not native.exists(): self.skipTest('Native interpreter unavailable')
        code = '''import sys, json
from pathlib import Path
sys.path.insert(0, '/tmp/parliament-smoke-614af0c')
from logic_puzzles_frameworks import generate
from run_objection_response_pilot import SavedNativeBackend
source = Path('diagnostics/argument_quality_pilot_fixes_20261006/competing_promise')
call = json.loads((source/'standard/care/model_calls.json').read_text())[0]
from run_objection_response_pilot import decode
saved = {'model': call['model'], 'response': {'argument_response': {}, 'native_submission': decode(call['response'])}}
quotes = json.loads((source/'standard/framework_commitments.json').read_text())
result = generate(source/'frozen_world_trace.json', Path(sys.argv[1]), ['care'], lambda _: SavedNativeBackend(saved), core_quotes=quotes, call_budget=1)
assert result['world_mutation'] is False
assert result['cycles'][0]['candidates'][0]['delegate_status'] != 'SPECIALIST_INTERNAL_ERROR'
assert json.loads((Path(sys.argv[1])/'care/model_calls.json').read_text())[0]['response_origin'] == 'SAVED_NATIVE_SUBMISSION_NO_API'
'''
        with tempfile.TemporaryDirectory() as folder:
            run = subprocess.run([str(native), '-c', code, folder], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)


if __name__ == '__main__': unittest.main()
