from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess

from inference_repair import make_contract, repair_schema, check_submission, annotated_saved_repair
from run_objection_response_pilot import decode, RevisionBackend


SOURCE = Path('diagnostics/specific_inference_probe_live_20261007')


def fixture():
    call = json.loads((SOURCE/'revision/utilitarian/revision_call.json').read_text())
    raw = decode(call['response'])
    frozen = json.loads((SOURCE/'frozen_world_trace.json').read_text())
    quotes = json.loads((SOURCE/'revision/framework_commitments.json').read_text())
    context = make_contract(raw['native_submission'], frozen, quotes)
    return context, annotated_saved_repair(context), raw, frozen, quotes


def cautious(raw):
    native = deepcopy(raw['native_submission'])
    native.update(ss='PROVISIONAL', ia='NONE', cc=False, esa=False, am={'A0': 'UNASSESSED', 'A1': 'UNASSESSED'},
                  x='Unspecified promise costs; no numerical welfare conversion',
                  dr='Preference conditional on a justified welfare comparison', ft='NONE',
                  l={'A0': 'Higher restricted expected survivor count', 'A1': 'Honours promise; welfare comparison unresolved'})
    return native


class InferenceRepairTests(unittest.TestCase):
    def test_conditionalization_cannot_be_selected_and_findings_survive(self):
        context, repair, raw, frozen, quotes = fixture()
        original = deepcopy(raw)
        result = check_submission(context, repair, raw['native_submission'], frozen, quotes)
        self.assertFalse(result['dispatch_native'])
        self.assertIn('SELECTION_STRENGTH', result['errors'])
        self.assertEqual(result['preserved_restricted_findings'], repair['preserved_restricted_findings'])
        self.assertEqual(raw, original)
        self.assertTrue(result['world_unchanged'])

    def test_cautious_conditional_submission_passes_without_changing_counts(self):
        context, repair, raw, frozen, quotes = fixture()
        native = cautious(raw)
        result = check_submission(context, repair, native, frozen, quotes)
        self.assertTrue(result['dispatch_native'], result)
        self.assertEqual(native['ev'], raw['native_submission']['ev'])

    def test_unit_conversion_and_numeric_reversal_stay_nonoperative(self):
        context, repair, raw, frozen, quotes = fixture()
        native = cautious(raw)
        native['ft'] = 'Promise cost exceeds 0.9 life-equivalents'
        repair['reversal_condition'].update(text=native['ft'], status='ILLUSTRATIVE')
        result = check_submission(context, repair, native, frozen, quotes)
        self.assertIn('UNLICENSED_WELFARE_UNIT', result['errors'])
        self.assertIn('NONOPERATIVE_REVERSAL', result['errors'])
        self.assertEqual(result['reversal_condition'], repair['reversal_condition'])

    def test_fake_grounded_reversal_or_count_ev_change_is_rejected(self):
        context, repair, raw, frozen, quotes = fixture()
        native = cautious(raw)
        native['ft'] = 'Promise cost exceeds 0.9'
        repair['reversal_condition'].update(text=native['ft'], status='SOURCE_GROUNDED', evidence_ids=['E2'])
        result = check_submission(context, repair, native, frozen, quotes)
        self.assertIn('UNLICENSED_NUMERIC_REVERSAL', result['errors'])
        native = cautious(raw); native['ev']['A0']['unit'] = 'UTILITY'
        result = check_submission(context, repair, native, frozen, quotes)
        self.assertIn('PROTECTED_NATIVE_FIELDS', result['errors'])

    def test_defend_supported_restricted_comparison_can_remain_selected(self):
        context, repair, raw, frozen, quotes = fixture()
        context = make_contract(raw['native_submission'], frozen, quotes, selection_scope='EXPECTED_SURVIVORS')
        repair = annotated_saved_repair(context)
        repair.update(repair_operation='DEFEND', conditional_assumptions=[], unresolved_claims=[])
        repair['supported_conclusion'].update(status='SUPPORTED', scope='EXPECTED_SURVIVORS', action_id='A0', text='A0 has greater expected survivor count')
        repair['conditional_preference'].update(action_id='NONE', assumption_ids=[], text='NONE')
        native = cautious(raw); native.update(ss='SELECTED', r='A0', cc=True, esa=True,
            da='expected survivor count', t='Compare restricted survivors', dr='Prefer larger expected survivor count', x='NONE')
        self.assertTrue(check_submission(context, repair, native, frozen, quotes)['dispatch_native'])

    def test_retract_cannot_reuse_stale_recommendation(self):
        context, repair, raw, frozen, quotes = fixture()
        repair['repair_operation'] = 'RETRACT'
        repair['supported_conclusion'].update(status='WITHDRAWN', action_id='NONE')
        repair['conditional_preference'].update(action_id='NONE', assumption_ids=[], text='NONE')
        result = check_submission(context, repair, cautious(raw), frozen, quotes)
        self.assertIn('WITHDRAWN_INFERENCE_NO_DISPATCH', result['errors'])

    def test_defend_cannot_hide_declared_assumptions_in_supported_status(self):
        context, repair, raw, frozen, quotes = fixture()
        repair['repair_operation'] = 'DEFEND'
        repair['supported_conclusion'].update(status='SUPPORTED', action_id='A0')
        self.assertIn('UNCONDITIONAL_FROM_ASSUMPTIONS', check_submission(context, repair, cautious(raw), frozen, quotes)['errors'])

    def test_world_commitments_locality_and_ids_are_preserved(self):
        context, repair, raw, frozen, quotes = fixture()
        changed = deepcopy(frozen); changed['scenario'] += ' Changed.'
        self.assertIn('SOURCE_CHANGED', check_submission(context, repair, cautious(raw), changed, quotes)['errors'])
        changed = deepcopy(quotes); changed['utilitarian']['text'] = 'invented'
        self.assertIn('COMMITMENTS_CHANGED', check_submission(context, repair, cautious(raw), frozen, changed)['errors'])
        native = cautious(raw); native['ct']['A0'][0]['wi'] = 'CRITICAL'
        self.assertIn('PROTECTED_NATIVE_FIELDS', check_submission(context, repair, native, frozen, quotes)['errors'])
        repair['challenged_inference_id'] = 'other'
        self.assertIn('INFERENCE_ID_MISMATCH', check_submission(context, repair, cautious(raw), frozen, quotes)['errors'])

    def test_answered_has_no_authority_and_missing_repair_fails_closed(self):
        context, repair, raw, frozen, quotes = fixture()
        self.assertFalse(check_submission(context, None, cautious(raw), frozen, quotes)['dispatch_native'])
        class Backend:
            model = 'test'
            def complete_json(self, prompt, **kwargs):
                return {'argument_response': {'disposition': 'answered'}, 'native_submission': raw['native_submission'], 'inference_repair': repair}
        with tempfile.TemporaryDirectory() as folder:
            packet = {'initial_argument': raw['native_submission']}
            wrapper = RevisionBackend(Backend(), packet, 'objection_focused', folder,
                repair_contract=context, frozen_world=frozen, commitments=quotes)
            with self.assertRaises(Exception): wrapper.complete_json('p', schema={}, max_tokens=4096)
            gate = json.loads((Path(folder)/'submission_consistency.json').read_text())
            self.assertFalse(gate['dispatch_native'])
            self.assertEqual(json.loads((Path(folder)/'argument_response.json').read_text())['record']['disposition'], 'answered')
            self.assertEqual(wrapper.calls, 1)

    def test_malformed_records_are_rejected_not_guessed(self):
        context, repair, raw, frozen, quotes = fixture()
        for change in ({'repair_operation': 'NEW'}, {'world_state': {}}, {'conditional_assumptions': 'none'}):
            bad = dict(repair, **change)
            self.assertIn('INVALID_REPAIR_SCHEMA', check_submission(context, bad, cautious(raw), frozen, quotes)['errors'])

    def test_scope_cannot_be_changed_to_evade_welfare_uncertainty(self):
        context, repair, raw, frozen, quotes = fixture()
        repair['repair_operation'] = 'DEFEND'
        repair['supported_conclusion'].update(scope='EXPECTED_SURVIVORS', status='SUPPORTED', action_id='A0')
        self.assertIn('CONCLUSION_SCOPE_MISMATCH', check_submission(context, repair, raw['native_submission'], frozen, quotes)['errors'])

    def test_unit_conversion_cannot_hide_in_referenced_assumption(self):
        context, repair, raw, frozen, quotes = fixture()
        repair['conditional_assumptions'][0]['text'] = 'Promise cost < 0.9 life-equivalents'
        result = check_submission(context, repair, cautious(raw), frozen, quotes)
        self.assertIn('UNLICENSED_WELFARE_UNIT', result['errors'])
        repair['conditional_assumptions'][0]['text'] = 'Promise costs are insufficient to offset survival benefit'
        native = cautious(raw); native['dr'] = 'utility = 0.9'
        self.assertIn('UNLICENSED_WELFARE_UNIT', check_submission(context, repair, native, frozen, quotes)['errors'])

    def test_framework_qualitative_reversal_and_weaken_are_allowed(self):
        context, repair, raw, frozen, quotes = fixture()
        repair['repair_operation'] = 'WEAKEN'
        repair['reversal_condition'].update(text='A justified welfare comparison could change the conditional ranking',
            status='FRAMEWORK_DERIVED', evidence_ids=[quotes['utilitarian']['passage_id']])
        native = cautious(raw); native['ft'] = repair['reversal_condition']['text']
        self.assertTrue(check_submission(context, repair, native, frozen, quotes)['dispatch_native'])

    def test_only_host_licensed_grounded_numeric_reversal_can_be_operative(self):
        context, repair, raw, frozen, quotes = fixture()
        context = make_contract(raw['native_submission'], frozen, quotes, selection_scope='EXPECTED_SURVIVORS')
        repair = annotated_saved_repair(context)
        text = 'Restricted ranking reverses if A1 exceeds 0.95 PERSON expected survivors'
        repair['reversal_condition'].update(text=text, status='SOURCE_GROUNDED', evidence_ids=['E2'], derivation_id='count_A0')
        native = cautious(raw); native['ft'] = text
        self.assertIn('UNLICENSED_NUMERIC_REVERSAL', check_submission(context, repair, native, frozen, quotes)['errors'])
        context['licensed_numeric_reversals']['count_A0'] = {'text': text, 'scope': 'EXPECTED_SURVIVORS', 'evidence_ids': ['E2']}
        self.assertTrue(check_submission(context, repair, native, frozen, quotes)['dispatch_native'])

    def test_native_boundary_blocks_inconsistent_output_and_retains_provisional_status(self):
        native_python = Path('/tmp/parliament-smoke-env/bin/python')
        if not native_python.exists(): self.skipTest('Native interpreter unavailable')
        code = '''import sys, json
from pathlib import Path
sys.path.insert(0, '/tmp/parliament-smoke-614af0c')
from test_inference_repair import fixture, cautious
from run_objection_response_pilot import RevisionBackend
from logic_puzzles_frameworks import generate
context, repair, raw, frozen, quotes = fixture()
class Backend:
 model = 'saved_fixture_no_api'
 def __init__(self, submission): self.submission = submission
 def complete_json(self, prompt, **kwargs):
  assert 'inference_repair' in kwargs['schema']['required']
  return {'argument_response': {'disposition': 'answered'}, 'native_submission': self.submission, 'inference_repair': repair}
for kind, submission in [('blocked', raw['native_submission']), ('cautious', cautious(raw))]:
 folder = Path(sys.argv[1])/kind
 result = generate('diagnostics/specific_inference_probe_live_20261007/frozen_world_trace.json', folder,
  ['utilitarian'], lambda _: RevisionBackend(Backend(submission), {}, 'objection_focused', folder/'utilitarian',
     repair_contract=context, frozen_world=frozen, commitments=quotes), core_quotes=quotes, call_budget=1)
 gate = json.loads((folder/'utilitarian/submission_consistency.json').read_text())
 candidates = result['cycles'][0]['candidates']
 assert result['world_mutation'] is False
 if kind == 'blocked':
  assert gate['dispatch_native'] is False
  assert not candidates or all(c['delegate_status'] == 'MODEL_ERROR' for c in candidates)
  assert all(not c.get('committed_native_ledger') for c in candidates)
  assert json.loads((folder/'utilitarian/revision_call.json').read_text())['status'] == 'RETURNED'
 else:
  assert gate['dispatch_native'] is True
  assert candidates[0]['delegate_status'] == 'VALID'
  assert candidates[0]['selection_status'] == 'PROVISIONAL'
'''
        with tempfile.TemporaryDirectory() as folder:
            run = subprocess.run([str(native_python), '-c', code, folder], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)


if __name__ == '__main__': unittest.main()
