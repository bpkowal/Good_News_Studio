from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from opposing_complaint import (ARMS, FOCUSED, bind_assignment, bind_reply,
    reply_request, selector_request)
from run_contractualist_pilot import digest, write_new
from run_opposing_complaint_probe import prepare, execute, freeze, verify_freeze


def packet():
    text = 'Ari needs aid. Bea needs aid. Under A, Ari gets aid. Under B, Bea gets aid.'
    return {'framework_id': 'contractualist', 'framework_instruction': 'Justify principles to individuals.',
        'source': {'id': 'test_case', 'scenario': text, 'source_text_sha256': digest(text.encode()),
                   'actions': [{'id': 'A', 'label': 'Aid Ari'}, {'id': 'B', 'label': 'Aid Bea'}]},
        'prior_argument': {'recommendation_action_id': 'A',
                           'conclusion': 'Choose Ari because the benefit justifies it.'},
        'prior_argument_origin': 'TEST_AUTHORED_PRIOR_NOT_MODEL_OUTPUT'}


def assignment():
    return {'status': 'GROUNDED_TARGET_PROPOSED', 'complainant': 'Bea',
        'recommended_action_id': 'A', 'alternative_action_id': 'B',
        'source_quotes': ['Bea needs aid.'], 'complaint': 'Why am I left without aid?',
        'target_path': '/conclusion', 'target_quote': 'the benefit justifies it',
        'challenged_inference': 'Why the benefit to Ari outweighs Bea\'s claim.',
        'why_prior_answer_is_insufficient': 'It asserts a priority without defending it.',
        'selection_limits': 'This is a proposed objection, not a validated strongest claim.'}


def reply():
    return {'complaint_restatement': 'Bea asks why she is left without aid.',
        'conceded_points': ['Bea has an actual need.'], 'priority_premise': 'A priority remains unproved.',
        'answer': 'I cannot establish that Ari\'s claim outweighs Bea\'s.',
        'countercomplaint_to_alternative': 'Ari also needs aid.',
        'comparative_justification': 'The relevant difference is unresolved.',
        'premises': [{'claim': 'Bea needs aid.', 'kind': 'SOURCE_FACT', 'source_quotes': ['Bea needs aid.']}],
        'conclusion': {'action_id': 'A', 'status': 'PROVISIONAL_PREFERENCE',
                       'rationale': 'Retain only a provisional proposal.'},
        'changed_inferences': [{'original_path': '/conclusion', 'original_quote': 'the benefit justifies it',
            'replacement': 'Priority is not established.', 'reason': 'The competing need remains.'}],
        'disposition': 'PARTLY_ANSWERED', 'remaining_objection': 'Bea\'s objection remains live.',
        'reversal_condition': 'A justified difference in claims could settle the comparison.'}


def prior_run(root):
    root.mkdir()
    write_new(root/'protocol.json', {'case_ids': ['test_case'], 'frameworks': ['contractualist'],
                                    'model': 'o3', 'reasoning_effort': 'low', 'max_completion_tokens': 8192})
    folder = root/'test_case'/'contractualist'; folder.mkdir(parents=True)
    p = packet()
    write_new(folder/'source.json', p['source'])
    write_new(folder/'response.json', p['prior_argument'])
    write_new(folder/'request.json', {'messages': [{'role': 'system', 'content': 'Prior test.'},
        {'role': 'user', 'content': json.dumps({'source': p['source'],
                                              'framework_instruction': p['framework_instruction']})}]})
    freeze(folder, 'output.freeze.json', list(folder.iterdir()))


class OpposingComplaintTests(unittest.TestCase):
    def test_assignment_must_target_saved_choice_and_exact_prior_inference(self):
        p = packet(); raw = assignment()
        self.assertEqual(bind_assignment(p, raw)['binding_issues'], [])
        raw['recommended_action_id'] = 'B'; raw['target_quote'] = 'an invented prior conclusion'
        errors = bind_assignment(p, raw)['binding_issues']
        self.assertIn('TARGET_NOT_PRIOR_RECOMMENDATION', errors)
        self.assertIn('TARGET_QUOTE_MISMATCH', errors)

    def test_assignment_citations_are_not_semantic_or_normative_certification(self):
        p = packet(); raw = assignment(); raw['source_quotes'] = ['Bea prefers winning.']
        bound = bind_assignment(p, raw)
        self.assertIn('SOURCE_QUOTE_MISMATCH', bound['binding_issues'])
        self.assertEqual(bound['semantic_support'], 'NOT_INDEPENDENTLY_ASSESSED')
        self.assertFalse(bound['operative'])
        self.assertEqual(bound['raw_assignment'], raw)

    def test_matched_requests_have_identical_packets_schema_model_and_ceiling(self):
        p = packet(); target = bind_assignment(p, assignment()); before = deepcopy(p)
        requests = [reply_request(p, target, arm, model='o3', reasoning_effort='low') for arm in ARMS]
        self.assertEqual(requests[0]['messages'][1], requests[1]['messages'][1])
        self.assertEqual(requests[0]['response_format'], requests[1]['response_format'])
        self.assertEqual(requests[0]['model'], requests[1]['model'])
        self.assertEqual(requests[0]['max_completion_tokens'], requests[1]['max_completion_tokens'])
        self.assertNotEqual(requests[0]['messages'][0], requests[1]['messages'][0])
        self.assertEqual(p, before)
        self.assertIn('conditional preference', requests[0]['messages'][0]['content'])
        self.assertNotIn('Bea', FOCUSED)
        self.assertNotIn('Jo', FOCUSED)
        self.assertNotIn('Sasha', FOCUSED)

    def test_selector_does_not_receive_assistant_review_or_another_framework(self):
        p = packet(); request = selector_request(p, model='o3', reasoning_effort='low')
        actual = json.loads(request['messages'][1]['content'])
        self.assertEqual(actual, p)
        self.assertNotIn('peer_arguments', actual)
        self.assertNotIn('tools', request)
        self.assertNotIn('CONTENT_REVIEW', json.dumps(request))

    def test_revised_conclusion_does_not_automatically_retire_objection_or_change_source(self):
        p = packet(); before = deepcopy(p); raw = reply(); raw['disposition'] = 'ANSWERED'
        bound = bind_reply(p, raw)
        self.assertEqual(bound['binding_issues'], [])
        self.assertFalse(bound['self_report_is_resolution'])
        self.assertFalse(bound['operative'])
        self.assertEqual(p, before)
        self.assertEqual(bound['raw_reply']['remaining_objection'], raw['remaining_objection'])

    def test_changed_inference_and_factual_claims_preserve_exact_source_bindings(self):
        raw = reply(); raw['changed_inferences'][0]['original_path'] = '/not_present'
        raw['premises'][0]['source_quotes'] = ['A fabricated fact']
        issues = bind_reply(packet(), raw)['binding_issues']
        self.assertIn('PRIOR_CHANGE_PATH_MISMATCH:0', issues)
        self.assertIn('PREMISE_QUOTE_MISMATCH:0', issues)

    def test_prepare_is_offline_and_freezes_original_argument_with_bounded_budget(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); prior_run(root/'prior')
            before = {p: p.read_bytes() for p in (root/'prior').rglob('*') if p.is_file()}
            protocol = prepare(root/'prior', root/'probe')
            self.assertEqual(protocol['maximum_provider_calls'], 3)
            verify_freeze(root/'probe', 'inputs.freeze.json')
            self.assertFalse(list((root/'probe').rglob('call.json')))
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_fake_complete_probe_makes_one_assignment_and_two_fresh_replies(self):
        calls = []
        class Response:
            def __init__(self, raw):
                self.raw = raw
                self.choices = [SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                    content=json.dumps(raw), refusal=None))]
            def model_dump(self, **kwargs):
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(self.raw)}}]}
        def create(**request):
            calls.append(request)
            schema = request['response_format']['json_schema']['schema']
            return Response(assignment() if 'complainant' in schema['properties'] else reply())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with TemporaryDirectory() as temp:
            root = Path(temp); prior_run(root/'prior')
            prepare(root/'prior', root/'probe')
            rows = execute(root/'probe', client_factory=lambda: client)
            self.assertEqual(len(calls), 3)
            self.assertEqual([r['status'] for r in rows], ['RETURNED', 'RETURNED'])
            self.assertEqual(calls[1]['messages'][1], calls[2]['messages'][1])
            verify_freeze(root/'probe/test_case/selector', 'outputs.freeze.json')
            verify_freeze(root/'probe/test_case', 'reply_inputs.freeze.json')
            with self.assertRaises(ValueError): execute(root/'probe', client_factory=lambda: client)
            self.assertEqual(len(calls), 3)

    def test_failed_assignment_is_retained_without_retry_or_fabricated_challenge(self):
        calls = []
        def create(**request):
            calls.append(request)
            raise RuntimeError('secret provider detail')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with TemporaryDirectory() as temp:
            root = Path(temp); prior_run(root/'prior')
            prepare(root/'prior', root/'probe')
            rows = execute(root/'probe', client_factory=lambda: client)
            self.assertEqual(len(calls), 1)
            self.assertTrue(all(r['status'] == 'NO_ASSIGNED_TARGET' for r in rows))
            self.assertNotIn('secret provider detail', (root/'probe/test_case/selector/result.json').read_text())


if __name__ == '__main__':
    unittest.main()
