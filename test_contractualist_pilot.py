from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from argument_construction import construct
from argument_frameworks import SHARED_GENERATION, SHARED_CRITIQUE
from argument_support import compile_argument, project
from contractualist_pilot import (FRAMEWORKS, compile_response, make_request,
                                  response_schema, source_packet)
from run_contractualist_pilot import prepare, execute, verify_inputs

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT/'fixtures/contractualist_pilot_cases.json'


def statement(handle, kind, text, hypotheses=(), quote=None):
    return {'handle': handle, 'kind': kind, 'text': text, 'polarity': 'AFFIRMED',
            'modality': 'ASSERTED', 'hypothesis_ids': list(hypotheses), 'citations': [] if quote is None else
            [{'evidence_id': 'SCENE', 'value_path': '', 'quote': quote}]}


def proposal(source):
    """Authored structural fixture, not a good-morals answer or model result."""
    rows = [statement('p', 'PRINCIPLE', 'Consider the claims of each person.'),
            statement('n', 'NORMATIVE_PREMISE', 'A person can object to unprovided protection.'),
            statement('f', 'FACTUAL_PREMISE', 'The source concerns a cooling plan.', quote='Plan B'),
            statement('r', 'PERSONAL_REASON', 'Jo has a reason to question the allocation.', ('p', 'n')),
            statement('cmp', 'EVALUATIVE_COMPARISON', 'This does not settle the comparison.', ('p',))]
    routes = [dict(operation='NORM_APPLICATION', inputs=['p', 'n', 'f'], output='r', explanation='Conditional objection.'),
              dict(operation='NORM_PRIORITY', inputs=['r', 'n'], output='cmp', explanation='Comparison remains unresolved.')]
    assessments = []
    for action in source['actions']:
        handle = 'conclusion_' + action['id']
        rows.append(statement(handle, 'NORMATIVE_JUDGMENT', 'Assessment is unresolved for ' + action['id'], ('p',)))
        routes.append(dict(operation='NORM_APPLICATION', inputs=['cmp', 'n'], output=handle,
                           explanation='Retain an unresolved judgment.'))
        assessments.append(dict(action_id=action['id'], conclusion_handle=handle, assessment='UNRESOLVED'))
    return {'persons': [{'id': 'jo', 'label': 'Jo', 'source_quotes': ["Jo's home"]}],
        'burdens': [{'id': 'heat', 'person_id': 'jo', 'description': 'Unrelieved heat exposure.'}],
        'burden_positions': [{'burden_id': 'heat', 'action_id': a['id'],
            'status': 'ABSENT' if a['id'] == 'C' else 'PRESENT',
            'position': 'Cooling under C; no cooling under A or B.', 'ground_handles': ['f']} for a in source['actions']],
        'principles': [{'id': 'principle', 'statement_handle': 'p', 'action_ids': [a['id'] for a in source['actions']]}],
        'statements': rows, 'supports': routes,
        'reasons': [{'id': 'reason', 'person_id': 'jo', 'principle_id': 'principle', 'action_id': 'B',
                    'stance': 'AGAINST', 'value': 'protection', 'conclusion_handle': 'r',
                    'ground_handles': ['f'], 'warrant_handles': ['n'], 'generic_basis': 'Any excluded person.'}],
        'comparisons': [{'id': 'comparison', 'reason_ids': ['reason'], 'judgment_handle': 'cmp',
                         'from_action_id': 'B', 'to_action_id': 'C', 'status': 'UNRESOLVED',
                         'relieved_burden_ids': ['heat'], 'introduced_burden_ids': [],
                         'retained_burden_ids': [], 'explanation': 'Not sufficient to settle.'}],
        'challenges': [{'id': 'challenge', 'kind': 'UNDERCUTTING', 'target_kind': 'SUPPORT',
                        'target_id': 'r', 'challenge_handle': 'cmp', 'response_handle': None, 'status': 'OPEN'}],
        'action_assessments': assessments, 'recommendation_action_id': None,
        'recommendation_status': 'UNRESOLVED', 'recommendation_comparison_ids': [],
        'remaining_complaint_ids': ['reason'],
        'conclusion': 'No settled recommendation.', 'strongest_objection': 'Incomplete comparison.',
        'answer_to_objection': 'Retain it.', 'reversal_condition': 'Further reasons could support a ranking.',
        'residual_disagreement': ['Comparison remains open.']}


class NeutralPilotTests(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads(FIXTURE.read_text())['cases']
        self.source = source_packet(self.cases[0])

    def test_all_cases_preserve_exact_source_and_exclude_evaluator_metadata(self):
        for case in self.cases:
            source = source_packet(case)
            self.assertEqual(source['scenario'], case['scenario'])
            self.assertNotIn('design_notes', source)
            self.assertNotIn('pair', source)
        changed = deepcopy(self.cases[0]); changed['scenario'] += ' Changed.'
        with self.assertRaises(ValueError): source_packet(changed)

    def test_same_common_instructions_schema_budget_and_source_across_frameworks(self):
        requests = [make_request(self.source, f) for f in FRAMEWORKS]
        for request in requests[1:]:
            self.assertEqual(request['messages'][0], requests[0]['messages'][0])
            self.assertEqual(request['response_format'], requests[0]['response_format'])
            self.assertEqual(request['max_completion_tokens'], requests[0]['max_completion_tokens'])
            self.assertEqual(json.loads(request['messages'][1]['content'])['source'], self.source)
        for request in requests:
            self.assertEqual(len(request['messages']), 2)
            self.assertNotIn('tools', request)
        for common in (SHARED_GENERATION, SHARED_CRITIQUE):
            self.assertNotIn('strongest competing duty', common)
            self.assertNotIn('instrumental belief', common)
            self.assertNotIn('Identify a candidate maxim', common)
        self.assertIn('candidate maxims', json.loads(requests[1]['messages'][1]['content'])['framework_instruction'])
        self.assertNotIn('candidate maxims', json.loads(requests[0]['messages'][1]['content'])['framework_instruction'])

    def test_legacy_construct_uses_framework_local_critique_without_deontological_requirement(self):
        source = dict(id='C', world_sha256='W', framework_id='utilitarian', framework='utilitarian',
                      scenario='A source.', evidence=[{'ref': 'SCENE', 'value': 'A source.'}])
        candidate = dict(label='conditional welfare',
            statements=[statement('n', 'NORMATIVE_PREMISE', 'Assume comparable welfare.'),
                        statement('c', 'NORMATIVE_JUDGMENT', 'Conditional conclusion.', ('n',))],
            supports=[dict(operation='NORM_APPLICATION', inputs=['n'], output='c', explanation='Conditional.')],
            conclusion_handle='c', strongest_objection='Comparability unknown.', answer_to_objection='Conditional only.',
            reversal_condition='Noncomparability.')
        prompts = []
        class Backend:
            def complete_json(inner, prompt, **kwargs):
                prompts.append(prompt)
                if len(prompts) == 1:
                    return {'candidates': [candidate]}
                return dict(assessments=[], revisions=[], strongest_remaining_conclusion='Conditional conclusion.', remaining_questions=[])
        result = construct(source, Backend())
        self.assertEqual(result['candidates'][0]['status'], 'STAGED_ARGUMENT')
        for prompt in prompts:
            self.assertIn('restricted endpoints', prompt)
            self.assertNotIn('strongest competing duty', prompt)

    def test_neutral_assumption_roots_propagate_without_maxim_or_kind_laundering(self):
        state = compile_argument('TEXT', self.source['evidence'], 'ISSUE',
            [statement('p', 'PRINCIPLE', 'A proposed norm.'),
             statement('i', 'INTERPRETIVE_PREMISE', 'A defeasible reading.'),
             statement('c', 'PERSONAL_REASON', 'A conditional complaint.', ('p', 'i'))],
            [dict(operation='PARAPHRASE', inputs=['p', 'i'], output='c', explanation='Conditional route.')])
        view = project(state)
        self.assertEqual(set(view['statements'][state['handles']['c']]['effective_hypotheses']),
                         {state['handles']['p'], state['handles']['i']})
        self.assertNotIn('MAXIM', [s['kind'] for s in state['statements'].values()])

    def test_three_options_compile_and_remain_nonoperative_with_unassessed_support(self):
        raw = proposal(self.source); original = deepcopy(raw)
        result = compile_response(self.source, 'contractualist', raw)
        self.assertEqual(result['status'], 'STAGED_ARGUMENT')
        self.assertFalse(result['operative'])
        self.assertFalse(result['paper_attribution']['formal_semantics_implemented'])
        self.assertEqual(raw, original)
        self.assertNotIn('SUPPORTED', [r['status'] for r in result['view']['support_routes'].values()])
        self.assertFalse(result['view']['operative'])

    def test_bad_citation_retains_raw_argument_and_good_other_statements(self):
        raw = proposal(self.source)
        raw['statements'][2]['citations'][0]['quote'] = 'not present in text'
        result = compile_response(self.source, 'contractualist', raw)
        self.assertEqual(result['status'], 'STAGED_PARTIAL_ARGUMENT')
        self.assertEqual(result['raw_response'], raw)
        self.assertIn('p', result['argument']['handles'])
        self.assertNotIn('f', result['argument']['handles'])

    def test_missing_person_and_option_are_visible_without_deleting_proposal(self):
        raw = proposal(self.source)
        raw['reasons'][0]['person_id'] = 'invented'
        raw['action_assessments'].pop()
        result = compile_response(self.source, 'contractualist', raw)
        kinds = {x['kind'] for x in result['issues']}
        self.assertIn('MISSING_PERSON_ID', kinds)
        self.assertIn('OPTION_COVERAGE_INCOMPLETE', kinds)
        self.assertEqual(result['raw_response'], raw)

    def test_comparison_and_terminal_checks_expose_disconnected_or_circular_moral_reasoning(self):
        raw = proposal(self.source)
        raw['supports'][1]['inputs'] = ['n']
        raw['supports'].append(dict(operation='PARAPHRASE', inputs=['conclusion_A'], output='f',
                                    explanation='Illicit verdict-to-fact route.'))
        result = compile_response(self.source, 'contractualist', raw)
        kinds = {x['kind'] for x in result['issues']}
        self.assertIn('DETACHED_COMPARISON', kinds)
        self.assertIn('VERDICT_USED_AS_PREMISE', kinds)
        # A genuine support cycle is rejected locally; the rest survives.
        cyclic = proposal(self.source)
        cyclic['supports'].append(dict(operation='PARAPHRASE', inputs=['conclusion_A'], output='f',
                                       explanation='Cycle back to the factual premise.'))
        self.assertIn('CONSTRUCTION', {x['kind'] for x in
            compile_response(self.source, 'contractualist', cyclic)['issues']})
        other = proposal(self.source)
        other['statements'].append(statement('extra', 'NORMATIVE_JUDGMENT', 'Another judgment.'))
        other['supports'].append(dict(operation='PARAPHRASE', inputs=['conclusion_A'], output='extra', explanation='Terminal reused.'))
        result = compile_response(self.source, 'contractualist', other)
        self.assertIn('VERDICT_USED_AS_PREMISE', {x['kind'] for x in result['issues']})

    def test_attack_binding_does_not_retire_objection_or_compute_defeat(self):
        raw = proposal(self.source); raw['challenges'][0]['status'] = 'ANSWERED'
        result = compile_response(self.source, 'contractualist', raw)
        self.assertEqual(result['raw_response']['challenges'][0]['status'], 'ANSWERED')
        self.assertNotIn('grounded_extension', result)
        self.assertNotIn('defeats', result)
        raw['challenges'][0]['target_kind'] = 'STATEMENT'
        result = compile_response(self.source, 'contractualist', raw)
        self.assertIn('ATTACK_TARGET_KIND_MISMATCH', {x['kind'] for x in result['issues']})

    def test_malformed_transport_is_retained(self):
        raw = proposal(self.source); raw['unsupported_override'] = True
        result = compile_response(self.source, 'care', raw)
        self.assertEqual(result['status'], 'MALFORMED_RESPONSE')
        self.assertEqual(result['raw_response'], raw)

    def test_prepare_all_cells_is_offline_frozen_and_metadata_free(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)/'prepared'
            protocol = prepare(FIXTURE, output)
            self.assertEqual(protocol['maximum_provider_calls'], 16)
            self.assertEqual(verify_inputs(output), protocol)
            for path in output.glob('*/*/request.json'):
                packet = json.loads(path.read_text())
                self.assertEqual(packet['response_format']['json_schema']['schema'], response_schema())
                self.assertNotIn('design_notes', json.loads(packet['messages'][1]['content'])['source'])
            self.assertFalse(list(output.glob('*/*/call.json')))
            request_path = next(output.glob('*/*/request.json'))
            request_path.write_text('{}')
            with self.assertRaises(ValueError): verify_inputs(output)

    def test_fake_execution_freezes_outputs_and_never_retries_a_cell(self):
        source = self.source
        class Response:
            choices = [SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps(proposal(source)), refusal=None))]
            def model_dump(self, **kwargs):
                return {'choices': [{'message': {'content': self.choices[0].message.content}}]}
        calls = []
        def create(**request):
            calls.append(request)
            return Response()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with TemporaryDirectory() as temp:
            output = Path(temp)/'run'
            prepare(FIXTURE, output, cases=[source['id']], frameworks=['contractualist'])
            result = execute(output, client_factory=lambda: client)
            self.assertEqual(result[0]['status'], 'STAGED_ARGUMENT')
            self.assertEqual(len(calls), 1)
            self.assertTrue((output/source['id']/'contractualist'/'output.freeze.json').exists())
            with self.assertRaises(ValueError): execute(output, client_factory=lambda: client)
            self.assertEqual(len(calls), 1)

    def test_transport_failure_consumes_one_attempt_and_excludes_secret_messages(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            raise RuntimeError('sensitive provider message')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with TemporaryDirectory() as temp:
            output = Path(temp)/'run'
            prepare(FIXTURE, output, cases=[self.source['id']], frameworks=['contractualist'])
            result = execute(output, client_factory=lambda: client)
            self.assertEqual(result[0]['status'], 'FAILED')
            self.assertEqual(len(calls), 1)
            saved = (output/self.source['id']/'contractualist'/'result.json').read_text()
            self.assertNotIn('sensitive provider message', saved)
            with self.assertRaises(ValueError): execute(output, client_factory=lambda: client)


if __name__ == '__main__':
    unittest.main()
