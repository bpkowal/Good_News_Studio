"""Authored adapter/roster checks; no API calls or gold moral verdicts."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from contractualist_world_adapter import world_source, generate, framework_view, fingerprint
from run_contractualist_demo import assemble_report, NATIVE_FRAMEWORKS
from run_opposing_complaint_probe import verify_freeze
from contractualist_pilot import compile_response
from test_contractualist_pilot import proposal, FIXTURE
from test_contractualist_questions import question
from export_parliament_viewer import presentation, export
from run_parliament_demo import write_report


def source():
    case = json.loads(FIXTURE.read_text())['cases'][0]
    world = {'parties': [{'party_id': 'P1', 'label': 'Jo', 'kind': 'PERSON'}],
        'actions': [{'action_id': a['id'], 'intervention': a['label'], 'actor_party_id': 'P1',
                     'effect_ids': []} for a in case['actions']],
        'effects': [], 'causal_links': [], 'counterfactual_links': [], 'conditions': []}
    return world_source(case['scenario'], world, trace_sha256='1'*64)


def raw_response(src):
    raw = proposal(src)
    raw['critical_questions'] = [question()]
    raw['conclusion'] = 'A comparative priority remains open; no ranking is claimed.'
    return raw


def result(src):
    raw = raw_response(src)
    compiled = compile_response(src, 'contractualist', raw)
    compiled.update(adapter_version='test', transport_status='RETURNED',
                    shared_world_sha256=src['world_sha256'], source_trace_sha256=src['source_trace_sha256'])
    return compiled


def native(src):
    return {'scenario': src['scenario'], 'action_source_grounding': {'world_model': deepcopy(src['admitted_world'])},
            'cycles': [{'candidates': [{'specialist': name, 'recommended_action': None,
                'rationale': 'No settled ranking.', 'framework_vote_status': 'ABSTAIN',
                'committed_native_ledger': None} for name in NATIVE_FRAMEWORKS]}],
            'judgment_status': 'NOT_ADJUDICATED', 'framework_runs': []}


class ContractualistDemoTests(unittest.TestCase):
    def test_shared_world_and_native_action_ids_survive_without_entity_merging(self):
        src = source(); before = deepcopy(src)
        compiled = result(src)
        self.assertEqual(compiled['argument']['world_sha256'], src['world_sha256'])
        self.assertEqual(src, before)
        self.assertEqual([a['id'] for a in src['actions']], [a['action_id'] for a in src['admitted_world']['actions']])

    def test_question_view_does_not_invent_a_rawls_score_vote_or_native_ledger(self):
        src = source(); view = framework_view(result(src), src)
        self.assertEqual(view['framework'], 'contractualist')
        self.assertEqual(view['position'], 'CLARIFICATION')
        self.assertIsNone(view['preference'])
        self.assertEqual(view['ledger_status'], 'NOT_APPLICABLE_ARGUMENT_ADAPTER')
        self.assertFalse(view['reasoning_provenance']['native_ledger_admission'])
        self.assertNotIn('rawls', json.dumps(view).lower())
        self.assertEqual(view['critical_questions'][0]['id'], 'q_priority')

    def test_one_live_transport_slot_is_frozen_and_never_retried(self):
        src = source(); before = deepcopy(src); calls = []
        class Response:
            choices = [SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                content=json.dumps(raw_response(src)), refusal=None))]
            def model_dump(self, **kwargs):
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': self.choices[0].message.content}}]}
        def create(**request):
            calls.append(request); return Response()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with TemporaryDirectory() as temp:
            folder = Path(temp)/'contract'
            compiled = generate(src, folder, client_factory=lambda: client)
            self.assertEqual(compiled['transport_status'], 'RETURNED')
            self.assertFalse(compiled['native_ledger_admission'])
            self.assertEqual(len(calls), 1)
            packet = json.loads(calls[0]['messages'][1]['content'])
            self.assertEqual(packet['source']['admitted_world'], src['admitted_world'])
            self.assertNotIn('peer_arguments', packet)
            verify_freeze(folder, 'inputs.freeze.json'); verify_freeze(folder, 'output.freeze.json')
            with self.assertRaises(FileExistsError): generate(src, folder, client_factory=lambda: client)
            self.assertEqual(len(calls), 1)
        self.assertEqual(src, before)

    def test_misbound_source_is_rejected_before_a_provider_call(self):
        src = source(); src['admitted_world']['actions'][0]['intervention'] = 'Changed action'
        with TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                generate(src, Path(temp), client_factory=lambda: self.fail('Must not call model'))

    def test_roster_replacement_preserves_input_world_and_native_records(self):
        src = source(); trace = native(src); before = deepcopy((src, trace))
        report = assemble_report(src['scenario'], src['admitted_world'], trace, result(src), src)
        self.assertEqual([v['framework'] for v in report['frameworks']], [*NATIVE_FRAMEWORKS, 'contractualist'])
        self.assertEqual(report['world'], src['admitted_world'])
        self.assertEqual((src, trace), before)
        self.assertEqual(report['agent_roster']['replaced_for_this_run'], 'rawlsian')
        self.assertTrue(any(h.get('role') == 'OPEN_QUESTION' for h in report['hinges']))
        self.assertFalse(any(a['framework'] in ('rawlsian', 'contractualist') for a in report['topography']['framework_arrows']))

    def test_mismatched_scenario_or_world_is_not_presented_as_shared(self):
        src = source(); trace = native(src)
        wrong = deepcopy(src); wrong['scenario'] += ' Changed.'
        with self.assertRaises(ValueError): assemble_report(src['scenario'], src['admitted_world'], trace, result(src), wrong)
        wrong = deepcopy(trace); wrong['action_source_grounding']['world_model']['actions'].pop()
        with self.assertRaises(ValueError): assemble_report(src['scenario'], src['admitted_world'], wrong, result(src), src)

    def test_saved_rawls_candidate_is_not_silently_relabeled(self):
        src = source(); trace = native(src)
        trace['cycles'][0]['candidates'].append({'specialist': 'rawlsian'})
        with self.assertRaises(ValueError):
            assemble_report(src['scenario'], src['admitted_world'], trace, result(src), src)

    def test_report_and_viewer_preserve_questions_and_full_original_argument(self):
        src = source(); compiled = result(src)
        report = assemble_report(src['scenario'], src['admitted_world'], native(src), compiled, src)
        before = deepcopy(report)
        payload = presentation(report, 'contract_test', 'Contractualist test')
        view = payload['frameworks'][-1]
        self.assertEqual(view['critical_questions'], compiled['raw_response']['critical_questions'])
        self.assertEqual(view['recorded_argument'], compiled['raw_response'])
        self.assertEqual(payload['agent_roster']['shared_world_sha256'], src['world_sha256'])
        self.assertEqual(report, before)
        with TemporaryDirectory() as temp:
            output = Path(temp)
            write_report(report, output)
            self.assertIn('Critical unanswered questions', (output/'arguments/contractualist.md').read_text())
            self.assertIn('critical unanswered question', (output/'DEMO_REPORT.md').read_text())


if __name__ == '__main__':
    unittest.main()
