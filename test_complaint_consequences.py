from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from complaint_consequences import (ARMS, build_consequence_record, consequence_request,
    bind_consequence_reply, render_consequences)
from opposing_complaint import bind_assignment
from run_consequence_complaint_probe import prepare, execute
from run_contractualist_pilot import digest, write_new
from run_opposing_complaint_probe import freeze, verify_freeze
from test_opposing_complaint import packet, assignment, reply


def rich_packet():
    p = packet()
    p['prior_argument'].update(persons=[{'id': 'ari', 'label': 'Ari', 'source_quotes': ['Ari needs aid.']},
                                      {'id': 'bea', 'label': 'Bea', 'source_quotes': ['Bea needs aid.']}],
        burdens=[{'id': 'lack_aid', 'person_id': 'bea', 'description': 'Lack of aid'}],
        burden_positions=[{'burden_id': 'lack_aid', 'action_id': 'A', 'status': 'PRESENT',
                           'position': 'Bea receives no aid', 'ground_handles': ['g1']},
                          {'burden_id': 'lack_aid', 'action_id': 'B', 'status': 'ABSENT',
                           'position': 'Bea receives aid', 'ground_handles': ['g2']}],
        statements=[{'handle': 'g1', 'kind': 'FACTUAL_PREMISE', 'text': 'Under A, Ari gets aid.',
                     'citations': [{'evidence_id': 'SCENE', 'value_path': '', 'quote': 'Under A, Ari gets aid.'}]},
                    {'handle': 'g2', 'kind': 'FACTUAL_PREMISE', 'text': 'Under B, Bea gets aid.',
                     'citations': [{'evidence_id': 'SCENE', 'value_path': '', 'quote': 'Under B, Bea gets aid.'}]}],
        comparisons=[{'id': 'ab', 'from_action_id': 'A', 'to_action_id': 'B', 'status': 'FAVORS_FROM'}])
    return p


def reviewed_reply():
    r = reply()
    r.update(factual_review=[{'target': {'origin': 'ASSIGNED_COMPLAINT',
        'path': '/raw_assignment/complaint', 'quote': 'left without aid'},
        'status': 'SUPPORTED', 'corrected_statement': 'Bea lacks aid under A.',
        'source_quotes': ['Under A, Ari gets aid.'], 'consequence_row_ids': ['row0']}],
        normative_warrant={'principle': 'Priority needs a justification.',
                           'justification': 'The difference is not established.',
                           'limits': 'Bea retains her objection.'})
    return r


def saved_probe(root):
    root.mkdir(); (root/'test_case/selector').mkdir(parents=True)
    write_new(root/'protocol.json', {'cases': ['test_case'], 'model': 'o3', 'reasoning_effort': 'low',
                                    'reply_max_tokens': 8192})
    write_new(root/'test_case/packet.json', rich_packet())
    freeze(root, 'inputs.freeze.json', [root/'protocol.json', root/'test_case/packet.json'])
    write_new(root/'test_case/selector/result.json', {
        'bound_assignment': bind_assignment(rich_packet(), assignment())})
    freeze(root/'test_case/selector', 'outputs.freeze.json', [root/'test_case/selector/result.json'])


class ConsequenceComplaintTests(unittest.TestCase):
    def test_quantitative_risks_remain_individual_and_option_specific(self):
        p = rich_packet()
        text = 'Ari has 70% survival under A and dies under B. Bea dies under A and has 30% survival under B.'
        p['source'].update(scenario=text, source_text_sha256=digest(text.encode()))
        prior = p['prior_argument']
        prior['burdens'] = [{'id': 'risk', 'person_id': 'bea', 'description': 'Death risk'}]
        prior['burden_positions'] = [
            {'burden_id': 'risk', 'action_id': 'A', 'status': 'PRESENT',
             'position': 'Bea: 100% death', 'ground_handles': ['g1']},
            {'burden_id': 'risk', 'action_id': 'B', 'status': 'PRESENT',
             'position': 'Bea: 70% death, 30% survival', 'ground_handles': ['g1']}]
        prior['statements'] = [{'handle': 'g1', 'kind': 'FACTUAL_PREMISE', 'text': text,
                               'citations': [{'evidence_id': 'SCENE', 'value_path': '', 'quote': text}]}]
        before = deepcopy(p); record = build_consequence_record(p)
        rows = record['position_rows']
        self.assertEqual([r['person_id'] for r in rows], ['bea', 'bea'])
        self.assertEqual([r['action_id'] for r in rows], ['A', 'B'])
        self.assertEqual(rows[1]['raw_positions'][0]['claimed_position'], 'Bea: 70% death, 30% survival')
        self.assertEqual(record['pair_changes'][0]['positions'][0]['change'],
                         'REMAINS_WITH_POSSIBLY_DIFFERENT_EXTENT')
        self.assertFalse(record['moral_ranking_computed'])
        self.assertNotIn('proposed_comparison_status', record['pair_changes'][0])
        self.assertEqual(p, before)

    def test_quoted_normative_claim_is_not_promoted_to_fact(self):
        p = rich_packet(); prior = p['prior_argument']
        prior['statements'][0]['kind'] = 'NORMATIVE_PREMISE'
        prior['statements'][0]['text'] = 'A is morally compulsory.'
        prior['burdens'][0]['description'] = 'Betrayal (model label)'
        record = build_consequence_record(p)
        ground = record['position_rows'][0]['raw_positions'][0]['grounds'][0]
        self.assertIn('GROUND_IS_NOT_A_FACTUAL_PREMISE', ground['binding_issues'])
        self.assertEqual(record['position_rows'][0]['proposed_burden_label'], 'Betrayal (model label)')
        self.assertNotIn('g1', [g['handle'] for g in record['factual_premise_claims']])
        self.assertEqual(record['authority'], 'MODEL_POSITION_PROJECTION_NOT_SOURCE_VERIFIED')

    def test_missing_and_duplicate_positions_remain_unknown_with_all_raw_rows(self):
        p = rich_packet()
        first = deepcopy(p['prior_argument']['burden_positions'][0])
        p['prior_argument']['burden_positions'] = [first, {**first, 'status': 'ABSENT'}]
        record = build_consequence_record(p)
        self.assertEqual([r['claimed_status'] for r in record['position_rows']], ['UNKNOWN', 'UNKNOWN'])
        self.assertEqual(len(record['position_rows'][0]['raw_positions']), 2)
        self.assertEqual(record['position_rows'][1]['raw_positions'], [])
        self.assertEqual(record['pair_changes'][0]['positions'][0]['change'], 'UNKNOWN')

    def test_unknown_ground_person_and_orphan_position_remain_inspectable(self):
        p = rich_packet(); prior = p['prior_argument']
        prior['persons'] = []
        prior['burden_positions'][0]['ground_handles'] = ['missing']
        prior['burden_positions'].append({'burden_id': 'unknown', 'action_id': 'Z',
            'status': 'PRESENT', 'position': 'An unsupported extra', 'ground_handles': []})
        record = build_consequence_record(p)
        self.assertIn('MISSING_OR_DUPLICATE_PERSON', record['position_rows'][0]['binding_issues'])
        self.assertIn('MISSING_OR_DUPLICATE_STATEMENT',
            record['position_rows'][0]['raw_positions'][0]['grounds'][0]['binding_issues'])
        self.assertEqual(record['unassigned_positions'][0]['raw']['position'], 'An unsupported extra')

    def test_exact_evidence_spans_without_repairing_bad_quotations(self):
        p = rich_packet(); p['source']['scenario'] += ' Under B, Bea gets aid.'
        p['source']['source_text_sha256'] = digest(p['source']['scenario'].encode())
        p['prior_argument']['statements'][0]['citations'][0]['quote'] = '"Under A, Ari gets aid."'
        record = build_consequence_record(p)
        g = record['position_rows'][0]['raw_positions'][0]['grounds'][0]
        self.assertIn('UNBOUND_SOURCE_QUOTE', g['binding_issues'])
        span = record['cited_source_spans'][0]
        self.assertEqual(len(span['occurrences']), 2)
        for hit in span['occurrences']:
            self.assertEqual(p['source']['scenario'][hit['start']:hit['end']], span['text'])
        self.assertIn('Fallible', render_consequences(record))

    def test_source_hash_mismatch_does_not_build_a_misbound_record(self):
        p = rich_packet(); p['source']['scenario'] += ' altered'
        with self.assertRaises(ValueError): build_consequence_record(p)

    def test_request_comparison_changes_only_record_and_fact_first_instruction(self):
        p = rich_packet(); a = bind_assignment(p, assignment()); before = deepcopy((p, a))
        requests = [consequence_request(p, a, arm, model='o3', reasoning_effort='low') for arm in ARMS]
        users = [json.loads(r['messages'][1]['content']) for r in requests]
        self.assertNotIn('consequence_record', users[0])
        record = users[1].pop('consequence_record')
        self.assertEqual(users[0], users[1])
        self.assertFalse(record['operative'])
        for key in ('model', 'reasoning_effort', 'max_completion_tokens', 'response_format'):
            self.assertEqual(requests[0][key], requests[1][key])
        self.assertNotIn('CONTENT_REVIEW', json.dumps(requests))
        self.assertNotIn('peer_arguments', users[0])
        self.assertEqual((p, a), before)

    def test_review_bindings_preserve_actual_complaint_without_retiring_it(self):
        p = rich_packet(); a = bind_assignment(p, assignment()); r = reviewed_reply()
        before = deepcopy((p, a)); r['disposition'] = 'ANSWERED'
        bound = bind_consequence_reply(p, a, r, build_consequence_record(p))
        self.assertEqual(bound['binding_issues'], [])
        self.assertFalse(bound['operative'])
        self.assertFalse(bound['self_report_is_resolution'])
        self.assertFalse(bound['review_status_is_semantic_certification'])
        self.assertEqual((p, a), before)

    def test_misquoted_corrections_and_unknown_rows_are_findings_not_factual_repairs(self):
        p = rich_packet(); a = bind_assignment(p, assignment()); r = reviewed_reply()
        review = r['factual_review'][0]
        review['source_quotes'] = ['An invented fact.']
        review['target']['quote'] = 'not in the complaint'
        review['consequence_row_ids'] = ['unknown']
        result = bind_consequence_reply(p, a, r, build_consequence_record(p))
        self.assertIn('REVIEW_TARGET_QUOTE_MISMATCH:0', result['binding_issues'])
        self.assertIn('REVIEW_SOURCE_QUOTE_MISMATCH:0', result['binding_issues'])
        self.assertIn('REVIEW_ROW_REFERENCE_MISMATCH:0', result['binding_issues'])
        self.assertEqual(result['raw_reply'], r)

    def test_baseline_cannot_cite_a_record_it_never_received(self):
        r = reviewed_reply(); r['factual_review'][0]['target'] = {
            'origin': 'CONSEQUENCE_RECORD', 'path': '/authority', 'quote': 'MODEL_POSITION'}
        issues = bind_consequence_reply(rich_packet(), bind_assignment(rich_packet(), assignment()), r)['binding_issues']
        self.assertIn('REVIEW_TARGET_PATH_MISMATCH:0', issues)
        self.assertIn('REVIEW_ROW_REFERENCE_MISMATCH:0', issues)

    def test_unestablished_premise_can_be_identified_without_manufacturing_an_absence_quote(self):
        r = reviewed_reply(); review = r['factual_review'][0]
        review.update(status='NOT_ESTABLISHED', source_quotes=[], consequence_row_ids=[])
        self.assertEqual(bind_consequence_reply(rich_packet(),
            bind_assignment(rich_packet(), assignment()), r)['binding_issues'], [])

    def test_preparation_is_offline_frozen_and_preserves_original_inputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); saved_probe(root/'saved')
            before = {p: p.read_bytes() for p in (root/'saved').rglob('*') if p.is_file()}
            protocol = prepare(root/'saved', root/'new', cases=['test_case'])
            self.assertEqual(protocol['maximum_provider_calls'], 2)
            self.assertEqual(protocol['selector_calls'], 0)
            verify_freeze(root/'new', 'inputs.freeze.json')
            self.assertFalse(list((root/'new').rglob('call.json')))
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_fake_execution_uses_saved_complaint_and_two_fresh_requests_only(self):
        calls = []
        class Response:
            def __init__(self, raw):
                self.raw = raw
                self.choices = [SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                    content=json.dumps(raw), refusal=None))]
            def model_dump(self, **kwargs):
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(self.raw)}}]}
        def create(**request):
            calls.append(request); raw = reviewed_reply()
            if len(calls) == 1: raw['factual_review'][0]['consequence_row_ids'] = []
            return Response(raw)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with TemporaryDirectory() as temp:
            root = Path(temp); saved_probe(root/'saved')
            prepare(root/'saved', root/'new', cases=['test_case'])
            rows = execute(root/'new', client_factory=lambda: client)
            self.assertEqual(len(calls), 2)
            self.assertEqual([r['status'] for r in rows], ['RETURNED', 'RETURNED'])
            for arm in ARMS: verify_freeze(root/'new/test_case'/arm, 'outputs.freeze.json')
            with self.assertRaises(ValueError): execute(root/'new', client_factory=lambda: client)
            self.assertEqual(len(calls), 2)

    def test_failed_call_is_retained_without_retry_or_secret_error_details(self):
        calls = []
        def create(**request):
            calls.append(request); raise RuntimeError('provider secret')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with TemporaryDirectory() as temp:
            root = Path(temp); saved_probe(root/'saved')
            prepare(root/'saved', root/'new', cases=['test_case'])
            rows = execute(root/'new', client_factory=lambda: client)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(r['status'] == 'FAILED' for r in rows))
            self.assertNotIn('provider secret', (root/'new/test_case/consequences_first/result.json').read_text())

    def test_tampered_input_is_detected_before_any_provider_call(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); saved_probe(root/'saved')
            prepare(root/'saved', root/'new', cases=['test_case'])
            (root/'new/test_case/packet.json').write_text('{}')
            with self.assertRaises(ValueError):
                execute(root/'new', client_factory=lambda: self.fail('Must not call provider'))


if __name__ == '__main__':
    unittest.main()
