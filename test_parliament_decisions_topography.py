import json
import unittest
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from parliament_decisions_topography import (
    JACKKNIFE_FOLDS, QUESTIONS, ablate_packet_text, apply_to_report,
    hinging_record, jackknife_interval, annotate_replicate_influence,
    parliament_packet, parse_answers, topography_from_decisions,
)


class DecisionsTopographyTests(unittest.TestCase):
    def test_original_five_framework_trace_is_recovered_from_manifest(self):
        from parliament_topography import native_trace_from_report
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ['utilitarian','deontological','virtue','care','rawlsian']
            trace = {'scenario':'The original scenario.', 'cycles':[{'candidates':[
                {'specialist':name} for name in names]}]}
            (root/'native.json').write_text(json.dumps(trace))
            (root/'preparation_manifest.json').write_text(json.dumps({
                'deliberation':{'trace':'native.json'}}))
            report = {'scenario':'The original scenario.',
                      'artifacts':['preparation_manifest.json']}
            self.assertEqual(native_trace_from_report(report, root), trace)
            report['scenario'] = 'A different scenario.'
            self.assertIsNone(native_trace_from_report(report, root))

    def actions(self):
        return [{'id': 'A0', 'label': 'give Ben'}, {'id': 'A1', 'label': 'give Cara'}]

    def answers(self, ident, care, welfare, obligation):
        return [
            {'type': 'predicate', 'name': ident + '_care', 'probability': care},
            {'type': 'predicate', 'name': ident + '_welfare', 'probability': welfare},
            {'type': 'predicate', 'name': ident + '_obligation', 'probability': obligation},
        ]

    def test_packet_omits_local_paths_and_keeps_framework_records(self):
        packet = parliament_packet(
            'Ada can give the medicine to Ben or Cara.',
            {'actions': [{'action_id': 'A0', 'intervention': 'give Ben', 'effect_ids': ['E1']}],
             'effects': [{'effect_id': 'E1', 'outcome': 'Ben lives', 'party_id': 'P3'}],
             'parties': [{'party_id': 'P1', 'label': 'Ada', 'kind': 'PERSON'}]},
            {'cycles': [{'candidates': [{'specialist': 'utilitarian',
                                         'recommended_action': '',
                                         'framework_vote_reason': 'framework did not select a live action',
                                         'trace_path': '/Users/benjaminkowal/secret.json',
                                         'expected_value_estimates': {'give Ben': {'value': 0.95}}}]}],
             'judgment_status': 'UNRESOLVED'},
            [{'framework': 'utilitarian', 'position': 'ABSTAIN', 'preference': None}])
        text = json.dumps(packet)
        self.assertNotIn('/Users/', text)
        self.assertNotIn('trace_path', text)
        self.assertIn('framework did not select a live action', text)
        self.assertEqual(packet['world']['actions'][0]['action_id'], 'A0')

    def test_refusal_is_a_hole_not_zero(self):
        points = parse_answers(self.actions()[0], [
            {'type': 'predicate', 'name': 'A0_care', 'probability': 0.8},
            {'type': 'refusal', 'name': 'A0_welfare'},
            {'type': 'predicate', 'name': 'A0_obligation', 'probability': 0.0},
        ])
        self.assertEqual(points['care']['probability'], 0.8)
        self.assertFalse(points['welfare']['present'])
        self.assertNotIn('probability', points['welfare'])
        self.assertEqual(points['obligation']['probability'], 0.0)
        self.assertTrue(points['obligation']['present'])

    def test_argument_adapter_keeps_questions_and_qualifications_in_packet(self):
        view = {'framework': 'contractualist', 'position': 'CLARIFICATION',
                'preference': None, 'ledger_status': 'NOT_APPLICABLE_ARGUMENT_ADAPTER',
                'critical_questions': [{'question': 'Was there reliance?'}],
                'assumption_records': [{'text': 'Priority remains contested.'}],
                'argument_technical_status': 'STAGED_PARTIAL_ARGUMENT',
                'recorded_argument': {'conclusion': 'No unconditional ranking.'},
                'dependencies': ['Reliance is unknown.'],
                'critical_question_findings': [],
                'trace_path': '/Users/private/trace.json'}
        packet = parliament_packet('A rescue.', {}, {}, [view])
        saved = packet['frameworks'][0]
        for key in ('critical_questions', 'assumption_records', 'recorded_argument',
                    'argument_technical_status', 'dependencies'):
            self.assertEqual(saved[key], view[key])
        self.assertIsNone(saved['preference'])
        self.assertEqual(saved['ledger_status'], 'NOT_APPLICABLE_ARGUMENT_ADAPTER')
        self.assertNotIn('/Users/', json.dumps(packet))
        self.assertNotIn('rawlsian', json.dumps(packet))

    def test_plotted_coordinates_are_probabilities_not_ordinals(self):
        per = {
            'A0': parse_answers(self.actions()[0], self.answers('A0', 0.81, 0.11, 0.77)),
            'A1': parse_answers(self.actions()[1], self.answers('A1', 0.40, 0.93, 0.22)),
        }
        native = {'actions': [{'id': 'A0', 'label': 'give Ben',
                               'care': {'present': True, 'ordinal': 2, 'verdict': 'RESPONSIVE'},
                               'welfare': {'present': True, 'value': 0.95},
                               'obligation': {'present': False}}],
                  'framework_arrows': [{'framework': 'care', 'action_id': 'A0'}]}
        topo = topography_from_decisions(self.actions(), per, native=native)
        self.assertEqual(topo['version'], 'parliament-topography/2')
        self.assertEqual(topo['axes']['care']['encoding'], 'probability')
        a0 = next(row for row in topo['actions'] if row['id'] == 'A0')
        self.assertEqual(a0['care']['probability'], 0.81)
        self.assertNotIn('ordinal', a0['care'])
        self.assertEqual(a0['native']['care']['verdict'], 'RESPONSIVE')
        self.assertEqual(a0['native']['welfare']['value'], 0.95)
        self.assertEqual(topo['flags']['maxima']['welfare'], 'A1')
        self.assertEqual(topo['flags']['maxima']['obligation'], 'A0')
        self.assertIn('A0', topo['flags']['attractors'])

    def test_missing_answers_do_not_invent_an_attractor(self):
        topo = topography_from_decisions(self.actions(), {})
        self.assertEqual(topo['status'], 'UNAVAILABLE')
        self.assertEqual(topo['actions'], [])
        self.assertEqual(topo['flags']['attractors'], [])

    def test_apply_uses_mocked_decisions_and_does_not_call_network(self):
        report = {'scenario': 'Ada can give the medicine to Ben or Cara.',
                  'status': 'FRAMEWORK_PARTIAL',
                  'world': {'actions': [{'action_id': 'A0', 'intervention': 'give Ben'},
                                        {'action_id': 'A1', 'intervention': 'give Cara'}]},
                  'frameworks': [{'framework': 'contractualist',
                                  'preference': 'give Ben',
                                  'ledger_status': 'NOT_APPLICABLE_ARGUMENT_ADAPTER',
                                  'critical_questions': [{'question': 'Was there reliance?'}]}],
                  'cycles': [{'candidates': [{'specialist': 'care'}]}]}
        payloads = []

        def opener(request, timeout=120):
            payloads.append(json.loads(request.data.decode()))
            ident = 'A0' if 'A0 —' in payloads[-1]['input'] or 'A0 —' in json.dumps(payloads[-1]) else None
            if ident is None:
                ident = 'A0' if len(payloads) == 1 else 'A1'
            body = json.dumps({'answers': self.answers(ident, 0.7, 0.6, 0.5)}).encode()
            return BytesIO(body)

        class Handle(BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        def wrapped(request, timeout=120):
            ident = 'A0' if len(payloads) == 0 else 'A1'
            payload = json.loads(request.data.decode())
            payloads.append(payload)
            names = [q['name'] for q in payload['questions']]
            ident = names[0].split('_')[0]
            return Handle(json.dumps({'answers': self.answers(ident, 0.7, 0.6, 0.5)}).encode())

        updated = apply_to_report(report, Path('.'), 'sk-test', opener=wrapped)
        self.assertEqual(updated['topography']['axes']['care']['encoding'], 'probability')
        self.assertEqual(len(payloads), 2)
        self.assertTrue(all(row['model'] == 'gpt-6-luna' for row in payloads))
        self.assertIn(QUESTIONS['obligation'][:40], json.dumps(payloads))
        self.assertIn('Was there reliance?', payloads[0]['input'])
        self.assertFalse(any(row['framework'] == 'contractualist'
                             for row in updated['topography']['framework_arrows']))
        self.assertEqual(set(updated['topography']['axes']), {'care', 'welfare', 'obligation'})

    def test_ablation_drops_disjoint_tenths_covering_the_packet(self):
        text = ''.join(chr(65 + (i % 26)) for i in range(103))
        omitted, rebuilt = [], ''
        for fold in range(JACKKNIFE_FOLDS):
            kept, meta = ablate_packet_text(text, fold)
            chunk = text[meta['omitted_start']:meta['omitted_end']]
            self.assertEqual(kept, text[:meta['omitted_start']] + text[meta['omitted_end']:])
            self.assertEqual(len(kept) + meta['omitted_chars'], len(text))
            omitted.append(chunk)
            rebuilt += chunk
        self.assertEqual(rebuilt, text)
        self.assertEqual(sum(len(chunk) for chunk in omitted), len(text))

    def test_jackknife_does_not_treat_missing_folds_as_zero(self):
        identical = jackknife_interval([0.4] * 10)
        self.assertAlmostEqual(identical['mean'], 0.4)
        self.assertEqual(identical['se'], 0.0)
        self.assertAlmostEqual(identical['low'], 0.4)
        self.assertAlmostEqual(identical['high'], 0.4)
        self.assertIsNone(jackknife_interval([0.8, 0.7, None, None, None, None, None, None, None, None]))
        self.assertIsNone(jackknife_interval([0.0] * 7 + [None] * 3))

    def test_overlapping_jackknife_intervals_do_not_invent_a_unique_maximum(self):
        def cell(mean, low, high):
            return {'present': True, 'probability': mean, 'encoding': 'probability', 'source': 'decisions/v1',
                    'interval': {'method': 'jackknife', 'low': low, 'high': high, 'mean': mean,
                                 'folds': 10, 'folds_used': 10}}
        per = {
            'A0': {'care': cell(0.50, 0.30, 0.70), 'welfare': cell(0.95, 0.90, 1.00),
                   'obligation': cell(0.45, 0.20, 0.70)},
            'A1': {'care': cell(0.55, 0.40, 0.80), 'welfare': cell(0.05, 0.00, 0.10),
                   'obligation': cell(0.50, 0.25, 0.75)},
        }
        topo = topography_from_decisions(self.actions(), per)
        self.assertIsNone(topo['flags']['maxima']['care'])
        self.assertEqual(topo['flags']['maxima']['welfare'], 'A0')
        self.assertIsNone(topo['flags']['maxima']['obligation'])
        self.assertEqual(topo['flags']['attractors'], [])
        a0 = next(row for row in topo['actions'] if row['id'] == 'A0')
        self.assertEqual(a0['care']['interval']['low'], 0.30)
        self.assertNotEqual(a0['care']['probability'], 0)

    def test_jackknife_evaluate_ablates_packet_text_ten_times_per_action(self):
        report = {'scenario': 'Trolley. ' + ('track ' * 80),
                  'status': 'FRAMEWORK_PARTIAL',
                  'world': {'actions': [{'action_id': 'A0', 'intervention': 'divert'},
                                        {'action_id': 'A1', 'intervention': 'refrain'}]},
                  'frameworks': [],
                  'cycles': [{'candidates': [{'specialist': 'utilitarian'}]}]}
        payloads = []

        class Handle(BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        def wrapped(request, timeout=120):
            payload = json.loads(request.data.decode())
            payloads.append(payload)
            ident = payload['questions'][0]['name'].split('_')[0]
            fold = sum(1 for row in payloads if row['questions'][0]['name'].startswith(ident + '_')) - 1
            body = json.dumps({'answers': self.answers(ident, 0.40 + fold * 0.02, 0.90, 0.30 + fold * 0.01)}).encode()
            return Handle(body)

        updated = apply_to_report(report, Path('.'), 'sk-test', opener=wrapped, jackknife_folds=JACKKNIFE_FOLDS)
        self.assertEqual(len(payloads), 20)
        a0_inputs = [row['input'] for row in payloads if row['questions'][0]['name'].startswith('A0_')]
        self.assertEqual(len(a0_inputs), 10)
        self.assertGreater(len(set(a0_inputs)), 1)
        self.assertTrue(all('ACTION UNDER EVALUATION: A0' in text for text in a0_inputs))
        care = next(row for row in updated['topography']['actions'] if row['id'] == 'A0')['care']
        self.assertTrue(care['present'])
        self.assertEqual(care['interval']['method'], 'jackknife')
        self.assertEqual(care['interval']['folds_used'], 10)
        self.assertEqual(care['interval']['low'] <= care['probability'] <= care['interval']['high'], True)
        self.assertEqual(updated['topography']['provenance']['new_model_calls'], 20)
        self.assertEqual(updated['topography']['axes']['care']['interval'], 'jackknife')
        blocks = updated['topography']['provenance']['jackknife']['packet_blocks']
        self.assertEqual(len(blocks), 10)
        self.assertEqual(''.join(block['omitted_text'] for block in blocks),
                         updated['topography']['provenance']['jackknife']['packet_text'])
        self.assertNotIn('/Users/', json.dumps(blocks))
        hinge = care['interval']['hinging']
        self.assertTrue(hinge['present'])
        self.assertIn('omitted_text', hinge)
        self.assertEqual(hinge['fold'], 0)
        self.assertAlmostEqual(hinge['abs_delta'], 0.09)
        self.assertEqual(len(hinge['ranking']), 10)
        self.assertNotIn('omitted_text', care['interval']['replicates'][0])
        self.assertIn('delta_from_mean', care['interval']['replicates'][0])

    def test_most_hinging_tenth_is_the_largest_move_and_does_not_treat_gaps_as_zero(self):
        replicates = [{'fold': i, 'probability': 0.5, 'omitted_text': 'block-%s' % i,
                       'omitted_start': i * 4, 'omitted_end': i * 4 + 4, 'omitted_chars': 4}
                      for i in range(10)]
        replicates[3]['probability'] = 0.1
        annotate_replicate_influence(replicates, 0.46)
        hinge = hinging_record(replicates, mean=0.46)
        self.assertEqual(hinge['fold'], 3)
        self.assertEqual(hinge['omitted_text'], 'block-3')
        self.assertAlmostEqual(hinge['abs_delta'], 0.36)
        self.assertEqual(hinge['effect_when_present'], 'raises_estimate')
        missing = [{'fold': i} for i in range(10)]
        missing[0]['probability'] = 0.8
        missing[1]['probability'] = 0.7
        annotate_replicate_influence(missing, 0.75)
        self.assertNotIn('abs_delta', missing[2])
        self.assertIsNone(missing[2].get('probability'))
        identical = [{'fold': i, 'probability': 1.0} for i in range(10)]
        annotate_replicate_influence(identical, 1.0)
        self.assertFalse(hinging_record(identical, mean=1.0)['present'])


if __name__ == '__main__':
    unittest.main()
