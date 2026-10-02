import json
import tempfile
import unittest
from pathlib import Path

import parsing_game_N as n
import parsing_game_O as o


REGIONS = ('left', 'argument_1', 'between', 'argument_2', 'right')


def without_context(value):
    if isinstance(value, dict):
        return {k: without_context(v) for k, v in value.items()
                if k not in {'argument_context', 'argument_contexts'}}
    if isinstance(value, list):
        return [without_context(v) for v in value]
    return value


class OTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy, _ = n.train_policy()

    def assert_lossless(self, text, context):
        self.assertEqual(context['status'], 'complete')
        regions = context['regions']
        self.assertEqual(''.join(regions[k]['text'] for k in REGIONS), context['window']['text'])
        for region in [context['window']] + list(regions.values()):
            self.assertEqual(region['text'], text[region['start']:region['end']])
            for token in region['tokens']:
                self.assertEqual(token['text'], text[token['start']:token['end']])
                self.assertTrue({'lemma', 'pos', 'dep', 'token_index', 'head_index'} <= token.keys())

    def test_active_passive_and_intervening_contrast(self):
        for text, between in (
            ('Rain caused flooding by noon.', ' caused '),
            ('Flooding was caused by rain.', ' was caused by '),
            ('Rain not snow caused flooding.', ' not snow caused '),
        ):
            with self.subTest(text=text):
                context = o.collect_evidence(text)['argument_context']
                self.assert_lossless(text, context)
                self.assertEqual(context['regions']['between']['text'], between)

    def test_reversed_input_roles_keep_surface_order_and_identity(self):
        text = 'Flooding was caused by rain.'
        doc = o.get_nlp()(text)
        start = text.index('rain')
        context = o.build_argument_context(doc, [dict(start=start, end=start + 4, role='cause'),
                                                dict(start=0, end=8, role='effect')])
        self.assert_lossless(text, context)
        self.assertEqual(context['regions']['argument_1']['input_slot'], 1)
        self.assertEqual(context['regions']['argument_2']['reference']['role'], 'cause')

    def test_multiword_repeated_mentions_unicode_spacing_and_punctuation(self):
        text = '“Heavy rain”  caused severe flooding; heavy rain persisted.\n'
        doc = o.get_nlp()(text)
        start = text.index('severe flooding')
        context = o.build_argument_context(doc, [dict(start=1, end=11),
                                                dict(start=start, end=start + len('severe flooding'))])
        self.assert_lossless(text, context)
        self.assertEqual(context['regions']['between']['text'], '”  caused ')
        self.assertEqual(context['regions']['right']['text'], '; heavy rain persisted.\n')
        start2 = text.index('heavy rain')
        repeated = o.build_argument_context(doc, [dict(start=1, end=11),
                                                 dict(start=start2, end=start2 + 10)])
        self.assert_lossless(text, repeated)
        self.assertEqual(repeated['regions']['argument_2']['start'], start2)

    def test_missing_overlapping_and_invalid_spans_are_explicit(self):
        doc = o.get_nlp()('Heavy rain caused flooding.')
        for args, status in (
            ([None, dict(start=18, end=26)], 'missing_arguments'),
            ([dict(start=0, end=10), dict(start=6, end=10)], 'overlapping_arguments'),
            ([dict(start=1, end=10), dict(start=18, end=26)], 'invalid_or_out_of_window_arguments'),
        ):
            context = o.build_argument_context(doc, args)
            self.assertEqual(context['status'], status)
            self.assertEqual(context['regions'], {})
            self.assertEqual(context['window']['text'], doc.text)
        self.assertEqual(o.collect_evidence('')['argument_context']['status'], 'missing_arguments')

    def test_local_sentence_window_has_global_offsets(self):
        text = 'Birds sang. Rain caused flooding.'
        record = next(r for r in o.collect_claim_evidence(text)
                      if r['argument_context']['window']['text'] == 'Rain caused flooding.')
        context = record['argument_context']
        self.assert_lossless(text, context)
        self.assertEqual(context['window']['text'], 'Rain caused flooding.')
        self.assertGreater(context['window']['start'], 0)

    def test_proposition_context_preserves_subtree_and_target_id(self):
        text = 'The flood caused the library to close.'
        result = o.parse_world_state(text, self.policy)
        parent, child = result['propositions']
        context, = parent['argument_contexts']
        self.assert_lossless(text, context)
        self.assertEqual(context['regions']['argument_2']['text'], 'the library to close')
        self.assertEqual(context['target_proposition_id'], child['frame_id'])
        self.assertEqual(context['regions']['argument_2']['reference']['kind'], 'proposition')
        self.assertEqual(child['argument_contexts'][0]['status'], 'missing_arguments')

    def test_nested_context_is_observational(self):
        text = 'The worker tried to persuade Alice to leave.'
        result = o.parse_world_state(text, self.policy)
        for frame in result['propositions'][:2]:
            self.assert_lossless(text, frame['argument_contexts'][0])
        self.assertEqual([r['relation_type'] for r in result['complement_relations']],
                         ['ATTEMPT', 'UNRESOLVED'])
        self.assertEqual([p['occurrence_status'] for p in result['propositions']],
                         ['asserted_in_text', 'unknown', 'unknown'])

    def test_unchanged_n_predictions_features_and_world_output(self):
        suites = (n.TRAIN_EXAMPLES, n.TEST_EXAMPLES, n.NEGATION_GENERALIZATION_EXAMPLES,
                  n.ROBUSTNESS_EXAMPLES, n.LEXICAL_HOLDOUT, n.UNKNOWN_HOLDOUT,
                  n.EPISTEMIC_HOLDOUT, n.MULTI_RELATION_HOLDOUT)
        texts = {example['sentence'] if isinstance(example, dict) else example[0]
                 for suite in suites for example in suite}
        texts.update(('The worker tried and left.', 'The worker may try to leave.',
                      'The worker tried to persuade Alice to leave.',
                      'The bat flew by the blind man.',
                      'The museum was forced to close by the outage.',
                      'If the outage caused the museum to close, refunds would follow.'))
        self.assertEqual(o.FEATURE_NAMES, n.FEATURE_NAMES)
        for text in sorted(texts):
            with self.subTest(text=text):
                expected = n.parse_world_state(text, self.policy)
                actual = without_context(o.parse_world_state(text, self.policy))
                self.assertEqual(actual.pop('schema_version'), 4)
                expected.pop('schema_version')
                self.assertEqual(actual, expected)

    def test_rolling_suite_serializes_context_and_retains_five(self):
        self.assertEqual(o.USER_SUITE_PATH.name, 'parsing_game_O_user_probes.json')
        entries = []
        for i in range(7):
            entries = o.append_user_probe(entries, f'Worker {i} tried to leave.')
        self.assertEqual(len(entries), 5)
        entries = o.evaluate_user_suite(entries, self.policy)
        self.assertIn('argument_contexts', entries[0]['last_result']['propositions'][0])
        self.assertIn('argument_context', entries[0]['last_result']['claims'][0]['evidence'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probes.json'
            o.save_user_suite(path, entries)
            self.assertEqual(o.load_user_suite(path), entries)
            json.loads(path.read_text())


if __name__ == '__main__':
    unittest.main()
