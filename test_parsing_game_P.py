import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import parsing_game_P as p
import test_parsing_game_N as previous
import test_parsing_game_M as legacy


class PInheritedTests(previous.NTests):
    @classmethod
    def setUpClass(cls):
        replacement = patch.object(previous, 'n', p)
        replacement.start()
        cls.addClassCleanup(replacement.stop)
        super().setUpClass()

    def test_path_adjunct_known_gap_not_a_patient(self):
        # Promote the repaired N expected failure to an ordinary regression.
        frame, = p.extract_proposition_frames('The bat flew by the blind man.')
        self.assertIsNone(frame.object)
        self.assertEqual(frame.attachments[0]['role'], 'unresolved')

    def test_quantity_and_legacy_causal_suites(self):
        for suite in (p.TRAIN_EXAMPLES, p.NEGATION_GENERALIZATION_EXAMPLES,
                      p.ROBUSTNESS_EXAMPLES, p.LEXICAL_HOLDOUT,
                      p.UNKNOWN_HOLDOUT, p.EPISTEMIC_HOLDOUT):
            self.assertEqual(p.evaluate_suite(suite, self.policy)[0]['end_to_end_correct'], 1)
        metrics, records = p.evaluate_suite(p.TEST_EXAMPLES, self.policy)
        failures = [r for r in records if not r['end_to_end_correct']]
        self.assertEqual([r['sentence'] for r in failures], ['Insomnia caused by stress yesterday.'])
        self.assertFalse(failures[0]['claim']['eligible_for_world_state'])
        self.assertEqual(p.evaluate_multi_suite(p.MULTI_RELATION_HOLDOUT, self.policy)[0]
                         ['exact_claim_list_accuracy'], 1)
        self.assertEqual(self.parse('Insulation reduces noise.')['events'][0]['change']['direction'], 'decrease')

    def test_rolling_results_serialize_entailment_and_display_it(self):
        self.assertEqual(p.USER_SUITE_PATH.name, 'parsing_game_P_user_probes.json')
        entries = p.evaluate_user_suite(p.append_user_probe([], 'After the game the players all went home'), self.policy)
        frame = entries[0]['last_result']['propositions'][0]
        self.assertEqual(len(frame['attachments']), 2)
        self.assertEqual(frame['quantification'][0]['text'], 'all')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probes.json'
            p.save_user_suite(path, entries)
            self.assertEqual(p.load_user_suite(path), entries)

    def test_twelve_contrastive_role_sentences(self):
        cases = [
            ('After the game the players all went home', None, 'asserted_in_text', ['temporal', 'destination']),
            ('The players went home after the game.', None, 'asserted_in_text', ['destination', 'temporal']),
            ('Before the rehearsal, the musicians went home.', None, 'asserted_in_text', ['temporal', 'destination']),
            ('The hikers walked to the shelter.', None, 'asserted_in_text', ['destination']),
            ('The players went home.', None, 'asserted_in_text', ['destination']),
            ('The players did not go home after the game.', None, 'denied_in_text', ['destination', 'temporal']),
            ('The players may go home after the game.', None, 'unknown', ['destination', 'temporal']),
            ('Each sailor returned home.', None, 'asserted_in_text', ['destination']),
            ('The bat flew by the blind man.', None, 'asserted_in_text', ['unresolved']),
            ('The ball was thrown by the player.', None, 'asserted_in_text', ['agent']),
            ('The players carried the trophy after the game.', 'the trophy', 'asserted_in_text', ['temporal']),
            ('The musicians carried the instruments before the rehearsal.', 'the instruments', 'asserted_in_text', ['temporal']),
        ]
        for text, obj, occurrence, roles in cases:
            with self.subTest(text=text):
                result = self.parse(text)
                frame, = result['propositions']
                self.assertEqual(frame['object']['text'] if frame['object'] else None, obj)
                self.assertEqual(frame['occurrence_status'], occurrence)
                self.assertEqual([a['role'] for a in frame['attachments']], roles)
                self.assertFalse(any(c['eligible_for_world_state'] for c in result['causal_claims']))
                for a in frame['attachments']:
                    context = a['argument_context']
                    if context['status'] == 'complete':
                        self.assertEqual(''.join(context['regions'][k]['text'] for k in
                                                ('left','argument_1','between','argument_2','right')),
                                         context['window']['text'])

    def test_quantified_causation_is_asserted_but_not_committed(self):
        result = self.parse('All storms cause flooding.')
        self.assertEqual(result['propositions'][0]['assertion']['status'], 'asserted')
        self.assertTrue(result['propositions'][0]['quantification'])
        claim, = result['causal_claims']
        self.assertFalse(claim['eligible_for_world_state'])
        self.assertIn('quantified_argument_requires_scope_validation', claim['validation_reasons'])


class PEmbeddingTests(legacy.AdversarialEmbeddingTests):
    @classmethod
    def setUpClass(cls):
        replacement = patch.object(legacy, 'm', p)
        replacement.start()
        cls.addClassCleanup(replacement.stop)
        super().setUpClass()

    def test_11_conditional_boundary_known_gap(self):
        # Removing positional object invention also repairs this boundary leak.
        super().test_11_conditional_boundary_known_gap()


if __name__ == '__main__':
    unittest.main()
