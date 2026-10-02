import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import parsing_game_K as k


class KTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy, _ = k.train_policy()

    def structure(self, text):
        return k.parse_world_state(text, self.policy)

    def role_text(self, result, event, role):
        mentions = {m['mention_id']: m for m in result['entities']}
        return [mentions[ref]['text'] for ref in event['roles'].get(role, [])]

    def test_existing_causal_suites(self):
        for suite in (k.TRAIN_EXAMPLES, k.TEST_EXAMPLES, k.NEGATION_GENERALIZATION_EXAMPLES,
                      k.ROBUSTNESS_EXAMPLES, k.LEXICAL_HOLDOUT, k.UNKNOWN_HOLDOUT, k.EPISTEMIC_HOLDOUT):
            self.assertEqual(k.evaluate_suite(suite, self.policy)[0]['end_to_end_correct'], 1)
        self.assertEqual(k.evaluate_multi_suite(k.MULTI_RELATION_HOLDOUT, self.policy)[0]
                         ['exact_claim_list_accuracy'], 1)

    def test_inventory_and_unary_remaining(self):
        result = self.structure('A warehouse has two energy-saving generators remaining.')
        self.assertEqual([e['meaning'] for e in result['events']], ['possession', 'remaining'])
        self.assertEqual(self.role_text(result, result['events'][0], 'holder'), ['A warehouse'])
        theme = next(m for m in result['entities'] if m['head_text'] == 'generators')
        self.assertEqual(theme['quantities'][0]['value'], 2)
        self.assertEqual(result['events'][1]['roles']['theme'], result['events'][0]['roles']['theme'])
        self.assertTrue(all(not e['eligible_for_world_state'] for e in result['events']))

    def test_alternatives_roles_quantities_and_references(self):
        text = ('A courier can carry the package to a customer who ordered it yesterday, '
                'or reroute it to a depot where five workers face identical exposure.')
        result = self.structure(text)
        events = {e['predicate']['lemma']: e for e in result['events']}
        self.assertEqual(set(events), {'carry', 'order', 'reroute', 'face'})
        for name in ('carry', 'reroute'):
            self.assertEqual(self.role_text(result, events[name], 'actor'), ['A courier'])
            self.assertEqual(events[name]['assertion']['status'], 'possible')
        self.assertEqual(self.role_text(result, events['carry'], 'destination'), ['a customer'])
        self.assertEqual(self.role_text(result, events['reroute'], 'destination'), ['a depot'])
        group = result['alternatives'][0]
        self.assertEqual(group['branches'], [events['carry']['frame_id'], events['reroute']['frame_id']])
        self.assertEqual(group['exclusivity'], 'unspecified')
        self.assertEqual(group['attachment_status'], 'ambiguous')
        self.assertEqual(events['face']['context']['branch_id'], events['reroute']['frame_id'])
        self.assertEqual(events['order']['context']['branch_id'], events['carry']['frame_id'])
        self.assertTrue(any(m['quantities'] and m['quantities'][0]['value'] == 5 for m in result['entities']))
        pronouns = [m for m in result['entities'] if m['text'] == 'it']
        self.assertEqual(len(pronouns), 2)
        self.assertTrue(all(m['reference_status'] == 'unresolved' for m in pronouns))
        for mention in result['entities']:
            self.assertEqual(text[mention['start']:mention['end']], mention['text'])
        json.dumps(result)

    def test_passive_transfer_roles(self):
        result = self.structure('The package was sent to a clinic by a courier.')
        event = result['events'][0]
        self.assertEqual(self.role_text(result, event, 'actor'), ['a courier'])
        self.assertEqual(self.role_text(result, event, 'theme'), ['The package'])
        self.assertEqual(self.role_text(result, event, 'destination'), ['a clinic'])

    def test_modal_denied_and_conditional_never_commit(self):
        for text, status in [
            ('A courier may send a package to a clinic.', 'possible'),
            ('A courier did not send a package to a clinic.', 'denied'),
            ('If a courier sends a package, the clinic opens.', 'conditional'),
        ]:
            with self.subTest(text=text):
                result = self.structure(text)
                event = next(e for e in result['events'] if e['predicate']['lemma'] == 'send')
                self.assertEqual(event['assertion']['status'], status)
                self.assertFalse(event['eligible_for_world_state'])

    def test_copular_state(self):
        result = self.structure('The vehicle is stationary.')
        event = next(e for e in result['events'] if e['meaning'] == 'property')
        self.assertEqual(self.role_text(result, event, 'theme'), ['The vehicle'])
        self.assertEqual(self.role_text(result, event, 'attribute'), ['stationary'])

    def test_unknown_predicate_keeps_structure_without_semantic_guess(self):
        result = self.structure('Light modulates growth.')
        self.assertEqual(result['events'][0]['meaning'], 'unresolved')
        self.assertIn('frame_semantics_unresolved', result['events'][0]['issues'])
        self.assertTrue(result['events'][0]['roles'])

    def test_probe_storage_is_separate_and_rotates(self):
        self.assertEqual(k.USER_SUITE_PATH.name, 'parsing_game_K_user_probes.json')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probes.json'
            entries = []
            for i in range(6):
                entries = k.append_user_probe(entries, f'Warehouse {i} has supplies.')
            evaluated = k.evaluate_user_suite(entries, self.policy)
            k.save_user_suite(path, evaluated)
            saved = k.load_user_suite(path)
            self.assertEqual(len(saved), 5)
            self.assertNotIn('Warehouse 0', path.read_text())
            self.assertIn('structure', saved[0]['last_result'])
            with redirect_stdout(io.StringIO()) as output:
                k.print_user_suite(saved)
            self.assertIn('EVENTS / STATES', output.getvalue())
            self.assertIn('CAUSAL CLAIMS', output.getvalue())

    def test_quantity_change_scope_reference_and_modifier(self):
        result = self.structure('This policy sharply reduces total demand across rural districts and the neighboring region.')
        event = next(e for e in result['events'] if e['meaning'] == 'quantity_change')
        self.assertEqual(event['change']['direction'], 'decrease')
        self.assertEqual(self.role_text(result, event, 'theme'), ['total demand'])
        self.assertEqual(self.role_text(result, event, 'scope'), ['rural districts and the neighboring region'])
        self.assertIsNone(event['change']['magnitude'])
        self.assertIn('unresolved_reference', event['issues'])
        self.assertEqual(event['modifiers'][0]['text'], 'sharply')
        self.assertTrue(all(c['relation_type'] != 'causal' for c in result['causal_claims']))

    def test_change_voice_scope_and_intransitive_roles(self):
        for text, direction, status in [
            ('Insulation reduces noise.', 'decrease', 'asserted'),
            ('Insulation may reduce noise.', 'decrease', 'possible'),
            ('Insulation does not reduce noise.', 'decrease', 'denied'),
            ('Heating increases pressure.', 'increase', 'asserted'),
        ]:
            event = next(e for e in self.structure(text)['events'] if e['meaning'] == 'quantity_change')
            self.assertEqual(event['change']['direction'], direction)
            self.assertEqual(event['assertion']['status'], status)
            self.assertFalse(event['eligible_for_world_state'])
        result = self.structure('Noise was reduced by insulation.')
        self.assertEqual(self.role_text(result, result['events'][0], 'theme'), ['Noise'])
        self.assertEqual(self.role_text(result, result['events'][0], 'influence'), ['insulation'])
        result = self.structure('Demand decreased.')
        self.assertEqual(self.role_text(result, result['events'][0], 'theme'), ['Demand'])
        self.assertEqual(result['events'][0]['change']['influence'], [])


if __name__ == '__main__':
    unittest.main()
