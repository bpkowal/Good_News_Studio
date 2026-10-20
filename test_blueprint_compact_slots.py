"""Semantic checks for reusing small cloze spans outside allocation/omission."""
import unittest

from blueprint_cloze_chooser import (
    _BY_ID, _cloze_prompt, _risk_graph, _score_template,
)
from blueprint_discourse import build_promise_reliance, build_disputed_report


class CompactSlotTests(unittest.TestCase):
    def score(self, family, text, answers):
        return _score_template(text, _BY_ID[family], answers, 0)

    def test_actor_is_the_condition_actor_not_the_outcome_victim(self):
        text = 'If Ada opens the gate, Ben might drown.'
        row = self.score('uncertain_risk', text, {
            'actor': text[:-1], 'action': text[:-1],
            'possible_outcome': text[:-1], 'affected_party': 'Ben', 'likelihood': 'might',
        })
        self.assertEqual(row['slots']['actor'], 'Ada')
        self.assertEqual(row['slots']['action'], 'opens the gate')
        self.assertEqual(row['slots']['possible_outcome'], 'Ben might drown')
        self.assertEqual(row['items'][0]['completion'], text[:-1])

    def test_long_and_compact_risk_fills_produce_the_same_scoped_world(self):
        text = 'If Ada opens the gate, Ben might drown.'
        short = {'actor': 'Ada', 'action': 'opens the gate',
                 'possible_outcome': 'Ben might drown', 'affected_party': 'Ben', 'likelihood': 'might'}
        long = {**short, **{key: text[:-1] for key in ('actor', 'action', 'possible_outcome')}}
        a = self.score('uncertain_risk', text, short)
        b = self.score('uncertain_risk', text, long)
        world, notes = _risk_graph(text, b['slots'])
        self.assertEqual((world, notes), _risk_graph(text, a['slots']))
        self.assertEqual(world['effects'][-1]['modality'], 'POSSIBLE')
        self.assertTrue(world['effects'][-1]['condition_ids'])
        self.assertEqual(world['parties'][0]['label'], 'Ada')

    def test_rescue_actions_and_outcomes_keep_their_own_branches(self):
        text = ('Maria can save either the child or the dog, but not both. '
                'If Maria saves the child, the child will live. '
                'If Maria saves the dog, the dog will live.')
        row = self.score('rescue_contrast', text, {
            'rescuer': 'Maria can save', 'first_saved': 'the child', 'second_saved': 'the dog',
            'rescue_exclusivity': 'but not both',
            'first_rescue_action': 'If Maria saves the child, the child will live',
            'second_rescue_action': 'If Maria saves the dog, the dog will live',
            'first_benefit': 'If Maria saves the child, the child will live',
            'second_benefit': 'If Maria saves the dog, the dog will live',
        })
        self.assertEqual(row['status'], 'FILLED')
        self.assertEqual(row['slots']['first_rescue_action'], 'saves the child')
        self.assertEqual(row['slots']['second_rescue_action'], 'saves the dog')
        self.assertEqual(row['slots']['first_benefit'], 'the child will live')
        self.assertEqual(row['slots']['second_benefit'], 'the dog will live')
        self.assertNotIn('first_harm', row['slots'])
        for item in row['items']:
            if item['verdict'] == 'accepted':
                self.assertIn(item['span'], text)
                self.assertEqual(item['hypotheses'][0]['source_span'], item['span'])

    def test_other_families_reuse_actor_spans_without_flattening_content(self):
        cases = [
            ('ability_permission', 'Ada may deliver medicine to Ben.', 'actor', 'Ada may deliver medicine to Ben'),
            ('deontic_rule', 'Ada must not deliver medicine to Ben.', 'bearer', 'Ada must not deliver medicine to Ben'),
            ('promise_reliance', 'Ada promised Ben that Ada would deliver medicine.', 'promisor',
             'Ada promised Ben that Ada would deliver medicine'),
            ('disputed_report', 'Ada claims Ben could die.', 'source', 'Ada claims Ben could die'),
            ('conditional_outcome', 'If Ada acts, Ben will live.', 'actor', 'If Ada acts, Ben will live'),
            ('diversion_redirection', 'Maria can divert the trolley toward one worker.', 'actor',
             'Maria can divert the trolley toward one worker'),
        ]
        for family, text, slot, fill in cases:
            with self.subTest(family=family):
                row = self.score(family, text, {slot: fill})
                self.assertEqual(row['slots'][slot], 'Ada' if family != 'diversion_redirection' else 'Maria')
        text = 'Ada promised Ben that Ada would not deliver medicine.'
        row = self.score('promise_reliance', text, {
            'promisor': text[:-1], 'promisee': 'Ben', 'commitment_event': 'promised',
            'commitment_content': 'that Ada would not deliver medicine',
        })
        world, _ = build_promise_reliance(text, row['slots'])
        self.assertEqual(world['propositions'][0]['predication'], 'that Ada would not deliver medicine')
        self.assertEqual(world['effects'], [])
        text = 'Ada claims Ben could die.'
        row = self.score('disputed_report', text, {
            'source': text[:-1], 'report_words': 'claims', 'reported_content': 'Ben could die',
        })
        world, _ = build_disputed_report(text, row['slots'])
        self.assertEqual(world['propositions'][0]['status'], 'ATTRIBUTED')
        self.assertEqual(world['propositions'][0]['predication'], 'Ben could die')
        self.assertEqual(world['effects'], [])

    def test_action_keeps_negation_modality_and_complement_content(self):
        text = 'Ada may not open the gate. Ada tries to rescue Ben.'
        row = self.score('uncertain_risk', text, {
            'actor': 'Ada', 'action': 'Ada may not open the gate',
            'second_action': 'Ada tries to rescue Ben',
        })
        self.assertEqual(row['slots']['action'], 'may not open the gate')
        # The one-predicate helper must not silently drop a required complement.
        self.assertEqual(row['slots']['second_action'], 'Ada tries to rescue Ben')

    def test_patient_is_not_the_harming_process_and_keeps_quantity(self):
        text = 'Maria can divert the trolley. The trolley might kill three workers.'
        row = self.score('diversion_redirection', text, {
            'actor': 'Maria', 'intervention': 'divert the trolley',
            'controllable_process': 'The trolley might kill three workers',
            'affected_party': 'The trolley might kill three workers',
            'outcome': 'The trolley might kill three workers', 'uncertainty': 'might',
        })
        self.assertEqual(row['slots']['controllable_process'], 'The trolley')
        self.assertEqual(row['slots']['affected_party'], 'three workers')
        self.assertEqual(row['slots']['outcome'], 'The trolley might kill three workers')

    def test_existing_strong_templates_and_unknown_fills_are_unchanged(self):
        for family in ('exclusive_allocation', 'omission_harm'):
            self.assertFalse(any(i.get('span_shape') for i in _BY_ID[family]['items']))
        for family in _BY_ID:
            row = self.score(family, 'Ada waits.', {})
            self.assertEqual(row['slots'], {})
        row = self.score('uncertain_risk', 'Ada opens the gate.', {'actor': 'invented person'})
        self.assertNotIn('actor', row['slots'])
        self.assertIn('only the action phrase', _cloze_prompt('', _BY_ID['uncertain_risk']))


if __name__ == '__main__':
    unittest.main()
