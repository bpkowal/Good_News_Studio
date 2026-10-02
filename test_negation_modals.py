import copy
import json
from pathlib import Path
import unittest

from parsing_game_T import export_candidate_graph, validate_candidate_selection
from test_scope_semantics import select


class NegationModalTests(unittest.TestCase):
    def test_frozen_combinations(self):
        cases = json.loads((Path(__file__).parent / 'fixtures/T_negation_modals.json').read_text())
        for case in cases:
            with self.subTest(text=case['text']):
                p = export_candidate_graph(case['text'])
                leave = next(n['id'] for n in p['nodes'] if n.get('predicate') == 'leave')
                local = [c for c in p['candidates'] if c['arguments'].get('proposition') == leave]
                modals = [c for c in local if c['type'] == 'MODALITY']
                self.assertEqual([c['value'] for c in modals], case['readings'])
                self.assertEqual([c['scope']['polarity'] for c in modals], [case['modal_polarity']] * len(modals))
                self.assertTrue(all(c['scope']['polarity'] == case['polarity'] for c in local if c['type'] != 'MODALITY'))
                scope_questions = [q for q in p['open_questions'] if q['kind'] == 'scope']
                self.assertEqual(bool(scope_questions), case['scope_open'])
                before = copy.deepcopy(p)
                for c in local:
                    selection = select(p, c['id'])
                    result = validate_candidate_selection(p, selection)
                    self.assertTrue(result['contract_valid'], result['errors'])
                    if case['scope_open']:
                        self.assertIn(c['id'], result['provisional_candidate_ids'])
                        q = scope_questions[0]
                        resolution = next(r for r in selection['question_resolutions'] if r['question_id'] == q['id'])
                        resolution.update(status='resolved_by_selection', selected_candidate_ids=[c['id']])
                        self.assertFalse(validate_candidate_selection(p, selection)['contract_valid'])
                self.assertEqual(p, before)

    def test_nested_scope_and_conditional_link_are_not_negated(self):
        for text in ['Officials said Maria cannot leave.',
                     'If Maria cannot leave, Anna will stay.']:
            with self.subTest(text=text):
                p = export_candidate_graph(text)
                for c in p['candidates']:
                    if c['type'] == 'CONDITIONAL_ON':
                        self.assertEqual(c['scope']['polarity'], 'positive')
                    result = validate_candidate_selection(p, select(p, c['id']))
                    self.assertTrue(result['contract_valid'], result['errors'])
                leave = next(n['id'] for n in p['nodes'] if n.get('predicate') == 'leave')
                anchor = next(c for c in p['candidates'] if c['type'] == 'PREDICATION' and c['arguments']['proposition'] == leave)
                expected = 'attributed' if text.startswith('Officials') else 'hypothetical'
                self.assertEqual(anchor['scope']['contexts'][0]['kind'], expected)

    def test_lexical_negation_does_not_become_prohibition(self):
        for text in ['Maria is not required to leave.', "Maria doesn't have to leave.",
                     'Maria is not permitted to leave.']:
            with self.subTest(text=text):
                p = export_candidate_graph(text)
                self.assertFalse(any(c['type'] == 'MODALITY' for c in p['candidates']))
                leave = next(n['id'] for n in p['nodes'] if n.get('predicate') == 'leave')
                anchor = next(c for c in p['candidates'] if c['type'] == 'PREDICATION' and c['arguments']['proposition'] == leave)
                result = validate_candidate_selection(p, select(p, anchor['id']))
                self.assertTrue(result['contract_valid'], result['errors'])
                self.assertIn(anchor['id'], result['provisional_candidate_ids'])


if __name__ == '__main__':
    unittest.main()
