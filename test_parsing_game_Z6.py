import copy
import unittest

import parsing_game_Z5 as z5
import parsing_game_Z6 as z6
from candidate_validation import validate_candidate_selection
from test_scope_semantics import select


class ConditionContentTests(unittest.TestCase):
    sentence = 'If Maria decides to pull the lever, the trolley will stop.'

    def test_every_consequence_path_requires_complement(self):
        package = z6.export_candidate_graph(self.sentence)
        complement = next(c for c in package['candidates'] if c['type'] == 'EVENT_LINK')
        consumers = [c for c in package['candidates'] if c['type'] == 'CONDITIONAL_ON' or
                     any(ctx['kind'] == 'conditional' for ctx in c['scope']['contexts'])]
        self.assertTrue({'PREDICATION', 'PARTICIPANT', 'MODALITY', 'CONDITIONAL_ON'} <= {c['type'] for c in consumers})
        for candidate in consumers:
            with self.subTest(candidate=candidate['type']):
                selection = select(package, candidate['id'])
                self.assertIn(complement['id'], selection['selected_candidate_ids'])
                result = validate_candidate_selection(package, selection)
                self.assertTrue(result['contract_valid'], result['errors'])
                selection['selected_candidate_ids'].remove(complement['id'])
                result = validate_candidate_selection(package, selection)
                self.assertIn('unselected_dependency', [e['code'] for e in result['errors']])

    def test_nested_complements_are_retained(self):
        package = z6.export_candidate_graph('If Maria decides to try to pull the lever, the trolley will stop.')
        links = [c for c in package['candidates'] if c['type'] == 'EVENT_LINK']
        self.assertGreaterEqual(len(links), 2)
        stop = next(n['id'] for n in package['nodes'] if n.get('predicate') == 'stop')
        anchor = next(c for c in package['candidates'] if c['type'] == 'PREDICATION' and c['arguments']['proposition'] == stop)
        selection = select(package, anchor['id'])
        self.assertTrue({c['id'] for c in links} <= set(selection['selected_candidate_ids']))
        result = validate_candidate_selection(package, selection)
        self.assertTrue(result['contract_valid'], result['errors'])

    def test_ambiguous_content_remains_provisional(self):
        package = z5.export_candidate_graph(self.sentence)
        link = next(c for c in package['candidates'] if c['type'] == 'EVENT_LINK')
        other = copy.deepcopy(link)
        other.update(id='alternative_link', value='purpose', exclusive_with=[link['id']])
        link['exclusive_with'] = [other['id']]
        package['candidates'].append(other)
        z6.preserve_condition_content(package)
        for c in package['candidates']:
            if c['type'] != 'CONDITIONAL_ON' and not any(ctx['kind'] == 'conditional' for ctx in c['scope']['contexts']):
                continue
            selection = select(package, c['id'])
            self.assertNotIn(link['id'], selection['selected_candidate_ids'])
            self.assertNotIn(other['id'], selection['selected_candidate_ids'])
            result = validate_candidate_selection(package, selection)
            self.assertTrue(result['contract_valid'], result['errors'])
            self.assertIn(c['id'], result['provisional_candidate_ids'])

    def test_simple_conditions_and_unrelated_sentences_are_unchanged(self):
        for text in ['If Maria pulls the lever, the trolley will stop.',
                     'Maria decides to pull the lever.',
                     'Maria can save the child, but not the dog.']:
            before = z5.export_candidate_graph(text, package_id='comparison')
            after = z6.export_candidate_graph(text, package_id='comparison')
            after['producer'] = before['producer']
            self.assertEqual(before, after)

    def test_scope_and_evidence_are_not_rewritten(self):
        before = z5.export_candidate_graph(self.sentence, package_id='comparison')
        after = z6.export_candidate_graph(self.sentence, package_id='comparison')
        self.assertEqual(before['nodes'], after['nodes'])
        self.assertEqual(before['evidence'], after['evidence'])
        for left, right in zip(before['candidates'], after['candidates']):
            self.assertEqual(left['scope'], right['scope'])
            self.assertEqual(left['arguments'], right['arguments'])

    def test_back_reference_abstains_instead_of_creating_dependency_cycle(self):
        package = z5.export_candidate_graph(self.sentence)
        stop = next(n['id'] for n in package['nodes'] if n.get('predicate') == 'stop')
        anchor = next(c for c in package['candidates'] if c['type'] == 'PREDICATION' and c['arguments']['proposition'] == stop)
        link = next(c for c in package['candidates'] if c['type'] == 'EVENT_LINK')
        link['arguments']['child'] = stop
        link['requires'].append(anchor['id'])
        z6.preserve_condition_content(package)
        selection = select(package, anchor['id'])
        result = validate_candidate_selection(package, selection)
        self.assertTrue(result['contract_valid'], result['errors'])
        self.assertIn(anchor['id'], result['provisional_candidate_ids'])
        self.assertNotIn(link['id'], selection['selected_candidate_ids'])


if __name__ == '__main__':
    unittest.main()
