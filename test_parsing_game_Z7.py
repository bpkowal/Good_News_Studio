import copy
import unittest

import parsing_game_Z6 as z6
import parsing_game_Z7 as z7
from candidate_validation import empty_selection, validate_candidate_selection
from test_scope_semantics import select


class CompleteConditionTests(unittest.TestCase):
    def consumers(self, p):
        return [c for c in p['candidates'] if c['type'] == 'CONDITIONAL_ON' or
                any(ctx['kind'] == 'conditional' for ctx in c['scope']['contexts'])]

    def check(self, p, selection):
        before = copy.deepcopy((p, selection))
        result = validate_candidate_selection(p, selection)
        self.assertTrue(result['contract_valid'], result['errors'])
        self.assertEqual((p, selection), before)
        return result

    def test_all_paths_keep_roles_complement_and_controller(self):
        p = z7.export_candidate_graph('If Maria decides to pull the lever, the trolley will stop.')
        bundle = p['condition_contents'][0]
        table = {c['id']: c for c in p['candidates']}
        required = [table[i] for i in bundle['required_candidate_ids']]
        self.assertTrue({'subject', 'object', 'controller', 'complement'} <= {c['value'] for c in required})
        controller = next(c for c in required if c['value'] == 'controller')
        q = next(q for q in p['open_questions'] if controller['id'] in q['candidate_ids'])
        for c in self.consumers(p):
            with self.subTest(path=c['type']):
                selection = select(p, c['id'])
                self.assertTrue(set(bundle['required_candidate_ids']) <= set(selection['selected_candidate_ids']))
                self.assertIn(c['id'], self.check(p, selection)['provisional_candidate_ids'])
                resolution = next(r for r in selection['question_resolutions'] if r['question_id'] == q['id'])
                resolution.update(status='resolved_by_selection', selected_candidate_ids=[controller['id']])
                self.assertNotIn(c['id'], self.check(p, selection)['provisional_candidate_ids'])

    def test_missing_role_cannot_bypass_bundle_check(self):
        p = z7.export_candidate_graph('If Maria pulls the lever, the trolley will stop.')
        omitted = next(c['id'] for c in p['candidates'] if c['type'] == 'PARTICIPANT' and c['value'] == 'object')
        # Even a package producer that omits the convenience dependency cannot
        # make an incomplete bundle look complete to the selection validator.
        for c in p['candidates']:
            c['requires'] = [i for i in c['requires'] if i != omitted]
        for c in self.consumers(p):
            selection = select(p, c['id'])
            self.assertNotIn(omitted, selection['selected_candidate_ids'])
            self.assertIn(c['id'], self.check(p, selection)['provisional_candidate_ids'])
            selection['selected_candidate_ids'].append(omitted)
            role = next(x for x in p['candidates'] if x['id'] == omitted)
            selection['selected_node_ids'] = sorted(set(selection['selected_node_ids']) | set(role['arguments'].values()))
            self.assertNotIn(c['id'], self.check(p, selection)['provisional_candidate_ids'])

    def test_quantities_and_nested_complements_survive(self):
        p = z7.export_candidate_graph('If Maria decides to try to save five workers, Anna will leave.')
        bundle = p['condition_contents'][0]
        table = {c['id']: c for c in p['candidates']}
        required = [table[i] for i in bundle['required_candidate_ids']]
        self.assertTrue(any(c['type'] == 'QUANTITY' and c['value']['amount'] == 5 for c in required))
        links = [c for c in required if c['type'] == 'EVENT_LINK']
        self.assertEqual(len(links), 2)
        self.assertTrue(any(a['arguments']['child'] == b['arguments']['parent']
                            for a in links for b in links if a is not b))
        for c in self.consumers(p):
            selection = select(p, c['id'])
            self.check(p, selection)
            self.assertTrue(set(bundle['required_candidate_ids']) <= set(selection['selected_candidate_ids']))

    def test_modal_condition_abstention_and_resolution(self):
        p = z7.export_candidate_graph('If Maria can pull the lever, the trolley will stop.')
        ability = next(c for c in p['candidates'] if c['type'] == 'MODALITY' and c['value'] == 'ability')
        q = next(q for q in p['open_questions'] if ability['id'] in q['candidate_ids'])
        for c in self.consumers(p):
            selection = select(p, c['id'])
            self.assertIn(c['id'], self.check(p, selection)['provisional_candidate_ids'])
            selection['selected_candidate_ids'].append(ability['id'])
            r = next(r for r in selection['question_resolutions'] if r['question_id'] == q['id'])
            r.update(status='resolved_by_selection', selected_candidate_ids=[ability['id']])
            self.assertNotIn(c['id'], self.check(p, selection)['provisional_candidate_ids'])

    def test_unlinked_embedded_content_is_explicit_gap(self):
        p = z7.export_candidate_graph('If Maria says Anna left, the trolley will stop.')
        bundle = p['condition_contents'][0]
        self.assertTrue(bundle['question_ids'])
        for c in self.consumers(p):
            self.assertIn(c['id'], self.check(p, select(p, c['id']))['provisional_candidate_ids'])

    def test_bundle_contract_rejects_missing_and_dangling_content(self):
        original = z7.export_candidate_graph('If Maria pulls the lever, the trolley will stop.')
        for mutation, code in [
            (lambda p: p.pop('condition_contents'), 'missing_condition_contents'),
            (lambda p: p.update(condition_contents=[]), 'missing_condition_content'),
            (lambda p: p['condition_contents'][0]['required_candidate_ids'].append('missing'), 'dangling_reference'),
            (lambda p: p.update(schema_version='0.2'), 'condition_contents_requires_schema_0.3')]:
            p = copy.deepcopy(original)
            mutation(p)
            result = validate_candidate_selection(p, empty_selection(p))
            self.assertFalse(result['contract_valid'])
            self.assertIn(code, [e['code'] for e in result['errors']])

    def test_negative_scope_is_unchanged_and_simple_condition_completes(self):
        text = 'If Maria does not pull the lever, the trolley will stop.'
        p = z7.export_candidate_graph(text, package_id='same')
        old = z6.export_candidate_graph(text, package_id='same')
        self.assertEqual(p['nodes'], old['nodes'])
        self.assertEqual(p['evidence'], old['evidence'])
        self.assertEqual([c['scope'] for c in p['candidates']], [c['scope'] for c in old['candidates']])
        for c in self.consumers(p):
            self.assertNotIn(c['id'], self.check(p, select(p, c['id']))['provisional_candidate_ids'])

    def test_nonconditional_graph_unchanged(self):
        text = 'Maria can save the child, but not the dog.'
        p = z7.export_candidate_graph(text, package_id='same')
        old = z6.export_candidate_graph(text, package_id='same')
        self.assertEqual(p.pop('condition_contents'), [])
        p['producer'], p['schema_version'] = old['producer'], old['schema_version']
        self.assertEqual(p, old)

    def test_ambiguous_condition_roles_do_not_force_incompatible_selections(self):
        p = z7.export_candidate_graph('If Lila helps Omar, Nora will leave.')
        self.assertTrue(p['condition_contents'][0]['question_ids'])
        for c in self.consumers(p):
            selection = select(p, c['id'])
            self.assertIn(c['id'], self.check(p, selection)['provisional_candidate_ids'])

    def test_unrelated_conditions_do_not_share_controller_block(self):
        p = z7.export_candidate_graph('If Maria decides to pull the lever, the trolley will stop. '
                                     'If Liam pulls the rope, the alarm will ring.')
        ring = next(n['id'] for n in p['nodes'] if n.get('predicate') == 'ring')
        anchor = next(c for c in p['candidates'] if c['type'] == 'PREDICATION' and c['arguments']['proposition'] == ring)
        self.assertEqual(len(p['condition_contents']), 2)
        self.assertNotIn(anchor['id'], self.check(p, select(p, anchor['id']))['provisional_candidate_ids'])


if __name__ == '__main__':
    unittest.main()
