import copy
import unittest

import parsing_game_Z7 as z7
import parsing_game_Z8 as z8
from candidate_validation import empty_selection, validate_candidate_selection
from test_scope_semantics import select


class ReconstructionIdentityTests(unittest.TestCase):
    def package(self, text):
        p = z8.export_candidate_graph(text, package_id='fixture_z8')
        self.check(p, empty_selection(p))
        for e in p['evidence']:
            self.assertEqual(text[e['start']:e['end']], e['text'])
        return p

    def check(self, p, selection):
        before = copy.deepcopy((p, selection))
        result = validate_candidate_selection(p, selection)
        self.assertTrue(result['contract_valid'], result['errors'])
        self.assertEqual((p, selection), before)
        return result

    def roles(self, p, prop):
        nodes = {n['id']: n for n in p['nodes']}
        return [(c['value'], nodes[c['arguments']['mention']]['label'], c)
                for c in p['candidates'] if c['type'] == 'PARTICIPANT' and c['arguments']['proposition'] == prop]

    def test_roles_group_under_distinct_propositions_and_both_select(self):
        p = self.package('Maria saved the child, but not the dog.')
        r = p['reconstructions'][0]
        self.assertNotEqual(r['proposition_id'], r['antecedent_proposition_id'])
        self.assertEqual([(role, label) for role, label, _ in self.roles(p, r['antecedent_proposition_id'])],
                         [('subject', 'Maria'), ('object', 'the child')])
        self.assertEqual([(role, label) for role, label, _ in self.roles(p, r['proposition_id'])],
                         [('subject', 'Maria'), ('object', 'the dog')])
        ids = [c['id'] for c in p['candidates'] if c['type'] == 'PARTICIPANT']
        selection = empty_selection(p)
        for ident in ids:
            fragment = select(p, ident)
            for key in ['selected_candidate_ids', 'selected_node_ids']:
                selection[key] = sorted(set(selection[key]) | set(fragment[key]))
        self.check(p, selection)

    def test_modals_are_local_and_do_not_require_positive_antecedent(self):
        for modal in ['can', 'will']:
            with self.subTest(modal=modal):
                p = self.package(f'Maria {modal} save the child, but not the dog.')
                r = p['reconstructions'][0]
                for _, _, c in self.roles(p, r['proposition_id']):
                    selection = select(p, c['id'])
                    result = self.check(p, selection)
                    self.assertNotIn(r['antecedent_candidate_id'], selection['selected_candidate_ids'])
                    self.assertIn(c['id'], result['provisional_candidate_ids'])
                for c in p['candidates']:
                    if c['type'] == 'MODALITY' and c['arguments']['proposition'] == r['proposition_id']:
                        self.check(p, select(p, c['id']))

    def test_unchanged_object_is_retained_for_recipient_replacement(self):
        p = self.package('Lila can give the medicine to Omar, but not to Nora.')
        r = p['reconstructions'][0]
        roles = self.roles(p, r['proposition_id'])
        self.assertEqual({(role, label) for role, label, _ in roles},
                         {('subject', 'Lila'), ('object', 'the medicine'), ('destination', 'Nora')})
        self.assertEqual(set(r['participant_candidate_ids']), {c['id'] for _, _, c in roles})
        for _, _, c in roles:
            self.assertEqual(c['scope']['polarity'], 'negative')
            self.check(p, select(p, c['id']))

    def test_ambiguous_roles_keep_exclusivity(self):
        p = self.package('Lila gives Omar the medicine, but not Nora.')
        r = p['reconstructions'][0]
        roles = [c for role, _, c in self.roles(p, r['proposition_id']) if role != 'subject']
        self.assertEqual({c['value'] for c in roles}, {'object', 'destination'})
        for c in roles:
            self.check(p, select(p, c['id']))
        selection = select(p, roles[0]['id'])
        selection['selected_candidate_ids'].append(roles[1]['id'])
        self.assertFalse(validate_candidate_selection(p, selection)['contract_valid'])

    def test_attribution_and_condition_survive(self):
        for text, context in [
            ('Officials said Maria saved the child, but not the dog.', 'attributed'),
            ('If Maria pulls the lever but not the brake, the trolley will stop.', 'hypothetical')]:
            p = self.package(text)
            r = p['reconstructions'][0]
            pred = next(c for c in p['candidates'] if c['id'] == r['predication_candidate_id'])
            self.assertIn(context, [ctx['kind'] for ctx in pred['scope']['contexts']])
            for _, _, c in self.roles(p, r['proposition_id']):
                self.check(p, select(p, c['id']))
            if context == 'hypothetical':
                for c in p['candidates']:
                    if any(ctx['kind'] == 'conditional' for ctx in c['scope']['contexts']):
                        self.assertIn(c['id'], self.check(p, select(p, c['id']))['provisional_candidate_ids'])

    def test_multiple_reconstructions_and_fixed_ids(self):
        text = 'Maria saved the child, but not the dog. Sam moved the lever, but not the brake.'
        p = self.package(text)
        self.assertEqual(p, z8.export_candidate_graph(text, package_id='fixture_z8'))
        self.assertEqual(len({r['proposition_id'] for r in p['reconstructions']}), 2)
        bundles = [set(r['participant_candidate_ids']) for r in p['reconstructions']]
        self.assertFalse(bundles[0] & bundles[1])

    def test_provenance_and_bundle_validation(self):
        original = self.package('Maria saved the child, but not the dog.')
        for mutation, code in [
            (lambda r: r.update(proposition_id=r['antecedent_proposition_id']), 'invalid_reconstruction_identity'),
            (lambda r: r.update(antecedent_candidate_id='missing'), 'invalid_reconstruction_anchor'),
            (lambda r: r['participant_candidate_ids'].pop(), 'invalid_reconstruction_participants')]:
            p = copy.deepcopy(original)
            mutation(p['reconstructions'][0])
            result = validate_candidate_selection(p, empty_selection(p))
            self.assertFalse(result['contract_valid'])
            self.assertIn(code, [e['code'] for e in result['errors']])

    def test_without_stripping_matches_z7(self):
        for text in ['If Maria decides to pull the lever, the trolley will stop.',
                     'Maria saved the child. Anna did too.']:
            p = self.package(text)
            before = z7.export_candidate_graph(text, package_id='fixture_z8')
            self.assertEqual(p.pop('reconstructions'), [])
            p['schema_version'], p['producer'] = before['schema_version'], before['producer']
            self.assertEqual(p, before)

    def test_antecedent_provenance_cannot_be_circular(self):
        p = self.package('Maria saved the child, but not the dog. Sam moved the lever, but not the brake.')
        a, b = p['reconstructions']
        a['antecedent_proposition_id'], a['antecedent_candidate_id'] = b['proposition_id'], b['predication_candidate_id']
        b['antecedent_proposition_id'], b['antecedent_candidate_id'] = a['proposition_id'], a['predication_candidate_id']
        result = validate_candidate_selection(p, empty_selection(p))
        self.assertFalse(result['contract_valid'])
        self.assertIn('reconstruction_provenance_cycle', [e['code'] for e in result['errors']])


if __name__ == '__main__':
    unittest.main()
