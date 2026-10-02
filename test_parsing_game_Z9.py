import copy
import unittest

import parsing_game_Z8 as z8
import parsing_game_Z9 as z9
from candidate_validation import empty_selection, validate_candidate_selection
from test_scope_semantics import select


class ModalEllipsisScopeTests(unittest.TestCase):
    def check(self, p, selection):
        before = copy.deepcopy((p, selection))
        result = validate_candidate_selection(p, selection)
        self.assertTrue(result['contract_valid'], result['errors'])
        self.assertEqual((p, selection), before)
        return result

    def local(self, p):
        prop = p['reconstructions'][0]['proposition_id']
        return [c for c in p['candidates'] if c['arguments'].get('proposition') == prop]

    def test_explicit_operator_question_and_all_selection_paths(self):
        for modal in ['can', 'may', 'must', 'will']:
            with self.subTest(modal=modal):
                text = f'Maria {modal} save the child, but not the dog.'
                p = z9.export_candidate_graph(text)
                local = self.local(p)
                q = next(q for q in p['open_questions'] if 'NOT MODAL(P)' in q['question'])
                self.assertIn('MODAL(NOT P)', q['question'])
                self.assertEqual(set(q['candidate_ids']), {c['id'] for c in local})
                self.assertEqual(set(q['blocking_for']), set(q['candidate_ids']))
                evidence = {e['id']: e['text'] for e in p['evidence']}
                self.assertIn(modal, [evidence[e] for e in q['evidence_ids']])
                self.assertIn('not', [evidence[e] for e in q['evidence_ids']])
                for c in local:
                    self.assertEqual(c['scope']['polarity'], 'unresolved')
                    selection = select(p, c['id'])
                    self.assertIn(c['id'], self.check(p, selection)['provisional_candidate_ids'])
                    r = next(r for r in selection['question_resolutions'] if r['question_id'] == q['id'])
                    r.update(status='resolved_by_selection', selected_candidate_ids=[c['id']])
                    result = validate_candidate_selection(p, selection)
                    self.assertFalse(result['contract_valid'])
                    self.assertIn('unresolved_candidate_is_not_resolution', [e['code'] for e in result['errors']])

    def test_spoken_antecedent_and_provenance_unchanged(self):
        text = 'Maria can save the child, but not the dog.'
        old = z8.export_candidate_graph(text, package_id='same')
        p = z9.export_candidate_graph(text, package_id='same')
        source = p['reconstructions'][0]['antecedent_proposition_id']
        self.assertEqual([c for c in p['candidates'] if c['arguments'].get('proposition') == source],
                         [c for c in old['candidates'] if c['arguments'].get('proposition') == source])
        for key in ['nodes', 'evidence', 'reconstructions', 'choice_sets']:
            self.assertEqual(p[key], old[key])

    def test_nonmodal_reconstructions_and_nonellipsis_unchanged(self):
        for text in ['Maria saved the child, but not the dog.',
                     'Maria cannot save the dog.',
                     'If Maria decides to pull the lever, the trolley will stop.']:
            old = z8.export_candidate_graph(text, package_id='same')
            p = z9.export_candidate_graph(text, package_id='same')
            p['producer'] = old['producer']
            self.assertEqual(p, old)
            self.check(p, empty_selection(p))

    def test_attribution_and_recipient_bundle_survive(self):
        p = z9.export_candidate_graph('Officials said Lila can give the medicine to Omar, but not to Nora.')
        local = self.local(p)
        roles = [c for c in local if c['type'] == 'PARTICIPANT']
        self.assertEqual({c['value'] for c in roles}, {'subject', 'object', 'destination'})
        for c in local:
            self.assertIn('attributed', [ctx['kind'] for ctx in c['scope']['contexts']])
            self.check(p, select(p, c['id']))

    def test_multiple_reconstructions_and_whitespace(self):
        text = 'Maria\ncan save the child, but\nnot the dog. Sam will move the lever, but not the brake.'
        p = z9.export_candidate_graph(text, package_id='fixed')
        self.assertEqual(p, z9.export_candidate_graph(text, package_id='fixed'))
        self.assertEqual(len(p['reconstructions']), 2)
        questions = [q for q in p['open_questions'] if 'NOT MODAL(P)' in q['question']]
        self.assertEqual(len(questions), 2)
        self.assertFalse(set(questions[0]['candidate_ids']) & set(questions[1]['candidate_ids']))
        self.check(p, empty_selection(p))


if __name__ == '__main__':
    unittest.main()
