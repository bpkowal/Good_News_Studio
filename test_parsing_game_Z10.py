import copy
import json
from pathlib import Path
import unittest

import parsing_game_Z10 as z10
from candidate_validation import empty_selection, validate_candidate_selection
from test_scope_semantics import select


def combined(package, ids):
    result = empty_selection(package)
    for ident in ids:
        fragment = select(package, ident)
        for field in ['selected_node_ids', 'selected_candidate_ids']:
            result[field] = sorted(set(result[field]) | set(fragment[field]))
    return result


class CompleteReadingsTests(unittest.TestCase):
    def test_whitespace_punctuation_and_source_offsets(self):
        fixtures = json.loads((Path(__file__).parent / 'fixtures/ellipsis_text_invariance.json').read_text())
        for fixture in fixtures:
            for text in fixture['texts']:
                with self.subTest(text=text):
                    p = z10.export_candidate_graph(text, package_id='invariance')
                    self.assertEqual(p, z10.export_candidate_graph(text, package_id='invariance'))
                    r = p['reconstructions'][0]
                    roles = {(role, ' '.join(label.split())) for role, label in self.roles(p, r['proposition_id'])}
                    self.assertEqual(roles, {tuple(role) for role in fixture['roles']})
                    for e in p['evidence']:
                        self.assertEqual(text[e['start']:e['end']], e['text'])
                    self.valid(p, combined(p, r['participant_candidate_ids']))

    def roles(self, p, prop):
        nodes = {n['id']: n for n in p['nodes']}
        return {(c['value'], nodes[c['arguments']['mention']]['label'])
                for c in p['candidates'] if c['type'] == 'PARTICIPANT' and c['arguments']['proposition'] == prop}

    def valid(self, p, selection):
        before = copy.deepcopy((p, selection))
        result = validate_candidate_selection(p, selection)
        self.assertTrue(result['contract_valid'], result['errors'])
        self.assertEqual((p, selection), before)
        return result

    def test_complete_subject_object_recipient_selections(self):
        for text, expected in [
            ('Maria left, but not Anna.', {('subject', 'Anna')}),
            ('Maria saved the child, but not the dog.', {('subject', 'Maria'), ('object', 'the dog')}),
            ('Lila gave the medicine to Omar, but not to Nora.',
             {('subject', 'Lila'), ('object', 'the medicine'), ('destination', 'Nora')}),
            ('Lila gives the medicine to Omar, but not the bandage.',
             {('subject', 'Lila'), ('object', 'the bandage'), ('destination', 'Omar')})]:
            with self.subTest(text=text):
                p = z10.export_candidate_graph(text)
                r = p['reconstructions'][0]
                self.assertEqual(self.roles(p, r['proposition_id']), expected)
                ids = [c['id'] for c in p['candidates'] if c['type'] == 'PARTICIPANT']
                selection = combined(p, ids)
                self.valid(p, selection)
                self.assertIn(r['antecedent_candidate_id'], selection['selected_candidate_ids'])
                self.assertIn(r['predication_candidate_id'], selection['selected_candidate_ids'])

    def test_ambiguous_bundles_complete_separate_and_exclusive(self):
        p = z10.export_candidate_graph('Lila gives Omar the medicine, but not Nora.')
        a, b, c = p['reconstructions']
        self.assertEqual(self.roles(p, a['antecedent_proposition_id']),
                         {('subject', 'Lila'), ('object', 'the medicine'), ('destination', 'Omar')})
        self.assertEqual({frozenset(self.roles(p, r['proposition_id'])) for r in [a, b, c]}, {
            frozenset({('subject', 'Lila'), ('object', 'the medicine'), ('destination', 'Nora')}),
            frozenset({('subject', 'Lila'), ('object', 'Nora'), ('destination', 'Omar')}),
            frozenset({('subject', 'Nora'), ('object', 'the medicine'), ('destination', 'Omar')})})
        spoken = [c['id'] for c in p['candidates'] if c['type'] == 'PARTICIPANT'
                  and c['arguments']['proposition'] == a['antecedent_proposition_id']]
        for r in [a, b, c]:
            self.valid(p, combined(p, spoken + r['participant_candidate_ids']))
        for left, right in [(a, b), (a, c), (b, c)]:
            mixed = combined(p, spoken + left['participant_candidate_ids'] + right['participant_candidate_ids'])
            self.assertFalse(validate_candidate_selection(p, mixed)['contract_valid'])
        for r in [a, b, c]:
            missing = combined(p, r['participant_candidate_ids'])
            missing['selected_candidate_ids'].remove(r['predication_candidate_id'])
            self.assertFalse(validate_candidate_selection(p, missing)['contract_valid'])

    def test_scope_gaps_survive_joint_selection_and_direct_consequence(self):
        for text in ['Officials said Maria can save the child, but not the dog.',
                     'If Maria can save the child but not the dog, Anna will leave.']:
            p = z10.export_candidate_graph(text)
            r = p['reconstructions'][0]
            result = self.valid(p, combined(p, [c['id'] for c in p['candidates'] if c['type'] == 'PARTICIPANT']))
            self.assertTrue(set(r['participant_candidate_ids']) <= set(result['provisional_candidate_ids']))
            self.assertTrue(any('NOT MODAL(P)' in q['question'] for q in p['open_questions']))
            for c in p['candidates']:
                if any(ctx['kind'] == 'conditional' for ctx in c['scope']['contexts']):
                    self.assertIn(c['id'], self.valid(p, select(p, c['id']))['provisional_candidate_ids'])

    def test_specific_missing_content_questions_and_controls(self):
        for text, prefix in [
            ('Maria saved the child. Anna did too.', 'Verb-phrase ellipsis:'),
            ('Maria ate rice, and Anna beans.', 'Possible gapping:'),
            ('Someone left, but I do not know who.', 'Possible sluicing:'),
            ('Susan works at night, and Bill too.', 'Stripping:')]:
            p = z10.export_candidate_graph(text)
            self.valid(p, empty_selection(p))
            questions = [q for q in p['open_questions'] if q['question'].startswith(prefix)]
            self.assertEqual(len(questions), 1, text)
            self.assertEqual(questions[0]['candidate_ids'], [])
            self.assertEqual(p['coverage']['status'], 'partial')
        for text in ['Maria saved the child and the dog.', 'Maria did the work.',
                     'I know who left.', 'Maria ate rice, and Anna ate beans.']:
            p = z10.export_candidate_graph(text)
            self.assertFalse(p['reconstructions'])
            self.assertFalse(any(q['question'].startswith(('Stripping:', 'Verb-phrase ellipsis:',
                'Possible gapping:', 'Possible sluicing:')) for q in p['open_questions']))


if __name__ == '__main__':
    unittest.main()
