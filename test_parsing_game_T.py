import unittest

import parsing_game_T as t


class CandidateTests(unittest.TestCase):
    def package(self, text):
        result = t.export_candidate_graph(text)
        candidates = {c['id']: c for c in result['candidates']}
        nodes = {n['id']: n for n in result['nodes']}
        for evidence in result['evidence']:
            self.assertEqual(text[evidence['start']:evidence['end']], evidence['text'])
        for candidate in candidates.values():
            self.assertTrue(all(x in nodes for x in candidate['arguments'].values()))
            self.assertTrue(all(x in candidates for x in candidate['requires']))
            for other in candidate['exclusive_with']:
                self.assertIn(candidate['id'], candidates[other]['exclusive_with'])
            def visit(ident, path):
                self.assertNotIn(ident, path)
                for dependency in candidates[ident]['requires']:
                    visit(dependency, path + [ident])
            visit(candidate['id'], [])
        return result

    def test_c4_preserves_event_and_destination_alternatives(self):
        p = self.package('They were too late to work.')
        self.assertTrue(any(n.get('predicate') == 'work' for n in p['nodes']))
        self.assertTrue(any(c['value'] == 'destination' for c in p['candidates']))
        self.assertTrue(any(c['type'] == 'EVENT_LINK' for c in p['candidates']))
        work = next(n['id'] for n in p['nodes'] if n.get('predicate') == 'work')
        anchor = next(c for c in p['candidates'] if c['type'] == 'PREDICATION' and c['arguments']['proposition'] == work)
        destination = next(c for c in p['candidates'] if c['value'] == 'destination')
        self.assertIn(anchor['id'], destination['exclusive_with'])
        self.assertTrue(any(q['kind'] == 'attachment' and q['blocking_for'] for q in p['open_questions']))

    def test_modal_meanings_are_distinct(self):
        for text, expected in [('Maria can leave.', 'ability'),
                               ('Maria will leave.', 'prediction'),
                               ('Maria must leave.', 'obligation')]:
            p = self.package(text)
            self.assertIn(expected, [c['value'] for c in p['candidates'] if c['type'] == 'MODALITY'])
            self.assertFalse(any('occurrence_status' in c for c in p['candidates']))

    def test_conditional_consequence_and_modal_share_scope(self):
        p = self.package('If Maria pulls the lever, the trolley will stop.')
        links = [c for c in p['candidates'] if c['type'] == 'CONDITIONAL_ON']
        self.assertEqual(len(links), 1)
        nodes = {n['id']: n for n in p['nodes']}
        self.assertEqual(nodes[links[0]['arguments']['condition']]['predicate'], 'pull')
        self.assertEqual(nodes[links[0]['arguments']['consequence']]['predicate'], 'stop')
        modal = next(c for c in p['candidates'] if c['type'] == 'MODALITY')
        self.assertEqual(modal['scope']['contexts'][0]['kind'], 'conditional')

    def test_embedded_if_is_not_a_conditional_outcome(self):
        p = self.package('Maria asked if Anna left.')
        self.assertFalse(any(c['type'] == 'CONDITIONAL_ON' for c in p['candidates']))
        self.assertTrue(any(q['kind'] == 'scope' for q in p['open_questions']))

    def test_negative_predication_and_document_offsets(self):
        p = self.package('Maria waited. Anna did not leave.')
        node = next(n for n in p['nodes'] if n.get('predicate') == 'leave')
        pred = next(c for c in p['candidates'] if c['type'] == 'PREDICATION' and c['arguments']['proposition'] == node['id'])
        self.assertEqual(pred['scope']['polarity'], 'negative')

    def test_purpose_and_whether_verbs_are_preserved(self):
        p = self.package('Maria can pull a lever to divert it. She must decide whether to act.')
        lemmas = {n.get('predicate') for n in p['nodes']}
        self.assertTrue({'divert', 'act', 'decide'} <= lemmas)

    def test_controllers_remain_explicit_child_subjects(self):
        for text, noun in [('The flood caused the library to close.', 'library'),
                           ('The manager allowed the workers to leave.', 'workers')]:
            p = self.package(text)
            self.assertTrue(any(noun in n['label'] for n in p['nodes'] if n['kind'] == 'mention'))
            self.assertTrue(any(c['type'] == 'EVENT_LINK' for c in p['candidates']))
            self.assertTrue(any(c['value'] == 'controller' for c in p['candidates']))


if __name__ == '__main__':
    unittest.main()
