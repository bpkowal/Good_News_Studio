import unittest

from parsing_game_T import export_candidate_graph


class ReferenceTests(unittest.TestCase):
    def links(self, text):
        package = export_candidate_graph(text)
        nodes = {n['id']: n for n in package['nodes']}
        links = [(nodes[c['arguments']['mention_a']]['label'],
                  nodes[c['arguments']['mention_b']]['label'], c)
                 for c in package['candidates'] if c['type'] == 'SAME_REFERENT']
        return package, links

    def test_trolley_references(self):
        p, links = self.links('A runaway trolley approached. Maria saw it. She waited. The trolley stopped.')
        pairs = {(a.lower(), b.lower()) for a, b, _ in links}
        self.assertIn(('it', 'a runaway trolley'), pairs)
        self.assertIn(('she', 'maria'), pairs)
        self.assertIn(('the trolley', 'a runaway trolley'), pairs)
        self.assertNotIn(('it', 'maria'), pairs)

    def test_ambiguous_people_remain_candidates(self):
        p, links = self.links('Maria saw Anna. She left.')
        candidates = [c for a, b, c in links if a == 'She']
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(c['assessment']['status'] == 'unresolved' for c in candidates))
        self.assertTrue(all(c['assessment']['score']['kind'] == 'uncalibrated_score' for c in candidates))
        self.assertTrue(any(q['kind'] == 'reference' and len(q['candidate_ids']) == 2 for q in p['open_questions']))

    def test_number_and_cardinality(self):
        _, links = self.links('Five workers waited. The one worker left. She waved.')
        self.assertFalse(any(a == 'The one worker' and b == 'Five workers' for a, b, _ in links))
        self.assertFalse(any(a == 'She' and b == 'Five workers' for a, b, _ in links))

    def test_no_antecedent_still_has_question(self):
        p, links = self.links('She left.')
        self.assertEqual(links, [])
        self.assertTrue(any(q['kind'] == 'reference' and not q['candidate_ids'] for q in p['open_questions']))

    def test_reference_does_not_rewrite_scope_or_merge_mentions(self):
        p, links = self.links('Maria waited. If she leaves, the trolley will stop.')
        self.assertTrue(any(a == 'she' and b == 'Maria' for a, b, _ in links))
        self.assertEqual(len([c for c in p['candidates'] if c['type'] == 'CONDITIONAL_ON']), 1)
        self.assertTrue(any(c['type'] == 'MODALITY' and c['value'] == 'prediction'
                            and c['scope']['contexts'] for c in p['candidates']))
        self.assertTrue(all(c['requires'] == [] for _, _, c in links))
        self.assertTrue(all(c['exclusive_with'] == [] for _, _, c in links))

    def test_local_binding_and_horizon(self):
        _, links = self.links('Maria saw her.')
        self.assertFalse(any(a == 'her' and b == 'Maria' for a, b, _ in links))
        _, links = self.links('Maria waited. Rain fell. Wind blew. Snow melted. She left.')
        self.assertFalse(any(a == 'She' and b == 'Maria' for a, b, _ in links))


if __name__ == '__main__':
    unittest.main()
