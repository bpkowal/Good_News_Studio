import copy
import unittest

import eval_ellipsis_coverage as evaluation
import parsing_game_Z10 as z10


class CoverageMetricsTests(unittest.TestCase):
    def test_frozen_construction_coverage_has_honest_denominators(self):
        report = evaluation.evaluate(z10.export_candidate_graph)
        groups = report['constructions']
        stripping = groups['stripping']
        self.assertEqual((stripping['gold'], stripping['matched'], stripping['false_proposals']), (12, 11, 0))
        self.assertEqual(stripping['scope_correct'], 11)
        self.assertEqual(stripping['role_correct'], stripping['role_count'])
        self.assertEqual(stripping['no_proposal_positive_cases'], 1)
        self.assertEqual(stripping['unresolved_proposal_rate'], 1)
        for kind in ['verb_phrase_ellipsis', 'gapping', 'sluicing']:
            self.assertEqual(groups[kind]['candidate_recall'], 0)
            self.assertEqual(groups[kind]['no_proposal_abstention'], 1)
            self.assertIsNone(groups[kind]['scope_preservation'])
            self.assertIsNone(groups[kind]['role_accuracy'])
        for group in groups.values():
            self.assertEqual(group['valid_packages'], group['cases'])
            self.assertEqual(group['specific_question_cases'], group['positive_cases'])
            self.assertEqual(group['question_control_false_positives'], 0)

    def test_wrong_scope_false_readings_and_duplicates_are_counted(self):
        gold = dict(predicate='save', roles=[['subject', 'Maria'], ['object', 'dog']],
                    scope=dict(polarity='negative', contexts=[]), modal=False, scope_question=False)
        actual = dict(copy.deepcopy(gold), unresolved=True)
        actual['scope']['polarity'] = 'positive'
        wrong = dict(copy.deepcopy(actual), predicate='leave')
        counts = evaluation.score_readings([gold], [actual, actual, wrong])
        self.assertEqual(counts['matched'], 1)
        self.assertEqual(counts['false_proposals'], 2)
        self.assertEqual(counts['scope_correct'], 0)
        self.assertEqual(counts['role_count'], 6)
        self.assertEqual(counts['role_correct'], 4)
        self.assertEqual(gold['scope']['polarity'], 'negative')

    def test_inconsistent_participant_scope_fails_projection(self):
        p = z10.export_candidate_graph('Maria saved the child, but not the dog.')
        role_id = p['reconstructions'][0]['participant_candidate_ids'][0]
        next(c for c in p['candidates'] if c['id'] == role_id)['scope']['polarity'] = 'positive'
        self.assertEqual(evaluation.project(p)[0]['scope'], {'inconsistent': True})


if __name__ == '__main__':
    unittest.main()
