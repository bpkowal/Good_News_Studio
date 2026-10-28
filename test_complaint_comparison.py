"""Authored structural contrasts; no asserted correct moral ranking."""
from copy import deepcopy
from itertools import permutations
import unittest

from complaint_comparison import review_comparisons, derive_position_changes


def packet():
    return {
        'persons': [{'id': 'excluded'}, {'id': 'contributor'}],
        'burdens': [{'id': 'unmet_need', 'person_id': 'excluded', 'description': 'Unmet protective need.'},
                    {'id': 'income_loss', 'person_id': 'contributor', 'description': 'Income loss.'}],
        'burden_positions': [
            {'burden_id': burden, 'action_id': action, 'status': status,
             'position': description, 'ground_handles': ['fact']}
            for burden, action, status, description in [
                ('unmet_need', 'X', 'PRESENT', 'Need remains.'),
                ('unmet_need', 'Y', 'PRESENT', 'Need remains.'),
                ('unmet_need', 'Z', 'ABSENT', 'Need met.'),
                ('income_loss', 'X', 'ABSENT', 'No imposed income loss.'),
                ('income_loss', 'Y', 'ABSENT', 'No imposed income loss.'),
                ('income_loss', 'Z', 'PRESENT', 'Eight-week income loss.')]],
        'reasons': [{'id': 'need_complaint', 'person_id': 'excluded', 'action_id': 'Y'},
                    {'id': 'income_complaint', 'person_id': 'contributor', 'action_id': 'Z'}],
        'comparisons': [{'id': 'pair', 'from_action_id': 'Y', 'to_action_id': 'Z',
            'status': 'UNRESOLVED', 'reason_ids': ['need_complaint', 'income_complaint'],
            'relieved_burden_ids': ['unmet_need'], 'introduced_burden_ids': ['income_loss'],
            'retained_burden_ids': []}],
        'recommendation_action_id': 'Y', 'recommendation_status': 'PROVISIONAL_PREFERENCE',
        'recommendation_comparison_ids': ['pair'], 'remaining_complaint_ids': ['need_complaint']}


class ComparisonTests(unittest.TestCase):
    def review(self, raw):
        return review_comparisons(raw, {'X', 'Y', 'Z'}, {'fact'})

    def codes(self, raw):
        return {f['code'] for f in self.review(raw)['findings']}

    def test_unresolved_comparison_can_retain_provisional_choice_and_objection(self):
        raw = packet(); before = deepcopy(raw)
        review = self.review(raw)
        self.assertEqual(review['findings'], [])
        self.assertEqual(review['remaining_complaint_ids'], ['need_complaint'])
        self.assertFalse(review['moral_ranking_computed'])
        self.assertFalse(review['operative'])
        self.assertEqual(raw, before)

    def test_eliminating_third_option_does_not_resolve_remaining_pair(self):
        raw = packet()
        raw['comparisons'].append({'id': 'other_pair', 'from_action_id': 'X', 'to_action_id': 'Y',
            'status': 'FAVORS_TO', 'reason_ids': [], 'relieved_burden_ids': [],
            'introduced_burden_ids': [], 'retained_burden_ids': ['unmet_need']})
        raw['recommendation_status'] = 'UNIQUE_DEFENSIBLE'
        raw['recommendation_comparison_ids'].append('other_pair')
        self.assertIn('UNIQUENESS_COMPARISON_NOT_ESTABLISHED', self.codes(raw))
        self.assertIn('OPEN_COMPARISON_IN_UNIQUENESS_CLAIM', self.codes(raw))

    def test_burden_remaining_in_both_options_cannot_be_reported_as_relieved(self):
        raw = packet()
        raw['comparisons'][0].update(from_action_id='X', to_action_id='Y',
            reason_ids=['need_complaint'], introduced_burden_ids=[], retained_burden_ids=[])
        codes = self.codes(raw)
        self.assertIn('BURDEN_TRANSITION_MISMATCH', codes)
        self.assertIn('RETAINED_BURDEN_OMITTED', codes)

    def test_unknown_risk_does_not_count_as_absence_or_relief(self):
        raw = packet()
        raw['burden_positions'][2]['status'] = 'UNKNOWN'
        raw['burden_positions'][2]['position'] = 'Whether this option meets the need is unknown.'
        self.assertIn('BURDEN_TRANSITION_MISMATCH', self.codes(raw))

    def test_risk_can_change_magnitude_while_still_remaining(self):
        raw = packet()
        raw['burden_positions'][2].update(status='PRESENT', position='20% mortality risk rather than certainty.')
        raw['burden_positions'][1]['position'] = 'Certain death.'
        raw['comparisons'][0].update(relieved_burden_ids=[], retained_burden_ids=['unmet_need'])
        self.assertEqual(self.review(raw)['findings'], [])
        self.assertEqual(raw['burden_positions'][2]['position'], '20% mortality risk rather than certainty.')

    def test_consent_does_not_erase_economic_cost_or_force_ranking(self):
        raw = packet()
        raw['burden_positions'][5]['position'] = 'Eight-week income loss; freely consented.'
        raw['comparisons'][0]['status'] = 'FAVORS_TO'
        self.assertEqual(self.review(raw)['findings'], [])
        self.assertFalse(self.review(raw)['moral_ranking_computed'])

    def test_irrelevant_third_option_reason_is_exposed(self):
        raw = packet(); raw['reasons'][0]['action_id'] = 'X'
        self.assertIn('REASON_OUTSIDE_COMPARISON_PAIR', self.codes(raw))

    def test_duplicate_or_missing_positions_remain_visible(self):
        raw = packet(); raw['burden_positions'].append(deepcopy(raw['burden_positions'][0]))
        raw['burden_positions'].pop(2)
        codes = self.codes(raw)
        self.assertIn('DUPLICATE_POSITION', codes)
        self.assertIn('MISSING_POSITION', codes)

    def test_rule_is_invariant_under_action_and_person_renaming(self):
        for names in permutations(('disclose', 'withhold', 'ask_consent')):
            raw = packet(); mapping = dict(zip(('X', 'Y', 'Z'), names))
            for row in raw['burden_positions']:
                row['action_id'] = mapping[row['action_id']]
            for row in raw['reasons']:
                row['action_id'] = mapping[row['action_id']]
            for row in raw['comparisons']:
                for field in ('from_action_id', 'to_action_id'):
                    row[field] = mapping[row[field]]
            raw['recommendation_action_id'] = mapping[raw['recommendation_action_id']]
            raw['persons'][0]['id'] = 'patient'
            raw['burdens'][0]['person_id'] = 'patient'
            raw['reasons'][0]['person_id'] = 'patient'
            review = review_comparisons(raw, set(names), {'fact'})
            self.assertEqual(review['findings'], [])

    def test_projection_constructs_changes_even_when_model_labels_reverse_them(self):
        raw = packet(); before = deepcopy(raw)
        raw['comparisons'][0]['relieved_burden_ids'] = ['income_loss']
        raw['comparisons'][0]['introduced_burden_ids'] = ['unmet_need']
        frame = derive_position_changes(raw)[0]
        changes = {p['burden_id']: p['change'] for p in frame['positions']}
        self.assertEqual(changes, {'unmet_need': 'RELIEVED', 'income_loss': 'INTRODUCED'})
        self.assertEqual(raw['burden_positions'], before['burden_positions'])
        self.assertEqual(frame['proposed_comparison_status'], 'UNRESOLVED')
        self.assertFalse(frame['moral_ranking_computed'])

    def test_projection_preserves_quantities_and_does_not_resolve_duplicate_cells(self):
        raw = packet()
        raw['burden_positions'][2].update(status='PRESENT', position='20% mortality rather than certainty.')
        frame = derive_position_changes(raw)[0]
        self.assertEqual(frame['positions'][0]['change'], 'REMAINS_WITH_POSSIBLY_DIFFERENT_EXTENT')
        self.assertIn('20%', frame['positions'][0]['to']['descriptions'][0])
        duplicate = deepcopy(raw['burden_positions'][2]); duplicate['status'] = 'ABSENT'
        raw['burden_positions'].append(duplicate)
        frame = derive_position_changes(raw)[0]
        self.assertEqual(frame['positions'][0]['change'], 'UNKNOWN')
        self.assertEqual(len(frame['positions'][0]['to']['raw_position_indices']), 2)

    def test_projection_cannot_promote_a_proposed_moral_ranking(self):
        raw = packet(); raw['comparisons'][0]['status'] = 'FAVORS_FROM'
        frame = derive_position_changes(raw)[0]
        self.assertEqual(frame['proposed_comparison_status'], 'FAVORS_FROM')
        self.assertFalse(frame['moral_ranking_computed'])
        self.assertEqual(frame['authority'], 'DERIVED_FROM_MODEL_MATRIX_NOT_SOURCE_VERIFIED')

    def test_explicit_comparisons_can_support_uniqueness_without_computing_morality(self):
        raw = packet(); raw['comparisons'][0]['status'] = 'FAVORS_FROM'
        raw['comparisons'].append({'id': 'other_pair', 'from_action_id': 'X', 'to_action_id': 'Y',
            'status': 'FAVORS_TO', 'reason_ids': [], 'relieved_burden_ids': [],
            'introduced_burden_ids': [], 'retained_burden_ids': ['unmet_need']})
        raw['recommendation_status'] = 'UNIQUE_DEFENSIBLE'
        raw['recommendation_comparison_ids'].append('other_pair')
        review = self.review(raw)
        self.assertEqual(review['findings'], [])
        self.assertEqual(review['semantic_support'], 'NOT_INDEPENDENTLY_ASSESSED')


if __name__ == '__main__':
    unittest.main()
