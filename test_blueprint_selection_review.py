from copy import deepcopy
import json
import unittest

from blueprint_selection_review import compare_admitted, review_selection, semantic_signature
from test_blueprint_world_beam import _admitted, _effect


class SelectionReviewTests(unittest.TestCase):
    def pair(self):
        a = _admitted('one', 1, ['act'], [_effect('DIE')])
        b = _admitted('two', 2, ['act'], [_effect('LIVE', 'BENEFICIAL')])
        return [a, b]

    def response(self, pair):
        return {'status': 'SELECTED', 'selected_candidate_id': 'candidate_0',
            'resolutions': [{'difference_id': d['id'], 'rationale': 'The source states death.',
                            'evidence': [{'clause_id': 'C0', 'excerpt': 'Ada will die.'}]} for d in compare_admitted(pair)['differences']],
            'unresolved_question': ''}

    def test_supported_selection_returns_existing_immutable_candidate(self):
        pair = self.pair(); before = deepcopy(pair)
        response = self.response(pair)
        seen = []
        result, selected = review_selection('Ada will die.', pair,
            lambda messages: seen.append(messages) or json.dumps(response))
        self.assertEqual(result['status'], 'SELECTED')
        self.assertIs(selected, pair[0])
        self.assertEqual(pair, before)
        self.assertIn('Do not add facts', seen[0][0]['content'])

    def test_missing_difference_false_citation_and_new_candidate_do_not_select(self):
        pair = self.pair()
        for change in ('missing', 'citation', 'new', 'empty'):
            response = self.response(pair)
            if change == 'missing': response['resolutions'] = []
            if change == 'citation': response['resolutions'][0]['evidence'][0]['excerpt'] = 'Ada will live.'
            if change == 'new': response['selected_candidate_id'] = 'new_graph'
            if change == 'empty': response['resolutions'][0]['rationale'] = ''
            result, selected = review_selection('Ada will die.', pair, lambda _: json.dumps(response))
            self.assertEqual(result['status'], 'UNRESOLVED')
            self.assertIsNone(selected)
            self.assertTrue(result['errors'])

    def test_abstention_and_failed_service_preserve_precise_questions(self):
        pair = self.pair()
        result, selected = review_selection('Ada will die.', pair, lambda _: json.dumps({
            'status': 'UNRESOLVED', 'unresolved_question': 'Does the conditional apply to both actions?'}))
        self.assertIsNone(selected)
        self.assertIn('conditional', result['unresolved_question'])
        def fail(_):raise RuntimeError('service unavailable')
        result, selected = review_selection('Ada will die.', pair, fail)
        self.assertIsNone(selected)
        self.assertTrue(result['errors'])
        self.assertTrue(result['request']['comparison']['differences'])

    def test_terminal_ellipsis_binds_exact_prefix_without_altering_saved_response(self):
        pair=self.pair();response=self.response(pair)
        for row in response['resolutions']:row['evidence'][0]['excerpt']='Ada will…'
        raw=json.dumps(response)
        result,selected=review_selection('Ada will die.',pair,lambda _:raw)
        self.assertIs(selected,pair[0])
        self.assertEqual(result['raw_response'],raw)
        self.assertTrue(all(b['bound_excerpt']=='Ada will' for b in result['evidence_bindings']))
        for bad in ('Ada will live…','Ada…die.','…'):
            wrong=deepcopy(response);wrong['resolutions'][0]['evidence'][0]['excerpt']=bad
            rejected,selected=review_selection('Ada will die.',pair,lambda _:json.dumps(wrong))
            self.assertIsNone(selected)
            self.assertEqual(rejected['status'],'UNRESOLVED')

    def test_names_and_ids_do_not_change_equivalence_but_quantities_scope_and_links_do(self):
        a = self.pair()[0]['proposal']; b = deepcopy(a)
        b['candidate']['world_model']['parties'][0]['party_id'] = 'P9'
        b['candidate']['world_model']['effects'][0]['party_id'] = 'P9'
        self.assertEqual(semantic_signature(a), semantic_signature(b))
        for field, value in [('quantities', ['five']), ('modality', 'POSSIBLE'),
                             ('likelihood_qualifiers', ['95% chance']), ('condition_ids', ['CND1'])]:
            changed = deepcopy(b); changed['candidate']['world_model']['effects'][0][field] = value
            self.assertNotEqual(semantic_signature(a), semantic_signature(changed))
        changed = deepcopy(a); changed['candidate']['world_model']['causal_links'] = [{'source_id':'E1','target_id':'E2','relation':'CAUSES'}]
        self.assertNotEqual(semantic_signature(a), semantic_signature(changed))


if __name__ == '__main__':unittest.main()
