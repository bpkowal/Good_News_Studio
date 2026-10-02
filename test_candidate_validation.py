import copy
import unittest

from candidate_validation import empty_selection, validate_candidate_selection
from parsing_game_T import export_candidate_graph


def select(package, candidate_ids):
    selection = empty_selection(package)
    table = {c['id']: c for c in package['candidates']}
    selected = set(candidate_ids)
    pending = list(selected)
    while pending:
        for dependency in table[pending.pop()]['requires']:
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    selection['selected_candidate_ids'] = sorted(selected)
    selection['selected_node_ids'] = sorted({v for ident in selected for v in table[ident]['arguments'].values()})
    return selection


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = export_candidate_graph('Maria saw Anna. She did not leave. If Maria pulls the lever, the trolley will stop.')

    def check(self, package, selection, code=None):
        before = copy.deepcopy((package, selection))
        result = validate_candidate_selection(package, selection)
        self.assertEqual((package, selection), before)
        if code:
            self.assertFalse(result['contract_valid'])
            self.assertIn(code, [e['code'] for e in result['errors']])
        else:
            self.assertTrue(result['contract_valid'], result['errors'])
        self.assertEqual(result['world_state_commitment'], 'not_authorized')
        return result

    def test_empty_partial_selection_is_valid(self):
        result = self.check(self.package, empty_selection(self.package))
        self.assertTrue(result['unresolved_question_ids'])

    def test_valid_scoped_selection(self):
        ident = next(c['id'] for c in self.package['candidates'] if c['type'] == 'MODALITY')
        self.check(self.package, select(self.package, [ident]))

    def test_invalid_dependency_and_endpoint(self):
        c = next(c for c in self.package['candidates'] if c['type'] == 'PARTICIPANT')
        selection = select(self.package, [c['id']])
        selection['selected_candidate_ids'] = [c['id']]
        self.check(self.package, selection, 'unselected_dependency')
        selection = select(self.package, [c['id']])
        selection['selected_node_ids'] = []
        self.check(self.package, selection, 'unselected_endpoint')

    def test_no_scope_override_or_actuality_fields(self):
        selection = empty_selection(self.package)
        selection['scope'] = {'polarity': 'positive'}
        self.check(self.package, selection, 'unexpected_fields')
        selection = empty_selection(self.package)
        selection['actually_happened'] = True
        self.check(self.package, selection, 'unexpected_fields')

    def test_dangling_span_version_and_cycle(self):
        p = copy.deepcopy(self.package)
        p['evidence'][0]['end'] += 1
        self.check(p, empty_selection(p), 'invalid_evidence_span')
        p = copy.deepcopy(self.package)
        p['schema_version'] = '99'
        self.check(p, empty_selection(p), 'unsupported_schema_version')
        p = copy.deepcopy(self.package)
        p['candidates'][0]['requires'] = [p['candidates'][0]['id']]
        self.check(p, empty_selection(p), 'dependency_cycle')
        selection = empty_selection(self.package)
        selection['selected_candidate_ids'] = ['invented']
        self.check(self.package, selection, 'dangling_reference')

    def test_conditional_dependency_cannot_disappear(self):
        p = copy.deepcopy(self.package)
        modal = next(c for c in p['candidates'] if c['type'] == 'MODALITY')
        anchor = next(c for c in p['candidates'] if c['id'] == modal['requires'][0])
        anchor['requires'] = []
        self.check(p, empty_selection(p), 'missing_condition_dependency')

    def test_c4_mutually_exclusive_readings(self):
        p = export_candidate_graph('They were too late to work.')
        candidates = [c['id'] for c in p['candidates'] if c['value'] == 'destination' or c['type'] == 'EVENT_LINK']
        self.check(p, select(p, candidates), 'exclusive_selection')

    def test_question_bookkeeping_and_provisional_propagation(self):
        q = next(q for q in self.package['open_questions'] if q['kind'] == 'reference' and q['blocking_for'])
        selection = select(self.package, q['blocking_for'])
        result = self.check(self.package, selection)
        self.assertTrue(set(q['blocking_for']) <= set(result['provisional_candidate_ids']))
        resolution = next(r for r in selection['question_resolutions'] if r['question_id'] == q['id'])
        resolution['status'] = 'resolved_by_selection'
        self.check(self.package, selection, 'unsupported_question_resolution')
        selection = empty_selection(self.package)
        selection['question_resolutions'].pop()
        self.check(self.package, selection, 'missing_question_resolutions')

    def test_extensions_stay_unverified(self):
        selection = empty_selection(self.package)
        selection['extensions'] = [dict(id='extension1', origin='llm_inferred',
            evidence_ids=[self.package['evidence'][0]['id']], description='Possible new interpretation', verification_status='unverified')]
        result = self.check(self.package, selection)
        self.assertEqual(result['unverified_extension_ids'], ['extension1'])
        selection['selected_candidate_ids'] = ['extension1']
        self.check(self.package, selection, 'dangling_reference')

    def test_reference_resolution_and_dependency_provisional_status(self):
        p = copy.deepcopy(self.package)
        q = next(q for q in p['open_questions'] if q['kind'] == 'reference' and q['blocking_for'])
        link = q['candidate_ids'][0]
        role = next(c for c in p['candidates'] if c['id'] == q['blocking_for'][0])
        role['requires'].append(link)
        selection = select(p, [role['id']])
        result = self.check(p, selection)
        self.assertIn(role['id'], result['provisional_candidate_ids'])
        resolution = next(r for r in selection['question_resolutions'] if r['question_id'] == q['id'])
        resolution.update(status='resolved_by_selection', selected_candidate_ids=[link])
        result = self.check(p, selection)
        self.assertNotIn(link, result['provisional_candidate_ids'])
        self.assertNotIn(role['id'], result['provisional_candidate_ids'])

    def test_scores_types_and_package_binding(self):
        p = copy.deepcopy(self.package)
        p['candidates'][0]['assessment']['score'] = dict(value=0.9, kind='calibrated_probability',
                                                       source='test', calibration_id=None)
        self.check(p, empty_selection(p), 'missing_calibration_artifact')
        p['candidates'][0]['assessment']['score']['value'] = float('inf')
        self.check(p, empty_selection(p), 'invalid_score')
        p = copy.deepcopy(self.package)
        p['candidates'][0]['type'] = 'INVENTED'
        self.check(p, empty_selection(p), 'unknown_candidate_type')
        selection = empty_selection(self.package)
        selection['package_id'] = 'another_export'
        self.check(self.package, selection, 'package_mismatch')

    def test_identity_transitive_conflict(self):
        p = copy.deepcopy(self.package)
        names = {}
        for node in p['nodes']:
            if node['kind'] == 'mention':
                names.setdefault(node['label'], node)
        links = [c for c in p['candidates'] if c['type'] == 'SAME_REFERENT'
                 and c['arguments']['mention_a'] == names['She']['id']]
        self.assertEqual(len(links), 2)
        p['identity_constraints'] = [dict(id='different_people', mention_a=names['Maria']['id'],
            mention_b=names['Anna']['id'], evidence_ids=names['Maria']['evidence_ids'] + names['Anna']['evidence_ids'],
            provenance=[dict(producer='test_fixture', version='1', method='explicit_fixture_constraint', resource_ids=[])])]
        self.check(p, select(p, [links[0]['id']]))
        self.check(p, select(p, [c['id'] for c in links]), 'identity_constraint_violated')
        p.pop('identity_constraints')
        result = self.check(p, select(p, [c['id'] for c in links]))
        self.assertTrue(any(len(group) == 3 for group in result['identity_components']))

    def test_malformed_payloads_return_errors(self):
        for payload in [None, [], {}, {'schema_version': '0.1'}]:
            self.assertFalse(validate_candidate_selection(payload, None)['contract_valid'])
        p = copy.deepcopy(self.package)
        p['candidates'][0]['scope'] = {'polarity': 'positive', 'contexts': [None]}
        self.check(p, empty_selection(p), 'invalid_context')


if __name__ == '__main__':
    unittest.main()
