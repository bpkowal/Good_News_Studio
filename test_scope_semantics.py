"""Hand-authored semantic projections, independent of exporter IDs and scores."""
import copy
import json
from pathlib import Path
import unittest

from parsing_game_T import export_candidate_graph, empty_selection, validate_candidate_selection


def select(package, ident):
    result = empty_selection(package)
    table = {c['id']: c for c in package['candidates']}
    pending, selected, nodes = [ident], set(), set()
    while pending:
        cid = pending.pop()
        if cid in selected:
            continue
        selected.add(cid)
        candidate = table[cid]
        pending.extend(candidate['requires'])
        nodes.update(candidate['arguments'].values())
        for ctx in candidate['scope']['contexts']:
            nodes.update(ctx[k] for k in ('source_mention_id', 'report_proposition_id', 'condition_proposition_id')
                         if ctx.get(k))
    result['selected_node_ids'] = sorted(nodes)
    result['selected_candidate_ids'] = sorted(selected)
    return result


class FrozenScopeTests(unittest.TestCase):
    def test_frozen_semantics_and_selection(self):
        fixtures = json.loads((Path(__file__).parent / 'fixtures/T_scope_semantics.json').read_text())
        for index, fixture in enumerate(fixtures):
            with self.subTest(text=fixture['text']):
                package = export_candidate_graph(fixture['text'], package_id=f'fixture_scope_{index}')
                self.assertEqual(package, export_candidate_graph(fixture['text'], package_id=f'fixture_scope_{index}'))
                nodes = {n['id']: n for n in package['nodes']}
                anchors = {c['arguments']['proposition']: c for c in package['candidates'] if c['type'] == 'PREDICATION'}
                def contexts(candidate):
                    return [ctx['kind'] + (':' + nodes[ctx['source_mention_id']]['label']
                            if ctx['kind'] == 'attributed' and ctx['source_mention_id'] else '')
                            for ctx in candidate['scope']['contexts']]
                actual = [[nodes[pid]['predicate'], c['scope']['polarity'], contexts(c)] for pid, c in anchors.items()]
                self.assertEqual(actual, fixture['predicates'])
                self.assertEqual([[c['value'], c['scope']['polarity']] for c in package['candidates']
                                  if c['type'] == 'MODALITY'], fixture['modalities'])
                before = copy.deepcopy(package)
                for c in package['candidates']:
                    if c['type'] == 'PARTICIPANT':
                        anchor = anchors[c['arguments']['proposition']]
                        self.assertEqual(c['scope']['polarity'], anchor['scope']['polarity'])
                        self.assertEqual([x for x in contexts(c) if x != 'modal'], contexts(anchor))
                    if c['type'] == 'CONDITIONAL_ON':
                        anchor = anchors[c['arguments']['consequence']]
                        self.assertEqual(contexts(c), [x for x in contexts(anchor) if x not in {'conditional', 'modal_choice'}])
                    selection = select(package, c['id'])
                    result = validate_candidate_selection(package, selection)
                    self.assertTrue(result['contract_valid'], result['errors'])
                self.assertEqual(package, before)

    def test_modal_abstention_remains_qualified_and_provisional(self):
        p = export_candidate_graph('Maria can leave.')
        subject = next(c for c in p['candidates'] if c['type'] == 'PARTICIPANT')
        selection = select(p, subject['id'])
        result = validate_candidate_selection(p, selection)
        self.assertTrue(result['contract_valid'], result['errors'])
        self.assertIn(subject['id'], result['provisional_candidate_ids'])
        self.assertEqual(subject['scope']['contexts'][0]['kind'], 'modal_choice')
        ability = next(c for c in p['candidates'] if c['value'] == 'ability')
        selection['selected_candidate_ids'].append(ability['id'])
        q = next(q for q in p['open_questions'] if q['kind'] == 'modality')
        r = next(r for r in selection['question_resolutions'] if r['question_id'] == q['id'])
        r.update(status='resolved_by_selection', selected_candidate_ids=[ability['id']])
        result = validate_candidate_selection(p, selection)
        self.assertTrue(result['contract_valid'], result['errors'])
        self.assertNotIn(subject['id'], result['provisional_candidate_ids'])

    def test_scope_references_and_versions_cannot_be_forged(self):
        p = export_candidate_graph('Officials said Maria can leave.')
        anchor = next(c for c in p['candidates'] if c['type'] == 'PREDICATION' and c['scope']['contexts'])
        anchor['requires'] = []
        self.assertIn('missing_report_dependency', [e['code'] for e in validate_candidate_selection(p, empty_selection(p))['errors']])
        p = export_candidate_graph('Maria can leave.')
        p['schema_version'] = '0.1'
        self.assertFalse(validate_candidate_selection(p, empty_selection(p))['contract_valid'])
        p = export_candidate_graph('Maria can leave. Anna can stay.')
        scoped = [c for c in p['candidates'] if c['type'] == 'PREDICATION']
        scoped[0]['scope']['contexts'][0]['modality_choice_set_id'] = scoped[1]['scope']['contexts'][0]['modality_choice_set_id']
        self.assertIn('modal_choice_proposition_mismatch', [e['code'] for e in validate_candidate_selection(p, empty_selection(p))['errors']])

    def test_lexical_modality_stays_open(self):
        p = export_candidate_graph('Maria is not required to leave.')
        leave = next(n['id'] for n in p['nodes'] if n.get('predicate') == 'leave')
        anchor = next(c for c in p['candidates'] if c['type'] == 'PREDICATION' and c['arguments']['proposition'] == leave)
        result = validate_candidate_selection(p, select(p, anchor['id']))
        self.assertTrue(result['contract_valid'], result['errors'])
        self.assertIn(anchor['id'], result['provisional_candidate_ids'])


if __name__ == '__main__':
    unittest.main()
