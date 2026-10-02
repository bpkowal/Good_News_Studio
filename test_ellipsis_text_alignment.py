import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import ellipsis_proposer as proposer
import parsing_game_S as s
import parsing_game_Z5 as z5
import parsing_game_Z8 as z8
from candidate_validation import empty_selection, validate_candidate_selection
from test_scope_semantics import select


CASES = json.loads((Path(__file__).parent / 'fixtures/ellipsis_text_invariance.json').read_text())


class OriginalTextAlignmentTests(unittest.TestCase):
    def test_proposals_index_original_document(self):
        model = proposer.load_proposer()
        for case in CASES:
            for text in case['texts']:
                with self.subTest(text=text):
                    doc = s.get_nlp()(text)
                    implicit = model.propose(text)
                    explicit = model.propose(text, doc=doc)
                    self.assertEqual(implicit, explicit)
                    self.assertTrue(explicit['stripping'])
                    for item in explicit['stripping']:
                        self.assertEqual(doc[item['verb_index']].text, item['verb'])
                        self.assertEqual(doc[item['not_index']].lower_, 'not')
                        remnant = doc[item['remnant_index']]
                        self.assertEqual(text[remnant.idx:remnant.idx + len(remnant.text)], remnant.text)
                        self.assertEqual(proposer._nominal_text(remnant, None if item['prep_index'] is None else doc[item['prep_index']]), item['remnant'])
                        if item['subject_index'] is not None:
                            self.assertEqual(proposer._nominal_text(doc[item['subject_index']]), item['subject'])

    def test_frozen_roles_scope_and_exact_evidence(self):
        for case in CASES:
            for text in case['texts']:
                with self.subTest(text=text):
                    old = z5.export_candidate_graph(text)
                    self.assertTrue(any(c['type'] == 'PREDICATION' and c['provenance'][0]['method'] == 'stripping_not_nominal'
                                        for c in old['candidates']))
                    p = z8.export_candidate_graph(text)
                    self.assertEqual(p['document']['text'], text)
                    self.assertEqual(len(p['reconstructions']), 1)
                    nodes = {n['id']: n for n in p['nodes']}
                    table = {c['id']: c for c in p['candidates']}
                    bundle = p['reconstructions'][0]
                    roles = [table[i] for i in bundle['participant_candidate_ids']]
                    actual = [[c['value'], ' '.join(nodes[c['arguments']['mention']]['label'].split())] for c in roles]
                    self.assertEqual(actual, case['roles'])
                    for c in roles:
                        self.assertEqual(c['scope']['polarity'], 'negative')
                        self.assertEqual([ctx['kind'] for ctx in c['scope']['contexts']], ['modal_choice'])
                        result = validate_candidate_selection(p, select(p, c['id']))
                        self.assertTrue(result['contract_valid'], result['errors'])
                    for e in p['evidence']:
                        self.assertEqual(text[e['start']:e['end']], e['text'])
                    self.assertTrue(validate_candidate_selection(p, empty_selection(p))['contract_valid'])

    def test_mismatched_document_rejected(self):
        doc = s.get_nlp()('Maria can save the child, but not the dog.')
        with self.assertRaisesRegex(ValueError, 'exact original text'):
            proposer.load_proposer().propose('Maria  can save the child, but not the dog.', doc=doc)

    def test_supplied_document_is_used_without_reparse(self):
        text = 'Maria\ncan save the child, but\nnot the dog.'
        doc = s.get_nlp()(text)
        gate = Mock()
        gate.decide.return_value = dict(present=True, score=1, features={})
        parser = Mock(side_effect=AssertionError('should reuse supplied Doc'))
        result = proposer.load_proposer().propose(text, nlp=parser, gate=gate, doc=doc)
        self.assertTrue(result['stripping'])
        parser.assert_not_called()

    def test_whitespace_does_not_turn_not_only_into_stripping(self):
        for text in ['Maria can save the child, but not only the dog.',
                     'Maria can save the child, but not\nonly the dog.']:
            self.assertEqual(proposer.load_proposer().propose(text)['stripping'], [])


if __name__ == '__main__':
    unittest.main()
