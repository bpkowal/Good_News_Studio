import json
from pathlib import Path
import unittest

from blueprint_cloze_chooser import choose_by_cloze
from blueprint_deontic_rules import source_rule_bundles, rule_force
from blueprint_discourse import build_deontic_rule, relent_workspace_split
from blueprint_proposal_contract import validate_proposal
from test_blueprint_cloze_chooser import Script


ROOT = Path(__file__).resolve().parent
FIXTURE = json.loads((ROOT / 'fixtures/deontic_lab_cloze.json').read_text())
TEXT = FIXTURE['source']


class DeonticRuleBundleTests(unittest.TestCase):
    def test_lab_preserves_two_distinct_scoped_norms_without_occurrences(self):
        world, notes = build_deontic_rule(TEXT, {})
        norms = world['normative_propositions']
        self.assertEqual([n['force'] for n in norms], ['obligation', 'permission'])
        self.assertEqual([p['predication'] for p in world['propositions']], [
            'keep a containment door closed while a hazardous experiment is running',
            'opening it only after the experiment has stopped',
        ])
        self.assertEqual(norms[0]['clause_ids'], ['C0'])
        self.assertEqual(norms[1]['clause_ids'], ['C1'])
        self.assertNotEqual(norms[0]['governed_proposition_id'], norms[1]['governed_proposition_id'])
        self.assertEqual(norms[1]['bearer_party_id'], '')
        self.assertTrue(any('unstated' in n for n in notes))
        self.assertTrue(any('unresolved: it' in n for n in notes))
        self.assertEqual(world['actions'], [])
        self.assertEqual(world['effects'], [])
        self.assertEqual(relent_workspace_split(world)['established'], [])
        self.assertTrue(all(p['kind'] == 'OTHER' for p in world['parties'] if 'rule' in p['label']))

    def test_exact_saved_bad_answers_now_recover_and_select_deontic(self):
        result = choose_by_cloze(TEXT, Script(FIXTURE['ranking'], FIXTURE['answers']))
        self.assertEqual(result['chosen_blueprint_id'], 'deontic_rule')
        self.assertEqual(result['status'], 'FILLED')
        self.assertEqual(validate_proposal(result['graph']), [])
        row = next(r for r in result['considered'] if r['blueprint_id'] == 'deontic_rule')
        self.assertEqual(row['slots']['deontic_words'], 'requires')
        original = next(i for i in row['items'] if i['id'] == 'deontic_words')
        self.assertEqual(original['completion'], 'requires, allows')
        self.assertEqual(original['reason'], 'recovered_from_source_rule_complement')
        ids = [h['hypothesis_id'] for item in row['items'] for h in item['hypotheses']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(result['graph']['slot_bindings']['deontic_rule_bindings']), 2)
        self.assertEqual(len(result['graph']['candidate']['world_model']['normative_propositions']), 2)

    def test_force_belongs_to_this_norm_not_an_unrelated_modal(self):
        package = {'candidates': [{'type': 'MODALITY', 'value': 'permission'}]}
        world, _ = build_deontic_rule('Mira must not open the door. Ben may leave.', {
            'deontic_words': 'must not', 'governed_action': 'not open the door', 'bearer': 'Mira',
        }, package)
        self.assertEqual(world['normative_propositions'][0]['force'], 'prohibition')

    def test_source_offsets_survive_whitespace_changes(self):
        for text in [TEXT, TEXT.replace(' requires ', '  requires\n'), TEXT.replace('. The rule', '.\nThe rule')]:
            bundles = source_rule_bundles(text)
            self.assertEqual(len(bundles), 2)
            for b in bundles:
                self.assertEqual(text[b['operator_start']:b['operator_end']], b['operator'])
                self.assertEqual(text[b['content_start']:b['content_end']], b['governed_action'])
                for scope in b['scope']:
                    self.assertEqual(text[scope['start']:scope['end']], scope['text'])

    def test_distinct_bearers_and_authorities_do_not_get_merged(self):
        text = ('The safety policy requires Mira to close the door. '
                'The rescue protocol permits Ben to open the gate only after the alarm stops.')
        world, _ = build_deontic_rule(text, {})
        parties = {p['party_id']: p['label'] for p in world['parties']}
        norms = world['normative_propositions']
        self.assertEqual([parties[n['bearer_party_id']] for n in norms], ['Mira', 'Ben'])
        self.assertEqual([parties[n['authority_party_id']] for n in norms], ['The safety policy', 'The rescue protocol'])
        self.assertIn('only after', world['propositions'][1]['predication'])

    def test_negated_requirement_is_not_a_prohibition(self):
        text = 'The rule does not require Mira to close the door.'
        self.assertEqual(source_rule_bundles(text), [])
        world, _ = build_deontic_rule(text, {
            'deontic_words': 'does not require', 'governed_action': 'close the door', 'bearer': 'Mira',
        })
        self.assertEqual(world['normative_propositions'][0]['force'], 'unresolved')

    def test_negation_inside_scope_does_not_negate_governed_action(self):
        text = 'The rule requires Mira to wait while the machine is not running.'
        world, _ = build_deontic_rule(text, {})
        self.assertEqual(world['propositions'][0]['polarity'], 'positive')
        self.assertEqual(world['propositions'][0]['predication'], 'wait while the machine is not running')

    def test_mechanical_numeric_reported_and_questioned_rules_are_not_extracted(self):
        cases = ['The hinge allows the door to open.',
                 'The experiment requires ten minutes.',
                 'Ada claims the rule requires Mira to close the door.',
                 'Does the rule require Mira to close the door?']
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(source_rule_bundles(text), [])

    def test_existing_single_rule_shape_is_preserved(self):
        world, _ = build_deontic_rule('Ada must deliver medicine to Ben.', {
            'deontic_words': 'must', 'governed_action': 'deliver medicine to Ben', 'bearer': 'Ada',
        })
        self.assertEqual(world['propositions'][0]['proposition_id'], 'PR1')
        self.assertEqual(world['normative_propositions'][0]['norm_id'], 'N0')
        self.assertEqual(world['normative_propositions'][0]['force'], 'obligation')
        self.assertEqual(len(world['normative_propositions']), 1)
        self.assertEqual(world['effects'], [])
        self.assertEqual(rule_force('allowed to'), 'permission')
        self.assertEqual(rule_force('not required to'), 'unresolved')
        self.assertEqual(rule_force('not permitted to'), 'prohibition')

    def test_explicit_secondary_cloze_rule_keeps_its_own_force(self):
        text = 'Mira must close the door. Ben is permitted to leave only after the alarm stops.'
        world, _ = build_deontic_rule(text, {
            'deontic_words': 'must', 'governed_action': 'close the door', 'bearer': 'Mira',
            'second_deontic_words': 'permitted to',
            'second_governed_action': 'leave only after the alarm stops', 'second_bearer': 'Ben',
        })
        self.assertEqual([n['force'] for n in world['normative_propositions']], ['obligation', 'permission'])
        self.assertEqual(world['propositions'][1]['predication'], 'leave only after the alarm stops')
        self.assertNotEqual(world['normative_propositions'][0]['bearer_party_id'],
                            world['normative_propositions'][1]['bearer_party_id'])


if __name__ == '__main__':
    unittest.main()
