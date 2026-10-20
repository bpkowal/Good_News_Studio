from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import parsing_game_Z10 as z10
from blueprint_cloze_chooser import assess_question, choose_by_cloze
from blueprint_explicit_choices import extract_explicit_choices
from blueprint_graph_amendments import expand_candidates
from blueprint_primitive_composition import extract_inventory, compose, append_composition
from blueprint_proposal_contract import validate_proposal
from run_blueprint_parliament import _admit_candidates, DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON
from test_blueprint_cloze_chooser import Script


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/deontic_lab_cloze.json').read_text())
TEXT = FIXTURE['source']


class DeonticChoiceCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = z10.export_candidate_graph(TEXT)
        cls.inventory = extract_inventory(TEXT, cls.package)
        cls.proposal = compose(TEXT, cls.inventory, assess_question(TEXT))

    def test_two_positive_choices_and_no_invented_omission(self):
        choices = self.inventory['explicit_choice']
        self.assertTrue(choices['complete'])
        self.assertEqual([b['action'] for b in choices['branches']],
                         ['open the door immediately', 'stop the experiment and wait'])
        self.assertEqual([b['outcomes'][0]['binding'] for b in choices['branches']],
                         ['explicit_action_anaphor', 'explicit_duration_anaphor'])
        self.assertEqual(choices['exclusivity'], 'UNKNOWN')
        self.assertFalse(choices['exhaustive'])

    def test_risks_duration_and_rules_survive_construction(self):
        self.assertEqual(validate_proposal(deepcopy(self.proposal)), [])
        world = self.proposal['candidate']['world_model']
        self.assertEqual(len(world['actions']), 2)
        self.assertEqual(len(world['effects']), 5)
        self.assertEqual([n['force'] for n in world['normative_propositions']], ['obligation', 'permission'])
        self.assertIn('while a hazardous experiment is running', world['propositions'][0]['predication'])
        self.assertIn('only after', world['propositions'][1]['predication'])
        effects = {e['effect_id']: e for e in world['effects']}
        exposure = next(e for e in effects.values() if e['predicate'] == 'expose')
        death = next(e for e in effects.values() if e['predicate'] == 'die')
        delay = effects[death['source_effect_ids'][0]]
        self.assertEqual(delay['quantities'], ['ten minutes'])
        self.assertEqual(delay['effect_kind'], 'OTHER')
        self.assertEqual(exposure['effect_kind'], 'OTHER')
        self.assertEqual(exposure['polarity'], 'ADVERSE')
        for effect in (exposure, death):
            self.assertEqual(effect['modality'], 'POSSIBLE')
            self.assertEqual(effect['likelihood_qualifiers'], ['may'])
        self.assertEqual(world['conditions'], [])
        self.assertFalse(any(e['predicate'] in {'save', 'live', 'survive'} for e in effects.values()))
        self.assertTrue(all(link['link_relation'] == 'ENABLES' for link in world['causal_links']))

    def test_rule_baseline_is_preserved_beside_amendment_and_composition(self):
        original = choose_by_cloze(TEXT, Script(FIXTURE['ranking'], FIXTURE['answers']))
        before = deepcopy(original)
        result = append_composition(TEXT, self.inventory, expand_candidates(TEXT, self.package, original, self.inventory))
        self.assertEqual(original, before)
        baseline = next(a for a in before['candidate_attempts'] if a['blueprint_id'] == 'deontic_rule')
        self.assertIn(baseline, result['candidate_attempts'])
        amendment = next(a for a in result['candidate_attempts'] if a.get('variant') == 'amended')
        self.assertEqual(amendment['blueprint_id'], 'deontic_rule')
        self.assertEqual(len(amendment['proposal']['candidate']['world_model']['actions']), 2)
        self.assertEqual(validate_proposal(amendment['proposal']), [])
        self.assertEqual(result['candidate_attempts'][-1]['variant'], 'composition')

    def test_removing_a_risk_keeps_a_specific_missing_branch(self):
        text = TEXT.replace('the technician may die during that delay', 'the technician remains inside')
        inventory = extract_inventory(text, z10.export_candidate_graph(text))
        proposal = compose(text, inventory, assess_question(text))
        self.assertEqual(proposal['status'], 'WITHHELD')
        self.assertIn('stop the experiment and wait', str(proposal['construction_problems']))

    def test_no_cross_object_or_conditional_scope_rebinding(self):
        for text in [TEXT.replace('Mira can open the door now', 'Mira can open the window now'),
                     TEXT.replace('Mira can open the door now', 'If the alarm rings, Mira can open the door now'),
                     TEXT.replace('Stopping the experiment first', 'Stopping the pump first')]:
            with self.subTest(text=text):
                self.assertFalse(extract_explicit_choices(text)['complete'])

    def test_negated_outcome_is_not_promoted_and_renaming_is_not_a_new_family(self):
        self.assertFalse(extract_explicit_choices(TEXT.replace('may die', 'may not die'))['complete'])
        variant = TEXT.replace('Mira', 'Nora').replace('ten minutes', 'seven minutes').replace('may die', 'could die')
        inventory = extract_explicit_choices(variant)
        self.assertTrue(inventory['complete'])
        self.assertEqual(inventory['branches'][0]['actor'], 'Nora')
        self.assertEqual(inventory['branches'][1]['duration']['quantity'], 'seven minutes')
        self.assertEqual(inventory['branches'][1]['outcomes'][0]['likelihood'], 'could')

    def test_different_action_vocabulary_uses_the_same_construction(self):
        text = ('Iris can unlock the gate now, but doing so might expose nearby staff to smoke. '
                'Pausing the trial first takes two hours; the technician could die during that delay. '
                'Should Iris unlock the gate immediately or pause the trial and wait?')
        inventory = extract_explicit_choices(text)
        self.assertTrue(inventory['complete'])
        self.assertEqual([b['predicate'] for b in inventory['branches']], ['unlock', 'pause'])
        self.assertEqual(inventory['branches'][1]['duration']['quantity'], 'two hours')

    def test_whitespace_does_not_change_choice_identity_or_duration(self):
        for text in [TEXT.replace(' may ', '  may '), TEXT.replace('. Mira', '.\nMira'),
                     TEXT.replace('the door', 'the  door', 1)]:
            choices = extract_explicit_choices(text)
            self.assertTrue(choices['complete'])
            self.assertEqual(choices['branches'][1]['duration']['quantity'], 'ten minutes')
            for branch in choices['branches']:
                self.assertIn(branch['action'], text)
                self.assertIn(branch['outcomes'][0]['text'], text)

    @unittest.skipUnless(DEFAULT_PARLIAMENT_ROOT.exists() and DEFAULT_PARLIAMENT_PYTHON.exists(), 'native runtime unavailable')
    def test_full_occurrence_native_admission_and_frozen_replay(self):
        proposal = deepcopy(self.proposal)
        blueprint = {'candidate_attempts': [{'rank': 0, 'blueprint_id': 'primitive_composition',
                     'variant': 'composition', 'selected': True, 'proposal': proposal}]}
        with tempfile.TemporaryDirectory() as temp:
            chosen, records = _admit_candidates(blueprint, TEXT, self.package, Path(temp),
                                               DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON)
            self.assertIsNotNone(chosen, records)
            self.assertEqual(records[0]['status'], 'ADMITTED')
            self.assertTrue(records[0]['admission']['frozen_trace_valid'])
            self.assertEqual(records[0]['admission']['effects'], 5)
            trace = json.loads(Path(records[0]['trace_path']).read_text())
            world = trace['action_source_grounding']['world_model_1_4']
            possible = [e for e in world['effects'] if e.get('modality') == 'POSSIBLE']
            self.assertEqual(len(possible), 2)
            self.assertEqual(len(world['normative_propositions']), 2)


if __name__ == '__main__':
    unittest.main()
