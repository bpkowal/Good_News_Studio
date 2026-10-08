"""Replay the failed live Cloze answers against the original, richer input."""
import json
from pathlib import Path
import unittest
import tempfile
from copy import deepcopy

from blueprint_cloze_chooser import choose_by_cloze
from test_blueprint_cloze_chooser import Script

ROOT = Path(__file__).resolve().parent


class SavedTrolleyRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import parsing_game_Z10 as z10
        cls.cases = json.loads((ROOT/'fixtures/trolley_scene_cloze_cases.json').read_text())['cases']
        saved = cls.cases[0]
        cls.source = saved['source']
        cls.package = z10.export_candidate_graph(cls.source, package_id='trolley_scene_test')
        cls.result = choose_by_cloze(cls.source, Script(saved['ranking'], saved['answers']))
        cls.world = cls.result['graph']['candidate']['world_model']

    def test_conditions_do_not_become_whole_sentence_actions(self):
        self.assertTrue(self.world['actions'])
        for action in self.world['actions']:
            self.assertNotIn(',', action['intervention'])
            self.assertNotIn('kill', action['intervention'])
            self.assertNotIn('spare', action['intervention'])

    def test_possible_outcomes_and_links_keep_condition_references(self):
        conditions = {c['condition_id'] for c in self.world['conditions']}
        for record in self.world['effects'] + self.world['causal_links']:
            if record['modality'] in {'POSSIBLE', 'PROBABILISTIC'}:
                self.assertTrue(record['condition_ids'], record)
                self.assertLessEqual(set(record['condition_ids']), conditions)

    def test_killed_party_is_not_the_trolley(self):
        parties = {p['party_id']:p for p in self.world['parties']}
        killed = [e for e in self.world['effects'] if e['predicate'] == 'kill' and e['polarity'] == 'ADVERSE']
        self.assertEqual(len(killed), 2)
        for effect in killed:
            self.assertNotIn('trolley', parties[effect['party_id']]['label'].lower())

    def test_spared_and_killed_outcomes_are_separate(self):
        for action in self.world['actions']:
            outcomes = [e for e in self.world['effects'] if e['action_id'] == action['action_id']]
            self.assertTrue(any(e['polarity'] == 'ADVERSE' for e in outcomes))
            self.assertTrue(any(e['polarity'] == 'BENEFICIAL' for e in outcomes))

    def test_instrument_and_switching_process_have_distinct_bearers(self):
        parties = {p['party_id']:p['label'] for p in self.world['parties']}
        transition = next(e for e in self.world['effects'] if e['predicate'] == 'switch')
        direct = next(e for e in self.world['effects'] if e['effect_id'] in transition['source_effect_ids'])
        self.assertIn('lever', parties[direct['party_id']])
        self.assertIn('trolley', parties[transition['party_id']])
        self.assertEqual(transition['outcome'], 'switches the trolley onto a side track')
        self.assertIn(transition['source_proposition'], self.source)

    def test_scene_and_inability_reach_existing_proposition_layer(self):
        claims = [p['predication'] for p in self.world['propositions']]
        self.assertTrue(any('main track' in p for p in claims))
        self.assertTrue(any('side track, where one worker stands' in p for p in claims))
        self.assertIn('Maria cannot stop the trolley.', claims)
        for p in self.world['propositions']:
            self.assertIn(p['predication'], self.source)
            self.assertTrue(p['clause_ids'])
        self.assertFalse(any(e['predicate'] == 'stop' for e in self.world['effects']))

    def test_counted_references_share_two_human_bearers(self):
        people = [p for p in self.world['parties'] if p['kind'] in {'PERSON','HUMAN_GROUP'}]
        self.assertEqual({p['label'] for p in people}, {'Maria','one worker','five workers'})
        for action in self.world['actions']:
            outcomes = [e for e in self.world['effects'] if e['action_id'] == action['action_id']
                        and e['polarity'] in {'ADVERSE','BENEFICIAL'}]
            self.assertEqual(len({e['party_id'] for e in outcomes}), 2)

    def test_ambiguous_counted_antecedents_remain_unmerged(self):
        from blueprint_physical_scene import counted_antecedent
        self.assertEqual(counted_antecedent('Five workers stand here. Five workers stand there. Save the five.', 'the five'), '')
        self.assertEqual(counted_antecedent('One worker stands here. One worker stands there. Save that worker.', 'that worker'), '')

    def test_omission_has_expanded_action_and_source_backed_process(self):
        action = next(a for a in self.world['actions'] if 'not' in a['intervention'])
        self.assertEqual(action['intervention'], 'does not pull the lever')
        owned = [e for e in self.world['effects'] if e['action_id'] == action['action_id']]
        direct = next(e for e in owned if e['directness'] == 'DIRECT')
        self.assertEqual(direct['source_proposition'], 'does not')
        process = next(e for e in owned if e['directness'] == 'DOWNSTREAM' and e['polarity'] == 'NEUTRAL')
        self.assertEqual(process['source_proposition'], 'it will kill the five')
        self.assertNotEqual(process['party_id'], direct['party_id'])
        self.assertEqual(process['source_effect_ids'], [direct['effect_id']])
        proof = next(p for p in self.result['graph']['construction_provenance'] if p['atom_id'] == 'action:'+action['action_id'])
        self.assertEqual(proof['origin'], 'STRUCTURALLY_DERIVED')
        self.assertEqual(proof['evidence'], ['does not', 'If she pulls the lever'])

    def test_scene_binding_is_invariant_to_action_and_condition_record_order(self):
        from blueprint_physical_scene import attach_physical_scene
        for collections in [('conditions',), ('actions',), ('conditions','actions')]:
            with self.subTest(collections=collections):
                shuffled = deepcopy(self.world)
                for collection in collections:
                    shuffled[collection].reverse()
                revised = attach_physical_scene(self.source, shuffled)
                self.assertEqual({a['action_id']:a['intervention'] for a in revised['actions']},
                                 {a['action_id']:a['intervention'] for a in self.world['actions']})
                self.assertEqual({e['effect_id']:(e['outcome'], e['party_id']) for e in revised['effects']},
                                 {e['effect_id']:(e['outcome'], e['party_id']) for e in self.world['effects']})

    def test_expanded_omission_keeps_its_diagram_scope(self):
        from run_blueprint_parliament import _mermaid_world
        diagram = '\n'.join(_mermaid_world(self.world, 'test'))
        self.assertIn('CND2 -.->|"branch scope"| A1', diagram)
        self.assertNotIn('CND2 -.->|"branch scope"| A0', diagram)

    def test_wording_variants_preserve_both_branches_and_human_bearers(self):
        variants = [
            ('explicit omission', [('If she does not,', 'If she does not pull the lever,')]),
            ('newlines', [('. ', '.\n')]),
            ('extra spaces', [(' ', '  ')]),
            ('redirect', [('switches', 'redirects')]),
            ('digits', [('five', '5'), ('one worker', '1 worker'), ('the one', 'the 1')]),
            ('renamed actor and vehicle', [('Maria', 'Dana'), ('trolley', 'tram')]),
        ]
        for name, replacements in variants:
            with self.subTest(name=name):
                saved = deepcopy(self.cases[0])
                def replace(value):
                    for old, new in replacements:
                        value = value.replace(old, new)
                    return value
                source = replace(saved['source'])
                sheets = {key:{slot:replace(value) for slot,value in answers.items()}
                          for key,answers in saved['answers'].items()}
                result = choose_by_cloze(source, Script(saved['ranking'], sheets))
                proposals = [a['proposal'] for a in result['candidate_attempts'] if a['proposal']['status'] == 'FILLED']
                self.assertEqual(len(proposals), 2)
                for proposal in proposals:
                    world = proposal['candidate']['world_model']
                    humans = [p for p in world['parties'] if p['kind'] in {'PERSON','HUMAN_GROUP'}]
                    self.assertEqual(len(humans), 3, humans)
                    self.assertEqual(len(world['propositions']), 3)
                    self.assertTrue(any(e['predicate'] in {'switch','redirect'} for e in world['effects']))
                    for action in world['actions']:
                        harms = [e for e in world['effects'] if e['action_id'] == action['action_id'] and e['polarity'] == 'ADVERSE']
                        benefits = [e for e in world['effects'] if e['action_id'] == action['action_id'] and e['polarity'] == 'BENEFICIAL']
                        self.assertEqual((len(harms),len(benefits)), (1,1))
                        self.assertNotEqual(harms[0]['party_id'], benefits[0]['party_id'])
                if Path('/tmp/parliament-smoke-env/bin/python').exists() and Path('/tmp/parliament-smoke-614af0c').exists():
                    from run_blueprint_parliament import _admit_candidates
                    import parsing_game_Z10 as z10
                    with tempfile.TemporaryDirectory() as folder:
                        _, attempts = _admit_candidates(result, source,
                            z10.export_candidate_graph(source, package_id='variant_test'),
                            Path(folder), Path('/tmp/parliament-smoke-614af0c'),
                            Path('/tmp/parliament-smoke-env/bin/python'))
                        self.assertEqual({a['blueprint_id'] for a in attempts if a['status'] == 'ADMITTED'},
                                         {'omission_harm','conditional_outcome'}, attempts)

    def test_uncertain_or_negated_switching_never_becomes_certain(self):
        from blueprint_physical_scene import mechanism_from_source
        for replacement in ('might switch', 'does not switch', 'could switch'):
            text = self.source.replace('switches', replacement)
            self.assertIsNone(mechanism_from_source(text)[1])

    def test_capability_and_conditional_wording_build_the_same_world(self):
        from blueprint_selection_review import compare_admitted
        saved = self.cases[1]
        replay = choose_by_cloze(saved['source'], Script(saved['ranking'], saved['answers']))
        rows = [{'attempt':r, 'proposal':r['proposal']} for r in replay['candidate_attempts']
                if r['proposal'].get('status') == 'FILLED']
        self.assertEqual(len(rows), 2)
        self.assertEqual(compare_admitted(rows)['status'], 'EQUIVALENT')

    def test_full_negative_clause_keeps_negation_when_normalizing_action(self):
        from blueprint_physical_scene import attach_physical_scene
        source = self.source.replace('If she does not,', 'If she does not pull the lever,')
        world = deepcopy(self.world)
        world['actions'][1]['intervention'] = 'she does not pull the lever'
        world['conditions'][1]['description'] = 'If she does not pull the lever'
        revised = attach_physical_scene(source, world)
        self.assertEqual(revised['actions'][1]['intervention'], 'does not pull the lever')
        direct = next(e for e in revised['effects'] if e['action_id'] == 'A1' and e['directness'] == 'DIRECT')
        self.assertIn('not', direct['source_proposition'])
        self.assertEqual(len([e for e in revised['effects'] if e['predicate'] == 'switch']), 1)

    @unittest.skipUnless(Path('/tmp/parliament-smoke-env/bin/python').exists() and
                         Path('/tmp/parliament-smoke-614af0c').exists(), 'native checkout unavailable')
    def test_recovered_candidates_reach_native_admission(self):
        from run_blueprint_parliament import _admit_candidates
        with tempfile.TemporaryDirectory() as folder:
            _, attempts = _admit_candidates(self.result, self.source, self.package,
                Path(folder), Path('/tmp/parliament-smoke-614af0c'),
                Path('/tmp/parliament-smoke-env/bin/python'))
            admitted = [a for a in attempts if a['status'] == 'ADMITTED']
            self.assertEqual({a['blueprint_id'] for a in admitted}, {'omission_harm', 'conditional_outcome'}, attempts)


if __name__ == '__main__':
    unittest.main()
