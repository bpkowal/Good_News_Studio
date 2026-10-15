from copy import deepcopy
import json
from pathlib import Path
import unittest
from blueprint_spatial_projection import spatial_cells,typed_reader_view,RELATIONS

ROOT=Path(__file__).resolve().parent
class SpatialReaderTests(unittest.TestCase):
    def setUp(self):
        self.report=json.loads((ROOT/'diagnostics/decision_topography_suite_20261009/trolley/DEMO_REPORT.json').read_text())
        self.text=self.report['scenario'];self.world=self.report['world']

    def test_source_cells_separate_entities_places_and_movement(self):
        cells=spatial_cells(self.text,self.world)
        self.assertEqual((cells['movement_entity'],cells['instrument'],cells['destination']),('P7','P6','P9'))
        edges={(e['type'],e['source'],e['target']) for e in cells['scene_relations']}
        self.assertIn(('ON','P4','P8'),edges);self.assertIn(('ON','P2','P9'),edges)
        self.assertIn(('HEADING_TOWARD','P7','P4'),edges)
        for e in cells['scene_relations']+cells['branch_transitions']:
            support=e['evidence'];self.assertEqual(self.text[support['start']:support['end']],support['text'])

    def test_view_does_not_rewrite_frozen_records_or_original_kinds(self):
        before=deepcopy(self.world);view=typed_reader_view(self.text,self.world)
        self.assertEqual(self.world,before)
        self.assertEqual(next(p for p in self.world['parties'] if p['party_id']=='P7')['kind'],'PROCESS')
        self.assertEqual(next(n for n in view['nodes'] if n['id']=='SC_P7')['kind'],'entity')
        self.assertIn('If she pulls the lever',view['mermaid']);self.assertIn('If she does not',view['mermaid'])
        self.assertTrue(any(n.get('display_reconstruction') for n in view['nodes']))

    def test_every_arrow_is_typed_and_has_the_declared_direction(self):
        view=typed_reader_view(self.text,self.world);nodes={n['id']:n for n in view['nodes']}
        self.assertEqual(len(nodes),len(view['nodes']))
        for edge in view['edges']:
            self.assertEqual((nodes[edge['source']]['kind'],nodes[edge['target']]['kind']),RELATIONS[edge['type']])
        self.assertEqual(sum(' -->|' in line or ' -.->|' in line for line in view['mermaid'].splitlines()),len(view['edges']))

    def test_omission_does_not_invent_a_route_or_positive_action_occurrence(self):
        view=typed_reader_view(self.text,self.world)
        self.assertNotIn('REMAINS_ON', {e['type'] for e in view['edges']})
        self.assertNotIn('AWAY_FROM', {e['type'] for e in view['edges']})
        self.assertIn('Physical continuation not recorded',view['mermaid'])
        self.assertEqual([q['action_id'] for q in view['cells']['open_questions']],['A1'])
        self.assertTrue(any(e['type']=='AFFECTS' and e['target']=='A0_P6' for e in view['edges']))
        self.assertFalse(any(e['type']=='AFFECTS' and e['target']=='A0_P7' for e in view['edges']))

    def test_negation_and_attribution_do_not_become_affirmative_locations(self):
        for first in ('It is not true that five workers are on the main track.',
                      'Alice says five workers are on the main track.'):
            text=first+self.text[self.text.index('.')+1:]
            cells=spatial_cells(text,self.world)
            self.assertFalse(any(e['type']=='ON' and e['source']=='P4' for e in cells['scene_relations']))

    def test_whitespace_does_not_change_location_bindings(self):
        base=spatial_cells(self.text,self.world)
        changed=spatial_cells(self.text.replace('on the main track','on\n  the main track'),self.world)
        shape=lambda cells:{(e['type'],e['source'],e['target']) for e in cells['scene_relations']}
        self.assertEqual(shape(base),shape(changed))

    def test_absent_or_ambiguous_mechanism_abstains_without_new_nodes(self):
        self.assertIsNone(spatial_cells('There are five workers.',self.world))
        self.assertIsNone(spatial_cells(self.text.replace('that switches','that might switch'),self.world))

    def test_cloze_reuses_cells_without_new_required_blanks(self):
        from blueprint_cloze_chooser import choose_by_cloze,_BY_ID
        from test_trolley_cloze_recovery import Script
        case=json.loads((ROOT/'fixtures/trolley_scene_cloze_cases.json').read_text())['cases'][0]
        original=len(_BY_ID['omission_harm']['items'])
        result=choose_by_cloze(case['source'],Script(case['ranking'],case['answers']))
        cells=result['graph']['slot_bindings']['physical_scene_cells']
        self.assertTrue(cells['observational_only']);self.assertEqual(len(_BY_ID['omission_harm']['items']),original)
        self.assertEqual(result['graph']['candidate']['world_model']['schema_version'],'1.4')

if __name__=='__main__':unittest.main()
