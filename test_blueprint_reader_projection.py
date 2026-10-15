from copy import deepcopy
import json
from pathlib import Path
import unittest

from blueprint_reader_projection import action_bundles, bundle_mermaid, context_propositions
from run_blueprint_parliament import _mermaid_world


class ReaderProjectionTests(unittest.TestCase):
    def world(self):
        return json.loads((Path(__file__).resolve().parent/'frontend/dist/data/medicine.json').read_text())['diagrams'][0]['world']

    def test_bundle_reduces_boxes_without_changing_saved_world(self):
        world=self.world(); before=deepcopy(world)
        bundles=action_bundles(world)
        self.assertEqual(bundles, {'A0':['A0','E1'],'A1':['A1','E3']})
        source='\n'.join(_mermaid_world(world,'Medicine'))
        reduced=bundle_mermaid(source,bundles)
        self.assertNotIn('    E1[',reduced)
        self.assertIn('    A0 -->|"ENABLES"| E2',reduced)
        self.assertIn('    E2[',reduced)
        self.assertEqual(world,before)

    def test_different_or_qualified_events_never_bundle(self):
        for update in ({'outcome':'a separate transition'}, {'condition_ids':['C1']},
                       {'modality':'POSSIBLE'}, {'polarity':'ADVERSE'}):
            world=self.world(); world['effects'][0].update(update)
            self.assertNotIn('A0',action_bundles(world))

    def test_alias_mapping_never_rewrites_evidence_labels(self):
        source='flowchart LR\n    E1["E1 evidence"]\n    E2["Text mentions E1"]\n    E1 -->|"E1 source reference"| E2'
        reduced=bundle_mermaid(source,{'A0':['A0','E1']})
        self.assertIn('E2["Text mentions E1"]',reduced)
        self.assertIn('A0 -->|"E1 source reference"| E2',reduced)

    def test_governed_or_unresolved_propositions_stay_in_topology(self):
        world={'propositions':[{'proposition_id':'PR1','status':'ASSERTED'},
                               {'proposition_id':'PR2','status':'ASSERTED'},
                               {'proposition_id':'PR3','status':'GOVERNED'}],
               'modal_operators':[{'governed_proposition_id':'PR2'}]}
        self.assertEqual(context_propositions(world),['PR1'])
        source='flowchart LR\n    PR1(["Scene"])\n    P1 -.->|"MENTIONED_IN"| PR1\n    M1 -.->|"GOVERNS"| PR2'
        reduced=bundle_mermaid(source,{},['PR1'])
        self.assertNotIn('PR1',reduced)
        self.assertIn('M1 -.->|"GOVERNS"| PR2',reduced)


if __name__ == '__main__':
    unittest.main()
