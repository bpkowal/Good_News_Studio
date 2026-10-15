"""Production producer-to-checker regression on the actual ten-worker Cloze input."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from blueprint_cloze_chooser import _graph
from blueprint_physical_scene import attach_physical_scene
from blueprint_spatial_projection import typed_reader_view
from parliament_runtime import DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON
from run_blueprint_parliament import _admit_candidates

ROOT=Path(__file__).resolve().parent
SAVED=ROOT/'diagnostics/tram_ten_workers_maxims_live_20261015_network/pipeline/blueprint_candidate.json'


def rebuild_ten_worker_proposal():
    saved=json.loads(SAVED.read_text());blueprint=saved['blueprint_result']
    winner=next(row for row in blueprint['considered'] if row['blueprint_id']==blueprint['chosen_blueprint_id'])
    proposal=_graph(saved['scenario'],deepcopy(winner),deepcopy(blueprint['question']))
    return saved,proposal


class TrolleyCausalProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.saved,cls.proposal=rebuild_ten_worker_proposal()

    def test_intervention_has_causal_spine_and_omission_keeps_dependency(self):
        world=self.proposal['candidate']['world_model']
        links={(l['source_id'],l['target_id']):l['link_relation'] for l in world['causal_links']}
        self.assertEqual(links[('E1','E2')],'CAUSES')
        self.assertEqual(links[('E2','E3')],'CAUSES')
        self.assertEqual(links[('E2','E4')],'CAUSES')
        self.assertEqual(links[('E5','E6')],'ENABLES')
        before=self.saved['blueprint_result']['graph']['candidate']['world_model']
        for key in ('parties','actions','effects','conditions'):
            self.assertEqual(world[key],before[key],key)
        self.assertEqual(len(world['causal_links']),len(before['causal_links']))
        proof=[r for r in self.proposal['construction_provenance'] if r['atom_id'] in {'relation:0','relation:1','relation:2'}]
        self.assertEqual(len(proof),3)
        self.assertTrue(all(r['origin']=='STRUCTURALLY_DERIVED' and r['evidence'] for r in proof))

    def test_uncertain_or_attributed_mechanism_does_not_upgrade_legacy_links(self):
        original=self.saved['blueprint_result']['graph']['candidate']['world_model']
        source=self.saved['scenario']
        for text in (source.replace('that redirects','that might redirect'),
                     source.replace('Dana can pull','Alice says Dana can pull'),
                     source.replace('that redirects','that does not redirect')):
            with self.subTest(text=text):
                before=deepcopy(original)
                actual=attach_physical_scene(text,original,[])
                self.assertEqual(actual,before)
                self.assertEqual(original,before)

    def test_native_admission_and_symmetric_reader_keep_correct_arrow_types(self):
        p=self.proposal
        with tempfile.TemporaryDirectory() as tmp:
            chosen,attempts=_admit_candidates({'chosen_blueprint_id':p['blueprint_id'],'proposals':[deepcopy(p)]},
                self.saved['scenario'],self.saved['package'],Path(tmp),DEFAULT_PARLIAMENT_ROOT,DEFAULT_PARLIAMENT_PYTHON)
            self.assertIsNotNone(chosen,attempts)
            trace=json.loads(Path(chosen['record']['trace_path']).read_text())
            world=trace['action_source_grounding']['world_model']
            view=typed_reader_view(self.saved['scenario'],world)
            self.assertTrue(any(e['type']=='CAUSES' and e['source']=='A0' for e in view['edges']))
            self.assertTrue(any(e['type']=='BRANCH_FOR' and e['source']=='A1' for e in view['edges']))
            self.assertFalse(any(e['type']=='CAUSES' and e['source']=='A1' for e in view['edges']))
            branch_groups=sorted({n['group'] for n in view['nodes'] if n['group'].startswith('branch')})
            self.assertEqual(branch_groups,['branch_A0','branch_A1'])
            counts=[sum(n['group']==g for n in view['nodes']) for g in branch_groups]
            self.assertEqual(counts[0],counts[1])


if __name__=='__main__':unittest.main()
