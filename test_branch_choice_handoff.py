from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from parliament_source_constructions import build_packet, render_advisory, validate_packet
from blueprint_cloze_chooser import choose_by_cloze
from test_blueprint_cloze_chooser import Script
from parliament_runtime import DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON
import parsing_game_Z10 as z10

ROOT = Path(__file__).resolve().parent


class BranchChoiceHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case=json.loads((ROOT/'fixtures/trolley_scene_cloze_cases.json').read_text())['cases'][0]
        cls.text=case['source']
        cls.result=choose_by_cloze(cls.text,Script(case['ranking'],case['answers']))
        cls.proposal=cls.result['graph'];cls.world=cls.proposal['candidate']['world_model']
        cls.package=z10.export_candidate_graph(cls.text)

    def test_choice_context_is_bound_to_world_without_promoting_it(self):
        before=deepcopy(self.world)
        packet=build_packet(self.package,self.proposal)
        validate_packet(packet,self.text,world=self.world)
        context=packet['branch_context'];omission=next(c for c in context['choices'] if c['kind']=='omission')
        self.assertEqual(omission['omitted_intervention_id'],'A0')
        self.assertEqual(omission['decision_maker_id'],'P1')
        self.assertEqual(context['physical_scene']['open_questions'],[])
        prompt=render_advisory(packet,self.text)
        self.assertIn('not physical causation',prompt)
        self.assertIn('legacy ENABLES',prompt)
        self.assertEqual(self.world,before)
        for change in ('world','participants','intervention'):
            broken=deepcopy(packet)
            if change=='world':broken['branch_context']['world_sha256']='bad'
            elif change=='participants':broken['branch_context']['choices'][1]['decision_maker_id']='unknown'
            else:broken['branch_context']['choices'][1]['intervention_description']='invented'
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_packet(broken,self.text,world=self.world)

    def test_quantity_and_conditional_scope_survive_continuation(self):
        continuation=next(e for e in self.world['effects'] if e['predicate']=='head')
        self.assertIn('five',continuation['quantities'])
        self.assertEqual(continuation['action_id'],'A1')
        self.assertTrue(any(c['polarity']=='NEGATED' for c in self.world['conditions']))

    def test_flexible_physical_state_closes_certain_conditions(self):
        from blueprint_graph_amendments import branch_world
        text='If Maria pulls the lever, the signal will activate.'
        world,_=branch_world(text,{'actor':'Maria','condition':'If Maria pulls the lever',
            'bearer':'the signal','outcome':'the signal will activate',
            'construction':'physical_state','predicate':'activate'})
        self.assertEqual(world['effects'][-1]['modality'],'CERTAIN')
        self.assertEqual(world['effects'][-1]['condition_ids'],[])
        self.assertEqual(world['causal_links'][-1]['condition_ids'],[])
        from blueprint_primitive_composition import extract_inventory
        inventory=extract_inventory(self.text,self.package)
        self.assertEqual(len(inventory['branches']),2)
        ambiguous=self.text.replace('Should she pull the lever?', 'Anna is beside Maria. Should she pull the lever?')
        inventory=extract_inventory(ambiguous,z10.export_candidate_graph(ambiguous))
        self.assertEqual(inventory['branches'],[])

    def test_native_framework_primary_repair_and_contingency_receive_context(self):
        # Captures real specialist prompts, not a mock of the transport itself.
        from run_blueprint_parliament import _admit_candidates
        from blueprint_graph_amendments import expand_candidates
        from blueprint_primitive_composition import extract_inventory, append_composition
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            inventory=extract_inventory(self.text,self.package)
            expanded=expand_candidates(self.text,self.package,deepcopy(self.result),inventory=inventory)
            expanded=append_composition(self.text,inventory,expanded)
            chosen,attempts=_admit_candidates(expanded,self.text,self.package,out,
                DEFAULT_PARLIAMENT_ROOT,DEFAULT_PARLIAMENT_PYTHON)
            self.assertIsNotNone(chosen,attempts)
            self.assertTrue(any(a['blueprint_id']=='flexible_graph' and a['status']=='ADMITTED' for a in attempts),attempts)
            self.assertTrue(any(a['blueprint_id']=='primitive_composition' and a['status']=='ADMITTED' for a in attempts),attempts)
            trace=json.loads(Path(chosen['record']['trace_path']).read_text())
            trace['source_construction_advisory']=build_packet(self.package,chosen['proposal'],
                world=trace['action_source_grounding']['world_model'])
            path=out/'trace.json';path.write_text(json.dumps(trace))
            result=subprocess.run([str(DEFAULT_PARLIAMENT_PYTHON),str(ROOT/'diagnostics/trolley_choice_pipeline_20261014/capture.py'),
                str(path),str(out/'capture.json')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            capture=json.loads((out/'capture.json').read_text())
            self.assertEqual(len(capture['frameworks']),5)
            self.assertTrue(capture['world_unchanged'])


if __name__=='__main__':unittest.main()
