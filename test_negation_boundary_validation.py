"""Saved-reply validator regression: scope is not a new world fact."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.runtime/parliament'))
from global_workspace.local_specialists import _candidate_from_data, _landscape_semantic_errors, extract_scenario_facts
from global_workspace.models import WorkspaceBroadcast


class NegationBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((ROOT/'fixtures/tram_negation_boundary_reply.json').read_text())

    def mapping(self,actions,cases):
        return _landscape_semantic_errors(actions,dict(zip(actions,cases)),'NONE','STATED_FACTS',
            'prefer fewer deaths','NONE')

    def parse(self,data):
        return _candidate_from_data('utilitarian',self.fixture['actions'],deepcopy(data),
            WorkspaceBroadcast(),'NONE',extract_scenario_facts(self.fixture['scenario']),
            scenario_text=self.fixture['scenario'],grounded_effects=self.fixture['grounded_effects'],
            available_propositions=self.fixture['proposition_ledger'])

    def test_negated_paraphrases_and_action_permutations(self):
        actions=self.fixture['actions'];reply=self.fixture['reply']
        for negative in (reply['l']['A1'],'By not pulling, Dana lets the tram continue toward ten workers.',
                         'Without pulling the lever, Dana leaves the tram moving toward ten workers.',
                         'PERMISSIBLE: Refraining from pulling avoids intervention but allows ten workers to die.',
                         'Abstaining from pulling leaves the tram on its original trajectory.'):
            cases=[reply['l']['A0'],negative]
            for order in ((0,1),(1,0)):
                with self.subTest(case=negative,order=order):
                    errors=self.mapping([actions[i] for i in order],[cases[i] for i in order])
                    self.assertFalse(any('opposing action' in e for e in errors),errors)

    def test_opposite_action_and_nonnegation_inversions_still_flagged(self):
        actions=self.fixture['actions'];reply=self.fixture['reply']
        errors=self.mapping(actions,[reply['l']['A1'],reply['l']['A0']])
        self.assertTrue(any('A0 tracks the opposing action' in e for e in errors),errors)
        # Choosing the correct verb does not excuse swapping typed outcomes or
        # citing the opposing branch's effect as this branch's evidence.
        for mode in ('counts','citations'):
            bad=deepcopy(reply)
            if mode=='counts':
                bad['ev']['A0']['value']=10;bad['ev']['A1']['value']=1
            else:
                bad['ev']['A0']['source_effect_ids']=['E7']
                bad['ev']['A1']['source_effect_ids']=['E3']
            candidate=self.parse(bad)
            self.assertTrue(candidate.expected_value_validation_errors,mode)
        self.assertTrue(any('A1 tracks the opposing action' in e for e in errors),errors)
        errors=self.mapping(['Protect the child','Protect the researcher'],
            ['Researcher receives the scarce treatment','Researcher receives treatment and survives'])
        self.assertTrue(any('A0 tracks the opposing action' in e for e in errors),errors)

    def test_saved_hypothetical_example_keeps_grounded_boundary(self):
        reply=deepcopy(self.fixture['reply']);before=deepcopy(reply)
        candidate=self.parse(reply)
        self.assertEqual(reply,before)
        self.assertFalse(any('numeric reversal threshold lacks' in e for e in candidate.framework_validation_errors),
                         candidate.framework_validation_errors)
        self.assertEqual(candidate.factual_reversal_threshold,reply['ft'])
        self.assertEqual(candidate.graph_update_proposal['operation'],'BOUNDARY')
        self.assertEqual(candidate.graph_update_proposal['clauses'][0]['threshold'],10)

    def test_unverified_parameter_is_not_committed_and_assertion_still_rejected(self):
        reply=deepcopy(self.fixture['reply']);reply['ft']='If pulling would kill 20 workers, reconsider.'
        reply['gu']['clauses'][0]['threshold']=20
        reply['gu']['clauses'][0]['source_text']='If pulling would kill 20 workers'
        candidate=self.parse(reply)
        self.assertEqual(candidate.factual_reversal_threshold,reply['ft'])
        self.assertEqual(candidate.graph_update_proposal['operation'],'NONE')
        self.assertTrue(any('hypothetical reversal boundary parameter is unverified' in e
                            for e in candidate.framework_validation_errors),candidate.framework_validation_errors)
        reply['ft']='There are currently 20 workers on the main track.'
        candidate=self.parse(reply)
        self.assertEqual(candidate.factual_reversal_threshold,'NONE')
        self.assertEqual(candidate.graph_update_proposal['operation'],'NONE')
        self.assertTrue(any('numeric reversal threshold lacks' in e for e in candidate.framework_validation_errors))


if __name__=='__main__':unittest.main()
