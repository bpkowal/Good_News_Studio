"""Physical causal support and proposed practical reasoning are independent."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.runtime/parliament'))
from global_workspace.deontology_ledger import CandidateMaxim, DutyAssessmentProposal, calibrate_deontological_adjudication, harm_relation_conflicts_with_graph
from global_workspace.scenario_semantics import compile_scenario_graph
import test_candidate_maxims as maxim_fixture
from test_candidate_maxims import maxim


class MeansEndReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        maxim_fixture.CandidateMaximTests.setUpClass()
        cls.parser=maxim_fixture.CandidateMaximTests()
        trace=json.loads(maxim_fixture.CandidateMaximTests.trace.read_text());cls.world=deepcopy(trace['action_source_grounding']['world_model'])
        # Declared oracle: intervention causes redirection and its two sibling outcomes.
        for link in cls.world['causal_links']:
            if link['action_id']=='A0':link['relation']='CAUSES'
        cls.graph=compile_scenario_graph(maxim_fixture.CandidateMaximTests.context['scenario'],maxim_fixture.CandidateMaximTests.context['actions'],world_model=cls.world)
        cls.actions={n.attributes['canonical_action_id']:n for n in cls.graph.nodes.values() if n.kind=='ACTION' and 'canonical_action_id' in n.attributes}
        cls.proposal=DutyAssessmentProposal(**cls.parser.parse(maxim_fixture.CandidateMaximTests.reply).deontological_ledger_proposal['assessments'][0])

    def item(self,**changes):
        mx=maxim();mx.update(instrumental_belief='DOES_NOT_RELY',instrumental_use='DOES_NOT_USE',
            practical_reasoning='The proposed policy employs the lever and redirects the threat, not the worker.',
            means_effect_ids=['E2'],end_effect_ids=['E4'],action_permissibility='IMPERMISSIBLE')
        mx.update(changes)
        return self.proposal.model_copy(update={'candidate_maxim':CandidateMaxim(**mx),
            'reason':'Whether the negative duty prohibits foreseen lethal redirection requires justification.'})

    def errors(self,item,graph=None):
        return calibrate_deontological_adjudication(graph or self.graph,self.actions['A0'],item).errors

    def test_intervention_causation_does_not_turn_omission_into_doing(self):
        self.assertEqual(harm_relation_conflicts_with_graph(self.graph,self.proposal,action=self.actions['A0']),'')
        omitted=self.proposal.model_copy(update={'action_id':'A1','protected_party':'ten workers','harm_relation':'ALLOWING_HARM'})
        self.assertEqual(harm_relation_conflicts_with_graph(self.graph,omitted,action=self.actions['A1']),'')
        bad=omitted.model_copy(update={'harm_relation':'DOING_HARM'})
        self.assertIn('agent-caused settled welfare harm',harm_relation_conflicts_with_graph(self.graph,bad,action=self.actions['A1']))
        bad=self.proposal.model_copy(update={'harm_relation':'ALLOWING_HARM'})
        self.assertIn('inconsistent with an agent-caused',harm_relation_conflicts_with_graph(self.graph,bad,action=self.actions['A0']))

    def test_sibling_death_is_not_means_to_rescue_and_refs_must_resolve(self):
        warning='candidate maxim chosen means has no admitted path to its stated end'
        self.assertNotIn(warning,self.errors(self.item()))
        self.assertIn(warning,self.errors(self.item(means='Tram strikes side-track worker',means_effect_ids=['E3'])))
        for refs in (['E7'],['invented']):
            self.assertIn('candidate maxim cites an unknown or other-action effect',self.errors(self.item(means_effect_ids=refs)))
        # An empty normative end binding does not invent a physical end or action.
        self.assertNotIn(warning,self.errors(self.item(end_effect_ids=[])))

    def test_physical_path_is_not_sufficient_for_purposive_use(self):
        # Make the injury a declared intermediate cause solely for this oracle.
        from global_workspace.semantic_graph import SemanticEdge
        graph=deepcopy(self.graph)
        graph.add_edge(SemanticEdge('A0:WORLD_EFFECT:E3','CAUSES','A0:WORLD_EFFECT:E4'))
        item=self.item(person_role='INTENDED_MEANS',means_effect_ids=['E3']).model_copy(update={'means_relation':'INTENDED_AS_MEANS'})
        warning='candidate maxim intended use lacks a proposed instrumental belief and purposive use'
        self.assertIn(warning,self.errors(item,graph))
        mx=item.candidate_maxim.model_copy(update={'instrumental_belief':'RELIES_ON_PERSON','instrumental_use':'USES_PERSON',
            'practical_reasoning':'In this hypothetical policy the agent relies on the person stopping the tram and employs that person to do so.'})
        supported=item.model_copy(update={'candidate_maxim':mx})
        errors=self.errors(supported,graph)
        self.assertNotIn(warning,errors)
        self.assertFalse(any('intended-as-means classification lacks' in e for e in errors),errors)
        self.assertIn('candidate maxim denies intended use while proposing instrumental belief and purposive use',
            self.errors(self.item(instrumental_belief='RELIES_ON_PERSON',instrumental_use='USES_PERSON')))

    def test_unknown_practical_reasoning_stays_unresolved_and_legacy_maxim_readable(self):
        errors=self.errors(self.item(instrumental_belief='UNRESOLVED',instrumental_use='UNRESOLVED'))
        self.assertIn('candidate maxim practical-reasoning qualification remains unresolved',errors)
        old=CandidateMaxim(**maxim())
        self.assertNotIn('instrumental_belief',old.model_dump(exclude_none=True))
        corrected=self.item().model_copy(update={'resolution_status':'CONTESTED','verdict':'CONFLICTED',
            'governing_norm':'UNRESOLVED','priority_basis':'UNRESOLVED'})
        self.assertFalse(any('agent-caused settled' in e for e in self.errors(corrected)))

    def test_saved_live_review_corrects_dependency_not_only_its_label(self):
        folder=ROOT/'diagnostics/causal_means_reasoning_20261015'
        initial=json.loads((folder/'live_frameworks/deontological/model_calls.json').read_text())[0]['response']
        revised=json.loads((folder/'live_review/deontological/model_calls.json').read_text())[0]['response']
        items=[]
        for response in (initial,revised):
            data=json.loads(response['choices'][0]['text'])
            items.append(DutyAssessmentProposal(**self.parser.parse(data).deontological_ledger_proposal['assessments'][0]))
        before,after=items
        self.assertEqual(before.means_relation,'INTENDED_AS_MEANS')
        self.assertIn('E3',before.candidate_maxim.means_effect_ids)
        self.assertIn('candidate maxim chosen means has no admitted path to its stated end',self.errors(before))
        self.assertEqual(after.means_relation,'FORESEEN_SIDE_EFFECT')
        self.assertEqual(after.candidate_maxim.means_effect_ids,['E2'])
        self.assertEqual(after.candidate_maxim.end_effect_ids,['E4'])
        self.assertEqual(after.candidate_maxim.instrumental_belief,'DOES_NOT_RELY')
        self.assertEqual(after.candidate_maxim.instrumental_use,'DOES_NOT_USE')
        self.assertEqual(self.errors(after),())
        # This is not a test that the agent must recommend pulling.
        self.assertEqual(after.verdict,before.verdict)


if __name__=='__main__':unittest.main()
