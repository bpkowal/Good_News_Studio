"""Candidate maxims survive native transport and receive bounded, attributed review."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'.runtime/parliament'))
from global_workspace.deontology_ledger import CandidateMaxim, DutyAssessmentProposal, calibrate_deontological_adjudication
from global_workspace.local_specialists import _candidate_from_data, extract_scenario_facts
from global_workspace.models import WorkspaceBroadcast
from global_workspace.scenario_semantics import compile_scenario_graph
from logic_puzzles_frameworks import generate
from logic_puzzles_review import review, review_jobs, reconcile
from parliament_framework_presentation import duty_explanation


def maxim(omission=False):
    return {
        'circumstance': 'An existing tram threat can be redirected, but cannot be stopped.',
        'policy': ('Refrain from redirecting an existing threat to avoid imposing it on another person.'
                   if omission else 'Redirect an existing threat to rescue the larger group without employing another person.'),
        'end': 'Respect persons while responding to an existing threat.',
        'means': 'Refrain from pulling the lever.' if omission else 'Redirect the tram by pulling the lever.',
        'person_role': 'NOT_USED' if omission else 'FORESEEN_AFFECTED',
        'consent_constraint': 'Consent is not established; a rescue purpose cannot substitute for consent to use.',
        'universalization': 'Whether this policy respects each person while answering the rescue claim remains contested.',
        'action_permissibility': 'UNRESOLVED', 'maxim_assessment': 'UNRESOLVED',
    }


class CandidateMaximTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = json.loads((ROOT/'fixtures/tram_negation_boundary_reply.json').read_text())
        response=json.loads((ROOT/'diagnostics/tram_ten_workers_20261015/frameworks/deontological/model_calls.json').read_text())[0]['response']
        cls.reply=json.loads(response['choices'][0]['text'])
        cls.trace=ROOT/'diagnostics/tram_ten_workers_20261015/handoff/frozen_world_trace.json'
        world=json.loads(cls.trace.read_text())['action_source_grounding']
        cls.graph=compile_scenario_graph(cls.context['scenario'],cls.context['actions'],world['actions'],world_model=world['world_model'])

    def parse(self, reply):
        c=self.context
        return _candidate_from_data('deontological',c['actions'],deepcopy(reply),WorkspaceBroadcast(),
            'NONE',extract_scenario_facts(c['scenario']),scenario_text=c['scenario'],
            grounded_effects=c['grounded_effects'],available_propositions=c['proposition_ledger'])

    def augmented(self):
        reply=deepcopy(self.reply)
        for action,row in reply['dp'].items(): row['mx']=maxim(action=='A1')
        return reply

    def test_roundtrip_and_native_prompt_keeps_maxims_out_of_world(self):
        reply=self.augmented();before=self.trace.read_bytes()
        class Saved:
            model='test-saved-with-candidate-maxims'
            def complete_json(inner,prompt,**kwargs):
                self.assertIn('CANDIDATE INTERPRETATION',prompt)
                self.assertIn('including omissions',prompt)
                self.assertIn('mx',kwargs['schema']['properties']['dp']['properties']['A0']['required'])
                return {'choices':[{'text':json.dumps(reply)}]}
        with tempfile.TemporaryDirectory() as tmp:
            result=generate(self.trace,Path(tmp),['deontological'],lambda name:Saved(),call_budget=1)
            candidate=result['cycles'][0]['candidates'][0]
            rows=candidate['committed_native_ledger']['records']
            self.assertEqual(len(rows),2)
            for row in rows:
                self.assertEqual(row['candidate_maxim'],reply['dp'][row['canonical_action_id']]['mx'])
                self.assertEqual(row['maxim_origin'],'CANDIDATE_INTERPRETATION')
            self.assertEqual(self.trace.read_bytes(),before)
            self.assertNotIn('candidate_maxim',json.dumps(result['action_source_grounding']))
            # Review localization carries actual prior maxims, for both options.
            jobs=review_jobs(result)
            self.assertTrue(jobs)
            self.assertEqual(len(jobs[0]['maxim_review']['options']),2)
            self.assertFalse(jobs[0]['maxim_review']['world_fact'])
            self.assertIn('retract that inference',jobs[0]['question'])
            text=duty_explanation(candidate,result['action_source_grounding']['world_model'])
            self.assertIn('not an established intention',text)
            self.assertIn('agent proposes the maxim',text)
            # Neither a maxim nor a self-reported resolution erases a native objection.
            candidate['challenge_response']={'issue_id':jobs[0]['issue_id'],'disposition':'RESOLVED','answer':'The maxim is defensible.'}
            reviewed=reconcile(result,jobs,{'deontological':candidate},result['framework_proposition_ledgers'])
            self.assertTrue(all(i['objection_check']=='OPEN' for i in reviewed['targeted_review']['issues']))
            class UnchangedReview:
                def complete_json(inner,prompt,**kwargs):
                    self.assertIn('Redirect an existing threat',prompt)
                    self.assertIn('Challenge whether the omission policy',prompt)
                    self.assertIn('retract that inference',prompt)
                    self.assertIn('Your assigned challenges:',prompt)
                    data=deepcopy(reply)
                    data['qa']={'issue_id':kwargs['schema']['properties']['qa']['properties']['issue_id']['enum'][0],
                        'disposition':'RESOLVED','effect':'NO_CHANGE','current_position_effect':'NO_CHANGE',
                        'boundary_effect':'NOT_APPLICABLE','answer':'I retain the same maxim and claim the inference is resolved.',
                        'follow_up_question':'No further question.'}
                    return {'choices':[{'text':json.dumps(data)}]}
            recurrent=review(Path(tmp)/'framework_generation.json',Path(tmp)/'review',
                lambda name:UnchangedReview(),max_calls=1,allow_partial=False,allow_residual=False)
            self.assertEqual(recurrent['targeted_review']['adapter_calls'],{'deontological':1})
            self.assertTrue(all(i['objection_check']=='OPEN' for i in recurrent['targeted_review']['issues']))
            self.assertEqual(recurrent['action_source_grounding'],result['action_source_grounding'])

    def test_specific_contradiction_is_challenged_and_correction_removes_only_it(self):
        reply=self.augmented()
        item=DutyAssessmentProposal(**self.parse(reply).deontological_ledger_proposal['assessments'][0])
        action=next(n for n in self.graph.nodes.values() if n.attributes.get('canonical_action_id')=='A0' and n.kind=='ACTION')
        initial=calibrate_deontological_adjudication(self.graph,action,item)
        warning='merely-as-means conclusion lacks intended-as-means classification'
        self.assertIn(warning,initial.errors)
        corrected=item.model_copy(update={'reason':'The death is foreseen; whether redirection is wrongful remains contested.',
                                          'resolution_status':'CONTESTED','verdict':'CONFLICTED',
                                          'governing_norm':'UNRESOLVED','priority_basis':'UNRESOLVED'})
        revised=calibrate_deontological_adjudication(self.graph,action,corrected)
        self.assertNotIn(warning,revised.errors)
        self.assertEqual(revised.assessment.verdict,'CONFLICTED')
        denied=corrected.model_copy(update={'reason':'She does not use the worker as a mere means.'})
        self.assertNotIn(warning,calibrate_deontological_adjudication(self.graph,action,denied).errors)
        # A contradictory explicit person-role is still caught even under abstention.
        bad=corrected.model_copy(update={'candidate_maxim':corrected.candidate_maxim.model_copy(update={'person_role':'INTENDED_MEANS'})})
        self.assertIn('candidate maxim person-role conflicts with means classification',
                      calibrate_deontological_adjudication(self.graph,action,bad).errors)

    def test_bad_maxim_does_not_alone_prohibit_action_and_legacy_reply_survives(self):
        old=self.parse(self.reply)
        self.assertTrue(old.deontological_ledger_proposal['assessments'])
        self.assertNotIn('candidate_maxim',old.deontological_ledger_proposal['assessments'][0])
        item=DutyAssessmentProposal(**old.deontological_ledger_proposal['assessments'][1])
        mx=maxim(True);mx.update(action_permissibility='PERMISSIBLE',maxim_assessment='INDEFENSIBLE')
        item=item.model_copy(update={'candidate_maxim':CandidateMaxim(**mx)})
        action=next(n for n in self.graph.nodes.values() if n.attributes.get('canonical_action_id')=='A1' and n.kind=='ACTION')
        calibrated=calibrate_deontological_adjudication(self.graph,action,item)
        self.assertFalse(any('candidate maxim' in e for e in calibrated.errors),calibrated.errors)
        self.assertEqual(calibrated.assessment.verdict,'PERMISSIBLE')
        for verdict, permissibility in (('PROHIBITED','PERMISSIBLE'),('PERMISSIBLE','IMPERMISSIBLE')):
            inconsistent=item.model_copy(update={'verdict':verdict,
                'candidate_maxim':item.candidate_maxim.model_copy(update={'action_permissibility':permissibility})})
            self.assertIn('candidate maxim action-permissibility conflicts with duty verdict',
                calibrate_deontological_adjudication(self.graph,action,inconsistent).errors)

    def test_maxim_schema_rejects_claimed_intention_and_missing_slots(self):
        from global_workspace.deontology_ledger import CandidateMaxim
        from pydantic import ValidationError
        for change in ({'actor_intention_established':True},{'person_role':'WORLD_FACT'}):
            with self.assertRaises(ValidationError):CandidateMaxim(**dict(maxim(),**change))
        missing=maxim();del missing['means']
        with self.assertRaises(ValidationError):CandidateMaxim(**missing)


if __name__=='__main__':unittest.main()
