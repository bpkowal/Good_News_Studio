from copy import deepcopy
import json
import unittest
from pathlib import Path
import sys

from argument_construction import construct
from argument_support import project
from argument_construction import source_from_trace


class ConstructionTests(unittest.TestCase):
    def source(self):
        return dict(id='CASE',world_sha256='WORLD',scenario='Jo holds the gate to help rescue.',
            framework='Test the rescue maxim conditionally.',evidence=[{'ref':'S','value':'Jo holds the gate to help rescue.'}])

    def proposal(self):
        def node(handle,kind,text,hypotheses):
            return dict(handle=handle,kind=kind,text=text,polarity='AFFIRMED',modality='ASSERTED',
                        hypothesis_ids=hypotheses,citations=[])
        return dict(label='rescue policy',statements=[node('m','MAXIM','Rescue without employing a bystander.',[]),
            node('c','INTENDED_USE','No bystander is employed within this conditional policy.',['m'])],
            supports=[dict(operation='ASSUMPTION',inputs=[],output='m',explanation='Proposed policy'),
                      dict(operation='PARAPHRASE',inputs=['m'],output='c',explanation='Conditional analysis')],
            conclusion_handle='c',strongest_objection='This is not known actual intent.',
            answer_to_objection='It is a conditional interpretation only.',reversal_condition='Evidence of purposive employment would require another interpretation.')

    def test_two_candidates_compile_and_receive_local_critique_without_world_changes(self):
        source=self.source();before=deepcopy(source);proposal=self.proposal();calls=[]
        class Backend:
            def complete_json(inner,prompt,**kwargs):
                calls.append(prompt)
                if len(calls)==1:return {'candidates':[proposal,deepcopy(proposal)]}
                context=json.loads(prompt.split('\n',1)[1]);state=context['canonical_argument']
                return dict(assessments=[dict(support_id=rid,status='CONDITIONAL',rationale='Valid under proposed maxim only.') for rid in state['routes']],
                    revisions=[],strongest_remaining_conclusion='Conditional policy conclusion.',remaining_questions=['Actual policy unknown.'])
        result=construct(source,Backend())
        self.assertEqual(len(calls),3)
        self.assertEqual(source,before)
        self.assertFalse(result['operative'])
        self.assertTrue(all(c['status']=='STAGED_ARGUMENT' for c in result['candidates']))
        for c in result['candidates']:
            sid=c['argument']['handles']['c']
            self.assertEqual(c['view']['statements'][sid]['support_status'],'CONDITIONAL')

    def test_invalid_critic_patch_preserves_original_candidate(self):
        calls=[];proposal=self.proposal()
        class Backend:
            def complete_json(inner,prompt,**kwargs):
                calls.append(prompt)
                if len(calls)==1:return {'candidates':[proposal]}
                return dict(assessments=[],revisions=[dict(action='withdraw',target='other_action',reason='invalid scope')],
                    strongest_remaining_conclusion='Not trusted.',remaining_questions=[])
        result=construct(self.source(),Backend())
        c=result['candidates'][0]
        self.assertEqual(c['status'],'STAGED_PRIOR_RETAINED')
        self.assertEqual(c['argument'],c['compiled_before'])

    def test_more_than_two_candidates_stops_before_critique(self):
        outer=self
        class Backend:
            def complete_json(inner,*args,**kwargs):return {'candidates':[outer.proposal()]*3}
        with self.assertRaises(ValueError):construct(self.source(),Backend())

    def test_validated_world_adapter_preserves_source(self):
        root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'.runtime/parliament'))
        path=root/'diagnostics/deontic_choice_composition_20261016/live_final/frozen_world_trace.json'
        before=path.read_bytes()
        source=source_from_trace(path,'Test framework commitment')
        self.assertEqual(source['source_status'],'VALIDATED_FROZEN_WORLD')
        self.assertTrue(source['admitted_world']['actions'])
        self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
