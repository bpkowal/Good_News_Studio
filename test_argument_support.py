from copy import deepcopy
import json
from pathlib import Path
import unittest

from argument_support import compile_argument,apply_revision,assess_support,project,resolve_citation,compile_legacy_review
from deontology_repair_cards import fingerprint


def statement(handle,kind='BELIEF',text=None,polarity='AFFIRMED',hypotheses=()):
    return dict(handle=handle,kind=kind,text=text or handle,polarity=polarity,modality='ASSERTED',
                hypothesis_ids=list(hypotheses),citations=['E1:/provenance/0/excerpt'])


class SupportTests(unittest.TestCase):
    evidence=[{'ref':'E1','value':{'provenance':[{'excerpt':'Jo agrees to hold the gate for rescue.'}]}}]

    def build(self,rows=None,routes=None):
        rows=rows or [statement('a'),statement('b'),statement('c')]
        routes=routes or [dict(operation='SOURCE',inputs=[],output='a',explanation='source'),
                          dict(operation='SOURCE',inputs=[],output='b',explanation='independent'),
                          dict(operation='PARAPHRASE',inputs=['a'],output='c',explanation='first route'),
                          dict(operation='PARAPHRASE',inputs=['b'],output='c',explanation='second route')]
        state=compile_argument('WORLD',self.evidence,'ISSUE',rows,routes)
        for rid in state['routes']:
            state=assess_support(state,rid,'SUPPORTED',reviewer='TEST_REVIEWER',rationale='Authored fixture support')
        return state

    def test_alias_resolution_preserves_location_and_does_not_certify_entailment(self):
        c=resolve_citation('E1:/provenance/0/excerpt',self.evidence)
        self.assertEqual(c['value_path'],'/provenance/0/excerpt')
        self.assertEqual(c['raw'],'E1:/provenance/0/excerpt')
        self.assertEqual(c['semantic_support'],'NOT_ASSESSED')
        with self.assertRaises(ValueError):resolve_citation('E1:/missing',self.evidence)
        with self.assertRaises(ValueError):resolve_citation({'evidence_id':'E1','value_path':'/provenance/0/excerpt','quote':'fabricated'},self.evidence)

    def test_source_citation_alone_does_not_promote_unreviewed_claim(self):
        state=compile_argument('WORLD',self.evidence,'ISSUE',[statement('a')],
            [dict(operation='SOURCE',inputs=[],output='a',explanation='source')])
        self.assertEqual(project(state)['statements'][state['handles']['a']]['support_status'],'UNASSESSED')

    def test_withdrawal_is_explicit_and_alternative_support_survives(self):
        state=self.build();before=deepcopy(state)
        revised=apply_revision(state,[dict(action='withdraw',target='a',reason='Premise withdrawn')],
            author='REVIEWER',expected_sha256=fingerprint(state))
        view=project(revised)
        self.assertEqual(view['statements'][state['handles']['a']]['lifecycle'],'WITHDRAWN')
        self.assertNotIn(state['handles']['a'],view['active_statement_ids'])
        self.assertEqual(view['statements'][state['handles']['c']]['support_status'],'SUPPORTED')
        self.assertEqual(state,before)
        revised=apply_revision(revised,[dict(action='withdraw',target='b',reason='Other route removed')],
            author='REVIEWER',expected_sha256=fingerprint(revised))
        self.assertEqual(project(revised)['statements'][state['handles']['c']]['support_status'],'REQUIRES_REASSESSMENT')

    def test_qualification_allocates_new_identity_without_rewriting_old_statement(self):
        state=self.build();replacement=statement('new','BELIEF','A qualified belief',polarity='UNKNOWN')
        del replacement['handle']
        revised=apply_revision(state,[dict(action='qualify',target='a',handle='qualified',statement=replacement,reason='Retreat from certainty')],
            author='REVIEWER',expected_sha256=fingerprint(state))
        self.assertNotEqual(revised['handles']['qualified'],state['handles']['a'])
        self.assertEqual(revised['statements'][state['handles']['a']],state['statements'][state['handles']['a']])
        self.assertEqual(project(revised)['statements'][state['handles']['a']]['lifecycle'],'SUPERSEDED')

    def test_conditional_ancestry_propagates_through_intermediate_claims(self):
        rows=[statement('maxim','MAXIM'),statement('a'),statement('b')]
        routes=[dict(operation='ASSUMPTION',inputs=[],output='maxim',explanation='Hypothesis'),
                dict(operation='PARAPHRASE',inputs=['maxim'],output='a',explanation='conditional'),
                dict(operation='PARAPHRASE',inputs=['a'],output='b',explanation='indirect')]
        state=self.build(rows,routes);view=project(state)
        self.assertEqual(view['statements'][state['handles']['b']]['support_status'],'CONDITIONAL')
        self.assertEqual(view['statements'][state['handles']['b']]['effective_hypotheses'],[state['handles']['maxim']])

    def test_supersession_preserves_attached_support_and_is_atomic_if_invalid(self):
        state=self.build();replacement=statement('new','BELIEF','Jo agrees to hold the gate.')
        del replacement['handle']
        operation=dict(action='supersede',target='a',handle='new',statement=replacement,
            support=dict(operation='SOURCE',inputs=[],output='new',explanation='Weaker exact source reading'),reason='Remove overclaim')
        revised=apply_revision(state,[operation],author='R',expected_sha256=fingerprint(state))
        self.assertTrue(any(r['output']==revised['handles']['new'] for r in revised['routes'].values()))
        before=deepcopy(state);operation['support']['output']='missing'
        with self.assertRaises(ValueError):apply_revision(state,[operation],author='R',expected_sha256=fingerprint(state))
        self.assertEqual(state,before)

    def test_denied_belief_cannot_supply_affirmative_practical_requirement(self):
        rows=[statement('goal','PURPOSE'),statement('belief','BELIEF',polarity='DENIED'),
              statement('use','PURPOSIVE_USE'),statement('conclusion','INTENDED_USE')]
        routes=[dict(operation='SOURCE',inputs=[],output=h,explanation='fixture') for h in ('goal','belief','use')]
        routes.append(dict(operation='PRACTICAL_REASONING',inputs=['goal','belief','use'],output='conclusion',explanation='invalid inference'))
        state=self.build(rows,routes);view=project(state)
        self.assertEqual(view['statements'][state['handles']['conclusion']]['support_status'],'REQUIRES_REASSESSMENT')
        self.assertIn('AFFIRMATIVE_PRACTICAL_PREMISES_MISSING',str(view))

    def test_unknown_under_hypothesis_remains_unknown_not_asserted_promotion(self):
        rows=[statement('maxim','MAXIM'),statement('claim',polarity='UNKNOWN')]
        state=self.build(rows,[dict(operation='ASSUMPTION',inputs=[],output='maxim',explanation='hypothesis'),
            dict(operation='PARAPHRASE',inputs=['maxim'],output='claim',explanation='unknown')])
        self.assertEqual(project(state)['statements'][state['handles']['claim']]['support_status'],'UNASSESSED')

    def test_withdrawn_hypothesis_reopens_even_a_directly_cited_scoped_claim(self):
        state=self.build([statement('m','MAXIM'),statement('a',hypotheses=['m'])],
            [dict(operation='ASSUMPTION',inputs=[],output='m',explanation='hypothesis'),
             dict(operation='SOURCE',inputs=[],output='a',explanation='scoped claim')])
        revised=apply_revision(state,[dict(action='withdraw',target='m',reason='Interpretation rejected')],
            author='REVIEWER',expected_sha256=fingerprint(state))
        self.assertEqual(project(revised)['statements'][state['handles']['a']]['support_status'],'REQUIRES_REASSESSMENT')

    def test_stale_revision_and_mutated_statement_rejected(self):
        state=self.build()
        with self.assertRaises(ValueError):apply_revision(state,[],author='R',expected_sha256='wrong')
        state['statements'][state['handles']['a']]['text']='rewritten'
        with self.assertRaises(ValueError):project(state)

    def test_withdrawing_support_keeps_statement_and_independent_route(self):
        state=self.build()
        route=next(rid for rid,r in state['routes'].items() if r['output']==state['handles']['c'])
        revised=apply_revision(state,[dict(action='withdraw',target=route,reason='Reject one derivation')],
            author='REVIEWER',expected_sha256=fingerprint(state))
        view=project(revised)
        self.assertEqual(view['support_routes'][route]['status'],'WITHDRAWN')
        self.assertEqual(view['statements'][state['handles']['c']]['lifecycle'],'ACTIVE')
        self.assertEqual(view['statements'][state['handles']['c']]['support_status'],'SUPPORTED')

    def test_partial_construction_preserves_valid_parts_and_missing_route_problem(self):
        state=compile_argument('WORLD',self.evidence,'ISSUE',[statement('a'),statement('b')],
            [dict(operation='ASSUMPTION',inputs=['a'],output='b',explanation='mistyped inference')],partial=True)
        self.assertEqual(len(state['construction_problems']),1)
        self.assertEqual(len(state['statements']),2)
        self.assertTrue(all(r['operation']=='SOURCE' for r in state['routes'].values()))
        self.assertTrue(all(v['support_status']=='UNASSESSED' for v in project(state)['statements'].values()))

    def test_saved_live_withdrawals_compile_without_rewriting_provider_records(self):
        root=Path(__file__).resolve().parent
        for case in ('means_not_automatically_prohibited','rule_not_automatically_priority'):
            path=root/'diagnostics/inference_boundary_matched_live_20261023'/case/'typed/call.json'
            raw=path.read_bytes();reply=json.loads(raw)['response'];old=reply['reasoning_revision']['claims'][-1]
            state=self.build([statement('old',old['kind'],old['text'])],
                [dict(operation='SOURCE',inputs=[],output='old',explanation='fixture baseline, not semantic certification')])
            # Replay an explicit, reviewed withdrawal. Never infer lifecycle
            # from the word 'unsupported' in free-form bridge prose.
            revised=apply_revision(state,[dict(action='withdraw',target='old',reason=reply['bridge'])],
                author='CONTENT_REVIEW',expected_sha256=fingerprint(state))
            self.assertEqual(project(revised)['active_statement_ids'],[])
            self.assertEqual(path.read_bytes(),raw)

    def test_legacy_adapter_applies_only_exact_named_retractions(self):
        root=Path(__file__).resolve().parent
        for case in ('means_not_automatically_prohibited','rule_not_automatically_priority'):
            p=json.loads((root/'diagnostics/inference_boundary_matched_live_20261023'/case/'typed/call.json').read_text())
            state=compile_legacy_review(p['challenge'],p['response'],'WORLD')
            retired=[r['text'] for r in project(state)['statements'].values() if r['lifecycle']=='WITHDRAWN']
            self.assertEqual(retired,p['response']['withdrawn_claims'])


if __name__=='__main__':unittest.main()
