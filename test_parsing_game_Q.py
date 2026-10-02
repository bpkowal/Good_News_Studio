import unittest
from unittest.mock import patch
import numpy as np
import parsing_game_Q as q
import test_parsing_game_P as previous
import test_parsing_game_M as legacy
import audit_parsing_game_Q as audit


class QTests(previous.PInheritedTests):
    @classmethod
    def setUpClass(cls):
        replacement = patch.object(previous, 'p', q)
        replacement.start()
        cls.addClassCleanup(replacement.stop)
        super().setUpClass()

    def test_quantity_and_legacy_causal_suites(self):
        for name in ('TRAIN_EXAMPLES', 'TEST_EXAMPLES', 'NEGATION_GENERALIZATION_EXAMPLES',
                     'ROBUSTNESS_EXAMPLES', 'LEXICAL_HOLDOUT', 'UNKNOWN_HOLDOUT', 'EPISTEMIC_HOLDOUT'):
            self.assertEqual(q.evaluate_suite(getattr(q, name), self.policy)[0]['end_to_end_correct'], 1)
        self.assertEqual(q.evaluate_multi_suite(q.MULTI_RELATION_HOLDOUT, self.policy)[0]['exact_claim_list_accuracy'], 1)
        self.assertEqual(self.parse('Insulation reduces noise.')['events'][0]['change']['direction'], 'decrease')

    def test_rolling_results_serialize_entailment_and_display_it(self):
        self.assertEqual(q.USER_SUITE_PATH.name, 'parsing_game_Q_user_probes.json')
        entries = q.evaluate_user_suite(q.append_user_probe([], 'Insomnia caused by stress yesterday.'), self.policy)
        self.assertEqual(entries[0]['last_result']['propositions'][0]['structural_hypothesis']['status'], 'provisional')

    def test_fragment_repair_preserves_original_parse_and_blocks_commitment(self):
        text = 'Insomnia caused by stress yesterday.'
        original = [(t.text,t.tag_,t.dep_,t.head.i) for t in q.get_nlp()(text)]
        result = self.parse(text)
        claim, = result['causal_claims']
        self.assertEqual((claim['source'],claim['target']), ('stress','Insomnia'))
        self.assertFalse(claim['eligible_for_world_state'])
        self.assertIn('provisional_structure_requires_validation',claim['validation_reasons'])
        frame, = result['propositions']
        self.assertIsNone(frame['object'])
        self.assertEqual(frame['occurrence_status'],'unknown')
        self.assertEqual(frame['structural_hypothesis']['original_tag'],'VBD')
        self.assertTrue(any(a['head_text']=='yesterday' and a['role']=='temporal'
                            for a in frame['attachments']))
        self.assertEqual(original,[(t.text,t.tag_,t.dep_,t.head.i) for t in q.get_nlp()(text)])
        with patch.object(q,'REPAIR_ENABLED',False):
            self.assertIsNone(self.parse(text)['causal_claims'][0]['decision'])

    def test_passive_contrasts_and_held_out_nouns(self):
        for text in ('Damage caused by noon.', 'The bat flew by the blind man.',
                     'Damage caused by cutting cables.', 'Damage not caused by corrosion.',
                     'Damage may be caused by corrosion.',
                     'Officials said damage was caused by corrosion.',
                     'Was damage caused by corrosion?'):
            with self.subTest(text=text):
                result=self.parse(text)
                self.assertFalse(any(c['eligible_for_world_state'] for c in result['causal_claims']))
                self.assertFalse(any(c.get('source')=='cables' for c in result['causal_claims']))
        for text, source in (('Flooding caused by rainfall overnight.','rainfall'),
                             ('Damage caused by corrosion.','corrosion'),
                             ('Erosion triggered by vibration.','vibration'),
                             ('Nausea induced by turbulence.','turbulence')):
            with self.subTest(text=text):
                claims=self.parse(text)['causal_claims']
                self.assertTrue(any(c.get('source','')==source for c in claims))
        result=self.parse('Rain caused flooding by noon.')
        self.assertEqual(result['causal_claims'][0]['target'],'flooding')
        self.assertTrue(result['causal_claims'][0]['eligible_for_world_state'])

    def test_audit_raw_baseline_and_annotation_boundaries(self):
        evidence=q.collect_evidence('Rain causes flooding.')
        annotation=dict(entity1='Rain',entity2='flooding',correct_action=0,expected_eligible=True)
        row=audit.audit_candidate(evidence,self.policy,annotation)
        self.assertTrue(row['pair_correct'])
        self.assertTrue(row['cem_correct'])
        self.assertEqual(row['baseline_action'],0)
        weights=np.zeros_like(self.policy.weights)
        weights[2,q.FEATURE_NAMES.index('bias')]=1
        bad=audit.audit_candidate(evidence,q.CEMPolicy(weights),annotation)
        self.assertFalse(bad['cem_correct'])
        self.assertFalse(bad['final_claim']['eligible_for_world_state'])
        self.assertIn('D_policy_error_candidate_requires_feature_review',bad['diagnostic_flags'])
        summary=audit.summarize([row,bad])
        self.assertEqual(summary['paired_comparison']['baseline_only_correct'],1)
        unlabelled=audit.audit_candidate(evidence,self.policy)
        self.assertIsNone(unlabelled['cem_correct'])
        self.assertIsNone(audit.summarize([unlabelled])['false_commitments']['rate'])
        contradiction=audit.audit_candidate(evidence,self.policy,dict(annotation,correct_action=1))
        self.assertEqual(len(audit.collision_candidates([row,contradiction])),1)
        mismatch=audit.audit_candidate(evidence,self.policy,dict(annotation,entity2='causes'))
        self.assertIn('A_argument_mismatch',mismatch['diagnostic_flags'])
        outside=audit.audit_candidate(evidence,self.policy,dict(relation_scope='outside_causal_actions'))
        self.assertIn('B_outside_action_space',outside['diagnostic_flags'])


class QEmbeddingTests(previous.PEmbeddingTests):
    @classmethod
    def setUpClass(cls):
        replacement = patch.object(previous, 'p', q)
        replacement.start()
        cls.addClassCleanup(replacement.stop)
        super().setUpClass()


if __name__ == '__main__':
    unittest.main()
