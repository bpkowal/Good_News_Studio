import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import patch

import parsing_game_R as r
import audit_parsing_game_R as audit
import test_parsing_game_Q as previous


class RTests(previous.QTests):
    @classmethod
    def setUpClass(cls):
        replacement = patch.object(previous, 'q', r)
        replacement.start()
        cls.addClassCleanup(replacement.stop)
        super().setUpClass()

    def test_rolling_results_serialize_entailment_and_display_it(self):
        self.assertEqual(r.USER_SUITE_PATH.name, 'parsing_game_R_user_probes.json')
        entries = r.evaluate_user_suite(r.append_user_probe([], 'Insomnia caused by stress yesterday.'), self.policy)
        hypothesis = entries[0]['last_result']['propositions'][0]['structural_hypothesis']
        self.assertEqual(hypothesis['interpretation_status'], 'provisional')
        self.assertTrue(all(hypothesis['required_evidence'].values()))
        self.assertFalse(hypothesis['eligible_for_world_state'])
        self.assertEqual(hypothesis['agent_or_cause']['text'], 'stress')
        self.assertEqual(hypothesis['affected']['text'], 'Insomnia')
        self.assertFalse(hypothesis['supporting_evidence']['parser_participle'])
        competing = {c['reading']: c['status'] for c in hypothesis['competing_interpretations']}
        self.assertEqual(competing['instrument_or_means'], 'unassessed')
        self.assertEqual(competing['active_or_elliptical'], 'unresolved')

    def test_rejected_repairs_retain_diagnostic_evidence(self):
        for text, failed in (
            ('Damage caused by noon.', 'one_non_temporal_nominal_source'),
            ('The bat flew by the blind man.', 'licensed_surface_form'),
            ('Rain caused flooding by noon.', 'no_competing_direct_object'),
            ('Damage caused by cutting cables.', 'one_non_temporal_nominal_source'),
        ):
            with self.subTest(text=text):
                record = next(e for e in r.collect_claim_evidence(text)
                              if 'passive_fragment_assessment' in e.get('selected_candidate', {}))
                assessment = record['selected_candidate']['passive_fragment_assessment']
                self.assertEqual(assessment['status'], 'rejected')
                self.assertIn(failed, assessment['rejection_reasons'])
                self.assertIsNone(assessment['proposed_direction'])
                self.assertFalse(assessment['eligible_for_world_state'])

    def test_gating_failure_attribution_for_both_pipelines(self):
        evidence = r.collect_evidence('Rain causes flooding.')
        labels = dict(entity1='Rain', entity2='flooding', correct_action=0)
        original = r._claim_from_evidence
        for expected, injected in ((True, False), (False, True)):
            def faulty_gate(evidence, policy):
                result = original(evidence, policy)
                result['eligible_for_world_state'] = injected
                return result
            with patch.object(r, '_claim_from_evidence', side_effect=faulty_gate):
                row = audit.audit_candidate(evidence, self.policy, dict(labels, expected_eligible=expected))
            for name in ('cem', 'baseline'):
                self.assertTrue(row['gate_outcomes'][name]['gating_failure'])
                kind = 'false_rejection' if expected else 'false_commitment'
                self.assertTrue(row['gate_outcomes'][name][kind])
                self.assertIn('GATING_FAILURE_' + name + '_' + kind, row['diagnostic_flags'])
            summary = audit.summarize([row])
            self.assertEqual(summary['pipelines']['cem']['gating_failure']['rate'], 1)

    def test_audit_does_not_blame_gate_for_extraction_or_policy_error(self):
        evidence = r.collect_evidence('Rain causes flooding.')
        label = dict(entity1='Rain', entity2='flooding', correct_action=0, expected_eligible=False)
        wrong = copy.deepcopy(evidence)
        wrong['entities'][0]['token_index'] = 1
        row = audit.audit_candidate(wrong, self.policy, label)
        self.assertIn('EXTRACTION_FAILURE_wrong_pair', row['diagnostic_flags'])
        self.assertIsNone(row['gate_outcomes']['cem']['gating_failure'])
        self.assertTrue(row['gate_outcomes']['cem']['false_commitment'])
        no_pair = r.collect_evidence('The players went home.')
        row = audit.audit_candidate(no_pair, self.policy,
                                    dict(entity1='players',entity2='home',correct_action=3,expected_eligible=True))
        self.assertIn('EXTRACTION_FAILURE_missing_pair',row['diagnostic_flags'])
        self.assertIsNone(row['gate_outcomes']['cem']['gating_failure'])
        row = audit.audit_candidate(evidence, self.policy, dict(label,correct_action=1))
        self.assertIsNone(row['gate_outcomes']['cem']['gating_failure'])
        self.assertIn('D_policy_error_candidate_requires_feature_review',row['diagnostic_flags'])

    def test_unlabelled_and_unsupported_cases_have_no_invented_accuracy(self):
        evidence = r.collect_evidence('Rain causes flooding.')
        row = audit.audit_candidate(evidence, self.policy)
        for pipeline in audit.summarize([row])['pipelines'].values():
            self.assertIsNone(pipeline['false_commitment']['rate'])
            self.assertIsNone(pipeline['gating_failure']['rate'])
        outside = audit.audit_candidate(evidence, self.policy,
                                       dict(relation_scope='outside_causal_actions',correct_action=0))
        self.assertIn('B_outside_action_space',outside['diagnostic_flags'])
        self.assertIsNone(outside['cem_correct'])
        self.assertIsNone(outside['pair_and_raw_action_correct'])
        summary = audit.summarize([row,outside])
        self.assertEqual(summary['conditional_raw_accuracy']['denominator'],0)
        weights = self.policy.weights * 0
        weights[2, r.FEATURE_NAMES.index('bias')] = 1
        disagreement = audit.audit_candidate(evidence, r.CEMPolicy(weights))
        self.assertEqual(audit.summarize([disagreement])['raw_disagreements']['rate'],1)
        self.assertIsNone(disagreement['cem_correct'])

    def test_explicit_head_annotations_disambiguate_repeated_mentions(self):
        evidence = r.collect_evidence('Rain causes flooding, and rain causes erosion.')
        ambiguous = audit.audit_candidate(evidence, self.policy,
                                          dict(entity1='rain',entity2='flooding',correct_action=0))
        self.assertIsNone(ambiguous['pair_correct'])
        labelled = audit.audit_candidate(evidence, self.policy,
                                         dict(argument_token_indices=[0,2],correct_action=0))
        self.assertTrue(labelled['pair_correct'])


class REmbeddingTests(previous.QEmbeddingTests):
    @classmethod
    def setUpClass(cls):
        replacement = patch.object(previous, 'q', r)
        replacement.start()
        cls.addClassCleanup(replacement.stop)
        super().setUpClass()


class ControllerPreservationTests(unittest.TestCase):
    """Added for the controller patch; execution deferred at the user's request."""
    @classmethod
    def setUpClass(cls):
        cls.policy, _ = r.train_policy()

    def test_controller_survives_without_parent_object(self):
        for text, expected, obj in (
            ('The flood caused the library to close.', 'the library', None),
            ('The manager allowed the workers to leave.', 'the workers', None),
            ('The storm forced the school to close.', 'the school', 'the school'),
        ):
            with self.subTest(text=text):
                result = r.parse_world_state(text, self.policy)
                parent, child = result['propositions']
                self.assertEqual((parent['object'] or {}).get('text'), obj)
                link, = parent['event_links']
                self.assertEqual(link['controller_candidate']['text'], expected)
                self.assertEqual(link['controller_source'], 'child_subject')
                self.assertEqual(link['parent_frame'], parent['frame_id'])
                self.assertEqual(link['child_frame'], child['frame_id'])
                self.assertEqual(link['controller_candidate']['token_index'], child['subject']['token_index'])
                self.assertFalse(link['eligible_for_world_state'])
                self.assertEqual(result['complement_relations'][0]['controller_link'], link)
                self.assertFalse(any(c['eligible_for_world_state'] for c in result['causal_claims']))
                json.dumps(result)

    def test_subject_control_is_local_to_the_immediate_event_link(self):
        result = r.parse_world_state('The worker tried to persuade Alice to leave.', self.policy)
        parent, middle, child = result['propositions']
        outer, = parent['event_links']
        inner, = middle['event_links']
        self.assertEqual(outer['controller_candidate']['head_text'], 'worker')
        self.assertEqual(outer['configuration'], 'subject_control')
        self.assertEqual(outer['controller_support'], 'adapter_subject_control')
        self.assertEqual(inner['child_frame'], child['frame_id'])
        self.assertEqual(inner['controller_candidate']['head_text'], 'Alice')
        self.assertNotEqual(inner['configuration'], 'subject_control')
        self.assertEqual([rel['relation_type'] for rel in result['complement_relations']], ['ATTEMPT','UNRESOLVED'])

    def test_adjuncts_and_passive_agents_do_not_become_controllers(self):
        for text in ('After the game the players went home.',
                     'The bat flew by the blind man.', 'The man was bit by the dog.'):
            with self.subTest(text=text):
                result = r.parse_world_state(text, self.policy)
                for frame in result['propositions']:
                    self.assertIsNone(frame['object'])
                    self.assertEqual(frame['event_links'], [])
                self.assertFalse(any(c['eligible_for_world_state'] for c in result['causal_claims']))

    def test_finite_reporting_subject_is_not_a_controller(self):
        frames = r.extract_proposition_frames('Officials said that the bridge collapsed.')
        link, = frames[0].event_links
        self.assertIsNone(link['controller_candidate'])
        self.assertEqual(link['controller_support'], 'not_a_control_construction')
        self.assertEqual(frames[1].subject['text'], 'the bridge')

    def test_finite_ccomp_with_auxiliary_is_not_mistaken_for_infinitive(self):
        frames = r.extract_proposition_frames('Officials said that the bridge had collapsed.')
        link, = frames[0].event_links
        self.assertEqual(link['dependency_role'], 'ccomp')
        self.assertIsNone(link['controller_candidate'])
        self.assertEqual(link['controller_support'], 'not_a_control_construction')
        self.assertEqual(frames[1].subject['text'], 'the bridge')

    def test_missing_subject_and_multiple_links_do_not_borrow_or_overwrite(self):
        text = 'The flood caused the library to close.'
        parent, child = r.extract_proposition_frames(text)
        missing = replace(child, frame_id='missing', subject=None)
        frames = [parent, child, missing]
        r.preserve_event_controllers(r.get_nlp()(text), frames, [])
        self.assertEqual(len(parent.event_links), 2)
        self.assertEqual(parent.event_links[0]['controller_candidate']['text'], 'the library')
        self.assertIsNone(parent.event_links[1]['controller_candidate'])
        self.assertIsNone(parent.object)

    def test_smoke_and_rolling_displays_separate_object_from_controller(self):
        text = 'The manager allowed the workers to leave.'
        with redirect_stdout(io.StringIO()) as output:
            r.print_proposition_frames(text)
        self.assertIn('syntactic object=?', output.getvalue())
        self.assertIn('controller_candidate=the workers', output.getvalue())
        self.assertNotIn('syntactic object/controller=', output.getvalue())
        entries = r.evaluate_user_suite(r.append_user_probe([], text), self.policy)
        with redirect_stdout(io.StringIO()) as output:
            r.print_user_suite(entries)
        self.assertIn('controller_candidate=the workers', output.getvalue())
        self.assertIn('child_subject', output.getvalue())


if __name__ == '__main__':
    unittest.main()
