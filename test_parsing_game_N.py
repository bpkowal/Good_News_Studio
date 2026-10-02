import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import patch

import parsing_game_N as n
import test_parsing_game_M as legacy


class NAdversarialTests(legacy.AdversarialEmbeddingTests):
    """Run M's unchanged structural expectations against N, including its 3 gaps."""
    @classmethod
    def setUpClass(cls):
        replacement = patch.object(legacy, 'm', n)
        replacement.start()
        cls.addClassCleanup(replacement.stop)
        super().setUpClass()


class NTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy, _ = n.train_policy()

    def parse(self, text):
        return n.parse_world_state(text, self.policy)

    def test_attempt_resource_and_subject_control(self):
        for text in ('The worker tried to leave.', 'The engineer attempted to open the hatch.'):
            with self.subTest(text=text):
                result = self.parse(text)
                relation, = result['complement_relations']
                by_id = {f['frame_id']: f for f in result['propositions']}
                parent, child = by_id[relation['parent_frame']], by_id[relation['child_frame']]
                self.assertEqual(relation['relation_type'], 'ATTEMPT')
                self.assertEqual(relation['namespace'], 'parliament:proposition')
                self.assertEqual(relation['child_entailment'], 'not_entailed')
                self.assertEqual(relation['provenance']['class_id'], 'try-61.1')
                self.assertEqual(parent['subject']['token_index'], child['subject']['token_index'])
                self.assertEqual(parent['occurrence_status'], 'asserted_in_text')
                self.assertEqual(child['occurrence_status'], 'unknown')
                self.assertTrue(all(not f['eligible_for_world_state'] for f in result['propositions']))

    def test_negation_modality_condition_and_attribution(self):
        cases = [
            ('The worker did not try to leave.', 'denied', 'denied_in_text'),
            ('The worker might try to leave.', 'possible', 'unknown'),
            ('If the worker tried to leave, the alarm sounded.', 'conditional', 'unknown'),
            ('Officials said the worker tried to leave.', 'attributed', 'unknown'),
        ]
        for text, status, occurrence in cases:
            with self.subTest(text=text):
                result = self.parse(text)
                relation = next(r for r in result['complement_relations'] if r['relation_type'] == 'ATTEMPT')
                by_id = {p['frame_id']: p for p in result['propositions']}
                self.assertEqual(relation['parent_assertion_status'], status)
                self.assertEqual(by_id[relation['parent_frame']]['occurrence_status'], occurrence)
                self.assertEqual(by_id[relation['child_frame']]['occurrence_status'], 'unknown')
                self.assertEqual(relation['child_entailment'], 'not_entailed')

    def test_independent_assertion_does_not_get_erased_by_attempt(self):
        result = self.parse('The worker tried to leave. The worker left.')
        leaving = [p for p in result['propositions'] if p['predicate_lemma'] == 'leave']
        self.assertEqual([p['occurrence_status'] for p in leaving], ['unknown', 'asserted_in_text'])

    def test_contrastive_attempt_micro_suite(self):
        cases = [
            ('The worker tried to leave.', 'asserted', 'asserted_in_text'),
            ('The worker attempted to leave.', 'asserted', 'asserted_in_text'),
            ('The worker did not try to leave.', 'denied', 'denied_in_text'),
            ('The worker may try to leave.', 'possible', 'unknown'),
        ]
        for text, assertion, occurrence in cases:
            with self.subTest(text=text):
                result = self.parse(text)
                relation, = result['complement_relations']
                parent, child = result['propositions']
                self.assertEqual(relation['relation_type'], 'ATTEMPT')
                self.assertEqual(relation['parent_assertion_status'], assertion)
                self.assertEqual(relation['child_entailment'], 'not_entailed')
                self.assertEqual(parent['occurrence_status'], occurrence)
                self.assertEqual(child['occurrence_status'], 'unknown')
                self.assertEqual(child['subject']['head_text'], 'worker')

    def test_coordinated_independent_leaving_is_asserted(self):
        result = self.parse('The worker tried and left.')
        self.assertEqual(result['complement_relations'], [])
        self.assertEqual([p['predicate_lemma'] for p in result['propositions']], ['try', 'leave'])
        for frame in result['propositions']:
            self.assertEqual(frame['subject']['head_text'], 'worker')
            self.assertEqual(frame['occurrence_status'], 'asserted_in_text')
            self.assertIsNone(frame['parent_frame_id'])
            self.assertFalse(frame['eligible_for_world_state'])

    def test_coordination_does_not_promote_scoped_leaving(self):
        for text in ('The worker may try and leave.',
                     'The worker did not try and leave.',
                     'If the worker tried and left, the alarm sounded.',
                     'The worker tried or left.'):
            with self.subTest(text=text):
                child = next(p for p in self.parse(text)['propositions']
                             if p['predicate_lemma'] == 'leave')
                self.assertEqual(child['occurrence_status'], 'unknown')

    def test_nested_control_is_local_and_lower_relation_abstains(self):
        for text, actor, recipient in (
            ('The worker tried to persuade Alice to leave.', 'worker', 'Alice'),
            ('The sailor attempted to persuade Nora to leave.', 'sailor', 'Nora'),
        ):
            with self.subTest(text=text):
                result = self.parse(text)
                parent, middle, child = result['propositions']
                self.assertEqual(middle['parent_frame_id'], parent['frame_id'])
                self.assertEqual(child['parent_frame_id'], middle['frame_id'])
                self.assertEqual(parent['complement_frame_ids'], [middle['frame_id']])
                self.assertEqual(middle['complement_frame_ids'], [child['frame_id']])
                self.assertEqual(middle['subject']['head_text'], actor)
                # This is a structural candidate, not a new persuade control policy.
                if child['subject'] is not None:
                    self.assertEqual(child['subject']['head_text'], recipient)
                self.assertEqual([r['relation_type'] for r in result['complement_relations']],
                                 ['ATTEMPT', 'UNRESOLVED'])
                self.assertEqual([r['child_entailment'] for r in result['complement_relations']],
                                 ['not_entailed', 'unknown'])
                self.assertEqual([p['occurrence_status'] for p in (middle, child)],
                                 ['unknown', 'unknown'])
                self.assertTrue(all(not c['eligible_for_world_state'] for c in result['causal_claims']))

    def test_registry_dispatches_adapter_policy_without_attempt_branch(self):
        adapter, = n.get_complement_adapters()
        # A test-only replacement proves dispatch uses adapter fields/callbacks.
        replacement = replace(adapter, relation_type='TEST_ONLY', child_entailment='unknown')
        with patch.object(n, 'get_complement_adapters', return_value=(replacement,)):
            relation, = self.parse('The worker tried to leave.')['complement_relations']
        self.assertEqual(relation['relation_type'], 'TEST_ONLY')
        self.assertEqual(relation['child_entailment'], 'unknown')
        rejecting = replace(adapter, recovered_frame_matcher=lambda parent, child: False)
        with patch.object(n, 'get_complement_adapters', return_value=(rejecting,)):
            relation, = self.parse('The worker tried to leave.')['complement_relations']
        self.assertEqual(relation['relation_type'], 'UNRESOLVED')

    def test_registry_empty_or_conflicting_matches_abstain(self):
        adapter, = n.get_complement_adapters()
        for registry in ((), (adapter, replace(adapter, relation_type='TEST_CONFLICT'))):
            with self.subTest(size=len(registry)):
                with patch.object(n, 'get_complement_adapters', return_value=registry):
                    relation, = self.parse('The worker tried to leave.')['complement_relations']
                self.assertEqual(relation['relation_type'], 'UNRESOLVED')
                self.assertEqual(relation['child_entailment'], 'unknown')
                self.assertFalse(relation['provenance']['resource_match'])

    def test_intend_and_unmatched_senses_abstain(self):
        result = self.parse('The worker intended to leave.')
        relation, = result['complement_relations']
        self.assertEqual(relation['relation_type'], 'UNRESOLVED')
        self.assertEqual(relation['child_entailment'], 'unknown')
        self.assertTrue(relation['provenance']['lexical_member'])
        for text in ('The chef tried the soup.', 'The worker tried leaving.',
                     'The worker tried the key to open the door.',
                     'The manager allowed the worker to leave.'):
            with self.subTest(text=text):
                result = self.parse(text)
                self.assertFalse(any(r['relation_type'] == 'ATTEMPT' for r in result['complement_relations']))

    def test_attempted_causal_action_never_becomes_actual_causal_edge(self):
        text = 'The worker tried to cause damage.'
        for claims in (self.parse(text)['causal_claims'], n.parse_claims(text, self.policy)):
            self.assertTrue(all(not c['eligible_for_world_state'] for c in claims))
            cause = next(c for c in claims if c['relation_type'] == 'causal')
            self.assertIn('embedded_proposition_requires_scope_validation', cause['validation_reasons'])

    def test_quantity_and_legacy_causal_suites(self):
        for suite in (n.TRAIN_EXAMPLES, n.TEST_EXAMPLES, n.NEGATION_GENERALIZATION_EXAMPLES,
                      n.ROBUSTNESS_EXAMPLES, n.LEXICAL_HOLDOUT, n.UNKNOWN_HOLDOUT, n.EPISTEMIC_HOLDOUT):
            self.assertEqual(n.evaluate_suite(suite, self.policy)[0]['end_to_end_correct'], 1)
        self.assertEqual(n.evaluate_multi_suite(n.MULTI_RELATION_HOLDOUT, self.policy)[0]
                         ['exact_claim_list_accuracy'], 1)
        self.assertEqual(self.parse('Insulation reduces noise.')['events'][0]['change']['direction'], 'decrease')

    def test_rolling_results_serialize_entailment_and_display_it(self):
        self.assertEqual(n.USER_SUITE_PATH.name, 'parsing_game_N_user_probes.json')
        entries = n.evaluate_user_suite(n.append_user_probe([], 'The worker tried to leave.'), self.policy)
        self.assertEqual(entries[0]['last_result']['complement_relations'][0]['child_entailment'], 'not_entailed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probes.json'
            n.save_user_suite(path, entries)
            self.assertEqual(n.load_user_suite(path), entries)
        with redirect_stdout(io.StringIO()) as output:
            n.print_user_suite(entries)
        self.assertIn('ATTEMPT', output.getvalue())
        self.assertIn('not_entailed', output.getvalue())

    def test_resource_namespaces_are_separate(self):
        root = Path(n.__file__).parent / 'resources'
        sample = json.loads((root / 'freebase_schema_sample.json').read_text())
        self.assertFalse(sample['parser_semantic_authority'])
        self.assertNotEqual(sample['namespace'], n.load_attempt_adapter()['namespace'])

    @unittest.expectedFailure
    def test_path_adjunct_known_gap_not_a_patient(self):
        frame, = n.extract_proposition_frames('The bat flew by the blind man.')
        self.assertIsNone(frame.object)

    def test_path_adjunct_never_creates_a_causal_commitment(self):
        self.assertTrue(all(not c['eligible_for_world_state'] for c in
                            n.parse_claims('The bat flew by the blind man.', self.policy)))


if __name__ == '__main__':
    unittest.main()
