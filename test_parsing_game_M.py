import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

import parsing_game_M as m


class MTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy, _ = m.train_policy()

    def test_event_target_exposed_and_noun_edge_blocked(self):
        text = 'The flood caused the library to close.'
        result = m.parse_world_state(text, self.policy)
        frames = {f['predicate_lemma']: f for f in result['propositions']}
        parent, child = frames['cause'], frames['close']
        self.assertEqual(parent['complement_frame_ids'], [child['frame_id']])
        self.assertEqual(child['parent_frame_id'], parent['frame_id'])
        self.assertEqual(child['subject']['text'], 'the library')
        self.assertIsNone(child['object'])
        for claim in (result['causal_claims'][0], m.parse_claims(text, self.policy)[0],
                      m.parse_sentence(text, self.policy)):
            self.assertFalse(claim['eligible_for_world_state'])
            self.assertIsNone(claim['target'])
            self.assertIsNone(claim['target_entity'])
            self.assertIsNone(claim['direction'])
            self.assertEqual(claim['target_proposition_ids'], [child['frame_id']])
        json.dumps(result)

    def test_missing_child_still_blocks_commitment(self):
        with patch.object(m, 'extract_proposition_frames', return_value=[]):
            claim = m.parse_claims('The flood caused the library to close.', self.policy)[0]
        self.assertFalse(claim['eligible_for_world_state'])
        self.assertIsNone(claim['target'])
        self.assertEqual(claim['target_proposition_ids'], [])

    def test_permission_and_force_are_structural_not_automatic_causation(self):
        for text, verb, child_verb in [
            ('The manager allowed the workers to leave.', 'allow', 'leave'),
            ('The storm forced the school to close.', 'force', 'close'),
        ]:
            result = m.parse_world_state(text, self.policy)
            frames = {f['predicate_lemma']: f for f in result['propositions']}
            self.assertEqual(frames[child_verb]['parent_frame_id'], frames[verb]['frame_id'])
            self.assertEqual(frames[verb]['event_object_frame_id'], frames[child_verb]['frame_id'])
            self.assertTrue(all(not c['eligible_for_world_state'] for c in result['causal_claims']))

    def test_parent_negation_and_modality_do_not_commit_child(self):
        for text, status in [
            ('The flood did not cause the library to close.', 'denied'),
            ('The flood may cause the library to close.', 'possible'),
        ]:
            result = m.parse_world_state(text, self.policy)
            parent = next(f for f in result['propositions'] if f['predicate_lemma'] == 'cause')
            self.assertEqual(parent['assertion']['status'], status)
            self.assertTrue(all(not c['eligible_for_world_state'] for c in result['causal_claims']))

    def test_multiple_complements_not_silently_overwritten(self):
        parent, child = m.extract_proposition_frames('The flood caused the library to close.')
        parent = replace(parent, complement_frame_ids=[], event_object_frame_id=None)
        other = replace(child, frame_id='other', predicate_index=100, complement_frame_ids=[])
        m.link_embedded_event_frames([parent, child, other])
        self.assertEqual(parent.complement_frame_ids, [child.frame_id, 'other'])
        self.assertIsNone(parent.event_object_frame_id)

    def test_original_causal_suites_and_quantity_change(self):
        for suite in (m.TRAIN_EXAMPLES, m.TEST_EXAMPLES, m.NEGATION_GENERALIZATION_EXAMPLES,
                      m.ROBUSTNESS_EXAMPLES, m.LEXICAL_HOLDOUT, m.UNKNOWN_HOLDOUT, m.EPISTEMIC_HOLDOUT):
            self.assertEqual(m.evaluate_suite(suite, self.policy)[0]['end_to_end_correct'], 1)
        self.assertEqual(m.evaluate_multi_suite(m.MULTI_RELATION_HOLDOUT, self.policy)[0]
                         ['exact_claim_list_accuracy'], 1)
        event = m.extract_event_state_structure('Insulation reduces noise.')['events'][0]
        self.assertEqual(event['change']['direction'], 'decrease')

    def test_versioned_probe_storage_contains_propositions(self):
        self.assertEqual(m.USER_SUITE_PATH.name, 'parsing_game_M_user_probes.json')
        entries = m.append_user_probe([], 'The storm forced the school to close.')
        entries = m.evaluate_user_suite(entries, self.policy)
        self.assertEqual(len(entries[0]['last_result']['propositions']), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probes.json'
            m.save_user_suite(path, entries)
            self.assertEqual(m.load_user_suite(path), entries)


class AdversarialEmbeddingTests(unittest.TestCase):
    """Sentence-level structural probes, excluded from CEM fitting/selection.

    Expected failures record desired behavior for documented unsupported cases;
    they must be removed when those cases are fixed (unexpected success fails CI).
    """

    @classmethod
    def setUpClass(cls):
        cls.policy, _ = m.train_policy()

    def parse(self, text):
        result = m.parse_world_state(text, self.policy)
        self.assertEqual(len({f['frame_id'] for f in result['propositions']}),
                         len(result['propositions']))
        return result

    def frame(self, result, lemma):
        matches = [f for f in result['propositions'] if f['predicate_lemma'] == lemma]
        self.assertEqual(len(matches), 1, f'Expected exactly one {lemma} frame')
        return matches[0]

    def linked(self, result, parent_lemma, child_lemma, child_subject):
        parent = self.frame(result, parent_lemma)
        child = self.frame(result, child_lemma)
        self.assertEqual(child['parent_frame_id'], parent['frame_id'])
        self.assertIn(child['frame_id'], parent['complement_frame_ids'])
        self.assertIsNotNone(child['subject'])
        self.assertEqual(child['subject']['text'], child_subject)
        self.assertIsNone(child['object'])
        return parent, child

    def no_commitments(self, result):
        self.assertTrue(result['causal_claims'])
        self.assertTrue(all(not c['eligible_for_world_state'] for c in result['causal_claims']))

    def causal_target(self, result, parent, child):
        claim = next(c for c in result['causal_claims']
                     if c['proposition_frame_id'] == parent['frame_id'])
        self.assertEqual(claim['target_kind'], 'proposition')
        self.assertEqual(claim['target_proposition_ids'], [child['frame_id']])
        self.assertIsNone(claim['target'])
        self.assertIsNone(claim['target_entity'])
        self.assertIn('event_complement_requires_proposition_reasoning', claim['validation_reasons'])

    def test_01_lexical_substitution_preserves_embedding(self):
        result = self.parse('The outage caused the museum to close.')
        parent, child = self.linked(result, 'cause', 'close', 'the museum')
        self.assertEqual(parent['subject']['text'], 'The outage')
        self.causal_target(result, parent, child)
        self.no_commitments(result)

    def test_02_negated_parent_does_not_assert_embedded_event(self):
        result = self.parse('The outage did not cause the museum to close.')
        parent, child = self.linked(result, 'cause', 'close', 'the museum')
        self.assertEqual(parent['assertion']['status'], 'denied')
        self.causal_target(result, parent, child)
        self.no_commitments(result)

    def test_03_modal_parent_preserves_proposed_event_target(self):
        result = self.parse('The outage might cause the museum to close.')
        parent, child = self.linked(result, 'cause', 'close', 'the museum')
        self.assertEqual(parent['assertion']['status'], 'possible')
        self.causal_target(result, parent, child)
        self.no_commitments(result)

    def test_04_permission_is_not_causation(self):
        result = self.parse('The supervisor allowed the engineers to leave.')
        self.linked(result, 'allow', 'leave', 'the engineers')
        self.assertFalse(any(c['relation_type'] == 'causal' for c in result['causal_claims']))
        self.no_commitments(result)

    def test_05_denied_permission_keeps_the_parent(self):
        result = self.parse('The supervisor did not allow the engineers to leave.')
        parent, _ = self.linked(result, 'allow', 'leave', 'the engineers')
        self.assertEqual(parent['assertion']['status'], 'denied')
        self.no_commitments(result)

    def test_06_reporting_complement_is_structural(self):
        result = self.parse('Officials said that the bridge collapsed.')
        self.linked(result, 'say', 'collapse', 'the bridge')
        self.assertFalse(any(c['relation_type'] == 'causal' for c in result['causal_claims']))
        self.no_commitments(result)

    @unittest.expectedFailure
    def test_07_passive_controller_known_gap(self):
        # Desired: museum controls close; outage must not become its subject.
        result = self.parse('The museum was forced to close by the outage.')
        self.linked(result, 'force', 'close', 'The museum')

    @unittest.expectedFailure
    def test_08_coordinated_complement_known_gap(self):
        # Desired: both events attach to cause; neither overwrites the other.
        result = self.parse('The outage caused the museum to close and the workers to leave.')
        parent, closing = self.linked(result, 'cause', 'close', 'the museum')
        _, leaving = self.linked(result, 'cause', 'leave', 'the workers')
        self.assertCountEqual(parent['complement_frame_ids'], [closing['frame_id'], leaving['frame_id']])
        self.assertIsNone(parent['event_object_frame_id'])

    def test_09_independent_clause_is_not_an_event_object(self):
        result = self.parse('The outage caused damage, but the museum stayed open.')
        parent = self.frame(result, 'cause')
        other = self.frame(result, 'stay')
        self.assertEqual(parent['complement_frame_ids'], [])
        self.assertIsNone(other['parent_frame_id'])
        claim = next(c for c in result['causal_claims'] if c['proposition_frame_id'] == parent['frame_id'])
        self.assertEqual(claim['target'], 'damage')
        self.assertEqual(claim['target_kind'], 'entity')
        self.assertTrue(claim['eligible_for_world_state'])

    def test_10_nearby_noun_is_not_an_event(self):
        result = self.parse('The outage caused damage near the museum.')
        self.assertEqual(len(result['propositions']), 1)
        parent = self.frame(result, 'cause')
        self.assertEqual(parent['complement_frame_ids'], [])
        claim = result['causal_claims'][0]
        self.assertEqual(claim['target'], 'damage')
        self.assertEqual(claim['target_proposition_ids'], [])
        self.assertTrue(claim['eligible_for_world_state'])

    @unittest.expectedFailure
    def test_11_conditional_boundary_known_gap(self):
        # Desired: refunds belongs to follow, not to the intransitive close.
        result = self.parse('If the outage caused the museum to close, refunds would follow.')
        parent, child = self.linked(result, 'cause', 'close', 'the museum')
        self.assertEqual(parent['assertion']['status'], 'conditional')
        self.causal_target(result, parent, child)
        self.no_commitments(result)

    def test_known_structural_gaps_still_block_commitment(self):
        # These must pass independently: expectedFailure above must never hide
        # an unsafe positive edge if argument recovery regresses further.
        for text in (
            'The museum was forced to close by the outage.',
            'The outage caused the museum to close and the workers to leave.',
            'If the outage caused the museum to close, refunds would follow.',
        ):
            with self.subTest(text=text):
                self.no_commitments(self.parse(text))

    def test_12_nested_reporting_preserves_both_links(self):
        result = self.parse('Officials said that the outage caused the museum to close.')
        reporting = self.frame(result, 'say')
        parent, child = self.linked(result, 'cause', 'close', 'the museum')
        self.assertEqual(parent['parent_frame_id'], reporting['frame_id'])
        self.assertEqual(reporting['complement_frame_ids'], [parent['frame_id']])
        self.assertEqual(parent['assertion']['status'], 'attributed')
        self.causal_target(result, parent, child)
        self.no_commitments(result)


if __name__ == '__main__':
    unittest.main()
