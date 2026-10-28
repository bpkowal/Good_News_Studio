from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from behavioral_policy import ContextualPolicy, ExperienceStore, values_with_memory
from contextual_moral_simulation import (
    ACTIONS, FEATURES, FEATURE_SCHEMA, build_cases, expected_scores, feedback,
    fixed_action, validate_fixture,
)
from run_contextual_moral_generalization import run


class ContextualLearningTests(unittest.TestCase):
    def setUp(self):
        self.fixture = build_cases()
        self.case = next(c for c in self.fixture['cases'] if c['split'] == 'train')
        self.policy = ContextualPolicy(ACTIONS, FEATURES)
        self.temp = tempfile.TemporaryDirectory()
        self.store = ExperienceStore(Path(self.temp.name)/'experience.sqlite3')

    def tearDown(self):
        self.store.close(); self.temp.cleanup()

    def record(self, case=None, action='disclose', arm='learned_values_memory', seed=11, step=0, learn=True):
        case = case or self.case
        self.store.record(arm=arm, seed=seed, phase=case['split'], step=step, case=case, action=action,
                          values=self.policy.values(case['features']), probabilities={a: 1/3 for a in ACTIONS},
                          feedback=feedback(case, action, seed, case['split'], step), retrieval={},
                          policy=self.policy, learn=learn)

    def test_frozen_fixture_holds_out_whole_contexts_not_just_wording(self):
        self.assertEqual(validate_fixture(self.fixture), {'train': 32, 'validation': 8, 'test': 8})
        saved = json.loads(Path('fixtures/contextual_moral_generalization_v01.json').read_text())
        self.assertEqual(self.fixture, saved)
        train = {tuple(c['features'].values()) for c in saved['cases'] if c['split'] == 'train'}
        test = {tuple(c['features'].values()) for c in saved['cases'] if c['split'] == 'test'}
        self.assertFalse(train & test)
        self.assertEqual(len(test), 2)

    def test_cross_split_duplicate_context_is_rejected(self):
        changed = deepcopy(self.fixture)
        changed['cases'][0]['split'] = 'test' if changed['cases'][0]['split'] != 'test' else 'train'
        with self.assertRaises(ValueError):
            validate_fixture(changed)

    def test_same_disclosure_has_different_function_and_coercion_is_not_a_sufficient_cue(self):
        tests = [c for c in self.fixture['cases'] if c['split'] == 'test']
        for case in tests:
            scores = expected_scores(case['features'])
            self.assertNotEqual(fixed_action(case['features']), max(scores, key=scores.get))
        helpful, harmful = tests[0], tests[-1]
        self.assertGreater(expected_scores(helpful['features'])['disclose'],
                           expected_scores(harmful['features'])['disclose'])

    def test_update_changes_shared_feature_model_and_round_trips(self):
        before = self.policy.values(self.case['features'])['disclose']
        self.policy.update(self.case['features'], 'disclose', 1)
        self.assertGreater(self.policy.values(self.case['features'])['disclose'], before)
        restored = ContextualPolicy.restore(self.policy.snapshot())
        self.assertEqual(restored.snapshot(), self.policy.snapshot())
        self.assertNotIn(self.case['id'], json.dumps(restored.snapshot()))

    def test_invalid_feedback_features_and_actions_are_rejected_before_update(self):
        before = self.policy.snapshot()
        for value in (float('nan'), float('inf'), -1, 2, True):
            with self.assertRaises(ValueError):
                self.policy.update(self.case['features'], 'disclose', value)
        with self.assertRaises(ValueError):
            self.policy.update(self.case['features'], 'invent_action', 1)
        with self.assertRaises(ValueError):
            self.policy.values(dict(self.case['features'], scenario_id=0))
        self.assertEqual(before, self.policy.snapshot())

    def test_softmax_selection_distribution_is_finite_and_normalized(self):
        values = self.policy.values(self.case['features'])
        action, probabilities = self.policy.choose(values, random.Random(11), epsilon=0.15)
        self.assertIn(action, ACTIONS)
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertTrue(all(math.isfinite(p) and p >= 0 for p in probabilities.values()))

    def test_bad_luck_does_not_relabel_expected_decision_or_process(self):
        case = next(c for c in self.fixture['cases'] if c['features']['protective_purpose'] == 1)
        chosen = 'disclose'
        unlucky = next(feedback(case, chosen, seed, 'train', 0) for seed in range(100)
                       if feedback(case, chosen, seed, 'train', 0)['outcome_score'] == 0)
        self.assertGreater(unlucky['decision_score'], 0.8)
        self.assertEqual(unlucky['process_score'], 1)
        self.assertEqual(unlucky['training_signal'], 'outcome_score_only')

    def test_held_out_observation_is_visible_but_cannot_train_or_enter_memory(self):
        case = next(c for c in self.fixture['cases'] if c['split'] == 'test')
        before = self.policy.snapshot()
        with self.assertRaises(ValueError):
            self.record(case=case, learn=True)
        self.record(case=case, learn=False)
        self.assertEqual(self.policy.snapshot(), before)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM experiences').fetchone()[0], 1)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM memory').fetchone()[0], 0)

    def test_duplicate_receipt_rolls_back_weight_update(self):
        self.record()
        before = self.policy.snapshot()
        with self.assertRaises(sqlite3.IntegrityError):
            self.record()
        self.assertEqual(self.policy.snapshot(), before)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM memory').fetchone()[0], 1)

    def test_retrieval_is_persistent_isolated_and_never_reads_held_out_feedback(self):
        self.record()
        case = next(c for c in self.fixture['cases'] if c['split'] == 'test')
        self.record(case=case, learn=False)
        path = Path(self.temp.name)/'experience.sqlite3'
        other = ExperienceStore(path)
        try:
            found = other.retrieve('learned_values_memory', 11, self.case['features'], 'disclose', FEATURE_SCHEMA)
            self.assertEqual([r['case_id'] for r in found], [self.case['id']])
            self.assertEqual(other.retrieve('learned_values', 11, self.case['features'], 'disclose', FEATURE_SCHEMA), [])
            self.assertEqual(other.retrieve('learned_values_memory', 23, self.case['features'], 'disclose', FEATURE_SCHEMA), [])
        finally:
            other.close()

    def test_memory_with_no_precedent_leaves_values_unchanged(self):
        values, audit = values_with_memory(self.policy, self.case['features'], self.store,
                                          'learned_values_memory', 11, FEATURE_SCHEMA)
        self.assertEqual(values, self.policy.values(self.case['features']))
        self.assertEqual(audit, {})

    def test_only_observed_outcome_trains_not_other_feedback_scores(self):
        observed = feedback(self.case, 'disclose', 11, 'train', 0)
        observed.update(decision_score=-999, process_score=999, evaluator_confidence=0.01)
        expected = ContextualPolicy(ACTIONS, FEATURES)
        expected.update(self.case['features'], 'disclose', observed['outcome_score'])
        self.store.record(arm='learned_values', seed=11, phase='train', step=0, case=self.case,
                          action='disclose', values={}, probabilities={}, feedback=observed,
                          retrieval={}, policy=self.policy, learn=True)
        self.assertEqual(self.policy.snapshot(), expected.snapshot())

    def test_end_to_end_is_reproducible_and_preserves_checkpoints_without_network(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('No API calls allowed')):
            first = run(self.fixture, Path(self.temp.name)/'first', seeds=(11, 23), epochs=2)
            second = run(self.fixture, Path(self.temp.name)/'second', seeds=(11, 23), epochs=2)
        self.assertEqual(first, second)
        folder = Path(self.temp.name)/'first'
        controls = json.loads((folder/'controls.json').read_text())
        self.assertEqual(controls['held_out_memory_records'], 0)
        self.assertEqual(controls['held_out_updates'], 0)
        for seed in (11, 23):
            self.assertEqual(len({r['schedule_sha256'] for r in controls['runs'] if r['seed'] == seed}), 1)
        db = sqlite3.connect(str(folder/'experience.sqlite3'))
        try:
            for arm, seed, saved in db.execute("SELECT arm,seed,policy FROM checkpoints WHERE phase='trained'"):
                for (later,) in db.execute("SELECT policy FROM checkpoints WHERE arm=? AND seed=? AND phase IN ('validation','test')", (arm, seed)):
                    self.assertEqual(saved, later)
        finally:
            db.close()
        with self.assertRaises(ValueError):
            run(self.fixture, folder, seeds=(11,), epochs=1)


if __name__ == '__main__':
    unittest.main()
