import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from behavioral_policy import ContextualPolicy
from contextual_moral_simulation import ACTIONS, FEATURES, build_cases
from run_weight_update_comparison import compare
from weight_failure_review import classify, export_failure_review


class WeightFailureReviewTests(unittest.TestCase):
    def test_unlucky_optimal_choice_is_not_called_suboptimal(self):
        record = {'feedback': {'outcome_score': 0, 'process_score': 1}}
        self.assertEqual(classify(record, {'regret': 0, 'expected_score': 0.95}),
                         ['observed_outcome_failure'])

    def test_lucky_suboptimal_choice_is_still_flagged(self):
        record = {'feedback': {'outcome_score': 1, 'process_score': 1}}
        self.assertEqual(classify(record, {'regret': 0.5, 'expected_score': 0.4}),
                         ['monitoring_suboptimal_choice'])

    def test_regression_and_process_measure_remain_separate(self):
        record = {'feedback': {'outcome_score': 1, 'process_score': 0}}
        self.assertEqual(classify(record, {'regret': 0.1, 'expected_score': 0.8}, {'expected_score': 0.9}),
                         ['monitoring_suboptimal_choice','monitoring_regression','process_proxy_zero'])
        self.assertEqual(classify(record), ['process_proxy_zero'])

    def test_no_counterfactual_grade_is_invented_for_training_failure(self):
        self.assertEqual(classify({'feedback': {'outcome_score': 0}}), ['observed_outcome_failure'])

    def test_auto_archive_and_readonly_replay_preserve_weights_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = ContextualPolicy(ACTIONS, FEATURES)
            compare(build_cases(), {11:policy.snapshot()}, {}, root/'source', epochs=1)
            source = root/'source'
            auto = json.loads((source/'failure_review/summary.json').read_text())
            self.assertGreater(auto['flagged_experiences'], 0)
            self.assertEqual(auto['policy_updates'], 0)
            before = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                      for name in ('experience.sqlite3','policy_checkpoints.json','evaluations.json')}
            replay = export_failure_review(source, root/'review', source_run_id='test_source')
            self.assertEqual(replay['counts_by_kind'], auto['counts_by_kind'])
            self.assertEqual(replay['flagged_experiences'], auto['flagged_experiences'])
            self.assertTrue(all(hashlib.sha256((source/name).read_bytes()).hexdigest() == value
                                for name,value in before.items()))
            entries = [json.loads(line) for line in (root/'review/failures.jsonl').read_text().splitlines()]
            self.assertEqual(len({e['experience_id'] for e in entries}), len(entries))
            self.assertEqual(sum(row['experiences'] for row in replay['grouped']), replay['total_experiences'])
            self.assertTrue(any(entry['phase'] == 'train' for entry in entries))
            self.assertTrue(any(entry['phase'] == 'validation' for entry in entries))
            for entry in entries:
                self.assertEqual(entry['source_run_id'], 'test_source')
                self.assertTrue(entry['record']['policy_before']['weights'])
                self.assertTrue(entry['record']['policy_after']['weights'])
                self.assertTrue(entry['chosen_value_contributions']['terms'])
                self.assertTrue(entry['record']['feedback']['evidence'])
                if entry['phase'] == 'train':
                    self.assertIsNone(entry['monitoring'])
                    self.assertNotIn('monitoring_suboptimal_choice', entry['kinds'])
            page = (root/'review/report.html').read_text()
            self.assertIn('data-kinds=', page)
            self.assertIn('id="search"', page)
            self.assertIn('not moral verdicts', page)
            with self.assertRaises(ValueError):
                export_failure_review(source, root/'review')

    def test_mismatched_saved_monitoring_measurements_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            compare(build_cases(), {11:ContextualPolicy(ACTIONS, FEATURES).snapshot()}, {}, root/'source', epochs=1)
            path = root/'source/evaluations.json'
            rows = json.loads(path.read_text())
            rows[0]['selected_action'] = 'fabricated_action'
            path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                export_failure_review(root/'source', root/'invalid')


if __name__ == '__main__':
    unittest.main()
