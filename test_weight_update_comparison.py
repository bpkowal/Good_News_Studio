import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from behavioral_policy import ContextualPolicy, ExperienceStore
from contextual_moral_simulation import ACTIONS, FEATURES, VERSION as SIM_VERSION, build_cases
from run_contextual_moral_generalization import dump, fingerprint
from run_weight_update_comparison import compare, load_parent, VERSION


class WeightUpdateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = build_cases()
        self.policy = ContextualPolicy(ACTIONS, FEATURES)
        # A noninitial model proves that continuation does not silently reset.
        self.policy.update(self.fixture['cases'][0]['features'], 'disclose', 1)

    def test_explanations_reconstruct_raw_values_and_clipped_predictions(self):
        context = self.fixture['cases'][0]['features']
        explanation = self.policy.explain(context)
        for action in ACTIONS:
            row = explanation[action]
            self.assertAlmostEqual(sum(t['contribution'] for t in row['terms']), row['raw_value'])
            self.assertEqual(row['value'], self.policy.values(context)[action])

    def test_invalid_revisions_cannot_resume(self):
        for revision in (-1, 1.5, True, None):
            saved = self.policy.snapshot(); saved['revision'] = revision
            with self.assertRaises(ValueError):
                ContextualPolicy.restore(saved)

    def test_memory_can_be_disabled_without_disabling_observation_journal(self):
        store = ExperienceStore(self.root/'store.sqlite3', retain_memory=False)
        try:
            case = next(c for c in self.fixture['cases'] if c['split'] == 'train')
            store.record(arm='updated_weights', seed=11, phase='train', step=0,
                         case=case, action='disclose', values={}, probabilities={},
                         feedback={'outcome_score': 1, 'evidence': {'source': 'test'}},
                         retrieval={}, policy=self.policy, learn=True)
            self.assertEqual(store.db.execute('SELECT count(*) FROM memory').fetchone()[0], 0)
            self.assertEqual(store.db.execute('SELECT count(*) FROM experiences').fetchone()[0], 1)
        finally:
            store.close()

    def test_pair_starts_identically_but_only_updated_arm_changes(self):
        output = self.root/'compare'
        start = self.policy.snapshot()
        with patch('behavioral_policy.values_with_memory', side_effect=AssertionError('Retrieval must remain off')):
            compare(self.fixture, {11: start}, {'parent_run_id': 'test'}, output, epochs=2)
        saved = json.loads((output/'policy_checkpoints.json').read_text())
        self.assertEqual(saved['frozen_weights:11'], start)
        self.assertEqual(saved['updated_weights:11']['revision'], start['revision'] + 64)
        self.assertNotEqual(saved['updated_weights:11']['weights'], start['weights'])
        controls = json.loads((output/'controls.json').read_text())
        self.assertEqual(len({r['initial_sha256'] for r in controls['pairs']}), 1)
        self.assertEqual(len({r['schedule_sha256'] for r in controls['pairs']}), 1)
        self.assertEqual(controls['memory_rows'], 0)
        db = sqlite3.connect(output/'experience.sqlite3')
        try:
            self.assertEqual(db.execute("SELECT count(*) FROM experiences WHERE learned=1 AND phase!='train'").fetchone()[0], 0)
            for arm, policy in db.execute("SELECT arm,policy FROM checkpoints WHERE phase='trained'"):
                for (later,) in db.execute("SELECT policy FROM checkpoints WHERE arm=? AND phase IN ('validation','test')", (arm,)):
                    self.assertEqual(later, policy)
        finally:
            db.close()
        with (output/'weight_changes.csv').open() as stream:
            changes = list(csv.DictReader(stream))
        self.assertTrue(all(float(r['delta']) == 0 for r in changes if r['arm'] == 'frozen_weights'))
        self.assertTrue(any(float(r['delta']) != 0 for r in changes if r['arm'] == 'updated_weights'))
        self.assertEqual(json.loads((output/'design.json').read_text())['evaluation_status'],
                         'REUSED_MONITORING_CASES_NOT_FRESH_HOLDOUT')

    def test_parent_integrity_and_automatic_weight_only_lineage(self):
        from parliament_run_provenance import record_execution, record_start
        parent = self.root/'parent'
        snapshot = self.policy.snapshot()
        def create(argv):
            record_start(parent)
            dump(parent/'design.json', {'version': SIM_VERSION, 'seeds': [11], 'fixture_sha256': fingerprint(self.fixture)})
            dump(parent/'cases.freeze.json', self.fixture)
            dump(parent/'policy_checkpoints.json', {'learned_values:11': snapshot})
            return 0
        record_execution(create, ['--output', str(parent)], entrypoint='run_contextual_moral_generalization.py', output_flag='--output')
        fixture, policies, origin = load_parent(parent)
        self.assertEqual(policies[11], snapshot)
        self.assertEqual(origin['source_arm'], 'learned_values')
        continued = self.root/'continued'
        def continuation(argv):
            compare(fixture, policies, origin, continued, epochs=1)
            return 0
        record_execution(continuation, ['--output', str(continued)], entrypoint='run_weight_update_comparison.py', output_flag='--output')
        _, second, second_origin = load_parent(continued)
        self.assertEqual(second[11]['revision'], snapshot['revision'] + 32)
        self.assertEqual(second_origin['source_arm'], 'updated_weights')
        manifest = json.loads((continued/'run_manifest.json').read_text())
        self.assertEqual(manifest['mode'], 'SIMULATION_NO_API')
        self.assertEqual(manifest['execution_path'], 'behavioral_simulation')
        (parent/'policy_checkpoints.json').write_text('{}')
        with self.assertRaises(ValueError):
            load_parent(parent)


if __name__ == '__main__':
    unittest.main()
