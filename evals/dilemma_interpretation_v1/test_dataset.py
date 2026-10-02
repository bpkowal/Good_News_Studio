import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('dilemma_dataset', Path(__file__).with_name('dataset.py'))
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


class DatasetTests(unittest.TestCase):
    def test_integrity_and_baseline_frozen(self):
        result = dataset.validate()
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['split_counts'], {'dev': 16, 'heldout': 16})
        for counts in result['construction_counts'].values():
            self.assertEqual(set(counts), dataset.CONSTRUCTIONS)

    def test_evidence_corruption_and_dangling_references_rejected(self):
        cases = copy.deepcopy(dataset.load_cases())
        cases[0]['evidence'][1]['start'] += 1
        cases[1]['gold']['readings'][0]['evidence_ids'] = ['absent']
        cases[2]['gold']['downstream_probes'][0]['required_reading_ids'] = ['absent']
        errors = dataset.validate_cases(cases)
        self.assertTrue(any('evidence text mismatch' in e for e in errors))
        self.assertTrue(any('dangling evidence' in e for e in errors))
        self.assertTrue(any('dangling probe' in e for e in errors))

    def test_leaking_families_and_duplicate_ids_rejected(self):
        cases = dataset.load_cases() + dataset.load_cases('heldout')
        cases[16]['template_family'] = cases[0]['template_family']
        cases[16]['scenario_family'] = cases[0]['scenario_family']
        cases[17]['id'] = cases[0]['id']
        errors = dataset.validate_cases(cases)
        self.assertTrue(any('cross-split template_family' in e for e in errors))
        self.assertTrue(any('cross-split scenario_family' in e for e in errors))
        self.assertTrue(any('duplicate case ID' in e for e in errors))

    def test_gold_free_export_defaults_to_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'inputs.jsonl'
            subprocess.run([sys.executable, str(dataset.ROOT / 'dataset.py'), '--export-inputs', str(output)],
                           check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(set(r) == {'id', 'text'} and r['id'].startswith('dev_') for r in rows))
        with self.assertRaises(ValueError):
            dataset.load_cases('all')

    def test_baseline_drift_detected_without_mutating_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dataset.validate(repo=Path(tmp))
        self.assertFalse(result['valid'])
        self.assertTrue(any('Z10 baseline changed: parsing_game_Z10.py' in e for e in result['errors']))

    def test_truncations_are_specific_abstentions_and_controls_have_readings(self):
        for case in dataset.load_cases() + dataset.load_cases('heldout'):
            if case['construction'] == 'truncation':
                self.assertEqual(case['gold']['readings'], [])
                self.assertTrue(case['gold']['open_questions'])
            if case['construction'] == 'nonellipsis_control':
                self.assertTrue(case['gold']['readings'])
                self.assertTrue(case['gold']['forbidden_inferences'])


if __name__ == '__main__':
    unittest.main()
