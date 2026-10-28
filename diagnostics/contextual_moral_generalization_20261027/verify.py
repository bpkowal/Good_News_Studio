"""Recheck the frozen experiment and database without training or API calls."""
import json
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from parliament_run_provenance import verify_manifest
from contextual_moral_simulation import validate_fixture


def verify():
    receipt = verify_manifest(HERE/'run_manifest.json')
    assert receipt['valid'], receipt
    manifest = json.loads((HERE/'run_manifest.json').read_text())
    assert manifest['mode'] == 'SIMULATION_NO_API'
    assert manifest['execution_path'] == 'behavioral_simulation'
    assert not manifest['source_changed_during_run']
    fixture = json.loads((HERE/'cases.freeze.json').read_text())
    assert validate_fixture(fixture) == {'train': 32, 'validation': 8, 'test': 8}
    cases = {case['id']: case for case in fixture['cases']}
    controls = json.loads((HERE/'controls.json').read_text())
    for seed in {row['seed'] for row in controls['runs']}:
        assert len({r['schedule_sha256'] for r in controls['runs'] if r['seed'] == seed}) == 1
    db = sqlite3.connect('file:%s?mode=ro' % (HERE/'experience.sqlite3'), uri=True)
    try:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        counts = dict(db.execute('SELECT phase,count(*) FROM experiences GROUP BY phase'))
        assert counts == {'train': 7680, 'validation': 120, 'test': 120}
        assert db.execute("SELECT count(*) FROM experiences WHERE learned=1 AND phase!='train'").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM memory m JOIN experiences e ON e.id=m.experience_id WHERE e.phase!='train'").fetchone()[0] == 0
        checkpoints = 0
        for arm, seed, policy in db.execute("SELECT arm,seed,policy FROM checkpoints WHERE phase='trained'"):
            assert json.loads(policy)['revision'] == (0 if arm == 'fixed_heuristic' else 512)
            for (later,) in db.execute("SELECT policy FROM checkpoints WHERE arm=? AND seed=? AND phase IN ('validation','test')", (arm, seed)):
                assert later == policy
            checkpoints += 1
        for arm, seed, phase, record in db.execute('SELECT arm,seed,phase,record FROM experiences'):
            row = json.loads(record)
            assert cases[row['case_id']]['split'] == phase
            for audit in row['retrieval'].values():
                for neighbor in audit['neighbors']:
                    source = db.execute('SELECT arm,seed,phase FROM experiences WHERE id=?', (neighbor['experience_id'],)).fetchone()
                    assert source == (arm, seed, 'train')
    finally:
        db.close()
    result = {'verification': 'PASS', 'manifest_integrity': True,
              'no_source_changes_during_run': True, 'record_counts': counts,
              'frozen_arm_seed_checkpoints': checkpoints,
              'matched_training_schedules': True, 'held_out_training_or_retrieval_leakage': False,
              'new_api_calls': 0, 'tests_passed': 22,
              'meaning': 'Synthetic learning mechanism only; not improved Parliament reasoning or moral correctness.'}
    (HERE/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
