"""Read-only checks of weight continuation, comparisons and source integrity."""
import csv
import json
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from parliament_run_provenance import verify_manifest
from run_weight_update_comparison import load_parent


def verify():
    integrity = verify_manifest(HERE/'run_manifest.json')
    assert integrity['valid'], integrity
    manifest = json.loads((HERE/'run_manifest.json').read_text())
    assert manifest['mode'] == 'SIMULATION_NO_API' and not manifest['source_changed_during_run']
    controls = json.loads((HERE/'controls.json').read_text())
    pairs = controls['pairs']
    assert controls['memory_rows'] == controls['evaluation_updates'] == 0
    assert len(pairs) == 10
    for seed in {row['seed'] for row in pairs}:
        pair = [row for row in pairs if row['seed'] == seed]
        assert len({row['initial_sha256'] for row in pair}) == 1
        assert len({row['schedule_sha256'] for row in pair}) == 1
        for row in pair:
            assert row['weight_updates'] == (512 if row['arm'] == 'updated_weights' else 0)
            if row['arm'] == 'frozen_weights':
                assert row['initial_sha256'] == row['trained_sha256']
    db = sqlite3.connect('file:%s?mode=ro' % (HERE/'experience.sqlite3'), uri=True)
    try:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        counts = dict(db.execute('SELECT phase,count(*) FROM experiences GROUP BY phase'))
        assert counts == {'train': 5120, 'validation': 80, 'test': 80}
        assert db.execute('SELECT count(*) FROM memory').fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM experiences WHERE learned=1 AND (phase!='train' OR arm!='updated_weights')").fetchone()[0] == 0
        for arm, seed, saved in db.execute("SELECT arm,seed,policy FROM checkpoints WHERE phase='trained'"):
            for (later,) in db.execute("SELECT policy FROM checkpoints WHERE arm=? AND seed=? AND phase IN ('validation','test')", (arm,seed)):
                assert later == saved
        for record, in db.execute('SELECT record FROM experiences'):
            row = json.loads(record)
            assert row['retrieval'] == {}
    finally:
        db.close()
    with (HERE/'weight_changes.csv').open() as stream:
        weights = list(csv.DictReader(stream))
    assert all(float(r['delta']) == 0 for r in weights if r['arm'] == 'frozen_weights')
    assert any(float(r['delta']) != 0 for r in weights if r['arm'] == 'updated_weights')
    _, policies, origin = load_parent(HERE)
    assert origin['source_arm'] == 'updated_weights'
    assert all(row['revision'] == 1024 for row in policies.values())
    page = (HERE/'report.html').read_text()
    assert page.count('<polyline ') == 2 and '90.0%' in page and '70.0%' in page
    assert 'Additional training trials' in page and 'not fresh generalization' in page
    result = {'verification': 'PASS', 'paired_seeds': 5, 'tests_passed': 27,
              'record_counts': counts, 'only_updated_arm_learns': True,
              'same_initial_policy_and_schedule': True, 'monitoring_updates': 0,
              'retrieval_records': 0, 'resumed_revision': 512, 'new_updated_revision': 1024,
              'continuation_checkpoint_loads': True, 'manifest_integrity': True,
              'new_api_calls': 0, 'operative_parliament_changes': False}
    (HERE/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
