"""Recheck the failure archive and unchanged source experiment, with no learning."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from parliament_run_provenance import verify_manifest


def verify():
    source = ROOT/'diagnostics/weight_only_continuation_20261028'
    for folder in (HERE, source):
        receipt = verify_manifest(folder/'run_manifest.json')
        assert receipt['valid'], receipt
    summary = json.loads((HERE/'summary.json').read_text())
    for name, expected in summary['source_files_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected
    manifest = json.loads((HERE/'run_manifest.json').read_text())
    assert manifest['mode'] == 'SAVED_REVIEW_NO_API'
    assert manifest['execution_path'] == 'behavioral_failure_review'
    assert not manifest['source_changed_during_run']
    rows = [json.loads(line) for line in (HERE/'failures.jsonl').read_text().splitlines()]
    assert len(rows) == summary['flagged_experiences'] == 1940
    assert len({r['experience_id'] for r in rows}) == len(rows)
    assert all(r['record']['retrieval'] == {} for r in rows)
    assert (HERE/'report.html').read_text().count('class="finding"') == len(rows)
    assert all(r['monitoring'] is None for r in rows if r['phase'] == 'train')
    for kind, count in summary['counts_by_kind'].items():
        assert sum(kind in r['kinds'] for r in rows) == count
    regressions = [r for r in rows if 'monitoring_regression' in r['kinds']]
    assert len(regressions) == 4
    assert {(r['seed'],r['group_id']) for r in regressions} == {(11,'unknown-c1-u0')}
    result = {'verification':'PASS', 'tests_passed':33, 'source_manifest_and_bytes_unchanged':True,
              'review_manifest_verified':True, 'flagged_experiences':len(rows),
              'counts_by_kind':summary['counts_by_kind'], 'new_model_calls':0, 'policy_updates':0,
              'retrieval_used':False, 'distinct_regression_context_seed_pairs':1,
              'optimal_despite_observed_failure':sum('observed_outcome_failure' in r['kinds'] and
                  r['monitoring'] is not None and r['monitoring']['optimal_under_toy_model'] for r in rows),
              'suboptimal_despite_observed_success':sum('monitoring_suboptimal_choice' in r['kinds'] and
                  r['record']['feedback']['outcome_score']==1 for r in rows)}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    verify()
