"""Resume weight-only learning, paired with an identical frozen policy. No API."""
import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import random
import statistics

from behavioral_policy import ContextualPolicy, ExperienceStore
from contextual_moral_simulation import ACTIONS, FEATURES, VERSION as SIM_VERSION, feedback, validate_fixture
from run_contextual_moral_generalization import decision, dump, evaluate, fingerprint

ROOT = Path(__file__).resolve().parent
DEFAULT_PARENT = ROOT/'diagnostics/contextual_moral_generalization_20261027'
ARMS = ('frozen_weights', 'updated_weights')
VERSION = 'weight-update-comparison/0.1'


def load_parent(folder):
    from parliament_run_provenance import verify_manifest
    folder = Path(folder)
    receipt = verify_manifest(folder/'run_manifest.json')
    if not receipt['valid']:
        raise ValueError('Parent experiment integrity check failed: ' + '; '.join(receipt['errors']))
    fixture = json.loads((folder/'cases.freeze.json').read_text())
    validate_fixture(fixture)
    design = json.loads((folder/'design.json').read_text())
    if design.get('version') not in (VERSION, SIM_VERSION):
        raise ValueError('Unsupported continuation design version.')
    if fingerprint(fixture) != design['fixture_sha256']:
        raise ValueError('Parent cue/case fixture differs from its recorded design.')
    saved = json.loads((folder/'policy_checkpoints.json').read_text())
    # Always continue the predeclared updated/weight-only arm, never select a
    # seed or checkpoint on validation performance.
    prefix = 'updated_weights' if design.get('version') == VERSION else 'learned_values'
    policies = {seed: saved['%s:%s' % (prefix, seed)] for seed in design['seeds']}
    for policy in policies.values():
        restored = ContextualPolicy.restore(policy)
        if restored.actions != ACTIONS or restored.features != FEATURES:
            raise ValueError('Checkpoint does not match this experiment action/feature vocabulary.')
    files = ('cases.freeze.json', 'design.json', 'policy_checkpoints.json', 'run_manifest.json')
    origin = {'parent_run_id': receipt['run_id'], 'source_arm': prefix,
              'parent_files': {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files}}
    return fixture, policies, origin


def summary(rows):
    result = {}
    for phase in ('validation', 'test'):
        result[phase] = {}
        for arm in ARMS:
            subset = [r for r in rows if r['split'] == phase and r['arm'] == arm]
            result[phase][arm] = {
                'expected_score': statistics.mean(r['expected_score'] for r in subset),
                'optimal_action_rate': statistics.mean(r['optimal_under_toy_model'] for r in subset),
                'regret': statistics.mean(r['regret'] for r in subset),
                'observed_success': statistics.mean(r['realized_outcome'] for r in subset)}
        paired = []
        for seed in sorted({row['seed'] for row in rows}):
            per_seed = {arm: statistics.mean(r['expected_score'] for r in rows
                                            if r['split'] == phase and r['seed'] == seed and r['arm'] == arm)
                        for arm in ARMS}
            paired.append({'seed': seed, 'expected_score_delta': per_seed['updated_weights'] - per_seed['frozen_weights']})
        result[phase]['paired_by_seed'] = paired
        result[phase]['mean_expected_score_delta'] = statistics.mean(r['expected_score_delta'] for r in paired)
    return result


def compare(fixture, initial_policies, origin, output, *, epochs=16):
    validate_fixture(fixture)
    if type(epochs) is not int or epochs < 1 or not initial_policies:
        raise ValueError('Require saved policies and a positive continuation budget.')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ('design.json', 'experience.sqlite3', 'results.json')):
        raise ValueError('Use a new comparison directory.')
    seeds = sorted(initial_policies)
    # Freeze before training; evaluation sets were already seen in the earlier
    # report, so they are monitoring sets, not a fresh generalization test.
    design = {'version': VERSION, 'mode': 'SIMULATION_NO_API', 'seeds': seeds, 'epochs': epochs,
              'arms': list(ARMS), 'fixture_sha256': fingerprint(fixture), 'origin': origin,
              'same_initial_weights_per_pair': True, 'retrieval_enabled': False,
              'training_signal': 'observed_outcome_only', 'new_api_calls': 0,
              'schedule_offset': 10000, 'outcome_seed_offset': 100000,
              'evaluation_status': 'REUSED_MONITORING_CASES_NOT_FRESH_HOLDOUT',
              'parliament_policy_changed': False}
    dump(output/'design.json', design)
    dump(output/'cases.freeze.json', fixture)
    dump(output/'initial_policies.json', {str(seed): p for seed, p in initial_policies.items()})
    from parliament_run_provenance import record_checkpoint, record_configuration
    record_configuration(design)
    record_checkpoint('weight_comparison_frozen', artifacts=[output/'design.json', output/'initial_policies.json'])
    cases, rows, curves, controls, final_policies, explanations = fixture['cases'], [], [], [], {}, []
    store = ExperienceStore(output/'experience.sqlite3', retain_memory=False)
    try:
        for seed in seeds:
            training = [case for case in cases if case['split'] == 'train']
            scheduler = random.Random(seed + 10000)
            schedule = []
            for _ in range(epochs):
                epoch = list(training); scheduler.shuffle(epoch); schedule += epoch
            for arm in ARMS:
                policy = ContextualPolicy.restore(initial_policies[seed])
                start = policy.snapshot()
                store.checkpoint(arm, seed, 'initial', policy)
                rng, window = random.Random(seed + 10000), []
                for step, case in enumerate(schedule):
                    action, values, probabilities, retrieved = decision(policy, case, arm, store, seed, rng)
                    assert not retrieved
                    observed = feedback(case, action, seed + 100000, 'train', step)
                    store.record(arm=arm, seed=seed, phase='train', step=step, case=case, action=action,
                                 values=values, probabilities=probabilities, feedback=observed,
                                 retrieval={}, policy=policy, learn=arm == 'updated_weights')
                    window.append(observed['decision_score'])
                    if (step + 1) % 64 == 0 or step + 1 == len(schedule):
                        curves.append({'arm': arm, 'seed': seed, 'step': step + 1,
                                       'expected_score': statistics.mean(window)})
                        window = []
                if arm == 'frozen_weights' and policy.snapshot() != start:
                    raise AssertionError('The frozen control changed.')
                store.checkpoint(arm, seed, 'trained', policy)
                trained = policy.snapshot()
                for phase in ('validation', 'test'):
                    assessed = evaluate(policy, cases, arm, store, seed, phase)
                    rows += assessed
                    store.checkpoint(arm, seed, phase, policy)
                    if policy.snapshot() != trained:
                        raise AssertionError('Monitoring changed the model.')
                final_policies['%s:%s' % (arm, seed)] = trained
                before_policy = ContextualPolicy.restore(start)
                for case in cases:
                    if case['surface_variant'] == 0 and case['split'] != 'train':
                        explanations.append({'seed': seed, 'arm': arm, 'group_id': case['group_id'],
                                             'features': case['features'], 'before': before_policy.explain(case['features']),
                                             'after': policy.explain(case['features']),
                                             'meaning': 'Weight contributions to predicted functional success, not ethical warrants.'})
                controls.append({'seed': seed, 'arm': arm, 'initial_sha256': fingerprint(start),
                                 'trained_sha256': fingerprint(trained),
                                 'schedule_sha256': fingerprint([case['id'] for case in schedule]),
                                 'starting_revision': start['revision'], 'ending_revision': policy.revision,
                                 'weight_updates': policy.revision - start['revision'],
                                 'training_trials': len(schedule), 'monitoring_weights_unchanged': True})
                print('%s seed=%s: revision %s -> %s' % (arm, seed, start['revision'], policy.revision), flush=True)
        memory_count = store.db.execute('SELECT count(*) FROM memory').fetchone()[0]
        illegal = store.db.execute("SELECT count(*) FROM experiences WHERE learned=1 AND phase!='train'").fetchone()[0]
        assert memory_count == illegal == 0
    finally:
        store.close()
    dump(output/'policy_checkpoints.json', final_policies)
    with (output/'weight_changes.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arm', 'seed', 'action', 'term', 'before', 'after', 'delta'])
        for seed in seeds:
            before = initial_policies[seed]
            for arm in ARMS:
                after = final_policies['%s:%s' % (arm, seed)]
                for action in ACTIONS:
                    for term, left, right in zip(before['terms'], before['weights'][action], after['weights'][action]):
                        writer.writerow([arm, seed, action, term, left, right, right-left])
    dump(output/'weight_explanations.json', explanations)
    dump(output/'controls.json', {'pairs': controls, 'memory_rows': memory_count, 'evaluation_updates': illegal})
    dump(output/'learning_curves.json', curves)
    dump(output/'evaluations.json', rows)
    result = {'version': VERSION, 'configuration': design, 'summary': summary(rows)}
    dump(output/'results.json', result)
    write_report(output, result, rows, curves)
    from weight_failure_review import export_failure_review
    from parliament_run_provenance import current_provenance
    export_failure_review(output, output/'failure_review', source_run_id=current_provenance().get('run_id'))
    record_checkpoint('weight_failures_recorded', artifacts=[output/'failure_review/summary.json', output/'failure_review/failures.jsonl'])
    record_checkpoint('weight_comparison_completed', artifacts=[output/'results.json', output/'controls.json'])
    return result


def write_report(output, result, rows, curves):
    lines = ['# Continued weight learning versus identical frozen weights', '',
             'Both arms resumed exactly the same trained checkpoints. Only one received additional weight updates.',
             'Retrieval is disabled. No Parliament agent, admitted world or operative selection changed.', '',
             '**These are reused monitoring cases, not a new held-out generalization claim.**', '',
             '| Monitoring split / arm | Expected functional score | Toy-optimal choices | Regret | Observed success |',
             '| --- | --- | --- | --- | --- |']
    html_rows = []
    for phase in ('validation', 'test'):
        for arm in ARMS:
            r = result['summary'][phase][arm]
            cells = [phase + ' / ' + arm, '%.4f' % r['expected_score'], '%.1f%%' % (100*r['optimal_action_rate']),
                     '%.4f' % r['regret'], '%.3f' % r['observed_success']]
            lines.append('| ' + ' | '.join(cells) + ' |')
            html_rows.append('<tr>' + ''.join('<td>%s</td>' % escape(cell) for cell in cells) + '</tr>')
    lines += ['', 'Each pair had the same initial weights, case schedule, random outcome draws and training budget.',
              'The policies may choose different actions, so their observed experience differs after they diverge.',
              'Only the selected action\u2019s sampled outcome updates weights; no oracle alternative enters learning.',
              'Frozen controls do not update. Neither arm updates during monitoring. No best checkpoint is chosen from monitoring scores.', '',
              'Improvement is not guaranteed: constant-rate updates from noisy outcomes can also cause regressions.',
              'Scores describe the specified simulator. They do not establish better Parliament reasoning or moral correctness.', '',
              '[Browser report](report.html) · [failure review](failure_review/report.html) · [paired results](results.json) · [weight changes](weight_changes.csv) · [weight explanations](weight_explanations.json) · [checkpoints](policy_checkpoints.json)', '',
              'Continue again from the updated arm, using a fresh output directory:', '', '```bash',
              '.venv/bin/python -B run_weight_update_comparison.py \\',
              '  --from-run ' + str(output) + ' \\',
              '  --output diagnostics/weight_learning_next', '```', '']
    (output/'RESULT.md').write_text('\n'.join(lines))
    with (output/'evaluation.csv').open('w', newline='') as stream:
        fields = ['arm','seed','split','case_id','selected_action','expected_score','realized_outcome','regret']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    chart = ['<svg viewBox="0 0 800 300" role="img" aria-label="Expected functional success during further training">',
             '<path d="M65 20 V250 H770" fill="none" stroke="#8b9d91"/>']
    for value in (0, 0.25, 0.5, 0.75, 1):
        y = 250 - 220*value
        chart += ['<path d="M65 %g H770" stroke="#ddd"/>' % y,
                  '<text x="52" y="%g" text-anchor="end">%.2f</text>' % (y+4, value)]
    max_step = max(row['step'] for row in curves)
    for step in sorted({0, max_step//2, max_step}):
        chart.append('<text x="%g" y="275" text-anchor="middle">%s</text>' % (65+700*step/max_step, step))
    for arm, color in zip(ARMS, ('#8b6654', '#306b8b')):
        points = []
        for step in sorted({row['step'] for row in curves}):
            values = [r['expected_score'] for r in curves if r['arm'] == arm and r['step'] == step]
            points.append('%g,%g' % (65+700*step/max_step, 250-220*statistics.mean(values)))
        chart.append('<polyline points="%s" stroke="%s" stroke-width="3" fill="none"/>' % (' '.join(points), color))
    chart.append('<text x="360" y="298">Additional training trials</text></svg>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Weight-only continuation comparison</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;
    font:16px/1.5 system-ui;background:#f4f5ef;color:#253d32}svg{width:100%;background:white}
    table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}
    .note{padding:15px;border-left:4px solid #a57d45;background:white}a{color:#235e88}</style>
    <h1>Do additional weight updates improve choices?</h1><p class="note">Identical trained starting policies.
    Retrieval off. Reused monitoring cases, not fresh generalization. No changes to Parliament's operative decisions.</p>
    <h2>Expected functional success during further training</h2>'''
    page += ''.join(chart) + '<p><span style="color:#8b6654">Frozen weights</span> · <span style="color:#306b8b">Updated weights</span></p>'
    page += '<h2>Frozen evaluation after training</h2><table><tr><th>Split / arm</th><th>Expected score</th><th>Toy-optimal</th><th>Regret</th><th>Observed</th></tr>'
    page += ''.join(html_rows) + '</table><p>Scores use a hand-specified simulator, not a moral truth scale.</p>'
    page += '<p><a href="RESULT.md">Read controls and limitations</a> · <a href="failure_review/report.html">Browse failures and regressions</a> · <a href="evaluation.csv">Inspect every evaluation choice</a> · <a href="weight_changes.csv">Compare individual weights</a> · <a href="weight_explanations.json">Inspect weight contributions</a></p></html>'
    (output/'report.html').write_text(page)


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-only', action='store_true', help='Archive failures from a saved comparison without training')
    args = parser.parse_args(argv)
    if args.review_only:
        from parliament_run_provenance import verify_manifest, record_resource, record_configuration, record_checkpoint
        receipt = verify_manifest(args.from_run/'run_manifest.json')
        if not receipt['valid']:
            raise ValueError('Source comparison failed integrity verification.')
        for name in ('run_manifest.json','experience.sqlite3','cases.freeze.json','design.json','evaluations.json'):
            record_resource(args.from_run/name)
        record_configuration(review_only=True, source_run_id=receipt['run_id'], new_model_calls=0, policy_updates=0)
        from weight_failure_review import export_failure_review
        result = export_failure_review(args.from_run, args.output, source_run_id=receipt['run_id'])
        record_checkpoint('weight_failures_recorded', artifacts=[args.output/'summary.json',args.output/'failures.jsonl'])
        print(json.dumps({k:v for k,v in result.items() if k not in ('grouped', 'kind_definitions')},indent=2))
        print('Failure report:', args.output/'report.html')
        return 0
    fixture, policies, origin = load_parent(args.from_run)
    from parliament_run_provenance import record_resource
    for name in origin['parent_files']:
        record_resource(args.from_run/name)
    result = compare(fixture, policies, origin, args.output)
    print(json.dumps(result['summary'], indent=2))
    print('Report:', args.output/'report.html')
    return 0


def main(argv=None):
    from parliament_run_provenance import record_execution
    return record_execution(_main, argv, entrypoint='run_weight_update_comparison.py', output_flag='--output')


if __name__ == '__main__':
    raise SystemExit(main())
