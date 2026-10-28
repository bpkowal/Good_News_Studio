"""One-command, no-API contextual learning experiment; never updates Parliament."""
import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import random
import statistics

from behavioral_policy import ContextualPolicy, ExperienceStore, values_with_memory
from contextual_moral_simulation import (
    ACTIONS, ARMS, FEATURES, FEATURE_SCHEMA, VERSION, expected_scores, feedback,
    fixed_action, validate_fixture,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURE = ROOT/'fixtures/contextual_moral_generalization_v01.json'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def decision(policy, case, arm, store, seed, rng=None):
    features = case['features']
    if arm == 'fixed_heuristic':
        action = fixed_action(features)
        values = {a: float(a == action) for a in ACTIONS}
        return action, values, values, {}
    if arm == 'learned_values_memory':
        values, retrieved = values_with_memory(policy, features, store, arm, seed, FEATURE_SCHEMA)
    else:
        values, retrieved = policy.values(features), {}
    if rng is not None:
        action, probabilities = policy.choose(values, rng, epsilon=0.15)
    else:
        action = max(ACTIONS, key=values.get)
        probabilities = {a: float(a == action) for a in ACTIONS}
    return action, values, probabilities, retrieved


def evaluate(policy, cases, arm, store, seed, phase):
    before, rows = policy.snapshot(), []
    for step, case in enumerate(c for c in cases if c['split'] == phase):
        action, values, probabilities, retrieved = decision(policy, case, arm, store, seed)
        observed = feedback(case, action, seed, phase, step)
        # Oracle alternatives are used only by the evaluator after selection.
        oracle = expected_scores(case['features'])
        best = max(oracle.values())
        store.record(arm=arm, seed=seed, phase=phase, step=step, case=case, action=action,
                     values=values, probabilities=probabilities, feedback=observed,
                     retrieval=retrieved, policy=policy, learn=False)
        rows.append({'arm': arm, 'seed': seed, 'split': phase, 'case_id': case['id'],
                     'group_id': case['group_id'], 'selected_action': action,
                     'values': values, 'greedy_selection': True,
                     'model_selection_distribution': policy.distribution(values),
                     'expected_score': oracle[action], 'realized_outcome': observed['outcome_score'],
                     'process_score': observed['process_score'],
                     'regret': best - oracle[action], 'best_actions': [a for a in ACTIONS if oracle[a] == best],
                     'optimal_under_toy_model': oracle[action] == best,
                     'retrieval': retrieved})
    if policy.snapshot() != before:
        raise AssertionError('Evaluation changed the policy.')
    return rows


def mean(rows, key):
    return statistics.mean(row[key] for row in rows)


def summarize(rows, seeds):
    summary = {}
    for phase in ('validation', 'test'):
        summary[phase] = {}
        for arm in ARMS:
            selected = [r for r in rows if r['arm'] == arm and r['split'] == phase]
            per_seed = [{key: mean([r for r in selected if r['seed'] == seed], key)
                         for key in ('expected_score', 'realized_outcome', 'regret', 'optimal_under_toy_model', 'process_score')}
                        for seed in seeds]
            summary[phase][arm] = {
                'seed_count': len(seeds),
                'mean_expected_score': statistics.mean(r['expected_score'] for r in per_seed),
                'seed_sd_expected_score': statistics.stdev(r['expected_score'] for r in per_seed) if len(seeds) > 1 else None,
                'mean_realized_outcome': mean(selected, 'realized_outcome'),
                'mean_regret': mean(selected, 'regret'),
                'optimal_action_rate': mean(selected, 'optimal_under_toy_model'),
                'mean_process_score': mean(selected, 'process_score'),
                'contexts': {group: {'expected_score': mean([r for r in selected if r['group_id'] == group], 'expected_score'),
                                     'regret': mean([r for r in selected if r['group_id'] == group], 'regret'),
                                     'actions': {a: sum(r['group_id'] == group and r['selected_action'] == a for r in selected)
                                                 for a in ACTIONS}}
                             for group in sorted({r['group_id'] for r in selected})}}
        baseline = summary[phase]['fixed_heuristic']['mean_expected_score']
        summary[phase]['paired_mean_differences_vs_fixed'] = {
            arm: summary[phase][arm]['mean_expected_score'] - baseline for arm in ARMS[1:]}
    return summary


def render_outputs(output, result, rows, curves):
    dump(output/'results.json', result)
    dump(output/'evaluations.json', rows)
    dump(output/'learning_curves.json', curves)
    with (output/'evaluation.csv').open('w', newline='') as stream:
        fields = ['arm','seed','split','case_id','group_id','selected_action','expected_score',
                  'realized_outcome','regret','optimal_under_toy_model','process_score']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Contextual Moral Generalization v0.1', '',
             'Offline simulated terminal-action weighting. No API calls, agent edits, live decisions or world-state mutations.', '',
             '48 synthetic scenarios in 12 cue-combination groups: 32 training, 8 validation, 8 test.',
             'Each held-out split has two unseen context groups and four surface variants per group.',
             'The variants are not independent contexts. Features and outcome contingencies are hand specified.', '',
             '## Held-out results', '',
             '| Split / arm | Expected functional score | Regret | Toy-optimal action rate | Observed success | Process proxy |',
             '| --- | --- | --- | --- | --- | --- |']
    for phase in ('validation', 'test'):
        for arm in ARMS:
            row = result['summary'][phase][arm]
            lines.append('| %s / %s | %.3f | %.3f | %.1f%% | %.3f | %.3f |' %
                         (phase, arm, row['mean_expected_score'], row['mean_regret'],
                          100 * row['optimal_action_rate'], row['mean_realized_outcome'], row['mean_process_score']))
    lines += ['', 'Expected score and regret use the stipulated simulator, not a moral truth metric.',
              'Observed success is sampled: a good expected choice may fail, and a poor one may succeed.',
              'Only observed outcome feedback updates weights. Decision and process scores stay separate.',
              '%s seeds estimate sensitivity to training exposure and luck; they do not create new independent ethical cases.' % len(result['configuration']['seeds']),
              'The fixed control is a coercion-based heuristic, not a fixed LLM or Parliament baseline.', '',
              '## Counterexamples to a familiar cue', '',
              '| Test context / arm | Expected score | Selected actions across seeds and surface variants |',
              '| --- | --- | --- |']
    for arm in ARMS:
        for group, record in result['summary']['test'][arm]['contexts'].items():
            selected = ', '.join('%s: %s' % (a, n) for a, n in record['actions'].items() if n)
            lines.append('| %s / %s | %.3f | %s |' % (group, arm, record['expected_score'], selected))
    lines += ['', '## Controls and interpretation', '',
              '- Split, configuration, evaluator source and input hashes were frozen before the first trial.',
              '- All arms received the same scenario schedule and common simulator random draws for each seed.',
              '- The policy sees only the shared feature schema and the three available actions; no case IDs, labels or oracle alternatives.',
              '- Retrieval uses only observed training outcomes from the same arm and seed. Success/failure examples remain inspectable.',
              '- Held-out evaluation is greedy, makes no updates, and preserves the exact frozen model checkpoint.',
              '- Retrieval and feature interactions were specified in advance; neither validation nor test tuned this run.',
              '- The experiment has one disclosure domain and only two test contexts. Results do not establish improved Parliament reasoning, moral correctness, or broad transfer.',
              '- Missing cues, new actions, learned agent recruitment and live Parliament integration are later increments.', '',
              '[Interactive record](report.html) · [results](results.json) · [all held-out selections](evaluation.csv) · [SQLite experiences](experience.sqlite3) · [frozen design](design.json)', '']
    (output/'RESULT.md').write_text('\n'.join(lines))
    colors = {'fixed_heuristic': '#8b6654', 'learned_values': '#305e88', 'learned_values_memory': '#347459'}
    svg = ['<svg viewBox="0 0 740 260" role="img" aria-label="Training expected functional score by trial window">',
           '<path d="M40 15 V220 H720" fill="none" stroke="#89988e"/>']
    max_step = max(c['step'] for c in curves)
    for arm in ARMS:
        points = []
        for step in sorted({c['step'] for c in curves}):
            window = [c for c in curves if c['arm'] == arm and c['step'] == step]
            points.append('%g,%g' % (40 + 680 * step / max_step, 220 - 200 * mean(window, 'expected_score')))
        svg.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="3"/>' % (' '.join(points), colors[arm]))
    svg.append('<text x="42" y="248">Training trials (window means across seeds; no held-out feedback)</text></svg>')
    detail_rows = []
    for row in rows:
        detail_rows.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.3f</td><td>%.3f</td></tr>' %
                           (escape(row['split']), escape(row['arm']), escape(row['case_id']),
                            escape(row['selected_action']), row['expected_score'], row['regret']))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Contextual action-learning pilot</title><style>
    body{max-width:1100px;margin:40px auto;padding:0 22px;background:#f5f5f0;color:#24392e;font:16px/1.5 system-ui}
    svg{width:100%;background:white;border:1px solid #ccc}table{border-collapse:collapse;width:100%;background:white}
    td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}summary,select{padding:12px;cursor:pointer}
    .note{border-left:4px solid #b8883c;padding:12px;background:#fff}a{color:#23577a}</style>
    <h1>Learning which response works in which context</h1>
    <p class="note">Synthetic simulation, not a moral verdict. Parliament's agents and world models are unchanged.
    Values measure predefined functional contingencies. Features are supplied by the experiment.</p>'''
    page += ''.join(svg) + '<p>' + ' · '.join('<span style="color:%s">%s</span>' % (colors[a], a) for a in ARMS) + '</p>'
    page += '<p><a href="RESULT.md">Read the results and limitations</a> · <a href="evaluation.csv">Download selections</a></p>'
    page += '<details><summary>Inspect held-out actions, including failures</summary><label>Filter arm <select id="arm"><option value="">All</option>'
    page += ''.join('<option>%s</option>' % a for a in ARMS) + '</select></label>'
    page += '<table><thead><tr><th>Split</th><th>Arm</th><th>Case</th><th>Action</th><th>Expected score</th><th>Regret</th></tr></thead><tbody id="rows">'
    page += ''.join(detail_rows) + '</tbody></table></details>'
    page += '''<script>document.getElementById('arm').addEventListener('change',e=>{
    for(const row of document.querySelectorAll('#rows tr'))row.hidden=!!e.target.value&&row.cells[1].textContent!==e.target.value;
    });</script></html>'''
    (output/'report.html').write_text(page)


def run(fixture, output, seeds=(11, 23, 37, 41, 53), epochs=16):
    counts = validate_fixture(fixture)
    if not seeds or len(set(seeds)) != len(seeds) or epochs < 1:
        raise ValueError('Distinct seeds and a positive epoch count are required.')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ('design.json', 'experience.sqlite3', 'results.json')):
        raise ValueError('Use a fresh experiment directory.')
    cases = fixture['cases']
    config = {'version': VERSION, 'mode': 'SIMULATION_NO_API', 'seeds': list(seeds), 'epochs': epochs,
              'alpha': 0.3, 'temperature': 0.15, 'epsilon_training': 0.15, 'memory_blend': 0.25,
              'memory_neighbors': 12, 'evaluation': 'FROZEN_GREEDY_NO_UPDATE',
              'training_signal': 'observed_outcome_only', 'feedback_revision': 'original_no_reassessment_yet',
              'fixture_sha256': fingerprint(fixture), 'split_counts': counts,
              'split_unit': 'entire_feature_combination', 'new_api_calls': 0,
              'agents_modified': False, 'operative_policy_changed': False}
    from parliament_run_provenance import record_checkpoint, record_configuration
    record_configuration(config)
    dump(output/'design.json', config)
    dump(output/'cases.freeze.json', fixture)
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                     ('behavioral_policy.py', 'contextual_moral_simulation.py', 'run_contextual_moral_generalization.py')}
    dump(output/'implementation_hashes.json', source_hashes)
    record_checkpoint('behavioral_design_frozen', artifacts=[output/'design.json', output/'cases.freeze.json'])
    store = ExperienceStore(output/'experience.sqlite3')
    rows, curves, checkpoints, controls = [], [], {}, []
    try:
        for seed in seeds:
            training = [case for case in cases if case['split'] == 'train']
            schedule, scheduler = [], random.Random(seed)
            for _ in range(epochs):
                epoch = list(training); scheduler.shuffle(epoch); schedule += epoch
            schedule_hash = fingerprint([c['id'] for c in schedule])
            for arm in ARMS:
                policy, rng = ContextualPolicy(ACTIONS, FEATURES), random.Random(seed)
                store.checkpoint(arm, seed, 'initial', policy)
                window = []
                for step, case in enumerate(schedule):
                    action, values, probabilities, retrieved = decision(policy, case, arm, store, seed, rng)
                    observed = feedback(case, action, seed, 'train', step)
                    store.record(arm=arm, seed=seed, phase='train', step=step, case=case, action=action,
                                 values=values, probabilities=probabilities, feedback=observed,
                                 retrieval=retrieved, policy=policy, learn=arm != 'fixed_heuristic')
                    window.append(observed['decision_score'])
                    if (step + 1) % 64 == 0 or step + 1 == len(schedule):
                        curves.append({'arm': arm, 'seed': seed, 'step': step + 1,
                                       'expected_score': statistics.mean(window)})
                        window = []
                store.checkpoint(arm, seed, 'trained', policy)
                frozen = policy.snapshot()
                for phase in ('validation', 'test'):
                    rows += evaluate(policy, cases, arm, store, seed, phase)
                    store.checkpoint(arm, seed, phase, policy)
                assert frozen == policy.snapshot()
                checkpoints['%s:%s' % (arm, seed)] = frozen
                controls.append({'arm': arm, 'seed': seed, 'schedule_sha256': schedule_hash,
                                 'policy_frozen_during_evaluation': True,
                                 'weight_updates': policy.revision, 'trials': len(schedule)})
                print('%s seed=%s completed: %s training trials; held-out weights unchanged' %
                      (arm, seed, len(schedule)), flush=True)
        illegal = store.db.execute("SELECT count(*) FROM experiences WHERE learned=1 AND phase!='train'").fetchone()[0]
        leakage = store.db.execute("SELECT count(*) FROM memory m JOIN experiences e ON m.experience_id=e.id WHERE e.phase!='train'").fetchone()[0]
        assert illegal == leakage == 0
        dump(output/'policy_checkpoints.json', checkpoints)
        dump(output/'controls.json', {'runs': controls, 'held_out_updates': illegal,
                                     'held_out_memory_records': leakage, 'split_counts': counts})
    finally:
        store.close()
    result = {'version': VERSION, 'configuration': config, 'summary': summarize(rows, seeds),
              'interpretation': 'Transfer within a hand-annotated disclosure simulator; not demonstrated improvement in Parliament or moral correctness.'}
    render_outputs(output, result, rows, curves)
    record_checkpoint('behavioral_comparison_completed', artifacts=[output/'results.json', output/'controls.json'])
    return result


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args(argv)
    from parliament_run_provenance import record_resource
    record_resource(args.fixture)
    result = run(json.loads(args.fixture.read_text()), args.output)
    print(json.dumps(result['summary'], indent=2))
    print('Saved report:', args.output/'report.html')
    return 0


def main(argv=None):
    from parliament_run_provenance import record_execution
    return record_execution(_main, argv, entrypoint='run_contextual_moral_generalization.py', output_flag='--output')


if __name__ == '__main__':
    raise SystemExit(main())
