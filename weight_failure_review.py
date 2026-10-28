"""Read-only failure review for weight-only experiments. Never trains/retrieves."""
from collections import Counter, defaultdict
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import sqlite3

from behavioral_policy import ContextualPolicy

VERSION = 'weight-failure-review/0.1'
KINDS = {
    'observed_outcome_failure': 'Observed functional outcome was zero; this alone does not establish a bad choice.',
    'monitoring_suboptimal_choice': 'Recorded monitoring regret was positive under the stipulated simulator.',
    'monitoring_regression': 'Updated policy had lower recorded expected success than its matched frozen control.',
    'process_proxy_zero': 'The recorded procedural proxy was zero; this is not a general ethics verdict.',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row['arm'], row['seed'], row['split'], row['case_id']


def classify(record, evaluation=None, baseline=None):
    """Classify saved measurements only; no new counterfactual scoring."""
    tags = []
    if record['feedback']['outcome_score'] == 0:
        tags.append('observed_outcome_failure')
    if evaluation is not None and evaluation['regret'] > 1e-12:
        tags.append('monitoring_suboptimal_choice')
    if evaluation is not None and baseline is not None and evaluation['expected_score'] < baseline['expected_score'] - 1e-12:
        tags.append('monitoring_regression')
    if record['feedback'].get('process_score') == 0:
        tags.append('process_proxy_zero')
    return tags


def export_failure_review(source, output, *, source_run_id=None):
    """Build a self-contained review from existing receipts, leaving them intact.

    Monitoring measurements are joined only after selection. Training records
    never acquire a counterfactual reward or an invented optimal-action label.
    """
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ('summary.json', 'failures.jsonl', 'report.html')):
        raise ValueError('Use a fresh failure-review directory.')
    names = ('experience.sqlite3', 'cases.freeze.json', 'design.json', 'evaluations.json')
    hashes = {name: sha(source/name) for name in names}
    design = json.loads((source/'design.json').read_text())
    if design.get('version') != 'weight-update-comparison/0.1':
        raise ValueError('Unsupported weight-comparison source.')
    cases = {c['id']: c for c in json.loads((source/'cases.freeze.json').read_text())['cases']}
    evaluations = json.loads((source/'evaluations.json').read_text())
    evaluated = {key(row): row for row in evaluations}
    if len(evaluated) != len(evaluations):
        raise ValueError('Duplicate monitoring measurements.')
    source_manifest = source/'run_manifest.json'
    if source_manifest.exists():
        source_run_id = json.loads(source_manifest.read_text())['run_id']
    run_ref = source_run_id or 'UNRECORDED_EXECUTION'
    db = sqlite3.connect('file:%s?mode=ro' % (source/'experience.sqlite3').resolve(), uri=True)
    db.row_factory = sqlite3.Row
    entries, totals, kind_counts = [], Counter(), Counter()
    groups = defaultdict(lambda: {'experiences': 0, 'flagged_experiences': 0, 'kinds': Counter()})
    seen_monitoring = set()
    try:
        # Receipt IDs are stable references within the identified source run.
        monitor_ids = {(r['arm'], r['seed'], r['phase'], r['case_id']): r['id']
                       for r in db.execute("SELECT id,arm,seed,phase,case_id FROM experiences WHERE phase!='train'")}
        for receipt in db.execute('SELECT * FROM experiences ORDER BY id'):
            record = json.loads(receipt['record'])
            case = cases[receipt['case_id']]
            if case['split'] != receipt['phase'] or case['features'] != record['features']:
                raise ValueError('Experience differs from its frozen case.')
            ref = (receipt['arm'], receipt['seed'], receipt['phase'], receipt['case_id'])
            evaluation = evaluated.get(ref) if receipt['phase'] != 'train' else None
            if receipt['phase'] != 'train':
                if evaluation is None or evaluation['selected_action'] != record['selected_action']:
                    raise ValueError('Monitoring receipt lacks a matching selection measurement.')
                if evaluation['realized_outcome'] != record['feedback']['outcome_score']:
                    raise ValueError('Monitoring outcome differs from the saved receipt.')
                seen_monitoring.add(ref)
            baseline_ref = ('frozen_weights', receipt['seed'], receipt['phase'], receipt['case_id'])
            baseline = evaluated.get(baseline_ref) if receipt['arm'] == 'updated_weights' and evaluation else None
            tags = classify(record, evaluation, baseline)
            group_key = (receipt['arm'], receipt['phase'], receipt['group_id'])
            group = groups[group_key]
            group['experiences'] += 1
            totals[receipt['phase']] += 1
            if not tags:
                continue
            kind_counts.update(tags)
            group['flagged_experiences'] += 1
            group['kinds'].update(tags)
            action = record['selected_action']
            policy = ContextualPolicy.restore(record['policy_before'])
            greedy = max(record['candidate_actions'], key=record['values'].get)
            comparison = None
            if baseline:
                comparison = {'frozen_experience_id': monitor_ids[baseline_ref],
                              'frozen_action': baseline['selected_action'],
                              'frozen_expected_score': baseline['expected_score'],
                              'updated_expected_score': evaluation['expected_score'],
                              'expected_score_delta': evaluation['expected_score'] - baseline['expected_score']}
            entries.append({'id': 'failure_%s' % receipt['id'], 'version': VERSION,
                            'source_run_id': run_ref, 'experience_id': receipt['id'],
                            'arm': receipt['arm'], 'seed': receipt['seed'], 'phase': receipt['phase'],
                            'step': receipt['step'], 'case_id': receipt['case_id'], 'group_id': receipt['group_id'],
                            'kinds': tags, 'scenario': case['scenario'], 'record': record,
                            'selection': {'greedy_action_at_time': greedy,
                                          'selected_action_was_greedy': action == greedy,
                                          'mode': 'sampled_training_response' if receipt['phase'] == 'train' else 'greedy_monitoring'},
                            'chosen_value_contributions': policy.explain(record['features'])[action],
                            'prediction_error': record['feedback']['outcome_score'] - record['values'][action],
                            'monitoring': evaluation, 'frozen_comparison': comparison,
                            'semantic_status': 'RECORDED_SIMULATION_FINDING_NOT_MORAL_VERDICT'})
        if seen_monitoring != set(evaluated):
            raise ValueError('Some monitoring measurements have no source receipt.')
    finally:
        db.close()
    if any(sha(source/name) != value for name, value in hashes.items()):
        raise ValueError('Source changed while reviewing failures.')
    with (output/'failures.jsonl').open('x') as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False)+'\n')
    grouped = [{'arm': arm, 'phase': phase, 'group_id': group, **row,
                'flagged_rate': row['flagged_experiences']/row['experiences']}
               for (arm, phase, group), row in sorted(groups.items())]
    summary = {'version': VERSION, 'source_run_id': run_ref, 'source_files_sha256': hashes,
               'new_model_calls': 0, 'policy_updates': 0, 'retrieval_used': False,
               'total_experiences': sum(totals.values()), 'experience_counts_by_phase': dict(totals),
               'flagged_experiences': len(entries), 'counts_by_kind': dict(kind_counts),
               'kind_definitions': KINDS, 'grouped': grouped,
               'counting_note': 'One entry per experience; kinds can overlap. Denominators include unflagged experiences.',
               'limits': 'Review uses original feedback and monitoring measurements; no automatic correction, retraining or new moral judgment.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (output/'failure_index.csv').open('w', newline='') as stream:
        fields = ['id','experience_id','arm','seed','phase','step','case_id','group_id','kinds','selected_action','outcome_score','regret','expected_score_delta']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for entry in entries:
            writer.writerow({**{k: entry[k] for k in fields if k in entry},
                             'kinds': ';'.join(entry['kinds']),
                             'selected_action': entry['record']['selected_action'],
                             'outcome_score': entry['record']['feedback']['outcome_score'],
                             'regret': (entry['monitoring'] or {}).get('regret'),
                             'expected_score_delta': (entry['frozen_comparison'] or {}).get('expected_score_delta')})
    render_review(output, summary, entries)
    return summary


def render_review(output, summary, entries):
    lines = ['# Weight-only failure review', '',
             '%s flagged experiences out of %s recorded selections.' % (summary['flagged_experiences'], summary['total_experiences']),
             'Records preserve the original feedback, weights before/after, case text and feature contributions.',
             'Review makes no policy updates or retrieval calls. It does not replace the original experiment.', '',
             '| Finding | Count | Meaning |', '| --- | --- | --- |']
    lines += ['| %s | %s | %s |' % (kind, summary['counts_by_kind'].get(kind, 0), meaning)
              for kind, meaning in KINDS.items()]
    lines += ['', 'Kinds overlap; their counts must not be added to obtain a failure total.',
              'A zero observed outcome can follow a good expected choice. A lucky outcome can hide a poor expected choice.',
              'Training records are not assigned counterfactual regret. Monitoring findings use only the original saved evaluator measurements.',
              'Non-greedy training selections are explicitly marked; not every such failure is a ranking error.', '',
              '[Browse and filter every finding](report.html) · [Index CSV](failure_index.csv) · [Full records](failures.jsonl) · [Counts and source hashes](summary.json)', '']
    (output/'RESULT.md').write_text('\n'.join(lines))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Weight-only failure review</title><style>body{max-width:1150px;margin:36px auto;padding:0 20px;
    font:16px/1.5 system-ui;color:#253b31;background:#f4f5ef}details{background:white;margin:10px 0;padding:14px;border:1px solid #ddd}
    summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:550px;overflow:auto}
    label{display:inline-block;margin:8px}select,input{padding:8px}table{border-collapse:collapse;width:100%;background:white}
    td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.note{padding:15px;border-left:4px solid #a97844;background:white}
    a{color:#245a82}</style><h1>Where weight-only learning did not work well</h1>
    <p class="note">Saved simulation findings for inspection. Bad outcomes, suboptimal choices and regressions remain distinct.
    Retrieval and automatic correction are off. These are not moral verdicts.</p>'''
    page += '<p>%s flagged experiences / %s selections. Categories overlap.</p><table><tr><th>Finding</th><th>Count</th></tr>' % (summary['flagged_experiences'],summary['total_experiences'])
    page += ''.join('<tr><td>%s</td><td>%s</td></tr>' % (escape(kind),summary['counts_by_kind'].get(kind,0)) for kind in KINDS) + '</table>'
    for ident, label, options in (
            ('kind','Finding',KINDS), ('arm','Arm',sorted({e['arm'] for e in entries})),
            ('phase','Phase',('train','validation','test'))):
        page += '<label>%s <select id="%s"><option value="">All</option>%s</select></label>' % (label,ident,
                 ''.join('<option>%s</option>' % escape(option) for option in options))
    page += '<label>Case or action <input id="search" type="search"></label><p id="visible"></p>'
    page += '<p><a href="failure_index.csv">Index CSV</a> · <a href="failures.jsonl">Complete saved failure records</a> · <a href="RESULT.md">Definitions and limitations</a></p>'
    for entry in entries:
        record = entry['record']
        searchable = ' '.join((entry['case_id'],entry['group_id'],record['selected_action'])).lower()
        page += '<details class="finding" id="%s" data-arm="%s" data-phase="%s" data-kinds="%s" data-search="%s">' % tuple(
            escape(value,quote=True) for value in (entry['id'],entry['arm'],entry['phase'],' '.join(entry['kinds']),searchable))
        page += '<summary>%s · %s · seed %s · %s · %s · %s</summary>' % (
            escape(entry['id']),escape(entry['phase']),entry['seed'],escape(entry['case_id']),
            escape(record['selected_action']),escape(', '.join(entry['kinds'])))
        page += '<p>%s</p><p>Observed score: %s. Selection-time greedy action: %s. Weight revision: %s → %s.</p>' % (
            escape(entry['scenario']),record['feedback']['outcome_score'],escape(entry['selection']['greedy_action_at_time']),
            record['policy_before']['revision'],record['policy_after']['revision'])
        page += '<pre>%s</pre></details>' % escape(json.dumps(entry,indent=2,ensure_ascii=False))
    page += '''<script>function filter(){let count=0;for(const row of document.querySelectorAll('.finding')){
    const kind=document.getElementById('kind').value,arm=document.getElementById('arm').value,phase=document.getElementById('phase').value;
    const query=document.getElementById('search').value.toLowerCase();
    row.hidden=!!((kind&&!row.dataset.kinds.split(' ').includes(kind))||(arm&&row.dataset.arm!==arm)||
    (phase&&row.dataset.phase!==phase)||(query&&!row.dataset.search.includes(query)));if(!row.hidden)count++;}
    document.getElementById('visible').textContent=count+' records match';}
    for(const id of ['kind','arm','phase','search'])document.getElementById(id).addEventListener('input',filter);filter();</script></html>'''
    (output/'report.html').write_text(page)
