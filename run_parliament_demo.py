"""A small reporting/orchestration layer over the existing Parliament pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def strings(value):
    if isinstance(value, str):
        return [value] if value.strip() and value not in ('NONE', 'UNKNOWN') else []
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    return []


def summarize(source, manifest, trace=None, failure=None):
    """Report recorded conclusions; missing reasons/hinges stay missing."""
    trace = trace or {}
    world = (trace.get('action_source_grounding') or {}).get('world_model') or {}
    rows = trace.get('cycles', [])
    candidates = rows[-1].get('candidates', []) if rows else []
    views, hinges, unknowns = [], [], []
    for index, c in enumerate(candidates):
        ledger = c.get('committed_native_ledger') or {}
        committed = str(ledger.get('transaction_status', '')).startswith('COMMITTED')
        warnings = list(dict.fromkeys([s for k,v in c.items() if k.endswith('validation_errors') for s in strings(v)] +
            strings(c.get('framework_specific_open_questions')) + strings(c.get('framework_internal_conflicts'))))
        position = ('ABSTAIN' if not committed or not c.get('recommended_action') or c.get('framework_vote_status') == 'ABSTAIN' else
                    'FULL' if c.get('selection_status') == 'SELECTED' and not warnings and
                    ledger.get('transaction_status') == 'COMMITTED' else 'ATTENUATED')
        reason = next(iter(strings(c.get('framework_application')) or strings(c.get('rationale'))), 'No reason recorded.')
        assumptions = strings(c.get('unsupported_assumption'))
        reversals = []
        for key in ('revised_reversal_condition', 'reversal_condition', 'normative_reversal_threshold',
                    'factual_reversal_threshold', 'boundary_switch_condition'):
            for text in strings(c.get(key)):
                if text not in reversals:
                    reversals.append(text)
                    if not any(h['framework'] == c['specialist'] for h in hinges):
                        hinges.append({'framework': c['specialist'], 'text': text,
                            'record': f'cycles/{len(rows)-1}/candidates/{index}/{key}',
                            'authority': 'MODEL_REPORTED_NOT_INDEPENDENTLY_VERIFIED'})
        views.append({'framework': c['specialist'], 'position': position,
            'preference': c.get('recommended_action') or None, 'reason': reason,
            'assumptions': assumptions, 'live_objections': warnings,
            'dependencies': strings(c.get('decision_critical_dependency_claims')),
            'reversal_conditions': reversals, 'ledger_status': ledger.get('transaction_status', 'NONE')})
        unknowns.extend(warnings)
        if c.get('specialist') == 'utilitarian':
            unknowns.extend(strings(c.get('utilitarian_missing_comparison')))
            unknowns.extend(strings(c.get('utilitarian_incommensurable_remainder')))
            if any(r.get('magnitude') == 'UNKNOWN' or r.get('duration') == 'UNKNOWN' for r in ledger.get('records', [])):
                unknowns.append('The utilitarian ledger contains unknown welfare magnitude or duration; survivor arithmetic does not settle total welfare.')
    for run in trace.get('framework_runs', []):
        if not any(v['framework'] == run['framework'] for v in views):
            views.append({'framework':run['framework'], 'position':'ABSTAIN', 'preference':None,
                'reason':'No candidate argument was produced.', 'assumptions':[], 'dependencies':[],
                'live_objections':['Framework generation did not produce an assessable candidate.'],
                'reversal_conditions':[], 'ledger_status':'NONE'})
    status = manifest.get('pipeline_status', 'NOT_STARTED')
    status = {'ADMITTED': 'FROZEN', 'WITHHELD': 'CONSTRUCTION_WITHHELD'}.get(status, status)
    if failure:
        status = 'SOFTWARE_FAILED'
    elif views:
        status = 'FRAMEWORK_PARTIAL' if any(v['position'] != 'FULL' for v in views) else 'COMPLETED'
    stop = {'stage': 'framework assessment' if views else 'construction/admission',
            'reason': failure or manifest.get('withholding_reason') or
                (manifest.get('parliament_admission') or {}).get('error') or status,
            'kind': 'software_failure' if failure else 'epistemic_abstention' if status in
                ('INTERPRETIVE_BEAM', 'CONSTRUCTION_WITHHELD') else 'recorded_pipeline_outcome'}
    supported = []
    for effect in world.get('effects', []):
        excerpts = [p.get('excerpt', '') for p in effect.get('provenance', []) if isinstance(p, dict)]
        supported.append({'id': effect['effect_id'], 'claim': effect.get('outcome'),
            'operation': effect.get('derivation_operation'),
            'source_excerpts': [e for e in excerpts if e and e in source],
            'source_check': 'EXACT_EXCERPT' if any(e and e in source for e in excerpts) else 'NO_EXACT_EXCERPT'})
    commitments = [p for p in trace.get('proposition_ledger', []) if p.get('epistemic_status') == 'STIPULATED']
    preferences = {v['preference'] for v in views if v['preference']}
    return {'version': 'parliament-demo/1', 'scenario': source,
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(), 'status': status, 'stop': stop,
        'world': world, 'grounded_effects': supported, 'stipulated_claims': commitments,
        'unknowns': list(dict.fromkeys(unknowns)), 'frameworks': views, 'hinges': hinges[:3],
        'disagreements': [{'framework':h['framework'], 'kind':'normative' if
            h['record'].endswith('normative_reversal_threshold') else 'empirical' if
            h['record'].endswith('factual_reversal_threshold') else 'unclassified',
            'dependency':h['text'], 'record':h['record']} for h in hinges[:3]],
        'agreement': 'Recorded preferences differ.' if len(preferences) > 1 else
            'Recorded preferences agree; this is not a collective verdict.' if preferences else 'No operative preference recorded.',
        'disagreement_types': 'Framework-local normative arguments and empirical dependencies are shown above; no automatic causal classification is claimed.',
        'missing_slots': manifest.get('missing_slots', []),
        'not_decided': ['This report does not adjudicate a governing moral verdict.',
            'Ledger admission does not prove the argument or supply missing welfare units.',
            'Unresolved objections remain live; reported reversal thresholds are not source facts.'],
        'judgment_status': trace.get('judgment_status', 'NOT_RUN')}


def attach_candidate_diagrams(report, folder):
    """Keep beam diagrams separate from a world supplied to the frameworks."""
    report['candidate_world_diagrams'] = []
    report['diagram_notices'] = []
    report['candidate_attempts'] = []
    report['failed_candidate_graphs_markdown'] = ''
    packet = read(folder/'blueprint_candidate.json', {})
    if packet and packet.get('scenario') == report['scenario']:
        for row in packet.get('blueprint_result', {}).get('considered', []):
            missing = row.get('unfilled', [])
            optional = {item['id'] for item in row.get('items', []) if item.get('core') is False}
            required_missing = [slot for slot in missing if slot not in optional]
            report['candidate_attempts'].append({'blueprint_id': row['blueprint_id'],
                'construction_status': row.get('status'), 'missing_slots': required_missing,
                'unfilled_optional_slots': [slot for slot in missing if slot in optional]})
            report['missing_slots'].extend(row['blueprint_id']+': '+str(slot) for slot in required_missing)
        report['missing_slots'] = list(dict.fromkeys(report['missing_slots']))
    if report['status'] in ('ADMISSION_FAILED', 'CONSTRUCTION_WITHHELD'):
        graph_path = folder/'candidate_graphs.md'
        if graph_path.exists() and packet.get('scenario') == report['scenario']:
            report['failed_candidate_graphs_markdown'] = graph_path.read_text()
        report['admission_attempts'] = read(folder/'preparation_manifest.json', {}).get('admission_attempts', [])
        report['stop']['kind'] = ('construction_failure' if report['status'] == 'ADMISSION_FAILED'
                                  else 'epistemic_abstention')
        return
    if report['status'] != 'INTERPRETIVE_BEAM':
        return
    beam = read(folder/'world_beam.json', {})
    for member in beam.get('members', []):
        if member.get('admission_status') != 'ADMITTED':
            continue
        path = Path(member.get('trace_path') or '')
        try:
            trace = read(path, {}) if path.is_file() else {}
            world = (trace.get('action_source_grounding') or {}).get('world_model') or {}
            if trace.get('scenario') != report['scenario'] or not world:
                raise ValueError('Matching source-bound candidate trace is unavailable.')
            report['candidate_world_diagrams'].append({'label':member['blueprint_id'],
                'world':world, 'trace_path':str(path), 'sent_to_agents':False})
        except (OSError, ValueError) as exc:
            report['diagram_notices'].append(member['blueprint_id']+': '+str(exc))


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/'DEMO_REPORT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    lines = ['# Parliament demo', '', f"Status: **{report['status']}** ({report['stop']['kind']}).",
        '', 'Native judgment status: '+str(report['judgment_status'])+'.',
        '', '## Scenario', '', report['scenario'], '', '## What Parliament thinks is happening', '']
    world = report['world']
    lines += ['Actions: ' + '; '.join(a['intervention'] for a in world.get('actions', [])) if world else 'No unique frozen world is reported.']
    if world:
        party_labels = {p['party_id']:p['label'] for p in world.get('parties', [])}
        lines += ['', 'Parties: ' + ', '.join(p['label'] for p in world.get('parties', [])), '',
            'These are alternative action branches and their stipulated outcomes, not claims that both actions occurred.', '']
        labels = {e['effect_id']: e['outcome'] for e in world.get('effects', [])}
        lines += [f"- Under {e.get('action_id')}, {party_labels.get(e.get('party_id'), '')}: {e.get('outcome')} ({e.get('modality')})" for e in world.get('effects', [])]
        lines += [f"- {labels.get(l['source_id'], l['source_id'])} → {l['relation']} → {labels.get(l['target_id'], l['target_id'])}" for l in world.get('causal_links', [])]
        scene = [p['predication'] for p in world.get('propositions', []) if p.get('status') == 'ASSERTED']
        if scene:
            lines += ['', 'Source scene and constraints:', ''] + ['- '+s for s in scene]
    candidates = report.get('candidate_world_diagrams', [])
    if world or candidates:
        from run_blueprint_parliament import _mermaid_world
        lines += ['', '## World-state diagram', '']
        if world:
            caption = ('Frozen world supplied to framework assessment' if report['frameworks'] else
                       'Frozen world — no framework assessment recorded')
            lines += _mermaid_world(world, caption)
        else:
            lines += ['No unique world was frozen. The following are competing admitted interpretations; **none was sent to agents**.', '']
            for candidate in candidates:
                lines += _mermaid_world(candidate['world'], candidate['label']+' — candidate only; not sent to agents')
    lines += ['Diagram unavailable: '+s for s in report.get('diagram_notices', [])]
    if report.get('failed_candidate_graphs_markdown'):
        lines += ['', '## Unaccepted candidate graphs', '',
            'These are construction attempts, not the shared world. None was sent to agents.', '',
            report['failed_candidate_graphs_markdown']]
    lines += ['', '## What is grounded', '',
        'Admitted source copies and structurally derived records follow. Exact excerpts establish traceability; they do not independently prove every interpretation.', '']
    lines += [f"- {e['claim']} — {e['operation'] or 'admitted record'}; source: “{'; '.join(e['source_excerpts'])}”" for e in report['grounded_effects'] if e['source_check'] == 'EXACT_EXCERPT']
    lines += [f"- Stipulated source claim: {p.get('claim')}" for p in report['stipulated_claims']]
    lines += ['', '## What remains uncertain', '']
    lines += (['- '+s for s in report['unknowns']] or ['No additional framework uncertainty was recorded; this is not proof of completeness.'])
    lines += ['', '## Framework views', '']
    for v in report['frameworks']:
        lines += [f"### {v['framework']}", '', f"Position: **{v['position']}**. Recorded preference: {v['preference'] or 'withheld'} (not necessarily operative).",
            '', str(v['reason']), '', 'Assumption: '+('; '.join(v['assumptions']) or 'Not separately recorded.'),
            '', 'Dependencies: '+('; '.join(v['dependencies']) or 'Not separately recorded.'),
            '', 'Live objection: '+('; '.join(v['live_objections']) or 'None separately recorded.'),
            '', 'Reversal: '+('; '.join(v['reversal_conditions']) or 'Not recorded.'), '']
    lines += ['## Agreement and disagreement', '', report['agreement'], '', report['disagreement_types'],
        '']
    lines += [f"- {d['framework']}: {d['kind']} dependency — {d['dependency']}" for d in report['disagreements']]
    lines += ['', '## Decision hinges', '']
    lines += ([f"- {h['framework']}: {h['text']} (reported argument, not verified premise; `{h['record']}`)." for h in report['hinges']] or ['No explicit reversal hinge was recorded. The report does not fabricate one.'])
    lines += ['', '## What Parliament did not decide', ''] + ['- '+s for s in report['not_decided']]
    lines += ['', '## Pipeline stop and inspection', '', str(report['stop']['reason']), '',
        'Missing slots: '+(', '.join(map(str, report['missing_slots'])) or 'none recorded'), '',
        'Competing admitted interpretations: '+('; '.join(report.get('competing_interpretations', [])) or 'none recorded'), '',
        'Inspection artifacts (saved replay references its original records):', '']
    for attempt in report.get('admission_attempts', []):
        lines += ['- '+attempt['blueprint_id']+': '+attempt['status']+' — '+
            str(attempt.get('error') or attempt.get('reason') or 'No rejection reason recorded.')]
    lines += [f"- [{p.name}]({p.resolve()})" for p in map(Path, report.get('artifacts', []))]
    (output/'DEMO_REPORT.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path, help='Inspect an existing run without API calls; source must match exactly')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--review', action='store_true', help='Run existing bounded targeted review after independent generation')
    parser.add_argument('--baseline', action='store_true', help='Optional direct model analysis; additional API call')
    parser.add_argument('--model', default='o3')
    parser.add_argument('--parliament-root', type=Path, default=Path('/tmp/parliament-smoke-614af0c'))
    parser.add_argument('--parliament-python', type=Path, default=Path('/tmp/parliament-smoke-env/bin/python'))
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory to preserve prior evidence.')
    args.output.mkdir(parents=True, exist_ok=True)
    source = ''
    manifest, trace, failure = {}, None, None
    folder = args.replay or args.output/'pipeline'
    if args.replay and not (folder/'preparation_manifest.json').exists() and (folder/'pipeline/preparation_manifest.json').exists():
        folder = folder/'pipeline'
    commands = []
    def run(command, label):
        commands.append(command)
        with (args.output/(label+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return result.returncode
    try:
        source = args.scenario.read_text().strip()
        if not source:
            raise ValueError('Scenario text is empty.')
        if not args.replay:
            code = run([sys.executable, str(ROOT/'run_blueprint_parliament.py'), '--scenario-file', str(args.scenario.resolve()),
                '--output-dir', str(folder.resolve()), '--prepare-only', '--parliament-root', str(args.parliament_root),
                '--parliament-python', str(args.parliament_python)], 'construction')
        else:
            code = 0
        manifest = read(folder/'preparation_manifest.json', {})
        frozen = read(folder/'frozen_world_trace.json', {})
        trace = read(folder/'framework_generation.json') or frozen
        if args.replay and manifest.get('deliberation', {}).get('trace'):
            trace = read(Path(manifest['deliberation']['trace'])) or trace
        if args.replay and not manifest and frozen:
            manifest = {'pipeline_status': 'ADMITTED'}
        if code and not manifest:
            raise RuntimeError('Construction process failed before its manifest. See construction.log.')
        recorded_source = manifest.get('scenario') or (trace or {}).get('scenario')
        if recorded_source != source:
            trace = None
            raise ValueError('Saved source does not match supplied scenario; no replay report can be attributed to it.')
        if not args.replay and frozen and not args.prepare_only:
            before = hashlib.sha256((folder/'frozen_world_trace.json').read_bytes()).hexdigest()
            generation = args.output/'frameworks'
            code = run([str(args.parliament_python), str(ROOT/'logic_puzzles_frameworks.py'), '--trace', str((folder/'frozen_world_trace.json').resolve()),
                '--output-dir', str(generation.resolve()), '--parliament-root', str(args.parliament_root), '--model', args.model], 'frameworks')
            trace = read(generation/'framework_generation.json') or frozen
            if code:
                failure = 'Framework process failed. Saved partial records remain available in frameworks.log and frameworks/.'
            if args.review and not code:
                review_dir = args.output/'review'
                code = run([str(args.parliament_python), str(ROOT/'logic_puzzles_review.py'), '--trace', str((generation/'framework_generation.json').resolve()),
                    '--output-dir', str(review_dir.resolve()), '--parliament-root', str(args.parliament_root), '--model', args.model], 'review')
                trace = read(review_dir/'targeted_review.json') or trace
                if code:
                    failure = 'Bounded review process failed; see review.log.'
            if hashlib.sha256((folder/'frozen_world_trace.json').read_bytes()).hexdigest() != before:
                raise RuntimeError('Frozen trace changed during framework execution.')
        if args.baseline:
            if args.replay:
                raise ValueError('Replay mode makes no API calls; baseline requires a live run.')
            code = run([str(args.parliament_python), str(ROOT/'parliament_demo_baseline.py'),
                '--scenario', str(args.scenario.resolve()), '--output', str(args.output.resolve()),
                '--parliament-root', str(args.parliament_root), '--model', args.model], 'baseline')
            if code:
                raise RuntimeError('Optional baseline failed; see baseline.log. Parliament records remain available.')
            (args.output/'BASELINE_COMPARISON.md').write_text('Compare BASELINE.md with DEMO_REPORT.md descriptively. No automatic score or unsupported verdict is assigned.\n\n- Unsupported factual assumptions?\n- Ambiguity collapsed?\n- Expected survivors equated with total welfare?\n- Reversal conditions exposed?\n- Live disagreement identified?\n')
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    report = summarize(source, manifest, trace, failure)
    report['execution'] = {'mode': 'SAVED_REPLAY_NO_API' if args.replay else 'LIVE', 'commands': commands,
        'replay_origin': str(args.replay) if args.replay else None}
    names = ('candidate_graphs.md', 'candidate_attempts.json', 'question_assessment.json',
        'world_beam.json', 'preparation_manifest.json', 'frozen_world_trace.json',
        'framework_generation.json', 'conflict_graph.md', 'targeted_review.json')
    report['artifacts'] = [str(p) for directory in (folder, args.output/'frameworks', args.output/'review')
        for name in names for p in [directory/name] if p.exists()]
    beam = read(folder/'world_beam.json', {})
    report['competing_interpretations'] = [m['blueprint_id']+': '+', '.join(m.get('assignment', []))
        for m in beam.get('members', []) if m.get('admission_status') == 'ADMITTED']
    attach_candidate_diagrams(report, folder)
    write_report(report, args.output)
    print(report['status'] + ': ' + str(args.output/'DEMO_REPORT.md'))
    return 1 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
