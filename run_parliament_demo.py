"""A small reporting/orchestration layer over the existing Parliament pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from parliament_framework_presentation import readable_prose, incomplete_prose, enrich_view, argument_markdown
from parliament_run_provenance import current_provenance, record_checkpoint, record_start, record_configuration, record_runtime, record_execution
from parliament_runtime import DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON, check_runtime

ROOT = Path(__file__).resolve().parent


def approve_followup(offer, depth, rounds=2, *, interactive=None, ask=None, emit=None):
    """A visible opt-in; noninteractive execution never answers on the user's behalf."""
    if depth not in {'ask','standard','deep'}:raise ValueError('Unknown review depth')
    if offer.get('status')!='AVAILABLE' or depth=='standard':return False
    if depth=='deep':return True
    emit=emit or print
    emit(offer['message'])
    for question in offer['questions']:emit(question['question'])
    interactive=sys.stdin.isatty() if interactive is None else interactive
    if not interactive:return False
    try:
        answer=(ask or input)(f'Run up to {rounds} additional review calls? [y/N] ')
    except EOFError:
        return False
    return answer.strip().lower() in {'y','yes'}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def strings(value):
    if isinstance(value, str):
        return [value] if value.strip() and value not in ('NONE', 'UNKNOWN') else []
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    return []


def first_text(*values):
    for value in values:
        for text in strings(value):
            if readable_prose(text):
                return text
    return ''


def recorded_rationale(candidate):
    """Prefer a longer recorded copy when the top-level rationale was cut off."""
    raw = candidate.get('model_argument_fields') or {}
    primary = first_text(raw.get('framework_application'), candidate.get('framework_application'),
                         raw.get('rationale'), candidate.get('rationale'))
    proposed = ((candidate.get('proposed_framework_state') or {}).get('rationale') or '').strip()
    if not readable_prose(proposed):
        return primary
    if not primary:
        return proposed
    stem = primary.split(':', 1)[-1].strip()[:60]
    if len(proposed) > len(primary) and (primary.rstrip() in proposed or proposed in primary or stem in proposed):
        return proposed
    return primary if len(primary) >= len(proposed) else proposed


def _cut_off(text):
    return incomplete_prose(text)


def landscape_comparison(candidate):
    cases = (candidate.get('model_argument_fields') or {}).get('landscape_cases', candidate.get('landscape_cases'))
    if not isinstance(cases, dict):
        return ''
    bits = [f'{action}: {line.rstrip(";")}' for action, text in cases.items() for line in strings(text) if not _cut_off(line) and readable_prose(line)]
    return 'Recorded comparison — ' + '; '.join(bits) if bits else ''


def _keep_reason_part(parts, text):
    text = (text or '').strip()
    if not readable_prose(text) or _cut_off(text):
        return
    for index, existing in enumerate(parts):
        if text in existing:
            return
        if existing in text:
            parts[index] = text
            return
    parts.append(text)


def candidate_reason(candidate, position, preference):
    """Compose the recorded explanation; do not invent a ranking or complete a truncated argument."""
    parts = []
    if position == 'ABSTAIN':
        _keep_reason_part(parts, first_text(candidate.get('framework_vote_reason')))
        _keep_reason_part(parts, first_text(candidate.get('investigative_claim')))
        if not preference and first_text(candidate.get('interim_action')):
            _keep_reason_part(parts, 'Interim working action recorded without becoming a ranking: '
                              + first_text(candidate.get('interim_action')) + '.')
    _keep_reason_part(parts, recorded_rationale(candidate))
    _keep_reason_part(parts, first_text(candidate.get('decision_rule')))
    _keep_reason_part(parts, landscape_comparison(candidate))
    for note in strings(candidate.get('epistemic_binding_notes')):
        _keep_reason_part(parts, note)
    if not parts:
        return 'No reason recorded.'
    sentences = []
    for part in parts:
        if sentences and sentences[-1][-1:] not in '.!?':
            sentences[-1] += '.'
        sentences.append(part)
    text = ' '.join(sentences)
    return text[0].upper() + text[1:]


def reversal_records(candidate):
    rows, seen = [], []
    for key in ('revised_reversal_condition', 'reversal_condition', 'normative_reversal_threshold',
                'factual_reversal_threshold', 'boundary_switch_condition'):
        for text in strings((candidate.get('model_argument_fields') or {}).get(key, candidate.get(key))):
            if text not in seen:
                seen.append(text)
                rows.append((key, text))
    return rows


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
        preference = c.get('recommended_action') or None
        if isinstance(preference, str) and not preference.strip():
            preference = None
        position = ('ABSTAIN' if not committed or not preference or c.get('framework_vote_status') == 'ABSTAIN' else
                    'FULL' if c.get('selection_status') == 'SELECTED' and not warnings and
                    ledger.get('transaction_status') == 'COMMITTED' else 'ATTENUATED')
        reason = candidate_reason(c, position, preference)
        assumptions = strings(c.get('unsupported_assumption'))
        reversals, boundaries = [], []
        for key, text in reversal_records(c):
            role = 'PREFERENCE_REVERSAL' if preference else 'ABSTENTION_BOUNDARY'
            (reversals if preference else boundaries).append(text)
            if not any(h['framework'] == c['specialist'] for h in hinges):
                hinges.append({'framework': c['specialist'], 'text': text,
                    'record': f'cycles/{len(rows)-1}/candidates/{index}/{key}',
                    'authority': 'MODEL_REPORTED_NOT_INDEPENDENTLY_VERIFIED',
                    'role': role})
        views.append(enrich_view({'framework': c['specialist'], 'position': position,
            'preference': preference, 'reason': reason,
            'assumptions': assumptions, 'live_objections': warnings,
            'dependencies': strings(c.get('decision_critical_dependency_claims')),
            'reversal_conditions': reversals, 'decision_boundaries': boundaries,
            'ledger_status': ledger.get('transaction_status', 'NONE')}, c, world))
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
                'reversal_conditions':[], 'decision_boundaries':[], 'ledger_status':'NONE'})
    status = manifest.get('pipeline_status', 'NOT_STARTED')
    status = {'ADMITTED': 'FROZEN', 'WITHHELD': 'CONSTRUCTION_WITHHELD'}.get(status, status)
    if status == 'RUNTIME_UNAVAILABLE':
        failure = failure or manifest.get('withholding_reason') or 'Runtime preflight failed.'
    if failure:
        status = 'SOFTWARE_FAILED'
    elif views:
        status = 'FRAMEWORK_PARTIAL' if any(v['position'] != 'FULL' for v in views) else 'COMPLETED'
    stop = {'stage': 'runtime_setup' if manifest.get('pipeline_status') == 'RUNTIME_UNAVAILABLE' else 'framework assessment' if views else 'construction/admission',
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
    from parliament_topography import topography_from_trace, unavailable
    if status == 'INTERPRETIVE_BEAM':
        topography = unavailable(
            'No unique frozen world was supplied to agents; no framework coordinates exist.')
    else:
        topography = topography_from_trace(trace, world, views)
    offer=trace.get('follow_up_offer') or {}
    if offer.get('status')=='AVAILABLE':
        unknowns.append('Optional deeper review is available for a decision-relevant question in the revised argument.')
    return {'version': 'parliament-demo/1', 'scenario': source,
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(), 'status': status, 'stop': stop,
        'world': world, 'grounded_effects': supported, 'stipulated_claims': commitments,
        'unknowns': list(dict.fromkeys(unknowns)), 'frameworks': views, 'hinges': hinges[:3],
        'disagreements': [{'framework':h['framework'], 'kind':'normative' if
            h['record'].endswith('normative_reversal_threshold') else 'empirical' if
            h['record'].endswith('factual_reversal_threshold') else 'unclassified',
            'dependency':h['text'], 'record':h['record']} for h in hinges[:3]
            if h.get('role') != 'ABSTENTION_BOUNDARY'],
        'agreement': 'Recorded preferences differ.' if len(preferences) > 1 else
            'Recorded preferences agree; this is not a collective verdict.' if preferences else 'No operative preference recorded.',
        'disagreement_types': 'Framework-local normative arguments and empirical dependencies are shown above; no automatic causal classification is claimed.',
        'missing_slots': manifest.get('missing_slots', []),
        'follow_up_offer': trace.get('follow_up_offer', {}),
        'reasoning_followup': trace.get('reasoning_followup', {}),
        'not_decided': ['This report does not adjudicate a governing moral verdict.',
            'Ledger admission does not prove the argument or supply missing welfare units.',
            'Unresolved objections remain live; reported reversal thresholds are not source facts.'],
        'judgment_status': trace.get('judgment_status', 'NOT_RUN'),
        'topography': topography}


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
    for view in report.get('frameworks', []):
        name = view['framework']
        if name not in {'utilitarian','deontological','care','virtue','rawlsian','contractualist'}:
            continue
        view['argument_markdown_path'] = 'arguments/'+name+'.md'
        (output/'arguments').mkdir(exist_ok=True)
        (output/view['argument_markdown_path']).write_text(argument_markdown(view))
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
    offer=report.get('follow_up_offer') or {}
    followup=report.get('reasoning_followup') or {}
    if offer.get('status')=='AVAILABLE' or followup:
        lines += ['', '## Optional deeper review', '',
                  'Follow-ups examine revised arguments. Extra calls require opt-in; a model answer is not an independently verified moral resolution.', '']
        lines += ['Interactive demo runs ask before proceeding. Use `--review-depth deep` to opt in in advance, '
                  'or `logic_puzzles_followup.py --trace <saved-review> --output-dir <new-folder> --execute` to continue a saved review.', '']
        if offer.get('status')=='AVAILABLE':
            lines += [offer['message'], '']+[q['question']+'\n' for q in offer['questions']]
        if followup:
            lines += ['Additional rounds: '+str(len(followup.get('history',[])))+
                      '. Stop: '+str(followup.get('stop_reason'))+'.', '']
            for item in followup.get('history',[]):
                answer=(item.get('native_review') or {}).get('response') or {}
                lines += ['Follow-up '+str(item['round'])+': '+item['question']['question'], '',
                          answer.get('answer','No admitted answer.'), '',
                          '[Recorded follow-up](<'+item['trace_path']+'>).', '']
    lines += ['', '## Framework views', '']
    for v in report['frameworks']:
        lines += [f"### {v['framework']}", '', f"Position: **{v['position']}**. Recorded preference: {v['preference'] or 'withheld'} (not necessarily operative).",
            '', str(v['reason']), '', '[Full recorded explanation (Markdown)]('+v.get('argument_markdown_path','#')+').',
            '', 'Assumptions / interpretive commitments: '+('; '.join(
                r['kind'].replace('_',' ').capitalize()+': '+r['text'] for r in v.get('assumption_records', []))
                or '; '.join(v['assumptions']) or 'Not separately recorded.'),
            '', v.get('source_basis',{}).get('text',''),
            '', 'Dependencies: '+('; '.join(v['dependencies']) or 'Not separately recorded.'),
            '', 'Live objection: '+('; '.join(v['live_objections']) or 'None separately recorded.'),
            '', ('Would reverse the recorded preference: '+('; '.join(v.get('reversal_explanations',v['reversal_conditions'])) or 'Not recorded.')
                 if v.get('preference') else
                 'Open comparison (no ranking to reverse): '+('; '.join(v.get('decision_boundaries') or []) or 'Not recorded.')), '']
        if v.get('critical_questions'):
            from contractualist_questions import render_questions
            lines += [render_questions({'recording_status': 'RECORDED', 'questions': v['critical_questions']})]
    lines += ['## Agreement and disagreement', '', report['agreement'], '', report['disagreement_types'],
        '']
    lines += [f"- {d['framework']}: {d['kind']} dependency — {d['dependency']}" for d in report['disagreements']]
    lines += ['', '## Decision hinges', '']
    lines += ([f"- {h['framework']}: {h['text']} ({'critical unanswered question' if h.get('role')=='OPEN_QUESTION' else 'abstention boundary, no ranking to reverse' if h.get('role')=='ABSTENTION_BOUNDARY' else 'would reverse a recorded preference'}; `{h['record']}`)." for h in report['hinges']] or ['No explicit reversal hinge was recorded. The report does not fabricate one.'])
    lines += ['', '## What Parliament did not decide', ''] + ['- '+s for s in report['not_decided']]
    topo = report.get('topography') or {}
    lines += ['', '## Topography', '', topo.get('caption') or '', '']
    if topo.get('status') != 'RECORDED':
        lines += [topo.get('unavailable_reason') or 'No native Care, welfare, or obligation coordinates were recorded.', '']
    else:
        lines += ['Recorded action loci only. Missing axes are omitted; UNKNOWN is not plotted as zero.', '']
        for action in topo.get('actions') or []:
            bits = []
            for axis in ('care', 'welfare', 'obligation'):
                point = action.get(axis) or {}
                if point.get('present'):
                    if isinstance(point.get('probability'), (int, float)):
                        value = '%.3f' % point['probability']
                        interval = point.get('interval') or {}
                        if all(isinstance(interval.get(k), (int, float)) for k in ('low','high')):
                            value += ' [%.3f, %.3f]' % (interval['low'], interval['high'])
                        bits.append(axis+': '+value)
                    else:
                        bits.append(axis+': '+str(point.get('verdict') or point.get('value')))
                else:
                    bits.append(axis+': omitted')
            lines += ['- '+action.get('id','')+' '+action.get('label','')+' — '+'; '.join(bits)]
        flags = topo.get('flags') or {}
        lines += ['', 'Degenerate: '+(', '.join(flags.get('degenerate') or []) or 'none'),
            'Attractors: '+(', '.join(flags.get('attractors') or []) or 'none'), '']
        lines += ['- '+n for n in topo.get('notices') or []]
    provenance=report.get('run_provenance') or {'status':'UNKNOWN_HISTORICAL'}
    application=provenance.get('application') or {}
    lines += ['', '## Run provenance', '',
        'Execution: '+str(provenance.get('execution_path', 'unknown historical run'))+'; run ID: '+str(provenance.get('run_id', 'not recorded'))+'.',
        'Application commit: '+str(application.get('commit', 'not recorded'))+'; source fingerprint: '+str(application.get('source_sha256', 'not recorded'))+'.',
        'Original generation: '+str((report.get('generation_provenance') or {}).get('run_id') or 'unknown historical provenance')+'.',
        'The finalized run_manifest.json records configuration, runtime checks, source snapshot, and output hashes.']
    lines += ['', '## Pipeline stop and inspection', '', str(report['stop']['reason']), '',
        'Missing slots: '+(', '.join(map(str, report['missing_slots'])) or 'none recorded'), '',
        'Competing admitted interpretations: '+('; '.join(report.get('competing_interpretations', [])) or 'none recorded'), '',
        'Inspection artifacts (saved replay references its original records):', '']
    for attempt in report.get('admission_attempts', []):
        lines += ['- '+attempt['blueprint_id']+': '+attempt['status']+' — '+
            str(attempt.get('error') or attempt.get('reason') or 'No rejection reason recorded.')]
    lines += [f"- [{p.name}]({p.resolve()})" for p in map(Path, report.get('artifacts', []))]
    (output/'DEMO_REPORT.md').write_text('\n'.join(lines)+'\n')


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path, help='Inspect an existing run without API calls; source must match exactly')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--review', action='store_true', help='Run existing bounded targeted review after independent generation')
    parser.add_argument('--review-depth', choices=('ask','standard','deep'), default='ask',
                        help='Ask before decision-relevant follow-ups; deep explicitly opts in')
    parser.add_argument('--followup-rounds', type=int, choices=(1,2), default=2)
    parser.add_argument('--baseline', action='store_true', help='Optional direct model analysis; additional API call')
    parser.add_argument('--model', default='o3')
    parser.add_argument('--chooser-model', default='gpt-4o-mini', help='Model for blueprint construction and interpretation selection')
    parser.add_argument('--parliament-root', type=Path, default=DEFAULT_PARLIAMENT_ROOT)
    parser.add_argument('--parliament-python', type=Path, default=DEFAULT_PARLIAMENT_PYTHON)
    parser.add_argument('--checkpoint-tag', action='append', help='Run label, inherited by child runs; repeatable')
    args = parser.parse_args(argv)
    if args.review_depth=='deep':
        args.review=True
    record_configuration(args)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory to preserve prior evidence.')
    args.output.mkdir(parents=True, exist_ok=True)
    record_start(args.output)
    source = ''
    framework_folder = None
    manifest, trace, failure, runtime = {}, None, None, None
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
            runtime = check_runtime(args.parliament_root, args.parliament_python)
            record_runtime(runtime)
            (args.output/'runtime_check.json').write_text(json.dumps(runtime,indent=2)+'\n')
            if not runtime['ready']:
                raise RuntimeError('Runtime preflight failed: ' + '; '.join(runtime['errors']) + '. ' + runtime['remediation'])
            code = run([sys.executable, str(ROOT/'run_blueprint_parliament.py'), '--scenario-file', str(args.scenario.resolve()),
                '--output-dir', str(folder.resolve()), '--prepare-only', '--parliament-root', str(args.parliament_root),
                '--parliament-python', str(args.parliament_python), '--chooser-model', args.chooser_model], 'construction')
        else:
            code = 0
        manifest = read(folder/'preparation_manifest.json', {})
        frozen = read(folder/'frozen_world_trace.json', {})
        trace = read(folder/'framework_generation.json') or frozen
        if args.replay:
            framework_folder = (args.replay/'frameworks' if (args.replay/'frameworks').is_dir()
                                else folder.parent/'frameworks')
            trace = read(framework_folder/'framework_generation.json') or trace
            if trace and trace.get('cycles'):
                from parliament_framework_presentation import preserve_model_arguments
                preserve_model_arguments(trace['cycles'][-1].get('candidates', []), framework_folder)
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
                    '--output-dir', str(review_dir.resolve()), '--parliament-root', str(args.parliament_root), '--model', args.model,
                    '--review-depth', 'deep' if args.review_depth=='deep' else 'standard',
                    '--followup-rounds', str(args.followup_rounds)], 'review')
                trace = read(review_dir/'targeted_review.json') or trace
                followup_path=review_dir/'deeper_review/reasoning_followup.json'
                offer=(trace or {}).get('follow_up_offer') or {}
                if not code and args.review_depth=='ask' and offer.get('status')=='AVAILABLE':
                    # Noninteractive runs retain the offer without silently spending more.
                    interactive=sys.stdin.isatty()
                    approved=approve_followup(offer,args.review_depth,args.followup_rounds,interactive=interactive)
                    offer['user_choice']='ACCEPTED' if approved else 'DECLINED' if interactive else 'OFFERED_NONINTERACTIVE'
                    record_configuration(followup_user_choice=offer['user_choice'])
                    if approved:
                        code=run([str(args.parliament_python),str(ROOT/'logic_puzzles_followup.py'),
                            '--trace',str((review_dir/'targeted_review.json').resolve()),
                            '--output-dir',str((review_dir/'deeper_review').resolve()),'--execute',
                            '--max-rounds',str(args.followup_rounds),'--model',args.model,
                            '--parliament-root',str(args.parliament_root)],'deeper_review')
                if followup_path.exists():trace=read(followup_path) or trace
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
    if manifest.get('runtime_check') or runtime:
        report['runtime_check'] = manifest.get('runtime_check') or runtime
        if not report['runtime_check']['ready']:
            report['stop']['stage'] = 'runtime_setup'
    if report.get('world'):
        record_checkpoint('saved_world_loaded' if args.replay else 'world_available', artifacts=[folder/'frozen_world_trace.json'])
    record_checkpoint('report_prepared',status=report['status'])
    report['run_provenance'] = current_provenance()
    report['generation_provenance'] = manifest.get('run_provenance') or {'status':'UNKNOWN_HISTORICAL','reason':'Original generation did not record an execution manifest.'}
    report['execution'] = {'mode': 'SAVED_REPLAY_NO_API' if args.replay else 'LIVE', 'commands': commands,
        'replay_origin': str(args.replay) if args.replay else None}
    names = ('run_manifest.json', 'runtime_check.json', 'candidate_graphs.md', 'candidate_attempts.json', 'question_assessment.json',
        'world_beam.json', 'preparation_manifest.json', 'frozen_world_trace.json',
        'framework_generation.json', 'conflict_graph.md', 'targeted_review.json',
        'reasoning_followup.json', 'reasoning_followup.md')
    report['artifacts'] = [str(p) for directory in (folder, args.output, args.output/'frameworks', args.output/'review', args.output/'review/deeper_review')
        for name in names for p in [directory/name] if p.exists()]
    if args.replay and framework_folder is not None:
        # A resumed preparation commonly has framework outputs in a sibling
        # directory. Retain that native trace for full arguments/topography.
        report['artifacts'].extend(str(p) for name in names
                                   for p in [framework_folder/name] if p.exists()
                                   and str(p) not in report['artifacts'])
    beam = read(folder/'world_beam.json', {})
    report['competing_interpretations'] = [m['blueprint_id']+': '+', '.join(m.get('assignment', []))
        for m in beam.get('members', []) if m.get('admission_status') == 'ADMITTED']
    attach_candidate_diagrams(report, folder)
    write_report(report, args.output)
    record_checkpoint('report_written',artifacts=[args.output/'DEMO_REPORT.json'])
    print(report['status'] + ': ' + str(args.output/'DEMO_REPORT.md'))
    return 1 if failure or report['status'] == 'SOFTWARE_FAILED' else 0


def main(argv=None):
    return record_execution(_main, argv, entrypoint='run_parliament_demo.py', output_flag='--output', defaults={'model': 'o3', 'chooser_model': 'gpt-4o-mini', 'runtime_profile': 'compact_frameworks'})


if __name__ == '__main__':
    raise SystemExit(main())
