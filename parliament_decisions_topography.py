"""Offline Decisions-API coordinates for Care × welfare × obligation.

Plotted values are predicate probabilities from /v1/decisions, not native
ordinals. The Parliament packet is the evidence. A refusal or missing answer
is omitted; it is never written as 0. Saved scenarios may replace the point estimate
with a jackknife interval from ten packet-text ablations.
"""
import hashlib
import json
import math
import urllib.error
import urllib.request

from parliament_topography import (
    action_catalog, last_candidates, native_trace_from_report, ranking,
    topography_from_trace, unique_max, unavailable,
)

DECISIONS_URL = 'https://api.openai.com/v1/decisions'
DECISIONS_MODEL = 'gpt-6-luna'
TOPOGRAPHY_VERSION = 'parliament-topography/2'
CAPTION = (
    'Plotted coordinates are Decisions-API probabilities that each action '
    'satisfies a named criterion, given the Parliament packet. They are not '
    'native ledger ordinals or total welfare. A refused or missing answer is '
    'a hole, not zero. Attractors are occupancy of recorded points.'
)
JACKKNIFE_CAPTION = (
    'Plotted coordinates are jackknife 95% intervals of Decisions-API '
    'probabilities. Each replicate ablates one contiguous tenth of the '
    'Parliament packet text. The marker is the jackknife mean. A refused or '
    'missing replicate is omitted, not zero. Attractors require a unique '
    'maximum whose interval sits entirely above the others on that axis.'
)
AXES = ('care', 'welfare', 'obligation')
JACKKNIFE_FOLDS = 10
JACKKNIFE_MIN_PRESENT = 8
JACKKNIFE_LEVEL = 0.95
# Student's t 0.975 quantile by degrees of freedom (n_folds_used - 1).
T_0975 = {
    1: 12.706204736432095, 2: 4.302652729911275, 3: 3.182446305284341,
    4: 2.7764451051977987, 5: 2.5705818366147395, 6: 2.446911851144969,
    7: 2.364624251592784, 8: 2.3060041352041663, 9: 2.2621571627409915,
    10: 2.228138851986272,
}
QUESTIONS = {
    'care': (
        'Given this Parliament packet, is this action adequately responsive '
        'to the relevant dependents under a care-ethics standard? Use the '
        'source, frozen world, specialist arguments, ledgers, objections, '
        'and abstentions. Do not treat a missing Care ledger as proof of neglect.'
    ),
    'welfare': (
        'Given this Parliament packet, does this action maximize expected '
        'survivors among the recorded alternatives? Use stipulated survival '
        'chances and any arithmetic-verified expected-value records. Unknown '
        'total-welfare magnitude does not by itself decide this question.'
    ),
    'obligation': (
        'Given this Parliament packet, is this action at least deontologically '
        'permissible (not prohibited)? Use the source, frozen world, duty '
        'arguments, ledgers, and live objections. A conflicted or contested '
        'duty should lower this probability relative to a clearly permissible '
        'or required action.'
    ),
}
CANDIDATE_FIELDS = (
    'specialist', 'recommended_action', 'framework_vote_status', 'framework_vote_reason',
    'rationale', 'decision_rule', 'investigative_claim', 'landscape_cases',
    'expected_value_estimates', 'committed_native_ledger', 'action_scores',
    'framework_specific_open_questions', 'framework_internal_conflicts',
    'normative_reversal_threshold', 'factual_reversal_threshold',
    'comparison_complete', 'interim_action', 'adjudication_status',
    'epistemic_binding_notes', 'selection_status', 'action_admissibility',
)


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()
                if k not in ('trace_path', 'commands', 'openai_env', 'api_key')}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, str) and ('/Users/' in value or '/tmp/' in value or 'OPENAI_API_KEY' in value):
        return '[omitted-local-path]'
    return value


def parliament_packet(source, world, trace, frameworks=None):
    """Shared evidence for Decisions questions. No local paths or credentials."""
    candidates = []
    for row in last_candidates(trace):
        slim = {key: _clean(row.get(key)) for key in CANDIDATE_FIELDS if key in row}
        candidates.append(slim)
    world_slice = {}
    if isinstance(world, dict):
        for key in ('parties', 'actions', 'effects', 'causal_links', 'conditions', 'propositions'):
            rows = world.get(key) or []
            if key == 'parties':
                world_slice[key] = [{'party_id': r.get('party_id'), 'label': r.get('label'), 'kind': r.get('kind')} for r in rows]
            elif key == 'actions':
                world_slice[key] = [{'action_id': r.get('action_id'), 'intervention': r.get('intervention'),
                                     'effect_ids': r.get('effect_ids')} for r in rows]
            elif key == 'effects':
                world_slice[key] = [{'effect_id': r.get('effect_id'), 'outcome': r.get('outcome'),
                                     'party_id': r.get('party_id')} for r in rows]
            else:
                world_slice[key] = _clean(rows)
    views = []
    for view in frameworks or []:
        views.append({k: view.get(k) for k in (
            'framework', 'position', 'preference', 'reason', 'live_objections',
            'reversal_conditions', 'decision_boundaries', 'ledger_status')})
    return {
        'source': source or (trace or {}).get('scenario') or '',
        'world': world_slice,
        'frameworks': views,
        'candidates': candidates,
        'judgment_status': (trace or {}).get('judgment_status'),
    }


def packet_body(packet):
    return json.dumps(packet, ensure_ascii=False, indent=2)


def packet_text(packet, action, body=None):
    body = packet_body(packet) if body is None else body
    return (
        'ACTION UNDER EVALUATION: %s — %s\n\n'
        'Evaluate only this action against the named predicate. The Parliament '
        'packet below is evidence, including abstentions and missing ledgers. '
        'Do not invent world facts that are not in the packet.\n\n'
        '%s' % (action['id'], action['label'], body)
    )


def packet_blocks(text, folds=JACKKNIFE_FOLDS):
    """Contiguous character slices covering `text`. Remainder goes to earlier folds."""
    n = len(text)
    sizes = [n // folds + (1 if i < n % folds else 0) for i in range(folds)]
    blocks, start = [], 0
    for size in sizes:
        blocks.append((text[start:start + size], start, start + size))
        start += size
    return blocks


def ablate_packet_text(text, fold, folds=JACKKNIFE_FOLDS):
    """Drop one contiguous tenth of the packet text. Empty slices are a no-op ablation."""
    blocks = packet_blocks(text, folds)
    kept = ''.join(chunk for i, (chunk, _, _) in enumerate(blocks) if i != fold)
    chunk, start, end = blocks[fold]
    return kept, {
        'fold': fold,
        'omitted_start': start,
        'omitted_end': end,
        'omitted_chars': end - start,
        'omitted_text': chunk,
    }


def packet_block_records(text, folds=JACKKNIFE_FOLDS):
    return [{
        'fold': fold,
        'omitted_start': start,
        'omitted_end': end,
        'omitted_chars': end - start,
        'omitted_text': chunk,
    } for fold, (chunk, start, end) in enumerate(packet_blocks(text, folds))]


def jackknife_interval(replicates, level=JACKKNIFE_LEVEL):
    """Leave-one-block-out interval. Missing replicates are omitted, never scored as 0."""
    present = [float(value) for value in replicates if isinstance(value, (int, float))]
    n = len(present)
    if n < JACKKNIFE_MIN_PRESENT:
        return None
    mean = sum(present) / n
    var_jack = ((n - 1) / n) * sum((value - mean) ** 2 for value in present)
    if var_jack < 1e-18:
        var_jack = 0.0
    se = math.sqrt(var_jack)
    t_crit = T_0975.get(n - 1, 1.959963984540054)
    half = t_crit * se
    return {
        'method': 'jackknife',
        'ablation': 'contiguous_tenth_of_packet_text',
        'level': level,
        'folds': len(replicates),
        'folds_used': n,
        'mean': mean,
        'se': se,
        't_crit': t_crit,
        'low': max(0.0, min(1.0, mean - half)),
        'high': max(0.0, min(1.0, mean + half)),
    }


def combine_jackknife_points(action, fold_rows, folds=JACKKNIFE_FOLDS):
    points = {}
    for axis in AXES:
        samples, replicate_rows = [], []
        for meta, parsed in fold_rows:
            cell = parsed.get(axis) or {}
            row = dict(meta)
            if cell.get('present') and isinstance(cell.get('probability'), (int, float)):
                samples.append(float(cell['probability']))
                row['probability'] = float(cell['probability'])
            else:
                samples.append(None)
                row['omit_reason'] = cell.get('omit_reason') or 'No Decisions probability was returned.'
            replicate_rows.append(row)
        used = sum(value is not None for value in samples)
        interval = jackknife_interval(samples)
        if interval is None:
            annotate_replicate_influence(replicate_rows, None)
            points[axis] = {
                'present': False,
                'omit_reason': 'Jackknife had %s/%s usable folds; missing folds are not treated as zero.' % (used, folds),
                'source': 'decisions/v1',
                'interval': {
                    'method': 'jackknife', 'ablation': 'contiguous_tenth_of_packet_text',
                    'folds': folds, 'folds_used': used, 'replicates': replicate_rows,
                    'hinging': hinging_record(replicate_rows),
                },
            }
            continue
        interval['replicates'] = replicate_rows
        annotate_replicate_influence(replicate_rows, interval['mean'])
        interval['hinging'] = hinging_record(replicate_rows, mean=interval['mean'])
        points[axis] = {
            'present': True,
            'probability': interval['mean'],
            'encoding': 'probability',
            'question': QUESTIONS[axis],
            'source': 'decisions/v1',
            'model': DECISIONS_MODEL,
            'interval': interval,
        }
    return points


def annotate_replicate_influence(replicates, mean):
    """Score how far each leave-one-tenth estimate sits from the jackknife mean.

    Missing folds stay without a numeric influence. They are never filled with 0.
    influence = (n-1) * (mean - loo): positive means the omitted tenth was raising
    the estimate when present.
    """
    present = [row for row in replicates if isinstance(row.get('probability'), (int, float))]
    n = len(present)
    if mean is None and n:
        mean = sum(float(row['probability']) for row in present) / n
    for row in replicates:
        if mean is None or n < 2 or not isinstance(row.get('probability'), (int, float)):
            continue
        probability = float(row['probability'])
        delta = probability - mean
        row['delta_from_mean'] = delta
        row['abs_delta'] = abs(delta)
        row['influence'] = (n - 1) * (mean - probability)
    return mean


def hinging_record(replicates, blocks=None, mean=None):
    """The tenth whose ablation moved the leave-one-out estimate the most."""
    scored = [row for row in replicates if isinstance(row.get('abs_delta'), (int, float))]
    if not scored:
        return {'present': False, 'omit_reason': 'No fold has a recorded leave-one-out probability.'}
    top = max(row['abs_delta'] for row in scored)
    if top < 1e-12:
        return {'present': False, 'omit_reason': 'No ablation moved the leave-one-out estimate.'}
    scored.sort(key=lambda row: (-row['abs_delta'], row.get('fold', 0)))
    winner = scored[0]
    tied = [row['fold'] for row in scored if abs(row['abs_delta'] - top) < 1e-12]
    influence = winner.get('influence')
    if isinstance(influence, (int, float)):
        effect = 'raises_estimate' if influence > 1e-12 else ('lowers_estimate' if influence < -1e-12 else 'none')
    else:
        effect = None
    omitted_text = winner.get('omitted_text')
    if omitted_text is None and blocks is not None:
        match = next((block for block in blocks if block.get('fold') == winner.get('fold')), None)
        if match:
            omitted_text = match.get('omitted_text')
    return {
        'present': True,
        'fold': winner.get('fold'),
        'omitted_start': winner.get('omitted_start'),
        'omitted_end': winner.get('omitted_end'),
        'omitted_chars': winner.get('omitted_chars'),
        'loo_probability': winner.get('probability'),
        'mean': mean,
        'delta_from_mean': winner.get('delta_from_mean'),
        'abs_delta': winner.get('abs_delta'),
        'influence': influence,
        'effect_when_present': effect,
        'tied_folds': tied if len(tied) > 1 else [],
        'ranking': [{
            'fold': row.get('fold'),
            'loo_probability': row.get('probability'),
            'delta_from_mean': row.get('delta_from_mean'),
            'abs_delta': row.get('abs_delta'),
            'influence': row.get('influence'),
            'omitted_start': row.get('omitted_start'),
            'omitted_end': row.get('omitted_end'),
            'omitted_chars': row.get('omitted_chars'),
        } for row in scored],
        **({'omitted_text': omitted_text} if omitted_text is not None else {}),
    }


def attach_jackknife_diagnostics(topography, packet=None):
    """Persist every omitted tenth, influence scores, and the most hinging fold.

    Does not call the API. Missing folds stay holes. Zero movement is not a hinge.
    """
    if not isinstance(topography, dict) or topography.get('status') != 'RECORDED':
        return topography
    jackknife = any(((row.get(axis) or {}).get('interval') or {}).get('method') == 'jackknife'
                    for row in topography.get('actions') or [] for axis in AXES)
    if not jackknife:
        return topography
    body = packet_body(packet) if packet is not None else None
    jack = topography.setdefault('provenance', {}).setdefault('jackknife', {
        'folds': JACKKNIFE_FOLDS, 'min_present': JACKKNIFE_MIN_PRESENT,
        'level': JACKKNIFE_LEVEL, 'ablation': 'contiguous_tenth_of_packet_text',
    })
    blocks = jack.get('packet_blocks')
    if body is not None:
        blocks = packet_block_records(body)
        jack['packet_chars'] = len(body)
        jack['packet_sha256'] = hashlib.sha256(body.encode('utf-8')).hexdigest()
        jack['packet_text'] = body
        jack['packet_blocks'] = blocks
        mismatch = False
        for row in topography.get('actions') or []:
            for axis in AXES:
                for replicate in ((row.get(axis) or {}).get('interval') or {}).get('replicates') or []:
                    start, end = replicate.get('omitted_start'), replicate.get('omitted_end')
                    if not isinstance(start, int) or not isinstance(end, int):
                        continue
                    if start < 0 or end > len(body) or (replicate.get('omitted_chars') not in (None, end - start)):
                        mismatch = True
                    elif replicate.get('omitted_text') is None:
                        replicate['omitted_text'] = body[start:end]
        if mismatch:
            jack['packet_offset_mismatch'] = True
        else:
            jack.pop('packet_offset_mismatch', None)
    global_best = None
    for row in topography.get('actions') or []:
        action_best = None
        for axis in AXES:
            interval = (row.get(axis) or {}).get('interval') or {}
            if interval.get('method') != 'jackknife':
                continue
            replicates = interval.get('replicates') or []
            annotate_replicate_influence(replicates, interval.get('mean'))
            hinge = hinging_record(replicates, blocks=blocks, mean=interval.get('mean'))
            interval['hinging'] = hinge
            if blocks:
                for replicate in replicates:
                    replicate.pop('omitted_text', None)
            if hinge.get('present') and (action_best is None or hinge['abs_delta'] > action_best['abs_delta']):
                action_best = {
                    'axis': axis,
                    'fold': hinge.get('fold'),
                    'abs_delta': hinge.get('abs_delta'),
                    'delta_from_mean': hinge.get('delta_from_mean'),
                    'loo_probability': hinge.get('loo_probability'),
                    'influence': hinge.get('influence'),
                    'effect_when_present': hinge.get('effect_when_present'),
                    'omitted_start': hinge.get('omitted_start'),
                    'omitted_end': hinge.get('omitted_end'),
                    'omitted_chars': hinge.get('omitted_chars'),
                }
        if action_best:
            row['jackknife_hinging'] = action_best
            if global_best is None or action_best['abs_delta'] > global_best['abs_delta']:
                global_best = {'action_id': row.get('id'), 'label': row.get('label'), **action_best}
        else:
            row.pop('jackknife_hinging', None)
    flags = topography.setdefault('flags', {})
    if global_best:
        flags['jackknife_hinging'] = global_best
        notices = topography.setdefault('notices', [])
        note = ('Most hinging packet tenth: %s %s fold %s (Δ=%.3f). Omitting that slice moved the leave-one-out estimate the most.'
                % (global_best.get('action_id'), global_best.get('axis'), global_best.get('fold'), global_best.get('abs_delta') or 0))
        notices[:] = [item for item in notices if not str(item).startswith('Most hinging packet tenth:')]
        notices.append(note)
    else:
        flags.pop('jackknife_hinging', None)
    return topography


def enrich_jackknife_report(report, root):
    """Attach omitted tenths and hinging ranks to an existing jackknife topography."""
    trace = report if report.get('cycles') else native_trace_from_report(report, root)
    if trace is None:
        from export_parliament_viewer import ident_fallback_trace
        trace = ident_fallback_trace(report)
    if trace is None:
        raise ValueError('No framework trace is available to rebuild the Parliament packet.')
    world = report.get('world') or (trace.get('action_source_grounding') or {}).get('world_model') or {}
    packet = parliament_packet(report.get('scenario') or report.get('source'), world, trace, report.get('frameworks'))
    attach_jackknife_diagnostics(report.get('topography') or {}, packet)
    return report


def _axis_span(point):
    if not point or not point.get('present'):
        return None
    interval = point.get('interval') or {}
    if isinstance(interval.get('low'), (int, float)) and isinstance(interval.get('high'), (int, float)):
        return float(interval['low']), float(interval['high'])
    probability = point.get('probability')
    if isinstance(probability, (int, float)):
        return float(probability), float(probability)
    return None


def unique_max_interval(actions, rows, axis):
    """Unique maximum only when one interval sits entirely above every other."""
    scored = []
    for action in actions:
        row = next((item for item in rows if item['id'] == action['id']), None)
        span = _axis_span((row or {}).get(axis))
        if span:
            scored.append((action['id'], span[0], span[1]))
    winner = None
    for ident, low, _high in scored:
        if all(ident == other or low > other_high for other, _other_low, other_high in scored):
            if winner is not None:
                return None
            winner = ident
    return winner


def questions_for(action):
    return [{'type': 'predicate', 'name': '%s_%s' % (action['id'], axis),
             'instructions': 'Action %s (%s). %s' % (action['id'], action['label'], QUESTIONS[axis])}
            for axis in AXES]


def parse_answers(action, answers):
    points = {}
    by_name = {row.get('name'): row for row in answers or [] if isinstance(row, dict)}
    for axis in AXES:
        name = '%s_%s' % (action['id'], axis)
        row = by_name.get(name)
        if not row or row.get('type') == 'refusal':
            reason = 'Decisions API refused this question.' if row and row.get('type') == 'refusal' else 'No Decisions probability was returned.'
            points[axis] = {'present': False, 'omit_reason': reason, 'source': 'decisions/v1'}
            continue
        probability = row.get('probability')
        if not isinstance(probability, (int, float)) or probability < 0 or probability > 1:
            points[axis] = {'present': False, 'omit_reason': 'Decisions answer was not a probability in [0, 1].',
                            'source': 'decisions/v1'}
            continue
        points[axis] = {
            'present': True,
            'probability': float(probability),
            'encoding': 'probability',
            'question': QUESTIONS[axis],
            'source': 'decisions/v1',
            'model': DECISIONS_MODEL,
        }
    return points


def _flags(actions, rows):
    def getter(axis):
        def value(ident):
            point = next((row[axis] for row in rows if row['id'] == ident), None)
            if point and point.get('present') and isinstance(point.get('probability'), (int, float)):
                return point['probability']
            return None
        return value

    has_interval = any((row.get(axis) or {}).get('interval') and row[axis].get('present')
                       for row in rows for axis in AXES)
    if has_interval:
        maxima = {axis: unique_max_interval(actions, rows, axis) for axis in AXES}
    else:
        maxima = {axis: unique_max(actions, getter(axis)) for axis in AXES}
    care_rank = ranking(actions, getter('care'))
    welfare_rank = ranking(actions, getter('welfare'))
    degenerate, notices = [], []
    if care_rank and care_rank == welfare_rank and len(care_rank) >= 2:
        degenerate.append('care_welfare_probability_copy')
        notices.append('Care and welfare probability rankings coincide; this is marked degenerate, not an attractor.')
    if has_interval:
        notices.append('Axis maxima use non-overlapping jackknife intervals; overlapping intervals do not yield a unique maximum.')
    counted, attractors, conflicts = {}, [], []
    for axis, ident in maxima.items():
        if ident:
            if 'care_welfare_probability_copy' in degenerate and axis in ('care', 'welfare'):
                counted.setdefault(ident, set()).add('care_welfare')
            else:
                counted.setdefault(ident, set()).add(axis)
    for ident, axes in counted.items():
        if len(axes) >= 2:
            attractors.append(ident)
    present = {axis: ident for axis, ident in maxima.items() if ident}
    names = list(present)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if present[left] != present[right]:
                if 'care_welfare_probability_copy' in degenerate and {left, right} == {'care', 'welfare'}:
                    continue
                conflicts.append({'axes': [left, right], 'maxima': {left: present[left], right: present[right]}})
    holes = [axis for axis in AXES if not any(row[axis].get('present') for row in rows)]
    return {'degenerate': degenerate, 'attractors': attractors, 'conflicts': conflicts,
            'holes': holes, 'maxima': maxima}, notices


def topography_from_decisions(actions, per_action_points, native=None, packet=None):
    """Assemble plotted probabilities. Native ordinals stay nested for inspection."""
    if not actions:
        return unavailable('No admitted actions were recorded.')
    native_rows = {row['id']: row for row in (native or {}).get('actions') or []}
    rows = []
    for action in actions:
        points = per_action_points.get(action['id']) or {}
        row = {'id': action['id'], 'label': action['label']}
        for axis in AXES:
            row[axis] = points.get(axis) or {
                'present': False, 'omit_reason': 'No Decisions probability was returned.', 'source': 'decisions/v1'}
        prior = native_rows.get(action['id'])
        if prior:
            row['native'] = {axis: prior.get(axis) for axis in AXES}
        rows.append(row)
    if not any(row[axis].get('present') for row in rows for axis in AXES):
        return unavailable('Decisions returned no usable axis probabilities.')
    flags, notices = _flags(actions, rows)
    jackknife = any((row.get(axis) or {}).get('interval', {}).get('method') == 'jackknife'
                    for row in rows for axis in AXES)
    notices.insert(0, 'Plotted axes are Decisions probabilities over the Parliament packet, not ledger ordinals.')
    available = {axis: any(row[axis].get('present') for row in rows) for axis in AXES}
    folds = 0
    for row in rows:
        for axis in AXES:
            interval = (row.get(axis) or {}).get('interval') or {}
            replicates = interval.get('replicates')
            if replicates:
                folds = max(folds, len(replicates))
    calls = len(actions) * folds if folds else len(actions)
    axis_extra = {'interval': 'jackknife', 'interval_level': JACKKNIFE_LEVEL,
                  'ablation': 'contiguous_tenth_of_packet_text'} if jackknife else {}
    result = {
        'version': TOPOGRAPHY_VERSION,
        'status': 'RECORDED',
        'unavailable_reason': None,
        'caption': JACKKNIFE_CAPTION if jackknife else CAPTION,
        'axes': {axis: {
            'available': available[axis],
            'encoding': 'probability',
            'source': 'decisions/v1',
            'question': QUESTIONS[axis],
            **axis_extra,
            **({} if available[axis] else {'omit_reason': 'No Decisions probability was recorded on this axis.'}),
        } for axis in AXES},
        'actions': rows,
        'framework_arrows': (native or {}).get('framework_arrows') or [],
        'flags': flags,
        'notices': notices,
        'provenance': {
            'origin': 'decisions/v1',
            'model': DECISIONS_MODEL,
            'new_model_calls': calls,
            **({'jackknife': {'folds': folds or JACKKNIFE_FOLDS, 'min_present': JACKKNIFE_MIN_PRESENT,
                              'level': JACKKNIFE_LEVEL, 'ablation': 'contiguous_tenth_of_packet_text'}}
               if jackknife else {}),
        },
    }
    if jackknife:
        attach_jackknife_diagnostics(result, packet)
    return result


def create_decision(api_key, payload, opener=None):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        DECISIONS_URL, data=body, method='POST',
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
    opener = opener or urllib.request.urlopen
    try:
        with opener(request, timeout=120) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:500]
        raise RuntimeError('Decisions HTTP %s: %s' % (exc.code, detail)) from exc


def evaluate_actions(packet, actions, api_key, opener=None, jackknife_folds=0):
    per_action = {}
    body = packet_body(packet)
    folds = jackknife_folds if jackknife_folds else 0
    for action in actions:
        if not folds:
            payload = {
                'model': DECISIONS_MODEL,
                'input': packet_text(packet, action, body=body),
                'questions': questions_for(action),
            }
            result = create_decision(api_key, payload, opener=opener)
            per_action[action['id']] = parse_answers(action, result.get('answers'))
            continue
        fold_rows = []
        for fold in range(folds):
            ablated, meta = ablate_packet_text(body, fold, folds)
            payload = {
                'model': DECISIONS_MODEL,
                'input': packet_text(packet, action, body=ablated),
                'questions': questions_for(action),
            }
            try:
                result = create_decision(api_key, payload, opener=opener)
                parsed = parse_answers(action, result.get('answers'))
            except RuntimeError as exc:
                parsed = {axis: {'present': False, 'omit_reason': str(exc)[:240], 'source': 'decisions/v1'}
                          for axis in AXES}
            fold_rows.append((meta, parsed))
        per_action[action['id']] = combine_jackknife_points(action, fold_rows, folds)
    return per_action


def run_for_report(report, trace, api_key, opener=None, jackknife_folds=0):
    """Replace plotted topography with Decisions probabilities. Native records stay nested."""
    if report.get('status') == 'INTERPRETIVE_BEAM':
        return unavailable('No unique frozen world was supplied to agents; no framework coordinates exist.')
    world = report.get('world') or (trace.get('action_source_grounding') or {}).get('world_model') or {}
    actions = action_catalog(world, last_candidates(trace))
    native = topography_from_trace(trace, world, report.get('frameworks') or [])
    packet = parliament_packet(report.get('scenario') or report.get('source'), world, trace, report.get('frameworks'))
    points = evaluate_actions(packet, actions, api_key, opener=opener, jackknife_folds=jackknife_folds)
    return topography_from_decisions(actions, points, native=native, packet=packet)


def apply_to_report(report, root, api_key, opener=None, jackknife_folds=0):
    trace = report if report.get('cycles') else native_trace_from_report(report, root)
    if trace is None:
        from export_parliament_viewer import ident_fallback_trace
        trace = ident_fallback_trace(report)
    if trace is None:
        raise ValueError('No framework trace is available for Decisions topography.')
    report = dict(report)
    report['topography'] = run_for_report(report, trace, api_key, opener=opener, jackknife_folds=jackknife_folds)
    return report
