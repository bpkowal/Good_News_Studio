"""Native Care × welfare × obligation coordinates for the saved-run viewer.

Coordinates are copied from committed ledgers and arithmetic-verified expected
value records. Missing axes stay missing. UNKNOWN is never written as 0.
Attractors are occupancy of recorded points, not interpolated terrain.
"""

CARE_ORDINAL = {'NEGLECTFUL': 0, 'MIXED': 1, 'RESPONSIVE': 2}
OBLIGATION_ORDINAL = {'PROHIBITED': 0, 'CONFLICTED': 1, 'PERMISSIBLE': 2, 'REQUIRED': 3}
CAPTION = (
    'Survivor arithmetic does not settle total welfare. Coordinates are native '
    'ledger records, not a calibrated moral physics. Attractors are occupancy of '
    'recorded points, not interpolated terrain.'
)


def unavailable(reason):
    return {
        'version': 'parliament-topography/1',
        'status': 'UNAVAILABLE',
        'unavailable_reason': reason,
        'caption': CAPTION,
        'axes': {
            'care': {'available': False, 'encoding': 'ordinal', 'omit_reason': reason},
            'welfare': {'available': False, 'encoding': 'numeric', 'omit_reason': reason},
            'obligation': {'available': False, 'encoding': 'ordinal', 'omit_reason': reason},
        },
        'actions': [],
        'framework_arrows': [],
        'flags': {'degenerate': [], 'attractors': [], 'conflicts': [], 'holes': ['care', 'welfare', 'obligation']},
        'notices': [reason],
    }


def last_candidates(trace):
    rows = (trace or {}).get('cycles') or []
    return list(rows[-1].get('candidates') or []) if rows else []


def specialist(candidates, name):
    for row in candidates:
        if row.get('specialist') == name:
            return row
    return None


def committed(ledger, kind):
    if not isinstance(ledger, dict) or not ledger:
        return False
    status = str(ledger.get('transaction_status') or '')
    return ledger.get('ledger_kind') == kind and status.startswith('COMMITTED')


def action_catalog(world, candidates):
    actions = []
    seen = set()
    for row in (world or {}).get('actions') or []:
        ident = row.get('action_id')
        label = row.get('intervention') or ident
        if ident and ident not in seen:
            actions.append({'id': ident, 'label': label})
            seen.add(ident)
    if actions:
        return actions
    labels = []
    for row in candidates:
        for key in row.get('action_scores') or {}:
            if key not in labels:
                labels.append(key)
        for key in row.get('expected_value_estimates') or {}:
            if key not in labels:
                labels.append(key)
    return [{'id': 'A%s' % i, 'label': label} for i, label in enumerate(labels)]


def match_action(actions, key):
    if key is None or key in ('', 'NONE', 'UNKNOWN'):
        return None
    for row in actions:
        if key in (row['id'], row['label']):
            return row
    return None


def care_axis(candidate, actions):
    omit = 'No CARE_RELATIONSHIP_LEDGER commit.'
    if not committed(candidate.get('committed_native_ledger') if candidate else None, 'CARE_RELATIONSHIP_LEDGER'):
        return False, omit, {}
    points = {}
    for record in candidate['committed_native_ledger'].get('records') or []:
        action = match_action(actions, record.get('canonical_action_id')) or match_action(actions, record.get('action_id'))
        if not action:
            continue
        verdict = record.get('verdict')
        if verdict not in CARE_ORDINAL:
            continue
        current = points.get(action['id'])
        if current and current['verdict'] != verdict:
            verdict = 'MIXED'
        points[action['id']] = {
            'present': True,
            'ordinal': CARE_ORDINAL[verdict],
            'verdict': verdict,
            'reason': record.get('reason') or '',
            'affected_party': record.get('affected_party') or '',
        }
    scores = candidate.get('action_scores') or {}
    for action in actions:
        score = scores.get(action['label'], scores.get(action['id']))
        if action['id'] in points and isinstance(score, (int, float)):
            points[action['id']]['score'] = score
    if not points:
        return False, 'Care ledger committed without per-action verdicts.', {}
    return True, None, points


def welfare_axis(candidate, actions):
    omit = 'No arithmetic-verified utilitarian expected-value record.'
    if not candidate:
        return False, omit, {}
    if str(candidate.get('expected_value_validation_status') or '') != 'ARITHMETIC_VERIFIED':
        return False, omit, {}
    points = {}
    for key, estimate in (candidate.get('expected_value_estimates') or {}).items():
        if not isinstance(estimate, dict):
            continue
        if estimate.get('validation_status') != 'ARITHMETIC_VERIFIED':
            continue
        if estimate.get('unit') in (None, '', 'NONE'):
            continue
        if not isinstance(estimate.get('value'), (int, float)):
            continue
        action = match_action(actions, key)
        if not action:
            continue
        unknown_total = any(
            projection.get('welfare_magnitude') == 'UNKNOWN' or projection.get('duration') == 'UNKNOWN'
            for projection in estimate.get('count_projections') or []
        )
        if 'NOT_TOTAL_WELFARE' in str(estimate.get('valuation_scope') or '') or estimate.get('unit') == 'EXPECTED_SURVIVORS':
            unknown_total = True
        point = {
            'present': True,
            'value': estimate['value'],
            'unit': estimate.get('unit'),
            'direction': estimate.get('direction'),
            'arithmetic': 'ARITHMETIC_VERIFIED',
            'calculation': estimate.get('calculation') or '',
            'scope': estimate.get('valuation_scope') or 'RESTRICTED_SURVIVOR_ENDPOINT_NOT_TOTAL_WELFARE',
        }
        if unknown_total:
            point['total_welfare'] = 'UNKNOWN'
        points[action['id']] = point
    if not points:
        return False, omit, {}
    return True, None, points


def obligation_axis(candidate, actions):
    omit = 'Deontological ledger NONE.'
    if not committed(candidate.get('committed_native_ledger') if candidate else None, 'DEONTOLOGICAL_DUTY_LEDGER'):
        status = ((candidate or {}).get('committed_native_ledger') or {}).get('transaction_status') or 'NONE'
        if status == 'NONE' or not candidate or not (candidate.get('committed_native_ledger') or {}):
            omit = 'Deontological ledger NONE.'
        else:
            omit = 'No DEONTOLOGICAL_DUTY_LEDGER commit.'
        return False, omit, {}
    points = {}
    for record in candidate['committed_native_ledger'].get('records') or []:
        action = match_action(actions, record.get('canonical_action_id')) or match_action(actions, record.get('action_id'))
        if not action:
            continue
        verdict = record.get('verdict')
        if verdict not in OBLIGATION_ORDINAL:
            continue
        current = points.get(action['id'])
        if current and current['verdict'] != verdict:
            verdict = 'CONFLICTED'
        points[action['id']] = {
            'present': True,
            'ordinal': OBLIGATION_ORDINAL[verdict],
            'verdict': verdict,
            'unresolved': verdict == 'CONFLICTED',
            'reason': record.get('reason') or '',
            'norm': record.get('norm') or '',
        }
    if not points:
        return False, 'Duty ledger committed without per-action verdicts.', {}
    return True, None, points


def welfare_rank_value(point):
    value = point['value']
    if point.get('direction') == 'HARM' or point.get('unit') == 'DEATHS':
        return -value
    return value


def unique_max(actions, getter):
    scored = []
    for action in actions:
        value = getter(action['id'])
        if value is not None:
            scored.append((value, action['id']))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def ranking(actions, getter):
    scored = [(getter(action['id']), action['id']) for action in actions if getter(action['id']) is not None]
    scored.sort(reverse=True)
    return tuple(ident for _value, ident in scored)


def framework_arrows(frameworks, actions):
    arrows = []
    for view in frameworks or []:
        target = match_action(actions, view.get('preference'))
        arrows.append({
            'framework': view.get('framework'),
            'preference': view.get('preference'),
            'action_id': target['id'] if target else None,
            'position': view.get('position'),
            'ledger_status': view.get('ledger_status'),
        })
    return arrows


def topography_from_trace(trace, world=None, frameworks=None):
    """Build a topography object from a native framework trace. Do not invent values."""
    if (trace or {}).get('status') == 'INTERPRETIVE_BEAM':
        return unavailable('No unique frozen world was supplied to agents; no framework coordinates exist.')
    candidates = last_candidates(trace)
    if not candidates:
        return unavailable('No framework candidates were recorded; no axis coordinates exist.')
    actions = action_catalog(world, candidates)
    if not actions:
        return unavailable('No admitted actions were recorded.')
    care_ok, care_omit, care_points = care_axis(specialist(candidates, 'care'), actions)
    welfare_ok, welfare_omit, welfare_points = welfare_axis(specialist(candidates, 'utilitarian'), actions)
    duty_ok, duty_omit, duty_points = obligation_axis(specialist(candidates, 'deontological'), actions)
    notices = []
    if welfare_ok:
        notices.append('Plotted welfare is a restricted survivor or count endpoint. It is not total welfare.')
        if any(point.get('total_welfare') == 'UNKNOWN' for point in welfare_points.values()):
            notices.append('Utilitarian magnitude or duration remains UNKNOWN; that gap is not plotted as zero.')
    holes = [name for name, ok in (('care', care_ok), ('welfare', welfare_ok), ('obligation', duty_ok)) if not ok]
    if not (care_ok or welfare_ok or duty_ok):
        return unavailable('No committed care, welfare, or obligation records.')
    rows = []
    for action in actions:
        care = care_points.get(action['id']) or {'present': False, 'omit_reason': care_omit}
        welfare = welfare_points.get(action['id']) or {'present': False, 'omit_reason': welfare_omit}
        obligation = duty_points.get(action['id']) or {'present': False, 'omit_reason': duty_omit}
        if welfare.get('present') and 'value' not in welfare:
            welfare = {'present': False, 'omit_reason': 'Welfare record lacked a numeric coordinate.'}
        rows.append({'id': action['id'], 'label': action['label'], 'care': care, 'welfare': welfare, 'obligation': obligation})

    def care_value(ident):
        point = care_points.get(ident)
        return point['ordinal'] if point and point.get('present') else None

    def welfare_value(ident):
        point = welfare_points.get(ident)
        return welfare_rank_value(point) if point and point.get('present') else None

    def obligation_value(ident):
        point = duty_points.get(ident)
        if not point or not point.get('present') or point.get('unresolved') or point.get('verdict') == 'CONFLICTED':
            return None
        return point['ordinal']

    maxima = {
        'care': unique_max(actions, care_value) if care_ok else None,
        'welfare': unique_max(actions, welfare_value) if welfare_ok else None,
        'obligation': unique_max(actions, obligation_value) if duty_ok else None,
    }
    degenerate = []
    if care_ok and welfare_ok:
        care_rank = ranking(actions, care_value)
        welfare_rank = ranking(actions, welfare_value)
        unit = next((point.get('unit') for point in welfare_points.values()), '')
        if care_rank and care_rank == welfare_rank and len(care_rank) >= 2:
            if unit in ('EXPECTED_SURVIVORS', 'DEATHS') or unit:
                degenerate.append('care_welfare_survival_copy')
                notices.append('Care ranking coincides with the welfare/count ranking; this is marked degenerate, not an attractor.')
    attractors = []
    counted = {}
    for axis, ident in maxima.items():
        if not ident:
            continue
        if 'care_welfare_survival_copy' in degenerate and axis in ('care', 'welfare'):
            counted.setdefault(ident, set()).add('care_welfare')
        else:
            counted.setdefault(ident, set()).add(axis)
    for ident, axes in counted.items():
        if len(axes) >= 2:
            attractors.append(ident)
    conflicts = []
    present_maxima = {axis: ident for axis, ident in maxima.items() if ident}
    names = list(present_maxima)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if present_maxima[left] != present_maxima[right]:
                if 'care_welfare_survival_copy' in degenerate and {left, right} == {'care', 'welfare'}:
                    continue
                conflicts.append({'axes': [left, right], 'maxima': {left: present_maxima[left], right: present_maxima[right]}})
    arrows = framework_arrows(frameworks, actions)
    if not frameworks:
        arrows = framework_arrows([
            {'framework': row['specialist'], 'preference': row.get('recommended_action'),
             'position': row.get('framework_vote_status'), 'ledger_status': (row.get('committed_native_ledger') or {}).get('transaction_status')}
            for row in candidates if row.get('specialist')
        ], actions)
    return {
        'version': 'parliament-topography/1',
        'status': 'RECORDED',
        'unavailable_reason': None,
        'caption': CAPTION,
        'axes': {
            'care': {'available': care_ok, 'encoding': 'ordinal', 'levels': ['NEGLECTFUL', 'MIXED', 'RESPONSIVE'],
                     **({} if care_ok else {'omit_reason': care_omit})},
            'welfare': {'available': welfare_ok, 'encoding': 'numeric',
                        'unit': next((point.get('unit') for point in welfare_points.values()), None),
                        'scope': 'RESTRICTED_SURVIVOR_ENDPOINT_NOT_TOTAL_WELFARE',
                        **({} if welfare_ok else {'omit_reason': welfare_omit})},
            'obligation': {'available': duty_ok, 'encoding': 'ordinal',
                           'levels': ['PROHIBITED', 'CONFLICTED', 'PERMISSIBLE', 'REQUIRED'],
                           **({} if duty_ok else {'omit_reason': duty_omit})},
        },
        'actions': rows,
        'framework_arrows': arrows,
        'flags': {
            'degenerate': degenerate,
            'attractors': attractors,
            'conflicts': conflicts,
            'holes': holes,
            'maxima': maxima,
        },
        'notices': notices,
    }


def native_trace_from_report(report, root):
    """Load an on-disk framework trace when the demo report omitted cycles. Paths never enter the payload."""
    if report.get('status') == 'INTERPRETIVE_BEAM':
        return None
    from pathlib import Path
    root = Path(root)
    candidates = []
    for artifact in report.get('artifacts') or []:
        text = str(artifact)
        if text.endswith('framework_generation.json'):
            candidates.append(root / text)
        elif text.endswith('preparation_manifest.json'):
            path = root / text
            if path.is_file():
                import json
                manifest = json.loads(path.read_text())
                native = (manifest.get('deliberation') or {}).get('trace')
                if native:
                    candidates.append(root / native)
    origin = (report.get('execution') or {}).get('replay_origin')
    if origin:
        folder = root / origin
        candidates.extend([folder / 'framework_generation.json', folder / 'frameworks' / 'framework_generation.json'])
    for path in candidates:
        if path.is_file():
            import json
            trace = json.loads(path.read_text())
            scenario = report.get('scenario') or report.get('source')
            if scenario and trace.get('scenario') and trace['scenario'] != scenario:
                continue
            if last_candidates(trace):
                return trace
    return None
