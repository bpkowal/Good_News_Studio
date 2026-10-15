"""Readable projections of recorded arguments, without changing their authority.

This module supplies presentation context, not replacement framework judgments.
Raw records remain available even when a malformed text field is not displayed.
"""
import re


def explanation_basis_sha256(report, view):
    """Bind an editorial explanation to its exact saved reasoning and decision."""
    import hashlib
    import json
    basis = {key: report.get(key) for key in ('scenario', 'world')}
    basis['view'] = {key: view.get(key) for key in
                     ('framework', 'position', 'preference', 'ledger_status',
                      'recorded_argument', 'live_objections', 'decision_boundaries')}
    return hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def apply_reviewed_explanations(report, views, catalog_path=None):
    """Present reviewed saved arguments; never apply them to a changed run."""
    import json
    from pathlib import Path
    path = Path(catalog_path) if catalog_path else Path(__file__).parent/'frontend/reviewed_explanations.json'
    if not path.is_file():
        return views
    catalog = json.loads(path.read_text())
    if catalog.get('version') != 'parliament-reviewed-explanations/1':
        raise ValueError('Unknown reviewed explanation version')
    by_basis = {row['basis_sha256']: row for row in catalog['explanations']}
    for view in views:
        row = by_basis.get(explanation_basis_sha256(report, view))
        if not row:
            continue
        view['original_presentation'] = {key: view.get(key) for key in
                                        ('reason', 'reason_authority', 'live_objections', 'reversal_explanations')}
        view['reason'] = row['reason']
        view['reason_authority'] = 'REVIEWED_PLAIN_LANGUAGE_SUMMARY_OF_SAVED_ARGUMENT'
        view['display_uncertainties'] = row['uncertainties']
        view['display_reversals'] = row.get('reversals', view.get('reversal_explanations', []))
        view['explanation_provenance'] = {key: row[key] for key in
                                         ('basis_sha256', 'support_fields', 'reviewer')}
    return views


def preserve_model_arguments(candidates, folder):
    """Keep pre-normalization prose alongside native decisions, not over them."""
    import json
    from pathlib import Path
    fields = {'w':'rationale', 'fa':'framework_application', 'dr':'decision_rule',
              'l':'landscape_cases', 'ft':'factual_reversal_threshold', 'nt':'normative_reversal_threshold'}
    for candidate in candidates:
        name = candidate.get('specialist', '')
        if not re.fullmatch(r'[a-z_]+', name):
            continue
        path = Path(folder)/name/'model_calls.json'
        if not path.is_file():
            continue
        for call in reversed(json.loads(path.read_text())):
            try:
                response = call['response']
                native = json.loads(response['choices'][0]['text'])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if not isinstance(native, dict) or not any(key in native for key in fields):
                continue
            candidate['model_argument_fields'] = {long:native[short] for short,long in fields.items() if short in native}
            break


def readable_prose(value):
    if not isinstance(value, str) or value.strip().upper() in {'', 'NONE', 'UNKNOWN'}:
        return False
    # Escaped JSON accidentally emitted inside a string is data corruption,
    # not an explanation. Ordinary quoted prose and mathematical comparisons
    # remain legal when accompanied by words.
    if re.search(r'"\s*:\s*(?:true|false|null|[\[{\"]|\d)', value.replace('\\"', '"')):
        return False
    if value.lstrip().startswith(('{', '[')):
        return False
    words = re.findall(r'[A-Za-z]{2,}', value)
    return bool(words) and not all(w.isupper() for w in words)


def incomplete_prose(value):
    """Detect evident dangling endings; never invent their missing continuation."""
    tail = str(value).strip().rstrip(';,.').lower()
    return tail.endswith((' so', ' but', ' but only', ' rather', ' and', ' or',
                          ' the', ' a', ' to', ' at', ' toward', ' with', ' because', '-'))


def action_labels(world):
    return {a['action_id']: a['intervention'] for a in world.get('actions', [])}


def expand_actions(text, world):
    labels = action_labels(world)
    return re.sub(r'\bA\d+\b', lambda m: '“'+labels[m[0]]+'”' if m[0] in labels else m[0], text)


def duty_explanation(candidate, world):
    ledger = candidate.get('committed_native_ledger') or {}
    records = ledger.get('records') or []
    if ledger.get('ledger_kind') != 'DEONTOLOGICAL_DUTY_LEDGER' or not records:
        return ''
    labels = action_labels(world)
    names = {p['label'].casefold():p['label'] for p in world.get('parties', [])}
    parts = []
    seen = set()
    for row in records:
        bearer = names.get(str(row.get('duty_bearer', '')).casefold(), row.get('duty_bearer')) or 'The recorded duty-bearer'
        norm, protected = row.get('norm'), row.get('protected_party')
        if norm and protected and (bearer, norm, protected) not in seen:
            parts.append(f'The agent assesses {bearer}’s duty “{norm}” toward {protected}. '
                         f'{bearer} is the duty-bearer; {protected} is the protected party.')
            seen.add((bearer, norm, protected))
            if row.get('competing_norm') and row.get('competing_protected_party'):
                parts.append(f'The competing obligation being assessed is {bearer}’s “{row["competing_norm"]}” '
                             f'toward {row["competing_protected_party"]}.')
        action = labels.get(row.get('canonical_action_id'), row.get('canonical_action_id', 'the action'))
        verdict = str(row.get('verdict', 'unresolved')).lower().replace('_', ' ')
        reason = row.get('reason', '')
        maxim = row.get('candidate_maxim')
        if isinstance(maxim, dict):
            clean = lambda key: str(maxim.get(key, 'unresolved')).rstrip('. ')
            parts.append(f'For “{action}”, the agent proposes the maxim “{clean("policy")}”. '
                         'This is an interpretation of possible practical reasoning, not an established intention. '
                         f'Its purpose is {clean("end")}; its chosen means are {clean("means")}. '
                         f'The proposed justification is: {clean("universalization")}. '
                         f'The agent assesses the maxim as {str(maxim.get("maxim_assessment", "unresolved")).lower()}, '
                         f'and the action as {str(maxim.get("action_permissibility", "unresolved")).lower()}.')
            if maxim.get('practical_reasoning'):
                parts.append(f'Its proposed means–ends reasoning is: {clean("practical_reasoning")}. '
                             f'Physical means references: {", ".join(maxim.get("means_effect_ids") or []) or "none"}; '
                             f'end references: {", ".join(maxim.get("end_effect_ids") or []) or "none"}.')
        if row.get('proposed_verdict') and row['proposed_verdict'] != row.get('verdict'):
            parts.append(f'The agent proposed {row["proposed_verdict"].lower()} for “{action}”; '
                         'the following ledger assessment retains the native qualification.')
        parts.append(f'For {bearer}’s option “{action}”, the ledger records {verdict}' +
                     (f'. The agent’s stated reason is: {reason}.' if readable_prose(reason) else '.'))
    if ledger.get('transaction_status') == 'COMMITTED_WITH_UNCERTAINTY':
        parts.append('The ledger retains unresolved issues; this does not establish that one duty overrides the other.')
    return ' '.join(parts)


def assumption_records(candidate):
    rows = []
    value = candidate.get('unsupported_assumption')
    if readable_prose(value):
        rows.append({'kind':'MODEL_REPORTED_ASSUMPTION', 'text':value, 'record':'unsupported_assumption'})
    for key in ('decision_rule', 'framework_numerical_justification'):
        value = (candidate.get('model_argument_fields') or {}).get(key, candidate.get(key))
        if readable_prose(value):
            rows.append({'kind':'RECORDED_INTERPRETIVE_COMMITMENT', 'text':value, 'record':key})
    # This is the experiment's factual premise, explicitly distinguished from
    # a model's own assertion or a new empirical claim.
    rows.append({'kind':'RUN_ASSUMPTION', 'record':'source_scenario',
        'text':'The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.'})
    return rows


def source_basis(world):
    counts = [p['label'] for p in world.get('parties', []) if p.get('kind') in {'PERSON','HUMAN_GROUP'}
              and (p.get('quantities') or re.search(r'\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b',p['label'],re.I))]
    statement = ('The counts (' + '; '.join(counts) + ') are stipulated by the scenario for this run. '
                 if counts else 'The scenario’s stated facts are treated as givens for this run. ')
    return {'status':'STIPULATED_FOR_THIS_RUN', 'how_known':'NOT_SEPARATELY_RECORDED',
        'text': statement + 'No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.'}


def enrich_view(view, candidate, world):
    """Add human-readable context, retaining original records separately."""
    view['recorded_argument'] = {key:candidate.get(key) for key in (
        'rationale','framework_application','decision_rule','landscape_cases',
        'proposed_framework_state','framework_vote_reason','investigative_claim',
        'committed_native_ledger','unsupported_assumption','normative_reversal_threshold',
        'factual_reversal_threshold','model_argument_fields')}
    explanation = duty_explanation(candidate, world)
    if explanation:
        view['reason'] = explanation
        view['reason_authority'] = 'READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER'
    else:
        view['reason_authority'] = 'RECORDED_ARGUMENT_FIELDS'
    if not readable_prose(view['reason']):
        view['reason'] = 'No readable main explanation was recorded. The saved native record remains available for inspection.'
        view['reason_authority'] = 'MISSING_READABLE_EXPLANATION'
    view['reason'] = expand_actions(view['reason'], world)
    view['assumption_records'] = assumption_records(candidate)
    for row in view['assumption_records']:
        row['text'] = expand_actions(row['text'], world)
    view['source_basis'] = source_basis(world)
    view['presentation_notices'] = []
    if any(v and not readable_prose(v) for v in (candidate.get('framework_application'),)
           if str(v).strip().upper() not in {'NONE','UNKNOWN'}):
        view['presentation_notices'].append('A malformed explanation field was excluded from the main reason and preserved in the full record.')
    cases = (candidate.get('model_argument_fields') or {}).get('landscape_cases', candidate.get('landscape_cases')) or {}
    if isinstance(cases, dict) and any(incomplete_prose(v) for v in cases.values()):
        view['presentation_notices'].append('An incomplete recorded comparison was omitted from the main reason. Its original fragment is preserved in the full record; the missing words cannot be recovered.')
    labels = list(action_labels(world).values())
    preference = view.get('preference')
    preference = action_labels(world).get(preference, preference)
    alternatives = [label for label in labels if label != preference]
    lead = ('The agent reports a switch from “'+preference+'” to “'+alternatives[0]+'”. '
            if preference and preference in labels and len(alternatives) == 1 else
            'The agent reports reconsidering its preference'+(' for “'+preference+'”' if preference else '')+'. ')
    view['reversal_explanations'] = [lead + 'Condition: '+expand_actions(text, world).rstrip(' .')+'.'+
        ' This is an agent-reported condition, not an independently established threshold.'
        for text in view.get('reversal_conditions', []) if readable_prose(text)]
    return view


def argument_markdown(view):
    import json
    lines = ['# '+view['framework'].capitalize()+' — recorded argument', '',
        'Participation: '+view['position']+'. Native ledger: '+view['ledger_status']+'.', '',
        '## Readable explanation', '', view['reason'], '',
        'Presentation basis: '+view.get('reason_authority','RECORDED_ARGUMENT_FIELDS')+'.', '',
        '## Assumptions and interpretive commitments', '']
    lines += ['- '+r['kind'].replace('_',' ').capitalize()+': '+r['text'] for r in view.get('assumption_records', [])]
    lines += ['', '## Source and uncertainty', '', view.get('source_basis',{}).get('text','')]
    lines += ['- '+t for t in view.get('display_uncertainties', view.get('live_objections', []))]
    if view.get('original_presentation'):
        lines += ['', '## Original presentation and objections', '', '```json',
                  json.dumps(view['original_presentation'], ensure_ascii=False, indent=2), '```',
                  '', '## Explanation provenance', '', '```json',
                  json.dumps(view['explanation_provenance'], ensure_ascii=False, indent=2), '```']
    lines += ['', '## Conditions for reconsidering', ''] + ['- '+t for t in view.get('display_reversals', view.get('reversal_explanations', []))]
    lines += ['', '## Original recorded fields', '', 'These fields are preserved verbatim as data. They can include incomplete or malformed model text.', '',
        '```json', json.dumps(view.get('recorded_argument',{}),ensure_ascii=False,indent=2), '```', '']
    return '\n'.join(lines)
