"""Burden-preserving comparison scaffolding, not a moral ranking algorithm.

Extends the person/value reason structure credited in CONTRACTUALIST_RESEARCH.md.
The burden matrix and scoped comparisons are our experimental additions. Model
annotations remain fallible; no preference, factual negation or defeat is inferred.
"""
from copy import deepcopy

VERSION = 'complaint-comparison/0.1'

INSTRUCTION = (
    'Build the comparison before deciding. First list relevant individual burdens, '
    'including loss of an available benefit, using stable burden IDs and person IDs. '
    'For each burden record its status under EVERY option as PRESENT, ABSENT or UNKNOWN '
    'with factual ground_handles and a short position description preserving quantities, '
    'risk and scope. PRESENT in both options does not mean equal magnitude: a death risk '
    'can remain while changing from certainty to a stated probability. These are '
    'counterfactual positions under the options, '
    'not claims that the actions occurred. Unknown severity or risk must stay unknown; '
    'missing information is not evidence that a burden is minor or absent. Keep consent '
    'and assurances distinct from physical outcomes: consent can change a complaint\'s '
    'warrant without removing an economic or physical burden. '
    'Then compare two explicit options at a time. from_action_id and to_action_id orient '
    'relieved_burden_ids (PRESENT to ABSENT), introduced_burden_ids (ABSENT to PRESENT) '
    'and retained_burden_ids (PRESENT in both). An UNKNOWN cell does not license a claim '
    'of established relief. Include every relevant retained burden in that pair even '
    'when it supports neither side. Compare the actual alternatives rather than an '
    'imaginary option that combines their benefits. Reasons for rejecting a third '
    'option do not by themselves settle this pair. Never count complaints or require '
    'a single scalar scale. A tied comparison differs from unresolved or incomparable '
    'claims. State the priority warrant or the precise premise left unresolved. '
    'Finally distinguish UNIQUE_DEFENSIBLE, PROVISIONAL_PREFERENCE, MULTIPLE_DEFENSIBLE '
    'and UNRESOLVED. A provisional recommendation may coexist with an open comparison. '
    'Unique defensibility needs a separately defended comparison against every '
    'alternative; unresolved comparisons cannot silently become favorable verdicts. '
    'Use recommendation_comparison_ids to identify that reasoning and '
    'remaining_complaint_ids to keep the strongest live objections in view. A person\'s '
    'burden can remain even if an argued answer limits their objection\'s moral force. '
)


def schema_fields(object_schema):
    text = {'type': 'string', 'minLength': 1}
    strings = {'type': 'array', 'items': text}
    burden = object_schema({'id': text, 'person_id': text, 'description': text})
    position = object_schema({'burden_id': text, 'action_id': text,
        'status': {'type': 'string', 'enum': ['PRESENT', 'ABSENT', 'UNKNOWN']},
        'position': text, 'ground_handles': strings})
    return {
        'burdens': {'type': 'array', 'items': burden, 'maxItems': 12},
        'burden_positions': {'type': 'array', 'items': position, 'maxItems': 36},
        'recommendation_status': {'type': 'string', 'enum': ['UNIQUE_DEFENSIBLE',
            'PROVISIONAL_PREFERENCE', 'MULTIPLE_DEFENSIBLE', 'UNRESOLVED']},
        'recommendation_comparison_ids': strings, 'remaining_complaint_ids': strings}


def comparison_fields():
    text = {'type': 'string', 'minLength': 1}
    strings = {'type': 'array', 'items': text}
    return {'from_action_id': text, 'to_action_id': text,
        'status': {'type': 'string', 'enum': ['FAVORS_FROM', 'FAVORS_TO', 'TIED',
                                             'INCOMPARABLE', 'UNRESOLVED']},
        'relieved_burden_ids': strings, 'introduced_burden_ids': strings,
        'retained_burden_ids': strings}


def review_comparisons(raw, actions, handles):
    """Check explicit annotations. Cannot verify the matrix or prose against reality."""
    findings = []
    def note(code, record, **detail):
        findings.append({'code': code, 'record_id': record, **detail})
    burdens = {b['id']: b for b in raw['burdens']}
    persons = {p['id'] for p in raw['persons']}
    reasons = {r['id']: r for r in raw['reasons']}
    comparisons = {c['id']: c for c in raw['comparisons']}
    if len(burdens) != len(raw['burdens']):
        note('DUPLICATE_BURDEN', 'burdens')
    positions = {}
    for cell in raw['burden_positions']:
        key = (cell['burden_id'], cell['action_id'])
        if key in positions:
            note('DUPLICATE_POSITION', str(key))
        positions[key] = cell['status']
        if cell['burden_id'] not in burdens or cell['action_id'] not in actions:
            note('UNKNOWN_POSITION_REFERENCE', str(key))
        if any(h not in handles for h in cell['ground_handles']):
            note('POSITION_GROUND_MISSING', str(key))
        if cell['status'] != 'UNKNOWN' and not cell['ground_handles']:
            note('POSITION_GROUND_MISSING', str(key))
    for bid, burden in burdens.items():
        if burden['person_id'] not in persons:
            note('UNKNOWN_BURDEN_PERSON', bid)
        for action in actions:
            if (bid, action) not in positions:
                note('MISSING_POSITION', bid, action_id=action)
    if not burdens:
        note('BURDEN_INVENTORY_EMPTY', 'burdens')
    for comparison in raw['comparisons']:
        cid = comparison['id']
        start, end = comparison['from_action_id'], comparison['to_action_id']
        if start == end or start not in actions or end not in actions:
            note('INVALID_COMPARISON_PAIR', cid)
        expected = {'relieved_burden_ids': ('PRESENT', 'ABSENT'),
                    'introduced_burden_ids': ('ABSENT', 'PRESENT'),
                    'retained_burden_ids': ('PRESENT', 'PRESENT')}
        for field, transition in expected.items():
            for bid in comparison[field]:
                observed = (positions.get((bid, start)), positions.get((bid, end)))
                if bid not in burdens or observed != transition:
                    note('BURDEN_TRANSITION_MISMATCH', cid, burden_id=bid,
                         claim=field, observed=list(observed), expected=list(transition))
        for bid in burdens:
            if (positions.get((bid, start)), positions.get((bid, end))) == ('PRESENT', 'PRESENT'):
                if bid not in comparison['retained_burden_ids']:
                    note('RETAINED_BURDEN_OMITTED', cid, burden_id=bid)
        for rid in comparison['reason_ids']:
            if rid in reasons and reasons[rid]['action_id'] not in (start, end):
                note('REASON_OUTSIDE_COMPARISON_PAIR', cid, reason_id=rid)
    for rid in raw['remaining_complaint_ids']:
        if rid not in reasons:
            note('UNKNOWN_REMAINING_COMPLAINT', rid)
    selected = raw['recommendation_comparison_ids']
    for cid in selected:
        if cid not in comparisons:
            note('UNKNOWN_RECOMMENDATION_COMPARISON', cid)
    recommendation = raw['recommendation_action_id']
    status = raw['recommendation_status']
    if status in ('UNIQUE_DEFENSIBLE', 'PROVISIONAL_PREFERENCE') and recommendation is None:
        note('RECOMMENDATION_MISSING', status)
    if status in ('MULTIPLE_DEFENSIBLE', 'UNRESOLVED') and recommendation is not None:
        note('RECOMMENDATION_STATUS_MISMATCH', status)
    if status == 'UNIQUE_DEFENSIBLE':
        for other in actions - {recommendation}:
            linked = [comparisons[cid] for cid in selected if cid in comparisons and
                      {comparisons[cid]['from_action_id'], comparisons[cid]['to_action_id']} == {recommendation, other}]
            favorable = any((c['from_action_id'] == recommendation and c['status'] == 'FAVORS_FROM') or
                            (c['to_action_id'] == recommendation and c['status'] == 'FAVORS_TO') for c in linked)
            if not favorable:
                note('UNIQUENESS_COMPARISON_NOT_ESTABLISHED', str(other))
            if any(c['status'] in ('UNRESOLVED', 'INCOMPARABLE', 'TIED') for c in linked):
                note('OPEN_COMPARISON_IN_UNIQUENESS_CLAIM', str(other))
    return {'version': VERSION, 'findings': findings,
            'status': 'REVIEW_REQUIRED' if findings else 'NO_STRUCTURAL_MISMATCH_FOUND',
            'matrix_authority': 'MODEL_PROPOSED_NOT_SOURCE_VERIFIED',
            'semantic_support': 'NOT_INDEPENDENTLY_ASSESSED',
            'moral_ranking_computed': False, 'operative': False,
            'recommendation_status': status,
            'remaining_complaint_ids': deepcopy(raw['remaining_complaint_ids'])}
