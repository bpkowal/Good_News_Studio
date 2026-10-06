"""Isolate bounded native classification changes from a rejected duty bundle."""
from copy import deepcopy
import json

# These are native calibration issue kinds, not inferred normative priorities.
CLASSIFICATION_FIELDS = {
    'HARM_RELATION_GRAPH_MISALIGN': 'hr',
    'INTENDED_AS_MEANS_LACKS_PATH': 'mr',
}
PRIOR_NATIVE_FIELDS = {'v': 'verdict', 'rel': 'relation', 'gv': 'governing_norm',
                       'pb': 'priority_basis', 'res': 'resolution_status'}


def isolate_classifications(prior_response, proposed_response, job):
    """Isolate classifications, preserving committed duty conclusions on uncertainty."""
    if job['framework'] != 'deontological':
        return None, []
    try:
        prior = json.loads(prior_response['choices'][0]['text'])
        proposed = json.loads(proposed_response['choices'][0]['text'])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None, []
    patches = []
    for issue in job['issues']:
        for claim in job['claims']:
            if claim['id'] not in issue['claim_ids']:
                continue
            record = claim['record']
            action = record.get('canonical_action_id')
            for calibration in record.get('calibration_issues', []):
                field = CLASSIFICATION_FIELDS.get(calibration.get('kind'))
                if not field or calibration.get('message') != issue['label']:
                    continue
                old = prior.get('dp', {}).get(action, {}).get(field)
                new = proposed.get('dp', {}).get(action, {}).get(field)
                companion = None
                if field == 'hr':
                    # Only explicit uncertainty can be isolated. No positive
                    # harm attribution or inferred normative conclusion here.
                    proposed_resolution = proposed.get('dp', {}).get(action, {}).get('res')
                    if new != 'UNRESOLVED' or proposed_resolution not in {'CONTESTED', 'UNKNOWN'}:
                        continue
                    old_resolution = record.get('resolution_status')
                    if not isinstance(old_resolution, str):
                        continue
                    if old_resolution != proposed_resolution:
                        companion = {'action_id': action, 'field': 'res',
                            'before': old_resolution, 'after': proposed_resolution,
                            'issue_id': issue['id'], 'authority': 'EXPLICIT_PROPOSED_UNCERTAINTY',
                            'requires_field': 'hr', 'semantic_support': 'NOT_INDEPENDENTLY_VERIFIED'}
                if isinstance(old, str) and isinstance(new, str) and old != new:
                    patch = {'action_id': action, 'field': field, 'before': old, 'after': new,
                             'issue_id': issue['id'], 'authority': 'PROPOSED_CLASSIFICATION',
                             'semantic_support': 'NOT_INDEPENDENTLY_VERIFIED'}
                    if patch not in patches:
                        patches.append(patch)
                    if companion is not None and companion not in patches:
                        patches.append(companion)
    if not patches or len({p['issue_id'] for p in patches}) > 2:
        return None, []
    isolated = deepcopy(prior)
    # Removing a calibration error must not resurrect the stronger verdict
    # originally emitted by the model. Replay the actual committed state.
    harm_actions = {p['action_id'] for p in patches if p['field'] == 'hr'}
    for action in harm_actions:
        record = next(c['record'] for c in job['claims'] if c['record'].get('canonical_action_id') == action)
        rendering = {}
        for short, full in PRIOR_NATIVE_FIELDS.items():
            if full not in record:
                return None, []
            old = isolated['dp'][action].get(short)
            isolated['dp'][action][short] = record[full]
            if old != record[full]:
                rendering[short] = {'response_value': old, 'committed_value': record[full]}
        for patch in patches:
            if patch['action_id'] == action:
                patch['prior_native_rendering'] = rendering
    for patch in patches:
        isolated['dp'][patch['action_id']][patch['field']] = patch['after']
    explanation = str((proposed.get('qa') or {}).get('answer', ''))
    isolated['qa'] = {'issue_id': job['issue_id'], 'disposition': 'UNRESOLVED',
        'current_position_effect': 'NO_CHANGE', 'boundary_effect': 'NO_SWITCH',
        'answer': 'Isolated duty classification review. Only the listed classifications are proposed for change: '
                  + json.dumps(patches) + '. Other proposed duty conclusions remain unresolved. '
                  + 'Attributed original framework explanation: ' + explanation,
        'follow_up_question': 'Does the full proposed duty conclusion have an explicit supported derivation?'}
    isolated['j'] = 'Native revalidation of isolated classifications; prior duty conclusions and commitments retained.'
    result = deepcopy(prior_response)
    result['choices'][0]['text'] = json.dumps(isolated)
    return result, patches


class IsolatedReplay:
    model = 'deterministic-isolation-replay'
    execution_mode = 'ISOLATED_CLASSIFICATION_REPLAY'
    def __init__(self, response):
        self.response = deepcopy(response)
    def complete_json(self, prompt, **kwargs):
        return deepcopy(self.response)
