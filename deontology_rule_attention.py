"""Source-bound rule context and attributed, nonoperative authored duties."""
from copy import deepcopy
import hashlib
import json


def source_rule_context(world):
    parties = {p['party_id']: p for p in world.get('parties', [])}
    propositions = {p['proposition_id']: p for p in world.get('propositions', [])}
    rows = []
    for norm in world.get('normative_propositions', []):
        content = propositions.get(norm.get('governed_proposition_id'), {})
        rows.append({'norm_id': norm['norm_id'], 'force': norm.get('force'),
                     'bearer': deepcopy(parties.get(norm.get('bearer_party_id'))),
                     'authority': deepcopy(parties.get(norm.get('authority_party_id'))),
                     'governed_proposition_id': norm.get('governed_proposition_id'),
                     'content': content.get('predication'),
                     'polarity': content.get('polarity'),
                     'clause_ids': deepcopy(norm.get('clause_ids', []))})
    return {'origin': 'FROZEN_WORLD_COPY', 'moral_authority': 'NOT_ASSESSED',
            'identity_resolution': 'NO_ADDITIONAL_MERGES', 'rules': rows}


def rule_attention_instruction(context):
    if not context or not context['rules']:
        return ''
    return ('SOURCE-RULE ASSESSMENT: ' + json.dumps(context, ensure_ascii=False) + '\n'
            'Before duty ranking, examine each source rule for BOTH actions: applicability '
            'under its stated conditions; the grounds and limits of institutional moral authority; '
            'and whether emergency rescue justifies an exception or leaves a conflict. '
            'Distinguish rule compliance from moral permissibility and independent ethical duties. '
            'Do not infer current conditions from a rule conditional, resolve unknown pronouns, '
            'or treat a stated rule as automatically morally binding. Stop-and-wait may itself '
            'be an active alternative. Use existing l/dr/rs/pr fields for this compact analysis '
            'and foq for unresolved questions; do not add response fields. A source norm ID '
            'supports its actual content, not a generic duty hierarchy. The existing coercion '
            'auth field is not a substitute for assessing institutional authority.\n')


def authored_duty_proposal(calls):
    for call in reversed(calls):
        response = call.get('response')
        try:
            raw = response['choices'][0]['text']
            data = json.loads(raw)
        except (TypeError, KeyError, IndexError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get('dp'), dict) and data['dp']:
            return {'origin': 'SAVED_MODEL_RESPONSE', 'operative': False,
                    'world_state_authority': 'NONE',
                    'response_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                    'duties': deepcopy(data['dp'])}
    return None
