"""Compare admitted semantic graphs and review only their source differences.

Selection is an attributed interpretation of an existing admitted proposal. It
never edits effects, promotes overlays, or establishes a normative verdict.
"""
import hashlib
import json
from collections import Counter
from itertools import combinations

from z10_world_model_adapter import segment_source_clauses


def _text(value):
    return ' '.join(str(value).casefold().split()).rstrip('. ')


def semantic_view(proposal):
    world = (proposal.get('candidate') or {}).get('world_model') or {}
    ids = {}
    for collection, key in [('parties', 'party_id'), ('actions', 'action_id'), ('conditions', 'condition_id'),
                            ('propositions', 'proposition_id')]:
        for row in world.get(collection, []):
            label = row.get('label') or row.get('intervention') or row.get('text') or row.get('predication') or row.get('predicate') or row.get('source_proposition') or ''
            ids[row.get(key)] = collection + ':' + _text(label)
    for row in world.get('modal_operators', []):
        ids[row.get('modal_id')] = 'modal:' + _text(row.get('force', '')) + ':' + ids.get(row.get('governed_proposition_id'), str(row.get('governed_proposition_id', '')))
    ignored = {'provenance', 'clause_ids', 'derivation_explanation', 'kind_origin'}
    def clean(value, *, exclude=()):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in sorted(value.items())
                    if k not in ignored and k not in exclude}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            return ids.get(value, _text(value))
        return value
    def serial(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    for row in world.get('effects', []):
        anchor = clean(row, exclude=('effect_id', 'source_effect_ids', 'condition_ids'))
        ids[row.get('effect_id')] = 'effect:' + hashlib.sha256(serial(anchor).encode()).hexdigest()
    result = {}
    for collection in ('parties', 'actions', 'conditions', 'propositions', 'modal_operators', 'effects', 'causal_links',
                       'temporal_relations', 'counterfactual_links'):
        rows = []
        for row in world.get(collection, []):
            own_id = {'parties':'party_id', 'actions':'action_id', 'conditions':'condition_id', 'effects':'effect_id',
                      'propositions':'proposition_id', 'modal_operators':'modal_id'}.get(collection)
            # Record IDs are bookkeeping; endpoint IDs remain semantic references.
            item = clean(row, exclude=(own_id, 'link_id', 'relation_id') if own_id else ('link_id', 'relation_id'))
            for key in ('effect_ids', 'recipient_party_ids', 'condition_ids', 'source_effect_ids', 'quantities'):
                if key in item:item[key] = sorted(item[key], key=serial)
            rows.append(item)
        result[collection] = sorted(rows, key=serial)
    # Discourse claims can distinguish worlds even when occurrence atoms match.
    for key, value in world.items():
        if key not in result and key not in {'schema_version', 'provenance'}:
            result[key] = clean(value)
    return result


def semantic_signature(proposal):
    return json.dumps(semantic_view(proposal), sort_keys=True, ensure_ascii=False)


def compare_admitted(admitted):
    candidates = []
    for i, item in enumerate(admitted):
        attempt = item.get('attempt') or {}
        candidates.append({'id': 'candidate_' + str(i), 'blueprint_id': attempt.get('blueprint_id'),
            'variant': (item.get('record') or {}).get('variant', 'baseline'),
            'view': semantic_view(item['proposal'])})
    differences = []
    for a, b in combinations(candidates, 2):
        for field in sorted(set(a['view']) | set(b['view'])):
            left, right = a['view'].get(field), b['view'].get(field)
            if left != right:
                if isinstance(left, list) and isinstance(right, list):
                    encode = lambda rows: Counter(json.dumps(r, sort_keys=True) for r in rows)
                    lrows, rrows = encode(left), encode(right)
                    left = [json.loads(r) for r, n in (lrows - rrows).items() for _ in range(n)]
                    right = [json.loads(r) for r, n in (rrows - lrows).items() for _ in range(n)]
                differences.append({'id': 'difference_' + str(len(differences)),
                    'left_id': a['id'], 'right_id': b['id'], 'field': field,
                    'left': left, 'right': right,
                    'question': 'Which ' + field + ' reading is supported by the source?'})
    return {'candidates': candidates, 'differences': differences,
            'status': 'EQUIVALENT' if not differences else 'SUBSTANTIVE_DIFFERENCES'}


def review_selection(scenario, admitted, complete):
    comparison = compare_admitted(admitted)
    clauses = segment_source_clauses(scenario)
    request = {'scenario': scenario, 'source_clauses': clauses, 'comparison': comparison}
    prompt = '''Review only the displayed differences between already admitted interpretations.
Choose an existing candidate if the original text supports its disputed readings;
otherwise leave unresolved. Compare source fidelity, not perfect completeness:
a supported partial graph may be selected if its missing content is recorded in
missing_content and does not invalidate the stated action/outcome interpretation.
Distinguish omission (e.g. a count not encoded) from contradiction (e.g. a lever
typed as a person). Do not repair either by adding world atoms. Do not prefer a blueprint name, rank, ethical verdict,
or admission status. Do not add facts, survival complements, actions or relations.
For each difference involving your chosen candidate, give a source-grounded reason.
Return JSON: {"status":"SELECTED" or "UNRESOLVED", "selected_candidate_id":
string or null, "resolutions":[{"difference_id":string,"rationale":string,
"evidence":[{"clause_id":string,"excerpt":exact source substring}]}],
"unresolved_question":string,"missing_content":[string]}. A selection must address EVERY displayed difference
involving that candidate. Explain which differences are supported, contradicted,
or unrepresented. If no candidate offers a supported action/outcome interpretation,
abstain and identify the precise disputed reading, not a generic selection question.
INPUT:\n''' + json.dumps(request, ensure_ascii=False)
    result = {'request': request, 'status': 'UNRESOLVED', 'selected_candidate_id': None,
              'errors': [], 'origin': 'MODEL_INTERPRETATION_REVIEW',
              'semantic_support': 'reviewer_assessed_not_proven'}
    try:
        raw = complete([{'role': 'user', 'content': prompt}])
        result['raw_response'] = raw
        response = json.loads(raw) if isinstance(raw, str) else raw
        result['response'] = response
        if response.get('status') != 'SELECTED':
            result['unresolved_question'] = response.get('unresolved_question') or 'Source does not resolve the displayed differences.'
            return result, None
        ident = response.get('selected_candidate_id')
        index = next(i for i, c in enumerate(comparison['candidates']) if c['id'] == ident)
        needed = {d['id'] for d in comparison['differences'] if ident in (d['left_id'], d['right_id'])}
        rows = response.get('resolutions') or []
        if {r.get('difference_id') for r in rows} != needed or len(rows) != len(needed):
            raise ValueError('Selection must address each difference once.')
        sources = {c['clause_id']: c['text'] for c in clauses}
        for row in rows:
            if not str(row.get('rationale') or '').strip() or not row.get('evidence'):
                raise ValueError('Every difference needs rationale and source evidence.')
            for proof in row['evidence']:
                excerpt = proof.get('excerpt')
                if not isinstance(excerpt, str) or not excerpt.strip() or excerpt not in sources.get(proof.get('clause_id'), ''):
                    raise ValueError('Review evidence must match the referenced original clause.')
        result.update(status='SELECTED', selected_candidate_id=ident)
        return result, admitted[index]
    except Exception as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
        result['unresolved_question'] = 'Which admitted interpretation preserves the disputed source content?'
        return result, None
