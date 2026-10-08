"""Local contract consistency, not semantic proof or native moral adjudication."""
from copy import deepcopy
import math
import re
import json

from logic_puzzles_frameworks import fingerprint
from parliament_recognition import survivor_count_projection, explicit_probability

OPERATIONS = ('DEFEND', 'CONDITIONALIZE', 'WEAKEN', 'RETRACT')
SCOPES = ('EXPECTED_SURVIVORS', 'TOTAL_WELFARE')
EDITABLE_FIELDS = frozenset(('r', 'ss', 'ia', 'am', 'cc', 'esa', 'c', 'u', 'w', 'j', 'x',
    'l', 'da', 't', 'tf', 'dr', 'ft', 'nt', 'z', 'scores', 'fic', 'foq'))
NUMERIC = re.compile(r'(?<![\w])[-+]?\d+(?:\.\d+)?(?:\s*%)?')
WELFARE_UNIT = r'(?:life[\s_-]*(?:equivalents?|years?)|utilit(?:y|ies)|utils?\b)'
NUMERIC_WELFARE = re.compile(NUMERIC.pattern + r'\s*(?:expected\s+)?'+WELFARE_UNIT, re.I)
WELFARE_NUMERIC = re.compile(WELFARE_UNIT + r'\s*(?:[<>=≤≥:]|is|of)?\s*' + NUMERIC.pattern, re.I)


def numeric_welfare_claim(text):
    return bool(NUMERIC_WELFARE.search(text) or WELFARE_NUMERIC.search(text))


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def repair_schema():
    text = {'type': 'string', 'maxLength': 1000}
    strings = {'type': 'array', 'items': text, 'maxItems': 16}
    action = {'type': 'string', 'enum': ['A0', 'A1', 'NONE']}
    return obj({
        'challenged_inference_id': text,
        'repair_operation': {'type': 'string', 'enum': list(OPERATIONS)},
        'supported_conclusion': obj({'text': text, 'scope': {'type': 'string', 'enum': list(SCOPES)},
            'status': {'type': 'string', 'enum': ['SUPPORTED', 'CONDITIONAL', 'UNRESOLVED', 'WITHDRAWN']}, 'action_id': action}),
        'conditional_assumptions': {'type': 'array', 'maxItems': 8, 'items': obj({'id': text, 'text': text,
            'provenance': {'type': 'string', 'enum': ['EMPIRICAL_ASSUMPTION', 'DEFEASIBLE_NORMATIVE_ASSUMPTION']},
            'evidence_ids': strings})},
        'conditional_preference': obj({'action_id': action, 'text': text, 'assumption_ids': strings}),
        'preserved_restricted_findings': {'type': 'array', 'maxItems': 2, 'items': obj({
            'action_id': action, 'value': {'type': 'number'}, 'unit': {'type': 'string', 'enum': ['PERSON']},
            'method': {'type': 'string', 'enum': ['EXPECTED_COUNT']}, 'source_effect_ids': strings})},
        'unresolved_claims': strings,
        'reversal_condition': obj({'text': text, 'status': {'type': 'string',
            'enum': ['SOURCE_GROUNDED', 'FRAMEWORK_DERIVED', 'ILLUSTRATIVE', 'UNSUPPORTED']},
            'evidence_ids': strings, 'derivation_id': {'type': ['string', 'null']}})})


def shape_valid(value, schema):
    """Validate only the small envelope's JSON shape, without adding a dependency."""
    kind = schema['type']
    if isinstance(kind, list):
        return any(shape_valid(value, dict(schema, type=t)) for t in kind)
    types = {'object': dict, 'array': list, 'string': str, 'number': (int, float), 'null': type(None)}
    if not isinstance(value, types[kind]): return False
    if kind == 'number' and (isinstance(value, bool) or not math.isfinite(value)): return False
    if 'enum' in schema and value not in schema['enum']: return False
    if kind == 'object':
        if set(value) != set(schema['properties']): return False
        return all(shape_valid(value[k], s) for k, s in schema['properties'].items())
    if kind == 'array':
        return len(value) <= schema['maxItems'] and all(shape_valid(v, schema['items']) for v in value)
    return kind != 'string' or len(value) <= schema.get('maxLength', 1000)


def make_contract(prior_native, frozen, commitments, *, selection_scope='TOTAL_WELFARE'):
    if selection_scope not in SCOPES: raise ValueError('Unknown selection scope')
    world = frozen['action_source_grounding']['world_model']
    effects = {e['effect_id']: e for e in world['effects']}
    parties = {p['party_id']: p for p in world['parties']}
    findings = []
    for action, ev in prior_native['ev'].items():
        if ev['unit'] != 'PERSON' or ev['method'] != 'EXPECTED_COUNT' or ev['grounded'] is not True:
            raise ValueError('This bounded contract requires the verified person-count endpoint')
        ids = ev['source_effect_ids']
        if len(ids) != 1: raise ValueError('Only the saved single-person comparison is supported')
        effect = effects[ids[0]]; party = parties[effect['party_id']]
        attributes = dict(effect, party_kind=party['kind'], party_label=party['label'])
        projection = survivor_count_projection(attributes, effect['outcome'])
        if (not projection or effect['action_id'] != action
                or not any(p['excerpt'] in frozen['scenario'] for p in effect['provenance'])
                or not math.isclose(float(explicit_probability(attributes)), ev['value'], abs_tol=1e-12)):
            raise ValueError('Saved count does not match the admitted single-person probability')
        findings.append({k: deepcopy(ev[k]) for k in ('value', 'unit', 'method', 'source_effect_ids')})
        findings[-1]['action_id'] = action
    if {f['action_id'] for f in findings} != {'A0', 'A1'}: raise ValueError('Both count rows are required')
    return {'version': 'bounded-inference-repair/0.1',
        'challenged_inference_id': 'promise_cost_to_total_welfare', 'selection_scope': selection_scope,
        'prior_native': deepcopy(prior_native), 'preserved_restricted_findings': findings,
        'source_sha256': fingerprint(frozen), 'world_sha256': fingerprint(world),
        'commitments_sha256': fingerprint(commitments),
        'evidence_ids': sorted(set(effects) | {'PROP:WORLD:'+i for i in effects}
                               | {q['passage_id'] for q in commitments.values()}),
        # No total-welfare conversion or reversal quantity is licensed in this case.
        'licensed_numeric_reversals': {}, 'licensed_welfare_units': [],
        'editable_native_fields': sorted(EDITABLE_FIELDS)}


def annotated_saved_repair(contract):
    """Explicit reviewer fixture; never presented as a model-authored repair."""
    return {'challenged_inference_id': contract['challenged_inference_id'], 'repair_operation': 'CONDITIONALIZE',
        'supported_conclusion': {'scope': contract['selection_scope'], 'status': 'UNRESOLVED',
            'action_id': 'NONE', 'text': 'Total welfare ranking is not established.'},
        'conditional_assumptions': [{'id': 'assumption_cost', 'text': 'Promise costs do not outweigh survival benefit.',
            'provenance': 'EMPIRICAL_ASSUMPTION', 'evidence_ids': []}],
        'conditional_preference': {'action_id': 'A0', 'text': 'A0 only if a justified welfare comparison supports it.',
            'assumption_ids': ['assumption_cost']},
        'preserved_restricted_findings': deepcopy(contract['preserved_restricted_findings']),
        'unresolved_claims': ['Total welfare comparison', 'Common welfare unit'],
        'reversal_condition': {'text': 'Promise costs offset the survival benefit under a justified welfare comparison.',
            'status': 'UNSUPPORTED', 'evidence_ids': [], 'derivation_id': None}}


def check_submission(contract, repair, native, frozen, commitments):
    errors = []
    world_ok = fingerprint(frozen['action_source_grounding']['world_model']) == contract['world_sha256']
    if not world_ok: errors.append('WORLD_CHANGED')
    if fingerprint(frozen) != contract['source_sha256']: errors.append('SOURCE_CHANGED')
    if fingerprint(commitments) != contract['commitments_sha256']: errors.append('COMMITMENTS_CHANGED')
    telemetry = {'dispatch_native': False, 'errors': errors, 'world_unchanged': world_ok,
        'result': 'CONTRACT_BLOCKED',
        'semantic_support': 'NOT_ASSESSED', 'self_report_authority': 'TELEMETRY_ONLY',
        'preserved_restricted_findings': deepcopy(contract['preserved_restricted_findings']),
        'native_fields_rewritten': False}
    if not shape_valid(repair, repair_schema()):
        errors.append('MISSING_REPAIR_RECORD' if repair is None else 'INVALID_REPAIR_SCHEMA')
        return telemetry
    telemetry.update(challenged_inference_id=repair['challenged_inference_id'], repair_operation=repair['repair_operation'],
        declared_assumptions=deepcopy(repair['conditional_assumptions']), unresolved_claims=deepcopy(repair['unresolved_claims']),
        reversal_condition=deepcopy(repair['reversal_condition']),
        withdrawn_claims=[repair['supported_conclusion']['text']] if repair['repair_operation'] == 'RETRACT' else [])
    conclusion = repair['supported_conclusion']; op = repair['repair_operation']; status = conclusion['status']
    if repair['challenged_inference_id'] != contract['challenged_inference_id']: errors.append('INFERENCE_ID_MISMATCH')
    if conclusion['scope'] != contract['selection_scope']: errors.append('CONCLUSION_SCOPE_MISMATCH')
    allowed = {'DEFEND': {'SUPPORTED'}, 'CONDITIONALIZE': {'CONDITIONAL', 'UNRESOLVED'},
               'WEAKEN': {'CONDITIONAL', 'UNRESOLVED'}, 'RETRACT': {'WITHDRAWN'}}
    if status not in allowed[op]: errors.append('OPERATION_STATUS_MISMATCH')
    if repair['preserved_restricted_findings'] != contract['preserved_restricted_findings']:
        errors.append('RESTRICTED_FINDINGS_CHANGED')
    assumptions = repair['conditional_assumptions']; ids = [a['id'] for a in assumptions]
    pref = repair['conditional_preference']
    if len(set(ids)) != len(ids) or not set(pref['assumption_ids']) <= set(ids): errors.append('ASSUMPTION_REFERENCES')
    if status == 'SUPPORTED' and assumptions: errors.append('UNCONDITIONAL_FROM_ASSUMPTIONS')
    if any(not set(a['evidence_ids']) <= set(contract['evidence_ids']) for a in assumptions): errors.append('EVIDENCE_REFERENCES')
    if native.get('ss') not in {'SELECTED', 'PROVISIONAL', 'UNSELECTED'}:
        errors.append('INVALID_NATIVE_STRENGTH')
    if status in {'CONDITIONAL', 'UNRESOLVED'}:
        if (native.get('ss') not in {'PROVISIONAL', 'UNSELECTED'} or native.get('esa') is not False
                or native.get('cc') is not False or native.get('ia') != 'NONE'
                or any(v in {'REQUIRED', 'REJECTED'} for v in native.get('am', {}).values())):
            errors.append('SELECTION_STRENGTH')
        if native.get('ss') == 'PROVISIONAL' and (pref['action_id'] == 'NONE' or not pref['assumption_ids']
                or native.get('r') != pref['action_id']): errors.append('CONDITIONAL_PREFERENCE_MISMATCH')
    if status == 'WITHDRAWN': errors.append('WITHDRAWN_INFERENCE_NO_DISPATCH')
    if status == 'SUPPORTED' and native.get('ss') == 'SELECTED' and native.get('r') != conclusion['action_id']:
        errors.append('SUPPORTED_ACTION_MISMATCH')
    prior = contract['prior_native']
    if set(native) != set(prior) or any(native[k] != prior[k] for k in prior if k not in EDITABLE_FIELDS):
        errors.append('PROTECTED_NATIVE_FIELDS')
    # Only operative native fields are guarded. Withdrawn/illustrative prose stays inspectable.
    operative = json.dumps({k: native.get(k) for k in ('x', 'l', 't', 'tf', 'dr', 'ft', 'nt', 'w')}, ensure_ascii=False)
    active_repair_text = [a['text'] for a in assumptions if a['id'] in pref['assumption_ids']]
    if pref['action_id'] != 'NONE': active_repair_text.append(pref['text'])
    if status in {'SUPPORTED', 'CONDITIONAL'}: active_repair_text.append(conclusion['text'])
    if (numeric_welfare_claim(operative) or any(numeric_welfare_claim(t) for t in active_repair_text)) and not contract['licensed_welfare_units']:
        errors.append('UNLICENSED_WELFARE_UNIT')
    reversal = repair['reversal_condition']; ft = native.get('ft', 'NONE')
    if ft != 'NONE':
        if ft != reversal['text']: errors.append('REVERSAL_TEXT_MISMATCH')
        if not set(reversal['evidence_ids']) <= set(contract['evidence_ids']): errors.append('REVERSAL_EVIDENCE_REFERENCES')
        if reversal['status'] == 'SOURCE_GROUNDED' and not reversal['evidence_ids']: errors.append('REVERSAL_EVIDENCE_REFERENCES')
        if reversal['status'] in {'ILLUSTRATIVE', 'UNSUPPORTED'}: errors.append('NONOPERATIVE_REVERSAL')
        if NUMERIC.search(ft):
            license = contract['licensed_numeric_reversals'].get(reversal['derivation_id'])
            if (not license or license['text'] != ft or license['scope'] != contract['selection_scope']
                    or license['evidence_ids'] != reversal['evidence_ids']):
                errors.append('UNLICENSED_NUMERIC_REVERSAL')
    telemetry['dispatch_native'] = not errors
    telemetry['result'] = 'CONSISTENT' if not errors else 'CONTRACT_BLOCKED'
    return telemetry
