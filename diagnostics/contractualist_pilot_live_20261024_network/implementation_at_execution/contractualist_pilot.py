"""Neutral, nonvoting argument adapter for two or three explicit text options.

Person/value reasons and attack annotations informed by Marcos-Vidal, Abbo and
Belpaeme (2026), arXiv:2608.01937v1, sections 5.1-5.3. This is not an ASPIC+
implementation and computes neither defeat nor reasonable rejection. See
CONTRACTUALIST_RESEARCH.md for credit and deliberate departures.
"""
from copy import deepcopy
import hashlib
import json

from argument_construction import object_schema, schemas, write_report
from argument_frameworks import FRAMEWORK_GUIDANCE, SHARED_GENERATION, VERSION as INSTRUCTION_VERSION
from argument_support import compile_argument, project
from deontology_repair_cards import fingerprint

VERSION = 'neutral-framework-pilot/0.1'
FRAMEWORKS = ('utilitarian', 'deontological', 'care', 'contractualist')
PAPER = {'authors': ['Luis Marcos-Vidal', 'Giulio Antonio Abbo', 'Tony Belpaeme'],
         'title': 'A Contractualist Argumentation Framework for Moral Decision-Making',
         'version': 'arXiv:2608.01937v1', 'url': 'https://arxiv.org/html/2608.01937v1',
         'used_sections': ['5.1', '5.2', '5.3'], 'formal_semantics_implemented': False}


def source_packet(case):
    """Allowlist excludes design notes, pair labels, outcomes of previous runs."""
    source = {key: deepcopy(case[key]) for key in ('id', 'scenario', 'actions')}
    digest = hashlib.sha256(source['scenario'].encode('utf-8')).hexdigest()
    if digest != case['source_text_sha256']:
        raise ValueError('Frozen source text changed')
    ids = [a['id'] for a in source['actions']]
    if len(ids) not in (2, 3) or len(ids) != len(set(ids)):
        raise ValueError('Pilot requires two or three distinct explicit options')
    source.update(source_text_sha256=digest, source_status='FROZEN_TEXT_NOT_ADMITTED_WORLD',
                  evidence=[{'ref': 'SCENE', 'kind': 'EXACT_SCENARIO', 'value': source['scenario']}])
    return source


def response_schema():
    text = {'type': 'string', 'minLength': 1}
    strings = {'type': 'array', 'items': text}
    nullable = lambda x: {'anyOf': [x, {'type': 'null'}]}
    array = lambda item, maximum: {'type': 'array', 'items': item, 'maxItems': maximum}
    candidate = schemas()[0]['properties']['candidates']['items']['properties']
    person = object_schema({'id': text, 'label': text, 'source_quotes': strings})
    principle = object_schema({'id': text, 'statement_handle': text, 'action_ids': strings})
    reason = object_schema({'id': text, 'person_id': nullable(text), 'principle_id': nullable(text),
        'action_id': text, 'stance': {'type': 'string', 'enum': ['FOR', 'AGAINST', 'UNRESOLVED']},
        'value': text, 'conclusion_handle': text, 'ground_handles': strings,
        'warrant_handles': strings, 'generic_basis': text})
    comparison = object_schema({'id': text, 'reason_ids': strings, 'judgment_handle': text,
                                'explanation': text})
    challenge = object_schema({'id': text,
        'kind': {'type': 'string', 'enum': ['UNDERMINING', 'REBUTTING', 'UNDERCUTTING']},
        'target_kind': {'type': 'string', 'enum': ['STATEMENT', 'SUPPORT']},
        'target_id': text, 'challenge_handle': text, 'response_handle': nullable(text),
        'status': {'type': 'string', 'enum': ['OPEN', 'ANSWERED', 'UNRESOLVED']}})
    assessment = object_schema({'action_id': text, 'conclusion_handle': text,
        'assessment': {'type': 'string', 'enum': ['DEFENSIBLE', 'REJECTED', 'UNRESOLVED']}})
    return object_schema({
        'persons': array(person, 12), 'principles': array(principle, 8),
        'statements': array(candidate['statements']['items'], 32),
        'supports': array(candidate['supports']['items'], 40),
        'reasons': array(reason, 16), 'comparisons': array(comparison, 12),
        'challenges': array(challenge, 8), 'action_assessments': array(assessment, 3),
        'recommendation_action_id': nullable(text), 'conclusion': text,
        'strongest_objection': text, 'answer_to_objection': text,
        'reversal_condition': text, 'residual_disagreement': strings})


COMMON_FORMAT = (
    'Return one compact argument record covering EVERY supplied option, not a new world. '
    'Use the common format in the way appropriate to your framework: principles and '
    'personal reasons are available to all frameworks but are not mandatory methods for '
    'noncontractualists. Use null when a reason is not individual or principle-specific; '
    'do not misdescribe an aggregate reason as one person\'s reason. source_quotes must '
    'be exact substrings of SCENE. citations use evidence_id SCENE and value_path "". '
    'Use declared statement handles in supports. Reference a support by its output handle '
    'when specifying an UNDERCUTTING challenge; this denotes the support bundle for that '
    'conclusion, not a guessed canonical ID. UNDERMINING challenges a premise; REBUTTING '
    'challenges a conclusion; UNDERCUTTING challenges an inference application. These '
    'annotations and ANSWERED labels are proposed model judgments, not established defeats. '
    'Reason ground_handles and warrant_handles must also be inputs or ancestors of the '
    'reason conclusion in supports. Comparative judgments must explicitly link the reasons '
    'compared. Action-assessment conclusions are terminal: do not feed a permissibility '
    'verdict back into premises. A claim that an option is permissible is not a fact that '
    'it occurred. Reasons, comparisons and challenges reference the same immutable '
    'statement handles, rather than detached prose. Explain the final preference or '
    'uncertainty in conclusion. Do not assert missing subjective preferences as facts. '
    'Keep reasons short enough to fit the common response budget. '
)


def make_request(source, framework, *, model='o3', reasoning_effort='low', max_tokens=8192):
    if framework not in FRAMEWORKS:
        raise ValueError('Unknown pilot framework')
    system = SHARED_GENERATION + COMMON_FORMAT
    user = {'framework_id': framework, 'framework_instruction': FRAMEWORK_GUIDANCE[framework],
            'source': source}
    return {'model': model, 'reasoning_effort': reasoning_effort,
            'max_completion_tokens': max_tokens,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)}],
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'neutral_framework_argument', 'strict': True, 'schema': response_schema()}}}


def validate_shape(value, schema, path='response'):
    """Validate exactly the small schema subset emitted above, without a dependency."""
    if 'anyOf' in schema:
        for option in schema['anyOf']:
            try:
                validate_shape(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(path + ': no allowed type')
    kind = schema['type']
    matches = {'object': isinstance(value, dict), 'array': isinstance(value, list),
               'string': isinstance(value, str), 'null': value is None}
    if not matches[kind] or ('enum' in schema and value not in schema['enum']):
        raise ValueError(path + ': wrong type or value')
    if kind == 'object':
        if set(value) != set(schema['required']):
            raise ValueError(path + ': unknown or missing fields')
        for key, item in value.items():
            validate_shape(item, schema['properties'][key], path + '/' + key)
    elif kind == 'array':
        if len(value) > schema.get('maxItems', len(value)) or len(value) < schema.get('minItems', 0):
            raise ValueError(path + ': array size')
        for index, item in enumerate(value):
            validate_shape(item, schema['items'], path + '/' + str(index))
    elif kind == 'string' and len(value) < schema.get('minLength', 0):
        raise ValueError(path + ': empty string')


def compile_response(source, framework, raw):
    """Compile what is usable, expose bad bindings, never accept a moral verdict."""
    result = {'version': VERSION, 'framework': framework, 'raw_response': deepcopy(raw),
              'source_text_sha256': source['source_text_sha256'], 'source_status': source['source_status'],
              'operative': False, 'world_state_authority': 'NONE',
              'semantic_support': 'NOT_INDEPENDENTLY_ASSESSED', 'paper_attribution': deepcopy(PAPER),
              'instruction_version': INSTRUCTION_VERSION, 'issues': []}
    try:
        validate_shape(raw, response_schema())
    except ValueError as error:
        result.update(status='MALFORMED_RESPONSE', shape_error=str(error))
        return result
    # The legacy compiler calls its evidence-boundary hash world_sha256. Here it
    # binds frozen text only; it does not establish or admit a Parliament world.
    state = compile_argument(source['source_text_sha256'], source['evidence'],
                             source['id'] + ':' + framework, raw['statements'], raw['supports'], partial=True)
    result.update(argument=state, view=project(state))
    issues = result['issues']
    issues.extend({'kind': 'CONSTRUCTION', **p} for p in state.get('construction_problems', []))
    def note(kind, record):
        issues.append({'kind': kind, 'record_id': record})
    def index(records, label):
        keyed = {r['id']: r for r in records}
        if len(keyed) != len(records):
            note('DUPLICATE_' + label, label)
        return keyed
    persons = index(raw['persons'], 'PERSON')
    principles = index(raw['principles'], 'PRINCIPLE')
    reasons = index(raw['reasons'], 'REASON')
    index(raw['comparisons'], 'COMPARISON'); index(raw['challenges'], 'CHALLENGE')
    handles = state['handles']
    actions = {a['id'] for a in source['actions']}
    def bind(handle, record):
        if handle not in handles:
            note('MISSING_STATEMENT_REFERENCE', record)
        return handles.get(handle)
    def ancestors(handle):
        sid = handles.get(handle)
        reached, pending = set(), [sid] if sid else []
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(state['statements'][current]['hypothesis_ids'])
            for route in state['routes'].values():
                if route['output'] == current:
                    pending.extend(route['inputs'])
        return reached - {sid}
    for person in persons.values():
        if not person['source_quotes'] or any(q not in source['scenario'] for q in person['source_quotes']):
            note('PERSON_SOURCE_BINDING_MISSING', person['id'])
    for principle in principles.values():
        sid = bind(principle['statement_handle'], principle['id'])
        if sid and state['statements'][sid]['kind'] != 'PRINCIPLE':
            note('PRINCIPLE_KIND_MISMATCH', principle['id'])
        if not principle['action_ids'] or not set(principle['action_ids']) <= actions:
            note('PRINCIPLE_ACTION_BINDING', principle['id'])
    for reason in reasons.values():
        bind(reason['conclusion_handle'], reason['id'])
        for field, lookup in [('person_id', persons), ('principle_id', principles)]:
            if reason[field] is not None and reason[field] not in lookup:
                note('MISSING_' + field.upper(), reason['id'])
        if reason['action_id'] not in actions:
            note('MISSING_ACTION', reason['id'])
        if framework == 'contractualist' and (reason['person_id'] is None or reason['principle_id'] is None):
            note('CONTRACTUALIST_REASON_BINDING_INCOMPLETE', reason['id'])
        ancestry = ancestors(reason['conclusion_handle'])
        for handle in reason['ground_handles'] + reason['warrant_handles']:
            sid = bind(handle, reason['id'])
            if sid and sid not in ancestry:
                note('DETACHED_REASON_SUPPORT', reason['id'])
        if not reason['ground_handles'] or not reason['warrant_handles']:
            note('REASON_GROUNDS_OR_WARRANT_MISSING', reason['id'])
    for comparison in raw['comparisons']:
        sid = bind(comparison['judgment_handle'], comparison['id'])
        if not comparison['reason_ids'] or not set(comparison['reason_ids']) <= reasons.keys():
            note('COMPARISON_REASON_BINDING', comparison['id'])
        elif sid and any(handles.get(reasons[r]['conclusion_handle']) not in
                         ancestors(comparison['judgment_handle']) for r in comparison['reason_ids']):
            note('DETACHED_COMPARISON', comparison['id'])
    for challenge in raw['challenges']:
        bind(challenge['challenge_handle'], challenge['id'])
        if challenge['response_handle'] is not None:
            bind(challenge['response_handle'], challenge['id'])
        if challenge['target_kind'] == 'STATEMENT':
            bind(challenge['target_id'], challenge['id'])
        elif not any(r['output'] == handles.get(challenge['target_id']) for r in state['routes'].values()):
            note('MISSING_SUPPORT_TARGET', challenge['id'])
        if (challenge['kind'] == 'UNDERCUTTING') != (challenge['target_kind'] == 'SUPPORT'):
            note('ATTACK_TARGET_KIND_MISMATCH', challenge['id'])
    assessments = raw['action_assessments']
    if {a['action_id'] for a in assessments} != actions or len(assessments) != len(actions):
        note('OPTION_COVERAGE_INCOMPLETE', 'action_assessments')
    terminals = {bind(a['conclusion_handle'], a['action_id']) for a in assessments}
    for route in state['routes'].values():
        if terminals.intersection(route['inputs']):
            note('VERDICT_USED_AS_PREMISE', route['output'])
    if raw['recommendation_action_id'] is not None and raw['recommendation_action_id'] not in actions:
        note('UNKNOWN_RECOMMENDATION', 'recommendation_action_id')
    if framework == 'contractualist' and (not principles or not reasons or not raw['comparisons']):
        note('CONTRACTUALIST_ANALYSIS_INCOMPLETE', framework)
    result['status'] = 'STAGED_PARTIAL_ARGUMENT' if issues else 'STAGED_ARGUMENT'
    return result


def write_argument_report(result, path):
    candidate = {'label': result['framework'], 'status': result['status']}
    if 'argument' in result:
        candidate.update(argument=result['argument'], view=result['view'],
                         strongest_remaining_conclusion=result['raw_response']['conclusion'],
                         remaining_questions=result['raw_response']['residual_disagreement'])
    write_report({'candidates': [candidate]}, path)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write('\n\n## Binding issues (not moral verdicts)\n\n')
        handle.write(json.dumps(result.get('issues', []), indent=2))
        handle.write('\n\nFull principles, reasons, comparisons and proposed attacks: [response.json](response.json).\n')
