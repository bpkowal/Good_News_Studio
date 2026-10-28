"""Evidence-facing projections and fact-first replies to opposing complaints.

Reuses the existing burden positions; it does not extract a new world or certify
that a cited passage entails a model's paraphrase. Moral labels remain proposals.
"""
from collections import Counter
from copy import deepcopy
import hashlib

from argument_construction import object_schema
from complaint_comparison import derive_position_changes
from contractualist_pilot import validate_shape
from contractualist_questions import (CONTRACTUALIST_GUIDANCE, QUESTION_FORMAT,
                                     question_schema, review_questions)
from inference_boundaries import value_at
from opposing_complaint import (FOCUSED, SHARED_REPLY, bind_reply, make_request,
                               reply_schema)

VERSION = 'complaint-consequence-reply/0.2'
ARMS = ('focused_baseline', 'consequences_first')
REFINEMENT = (
    'Before defending your conclusion, compare the assigned complaint and your own '
    'saved reasoning against the source and the person-by-option consequence_record. '
    'The record is a projection of FALLIBLE model annotations, not an admitted world '
    'or a semantic oracle. Exact source text takes priority over every paraphrase. '
    'Correct factual errors in either side, and distinguish contradictions from '
    'claims that are simply not established. Unsupported severity, knowledge, '
    'reliance, institutional entitlement or assent must not become facts. Consent '
    'may change a complaint\'s warrant without removing the physical or financial '
    'loss. Distinguish the actual assurance and its wording from a future policy '
    'that could qualify assurances. Do not retroactively insert such a qualification. '
    'For every quantitative comparison keep the same identified person, option, '
    'endpoint, unit and conditional probability. A risk PRESENT under both options '
    'can change in magnitude. A recipient\'s probability is not everyone\'s probability; '
    'do not replace identified people with an unstated identity lottery. Distinguish '
    'guaranteed outcomes from possible outcomes. Do not erase a complaint because '
    'one factual premise needs correction: restate the strongest objection that '
    'survives. Then defend the NORMATIVE warrant for your comparative priority, '
    'including why the disadvantaged person should accept it and its strongest '
    'counterreason. Neither the matrix, a number, a count of complainants nor the '
    'name of a theory supplies this warrant. Separate a justification for this '
    'actual choice from a proposed future policy or a hypothetical boundary test. '
    'If a warrant remains contestable, retain that objection and calibrate the '
    'inferential conclusion itself, not merely a confidence label. No choice is '
    'prescribed by the record or by the critic.'
)
REVIEW_FORMAT = (
    'In factual_review cite an exact quote and JSON pointer in the named origin: '
    'PRIOR_ARGUMENT means prior_argument; ASSIGNED_COMPLAINT means assigned_complaint '
    '(including raw_assignment); CONSEQUENCE_RECORD means consequence_record when '
    'provided. source_quotes are exact contiguous source substrings, without added '
    'quotation marks. SUPPORTED is your proposed support assessment, not certification. '
    'CONTRADICTED, NOT_ESTABLISHED, NORMATIVE_NOT_FACT and CONDITIONAL are distinct. '
    'Keep corrected_statement faithful even when the target is morally serious. '
    'Record row IDs only if a consequence record is supplied. Give the normative '
    'warrant, its justification and its limits separately from factual corrections.'
)


def build_consequence_record(packet):
    """Project every supplied person/burden across options; never fill missing cells.

    Bound quotations are source spans, not proofs of the associated interpretation.
    All input bytes/structures are preserved. No moral comparisons are computed.
    """
    source, prior = packet['source'], packet['prior_argument']
    text = source['scenario']
    if hashlib.sha256(text.encode()).hexdigest() != source['source_text_sha256']:
        raise ValueError('Source text hash mismatch')
    actions = source['actions']
    if len({a['id'] for a in actions}) != len(actions):
        raise ValueError('Duplicate source action')
    statement_counts = Counter(s['handle'] for s in prior.get('statements', []))
    statements = {s['handle']: (i, s) for i, s in enumerate(prior.get('statements', []))}
    anchors, ground_cache = {}, {}

    def ground(handle):
        if handle in ground_cache:
            return deepcopy(ground_cache[handle])
        result = {'handle': handle, 'prior_path': None, 'claimed_kind': None,
                  'claimed_text': None, 'source_anchor_ids': [], 'binding_issues': []}
        if statement_counts[handle] != 1:
            result['binding_issues'].append('MISSING_OR_DUPLICATE_STATEMENT')
        else:
            index, statement = statements[handle]
            result.update(prior_path=f'/statements/{index}',
                          claimed_kind=statement['kind'], claimed_text=statement['text'])
            if statement['kind'] != 'FACTUAL_PREMISE':
                result['binding_issues'].append('GROUND_IS_NOT_A_FACTUAL_PREMISE')
            for citation in statement.get('citations', []):
                quote = citation.get('quote', '')
                if (citation.get('evidence_id') != 'SCENE' or citation.get('value_path') != ''
                        or not quote or quote not in text):
                    result['binding_issues'].append('UNBOUND_SOURCE_QUOTE')
                    continue
                if quote not in anchors:
                    offsets, start = [], 0
                    while (start := text.find(quote, start)) >= 0:
                        offsets.append({'start': start, 'end': start + len(quote)})
                        start += 1
                    anchors[quote] = {'id': f'e{len(anchors)}', 'text': quote,
                                      'occurrences': offsets}
                result['source_anchor_ids'].append(anchors[quote]['id'])
            if not result['source_anchor_ids']:
                result['binding_issues'].append('NO_BOUND_SOURCE_SPAN')
        ground_cache[handle] = result
        return deepcopy(result)

    rows = []
    person_counts = Counter(p['id'] for p in prior.get('persons', []))
    burden_counts = Counter(b['id'] for b in prior.get('burdens', []))
    cells = prior.get('burden_positions', [])
    for bi, burden in enumerate(prior.get('burdens', [])):
        for action in actions:
            matches = [(i, c) for i, c in enumerate(cells)
                       if c['burden_id'] == burden['id'] and c['action_id'] == action['id']]
            issues = []
            if person_counts[burden['person_id']] != 1:
                issues.append('MISSING_OR_DUPLICATE_PERSON')
            if burden_counts[burden['id']] != 1:
                issues.append('DUPLICATE_BURDEN')
            if len(matches) != 1:
                issues.append('MISSING_OR_DUPLICATE_POSITION')
            rows.append({'id': f'row{len(rows)}', 'person_id': burden['person_id'],
                'burden_id': burden['id'], 'proposed_burden_label': burden['description'],
                'burden_path': f'/burdens/{bi}', 'action_id': action['id'],
                'claimed_status': matches[0][1]['status'] if len(matches) == 1 else 'UNKNOWN',
                'raw_positions': [{'prior_path': f'/burden_positions/{i}',
                    'claimed_status': c['status'], 'claimed_position': c['position'],
                    'grounds': [ground(h) for h in c['ground_handles']]} for i, c in matches],
                'binding_issues': issues})
    # Include cited factual claims even when omitted from the positions (e.g.,
    # explicit absence of a promise). A normative label is never promoted here.
    factual_claims = [ground(s['handle']) for s in prior.get('statements', [])
                      if s['kind'] == 'FACTUAL_PREMISE']
    pair_changes = derive_position_changes({
        'burdens': prior.get('burdens', []), 'burden_positions': cells,
        'comparisons': prior.get('comparisons', [])})
    for frame in pair_changes:
        frame.pop('proposed_comparison_status', None)
    known_burdens = {b['id'] for b in prior.get('burdens', [])}
    known_actions = {a['id'] for a in actions}
    unassigned = [{'prior_path': f'/burden_positions/{i}', 'raw': deepcopy(c)}
                  for i, c in enumerate(cells)
                  if c['burden_id'] not in known_burdens or c['action_id'] not in known_actions]
    return {'version': VERSION, 'source_text_sha256': source['source_text_sha256'],
        'authority': 'MODEL_POSITION_PROJECTION_NOT_SOURCE_VERIFIED',
        'source_priority': 'Exact source overrides projected interpretations; a bound quote does not prove entailment.',
        'actions': deepcopy(actions), 'proposed_person_labels': deepcopy(prior.get('persons', [])),
        'position_rows': rows, 'cited_source_spans': list(anchors.values()),
        'factual_premise_claims': factual_claims, 'pair_changes': pair_changes,
        'unassigned_positions': unassigned,
        'coverage': 'Only supplied burdens; omitted burdens and missing facts remain unknown.',
        'moral_ranking_computed': False, 'operative': False}


def consequence_reply_schema():
    schema = deepcopy(reply_schema())
    text = {'type': 'string', 'minLength': 1}
    strings = {'type': 'array', 'items': text}
    target = object_schema({'origin': {'type': 'string', 'enum': [
        'PRIOR_ARGUMENT', 'ASSIGNED_COMPLAINT', 'CONSEQUENCE_RECORD']},
        'path': text, 'quote': text})
    review = object_schema({'target': target,
        'status': {'type': 'string', 'enum': ['SUPPORTED', 'CONTRADICTED', 'NOT_ESTABLISHED',
                                             'NORMATIVE_NOT_FACT', 'CONDITIONAL']},
        'corrected_statement': text, 'source_quotes': strings, 'consequence_row_ids': strings})
    # Put factual review first in the requested output, before the argument.
    fields = {'factual_review': {'type': 'array', 'items': review},
              'normative_warrant': object_schema({'principle': text, 'justification': text, 'limits': text}),
              'critical_questions': question_schema(object_schema),
              **schema['properties']}
    return object_schema(fields)


def consequence_request(packet, assignment, arm, *, model, reasoning_effort, max_tokens=8192):
    if arm not in ARMS:
        raise ValueError('Unknown consequence-reply arm')
    shared = {**deepcopy(packet), 'assigned_complaint': deepcopy(assignment)}
    system = SHARED_REPLY + '\n' + FOCUSED + '\n' + REVIEW_FORMAT + '\n' + QUESTION_FORMAT
    if packet['framework_id'] == 'contractualist':
        # The saved framework instruction remains immutable. This is the current
        # task guidance for a fresh response, not a rewrite of the old opening.
        system += '\n' + CONTRACTUALIST_GUIDANCE
    if arm == 'consequences_first':
        shared['consequence_record'] = build_consequence_record(packet)
        system += '\n' + REFINEMENT
    return make_request(system, shared, consequence_reply_schema(), model, reasoning_effort, max_tokens)


def bind_consequence_reply(packet, assignment, raw, record=None):
    """Bind review targets; no semantic grading, factual rewriting or new gates."""
    shape_input = deepcopy(raw)
    recorded_questions = 'critical_questions' in shape_input
    shape_input.setdefault('critical_questions', [])
    validate_shape(shape_input, consequence_reply_schema())
    legacy = {k: deepcopy(raw[k]) for k in reply_schema()['properties']}
    base = bind_reply(packet, legacy)
    origins = {'PRIOR_ARGUMENT': packet['prior_argument'], 'ASSIGNED_COMPLAINT': assignment}
    if record is not None:
        origins['CONSEQUENCE_RECORD'] = record
    row_ids = {r['id'] for r in (record or {}).get('position_rows', [])}
    for i, review in enumerate(raw['factual_review']):
        target = review['target']
        try:
            value = value_at(origins[target['origin']], target['path'])
            if not isinstance(value, str) or target['quote'] not in value:
                base['binding_issues'].append(f'REVIEW_TARGET_QUOTE_MISMATCH:{i}')
        except (KeyError, IndexError, TypeError, ValueError):
            base['binding_issues'].append(f'REVIEW_TARGET_PATH_MISMATCH:{i}')
        if any(q not in packet['source']['scenario'] for q in review['source_quotes']):
            base['binding_issues'].append(f'REVIEW_SOURCE_QUOTE_MISMATCH:{i}')
        if not set(review['consequence_row_ids']) <= row_ids:
            base['binding_issues'].append(f'REVIEW_ROW_REFERENCE_MISMATCH:{i}')
    base.update(raw_reply=deepcopy(raw), review_status_is_semantic_certification=False)
    base['critical_question_review'] = review_questions(shape_input['critical_questions'],
        packet['prior_argument'], {a['id'] for a in packet['source']['actions']},
        raw['conclusion']['status'], recorded=recorded_questions)
    return base


def render_consequences(record):
    lines = ['# Person-by-option consequence record', '',
             'Fallible projection of saved model positions. Exact source takes priority.',
             'Labels and bound quotations do not establish semantic support or moral ranking.', '',
             '| Row | Person | Option | Proposed burden | Claimed position |',
             '| --- | --- | --- | --- | --- |']
    escape = lambda s: str(s).replace('|', '\\|').replace('\n', ' ')
    for row in record['position_rows']:
        position = '; '.join(x['claimed_position'] for x in row['raw_positions']) or 'UNKNOWN (missing)'
        lines.append('| ' + ' | '.join(map(escape, [row['id'], row['person_id'], row['action_id'],
                                                    row['proposed_burden_label'], position])) + ' |')
    lines += ['', '## Cited source spans', '']
    for anchor in record['cited_source_spans']:
        lines += ['- ' + anchor['id'] + ': ' + anchor['text']]
    return '\n'.join(lines) + '\n'
