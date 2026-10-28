"""Grounded opposing-complaint assignments and nonoperative argument replies.

Builds on the saved-reply, matched-arm design in run_objection_response_pilot.py
and the focused-inference idea in run_specific_inference_probe.py. This module
does not compute moral truth or retire objections from any native ledger.
"""
from copy import deepcopy
import json

from argument_construction import object_schema
from contractualist_pilot import validate_shape
from inference_boundaries import value_at

VERSION = 'opposing-complaint-response/0.1'
ARMS = ('ordinary', 'focused')

SELECTOR_INSTRUCTION = (
    'Act as an objection proposer, not a verdict authority. Read the exact source and '
    'the saved framework argument. Identify ONE of the strongest grounded complaints '
    'against its ACTUAL recommendation from a person disadvantaged by that choice '
    'relative to a feasible alternative. Do not just copy its strongest_objection '
    'field: that field may name the wrong person or action. Reconstruct the person\'s '
    'reason in its strongest defensible form without inventing facts, preferences, '
    'mental states or a fixed numerical moral weight. State the available alternative, '
    'the exact inferential step challenged, and why the prior answer does not settle '
    'that complaint. Cite contiguous source quotations and one exact quoted span '
    'at a JSON pointer in the prior argument. Distinguish facts, moral warrants and '
    'disputed priority. A serious complaint is not automatically reasonable rejection '
    'or a veto. If complaints are incomparable, explain why this is one important '
    'challenge rather than declaring it objectively strongest. If no grounded '
    'opposing complaint is identifiable, say so; do not fabricate one.'
)

SHARED_REPLY = (
    'Respond from the same assigned ethical framework to the supplied opposing '
    'complaint about your saved argument. The source is unchanged; the prior reply '
    'and objection are fallible attributed arguments, not new facts or authoritative '
    'verdicts. You may retain, qualify or change your preference. A changed choice '
    'earns no credit by itself. Give a substantive answer, not a declaration that '
    'the objection has been resolved. Separate source facts, normative commitments '
    'and conditional assumptions. Do not invent empirical magnitudes, people\'s '
    'knowledge or consent to make an argument work. A conditional preference is '
    'useful; mere uncertainty is not an instruction to abstain. Self-reported '
    'ANSWERED is not ledger acceptance or moral proof. original_quote fields must '
    'match a contiguous span at original_path in the saved argument; source_quotes '
    'must match the exact source. Keep the final conclusion, its status and remaining '
    'objection consistent. Do not reconstruct the entire world or argument graph.'
)

ORDINARY = 'Revise your reasoning in light of the assigned complaint, using your framework.'
FOCUSED = (
    'First restate the disadvantaged person\'s actual complaint fairly, including '
    'what they lose under your recommendation and what the feasible alternative '
    'would offer them. Concede what is true or morally serious about it. Identify '
    'the exact priority premise your answer must defend. Then answer WHY the '
    'complainant has reason to accept that priority despite their own burden; '
    'repeating your preferred principle or pointing out benefits to a different '
    'person is not yet that explanation. Compare the strongest countercomplaint '
    'to the alternative and explain which consideration makes a difference. If '
    'you rely on numbers, explain why counting is licensed by your framework; '
    'if you rely on consent or an assurance, explain its force and limits. '
    'Acknowledge what your proposed rule asks the losing person to bear and what '
    'remains objectionable after your answer. If you cannot defend the comparative '
    'premise, revise the inferential claim and the strength of the conclusion, not '
    'just its confidence label. State the strongest useful conclusion that survives. '
    'Do not force agreement, a winner, or an automatic priority of imposition over '
    'omission, promise over rescue, or rescue effectiveness over assurance.'
)


def assignment_schema():
    text = {'type': 'string', 'minLength': 1}
    return object_schema({
        'status': {'type': 'string', 'enum': ['GROUNDED_TARGET_PROPOSED', 'NO_GROUNDED_TARGET']},
        'complainant': text, 'recommended_action_id': text, 'alternative_action_id': text,
        'source_quotes': {'type': 'array', 'items': text}, 'complaint': text,
        'target_path': text, 'target_quote': text, 'challenged_inference': text,
        'why_prior_answer_is_insufficient': text, 'selection_limits': text})


def reply_schema():
    text = {'type': 'string', 'minLength': 1}
    strings = {'type': 'array', 'items': text}
    nullable = lambda value: {'anyOf': [value, {'type': 'null'}]}
    premise = object_schema({'claim': text,
        'kind': {'type': 'string', 'enum': ['SOURCE_FACT', 'NORMATIVE_COMMITMENT',
                                           'EMPIRICAL_ASSUMPTION', 'CONDITIONAL_INTERPRETATION']},
        'source_quotes': strings})
    change = object_schema({'original_path': text, 'original_quote': text, 'replacement': text,
                             'reason': text})
    return object_schema({'complaint_restatement': text, 'conceded_points': strings,
        'priority_premise': text, 'answer': text, 'countercomplaint_to_alternative': text,
        'comparative_justification': text, 'premises': {'type': 'array', 'items': premise},
        'conclusion': object_schema({'action_id': nullable(text),
            'status': {'type': 'string', 'enum': ['PROVISIONAL_PREFERENCE', 'UNIQUE_DEFENSIBLE',
                                                 'MULTIPLE_DEFENSIBLE', 'UNRESOLVED']}, 'rationale': text}),
        'changed_inferences': {'type': 'array', 'items': change},
        'disposition': {'type': 'string', 'enum': ['ANSWERED', 'PARTLY_ANSWERED', 'UNANSWERED']},
        'remaining_objection': text, 'reversal_condition': text})


def make_request(system, packet, schema, model, reasoning_effort, max_tokens):
    return {'model': model, 'reasoning_effort': reasoning_effort,
        'max_completion_tokens': max_tokens,
        'messages': [{'role': 'system', 'content': system},
                     {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}],
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'opposing_complaint', 'strict': True, 'schema': schema}}}


def selector_request(packet, *, model, reasoning_effort, max_tokens=4096):
    return make_request(SELECTOR_INSTRUCTION, packet, assignment_schema(), model, reasoning_effort, max_tokens)


def bind_assignment(packet, raw):
    validate_shape(raw, assignment_schema())
    issues = []
    actions = {a['id'] for a in packet['source']['actions']}
    if raw['status'] == 'GROUNDED_TARGET_PROPOSED':
        if raw['recommended_action_id'] != packet['prior_argument']['recommendation_action_id']:
            issues.append('TARGET_NOT_PRIOR_RECOMMENDATION')
        if raw['alternative_action_id'] not in actions or raw['alternative_action_id'] == raw['recommended_action_id']:
            issues.append('INVALID_ALTERNATIVE')
        if not raw['source_quotes'] or any(q not in packet['source']['scenario'] for q in raw['source_quotes']):
            issues.append('SOURCE_QUOTE_MISMATCH')
        try:
            original = value_at(packet['prior_argument'], raw['target_path'])
            if not isinstance(original, str) or raw['target_quote'] not in original:
                issues.append('TARGET_QUOTE_MISMATCH')
        except (KeyError, IndexError, TypeError, ValueError):
            issues.append('TARGET_PATH_MISMATCH')
    return {'raw_assignment': deepcopy(raw), 'binding_issues': issues,
            'origin': 'MODEL_PROPOSED_OPPOSING_COMPLAINT', 'semantic_support': 'NOT_INDEPENDENTLY_ASSESSED',
            'operative': False}


def reply_request(packet, assignment, arm, *, model, reasoning_effort, max_tokens=8192):
    if arm not in ARMS:
        raise ValueError('Unknown response arm')
    shared = {**deepcopy(packet), 'assigned_complaint': deepcopy(assignment)}
    return make_request(SHARED_REPLY + '\n' + (FOCUSED if arm == 'focused' else ORDINARY),
                        shared, reply_schema(), model, reasoning_effort, max_tokens)


def bind_reply(packet, raw):
    """Citation and action bookkeeping only; does not grade objection handling."""
    validate_shape(raw, reply_schema())
    issues = []
    actions = {a['id'] for a in packet['source']['actions']}
    action, status = raw['conclusion']['action_id'], raw['conclusion']['status']
    if action is not None and action not in actions:
        issues.append('UNKNOWN_RECOMMENDATION')
    if (status in ('PROVISIONAL_PREFERENCE', 'UNIQUE_DEFENSIBLE')) != (action is not None):
        issues.append('CONCLUSION_STATUS_MISMATCH')
    for index, premise in enumerate(raw['premises']):
        if any(q not in packet['source']['scenario'] for q in premise['source_quotes']):
            issues.append('PREMISE_QUOTE_MISMATCH:' + str(index))
        if premise['kind'] == 'SOURCE_FACT' and not premise['source_quotes']:
            issues.append('SOURCE_FACT_WITHOUT_QUOTE:' + str(index))
    for index, change in enumerate(raw['changed_inferences']):
        try:
            original = value_at(packet['prior_argument'], change['original_path'])
            if not isinstance(original, str) or change['original_quote'] not in original:
                issues.append('PRIOR_CHANGE_QUOTE_MISMATCH:' + str(index))
        except (KeyError, IndexError, TypeError, ValueError):
            issues.append('PRIOR_CHANGE_PATH_MISMATCH:' + str(index))
    return {'raw_reply': deepcopy(raw), 'binding_issues': issues, 'operative': False,
            'world_state_authority': 'NONE', 'self_report_is_resolution': False,
            'semantic_support': 'NOT_INDEPENDENTLY_ASSESSED'}
