"""Explicit unanswered questions as useful, nonvoting argument contributions.

This is our argument-adapter extension, not a decision procedure for contractualism.
It binds annotations without declaring that a theory cannot answer the question.
"""
from copy import deepcopy

VERSION = 'critical-argument-questions/0.1'

CONTRACTUALIST_GUIDANCE = (
    'Identifying the critical unanswered questions is a substantive contribution; '
    'an action recommendation is NOT required. Do not assume a winner must emerge. '
    'When comparative priority is not defended, identify the exact unresolved '
    'question, affected individuals, competing principles and available actions. '
    'Separate missing facts, disputed interpretations and normative priority: more '
    'precise facts alone need not resolve a normative comparison. Explain why the '
    'question matters, what evidence or argument could help, and what follows '
    'conditionally from different answers. Do not invent a numerical exchange '
    'rate between unlike reasons, treat uncertainty as a tie, or let a proposed '
    'tie-breaker become justified merely because a decision is wanted. Test the '
    'actual competing principles; rejecting an absolute rule does not defeat '
    'a qualified opposing principle. Hypothetical acceptance by a reasonable '
    'person is a proposed argument, not actual consent, a prediction of majority '
    'opinion, or proof that dissent is unreasonable. Do not infer consent from '
    'role-play. An open question can be decision-relevant without being answerable '
    'on this record. Do not declare it unanswerable in principle or a failure of '
    'contractualism. Use a null recommendation and UNRESOLVED when no ranking '
    'is justified. A conditional preference is allowed only with its disputed '
    'priority premise explicit; it does not resolve the question. A genuinely '
    'defended ranking may still support a recommendation. Do not force abstention '
    'when independent reasons settle the comparison.'
)

QUESTION_FORMAT = (
    'Use critical_questions for specific questions still open on this record, '
    'not generic requests for more information. Each has a kind (FACTUAL, '
    'INTERPRETIVE or NORMATIVE), affected person/action/principle IDs, relevant '
    'statement target_handles, why_it_matters, needed_to_resolve and '
    'conditional_implications. Use only supplied or declared IDs. In a follow-up '
    'use IDs/handles from prior_argument. DECISION_DEPENDENCY means the proposed '
    'justification needs an answer; RESIDUAL_CONCERN means the issue remains but '
    'is not claimed to control the recommendation. These are model annotations, '
    'not automatic blockers. Empty critical_questions is allowed if no specific '
    'question remains. A question-only contribution can use null action, '
    'UNRESOLVED assessments and a conclusion explaining what the decision turns '
    'on. Do not fill verdict fields with an invented preference. Each conditional '
    'implication describes an assumption and what would follow if defended; '
    'neither is a source fact or an accepted moral priority.'
)


def question_schema(object_schema):
    text = {'type': 'string', 'minLength': 1}
    strings = {'type': 'array', 'items': text}
    return {'type': 'array', 'maxItems': 6, 'items': object_schema({
        'id': text, 'kind': {'type': 'string', 'enum': ['FACTUAL', 'INTERPRETIVE', 'NORMATIVE']},
        'role': {'type': 'string', 'enum': ['DECISION_DEPENDENCY', 'RESIDUAL_CONCERN']},
        'question': text, 'person_ids': strings, 'action_ids': strings,
        'principle_ids': strings, 'target_handles': strings,
        'why_it_matters': text, 'needed_to_resolve': text,
        'conditional_implications': {'type': 'array', 'items': object_schema({
            'assumption': text, 'implication': text})}})}


def review_questions(questions, argument, actions, recommendation_status, *, recorded=True):
    """Reference bookkeeping only. Retain questions and choices despite findings."""
    findings, seen = [], set()
    indexes = {'person_ids': {p['id'] for p in argument.get('persons', [])},
               'action_ids': set(actions),
               'principle_ids': {p['id'] for p in argument.get('principles', [])},
               'target_handles': {s['handle'] for s in argument.get('statements', [])}}
    for question in questions:
        qid = question['id']
        if qid in seen:
            findings.append({'code': 'DUPLICATE_QUESTION_ID', 'question_id': qid})
        seen.add(qid)
        for field, known in indexes.items():
            for ref in question[field]:
                if ref not in known:
                    findings.append({'code': 'UNKNOWN_QUESTION_REFERENCE', 'question_id': qid,
                                     'field': field, 'reference': ref})
        if question['role'] == 'DECISION_DEPENDENCY' and recommendation_status == 'UNIQUE_DEFENSIBLE':
            findings.append({'code': 'OPEN_DEPENDENCY_WITH_UNIQUENESS_CLAIM', 'question_id': qid})
    return {'version': VERSION, 'questions': deepcopy(questions), 'findings': findings,
            'recording_status': 'RECORDED' if recorded else 'NOT_RECORDED_LEGACY',
            'semantic_support': 'NOT_INDEPENDENTLY_ASSESSED', 'operative': False,
            'question_is_automatic_blocker': False, 'unanswerable_in_principle': 'NOT_ESTABLISHED'}


def render_questions(review):
    lines = ['## Critical unanswered questions', '']
    if review['recording_status'] == 'NOT_RECORDED_LEGACY':
        lines += ['This older response did not record structured questions; absence is not evidence that none remain.', '']
    elif not review['questions']:
        lines += ['The model recorded no specific open questions. This is not independent certification of completeness.', '']
    for q in review['questions']:
        lines += [f"**{q['id']} — {q['kind']} / {q['role']}**", '', q['question'], '',
                  'Affected people: ' + ', '.join(q['person_ids']) + '.',
                  'Options: ' + ', '.join(q['action_ids']) + '.', '',
                  'Why it matters: ' + q['why_it_matters'], '',
                  'What could help: ' + q['needed_to_resolve'], '']
        for implication in q['conditional_implications']:
            lines += ['- If ' + implication['assumption'] + ': ' + implication['implication']]
        lines += ['']
    lines += ['These are proposed questions and conditional arguments, not settled moral findings.', '']
    return '\n'.join(lines)
