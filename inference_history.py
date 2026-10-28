"""Persistent argument dependencies and bounded review history, outside the world.

Claim support statuses are attributed content-review judgments. Neither native
admission nor a model's disposition can turn them into shared world facts.
"""
from copy import deepcopy

from deontology_repair_cards import fingerprint
from inference_review import check_challenge, assess_response

VERSION='inference-history/0.1'
STATUSES={'SUPPORTED','CONDITIONAL','WITHDRAWN','UNKNOWN'}
OUTCOMES={'CORRECTED','PARTIAL','DEFENDED','NO_PROGRESS'}


def _claims_valid(claims):
    if len({c['id'] for c in claims}) != len(claims):
        raise ValueError('Duplicate claim identity')
    by_id={c['id']:c for c in claims}
    visiting,done=set(),set()
    def visit(cid):
        if cid not in by_id:raise ValueError('Dangling argument dependency')
        if cid in visiting:raise ValueError('Cyclic argument dependency')
        if cid in done:return
        visiting.add(cid)
        row=by_id[cid]
        if row['status'] not in STATUSES or not row['text'].strip():
            raise ValueError('Invalid claim status or text')
        for parent in row['depends_on']:visit(parent)
        visiting.remove(cid);done.add(cid)
    for cid in by_id:visit(cid)


def _event(state,kind,payload):
    event={'sequence':len(state['events'])+1,'kind':kind,'payload':deepcopy(payload),
           'parent_sha256':state['events'][-1]['event_sha256'] if state['events'] else fingerprint(state['root'])}
    event['event_sha256']=fingerprint(event)
    state['events'].append(event)
    return state


def validate_history(state):
    if state['version'] != VERSION:raise ValueError('Unknown history version')
    if state['root']['maximum_rounds'] != 2:raise ValueError('Unsupported review budget')
    _claims_valid(state['root']['claims'])
    previous=fingerprint(state['root'])
    responses={}
    reviews=set()
    for sequence,event in enumerate(state['events'],1):
        body={k:v for k,v in event.items() if k!='event_sha256'}
        if (event['sequence']!=sequence or event['parent_sha256']!=previous
                or fingerprint(body)!=event['event_sha256']):
            raise ValueError('Altered history event or parent')
        payload=event['payload']
        if event['kind']=='RESPONSE':
            digest=fingerprint(payload['response'])
            if digest in responses:raise ValueError('Duplicate response')
            responses[digest]=payload
            if len(responses)>2:raise ValueError('Review budget exceeded')
        elif event['kind']=='CONTENT_REVIEW':
            digest=payload['response_sha256']
            if digest not in responses or digest in reviews:raise ValueError('Unknown or already reviewed response')
            reviews.add(digest)
            _claims_valid(payload['claims'])
        else:raise ValueError('Unknown history event')
        previous=event['event_sha256']
    return True


def start_history(world_sha256, challenge, prior_reply, *, dependent_claims=()):
    check_challenge(challenge,prior_reply)
    target='CLAIM:'+fingerprint([challenge['challenge_id'],'target'])[:24]
    claims=[{'id':target,'text':challenge['quote'],'status':'UNKNOWN','depends_on':[],
             'evidence_refs':[],'origin':'CHALLENGED_ARGUMENT'}]+deepcopy(list(dependent_claims))
    _claims_valid(claims)
    return {'version':VERSION,'root':{'world_sha256':world_sha256,
            'issue_id':challenge['challenge_id'],'target_claim_id':target,
            'challenge':deepcopy(challenge),'prior_reply_sha256':fingerprint(prior_reply),
            'maximum_rounds':2,'claims':claims,'operative':False,
            'authority':'ATTRIBUTED_ARGUMENT_REVIEW_NOT_WORLD_EVIDENCE'},'events':[]}


def current_claims(state):
    validate_history(state)
    reviews=[e for e in state['events'] if e['kind']=='CONTENT_REVIEW']
    return deepcopy(reviews[-1]['payload']['claims'] if reviews else state['root']['claims'])


def dependency_projection(state):
    """Conservative support invalidation; never infer a fresh normative verdict."""
    claims=current_claims(state)
    by_id={c['id']:c for c in claims}
    projected={}
    def status(cid):
        if cid in projected:return projected[cid]
        row=by_id[cid]; parents=[status(p) for p in row['depends_on']]
        effective=row['status']
        if effective not in ('WITHDRAWN','UNKNOWN'):
            if any(p in ('WITHDRAWN','UNKNOWN','REQUIRES_REASSESSMENT') for p in parents):
                effective='REQUIRES_REASSESSMENT'
            elif 'CONDITIONAL' in parents:effective='CONDITIONAL'
        projected[cid]=effective
        return effective
    return [{**c,'effective_status':status(c['id'])} for c in claims]


def followup_state(state):
    validate_history(state)
    responses=[e for e in state['events'] if e['kind']=='RESPONSE']
    reviews=[e for e in state['events'] if e['kind']=='CONTENT_REVIEW']
    if not responses:return {'status':'AVAILABLE','rounds_remaining':2,'remaining_gap':state['root']['challenge']['missing_premise']}
    latest=fingerprint(responses[-1]['payload']['response'])
    review=next((e['payload'] for e in reversed(reviews) if e['payload']['response_sha256']==latest),None)
    if review is None:status='AWAITING_CONTENT_REVIEW'
    elif review['outcome'] in ('CORRECTED','DEFENDED'):status='STOP_REVIEWED'
    elif review['outcome']=='NO_PROGRESS':status='STOP_NO_PROGRESS'
    elif len(responses)>=2:status='STOP_ROUND_LIMIT'
    else:status='AVAILABLE'
    return {'status':status,'rounds_remaining':2-len(responses),
            'remaining_gap':review['remaining_gap'] if review else None,
            'review_outcome':review['outcome'] if review else None}


def revision_context(state):
    availability=followup_state(state)
    if availability['status']!='AVAILABLE':
        raise ValueError('Follow-up unavailable: '+availability['status'])
    reviews=[e['payload'] for e in state['events'] if e['kind']=='CONTENT_REVIEW']
    return {'issue_id':state['root']['issue_id'],'round':3-availability['rounds_remaining'],
            'remaining_gap':availability['remaining_gap'],
            'previous_corrections':[r['corrected_points'] for r in reviews],
            'newly_identified_unsupported_claims':[r['new_unsupported_claims'] for r in reviews],
            'argument_dependencies':dependency_projection(state),
            'instruction':'Address the remaining gap; preserve supported corrections. Do not regenerate the whole argument.'}


def review_context_for(state,challenge,prior_reply,world_sha256):
    context=revision_context(state)  # checked before callers make another request
    check_challenge(challenge,prior_reply)
    root=state['root']
    if world_sha256!=root['world_sha256']:raise ValueError('Frozen world changed')
    if (challenge['evidence']!=root['challenge']['evidence'] or not set(challenge['affected_paths'])<=set(root['challenge']['affected_paths'])):
        raise ValueError('Follow-up expands original evidence or edit scope')
    responses=[e for e in state['events'] if e['kind']=='RESPONSE']
    expected=responses[-1]['payload']['response']['local_response'] if responses else None
    if fingerprint(prior_reply)!=(fingerprint(expected) if expected is not None else root['prior_reply_sha256']):
        raise ValueError('Follow-up is not based on latest saved reply')
    return context


def record_response(state,challenge,prior_reply,response,*,native_audit):
    context=review_context_for(state,challenge,prior_reply,state['root']['world_sha256'])
    assessment=assess_response(challenge,prior_reply,response)
    updated=deepcopy(state)
    return _event(updated,'RESPONSE',{'challenge':challenge,'prior_reply_sha256':fingerprint(prior_reply),
        'response':response,'contract_assessment':assessment,'native_status':native_audit.get('status'),
        'native_audit_sha256':fingerprint(native_audit),'revision_context':context,
        'effective_status':'OPEN_PENDING_CONTENT_REVIEW'})


def review_response(state,response_sha256,*,reviewer,outcome,rationale,remaining_gap,
                    corrected_points, new_unsupported_claims,claim_updates=(),new_claims=()):
    validate_history(state)
    responses=[e['payload'] for e in state['events'] if e['kind']=='RESPONSE']
    if not responses or fingerprint(responses[-1]['response'])!=response_sha256:
        raise ValueError('Review must assess latest saved response')
    if any(e['kind']=='CONTENT_REVIEW' and e['payload']['response_sha256']==response_sha256 for e in state['events']):
        raise ValueError('Response already reviewed; preserve its history')
    if not reviewer or not rationale or outcome not in OUTCOMES:
        raise ValueError('Attributed reviewer and explicit outcome required')
    if outcome=='PARTIAL' and not remaining_gap:raise ValueError('Partial correction requires remaining gap')
    if outcome in ('CORRECTED','DEFENDED') and (remaining_gap or new_unsupported_claims):
        raise ValueError('Outstanding issues cannot be reported as completed')
    if outcome in ('CORRECTED','DEFENDED') and not responses[-1]['contract_assessment']['contract_valid']:
        raise ValueError('Malformed response cannot complete review')
    claims=current_claims(state); by_id={c['id']:c for c in claims}
    for update in claim_updates:
        if update['id'] not in by_id:raise ValueError('Unknown updated claim')
        by_id[update['id']].update(deepcopy(update))
    if set(by_id)&{c['id'] for c in new_claims}:raise ValueError('New claims require new identities')
    claims+=deepcopy(list(new_claims)); _claims_valid(claims)
    updated=deepcopy(state)
    return _event(updated,'CONTENT_REVIEW',dict(response_sha256=response_sha256,reviewer=reviewer,
        outcome=outcome,rationale=rationale,remaining_gap=remaining_gap,
        corrected_points=deepcopy(corrected_points),new_unsupported_claims=deepcopy(new_unsupported_claims),claims=claims,
        authority='ATTRIBUTED_CONTENT_REVIEW_NOT_AUTOMATIC_TRUTH'))
