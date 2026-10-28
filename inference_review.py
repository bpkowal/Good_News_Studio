"""Reusable, reviewer-triggered inference challenges; no semantic truth oracle.

Reviewers identify a precise unsupported bridge. This protocol is independent of
scenario words and frameworks; adapters supply the permitted local-edit schema.
"""
from copy import deepcopy
import json

from deontology_repair_cards import fingerprint, get_path

VERSION = 'inference-review/0.1'
BASES = ['SOURCE_EVIDENCE', 'INTERPRETIVE_ASSUMPTION', 'NORMATIVE_ASSUMPTION', 'UNKNOWN']


def make_challenge(prior_reply, *, claim_path, quote, defect_kind, missing_premise,
                   evidence, affected_paths, reviewer, inference_map=None):
    if not quote or quote not in get_path(prior_reply, claim_path):
        raise ValueError('Challenge must quote the exact saved claim')
    if not reviewer or not missing_premise or not affected_paths:
        raise ValueError('Reviewer, missing premise and affected paths are required')
    if defect_kind not in ('UNSUPPORTED_FACT', 'MISSING_BRIDGE', 'SCOPE_OVERREACH',
                           'NORMATIVE_PRIORITY', 'ABSENCE_AS_EVIDENCE'):
        raise ValueError('Unknown investigative defect kind')
    packet = dict(version=VERSION, prior_reply_sha256=fingerprint(prior_reply),
                  claim_path=claim_path, quote=quote, defect_kind=defect_kind,
                  missing_premise=missing_premise, evidence=deepcopy(evidence),
                  affected_paths=list(affected_paths), reviewer=reviewer,
                  authority='REVIEWER_OBJECTION_NOT_ESTABLISHED_FACT', status='OPEN')
    if inference_map is not None:
        from inference_boundaries import boundary_card
        boundary_card(inference_map,evidence)
        packet['inference_map']=deepcopy(inference_map)
    packet['challenge_id']='INFERENCE:'+fingerprint(packet)[:24]
    return packet


def check_challenge(challenge, prior_reply):
    rebuilt=make_challenge(prior_reply, **{k:challenge[k] for k in
        ('claim_path','quote','defect_kind','missing_premise','evidence','affected_paths','reviewer')},
        inference_map=challenge.get('inference_map'))
    if rebuilt != challenge:
        raise ValueError('Stale or altered inference challenge')


def review_schema(local_schema, *, typed_reasoning=False):
    text={'type':'string','minLength':1}
    strings={'type':'array','items':text}
    premise={'type':'object','additionalProperties':False,
             'properties':{'claim':text,'basis':{'type':'string','enum':BASES},
                           'evidence_refs':strings},
             'required':['claim','basis','evidence_refs']}
    props={'challenge_id':text,
           'disposition':{'type':'string','enum':['DEFENDED','QUALIFIED','WITHDRAWN','UNRESOLVED']},
           'premises':{'type':'array','items':premise,'minItems':1},
           'bridge':text, 'withdrawn_claims':strings, 'qualifications':strings,
           'strongest_conclusion':text, 'remaining_questions':strings,
           'local_response':deepcopy(local_schema)}
    if typed_reasoning:
        from inference_boundaries import reasoning_schema
        props['reasoning_revision']=reasoning_schema()
    return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}


def request_review(challenge, prior_reply, local_schema, backend, *, max_tokens=2048, revision_context=None):
    check_challenge(challenge,prior_reply)
    boundary=None
    if 'inference_map' in challenge:
        from inference_boundaries import boundary_card
        boundary=boundary_card(challenge['inference_map'],challenge['evidence'])
    prompt=('Review one challenged inference in your saved argument. The reviewer may be mistaken. '
            'Separate source-supported premises, interpretive assumptions, normative assumptions and unknowns. '
            'Defend the bridge with the supplied evidence, qualify it explicitly, withdraw it, or leave it unresolved. '
            'A missing graph edge is not proof of absence. Causal structure alone does not establish intention. '
            'A named principle alone does not establish its priority. Do not introduce unsupported facts to rescue a conclusion. '
            'State the strongest conclusion that survives without the unsupported premise and propagate uncertainty '
            'to the permitted local fields. Do not choose a verdict or alter unrelated records. '
            'Your disposition is a self-report, not closure of the objection.\n'+
            json.dumps({'challenge':challenge,'prior_reply':prior_reply,
                        'revision_history':revision_context,'boundary_review':boundary},ensure_ascii=False))
    if boundary:
        prompt+='\nRevise the typed claims and bridge dependencies in reasoning_revision. Preserve existing claim IDs and exact source locations; use new IDs for newly introduced claims. '
        prompt+='A conditional maxim can be useful without establishing actual intention. A reviewer can be mistaken; do not force uncertainty when the source supplies the relevant practical reasoning.'
    raw=backend.complete_json(prompt,schema=review_schema(local_schema,typed_reasoning=boundary is not None),
                              max_tokens=max_tokens,temperature=0.0)
    return json.loads(raw['choices'][0]['text']) if 'choices' in raw else raw


def assess_response(challenge, prior_reply, response):
    """Check bookkeeping only. Never equate a model disposition with support."""
    check_challenge(challenge,prior_reply)
    errors=[]
    required=set(review_schema({},typed_reasoning='inference_map' in challenge)['required'])
    if not isinstance(response,dict) or set(response) != required:
        return {'contract_valid':False,'errors':['INVALID_RESPONSE_FIELDS'],
                'effective_status':'OPEN_PENDING_CONTENT_REVIEW',
                'semantic_support':'NOT_INDEPENDENTLY_VERIFIED','world_state_authority':'NONE'}
    if response.get('challenge_id') != challenge['challenge_id']:
        errors.append('WRONG_CHALLENGE_ID')
    disposition=response.get('disposition')
    if disposition not in ('DEFENDED','QUALIFIED','WITHDRAWN','UNRESOLVED'):
        errors.append('INVALID_DISPOSITION')
    if disposition=='WITHDRAWN' and challenge['quote'] not in response.get('withdrawn_claims',[]):
        errors.append('WITHDRAWAL_MUST_IDENTIFY_TARGET_CLAIM')
    if disposition=='QUALIFIED' and not response.get('qualifications'):
        errors.append('QUALIFICATION_MUST_BE_EXPLICIT')
    if disposition=='UNRESOLVED' and not response.get('remaining_questions'):
        errors.append('UNRESOLVED_REQUIRES_QUESTION')
    for field in ('bridge','strongest_conclusion'):
        if not isinstance(response[field],str) or not response[field].strip():
            errors.append('MISSING_'+field.upper())
    for field in ('withdrawn_claims','qualifications','remaining_questions'):
        if not isinstance(response[field],list) or any(not isinstance(v,str) for v in response[field]):
            errors.append('INVALID_'+field.upper())
    if not isinstance(response['local_response'],dict):
        errors.append('INVALID_LOCAL_RESPONSE')
    if not isinstance(response.get('premises'),list) or not response['premises']:
        errors.append('MISSING_PREMISE_ACCOUNT')
    citation_bindings=[]
    from argument_support import resolve_citation
    for premise in response['premises'] if isinstance(response['premises'],list) else []:
        if not isinstance(premise,dict):
            errors.append('INVALID_PREMISE');continue
        if premise.get('basis') not in BASES:
            errors.append('UNKNOWN_PREMISE_BASIS')
        citations=premise.get('evidence_refs',[])
        if not isinstance(citations,list):errors.append('INVALID_CITATIONS');continue
        for citation in citations:
            try:citation_bindings.append(resolve_citation(citation,challenge['evidence']))
            except (ValueError,TypeError):errors.append('UNKNOWN_EVIDENCE_REFERENCE')
        if premise.get('basis')=='SOURCE_EVIDENCE' and not premise.get('evidence_refs'):
            errors.append('SOURCE_PREMISE_REQUIRES_REFERENCE')
    boundary=None
    if 'inference_map' in challenge:
        from inference_boundaries import boundary_card
        try:
            boundary=boundary_card(response['reasoning_revision'],challenge['evidence'])
            old_ids={c['id'] for c in challenge['inference_map']['claims']}
            new_ids={c['id'] for c in response['reasoning_revision']['claims']}
            if not old_ids<=new_ids:errors.append('MISSING_PRIOR_CLAIM_IDENTITIES')
        except (ValueError,KeyError,TypeError) as error:errors.append('INVALID_REASONING_REVISION: '+str(error))
    canonical=None
    if boundary is not None and not errors:
        from argument_support import compile_legacy_review
        # The source fingerprint identifies this argument overlay, not an
        # additional admitted world. Never grant an operative vote here.
        canonical=compile_legacy_review(challenge,response,fingerprint(challenge['evidence']))
    return {'contract_valid':not errors,'errors':errors,'boundary_review':boundary,'citation_bindings':citation_bindings,
            'canonical_argument':canonical,
            'reported_disposition':disposition,'effective_status':'OPEN_PENDING_CONTENT_REVIEW',
            'semantic_support':'NOT_INDEPENDENTLY_VERIFIED', 'world_state_authority':'NONE'}
