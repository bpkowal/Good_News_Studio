"""Canonical argument construction and revision. Never edits an admitted world.

Statements are immutable; lifecycle and support assessments live in events.
Model proposals and reviewed support are distinct. The compiler handles IDs,
citations, hypothesis propagation and alternative support routes.
"""
from copy import deepcopy

from deontology_repair_cards import fingerprint
from inference_boundaries import KINDS,value_at

VERSION='argument-support/0.1'
ARGUMENT_KINDS=tuple(KINDS)+('FACTUAL_PREMISE','INTERPRETIVE_PREMISE','PRINCIPLE',
                            'PERSONAL_REASON','EVALUATIVE_COMPARISON')
ASSUMPTION_KINDS=('MAXIM','NORMATIVE_PREMISE','INTERPRETIVE_PREMISE','PRINCIPLE')
OPERATIONS=('SOURCE','ASSUMPTION','PARAPHRASE','CALCULATION','PHYSICAL_CAUSATION','PRACTICAL_REASONING',
            'NORM_APPLICATION','NORM_PRIORITY')


def resolve_citation(raw,evidence):
    sources={e['ref']:e for e in evidence}
    if isinstance(raw,str):
        if raw in sources:ref,path=raw,''
        else:
            ref,sep,path=raw.partition(':')
            if not sep or ref not in sources or not path.startswith('/'):
                raise ValueError('Unknown evidence reference')
        quote=None
    elif isinstance(raw,dict) and set(raw)=={'evidence_id','value_path','quote'}:
        ref,path,quote=raw['evidence_id'],raw['value_path'],raw['quote']
    else:raise ValueError('Invalid citation shape')
    try:target=value_at(sources[ref]['value'],path)
    except (KeyError,IndexError,TypeError):raise ValueError('Unknown evidence location')
    if quote is not None and (not isinstance(quote,str) or not quote or not isinstance(target,str) or quote not in target):
        raise ValueError('Citation quote does not match exact location')
    return {'evidence_id':ref,'value_path':path,'quote':quote,'raw':deepcopy(raw),
            'resolved_value_sha256':fingerprint(target),'semantic_support':'NOT_ASSESSED'}


def new_argument(world_sha256,evidence,issue_id):
    if len({e['ref'] for e in evidence})!=len(evidence):raise ValueError('Duplicate evidence ID')
    return {'version':VERSION,'world_sha256':world_sha256,'evidence':deepcopy(evidence),
            'issue_id':issue_id,'statements':{},'handles':{},'routes':{},'events':[],
            'operative':False,'world_state_authority':'NONE'}


def _event(state,kind,payload):
    event={'sequence':len(state['events'])+1,'kind':kind,'payload':deepcopy(payload),
           'parent_sha256':state['events'][-1]['sha256'] if state['events'] else fingerprint(
               [state['version'],state['world_sha256'],state['evidence'],state['issue_id']])}
    event['sha256']=fingerprint(event);state['events'].append(event)


def _statement(state,handle,proposal):
    if not isinstance(handle,str) or not handle or handle in state['handles']:
        raise ValueError('Statement handle must be new')
    required={'kind','text','polarity','modality','hypothesis_ids','citations'}
    if not isinstance(proposal,dict) or set(proposal)!=required:raise ValueError('Invalid statement fields')
    if proposal['kind'] not in ARGUMENT_KINDS:raise ValueError('Unknown semantic domain')
    if not isinstance(proposal['text'],str) or not proposal['text'].strip():raise ValueError('Empty statement text')
    if proposal['polarity'] not in ('AFFIRMED','DENIED','UNKNOWN'):raise ValueError('Unknown polarity')
    if proposal['modality'] not in ('ASSERTED','POSSIBLE','UNKNOWN'):raise ValueError('Unknown modality')
    hypotheses=[state['handles'].get(h,h) for h in proposal['hypothesis_ids']]
    if any(h not in state['statements'] or state['statements'][h]['kind'] not in ASSUMPTION_KINDS for h in hypotheses):
        raise ValueError('Hypothesis must refer to an existing explicit assumption root')
    citations=[resolve_citation(c,state['evidence']) for c in proposal['citations']]
    row={**deepcopy(proposal),'hypothesis_ids':hypotheses,'citations':citations}
    # Raw transport aliases are provenance; canonical identity uses locations.
    identity={**row,'citations':[{k:v for k,v in c.items() if k!='raw'} for c in citations]}
    sid='STATEMENT:'+fingerprint([state['issue_id'],identity])[:24]
    if sid not in state['statements']:
        state['statements'][sid]=row;_event(state,'ASSERT',{'statement_id':sid})
    state['handles'][handle]=sid
    return sid


def _route(state,spec):
    if set(spec)!={'operation','inputs','output','explanation'} or spec['operation'] not in OPERATIONS:
        raise ValueError('Invalid support operation')
    inputs=[state['handles'].get(h,h) for h in spec['inputs']]
    output=state['handles'].get(spec['output'],spec['output'])
    if output not in state['statements'] or any(i not in state['statements'] for i in inputs):
        raise ValueError('Dangling support reference')
    if output in inputs:raise ValueError('Self-dependent support')
    if spec['operation']=='SOURCE' and (inputs or not state['statements'][output]['citations']):
        raise ValueError('Source route requires citations and no argument premises')
    if spec['operation']=='ASSUMPTION' and (inputs or state['statements'][output]['kind'] not in ASSUMPTION_KINDS):
        raise ValueError('Assumptions must be explicit interpretive/normative roots')
    row={**deepcopy(spec),'inputs':inputs,'output':output}
    rid='SUPPORT:'+fingerprint(row)[:24]
    state['routes'][rid]=row
    _event(state,'PROPOSE_SUPPORT',{'support_id':rid})
    # Validate the whole candidate graph, including alternative routes.
    project(state)
    return rid


def compile_argument(world_sha256,evidence,issue_id,statements,routes,*,partial=False):
    state=new_argument(world_sha256,evidence,issue_id)
    problems=[]
    for index,row in enumerate(statements):
        attempt=deepcopy(state)
        try:_statement(attempt,row['handle'],{k:v for k,v in row.items() if k!='handle'})
        except (ValueError,KeyError,TypeError) as error:
            if not partial:raise
            problems.append({'path':'statements/'+str(index),'proposal':deepcopy(row),'error':str(error)})
        else:state=attempt
    for index,row in enumerate(routes):
        attempt=deepcopy(state)
        try:_route(attempt,row)
        except (ValueError,KeyError,TypeError) as error:
            if not partial:raise
            problems.append({'path':'supports/'+str(index),'proposal':deepcopy(row),'error':str(error)})
        else:state=attempt
    # Citation binding is bookkeeping, not a semantic endorsement. Do not make
    # the model repeat SOURCE routes just to make its cited premises inspectable.
    for sid,row in state['statements'].items():
        if row['citations'] and not any(r['output']==sid for r in state['routes'].values()):
            _route(state,{'operation':'SOURCE','inputs':[],'output':sid,'explanation':'Compiler-generated source binding; semantic review required.'})
        elif not row['citations'] and row['kind'] in ASSUMPTION_KINDS and not any(r['output']==sid for r in state['routes'].values()):
            _route(state,{'operation':'ASSUMPTION','inputs':[],'output':sid,'explanation':'Explicit candidate interpretive/normative root, never a world fact.'})
    if partial:state['construction_problems']=problems
    project(state)
    return state


def apply_revision(state,operations,*,author,expected_sha256):
    if fingerprint(state)!=expected_sha256:raise ValueError('Stale argument revision')
    if not author:raise ValueError('Revision author required')
    result=deepcopy(state)
    for op in operations:
        action=op['action'];target=result['handles'].get(op.get('target'),op.get('target'))
        if action=='assert':
            _statement(result,op['handle'],op['statement'])
            if op.get('support') is not None:_route(result,op['support'])
        elif action in ('withdraw','supersede','qualify'):
            if action=='withdraw' and target in result['routes']:
                _event(result,'WITHDRAW_SUPPORT',{'support_id':target,'author':author,'reason':op['reason']})
                continue
            if target not in result['statements']:raise ValueError('Unknown revised statement')
            prior=project(result)['statements'][target]
            if prior['lifecycle']!='ACTIVE':raise ValueError('Inactive statement cannot be revised again')
            replacement=None
            if action!='withdraw':replacement=_statement(result,op['handle'],op['statement'])
            if replacement==target:raise ValueError('Semantic revision requires a different statement')
            _event(result,'WITHDRAW' if action=='withdraw' else 'SUPERSEDE',
                   {'statement_id':target,'replacement_id':replacement,'author':author,'reason':op['reason']})
            if action!='withdraw' and op.get('support') is not None:_route(result,op['support'])
        elif action=='add_support':_route(result,op['support'])
        else:raise ValueError('Unknown revision operation')
    project(result)
    return result


def assess_support(state,support_id,status,*,reviewer,rationale):
    if support_id not in state['routes'] or status not in ('SUPPORTED','CONDITIONAL','CHALLENGED','UNASSESSED'):
        raise ValueError('Invalid support assessment')
    if not reviewer or not rationale:raise ValueError('Attributed support assessment required')
    result=deepcopy(state)
    _event(result,'ASSESS_SUPPORT',{'support_id':support_id,'status':status,'reviewer':reviewer,
                                   'rationale':rationale,'authority':'ATTRIBUTED_REVIEW_NOT_WORLD_FACT'})
    return result


def compile_legacy_review(challenge,response,world_sha256):
    """Lossless old-response adapter: exact named withdrawals only, no prose NLP.

    Missing legacy polarity is UNKNOWN, not guessed. The returned overlay can
    repair lifecycle without rewriting the archived response or world graph.
    """
    mapping=response['reasoning_revision']
    rows=[]
    # Existing old fixtures declare their conditional roots first. Partial
    # compilation records unresolved forward/missing scope references explicitly.
    for claim in mapping['claims']:
        rows.append({'handle':claim['id'],'kind':claim['kind'],'text':claim['text'],
            'polarity':claim.get('polarity','UNKNOWN'),'modality':claim.get('modality','ASSERTED'),
            'hypothesis_ids':claim['condition_ids'],
            'citations':[{'evidence_id':c['evidence_ref'],'value_path':c['value_path'],'quote':c['quote']}
                         for c in claim['supports']]})
    operations={'SOURCE_TO_INTERPRETATION':'PARAPHRASE','CAUSAL_ROLE_TO_INTENDED_USE':'PRACTICAL_REASONING',
                'INTENDED_USE_TO_PROHIBITION':'NORM_APPLICATION','SOURCE_RULE_TO_NORMATIVE_PRIORITY':'NORM_PRIORITY'}
    routes=[{'operation':operations[b['rule_id']],'inputs':b['premise_ids']+b['assumption_ids'],
             'output':b['conclusion_id'],'explanation':b['explanation']} for b in mapping['bridges']]
    state=compile_argument(world_sha256,challenge['evidence'],challenge['challenge_id'],rows,routes,partial=True)
    targets={c['id'] for c in mapping['claims'] if c['text'] in response['withdrawn_claims']}
    if targets:
        state=apply_revision(state,[{'action':'withdraw','target':cid,'reason':'Exact claim explicitly named in saved withdrawn_claims.'}
                                    for cid in sorted(targets)],author='SAVED_MODEL_RETRACTION',expected_sha256=fingerprint(state))
    return state


def project(state):
    if state['version']!=VERSION:raise ValueError('Unknown argument version')
    lifecycle={sid:'ACTIVE' for sid in state['statements']};route_lifecycle={rid:'ACTIVE' for rid in state['routes']};assessments={}
    parent=fingerprint([state['version'],state['world_sha256'],state['evidence'],state['issue_id']])
    for sequence,event in enumerate(state['events'],1):
        if event['sequence']!=sequence or event['parent_sha256']!=parent or event['sha256']!=fingerprint({k:v for k,v in event.items() if k!='sha256'}):
            raise ValueError('Altered argument history')
        parent=event['sha256'];p=event['payload']
        if event['kind'] in ('WITHDRAW','SUPERSEDE'):lifecycle[p['statement_id']]='WITHDRAWN' if event['kind']=='WITHDRAW' else 'SUPERSEDED'
        elif event['kind']=='ASSESS_SUPPORT':assessments[p['support_id']]=p
        elif event['kind']=='WITHDRAW_SUPPORT':route_lifecycle[p['support_id']]='WITHDRAWN'
    # Statements/routes are immutable outside the event chain as well.
    for sid,row in state['statements'].items():
        if any(resolve_citation(c['raw'],state['evidence'])!=c for c in row['citations']):
            raise ValueError('Altered citation binding')
        identity={**row,'citations':[{k:v for k,v in c.items() if k!='raw'} for c in row['citations']]}
        if sid!='STATEMENT:'+fingerprint([state['issue_id'],identity])[:24]:raise ValueError('Altered statement')
    for rid,row in state['routes'].items():
        if rid!='SUPPORT:'+fingerprint(row)[:24]:raise ValueError('Altered support route')
    by_output={sid:[] for sid in state['statements']}
    for rid,row in state['routes'].items():by_output[row['output']].append((rid,row))
    visiting=set();views={};route_views={}
    def view(sid):
        if sid in visiting:raise ValueError('Cyclic support graph')
        if sid in views:return views[sid]
        visiting.add(sid);row=state['statements'][sid];options=[]
        for rid,route in by_output[sid]:
            if route_lifecycle[rid]!='ACTIVE':
                route_views[rid]={'status':'WITHDRAWN','issues':[],'effective_hypotheses':[]};continue
            parents=[view(i) for i in route['inputs']]
            hypotheses=set(row['hypothesis_ids'])
            for p in parents:hypotheses.update(p['effective_hypotheses'])
            status=assessments.get(rid,{}).get('status','UNASSESSED');issues=[]
            if route['operation']=='ASSUMPTION':
                if row['kind'] not in ASSUMPTION_KINDS or parents:raise ValueError('Assumptions must be explicit interpretive/normative roots')
                hypotheses.add(sid);status='CONDITIONAL'
            if any(p['lifecycle']!='ACTIVE' or p['support_status'] not in ('SUPPORTED','CONDITIONAL') for p in parents):
                issues.append('PREMISE_REQUIRES_REASSESSMENT')
            if any(lifecycle[h]!='ACTIVE' for h in hypotheses):issues.append('HYPOTHESIS_WITHDRAWN_OR_SUPERSEDED')
            if route['operation']=='CALCULATION':issues.append('NUMERIC_VERIFIER_REQUIRED')
            if route['operation']=='PRACTICAL_REASONING':
                affirmed={p['kind'] for p in parents if p['polarity']=='AFFIRMED' and p['modality']!='UNKNOWN'}
                if row['polarity']=='AFFIRMED' and not {'PURPOSE','BELIEF','PURPOSIVE_USE'}<=affirmed:
                    issues.append('AFFIRMATIVE_PRACTICAL_PREMISES_MISSING')
            if route['operation'] in ('NORM_APPLICATION','NORM_PRIORITY') and not any(p['kind']=='NORMATIVE_PREMISE' for p in parents):
                issues.append('NORMATIVE_PREMISE_MISSING')
            if route['operation'] in ('SOURCE','PARAPHRASE','PHYSICAL_CAUSATION') and row['modality']=='ASSERTED' and any(
                    p['kind']==row['kind'] and p['modality']=='POSSIBLE' for p in parents):
                issues.append('MODALITY_STRENGTHENED')
            if row['polarity']=='UNKNOWN' or row['modality']=='UNKNOWN':status='UNASSESSED'
            if issues:status='REQUIRES_REASSESSMENT'
            if status=='SUPPORTED' and (hypotheses or any(p['support_status']=='CONDITIONAL' for p in parents)):status='CONDITIONAL'
            route_views[rid]={'status':status,'issues':issues,'effective_hypotheses':sorted(hypotheses)}
            options.append(route_views[rid])
        rank={'SUPPORTED':0,'CONDITIONAL':1,'UNASSESSED':2,'CHALLENGED':3,'REQUIRES_REASSESSMENT':4}
        best=min(options,key=lambda p:rank[p['status']]) if options else {'status':'UNASSESSED','effective_hypotheses':row['hypothesis_ids']}
        status=best['status'] if lifecycle[sid]=='ACTIVE' else 'REQUIRES_REASSESSMENT'
        views[sid]={**deepcopy(row),'id':sid,'lifecycle':lifecycle[sid],'support_status':status,
                    'effective_hypotheses':best['effective_hypotheses']}
        visiting.remove(sid);return views[sid]
    for sid in state['statements']:view(sid)
    return {'statements':views,'support_routes':route_views,'active_statement_ids':[sid for sid in views if lifecycle[sid]=='ACTIVE'],
            'operative':False,'semantic_support':'ATTRIBUTED_ASSESSMENTS_NOT_AUTOMATIC_TRUTH'}
