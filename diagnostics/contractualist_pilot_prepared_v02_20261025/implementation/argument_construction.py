"""Bounded, nonvoting argument proposals and local critique over a frozen source."""
from copy import deepcopy
import json

from argument_support import OPERATIONS,compile_argument,project,apply_revision,assess_support
from deontology_repair_cards import fingerprint
from argument_support import ARGUMENT_KINDS
from argument_frameworks import SHARED_GENERATION, SHARED_CRITIQUE, framework_instruction
import html


def source_from_trace(path,framework):
    """Consume the same validated frozen-world boundary as native frameworks."""
    from pathlib import Path
    from global_workspace.frozen_world_replay import load_frozen_world_trace
    payload=json.loads(Path(path).read_text())
    replay=load_frozen_world_trace(path,expected_scenario=payload['scenario'])
    world=deepcopy(replay.action_source_grounding['world_model'])
    evidence=[{'ref':'SCENE','kind':'EXACT_SCENARIO','value':replay.scenario}]
    for effect in world.get('effects',[]):evidence.append({'ref':effect['effect_id'],'kind':'FROZEN_WORLD_COPY','value':effect})
    for index,link in enumerate(world.get('causal_links',[])):
        evidence.append({'ref':'LINK:'+str(index),'kind':'FROZEN_WORLD_COPY','value':link})
    return {'id':'frozen_'+fingerprint(world)[:12],'world_sha256':fingerprint(world),
            'source_status':'VALIDATED_FROZEN_WORLD','scenario':replay.scenario,
            'framework':framework,'evidence':evidence,'admitted_world':world,
            'source_trace_sha256':fingerprint(payload)}


def object_schema(props):
    return {'type':'object','additionalProperties':False,'properties':props,'required':list(props)}


def schemas():
    text={'type':'string','minLength':1};strings={'type':'array','items':text}
    citation=object_schema({'evidence_id':text,'value_path':{'type':'string'},'quote':text})
    statement=object_schema({'kind':{'type':'string','enum':list(ARGUMENT_KINDS)},'text':text,
        'polarity':{'type':'string','enum':['AFFIRMED','DENIED','UNKNOWN']},
        'modality':{'type':'string','enum':['ASSERTED','POSSIBLE','UNKNOWN']},
        'hypothesis_ids':strings,'citations':{'type':'array','items':citation}})
    declared=deepcopy(statement);declared['properties']['handle']=text;declared['required'].append('handle')
    route=object_schema({'operation':{'type':'string','enum':list(OPERATIONS)},'inputs':strings,
                         'output':text,'explanation':text})
    candidate=object_schema({'label':text,'statements':{'type':'array','items':declared,'minItems':2,'maxItems':12},
        'supports':{'type':'array','items':route,'minItems':1,'maxItems':16},
        'conclusion_handle':text,'strongest_objection':text,'answer_to_objection':text,'reversal_condition':text})
    proposal=object_schema({'candidates':{'type':'array','items':candidate,'minItems':1,'maxItems':2}})
    nullable=lambda spec:{'anyOf':[spec,{'type':'null'}]}
    revision=object_schema({'action':{'type':'string','enum':['assert','withdraw','supersede','qualify','add_support']},
        'target':nullable(text),'handle':nullable(text),'statement':nullable(statement),'support':nullable(route),'reason':text})
    critique=object_schema({'assessments':{'type':'array','items':object_schema({'support_id':text,
        'status':{'type':'string','enum':['SUPPORTED','CONDITIONAL','CHALLENGED','UNASSESSED']},'rationale':text})},
        'revisions':{'type':'array','items':revision,'maxItems':8},'strongest_remaining_conclusion':text,
        'remaining_questions':strings})
    return proposal,critique


def _response(backend,prompt,schema):
    result=backend.complete_json(prompt,schema=schema,max_tokens=4096,temperature=0.0)
    return json.loads(result['choices'][0]['text']) if 'choices' in result else result


def construct(source,backend,*,saved_generation=None,premise_audit=False,saved_audit=None):
    """One generation plus one isolated critique per candidate, at most 3 calls.

    Opt-in premise auditing adds one source-only review per candidate (at most 5
    total calls). Saved generation skips the first call in either mode.

    Content-critic assessments remain model judgments; they cannot vote or change
    the world. Raw proposals, revisions and compilation failures are retained.
    """
    generation_schema,critique_schema=schemas()
    prompt=(SHARED_GENERATION+'Construct one or at most two candidate interpretations, '
        'using alternatives only for material uncertainty. Framework instruction (not '
        'source evidence): '+framework_instruction(source)+'\n'+json.dumps(source))
    raw=deepcopy(saved_generation) if saved_generation is not None else _response(backend,prompt,generation_schema)
    if not isinstance(raw.get('candidates'),list) or not 1<=len(raw['candidates'])<=2:
        raise ValueError('One or at most two candidate interpretations required')
    if saved_audit is not None and (not premise_audit or len(raw['candidates'])!=1):
        raise ValueError('Saved audit requires premise-audit mode and one candidate')
    results=[]
    for index,candidate in enumerate(raw['candidates']):
        row={'label':candidate['label'],'raw_proposal':deepcopy(candidate),'operative':False}
        try:
            state=compile_argument(source['world_sha256'],source['evidence'],source['id']+':'+str(index),
                                   candidate['statements'],candidate['supports'],partial=True)
            if candidate['conclusion_handle'] not in state['handles']:raise ValueError('Unknown conclusion handle')
            row['compiled_before']=deepcopy(state)
            if premise_audit:
                from premise_audit import audit_packet,audit_schema,AUDIT_INSTRUCTION,bind_audit,apply_source_findings
                packet=audit_packet(source,state,candidate['conclusion_handle'])
                row['premise_audit_packet']=deepcopy(packet)
                audit_raw=(deepcopy(saved_audit) if saved_audit is not None else
                           _response(backend,AUDIT_INSTRUCTION+json.dumps(packet),audit_schema(packet)))
                row['raw_premise_audit']=deepcopy(audit_raw)
                audit=bind_audit(packet,audit_raw)
                row['premise_audit']=audit
                state=apply_source_findings(state,audit)
                row['audited_before']=deepcopy(state)
            critique_prompt=(SHARED_CRITIQUE+'Framework instruction (not source evidence): '
                +framework_instruction(source)+' '+(
                'Address the premise audit\'s precise findings rather than merely adding caveats. '
                'Separate factual applicability from moral authority. A rule cannot be declared violated when '
                'a required case fact is unknown. Preserve useful conditional branches with ALL unresolved '
                'factual antecedents and normative assumptions; do not replace unknown facts with their negations. '
                'Revise dependent conclusions and their support routes explicitly, retaining independent good '
                'reasoning. The audit is fallible; explain any disagreement with its source interpretation. '
                if premise_audit else '')+'\n'+
                json.dumps({'source':source,'candidate':candidate,'canonical_argument':state,'view':project(state),
                            'premise_audit':row.get('premise_audit')}))
            critique=_response(backend,critique_prompt,critique_schema)
            row['raw_critique']=deepcopy(critique)
            revision_problems=[]
            if premise_audit:
                revised=state
                for operation_index,operation in enumerate(critique['revisions']):
                    try:
                        revised=apply_revision(revised,[operation],author='MODEL_CRITIC',expected_sha256=fingerprint(revised))
                    except (ValueError,KeyError,TypeError) as error:
                        revision_problems.append({'index':operation_index,'proposal':deepcopy(operation),'error':str(error)})
            else:
                revised=apply_revision(state,critique['revisions'],author='MODEL_CRITIC',expected_sha256=fingerprint(state))
            for assessment in critique['assessments']:
                revised=assess_support(revised,assessment['support_id'],assessment['status'],
                    reviewer='MODEL_CRITIC_NOT_INDEPENDENT_ORACLE',rationale=assessment['rationale'])
            row.update(status='STAGED_PARTIAL_REVISION' if revision_problems else 'STAGED_ARGUMENT',
                       revision_problems=revision_problems,argument=revised,view=project(revised),
                       strongest_remaining_conclusion=critique['strongest_remaining_conclusion'],
                       remaining_questions=critique['remaining_questions'])
        except (ValueError,KeyError,TypeError) as error:
            row.update(status='UNRESOLVED_CONSTRUCTION',error=str(error))
            if 'compiled_before' in row:
                retained=row.get('audited_before',row['compiled_before'])
                row.update(status='STAGED_PRIOR_RETAINED',argument=retained,view=project(retained))
        results.append(row)
    return {'source_sha256':fingerprint(source),'world_sha256':source['world_sha256'],
            'candidates':results,'operative':False,'native_admission':'NOT_AUTHORIZED_BY_THIS_ARGUMENT_VIEW'}


def write_report(result,path):
    """Inspect successful and failed proposals without confusing them with worlds."""
    from pathlib import Path
    lines=['# Candidate arguments','','Nonvoting argument overlays. These are not admitted world-state graphs.']
    for candidate in result.get('candidates',[]):
        lines+=['','## '+candidate['label'],'',candidate['status']]
        if candidate.get('error'):lines+=['','Construction/revision issue: '+candidate['error']]
        if candidate.get('premise_audit'):
            lines+=['','Source-applicability audit (attributed model review, not proof):']
            for finding in candidate['premise_audit']['findings']:
                for component in finding['components']:
                    lines+=['','- '+component['status']+': '+component['claim_span']+' — '+component['reason']]
        lines+=['','Critic\'s proposed remaining conclusion (check accepted graph below): '+str(candidate.get('strongest_remaining_conclusion','Not independently established.'))]
        if candidate.get('revision_problems'):
            lines+=['','Rejected local operations:']
            for problem in candidate['revision_problems']:
                lines+=['','- Operation '+str(problem['index'])+': '+problem['error']]
        if 'view' not in candidate:continue
        view=candidate['view'];state=candidate['argument']
        ids={sid:'n'+str(i) for i,sid in enumerate(view['statements'])}
        lines+=['','```mermaid','flowchart TB']
        for sid,row in view['statements'].items():
            label=html.escape(row['kind']+' / '+row['polarity']+' / '+row['support_status']+': '+row['text'][:150],quote=True).replace('\n',' ')
            lines.append(ids[sid]+'["'+label+'"]')
        for rid,route in state['routes'].items():
            for parent in route['inputs']:
                lines.append(ids[parent]+' -->|'+route['operation']+' / '+view['support_routes'][rid]['status']+'| '+ids[route['output']])
        retired=[ids[sid] for sid,r in view['statements'].items() if r['lifecycle']!='ACTIVE']
        if retired:lines+=['classDef retired fill:#eeeeee,stroke:#777,color:#777','class '+','.join(retired)+' retired']
        lines+=['```','','Remaining questions: '+ '; '.join(candidate.get('remaining_questions',[]))]
        if state.get('construction_problems'):lines+=['','Original construction problems retained: '+str(len(state['construction_problems']))]
    Path(path).write_text('\n'.join(lines)+'\n')
