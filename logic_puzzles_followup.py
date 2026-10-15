"""Opt-in, bounded follow-ups to revised arguments; no automatic moral verdict."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

from logic_puzzles_frameworks import fingerprint


def significant_questions(assessment):
    """First supported follow-up: application of a prohibition to foreseen harm.

    This identifies a decision-relevant question, not a proven contradiction.
    More constructions can be added without rewriting the admitted world.
    """
    from run_logic_puzzles import project_conflicts
    graph=project_conflicts(assessment)
    claims=[n for n in graph['nodes'] if n['kind']=='CLAIM']
    questions=[]
    for claim in claims:
        row=claim['record']
        if claim['origin']!='deontological':
            continue
        application=(row.get('verdict')=='PROHIBITED' and row.get('duty_type')=='PERFECT_NEGATIVE'
                     and row.get('means_relation')=='FORESEEN_SIDE_EFFECT')
        premise_errors=[e for e in row.get('calibration_errors',[])
                        if re.search(r'means|duty|premise|harm',str(e),re.I)]
        if not application and not premise_errors:continue
        action=row.get('canonical_action_id')
        world=assessment['action_source_grounding']['world_model']
        adverse=[e for e in world.get('effects',[]) if e.get('action_id')==action and e.get('polarity')=='ADVERSE']
        if not adverse:
            continue
        intent_norm=bool(re.search(r'\bintentional\w*\b',str(row.get('norm','')),re.I))
        if not application:
            question=('Your revised argument retains these decision-relevant premise questions: '
                      + '; '.join(premise_errors)+'. Address the specific missing inference, '
                      'revise the affected classification or preserve uncertainty. Do not invent '
                      'a causal path, intention, consent, or normative priority to remove the objection.')
            kind='NATIVE_PREMISE'
        else:
            kind='NORM_APPLICATION'
            question=(
            f'For {action}, your revised interpretation classifies the death or harm as a foreseen side effect, '
            'not an intended means. Distinguish intending the intervention, intending harm as an end, '
            'intending harm as a means, and foreseeing harm. Not being a means does not establish '
            'that an outcome is unintended in every respect. '
            + ('Your governing norm prohibits intentional harm or killing. Explain why the action falls '
               'under that norm despite your revised means classification. ' if intent_norm else
               'Explain the independent prohibition that you believe covers this foreseen harm. ')
            + 'If another prohibition governs foreseeable lethal or serious harm, identify and defend '
            'it explicitly rather than substituting its label. Test the particular rescue maxim and '
            'the omission maxim against universalization and the strongest rescue-duty objection. '
            'Explain what justifies the duty priority in this case. The physical world establishes '
            'causal relationships, not the actor\'s actual intentions or the truth of the normative rule. '
            'You may retain or revise your conclusion, or acknowledge an unresolved application. '
            'A RESOLVED label alone does not settle this philosophical question.')
        questions.append({'id':'FOLLOWUP:'+fingerprint([claim['origin'],action,kind])[:16],
            'framework':claim['origin'],'action_id':action,'kind':kind,
            'significance':'DECISION_RELEVANT','status':'OPEN','authority':'INVESTIGATIVE_ONLY',
            'source':'SYSTEM_REVIEW_QUESTION','claim':deepcopy(claim),
            'question':question,'premise_fingerprint':fingerprint({k:row.get(k) for k in
                ('norm','priority_rule','reason','public_justification','verdict','means_relation','candidate_maxim')})})
    return questions[:2]


def followup_offer(assessment):
    questions=significant_questions(assessment)
    return {'version':'reasoning-followup/1','status':'AVAILABLE' if questions else 'NO_SUPPORTED_TRIGGER',
        'requires_user_opt_in':True,'maximum_additional_rounds':2,
        'maximum_model_calls_per_round':1,'questions':questions,
        'message':('A decision-relevant question remains in the revised argument. Review it more deeply? '
                   'Up to two additional model calls; the conclusion may remain unresolved.' if questions else
                   'No supported follow-up trigger was detected; this does not establish argument soundness.'),
        'semantic_resolution':'NOT_INDEPENDENTLY_VERIFIED'}


def ledger_signature(candidate):
    rows=deepcopy((candidate.get('committed_native_ledger') or {}).get('records',[]))
    for row in rows:
        for key in ('cycle','provenance'):row.pop(key,None)
    return fingerprint(sorted(rows,key=lambda row:str(row.get('assessment_node_id',''))))


def save_resume_state(source_path, output_dir, result, shared, quotes, *, artifacts_dir=None):
    """Retain the accepted reply, not a rejected revision, for native replay."""
    output_dir=Path(output_dir);source_path=Path(source_path)
    artifacts=Path(artifacts_dir) if artifacts_dir is not None else output_dir
    prior_state=source_path.parent/'review_resume.json'
    prior=json.loads(prior_state.read_text()).get('responses',{}) if prior_state.exists() else {}
    responses={}
    for candidate in result['cycles'][-1]['candidates']:
        owner=candidate['specialist'];response=None;origin=None
        reconciled=any(i['framework']==owner and i['review_status']=='RECONCILED'
                       for i in result.get('targeted_review',{}).get('issues',[]))
        if reconciled:
            calls=artifacts/owner/'model_calls.json'
            if not calls.exists():calls=artifacts/'residual_round'/owner/'model_calls.json'
            if calls.exists():
                row=next((r for r in reversed(json.loads(calls.read_text())) if r.get('response')),None)
                if row:
                    response=deepcopy(row.get('normalized_response') or row['response']);origin=str(calls)
        if response is None and owner in prior and prior[owner].get('ledger_signature')==ledger_signature(candidate):
            response=deepcopy(prior[owner]['response']);origin=prior[owner].get('origin')
        if response is None:
            replay=artifacts/owner/'prior_replay.json'
            if replay.exists():response=json.loads(replay.read_text()).get('response');origin=str(replay)
            else:
                calls=source_path.parent/owner/'model_calls.json'
                if calls.exists():
                    row=next((r for r in reversed(json.loads(calls.read_text())) if r.get('response')),None)
                    if row:response=deepcopy(row.get('normalized_response') or row['response']);origin=str(calls)
        if response:
            responses[owner]={'response':response,'response_sha256':fingerprint(response),
                'ledger_signature':ledger_signature(candidate),'origin':origin}
    for name,value in [('shared_input.json',shared),('framework_commitments.json',quotes),
                       ('review_resume.json',{'responses':responses})]:
        (output_dir/name).write_text(json.dumps(value,indent=2)+'\n')


def stage_historical_review(source_path, output_dir):
    """Copy a saved review into a new continuation folder; never edit its origin."""
    source_path=Path(source_path);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    result=json.loads(source_path.read_text());expected=result['shared_input_sha256']
    inputs=[]
    candidates=[source_path.parent/'shared_input.json',
                *sorted(source_path.parent.parent.glob('*/shared_input.json'))]
    for path in candidates:
        quotes=path.parent/'framework_commitments.json'
        if path.exists() and quotes.exists():
            shared=json.loads(path.read_text());pack=json.loads(quotes.read_text())
            if fingerprint(shared)==expected:inputs.append((shared,pack))
    if not inputs or len({fingerprint(pair) for pair in inputs})!=1:
        raise ValueError('Exact shared input and quote pack cannot be uniquely recovered')
    shared,quotes=inputs[0]
    save_resume_state(source_path,out,result,shared,quotes,artifacts_dir=source_path.parent)
    staged=out/'targeted_review.json';staged.write_bytes(source_path.read_bytes())
    (out/'resume_origin.json').write_text(json.dumps({'path':str(source_path.resolve()),
        'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest()},indent=2)+'\n')
    return staged


def _job(question, assessment):
    from logic_puzzles_review import candidate_maxim_review
    owner=question['framework'];claim=question['claim']
    candidate=next(c for c in assessment['cycles'][-1]['candidates'] if c['specialist']==owner)
    issue={'id':question['id'],'kind':'UNRESOLVED_CONFLICT','origin':owner,
        'label':question['question'],'conflict_type':'REVISED_ARGUMENT_QUESTION',
        'status':'OPEN','source':'SYSTEM_REVIEW_QUESTION','claim_ids':[claim['id']]}
    return {'framework':owner,'issue_id':'CHALLENGE:'+fingerprint(
                [question['id'],question['premise_fingerprint']])[:16],
        'prior_selection_status':candidate.get('selection_status'),
        'issues':[issue],'claims':[deepcopy(claim)],'support_and_dependencies':[],
        'prior_challenge_response':deepcopy(candidate.get('challenge_response') or {}),
        'question':question['question'],'maxim_review':candidate_maxim_review(candidate,
            assessment['action_source_grounding']['world_model'])}


def deepen(assessment_path, output_dir, backend_factory, *, opt_in=False, max_rounds=2):
    if type(max_rounds) is not int or not 1<=max_rounds<=2:
        raise ValueError('Choose one or two additional rounds')
    from logic_puzzles_review import review
    source=Path(assessment_path);original_bytes=source.read_bytes();current=json.loads(original_bytes)
    world=deepcopy(current['action_source_grounding']);out=Path(output_dir)
    if (out/'reasoning_followup.json').exists() or list(out.glob('followup_*')):
        raise ValueError('Use a new output folder to preserve earlier follow-up evidence')
    out.mkdir(parents=True,exist_ok=True)
    history=[];seen=set();source_path=source
    stop=('USER_OPT_IN_REQUIRED' if significant_questions(current) else 'NO_SUPPORTED_TRIGGER') if not opt_in else 'ROUND_LIMIT'
    if opt_in:
        for index in range(max_rounds):
            questions=significant_questions(current)
            if not questions:stop='NO_SUPPORTED_TRIGGER';break
            if not (source_path.parent/'review_resume.json').exists():
                source_path=stage_historical_review(source_path,out/'resume_input')
            fresh=[q for q in questions if (q['id'],q['premise_fingerprint']) not in seen]
            if not fresh:stop='NO_SUBSTANTIVE_CHANGE';break
            question=fresh[0];signature=(question['id'],question['premise_fingerprint'])
            seen.add(signature)
            resume_path=source_path.parent/'review_resume.json'
            if not resume_path.exists():stop='RESUME_EVIDENCE_UNAVAILABLE';break
            resume=json.loads(resume_path.read_text()).get('responses',{});owner=question['framework']
            candidate=next(c for c in current['cycles'][-1]['candidates'] if c['specialist']==owner)
            prior=resume.get(owner)
            if (not prior or prior.get('ledger_signature')!=ledger_signature(candidate)
                    or fingerprint(prior.get('response'))!=prior.get('response_sha256')):
                stop='RESUME_EVIDENCE_MISMATCH';break
            folder=out/f'followup_{index+1:02d}'
            # Each round consumes the preceding admitted revision, never the opening.
            try:
                result=review(source_path,folder,backend_factory,max_calls=1,allow_partial=False,
                    allow_residual=False,jobs_override=[_job(question,current)],
                    prior_responses={owner:prior['response']},verify_prior=True)
            except RuntimeError as exc:
                if not str(exc).startswith('Prior reply does not reproduce'):
                    raise
                stop='PRIOR_STATE_REPLAY_MISMATCH';break
            if result['action_source_grounding']!=world:raise RuntimeError('Follow-up changed admitted world')
            answer=result['targeted_review']['issues'][0]
            history.append({'round':index+1,'question':question,'native_review':deepcopy(answer),
                'response_status':'RECORDED_NOT_INDEPENDENTLY_VERIFIED',
                'trace_path':str(folder/'targeted_review.json')})
            current=result;source_path=folder/'targeted_review.json'
            if answer['review_status']!='RECONCILED':stop='NATIVE_REVISION_NOT_ACCEPTED';break
        else:
            stop='ROUND_LIMIT' if significant_questions(current) else 'NO_SUPPORTED_TRIGGER'
    if source.read_bytes()!=original_bytes:raise RuntimeError('Source argument was modified')
    current['reasoning_followup']={'version':'reasoning-followup/1','user_opt_in':opt_in,
        'source_trace':{'path':str(source.resolve()),'sha256':hashlib.sha256(original_bytes).hexdigest()},
        'maximum_additional_rounds':max_rounds,'history':history,'stop_reason':stop,
        'remaining_questions':significant_questions(current),
        'semantic_resolution':'NOT_INDEPENDENTLY_VERIFIED','world_mutation':False,
        'total_adapter_calls':sum(json.loads((Path(h['trace_path']).parent/'targeted_review.json').read_text())
            ['targeted_review']['coverage']['total_adapter_calls'] for h in history)}
    current['follow_up_offer']=followup_offer(current)
    if opt_in and (source_path.parent/'shared_input.json').exists():
        shared=json.loads((source_path.parent/'shared_input.json').read_text())
        quotes=json.loads((source_path.parent/'framework_commitments.json').read_text())
        save_resume_state(source_path,out,current,shared,quotes)
    (out/'reasoning_followup.json').write_text(json.dumps(current,indent=2)+'\n')
    lines=['# Deeper review of revised arguments','','Stop: '+stop+'.',
        'These are attributed questions and replies, not established world facts or a verified moral verdict.','']
    for h in history:
        lines+=['## Follow-up '+str(h['round']),'',h['question']['question'],'',
                h['native_review']['response'].get('answer','No admitted answer.'),'',
                'Native revision: '+h['native_review']['review_status']+'.',
                'Philosophical adequacy: not independently verified.','']
    lines+=['## Remaining questions','']+[q['question'] for q in current['reasoning_followup']['remaining_questions']]
    (out/'reasoning_followup.md').write_text('\n'.join(lines)+'\n')
    return current


def _main(argv=None):
    from parliament_runtime import check_runtime,DEFAULT_PARLIAMENT_ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Opt in to additional live model calls')
    p.add_argument('--max-rounds',type=int,choices=(1,2),default=2)
    p.add_argument('--model',default='o3')
    p.add_argument('--parliament-root',type=Path,default=DEFAULT_PARLIAMENT_ROOT)
    args=p.parse_args(argv)
    from parliament_run_provenance import record_start,record_configuration,record_runtime
    record_start(args.output_dir);record_configuration(args)
    backend=None
    if args.execute:
        status=check_runtime(args.parliament_root);record_runtime(status)
        if not status['ready']:raise RuntimeError(status['errors'])
        sys.path.insert(0,str(args.parliament_root.resolve()))
        from dotenv import load_dotenv
        from global_workspace.openai_backend import OpenAIWorkspaceLLM
        load_dotenv(Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env'))
        backend=OpenAIWorkspaceLLM(args.model,timeout=120)
    result=deepen(args.trace,args.output_dir,lambda name:backend,opt_in=args.execute,max_rounds=args.max_rounds)
    print(result['reasoning_followup']['stop_reason']+': '+str(args.output_dir/'reasoning_followup.md'))


def main(argv=None):
    from parliament_run_provenance import record_execution
    return record_execution(_main,argv,entrypoint='logic_puzzles_followup.py',output_flag='--output-dir',
        defaults={'runtime_profile':'compact_frameworks','maximum_additional_rounds':2})

if __name__=='__main__':main()
