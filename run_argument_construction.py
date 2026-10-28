"""Saved-source argument experiment; optional source-premise audit before critique."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import hashlib

from argument_construction import construct,write_report,source_from_trace
from deontology_repair_cards import fingerprint


def run(args):
    sys.path.insert(0,str(Path('.runtime/parliament').resolve()))
    fixture=json.loads(Path(args.source).read_text())
    if args.trace:
        framework=fixture['cases'][0]['framework']
        fixture={'origin':'VALIDATED_FROZEN_WORLD','cases':[source_from_trace(args.trace,framework)]}
    if args.case:
        fixture['cases']=[c for c in fixture['cases'] if c['id']==args.case]
        if len(fixture['cases'])!=1:raise ValueError('Unknown case')
    if args.saved_audit and not (args.premise_audit and args.saved_generation and args.candidate_index is not None):
        raise ValueError('Saved audit needs audit mode and one saved candidate')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    candidate_count=1 if args.candidate_index is not None else 2
    call_limit=(0 if args.saved_generation else 1)+candidate_count*(2 if args.premise_audit else 1)-(1 if args.saved_audit else 0)
    (output/'design.json').write_text(json.dumps({'fixture_sha256':fingerprint(fixture),
        'maximum_provider_requests':call_limit*len(fixture['cases']),'model':'o3',
        'premise_audit':args.premise_audit,
        'saved_candidate_index':args.candidate_index,
        'saved_generation_sha256':hashlib.sha256(Path(args.saved_generation).read_bytes()).hexdigest() if args.saved_generation else None,
        'saved_audit_sha256':hashlib.sha256(Path(args.saved_audit).read_bytes()).hexdigest() if args.saved_audit else None,
        'operative':False,'schema_stage':'canonical-argument-prototype',
        'code_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('argument_support.py','argument_construction.py','premise_audit.py','run_argument_construction.py','pipeline.lock.json')}},indent=2)+'\n')
    sys.path.insert(0,str(Path('.runtime/parliament').resolve()))
    from openai import OpenAI
    from dotenv import load_dotenv
    from global_workspace.openai_backend import OpenAIWorkspaceLLM
    load_dotenv('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env')
    results=[]
    for source in fixture['cases']:
        folder=output/source['id'];folder.mkdir()
        calls=[]
        def save():(folder/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
        class Recorded(OpenAIWorkspaceLLM):
            def complete_json(inner,prompt,**kwargs):
                if len(calls)>=call_limit:raise RuntimeError('Case call limit reached')
                calls.append({'prompt':prompt,'schema':kwargs['schema'],'status':'STARTED','provider_attempts':0});save()
                value=super().complete_json(prompt,**kwargs)
                calls[-1].update(status='RETURNED',response=deepcopy(value));save()
                return value
            def _create(inner,request):
                if calls[-1]['provider_attempts']:raise RuntimeError('No automatic retry')
                calls[-1].update(provider_attempts=1,request=deepcopy(request));save()
                response=super()._create(request)
                calls[-1]['provider_response']=response.model_dump(mode='json');save()
                return response
        started=time.monotonic()
        try:
            backend=Recorded('o3',timeout=120,client=OpenAI(max_retries=0))
            saved=None
            if args.saved_generation:
                if not args.case:raise ValueError('Saved generation needs a specific case')
                previous=json.loads(Path(args.saved_generation).read_text())[0]
                previous_source=json.loads(previous['prompt'].split('\n',1)[1])
                if fingerprint(previous_source)!=fingerprint(source):raise ValueError('Saved generation source changed')
                saved=json.loads(previous['response']['choices'][0]['text'])
                if args.candidate_index is not None:
                    saved={'candidates':[saved['candidates'][args.candidate_index]]}
            elif args.candidate_index is not None:raise ValueError('Candidate selection needs saved generation')
            audit=None
            if args.saved_audit:
                from argument_support import compile_argument
                from premise_audit import audit_packet
                candidate=saved['candidates'][0]
                state=compile_argument(source['world_sha256'],source['evidence'],source['id']+':0',
                                       candidate['statements'],candidate['supports'],partial=True)
                previous=json.loads(Path(args.saved_audit).read_text())[0]
                packet=json.loads(previous['prompt'].split('\n',1)[1])
                if fingerprint(packet)!=fingerprint(audit_packet(source,state,candidate['conclusion_handle'])):
                    raise ValueError('Saved premise audit source or argument changed')
                audit=json.loads(previous['response']['choices'][0]['text'])
            result=construct(source,backend,saved_generation=saved,premise_audit=args.premise_audit,saved_audit=audit)
        except Exception as error:result={'status':'FAILED','error_type':type(error).__name__,'error':str(error),'operative':False}
        result.update(case_id=source['id'],elapsed_seconds=round(time.monotonic()-started,3),
                      provider_requests=sum(c['provider_attempts'] for c in calls))
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        write_report(result,folder/'candidate_arguments.md')
        results.append(result)
        print(source['id'],[(r['label'],r['status']) for r in result.get('candidates',[])],flush=True)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    return {'cases':len(results),'provider_requests':sum(r['provider_requests'] for r in results)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='fixtures/argument_construction_heldout.json')
    parser.add_argument('--trace',help='Optional admitted frozen-world trace; validated through native replay before construction')
    parser.add_argument('--output',required=True)
    parser.add_argument('--case')
    parser.add_argument('--saved-generation')
    parser.add_argument('--candidate-index',type=int,choices=[0,1],help='Audit one saved proposal without regeneration')
    parser.add_argument('--premise-audit',action='store_true',help='Add source-applicability audit before each critique; at most five calls per case')
    parser.add_argument('--saved-audit',help='Replay an exact source/argument-bound audit and request only the local critique')
    print(json.dumps(run(parser.parse_args())))
