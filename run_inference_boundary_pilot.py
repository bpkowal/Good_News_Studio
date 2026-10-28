"""Matched output-format pilot: equal inputs, ordinary vs typed bridge revision."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time

from deontology_repair_cards import fingerprint
from inference_boundaries import boundary_card
from inference_review import make_challenge,review_schema,assess_response

LOCAL_SCHEMA={'type':'object','additionalProperties':False,
              'properties':{'classification':{'type':'string','enum':['SUPPORTED','CONDITIONAL','UNRESOLVED','WITHDRAWN']}},
              'required':['classification']}


def prepare(case,spec):
    challenge=make_challenge(case['prior_reply'],claim_path='explanation',
        quote=case['prior_reply']['explanation'],defect_kind='MISSING_BRIDGE',
        missing_premise=spec['objection'],evidence=case['evidence'],
        affected_paths=['classification'],reviewer='EXPERIMENT_REVIEWER_MAY_BE_MISTAKEN')
    prompt=('Review the quoted argument against the exact supplied evidence. The reviewer may be mistaken. '
            'Defend, qualify, withdraw or leave the inference unresolved. Separate source evidence, '
            'interpretive assumptions, normative assumptions and unknowns. Preserve defensible claims '
            'and useful conditional analysis; do not equate caution with correctness or agreement with the reviewer. '
            'Do not manufacture facts or a moral verdict. Cite exact support, distinguish physical contribution '
            'from belief and purposive use, and state the strongest justified conclusion. '
            'The argument annotations can themselves be wrong. If your schema requests reasoning_revision, '
            'revise those annotations while retaining claim IDs and exact evidence locations.\n'+
            json.dumps({'challenge':challenge,'prior_reply':case['prior_reply'],
                        'argument_annotation':case['inference_map']},ensure_ascii=False))
    return challenge,prompt


def assess(case,challenge,response,arm):
    base=deepcopy(response)
    mapping=base.pop('reasoning_revision',None) if arm=='typed' else None
    result={'ordinary_contract':assess_response(challenge,case['prior_reply'],base),
            'native_admission':'NOT_ASSESSED_ARGUMENT_ONLY','semantic_support':'NOT_INDEPENDENTLY_ASSESSED'}
    if arm=='typed':
        try:
            result['typed_boundary']=boundary_card(mapping,case['evidence'])
            before={c['id'] for c in case['inference_map']['claims']}
            after={c['id'] for c in mapping['claims']}
            result['claim_ids_preserved']=before<=after
        except Exception as error:
            result['typed_error']=type(error).__name__+': '+str(error)
    return result


def run(args):
    manifest=json.loads(Path(args.manifest).read_text())
    fixture=json.loads(Path(args.fixtures).read_text())
    if fingerprint(fixture)!=manifest['fixture_sha256']:raise ValueError('Frozen fixtures changed')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    cases={c['id']:c for c in fixture['cases']}
    jobs=[];controls=[]
    for spec in manifest['cases']:
        case=cases[spec['case_id']];challenge,prompt=prepare(case,spec)
        controls.append({'case_id':case['id'],'prompt_sha256':fingerprint(prompt),
                         'input_sha256':fingerprint(case),'same_prompt_both_arms':True})
        for arm in manifest['arms']:
            folder=output/case['id']/arm;folder.mkdir(parents=True)
            record={'case_id':case['id'],'arm':arm,'prompt':prompt,'challenge':challenge,
                    'schema':review_schema(LOCAL_SCHEMA,typed_reasoning=arm=='typed'),
                    'provider_attempts':0,'operative':False,'status':'PREPARED'}
            (folder/'call.json').write_text(json.dumps(record,indent=2)+'\n')
            jobs.append((case,arm,folder,record))
    random.Random(manifest['order_seed']).shuffle(jobs)
    (output/'controls.json').write_text(json.dumps({'manifest':manifest,'controls':controls,
        'frozen_before_calls':True,'order':[(j[0]['id'],j[1]) for j in jobs]},indent=2)+'\n')
    if not args.execute:return {'status':'PREPARED','calls':0}
    sys.path.insert(0,str(Path(args.parliament_root).resolve()))
    from dotenv import load_dotenv
    from openai import OpenAI
    from global_workspace.openai_backend import OpenAIWorkspaceLLM
    load_dotenv('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env')
    def job_runner(job):
        case,arm,folder,record=job
        def save():(folder/'call.json').write_text(json.dumps(record,indent=2)+'\n')
        class OneCall(OpenAIWorkspaceLLM):
            def _create(inner,request):
                if record['provider_attempts']:raise RuntimeError('No automatic retries in pilot')
                record.update(provider_attempts=1,request=deepcopy(request),status='REQUESTED');save()
                raw=super()._create(request)
                record.update(provider_response=raw.model_dump(mode='json'),status='RETURNED');save()
                return raw
        started=time.monotonic()
        try:
            backend=OneCall(manifest['model'],timeout=90,client=OpenAI(max_retries=0))
            raw=backend.complete_json(record['prompt'],schema=record['schema'],
                                      max_tokens=manifest['max_tokens_argument'],temperature=0.0)
            response=json.loads(raw['choices'][0]['text'])
            record.update(response=response,assessment=assess(case,record['challenge'],response,arm),status='ASSESSED')
        except Exception as error:
            record.update(status='FAILED',error_type=type(error).__name__,error=str(error))
        record['elapsed_seconds']=round(time.monotonic()-started,3);save()
        print(case['id'],arm,record['status'],flush=True)
        return record
    with ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
        results=[future.result() for future in as_completed([pool.submit(job_runner,j) for j in jobs])]
    summary={'provider_attempts':sum(r['provider_attempts'] for r in results),
             'responses':sum(r['status']=='ASSESSED' for r in results),'operative':False,
             'manifest_sha256':fingerprint(manifest),'results':[{k:r[k] for k in
                ('case_id','arm','status','elapsed_seconds')} for r in results]}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='fixtures/inference_boundary_pilot_manifest.json')
    parser.add_argument('--fixtures',default='fixtures/inference_boundary_contrasts.json')
    parser.add_argument('--parliament-root',default='.runtime/parliament')
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true')
    print(json.dumps(run(parser.parse_args())))
