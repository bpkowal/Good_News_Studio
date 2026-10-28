"""One live local-repair request, saved separately from operative Parliament runs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from deontology_repair_cards import build_cards, request_repair, apply_repair, fingerprint, response_schema


def run(args):
    sys.path.insert(0, str(Path(args.parliament_root).resolve()))
    from dotenv import load_dotenv
    from openai import OpenAI
    from global_workspace.openai_backend import OpenAIWorkspaceLLM
    load_dotenv(Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env'))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.source)
    generation = json.loads(source.read_text())
    ledger = next(c for c in generation['cycles'][-1]['candidates']
                  if c['specialist']=='deontological')['committed_native_ledger']
    world = generation['action_source_grounding']['world_model']
    before_world, before_ledger = fingerprint(world), fingerprint(ledger)
    card = next(c for c in build_cards(world,ledger) if c.family==args.family)
    record = {'model':args.model, 'family':args.family, 'operative':False,
              'source':str(source), 'source_sha256':fingerprint(generation),
              'card':card.to_dict(), 'provider_attempts':0, 'status':'PREPARED'}

    def save():
        (output/'live_probe.json').write_text(json.dumps(record,indent=2)+'\n')

    class OneRequestBackend(OpenAIWorkspaceLLM):
        def _create(self, request):
            if record['provider_attempts']:
                raise RuntimeError('Single-request probe: automatic retry withheld')
            record.update(provider_attempts=1, request=deepcopy(request), status='REQUESTED')
            save()
            response = super()._create(request)
            record.update(provider_response=response.model_dump(mode='json'), status='RETURNED')
            save()
            return response

    save()
    started=time.monotonic()
    try:
        # Disable SDK retries as well as the native adapter's completion retry.
        backend=OneRequestBackend(args.model,timeout=90,client=OpenAI(max_retries=0))
        if args.inference_challenge:
            from inference_review import request_review, assess_response
            from inference_history import start_history, review_context_for, record_response
            challenge=json.loads(Path(args.inference_challenge).read_text())
            prior=json.loads(Path(args.prior_probe).read_text())
            if 'inference_response' in prior and not getattr(args,'history',None):
                raise ValueError('Resuming an inference review requires its history; budget cannot reset')
            if prior['card'] != json.loads(json.dumps(card.to_dict())):
                raise ValueError('Prior repair scope differs from original source card')
            if challenge['evidence'] != json.loads(json.dumps(card.evidence)) or not set(
                    challenge['affected_paths']) <= set(card.target_paths):
                raise ValueError('Challenge expands original evidence or mutation scope')
            record.update(challenge=challenge, prior_probe=str(args.prior_probe),
                          prior_probe_sha256=fingerprint(prior))
            if getattr(args,'history',None):
                history=json.loads(Path(args.history).read_text())
                record.update(parent_history=str(args.history),parent_history_sha256=fingerprint(history))
            else:
                history=start_history(before_world,challenge,prior['local_response'])
            context=review_context_for(history,challenge,prior['local_response'],before_world)
            record['revision_context']=context
            reviewed=request_review(challenge,prior['local_response'],response_schema(card),backend,
                                    revision_context=context)
            record['inference_response']=reviewed
            check=assess_response(challenge,prior['local_response'],reviewed)
            record['inference_contract']=check
            history=record_response(history,challenge,prior['local_response'],reviewed,
                                    native_audit={'status':'NOT_YET_ASSESSED'})
            (output/'inference_history_pending_native.json').write_text(json.dumps(history,indent=2)+'\n')
            save()
            if not check['contract_valid']:
                raise ValueError('Inference response bookkeeping invalid: '+str(check['errors']))
            response=reviewed['local_response']
            # Restage from the same immutable source. Preserve the prior staged
            # result as a parent artifact, rather than editing it in place.
        else:
            response=request_repair(card,backend,max_tokens=2048)
        record['local_response']=response
        staged,audit=apply_repair(world,ledger,card,response,scenario=generation['scenario'],
            action_grounding=generation['action_source_grounding']['actions'])
        record.update(status=audit['status'],audit=audit,staged_ledger=staged,
                      remaining_cards=[c.to_dict() for c in build_cards(world,staged)])
        if args.inference_challenge:
            # Recreate the response event with the final native audit. The
            # pending artifact above remains available if validation fails.
            parent=(json.loads(Path(args.history).read_text()) if getattr(args,'history',None)
                    else start_history(before_world,challenge,prior['local_response']))
            history=record_response(parent,challenge,prior['local_response'],reviewed,native_audit=audit)
            (output/'inference_history.json').write_text(json.dumps(history,indent=2)+'\n')
    except Exception as error:
        record.update(status='PROBE_FAILED',error_type=type(error).__name__,error=str(error))
    record.update(elapsed_seconds=round(time.monotonic()-started,3),
                  source_world_unchanged=fingerprint(world)==before_world,
                  source_ledger_unchanged=fingerprint(ledger)==before_ledger)
    save()
    print(json.dumps({k:record[k] for k in ('status','provider_attempts','elapsed_seconds',
                                          'source_world_unchanged','source_ledger_unchanged')}))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='diagnostics/lab_all_agents_20261019/frameworks/framework_generation.json')
    parser.add_argument('--parliament-root',default='.runtime/parliament')
    parser.add_argument('--family',choices=['PERSON_ROLE_CONSISTENCY','HARM_CLASSIFICATION'],default='PERSON_ROLE_CONSISTENCY')
    parser.add_argument('--model',default='o3')
    parser.add_argument('--inference-challenge')
    parser.add_argument('--prior-probe')
    parser.add_argument('--history',help='Reviewed history from the prior round; preserves the two-round budget')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if bool(args.inference_challenge) != bool(args.prior_probe):
        parser.error('--inference-challenge and --prior-probe must be supplied together')
    if args.history and not args.inference_challenge:
        parser.error('--history requires an inference challenge')
    run(args)
