"""Build persistent review history from saved probes and attributed content review.

No model calls. Outputs a new history and optional next challenge; never rewrites
the probe, admitted world or native ledger.
"""
import argparse
import json
from pathlib import Path

from deontology_repair_cards import fingerprint
from inference_review import make_challenge
from inference_history import (start_history,record_response,review_response,
    followup_state,dependency_projection,revision_context)


def replay(probe_path,review_path,output):
    probe=json.loads(Path(probe_path).read_text())
    prior=json.loads(Path(probe['prior_probe']).read_text())
    if fingerprint(prior)!=probe['prior_probe_sha256']:
        raise ValueError('Prior probe changed')
    review=json.loads(Path(review_path).read_text())
    if review['response_sha256']!=fingerprint(probe['inference_response']):
        raise ValueError('Content review is for another response')
    if probe.get('parent_history'):
        state=json.loads(Path(probe['parent_history']).read_text())
        if fingerprint(state)!=probe['parent_history_sha256']:
            raise ValueError('Parent history changed')
        if review.get('dependent_claims'):raise ValueError('New dependencies belong in reviewed claim revisions')
    else:
        state=start_history(probe['card']['world_sha256'],probe['challenge'],prior['local_response'],
                            dependent_claims=review.get('dependent_claims',[]))
    state=record_response(state,probe['challenge'],prior['local_response'],probe['inference_response'],
                          native_audit=probe['audit'])
    state=review_response(state,**{k:review[k] for k in
        ('response_sha256','reviewer','outcome','rationale','remaining_gap',
         'corrected_points','new_unsupported_claims','claim_updates','new_claims')})
    summary={'api_calls':0,'operative':False,'source_probe_sha256':fingerprint(probe),
             'content_review_sha256':fingerprint(review),'followup':followup_state(state),
             'argument_dependencies':dependency_projection(state)}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'inference_history.json').write_text(json.dumps(state,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if followup_state(state)['status']=='AVAILABLE':
        context=revision_context(state)
        next_spec=review['next_target']
        challenge=make_challenge(probe['local_response'],claim_path=next_spec['claim_path'],
            quote=next_spec['quote'],defect_kind=next_spec['defect_kind'],
            missing_premise=context['remaining_gap'],evidence=probe['challenge']['evidence'],
            affected_paths=probe['challenge']['affected_paths'],reviewer=review['reviewer'],
            inference_map=review.get('next_inference_map',probe['inference_response'].get('reasoning_revision')))
        (output/'next_challenge.json').write_text(json.dumps(challenge,indent=2)+'\n')
        (output/'next_context.json').write_text(json.dumps(context,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',required=True)
    parser.add_argument('--review',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(replay(args.probe,args.review,args.output)['followup']))
