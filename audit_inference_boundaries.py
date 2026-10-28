"""Adversarial diagnostics; author-labeled review obligations, not truth tests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from inference_boundaries import boundary_card


def scenarios(fixture):
    cases={c['id']:c for c in fixture['cases']}; rows=[]
    def add(name,case,wanted,lesson,clean_error=False):
        rows.append(dict(id=name,mapping=case['inference_map'],evidence=case['evidence'],
                         review_finding_expected=wanted,clean_error_expected=clean_error,lesson=lesson))
    c=deepcopy(cases['contribution_without_awareness'])
    base=c['inference_map']['claims'][0]
    for cid,kind in [('goal','PURPOSE'),('belief','BELIEF'),('use','PURPOSIVE_USE')]:
        c['inference_map']['claims'].insert(-1,{**deepcopy(base),'id':cid,'kind':kind})
    c['inference_map']['bridges'][0]['premise_ids']=['physical','goal','belief','use']
    add('kind_laundering',c,True,'Relabeling the same causal text as belief/use should not establish those claims.')
    c=deepcopy(cases['mistaken_means_belief'])
    text='The operator does not believe Jo can help and does not employ Jo for the rescue; Jo merely happens to be present.'
    c['evidence'][0]['value']['provenance'][0]['excerpt']=text
    for claim in c['inference_map']['claims']:
        for support in claim['supports']:support['quote']=text
        if claim['id']=='belief':claim['text']='The operator does not believe employing Jo will help.'
        if claim['id']=='use':claim['text']='The operator does not employ Jo for the rescue.'
    add('negated_practical_premises',c,True,'BELIEF/USE kinds do not encode affirmative versus denied claims.')
    c=deepcopy(cases['conditional_rescue_maxim'])
    for claim in c['inference_map']['claims']:
        if claim['id']!='maxim':claim['interpretation_mode']='ATTRIBUTED_INFERENCE'
    c['inference_map']['claims'][-1]['condition_ids']=[]
    c['inference_map']['bridges'][0]['assumption_ids']=[]
    add('indirect_conditional_promotion',c,True,'Conditional ancestry must propagate through intermediate claims.')
    add('matching_quote_overclaim',deepcopy(cases['provenance_purpose_not_exclusivity']),True,
        'An exact quote supporting purpose does not establish sole purpose.')
    c=deepcopy(cases['conditional_rescue_maxim'])
    c['inference_map']['claims'][-1]['interpretation_mode']='UNKNOWN'
    add('unknown_is_not_asserted_promotion',c,False,'An explicitly unknown conclusion is not an asserted unqualified conclusion.')
    c=deepcopy(cases['mistaken_means_belief']);c['inference_map']['claims'][0]['text']=None
    add('malformed_null_text',c,False,'Malformed input should give a controlled contract error.',True)
    c=deepcopy(cases['mistaken_means_belief']);c['inference_map']['bridges'][0]['premise_ids']=['missing']
    add('dangling_reference_control',c,False,'Dangling references should be rejected.',True)
    c=deepcopy(cases['mistaken_means_belief']);c['inference_map']['claims'][0]['supports'][0]['quote']='invented quotation'
    add('forged_quote_control',c,False,'Fabricated quote text should be rejected.',True)
    add('valid_conditional_control',deepcopy(cases['conditional_rescue_maxim']),False,'Conditional candidates should remain available.')
    add('valid_mistaken_belief_control',deepcopy(cases['mistaken_means_belief']),False,'Failed efficacy must not erase stipulated practical reasoning.')
    return rows


def audit(fixture):
    results=[]
    for row in scenarios(fixture):
        out=deepcopy(row)
        try:
            card=boundary_card(row['mapping'],row['evidence'])
            out.update(findings=card['findings'],exception_type=None)
            out['expectation_met']=not row['clean_error_expected'] and bool(card['findings'])==row['review_finding_expected']
        except Exception as error:
            out.update(findings=[],exception_type=type(error).__name__,error=str(error))
            out['expectation_met']=row['clean_error_expected'] and isinstance(error,ValueError)
        results.append(out)
    return {'origin':'AUTHOR_LABELED_ADVERSARIAL_DIAGNOSTICS_NOT_PREVALENCE_ESTIMATE',
            'api_calls':0,'cases':results,'expectations_met':sum(r['expectation_met'] for r in results),
            'gaps':[r['id'] for r in results if not r['expectation_met']]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=audit(json.loads(Path('fixtures/inference_boundary_contrasts.json').read_text()))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'expectations_met':result['expectations_met'],'cases':len(result['cases']),'gaps':result['gaps']}))
