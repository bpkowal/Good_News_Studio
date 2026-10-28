"""Export frozen boundary-card contrasts for inspection, without model calls."""
import argparse
import json
from pathlib import Path

from deontology_repair_cards import fingerprint
from inference_boundaries import boundary_card
from inference_review import make_challenge


def replay(source,output):
    source=Path(source); fixture=json.loads(source.read_text())
    rows=[]
    for case in fixture['cases']:
        card=boundary_card(case['inference_map'],case['evidence'])
        challenge=make_challenge(case['prior_reply'],claim_path='explanation',
            quote=case['prior_reply']['explanation'],defect_kind='MISSING_BRIDGE',
            missing_premise=case['reviewer_expected_distinction'],evidence=case['evidence'],
            affected_paths=['classification'],reviewer='AUTHOR_WRITTEN_CONTRACT_FIXTURE',
            inference_map=case['inference_map'])
        rows.append({'id':case['id'],'card':card,'challenge':challenge,
                     'expected_review_distinction':case['reviewer_expected_distinction'],
                     'structural_expectation_matches':[f['code'] for f in card['findings']]==case['expected_findings']})
    result={'api_calls':0,'operative':False,'origin':fixture['origin'],
            'fixture_sha256':fingerprint(fixture),'cards':rows,
            'semantic_support':'NOT_EVALUATED_BY_THIS_REPLAY'}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'boundary_cards.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Causal-role/intended-use boundary cards','','Offline export of eight author-written contrasts. '
           'No model responses, native admission or moral judgments are generated.','',
           '| Contrast | Structural review finding |', '| --- | --- |']
    lines+=['| '+r['id']+' | '+(', '.join(f['code'] for f in r['card']['findings']) or
                                'No structural bridge finding; semantic support remains unassessed')+' |' for r in rows]
    lines+=['','Mistaken instrumental belief and explicit purposeful employment remain representable without '
            'requiring an actual causal path. Conditional maxims remain conditional. A matching provenance '
            'quote verifies its location, not every qualifier in a cited claim. In particular, the purpose/exclusivity '
            'fixture deliberately illustrates an overstatement that still needs semantic content review.','',
            'Native validation remains unchanged. Its path requirement for intended-means classifications '
            'is recorded by a characterization test, not relaxed in this increment.','',
            'Inspect [typed claims, exact support locations, bridges and generated challenges](boundary_cards.json).']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',default='fixtures/inference_boundary_contrasts.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=replay(args.fixtures,args.output)
    print(json.dumps({'cards':len(result['cards']),'api_calls':0,
                      'all_structural_expectations_match':all(r['structural_expectation_matches'] for r in result['cards'])}))
