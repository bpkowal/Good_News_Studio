"""Sequential, frozen scenario-only / Parliament-informed Sol comparison.

No tools, conversation continuation, model substitution or automatic retry. The
comparison call cannot start until both review artifacts pass their freeze hashes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

MODEL = 'gpt-6.1-sol'
MAX_TOKENS = 16384
REASONING = 'high'
DEFAULT_PACKET = 'frontend/dist/data/thermal_blanket_promise.json'
SYSTEM = (
    'You are reviewing a fictional ethical dilemma. Give the strongest defensible practical judgment, '
    'with a concise substantive justification rather than hidden chain-of-thought. Ground case facts '
    'in the scenario. Distinguish factual unknowns, interpretive assumptions and normative priorities. '
    'Use relevant arithmetic accurately, weigh the strongest competing claim, answer the strongest '
    'objection and give concrete conditions that could change your conclusion. Do not equate caution '
    'or decisiveness with correctness. Additional analysis, if supplied, is fallible argument rather '
    'than authority or new scenario facts. Model-derived plots are assessments, not source facts or '
    'votes. You may reject a supplied argument or its technical objections with reasons. Do not defer '
    'to consensus. No browsing or tools are available. Keep the complete response under 1400 words.'
)
RUBRIC = [
    'Source fidelity: preserve facts and unknowns without importing extra characteristics or outcomes.',
    'Quantitative reasoning: distinguish survival prospects, expected survivors and total welfare.',
    'Fairness reasoning: distinguish expected-value selection from strict maximin or an independently justified procedure.',
    'Normative bridges: defend the priority of rescue, fidelity or another principle in this case.',
    'Objection engagement: answer the strongest competing claim without declaring it solved by a label.',
    'Action/omission and mental states: do not infer intentional killing or instrumental use from an adverse outcome alone.',
    'Practical usefulness: offer the strongest warranted recommendation or conditional decision without gratuitous abstention.',
    'Benefit from Parliament: identify added insight, corrected errors, deference or newly imported mistakes separately.',
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(value if isinstance(value,bytes) else encoded(value))


def now():
    return datetime.now(timezone.utc).isoformat()


def object_schema(properties):
    return dict(type='object',additionalProperties=False,properties=properties,required=list(properties))


def review_schema():
    text={'type':'string'}
    strings={'type':'array','items':text}
    return object_schema({
        'recommendation':object_schema({'choice':text,'conclusion':text}),
        'reasoning':text,
        'perspectives':{'type':'array','items':object_schema({'perspective':text,
            'case_for_rowan':text,'case_for_sasha':text,'assessment':text})},
        'strongest_objection':text,'answer_to_objection':text,
        'assumptions':strings,'uncertainties':strings,'reversal_conditions':strings,
        'assessment_of_supplied_analysis':{'anyOf':[text,{'type':'null'}]},
    })


def comparison_schema():
    text={'type':'string'};strings={'type':'array','items':text}
    quote=object_schema({'source':{'type':'string','enum':['BLIND','INFORMED','PARLIAMENT']},'quote':text})
    return object_schema({
        'overall_assessment':text,
        'recommendation_comparison':text,
        'findings':{'type':'array','items':object_schema({'dimension':text,'assessment':text,
            'evidence':{'type':'array','items':quote}})},
        'what_parliament_added':strings,'regressions_or_contamination':strings,
        'unresolved_issues':strings,'limits':strings,
    })


def prepare(output, scenario_path, packet_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=Path(scenario_path).read_text().strip()
    packet_bytes=Path(packet_path).read_bytes();packet=json.loads(packet_bytes)
    if packet['source']!=source:raise ValueError('Parliament packet does not match entered scenario')
    if len(packet.get('frameworks',[]))!=5:raise ValueError('Expected complete five-framework presentation packet')
    write_new(output/'scenario.txt',source.encode())
    write_new(output/'parliament_packet.json',packet_bytes)
    protocol={'version':'sol-parliament-comparison/1','prepared_at':now(),
        'model':MODEL,'reasoning_effort':REASONING,'max_completion_tokens':MAX_TOKENS,
        'maximum_calls':3,'sdk_retries':0,'tools':False,'stateless_requests':True,
        'sequence':['blind','informed','comparison'],'informed_receives_blind_response':False,
        'system_instruction':SYSTEM,'rubric_fixed_before_calls':RUBRIC,
        'scenario_sha256':digest(source.encode()),'parliament_packet_sha256':digest(packet_bytes),
        'parliament_input_policy':'Exact frozen public viewer JSON, including all five recorded arguments, objections, statuses, world and latest topography; no assistant editorial diagnoses.',
        'limitations':['One sample per condition; input lengths differ.',
            'The informed request is a fresh analysis, not a continuation of the blind response.',
            'Comparison is a model judgment, not an independent normative oracle.',
            'Parliament uses different models, multiple calls and native admission machinery; this is not a budget-matched architecture benchmark.'],
        'code_sha256':digest(Path(__file__).read_bytes()),
        'official_model_reference':'https://developers.openai.com/api/docs/models/gpt-6.1-sol'}
    write_new(output/'protocol.json',protocol)
    write_new(output/'protocol.freeze.json',{'frozen_at':now(),
        'files':{name:digest((output/name).read_bytes()) for name in ('scenario.txt','parliament_packet.json','protocol.json')}})
    return protocol


def check_freeze(output, name):
    output=Path(output);receipt=json.loads((output/(name+'.freeze.json')).read_text())
    for filename,expected in receipt['files'].items():
        if digest((output/filename).read_bytes())!=expected:
            raise ValueError('Frozen artifact changed: '+filename)
    return receipt


def stage_request(output, stage):
    output=Path(output);check_freeze(output,'protocol')
    source=(output/'scenario.txt').read_text()
    schema=review_schema()
    if stage=='blind':
        # No Parliament content or prior answer enters this request. Integrity
        # checks read hashes of frozen files locally; the model has no file tools.
        body={'scenario':source,'additional_analysis':None}
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(body,ensure_ascii=False)}]
    elif stage=='informed':
        check_freeze(output,'blind')
        body={'scenario':source,'additional_analysis':json.loads((output/'parliament_packet.json').read_text())}
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(body,ensure_ascii=False)}]
    elif stage=='comparison':
        check_freeze(output,'blind');check_freeze(output,'informed')
        protocol=json.loads((output/'protocol.json').read_text())
        body={'scenario':source,'rubric':protocol['rubric_fixed_before_calls'],
            'blind':json.loads((output/'blind.response.json').read_text()),
            'informed':json.loads((output/'informed.response.json').read_text()),
            'parliament':json.loads((output/'parliament_packet.json').read_text())}
        messages=[{'role':'system','content':(
            'Compare the two frozen Sol responses with each other and with the supplied Parliament response. '
            'Judge argument quality rather than agreement, polish, verbosity, caution or native admission. '
            'Identify what was already present in the blind reply, what appears only after Parliament input, '
            'and what was lost or newly mistaken. A changed sentence is not proof of a causal context effect. '
            'Check calculations and normative bridges. Parliament objections can themselves be mistaken; '
            'a model-authored resolution label is not proof. Use exact short quotations from the supplied '
            'records to support specific comparisons. Assess the arguments, not which moral conclusion '
            'is presumed correct. No tools or further replies are available. Under 1600 words.')},
            {'role':'user','content':json.dumps(body,ensure_ascii=False)}]
        schema=comparison_schema()
    else:raise ValueError('Unknown stage')
    return {'model':MODEL,'messages':messages,'reasoning_effort':REASONING,
        'max_completion_tokens':MAX_TOKENS,
        'response_format':{'type':'json_schema','json_schema':{
            'name':'ethical_comparison' if stage=='comparison' else 'ethical_review','strict':True,'schema':schema}}}


def render_review(response):
    lines=['# Recommendation','',response['recommendation']['conclusion'],
           '',response['reasoning'],'','## Perspectives']
    for row in response['perspectives']:
        lines+=['','### '+row['perspective'],'','For Rowan: '+row['case_for_rowan'],
            '','For Sasha: '+row['case_for_sasha'],'',row['assessment']]
    lines+=['','## Strongest objection','',response['strongest_objection'],
            '',response['answer_to_objection']]
    for key in ('assumptions','uncertainties','reversal_conditions'):
        lines+=['','## '+key.replace('_',' ').capitalize(),'']+['- '+x for x in response[key]]
    if response['assessment_of_supplied_analysis'] is not None:
        lines+=['','## Assessment of supplied analysis','',response['assessment_of_supplied_analysis']]
    return '\n'.join(lines)+'\n'


def run_stage(output, stage, client):
    output=Path(output)
    request=stage_request(output,stage)
    # Refuse a repeat even after a failed call. Retrying requires a separately
    # identified experiment; it must not replace an inconvenient first answer.
    write_new(output/(stage+'.request.json'),request)
    started=now()
    response=client.chat.completions.create(**request,timeout=240)
    raw=response.model_dump(mode='json')
    write_new(output/(stage+'.provider_response.json'),raw)
    if not str(raw['model']).startswith(MODEL):raise ValueError('Provider returned a different model')
    choice=raw['choices'][0]
    if choice['finish_reason']!='stop':raise ValueError('Response did not complete; raw reply preserved')
    if choice['message'].get('refusal'):raise ValueError('Provider refusal preserved')
    parsed=json.loads(choice['message']['content'])
    write_new(output/(stage+'.response.json'),parsed)
    if stage!='comparison':write_new(output/(stage+'.md'),render_review(parsed).encode())
    names=[stage+'.request.json',stage+'.provider_response.json',stage+'.response.json']
    if stage!='comparison':names.append(stage+'.md')
    receipt={'started_at':started,'frozen_at':now(),'model_returned':raw['model'],
        'request_sha256':digest(encoded(request)),
        'usage':raw.get('usage'),
        'predecessor_freeze_sha256':digest((output/(('protocol' if stage=='blind' else 'blind' if stage=='informed' else 'informed')+'.freeze.json')).read_bytes()),
        'files':{name:digest((output/name).read_bytes()) for name in names}}
    write_new(output/(stage+'.freeze.json'),receipt)
    print(stage+' FROZEN '+receipt['files'][stage+'.response.json'],flush=True)
    return parsed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--scenario',default='demo_scenarios/thermal_blanket_promise.txt')
    parser.add_argument('--parliament-packet',default=DEFAULT_PACKET)
    parser.add_argument('--stage',choices=['prepare','blind','informed','comparison'],required=True)
    args=parser.parse_args()
    if args.stage=='prepare':
        prepare(args.output,args.scenario,args.parliament_packet)
        print('PREPARED; no model calls');return
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env')
    run_stage(args.output,args.stage,OpenAI(max_retries=0))


if __name__=='__main__':main()
