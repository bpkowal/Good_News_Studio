"""Verify served data equals the frozen world and live framework/review inputs."""
import hashlib
import json
from pathlib import Path
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
report=json.loads((HERE/'report/DEMO_REPORT.json').read_text())
data=json.load(urllib.request.urlopen('http://127.0.0.1:7860/data/thermal_blanket_promise.json'))
initial=json.loads((HERE/'frameworks/framework_generation.json').read_text())
review=json.loads((HERE/'review/targeted_review.json').read_text())
frozen=json.loads((HERE/'pipeline/frozen_world_trace.json').read_text())
source=(ROOT/'demo_scenarios/thermal_blanket_promise.txt').read_text().strip()
assert data['source']==report['scenario']==initial['scenario']==review['scenario']==source
assert (data['diagrams'][0]['world']==report['world']==initial['action_source_grounding']['world_model']
        ==review['action_source_grounding']['world_model']==frozen['action_source_grounding']['world_model'])
assert data['diagrams'][0]['sent_to_agents']
assert len(data['frameworks'])==5
assert '/Users/' not in json.dumps(data)
for framework in data['frameworks']:
    with urllib.request.urlopen('http://127.0.0.1:7860/'+framework['argument_url']) as response:
        assert response.status==200
result={'source_matches_all_stages':True,'world_matches_all_stages':True,
        'sent_to_five_frameworks':True,'browser_payload_http_status':200,
        'final_status':report['status'],
        'framework_adapter_calls':sum(r['adapter_calls'] for r in initial['framework_runs']),
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'open_review_issues':sum(i['objection_check']=='OPEN' for i in review['targeted_review']['issues']),
        'world_mutation':review['world_mutation']}
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
