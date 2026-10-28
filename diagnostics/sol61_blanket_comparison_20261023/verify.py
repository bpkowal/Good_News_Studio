"""Verify isolation, order, matching settings and all frozen provider artifacts."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from run_sol_parliament_comparison import check_freeze

protocol=check_freeze(HERE,'protocol')
design=json.loads((HERE/'protocol.json').read_text())
receipts={stage:check_freeze(HERE,stage) for stage in ('blind','informed','comparison')}
requests={stage:json.loads((HERE/(stage+'.request.json')).read_text()) for stage in receipts}
replies={stage:json.loads((HERE/(stage+'.response.json')).read_text()) for stage in receipts}
source=(HERE/'scenario.txt').read_text()
packet=json.loads((HERE/'parliament_packet.json').read_text())
assert json.loads(requests['blind']['messages'][1]['content'])=={'scenario':source,'additional_analysis':None}
assert json.loads(requests['informed']['messages'][1]['content'])=={'scenario':source,'additional_analysis':packet}
assert requests['blind']['messages'][0]==requests['informed']['messages'][0]
assert requests['blind']['messages'][0]['content']==design['system_instruction']
for key in ('model','reasoning_effort','max_completion_tokens','response_format'):
    assert requests['blind'][key]==requests['informed'][key]
for stage in requests:
    assert len(requests[stage]['messages'])==2 and 'tools' not in requests[stage]
    assert 'previous_response_id' not in requests[stage] and 'conversation' not in requests[stage]
    raw=json.loads((HERE/(stage+'.provider_response.json')).read_text())
    assert json.loads(raw['choices'][0]['message']['content'])==replies[stage]
    assert raw['model'].startswith('gpt-6.1-sol')
comparison=json.loads(requests['comparison']['messages'][1]['content'])
assert comparison['blind']==replies['blind'] and comparison['informed']==replies['informed']
assert comparison['parliament']==packet and comparison['rubric']==design['rubric_fixed_before_calls']
assert datetime.fromisoformat(receipts['blind']['frozen_at']) < datetime.fromisoformat(receipts['informed']['started_at'])
assert datetime.fromisoformat(receipts['informed']['frozen_at']) < datetime.fromisoformat(receipts['comparison']['started_at'])
assert hashlib.sha256((ROOT/'run_sol_parliament_comparison.py').read_bytes()).hexdigest()==design['code_sha256']
controls={'model':'gpt-6.1-sol','model_calls':3,'blind_input_scenario_only':True,
    'informed_did_not_receive_blind_answer':True,'same_review_instructions_schema_effort_and_ceiling':True,
    'blind_frozen_before_informed_started':True,'both_frozen_before_comparison_started':True,
    'frozen_provider_responses_match_parsed_artifacts':True,'all_freeze_hashes_verified':True,
    'no_tools_or_conversation_continuation':True,
    'usage':{stage:receipts[stage]['usage'] for stage in receipts},
    'world_or_ledger_authority':'NONE'}
(HERE/'verification.json').write_text(json.dumps(controls,indent=2)+'\n')
print(json.dumps({k:v for k,v in controls.items() if k!='usage'},indent=2))
