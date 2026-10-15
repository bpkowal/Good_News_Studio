"""Native prompt transport probe with deliberately invalid fake replies; no API."""
import json
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime/parliament'))
from global_workspace.frozen_world_replay import load_frozen_world_trace
from global_workspace.local_specialists import CompactLocalSpecialist
from global_workspace.models import WorkspaceBroadcast

def main():
    path=Path(sys.argv[1]);source=json.loads(path.read_text())
    replay=load_frozen_world_trace(path,expected_scenario=source['scenario'])
    before=deepcopy(replay.action_source_grounding)
    class Capture:
        def __init__(self):self.prompts=[]
        def complete_json(self,prompt,**kwargs):
            self.prompts.append(prompt)
            return {'choices':[{'text':'{}'}]}
    records=[]
    for name in ('utilitarian','deontological','care','virtue','rawlsian'):
        backend=Capture()
        agent=CompactLocalSpecialist(name,backend,source_construction_advisory=deepcopy(replay.source_construction_advisory),
            canonical_action_records=list(replay.canonical_action_records))
        for cycle in (1,2):agent.evaluate(replay.scenario,list(replay.canonical_actions),WorkspaceBroadcast())
        normal=list(backend.prompts)
        contingency=WorkspaceBroadcast(constraint='CONTINGENCY_REVIEW',
            contingency_fallback_actions=tuple(replay.canonical_actions))
        agent.evaluate(replay.scenario,list(replay.canonical_actions),contingency)
        assert len(normal)>=4,(name,len(normal))
        assert len(backend.prompts)>len(normal),name
        assert all('BRANCH CHOICE INTERPRETATION' in p and 'omitted_intervention_id' in p
                   for p in backend.prompts),(name,len(backend.prompts))
        records.append({'framework':name,'normal_and_repair_prompts':len(normal),
            'contingency_and_repair_prompts':len(backend.prompts)-len(normal),'prompts':backend.prompts})
    assert before==replay.action_source_grounding
    Path(sys.argv[2]).write_text(json.dumps({'world_unchanged':True,'frameworks':records,
        'mode':'capture-only fake responses; not semantic evaluation'},indent=2)+'\n')
    print('All five native frameworks received branch context in primary, repair and contingency prompts.')

if __name__=='__main__':main()
