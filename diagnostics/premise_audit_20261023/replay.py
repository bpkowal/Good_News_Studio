"""Reapply exact saved audit and revision replies with zero provider calls."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from argument_construction import construct,write_report
from deontology_repair_cards import fingerprint

fixture=json.loads((ROOT/'fixtures/argument_construction_heldout.json').read_text())
results=[]
for case,index,stem in [('ineffective_volunteer',1,'volunteer'),('privacy_emergency',0,'privacy')]:
    source=next(c for c in fixture['cases'] if c['id']==case)
    generation=ROOT/'diagnostics/argument_construction_live_20261023'/case/'calls.json'
    audit=ROOT/('diagnostics/premise_audit_'+stem+'_live_20261023')/case/'calls.json'
    revision=ROOT/('diagnostics/premise_audit_'+stem+'_replay_20261023')/case/'calls.json'
    paths=[generation,audit,revision]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prior=json.loads(generation.read_text())[0]
    assert fingerprint(json.loads(prior['prompt'].split('\n',1)[1]))==fingerprint(source)
    proposal={'candidates':[json.loads(prior['response']['choices'][0]['text'])['candidates'][index]]}
    raw_audit=json.loads(json.loads(audit.read_text())[0]['response']['choices'][0]['text'])
    class Replay:
        count=0
        def complete_json(self,prompt,**kwargs):
            self.count+=1
            assert self.count==1
            return json.loads(revision.read_text())[0]['response']
    backend=Replay()
    result=construct(source,backend,saved_generation=proposal,premise_audit=True,saved_audit=raw_audit)
    saved_packet=json.loads(json.loads(audit.read_text())[0]['prompt'].split('\n',1)[1])
    assert fingerprint(saved_packet)==fingerprint(result['candidates'][0]['premise_audit_packet'])
    result['case_id']=case
    result['replay']={'provider_requests':0,'original_call_hashes':hashes,'saved_candidate_index':index}
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    folder=Path(__file__).resolve().parent
    (folder/(case+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    write_report(result,folder/(case+'.md'))
    results.append({'case':case,'status':result['candidates'][0]['status'],
                    'rejected_operations':result['candidates'][0].get('revision_problems',[])})
print(json.dumps(results,indent=2))
