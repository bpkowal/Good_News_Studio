"""Reassess exact saved provider replies with the current native validators."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.runtime/parliament'))
from parliament_run_provenance import record_execution,record_start,record_resource,record_artifacts,record_runtime
from parliament_runtime import check_runtime
from logic_puzzles_frameworks import generate,FRAMEWORKS

def main(argv):
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv);record_start(a.output_dir)
    ready=check_runtime();record_runtime(ready)
    if not ready['ready']:raise RuntimeError(ready['errors'])
    record_resource(a.trace);record_resource(a.replay/'run_manifest.json')
    controls=[]
    class SavedReply:
        model='saved_o3_reply'
        last_call_origin='SAVED_PROVIDER_RESPONSE_REPLAY'
        def __init__(self,name):
            self.name=name;self.calls=0;path=a.replay/name/'model_calls.json';record_resource(path)
            records=json.loads(path.read_text())
            if len(records)!=1 or records[0]['status']!='RETURNED':
                raise ValueError('This replay requires one intact saved reply per framework')
            self.response=records[0]['response']
            controls.append({'framework':name,'response_sha256':hashlib.sha256(json.dumps(self.response,sort_keys=True).encode()).hexdigest(),
                             'original_calls':len(records),'new_provider_calls':0})
        def complete_json(self,prompt,**kwargs):
            self.calls+=1
            if self.calls>1:raise AssertionError('Never generate a new repair in saved-response reassessment')
            return deepcopy(self.response)
    generate(a.trace,a.output_dir,FRAMEWORKS,SavedReply,call_budget=1,
             core_root=Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL'))
    control=a.output_dir/'replay_controls.json';control.write_text(json.dumps(controls,indent=2)+'\n')
    before=json.loads((a.replay/'framework_generation.json').read_text())
    after=json.loads((a.output_dir/'framework_generation.json').read_text())
    assert before['action_source_grounding']==after['action_source_grounding']
    differences=[]
    prior={c['specialist']:c for c in before['cycles'][-1]['candidates']}
    current={c['specialist']:c for c in after['cycles'][-1]['candidates']}
    for name in FRAMEWORKS:
        b=prior[name];c=current[name]
        row={'framework':name}
        for field in ('landscape_validation_errors','framework_validation_errors','factual_reversal_threshold',
                      'graph_update_proposal','expected_value_validation_errors','recommended_action',
                      'deontological_validation_errors','care_validation_errors'):
            if b.get(field)!=c.get(field):row[field]={'before':b.get(field),'after':c.get(field)}
        differences.append(row)
    path=a.output_dir/'validation_comparison.json';path.write_text(json.dumps(differences,indent=2)+'\n')
    record_artifacts([control,path])

if __name__=='__main__':
    record_execution(main,sys.argv[1:],entrypoint='replay_saved_framework_validation.py',output_flag='--output-dir',
                     defaults={'new_provider_calls':0,'response_origin':'saved_provider_replies'})
