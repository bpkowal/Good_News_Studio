"""Reproject advisory context after the native-predicate display fix; no new world."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from parliament_run_provenance import record_execution,record_start,record_resource,record_artifacts,record_checkpoint
from parliament_source_constructions import build_packet,validate_packet

def run(argv):
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--proposal',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv);record_start(a.output_dir)
    record_resource(a.trace);record_resource(a.trace.parent/'run_manifest.json');record_resource(a.proposal)
    original=json.loads(a.trace.read_text());trace=deepcopy(original)
    candidate=json.loads(a.proposal.read_text())
    packet=build_packet(candidate['package'],candidate['blueprint_result']['graph'],
        world=trace['action_source_grounding']['world_model'])
    validate_packet(packet,trace['scenario'],world=trace['action_source_grounding']['world_model'])
    assert not packet['branch_context']['physical_scene']['open_questions']
    trace['source_construction_advisory']=packet
    assert {k:v for k,v in trace.items() if k!='source_construction_advisory'}=={
        k:v for k,v in original.items() if k!='source_construction_advisory'}
    output=a.output_dir/'frozen_world_trace.json';output.write_text(json.dumps(trace,indent=2)+'\n')
    verification=a.output_dir/'handoff_revision.json'
    verification.write_text(json.dumps({'original_trace':str(a.trace),'original_sha256':hashlib.sha256(a.trace.read_bytes()).hexdigest(),
        'world_unchanged':True,'reason':'Reproject exact source-backed continuation after native normalization omits predicate fields.',
        'new_model_calls':0,'original_trace_unchanged':True},indent=2)+'\n')
    record_artifacts([output,verification]);record_checkpoint('advisory_reprojected',artifacts=[output])

if __name__=='__main__':
    record_execution(run,sys.argv[1:],entrypoint='refresh_handoff.py',output_flag='--output-dir')
