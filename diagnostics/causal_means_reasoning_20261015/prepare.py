"""Rebuild the saved ten-worker Cloze proposal and admit it, without model calls."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from parliament_run_provenance import record_execution,record_start,record_resource,record_runtime,record_checkpoint
from parliament_runtime import check_runtime,DEFAULT_PARLIAMENT_ROOT,DEFAULT_PARLIAMENT_PYTHON


def main(argv):
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args(argv)
    record_start(args.output_dir);runtime=check_runtime();record_runtime(runtime)
    if not runtime['ready']:raise RuntimeError(runtime['errors'])
    from blueprint_cloze_chooser import _graph
    from run_blueprint_parliament import _admit_candidates
    from blueprint_spatial_projection import typed_reader_view
    source=ROOT/'diagnostics/tram_ten_workers_maxims_live_20261015_network/pipeline/blueprint_candidate.json'
    record_resource(source);raw=source.read_bytes();saved=json.loads(raw);prior=saved['blueprint_result']
    winner=next(row for row in prior['considered'] if row['blueprint_id']==prior['chosen_blueprint_id'])
    proposal=_graph(saved['scenario'],deepcopy(winner),deepcopy(prior['question']))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selected,attempts=_admit_candidates({'chosen_blueprint_id':proposal['blueprint_id'],'proposals':[proposal]},
        saved['scenario'],saved['package'],args.output_dir,DEFAULT_PARLIAMENT_ROOT,DEFAULT_PARLIAMENT_PYTHON)
    (args.output_dir/'admission_attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
    if selected is None:raise RuntimeError('Rebuilt candidate did not admit; inspect admission_attempts.json')
    shutil.copyfile(selected['record']['trace_path'],args.output_dir/'frozen_world_trace.json')
    trace=json.loads((args.output_dir/'frozen_world_trace.json').read_text());world=trace['action_source_grounding']['world_model']
    old=prior['graph']['candidate']['world_model'];new=proposal['candidate']['world_model']
    assert all(old[k]==new[k] for k in ('actions','effects','parties','conditions'))
    assert source.read_bytes()==raw
    rows=[{'source':a['source_id'],'target':a['target_id'],'before':a['link_relation'],'after':b['link_relation']}
        for a,b in zip(old['causal_links'],new['causal_links']) if a!=b]
    (args.output_dir/'relation_changes.json').write_text(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),
        'new_provider_calls':0,'changes':rows,'construction_provenance':proposal['construction_provenance']},indent=2)+'\n')
    view=typed_reader_view(saved['scenario'],world)
    (args.output_dir/'world_state_topology.md').write_text('# Rebuilt ten-worker world\n\n```mermaid\n'+view['mermaid']+'\n```\n')
    (args.output_dir/'world_state_topology.json').write_text(json.dumps(view,indent=2)+'\n')
    record_checkpoint('world_available',artifacts=[args.output_dir/'frozen_world_trace.json',args.output_dir/'relation_changes.json'])
    print('ADMITTED: '+str(args.output_dir/'frozen_world_trace.json'))

if __name__=='__main__':
    record_execution(main,sys.argv[1:],entrypoint='causal_means_saved_cloze_rebuild.py',output_flag='--output-dir',defaults={'new_provider_calls':0,'runtime_profile':'compact_frameworks'})
