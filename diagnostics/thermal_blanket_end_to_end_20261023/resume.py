"""Replay one exact saved source-selection response, then preserve native admission.

No chooser calls, manual world choice, or world-atom edits. A changed comparison
or unsupported quotation leaves selection unresolved. Original runs remain intact.
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_blueprint_parliament import _admit_candidates,_write_world_topology
from parliament_source_constructions import build_packet
from parliament_run_provenance import record_execution,record_resource,record_configuration,current_provenance

ORIGIN=ROOT/'diagnostics/thermal_blanket_demo_o3_fixed_live_20261023/pipeline'
OUT=Path(__file__).resolve().parent/'pipeline'


def resume(_):
    original=ORIGIN/'blueprint_candidate.json'
    review_path=ORIGIN/'selection_review.json'
    for path in (original,review_path):record_resource(path)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (original,review_path)}
    payload=json.loads(original.read_text())
    source=payload['scenario'];package=payload['package'];blueprint=deepcopy(payload['blueprint_result'])
    saved=json.loads(review_path.read_text())
    def replay(messages):
        request=json.loads(messages[0]['content'].split('INPUT:\n',1)[1])
        if request!=saved['request']:raise ValueError('Saved review source or comparison changed')
        return saved['raw_response']
    record_configuration(selection_mode='EXACT_SAVED_MODEL_RESPONSE_REPLAY',new_chooser_calls=0,
                         original_input_sha256=hashes)
    chosen,attempts=_admit_candidates(blueprint,source,package,OUT,ROOT/'.runtime/parliament',
                                    ROOT/'.runtime/venv/bin/python',selection_reviewer=replay)
    if chosen is None:raise ValueError('Replayed source review did not select an admitted candidate')
    proposal=chosen['proposal']
    target=OUT/'frozen_world_trace.json'
    shutil.copyfile(chosen['record']['trace_path'],target)
    frozen=json.loads(target.read_text())
    advisory=build_packet(package,proposal,world=frozen['action_source_grounding']['world_model'])
    if advisory:frozen['source_construction_advisory']=advisory
    target.write_text(json.dumps(frozen,indent=2)+'\n')
    (OUT/'source_construction_advisory.json').write_text(json.dumps(advisory,indent=2)+'\n')
    topology_md,topology_json=_write_world_topology(target,OUT)
    manifest=json.loads((ORIGIN/'preparation_manifest.json').read_text())
    manifest.update(pipeline_status='ADMITTED',chosen_blueprint_id=chosen['attempt']['blueprint_id'],
        actions=proposal['assignment'],parliament_admission=chosen['record']['admission'],
        admission_attempts=attempts,collapse_status=blueprint['collapse_status'],
        world_beam=blueprint['world_beam_path'],selection_review=blueprint['selection_review'],
        admitted_graph_comparison=blueprint['admitted_graph_comparison'],frozen_world_trace=str(target),
        world_state_topology_markdown=str(topology_md),world_state_topology_json=str(topology_json),
        selection_replay_provenance=current_provenance(),original_input_sha256=hashes,
        continuation_mode='EXACT_SAVED_SELECTION_REPLAY_NO_NEW_CHOOSER_CALLS')
    (OUT/'preparation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (original,review_path)}
    print('ADMITTED from exact saved source review:',target)
    return 0


if __name__=='__main__':
    raise SystemExit(record_execution(resume,['--output-dir',str(OUT)],
        entrypoint='resume_saved_selection.py',output_flag='--output-dir',
        defaults={'runtime_profile':'compact_frameworks'}))
