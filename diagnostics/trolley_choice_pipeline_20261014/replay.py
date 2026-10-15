"""Saved-answer wording/construction probes; no model calls or forced beam selection."""
import json
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blueprint_cloze_chooser import choose_by_cloze
from blueprint_primitive_composition import extract_inventory, append_composition
from blueprint_graph_amendments import expand_candidates
from blueprint_semantic_coverage import attach_coverage
from blueprint_spatial_projection import typed_reader_view
from parliament_source_constructions import build_packet, validate_packet
from run_blueprint_parliament import _admit_candidates, DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON
from test_blueprint_cloze_chooser import Script
import parsing_game_Z10 as z10


def run():
    out = Path(__file__).parent
    case = json.loads((ROOT/'fixtures/trolley_scene_cloze_cases.json').read_text())['cases'][0]
    variants = [('base', [] , ''),
                ('renamed_redirect', [('Maria','Dana'), ('trolley','tram'), ('switches','redirects')], ''),
                ('explicit_omission', [('If she does not,','If she does not pull the lever,')], ''),
                ('additional_signal', [], ' If Maria pulls the lever, the signal will activate.')]
    records=[]
    for name, substitutions, extra in variants:
        def transform(value):
            if isinstance(value,str):
                for a,b in substitutions:value=value.replace(a,b)
                return value
            if isinstance(value,dict):return {k:transform(v) for k,v in value.items()}
            return deepcopy(value)
        text=transform(case['source'])+extra
        folder=out/name;folder.mkdir(exist_ok=True)
        package=z10.export_candidate_graph(text,package_id=name)
        result=choose_by_cloze(text,Script(case['ranking'],transform(case['answers'])))
        inventory=extract_inventory(text,package)
        result=expand_candidates(text,package,result,inventory=inventory)
        result=append_composition(text,inventory,result)
        result=attach_coverage(inventory,result)
        chosen,attempts=_admit_candidates(result,text,package,folder,DEFAULT_PARLIAMENT_ROOT,DEFAULT_PARLIAMENT_PYTHON)
        graphs=[]
        for i,a in enumerate(result['candidate_attempts']):
            world=(a['proposal'].get('candidate') or {}).get('world_model')
            if not world:continue
            view=typed_reader_view(text,world)
            if view:
                graphs.append('## '+str(i)+' '+a['blueprint_id']+' / '+a.get('variant','baseline')+'\n\n```mermaid\n'+view['mermaid']+'\n```')
        (folder/'candidate_graphs.md').write_text('# '+name+'\n\n'+text+'\n\n'+'\n\n'.join(graphs)+'\n')
        record={'case':name,'scenario':text,'collapse_status':result.get('collapse_status'),'attempts':attempts,
                'selected_blueprint':chosen['attempt']['blueprint_id'] if chosen else None}
        if chosen:
            trace=json.loads(Path(chosen['record']['trace_path']).read_text())
            world=trace['action_source_grounding']['world_model']
            packet=build_packet(package,chosen['proposal'],world=world)
            validate_packet(packet,text,world=world)
            trace['source_construction_advisory']=packet
            (folder/'frozen_world_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
            (folder/'source_construction_advisory.json').write_text(json.dumps(packet,indent=2)+'\n')
            record['branch_choices']=(packet.get('branch_context') or {}).get('choices',[])
            record['effects']=len(world['effects'])
        (folder/'blueprint_result.json').write_text(json.dumps(result,indent=2)+'\n')
        records.append(record)
        print(name,record['collapse_status'],[(a['blueprint_id'],a.get('variant'),a['status']) for a in attempts],flush=True)
    (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':run()
