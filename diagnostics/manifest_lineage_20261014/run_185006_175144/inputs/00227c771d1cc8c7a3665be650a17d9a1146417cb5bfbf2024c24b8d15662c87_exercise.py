"""Real subprocess replay/export lineage exercise; no model calls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime,timezone

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
from parliament_run_provenance import record_execution, record_start, record_resource, current_provenance, load_manifest, verify_manifest, sha
BASE=ROOT/'diagnostics/manifest_lineage_20261014'
OUT=BASE/('run_'+datetime.now(timezone.utc).strftime('%H%M%S_%f'))
SOURCE=ROOT/'diagnostics/live_pipeline_latest_20261006/medicine'
CATALOG=ROOT/'frontend/saved_runs.json'
reports=[ROOT/p for p in json.loads(CATALOG.read_text())['reports'].values()]
before={str(p):sha(p.read_bytes()) for p in reports}


def exercise(argv):
    record_start(OUT)
    record_resource(Path(__file__))
    record_resource(CATALOG)
    scenario=OUT/'scenario.txt'
    original=json.loads(reports[0].read_text())
    scenario.write_text(original['scenario'])
    commands=[
        [sys.executable,str(ROOT/'run_parliament_demo.py'),'--scenario',str(scenario),'--replay',str(SOURCE),'--output',str(OUT/'replay')],
        [sys.executable,str(ROOT/'export_parliament_viewer.py'),'--output',str(OUT/'viewer/data')],
    ]
    for index,command in enumerate(commands):
        with (OUT/f'child_{index}.log').open('w') as stream:
            subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
    replay_path=OUT/'replay/run_manifest.json'
    export_path=next(p for p in (OUT/'viewer/data/provenance').glob('run_*.json') if p.stem.count('_')==1)
    children=[load_manifest(p) for p in (replay_path,export_path)]
    root_id=current_provenance()['run_id']
    assert all(c['parent_run_id']==root_id for c in children)
    assert len({root_id,*[c['run_id'] for c in children]})==3
    checks=[verify_manifest(p) for p in (replay_path,export_path)]
    assert all(c['valid'] for c in checks),checks
    replay=json.loads((OUT/'replay/DEMO_REPORT.json').read_text())
    assert replay['execution']['commands']==[]
    assert replay['world']==original['world']
    assert all(sha(p.read_bytes())==before[str(p)] for p in reports)
    assert replay['generation_provenance']['status']=='UNKNOWN_HISTORICAL'
    with tempfile.TemporaryDirectory() as temporary:
        copy=Path(temporary)/'copy';shutil.copytree(OUT/'replay',copy)
        report=copy/'DEMO_REPORT.json';report.write_text(report.read_text()+'\n')
        tamper=verify_manifest(copy/'run_manifest.json')
        assert not tamper['valid'] and any('DEMO_REPORT.json' in e for e in tamper['errors'])
    result={'parent_run_id':root_id,'children':[{'run_id':c['run_id'],'parent_run_id':c['parent_run_id'],'execution_path':c['execution_path'],'mode':c['mode'],'manifest':str(p.relative_to(OUT))} for c,p in zip(children,(replay_path,export_path))],
            'child_integrity':checks,'tamper_detection':tamper,'admitted_world_unchanged':True,'original_reports_unchanged':True,'original_generation':'UNKNOWN_HISTORICAL','llm_api_calls':0}
    (OUT/'LINEAGE.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0

record_execution(exercise,['--output',str(OUT),'--replay',str(SOURCE)],entrypoint='offline_lineage_exercise',output_flag='--output')
parent_check=verify_manifest(OUT/'run_manifest.json')
assert parent_check['valid'],parent_check
summary={'output':str(OUT.relative_to(ROOT)),'parent_integrity':parent_check,**json.loads((OUT/'LINEAGE.json').read_text())}
(BASE/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
