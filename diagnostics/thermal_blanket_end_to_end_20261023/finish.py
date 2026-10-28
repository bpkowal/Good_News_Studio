"""Assemble a new saved report input without changing generation or review records."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
pipeline=HERE/'pipeline'
review=HERE/'review/targeted_review.json'
frameworks=HERE/'frameworks/framework_generation.json'
source=(ROOT/'demo_scenarios/thermal_blanket_promise.txt').read_text().strip()
frozen=pipeline/'frozen_world_trace.json'
paths=[frozen,review,frameworks]
before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
world=json.loads(frozen.read_text())['action_source_grounding']['world_model']
for p in (review,frameworks):
    payload=json.loads(p.read_text())
    assert payload['scenario']==source
    assert payload['action_source_grounding']['world_model']==world
    assert json.loads((p.parent/'run_manifest.json').read_text())['status']=='COMPLETED'
inputs=HERE/'report_input'
inputs.mkdir(exist_ok=False)
manifest=json.loads((pipeline/'preparation_manifest.json').read_text())
manifest['deliberation']={'trace':str(review),'mode':'LIVE_INDEPENDENT_FRAMEWORKS_PLUS_BOUNDED_REVIEW'}
manifest['report_input_assembly']={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                                 'immutable_inputs':before,'new_model_calls':0}
(inputs/'preparation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copyfile(frozen,inputs/frozen.name)
shutil.copyfile(review,inputs/'framework_generation.json')
shutil.copyfile(review,inputs/review.name)
code=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'run_parliament_demo.py'),
    '--scenario',str(ROOT/'demo_scenarios/thermal_blanket_promise.txt'),
    '--replay',str(inputs),'--output',str(HERE/'report'),
    '--checkpoint-tag','thermal-blanket-browser-report'],cwd=ROOT).returncode
assert code==0
assert before=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
print('Source, frozen world, initial arguments and native review preserved.')
