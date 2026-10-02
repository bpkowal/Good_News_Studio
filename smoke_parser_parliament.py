import argparse, copy, json, runpy, subprocess, sys
from pathlib import Path
import parsing_game_Z10 as z10
from candidate_validation import empty_selection, validate_candidate_selection
from test_scope_semantics import select
cli=argparse.ArgumentParser(description='Offline Z10/native Parliament boundary probe; development cases only.')
cli.add_argument('--parliament-root', type=Path, required=True)
cli.add_argument('--output', type=Path, default=Path('diagnostics/parliament_Z10_boundary_smoke.json'))
args=cli.parse_args()
root=args.parliament_root
revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
build=runpy.run_path(str(root/'global_workspace/stage_one_evidence.py'))['build_stage_one_evidence_packet']
cases=[json.loads(s) for s in Path('evals/dilemma_interpretation_v1/dev.jsonl').read_text().splitlines()]
report={'parliament_commit':revision,'scope':'offline development smoke; no semantic accuracy scoring, no heldout inference','parser_python':sys.version.split()[0],'cases':[]}
for case in cases:
 p=z10.export_candidate_graph(case['text'],package_id='smoke_'+case['id'])
 before=copy.deepcopy(p)
 checks=[validate_candidate_selection(p,empty_selection(p))]
 for c in p['candidates']:
  checks.append(validate_candidate_selection(p,select(p,c['id'])))
 packet=build(p)
 report['cases'].append({'id':case['id'],'candidates':len(p['candidates']), 'reconstructions':len(p['reconstructions']), 'selections_checked':len(checks),'valid_selections':sum(x['contract_valid'] for x in checks),'errors':[e for x in checks for e in x['errors']], 'unchanged':p==before,'exact_evidence':all(case['text'][e['start']:e['end']]==e['text'] for e in p['evidence']), 'direct_packet_observations':len(packet['source_observations']), 'direct_packet_hypotheses':len(packet['derived_hypotheses'])})
# Positive control exercises the existing Stage-1 API, not a Z10 adapter.
source={'neutral_skeleton':{'propositions':[{'proposition_id':'p1','outcome':'may leave','modality':'POSSIBLE','polarity':'POSITIVE','clause_ids':['c1']}]},'errors':['unresolved operator scope']}
snapshot=copy.deepcopy(source)
packet=build(source)
assert len(packet['source_observations'])==1 and packet['authority']=='ADVISORY_EVIDENCE_ONLY'
assert packet['unresolved']['validation_findings']==source['errors']
packet['source_observations'][0]['clause_ids'].append('mutation')
assert source==snapshot
report['native_packet_positive_control']='passed: observations, cautions, advisory authority, deep-copy isolation'
report['boundary_finding']='Raw Z10 packages are silently ignored by the native Stage-1 evidence builder; an explicit adapter/injection point is required.'
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

assert all(c['valid_selections']==c['selections_checked'] and c['unchanged'] and c['exact_evidence'] for c in report['cases'])
