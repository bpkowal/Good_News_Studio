"""Exercise the pinned native retrieval path offline, without an LLM or credentials."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from parliament_runtime import check_runtime

PROBE = r'''
import importlib,json,sys
sys.path.insert(0,sys.argv[1])
from langchain_huggingface import HuggingFaceEmbeddings
from global_workspace.framework_retrieval import load_corpus_passages,retrieve_framework_evidence
from global_workspace.legacy_bridge import AGENT_MODULES
embedder=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':'cpu','local_files_only':True})
rows=[]
for framework,module_name in AGENT_MODULES.items():
 module=importlib.import_module(module_name)
 prefix={'utilitarian':'UTILITARIAN','deontological':'DEONTOLOGY','virtue':'VIRTUE','care':'CARE','rawlsian':'RAWLS'}[framework]
 corpus=getattr(module,prefix+'_CORPUS_DIR')
 lens=getattr(module,prefix+'_QUERY_LENS')
 tags=getattr(module,prefix+'_IDENTITY_TAGS')
 thresholds=getattr(module,prefix+'_EVIDENCE_THRESHOLDS')
 passages=load_corpus_passages(corpus,framework=framework,max_chars=250)
 result=retrieve_framework_evidence(passages,query='One dose can save either patient. A prior promise competes with a better chance of survival.',embedder=embedder,query_lens=lens,identity_tags=tags,thresholds=thresholds,limit=3,prefer_direct_quotes=True,separate_identity_scoring=True)
 rows.append({'framework':framework,'passages':len(passages),'retrieved':len(result.evidence),'tiers':[x.tier.value for x in result.evidence],'sources':[x.passage.source_file for x in result.evidence]})
assert all(r['passages'] and r['retrieved'] for r in rows),rows
print(json.dumps({'ready':True,'llm_calls':0,'frameworks':rows}))
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path)
 args=parser.parse_args()
 runtime=check_runtime(capability='full_relent')
 result={'ready':False,'runtime':runtime,'llm_calls':0}
 if runtime['ready']:
  completed=subprocess.run([runtime['python'],'-I','-B','-c',PROBE,runtime['parliament_root']],cwd=runtime['parliament_root'],env={**os.environ,**runtime.get('environment',{})},capture_output=True,text=True,timeout=300)
  result['exit_code']=completed.returncode
  if completed.returncode==0:
   result.update(json.loads(completed.stdout.splitlines()[-1]))
  else:
   result['error']=completed.stderr
 if args.output:
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
 return 0 if result['ready'] else 1

if __name__=='__main__':
 raise SystemExit(main())
