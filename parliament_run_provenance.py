"""Immutable execution manifests and source snapshots; no credentials or model calls."""
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

LOCAL_RUN = subprocess.run
ROOT = Path(__file__).resolve().parent
CURRENT = ContextVar('parliament_run', default=None)
VERSION = 'parliament-run/1'
SUFFIXES = {'.py','.json','.yaml','.yml','.toml','.lock','.sh','.js','.mjs','.html','.css','.patch','.md'}
EXCLUDED = {'diagnostics','eval_outputs','.runtime','.venv','.git','node_modules','__pycache__'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, root=ROOT):
    return LOCAL_RUN(['git','-C',str(root),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=30).stdout.decode().rstrip('\n')


def source_snapshot(root=ROOT, excluded_paths=()):
    """Capture executable/config inputs, including untracked work, once as bytes."""
    root=Path(root)
    names=git('ls-files','--cached','--others','--exclude-standard','-z',root=root).split('\0')
    files={}
    for name in sorted(set(names)):
        path=Path(name)
        if name.startswith('frontend/dist/data/') or not name or any(part in EXCLUDED for part in path.parts) or path.suffix not in SUFFIXES:
            continue
        if any(part.startswith('.env') or 'credential' in part.lower() or 'secret' in part.lower() for part in path.parts):
            continue
        file=root/path
        if any(file.resolve().is_relative_to(Path(excluded).resolve()) for excluded in excluded_paths):continue
        if file.is_file() and not file.is_symlink():
            files[name]=file.read_bytes()
    rows={name:sha(data) for name,data in files.items()}
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items():
            archive.writestr(name,data)
    changed=git('status','--porcelain=v1','-z','--untracked-files=all',root=root).split('\0')
    return {'commit':git('rev-parse','HEAD',root=root),
            'branch':git('rev-parse','--abbrev-ref','HEAD',root=root),
            'dirty':bool(any(changed)), 'working_tree_status':[r for r in changed if r],
            'source_files':rows, 'source_sha256':sha(json.dumps(rows,sort_keys=True).encode()),
            'archive_sha256':sha(bundle.getvalue())},bundle.getvalue()


def current_provenance():
    record=CURRENT.get()
    if not record:
        return {'status':'UNKNOWN_HISTORICAL','reason':'No execution manifest was recorded.'}
    summary={key:record[key] for key in ('schema_version','run_id','parent_run_id','execution_path','mode','started_at','configuration','pipeline_lock_sha256','inputs','manifest_file','checkpoint_tags','checkpoints')}
    summary['application']={key:record['application'][key] for key in ('commit','branch','dirty','source_sha256','archive_sha256')}
    return summary




def record_checkpoint(stage, *, status='recorded', artifacts=()):
    """Append a stage receipt; tags describe execution, never semantic approval."""
    record=CURRENT.get()
    if record is None:return
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',stage):raise ValueError('Invalid checkpoint stage')
    folder=Path(record['_folder']);folder.mkdir(parents=True,exist_ok=True)
    evidence=[]
    for path in map(Path,artifacts):
        if not path.is_file():continue
        data=path.read_bytes();name=sha(data)+'_'+path.name
        snapshot=folder/'inputs'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists():
            with snapshot.open('xb') as stream:stream.write(data)
        evidence.append({'name':path.name,'sha256':sha(data),'snapshot':'inputs/'+name})
    event={'schema_version':'parliament-checkpoint/1','id':record['run_id']+':cp:'+str(len(record['checkpoints'])+1),
           'run_id':record['run_id'],'stage':stage,'status':status,
           'at':datetime.now(timezone.utc).isoformat(),'tags':list(record['checkpoint_tags']),'artifacts':evidence}
    record['checkpoints'].append(event)
    with (folder/record['checkpoint_log']['path']).open('a') as stream:stream.write(json.dumps(event)+'\n')


def checkpoint_arguments(argv):
    tags=json.loads(os.environ.get('PARLIAMENT_CHECKPOINT_TAGS','[]'))
    if not isinstance(tags,list):raise ValueError('Checkpoint tags must be a list')
    clean=[];index=0
    while index<len(argv):
        arg=argv[index]
        if arg=='--checkpoint-tag':
            if index+1>=len(argv):raise ValueError('--checkpoint-tag needs a value')
            tags.append(argv[index+1]);index+=2
        elif arg.startswith('--checkpoint-tag='):
            tags.append(arg.split('=',1)[1]);index+=1
        else:clean.append(arg);index+=1
    if any(not isinstance(t,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',t) for t in tags):
        raise ValueError('Checkpoint tags need 1–64 letters, digits, dots, underscores or hyphens')
    return clean,list(dict.fromkeys(tags))

def record_configuration(values=None, **extra):
    record=CURRENT.get()
    if record is None:return
    values=vars(values) if values is not None and not isinstance(values,dict) else values or {}
    allowed={'model','chooser_model','openai_model','agents','max_cycles','time_budget','agent_timeout',
             'use_rag','prepare_only','review','baseline','frameworks','core_root','parliament_root',
             'parliament_python','jackknife','annotate_hinges','prototype_deliberation',
             'review_depth','followup_rounds','max_rounds','execute'}
    record['configuration'].update({k:str(v) if isinstance(v,Path) else v for k,v in values.items() if k in allowed})
    record['configuration'].update(extra)


def record_runtime(runtime):
    if CURRENT.get() is not None:
        CURRENT.get()['runtime_check']=runtime
        record_checkpoint('runtime_checked',status='ready' if runtime.get('ready') else 'unavailable')


def record_resource(path):
    path=Path(path)
    if CURRENT.get() is not None and path.is_file():
        data=path.read_bytes()
        CURRENT.get().setdefault('resource_inputs',{})[sha(data)+'_'+path.name]={'sha256':sha(data),'snapshot':'inputs/'+sha(data)+'_'+path.name}
        CURRENT.get().setdefault('_resource_bytes',{})[sha(data)+'_'+path.name]=data



def record_artifacts(paths):
    if CURRENT.get() is not None:CURRENT.get().setdefault('_additional_outputs',[]).extend(str(p) for p in paths)

def record_start(folder):
    record=CURRENT.get()
    if record is None:return
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    archive_path=folder/'application_source.zip'
    with archive_path.open('xb') as stream:stream.write(record['_archive_bytes'])
    started={k:v for k,v in record.items() if not k.startswith('_')}
    with (folder/'run_started.json').open('x') as stream:json.dump(started,stream,indent=2);stream.write('\n')
    record_checkpoint('execution_started')

def argument_config(argv, defaults):
    # Explicit allowlist: never serialize environment values or unknown CLI options.
    result=dict(defaults)
    flags={'--model','--chooser-model','--openai-model','--parliament-root','--parliament-python',
           '--prepare-only','--review','--baseline','--prototype-deliberation','--jackknife','--annotate-hinges',
           '--review-depth','--followup-rounds','--max-rounds','--execute'}
    for index,arg in enumerate(argv):
        key,sep,value=arg.partition('=')
        if key not in flags:
            continue
        result[key[2:].replace('-','_')]=value if sep else (argv[index+1] if index+1<len(argv) and not argv[index+1].startswith('--') else True)
    return result


def option(argv, flag):
    for index,arg in enumerate(argv):
        if arg.startswith(flag+'='):return arg.split('=',1)[1]
        if arg==flag and index+1<len(argv):return argv[index+1]
    return None


def record_execution(function, argv, *, entrypoint, output_flag, defaults=None, in_place=False):
    """Capture before work; finalize on success, ordinary failure, or exception."""
    argv,tags=checkpoint_arguments(list(sys.argv[1:] if argv is None else argv))
    output=option(argv,output_flag)
    if not output:
        # Entrypoints with generated output directories set this flag before calling.
        return function(argv)
    target=Path(output).expanduser().resolve()
    report_before=target.read_bytes() if in_place and target.is_file() else None
    folder=target.parent/'provenance' if in_place else target
    run_id='run_'+uuid.uuid4().hex
    manifest_path=folder/(run_id+'.json' if in_place else 'run_manifest.json')
    # A rejected rerun cannot overwrite a previous run's evidence.
    if not in_place and folder.exists() and any(folder.iterdir()):
        if entrypoint=='run_parliament_demo.py':return function(argv)
        raise ValueError('Use a new empty output directory to preserve prior evidence.')
    application,archive=source_snapshot(excluded_paths=[folder])
    configuration=argument_config(argv,defaults or {})
    replay=option(argv,'--replay')
    mode='SAVED_REPLAY_NO_API' if replay or option(argv,'--answers') or '--annotate-hinges' in argv or entrypoint=='export_parliament_viewer.py' else 'LIVE'
    if entrypoint=='logic_puzzles_followup.py' and '--execute' not in argv:
        mode='SAVED_REPLAY_NO_API'
    path=('saved_replay' if replay else 'compact_demo' if entrypoint=='run_parliament_demo.py' else
          ('world_preparation' if '--prepare-only' in argv else 'full_relent') if entrypoint=='run_blueprint_parliament.py' else 'viewer_export' if entrypoint=='export_parliament_viewer.py' else 'independent_frameworks' if entrypoint=='logic_puzzles_frameworks.py' else 'targeted_review' if entrypoint in ('logic_puzzles_review.py','logic_puzzles_followup.py') else 'report_postprocessing')
    inputs={}
    for flag in ('--scenario','--scenario-file','--report','--answers','--trace'):
        value=option(argv,flag)
        if value and Path(value).is_file():inputs[flag[2:]]={'sha256':sha(Path(value).read_bytes())}
    text=option(argv,'--text')
    if text is not None:inputs['text']={'sha256':sha(text.encode())}
    if replay:
        origin=Path(replay)/'run_manifest.json'
        inputs['replay_origin']={'manifest_sha256':sha(origin.read_bytes()) if origin.is_file() else None,
                                'status':'RECORDED' if origin.is_file() else 'UNKNOWN_HISTORICAL'}
    lock=ROOT/'pipeline.lock.json'
    record={'schema_version':VERSION,'run_id':run_id,'parent_run_id':os.environ.get('PARLIAMENT_PARENT_RUN_ID'),
            'entrypoint':entrypoint,'execution_path':path,'mode':mode,
            'started_at':datetime.now(timezone.utc).isoformat(),'application':application,
            'configuration':configuration,'inputs':inputs,
            'pipeline_lock_sha256':sha(lock.read_bytes()) if lock.is_file() else None,
            'launcher':{'python':sys.version.split()[0],'executable':sys.executable,'platform':platform.platform(),'machine':platform.machine(),
                        'distributions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}},
            'status':'RUNNING','checkpoint_tags':tags,'checkpoints':[],
            'checkpoint_log':{'path':run_id+'_checkpoints.jsonl' if in_place else 'checkpoints.jsonl'},'manifest_file':manifest_path.name,'artifact_base':'..' if in_place else '.'}
    record['_archive_bytes']=archive
    record['_folder']=str(folder)
    record['configuration'].setdefault('runtime_profile', 'full_relent' if path=='full_relent' else 'compact_frameworks' if path in ('compact_demo','world_preparation') else None)
    token=CURRENT.set(record)
    previous_tags=os.environ.get('PARLIAMENT_CHECKPOINT_TAGS')
    os.environ['PARLIAMENT_CHECKPOINT_TAGS']=json.dumps(tags)
    previous=os.environ.get('PARLIAMENT_PARENT_RUN_ID')
    os.environ['PARLIAMENT_PARENT_RUN_ID']=run_id
    try:
        if in_place:
            record_checkpoint('execution_started')
            if target.is_file():record_checkpoint('input_recorded',artifacts=[target])
        code=function(argv)
        record['exit_code']=code if isinstance(code,int) else 0
        record['status']='COMPLETED' if record['exit_code']==0 else 'NONZERO_EXIT'
        return code
    except BaseException as exc:
        record['status']=('COMPLETED' if isinstance(exc,SystemExit) and exc.code in (None,0) else 'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'EXCEPTION')
        record['error_type']=type(exc).__name__
        raise
    finally:
        record_checkpoint('execution_finished',status=record['status'].lower())
        CURRENT.reset(token)
        if previous_tags is None:os.environ.pop('PARLIAMENT_CHECKPOINT_TAGS',None)
        else:os.environ['PARLIAMENT_CHECKPOINT_TAGS']=previous_tags
        record.pop('_folder',None)
        journal=folder/record['checkpoint_log']['path']
        record['checkpoint_log']['sha256']=sha(journal.read_bytes())
        if previous is None:os.environ.pop('PARLIAMENT_PARENT_RUN_ID',None)
        else:os.environ['PARLIAMENT_PARENT_RUN_ID']=previous
        folder.mkdir(parents=True,exist_ok=True)
        archive_path=folder/(run_id+'_source.zip' if in_place else 'application_source.zip')
        if not archive_path.exists():
            with archive_path.open('xb') as stream:stream.write(archive)
        record.pop('_archive_bytes',None)
        if report_before is not None:
            before=folder/(run_id+'_input_report.json')
            with before.open('xb') as stream:stream.write(report_before)
            record['input_report_snapshot']={'path':before.name,'sha256':sha(report_before)}
        for name,data in record.pop('_resource_bytes',{}).items():
            resource=folder/'inputs'/name;resource.parent.mkdir(parents=True,exist_ok=True)
            if resource.exists():
                if resource.read_bytes()!=data:raise ValueError('Resource snapshot changed: '+name)
            else:
                with resource.open('xb') as stream:stream.write(data)
        record['source_changed_during_run']=source_snapshot(excluded_paths=[folder])[0]['source_sha256']!=record['application']['source_sha256']
        record['finished_at']=datetime.now(timezone.utc).isoformat()
        paths=[target] if in_place and target.is_file() else sorted(folder.rglob('*'))
        record['artifacts']={str(p.relative_to(folder) if p.is_relative_to(folder) else p.name):sha(p.read_bytes())
                             for p in paths if p.is_file() and p!=manifest_path and p.suffix!='.zip' and 'provenance' not in p.relative_to(folder).parts} if not in_place else {target.name:sha(target.read_bytes())} if target.is_file() else {}
        base=folder.parent if in_place else folder
        for name in record.pop('_additional_outputs',[]):
            file=Path(name).resolve()
            if file.is_file():record['artifacts'][str(file.relative_to(base.resolve()))]=sha(file.read_bytes())
        record['source_archive']=archive_path.name
        with manifest_path.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')


def load_manifest(path):
    record=json.loads(Path(path).read_text())
    if record.get('schema_version')!=VERSION:
        raise ValueError('Unsupported execution manifest version')
    return record


def verify_manifest(path):
    path=Path(path);record=load_manifest(path);errors=[]
    archive=path.parent/record['source_archive']
    if not archive.is_file() or sha(archive.read_bytes())!=record['application']['archive_sha256']:
        errors.append('Source archive missing or changed')
    else:
        with zipfile.ZipFile(archive) as source:
            expected=record['application']['source_files']
            if set(source.namelist())!=set(expected):errors.append('Source archive inventory changed')
            for name,value in expected.items():
                if name not in source.namelist() or sha(source.read(name))!=value:errors.append('Source changed: '+name)
    for name,value in record.get('artifacts',{}).items():
        artifact=(path.parent/record.get('artifact_base','.')/name)
        if not artifact.is_file() or sha(artifact.read_bytes())!=value:errors.append('Artifact missing or changed: '+name)
    journal=record.get('checkpoint_log')
    if journal:
        file=path.parent/journal['path']
        if not file.is_file() or sha(file.read_bytes())!=journal['sha256']:errors.append('Checkpoint journal missing or changed')
        else:
            events=[json.loads(line) for line in file.read_text().splitlines()]
            if events!=record.get('checkpoints'):errors.append('Checkpoint journal differs from manifest')
        for event in record.get('checkpoints',[]):
            if event.get('schema_version')!='parliament-checkpoint/1':errors.append('Unsupported checkpoint version')
            for artifact in event.get('artifacts',[]):
                file=path.parent/artifact['snapshot']
                if not file.is_file() or sha(file.read_bytes())!=artifact['sha256']:errors.append('Checkpoint snapshot missing or changed: '+artifact['name'])
    prior=record.get('input_report_snapshot')
    if prior:
        file=path.parent/prior['path']
        if not file.is_file() or sha(file.read_bytes())!=prior['sha256']:errors.append('Input report snapshot missing or changed')
    for resource in record.get('resource_inputs',{}).values():
        file=path.parent/resource['snapshot']
        if not file.is_file() or sha(file.read_bytes())!=resource['sha256']:errors.append('Resource missing or changed: '+resource['snapshot'])
    return {'valid':not errors,'run_id':record['run_id'],'errors':errors,
            'note':'Integrity of saved bytes only; not semantic correctness or proof of reproducible model output.'}


def compare_manifests(left, right):
    first=load_manifest(left);second=load_manifest(right)
    fields={'execution_path':lambda r:r['execution_path'],'mode':lambda r:r['mode'],'checkpoint_tags':lambda r:r.get('checkpoint_tags',[]),
            'application_commit':lambda r:r['application']['commit'],
            'application_source':lambda r:r['application']['source_sha256'],
            'runtime_lock':lambda r:r.get('pipeline_lock_sha256'),
            'configuration':lambda r:r['configuration'],'inputs':lambda r:r['inputs'],
            'launcher':lambda r:r.get('launcher'),
            'native_runtime':lambda r:r.get('runtime_check')}
    return {'left':first['run_id'],'right':second['run_id'],
            'differences':{key:{'left':get(first),'right':get(second)} for key,get in fields.items() if get(first)!=get(second)}}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    verify=sub.add_parser('verify');verify.add_argument('manifest',type=Path)
    compare=sub.add_parser('compare');compare.add_argument('left',type=Path);compare.add_argument('right',type=Path)
    args=parser.parse_args()
    result=verify_manifest(args.manifest) if args.action=='verify' else compare_manifests(args.left,args.right)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result.get('valid',True) else 1)
