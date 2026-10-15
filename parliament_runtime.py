"""Pin, reconstruct and inspect the native Parliament runtime without model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / 'pipeline.lock.json'
DEFAULT_PARLIAMENT_ROOT = ROOT / '.runtime/parliament'
DEFAULT_PARLIAMENT_PYTHON = ROOT / '.runtime/venv/bin/python'
DEFAULT_RELENT_PYTHON = ROOT / '.runtime/relent-venv/bin/python'

def python_spec(lock, capability):
    if capability == 'compact_frameworks':
        return lock['python']
    if capability not in lock.get('profiles', {}):
        raise ValueError('Runtime profile is not locked: '+capability)
    return lock['profiles'][capability]['python']

def profile_python(capability):
    return DEFAULT_RELENT_PYTHON if capability == 'full_relent' else DEFAULT_PARLIAMENT_PYTHON



def profile_environment(lock, capability, workspace=ROOT):
    """Only declared runtime settings; never read credentials."""
    settings=lock.get('profiles', {}).get(capability, {}).get('environment', {})
    return {key: str(Path(workspace)/value) if key == 'HF_HUB_CACHE' else value
            for key,value in settings.items()}


def install_resources(lock, capability, python, workspace):
    profile=lock.get('profiles', {}).get(capability, {})
    model=profile.get('embedding_model')
    if not model:
        return
    resources=profile.get('resources', [])
    if all((Path(workspace)/r['path']).is_file() and digest(Path(workspace)/r['path'])==r['sha256'] for r in resources):
        return
    # Fetch the exact revision, never a floating main. Check hashes afterwards.
    script="""from huggingface_hub import snapshot_download
import sys
snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],cache_dir=sys.argv[3],allow_patterns=sys.argv[4:])
"""
    cache=Path(workspace)/profile['environment']['HF_HUB_CACHE']
    command([python,'-I','-c',script,model['repository'],model['revision'],cache,*model['files']],timeout=600)
    ref=cache/('models--'+model['repository'].replace('/','--'))/'refs/main'
    ref.parent.mkdir(parents=True,exist_ok=True)
    ref.write_text(model['revision'])

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True,
                          text=True, timeout=kwargs.pop('timeout', 120), **kwargs).stdout.strip()


def load_lock(path=LOCK_PATH):
    lock = json.loads(Path(path).read_text())
    if lock.get('lock_version') not in ('parliament-runtime/1','parliament-runtime/2'):
        raise ValueError('Unsupported runtime lock version')
    if lock['lock_version']=='parliament-runtime/1' and (lock.get('profiles') or set(lock.get('capabilities', []))-{'compact_frameworks'}):
        raise ValueError('Multiple runtime profiles require parliament-runtime/2')
    source = lock['parliament']
    if not re.fullmatch(r'https://github.com/[\w.-]+/[\w.-]+\.git', source['repository']):
        raise ValueError('Runtime source must be an HTTPS GitHub repository')
    if not re.fullmatch(r'[0-9a-f]{40}', source['commit']):
        raise ValueError('Runtime source requires an exact commit SHA')
    return lock


def source_files(root):
    """All tracked inputs plus patch-added Python modules, excluding generated logs."""
    tracked = command(['git','-C',root,'ls-files','-z']).split('\0')
    added = command(['git','-C',root,'ls-files','--others','--exclude-standard','-z']).split('\0')
    return sorted(set(p for p in tracked if p) | {p for p in added if p.endswith('.py')})


def tree_digest(root):
    rows = [(name, digest(Path(root)/name) if (Path(root)/name).is_file() else 'MISSING')
            for name in source_files(root)]
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def recipe_errors(lock, workspace=ROOT, capability="compact_frameworks"):
    errors=[]
    for item in lock['parliament']['patches'] + [python_spec(lock, capability)['requirements']]:
        path = Path(workspace)/item['path']
        if not path.is_file():
            errors.append('Missing locked input: '+item['path'])
        elif digest(path) != item['sha256']:
            errors.append('Locked input changed: '+item['path'])
    return errors


def check_runtime(root=DEFAULT_PARLIAMENT_ROOT, python=None,
                  *, lock_path=LOCK_PATH, source_only=False, workspace=ROOT, capability='compact_frameworks'):
    root=Path(root).expanduser().resolve()
    python=Path(python or profile_python(capability)).expanduser().absolute()  # retain the venv launcher symlink
    result={'check_version':'parliament-runtime-check/1','ready':False,
            'profile':'source_only' if source_only else capability,
            'parliament_root':str(root),'python':str(python),'errors':[]}
    errors=result['errors']
    try:
        lock=load_lock(lock_path)
        result['lock_sha256']=digest(lock_path)
        result['capability']=capability
        if not source_only and capability not in lock.get('capabilities', ['compact_frameworks']):
            errors.append('Runtime capability is not locked: '+capability+'. Optional original-agent/RAG dependencies need a separate pinned profile.')
            result['remediation']='Use the compact live-demo entrypoint, or add and verify a separate '+capability+' profile in pipeline.lock.json.'
        if source_only or capability in lock.get("capabilities", ["compact_frameworks"]):
            errors.extend(recipe_errors(lock,workspace, capability if not source_only else "compact_frameworks"))
        if not (root/'.git').exists():
            errors.append('Missing native Git checkout: '+str(root))
        else:
            head=command(['git','-C',root,'rev-parse','HEAD'])
            result['commit']=head
            if head!=lock['parliament']['commit']:
                errors.append('Native commit does not match pipeline.lock.json')
            remote=command(['git','-C',root,'remote','get-url','origin'])
            if remote.rstrip('/').removesuffix('.git') != lock['parliament']['repository'].removesuffix('.git'):
                errors.append('Native origin does not match the lock')
            for name in lock['parliament']['required_files']:
                if not (root/name).is_file():errors.append('Missing native file: '+name)
            actual=tree_digest(root)
            result['source_tree_sha256']=actual
            if actual!=lock['parliament']['patched_tree_sha256']:
                errors.append('Native source differs from the locked patched tree')
                result['source_status']='DRIFTED'
            else:
                result['source_status']='LOCKED_PATCH_SET'
        if not source_only:
            if not python.is_file():
                errors.append('Missing persistent Python environment: '+str(python))
            elif not errors:
                # Isolated imports; no credential loading, model construction or API calls.
                probe = '''import importlib,importlib.metadata as m,json,sys
sys.path.insert(0,sys.argv[1])
spec=json.loads(sys.argv[2]);issues=[]
if list(sys.version_info[:2])!=spec['version']:issues.append('Python major/minor differs from lock')
expected=spec['distributions']
actual={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions() if d.metadata['Name'].lower() not in {'pip','setuptools','wheel'}}
for name,version in expected.items():
 if actual.get(name)!=version:issues.append('Dependency '+name+': expected '+version+', found '+actual.get(name,'missing'))
extra=sorted(set(actual)-set(expected))
if extra:issues.append('Unpinned dependencies: '+', '.join(extra))
if not issues:
 for name in spec['imports']:
  try:importlib.import_module(name)
  except Exception as exc:issues.append(name+': '+type(exc).__name__+(' (missing '+str(exc.name)+')' if isinstance(exc,ModuleNotFoundError) else ''))
print(json.dumps({'errors':issues,'python_version':sys.version.split()[0]}))
'''
                spec=python_spec(lock, capability)
                environment=profile_environment(lock,capability,workspace)
                result['environment']=environment
                checked=json.loads(command([python,'-I','-B','-c',probe,root,json.dumps(spec)], env={**os.environ,**environment}))
                result['python_version']=checked['python_version']
                errors.extend(checked['errors'])
                for item in lock.get('profiles', {}).get(capability, {}).get('resources', []):
                    path=Path(workspace)/item['path']
                    if not path.is_file() or digest(path)!=item['sha256']:
                        errors.append('Missing or changed retrieval resource: '+item['path'])
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        errors.append('Runtime inspection failed: '+type(exc).__name__)
    result['ready']=not errors
    result.setdefault('remediation','Run: .venv/bin/python parliament_runtime.py setup --profile '+capability+' --python python3.11')
    return result


def setup_runtime(*, lock_path=LOCK_PATH, workspace=ROOT, bootstrap='python3.11',
                  source_only=False, root=DEFAULT_PARLIAMENT_ROOT, python=None, capability="compact_frameworks"):
    lock=load_lock(lock_path)
    errors=recipe_errors(lock,workspace,capability)
    if errors:raise ValueError('; '.join(errors))
    root=Path(root);python=Path(python or profile_python(capability))
    spec=python_spec(lock,capability)
    if root.exists():
        status=check_runtime(root,python,lock_path=lock_path,source_only=True,workspace=workspace)
        if not status['ready']:
            raise ValueError('Existing runtime is not overwritten. '+ '; '.join(status['errors']))
    else:
        root.parent.mkdir(parents=True,exist_ok=True)
        source=lock['parliament']
        command(['git','clone','--single-branch','--branch',source['branch'],source['repository'],root],timeout=300)
        command(['git','-C',root,'checkout','--detach',source['commit']])
        for item in source['patches']:
            path=Path(workspace)/item['path']
            try:
                command(['git','-C',root,'apply','--check',path])
                command(['git','-C',root,'apply',path])
            except subprocess.CalledProcessError as exc:
                raise ValueError('Patch could not be applied: '+item['path']+
                                 '. Incomplete checkout preserved at '+str(root)) from exc
        status=check_runtime(root,python,lock_path=lock_path,source_only=True,workspace=workspace)
        if not status['ready']:raise ValueError('; '.join(status['errors']))
    if not source_only:
        if python.exists():
            ready=check_runtime(root,python,lock_path=lock_path,workspace=workspace,capability=capability)
            if ready['ready']:return ready
        executable=shutil.which(bootstrap)
        if not executable:raise ValueError('Python bootstrap not found: '+bootstrap)
        version=json.loads(command([executable,'-I','-c','import sys,json;print(json.dumps(list(sys.version_info[:2])))']))
        if version!=spec['version']:raise ValueError('Bootstrap Python version differs from lock')
        if not python.exists():command([executable,'-m','venv',python.parent.parent])
        if not (python.parent.parent/'pyvenv.cfg').is_file():
            raise ValueError('Dependency installation requires a dedicated virtual environment')
        command([python,'-m','pip','install','--disable-pip-version-check','--requirement',Path(workspace)/spec['requirements']['path']],timeout=600)
        install_resources(lock,capability,python,workspace)
    status=check_runtime(root,python,lock_path=lock_path,source_only=source_only,workspace=workspace,capability=capability)
    return status


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','setup'])
    p.add_argument('--profile',choices=['compact_frameworks','full_relent'],default='compact_frameworks')
    p.add_argument('--source-only',action='store_true')
    p.add_argument('--python',default='python3.11',help='Bootstrap interpreter for setup')
    p.add_argument('--output',type=Path,help='Save the inspection JSON')
    args=p.parse_args(argv)
    try:
        result=setup_runtime(bootstrap=args.python,source_only=args.source_only,capability=args.profile) if args.action=='setup' else check_runtime(source_only=args.source_only,capability=args.profile)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        result={'ready':False,'errors':[str(exc) if isinstance(exc,ValueError) else type(exc).__name__]}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['ready'] else 1


if __name__=='__main__':
    raise SystemExit(main())
