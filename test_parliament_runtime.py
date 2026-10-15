import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import parliament_runtime as runtime


class RuntimeLockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.workspace=Path(self.temp.name)
        self.root=self.workspace/'native';self.root.mkdir()
        subprocess.run(['git','init','-q',str(self.root)],check=True)
        for key,value in [('user.name','Runtime test'),('user.email','runtime@example.invalid')]:
            runtime.command(['git','-C',self.root,'config',key,value])
        self.module=self.root/'native.py';self.module.write_text('VALUE = 1\n')
        runtime.command(['git','-C',self.root,'add','.'])
        runtime.command(['git','-C',self.root,'commit','-qm','fixture'])
        runtime.command(['git','-C',self.root,'remote','add','origin','https://github.com/example/runtime.git'])
        self.requirements=self.workspace/'requirements.lock';self.requirements.write_text('')
        self.lock={'lock_version':'parliament-runtime/1','parliament':{
            'repository':'https://github.com/example/runtime.git','branch':'main',
            'commit':runtime.command(['git','-C',self.root,'rev-parse','HEAD']),
            'patches':[],'required_files':['native.py'],
            'patched_tree_sha256':runtime.tree_digest(self.root)},
            'python':{'version':[3,11],'requirements':{'path':'requirements.lock',
                'sha256':runtime.digest(self.requirements)},'imports':[],'distributions':{}}}
        self.lock_path=self.workspace/'lock.json';self.save()

    def save(self):
        self.lock_path.write_text(json.dumps(self.lock))

    def check(self,source_only=True):
        return runtime.check_runtime(self.root,self.workspace/'absent-python',
            lock_path=self.lock_path,source_only=source_only,workspace=self.workspace)

    def test_profiles_use_distinct_interpreters_and_specs(self):
        self.assertNotEqual(runtime.profile_python('full_relent'),runtime.profile_python('compact_frameworks'))
        self.lock['lock_version']='parliament-runtime/2'
        self.lock['capabilities']=['compact_frameworks','full_relent']
        self.lock['profiles']={'full_relent':{'python':dict(self.lock['python'],imports=['native'])}}
        self.save()
        original=runtime.command
        def probe(args,**kwargs):
            if '-I' in args:
                spec=json.loads(args[-1])
                self.assertEqual(spec['imports'],['native'])
                return json.dumps({'errors':[],'python_version':'3.11.11'})
            return original(args,**kwargs)
        with mock.patch.object(runtime,'command',side_effect=probe):
            result=runtime.check_runtime(self.root,sys.executable,lock_path=self.lock_path,workspace=self.workspace,capability='full_relent')
        self.assertTrue(result['ready'],result)
        self.assertEqual(result['profile'],'full_relent')

    def test_full_profile_detects_resource_drift_without_affecting_compact(self):
        resource=self.workspace/'model.bin';resource.write_bytes(b'pinned')
        self.lock['lock_version']='parliament-runtime/2'
        self.lock['capabilities']=['compact_frameworks','full_relent']
        self.lock['profiles']={'full_relent':{'python':self.lock['python'],'resources':[{'path':'model.bin','sha256':runtime.digest(resource)}]}}
        self.save();resource.write_bytes(b'changed')
        original=runtime.command
        def probe(args,**kwargs):
            return json.dumps({'errors':[],'python_version':'3.11.11'}) if '-I' in args else original(args,**kwargs)
        with mock.patch.object(runtime,'command',side_effect=probe):
            full=runtime.check_runtime(self.root,sys.executable,lock_path=self.lock_path,workspace=self.workspace,capability='full_relent')
            compact=runtime.check_runtime(self.root,sys.executable,lock_path=self.lock_path,workspace=self.workspace)
        self.assertFalse(full['ready'])
        self.assertIn('retrieval resource',full['errors'][0])
        self.assertTrue(compact['ready'],compact)

    def test_exact_source_is_valid_without_claiming_executable_readiness(self):
        result=self.check();self.assertTrue(result['ready']);self.assertEqual(result['profile'],'source_only')
        self.assertFalse(self.check(False)['ready'])

    def test_source_edit_missing_file_and_added_module_are_detected(self):
        self.module.write_text('VALUE = 2\n');self.assertFalse(self.check()['ready'])
        self.module.write_text('VALUE = 1\n');(self.root/'injected.py').write_text('raise RuntimeError()\n')
        self.assertFalse(self.check()['ready']);(self.root/'injected.py').unlink()
        self.module.unlink();self.assertIn('Missing native file: native.py',self.check()['errors'])

    def test_wrong_revision_and_patch_changes_are_detected(self):
        self.lock['parliament']['commit']='0'*40;self.save()
        self.assertIn('Native commit does not match pipeline.lock.json',self.check()['errors'])
        self.requirements.write_text('unreviewed-package==1\n')
        self.assertIn('Locked input changed: requirements.lock',self.check()['errors'])

    def test_patch_hash_failure_prevents_any_clone_or_install(self):
        patch=self.workspace/'change.patch';patch.write_text('old')
        self.lock['parliament']['patches']=[{'path':'change.patch','sha256':runtime.digest(patch)}]
        self.save();patch.write_text('modified')
        with mock.patch.object(runtime,'command') as command:
            with self.assertRaisesRegex(ValueError,'Locked input changed'):
                runtime.setup_runtime(lock_path=self.lock_path,workspace=self.workspace,
                    root=self.workspace/'new-runtime',source_only=True)
        command.assert_not_called()

    def test_unknown_lock_version_and_non_https_source_rejected(self):
        self.lock['lock_version']='future';self.save()
        self.assertFalse(self.check()['ready'])
        self.lock['lock_version']='parliament-runtime/1';self.lock['parliament']['repository']='ext::untrusted';self.save()
        with self.assertRaisesRegex(ValueError,'HTTPS'):runtime.load_lock(self.lock_path)

    def test_unlocked_capability_does_not_claim_readiness(self):
        result=runtime.check_runtime(self.root,self.workspace/'python',lock_path=self.lock_path,
            workspace=self.workspace,capability='full_relent')
        self.assertFalse(result['ready'])
        self.assertTrue(any('not locked: full_relent' in error for error in result['errors']))

    def test_dependency_drift_is_named_before_native_import(self):
        marker=self.workspace/'imported'
        self.module.write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("imported")\n')
        self.lock['parliament']['patched_tree_sha256']=runtime.tree_digest(self.root)
        self.lock['python'].update(version=list(sys.version_info[:2]),
            distributions={'runtime-missing-test-dependency':'0'},imports=['native'])
        self.save()
        result=runtime.check_runtime(self.root,sys.executable,lock_path=self.lock_path,workspace=self.workspace)
        self.assertFalse(result['ready']);self.assertFalse(marker.exists())
        self.assertIn('Dependency runtime-missing-test-dependency: expected 0, found missing',result['errors'])

    def test_setup_applies_verified_patch_once_and_preserves_existing_tree(self):
        patch=self.workspace/'change.patch'
        patch.write_text('--- a/native.py\n+++ b/native.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n')
        self.lock['parliament']['patches']=[{'path':'change.patch','sha256':runtime.digest(patch)}]
        self.module.write_text('VALUE = 2\n')
        self.lock['parliament']['patched_tree_sha256']=runtime.tree_digest(self.root)
        self.module.write_text('VALUE = 1\n');self.save()
        dest=self.workspace/'installed';real_command=runtime.command
        def local_clone(args,**kwargs):
            if list(args)[:2]==['git','clone']:
                shutil.copytree(self.root,Path(args[-1]));return ''
            return real_command(args,**kwargs)
        with mock.patch.object(runtime,'command',side_effect=local_clone):
            first=runtime.setup_runtime(lock_path=self.lock_path,workspace=self.workspace,
                root=dest,source_only=True)
            second=runtime.setup_runtime(lock_path=self.lock_path,workspace=self.workspace,
                root=dest,source_only=True)
        self.assertTrue(first['ready']);self.assertTrue(second['ready'])
        self.assertEqual((dest/'native.py').read_text(),'VALUE = 2\n')
        (dest/'native.py').write_text('local work\n')
        with self.assertRaisesRegex(ValueError,'not overwritten'):
            runtime.setup_runtime(lock_path=self.lock_path,workspace=self.workspace,root=dest,source_only=True)
        self.assertEqual((dest/'native.py').read_text(),'local work\n')

    def test_bad_runtime_blocks_chooser_before_credentials_are_loaded(self):
        import run_blueprint_parliament as runner
        output=self.workspace/'output'
        with mock.patch.object(runner,'_load_environment') as env, mock.patch.object(runner,'openai_complete') as provider:
            code=runner.main(['--text','Example scenario.', '--parliament-root',str(self.workspace/'missing'),
                '--parliament-python',str(self.workspace/'missing-python'),'--output-dir',str(output),'--prepare-only'])
        self.assertEqual(code,3);env.assert_not_called();provider.assert_not_called()
        manifest=json.loads((output/'preparation_manifest.json').read_text())
        self.assertEqual(manifest['pipeline_status'],'RUNTIME_UNAVAILABLE')
        self.assertEqual(manifest['scenario'],'Example scenario.')

    def test_runtime_failure_is_not_reported_as_semantic_abstention(self):
        from run_parliament_demo import summarize
        report=summarize('Scenario',{'pipeline_status':'RUNTIME_UNAVAILABLE',
                                    'withholding_reason':'Missing native runtime.'})
        self.assertEqual(report['status'],'SOFTWARE_FAILED')
        self.assertEqual(report['stop']['kind'],'software_failure')
        self.assertEqual(report['stop']['stage'],'runtime_setup')
        self.assertEqual(report['frameworks'],[])


if __name__=='__main__':
    unittest.main()
