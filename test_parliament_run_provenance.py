import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile
import parliament_run_provenance as provenance


class RunProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_tags_inherit_and_checkpoints_preserve_intermediate_bytes(self):
        output=self.root/'tagged'
        def child(argv):
            provenance.record_start(output/'child');return 0
        def run(argv):
            provenance.record_start(output)
            file=output/'mutable.json';file.write_text('before')
            provenance.record_checkpoint('world_frozen',artifacts=[file])
            file.write_text('after')
            provenance.record_execution(child,['--output',str(output/'child'),'--checkpoint-tag','child-tag'],entrypoint='child',output_flag='--output')
            return 0
        provenance.record_execution(run,['--output',str(output),'--checkpoint-tag','demo-v1'],entrypoint='fixture',output_flag='--output')
        record=provenance.load_manifest(output/'run_manifest.json')
        child_record=provenance.load_manifest(output/'child/run_manifest.json')
        self.assertEqual(child_record['checkpoint_tags'],['demo-v1','child-tag'])
        self.assertEqual(record['checkpoint_tags'],['demo-v1'])
        event=next(e for e in record['checkpoints'] if e['stage']=='world_frozen')
        self.assertEqual((output/event['artifacts'][0]['snapshot']).read_text(),'before')
        self.assertTrue(provenance.verify_manifest(output/'run_manifest.json')['valid'])
        with (output/'checkpoints.jsonl').open('a') as stream:stream.write('{}\n')
        self.assertFalse(provenance.verify_manifest(output/'run_manifest.json')['valid'])

    def test_invalid_tag_is_rejected_before_execution(self):
        with mock.patch.object(provenance,'source_snapshot') as snapshot:
            with self.assertRaises(ValueError):
                provenance.record_execution(lambda argv:0,['--output',str(self.root/'bad'),'--checkpoint-tag','not a tag'],entrypoint='fixture',output_flag='--output')
            snapshot.assert_not_called()

    def test_archive_includes_untracked_source_but_not_credentials_or_diagnostics(self):
        for name in ('tracked.py','new.py','.env','diagnostics/result.json'):
            file=self.root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('original')
        def git(*args,**kwargs):
            if args[0]=='ls-files':return 'tracked.py\0new.py\0.env\0diagnostics/result.json\0'
            if args[0]=='status':return '?? new.py\0'
            return 'fixture'
        with mock.patch.object(provenance,'git',side_effect=git):
            snapshot,archive=provenance.source_snapshot(self.root)
        self.assertEqual(set(snapshot['source_files']),{'tracked.py','new.py'})
        self.assertTrue(snapshot['dirty'])
        import io
        with zipfile.ZipFile(io.BytesIO(archive)) as source:self.assertEqual(source.read('new.py'),b'original')

    def test_parent_child_integrity_and_source_bundle(self):
        output=self.root/'parent'
        def child(argv):
            provenance.record_start(output/'child')
            (output/'child'/'result.json').write_text('{}');return 0
        def parent(argv):
            provenance.record_start(output)
            provenance.record_execution(child,['--output',str(output/'child')],entrypoint='child',output_flag='--output')
            (output/'result.json').write_text('{}');return 0
        provenance.record_execution(parent,['--output',str(output)],entrypoint='parent',output_flag='--output')
        record=provenance.load_manifest(output/'run_manifest.json')
        nested=provenance.load_manifest(output/'child/run_manifest.json')
        self.assertEqual(nested['parent_run_id'],record['run_id'])
        self.assertTrue(provenance.verify_manifest(output/'run_manifest.json')['valid'])
        (output/'result.json').write_text('altered')
        self.assertFalse(provenance.verify_manifest(output/'run_manifest.json')['valid'])

    def test_exception_finalizes_and_preserves_source_without_env_secrets(self):
        output=self.root/'failure'
        def fail(argv):
            provenance.record_start(output)
            raise RuntimeError('message may contain a secret; do not record it')
        with mock.patch.dict(os.environ,{'OPENAI_API_KEY':'private-token'}):
            with self.assertRaises(RuntimeError):
                provenance.record_execution(fail,['--output',str(output),'--unknown','private-token'],entrypoint='fixture',output_flag='--output')
        text=(output/'run_manifest.json').read_text()
        self.assertNotIn('private-token',text)
        self.assertNotIn('message may contain',text)
        self.assertEqual(json.loads(text)['status'],'EXCEPTION')
        self.assertTrue(provenance.verify_manifest(output/'run_manifest.json')['valid'])

    def test_replay_has_unknown_original_and_own_identity(self):
        output=self.root/'replay'
        provenance.record_execution(lambda argv:0,['--output',str(output),'--replay',str(self.root/'historical')],entrypoint='run_parliament_demo.py',output_flag='--output')
        record=provenance.load_manifest(output/'run_manifest.json')
        self.assertEqual(record['inputs']['replay_origin']['status'],'UNKNOWN_HISTORICAL')
        self.assertEqual(record['execution_path'],'saved_replay')
        self.assertEqual(record['mode'],'SAVED_REPLAY_NO_API')

    def test_revision_keeps_input_and_original_generation(self):
        report=self.root/'DEMO_REPORT.json';report.write_text('{"generation":"old"}')
        def update(argv):report.write_text('{"generation":"old","view":"new"}');return 0
        provenance.record_execution(update,['--report',str(report),'--answers','saved.json'],entrypoint='run_parliament_decisions_topography.py',output_flag='--report',in_place=True)
        manifest=next(p for p in (self.root/'provenance').glob('run_*.json') if p.stem.count('_')==1)
        record=provenance.load_manifest(manifest)
        self.assertEqual(json.loads((manifest.parent/record['input_report_snapshot']['path']).read_text()),{'generation':'old'})
        self.assertTrue(provenance.verify_manifest(manifest)['valid'])

    def test_existing_run_is_not_rewritten_and_paths_record_distinct_profiles(self):
        paths=[]
        for prepared in (False,True):
            output=self.root/str(prepared)
            argv=['--output-dir',str(output)]+(['--prepare-only'] if prepared else [])
            provenance.record_execution(lambda argv:0,argv,entrypoint='run_blueprint_parliament.py',output_flag='--output-dir')
            path=output/'run_manifest.json';before=path.read_bytes()
            with self.assertRaises(ValueError):
                provenance.record_execution(lambda argv:0,argv,entrypoint='run_blueprint_parliament.py',output_flag='--output-dir')
            self.assertEqual(path.read_bytes(),before)
            paths.append(path)
        full,prepared=map(provenance.load_manifest,paths)
        self.assertEqual(full['configuration']['runtime_profile'],'full_relent')
        self.assertEqual(prepared['configuration']['runtime_profile'],'compact_frameworks')

    def test_compare_distinguishes_configuration_and_rejects_unknown_version(self):
        paths=[]
        for model in ('first','second'):
            output=self.root/model
            provenance.record_execution(lambda argv:0,['--output',str(output),'--model',model],entrypoint='fixture',output_flag='--output')
            paths.append(output/'run_manifest.json')
        self.assertIn('configuration',provenance.compare_manifests(*paths)['differences'])
        paths[0].write_text('{"schema_version":"future"}')
        with self.assertRaises(ValueError):provenance.load_manifest(paths[0])


if __name__=='__main__':unittest.main()
