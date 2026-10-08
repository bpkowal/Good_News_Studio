import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_parliament_demo import main, summarize, write_report, attach_candidate_diagrams


class DemoTests(unittest.TestCase):
    def test_failed_candidates_remain_inspectable_without_becoming_world(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'blueprint_candidate.json').write_text(json.dumps({'scenario':'same',
                'blueprint_result':{'considered':[{'blueprint_id':'omission',
                    'status':'WITHHELD', 'unfilled':['actor']}]}}))
            (folder/'candidate_graphs.md').write_text('```mermaid\nflowchart LR\nA-->B\n```')
            (folder/'preparation_manifest.json').write_text(json.dumps({'admission_attempts':[
                {'blueprint_id':'omission','status':'REJECTED','error':'Missing condition'}]}))
            report = summarize('same', {'pipeline_status':'ADMISSION_FAILED'})
            attach_candidate_diagrams(report, folder)
            write_report(report, folder)
            text = (folder/'DEMO_REPORT.md').read_text()
            self.assertFalse(report['world'])
            self.assertEqual(report['stop']['kind'], 'construction_failure')
            self.assertIn('omission: actor', report['missing_slots'])
            self.assertIn('None was sent to agents', text)
            self.assertIn('Missing condition', text)
            self.assertIn('```mermaid', text)

    def test_suite_reports_failure_after_successful_freeze(self):
        from run_parliament_demo_suite import suite_failed
        self.assertTrue(suite_failed([{'parse_status':'PACKAGE_WITH_VISIBLE_GAPS',
            'exact_spans':True, 'freeze_status':'FROZEN', 'expectation_match':True,
            'pipeline_status':'SOFTWARE_FAILED'}]))

    def test_frozen_diagram_uses_recorded_world_and_keeps_it_unchanged(self):
        world={'parties':[{'party_id':'P1','label':'Ada','kind':'PERSON'}],
               'actions':[{'action_id':'A0','intervention':'Ada acts','actor_party_id':'P1','effect_ids':['E1']}],
               'effects':[{'effect_id':'E1','outcome':'Ben lives','party_id':'P1'}]}
        original=json.dumps(world,sort_keys=True)
        report=summarize('Ada acts.', {'pipeline_status':'ADMITTED'},
            {'action_source_grounding':{'world_model':world}})
        with tempfile.TemporaryDirectory() as temp:
            write_report(report,Path(temp))
            text=(Path(temp)/'DEMO_REPORT.md').read_text()
            self.assertIn('```mermaid\nflowchart LR',text)
            self.assertIn('A0 — Ada acts',text)
            self.assertIn('E1 — Ben lives',text)
            self.assertIn('no framework assessment recorded',text)
        self.assertEqual(json.dumps(world,sort_keys=True),original)

    def test_beam_diagrams_never_become_selected_world(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            path=folder/'candidate.json'
            path.write_text(json.dumps({'scenario':'same','action_source_grounding':{'world_model':
                {'actions':[{'action_id':'A0','intervention':'pull lever'}]}}}))
            before=path.read_bytes()
            (folder/'world_beam.json').write_text(json.dumps({'members':[
                {'blueprint_id':'omission','admission_status':'ADMITTED','trace_path':str(path)}]}))
            report=summarize('same',{'pipeline_status':'INTERPRETIVE_BEAM'})
            attach_candidate_diagrams(report,folder)
            write_report(report,folder)
            text=(folder/'DEMO_REPORT.md').read_text()
            self.assertFalse(report['world'])
            self.assertIn('none was sent to agents',text)
            self.assertIn('```mermaid',text)
            self.assertEqual(path.read_bytes(),before)

    def test_candidate_diagram_rejects_source_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); path=folder/'candidate.json'
            path.write_text(json.dumps({'scenario':'wrong','action_source_grounding':{'world_model':{'actions':[]}}}))
            (folder/'world_beam.json').write_text(json.dumps({'members':[
                {'blueprint_id':'omission','admission_status':'ADMITTED','trace_path':str(path)}]}))
            report=summarize('same',{'pipeline_status':'INTERPRETIVE_BEAM'})
            attach_candidate_diagrams(report,folder)
            self.assertEqual(report['candidate_world_diagrams'],[])
            self.assertTrue(report['diagram_notices'])

    def test_beam_and_software_failure_are_distinct(self):
        beam = summarize('s', {'pipeline_status':'INTERPRETIVE_BEAM'})
        failed = summarize('s', {}, failure='parser crashed')
        self.assertEqual(beam['stop']['kind'], 'epistemic_abstention')
        self.assertEqual(failed['status'], 'SOFTWARE_FAILED')
        self.assertFalse(beam['world'])

    def test_missing_hinges_are_not_invented(self):
        self.assertEqual(summarize('s', {})['hinges'], [])

    def test_rejected_ledger_cannot_appear_full(self):
        trace = {'cycles':[{'candidates':[{'specialist':'care', 'recommended_action':'A0',
            'selection_status':'SELECTED', 'rationale':'model preference'}]}]}
        r = summarize('s', {}, trace)
        self.assertEqual(r['frameworks'][0]['position'], 'ABSTAIN')
        self.assertEqual(r['status'], 'FRAMEWORK_PARTIAL')

    def test_native_abstention_overrides_authored_preference(self):
        trace = {'cycles':[{'candidates':[{'specialist':'deontological', 'recommended_action':'A0',
            'selection_status':'SELECTED', 'framework_vote_status':'ABSTAIN',
            'committed_native_ledger':{'transaction_status':'COMMITTED'}}]}]}
        self.assertEqual(summarize('s', {}, trace)['frameworks'][0]['position'], 'ABSTAIN')

    def test_hinges_have_record_paths_and_keep_framework_diversity(self):
        cs = [{'specialist':f, 'normative_reversal_threshold':'priority changes',
               'factual_reversal_threshold':'new evidence'} for f in ('utilitarian','deontological','care')]
        r = summarize('s', {}, {'cycles':[{'candidates':cs}]})
        self.assertEqual(len(r['hinges']), 3)
        self.assertEqual({h['framework'] for h in r['hinges']}, {'utilitarian','deontological','care'})
        self.assertTrue(all(h['record'].endswith('normative_reversal_threshold') for h in r['hinges']))

    def test_nonexact_evidence_is_not_claimed_source_grounded(self):
        t = {'action_source_grounding':{'world_model':{'effects':[{'effect_id':'E1',
            'outcome':'invented', 'provenance':[{'excerpt':'missing'}]}]}}}
        e = summarize('source', {}, t)['grounded_effects'][0]
        self.assertEqual(e['source_check'], 'NO_EXACT_EXCERPT')

    def test_replay_never_calls_provider_and_keeps_source_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); saved = root/'saved'; saved.mkdir()
            source = root/'s.txt'; source.write_text('same\n')
            (saved/'preparation_manifest.json').write_text(json.dumps({'scenario':'same','pipeline_status':'INTERPRETIVE_BEAM'}))
            original = (saved/'preparation_manifest.json').read_bytes()
            with patch('subprocess.run', side_effect=AssertionError('No process in replay')):
                self.assertEqual(main(['--scenario',str(source),'--output',str(root/'out'),'--replay',str(saved)]),0)
            self.assertEqual(original,(saved/'preparation_manifest.json').read_bytes())
            self.assertEqual(json.loads((root/'out'/'DEMO_REPORT.json').read_text())['status'],'INTERPRETIVE_BEAM')

    def test_mismatched_source_does_not_report_foreign_world(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); saved=root/'saved'; saved.mkdir(); source=root/'s'; source.write_text('different')
            (saved/'frozen_world_trace.json').write_text(json.dumps({'scenario':'old','action_source_grounding':{'world_model':{'parties':[{'label':'foreign'}]}}}))
            self.assertEqual(main(['--scenario',str(source),'--output',str(root/'out'),'--replay',str(saved)]),1)
            self.assertFalse(json.loads((root/'out'/'DEMO_REPORT.json').read_text())['world'])

    def test_suite_sources_are_frozen(self):
        rows=json.loads(Path('demo_scenarios/manifest.json').read_text())['scenarios']
        self.assertEqual(len(rows),8)
        for row in rows:
            source=Path(row['scenario']).read_text().strip()
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),row['source_sha256'])
            self.assertNotIn('expected_verdict',row)

    def test_live_orchestration_handoff_preserves_world(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; source.write_text('same'); out=root/'out'
            frozen={'scenario':'same','action_source_grounding':{'world_model':{'admission':{'status':'COMMITTED'}}}}
            def process(command, **kwargs):
                from types import SimpleNamespace
                folder=Path(command[command.index('--output-dir')+1]); folder.mkdir(parents=True)
                if 'run_blueprint_parliament.py' in command[1]:
                    (folder/'preparation_manifest.json').write_text(json.dumps({'scenario':'same','pipeline_status':'ADMITTED'}))
                    (folder/'frozen_world_trace.json').write_text(json.dumps(frozen))
                else:
                    payload=dict(frozen, cycles=[{'candidates':[{'specialist':'care','recommended_action':'A0',
                        'selection_status':'SELECTED','committed_native_ledger':{'transaction_status':'COMMITTED'}}]}])
                    (folder/'framework_generation.json').write_text(json.dumps(payload))
                return SimpleNamespace(returncode=0)
            with patch('subprocess.run', side_effect=process) as mocked:
                self.assertEqual(main(['--scenario',str(source),'--output',str(out)]),0)
            self.assertEqual(mocked.call_count,2)
            self.assertEqual(json.loads((out/'DEMO_REPORT.json').read_text())['status'],'COMPLETED')
            self.assertEqual(json.loads((out/'pipeline/frozen_world_trace.json').read_text()),frozen)

    def test_construction_withheld_does_not_call_frameworks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; source.write_text('same'); out=root/'out'
            def process(command, **kwargs):
                from types import SimpleNamespace
                folder=Path(command[command.index('--output-dir')+1]); folder.mkdir(parents=True)
                (folder/'preparation_manifest.json').write_text(json.dumps({'scenario':'same',
                    'pipeline_status':'WITHHELD','missing_slots':['exclusivity'],'withholding_reason':'No evidence for exclusivity.'}))
                return SimpleNamespace(returncode=2)
            with patch('subprocess.run', side_effect=process) as mocked:
                self.assertEqual(main(['--scenario',str(source),'--output',str(out)]),0)
            self.assertEqual(mocked.call_count,1)
            report=json.loads((out/'DEMO_REPORT.json').read_text())
            self.assertEqual(report['status'],'CONSTRUCTION_WITHHELD')
            self.assertEqual(report['missing_slots'],['exclusivity'])

    def test_missing_framework_candidate_is_partial(self):
        r=summarize('s', {'pipeline_status':'ADMITTED'}, {'framework_runs':[{'framework':'care'}]})
        self.assertEqual(r['status'],'FRAMEWORK_PARTIAL')
        self.assertEqual(r['frameworks'][0]['position'],'ABSTAIN')

    def test_unreadable_scenario_still_reports_software_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out'
            self.assertEqual(main(['--scenario',str(Path(temp)/'missing'),'--output',str(out)]),1)
            self.assertEqual(json.loads((out/'DEMO_REPORT.json').read_text())['status'],'SOFTWARE_FAILED')


if __name__ == '__main__':
    unittest.main()
