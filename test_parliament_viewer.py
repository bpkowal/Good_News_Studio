import copy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from export_parliament_viewer import DEFAULT_REPORTS, ROOT, export, presentation, compact_world


class ViewerExportTests(unittest.TestCase):
    def test_promise_overlay_is_separate_and_uses_recorded_support(self):
        report=self.load('promise'); before=copy.deepcopy(report)
        diagram=presentation(report,'promise','Promise')['diagrams'][0]
        self.assertEqual(len(diagram['source_ledger_nodes']),1)
        record=diagram['source_ledger_nodes'][0]['record']
        self.assertEqual(record['claim'],'Ada promised Cara that she would give Cara the medicine.')
        self.assertIn('ledger support reference',diagram['source_ledger_mermaid'])
        self.assertIn('classDef sourceLedger',diagram['source_ledger_mermaid'])
        self.assertNotIn('SOURCE_SPAN:',diagram['source_ledger_mermaid'])
        self.assertEqual(diagram['world'],report['world'])
        self.assertEqual(report,before)

    def test_overlay_does_not_duplicate_world_claims_or_attach_to_candidates(self):
        medicine=presentation(self.load('medicine'),'medicine','Medicine')
        self.assertEqual(medicine['diagrams'][0]['source_ledger_nodes'],[])
        trolley=self.load('trolley')
        trolley['stipulated_claims']=[{'claim':'Test source claim','introduced_by':'SOURCE_TEXT'}]
        self.assertTrue(all(d['source_ledger_nodes']==[] for d in presentation(trolley,'trolley','Trolley')['diagrams']))

    def test_compact_view_retains_nodes_relations_and_original_world(self):
        world=self.load('medicine')['world']; before=copy.deepcopy(world)
        display, hidden=compact_world(world)
        self.assertEqual(len(hidden),6)
        self.assertEqual(display['effects'],world['effects'])
        self.assertEqual(display['causal_links'],world['causal_links'])
        self.assertEqual([a['effect_ids'] for a in display['actions']],[['E1'],['E3']])
        self.assertEqual(world,before)

    def test_compact_view_does_not_hide_unconnected_or_other_branch_effects(self):
        world={'actions':[{'action_id':'A0','effect_ids':['E1','E2','E3']}],
            'effects':[{'effect_id':'E1','directness':'DIRECT'},
                       {'effect_id':'E2','directness':'DOWNSTREAM'},
                       {'effect_id':'E3','directness':'DOWNSTREAM'}],
            'causal_links':[{'source_id':'E1','target_id':'E2','action_id':'A1'}]}
        display, hidden=compact_world(world)
        self.assertEqual(hidden,[])
        self.assertEqual(display,world)

    def load(self, name):
        return json.loads((DEFAULT_REPORTS/name/'DEMO_REPORT.json').read_text())

    def test_export_preserves_original_world_and_source(self):
        report=self.load('medicine'); before=copy.deepcopy(report)
        result=presentation(report,'medicine','Medicine')
        self.assertEqual(result['source'],report['scenario'])
        self.assertEqual(result['source_sha256'],hashlib.sha256(report['scenario'].encode()).hexdigest())
        self.assertEqual(result['diagrams'][0]['world'],report['world'])
        self.assertTrue(result['diagrams'][0]['sent_to_agents'])
        self.assertEqual(report,before)

    def test_beam_is_three_candidates_without_agent_participation(self):
        result=presentation(self.load('trolley'),'trolley','Trolley')
        self.assertEqual(len(result['diagrams']),3)
        self.assertTrue(all(d['authority']=='CANDIDATE' and not d['sent_to_agents'] for d in result['diagrams']))
        self.assertEqual(result['frameworks'],[])
        self.assertTrue(all(d['mermaid'].startswith('flowchart LR') for d in result['diagrams']))

    def test_contradictory_beam_cannot_export_as_frozen(self):
        report=self.load('medicine'); report['status']='INTERPRETIVE_BEAM'
        with self.assertRaises(ValueError):presentation(report,'medicine','Medicine')

    def test_framework_abstention_and_hinges_survive_export(self):
        report=self.load('promise'); result=presentation(report,'promise','Promise')
        self.assertEqual(result['frameworks'],report['frameworks'])
        self.assertTrue(any(v['position']=='ABSTAIN' for v in result['frameworks']))
        self.assertEqual(result['hinges'],report['hinges'])
        self.assertTrue(all(h['record'] for h in result['hinges']))

    def test_unknown_schema_and_unsafe_filename_rejected(self):
        report=self.load('medicine')
        for ident in ('../run','a/b','x.json',''):
            with self.assertRaises(ValueError):presentation(report,ident,'Medicine')
        report['version']='unknown'
        with self.assertRaises(ValueError):presentation(report,'medicine','Medicine')

    def test_bundle_does_not_copy_local_paths_or_commands(self):
        for name in ('medicine','promise','trolley'):
            text=json.dumps(presentation(self.load(name),name,name))
            self.assertNotIn('/Users/',text)
            self.assertNotIn('/tmp/parliament',text)
            self.assertNotIn('OPENAI_API_KEY',text)
            self.assertNotIn('trace_path',text)
            self.assertNotIn('"commands"',text)

    def test_export_writes_complete_index_without_editing_reports(self):
        before={p:p.read_bytes() for p in DEFAULT_REPORTS.rglob('DEMO_REPORT.json')}
        with tempfile.TemporaryDirectory() as temp:
            rows=export(output=Path(temp))
            self.assertEqual(len(rows),3)
            index=json.loads((Path(temp)/'runs.json').read_text())
            self.assertEqual(index['schema_version'],'parliament-view-index/1')
            for row in rows:
                payload=json.loads((Path(temp)/row['file']).read_text())
                self.assertEqual(payload['id'],row['id'])
                self.assertEqual(payload['provenance']['new_model_calls'],0)
        for p,content in before.items():self.assertEqual(p.read_bytes(),content)

    def test_assets_are_packaged_locally(self):
        class Assets(HTMLParser):
            refs=[]
            def handle_starttag(self,tag,attrs):
                attrs=dict(attrs)
                if tag=='script' and attrs.get('src'):self.refs.append(attrs['src'])
                if tag=='link' and attrs.get('href'):self.refs.append(attrs['href'])
        root=ROOT/'frontend/dist'; parser=Assets(); parser.feed((root/'index.html').read_text())
        for ref in parser.refs:
            self.assertNotIn('://',ref)
            self.assertTrue((root/ref).is_file())
        self.assertTrue((root/'vendor/MERMAID_LICENSE.txt').is_file())
        self.assertGreater((root/'vendor/mermaid-10.9.3.min.js').stat().st_size,1000000)


if __name__=='__main__':
    unittest.main()
