import copy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from export_parliament_viewer import DEFAULT_CATALOG, DEFAULT_REPORTS, ROOT, export, presentation, compact_world


class ViewerExportTests(unittest.TestCase):
    def test_provenance_preserves_generation_and_does_not_backfill_history(self):
        report=self.load('medicine')
        historical=presentation(report,'medicine','Medicine')
        self.assertEqual(historical['generation_provenance']['status'],'UNKNOWN_HISTORICAL')
        report['generation_provenance']={'run_id':'original'}
        report['run_provenance']={'run_id':'replay'}
        report['presentation_revisions']=[{'run_id':'coordinates'}]
        result=presentation(report,'medicine','Medicine')
        self.assertEqual(result['generation_provenance']['run_id'],'original')
        self.assertEqual(result['run_provenance']['run_id'],'replay')
        self.assertEqual(result['presentation_revisions'][0]['run_id'],'coordinates')
        self.assertIn('frontend/dist/app.js',result['viewer_provenance']['presentation_files'])

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
        # Explanations can be refreshed from matching native ledgers; operative
        # decisions and objections must survive a legacy export unchanged.
        for actual, saved in zip(result['frameworks'], report['frameworks']):
            for key in ('framework','position','preference','ledger_status','live_objections'):
                self.assertEqual(actual[key], saved[key])
        self.assertTrue(any(v['position']=='ABSTAIN' for v in result['frameworks']))
        self.assertEqual(result['hinges'],report['hinges'])
        self.assertTrue(all(h['record'] for h in result['hinges']))
        utilitarian=next(v for v in result['frameworks'] if v['framework']=='utilitarian')
        self.assertIsNone(utilitarian['preference'])
        self.assertEqual(utilitarian.get('reversal_conditions'),[])
        self.assertTrue(utilitarian.get('decision_boundaries'))
        if utilitarian.get('original_presentation'):
            self.assertIn('withholds a comparative ranking', utilitarian['reason'])
            self.assertIn('did not select a live action', utilitarian['original_presentation']['reason'])
        else:
            self.assertIn('did not select a live action', utilitarian['reason'])
        self.assertNotEqual(utilitarian['reason'],'Unresolved contestation: Max expected survival')
        self.assertTrue(any(h.get('role')=='ABSTENTION_BOUNDARY' for h in result['hinges']))

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
        catalog_before={ROOT/path:(ROOT/path).read_bytes()
                        for path in json.loads(DEFAULT_CATALOG.read_text())['reports'].values()}
        with tempfile.TemporaryDirectory() as temp:
            rows=export(output=Path(temp))
            self.assertEqual(len(rows),len(catalog_before))
            index=json.loads((Path(temp)/'runs.json').read_text())
            self.assertEqual(index['schema_version'],'parliament-view-index/1')
            for row in rows:
                payload=json.loads((Path(temp)/row['file']).read_text())
                self.assertEqual(payload['id'],row['id'])
                self.assertEqual(payload['provenance']['new_model_calls'],0)
                self.assertNotIn('/Users/', json.dumps(payload))
                for view in payload['frameworks']:
                    self.assertNotIn('argument_markdown_path', view)
                    url=view.get('argument_url')
                    self.assertRegex(url, r'^data/[a-z_]+_argument\.md$')
                    markdown=Path(temp)/url.split('/',1)[1]
                    self.assertTrue(markdown.is_file())
                    self.assertNotIn('/Users/', markdown.read_text())
        for p,content in before.items():self.assertEqual(p.read_bytes(),content)
        for p,content in catalog_before.items():self.assertEqual(p.read_bytes(),content)

    def test_catalog_keeps_latest_trolley_on_default_refresh(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            latest = self.load('medicine')
            latest['scenario'] = 'A pinned, admitted trolley scenario.'
            (root/'latest.json').write_text(json.dumps(latest))
            pinned = {name:str((DEFAULT_REPORTS/name/'DEMO_REPORT.json').resolve())
                      for name in ('medicine','promise','trolley')}
            pinned['trolley'] = str(root/'latest.json')
            (root/'catalog.json').write_text(json.dumps({'reports':pinned}))
            export(output=root/'out', catalog=root/'catalog.json')
            loaded = json.loads((root/'out/trolley.json').read_text())
            self.assertEqual(loaded['source'], latest['scenario'])
            self.assertTrue(loaded['diagrams'][0]['sent_to_agents'])
            # An explicitly requested historical folder still wins over the catalog.
            export(reports=DEFAULT_REPORTS, output=root/'out', catalog=root/'catalog.json')
            historical = json.loads((root/'out/trolley.json').read_text())
            self.assertEqual(historical['status'], 'INTERPRETIVE_BEAM')

    def test_catalog_exports_new_demo_without_overwriting_existing_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            report=self.load('medicine')
            (root/'novel.json').write_text(json.dumps(report))
            before=(root/'novel.json').read_bytes()
            pins=json.loads(DEFAULT_CATALOG.read_text())['reports']
            pins['new_rescue_case']=str(root/'novel.json')
            (root/'catalog.json').write_text(json.dumps({'version':'parliament-saved-runs/1','reports':pins}))
            rows=export(output=root/'out',catalog=root/'catalog.json')
            self.assertTrue({'medicine','promise','trolley','new_rescue_case'} <= {r['id'] for r in rows})
            payload=json.loads((root/'out/new_rescue_case.json').read_text())
            self.assertEqual(payload['source'],report['scenario'])
            self.assertEqual(payload['title'],'New rescue case')
            self.assertEqual(payload['diagrams'][0]['world'],report['world'])
            self.assertEqual((root/'novel.json').read_bytes(),before)

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
        js=(root/'app.js').read_text()
        self.assertIn('function scheduleFit',js)
        self.assertIn("if(view==='world')scheduleFit()",js)
        self.assertIn('ResizeObserver',js)
        self.assertIn('panel.hidden',js)
        self.assertIn('point.probability',js)
        self.assertIn('axisCoordinate',js)
        self.assertIn('axisInterval',js)
        self.assertIn('drawErrorCross',js)
        self.assertIn('drawIntervalBox',js)
        self.assertIn('jackknife',js)
        self.assertIn('Most hinging tenth',js)
        self.assertIn('jackknife_hinging',js)
        self.assertIn('drawLandscape',js)
        self.assertIn('interpolateVertical',js)
        self.assertIn('function hasLandscape',js)
        self.assertNotIn("id==='trolley'",js)
        self.assertNotIn("drawLandscape($('topo-p", js.replace('"', "'"))

    def medicine_native(self):
        world={'actions':[{'action_id':'A0','intervention':'Ada gives the medicine to Ben'},
                          {'action_id':'A1','intervention':'Ada gives the medicine to Cara'}]}
        def ev(value):
            return {'value':value,'unit':'EXPECTED_SURVIVORS','direction':'BENEFIT',
                'validation_status':'ARITHMETIC_VERIFIED','calculation':'1 × %s'%value,
                'valuation_scope':'RESTRICTED_SURVIVOR_ENDPOINT_NOT_TOTAL_WELFARE',
                'count_projections':[{'welfare_magnitude':'UNKNOWN','duration':'UNKNOWN'}]}
        return world, {'cycles':[{'candidates':[
            {'specialist':'utilitarian','expected_value_validation_status':'ARITHMETIC_VERIFIED',
             'recommended_action':'Ada gives the medicine to Ben',
             'expected_value_estimates':{'Ada gives the medicine to Ben':ev(0.95),'Ada gives the medicine to Cara':ev(0.05)},
             'committed_native_ledger':{'ledger_kind':'UTILITARIAN_CONSEQUENCE_LEDGER','transaction_status':'COMMITTED','records':[]}},
            {'specialist':'deontological','recommended_action':'Ada gives the medicine to Ben','committed_native_ledger':{}},
            {'specialist':'care','action_scores':{'Ada gives the medicine to Ben':0.76,'Ada gives the medicine to Cara':0.24},
             'recommended_action':'Ada gives the medicine to Ben',
             'committed_native_ledger':{'ledger_kind':'CARE_RELATIONSHIP_LEDGER','transaction_status':'COMMITTED','records':[
                 {'canonical_action_id':'A0','verdict':'RESPONSIVE','reason':'competent care'},
                 {'canonical_action_id':'A1','verdict':'NEGLECTFUL','reason':'incompetent response'}]}}
        ]}]}

    def test_topography_records_medicine_loci_without_inventing_welfare_zero(self):
        from parliament_topography import topography_from_trace
        world, trace=self.medicine_native()
        topo=topography_from_trace(trace, world, [{'framework':'care','preference':'Ada gives the medicine to Ben','position':'ATTENUATED'}])
        self.assertEqual(topo['status'],'RECORDED')
        self.assertEqual(len(topo['actions']),2)
        values={row['id']:row['welfare']['value'] for row in topo['actions']}
        self.assertEqual(values,{'A0':0.95,'A1':0.05})
        self.assertTrue(all(row['welfare'].get('total_welfare')=='UNKNOWN' for row in topo['actions']))
        self.assertTrue(all(not row['obligation']['present'] for row in topo['actions']))
        self.assertIn('care_welfare_survival_copy',topo['flags']['degenerate'])
        self.assertEqual(topo['flags']['attractors'],[])
        self.assertEqual(set(values.values()),{0.95,0.05})
        self.assertNotIn(0,[row['welfare'].get('value') for row in topo['actions']])
        self.assertNotIn(0.0,[row['welfare'].get('value') for row in topo['actions']])

    def test_unknown_welfare_is_omitted_rather_than_zero(self):
        from parliament_topography import topography_from_trace
        world={'actions':[{'action_id':'A0','intervention':'act 0'},{'action_id':'A1','intervention':'act 1'}]}
        blank={'value':0.0,'unit':'NONE','validation_status':'NOT_CLAIMED','count_projections':[
            {'welfare_magnitude':'UNKNOWN','duration':'UNKNOWN'}]}
        topo=topography_from_trace({'cycles':[{'candidates':[
            {'specialist':'utilitarian','expected_value_validation_status':'NOT_CLAIMED',
             'expected_value_estimates':{'act 0':blank,'act 1':blank},
             'committed_native_ledger':{'ledger_kind':'UTILITARIAN_CONSEQUENCE_LEDGER','transaction_status':'COMMITTED','records':[]}},
            {'specialist':'care','committed_native_ledger':{'ledger_kind':'CARE_RELATIONSHIP_LEDGER','transaction_status':'COMMITTED','records':[
                {'canonical_action_id':'A0','verdict':'RESPONSIVE'},{'canonical_action_id':'A1','verdict':'MIXED'}]}}
        ]}]}, world)
        self.assertTrue(topo['axes']['welfare']['available'] is False)
        for row in topo['actions']:
            self.assertFalse(row['welfare']['present'])
            self.assertNotIn('value',row['welfare'])
        self.assertNotIn(0, [row['welfare'].get('value') for row in topo['actions']])

    def test_beam_export_has_no_topography_points_or_attractors(self):
        result=presentation(self.load('trolley'),'trolley','Trolley')
        topo=result['topography']
        self.assertEqual(topo['status'],'UNAVAILABLE')
        self.assertEqual(topo['actions'],[])
        self.assertEqual(topo['flags']['attractors'],[])
        self.assertIn('no framework coordinates',topo['unavailable_reason'])

    def test_topography_is_additive_on_parliament_view_one(self):
        world, trace=self.medicine_native()
        report=self.load('promise')
        report=copy.deepcopy(report)
        report['topography']=None
        report['cycles']=trace['cycles']
        result=presentation(report,'promise','Promise')
        self.assertEqual(result['schema_version'],'parliament-view/1')
        self.assertEqual(len(result['topography']['actions']),2)
        html=(ROOT/'frontend/dist/index.html').read_text()
        self.assertIn('view-topography',html)
        self.assertIn('data-view="topography"',html)


if __name__=='__main__':
    unittest.main()
