"""Export a small, path-free presentation record from saved demo reports."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import copy

from parliament_topography import native_trace_from_report, topography_from_trace, unavailable

ROOT = Path(__file__).resolve().parent
DEFAULT_REPORTS = ROOT/'diagnostics/parliament_demo_20261007/reports'
DEFAULT_OUTPUT = ROOT/'frontend/dist/data'
DEFAULT_CATALOG = ROOT/'frontend/saved_runs.json'
MEDICINE_FALLBACK_TRACE = ROOT/'diagnostics/argument_quality_pilot_rerun_live_20261007/baseline/standard/framework_generation.json'


def compact_world(world):
    """Hide redundant display membership edges; never change saved world records."""
    display = copy.deepcopy(world)
    hidden = []
    for action in display.get('actions', []):
        members = set(action.get('effect_ids', []))
        roots = {e['effect_id'] for e in world.get('effects', [])
                 if e['effect_id'] in members and e.get('directness') == 'DIRECT'}
        reachable = set(roots)
        while True:
            added = {link['target_id'] for link in world.get('causal_links', [])
                     if link.get('action_id') == action['action_id']
                     and link.get('source_id') in reachable
                     and link.get('target_id') in members}
            if added <= reachable:
                break
            reachable |= added
        omitted = reachable - roots
        action['effect_ids'] = [e for e in action.get('effect_ids', []) if e not in omitted]
        hidden.extend({'action_id':action['action_id'], 'effect_id':e}
                      for e in sorted(omitted))
    return display, hidden


def source_ledger_layer(world, claims):
    """Present saved source-text ledger entries separately from world topology."""
    from run_blueprint_parliament import _label
    node_keys = {'parties':'party_id', 'actions':'action_id', 'effects':'effect_id',
                 'conditions':'condition_id', 'propositions':'proposition_id',
                 'reports':'report_id', 'commitments':'commitment_id',
                 'modal_operators':'modal_id', 'normative_propositions':'normative_proposition_id'}
    ids = {row[key] for array, key in node_keys.items() for row in world.get(array, [])
           if isinstance(row.get(key), str)}
    nodes, lines = [], []
    for claim in claims:
        if claim.get('introduced_by') != 'SOURCE_TEXT' or not claim.get('claim'):
            continue
        ident = 'SL_' + hashlib.sha256(str(claim.get('proposition_id', claim['claim'])).encode()).hexdigest()[:16]
        while ident in ids:
            ident += '_ledger'
        ids.add(ident)
        nodes.append({'id':ident, 'record':copy.deepcopy(claim)})
        lines.append(f'    {ident}(["SOURCE LEDGER — {_label(claim["claim"])}<br/>Not a frozen-world node"]):::sourceLedger')
        for support in claim.get('support_ids', []):
            if support in ids and support != ident:
                lines.append(f'    {ident} -.->|"ledger support reference"| {support}')
    if nodes:
        lines.append('    classDef sourceLedger fill:#f1e8ff,stroke:#7950a5,stroke-width:2px,stroke-dasharray:5 3,color:#40245c')
    return nodes, '\n'.join(lines)


def public_provenance(value):
    """Keep run identity and hashes while omitting machine-specific absolute paths."""
    if isinstance(value,dict):return {k:public_provenance(v) for k,v in value.items()}
    if isinstance(value,list):return [public_provenance(v) for v in value]
    if isinstance(value,str) and Path(value).is_absolute():
        try:return str(Path(value).relative_to(ROOT))
        except ValueError:return 'external-path:'+Path(value).name
    return value


def presentation(report, ident, title):
    if report.get('version') != 'parliament-demo/1':
        raise ValueError('Unknown demo report version')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', ident):
        raise ValueError('Run ID must be a safe filename')
    if report.get('status') == 'INTERPRETIVE_BEAM' and report.get('world'):
        raise ValueError('An interpretive beam cannot export a selected frozen world')
    from run_blueprint_parliament import _mermaid_world
    diagrams = []
    if report.get('world'):
        diagrams.append({'id':'frozen', 'label':'Frozen world', 'authority':'FROZEN',
            'sent_to_agents':bool(report.get('frameworks')), 'world':report['world']})
    else:
        for i, candidate in enumerate(report.get('candidate_world_diagrams', [])):
            diagrams.append({'id':'candidate-'+str(i), 'label':candidate['label'],
                'authority':'CANDIDATE', 'sent_to_agents':False, 'world':candidate['world']})
    for diagram in diagrams:
        lines = _mermaid_world(diagram['world'], diagram['label'])
        diagram['mermaid'] = '\n'.join(lines[lines.index('```mermaid')+1:lines.index('```')])
        display, hidden = compact_world(diagram['world'])
        compact_lines = _mermaid_world(display, diagram['label'])
        diagram['compact_mermaid'] = '\n'.join(compact_lines[compact_lines.index('```mermaid')+1:compact_lines.index('```')])
        diagram['hidden_membership_edges'] = hidden
        from blueprint_reader_projection import action_bundles, bundle_mermaid, context_propositions
        diagram['reader_bundles'] = action_bundles(diagram['world'])
        # Ledger support references must remain visible as graph endpoints.
        ledger_refs={ident for c in report.get('stipulated_claims',[]) if c.get('introduced_by')=='SOURCE_TEXT' for ident in c.get('support_ids',[])}
        diagram['reader_context_ids'] = [ident for ident in context_propositions(diagram['world']) if ident not in ledger_refs]
        diagram['compact_mermaid'] = bundle_mermaid(diagram['compact_mermaid'], diagram['reader_bundles'], diagram['reader_context_ids'])
        nodes, layer = source_ledger_layer(diagram['world'], report.get('stipulated_claims', [])) if diagram['authority'] == 'FROZEN' else ([], '')
        diagram['source_ledger_nodes'] = nodes
        diagram['source_ledger_mermaid'] = layer
        diagram['compact_source_ledger_mermaid'] = bundle_mermaid(layer, diagram['reader_bundles'])
        from blueprint_spatial_projection import typed_reader_view
        try:
            diagram['typed_reader_view'] = typed_reader_view(report['scenario'], diagram['world'])
        except (ImportError, OSError):
            diagram['typed_reader_view'] = None
        diagram['world_sha256'] = hashlib.sha256(json.dumps(diagram['world'],sort_keys=True).encode()).hexdigest()
    topography = topography_for_report(report)
    from parliament_framework_presentation import apply_reviewed_explanations
    views = apply_reviewed_explanations(report, framework_views_for_report(report))
    return {'schema_version':'parliament-view/1', 'id':ident, 'title':title,
        'source':report['scenario'], 'source_sha256':hashlib.sha256(report['scenario'].encode()).hexdigest(),
        'status':report['status'], 'stop':report['stop'], 'judgment_status':report.get('judgment_status'),
        'mode':report.get('execution',{}).get('mode','SAVED_RECORD'), 'diagrams':diagrams,
        'frameworks':views, 'hinges':report.get('hinges',[]),
        'agent_roster':copy.deepcopy(report.get('agent_roster')),
        'unknowns':report.get('unknowns',[]), 'disagreements':report.get('disagreements',[]),
        'agreement':report.get('agreement'), 'not_decided':report.get('not_decided',[]),
        'stipulated_claims':report.get('stipulated_claims',[]), 'missing_slots':report.get('missing_slots',[]),
        'diagram_notices':report.get('diagram_notices',[]), 'topography': topography,
        'run_provenance':public_provenance(report.get('run_provenance') or {'status':'UNKNOWN_HISTORICAL','reason':'No original execution manifest was recorded.'}),
        'generation_provenance':public_provenance(report.get('generation_provenance') or {'status':'UNKNOWN_HISTORICAL'}),
        'presentation_revisions':public_provenance(report.get('presentation_revisions', [])),
        'viewer_provenance':viewer_provenance(),
        'provenance':{'origin':'SAVED_DEMO_REPORT', 'new_model_calls':0,
            'report_sha256':hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()}}



def viewer_provenance():
    from parliament_run_provenance import current_provenance
    files=['export_parliament_viewer.py','blueprint_reader_projection.py','frontend/dist/app.js','frontend/dist/provenance-receipts.mjs','frontend/dist/index.html','frontend/dist/styles.css','frontend/dist/vendor/mermaid-10.9.3.min.js','frontend/reviewed_explanations.json','frontend/saved_runs.json']
    execution=public_provenance(current_provenance())
    run_id=execution.get('run_id')
    return {'execution':execution, 'manifest_file':execution.get('manifest_file') if run_id else None,
            'presentation_files':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files if (ROOT/name).is_file()}}


def export(reports=None, output=DEFAULT_OUTPUT, trolley_report=None, catalog=None, checkpoint_tags=None):
    from parliament_run_provenance import record_execution
    output=Path(output)
    return record_execution(lambda argv:_export(reports,output,trolley_report,catalog),
                            ['--report',str(output/'runs.json')]+[part for tag in (checkpoint_tags or []) for part in ('--checkpoint-tag',tag)],entrypoint='export_parliament_viewer.py',
                            output_flag='--report',in_place=True)

def pinned_report_path(pinned, ident):
    raw = Path(pinned[ident])
    return raw if raw.is_absolute() else ROOT / raw


def catalog_pins(catalog):
    data = json.loads(catalog.read_text())
    if data.get('version') not in (None, 'parliament-saved-runs/1'):
        raise ValueError('Unknown saved-run catalog version')
    pinned = data.get('reports')
    if not isinstance(pinned, dict):
        raise ValueError('Saved-run catalog must list reports')
    return pinned


def framework_views_for_report(report):
    """Refresh legacy prose from its own saved ledger, preserving all decisions.

    Exporting an old demo must not resurrect the protected-party/duty-bearer
    formatter bug. Never borrow a ledger from a different source or world.
    """
    views = copy.deepcopy(report.get('frameworks', []))
    for view in views:
        view.pop('argument_markdown_path', None)
    if not any(not v.get('recorded_argument') for v in views):
        return views
    trace = native_trace_from_report(report, ROOT)
    if (not trace or trace.get('scenario') != report.get('scenario')
            or (trace.get('action_source_grounding') or {}).get('world_model') != report.get('world')):
        return views
    from run_parliament_demo import candidate_reason
    from parliament_framework_presentation import enrich_view, preserve_model_arguments
    candidates = copy.deepcopy(trace['cycles'][-1].get('candidates', []))
    for artifact in report.get('artifacts') or []:
        path = ROOT / artifact
        if str(path).endswith('framework_generation.json') and path.is_file():
            if json.loads(path.read_text()) == trace:
                preserve_model_arguments(candidates, path.parent)
                break
    by_name = {c['specialist']: c for c in candidates}
    for view in views:
        c = by_name.get(view['framework'])
        if view.get('recorded_argument') or not c:
            continue
        ledger = c.get('committed_native_ledger') or {}
        if ((c.get('recommended_action') or None) != view.get('preference')
                or ledger.get('transaction_status', 'NONE') != view.get('ledger_status')):
            continue
        view['reason'] = candidate_reason(c, view['position'], view.get('preference'))
        enrich_view(view, c, report['world'])
        view['presentation_revision'] = 'FROM_MATCHING_SAVED_NATIVE_RECORD'
    return views


def topography_for_report(report):
    """Copy native topography; never invent coordinates or keep local paths."""
    if report.get('status') == 'INTERPRETIVE_BEAM':
        return unavailable('No unique frozen world was supplied to agents; no framework coordinates exist.')
    existing = report.get('topography')
    if isinstance(existing, dict) and existing.get('status') == 'RECORDED' and existing.get('actions'):
        return copy.deepcopy(existing)
    trace = report if report.get('cycles') else native_trace_from_report(report, ROOT)
    if trace is None:
        trace = ident_fallback_trace(report)
    if trace is None:
        return unavailable('Native axis records were not stored in this demo report.')
    return topography_from_trace(trace, report.get('world') or {}, report.get('frameworks') or [])


def ident_fallback_trace(report):
    """Medicine's saved demo report predates stored EV; use the matching-scenario native trace if present."""
    if not MEDICINE_FALLBACK_TRACE.is_file():
        return None
    if (report.get('scenario') or '') != (
        'A clinic has one dose of medicine. Ada can give the medicine to Ben or Cara, but not both. '
        'If Ada gives the medicine to Ben, Ben has a 95% chance of survival. If Ada gives the medicine '
        'to Cara, Cara has a 5% chance of survival. The patient who does not get the medicine will die.'
    ):
        return None
    trace = json.loads(MEDICINE_FALLBACK_TRACE.read_text())
    return trace if trace.get('scenario') == report.get('scenario') else None


def _export(reports=None, output=DEFAULT_OUTPUT, trolley_report=None, catalog=None):
    output.mkdir(parents=True, exist_ok=True)
    pinned = {}
    catalog = Path(catalog) if catalog is not None else DEFAULT_CATALOG
    if reports is None and catalog.is_file():
        pinned = catalog_pins(catalog)
    reports = Path(reports) if reports is not None else DEFAULT_REPORTS
    index = []
    titles = {'medicine':'One dose · unequal survival', 'promise':'One dose · competing promise',
              'trolley':'Trolley · competing interpretations'}
    # Keep the original display names while allowing new saved demos through
    # the same catalog and presentation contract.
    for ident in pinned:
        if ident not in titles:
            titles[ident] = ident.replace('_', ' ').replace('-', ' ').capitalize()
    for ident, title in titles.items():
        path = (trolley_report if ident == 'trolley' and trolley_report else
                pinned_report_path(pinned, ident) if ident in pinned else reports/ident/'DEMO_REPORT.json')
        from parliament_run_provenance import record_resource
        record_resource(path)
        report = json.loads(path.read_text())
        if ident == 'trolley' and report.get('world'):
            title = 'Trolley · lever and tracks'
        payload = presentation(report,ident,title)
        from parliament_framework_presentation import argument_markdown
        for view in payload['frameworks']:
            name = view['framework']
            if name not in {'utilitarian','deontological','care','virtue','rawlsian','contractualist'}:
                continue
            filename = ident+'_'+name+'_argument.md'
            (output/filename).write_text(argument_markdown(view), encoding='utf-8')
            view['argument_url'] = 'data/'+filename
        (output/(ident+'.json')).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
        index.append({'id':ident,'title':title,'status':payload['status'],'file':ident+'.json'})
    (output/'runs.json').write_text(json.dumps({'schema_version':'parliament-view-index/1','runs':index},indent=2)+'\n')
    from parliament_run_provenance import record_artifacts
    record_artifacts(list(output.glob('*.json'))+list(output.glob('*_argument.md')))
    from parliament_run_provenance import record_checkpoint
    record_checkpoint('viewer_exported',artifacts=[output/'runs.json'])
    return index


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,help='Override the pinned saved-run catalog with a report directory')
    p.add_argument('--catalog',type=Path,help='Saved-run catalog; defaults to frontend/saved_runs.json')
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--trolley-report',type=Path,help='Use a newer saved trolley report without overwriting historical runs')
    p.add_argument('--checkpoint-tag',action='append',help='Run label, inherited by child runs; repeatable')
    a=p.parse_args()
    print('Exported '+str(len(export(a.reports,a.output,a.trolley_report,a.catalog,a.checkpoint_tag)))+' saved runs; no model calls.')
