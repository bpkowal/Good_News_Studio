"""Export a small, path-free presentation record from saved demo reports."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import copy

ROOT = Path(__file__).resolve().parent
DEFAULT_REPORTS = ROOT/'diagnostics/parliament_demo_20261007/reports'
DEFAULT_OUTPUT = ROOT/'frontend/dist/data'


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
        nodes, layer = source_ledger_layer(diagram['world'], report.get('stipulated_claims', [])) if diagram['authority'] == 'FROZEN' else ([], '')
        diagram['source_ledger_nodes'] = nodes
        diagram['source_ledger_mermaid'] = layer
        diagram['world_sha256'] = hashlib.sha256(json.dumps(diagram['world'],sort_keys=True).encode()).hexdigest()
    return {'schema_version':'parliament-view/1', 'id':ident, 'title':title,
        'source':report['scenario'], 'source_sha256':hashlib.sha256(report['scenario'].encode()).hexdigest(),
        'status':report['status'], 'stop':report['stop'], 'judgment_status':report.get('judgment_status'),
        'mode':report.get('execution',{}).get('mode','SAVED_RECORD'), 'diagrams':diagrams,
        'frameworks':report.get('frameworks',[]), 'hinges':report.get('hinges',[]),
        'unknowns':report.get('unknowns',[]), 'disagreements':report.get('disagreements',[]),
        'agreement':report.get('agreement'), 'not_decided':report.get('not_decided',[]),
        'stipulated_claims':report.get('stipulated_claims',[]), 'missing_slots':report.get('missing_slots',[]),
        'diagram_notices':report.get('diagram_notices',[]),
        'provenance':{'origin':'SAVED_DEMO_REPORT', 'new_model_calls':0,
            'report_sha256':hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()}}


def export(reports=DEFAULT_REPORTS, output=DEFAULT_OUTPUT, trolley_report=None):
    output.mkdir(parents=True, exist_ok=True)
    index = []
    titles = {'medicine':'One dose · unequal survival', 'promise':'One dose · competing promise',
              'trolley':'Trolley · competing interpretations'}
    for ident, title in titles.items():
        path = trolley_report if ident == 'trolley' and trolley_report else reports/ident/'DEMO_REPORT.json'
        report = json.loads(path.read_text())
        if ident == 'trolley' and report.get('world'):
            title = 'Trolley · lever and tracks'
        payload = presentation(report,ident,title)
        (output/(ident+'.json')).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
        index.append({'id':ident,'title':title,'status':payload['status'],'file':ident+'.json'})
    (output/'runs.json').write_text(json.dumps({'schema_version':'parliament-view-index/1','runs':index},indent=2)+'\n')
    return index


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,default=DEFAULT_REPORTS)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--trolley-report',type=Path,help='Use a newer saved trolley report without overwriting historical runs')
    a=p.parse_args()
    print('Exported '+str(len(export(a.reports,a.output,a.trolley_report)))+' saved runs; no model calls.')
