"""Audit frozen demo sources offline, or explicitly execute selected live cases."""
import argparse
import hashlib
import json
from pathlib import Path


def suite_failed(results):
    """World admission alone cannot hide a later execution failure."""
    return any(r.get('parse_status') == 'SOFTWARE_FAILED' or
               r.get('pipeline_status') == 'SOFTWARE_FAILED' or
               not r.get('exact_spans', False) or
               r.get('expectation_match') is False for r in results)


def write_summary(output, results):
    lines = ['# Demo suite results', '',
        'Offline audits verify parsing and spans only. Live freeze checks do not certify argument quality.', '',
        '| Case | Parse | Freeze | Final pipeline | Freeze expectation |',
        '| --- | --- | --- | --- | --- |']
    for row in results:
        lines.append('| '+ ' | '.join(str(row.get(key, 'NOT_RUN')) for key in
            ('id', 'parse_status', 'freeze_status', 'pipeline_status', 'expectation_match'))+' |')
    lines += ['', 'Topology, unknowns, and hinge expectations still require content review; they are not automatically scored.']
    (output/'SUITE_RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true', help='Run live API pipeline; default only audits Z10/inventory')
    p.add_argument('--cases', nargs='+')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use a new output directory.')
    rows=json.loads(Path('demo_scenarios/manifest.json').read_text())['scenarios']
    if a.cases and set(a.cases)-{r['id'] for r in rows}:
        p.error('Unknown case ID.')
    a.output.mkdir(parents=True, exist_ok=True)
    import parsing_game_Z10 as z10
    from blueprint_primitive_composition import extract_inventory
    from run_parliament_demo import main as run
    results=[]
    for fixture in rows:
        if a.cases and fixture['id'] not in a.cases:
            continue
        folder=a.output/fixture['id']; text=Path(fixture['scenario']).read_text().strip()
        if hashlib.sha256(text.encode()).hexdigest()!=fixture['source_sha256']:
            raise ValueError('Frozen demo source changed: '+fixture['id'])
        try:
            package=z10.export_candidate_graph(text, package_id='demo_'+fixture['id'])
            inventory=extract_inventory(text, package)
            folder.mkdir()
            (folder/'z10_package.json').write_text(json.dumps(package, indent=2)+'\n')
            (folder/'primitive_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
            spans_ok=all(text[e['start']:e['end']]==e['text'] for e in package['evidence'])
            result={'id':fixture['id'], 'parse_status':'PACKAGE_WITH_VISIBLE_GAPS',
                'exact_spans':spans_ok,'open_question_count':len(package['open_questions']),
                'expected_freeze_status':fixture['expected_freeze_status'], 'freeze_status':'NOT_RUN',
                'expectation_match':'NOT_ASSESSED_OFFLINE'}
            if a.execute:
                run(['--scenario',fixture['scenario'],'--output',str(folder/'live')])
                report=json.loads((folder/'live/DEMO_REPORT.json').read_text())
                result['pipeline_status']=report['status']
                manifest_path=folder/'live/pipeline/preparation_manifest.json'
                native=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                actual=native.get('pipeline_status','SOFTWARE_FAILED')
                actual={'ADMITTED':'FROZEN','WITHHELD':'CONSTRUCTION_WITHHELD'}.get(actual,actual)
                result['freeze_status']=actual
                result['expectation_match']=actual==fixture['expected_freeze_status']
        except Exception as exc:
            result={'id':fixture['id'],'parse_status':'SOFTWARE_FAILED','error':str(exc)}
        results.append(result)
        (a.output/'SUITE_RESULTS.json').write_text(json.dumps(results, indent=2)+'\n')
        write_summary(a.output, results)
        print(fixture['id']+': '+result['parse_status']+'; freeze='+result.get('freeze_status','NOT_RUN'),flush=True)
    return 1 if suite_failed(results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
