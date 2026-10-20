"""Offline replay of saved laboratory duty submissions; no live backend."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parliament-root', type=Path, default=Path('.runtime/parliament'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.parliament_root.resolve()))
    from logic_puzzles_frameworks import generate
    from logic_puzzles_acceptance import SavedResponses
    from logic_puzzles_review import review, close_deontological_map
    from run_logic_puzzles import project_conflicts
    source = Path('diagnostics/deontic_choice_composition_20261016/live_final/frozen_world_trace.json')
    original = Path('diagnostics/deontic_choice_composition_20261016/deliberation/frameworks')
    live = Path('diagnostics/deontology_rule_attention_20261019/live')
    quotes = json.loads((original/'framework_commitments.json').read_text())
    before = source.read_bytes()
    results = []
    for name, folder in [('original', original), ('rule_context', live/'rule_context'),
                         ('rule_context_display_closed', live/'rule_context'), ('accepted_control', live/'control')]:
        calls_path = folder/'deontological/model_calls.json'
        calls = json.loads(calls_path.read_text())
        backend = SavedResponses([c['response'] for c in calls])
        result = generate(source, args.output_dir/name, ['deontological'], lambda _: backend,
                          call_budget=2, core_quotes=quotes, rule_attention=name.startswith('rule_context'),
                          response_normalizers={'deontological':close_deontological_map}
                          if name == 'rule_context_display_closed' else None)
        c = result['cycles'][-1]['candidates'][0]
        results.append({'case': name, 'delegate_status': c['delegate_status'],
                        'ledger': c.get('committed_native_ledger') or {},
                        'validation_errors': c.get('validation_errors', []),
                        'framework_warnings': c.get('framework_validation_errors', []),
                        'saved_response_sha256': hashlib.sha256(calls_path.read_bytes()).hexdigest(),
                        'world_unchanged': result['action_source_grounding'] == json.loads(before)['action_source_grounding']})
    assessment = original/'framework_generation.json'
    issues = [n['id'] for n in project_conflicts(json.loads(assessment.read_text()))['nodes']
              if n.get('origin') == 'deontological' and n['kind'] == 'UNRESOLVED_CONFLICT'][:2]
    path = live/'saved_argument_review/deontological/model_calls.json'
    backend = SavedResponses([c['response'] for c in json.loads(path.read_text())])
    result = review(assessment, args.output_dir/'accepted_review', lambda _: backend,
                    issue_ids=issues, max_calls=2, allow_partial=False, allow_residual=False)
    c = next(c for c in result['cycles'][-1]['candidates'] if c['specialist']=='deontological')
    results.append({'case':'accepted_review', 'delegate_status':c['delegate_status'],
                    'ledger':c.get('committed_native_ledger') or {},
                    'framework_warnings':c.get('framework_validation_errors', []),
                    'saved_response_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'world_unchanged':result['action_source_grounding']==json.loads(before)['action_source_grounding']})
    assert source.read_bytes() == before and all(r['world_unchanged'] for r in results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'comparison.json').write_text(json.dumps({'mode':'SAVED_RESPONSE_REPLAY',
        'new_api_calls':0,'source_sha256':hashlib.sha256(before).hexdigest(),'results':results},indent=2)+'\n')
    print([(r['case'],r['delegate_status'],r['ledger'].get('transaction_status','NONE')) for r in results])


if __name__ == '__main__':
    main()
