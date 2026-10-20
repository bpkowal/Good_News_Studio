"""Matched deontology rule-attention pair and one saved-argument challenge."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from parliament_runtime import DEFAULT_PARLIAMENT_ROOT, check_runtime


def run(args, backend_factory):
    from logic_puzzles_frameworks import generate, fingerprint
    from logic_puzzles_review import review
    from run_logic_puzzles import project_conflicts
    source_bytes = args.trace.read_bytes()
    source = json.loads(source_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quotes = json.loads((args.saved_assessment.parent/'framework_commitments.json').read_text())
    runs = {}
    for name, attention in [('control', False), ('rule_context', True)]:
        runs[name] = generate(args.trace, args.output_dir/name, ['deontological'],
                              backend_factory, tokens=3072, call_budget=2,
                              core_quotes=quotes, rule_attention=attention)
    original = json.loads(args.saved_assessment.read_text())
    issues = [n['id'] for n in project_conflicts(original)['nodes']
              if n.get('origin') == 'deontological' and n['kind'] == 'UNRESOLVED_CONFLICT'][:2]
    if not issues:
        raise ValueError('Saved deontological argument has no reviewable issue')
    reviewed = review(args.saved_assessment, args.output_dir/'saved_argument_review',
                      backend_factory, tokens=3072, max_calls=2,
                      allow_partial=False, allow_residual=False, issue_ids=issues)
    shared = {name: json.loads((args.output_dir/name/'shared_input.json').read_text()) for name in runs}
    calls = {name: json.loads((args.output_dir/name/'deontological/model_calls.json').read_text()) for name in runs}
    review_calls = json.loads((args.output_dir/'saved_argument_review/deontological/model_calls.json').read_text())
    checks = {'identical_shared_input': shared['control'] == shared['rule_context'],
              'identical_core_pack': (args.output_dir/'control/framework_commitments.json').read_bytes() ==
                                     (args.output_dir/'rule_context/framework_commitments.json').read_bytes(),
              'source_bytes_unchanged': args.trace.read_bytes() == source_bytes,
              'worlds_unchanged': all(r['action_source_grounding'] == source['action_source_grounding']
                                      for r in [*runs.values(), reviewed]),
              'same_model_and_budgets': all(c['model'] == args.model and c['max_tokens'] == 3072
                  for records in [*calls.values(), review_calls] for c in records) and
                  all(len(records) <= 2 for records in [*calls.values(), review_calls])}
    controls = {'checks': checks, 'model': args.model, 'tokens': 3072,
                'maximum_adapter_calls_per_arm': 2, 'maximum_saved_review_calls': 2,
                'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
                'saved_assessment_sha256': hashlib.sha256(args.saved_assessment.read_bytes()).hexdigest(),
                'shared_input_sha256': {k: fingerprint(v) for k,v in shared.items()},
                'treatment': 'Source-bound rule context plus rule assessment instruction only; admission unchanged.',
                'limits': 'One unblinded pair, not a treatment-effect estimate. Saved-argument review is a separate probe.'}
    (args.output_dir/'controls.json').write_text(json.dumps(controls, indent=2)+'\n')
    if not all(checks.values()):
        raise RuntimeError('Matched controls failed; inspect controls.json')
    rows = []
    for name, result in [*runs.items(), ('saved_argument_review', reviewed)]:
        c = result['cycles'][-1]['candidates'][0] if name != 'saved_argument_review' else next(
            c for c in result['cycles'][-1]['candidates'] if c['specialist']=='deontological')
        rows.append({'arm': name, 'preference': c.get('recommended_action'),
                     'native_ledger': (c.get('committed_native_ledger') or {}).get('transaction_status', 'NONE'),
                     'warnings': c.get('framework_validation_errors', []),
                     'semantic_quality': 'REQUIRES_CONTENT_REVIEW'})
    (args.output_dir/'comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
    return 0


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--saved-assessment', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='o3')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error('--execute is required for live API calls')
    from parliament_run_provenance import record_configuration, record_runtime
    record_configuration(args)
    from parliament_run_provenance import record_resource
    record_resource(args.saved_assessment)
    runtime = check_runtime(DEFAULT_PARLIAMENT_ROOT, Path(sys.executable))
    record_runtime(runtime)
    if not runtime['ready']:
        raise RuntimeError('Pinned native runtime is not ready')
    sys.path.insert(0, str(DEFAULT_PARLIAMENT_ROOT))
    from dotenv import load_dotenv
    from global_workspace.openai_backend import OpenAIWorkspaceLLM
    load_dotenv(Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env'))
    backend = OpenAIWorkspaceLLM(args.model, timeout=120)
    backend.execution_mode = 'LIVE_OPENAI'
    return run(args, lambda name: backend)


def main(argv=None):
    from parliament_run_provenance import record_execution
    return record_execution(_main, argv, entrypoint='run_deontology_rule_attention_pilot.py',
                            output_flag='--output-dir', defaults={'model':'o3','runtime_profile':'compact_frameworks'})


if __name__ == '__main__':
    main()
