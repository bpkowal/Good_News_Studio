"""Five-perspective demo: four native specialists plus Contractualist questions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from contractualist_world_adapter import framework_view, fingerprint, VERSION as ADAPTER_VERSION
from parliament_runtime import DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON, check_runtime
from parliament_run_provenance import (current_provenance, record_checkpoint,
    record_configuration, record_execution, record_resource, record_runtime, record_start)
from run_parliament_demo import summarize, write_report

ROOT = Path(__file__).resolve().parent
NATIVE_FRAMEWORKS = ('utilitarian', 'deontological', 'care', 'virtue')


def assemble_report(scenario, world, native_trace, contractualist_result, contractualist_source):
    if native_trace.get('scenario') != scenario:
        raise ValueError('Native framework source differs from the frozen scenario')
    if (native_trace.get('action_source_grounding') or {}).get('world_model') != world:
        raise ValueError('Native frameworks did not retain the shared frozen world')
    if contractualist_source['scenario'] != scenario or contractualist_source['admitted_world'] != world:
        raise ValueError('Contractualist adapter did not receive the same source and world')
    if contractualist_result['shared_world_sha256'] != fingerprint(world):
        raise ValueError('Contractualist result is bound to another world')
    candidates = [c for cycle in native_trace.get('cycles', []) for c in cycle.get('candidates', [])]
    if any(c['specialist'] not in NATIVE_FRAMEWORKS for c in candidates):
        raise ValueError('Unexpected specialist in this replacement run; preserve it as its own run')
    report = summarize(scenario, {'pipeline_status': 'ADMITTED'}, native_trace)
    report['frameworks'].append(framework_view(contractualist_result, contractualist_source))
    report['agent_roster'] = {'native_specialists': list(NATIVE_FRAMEWORKS),
        'argument_specialists': ['contractualist'], 'replaced_for_this_run': 'rawlsian',
        'mode': 'INDEPENDENT_PASSES_NO_COLLECTIVE_VERDICT',
        'shared_world_sha256': fingerprint(world), 'new_world_construction': False}
    report['status'] = 'FRAMEWORK_PARTIAL' if any(v['position'] != 'FULL' for v in report['frameworks']) else 'COMPLETED'
    report['stop'] = {'stage': 'five-perspective assessment', 'kind': 'recorded_pipeline_outcome',
        'reason': 'Four native framework passes plus a question-capable Contractualist argument. Open comparisons remain visible.'}
    view = report['frameworks'][-1]
    for i, q in enumerate(view['critical_questions']):
        record = 'contractualist/response.json#/critical_questions/' + str(i)
        report['hinges'].append({'framework': 'contractualist', 'text': q['question'], 'record': record,
            'authority': 'MODEL_REPORTED_NOT_INDEPENDENTLY_VERIFIED', 'role': 'OPEN_QUESTION'})
        report['unknowns'].append('Contractualist ' + q['kind'].lower() + ' question: ' + q['question'])
        report['disagreements'].append({'framework': 'contractualist', 'kind': q['kind'].lower(),
                                       'dependency': q['question'], 'record': record})
    report['agreement'] += ' A Contractualist question-focused contribution is not counted as agreement or a contrary vote.'
    report['not_decided'].append('The Contractualist adapter contributes attributed arguments and questions, not a committed native ledger or an operative vote.')
    from parliament_topography import topography_from_trace
    report['topography'] = topography_from_trace(native_trace, world, report['frameworks'])
    report['topography']['framework_arrows'] = [a for a in report['topography'].get('framework_arrows', [])
                                               if a['framework'] in NATIVE_FRAMEWORKS]
    report['topography'].setdefault('notices', []).append(
        'Contractualist questions appear in Frameworks and Hinges; no Rawls score or Contractualist coordinate is inferred.')
    return report


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True, help='An already admitted frozen-world trace')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parliament-root', type=Path, default=DEFAULT_PARLIAMENT_ROOT)
    parser.add_argument('--parliament-python', type=Path, default=DEFAULT_PARLIAMENT_PYTHON)
    parser.add_argument('--model', default='o3')
    parser.add_argument('--checkpoint-tag', action='append')
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory to preserve saved evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    record_configuration(args); record_start(args.output); record_resource(args.trace)
    before = args.trace.read_bytes(); frozen = json.loads(before)
    scenario, world = frozen['scenario'], frozen['action_source_grounding']['world_model']
    runtime = check_runtime(args.parliament_root, args.parliament_python)
    record_runtime(runtime)
    (args.output/'runtime_check.json').write_text(json.dumps(runtime, indent=2)+'\n')
    if not runtime['ready']:
        raise RuntimeError('Runtime preflight failed: ' + '; '.join(runtime['errors']))
    commands = []
    def run(command, label):
        commands.append(command)
        print(label + '...', flush=True)
        with (args.output/(label+'.log')).open('w') as log:
            return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    native_folder, contract_folder = args.output/'frameworks', args.output/'contractualist'
    native_code = run([str(args.parliament_python), str(ROOT/'logic_puzzles_frameworks.py'),
        '--trace', str(args.trace.resolve()), '--output-dir', str(native_folder.resolve()),
        '--parliament-root', str(args.parliament_root), '--model', args.model,
        '--frameworks', *NATIVE_FRAMEWORKS], 'native_frameworks')
    contract_code = run([str(args.parliament_python), str(ROOT/'contractualist_world_adapter.py'),
        '--trace', str(args.trace.resolve()), '--output-dir', str(contract_folder.resolve()),
        '--parliament-root', str(args.parliament_root), '--model', args.model], 'contractualist_argument')
    if args.trace.read_bytes() != before:
        raise RuntimeError('Input frozen trace changed')
    if native_code or contract_code:
        print('A process failed; saved logs and any partial arguments remain in ' + str(args.output), flush=True)
        return 1
    native = json.loads((native_folder/'framework_generation.json').read_text())
    result = json.loads((contract_folder/'result.json').read_text())
    source = json.loads((contract_folder/'source.json').read_text())
    report = assemble_report(scenario, world, native, result, source)
    report['run_provenance'] = current_provenance()
    report['generation_provenance'] = current_provenance()
    report['execution'] = {'mode': 'LIVE_FRAMEWORKS_ON_SAVED_FROZEN_WORLD', 'commands': commands,
                           'frozen_trace_sha256': hashlib.sha256(before).hexdigest()}
    report['runtime_check'] = runtime
    report['artifacts'] = [str(args.trace.resolve())] + [str(p.resolve()) for p in
        (native_folder/'framework_generation.json', contract_folder/'result.json',
         contract_folder/'source.json', contract_folder/'critical_questions.md',
         native_folder/'run_manifest.json', contract_folder/'run_manifest.json') if p.is_file()]
    write_report(report, args.output/'report')
    record_checkpoint('five_perspective_report_written', artifacts=[args.output/'report/DEMO_REPORT.json'])
    print('Five perspectives recorded: ' + str(args.output/'report/DEMO_REPORT.md'), flush=True)
    return 0


def main(argv=None):
    return record_execution(_main, argv, entrypoint='run_contractualist_demo.py', output_flag='--output',
                            defaults={'model': 'o3', 'runtime_profile': 'compact_frameworks'})


if __name__ == '__main__':
    raise SystemExit(main())
