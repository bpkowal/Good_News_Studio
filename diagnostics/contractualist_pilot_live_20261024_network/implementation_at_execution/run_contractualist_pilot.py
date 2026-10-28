"""Prepare frozen neutral framework requests; execute only with --execute."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from contractualist_pilot import (FRAMEWORKS, PAPER, VERSION, source_packet,
                                  make_request, compile_response, write_argument_report)
from argument_frameworks import FRAMEWORK_GUIDANCE, VERSION as INSTRUCTION_VERSION

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(fixture_path, output, *, cases=None, frameworks=FRAMEWORKS,
            model='o3', reasoning_effort='low', max_tokens=8192):
    fixture_bytes = Path(fixture_path).read_bytes()
    fixture = json.loads(fixture_bytes)
    all_cases = {c['id']: c for c in fixture['cases']}
    if len(all_cases) != len(fixture['cases']):
        raise ValueError('Duplicate case IDs')
    chosen = list(cases) if cases else list(all_cases)
    frameworks = list(frameworks)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= all_cases.keys():
        raise ValueError('Unknown or duplicate cases')
    if not frameworks or len(set(frameworks)) != len(frameworks) or not set(frameworks) <= set(FRAMEWORKS):
        raise ValueError('Unknown or duplicate frameworks')
    packets = [source_packet(all_cases[c]) for c in chosen]
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', p['id']) for p in packets):
        raise ValueError('Unsafe case ID')
    requests = [(p, f, make_request(p, f, model=model, reasoning_effort=reasoning_effort,
                                   max_tokens=max_tokens)) for p in packets for f in frameworks]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/'fixture.json', fixture)
    protocol = {'version': VERSION, 'instruction_version': INSTRUCTION_VERSION, 'prepared_at': now(),
        'model': model, 'reasoning_effort': reasoning_effort, 'max_completion_tokens': max_tokens,
        'maximum_provider_calls': len(requests), 'sdk_retries': 0, 'tools': False,
        'peer_input': False, 'repairs': False, 'world_admission': False,
        'case_ids': chosen, 'frameworks': frameworks, 'framework_instructions':
        {f: FRAMEWORK_GUIDANCE[f] for f in frameworks}, 'paper_attribution': PAPER,
        'fixture_file_sha256': digest(fixture_bytes),
        'code_sha256': {name: digest((ROOT/name).read_bytes()) for name in
            ('argument_frameworks.py', 'argument_support.py', 'argument_construction.py',
             'contractualist_pilot.py', 'run_contractualist_pilot.py')},
        'limits': ['One sample per cell; exploratory reasoning comparison.',
            'Text fixtures are not admitted worlds. No operative framework roster change.',
            'Binding checks and model-authored comparisons do not establish moral correctness.',
            'Common envelope and budgets match; framework instructions intentionally differ.']}
    write_new(output/'protocol.json', protocol)
    files = ['fixture.json', 'protocol.json']
    for source, framework, request in requests:
        folder = output/source['id']/framework
        folder.mkdir(parents=True)
        write_new(folder/'source.json', source)
        write_new(folder/'request.json', request)
        files.extend([str((folder/name).relative_to(output)) for name in ('source.json', 'request.json')])
    write_new(output/'inputs.freeze.json', {'frozen_at': now(),
        'files': {name: digest((output/name).read_bytes()) for name in files}})
    (output/'RESULT.md').write_text('# Neutral framework pilot\n\nPrepared only; no API calls.\n')
    return protocol


def verify_inputs(output):
    output = Path(output)
    frozen = json.loads((output/'inputs.freeze.json').read_text())
    for name, expected in frozen['files'].items():
        if digest((output/name).read_bytes()) != expected:
            raise ValueError('Frozen input changed: ' + name)
    protocol = json.loads((output/'protocol.json').read_text())
    for name, expected in protocol['code_sha256'].items():
        if digest((ROOT/name).read_bytes()) != expected:
            raise ValueError('Prepared implementation changed; use a fresh study directory')
    return protocol


def execute(output, client_factory=None):
    output = Path(output)
    protocol = verify_inputs(output)
    if client_factory is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(ROOT.parent/'RAGAIMODEL'/'.env')
        client_factory = lambda: OpenAI(max_retries=0, timeout=180)
    client = None
    results = []
    for case in protocol['case_ids']:
        for framework in protocol['frameworks']:
            folder = output/case/framework
            if (folder/'call.json').exists():
                # Even interrupted/provider-error attempts consume the one-call budget.
                raise ValueError('Cell already attempted; inspect saved output instead of retrying')
            if client is None:
                client = client_factory()
            request = json.loads((folder/'request.json').read_text())
            write_new(folder/'call.json', {'started_at': now(), 'provider_attempts': 1,
                                          'request_sha256': digest((folder/'request.json').read_bytes())})
            print(case + ' / ' + framework + ': calling ' + protocol['model'], flush=True)
            try:
                response = client.chat.completions.create(**request)
                write_new(folder/'provider_response.json', response.model_dump(mode='json'))
                choice = response.choices[0]
                message = choice.message
                if choice.finish_reason != 'stop' or getattr(message, 'refusal', None):
                    result = {'framework': framework, 'status': 'INCOMPLETE_OR_REFUSED',
                              'finish_reason': choice.finish_reason, 'operative': False}
                else:
                    raw = json.loads(message.content)
                    write_new(folder/'response.json', raw)
                    source = json.loads((folder/'source.json').read_text())
                    result = compile_response(source, framework, raw)
                    write_argument_report(result, folder/'candidate_arguments.md')
            except Exception as error:
                # Do not log provider exception messages, which can contain credentials.
                result = {'framework': framework, 'status': 'FAILED',
                          'error_type': type(error).__name__, 'operative': False}
            write_new(folder/'result.json', result)
            files = [p for p in folder.iterdir() if p.is_file() and p.name != 'output.freeze.json']
            write_new(folder/'output.freeze.json', {'frozen_at': now(),
                'files': {p.name: digest(p.read_bytes()) for p in files}})
            results.append({'case_id': case, 'framework': framework, 'status': result['status'],
                            'binding_issues': len(result.get('issues', []))})
            print(case + ' / ' + framework + ': ' + result['status'], flush=True)
            write_new_summary(output, results)
    return results


def write_new_summary(output, results):
    (output/'results.json').write_text(json.dumps(results, indent=2) + '\n')
    lines = ['# Neutral framework pilot', '',
        'One independent generation per cell. No repairs, peer exposure, world admission, or moral-quality verdict.', '',
        '| Case | Framework | Technical status | Binding issues |', '| --- | --- | --- | --- |']
    for row in results:
        link = row['case_id'] + '/' + row['framework'] + '/result.json'
        lines.append('| ' + ' | '.join([row['case_id'], row['framework'],
            '[' + row['status'] + '](' + link + ')', str(row['binding_issues'])]) + ' |')
    lines += ['', 'Raw responses, compiled graphs, and per-cell output hashes remain in each case/framework directory.',
              'Technical status is separate from the content review specified in CONTRACTUALIST_PILOT.md.', '']
    (output/'RESULT.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'fixtures/contractualist_pilot_cases.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--frameworks', nargs='+', choices=FRAMEWORKS, default=list(FRAMEWORKS))
    parser.add_argument('--model', default='o3')
    parser.add_argument('--reasoning-effort', choices=('low', 'medium', 'high'), default='low')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.max_tokens < 1024:
        parser.error('Use at least 1024 completion tokens')
    protocol = prepare(args.source, args.output, cases=args.case, frameworks=args.frameworks,
                       model=args.model, reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)
    if args.execute:
        execute(args.output)
    else:
        print('Prepared ' + str(protocol['maximum_provider_calls']) + ' requests; no API calls.')


if __name__ == '__main__':
    main()
