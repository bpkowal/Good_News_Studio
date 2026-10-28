"""Freeze one opposing complaint per saved argument; compare two reply arms."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import re
import sys

from opposing_complaint import (VERSION, ARMS, SELECTOR_INSTRUCTION, SHARED_REPLY,
    ORDINARY, FOCUSED, selector_request, bind_assignment, reply_request, bind_reply)
from run_contractualist_pilot import digest, now, write_new

ROOT = Path(__file__).resolve().parent


def freeze(folder, name, paths):
    write_new(folder/name, {'frozen_at': now(),
        'files': {str(p.relative_to(folder)): digest(p.read_bytes()) for p in paths}})


def verify_freeze(folder, name):
    receipt = json.loads((folder/name).read_text())
    for filename, expected in receipt['files'].items():
        if digest((folder/filename).read_bytes()) != expected:
            raise ValueError('Frozen artifact changed: ' + filename)


def prepare(source_run, output, *, cases=None, framework='contractualist'):
    source_run, output = Path(source_run), Path(output)
    prior_protocol = json.loads((source_run/'protocol.json').read_text())
    chosen = list(cases) if cases else prior_protocol['case_ids']
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(prior_protocol['case_ids']):
        raise ValueError('Unknown or duplicate cases')
    if framework not in prior_protocol['frameworks']:
        raise ValueError('Framework not in prior run')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in [framework, *chosen]):
        raise ValueError('Unsafe case or framework ID')
    packets, origins = {}, {}
    for case in chosen:
        folder = source_run/case/framework
        verify_freeze(folder, 'output.freeze.json')
        request = json.loads((folder/'request.json').read_text())
        prior_user = json.loads(request['messages'][1]['content'])
        source = json.loads((folder/'source.json').read_text())
        if prior_user['source'] != source:
            raise ValueError('Prior request source mismatch')
        if digest(source['scenario'].encode()) != source['source_text_sha256']:
            raise ValueError('Source text hash mismatch')
        prior = json.loads((folder/'response.json').read_text())
        packets[case] = {'framework_id': framework,
            'framework_instruction': prior_user['framework_instruction'],
            'source': source, 'prior_argument': prior,
            'prior_argument_origin': 'SAVED_MODEL_ARGUMENT_NOT_WORLD_FACT'}
        origins[case] = {'folder': str(folder), 'source_sha256': digest((folder/'source.json').read_bytes()),
            'prior_response_sha256': digest((folder/'response.json').read_bytes()),
            'prior_request_sha256': digest((folder/'request.json').read_bytes())}
    output.mkdir(parents=True, exist_ok=False)
    implementation = output/'implementation'; implementation.mkdir()
    names = ('opposing_complaint.py', 'run_opposing_complaint_probe.py',
             'run_contractualist_pilot.py', 'contractualist_pilot.py', 'argument_construction.py',
             'argument_support.py', 'argument_frameworks.py', 'inference_boundaries.py',
             'deontology_repair_cards.py')
    versions = {}
    for package in ('openai', 'python-dotenv'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    protocol = {'version': VERSION, 'prepared_at': now(), 'framework': framework, 'cases': chosen,
        'model': prior_protocol['model'], 'reasoning_effort': prior_protocol['reasoning_effort'],
        'selector_max_tokens': 4096, 'reply_max_tokens': prior_protocol['max_completion_tokens'],
        'maximum_provider_calls': 3*len(chosen), 'sdk_retries': 0,
        'stages': ['freeze_original_inputs', 'all_complaint_assignments', 'matched_reply_arms'],
        'arms': list(ARMS), 'source_run': str(source_run), 'origins': origins,
        'instructions': {'selector': SELECTOR_INSTRUCTION, 'common_reply': SHARED_REPLY,
                         'ordinary': ORDINARY, 'focused': FOCUSED},
        'code_sha256': {name: digest((ROOT/name).read_bytes()) for name in names},
        'runtime': {'python': sys.version, 'packages': versions},
        'world_mutation': False, 'native_dispatch': False, 'peer_input': False,
        'limits': ['One assignment and one response per arm per case; exploratory.',
            'Assignment is a fallible model-proposed objection, not a verified strongest complaint.',
            'Both arms receive the same assignment and saved argument; no cross-arm replies.',
            'Extra review versus the opening is not a matched-cost treatment comparison.',
            'Self-reported answer status does not retire any objection or commit a ledger.']}
    write_new(output/'protocol.json', protocol)
    paths = [output/'protocol.json']
    for name in names:
        (implementation/name).write_bytes((ROOT/name).read_bytes()); paths.append(implementation/name)
    for case, packet in packets.items():
        folder = output/case; (folder/'selector').mkdir(parents=True)
        write_new(folder/'packet.json', packet)
        request = selector_request(packet, model=protocol['model'], reasoning_effort=protocol['reasoning_effort'],
                                   max_tokens=protocol['selector_max_tokens'])
        write_new(folder/'selector/request.json', request)
        paths += [folder/'packet.json', folder/'selector/request.json']
    freeze(output, 'inputs.freeze.json', paths)
    (output/'RESULT.md').write_text('# Opposing-complaint response probe\n\nPrepared; no API calls.\n')
    return protocol


def call_once(folder, client):
    """One provider attempt; preserve every refusal, truncation and transport error."""
    if (folder/'call.json').exists():
        raise ValueError('Attempt already recorded; no automatic rerun')
    request = json.loads((folder/'request.json').read_text())
    write_new(folder/'call.json', {'started_at': now(), 'provider_attempts': 1,
                                  'request_sha256': digest((folder/'request.json').read_bytes())})
    try:
        response = client.chat.completions.create(**request)
        write_new(folder/'provider_response.json', response.model_dump(mode='json'))
        choice = response.choices[0]
        if choice.finish_reason != 'stop' or getattr(choice.message, 'refusal', None):
            result = {'status': 'INCOMPLETE_OR_REFUSED', 'finish_reason': choice.finish_reason}
        else:
            raw = json.loads(choice.message.content)
            write_new(folder/'response.json', raw)
            result = {'status': 'RETURNED', 'response': raw}
    except Exception as error:
        result = {'status': 'FAILED', 'error_type': type(error).__name__}
    return result


def render_reply(raw):
    lines = ['# Response to opposing complaint', '', raw['complaint_restatement'], '',
             '## Conceded points', ''] + ['- '+x for x in raw['conceded_points']]
    for title, key in [('Priority premise', 'priority_premise'), ('Answer', 'answer'),
                       ('Countercomplaint to the alternative', 'countercomplaint_to_alternative'),
                       ('Comparative justification', 'comparative_justification')]:
        lines += ['', '## '+title, '', raw[key]]
    c = raw['conclusion']
    lines += ['', '## Revised conclusion', '', str(c['action_id'])+' / '+c['status'], '', c['rationale'],
              '', '## Remaining objection', '', raw['remaining_objection'],
              '', '## Reversal condition', '', raw['reversal_condition'],
              '', 'Model self-report: '+raw['disposition']+' (not independently accepted).', '']
    return '\n'.join(lines)


def execute(output, client_factory=None):
    output = Path(output); verify_freeze(output, 'inputs.freeze.json')
    protocol = json.loads((output/'protocol.json').read_text())
    for name, expected in protocol['code_sha256'].items():
        if digest((ROOT/name).read_bytes()) != expected:
            raise ValueError('Prepared implementation changed')
    if list(output.glob('*/selector/call.json')) or list(output.glob('*/*/call.json')):
        raise ValueError('Study already attempted; preserve it and use a fresh experiment')
    if client_factory is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(ROOT.parent/'RAGAIMODEL'/'.env')
        client_factory = lambda: OpenAI(max_retries=0, timeout=180)
    client = client_factory()
    assignments, rows = {}, []
    # Freeze every assignment before producing either response arm.
    for case in protocol['cases']:
        folder = output/case; packet = json.loads((folder/'packet.json').read_text())
        print(case+' / assignment', flush=True)
        result = call_once(folder/'selector', client)
        if result['status'] == 'RETURNED':
            try:
                result['bound_assignment'] = bind_assignment(packet, result['response'])
                assignments[case] = result['bound_assignment']
            except ValueError as error:
                result.update(status='MALFORMED_ASSIGNMENT', binding_error=str(error))
        write_new(folder/'selector/result.json', result)
        freeze(folder/'selector', 'outputs.freeze.json', list((folder/'selector').iterdir()))
    for case in protocol['cases']:
        folder = output/case
        assignment = assignments.get(case)
        if not assignment or assignment['raw_assignment']['status'] != 'GROUNDED_TARGET_PROPOSED':
            for arm in ARMS:
                rows.append({'case': case, 'arm': arm, 'status': 'NO_ASSIGNED_TARGET', 'binding_issues': None})
            continue
        packet = json.loads((folder/'packet.json').read_text())
        paths = []
        for arm in ARMS:
            destination = folder/arm; destination.mkdir()
            request = reply_request(packet, assignment, arm, model=protocol['model'],
                reasoning_effort=protocol['reasoning_effort'], max_tokens=protocol['reply_max_tokens'])
            write_new(destination/'request.json', request); paths.append(destination/'request.json')
        freeze(folder, 'reply_inputs.freeze.json', paths + [folder/'selector/result.json'])
        for arm in ARMS:
            destination = folder/arm; print(case+' / '+arm, flush=True)
            result = call_once(destination, client)
            if result['status'] == 'RETURNED':
                try:
                    result['bound_reply'] = bind_reply(packet, result['response'])
                    (destination/'reply.md').write_text(render_reply(result['response']))
                except ValueError as error:
                    result.update(status='MALFORMED_REPLY', binding_error=str(error))
            result.update(operative=False, semantic_support='NOT_INDEPENDENTLY_ASSESSED')
            write_new(destination/'result.json', result)
            freeze(destination, 'outputs.freeze.json', list(destination.iterdir()))
            raw = result.get('response', {})
            rows.append({'case': case, 'arm': arm, 'status': result['status'],
                'binding_issues': result.get('bound_reply', {}).get('binding_issues', []),
                'choice': raw.get('conclusion', {}).get('action_id'),
                'conclusion_status': raw.get('conclusion', {}).get('status'),
                'self_report': raw.get('disposition')})
            summarize(output, rows)
    summarize(output, rows)
    return rows


def summarize(output, rows):
    (output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    lines = ['# Matched responses to opposing complaints', '',
        'Each pair uses the same assigned objection, source, saved reply, schema, model and budget.',
        'Self-reported resolution is not content assessment or ledger acceptance.', '',
        '| Case | Arm | Transport | Choice | Conclusion status | Self-report |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append('| '+' | '.join(str(row.get(k, '')) for k in
            ('case', 'arm', 'status', 'choice', 'conclusion_status', 'self_report'))+' |')
    lines += ['', 'Inspect each case/selector assignment and case/arm/reply.md. '
              'Substantive assessment is separate from these transport and self-report fields.', '']
    (output/'RESULT.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path,
        default=ROOT/'diagnostics/contractualist_comparison_v03_live_20261025')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--framework', default='contractualist')
    parser.add_argument('--case', action='append')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    protocol = prepare(args.source_run, args.output, cases=args.case, framework=args.framework)
    if args.execute:
        execute(args.output)
    else:
        print('Prepared; maximum calls on execution:', protocol['maximum_provider_calls'], '; no API calls.')


if __name__ == '__main__':
    main()
