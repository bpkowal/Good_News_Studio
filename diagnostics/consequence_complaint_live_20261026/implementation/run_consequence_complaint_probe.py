"""Compare saved-complaint replies with and without a consequence projection."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from complaint_consequences import (ARMS, VERSION, REFINEMENT, REVIEW_FORMAT,
    build_consequence_record, consequence_request, bind_consequence_reply, render_consequences)
from run_contractualist_pilot import digest, now, write_new
from run_opposing_complaint_probe import call_once, freeze, verify_freeze, render_reply

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ('cooling_contributor_consents', 'blanket_prior_assurance')


def prepare(source_probe, output, *, cases=None):
    """Reuse frozen assignments, not previous replies or an assistant's assessment."""
    source_probe, output = Path(source_probe), Path(output)
    verify_freeze(source_probe, 'inputs.freeze.json')
    previous = json.loads((source_probe/'protocol.json').read_text())
    chosen = list(DEFAULT_CASES if cases is None else cases)
    if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(previous['cases']):
        raise ValueError('Unknown, empty or duplicate cases')
    packets, assignments, records, origins = {}, {}, {}, {}
    for case in chosen:
        folder = source_probe/case
        verify_freeze(folder/'selector', 'outputs.freeze.json')
        packet = json.loads((folder/'packet.json').read_text())
        result = json.loads((folder/'selector/result.json').read_text())
        assignment = result.get('bound_assignment')
        if not assignment or assignment['raw_assignment']['status'] != 'GROUNDED_TARGET_PROPOSED':
            raise ValueError('No saved assignment for ' + case)
        packets[case], assignments[case] = packet, assignment
        records[case] = build_consequence_record(packet)
        origins[case] = {'packet_sha256': digest((folder/'packet.json').read_bytes()),
                         'assignment_result_sha256': digest((folder/'selector/result.json').read_bytes()),
                         'assignment_output_receipt_sha256': digest((folder/'selector/outputs.freeze.json').read_bytes())}
    output.mkdir(parents=True, exist_ok=False)
    implementation = output/'implementation'; implementation.mkdir()
    names = ('complaint_consequences.py', 'run_consequence_complaint_probe.py',
             'complaint_comparison.py', 'opposing_complaint.py', 'run_opposing_complaint_probe.py',
             'run_contractualist_pilot.py', 'contractualist_pilot.py', 'argument_construction.py',
             'argument_support.py', 'argument_frameworks.py', 'inference_boundaries.py', 'deontology_repair_cards.py')
    versions = {}
    for package in ('openai', 'python-dotenv'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    protocol = {'version': VERSION, 'prepared_at': now(), 'source_probe': str(source_probe.resolve()),
        'cases': chosen, 'origins': origins, 'arms': list(ARMS), 'model': previous['model'],
        'reasoning_effort': previous['reasoning_effort'], 'max_completion_tokens': previous['reply_max_tokens'],
        'maximum_provider_calls': 2*len(chosen), 'selector_calls': 0, 'sdk_retries': 0,
        'runtime': {'python': sys.version, 'packages': versions},
        'code_sha256': {name: digest((ROOT/name).read_bytes()) for name in names},
        'instructions': {'refinement': REFINEMENT, 'review_format': REVIEW_FORMAT},
        'world_mutation': False, 'native_dispatch': False, 'peer_input': False,
        'limits': ['Same source, own opening, assigned complaint, schema, model, effort and reply ceiling.',
            'Treatment adds a fallible consequence projection and fact-first instruction together; effects are not isolated.',
            'Both arms share new factual-review/warrant fields; baseline differs from the previous experiment.',
            'No assistant content review, previous response-arm outputs or desired recommendation enters either request.',
            'Correcting a fact or changing a conclusion is not proof of better normative reasoning.']}
    write_new(output/'protocol.json', protocol); paths = [output/'protocol.json']
    for name in names:
        (implementation/name).write_bytes((ROOT/name).read_bytes()); paths.append(implementation/name)
    for case in chosen:
        folder = output/case; folder.mkdir()
        write_new(folder/'packet.json', packets[case])
        write_new(folder/'assignment.json', assignments[case])
        write_new(folder/'consequence_record.json', records[case])
        (folder/'consequences.md').write_text(render_consequences(records[case]))
        paths += [folder/name for name in ('packet.json', 'assignment.json', 'consequence_record.json', 'consequences.md')]
        for arm in ARMS:
            cell = folder/arm; cell.mkdir()
            request = consequence_request(packets[case], assignments[case], arm,
                model=protocol['model'], reasoning_effort=protocol['reasoning_effort'],
                max_tokens=protocol['max_completion_tokens'])
            write_new(cell/'request.json', request); paths.append(cell/'request.json')
    freeze(output, 'inputs.freeze.json', paths)
    summarize(output, [])
    return protocol


def render_response(raw):
    lines = ['# Factual review (model-proposed, not certified)', '']
    for item in raw['factual_review']:
        target = item['target']
        lines += [f"- {item['status']}: {target['origin']} {target['path']}",
                  '  - Target: ' + target['quote'], '  - Response: ' + item['corrected_statement']]
    warrant = raw['normative_warrant']
    lines += ['', '## Normative warrant', '', warrant['principle'], '', warrant['justification'],
              '', 'Limits: ' + warrant['limits'], '', render_reply(raw)]
    return '\n'.join(lines)


def execute(output, client_factory=None):
    output = Path(output); verify_freeze(output, 'inputs.freeze.json')
    protocol = json.loads((output/'protocol.json').read_text())
    for name, expected in protocol['code_sha256'].items():
        if digest((ROOT/name).read_bytes()) != expected:
            raise ValueError('Prepared implementation changed: ' + name)
    if list(output.glob('*/*/call.json')):
        raise ValueError('Study already attempted; no automatic reruns')
    if client_factory is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(ROOT.parent/'RAGAIMODEL'/'.env')
        client_factory = lambda: OpenAI(max_retries=0, timeout=180)
    client = client_factory(); rows = []
    for case in protocol['cases']:
        folder = output/case
        packet = json.loads((folder/'packet.json').read_text())
        assignment = json.loads((folder/'assignment.json').read_text())
        record = json.loads((folder/'consequence_record.json').read_text())
        for arm in ARMS:
            cell = folder/arm; print(case + ' / ' + arm, flush=True)
            result = call_once(cell, client)
            if result['status'] == 'RETURNED':
                try:
                    result['bound_reply'] = bind_consequence_reply(packet, assignment, result['response'],
                        record if arm == 'consequences_first' else None)
                    (cell/'reply.md').write_text(render_response(result['response']))
                except ValueError as error:
                    result.update(status='MALFORMED_REPLY', binding_error=str(error))
            result.update(operative=False, semantic_support='NOT_INDEPENDENTLY_ASSESSED')
            write_new(cell/'result.json', result)
            freeze(cell, 'outputs.freeze.json', list(cell.iterdir()))
            raw = result.get('response', {})
            rows.append({'case': case, 'arm': arm, 'status': result['status'],
                'binding_issues': result.get('bound_reply', {}).get('binding_issues', []),
                'choice': raw.get('conclusion', {}).get('action_id'),
                'conclusion_status': raw.get('conclusion', {}).get('status'),
                'self_report': raw.get('disposition'),
                'factual_review_count': len(raw.get('factual_review', []))})
            summarize(output, rows)
    return rows


def summarize(output, rows):
    write = lambda name, data: (output/name).write_text(json.dumps(data, indent=2)+'\n')
    write('results.json', rows)
    lines = ['# Consequence-first opposing-complaint pilot', '',
             'Model-proposed corrections, choices and ANSWERED labels are not content-quality scores.', '',
             '| Case | Arm | Transport | Choice | Conclusion | Self-report |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(k)) for k in
            ('case', 'arm', 'status', 'choice', 'conclusion_status', 'self_report')) + ' |')
    if not rows:
        lines += ['', 'Prepared offline. No API calls.']
    lines += ['', 'Per-case consequences.md shows the existing matrix and source passages. '
              'Each arm has reply.md plus unedited requests and provider responses.', '']
    (output/'RESULT.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-probe', type=Path,
        default=ROOT/'diagnostics/opposing_complaint_live_20261025')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    protocol = prepare(args.source_probe, args.output, cases=args.case)
    if args.execute:
        execute(args.output)
    else:
        print('Prepared offline; maximum calls on execution:', protocol['maximum_provider_calls'])


if __name__ == '__main__':
    main()
