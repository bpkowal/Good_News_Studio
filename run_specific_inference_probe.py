"""One saved utilitarian inference, one corrective follow-up, unchanged native assessment."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from logic_puzzles_frameworks import fingerprint
from run_objection_response_pilot import RevisionBackend, decode, write
from inference_repair import make_contract

CHALLENGE = '''Your previous bridge says typical disutility from one broken promise is
smaller than the benefit of 0.90 expected survivors, and invokes unspecified empirical
studies. The admitted scenario establishes neither that bound nor a common welfare
unit. Do not replace it with another unsupported empirical generalization. Remove
the bound, or explicitly label any retained premise as a defeasible assumption rather
than source evidence. State the strongest conclusion justified without it and whether
total-welfare ranking remains unresolved. You may retain a conditional or provisional
preference; you need not reverse your recommendation. Keep expected survivors distinct
from utility and life-years; no duration is given. Reflect the same limits in your
native submission, decision rule, and reversal condition. Keep each explanation field
concise and complete (under 1000 characters).'''


def prepare(source, output):
    source, output = Path(source), Path(output)
    if (output/'manifest.json').exists(): raise ValueError('Use a fresh output directory; preserve existing diagnostics')
    folder = source/'competing_promise/objection_focused'
    paths = {'provider_reply': folder/'utilitarian/revision_call.json',
             'shared_input': folder/'shared_input.json',
             'commitments': folder/'framework_commitments.json',
             'frozen_world': source/'competing_promise/frozen_world_trace.json'}
    saved = {k: p.read_bytes() for k, p in paths.items()}
    call = json.loads(saved['provider_reply'])
    if call['status'] != 'RETURNED': raise ValueError('Saved reply unavailable')
    response = decode(call['response'])
    shared, frozen = json.loads(saved['shared_input']), json.loads(saved['frozen_world'])
    if shared['action_source_grounding'] != frozen['action_source_grounding'] or shared['scenario'] != frozen['scenario']:
        raise ValueError('Reply and frozen world differ')
    packet = {'framework': 'utilitarian', 'initial_argument': deepcopy(response['native_submission']),
              'previous_argument_response': deepcopy(response['argument_response']),
              'objection': CHALLENGE, 'objection_origin': 'EXPERIMENT_ASSIGNED_CHALLENGE_NOT_WORLD_FACT',
              'shared_input_sha256': fingerprint(shared)}
    manifest = {'version': 'specific-inference-probe/2', 'model': call['model'],
                'max_tokens': call['max_tokens'], 'maximum_provider_adapter_calls': 1,
                'source_paths': {k: str(p) for k, p in paths.items()},
                'source_sha256': {k: fingerprint(json.loads(b)) for k, b in saved.items()},
                'packet_sha256': fingerprint(packet), 'world_sha256': fingerprint(frozen['action_source_grounding']['world_model']),
                'limits': ['One instructed follow-up, no control arm or treatment-effect estimate.',
                           'Fresh native assessment, not a continuity-checked accepted update.',
                           'Structured inference repair must pass the local consistency gate before native dispatch.',
                           'Self-report is not independent assessment or conflict retirement.']}
    write(output/'manifest.json', manifest)
    write(output/'revision_input.json', packet)
    write(output/'frozen_world_trace.json', frozen)
    write(output/'original_reply.json', response)
    contract = make_contract(response['native_submission'], frozen, json.loads(saved['commitments']))
    write(output/'repair_contract.json', contract)
    return manifest, packet, json.loads(saved['commitments'])


def summarize(output, manifest):
    output = Path(output)
    trace = json.loads((output/'revision/framework_generation.json').read_text())
    if fingerprint(trace['action_source_grounding']['world_model']) != manifest['world_sha256']:
        raise RuntimeError('World mutated')
    for key, path in manifest['source_paths'].items():
        if fingerprint(json.loads(Path(path).read_text())) != manifest['source_sha256'][key]:
            raise RuntimeError('Saved source changed: '+key)
    candidates = trace['cycles'][0]['candidates']
    c = candidates[0] if candidates else {}
    path = output/'revision/utilitarian/argument_response.json'
    result = {'submission': c.get('delegate_status', 'NO_CANDIDATE'),
              'native_ledger': (c.get('committed_native_ledger') or {}).get('transaction_status', 'NONE'),
              'preferred_action': c.get('recommended_action'), 'selection_status': c.get('selection_status'),
              'warnings': c.get('framework_validation_errors', []),
              'adapter_calls': trace['framework_runs'][0]['adapter_calls'],
              'world_unchanged': True, 'saved_inputs_unchanged': True,
              'argument_response': json.loads(path.read_text()) if path.exists() else None,
              'content_assessment': 'REQUIRES_REVIEW'}
    gate_path = output/'revision/utilitarian/submission_consistency.json'
    if gate_path.exists():
        result['submission_consistency'] = json.loads(gate_path.read_text())
        if not result['submission_consistency']['dispatch_native']:
            result['native_assessment_status'] = result['submission']
            result['submission'] = 'CONTRACT_BLOCKED'
            result['preferred_action'] = ''
    write(output/'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('diagnostics/objection_response_pilot_live_20261007'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parliament-root', type=Path, default=Path('/tmp/parliament-smoke-614af0c'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.parliament_root.resolve()))
    if args.execute and (args.output_dir/'revision').exists(): raise ValueError('Use a fresh output directory')
    manifest, packet, quotes = prepare(args.source_dir, args.output_dir)
    if args.execute:
        from dotenv import load_dotenv
        from global_workspace.openai_backend import OpenAIWorkspaceLLM
        from logic_puzzles_frameworks import generate
        load_dotenv('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env')
        generate(args.output_dir/'frozen_world_trace.json', args.output_dir/'revision', ['utilitarian'],
                 lambda _: RevisionBackend(OpenAIWorkspaceLLM(manifest['model'], timeout=120), packet,
                     'objection_focused', args.output_dir/'revision/utilitarian',
                     repair_contract=json.loads((args.output_dir/'repair_contract.json').read_text()),
                     frozen_world=json.loads((args.output_dir/'frozen_world_trace.json').read_text()), commitments=quotes),
                 tokens=manifest['max_tokens'], core_quotes=quotes, call_budget=1)
        summarize(args.output_dir, manifest)
    print(args.output_dir/'manifest.json')


if __name__ == '__main__': main()
