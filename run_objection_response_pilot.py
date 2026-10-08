"""Matched objection revisions; attributed arguments never modify the admitted world."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from logic_puzzles_frameworks import fingerprint

FRAMEWORKS = ('utilitarian', 'deontological', 'care')
CASES = ('competing_promise', 'probability_swap')
ARMS = ('ordinary_revision', 'objection_focused')
OBJECTIONS = {
    'competing_promise': {
        'utilitarian': 'The expected-survivor advantage does not establish total welfare. How does your ranking address the explicit promise to Cara without inventing the magnitude of trust or other welfare effects?',
        'deontological': 'Why should your preferred duty take priority over the competing rescue or fidelity claim? Explain why the classification and priority apply here rather than merely naming a hierarchy.',
        'care': 'Why should treatment effectiveness or promise-based trust take priority over the competing patient claim? Address both needs and defend your provisional ranking without assuming an unprovided relationship.'},
    'probability_swap': {
        'utilitarian': 'Higher expected survivors is a restricted endpoint, not total welfare. Explain why it supports your current preference, what it cannot settle, and how the swapped probabilities affect your reason.',
        'deontological': 'Does changing survival probabilities change which duty applies, or only the effectiveness of helping? Explain why your preference changes or stays while both patients have a rescue claim.',
        'care': 'Both patients need care. Explain why the swapped treatment effectiveness should or should not outweigh the competing need without inventing relational priority.'}}
TREATMENT = '''Answer the assigned objection directly. Identify the precise inference it
challenges. Separate admitted facts, framework commitments and contestable normative
bridges. Defend why your bridge applies in this case; repeating a priority slogan is
not a defence. State the force of the competing claim, the remaining uncertainty and
a concrete reversal condition. A substantive answer need not defeat the objection.
Use answered, partly_answered or unanswered honestly. Keep a preference provisional
when the competing claim remains unresolved. Do not invent facts to obtain admission.'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def decode(response):
    if isinstance(response, dict) and response.get('choices'):
        return json.loads(response['choices'][0]['text'])
    return response


def argument_schema():
    names = ('target', 'reply', 'framework_commitment', 'reasoning_bridge',
             'remaining_uncertainty', 'reversal_condition')
    props = {n: {'type': 'string', 'maxLength': 1000} for n in names}
    props['evidence_ids'] = {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 12}
    props['disposition'] = {'type': 'string', 'enum': ['answered', 'partly_answered', 'unanswered']}
    return {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}


class RevisionBackend:
    def __init__(self, backend, packet, arm, destination, *, repair_contract=None, frozen_world=None, commitments=None):
        self.backend, self.packet, self.arm = backend, deepcopy(packet), arm
        self.model = backend.model
        self.destination = Path(destination)
        self.calls = 0
        self.repair_contract = deepcopy(repair_contract)
        self.frozen_world, self.commitments = deepcopy(frozen_world), deepcopy(commitments)

    def complete_json(self, prompt, **kwargs):
        if self.calls:
            raise RuntimeError('One revision call per cell; no hidden repair')
        self.calls += 1
        instruction = TREATMENT if self.arm == 'objection_focused' else 'Revise your initial argument in light of the assigned objection using your framework.'
        effective = prompt+'\nFROZEN REVISION INPUT (prior argument is attributed, not source evidence):\n'+json.dumps(self.packet)+'\n'+instruction+'''\nReturn argument_response and native_submission. The former is an attributed
explanation, not an accepted conflict resolution. The latter must use the unchanged
native schema. Use the same reasoning in both; do not claim resolution merely to pass
validation. Evidence IDs refer only to the shared input; unsupported assumptions must
remain explicit in remaining_uncertainty.'''
        schema = {'type': 'object', 'properties': {'argument_response': argument_schema(),
                  'native_submission': deepcopy(kwargs['schema'])},
                  'required': ['argument_response', 'native_submission'], 'additionalProperties': False}
        if self.repair_contract is not None:
            from inference_repair import repair_schema
            schema['properties']['inference_repair'] = repair_schema()
            schema['required'].append('inference_repair')
            effective += '\nBOUNDED INFERENCE REPAIR CONTRACT: '+json.dumps(self.repair_contract)+'''
Choose DEFEND, CONDITIONALIZE, WEAKEN or RETRACT for exactly the assigned inference.
Keep the restricted findings verbatim. Total welfare is a separate conclusion scope.
Conditional or unresolved conclusions cannot be submitted as SELECTED; cc/esa must
be false, ia NONE, and no action REQUIRED or REJECTED. A provisional preference must
cite the explicit assumptions. Unsupported or illustrative reversal text stays only
in the repair record; native ft must be NONE. Use an operative qualitative reversal
only if you declare its grounded/framework derivation. RETRACT retains the repair
record but dispatches no native selection through this inference. No field outside
the contract's edit list may change. No PERSON-to-welfare conversion is licensed.'''
        write(self.destination/'revision_call.json', {'prompt': effective, 'schema': schema,
              'model': self.model, 'max_tokens': kwargs['max_tokens'], 'status': 'STARTED'})
        try:
            raw = self.backend.complete_json(effective, **dict(kwargs, schema=schema))
            result = decode(raw)
            write(self.destination/'revision_call.json', {'prompt': effective, 'schema': schema,
                  'model': self.model, 'max_tokens': kwargs['max_tokens'], 'status': 'RETURNED', 'response': raw})
            expected = {'argument_response', 'native_submission'} | ({'inference_repair'} if self.repair_contract is not None else set())
            # A missing new record is a contract failure, not a guessed prose repair.
            if not isinstance(result, dict) or set(result) not in (expected, {'argument_response', 'native_submission'}):
                raise ValueError('Malformed revision envelope')
            write(self.destination/'argument_response.json', {'origin': 'MODEL_AUTHORED_ARGUMENT',
                  'authority': 'ATTRIBUTED_NOT_WORLD_STATE', 'assessment': 'REQUIRES_CONTENT_REVIEW',
                  'self_report_is_not_conflict_retirement': True, 'record': result['argument_response']})
            if self.repair_contract is not None:
                from inference_repair import check_submission
                repair = result.get('inference_repair')
                write(self.destination/'inference_repair.json', {'origin': 'MODEL_AUTHORED_REPAIR',
                      'semantic_support': 'NOT_INDEPENDENTLY_VERIFIED', 'record': repair})
                gate = check_submission(self.repair_contract, repair, result['native_submission'], self.frozen_world, self.commitments)
                write(self.destination/'submission_consistency.json', gate)
                if not gate['dispatch_native']:
                    # Reuse native bounded-call handling; this is explicitly a local gate,
                    # not a provider failure. No native fields are rewritten or parsed.
                    try:
                        from global_workspace.structured_io import ModelCallUnavailable
                    except ImportError:
                        raise ValueError('CONTRACT_BLOCKED: '+', '.join(gate['errors']))
                    raise ModelCallUnavailable('CONTRACT_BLOCKED: '+', '.join(gate['errors']),
                                               category='local_inference_repair_gate', terminal=True)
            return {'choices': [{'text': json.dumps(result['native_submission'])}]}
        except Exception as error:
            write(self.destination/'revision_error.json', {'exception_type': type(error).__name__})
            raise


def prepare(source, output):
    source, output = Path(source), Path(output)
    manifest = {'version': 'objection-response-pilot/1', 'cases': {}, 'frameworks': list(FRAMEWORKS),
                'arms': list(ARMS), 'calls_per_cell': 1, 'total_revision_calls': 12,
                'limits': ['Single sample per cell; exploratory.', 'Fresh native assessment, not an accepted ledger update.',
                           'Disposition is author self-report, not semantic evaluation or conflict retirement.',
                           'No peer arguments, world mutation, hidden repair or collective verdict.']}
    for case in CASES:
        opening = source/case/'standard'
        shared = json.loads((opening/'shared_input.json').read_text())
        quotes = json.loads((opening/'framework_commitments.json').read_text())
        frozen = json.loads((source/case/'frozen_world_trace.json').read_text())
        if shared['scenario'] != frozen['scenario'] or shared['action_source_grounding'] != frozen['action_source_grounding']:
            raise ValueError('Opening and source world differ')
        write(output/case/'frozen_world_trace.json', frozen)
        packets = {}
        for framework in FRAMEWORKS:
            calls = json.loads((opening/framework/'model_calls.json').read_text())
            if len(calls) != 1 or calls[0]['status'] != 'RETURNED':
                raise ValueError('Requires one completed opening response')
            packet = {'framework': framework, 'initial_argument': decode(calls[0]['response']),
                      'objection': OBJECTIONS[case][framework],
                      'objection_origin': 'EXPERIMENT_ASSIGNED_CHALLENGE_NOT_WORLD_FACT',
                      'shared_input_sha256': fingerprint(shared)}
            write(output/case/(framework+'_revision_input.json'), packet)
            packets[framework] = {'packet_sha256': fingerprint(packet), 'opening_model': calls[0]['model']}
        manifest['cases'][case] = {'source': str(source/case), 'shared_input_sha256': fingerprint(shared),
                                  'world_sha256': fingerprint(frozen['action_source_grounding']['world_model']),
                                  'packets': packets}
    write(output/'manifest.json', manifest)
    return manifest


def summarize(output, manifest):
    output = Path(output)
    rows, controls = [], []
    for case in CASES:
        folders = [output/case/arm for arm in ARMS]
        if all((f/'shared_input.json').exists() for f in folders):
            controls.append({'case': case, 'shared_input_equal': (folders[0]/'shared_input.json').read_bytes() == (folders[1]/'shared_input.json').read_bytes(),
                'commitments_equal': (folders[0]/'framework_commitments.json').read_bytes() == (folders[1]/'framework_commitments.json').read_bytes()})
        for arm in ARMS:
            folder = output/case/arm
            if not (folder/'framework_generation.json').exists(): continue
            assessed = folder/'native_replay/framework_generation.json'
            trace = json.loads((assessed if assessed.exists() else folder/'framework_generation.json').read_text())
            if fingerprint(trace['action_source_grounding']['world_model']) != manifest['cases'][case]['world_sha256']:
                raise RuntimeError('World changed')
            by_name = {c['specialist']: c for c in trace['cycles'][0]['candidates']}
            for name in FRAMEWORKS:
                path = folder/name/'revision_call.json'
                if not path.exists(): continue
                call = json.loads(path.read_text())
                response = folder/name/'argument_response.json'
                candidate = by_name.get(name, {})
                rows.append({'case': case, 'arm': arm, 'framework': name, 'model': call['model'],
                    'max_tokens': call['max_tokens'], 'call_status': call['status'],
                    'submission': candidate.get('delegate_status', 'NO_CANDIDATE'),
                    'native_ledger': (candidate.get('committed_native_ledger') or {}).get('transaction_status', 'NONE'),
                    'preferred_action': candidate.get('recommended_action'),
                    'warnings': candidate.get('framework_validation_errors', []),
                    'argument': json.loads(response.read_text())['record'] if response.exists() else None,
                    'content_score': None, 'world_unchanged': True})
    for case in CASES:
        for name in FRAMEWORKS:
            pair = [r for r in rows if r['case'] == case and r['framework'] == name]
            if len(pair) == 2:
                saved_calls = [json.loads((output/case/arm/name/'revision_call.json').read_text()) for arm in ARMS]
                marker = '\nFROZEN REVISION INPUT (prior argument is attributed, not source evidence):\n'
                actual_packets = [json.loads(c['prompt'].split(marker, 1)[1].split('\n', 1)[0]) for c in saved_calls]
                controls.append({'case': case, 'framework': name, 'model_and_tokens_equal':
                    (pair[0]['model'], pair[0]['max_tokens']) == (pair[1]['model'], pair[1]['max_tokens']),
                    'same_frozen_opening_and_objection': actual_packets[0] == actual_packets[1],
                    'packet_matches_manifest': all(fingerprint(p) == manifest['cases'][case]['packets'][name]['packet_sha256'] for p in actual_packets),
                    'native_prompt_equal': saved_calls[0]['prompt'].split(marker, 1)[0] == saved_calls[1]['prompt'].split(marker, 1)[0],
                    'response_schema_equal': saved_calls[0]['schema'] == saved_calls[1]['schema']})
    write(output/'results.json', rows)
    write(output/'controls.json', controls)
    lines = ['# Matched objection-response pilot', '', 'Admission and argument quality are separate. Disposition below is model self-report.', '',
             '| Case | Arm | Framework | Submission | Ledger | Self-report |', '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append('| '+' | '.join(str(r[k]) for k in ('case', 'arm', 'framework', 'submission', 'native_ledger'))+' | '+str((r['argument'] or {}).get('disposition', 'unavailable'))+' |')
    lines += ['', 'Raw arguments and failures are retained. Content scores are a separate review, not inferred from admission.', '', *manifest['limits']]
    if (output/'CONTENT_REVIEW.md').exists():
        lines += ['', 'Content assessment: [review](CONTENT_REVIEW.md).']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')


class SavedNativeBackend:
    """Re-assess exact saved native fields; zero provider calls or content edits."""
    def __init__(self, call):
        self.model = call['model']
        self.response = deepcopy(decode(call['response'])['native_submission'])
        self.last_call_origin = 'SAVED_NATIVE_SUBMISSION_NO_API'
        self.calls = 0

    def complete_json(self, prompt, **kwargs):
        if self.calls: raise RuntimeError('One saved replay only')
        self.calls += 1
        return {'choices': [{'text': json.dumps(self.response)}]}


def replay_native(output, source, tokens):
    from logic_puzzles_frameworks import generate
    for case in CASES:
        for arm in ARMS:
            folder = output/case/arm
            quotes = json.loads((source/case/'standard/framework_commitments.json').read_text())
            calls = {n: json.loads((folder/n/'revision_call.json').read_text()) for n in FRAMEWORKS}
            if any(c['status'] != 'RETURNED' for c in calls.values()):
                raise ValueError('Cannot replay missing provider responses')
            generate(output/case/'frozen_world_trace.json', folder/'native_replay', FRAMEWORKS,
                     lambda n: SavedNativeBackend(calls[n]), core_quotes=quotes, tokens=tokens, call_budget=1)
            write(folder/'native_replay/replay_provenance.json', {'provider_calls': 0,
                  'native_fields_unchanged': True, 'saved_response_sha256': {n: fingerprint(decode(c['response'])['native_submission']) for n, c in calls.items()},
                  'reason': 'Native transport envelope correction; original assessment retained.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opening-dir', type=Path, default=Path('diagnostics/argument_quality_pilot_fixes_20261006'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parliament-root', type=Path, default=Path('/tmp/parliament-smoke-614af0c'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--replay-native', action='store_true')
    parser.add_argument('--model', default='o3')
    parser.add_argument('--tokens', type=int, default=4096)
    args = parser.parse_args()
    sys.path.insert(0, str(args.parliament_root.resolve()))
    manifest = prepare(args.opening_dir, args.output_dir)
    if args.execute and args.replay_native: raise ValueError('Choose live execution or saved replay')
    if args.execute:
        from dotenv import load_dotenv
        from global_workspace.openai_backend import OpenAIWorkspaceLLM
        from logic_puzzles_frameworks import generate
        load_dotenv('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env')
        for case in CASES:
            for arm in ARMS:
                destination = args.output_dir/case/arm
                if (destination/'framework_generation.json').exists():
                    raise ValueError('Use a fresh output directory; no repeated paid calls')
                packets = {n: json.loads((args.output_dir/case/(n+'_revision_input.json')).read_text()) for n in FRAMEWORKS}
                if any(p['opening_model'] != args.model for p in manifest['cases'][case]['packets'].values()):
                    raise ValueError('Model must match frozen openings')
                quotes = json.loads((args.opening_dir/case/'standard/framework_commitments.json').read_text())
                print(case+' / '+arm, flush=True)
                generate(args.output_dir/case/'frozen_world_trace.json', destination, FRAMEWORKS,
                    lambda n: RevisionBackend(OpenAIWorkspaceLLM(args.model, timeout=120), packets[n], arm, destination/n),
                    tokens=args.tokens, core_quotes=quotes, call_budget=1)
                summarize(args.output_dir, manifest)
    if args.replay_native:
        replay_native(args.output_dir, args.opening_dir, args.tokens)
    summarize(args.output_dir, manifest)
    print(args.output_dir/'RESULT.md')


if __name__ == '__main__': main()
