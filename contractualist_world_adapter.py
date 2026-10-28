"""One question-capable Contractualist argument over a validated frozen world."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from contractualist_pilot import make_request, compile_response, write_argument_report
from run_contractualist_pilot import write_new
from run_opposing_complaint_probe import call_once, freeze
from parliament_run_provenance import (current_provenance, record_checkpoint,
    record_configuration, record_execution, record_resource, record_start)

VERSION = 'contractualist-world-adapter/0.1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def world_source(scenario, world, *, trace_sha256, canonical_action_records=(), advisory=None):
    """Called after native replay validation; retain the same world and its IDs."""
    actions = [{'id': a['action_id'], 'label': a['intervention']} for a in world['actions']]
    if len(actions) not in (2, 3) or len({a['id'] for a in actions}) != len(actions):
        raise ValueError('Contractualist adapter needs two or three distinct admitted actions')
    evidence = [{'ref': 'SCENE', 'kind': 'EXACT_SCENARIO', 'value': scenario}]
    evidence += [{'ref': e['effect_id'], 'kind': 'FROZEN_WORLD_COPY', 'value': deepcopy(e)}
                 for e in world.get('effects', [])]
    return {'id': 'frozen_' + trace_sha256[:12], 'scenario': scenario, 'actions': actions,
        'source_text_sha256': hashlib.sha256(scenario.encode()).hexdigest(),
        'source_status': 'VALIDATED_FROZEN_WORLD', 'world_sha256': fingerprint(world),
        'source_trace_sha256': trace_sha256, 'admitted_world': deepcopy(world),
        'canonical_action_records': deepcopy(list(canonical_action_records)),
        'source_construction_advisory': deepcopy(advisory or {}), 'evidence': evidence}


def generate(source, output, *, model='o3', client_factory=None):
    """Single isolated model call. No native ledger or peer debate is fabricated."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if fingerprint(source['admitted_world']) != source['world_sha256']:
        raise ValueError('World hash mismatch')
    if hashlib.sha256(source['scenario'].encode()).hexdigest() != source['source_text_sha256']:
        raise ValueError('Source hash mismatch')
    request = make_request(source, 'contractualist', model=model, reasoning_effort='low', max_tokens=8192)
    write_new(output/'source.json', source)
    write_new(output/'request.json', request)
    write_new(output/'protocol.json', {'version': VERSION, 'maximum_provider_calls': 1, 'sdk_retries': 0,
        'world_sha256': source['world_sha256'], 'source_trace_sha256': source['source_trace_sha256'],
        'peer_input': False, 'world_mutation': False, 'native_ledger_admission': False,
        'model': model, 'contribution': 'ARGUMENT_AND_CRITICAL_QUESTIONS_NOT_AN_OPERATIVE_VOTE'})
    freeze(output, 'inputs.freeze.json', [output/name for name in ('source.json', 'request.json', 'protocol.json')])
    record_checkpoint('contractualist_input_frozen', artifacts=[output/'source.json', output/'request.json'])
    if client_factory is None:
        from openai import OpenAI
        client_factory = lambda: OpenAI(max_retries=0, timeout=180)
    call = call_once(output, client_factory())
    if call['status'] == 'RETURNED':
        try:
            result = compile_response(source, 'contractualist', call['response'])
            write_argument_report(result, output/'candidate_arguments.md')
        except Exception as error:
            result = {'status': 'COMPILATION_FAILED', 'error_type': type(error).__name__,
                      'raw_response': deepcopy(call['response']), 'operative': False}
    else:
        result = call
    if fingerprint(source['admitted_world']) != source['world_sha256']:
        raise RuntimeError('Contractualist adapter changed the frozen world')
    result.update(adapter_version=VERSION, transport_status=call['status'],
        shared_world_sha256=source['world_sha256'], source_trace_sha256=source['source_trace_sha256'],
        run_provenance=current_provenance(), native_ledger_admission=False)
    write_new(output/'result.json', result)
    names = ('source.json', 'request.json', 'protocol.json', 'call.json', 'provider_response.json',
             'response.json', 'result.json', 'candidate_arguments.md', 'critical_questions.md')
    freeze(output, 'output.freeze.json', [output/n for n in names if (output/n).is_file()])
    record_checkpoint('contractualist_argument_recorded', status=result['status'].lower(),
                      artifacts=[output/'result.json', output/'output.freeze.json'])
    return result


def framework_view(result, source):
    raw = result.get('raw_response') or {}
    actions = {a['id']: a['label'] for a in source['actions']}
    chosen = raw.get('recommendation_action_id')
    questions = deepcopy(raw.get('critical_questions', []))
    review = result.get('critical_question_review') or {}
    warnings = list(raw.get('residual_disagreement') or [])
    if result.get('issues'):
        warnings.append('The argument graph has ' + str(len(result['issues'])) +
                        ' binding findings; the original argument and questions remain inspectable.')
    if result.get('transport_status') != 'RETURNED':
        warnings.append('The Contractualist call did not return a complete argument.')
    return {'framework': 'contractualist',
        'position': 'CLARIFICATION' if chosen is None and questions else 'ATTENUATED' if chosen in actions else 'ABSTAIN',
        'preference': actions.get(chosen), 'reason': raw.get('conclusion') or 'No complete Contractualist explanation was recorded.',
        'reason_authority': 'RECORDED_CONTRACTUALIST_ARGUMENT',
        'ledger_status': 'NOT_APPLICABLE_ARGUMENT_ADAPTER', 'assumptions': [],
        'assumption_records': [{'kind': 'model_proposed_' + s['kind'].lower(), 'text': s['text']}
            for s in raw.get('statements', []) if s['kind'] in ('PRINCIPLE', 'NORMATIVE_PREMISE', 'INTERPRETIVE_PREMISE')],
        'live_objections': warnings, 'dependencies': [q['question'] for q in questions if q['role'] == 'DECISION_DEPENDENCY'],
        'reversal_conditions': [raw['reversal_condition']] if chosen in actions and raw.get('reversal_condition') else [],
        'decision_boundaries': [q['needed_to_resolve'] for q in questions if q['role'] == 'DECISION_DEPENDENCY'],
        'critical_questions': questions, 'critical_question_findings': deepcopy(review.get('findings', [])),
        'person_labels': {p['id']: p['label'] for p in raw.get('persons', [])},
        'recorded_argument': deepcopy(raw), 'argument_technical_status': result['status'],
        'source_basis': {'text': 'The exact scenario and the same validated frozen world supplied to the other four specialists.'},
        'presentation_notices': ['Contractualist argument adapter: questions and conditional reasons, not a native ledger or an operative vote.'],
        'reasoning_provenance': {'adapter_version': result['adapter_version'],
            'world_sha256': result['shared_world_sha256'], 'source_trace_sha256': result['source_trace_sha256'],
            'source_text_sha256': source['source_text_sha256'], 'peer_input': False,
            'native_ledger_admission': False}}


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parliament-root', type=Path, required=True)
    parser.add_argument('--model', default='o3')
    parser.add_argument('--checkpoint-tag', action='append')
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a fresh output directory')
    record_configuration(args); record_start(args.output_dir); record_resource(args.trace)
    before = args.trace.read_bytes()
    payload = json.loads(before)
    sys.path.insert(0, str(args.parliament_root.resolve()))
    from global_workspace.frozen_world_replay import load_frozen_world_trace
    replay = load_frozen_world_trace(args.trace, expected_scenario=payload['scenario'])
    source = world_source(replay.scenario, replay.action_source_grounding['world_model'],
        trace_sha256=hashlib.sha256(before).hexdigest(), canonical_action_records=replay.canonical_action_records,
        advisory=replay.source_construction_advisory)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent/'RAGAIMODEL/.env')
    result = generate(source, args.output_dir, model=args.model)
    if args.trace.read_bytes() != before:
        raise RuntimeError('Frozen trace changed')
    print('Contractualist: ' + result['status'] + '; questions: ' +
          str(len((result.get('raw_response') or {}).get('critical_questions', []))), flush=True)
    return 0 if result['transport_status'] == 'RETURNED' else 1


def main(argv=None):
    return record_execution(_main, argv, entrypoint='contractualist_world_adapter.py', output_flag='--output-dir',
                            defaults={'model': 'o3', 'runtime_profile': 'compact_frameworks'})


if __name__ == '__main__':
    raise SystemExit(main())
