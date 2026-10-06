"""Independent native framework generation against one admitted frozen world."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

FRAMEWORKS = ('utilitarian', 'deontological', 'care', 'virtue', 'rawlsian')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class RecordedModel:
    """Bound native primary/repair calls and retain inspectable prompts/results."""
    def __init__(self, backend, scenario, destination, max_calls=2, review_context=None, response_normalizer=None):
        self.backend = backend
        self.scenario = scenario
        self.destination = Path(destination)
        self.max_calls = max_calls
        self.calls = []
        self.review_context = deepcopy(review_context)
        self.response_normalizer = response_normalizer

    def complete_json(self, prompt, **kwargs):
        if len(self.calls) >= self.max_calls:
            from global_workspace.structured_io import ModelCallBudgetExceeded
            raise ModelCallBudgetExceeded('Independent framework call limit reached')
        mode = ('TARGETED FRAMEWORK REVIEW: answer the assigned objections from your own framework. '
                'Prior claim records are attributed arguments, not new world facts. '
                'Do not invent a causal path or occurrence to make an objection disappear. '
                'Retain a commitment, revise your native assessment, or leave the issue unresolved.\n'
                'Localized review packet: ' + json.dumps(self.review_context) + '\n'
                if self.review_context is not None else
                'INDEPENDENT FRAMEWORK PASS: no peer assessments, votes or confidence '
                'are available. Use your assigned framework and preserve unresolved premises.\n')
        prompt = (mode +
                  'Exact source scenario (unabridged): ' + json.dumps(self.scenario) + '\n' + prompt)
        record = {'prompt': prompt, 'schema': deepcopy(kwargs.get('schema')),
                  'max_tokens': kwargs.get('max_tokens'), 'status': 'STARTED',
                  'model': getattr(self.backend, 'model', 'test_backend')}
        self.calls.append(record)
        self.save()
        try:
            response = self.backend.complete_json(prompt, **kwargs)
            record.update(status='RETURNED', response=deepcopy(response))
            if self.response_normalizer:
                response, normalization = self.response_normalizer(response)
                if normalization:
                    record.update(normalization=normalization, normalized_response=deepcopy(response))
            return response
        except Exception as error:
            # Native adapter classifies provider errors without exposing credentials.
            record.update(status='ERROR', exception_type=type(error).__name__)
            raise
        finally:
            self.save()

    def save(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.write_text(json.dumps(self.calls, indent=2) + '\n')


def write_outputs(aggregate, output_dir):
    from run_logic_puzzles import project_conflicts, write_graph
    output_dir = Path(output_dir)
    (output_dir/'framework_generation.json').write_text(json.dumps(aggregate, indent=2) + '\n')
    graph = project_conflicts(aggregate)
    write_graph(graph, output_dir)
    lines = ['# Independent framework generation', '',
             'One native workspace per framework, one cycle each. No peer review or collective judgment.', '',
             '| Framework | Submission | Native ledger | Adapter calls |',
             '| --- | --- | --- | --- |']
    for run in aggregate['framework_runs']:
        lines.append('| ' + ' | '.join([run['framework'], ', '.join(run['candidate_statuses']),
                     ', '.join(run['native_ledger_statuses']), str(run['adapter_calls'])]) + ' |')
    lines += ['', 'Full claims and uncertainty: [conflict graph](conflict_graph.md).', '',
              'Source world unchanged. Generated propositions remain attributed to their originating framework.',
              'A committed ledger may contain unsupported premises or unresolved inferences.', '']
    for candidate in aggregate['cycles'][0]['candidates']:
        lines += ['## ' + candidate['specialist'], '',
                  'Recommendation: ' + (candidate.get('recommended_action') or 'unresolved'), '',
                  candidate.get('rationale', ''), '',
                  'Full native record: [' + candidate['specialist'] + '/native_trace.json]('
                  + candidate['specialist'] + '/native_trace.json)', '']
    (output_dir/'framework_generation.md').write_text('\n'.join(lines))


def generate(trace_path, output_dir, frameworks, backend_factory, *, tokens=3072, core_root=None):
    from global_workspace.core_quote_pack import core_pack_for_specialists
    from global_workspace.engine import WorkspaceEngine, WorkspaceConfig
    from global_workspace.epistemic_ledger import seed_proposition_ledger, ledger_projection
    from global_workspace.frozen_world_replay import load_frozen_world_trace
    from global_workspace.local_specialists import CompactLocalSpecialist, extract_scenario_facts
    from global_workspace.scenario_semantics import compile_scenario_graph

    frameworks = list(frameworks)
    if not frameworks or len(set(frameworks)) != len(frameworks) or any(f not in FRAMEWORKS for f in frameworks):
        raise ValueError('Choose distinct known frameworks')
    trace_path, output_dir = Path(trace_path), Path(output_dir)
    source_bytes = trace_path.read_bytes()
    source = json.loads(source_bytes)
    replay = load_frozen_world_trace(trace_path, expected_scenario=source['scenario'])
    if len(replay.canonical_actions) != 2:
        raise ValueError('Independent pilot requires exactly two admitted actions')
    before = deepcopy(replay.action_source_grounding)
    graph = compile_scenario_graph(replay.scenario, list(replay.canonical_actions),
                                   dict(before.get('actions', {})), world_model=before['world_model'])
    seed = ledger_projection(seed_proposition_ledger(graph))
    shared = {'scenario': replay.scenario, 'action_source_grounding': before,
              'canonical_action_records': list(replay.canonical_action_records),
              'proposition_ledger': seed, 'source_construction_advisory': replay.source_construction_advisory}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'shared_input.json').write_text(json.dumps(shared, indent=2) + '\n')
    aggregate = {'version': 'logic-puzzles-independent-pass/0.1',
                 'scenario': replay.scenario, 'actions': list(replay.canonical_actions),
                 'presentation_actions': list(replay.presentation_actions),
                 'canonical_action_records': deepcopy(list(replay.canonical_action_records)),
                 'action_source_grounding': deepcopy(before), 'proposition_ledger': seed,
                 'source_construction_advisory': deepcopy(replay.source_construction_advisory),
                 'framework_proposition_ledgers': {}, 'cycles': [{'cycle': 1, 'candidates': []}],
                 'judgment_status': 'NOT_ADJUDICATED', 'framework_runs': [],
                 'world_mutation': False, 'shared_input_sha256': fingerprint(shared),
                 'limits': ['Independent generation only; no peer review or collective judgment.',
                            'Native ledger admission does not establish every generated premise.',
                            'At most two adapter calls per framework; provider retry behavior is unchanged.']}
    quotes = (core_pack_for_specialists(frameworks, root=Path(core_root))
              if core_root else core_pack_for_specialists(frameworks))
    (output_dir/'framework_commitments.json').write_text(json.dumps(quotes, indent=2) + '\n')
    for name in frameworks:
        print('Independent ' + name + ' generation...', flush=True)
        folder = output_dir/name
        folder.mkdir(parents=True, exist_ok=True)
        recorded = RecordedModel(backend_factory(name), replay.scenario, folder/'model_calls.json')
        specialist = CompactLocalSpecialist(
            name, recorded, max_tokens=tokens,
            scenario_facts=extract_scenario_facts(replay.scenario),
            source_construction_advisory=deepcopy(replay.source_construction_advisory),
            canonical_action_records=deepcopy(list(replay.canonical_action_records)),
            core_quote_pack=deepcopy(quotes.get(name, {})))
        engine = WorkspaceEngine([specialist], WorkspaceConfig(
            max_cycles=1, max_cycle_extensions=0, enable_synthesis=False,
            enable_planning=False, enable_consensus_audit=False,
            enable_problem_state_audit=False, enable_reversal_audit=False))
        started = time.monotonic()
        result = engine.run(
            replay.scenario, list(replay.canonical_actions),
            scenario_facts=deepcopy(specialist.scenario_facts),
            source_action_legend=deepcopy(replay.source_action_legend),
            action_source_grounding=deepcopy(before),
            presentation_actions=list(replay.presentation_actions),
            canonical_action_records=deepcopy(list(replay.canonical_action_records)))
        payload = result.to_dict()
        (folder/'native_trace.json').write_text(json.dumps(payload, indent=2) + '\n')
        if payload['action_source_grounding']['world_model'] != before['world_model']:
            raise RuntimeError('Independent framework changed admitted world')
        # Preserve local hypotheses but never pass them to the next framework.
        aggregate['framework_proposition_ledgers'][name] = payload['proposition_ledger']
        candidates = payload['cycles'][-1]['candidates'] if payload['cycles'] else []
        aggregate['cycles'][0]['candidates'].extend(deepcopy(candidates))
        run = {'framework': name, 'adapter_calls': len(recorded.calls),
               'model': getattr(recorded.backend, 'model', 'test_backend'),
               'elapsed_seconds': round(time.monotonic()-started, 3),
               'candidate_statuses': [c['delegate_status'] for c in candidates],
               'native_ledger_statuses': [(c.get('committed_native_ledger') or {}).get('transaction_status', 'NONE') for c in candidates],
               'shared_input_sha256': fingerprint(shared), 'peer_input': False,
               'core_quote_present': bool(quotes.get(name))}
        aggregate['framework_runs'].append(run)
        write_outputs(aggregate, output_dir)
        print(name + ': ' + ', '.join(run['candidate_statuses']), flush=True)
    if replay.action_source_grounding != before or trace_path.read_bytes() != source_bytes:
        raise RuntimeError('Frozen source was mutated')
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parliament-root', type=Path, required=True)
    parser.add_argument('--frameworks', nargs='+', default=list(FRAMEWORKS))
    parser.add_argument('--model', default='o3')
    parser.add_argument('--core-root', type=Path,
                        default=Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.parliament_root.resolve()))
    from global_workspace.openai_backend import OpenAIWorkspaceLLM
    # Existing environment loading is read-only and never records the key.
    from dotenv import load_dotenv
    load_dotenv(Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env'))
    backend = OpenAIWorkspaceLLM(args.model, timeout=120)
    generate(args.trace, args.output_dir, args.frameworks, lambda name: backend,
             core_root=args.core_root if args.core_root.exists() else None)


if __name__ == '__main__':
    main()
