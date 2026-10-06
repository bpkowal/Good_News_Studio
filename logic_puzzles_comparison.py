"""Bounded matched-opening comparison of native deliberation and targeted review."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from logic_puzzles_frameworks import FRAMEWORKS, RecordedModel, fingerprint, generate
from logic_puzzles_review import close_deontological_map, review
from run_logic_puzzles import project_conflicts, write_graph


class OpeningThenLive:
    def __init__(self, opening, backend=None):
        self.opening, self.backend = deepcopy(opening), backend
        self.model = getattr(backend, 'model', 'o3')
        self.count = 0

    def complete_json(self, prompt, **kwargs):
        self.count += 1
        if self.count == 1:
            self.last_call_origin = 'SAVED_OPENING_RESPONSE'
            return deepcopy(self.opening)
        if self.backend is None:
            raise RuntimeError('Opening replay has no follow-up backend')
        self.last_call_origin = 'LIVE_FOLLOWUP'
        return self.backend.complete_json(prompt, **kwargs)


def metrics(trace):
    candidates = trace['cycles'][-1]['candidates'] if trace.get('cycles') else []
    graph = project_conflicts(trace)
    claims = [n for n in graph['nodes'] if n['kind'] == 'CLAIM']
    return {
        'valid_submissions': sum(c.get('delegate_status') == 'VALID' for c in candidates),
        'committed_frameworks': sum(str((c.get('committed_native_ledger') or {}).get('transaction_status', '')).startswith('COMMITTED') for c in candidates),
        'missing_frameworks': sorted(set(FRAMEWORKS)-{c['specialist'] for c in candidates if str((c.get('committed_native_ledger') or {}).get('transaction_status', '')).startswith('COMMITTED')}),
        'claims': len(claims),
        'calibration_errors': sum(len(n['record'].get('calibration_errors', [])) for n in claims),
        'framework_warnings': sum(len(c.get('framework_validation_errors', [])) for c in candidates),
        'internal_conflicts': sum(len(c.get('framework_internal_conflicts', [])) for c in candidates),
        'rejected_update_warnings': sum(sum(str(w).startswith('rejected update:') for w in c.get('framework_validation_errors', [])) for c in candidates),
        'operative_record_cycles': sorted({n['record'].get('cycle', 0) for n in claims}),
        'recommendations': {c['specialist']: c.get('recommended_action') for c in candidates},
        'deontological_records': [n['record'] for n in claims if n['origin'] == 'deontological'],
        'semantic_support': 'NOT_INDEPENDENTLY_ASSESSED',
    }


def compare(opening_dir, output_dir, backend_factory, *, tokens=3072):
    from global_workspace.engine import WorkspaceEngine, WorkspaceConfig
    from global_workspace.frozen_world_replay import load_frozen_world_trace
    from global_workspace.local_specialists import CompactLocalSpecialist, extract_scenario_facts
    opening_dir, output_dir = Path(opening_dir), Path(output_dir)
    original = json.loads((opening_dir/'framework_generation.json').read_text())
    shared = json.loads((opening_dir/'shared_input.json').read_text())
    quotes = json.loads((opening_dir/'framework_commitments.json').read_text())
    if fingerprint(shared) != original['shared_input_sha256']:
        raise ValueError('Opening input fingerprint mismatch')
    openings = {name: json.loads((opening_dir/name/'model_calls.json').read_text())[0]['response'] for name in FRAMEWORKS}
    opening_models = {run['model'] for run in original['framework_runs']}
    followup_models = {}
    def matched_backend(name):
        backend = backend_factory(name)
        model = getattr(backend, 'model', None)
        if model not in opening_models or len(opening_models) != 1:
            raise ValueError('Follow-up model differs from saved openings')
        followup_models[name] = model
        return backend
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = deepcopy(original)
    frozen['canonical_action_records'] = shared['canonical_action_records']
    frozen_path = output_dir/'frozen_input.json'
    frozen_path.write_text(json.dumps(frozen, indent=2)+'\n')
    normalizers = {'deontological': close_deontological_map}
    independent = generate(frozen_path, output_dir/'independent', FRAMEWORKS,
        lambda name: OpeningThenLive(openings[name]), tokens=tokens,
        core_quotes=quotes, response_normalizers=normalizers)
    replay = load_frozen_world_trace(frozen_path, expected_scenario=shared['scenario'])
    models, specialists = {}, []
    folder = output_dir/'legacy'; folder.mkdir()
    for name in FRAMEWORKS:
        models[name] = RecordedModel(OpeningThenLive(openings[name], matched_backend(name)),
            replay.scenario, folder/name/'model_calls.json', max_calls=2,
            response_normalizer=normalizers.get(name), independent=False)
        specialists.append(CompactLocalSpecialist(name, models[name], max_tokens=tokens,
            scenario_facts=extract_scenario_facts(replay.scenario),
            source_construction_advisory=deepcopy(replay.source_construction_advisory),
            canonical_action_records=deepcopy(shared['canonical_action_records']),
            core_quote_pack=deepcopy(quotes.get(name, {}))))
    print('Legacy native loop: identical openings, then one peer-exposed cycle...', flush=True)
    engine = WorkspaceEngine(specialists, WorkspaceConfig(max_cycles=2, high_urgency_cycles=2,
        stop_redundant_consensus_cycles=False, max_cycle_extensions=0,
        enable_synthesis=False, enable_planning=False, enable_consensus_audit=False,
        enable_problem_state_audit=False, enable_reversal_audit=False))
    legacy = engine.run(replay.scenario, list(replay.canonical_actions),
        scenario_facts=extract_scenario_facts(replay.scenario),
        source_action_legend=deepcopy(replay.source_action_legend),
        action_source_grounding=deepcopy(replay.action_source_grounding),
        presentation_actions=list(replay.presentation_actions),
        canonical_action_records=deepcopy(shared['canonical_action_records'])).to_dict()
    (folder/'native_trace.json').write_text(json.dumps(legacy, indent=2)+'\n')
    write_graph(project_conflicts(legacy), folder)
    targeted = review(output_dir/'independent/framework_generation.json', output_dir/'targeted',
        matched_backend, tokens=tokens, max_calls=1)
    traces = {'independent': independent, 'legacy': legacy, 'targeted': targeted}
    result = {'design': 'matched openings; equal maximum allowance, unequal actual usage',
        'controls': {'shared_input_sha256': fingerprint(shared), 'commitments_sha256': fingerprint(quotes),
            'opening_response_sha256': {n: fingerprint(r) for n,r in openings.items()},
            'opening_models': sorted(opening_models), 'followup_models': followup_models,
            'max_tokens': tokens, 'maximum_logical_calls_per_arm': 10,
            'uniform_display_closure': 'deontological fm from unchanged dp',
            'legacy_auxiliary_stages': 'disabled; native two-cycle deliberation and ledger validation'},
        'arms': {}, 'limitations': ['One scenario, one follow-up sample per framework; no statistical winner.',
            'Native warnings are diagnostic proxies, not independently scored semantic correctness.',
            'Fewer errors with missing ledgers is not an improvement. Agreement is not a success criterion.',
            'Legacy applies native continuity checks; targeted review uses a fresh owner workspace then explicit reconciliation. Acceptance policy is part of this comparison.',
            'Original openings are replayed, not fresh API calls; provider retries are outside logical call counts.']}
    for arm, trace in traces.items():
        item = metrics(trace)
        item['world_unchanged'] = trace['action_source_grounding']['world_model'] == shared['action_source_grounding']['world_model']
        if not item['world_unchanged']:
            raise RuntimeError(arm+' mutated admitted world')
        calls = [c for path in (output_dir/arm).glob('*/model_calls.json') for c in json.loads(path.read_text())]
        item['saved_opening_calls'] = sum(c.get('response_origin') == 'SAVED_OPENING_RESPONSE' for c in calls)
        item['followup_attempts'] = len(calls)-item['saved_opening_calls']
        item['logical_calls_including_openings'] = len(calls)+(5 if arm == 'targeted' else 0)
        result['arms'][arm] = item
    (output_dir/'matched_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Matched-opening pilot', '', result['design']+'.', '',
        '| Arm | Valid | Committed | Claims | Calibration errors | Framework warnings | Calls including openings |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for arm,m in result['arms'].items():
        lines.append('| '+arm+' | '+' | '.join(str(m[k]) for k in ('valid_submissions','committed_frameworks','claims','calibration_errors','framework_warnings','logical_calls_including_openings'))+' |')
    lines += ['', 'All admitted worlds unchanged. Framework warnings and retained uncertainty must be inspected alongside admission.', '']
    for arm in traces:
        lines += ['- ['+arm+' graph]('+arm+'/conflict_graph.md)']
    lines += ['', *result['limitations'], '']
    (output_dir/'matched_comparison.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opening-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parliament-root', type=Path, default=Path('/tmp/parliament-smoke-614af0c'))
    parser.add_argument('--model', default='o3')
    args = parser.parse_args()
    sys.path.insert(0, str(args.parliament_root))
    from dotenv import load_dotenv
    from global_workspace.openai_backend import OpenAIWorkspaceLLM
    load_dotenv(Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env'))
    def factory(name):
        backend = OpenAIWorkspaceLLM(args.model, timeout=120)
        backend.execution_mode = 'LIVE_OPENAI'
        return backend
    compare(args.opening_dir, args.output_dir, factory)


if __name__ == '__main__':
    main()
