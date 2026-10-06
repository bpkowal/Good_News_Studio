"""Replay one identical saved revision through fresh and recurrent acceptance."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from logic_puzzles_frameworks import RecordedModel, fingerprint
from logic_puzzles_review import close_deontological_map, review, review_jobs
from run_logic_puzzles import project_conflicts, write_graph


class SavedResponses:
    model = 'o3'
    execution_mode = 'SAVED_RESPONSE_REPLAY'
    def __init__(self, responses):
        self.responses = deepcopy(responses)
        self.count = 0

    def complete_json(self, prompt, **kwargs):
        if self.count >= len(self.responses):
            raise RuntimeError('No additional saved response; no live backend available')
        response = deepcopy(self.responses[self.count])
        self.count += 1
        self.last_call_origin = 'SAVED_RESPONSE_REPLAY'
        return response


def compare_paths(comparison_dir, output_dir):
    from global_workspace.engine import WorkspaceEngine, WorkspaceConfig
    from global_workspace.frozen_world_replay import load_frozen_world_trace
    from global_workspace.local_specialists import CompactLocalSpecialist, extract_scenario_facts
    from global_workspace.models import WorkspaceBroadcast

    comparison_dir, output_dir = Path(comparison_dir), Path(output_dir)
    baseline_path = comparison_dir/'independent/framework_generation.json'
    baseline = json.loads(baseline_path.read_text())
    shared = json.loads((baseline_path.parent/'shared_input.json').read_text())
    quotes = json.loads((baseline_path.parent/'framework_commitments.json').read_text())
    job = next(j for j in review_jobs(baseline) if j['framework'] == 'deontological')
    opening = json.loads((baseline_path.parent/'deontological/model_calls.json').read_text())[0]['response']
    revision = json.loads((comparison_dir/'targeted/deontological/model_calls.json').read_text())[0]['response']
    source_hashes = {str(p): fingerprint(json.loads(p.read_text())) for p in (
        baseline_path, baseline_path.parent/'shared_input.json',
        comparison_dir/'targeted/deontological/model_calls.json')}
    output_dir.mkdir(parents=True, exist_ok=True)
    replay = load_frozen_world_trace(baseline_path, expected_scenario=shared['scenario'])
    # Immutable historical control: production review now uses prior-state native review.
    fresh = json.loads((comparison_dir/'targeted/targeted_review.json').read_text())
    fresh_folder = output_dir/'fresh_targeted'
    (fresh_folder/'deontological').mkdir(parents=True, exist_ok=True)
    (fresh_folder/'targeted_review.json').write_text(json.dumps(fresh, indent=2)+'\n')
    (fresh_folder/'deontological/model_calls.json').write_text(
        (comparison_dir/'targeted/deontological/model_calls.json').read_text())
    write_graph(project_conflicts(fresh), fresh_folder)
    traces = {'fresh_targeted': fresh}
    # Observation-only wrapper: all native audit and commit code is unchanged.
    class ObservedSpecialist(CompactLocalSpecialist):
        def _audit_framework_state_change(self, candidate, broadcast):
            before = deepcopy(candidate.framework_validation_errors)
            event = {'prior_state_present': bool(self.previous_framework_state),
                'prior_state': deepcopy(self.previous_framework_state),
                'constraint': broadcast.constraint,
                'challenge_agenda': deepcopy(list(broadcast.challenge_agenda)),
                'challenge_response': deepcopy(candidate.challenge_response),
                'workspace_reasoning_effect': candidate.workspace_reasoning_effect,
                'change_justification': candidate.change_justification,
                'framework_application': candidate.framework_application}
            super()._audit_framework_state_change(candidate, broadcast)
            event['new_warnings'] = [w for w in candidate.framework_validation_errors if w not in before]
            event['framework_retention_status'] = candidate.framework_retention_status
            self.observations.append(event)

    agenda = {'issue_id': job['issue_id'], 'question': job['question'],
        'generated_by': 'NATIVE_LEDGER_CALIBRATION_REVIEW', 'about_specialist': 'deontological',
        'target_specialists': ['deontological'], 'challenge_kind': 'INFERENCE',
        'grounding_status': 'ATTRIBUTED_OBJECTION', 'raised_by': ''}
    for arm, constraint, assigned in (
        ('recurrent_ordinary', None, False),
        ('recurrent_assigned_open', 'OPEN_DELIBERATION', True),
        ('recurrent_assigned_review', 'PROPOSAL_REVIEW', True)):
        folder = output_dir/arm
        folder.mkdir(parents=True, exist_ok=True)
        class ObservedModel(RecordedModel):
            def complete_json(self, prompt, **kwargs):
                if len(self.calls) >= self.max_calls:
                    (folder/'denied_repair.json').write_text(json.dumps({
                        'prompt': prompt, 'reason': 'SAVED_REPLAY_CALL_LIMIT',
                        'model_called': False}, indent=2)+'\n')
                return super().complete_json(prompt, **kwargs)
        model = ObservedModel(SavedResponses([opening, revision]), replay.scenario,
            folder/'model_calls.json', max_calls=2, response_normalizer=close_deontological_map,
            independent=False)
        specialist = ObservedSpecialist('deontological', model, max_tokens=3072,
            scenario_facts=extract_scenario_facts(replay.scenario),
            source_construction_advisory=deepcopy(replay.source_construction_advisory),
            canonical_action_records=deepcopy(shared['canonical_action_records']),
            core_quote_pack=deepcopy(quotes['deontological']))
        specialist.observations = []
        cycle_inputs = []
        class ReplayEngine(WorkspaceEngine):
            def prepare_cycle_input(self, **kwargs):
                if kwargs['cycle_number'] == 2 and assigned:
                    broadcast = deepcopy(kwargs['broadcast'])
                    broadcast.constraint = constraint
                    broadcast.challenge_agenda = (deepcopy(agenda),)
                    kwargs['broadcast'] = broadcast
                cycle_inputs.append({'cycle': kwargs['cycle_number'],
                    'constraint': kwargs['broadcast'].constraint,
                    'challenge_agenda': deepcopy(list(kwargs['broadcast'].challenge_agenda))})
                return super().prepare_cycle_input(**kwargs)
        engine = ReplayEngine([specialist], WorkspaceConfig(max_cycles=2, high_urgency_cycles=2,
            min_valid_specialists=1,
            stop_redundant_consensus_cycles=False, max_cycle_extensions=0,
            enable_synthesis=False, enable_planning=False, enable_consensus_audit=False,
            enable_problem_state_audit=False, enable_reversal_audit=False))
        trace = engine.run(replay.scenario, list(replay.canonical_actions),
            scenario_facts=extract_scenario_facts(replay.scenario),
            source_action_legend=deepcopy(replay.source_action_legend),
            action_source_grounding=deepcopy(replay.action_source_grounding),
            presentation_actions=list(replay.presentation_actions),
            canonical_action_records=deepcopy(shared['canonical_action_records'])).to_dict()
        (folder/'native_trace.json').write_text(json.dumps(trace, indent=2)+'\n')
        (folder/'continuity_audit.json').write_text(json.dumps(specialist.observations, indent=2)+'\n')
        (folder/'cycle_inputs.json').write_text(json.dumps(cycle_inputs, indent=2)+'\n')
        write_graph(project_conflicts(trace), folder)
        traces[arm] = trace
        print(arm+' replay complete', flush=True)
    original = next(c for c in baseline['cycles'][-1]['candidates'] if c['specialist'] == 'deontological')
    original_records = original['committed_native_ledger']['records']
    result = {'revision_sha256': fingerprint(revision), 'shared_input_sha256': fingerprint(shared),
              'fresh_api_calls': 0, 'native_validation_modified': False,
              'controls': {'scope': 'single-owner acceptance replay',
                           'recurrent_quorum': 1, 'original_pilot_quorum': 2,
                           'quorum_note': 'One valid owner required to reach cycle two; ledger validation and continuity code unchanged.'},
              'arms': {}}
    for arm, trace in traces.items():
        candidate = next(c for c in trace['cycles'][-1]['candidates'] if c['specialist'] == 'deontological')
        ledger = candidate.get('committed_native_ledger') or {}
        records = ledger.get('records', [])
        diffs = []
        for new in records:
            old = next((r for r in original_records if r['canonical_action_id'] == new['canonical_action_id'] and r.get('assessment_role') == new.get('assessment_role')), {})
            diffs.append({'action_id': new['canonical_action_id'], 'changes': {
                key: {'before': old.get(key), 'after': new.get(key)}
                for key in sorted(set(old)|set(new)) if old.get(key) != new.get(key)}})
        call_path = output_dir/arm/('deontological/model_calls.json' if arm == 'fresh_targeted' else 'model_calls.json')
        calls = json.loads(call_path.read_text())
        if fingerprint(calls[-1]['response']) != fingerprint(revision):
            raise RuntimeError('Revision payload differs across acceptance paths')
        world_same = trace['action_source_grounding']['world_model'] == shared['action_source_grounding']['world_model']
        if not world_same: raise RuntimeError('Replay changed admitted world')
        result['arms'][arm] = {'transaction_status': ledger.get('transaction_status', 'NONE'),
            'completed_cycles': 1 if arm == 'fresh_targeted' else len(trace['cycles']),
            'halted_by': trace.get('halted_by'),
            'operative_means': {r['canonical_action_id']: r.get('means_relation') for r in records},
            'framework_warnings': candidate.get('framework_validation_errors', []),
            'retention_status': candidate.get('framework_retention_status'),
            'world_unchanged': world_same, 'all_record_changes': diffs,
            'recommendation': candidate.get('recommended_action')}
    for p, digest in source_hashes.items():
        if fingerprint(json.loads(Path(p).read_text())) != digest: raise RuntimeError('Source artifact changed')
    (output_dir/'acceptance_paths.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Identical-revision acceptance replay', '',
        'No new model calls. Same saved targeted revision in every path; native validators unchanged.', '',
        '| Path | Completed cycles | A0 operative means relation | Retention status |', '| --- | --- | --- | --- |']
    for arm,item in result['arms'].items():
        lines.append('| '+arm+' | '+str(item['completed_cycles'])+' | '+str(item['operative_means'].get('A0'))+' | '+str(item['retention_status'])+' |')
    lines += ['', 'All admitted worlds and source artifacts unchanged.', '',
        'Full field changes and warnings: [JSON](acceptance_paths.json).', '']
    for arm in traces: lines.append('- ['+arm+' graph]('+arm+'/conflict_graph.md)')
    (output_dir/'acceptance_paths.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', type=Path, default=Path('diagnostics/logic_puzzles_matched_live'))
    parser.add_argument('--output-dir', type=Path, default=Path('diagnostics/logic_puzzles_acceptance_replay'))
    parser.add_argument('--parliament-root', type=Path, default=Path('/tmp/parliament-smoke-614af0c'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.parliament_root))
    compare_paths(args.comparison_dir, args.output_dir)


if __name__ == '__main__': main()
