"""One bounded native review round; immutable world, attributed revisions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from logic_puzzles_frameworks import RecordedModel, fingerprint
from run_logic_puzzles import project_conflicts, write_graph


def close_deontological_map(response):
    """Render the duplicate display map from typed duties, without changing them."""
    result = deepcopy(response)
    try:
        raw = result['choices'][0]['text']
        data = json.loads(raw)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return result, None
    duties, display = data.get('dp'), data.get('fm')
    if not isinstance(duties, dict) or not duties or not isinstance(display, dict):
        return result, None
    rendered = {}
    for action, duty in duties.items():
        if not isinstance(duty, dict) or not all(isinstance(duty.get(key), str) for key in ('v', 'n', 'dt', 'rel', 'rs')):
            return result, None
        rendered[action] = (duty['v'] + ': duty ' + duty['dt'] + ' under ' + duty['n'] +
                            '; relation ' + duty['rel'] + '; ' + duty['rs'])
    if set(rendered) != set(display):
        return result, None
    data['fm'] = rendered
    result['choices'][0]['text'] = json.dumps(data)
    return result, {'method': 'DISPLAY_MAP_FROM_TYPED_DUTY_ASSESSMENTS',
                    'changed_fields': ['fm'], 'before': display, 'after': rendered,
                    'semantic_fields_changed': False}


def review_jobs(assessment):
    graph = project_conflicts(assessment)
    nodes = {n['id']: n for n in graph['nodes']}
    groups = {}
    for issue_id in graph['review_queue'][:2]:
        issue = nodes[issue_id]
        groups.setdefault(issue['origin'], []).append(issue)
    jobs = []
    for origin, issues in groups.items():
        claim_ids = {ident for issue in issues for ident in issue['claim_ids']}
        related = set(claim_ids)
        for edge in graph['edges']:
            if edge['kind'] in {'SUPPORT', 'DEPENDENCY'} and (edge['target'] in claim_ids or edge['source'] in claim_ids):
                related.update((edge['source'], edge['target']))
        # Group questions about one framework to avoid competing rewrites of
        # the same native record. Native qa supplies one answer to this bundle.
        jobs.append({'framework': origin, 'issue_id': 'CHALLENGE:' + fingerprint([i['id'] for i in issues])[:16],
                     'issues': deepcopy(issues), 'claims': [deepcopy(nodes[i]) for i in sorted(claim_ids)],
                     'support_and_dependencies': [deepcopy(nodes[i]) for i in sorted(related - claim_ids)],
                     'question': 'Review each of these objections separately in your answer: ' +
                     '; '.join(i['label'] for i in issues) +
                     '. Identify any native classification you revise, retain, or leave unresolved. '
                     'A missing factual link remains missing; a framework commitment cannot supply it.'})
    return jobs


def reconcile(assessment, jobs, review_candidates, local_ledgers):
    """Replace only admitted owned proposals; preserve failed attempts separately."""
    result = deepcopy(assessment)
    original = assessment['cycles'][-1]['candidates']
    candidates = deepcopy(original)
    records = []
    for job in jobs:
        origin = job['framework']
        proposed = review_candidates.get(origin)
        ledger = (proposed or {}).get('committed_native_ledger') or {}
        response = (proposed or {}).get('challenge_response') or {}
        accepted = bool(proposed and proposed.get('specialist') == origin and proposed.get('schema_valid') and
                        ledger.get('transaction_status') in {'COMMITTED', 'COMMITTED_WITH_UNCERTAINTY'} and
                        response.get('issue_id') == job['issue_id'])
        prior = next(c for c in original if c['specialist'] == origin)
        if accepted:
            candidates = [deepcopy(proposed) if c['specialist'] == origin else c for c in candidates]
            result['framework_proposition_ledgers'][origin] = deepcopy(local_ledgers[origin])
        for issue in job['issues']:
            claim = next(c for c in job['claims'] if c['id'] in issue['claim_ids'])
            old = claim['record']
            replacements = [r for r in ledger.get('records', [])
                            if r.get('assessment_node_id') == old.get('assessment_node_id')]
            retained_error = any(issue['label'] in r.get('calibration_errors', []) for r in replacements)
            removed = bool(accepted and replacements and issue['conflict_type'] == 'UNSUPPORTED_INFERENCE' and not retained_error)
            changes = [{field: {'before': old.get(field), 'after': row.get(field)}
                        for field in ('harm_relation', 'means_relation', 'verdict', 'resolution_status')
                        if old.get(field) != row.get(field)} for row in replacements]
            unresolved_fields = [{field: value for field, value in row.items()
                                  if isinstance(value, str) and value in {'UNRESOLVED', 'CONTESTED', 'UNCERTAIN'}}
                                 for row in replacements]
            records.append({'original_issue_id': issue['id'], 'bundle_issue_id': job['issue_id'],
                            'framework': origin, 'objection': issue['label'],
                            'original_claim_id': claim['id'], 'original_record': deepcopy(old),
                            'proposed_records': deepcopy(replacements),
                            'review_status': 'RECONCILED' if accepted else 'REJECTED_OR_UNAVAILABLE',
                            'objection_check': 'NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR' if removed else 'OPEN',
                            'semantic_resolution': 'NOT_INDEPENDENTLY_VERIFIED',
                            'classification_changes': changes,
                            'remaining_unresolved_fields': unresolved_fields,
                            'framework_validation_errors': deepcopy((proposed or {}).get('framework_validation_errors', [])),
                            'response': deepcopy(response),
                            'recommendation_before': prior.get('recommended_action'),
                            'recommendation_after': proposed.get('recommended_action') if accepted else prior.get('recommended_action')})
    result['cycles'].append({'cycle': 2, 'phase': 'TARGETED_REVIEW', 'candidates': candidates})
    result['targeted_review'] = {'rounds': 1, 'issues': records,
                                'proposal_attempts': deepcopy(review_candidates),
                                'world_mutation': False, 'judgment_status': 'NOT_ADJUDICATED'}
    return result


def write_review(result, output_dir):
    output_dir = Path(output_dir)
    (output_dir/'targeted_review.json').write_text(json.dumps(result, indent=2)+'\n')
    graph = project_conflicts(result)
    graph['limits'][0] = 'One targeted native review round completed; no collective judgment.'
    write_graph(graph, output_dir)
    lines = ['# Targeted framework review', '',
             'One round, at most two objections. Native revisions remain attributed; the shared world is unchanged.', '',
             'Execution modes: ' + json.dumps(result['targeted_review'].get('execution_modes', {})) + '.', '',
             'Removing a validator objection is not independent proof of semantic correctness.', '']
    for issue in result['targeted_review']['issues']:
        lines += ['## ' + issue['objection'], '',
                  'Framework: ' + issue['framework'] + '. Review: ' + issue['review_status'] + '.', '',
                  'Objection check: ' + issue['objection_check'] + '.', '',
                  'Recommendation: ' + str(issue['recommendation_before']) + ' → ' + str(issue['recommendation_after']), '',
                  'Reviewer answer: ' + issue['response'].get('answer', 'No admitted response.'), '',
                  'Follow-up: ' + issue['response'].get('follow_up_question', 'unresolved'), '',
                  'Classification changes and remaining uncertainty:', '', '```json',
                  json.dumps({'changes': issue['classification_changes'],
                              'unresolved': issue['remaining_unresolved_fields'],
                              'native_warnings': issue['framework_validation_errors']}, indent=2), '```', '',
                  'Original record and proposed replacement:', '', '```json',
                  json.dumps({'original': issue['original_record'], 'proposed': issue['proposed_records']}, indent=2), '```', '']
    lines += ['Residual claims and conflicts: [graph](conflict_graph.md).', '']
    (output_dir/'targeted_review.md').write_text('\n'.join(lines))


def review(assessment_path, output_dir, backend_factory, *, tokens=3072):
    from global_workspace.engine import WorkspaceEngine, WorkspaceConfig
    from global_workspace.frozen_world_replay import load_frozen_world_trace
    from global_workspace.local_specialists import CompactLocalSpecialist, extract_scenario_facts
    from global_workspace.models import WorkspaceBroadcast
    assessment_path, output_dir = Path(assessment_path), Path(output_dir)
    source_bytes = assessment_path.read_bytes()
    assessment = json.loads(source_bytes)
    shared = json.loads((assessment_path.parent/'shared_input.json').read_text())
    quotes = json.loads((assessment_path.parent/'framework_commitments.json').read_text())
    if fingerprint(shared) != assessment['shared_input_sha256'] or shared['scenario'] != assessment['scenario'] or shared['action_source_grounding'] != assessment['action_source_grounding']:
        raise ValueError('Review shared input differs from assessed input')
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_payload = deepcopy(assessment)
    replay_payload['canonical_action_records'] = deepcopy(shared['canonical_action_records'])
    replay_path = output_dir/'frozen_review_input.json'
    replay_path.write_text(json.dumps(replay_payload, indent=2)+'\n')
    replay = load_frozen_world_trace(replay_path, expected_scenario=assessment['scenario'])
    if len(replay.canonical_actions) != 2:
        raise ValueError('Targeted review pilot requires exactly two actions')
    jobs = review_jobs(assessment)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'review_jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    source_graph = project_conflicts(assessment)
    write_graph(source_graph, output_dir/'before')
    attempts, ledgers, execution_modes = {}, {}, {}
    for job in jobs:
        name = job['framework']
        folder = output_dir/name
        folder.mkdir(parents=True, exist_ok=True)
        print('Targeted review: ' + name + ' (' + str(len(job['issues'])) + ' objections)...', flush=True)
        model = RecordedModel(backend_factory(name), replay.scenario, folder/'model_calls.json', review_context=job,
                              response_normalizer=close_deontological_map if name == 'deontological' else None)
        execution_modes[name] = getattr(model.backend, 'execution_mode', 'MODEL_BACKEND')
        specialist = CompactLocalSpecialist(name, model, max_tokens=tokens,
            scenario_facts=extract_scenario_facts(replay.scenario),
            source_construction_advisory=deepcopy(replay.source_construction_advisory),
            canonical_action_records=deepcopy(shared['canonical_action_records']),
            core_quote_pack=deepcopy(quotes.get(name, {})))
        agenda = {'issue_id': job['issue_id'], 'question': job['question'],
                  'generated_by': 'NATIVE_LEDGER_CALIBRATION_REVIEW', 'about_specialist': name,
                  'target_specialists': [name], 'challenge_kind': 'INFERENCE',
                  'grounding_status': 'ATTRIBUTED_OBJECTION', 'raised_by': ''}
        engine = WorkspaceEngine([specialist], WorkspaceConfig(max_cycles=1, max_cycle_extensions=0,
            enable_synthesis=False, enable_planning=False, enable_consensus_audit=False,
            enable_problem_state_audit=False, enable_reversal_audit=False))
        native = engine.run(replay.scenario, list(replay.canonical_actions),
            initial_broadcast=WorkspaceBroadcast(challenge_agenda=(agenda,)),
            scenario_facts=deepcopy(specialist.scenario_facts),
            source_action_legend=deepcopy(replay.source_action_legend),
            action_source_grounding=deepcopy(replay.action_source_grounding),
            presentation_actions=list(replay.presentation_actions),
            canonical_action_records=deepcopy(shared['canonical_action_records'])).to_dict()
        (folder/'native_trace.json').write_text(json.dumps(native, indent=2)+'\n')
        if native['action_source_grounding']['world_model'] != replay.action_source_grounding['world_model']:
            raise RuntimeError('Targeted review changed admitted world')
        if native['cycles']:
            attempts[name] = native['cycles'][-1]['candidates'][0]
            ledgers[name] = native['proposition_ledger']
        result = reconcile(assessment, jobs, attempts, ledgers)
        write_review(result, output_dir)
    result = reconcile(assessment, jobs, attempts, ledgers)
    result['targeted_review']['adapter_calls'] = {
        job['framework']: len(json.loads((output_dir/job['framework']/'model_calls.json').read_text())) for job in jobs}
    result['targeted_review']['execution_modes'] = execution_modes
    if assessment_path.read_bytes() != source_bytes:
        raise RuntimeError('Original assessment changed during review')
    write_review(result, output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parliament-root', type=Path, required=True)
    parser.add_argument('--model', default='o3')
    args = parser.parse_args()
    sys.path.insert(0, str(args.parliament_root.resolve()))
    from dotenv import load_dotenv
    from global_workspace.openai_backend import OpenAIWorkspaceLLM
    load_dotenv(Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env'))
    backend = OpenAIWorkspaceLLM(args.model, timeout=120)
    backend.execution_mode = 'LIVE_OPENAI'
    review(args.trace, args.output_dir, lambda name: backend)


if __name__ == '__main__':
    main()
