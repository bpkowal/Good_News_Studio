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
        prior = next(c for c in original if c['specialist'] == origin)
        native_rejected = bool(proposed and ledger == prior.get('committed_native_ledger') and
            any(str(w).startswith('rejected update:') for w in proposed.get('framework_validation_errors', [])))
        accepted = bool(proposed and proposed.get('specialist') == origin and proposed.get('schema_valid') and
                        ledger.get('transaction_status') in {'COMMITTED', 'COMMITTED_WITH_UNCERTAINTY'} and
                        response.get('issue_id') == job['issue_id'] and not native_rejected)
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
                        for field in sorted(set(old) | set(row))
                        if old.get(field) != row.get(field)} for row in replacements]
            unresolved_fields = [{field: value for field, value in row.items()
                                  if isinstance(value, str) and value in {'UNRESOLVED', 'CONTESTED', 'UNCERTAIN', 'UNKNOWN'}}
                                 for row in replacements]
            records.append({'original_issue_id': issue['id'], 'bundle_issue_id': job['issue_id'],
                            'framework': origin, 'objection': issue['label'],
                            'original_claim_id': claim['id'], 'original_record': deepcopy(old),
                            'proposed_records': deepcopy(replacements),
                            'review_status': 'RECONCILED' if accepted else
                                'RETAINED_AFTER_NATIVE_REJECTION' if native_rejected else 'REJECTED_OR_UNAVAILABLE',
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
                                'framework_changes': {job['framework']: {
                                    key: {'before': prior.get(key), 'after': proposed.get(key)}
                                    for key in sorted(set(prior) | set(proposed))
                                    if prior.get(key) != proposed.get(key)}
                                    for job in jobs
                                    for prior in [next(c for c in original if c['specialist'] == job['framework'])]
                                    for proposed in [review_candidates.get(job['framework'], {})]},
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
             'Acceptance path: ' + result['targeted_review'].get('acceptance_path', 'see native trace') + '.', '',
             'Removing a validator objection is not independent proof of semantic correctness.', '']
    for issue in result['targeted_review']['issues']:
        lines += ['## ' + issue['objection'], '',
                  'Framework: ' + issue['framework'] + '. Review: ' + issue['review_status'] + '.', '',
                  'Objection check: ' + issue['objection_check'] + '.', '',
                  'Recommendation: ' + str(issue['recommendation_before']) + ' → ' + str(issue['recommendation_after']), '',
                  'Reviewer answer: ' + issue['response'].get('answer', 'No admitted response.'), '',
                  'Follow-up: ' + issue['response'].get('follow_up_question', 'unresolved'), '',
                  'All native record changes and remaining uncertainty:', '', '```json',
                  json.dumps({'changes': issue['classification_changes'],
                              'unresolved': issue['remaining_unresolved_fields'],
                              'native_warnings': issue['framework_validation_errors']}, indent=2), '```', '',
                  'Original record and proposed replacement:', '', '```json',
                  json.dumps({'original': issue['original_record'], 'proposed': issue['proposed_records']}, indent=2), '```', '']
    lines += ['Residual claims and conflicts: [graph](conflict_graph.md).', '']
    if result['targeted_review'].get('retained_proposals'):
        lines += ['## Disputed full proposals', '',
                  'These remain attributed and nonoperative. No conclusion is promoted to the shared world.', '',
                  '```json', json.dumps(result['targeted_review']['retained_proposals'], indent=2), '```', '']
    if result['targeted_review'].get('isolated_classification_reviews'):
        lines += ['## Isolated native revalidation', '', '```json',
                  json.dumps(result['targeted_review']['isolated_classification_reviews'], indent=2), '```', '']
    for name, changes in result['targeted_review'].get('framework_changes', {}).items():
        lines += ['## All changed fields: ' + name, '', '```json',
                  json.dumps(changes, indent=2), '```', '']
    (output_dir/'targeted_review.md').write_text('\n'.join(lines))


def review(assessment_path, output_dir, backend_factory, *, tokens=3072, max_calls=2, allow_partial=True):
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
    retained_proposals, partial_reviews = [], []
    for job in jobs:
        name = job['framework']
        folder = output_dir/name
        folder.mkdir(parents=True, exist_ok=True)
        print('Targeted review: ' + name + ' (' + str(len(job['issues'])) + ' objections)...', flush=True)
        prior_calls = assessment_path.parent/name/'model_calls.json'
        saved_calls = json.loads(prior_calls.read_text()) if prior_calls.exists() else []
        saved = next((c.get('response') for c in reversed(saved_calls) if c.get('response') is not None), None)
        if saved is None:
            (folder/'prior_state_problem.json').write_text(json.dumps({
                'status': 'PRIOR_REPLAY_UNAVAILABLE', 'framework': name,
                'retained_original': True}, indent=2)+'\n')
            continue
        model = RecordedModel(backend_factory(name), replay.scenario, folder/'model_calls.json', review_context=job,
                              response_normalizer=close_deontological_map if name == 'deontological' else None,
                              max_calls=max_calls)
        execution_modes[name] = getattr(model.backend, 'execution_mode', 'MODEL_BACKEND')
        class PriorThenReview:
            reviewing = False
            seeded = False
            def complete_json(self, prompt, **kwargs):
                if self.reviewing:
                    return model.complete_json(prompt, **kwargs)
                if self.seeded:
                    raise RuntimeError('Prior native ledger could not be replayed without repair')
                self.seeded = True
                response = deepcopy(saved)
                if name == 'deontological':
                    response, _ = close_deontological_map(response)
                (folder/'prior_replay.json').write_text(json.dumps({
                    'origin': 'SAVED_PRIOR_RESPONSE', 'response': saved,
                    'response_sha256': fingerprint(saved), 'new_model_call': False}, indent=2)+'\n')
                return response
        staged = PriorThenReview()
        specialist = CompactLocalSpecialist(name, staged, max_tokens=tokens,
            scenario_facts=extract_scenario_facts(replay.scenario),
            source_construction_advisory=deepcopy(replay.source_construction_advisory),
            canonical_action_records=deepcopy(shared['canonical_action_records']),
            core_quote_pack=deepcopy(quotes.get(name, {})))
        agenda = {'issue_id': job['issue_id'], 'question': job['question'],
                  'generated_by': 'NATIVE_LEDGER_CALIBRATION_REVIEW', 'about_specialist': name,
                  'target_specialists': [name], 'challenge_kind': 'INFERENCE',
                  'grounding_status': 'ATTRIBUTED_OBJECTION', 'raised_by': ''}
        class NativeReviewEngine(WorkspaceEngine):
            def prepare_cycle_input(self, **kwargs):
                if kwargs['cycle_number'] == 2:
                    staged.reviewing = True
                    kwargs['broadcast'] = WorkspaceBroadcast(constraint='PROPOSAL_REVIEW',
                        challenge_agenda=(deepcopy(agenda),))
                    (folder/'prior_framework_state.json').write_text(json.dumps({
                        'prior_framework_state': deepcopy(specialist.previous_framework_state),
                        'prior_candidate': next(c for c in assessment['cycles'][-1]['candidates'] if c['specialist'] == name),
                        'review_constraint': 'PROPOSAL_REVIEW'}, indent=2)+'\n')
                return super().prepare_cycle_input(**kwargs)
        engine = NativeReviewEngine([specialist], WorkspaceConfig(max_cycles=2, high_urgency_cycles=2,
            min_valid_specialists=1, stop_redundant_consensus_cycles=False, max_cycle_extensions=0,
            enable_synthesis=False, enable_planning=False, enable_consensus_audit=False,
            enable_problem_state_audit=False, enable_reversal_audit=False))
        native = engine.run(replay.scenario, list(replay.canonical_actions),
            scenario_facts=deepcopy(specialist.scenario_facts),
            source_action_legend=deepcopy(replay.source_action_legend),
            action_source_grounding=deepcopy(replay.action_source_grounding),
            presentation_actions=list(replay.presentation_actions),
            canonical_action_records=deepcopy(shared['canonical_action_records'])).to_dict()
        (folder/'native_trace.json').write_text(json.dumps(native, indent=2)+'\n')
        if native['action_source_grounding']['world_model'] != replay.action_source_grounding['world_model']:
            raise RuntimeError('Targeted review changed admitted world')
        if len(native['cycles']) >= 2:
            attempts[name] = native['cycles'][-1]['candidates'][0]
            ledgers[name] = native['proposition_ledger']
        interim = reconcile(assessment, [job], attempts, ledgers)
        rejected = any(i['review_status'] == 'RETAINED_AFTER_NATIVE_REJECTION'
                       for i in interim['targeted_review']['issues'])
        if rejected and model.calls and model.calls[-1].get('response'):
            raw = deepcopy(model.calls[-1]['response'])
            retained_proposals.append({'framework': name, 'issue_id': job['issue_id'],
                'status': 'DISPUTED_NONOPERATIVE', 'response': raw,
                'native_attempt': deepcopy(attempts[name]),
                'world_state_authority': 'NONE', 'semantic_support': 'NOT_INDEPENDENTLY_VERIFIED'})
            if allow_partial:
                from logic_puzzles_partial_review import isolate_classifications, IsolatedReplay
                isolated, patches = isolate_classifications(saved, raw, job)
                if isolated is not None:
                    partial_folder = folder/'isolated_classifications'
                    partial = review(assessment_path, partial_folder,
                        lambda owner: IsolatedReplay(isolated), tokens=tokens, max_calls=1,
                        allow_partial=False)
                    accepted_partial = all(i['review_status'] == 'RECONCILED'
                        for i in partial['targeted_review']['issues'] if i['framework'] == name)
                    partial_reviews.append({'framework': name, 'patches': patches,
                        'response': isolated, 'source_response_sha256': fingerprint(raw),
                        'native_accepted': accepted_partial, 'new_api_calls': 0,
                        'path': str(partial_folder), 'full_bundle_accepted': False})
                    if accepted_partial:
                        attempts[name] = next(c for c in partial['cycles'][-1]['candidates'] if c['specialist'] == name)
                        ledgers[name] = partial['framework_proposition_ledgers'][name]
        result = reconcile(assessment, jobs, attempts, ledgers)
        write_review(result, output_dir)
    result = reconcile(assessment, jobs, attempts, ledgers)
    result['targeted_review']['adapter_calls'] = {
        job['framework']: len(json.loads((output_dir/job['framework']/'model_calls.json').read_text()))
        if (output_dir/job['framework']/'model_calls.json').exists() else 0 for job in jobs}
    result['targeted_review']['acceptance_path'] = 'NATIVE_RECURRENT_PROPOSAL_REVIEW'
    result['targeted_review']['prior_replay_calls'] = {job['framework']: 1 if
        (output_dir/job['framework']/'prior_replay.json').exists() else 0 for job in jobs}
    result['targeted_review']['execution_modes'] = execution_modes
    result['targeted_review']['retained_proposals'] = retained_proposals
    result['targeted_review']['isolated_classification_reviews'] = partial_reviews
    for partial in partial_reviews:
        if partial['native_accepted']:
            patch_issue_ids = {p['issue_id'] for p in partial['patches']}
            for issue in result['targeted_review']['issues']:
                if issue['framework'] == partial['framework']:
                    issue['review_status'] = ('ISOLATED_CLASSIFICATION_ACCEPTED'
                        if issue['original_issue_id'] in patch_issue_ids else 'ORIGINAL_CLASSIFICATION_RETAINED')
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
