"""Matched native reasoning pilot; no changes to admission or normative rules."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

FRAMEWORKS = ('utilitarian', 'deontological', 'care')
CHECKLIST = '''ARGUMENT QUALITY EXPERIMENT — use the existing response schema.
Distinguish source facts, framework commitments, and contestable normative bridges.
For utilitarian reasoning, show action-by-action arithmetic for expected survivors
as a restricted endpoint. Do not equate survivors with total welfare, invent utility
or duration, or supply an unsupported numeric welfare EV. Explain what can and
cannot be concluded from this restricted comparison.
For deontology, state each duty, bearer, beneficiary, source, classification, and
priority rule. Do not classify an imperfect rescue duty as perfect merely because
one action has a higher survival probability. If duties conflict, explain the
priority argument or leave the normative conflict unresolved.
For care, explain the relationship or entrusted responsibility that supports your
comparison. If effectiveness matters, justify that bridge without inventing a
relationship. Address the strongest objection to your own recommendation and
give a reversal condition. Agreement with another framework is neither required
nor penalized. An explicit promise can be considered from the exact source text;
it does not automatically override every other consideration.
'''


def transform(value, fn):
    if isinstance(value, str): return fn(value)
    if isinstance(value, list): return [transform(v, fn) for v in value]
    if isinstance(value, dict): return {k: transform(v, fn) for k, v in value.items()}
    return value


def prepare(output):
    from global_workspace.world_admission import restore_admitted_world, serialize_world_for_admission, admit_world_payload
    from global_workspace.frozen_world_replay import load_frozen_world_trace
    from logic_puzzles_frameworks import fingerprint
    baseline = json.loads(Path('fixtures/recognition_gap_medicine_frozen.json').read_text())
    swap = lambda s: s.replace('95%', '__SWAP__').replace('5%', '95%').replace('__SWAP__', '5%')
    cases = {'baseline': deepcopy(baseline), 'probability_swap': transform(baseline, swap),
             'competing_promise': deepcopy(baseline)}
    cases['competing_promise']['scenario'] += ' Ada promised Cara that she would give Cara the medicine.'
    manifest = {'version': 'argument-quality-pilot/1', 'frameworks': list(FRAMEWORKS),
                'arms': ['standard', 'argument_checklist'], 'calls_per_framework_arm_case': 1,
                'cases': {}, 'limits': ['One sample per cell; exploratory, not a causal estimate.',
                'No response normalization, forced verdict, or altered native validator.',
                'Promise is an explicit source stipulation, not an added world effect.',
                'Native ledger admission and argument quality are reported separately.']}
    for name, frozen in cases.items():
        grounding = frozen['action_source_grounding']
        world = restore_admitted_world(grounding['world_model'])
        clauses = [{'clause_id': 'C'+str(i), 'text': s.strip()+'.'}
                   for i, s in enumerate(frozen['scenario'].split('.')) if s.strip()]
        admitted = admit_world_payload(serialize_world_for_admission(world), clauses=clauses,
                       action_ids=['A0', 'A1'], action_texts=dict(zip(['A0', 'A1'], frozen['actions'])))
        if admitted.admission.status not in ('COMMITTED', 'COMMITTED_WITH_UNCERTAINTY'):
            raise ValueError(name+' did not re-admit')
        # Keep the known frozen serialization, after a fresh native admission check.
        path = output/name/'frozen_world_trace.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(frozen, indent=2)+'\n')
        replay = load_frozen_world_trace(path, expected_scenario=frozen['scenario'])
        manifest['cases'][name] = {'trace': str(path), 'fingerprint': replay.fingerprint,
            'world_sha256': fingerprint(grounding['world_model']), 'admission': admitted.admission.status,
            'action_labels': frozen['actions'], 'source': frozen['scenario'],
            'expected_survivor_endpoint': {'A0': .05 if name == 'probability_swap' else .95,
                                           'A1': .95 if name == 'probability_swap' else .05},
            'promise_graph_coverage': 'SOURCE_ONLY_NOT_PROJECTED' if name == 'competing_promise' else 'NOT_APPLICABLE'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'treatment_prompt.txt').write_text(CHECKLIST)
    return manifest


class ChecklistBackend:
    def __init__(self, backend): self.backend, self.model = backend, backend.model
    def complete_json(self, prompt, **kwargs):
        return self.backend.complete_json(prompt+'\n'+CHECKLIST, **kwargs)


def summarize(output, manifest):
    from logic_puzzles_frameworks import fingerprint
    rows = []
    controls = []
    for case in manifest['cases']:
        for arm in manifest['arms']:
            folder = output/case/arm
            if not (folder/'framework_generation.json').exists(): continue
            trace = json.loads((folder/'framework_generation.json').read_text())
            for c in trace['cycles'][0]['candidates']:
                native = c.get('committed_native_ledger') or {}
                calls = json.loads((folder/c['specialist']/'model_calls.json').read_text())
                authored = calls[-1].get('response') or {}
                if authored.get('choices'):
                    try:
                        authored = json.loads(authored['choices'][0]['text'])
                    except (ValueError, KeyError, IndexError, TypeError):
                        authored = {}
                work_record = {'origin': 'MODEL_AUTHORED_ARGUMENT', 'authority': 'ATTRIBUTED_NOT_WORLD_STATE',
                    'raw_fields': {k: authored.get(k) for k in ('ev', 'dp', 'cl', 'ep', 'dr', 'ft', 'nt', 'fic', 'foq')},
                    'native_submission': c.get('delegate_status'),
                    'native_ledger': native.get('transaction_status', 'NONE'),
                    'native_warnings': c.get('framework_validation_errors', []),
                    'semantic_support': 'NOT_INDEPENDENTLY_VERIFIED'}
                (folder/c['specialist']/'argument_work_record.json').write_text(json.dumps(work_record, indent=2)+'\n')
                rows.append({'case': case, 'arm': arm, 'framework': c['specialist'],
                    'adapter_calls': len(calls), 'submission': c.get('delegate_status'),
                    'ledger_status': native.get('transaction_status', 'NONE'),
                    'acceptance': c.get('selection_status'), 'preferred_action': c.get('recommended_action'),
                    'validation_errors': {k: v for k, v in c.items() if k.endswith('validation_errors') and v},
                    'rationale': c.get('rationale'), 'decision_rule': c.get('decision_rule'),
                    'argument': {k: c.get(k) for k in ('framework_application', 'framework_insights',
                        'reversal_condition', 'challenge_response', 'framework_action_map',
                        'factual_reversal_threshold', 'normative_reversal_threshold',
                        'expected_value_estimates', 'utilitarian_ranking_certificate',
                        'deontological_ledger_proposal', 'care_ledger_proposal',
                        'framework_internal_conflicts', 'framework_specific_open_questions')},
                    'native_record': str(folder/c['specialist']/'native_trace.json'),
                    'argument_work_record': str(folder/c['specialist']/'argument_work_record.json'),
                    'argument_quality': 'REQUIRES_CONTENT_REVIEW'})
    (output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    for case in manifest['cases']:
        standard, treatment = [output/case/a for a in manifest['arms']]
        if not all((p/'shared_input.json').exists() for p in (standard, treatment)): continue
        controls.append({'case': case,
            'shared_input_identical': (standard/'shared_input.json').read_bytes() == (treatment/'shared_input.json').read_bytes(),
            'framework_commitments_identical': (standard/'framework_commitments.json').read_bytes() == (treatment/'framework_commitments.json').read_bytes(),
            'treatment_sha256': fingerprint(CHECKLIST),
            'model_and_token_ceiling_equal': all(
                [(r['model'], r['max_tokens']) for r in json.loads((standard/f/'model_calls.json').read_text())] ==
                [(r['model'], r['max_tokens']) for r in json.loads((treatment/f/'model_calls.json').read_text())]
                for f in FRAMEWORKS if (standard/f/'model_calls.json').exists() and (treatment/f/'model_calls.json').exists()),
            'adapter_calls_equal': all(len(json.loads((standard/f/'model_calls.json').read_text())) ==
                 len(json.loads((treatment/f/'model_calls.json').read_text())) == 1
                 for f in FRAMEWORKS if (standard/f/'model_calls.json').exists() and (treatment/f/'model_calls.json').exists())})
    (output/'controls.json').write_text(json.dumps(controls, indent=2)+'\n')
    lines = ['# Matched argument-quality pilot', '',
        'Same model, source world, framework commitments, token ceiling and one adapter call per cell.',
        'Counterfactuals change only probabilities or add an explicit promise. No collective verdict is computed.', '',
        '| Case | Arm | Framework | Submission | Native ledger | Calls |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append('| '+' | '.join(str(r[k]) for k in ('case','arm','framework','submission','ledger_status','adapter_calls'))+' |')
    lines += ['', 'Content review must assess reproducible arithmetic, duty consistency, substantive objection answers,',
              'and explained counterfactual sensitivity. Ledger success alone earns no argument-quality credit.', '',
              'Full inspectable records: [results](results.json), [design and frozen inputs](manifest.json).']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('diagnostics/argument_quality_pilot'))
    parser.add_argument('--parliament-root', type=Path, default=Path('/tmp/parliament-smoke-614af0c'))
    parser.add_argument('--model', default='o3')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.parliament_root.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = prepare(args.output_dir)
    if args.execute:
        from dotenv import load_dotenv
        from global_workspace.openai_backend import OpenAIWorkspaceLLM
        from logic_puzzles_frameworks import generate
        load_dotenv('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env')
        backend = OpenAIWorkspaceLLM(args.model, timeout=120)
        for case, record in manifest['cases'].items():
            for arm in manifest['arms']:
                destination = args.output_dir/case/arm
                if (destination/'framework_generation.json').exists():
                    saved = json.loads((destination/'framework_generation.json').read_text())
                    if len(saved.get('framework_runs', [])) == len(FRAMEWORKS): continue
                    raise ValueError('Partial run exists; use a fresh output directory')
                print(case+' / '+arm, flush=True)
                chosen = ChecklistBackend(backend) if arm == 'argument_checklist' else backend
                generate(record['trace'], destination, FRAMEWORKS, lambda _: chosen,
                         call_budget=1, core_root=Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL'))
                summarize(args.output_dir, manifest)
    summarize(args.output_dir, manifest)
    print(str(args.output_dir/'RESULT.md'))


if __name__ == '__main__': main()
