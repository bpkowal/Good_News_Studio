"""First Logic_Puzzles increment: inspect native claims before changing debate."""
import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path


def _id(kind, value):
    return kind + ':' + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def project_conflicts(trace):
    world = trace['action_source_grounding']['world_model']
    if len(world['actions']) > 2:
        raise ValueError('Logic_Puzzles pilot supports at most two actions')
    nodes, edges, diagnostics = {}, [], []
    def node(ident, kind, **attrs):
        nodes[ident] = {'id': ident, 'kind': kind, **deepcopy(attrs)}
    def edge(source, target, kind, **attrs):
        edges.append({'source': source, 'target': target, 'kind': kind, **attrs})
    for effect in world['effects']:
        node('WORLD:' + effect['effect_id'], 'WORLD_EFFECT', record=effect,
             epistemic_status='ADMITTED', label=effect['outcome'])
    for proposition in trace.get('proposition_ledger', []):
        node(proposition['proposition_id'], 'PROPOSITION', record=proposition,
             epistemic_status=proposition['epistemic_status'], label=proposition['claim'])
    scoped_dependencies = {}
    for origin, propositions in trace.get('framework_proposition_ledgers', {}).items():
        for proposition in propositions:
            local_id = proposition['proposition_id']
            # Generated premises from isolated runs must not alias one another.
            if local_id not in nodes or nodes[local_id]['record'] != proposition:
                ident = origin + '::' + local_id
                node(ident, 'PROPOSITION', origin=origin, native_proposition_id=local_id,
                     record=proposition, epistemic_status=proposition['epistemic_status'],
                     label=proposition['claim'])
                scoped_dependencies[(origin, local_id)] = ident
    candidates = trace.get('cycles', [])[-1]['candidates'] if trace.get('cycles') else []
    issues = []
    for proposal in trace.get('targeted_review', {}).get('retained_proposals', []):
        ident = _id('PROPOSED_REVISION', proposal)
        node(ident, 'PROPOSED_REVISION', origin=proposal['framework'],
             status=proposal['status'], world_state_authority='NONE',
             semantic_support='NOT_INDEPENDENTLY_VERIFIED', record=proposal,
             label=proposal['framework'] + ': disputed full revision; not operative')
    for candidate in candidates:
        origin = candidate['specialist']
        ledger = candidate.get('committed_native_ledger') or {}
        if not str(ledger.get('transaction_status', '')).startswith('COMMITTED'):
            diagnostics.append({'specialist': origin, 'status': 'NO_COMMITTED_LEDGER',
                                'candidate': deepcopy(candidate)})
            continue
        claim_ids = []
        for index, record in enumerate(ledger.get('records', [])):
            ident = _id('CLAIM', [origin, ledger['ledger_kind'], index, record])
            claim_ids.append(ident)
            node(ident, 'CLAIM', origin=origin, native_ledger_kind=ledger['ledger_kind'],
                 native_record_id=record.get('assessment_node_id'), record=record,
                 action_id=record.get('canonical_action_id'),
                 epistemic_status=record.get('epistemic_status', 'NOT_ASSESSED'),
                 label=origin + ': ' + str(record.get('canonical_action_id')) + ' ' + str(record.get('verdict', 'assessment')))
            # An effect reference supports only the referenced consequence,
            # not every relationship or normative inference in the record.
            for effect_id in record.get('grounded_effect_ids', []):
                target = 'WORLD:' + effect_id
                if target in nodes:
                    edge(target, ident, 'SUPPORT', extent='REFERENCED_EFFECT_ONLY')
                else:
                    diagnostics.append({'claim_id': ident, 'unresolved_effect_id': effect_id})
            for dependency in candidate.get('decision_critical_proposition_ids', []):
                target = scoped_dependencies.get((origin, dependency), dependency)
                if target in nodes:
                    edge(ident, target, 'DEPENDENCY', scope='CANDIDATE_LEVEL',
                         epistemic_status=nodes[target]['epistemic_status'])
                else:
                    diagnostics.append({'claim_id': ident, 'unresolved_dependency_id': dependency})
            for field in ('norm', 'priority_rule', 'responsibility_basis', 'ranking_basis'):
                if record.get(field):
                    commitment = _id('COMMITMENT', [origin, field, record[field]])
                    node(commitment, 'FRAMEWORK_COMMITMENT', origin=origin, field=field,
                         value=record[field], label=record[field], authority='FRAMEWORK_LOCAL')
                    edge(commitment, ident, 'SUPPORT', extent='NORMATIVE_INTERPRETATION')
            for error in record.get('calibration_errors', []):
                conflict = _id('CONFLICT', [ident, error])
                node(conflict, 'UNRESOLVED_CONFLICT', origin=origin, label=error,
                     conflict_type='UNSUPPORTED_INFERENCE', status='OPEN',
                     source='NATIVE_LEDGER_CALIBRATION', claim_ids=[ident])
                edge(conflict, ident, 'OBJECTION', target_kind='INFERENCE')
                issues.append(conflict)
        for conflict_text in candidate.get('framework_internal_conflicts', []):
            conflict = _id('CONFLICT', [origin, conflict_text])
            node(conflict, 'UNRESOLVED_CONFLICT', origin=origin, label=conflict_text,
                 conflict_type='REPORTED_INTERNAL_CONFLICT', status='OPEN',
                 source='FRAMEWORK_REPORT', claim_ids=claim_ids,
                 localization_status='CANDIDATE_LEVEL_NOT_PRECISE_CLAIM_COLLISION')
            issues.append(conflict)
        for warning in candidate.get('framework_validation_errors', []):
            conflict = _id('CONFLICT', [origin, 'native_framework_warning', warning])
            node(conflict, 'UNRESOLVED_CONFLICT', origin=origin, label=warning,
                 conflict_type='NATIVE_FRAMEWORK_WARNING', status='OPEN',
                 source='NATIVE_FRAMEWORK_VALIDATION', claim_ids=claim_ids,
                 localization_status='CANDIDATE_LEVEL_NOT_PRECISE_CLAIM_COLLISION')
            issues.append(conflict)
        condition = candidate.get('reversal_condition')
        if condition and condition != 'NONE':
            ident = _id('REVERSAL', [origin, condition])
            node(ident, 'REVERSAL_CONDITION', origin=origin, label=condition,
                 claim_ids=claim_ids, authority='UNVERIFIED_BOUNDARY')
    return {'version': 'logic-puzzles-conflict-projection/0.1',
            'mode': 'READ_ONLY_PROJECTION', 'world_mutation': False,
            'source_judgment_status': trace.get('judgment_status'),
            'nodes': list(nodes.values()), 'edges': edges, 'diagnostics': diagnostics,
            'review_queue': sorted(dict.fromkeys(issues), key=lambda ident: {
                'UNSUPPORTED_INFERENCE': 0, 'REPORTED_INTERNAL_CONFLICT': 1,
                'NATIVE_FRAMEWORK_WARNING': 2}.get(nodes[ident]['conflict_type'], 3))[:2],
            'limits': [('Targeted native revisions are projected; no collective judgment.' if trace.get('targeted_review') else
                        'Independent assessments are projected; no targeted review executed.' if trace.get('framework_runs') else
                        'No independent model pass or targeted review executed.'),
                       'Different preferences do not establish a contradiction.',
                       'Native record commitment is not proof of every asserted premise.']}


def write_graph(graph, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'conflict_graph.json').write_text(json.dumps(graph, indent=2)+'\n')
    aliases = {n['id']: 'n'+str(i) for i,n in enumerate(graph['nodes'])}
    lines = ['# Logic_Puzzles claim/conflict projection', '',
             'Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.', '',
             '```mermaid', 'flowchart LR']
    for n in graph['nodes']:
        label = html.escape(str(n.get('label', n['id'])), quote=True).replace('\n', ' ')
        lines.append(f'    {aliases[n["id"]]}["{n["kind"]}: {label}"]')
    for e in graph['edges']:
        lines.append(f'    {aliases[e["source"]]} -->|"{e["kind"]}"| {aliases[e["target"]]}')
    lines += ['```', '', 'Proposed review queue (at most two issues):', '']
    by_id = {n['id']: n for n in graph['nodes']}
    lines += ['- '+by_id[i]['label'] for i in graph['review_queue']]
    (output_dir/'conflict_graph.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('diagnostics/logic_puzzles_projection'))
    parser.add_argument('--generate-frameworks', action='store_true')
    parser.add_argument('--targeted-review', action='store_true')
    parser.add_argument('--matched-comparison', action='store_true', help='Use --trace parent as saved opening directory')
    parser.add_argument('--frameworks', nargs='+', default=['utilitarian', 'deontological', 'care', 'virtue', 'rawlsian'])
    parser.add_argument('--model', default='o3')
    parser.add_argument('--parliament-root', type=Path, default=Path('/tmp/parliament-smoke-614af0c'))
    parser.add_argument('--parliament-python', type=Path, default=Path('/tmp/parliament-smoke-env/bin/python'))
    parser.add_argument('--core-root', type=Path)
    args = parser.parse_args()
    if sum((args.targeted_review, args.generate_frameworks, args.matched_comparison)) > 1:
        parser.error('Choose generation, review, or comparison for this invocation')
    if args.matched_comparison:
        import subprocess
        raise SystemExit(subprocess.call([
            str(args.parliament_python), str(Path(__file__).with_name('logic_puzzles_comparison.py')),
            '--opening-dir', str(args.trace.parent), '--output-dir', str(args.output_dir),
            '--parliament-root', str(args.parliament_root), '--model', args.model]))
    if args.targeted_review:
        import subprocess
        raise SystemExit(subprocess.call([
            str(args.parliament_python), str(Path(__file__).with_name('logic_puzzles_review.py')),
            '--trace', str(args.trace), '--output-dir', str(args.output_dir),
            '--parliament-root', str(args.parliament_root), '--model', args.model]))
    if args.generate_frameworks:
        import subprocess
        command = [str(args.parliament_python), str(Path(__file__).with_name('logic_puzzles_frameworks.py')),
                   '--trace', str(args.trace), '--output-dir', str(args.output_dir),
                   '--parliament-root', str(args.parliament_root), '--model', args.model,
                   '--frameworks', *args.frameworks]
        if args.core_root:
            command += ['--core-root', str(args.core_root)]
        raise SystemExit(subprocess.call(command))
    graph = project_conflicts(json.loads(args.trace.read_text()))
    write_graph(graph, args.output_dir)
    print(f'{len(graph["nodes"])} nodes; {len(graph["edges"])} edges; {len(graph["review_queue"])} proposed review issues. No model calls.')


if __name__ == '__main__':
    main()
