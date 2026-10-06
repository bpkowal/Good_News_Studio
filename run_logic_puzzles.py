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
    candidates = trace.get('cycles', [])[-1]['candidates'] if trace.get('cycles') else []
    issues = []
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
                if dependency in nodes:
                    edge(ident, dependency, 'DEPENDENCY', scope='CANDIDATE_LEVEL',
                         epistemic_status=nodes[dependency]['epistemic_status'])
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
        condition = candidate.get('reversal_condition')
        if condition and condition != 'NONE':
            ident = _id('REVERSAL', [origin, condition])
            node(ident, 'REVERSAL_CONDITION', origin=origin, label=condition,
                 claim_ids=claim_ids, authority='UNVERIFIED_BOUNDARY')
    return {'version': 'logic-puzzles-conflict-projection/0.1',
            'mode': 'READ_ONLY_PROJECTION', 'world_mutation': False,
            'source_judgment_status': trace.get('judgment_status'),
            'nodes': list(nodes.values()), 'edges': edges, 'diagnostics': diagnostics,
            'review_queue': list(dict.fromkeys(issues))[:2],
            'limits': ['No independent model pass or targeted review executed.',
                       'Different preferences do not establish a contradiction.',
                       'Native record commitment is not proof of every asserted premise.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('diagnostics/logic_puzzles_projection'))
    args = parser.parse_args()
    graph = project_conflicts(json.loads(args.trace.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'conflict_graph.json').write_text(json.dumps(graph, indent=2)+'\n')
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
    (args.output_dir/'conflict_graph.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(graph["nodes"])} nodes; {len(graph["edges"])} edges; {len(graph["review_queue"])} proposed review issues. No model calls.')


if __name__ == '__main__':
    main()
