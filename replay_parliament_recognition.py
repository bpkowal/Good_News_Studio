"""Replay recognition and accounting against saved inputs; never call a model."""
import argparse
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parliament-root', type=Path, default=Path('/tmp/parliament-smoke-614af0c'))
    parser.add_argument('--input-dir', type=Path, default=Path('diagnostics/live_pipeline_latest_20261006/medicine'))
    parser.add_argument('--output-dir', type=Path, default=Path('diagnostics/recognition_gap_saved_replay'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.parliament_root))
    from global_workspace.scenario_semantics import compile_scenario_graph
    from global_workspace.epistemic_ledger import seed_proposition_ledger, attach_candidate_dependencies, _record_from_mapping
    from global_workspace.models import CandidateChunk
    from global_workspace.utilitarian_ledger import utilitarian_accounting
    from global_workspace.recognition_bridge import component_support

    frozen_path = args.input_dir / 'frozen_world_trace.json'
    trace_path = next((args.input_dir / 'parliament').glob('workspace*.json'))
    frozen_bytes, trace_bytes = frozen_path.read_bytes(), trace_path.read_bytes()
    frozen, trace = json.loads(frozen_bytes), json.loads(trace_bytes)
    world = frozen['action_source_grounding']['world_model']
    graph = compile_scenario_graph(frozen['scenario'], frozen['actions'], world_model=world)
    ledger = seed_proposition_ledger(graph)
    cards = graph.discourse_collections.get('recognition_records', [])
    # Retain original hypotheses to test whether supported source rules take
    # precedence, rather than making old unestablished dependencies disappear.
    for row in trace['proposition_ledger']:
        if row['proposition_type'] == 'HYPOTHESIS':
            ledger[row['proposition_id']] = _record_from_mapping(row)
    claims = []
    allowed = {f.name for f in fields(CandidateChunk)}
    for old in trace['cycles'][-1]['candidates']:
        candidate = CandidateChunk(**{k: deepcopy(v) for k, v in old.items() if k in allowed})
        attach_candidate_dependencies(ledger, candidate)
        for row in candidate.material_empirical_claims:
            old_row = next((r for r in old['material_empirical_claims'] if r['claim'] == row['claim']), {})
            ident = row.get('canonical_proposition') or row.get('proposition_id')
            claims.append({'framework': candidate.specialist, 'claim': row['claim'],
                'before_id': old_row.get('canonical_proposition'), 'after_id': ident,
                'after_status': ledger[ident].epistemic_status if ident in ledger else 'UNAVAILABLE',
                'components': row.get('recognized_components', component_support(row['claim'], cards))})
    accounting = []
    for node in graph.nodes.values():
        if node.kind != 'CONSEQUENCE':continue
        eid = node.attributes.get('world_effect_id')
        if not node.attributes.get('likelihood_qualifiers'):continue
        old = next((r for r in trace['utilitarian_consequence_ledger'] if r.get('world_effect_id') == eid), {})
        direction, probability = utilitarian_accounting(graph, node.id)
        accounting.append({'effect_id': eid, 'outcome': node.label, 'before_probability': old.get('probability'),
            'after_probability': probability, 'modality': node.attributes['modality'],
            'direction': direction, 'qualifiers': node.attributes['likelihood_qualifiers']})
    assert frozen_bytes == frozen_path.read_bytes() and trace_bytes == trace_path.read_bytes()
    result = {'new_api_calls': 0, 'world_unchanged': True, 'judgment_recomputed': False,
        'source_sha256': hashlib.sha256(frozen_bytes).hexdigest(),
        'recognition_records': cards, 'accounting': accounting, 'saved_claim_rebindings': claims,
        'limits': ['Recognition/accounting replay only; no new normative verdict.',
                   'A partially supported compound remains hypothetical.',
                   'Numeric probability does not invent welfare magnitude or duration.']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Recognition evidence graph', '', 'Cards are source rules/constraints, not new events.', '', '```mermaid', 'flowchart LR']
    for index, card in enumerate(cards):
        label = card['claim'].replace('"', '&quot;')
        lines.append(f'  c{index}["{card["kind"]}: {label}"]')
        for ident in card['support_ids']:
            lines.append(f'  c{index} -->|supported by| {ident}')
    lines.append('```')
    (args.output_dir / 'recognition_graph.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'accounting': accounting, 'cards': len(cards),
        'changed_bindings': [c for c in claims if c['before_id'] != c['after_id']],
        'world_unchanged': True, 'new_api_calls': 0}, indent=2))


if __name__ == '__main__':main()
