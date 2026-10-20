"""Offline comparison against the backed-up cloze implementation; no model calls."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import blueprint_cloze_chooser as current

BASELINE = '5230184a08ea66c7befb095e14e5e1b3a7d60ea5'
before = types.ModuleType('cloze_before_compact_slots')
code = subprocess.check_output(['git', 'show', BASELINE + ':blueprint_cloze_chooser.py'], cwd=ROOT)
exec(compile(code, 'cloze_before_compact_slots', 'exec'), before.__dict__)


def compare(case, family, text, answers):
    a = before._score_template(text, before._BY_ID[family], answers, 0)
    b = current._score_template(text, current._BY_ID[family], answers, 0)
    return {
        'case': case, 'family': family,
        'before_status': a['status'], 'after_status': b['status'],
        'before_core': a['core_filled'], 'after_core': b['core_filled'],
        'changed_slots': {
            key: {'before': a['slots'].get(key), 'after': b['slots'].get(key)}
            for key in sorted(a['slots'].keys() | b['slots'].keys())
            if a['slots'].get(key) != b['slots'].get(key)
        },
    }


# Read only the data literal: the historical battery CLI has stale runtime imports.
variants = next(ast.literal_eval(node.value)
                for node in ast.parse((ROOT / 'run_blueprint_variants.py').read_text()).body
                if isinstance(node, ast.AnnAssign) and getattr(node.target, 'id', '') == 'VARIANTS')
rows = []
for case in variants:
    path = ROOT / 'diagnostics/blueprint_variant_battery' / (case['id'] + '_cloze.json')
    if not path.exists():
        continue
    for attempt in json.loads(path.read_text()).get('considered', []):
        family = attempt['blueprint_id']
        if family in current._BY_ID:
            rows.append(compare(case['id'], family, case['text'], {
                item['id']: item.get('completion', 'NONE') for item in attempt.get('items', [])
            }))
output = Path(__file__).resolve().parent
(output / 'saved_completion_comparison.json').write_text(json.dumps({
    'mode': 'OFFLINE_SAVED_COMPLETION_REPLAY', 'baseline_commit': BASELINE, 'api_calls': 0,
    'cases': len({row['case'] for row in rows}), 'template_attempts': len(rows),
    'core_regressions': [row for row in rows if row['after_core'] < row['before_core']],
    'attempts': rows,
}, indent=2) + '\n')

# Controlled copied-answer probes, explicitly not sampled provider responses.
probes = [
    ('risk_condition', 'uncertain_risk', 'If Ada opens the gate, Ben might drown.', {
        'actor': 'If Ada opens the gate, Ben might drown',
        'action': 'If Ada opens the gate, Ben might drown',
        'possible_outcome': 'If Ada opens the gate, Ben might drown',
        'affected_party': 'Ben', 'likelihood': 'might',
    }),
    ('rescue_condition', 'rescue_contrast',
     'Maria can save the child. If Maria saves the child, the child will live.', {
         'rescuer': 'Maria can save the child', 'first_saved': 'the child',
         'first_rescue_action': 'If Maria saves the child, the child will live',
         'first_benefit': 'If Maria saves the child, the child will live',
     }),
    ('promise_speaker', 'promise_reliance', 'Ada promised Ben that Ada would deliver medicine.', {
        'promisor': 'Ada promised Ben that Ada would deliver medicine',
        'promisee': 'Ben', 'commitment_event': 'promised',
        'commitment_content': 'that Ada would deliver medicine',
    }),
    ('diversion_patient', 'diversion_redirection',
     'Maria can divert the trolley. The trolley might kill three workers.', {
         'actor': 'Maria', 'intervention': 'divert the trolley',
         'controllable_process': 'The trolley might kill three workers',
         'affected_party': 'The trolley might kill three workers',
         'outcome': 'The trolley might kill three workers', 'uncertainty': 'might',
     }),
]
results = [compare(*probe) for probe in probes]
(output / 'controlled_span_comparison.json').write_text(json.dumps({
    'mode': 'CONTROLLED_COPIED_ANSWER_PROBES', 'baseline_commit': BASELINE,
    'api_calls': 0, 'attempts': results,
}, indent=2) + '\n')
print(f'{len(rows)} saved attempts; {len(probes)} controlled probes; zero API calls.')
