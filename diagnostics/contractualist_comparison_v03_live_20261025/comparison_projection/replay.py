"""Post-hoc deterministic projection of the original 16 unmodified replies."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from complaint_comparison import derive_position_changes, VERSION

RUN = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    for path in sorted(RUN.glob('*/*/response.json')):
        raw = json.loads(path.read_text())
        frames = derive_position_changes(raw)
        case, framework = path.parent.parent.name, path.parent.name
        destination = OUT/(case + '__' + framework)
        payload = {'projection_version': VERSION, 'source_response': str(path.relative_to(RUN)),
                   'source_response_sha256': sha(path), 'provider_calls': 0,
                   'semantic_support': 'NOT_INDEPENDENTLY_ASSESSED',
                   'model_recommendation_unchanged': raw['recommendation_action_id'],
                   'frames': frames}
        destination.with_suffix('.json').write_text(json.dumps(payload, indent=2) + '\n')
        lines = ['# Comparison positions: ' + case + ' / ' + framework, '',
                 'Derived from the saved model matrix, not verified against source meaning. '
                 'Original model annotations and recommendation remain unchanged. Zero API calls.', '']
        for frame in frames:
            lines += ['## ' + frame['comparison_id'] + ': ' + frame['from_action_id'] + ' → ' + frame['to_action_id'], '',
                'Model-proposed ranking: ' + frame['proposed_comparison_status'], '',
                '| Person / burden | From | To | Derived change |', '| --- | --- | --- | --- |']
            for row in frame['positions']:
                values = [row['person_id'] + ' / ' + row['description'],
                          '; '.join(row['from']['descriptions']) or 'UNKNOWN',
                          '; '.join(row['to']['descriptions']) or 'UNKNOWN', row['change']]
                lines.append('| ' + ' | '.join(v.replace('|', '/').replace('\n', ' ') for v in values) + ' |')
            lines += ['']
        destination.with_suffix('.md').write_text('\n'.join(lines))
        records.append({'case': case, 'framework': framework, 'source_response_sha256': sha(path),
                        'projection_sha256': sha(destination.with_suffix('.json'))})
    manifest = {'provider_calls': 0, 'original_response_mutation': False,
                'timing': 'POST_HOC_AFTER_ALL_16_GENERATIONS', 'projection_version': VERSION,
                'code_sha256': {name: sha(ROOT/name) for name in ('complaint_comparison.py',)},
                'records': records}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Projected', len(records), 'saved replies; zero API calls; original responses untouched.')


if __name__ == '__main__':
    main()
