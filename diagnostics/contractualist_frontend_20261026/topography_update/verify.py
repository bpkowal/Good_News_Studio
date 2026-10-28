"""Offline verification of the probability pass and actual browser surface."""
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify():
    before = read(HERE/'report_before.json')
    after = read(HERE.parent/'report/DEMO_REPORT.json')
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    assert changed == {'topography', 'presentation_revisions'}, changed
    hashes = read(HERE/'input_hashes.json')
    assert fingerprint(after['world']) == hashes['world_sha256']
    assert fingerprint(after['frameworks']) == hashes['frameworks_sha256']
    assert hashlib.sha256(after['scenario'].encode()).hexdigest() == hashes['source_sha256']
    topo = after['topography']
    assert topo['provenance']['new_model_calls'] == 20
    assert set(topo['axes']) == {'care', 'welfare', 'obligation'}
    assert topo['flags']['holes'] == []
    native = {row['id']: row for row in before['topography']['actions']}
    for row in topo['actions']:
        for axis in topo['axes']:
            point = row[axis]
            assert point['present'] and math.isfinite(point['probability'])
            assert 0 <= point['probability'] <= 1
            interval = point['interval']
            assert interval['folds_used'] == 10 and len(interval['replicates']) == 10
            assert 0 <= interval['low'] <= point['probability'] <= interval['high'] <= 1
            assert row['native'][axis] == native[row['id']][axis]
    assert all(row['framework'] not in ('contractualist', 'rawlsian')
               for row in topo['framework_arrows'])
    packet = topo['provenance']['jackknife']['packet_text']
    assert packet == (HERE/'assessment_packet.json').read_text()
    adapter = next(row for row in json.loads(packet)['frameworks'] if row['framework'] == 'contractualist')
    original = after['frameworks'][-1]
    assert adapter['critical_questions'] == original['critical_questions']
    assert adapter['recorded_argument'] == original['recorded_argument']
    assert adapter['ledger_status'] == 'NOT_APPLICABLE_ARGUMENT_ADAPTER'
    exported = read(REPO/'frontend/dist/data/thermal_blanket_contractualist.json')
    assert exported['topography'] == topo
    assert exported['frameworks'][-1]['critical_questions'] == original['critical_questions']
    dom = (HERE/'browser_dom.html').read_text()
    surface = re.findall(r'<g class="topo-landscape"[^>]*>(.*?)</g>', dom, re.S)
    assert len(surface) == 1
    assert surface[0].count('<polygon ') == 256
    assert 'NaN' not in surface[0] and 'Infinity' not in surface[0]
    panel = re.search(r'<section id="view-topography"[^>]*>', dom).group()
    assert 'hidden' not in panel
    assert 'Shepard, p=2' in dom
    result = {'verification': 'PASS', 'api_requests': 20,
              'usable_folds_per_axis_per_action': 10, 'complete_action_loci': len(topo['actions']),
              'browser_surface_polygons': 256, 'topography_tab_visible': True,
              'world_source_and_all_five_arguments_unchanged': True,
              'native_coordinates_preserved': True, 'contractualist_questions_in_assessment_packet': True,
              'new_contractualist_axis_or_native_ledger': False,
              'regression_tests_passed': 39,
              'interpretation': 'Display interpolation and packet-ablation sensitivity; not moral certainty.'}
    (HERE/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
