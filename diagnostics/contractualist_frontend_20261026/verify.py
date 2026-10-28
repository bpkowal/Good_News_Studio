"""Verify the saved five-perspective run; optional read-only localhost checks."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def read(path):
    return json.loads(path.read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class BrowserPage(HTMLParser):
    def __init__(self):
        super().__init__(); self.frameworks = []; self.questions = 0; self.graph_nodes = 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs); classes = attrs.get('class', '').split()
        if tag == 'article' and attrs.get('data-framework'):
            self.frameworks.append(attrs['data-framework'])
        if 'critical-question' in classes:
            self.questions += 1
        if tag == 'g' and 'node' in classes:
            self.graph_nodes += 1


def receipt(folder, filename):
    for name, expected in read(folder/filename)['files'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected, name


def verify(http=False):
    report = read(ROOT/'report/DEMO_REPORT.json')
    trace_path = Path(report['artifacts'][0])
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == report['execution']['frozen_trace_sha256']
    frozen = read(trace_path); world = frozen['action_source_grounding']['world_model']
    assert report['scenario'] == frozen['scenario'] and report['world'] == world
    native = read(ROOT/'frameworks/framework_generation.json')
    shared = read(ROOT/'frameworks/shared_input.json')
    assert native['action_source_grounding']['world_model'] == world
    assert native['shared_input_sha256'] == fingerprint(shared)
    names = ['utilitarian', 'deontological', 'care', 'virtue']
    calls = {}
    for name in names:
        rows = read(ROOT/'frameworks'/name/'model_calls.json')
        assert all(row['status'] == 'RETURNED' for row in rows)
        assert read(ROOT/'frameworks'/name/'native_trace.json')['action_source_grounding']['world_model'] == world
        calls[name] = len(rows)
    source = read(ROOT/'contractualist/source.json')
    request = read(ROOT/'contractualist/request.json')
    user = json.loads(request['messages'][1]['content'])
    assert user['source'] == source
    assert source['admitted_world'] == world and source['scenario'] == frozen['scenario']
    assert source['world_sha256'] == fingerprint(world)
    assert {a['id'] for a in source['actions']} == {a['action_id'] for a in world['actions']}
    assert 'peer_arguments' not in user
    receipt(ROOT/'contractualist', 'inputs.freeze.json')
    receipt(ROOT/'contractualist', 'output.freeze.json')
    assert read(ROOT/'contractualist/call.json')['provider_attempts'] == 1
    provider = read(ROOT/'contractualist/provider_response.json')
    assert provider['choices'][0]['finish_reason'] == 'stop'
    raw = read(ROOT/'contractualist/response.json')
    assert raw == json.loads(provider['choices'][0]['message']['content'])
    contract_view = report['frameworks'][-1]
    assert contract_view['recorded_argument'] == raw
    assert contract_view['critical_questions'] == raw['critical_questions']
    assert contract_view['ledger_status'] == 'NOT_APPLICABLE_ARGUMENT_ADAPTER'
    assert [v['framework'] for v in report['frameworks']] == names + ['contractualist']
    assert all(a['framework'] not in ('rawlsian', 'contractualist') for a in report['topography']['framework_arrows'])
    payload = read(REPO/'frontend/dist/data/thermal_blanket_contractualist.json')
    assert payload['source'] == frozen['scenario'] and payload['diagrams'][0]['world'] == world
    assert payload['frameworks'][-1]['recorded_argument'] == raw
    assert '/Users/' not in json.dumps(payload)
    historical = read(REPO/'diagnostics/thermal_blanket_end_to_end_20261023/report/DEMO_REPORT.json')
    assert any(v['framework'] == 'rawlsian' for v in historical['frameworks'])
    parent = read(ROOT/'run_manifest.json')['run_id']
    for folder in ('frameworks', 'contractualist'):
        assert read(ROOT/folder/'run_manifest.json')['parent_run_id'] == parent
    browser = BrowserPage(); browser.feed((ROOT/'browser_overview_dom.html').read_text())
    assert browser.frameworks == names + ['contractualist'] and browser.questions == len(raw['critical_questions'])
    assert browser.graph_nodes > 0
    if http:
        with urllib.request.urlopen('http://127.0.0.1:7860/data/thermal_blanket_contractualist.json', timeout=10) as response:
            assert json.load(response) == payload
        for view in payload['frameworks']:
            with urllib.request.urlopen('http://127.0.0.1:7860/' + view['argument_url'], timeout=10) as response:
                assert response.status == 200
    return {'verification': 'PASS', 'recorded_native_adapter_calls': calls,
        'contractualist_provider_calls': 1, 'recorded_adapter_call_total': sum(calls.values()) + 1,
        'same_exact_source_and_frozen_world': True, 'input_trace_unchanged': True,
        'rawlsian_history_preserved': True, 'no_contractualist_native_ledger_or_coordinate_invented': True,
        'child_run_lineage_matches': True, 'browser_framework_cards': browser.frameworks,
        'browser_question_count': browser.questions, 'mermaid_node_count': browser.graph_nodes,
        'http_payload_and_five_argument_links': 'PASS' if http else 'NOT_CHECKED',
        'semantic_support': 'NOT_CERTIFIED_BY_THIS_VERIFIER',
        'tests': {'demo_adapter_report_viewer_presentation_and_topography': 99,
                  'note': 'Passed before live calls; a later focused subset of 38 also passed. This verifier does not rerun tests.'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--http', action='store_true')
    result = verify(parser.parse_args().http)
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
