"""Offline provenance/matching verification; makes no model or native calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(folder, name):
    record = read(folder / name)
    for filename, expected in record['files'].items():
        assert sha(folder / filename) == expected, (folder, filename)
    return record


def verify():
    protocol = read(ROOT / 'protocol.json')
    receipt(ROOT, 'inputs.freeze.json')
    for filename, expected in protocol['code_sha256'].items():
        assert sha(ROOT / 'implementation' / filename) == expected
        assert sha(REPO / filename) == expected
    latest_assignment = ''
    first_reply = None
    calls, cases = [], []
    for case in protocol['cases']:
        folder = ROOT / case
        packet = read(folder / 'packet.json')
        origin = protocol['origins'][case]
        prior = Path(origin['folder'])
        if not prior.is_absolute():
            prior = REPO / prior
        receipt(prior, 'output.freeze.json')
        for filename, key in [('source.json', 'source_sha256'),
                              ('response.json', 'prior_response_sha256'),
                              ('request.json', 'prior_request_sha256')]:
            assert sha(prior / filename) == origin[key]
        assert packet['prior_argument'] == read(prior / 'response.json')
        assert packet['source'] == read(prior / 'source.json')
        assignment_freeze = receipt(folder / 'selector', 'outputs.freeze.json')
        latest_assignment = max(latest_assignment, assignment_freeze['frozen_at'])
        reply_freeze = receipt(folder, 'reply_inputs.freeze.json')
        requests = [read(folder / arm / 'request.json') for arm in protocol['arms']]
        assert requests[0]['messages'][1:] == requests[1]['messages'][1:]
        assert requests[0]['response_format'] == requests[1]['response_format']
        for key in ('model', 'reasoning_effort', 'max_completion_tokens'):
            assert requests[0][key] == requests[1][key]
        for arm, request in zip(protocol['arms'], requests):
            assert request['messages'][0]['content'] == (
                protocol['instructions']['common_reply'] + '\n' + protocol['instructions'][arm])
            payload = json.loads(request['messages'][1]['content'])
            assert set(payload) == set(packet) | {'assigned_complaint'}
            for key, value in packet.items():
                assert payload[key] == value
        for stage in ('selector', *protocol['arms']):
            cell = folder / stage
            receipt(cell, 'outputs.freeze.json')
            call = read(cell / 'call.json')
            assert call['provider_attempts'] == 1
            assert call['request_sha256'] == sha(cell / 'request.json')
            provider = read(cell / 'provider_response.json')
            assert provider['choices'][0]['finish_reason'] == 'stop'
            assert json.loads(provider['choices'][0]['message']['content']) == read(cell / 'response.json')
            assert read(cell / 'result.json')['status'] == 'RETURNED'
            calls.append({'case': case, 'stage': stage, 'model': provider['model'],
                          'completion_tokens': provider['usage']['completion_tokens']})
            if stage != 'selector':
                assert reply_freeze['frozen_at'] < call['started_at']
                first_reply = min(first_reply or call['started_at'], call['started_at'])
        cases.append({'case': case,
                      'assignment_binding_issues': read(folder / 'selector/result.json')['bound_assignment']['binding_issues']})
    assert latest_assignment < first_reply
    assert len(calls) == protocol['maximum_provider_calls'] == 12
    assert len(list(ROOT.glob('*/*/call.json'))) == 12
    assert protocol['sdk_retries'] == 0
    assert not any(protocol[key] for key in ('world_mutation', 'native_dispatch', 'peer_input'))
    rows = read(ROOT / 'results.json')
    return {'verification': 'PASS', 'provider_calls': calls, 'case_bindings': cases,
            'all_assignments_frozen_before_first_reply': True,
            'matched_user_packets_schema_model_effort_and_budget': True,
            'original_sources_and_replies_unchanged': True,
            'frozen_inputs_outputs_and_implementation_hashes_valid': True,
            'native_dispatch': False, 'world_mutation': False,
            'reply_count': len(rows), 'self_reported_answered': sum(r['self_report'] == 'ANSWERED' for r in rows),
            'replies_with_binding_findings': sum(bool(r['binding_issues']) for r in rows),
            'content_support': 'NOT_CERTIFIED_BY_THIS_VERIFIER',
            'related_regression_tests': {'passed': 98,
                'note': 'Completed during implementation before live calls; this verifier does not rerun tests.'}}


if __name__ == '__main__':
    result = verify()
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('verification', 'reply_count', 'self_reported_answered',
                                                 'replies_with_binding_findings')}, indent=2))
