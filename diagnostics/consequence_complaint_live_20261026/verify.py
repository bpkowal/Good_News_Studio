"""Offline verification of the four-call consequence-first experiment."""
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
    record = read(folder/name)
    for filename, expected in record['files'].items():
        assert sha(folder/filename) == expected, (folder, filename)
    return record


def verify():
    frozen = receipt(ROOT, 'inputs.freeze.json')
    protocol = read(ROOT/'protocol.json')
    source = Path(protocol['source_probe'])
    receipt(source, 'inputs.freeze.json')
    for filename, expected in protocol['code_sha256'].items():
        assert sha(ROOT/'implementation'/filename) == expected
        assert sha(REPO/filename) == expected
    calls = []
    for case in protocol['cases']:
        folder = ROOT/case
        previous = source/case
        receipt(previous/'selector', 'outputs.freeze.json')
        origin = protocol['origins'][case]
        assert sha(previous/'packet.json') == origin['packet_sha256']
        assert sha(previous/'selector/result.json') == origin['assignment_result_sha256']
        assert sha(previous/'selector/outputs.freeze.json') == origin['assignment_output_receipt_sha256']
        packet = read(folder/'packet.json')
        assignment = read(folder/'assignment.json')
        assert packet == read(previous/'packet.json')
        assert assignment == read(previous/'selector/result.json')['bound_assignment']
        assert hashlib.sha256(packet['source']['scenario'].encode()).hexdigest() == packet['source']['source_text_sha256']
        record = read(folder/'consequence_record.json')
        assert record['source_text_sha256'] == packet['source']['source_text_sha256']
        for span in record['cited_source_spans']:
            for hit in span['occurrences']:
                assert packet['source']['scenario'][hit['start']:hit['end']] == span['text']
        requests = [read(folder/arm/'request.json') for arm in protocol['arms']]
        users = [json.loads(request['messages'][1]['content']) for request in requests]
        assert users[1].pop('consequence_record') == record
        assert users[0] == users[1] == {**packet, 'assigned_complaint': assignment}
        assert requests[1]['messages'][0]['content'] == (
            requests[0]['messages'][0]['content'] + '\n' + protocol['instructions']['refinement'])
        for key in ('model', 'reasoning_effort', 'max_completion_tokens', 'response_format'):
            assert requests[0][key] == requests[1][key]
        for arm in protocol['arms']:
            cell = folder/arm
            receipt(cell, 'outputs.freeze.json')
            call = read(cell/'call.json')
            assert frozen['frozen_at'] < call['started_at']
            assert call['provider_attempts'] == 1
            assert call['request_sha256'] == sha(cell/'request.json')
            provider = read(cell/'provider_response.json')
            assert provider['choices'][0]['finish_reason'] == 'stop'
            assert not provider['choices'][0]['message'].get('refusal')
            assert read(cell/'response.json') == json.loads(provider['choices'][0]['message']['content'])
            assert read(cell/'result.json')['status'] == 'RETURNED'
            calls.append({'case': case, 'arm': arm, 'model': provider['model'], 'finish_reason': 'stop',
                          'usage': provider['usage']})
    assert len(calls) == protocol['maximum_provider_calls'] == 4
    assert len(list(ROOT.glob('*/*/call.json'))) == 4
    assert protocol['selector_calls'] == 0 and protocol['sdk_retries'] == 0
    assert not any(protocol[key] for key in ('native_dispatch', 'world_mutation', 'peer_input'))
    results = read(ROOT/'results.json')
    return {'verification': 'PASS', 'calls': calls,
            'inputs_frozen_before_all_calls': True, 'original_arguments_assignments_and_source_unchanged': True,
            'same_source_prior_assignment_schema_model_effort_and_reply_ceiling': True,
            'treatment_difference': 'Additional consequence projection and fact-first instruction together.',
            'no_previous_arm_replies_assistant_review_or_peer_arguments_in_requests': True,
            'valid_source_span_offsets_do_not_establish_entailment': True,
            'binding_findings_by_reply': [{'case': row['case'], 'arm': row['arm'],
                                         'findings': row['binding_issues']} for row in results],
            'self_report_is_resolution': False, 'semantic_support': 'NOT_CERTIFIED_BY_THIS_VERIFIER',
            'operative': False,
            'regression_tests': {'passed': 113, 'note': 'Completed before live calls; not rerun by this verifier.'}}


if __name__ == '__main__':
    result = verify()
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'verification': result['verification'], 'provider_calls': len(result['calls'])}))
