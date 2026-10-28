"""Offline receipts for the critical-question increment and unchanged old runs."""
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
    run = ROOT/'live_blanket'
    inputs = receipt(run, 'inputs.freeze.json')
    protocol = read(run/'protocol.json')
    for filename, expected in protocol['code_sha256'].items():
        assert sha(run/'implementation'/filename) == expected
        assert sha(REPO/filename) == expected
    cell = run/'blanket_prior_assurance/contractualist'
    receipt(cell, 'output.freeze.json')
    call = read(cell/'call.json')
    assert inputs['frozen_at'] < call['started_at']
    assert call['provider_attempts'] == 1
    assert call['request_sha256'] == sha(cell/'request.json')
    request = read(cell/'request.json')
    payload = json.loads(request['messages'][1]['content'])
    assert set(payload) == {'framework_id', 'framework_instruction', 'source'}
    assert payload['source'] == read(cell/'source.json')
    assert set(payload['source']) == {'id', 'scenario', 'actions', 'source_text_sha256', 'source_status', 'evidence'}
    assert 'critical_questions' in request['response_format']['json_schema']['schema']['properties']
    provider = read(cell/'provider_response.json')
    assert provider['choices'][0]['finish_reason'] == 'stop'
    assert not provider['choices'][0]['message'].get('refusal')
    raw = read(cell/'response.json')
    assert raw == json.loads(provider['choices'][0]['message']['content'])
    result = read(cell/'result.json')
    assert raw == result['raw_response']
    assert protocol['maximum_provider_calls'] == 1
    assert len(list(run.glob('*/*/call.json'))) == 1
    assert protocol['sdk_retries'] == 0
    assert not any(protocol[key] for key in ('world_admission', 'peer_input', 'repairs'))
    old = REPO/'diagnostics/contractualist_comparison_v03_live_20261025'
    receipt(old, 'inputs.freeze.json')
    old_cells = list(old.glob('*/*/output.freeze.json'))
    assert len(old_cells) == 16
    for output in old_cells:
        receipt(output.parent, output.name)
    for dirname in ('opposing_complaint_live_20261025', 'consequence_complaint_live_20261026'):
        previous = REPO/'diagnostics'/dirname
        receipt(previous, 'inputs.freeze.json')
        for output in previous.glob('*/*/outputs.freeze.json'):
            receipt(output.parent, output.name)
    # Old implementation snapshots remain valid history. They need not match
    # the current checkout after an intentional new implementation version.
    return {'verification': 'PASS', 'provider_calls': 1, 'model': provider['model'],
        'finish_reason': provider['choices'][0]['finish_reason'],
        'current_version': protocol['version'], 'instruction_version': protocol['instruction_version'],
        'recommendation_action_id': raw['recommendation_action_id'],
        'recommendation_status': raw['recommendation_status'],
        'critical_question_count': len(raw['critical_questions']),
        'technical_status': result['status'], 'binding_issue_count': len(result['issues']),
        'comparison_finding_count': len(result['comparison_review']['findings']),
        'question_findings': result['critical_question_review']['findings'],
        'source_and_old_output_receipts_valid': True,
        'no_previous_replies_review_notes_or_intended_winner_in_payload': True,
        'semantic_support': 'NOT_CERTIFIED_BY_THIS_VERIFIER', 'operative': False,
        'regression_tests': {'passed': 126, 'note': 'Completed before live call; not rerun by this verifier.'}}


if __name__ == '__main__':
    result = verify()
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
