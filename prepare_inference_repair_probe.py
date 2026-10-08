"""Prepare and audit a saved repair fixture. This program has no provider-call path."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

from inference_repair import make_contract, annotated_saved_repair, check_submission, repair_schema
from run_objection_response_pilot import decode, write


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists(): raise ValueError('Use a fresh output directory; existing diagnostics are immutable')
    call = json.loads((source/'revision/utilitarian/revision_call.json').read_text())
    raw = decode(call['response'])
    frozen = json.loads((source/'frozen_world_trace.json').read_text())
    quotes = json.loads((source/'revision/framework_commitments.json').read_text())
    contract = make_contract(raw['native_submission'], frozen, quotes)
    annotated = annotated_saved_repair(contract)
    annotated['reversal_condition']['text'] = raw['native_submission']['ft']
    original_audit = check_submission(contract, None, raw['native_submission'], frozen, quotes)
    annotated_audit = check_submission(contract, annotated, raw['native_submission'], frozen, quotes)
    cautious = deepcopy(raw['native_submission'])
    cautious.update(ss='PROVISIONAL', ia='NONE', cc=False, esa=False, am={'A0': 'UNASSESSED', 'A1': 'UNASSESSED'},
        x='Promise costs are unknown; no common welfare conversion is established', ft='NONE',
        w='Restricted survivors favour A0', da='Total welfare unresolved', t='Restricted count is not total welfare',
        dr='A0 preference conditional on a justified welfare comparison',
        l={'A0': 'Higher restricted expected survivor count', 'A1': 'Honours promise; welfare comparison unresolved'})
    example = annotated_saved_repair(contract)
    write(output/'repair_contract.json', contract)
    write(output/'repair_schema.json', repair_schema())
    write(output/'frozen_world_trace.json', frozen)
    write(output/'framework_commitments.json', quotes)
    write(output/'saved_reply.json', raw)
    write(output/'annotated_fixture.json', {'origin': 'REVIEWER_TEST_FIXTURE_NOT_MODEL_OUTPUT', 'record': annotated})
    write(output/'prepared_replay_packet.json', {'execute': False, 'provider_calls': 0,
        'challenge': 'Separate supported survivors from unresolved welfare. Choose exactly one repair operation.',
        'previous_reply': raw, 'repair_contract': contract,
        'example_only': {'origin': 'REVIEWER_TEST_FIXTURE_NOT_MODEL_OUTPUT', 'inference_repair': example, 'native_submission': cautious}})
    result = {'provider_calls': 0, 'saved_reply_without_record': original_audit,
              'annotated_saved_reply': annotated_audit,
              'cautious_example': check_submission(contract, example, cautious, frozen, quotes)}
    write(output/'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('diagnostics/specific_inference_probe_live_20261007'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir)
    print(args.output_dir/'audit.json')


if __name__ == '__main__': main()
