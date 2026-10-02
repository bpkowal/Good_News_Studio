"""Read-only diagnostic evaluation; no user probes or fitting on audit labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import parsing_game_R as q


def baseline_action(evidence):
    if evidence['features'] is None:
        return None
    semantic = evidence['selected_candidate']['semantic_class']
    if semantic == 'causal':
        return int(bool(evidence['features']['causal_reverse_structure']))
    return 2 if semantic in {'association', 'explicit_no_relation'} else 3


def audit_candidate(evidence, policy, annotation=None):
    annotation = annotation or {}
    features = evidence['features']
    baseline = baseline_action(evidence)
    raw, scores, margin = None, None, None
    pair_correct = None
    if 'argument_token_indices' in annotation:
        indices = annotation['argument_token_indices']
        if len(indices) != 2 or any(type(i) is not int for i in indices):
            raise ValueError('argument_token_indices must contain two integer head indices')
        pair_correct = sorted(e['token_index'] for e in evidence['entities']) == sorted(indices)
    elif 'entity1' in annotation and 'entity2' in annotation:
        doc = q.get_nlp()(evidence['sentence'])
        expected = [q.find_entity_spans(doc, annotation[k]) for k in ('entity1', 'entity2')]
        # Repeated mentions need explicit gold offsets; do not guess which occurrence.
        if all(len(spans) == 1 for spans in expected):
            pair_correct = sorted(e['token_index'] for e in evidence['entities']) == sorted(
                spans[0]['token_index'] for spans in expected)
    baseline_claim = None
    if features is not None:
        vector = np.array([features[n] for n in q.FEATURE_NAMES])
        logits = policy.weights @ vector
        raw, scores = int(logits.argmax()), logits.tolist()
        margin = float(np.sort(logits)[-1] - np.sort(logits)[-2])
        weights = np.zeros_like(policy.weights)
        weights[baseline, q.FEATURE_NAMES.index('bias')] = 1
        baseline_claim = q._claim_from_evidence(evidence, q.CEMPolicy(weights))
    claim = q._claim_from_evidence(evidence, policy)
    if baseline_claim is None:
        baseline_claim = claim  # No observation: neither pipeline has a decision.
    gold = annotation.get('correct_action')
    scope = annotation.get('relation_scope', 'causal_four_action' if gold is not None else 'unreviewed')
    valid = pair_correct is True and gold is not None and scope == 'causal_four_action' and raw is not None
    flags = []
    candidate_recovered = len(evidence['entities']) == 2
    if pair_correct is False:
        flags.append('EXTRACTION_FAILURE_wrong_pair' if candidate_recovered else 'EXTRACTION_FAILURE_missing_pair')
    if candidate_recovered and scope == 'outside_causal_actions':
        flags.append('B_outside_action_space')
    if valid and raw != gold:
        flags.append('D_policy_error_candidate_requires_feature_review')
    if not annotation:
        flags.append('unreviewed')
    expected_eligible = annotation.get('expected_eligible')
    if expected_eligible is not None and type(expected_eligible) is not bool:
        raise ValueError('expected_eligible must be Boolean or absent')
    gate_outcomes = {}
    for name, final, action in (('cem', claim, raw), ('baseline', baseline_claim, baseline)):
        eligible = final['eligible_for_world_state']
        assessed = expected_eligible is not None
        interpretation_correct = valid and action == gold
        mismatch = eligible != expected_eligible if assessed else None
        gate_failure = mismatch if assessed and interpretation_correct else None
        gate_outcomes[name] = dict(
            eligible=eligible,
            false_commitment=(eligible and not expected_eligible) if assessed else None,
            false_rejection=(not eligible and expected_eligible) if assessed else None,
            interpretation_correct=interpretation_correct if valid else None,
            gating_failure=gate_failure,
            gating_failure_kind=('false_commitment' if eligible else 'false_rejection') if gate_failure else None,
            rejection_reasons=final['validation_reasons'],
        )
        if gate_failure:
            flags.append('GATING_FAILURE_' + name + '_' + gate_outcomes[name]['gating_failure_kind'])
    return dict(
        schema_version=2, sentence=evidence['sentence'], candidate=evidence.get('selected_candidate'),
        arguments=evidence['entities'], pair_source=evidence.get('pair_source'),
        argument_context=evidence['argument_context'], features=features,
        candidate_recovered=candidate_recovered,
        observation_available=features is not None, pair_correct=pair_correct,
        annotation=annotation, relation_scope=scope, raw_cem_action=raw,
        scores=scores, margin=margin, margin_interpretation='uncalibrated',
        baseline_action=baseline, final_claim=claim, baseline_final_claim=baseline_claim,
        cem_correct=(raw == gold) if valid else None,
        baseline_correct=(baseline == gold) if valid else None,
        pair_and_raw_action_correct=((claim.get('decision') or {}).get('action') == gold
                              and pair_correct is True) if claim.get('decision') and gold is not None
                              and scope == 'causal_four_action'
                              else False if gold is not None and scope == 'causal_four_action' else None,
        false_commitment=gate_outcomes['cem']['false_commitment'],
        gate_outcomes=gate_outcomes, diagnostic_flags=flags,
    )


def summarize(records):
    valid = [r for r in records if r['cem_correct'] is not None]
    comparable = [r for r in records if r['raw_cem_action'] is not None and r['baseline_action'] is not None]
    annotated_pairs = [r for r in records if r['pair_correct'] is not None]
    eligibility_labels = [r for r in records if r['false_commitment'] is not None]
    labelled = [r for r in records if r['pair_and_raw_action_correct'] is not None]
    def fraction(correct, total):
        return dict(numerator=correct, denominator=total, rate=correct / total if total else None)
    buckets = dict(both_correct=0, cem_only_correct=0, baseline_only_correct=0, both_wrong=0)
    for r in valid:
        key = ('both_correct' if r['cem_correct'] and r['baseline_correct'] else
               'cem_only_correct' if r['cem_correct'] else
               'baseline_only_correct' if r['baseline_correct'] else 'both_wrong')
        buckets[key] += 1
    return dict(
        candidates=len(records),
        annotated_cases=sum(bool(r['annotation']) for r in records),
        candidate_extraction_coverage=fraction(sum(r['candidate_recovered'] for r in records),len(records)),
        coverage=fraction(sum(r['observation_available'] for r in records), len(records)),
        pair_accuracy=fraction(sum(r['pair_correct'] for r in annotated_pairs), len(annotated_pairs)),
        conditional_raw_accuracy=fraction(sum(r['cem_correct'] for r in valid), len(valid)),
        baseline_conditional_accuracy=fraction(sum(r['baseline_correct'] for r in valid),len(valid)),
        raw_disagreements=fraction(sum(r['raw_cem_action'] != r['baseline_action'] for r in comparable),len(comparable)),
        pair_and_raw_action_accuracy=fraction(sum(r['pair_and_raw_action_correct'] for r in labelled), len(labelled)),
        commitment_abstention=fraction(sum(not r['final_claim']['eligible_for_world_state'] for r in records), len(records)),
        false_commitments=fraction(sum(r['false_commitment'] for r in eligibility_labels), len(eligibility_labels)),
        paired_comparison=buckets,
        pipelines={name: {
            metric: fraction(sum(r['gate_outcomes'][name][metric] for r in records
                                 if r['gate_outcomes'][name][metric] is not None),
                             sum(r['gate_outcomes'][name][metric] is not None for r in records))
            for metric in ('false_commitment','false_rejection','gating_failure')
        } | {'abstention': fraction(sum(not r['gate_outcomes'][name]['eligible'] for r in records),len(records))}
                   for name in ('cem','baseline')},
        stage_counts=dict(
            no_candidate=sum(not r['candidate_recovered'] for r in records),
            annotated_wrong_pair=sum(r['pair_correct'] is False and r['candidate_recovered'] for r in records),
            available_outside_scope=sum(r['candidate_recovered'] and r['relation_scope']=='outside_causal_actions' for r in records),
            eligible_for_prediction_scoring=len(valid),
            unreviewed_pair=sum(r['pair_correct'] is None for r in records)),
        note='Conditional accuracies require an annotated correct, in-scope pair. Gate attribution additionally requires a correct raw action and explicit eligibility label. Null means unscored.',
    )


def collision_candidates(records):
    groups = {}
    for index, r in enumerate(records):
        if r['pair_correct'] is True and r['cem_correct'] is not None:
            key = tuple(r['features'][n] for n in q.FEATURE_NAMES)
            groups.setdefault(key, []).append(index)
    return [dict(record_indices=indices, status='C_candidate_requires_label_and_scope_review')
            for indices in groups.values()
            if len({records[i]['annotation']['correct_action'] for i in indices}) > 1]


def run_audit(policy):
    records, extras, end_to_end = [], [], {}
    for name in ('TEST_EXAMPLES', 'NEGATION_GENERALIZATION_EXAMPLES', 'ROBUSTNESS_EXAMPLES',
                 'LEXICAL_HOLDOUT', 'UNKNOWN_HOLDOUT', 'EPISTEMIC_HOLDOUT'):
        suite = getattr(q, name)
        end_to_end[name] = q.evaluate_suite(suite, policy)[0]
        for example in suite:
            primary = q.collect_evidence(example['sentence'])
            record = audit_candidate(primary, policy, example)
            record['suite'] = name
            world = q.parse_world_state(example['sentence'], policy)
            record['complement_relations'] = world['complement_relations']
            record['propositions'] = world['propositions']
            records.append(record)
            for evidence in q.collect_claim_evidence(example['sentence']):
                if evidence.get('selected_candidate', {}).get('index') != primary.get('selected_candidate', {}).get('index'):
                    extras.append(audit_candidate(evidence, policy))
    fixture = json.loads((Path(__file__).resolve().parent / 'resources' / 'R_commitment_audit.json').read_text())
    commitment_records = [audit_candidate(q.collect_evidence(case['sentence']), policy, case)
                          for case in fixture['cases']]
    return dict(schema_version=2, repair_enabled=q.REPAIR_ENABLED, summary=summarize(records),
                end_to_end=end_to_end, collision_candidates=collision_candidates(records),
                records=records, additional_unlabelled_candidates=extras,
                commitment_audit=dict(annotation_basis=fixture['annotation_basis'],
                                      summary=summarize(commitment_records), records=commitment_records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 42, 91])
    parser.add_argument('--no-repair', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    q.REPAIR_ENABLED = not args.no_repair
    runs = []
    for seed in args.seeds:
        policy, _ = q.train_policy(seed=seed)
        result = dict(seed=seed, **run_audit(policy))
        runs.append(result)
        print(json.dumps(dict(seed=seed, summary=result['summary'],
                             commitment_audit=result['commitment_audit']['summary'])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(schema_version=2, runs=runs), indent=2) + '\n')


if __name__ == '__main__':
    main()
