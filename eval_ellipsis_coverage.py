"""Evaluate frozen semantic readings at the typed graph-export boundary."""
import argparse
import importlib
import json
from pathlib import Path

from candidate_validation import empty_selection, validate_candidate_selection

FIXTURES = Path(__file__).parent / 'fixtures/ellipsis_semantic_coverage.json'
CONSTRUCTIONS = ('stripping', 'verb_phrase_ellipsis', 'gapping', 'sluicing')


def project(package):
    nodes = {n['id']: n for n in package['nodes']}
    candidates = {c['id']: c for c in package['candidates']}
    readings = []
    for reconstruction in package.get('reconstructions', []):
        prop = reconstruction['proposition_id']
        pred = candidates[reconstruction['predication_candidate_id']]
        local = [c for c in candidates.values() if c['arguments'].get('proposition') == prop]
        participants = [c for c in local if c['type'] == 'PARTICIPANT']
        scoped = [pred] + participants
        # Inconsistency across the bundle must fail scope preservation too.
        scopes = [dict(polarity=c['scope']['polarity'],
                       contexts=[ctx['kind'] for ctx in c['scope']['contexts']
                                 if ctx['kind'] not in {'modal', 'modal_choice'}]) for c in scoped]
        readings.append(dict(
            predicate=nodes[prop]['predicate'],
            roles=sorted([[c['value'], nodes[c['arguments']['mention']]['label']] for c in participants]),
            scope=scopes[0] if all(scope == scopes[0] for scope in scopes) else {'inconsistent': True},
            modal=any(c['type'] == 'MODALITY' for c in local),
            scope_question=any(q['kind'] == 'scope' and pred['id'] in q['blocking_for']
                               and 'NOT MODAL(P)' in q['question'] for q in package['open_questions']),
            unresolved=any(c['assessment']['status'] == 'unresolved' for c in local)))
    return readings


def score_readings(expected, actual):
    """One-to-one exact predicate/role matching; scope is scored separately."""
    remaining = list(range(len(expected)))
    matches, scope_correct, role_correct, role_count = 0, 0, 0, 0
    for reading in actual:
        roles = {tuple(r) for r in reading['roles']}
        role_count += len(reading['roles'])
        role_correct += max((len(roles & {tuple(r) for r in gold['roles']}) for gold in expected
                             if gold['predicate'] == reading['predicate']), default=0)
        index = next((i for i in remaining if expected[i]['predicate'] == reading['predicate']
                      and sorted(expected[i]['roles']) == sorted(reading['roles'])), None)
        if index is None:
            continue
        remaining.remove(index)
        matches += 1
        gold = expected[index]
        scope_correct += all(reading[key] == gold[key] for key in ['scope', 'modal', 'scope_question'])
    return dict(gold=len(expected), proposed=len(actual), matched=matches,
                false_proposals=len(actual) - matches, scope_correct=scope_correct,
                role_correct=role_correct, role_count=role_count,
                unresolved_proposals=sum(r['unresolved'] for r in actual))


def evaluate(exporter, fixtures=None):
    fixtures = fixtures if fixtures is not None else json.loads(FIXTURES.read_text())
    groups = {kind: dict(cases=0, positive_cases=0, controls=0, no_proposal_positive_cases=0,
        specific_question_cases=0, question_control_false_positives=0, valid_packages=0, gold=0, proposed=0, matched=0, false_proposals=0,
        scope_correct=0, role_correct=0, role_count=0, unresolved_proposals=0) for kind in CONSTRUCTIONS}
    cases = []
    for fixture in fixtures:
        package = exporter(fixture['text'], package_id='eval_' + fixture['id'])
        actual = project(package)
        counts = score_readings(fixture['readings'], actual)
        valid = validate_candidate_selection(package, empty_selection(package))['contract_valid']
        specific = any(q['question'].startswith(fixture['question_prefix']) for q in package['open_questions'])
        group = groups[fixture['construction']]
        for key, value in counts.items():
            group[key] += value
        group['cases'] += 1
        group['positive_cases'] += bool(fixture['readings'])
        group['controls'] += not fixture['readings']
        group['no_proposal_positive_cases'] += bool(fixture['readings']) and not actual
        group['specific_question_cases'] += bool(fixture['readings']) and specific
        group['question_control_false_positives'] += not fixture['readings'] and specific
        group['valid_packages'] += valid
        cases.append(dict(id=fixture['id'], construction=fixture['construction'], text=fixture['text'],
                          counts=counts, specific_question=specific, contract_valid=valid, actual=actual))
    ratio = lambda n, d: n / d if d else None
    for group in groups.values():
        group.update(candidate_recall=ratio(group['matched'], group['gold']),
                     candidate_precision=ratio(group['matched'], group['proposed']),
                     false_proposal_rate=ratio(group['false_proposals'], group['proposed']),
                     role_accuracy=ratio(group['role_correct'], group['role_count']),
                     scope_preservation=ratio(group['scope_correct'], group['matched']),
                     no_proposal_abstention=ratio(group['no_proposal_positive_cases'], group['positive_cases']),
                     unresolved_proposal_rate=ratio(group['unresolved_proposals'], group['proposed']))
    return dict(exporter=exporter.__module__, boundary='typed reconstruction graph; not raw proposer text',
                fixture_set='hand_authored_development_probes_not_a_held_out_benchmark',
                scope_denominator='structurally matched readings; unexported scope is not scored as correct',
                zero_denominator='null, not perfect coverage', constructions=groups, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exporter', choices=['Z8', 'Z9', 'Z10'], default='Z10')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    exporter = importlib.import_module('parsing_game_' + args.exporter).export_candidate_graph
    report = evaluate(exporter)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['constructions'], indent=2))


if __name__ == '__main__':
    main()
