"""Z10: complete alternative stripping bundles and construction-specific gaps."""
import copy

import parsing_game_S as s
import parsing_game_Z5 as z5
import parsing_game_Z6 as z6
import parsing_game_Z7 as z7
import parsing_game_Z8 as z8
import parsing_game_Z9 as z9
from ellipsis_coverage import add_construction_questions


def prepare_role_readings(package):
    """Split replacement-role alternatives before Z8 copies unchanged roles."""
    candidates = package['candidates']
    doc = s.get_nlp()(package['document']['text'])
    stripping = lambda c: any(p['method'] == 'stripping_not_nominal' for p in c['provenance'])
    predicates = [c for c in candidates if c['type'] == 'PREDICATION' and stripping(c)]
    for pred in predicates:
        roles = [c for c in candidates if c['type'] == 'PARTICIPANT' and stripping(c)
                 and pred['id'] in c['requires']]
        question = next(q for q in package['open_questions'] if pred['id'] in q['candidate_ids'])
        # A bare name can contrast with a named subject as well as an object
        # or recipient. Keep the alternatives; names are not entity identity.
        subjects = [r for r in roles if r['value'] == 'subject']
        replacements = [r for r in roles if r['value'] != 'subject']
        if subjects and replacements:
            remnant = doc[int(replacements[0]['arguments']['mention'][1:])]
            subject = doc[int(subjects[0]['arguments']['mention'][1:])]
            previous = next((t for t in reversed(list(doc[:remnant.i]))
                             if not t.is_space and not t.is_punct), None)
            if remnant.pos_ == subject.pos_ == 'PROPN' and previous is not None and previous.lower_ == 'not':
                alternative = copy.deepcopy(replacements[0])
                alternative['id'] = z5._next_id(candidates, 'c')
                alternative['value'] = 'subject'
                alternative['provenance'].append(dict(producer='parsing_game_Z10', version='Z10',
                    method='bare_named_subject_contrast', resource_ids=[]))
                candidates.append(alternative)
                roles.append(alternative)
                replacements.append(alternative)
                question['candidate_ids'].append(alternative['id'])
                for role in replacements:
                    role['exclusive_with'] = [r['id'] for r in replacements if r is not role]
        # A parser's conjunct attachment must not add the contrasting occurrence
        # to the spoken clause as another positive participant.
        remnant_mentions = {r['arguments']['mention'] for r in roles}
        source_roles = [c for c in candidates if c['type'] == 'PARTICIPANT' and not stripping(c)
                        and c['arguments']['proposition'] == pred['arguments']['proposition']]
        # The shared subject also carries reconstruction support. Only remove
        # occurrences whose source span actually follows the negation cue.
        evidence = {e['id']: e for e in package['evidence']}
        nodes = {n['id']: n for n in package['nodes']}
        neg_end = max(evidence[e]['end'] for e in pred['evidence_ids']
                      if evidence[e]['text'].lower() == 'not')
        removed = {c['id'] for c in source_roles if c['arguments']['mention'] in remnant_mentions
                   and min(evidence[e]['start'] for e in nodes[c['arguments']['mention']]['evidence_ids']) >= neg_end}
        candidates[:] = [c for c in candidates if c['id'] not in removed]
        for c in candidates:
            for field in ['requires', 'exclusive_with']:
                c[field] = [i for i in c[field] if i not in removed]
        for q in package['open_questions']:
            for field in ['candidate_ids', 'blocking_for']:
                q[field] = [i for i in q[field] if i not in removed]
        for group in package['choice_sets']:
            group['candidate_ids'] = [i for i in group['candidate_ids'] if i not in removed]

        alternatives = [r for r in roles if r['exclusive_with']]
        if len(alternatives) < 2:
            continue
        common = [r for r in roles if r not in alternatives]
        branches = [pred]
        for role in alternatives[1:]:
            branch = copy.deepcopy(pred)
            branch['id'] = z5._next_id(candidates, 'c')
            candidates.append(branch)
            branches.append(branch)
            role['requires'] = [branch['id'] if i == pred['id'] else i for i in role['requires']]
            question['candidate_ids'].append(branch['id'])
            for shared in common:
                if shared['value'] == role['value']:
                    continue
                cloned = copy.deepcopy(shared)
                cloned['id'] = z5._next_id(candidates, 'c')
                cloned['requires'] = [branch['id'] if i == pred['id'] else i for i in cloned['requires']]
                candidates.append(cloned)
                question['candidate_ids'].append(cloned['id'])
        for branch in branches:
            branch['exclusive_with'] = [b['id'] for b in branches if b is not branch]
        package['choice_sets'].append(dict(id=z5._next_id(package['choice_sets'], 'choice'),
            kind='interpretation', candidate_ids=[b['id'] for b in branches],
            selection_rule='at_most_one', exhaustive=False, evidence_ids=pred['evidence_ids'][:]))
        question['kind'] = 'attachment'
        question['question'] = ('Which participant does the stripping remnant contrast with? '
                                'Each alternative has its own proposition and unchanged participants.')


def export_candidate_graph(text, *, package_id=None):
    package = z5.export_candidate_graph(text, package_id=package_id)
    prepare_role_readings(package)
    z8.separate_reconstructions(package)
    z9.mark_reconstruction_scope_gaps(package)
    add_construction_questions(text, package)
    z6.preserve_condition_content(package)
    z7.add_condition_contents(package)
    package['schema_version'] = '0.4'
    package['producer'].update(name='parsing_game_Z10', version='Z10')
    return package
