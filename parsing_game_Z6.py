"""Z6: preserve represented complement content on every conditional selection path.

Z5 remains unchanged. Ambiguous content stays provisional rather than forcing
all competing readings into a conjunctive dependency list.
"""
import parsing_game_Z5 as z5


def preserve_condition_content(package):
    candidates = package['candidates']
    table = {c['id']: c for c in candidates}
    outgoing = {}
    for c in candidates:
        if c['type'] == 'EVENT_LINK':
            outgoing.setdefault(c['arguments']['parent'], []).append(c)
    # Replace Z4's conditional-link-only shortcut with the shared content policy.
    for c in candidates:
        if c['type'] == 'CONDITIONAL_ON':
            c['requires'] = [ident for ident in c['requires'] if table[ident]['type'] != 'EVENT_LINK']

    def closure(ident):
        found, pending = set(), [ident]
        while pending:
            item = pending.pop()
            if item not in found:
                found.add(item)
                pending.extend(table[item]['requires'])
        return found

    def content(proposition, path=()):
        if proposition in path:
            return set(), True
        required, incomplete = set(), False
        grouped = {}
        for link in outgoing.get(proposition, []):
            grouped.setdefault(link['arguments']['child'], []).append(link)
        for child, links in grouped.items():
            # A sole event edge can still compete with a nominal/destination reading.
            if len(links) != 1 or links[0]['exclusive_with']:
                incomplete = True
                continue
            link = links[0]
            if link['value'] not in {'complement', 'attempt'}:
                incomplete = True
                continue
            required.add(link['id'])
            nested, gap = content(child, path + (proposition,))
            required.update(nested)
            incomplete |= gap
        return required, incomplete

    for candidate in candidates:
        conditions = {ctx['condition_proposition_id']
                      for ctx in candidate['scope']['contexts'] if ctx['kind'] == 'conditional'}
        if candidate['type'] == 'CONDITIONAL_ON':
            conditions.add(candidate['arguments']['condition'])
        for condition in sorted(conditions):
            required, incomplete = content(condition)
            # Avoid adding a back edge when unusual/nested parses reuse an anchor.
            for ident in sorted(required):
                if candidate['id'] in closure(ident):
                    incomplete = True
                elif ident not in candidate['requires']:
                    candidate['requires'].append(ident)
            if incomplete:
                # Empty candidates deliberately prevent a superficial resolution
                # by selecting just one fragment of a condition interpretation.
                package['open_questions'].append(dict(
                    id=next_question_id(package), kind='scope',
                    evidence_ids=list(candidate['evidence_ids']), candidate_ids=[],
                    question='The condition has ambiguous or unsupported embedded content; '
                             'a complete condition interpretation is still required.',
                    blocking_for=[candidate['id']]))


def next_question_id(package):
    occupied = {x['id'] for name in ('nodes', 'evidence', 'candidates', 'choice_sets', 'open_questions')
                for x in package[name]}
    index = len(package['open_questions'])
    while f'q{index}' in occupied:
        index += 1
    return f'q{index}'


def export_candidate_graph(text, *, package_id=None):
    package = z5.export_candidate_graph(text, package_id=package_id)
    preserve_condition_content(package)
    package['producer'].update(name='parsing_game_Z6', version='Z6')
    return package
