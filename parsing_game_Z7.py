"""Z7: explicit condition content, with conservative partial-selection validation."""
import parsing_game_Z6 as z6


def add_condition_contents(package):
    candidates = package['candidates']
    table = {c['id']: c for c in candidates}
    nodes = {n['id']: n for n in package['nodes']}
    conditions = {ctx['condition_proposition_id'] for c in candidates
                  for ctx in c['scope']['contexts'] if ctx['kind'] == 'conditional'}
    conditions.update(c['arguments']['condition'] for c in candidates if c['type'] == 'CONDITIONAL_ON')
    anchors, edges = {}, {}
    for c in candidates:
        if c['type'] == 'PREDICATION':
            anchors.setdefault(c['arguments']['proposition'], []).append(c)
        if c['type'] == 'EVENT_LINK':
            edges.setdefault(c['arguments']['parent'], []).append(c)
    competing = {c['id'] for c in candidates if c['exclusive_with']}
    for group in package['choice_sets']:
        if group['kind'] == 'interpretation' and group['selection_rule'] == 'at_most_one' and len(group['candidate_ids']) > 1:
            competing.update(group['candidate_ids'])

    def closure(ident):
        result, pending = set(), [ident]
        while pending:
            current = pending.pop()
            if current not in result:
                result.add(current)
                pending.extend(table[current]['requires'])
        return result

    package['condition_contents'] = []
    for index, condition in enumerate(sorted(conditions)):
        required, visited, gaps = set(), set(), []
        pending = [condition]
        while pending:
            prop = pending.pop()
            if prop in visited:
                continue
            visited.add(prop)
            readings = anchors.get(prop, [])
            if not readings:
                gaps.append('missing predication')
                continue
            # Keep the source anchor while flagging alternative reconstructions.
            required.add(readings[0]['id'])
            if len(readings) != 1 or readings[0]['id'] in competing:
                gaps.append('competing predications')
            for link in edges.get(prop, []):
                if link['id'] in competing or link['value'] not in {'complement', 'attempt'}:
                    gaps.append('ambiguous or unsupported event link')
                else:
                    required.add(link['id'])
                    pending.append(link['arguments']['child'])
            roles = [c for c in candidates if c['type'] == 'PARTICIPANT' and c['arguments']['proposition'] == prop]
            for role in roles:
                if role['id'] in competing:
                    gaps.append('competing participant readings')
                    continue
                required.add(role['id'])
                for quantity in candidates:
                    if quantity['type'] == 'QUANTITY' and quantity['arguments']['mention'] == role['arguments']['mention']:
                        if quantity['id'] in competing:
                            gaps.append('competing quantities')
                        else:
                            required.add(quantity['id'])
            # A unique modal must survive even when there are no role candidates.
            modals = [c for c in candidates if c['type'] == 'MODALITY' and c['arguments']['proposition'] == prop]
            if len(modals) == 1:
                required.add(modals[0]['id'])
            elif modals:
                # Existing modality question records the ambiguity; never require all readings.
                pass

        # Detect represented embedded predicates for which no supported link exists.
        markers = {e for a in anchors[condition] for ctx in a['scope']['contexts']
                   if ctx['kind'] == 'hypothetical' for e in ctx['evidence_ids']}
        for prop, readings in anchors.items():
            if prop not in visited and any(markers.intersection(ctx['evidence_ids'])
                    for a in readings for ctx in a['scope']['contexts'] if ctx['kind'] == 'hypothetical'):
                gaps.append('embedded predicate without retained link')

        expanded = set().union(*(closure(ident) for ident in required))
        if any(set(table[ident]['exclusive_with']).intersection(expanded) for ident in expanded):
            # Do not make every consequence impossible to select on an inconsistent parse.
            required = {anchors[condition][0]['id']}
            gaps.append('incompatible required content')
            expanded = closure(anchors[condition][0]['id'])
        question_ids = [q['id'] for q in package['open_questions']
                        if expanded.intersection(q['blocking_for'] + q['candidate_ids'])]
        if gaps:
            qid = z6.next_question_id(package)
            package['open_questions'].append(dict(id=qid, kind='scope',
                evidence_ids=list(nodes[condition]['evidence_ids']), candidate_ids=[],
                question='Complete condition interpretation remains open: ' + ', '.join(sorted(set(gaps))) + '.',
                blocking_for=[]))
            question_ids.append(qid)
        package['condition_contents'].append(dict(id=f'condition_content_{index}',
            condition_proposition_id=condition, required_candidate_ids=sorted(required),
            question_ids=question_ids, evidence_ids=list(nodes[condition]['evidence_ids'])))

    # Materialize safe requirements for convenient consumers. The validator also
    # checks bundles independently, so a partial selection cannot lose this guard.
    bundles = {b['condition_proposition_id']: b for b in package['condition_contents']}
    for c in candidates:
        carried = {ctx['condition_proposition_id'] for ctx in c['scope']['contexts'] if ctx['kind'] == 'conditional'}
        if c['type'] == 'CONDITIONAL_ON':
            carried.add(c['arguments']['condition'])
        for condition in sorted(carried):
            for ident in bundles[condition]['required_candidate_ids']:
                if c['id'] not in closure(ident) and ident not in c['requires']:
                    c['requires'].append(ident)


def export_candidate_graph(text, *, package_id=None):
    package = z6.export_candidate_graph(text, package_id=package_id)
    add_condition_contents(package)
    package['schema_version'] = '0.3'
    package['producer'].update(name='parsing_game_Z7', version='Z7')
    return package
