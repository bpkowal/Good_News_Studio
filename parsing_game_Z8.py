"""Z8: separate proposition identities for Z5 stripping reconstructions."""
import copy

import parsing_game_Z5 as z5
import parsing_game_Z6 as z6
import parsing_game_Z7 as z7


def separate_reconstructions(package):
    original_candidates = list(package['candidates'])
    original_questions = list(package['open_questions'])
    original_choices = list(package['choice_sets'])
    ambiguous_roles = {i for group in original_choices
                       if group['kind'] == 'interpretation' and group['selection_rule'] == 'at_most_one'
                       and len(group['candidate_ids']) > 1 for i in group['candidate_ids']}
    nodes = {n['id']: n for n in package['nodes']}
    package['reconstructions'] = []

    def fresh(collection, prefix):
        occupied = {x['id'] for key in ('nodes', 'candidates', 'evidence', 'choice_sets', 'open_questions', 'reconstructions')
                    for x in package[key]}
        index = len(package[collection])
        while prefix + str(index) in occupied:
            index += 1
        return prefix + str(index)

    def stripping(c):
        return any(p['method'] == 'stripping_not_nominal' for p in c['provenance'])

    for pred in original_candidates:
        if pred['type'] != 'PREDICATION' or not stripping(pred):
            continue
        antecedent = pred['arguments']['proposition']
        source = next(c for c in original_candidates if c['type'] == 'PREDICATION'
                      and c['arguments']['proposition'] == antecedent and not stripping(c))
        roles = [c for c in original_candidates if c['type'] == 'PARTICIPANT' and stripping(c)
                 and pred['id'] in c['requires']]
        prop = copy.deepcopy(nodes[antecedent])
        prop['id'] = fresh('nodes', 'p_reconstructed_')
        prop['label'] = 'reconstructed ' + prop['label']
        prop['evidence_ids'] = list(dict.fromkeys(prop['evidence_ids'] + pred['evidence_ids']))
        package['nodes'].append(prop)
        new_id = prop['id']
        mapping = {source['id']: pred['id']}
        modals = []
        for modal in original_candidates:
            if modal['type'] == 'MODALITY' and modal['arguments']['proposition'] == antecedent:
                cloned = copy.deepcopy(modal)
                cloned['id'] = fresh('candidates', 'c')
                cloned['arguments']['proposition'] = new_id
                mapping[modal['id']] = cloned['id']
                package['candidates'].append(cloned)
                modals.append(cloned)
        choices = {}
        for group in original_choices:
            if group['candidate_ids'] and all(i in mapping and i != source['id'] for i in group['candidate_ids']):
                cloned = copy.deepcopy(group)
                cloned['id'] = fresh('choice_sets', 'choice')
                cloned['candidate_ids'] = [mapping[i] for i in group['candidate_ids']]
                choices[group['id']] = cloned['id']
                package['choice_sets'].append(cloned)

        # Retain compatible unchanged participants, but never import the slot
        # replaced by the remnant. Ambiguous replacement slots stay incomplete.
        replaced = {r['value'] for r in roles}
        for role in original_candidates:
            if (role['type'] != 'PARTICIPANT' or stripping(role) or
                role['arguments']['proposition'] != antecedent or role['value'] in replaced):
                continue
            if role['exclusive_with'] or role['id'] in ambiguous_roles:
                continue
            cloned = copy.deepcopy(role)
            cloned['id'] = fresh('candidates', 'c')
            cloned['scope'] = copy.deepcopy(pred['scope'])
            cloned['assessment'] = dict(status='unresolved', score=None)
            cloned['provenance'] = [dict(producer='parsing_game_Z8', version='Z8',
                method='ellipsis_unchanged_participant', resource_ids=[])]
            package['candidates'].append(cloned)
            roles.append(cloned)

        for item in [pred] + roles + modals:
            item['arguments']['proposition'] = new_id
            item['requires'] = list(dict.fromkeys(mapping.get(i, i) for i in item['requires']))
            item['exclusive_with'] = [mapping.get(i, i) for i in item['exclusive_with']]
            for ctx in item['scope']['contexts']:
                if ctx['kind'] == 'modal':
                    ctx['modality_candidate_id'] = mapping.get(ctx['modality_candidate_id'], ctx['modality_candidate_id'])
                if ctx['kind'] == 'modal_choice':
                    ctx['modality_choice_set_id'] = choices.get(ctx['modality_choice_set_id'], ctx['modality_choice_set_id'])
        # An anchor licenses a reading; it cannot depend on its own modal.
        own_modals = {m['id'] for m in modals}
        pred['requires'] = [i for i in pred['requires'] if i not in own_modals and i != pred['id']]
        pred['scope']['contexts'] = [ctx for ctx in pred['scope']['contexts'] if not
            (ctx['kind'] == 'modal' and ctx['modality_candidate_id'] in own_modals)]
        for q in original_questions:
            modal_ids = [i for i in q['candidate_ids'] if i in mapping and i != source['id']]
            if not modal_ids:
                continue
            cloned = copy.deepcopy(q)
            cloned['id'] = fresh('open_questions', 'q')
            cloned['candidate_ids'] = [mapping[i] for i in modal_ids]
            cloned['blocking_for'] = [mapping[i] for i in q['blocking_for'] if i in mapping]
            package['open_questions'].append(cloned)
        question = next(q for q in package['open_questions'] if pred['id'] in q['candidate_ids'])
        question['candidate_ids'] = list(dict.fromkeys(question['candidate_ids'] + [r['id'] for r in roles]))
        package['reconstructions'].append(dict(id=fresh('reconstructions', 'reconstruction_'),
            proposition_id=new_id, antecedent_proposition_id=antecedent,
            antecedent_candidate_id=source['id'], predication_candidate_id=pred['id'],
            participant_candidate_ids=[r['id'] for r in roles], evidence_ids=list(prop['evidence_ids']),
            method='stripping_not_nominal'))


def export_candidate_graph(text, *, package_id=None):
    # Identity separation must precede condition-content analysis.
    package = z5.export_candidate_graph(text, package_id=package_id)
    separate_reconstructions(package)
    z6.preserve_condition_content(package)
    z7.add_condition_contents(package)
    package['schema_version'] = '0.4'
    package['producer'].update(name='parsing_game_Z8', version='Z8')
    return package
