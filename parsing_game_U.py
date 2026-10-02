"""U0 candidate export: condition complements, inherited hypothetical scope, control candidates.

U does not modify T. S supplies tokenization and attachment hypotheses. This module
never changes S's globals, fits CEM, chooses a graph, or authorizes world-state
commitment.
"""
import argparse
import json
import uuid
import copy

import parsing_game_S as s
from parsing_references import add_reference_candidates
from candidate_validation import empty_selection, validate_candidate_selection


def export_candidate_graph(text, *, package_id=None):
    doc = s.get_nlp()(text)
    package = dict(
        schema_version="0.2", package_id=package_id or "pkg_" + uuid.uuid4().hex,
        document=dict(id="document", text=text),
        producer=dict(name="parsing_game_U", version="U0", resources=[]),
        evidence=[], nodes=[], candidates=[], choice_sets=[], open_questions=[],
        coverage=dict(status="partial", unrepresented_evidence_ids=[], limitations=[
            "reference_candidates_only_no_identity_merging", "no_world_state_commitment",
            "bounded_if_and_local_modal_scope", "no_semantic_support_validation",
            "purpose_and_attachment_readings_are_candidates",
            "subject_control_is_an_unresolved_candidate",
            "unambiguous_condition_complements_are_required"]),
    )
    spans, mentions, predicates = {}, {}, {}

    def evidence(tokens):
        tokens = list(tokens)
        start, end = min(t.idx for t in tokens), max(t.idx + len(t.text) for t in tokens)
        key = (start, end)
        if key not in spans:
            ident = f"e{len(spans)}"
            spans[key] = ident
            package['evidence'].append(dict(id=ident, start=start, end=end, text=text[start:end]))
        return spans[key]

    def candidate(kind, arguments, value, support, requires=(), scope=None):
        item = dict(id=f"c{len(package['candidates'])}", type=kind,
                    arguments=arguments, value=value, evidence_ids=list(support),
                    scope=scope or dict(polarity="positive", contexts=[]),
                    provenance=[dict(producer="parsing_game_U", version="U0",
                                     method="bounded_dependency_rules", resource_ids=[])],
                    assessment=dict(status="proposed", score=None),
                    requires=list(requires), exclusive_with=[])
        package['candidates'].append(item)
        return item

    def question(kind, support, candidates, message, blocking=()):
        package['open_questions'].append(dict(
            id=f"q{len(package['open_questions'])}", kind=kind,
            evidence_ids=list(support), candidate_ids=[c['id'] for c in candidates],
            question=message, blocking_for=list(blocking)))

    def mention(token):
        if token.i not in mentions:
            words = [token] + [c for c in token.children if c.dep_ in {'det', 'amod', 'compound', 'nummod', 'poss'}]
            ident = f"m{token.i}"
            mentions[token.i] = ident
            support = [evidence(words)]
            package['nodes'].append(dict(id=ident, kind="mention", label=doc[min(t.i for t in words):max(t.i for t in words)+1].text,
                                         evidence_ids=support))
        return mentions[token.i]

    # Copular auxiliaries anchor states; modal auxiliaries are not separate events.
    for token in doc:
        if token.pos_ == 'VERB' or (token.pos_ == 'AUX' and token.dep_ in {'ROOT', 'conj', 'advcl', 'ccomp'}):
            ident = f"p{token.i}"
            support = [evidence([token])]
            package['nodes'].append(dict(id=ident, kind="proposition", label=token.text,
                                         predicate=token.lemma_, evidence_ids=support,
                                         predicate_evidence_ids=support))
            pred = candidate('PREDICATION', dict(proposition=ident), None, support)
            predicates[token.i] = pred

    def pid(index):
        return predicates[index]['arguments']['proposition']

    def modal_lemma(token):
        # spaCy splits won't/can't into wo/ca + n't.
        return {'wo': 'will', 'ca': 'can', 'sha': 'shall'}.get(token.lower_, token.lower_)

    def local_negatives(token):
        return [c for c in token.children if c.dep_ == 'neg' and not
                (c.lower_ == 'not' and c.i + 1 < len(doc) and doc[c.i + 1].lower_ == 'only')]

    # Build contexts independently of identity resolution and occurrence.
    contexts = {i: [] for i in predicates}
    modal_dependencies = {i: [] for i in predicates}
    reporting = {'say', 'report', 'claim', 'tell', 'state'}
    for i, pred in predicates.items():
        token = doc[i]
        negatives = local_negatives(token)
        if negatives:
            pred['scope']['polarity'] = 'negative'
            pred['evidence_ids'].extend(evidence([c]) for c in negatives)
        if token.sent.text.rstrip().endswith('?'):
            contexts[i].append(dict(kind='questioned', evidence_ids=[evidence([token.sent[-1]])]))
        # Only clausal complements of reporting predicates license attribution.
        branch = token
        inherited = []
        for ancestor in token.ancestors:
            if ancestor.lemma_ in reporting and branch.dep_ == 'ccomp':
                subjects = [c for c in ancestor.children if c.dep_ == 'nsubj']
                inherited.append(dict(kind='attributed',
                    source_mention_id=mention(subjects[0]) if len(subjects) == 1 else None,
                    report_proposition_id=pid(ancestor.i),
                    evidence_ids=[evidence([ancestor])]))
                pred['requires'].append(predicates[ancestor.i]['id'])
            branch = ancestor
        contexts[i].extend(reversed(inherited))

    # Only direct if-advcl attachments are promoted to conditional links.
    for token in doc:
        if token.lower_ != 'if':
            continue
        child = token.head
        if child.i in predicates and child.dep_ == 'advcl' and child.head.i in predicates:
            parent = child.head.i
            support = [evidence([token])]
            hypothetical = dict(kind='hypothetical', evidence_ids=support)
            contexts[child.i].append(hypothetical)
            # Complements inside the if-clause inherit its hypothetical scope.
            # They are not a second conditional, and they are not actual events.
            pending = [child]
            seen = {child.i}
            while pending:
                governor = pending.pop()
                for embedded in governor.children:
                    if embedded.i in seen or embedded.i not in predicates:
                        continue
                    if embedded.dep_ not in {'xcomp', 'ccomp', 'pcomp', 'advcl'}:
                        continue
                    seen.add(embedded.i)
                    contexts[embedded.i].append(dict(kind='hypothetical', evidence_ids=list(support)))
                    pending.append(embedded)
            link = candidate('CONDITIONAL_ON', dict(consequence=pid(parent), condition=pid(child.i)),
                             None, support, [predicates[parent]['id'], predicates[child.i]['id']])
            for i in predicates:
                if i == parent or (doc[i].sent == child.sent and child.head in doc[i].ancestors
                                   and child not in doc[i].ancestors and i != child.i):
                    contexts[i].append(dict(kind='conditional', condition_proposition_id=pid(child.i), evidence_ids=support))
                    # Conditional scope depends on the condition anchor, not on the link itself.
                    predicates[i]['requires'].append(predicates[child.i]['id'])
        else:
            question('scope', [evidence([token])], [], 'Resolve the scope of this if-clause.')

    # Order outer contexts by the depth of their governing predicate.
    def context_depth(ctx):
        if ctx['kind'] == 'questioned':
            return (-1, 0)
        ev = next(e for e in package['evidence'] if e['id'] == ctx['evidence_ids'][0])
        marker = next(t for t in doc if t.idx == ev['start'])
        governor = marker if ctx['kind'] == 'attributed' else marker.head.head
        return (len(list(governor.ancestors)), int(ctx['kind'] == 'attributed'))
    for scope_contexts in contexts.values():
        scope_contexts.sort(key=context_depth)

    readings = {'will': ['prediction'], 'would': ['prediction', 'possibility'],
                'can': ['ability', 'permission', 'possibility'],
                'could': ['ability', 'permission', 'possibility'],
                'may': ['permission', 'possibility'], 'might': ['possibility'],
                'must': ['obligation', 'unresolved'], 'should': ['obligation', 'prediction']}
    for token in doc:
        if token.tag_ != 'MD':
            continue
        i = token.head.i
        if i not in predicates:
            question('modality', [evidence([token])], [], 'Resolve this modal governor.')
            continue
        options = []
        for value in readings.get(modal_lemma(token), ['unresolved']):
            option = candidate('MODALITY', dict(proposition=pid(i)), value,
                               [evidence([token])], [predicates[i]['id']],
                               dict(polarity='positive', contexts=list(contexts[i])))
            options.append(option)
        if len(options) > 1 or options[0]['value'] == 'unresolved':
            for option in options:
                option['assessment']['status'] = 'unresolved'
            question('modality', [evidence([token])], options,
                     'Select the modal meaning; prediction, ability and obligation are distinct.',
                     [predicates[i]['id']])
            for option in options:
                option['exclusive_with'] = [c['id'] for c in options if c is not option]
            package['choice_sets'].append(dict(id=f"choice{len(package['choice_sets'])}", kind='interpretation',
                candidate_ids=[c['id'] for c in options], selection_rule='at_most_one', exhaustive=False,
                evidence_ids=[evidence([token])]))
            contexts[i].append(dict(kind='modal_choice',
                modality_choice_set_id=package['choice_sets'][-1]['id'],
                evidence_ids=[evidence([token])]))
        else:
            contexts[i].append(dict(kind='modal', modality_candidate_id=options[0]['id'], evidence_ids=[evidence([token])]))
            modal_dependencies[i].append(options[0]['id'])
        # Predications are anchors, so they never require their own modality.

    for i, pred in predicates.items():
        # Keep modal interpretations on role/link claims; anchor scope carries outer contexts.
        pred['scope']['contexts'] = [c for c in contexts[i] if c['kind'] not in {'modal', 'modal_choice'}]
        for token in doc[i].children:
            role = {'nsubj': 'subject', 'nsubjpass': 'subject', 'dobj': 'object', 'obj': 'object'}.get(token.dep_)
            if role:
                candidate('PARTICIPANT', dict(proposition=pid(i), mention=mention(token)), role,
                          [evidence([token])], [pred['id']] + modal_dependencies[i],
                          dict(polarity='positive', contexts=list(contexts[i])))

    frames = s.extract_proposition_frames(text)
    by_frame = {frame.frame_id: frame for frame in frames}
    for frame in frames:
        for link in frame.event_links:
            child_frame = by_frame[link['child_frame']]
            index = child_frame.predicate_index
            controller = link['controller_candidate']
            if controller is not None and index in predicates:
                item = candidate('PARTICIPANT', dict(proposition=pid(index),
                    mention=mention(doc[controller['token_index']])), 'controller',
                    [evidence([doc[controller['token_index']]])], [predicates[index]['id']])
                item['provenance'][0]['method'] = 'S_controller_candidate_preservation'
                item['assessment']['status'] = 'unresolved'
                question('reference', item['evidence_ids'], [item],
                         'Validate this preserved child-event controller candidate.')

    # spaCy also labels explicit-subject infinitives ccomp. Preserve their
    # controller evidence without treating finite ccomp subjects as controllers.
    for i, pred in predicates.items():
        token = doc[i]
        if token.dep_ not in {'xcomp', 'ccomp'} or token.tag_ != 'VB':
            continue
        markers = [c for c in token.children if c.lower_ == 'to' and c.dep_ in {'aux', 'mark'}]
        subjects = [c for c in token.children if c.dep_ in {'nsubj', 'nsubjpass'}]
        if not markers or len(subjects) != 1:
            continue
        ref = mention(subjects[0])
        if any(c['type'] == 'PARTICIPANT' and c['value'] == 'controller'
               and c['arguments'] == dict(proposition=pid(i), mention=ref)
               for c in package['candidates']):
            continue
        item = candidate('PARTICIPANT', dict(proposition=pid(i), mention=ref), 'controller',
                         [evidence([subjects[0]]), evidence(markers)], [pred['id']])
        item['assessment']['status'] = 'unresolved'
        item['provenance'][0]['method'] = 'explicit_subject_infinitive_controller_candidate'
        question('reference', item['evidence_ids'], [item],
                 'Validate this explicit-subject infinitive controller candidate.')

    for analysis in s.classify_to_attachments(doc):
        child = analysis['complement_index']
        head = doc[analysis['head_index']]
        # Adjectival state head is represented by its copula anchor.
        parent = head.i if head.i in predicates else head.head.i
        support = [evidence([doc[analysis['marker_index']], doc[child]])]
        if parent not in predicates:
            question('attachment', support, [], 'No supported parent proposition anchor.')
            continue
        alternatives = []
        for reading in analysis['structural_candidates']:
            if reading == 'directional_pp':
                alternatives.append(candidate('PARTICIPANT', dict(proposition=pid(parent), mention=mention(doc[child])),
                                              'destination', support, [predicates[parent]['id']]))
            elif reading in {'infinitival_complement', 'degree_result_infinitive'} and child in predicates:
                alternatives.append(candidate('EVENT_LINK', dict(parent=pid(parent), child=pid(child)),
                                              'complement' if reading == 'infinitival_complement' else 'unresolved',
                                              support, [predicates[parent]['id'], predicates[child]['id']]))
        # Purpose is offered only for an infinitival advcl, never selected automatically.
        if child in predicates and doc[child].dep_ == 'advcl':
            alternatives.append(candidate('EVENT_LINK', dict(parent=pid(parent), child=pid(child)),
                                          'purpose', support, [predicates[parent]['id'], predicates[child]['id']]))
        for item in alternatives:
            item['assessment']['status'] = 'unresolved'
            item['scope']['contexts'] = list(contexts[parent])
            item['requires'].extend(modal_dependencies[parent])
            item['exclusive_with'] = [c['id'] for c in alternatives if c is not item]
            if item['value'] == 'destination' and child in predicates:
                # Selecting the noun reading must also exclude a bare event anchor,
                # not only the complement edge that requires that anchor.
                item['exclusive_with'].append(predicates[child]['id'])
                predicates[child]['exclusive_with'].append(item['id'])
        if len(alternatives) == 1:
            # One structural reading is not an ambiguity. It stays a candidate
            # and does not block its event anchor.
            alternatives[0]['assessment']['status'] = 'proposed'
        else:
            package['choice_sets'].append(dict(id=f"choice{len(package['choice_sets'])}", kind='interpretation',
                candidate_ids=[c['id'] for c in alternatives], selection_rule='at_most_one', exhaustive=False,
                evidence_ids=support))
            question('attachment', support, alternatives,
                     'Select an attachment reading or retain the unresolved alternatives.',
                     [predicates[child]['id']] if child in predicates else [])

    # An if-clause condition keeps its head proposition. When that head has one
    # embedded complement reading, the conditional requires it, so "decides to
    # pull" is not reduced to "decides".
    for item in package['candidates']:
        if item['type'] != 'CONDITIONAL_ON':
            continue
        grouped = {}
        for link in package['candidates']:
            if link['type'] == 'EVENT_LINK' and link['arguments']['parent'] == item['arguments']['condition']:
                grouped.setdefault(link['arguments']['child'], []).append(link)
        for options in grouped.values():
            if len(options) == 1 and options[0]['id'] not in item['requires']:
                item['requires'].append(options[0]['id'])

    # Subject control is a candidate, not a copied subject role. An infinitive
    # with no subject of its own may be controlled by the matrix subject.
    for i, pred in predicates.items():
        token = doc[i]
        if token.dep_ not in {'xcomp', 'ccomp'} or token.tag_ != 'VB':
            continue
        if any(c.dep_ in {'nsubj', 'nsubjpass'} for c in token.children):
            continue
        if not any(c.lower_ == 'to' and c.dep_ in {'aux', 'mark'} for c in token.children):
            continue
        governor = token.head
        subjects = [c for c in governor.children if c.dep_ in {'nsubj', 'nsubjpass'}]
        if len(subjects) != 1 and governor.pos_ == 'ADJ' and governor.head.lemma_ == 'be':
            subjects = [c for c in governor.head.children if c.dep_ == 'nsubj']
        # A direct object is object control. Do not offer the matrix subject.
        if len(subjects) != 1 or any(c.dep_ in {'obj', 'dobj', 'dative'} for c in governor.children):
            continue
        ref = mention(subjects[0])
        if any(c['type'] == 'PARTICIPANT' and c['value'] == 'controller'
               and c['arguments'] == dict(proposition=pid(i), mention=ref)
               for c in package['candidates']):
            continue
        requires = [pred['id']]
        child_links = [c for c in package['candidates'] if c['type'] == 'EVENT_LINK'
                       and c['arguments'].get('child') == pid(i)]
        if len(child_links) == 1:
            requires.append(child_links[0]['id'])
        item = candidate('PARTICIPANT', dict(proposition=pid(i), mention=ref), 'controller',
                         [evidence([subjects[0]]), evidence([c for c in token.children if c.lower_ == 'to'][:1])],
                         requires)
        item['assessment']['status'] = 'unresolved'
        item['provenance'][0]['method'] = 'subject_control_candidate'
        question('reference', item['evidence_ids'], [item],
                 'The matrix subject is a candidate controller of this infinitive, not a resolved identity.')

    # Include oblique mentions even when no participant role has been recovered.
    # Scope on roles/links qualifies their proposition reading, not the existence
    # of a mention or the syntactic role itself. Modality retains outer polarity.
    for item in package['candidates']:
        if item['type'] == 'CONDITIONAL_ON':
            index = next(i for i in predicates if pid(i) == item['arguments']['consequence'])
            item['scope']['contexts'] = copy.deepcopy([ctx for ctx in contexts[index]
                if ctx['kind'] not in {'modal', 'modal_choice'} and not
                (ctx['kind'] == 'conditional' and ctx['condition_proposition_id'] == item['arguments']['condition'])])
        if item['type'] not in {'PARTICIPANT', 'EVENT_LINK'}:
            continue
        proposition = item['arguments'].get('proposition', item['arguments'].get('parent'))
        index = next(i for i in predicates if pid(i) == proposition)
        item['scope'] = dict(polarity=predicates[index]['scope']['polarity'],
                             contexts=copy.deepcopy(contexts[index]))
        item['requires'] = list(dict.fromkeys(item['requires'] + modal_dependencies[index]))
    # Ambiguous modal scope is retained even on a bare predication anchor. Its
    # choice is a qualification, not a dependency back to the anchor.
    for i, pred in predicates.items():
        pred['scope']['contexts'].extend(copy.deepcopy(c) for c in contexts[i] if c['kind'] == 'modal_choice')
        negatives = local_negatives(doc[i])
        modals = [c for c in doc[i].children if c.tag_ == 'MD']
        # Only a single local negator under must/will/should has a supported
        # operator order here. Other combinations retain evidence and abstain.
        if negatives and modals and (len(negatives) != 1 or len(modals) != 1 or
                                     modal_lemma(modals[0]) not in {'must', 'will', 'should'}):
            affected = [c for c in package['candidates'] if
                        c['arguments'].get('proposition', c['arguments'].get('parent')) == pid(i)]
            for item in affected:
                item['scope']['polarity'] = 'unresolved'
            question('scope', [evidence(negatives + modals)], affected,
                     'Resolve negation versus modal operator scope.', [c['id'] for c in affected])
        lexical_modal = doc[i].lemma_ in {'require', 'oblige', 'permit'} or (
            doc[i].lemma_ == 'have' and any(child.dep_ == 'xcomp' and
                any(marker.lower_ == 'to' for marker in child.children) for child in doc[i].children))
        if lexical_modal:
            affected = [c for c in package['candidates'] if any(
                pid(j) in c['arguments'].values() for j in predicates
                if j == i or doc[i] in doc[j].ancestors)]
            question('scope', [evidence(doc[i].subtree)], [],
                     'Lexical modality and its negation scope require an extended interpretation.',
                     [c['id'] for c in affected])

    for token in doc:
        if token.pos_ in {'NOUN', 'PROPN', 'PRON'} and token.dep_ not in {'compound', 'poss', 'expl'}:
            mention(token)
    add_reference_candidates(doc, package, mentions, candidate, question)
    for token in doc:
        if token.pos_ == 'PRON' and token.i in mentions and token.lower_ not in {'she', 'her', 'he', 'him', 'it', 'they', 'them'}:
            question('reference', [evidence([token])], [], 'Reference form outside the bounded reference policy.')

    # Explicitly expose unsupported semantic/scope work instead of claiming completeness.
    for sent in doc.sents:
        support = [evidence(sent)]
        package['coverage']['unrepresented_evidence_ids'].extend(support)
        question('missing_representation', support, [],
                 'Review unrepresented semantics and scope, including attribution, choice and entailment.')
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sentence', required=True)
    args = parser.parse_args()
    print(json.dumps(export_candidate_graph(args.sentence), indent=2))


if __name__ == '__main__':
    main()
