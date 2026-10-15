"""Retain a copied physical scene and connect an explicit instrument mechanism.

Uses the existing 1.4 proposition and occurrence collections. Scene assertions
retain their entire modal wording; they are not extra performed actions.
"""
from copy import deepcopy
import re

from z10_world_model_adapter import segment_source_clauses


def counted_antecedent(text, reference):
    """Resolve only unique explicit human counts before a definite remnant.

    Returns the exact antecedent span. Repeated equal-size groups stay unresolved.
    This is intentionally narrower than general coreference.
    """
    from parsing_game_S import get_nlp
    match = re.fullmatch(r'(?:the\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)|that\s+(\w+))', reference, re.I)
    position = text.find(reference)
    if not match or position < 0:
        return ''
    matches = []
    for chunk in get_nlp()(text[:position]).noun_chunks:
        if chunk.root.lemma_.lower() not in {'worker', 'patient', 'person', 'child', 'resident'}:
            continue
        number = next((t.lower_ for t in chunk if t.dep_ == 'nummod'), '')
        if (match[1] and number == match[1].lower()) or (
            match[2] and number in {'one', '1'} and chunk.root.lemma_.lower() == match[2].lower()):
            matches.append(_nominal(chunk.root))
    return matches[0] if len(matches) == 1 else ''


def _resolve_scene_parties(text, world, provenance=None):
    """Unify uniquely counted references; retain antecedent and usage evidence."""
    clauses = segment_source_clauses(text)
    remap, canonical, resolved = {}, {}, {}
    for party in world['parties']:
        antecedent = counted_antecedent(text, party['label'])
        label = antecedent or party['label']
        if antecedent:
            resolved[party['party_id']] = (party['label'], antecedent)
            party['label'] = antecedent
            party['clause_ids'] = list(dict.fromkeys(party['clause_ids'] + [
                c['clause_id'] for c in clauses if antecedent in c['text']]))
            number = next((word for word in antecedent.split() if word.isdigit() or word in
                           {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}), '')
            party['quantities'] = [number] if number else party.get('quantities', [])
        if label.casefold() in canonical:
            target = canonical[label.casefold()]
            target['clause_ids'] = list(dict.fromkeys(target['clause_ids'] + party['clause_ids']))
            remap[party['party_id']] = target['party_id']
        else:
            canonical[label.casefold()] = party
    world['parties'] = list(canonical.values())
    for effect in world['effects']:
        reference = resolved.get(effect['party_id'])
        was_resolved = effect['party_id'] in resolved
        effect['party_id'] = remap.get(effect['party_id'], effect['party_id'])
        party = next(p for p in world['parties'] if p['party_id'] == effect['party_id'])
        if was_resolved:
            effect['clause_ids'] = list(dict.fromkeys(effect['clause_ids'] + party['clause_ids']))
            if effect['polarity'] in {'ADVERSE', 'BENEFICIAL'}:
                effect['quantities'] = list(dict.fromkeys(effect.get('quantities', []) + party.get('quantities', [])))
                # Render the resolved bearer in the outcome, leaving the exact
                # original pronoun/remnant in source_proposition. This also
                # binds the inherited cardinality locally for native admission.
                old = effect['outcome']
                effect['outcome'] = re.sub(r'\b' + re.escape(reference[0]) + r'\b',
                                           lambda _: reference[1], old, flags=re.I)
                if provenance is not None and effect['outcome'] != old:
                    provenance.append({'atom_id': 'effect:' + effect['effect_id'], 'atom_kind': 'effect',
                        'origin': 'STRUCTURALLY_DERIVED', 'clause_ids': effect['clause_ids'],
                        'evidence': [effect['source_proposition'], reference[1]],
                        'explanation': 'Unique counted antecedent resolves the outcome bearer; original source wording remains unchanged.'})
    return world


def _nominal(token):
    parts = [token] + [c for c in token.children if c.dep_ in
                      {'det', 'amod', 'compound', 'nummod'}]
    return token.doc[min(t.i for t in parts):max(t.i for t in parts)+1].text


def mechanism_from_source(text):
    from parsing_game_S import get_nlp
    doc = get_nlp()(text)
    matches = []
    for verb in doc:
        if verb.dep_ != 'relcl' or verb.lemma_.lower() not in {'switch', 'divert', 'redirect'}:
            continue
        instrument = verb.head
        governing = instrument.head
        if governing.lemma_.lower() not in {'pull', 'push', 'turn', 'press', 'flip'}:
            continue
        if governing.dep_ != 'ROOT' or any(c in governing.sent.text for c in '?"“”'):
            continue
        # A negated, uncertain, attributed or conditional mechanism cannot be
        # silently turned into a certain physical transition on an action branch.
        if any(t.dep_ == 'neg' or t.lower_ in {'if', 'unless', 'may', 'might', 'could', 'perhaps', 'according'}
               for t in governing.sent):
            continue
        objects = [c for c in verb.children if c.dep_ in {'dobj', 'obj'}]
        destinations = [p for c in verb.children if c.dep_ == 'prep' and c.lower_ in {'onto', 'into', 'to'}
                        for p in c.children if p.dep_ == 'pobj']
        if len(objects) != 1 or len(destinations) != 1:
            continue
        end = max(t.i for t in destinations[0].subtree if t.dep_ not in {'punct'})+1
        # A following locative relative clause describes the destination, not
        # an event caused by the instrument.
        tail = doc[verb.i:end].text.split(',')[0]
        matches.append({'instrument': _nominal(instrument), 'process': _nominal(objects[0]),
            'destination': _nominal(destinations[0]), 'transition': tail,
            'predicate': verb.lemma_.lower(), 'action_predicate': governing.lemma_.lower(),
            'sentence': governing.sent.text, 'instrument_head': instrument.lemma_.lower()})
    return (doc, matches[0]) if len(matches) == 1 else (doc, None)


def _branch_condition(world, action, negative):
    """Bind by branch evidence and polarity, never collection order."""
    owned = [e for e in world['effects'] if e['action_id'] == action['action_id']]
    outcome_clauses = {cid for e in owned if e['polarity'] in {'ADVERSE', 'BENEFICIAL'}
                       for cid in e.get('clause_ids', [])}
    matches = [c for c in world.get('conditions', [])
               if (c.get('polarity') == 'NEGATED') == negative
               and outcome_clauses.intersection(c.get('clause_ids', []))]
    return matches[0]['description'] if len(matches) == 1 else ''


def _process_subject(span, process):
    from parsing_game_S import get_nlp
    parsed = get_nlp()(span)
    root = next((t for t in parsed if t.dep_ == 'ROOT'), None)
    subjects = [t for t in root.children if t.dep_ in {'nsubj', 'nsubjpass'}] if root else []
    if len(subjects) != 1 or subjects[0].dep_ == 'nsubjpass':
        return ''
    subject = subjects[0]
    if _nominal(subject).casefold() == process.casefold():
        return 'explicit'
    return 'pronoun' if subject.lower_ == 'it' else ''


def _coordinated_process_outcome(clauses, effect, process):
    """Recover a shared subject from source coordination, not the outcome label."""
    from parsing_game_S import get_nlp
    for clause in clauses:
        if clause['clause_id'] not in effect['clause_ids']:
            continue
        for token in get_nlp()(clause['text']):
            if token.lemma_.lower() != effect.get('predicate') or token.dep_ != 'conj':
                continue
            head = token.head
            if any(c.dep_ == 'neg' for c in list(token.children) + list(head.children)):
                continue
            subjects = [c for c in head.children if c.dep_ == 'nsubj']
            if len(subjects) == 1 and _nominal(subjects[0]).casefold() == process.casefold():
                return True
    return False


def omission_continuation(text, process, harm):
    """Bind the stated default motion to its same-patient omission outcome.

    This is a structural inference, not a new assertion of movement or a claim
    that absence of intervention physically causes it.
    """
    from parsing_game_S import get_nlp
    doc = get_nlp()(text)
    harm_doc = get_nlp()(harm)
    objects = [t for t in harm_doc if t.dep_ in {'dobj', 'obj'}]
    if len(objects) != 1:
        return None
    patient_span = _nominal(objects[0])
    patient = counted_antecedent(text, patient_span) or patient_span
    matches = []
    for sent in doc.sents:
        root = sent.root
        if root.lemma_ != 'head' or any(t.dep_ == 'neg' or t.lower_ in
                {'if', 'unless', 'may', 'might', 'could', 'according'} for t in sent):
            continue
        subjects = [c for c in root.children if c.dep_ == 'nsubj']
        targets = [p for c in root.children if c.lower_ in {'toward', 'towards'}
                   for p in c.children if p.dep_ == 'pobj']
        if len(subjects) != 1 or len(targets) != 1 or '?' in sent.text:
            continue
        if _nominal(subjects[0]).split()[-1].lower() != process.split()[-1].lower():
            continue
        if not patient or _nominal(targets[0]).lower() != patient.lower():
            continue
        places = [p for c in targets[0].children if c.lower_ == 'on'
                  for p in c.children if p.dep_ == 'pobj']
        if len(places) != 1:
            continue
        end = max(t.idx + len(t.text) for t in places[0].subtree if not t.is_punct)
        matches.append({'transition': text[root.idx:end], 'predicate': 'head',
                        'destination': _nominal(places[0]), 'evidence': sent.text})
    return matches[0] if len(matches) == 1 else None


def attach_physical_scene(text, original, provenance=None):
    from blueprint_cloze_chooser import _party, _clause_for
    doc, mechanism = mechanism_from_source(text)
    if not mechanism:
        return original
    # Require a matching action already constructed by the shared builder.
    matched = []
    from parsing_game_S import get_nlp
    for action in original.get('actions', []):
        tokens = get_nlp()(action['intervention'])
        if any(t.lemma_.lower() == mechanism['action_predicate'] for t in tokens) and any(
                t.lemma_.lower() == mechanism['instrument_head'] for t in tokens):
            if not any(t.dep_ == 'neg' for t in tokens):
                matched.append(action['action_id'])
    if len(matched) != 1 or len(original.get('actions', [])) != 2:
        return original
    other = next(a for a in original['actions'] if a['action_id'] not in matched)
    other_tokens = get_nlp()(other['intervention'])
    if not any(t.dep_ == 'neg' for t in other_tokens) or any(
        t.pos_ == 'VERB' and t.lemma_.lower() not in {'do', mechanism['action_predicate']}
        for t in other_tokens):
        return original
    world = deepcopy(original)
    parties = world['parties']
    index = {'n': max([int(p['party_id'][1:]) for p in parties] or [0])}
    instrument = _party(text, parties, index, mechanism['instrument'], role='instrument',
                        construction='conditional_outcome')
    process = _party(text, parties, index, mechanism['process'], role='process',
                     construction='conditional_outcome')
    clauses = segment_source_clauses(text)
    mechanism_host = _clause_for(text, mechanism['transition'])
    retired = set()
    for action in world['actions']:
        owned = [e for e in world['effects'] if e['action_id'] == action['action_id']]
        direct = next((e for e in owned if e['directness'] == 'DIRECT'), None)
        if not direct:
            continue
        # Reuse the builder's bound if-clause as the branch identity. The
        # capability sentence describes the instrument but is not a different
        # choice from the same instrument action in the conditional.
        condition = _branch_condition(world, action, action['action_id'] not in matched)
        parsed = get_nlp()(condition)
        root = next((t for t in parsed if t.dep_ == 'ROOT'), None)
        branch_copy = ''
        if root is not None:
            objects = [c for c in root.children if c.dep_ in {'dobj','obj'}]
            negative = any(c.dep_ == 'neg' for c in root.children)
            if root.lemma_.lower() == mechanism['action_predicate'] and len(objects) == 1 and objects[0].lemma_.lower() == mechanism['instrument_head']:
                nominal = _nominal(objects[0])
                end = condition.find(nominal, root.idx) + len(nominal)
                start = min(t.idx for t in [root]+list(root.children) if t == root or t.dep_ in {'aux','neg'})
                branch_copy = condition[start:end]
            elif root.lemma_.lower() == 'do' and negative and len(world['actions']) == 2:
                end = max(c.idx+len(c.text) for c in [root]+list(root.children) if c == root or c.dep_ == 'neg')
                branch_copy = condition[root.idx:end]
        if branch_copy and branch_copy in text:
            action['intervention'] = ' '.join(branch_copy.split())
            host = _clause_for(text, condition)
            action['clause_ids'] = list(dict.fromkeys(action['clause_ids'] + [host['clause_id']]))
            direct.update(outcome=branch_copy, source_proposition=branch_copy,
                          predicate=root.lemma_.lower(), clause_ids=[host['clause_id']])
            if root.lemma_.lower() == 'do' and negative:
                antecedent = _branch_condition(world, next(a for a in world['actions']
                    if a['action_id'] in matched), False)
                antecedent_doc = get_nlp()(antecedent)
                antecedent_root = next((t for t in antecedent_doc if t.dep_ == 'ROOT'), None)
                objects = [c for c in antecedent_root.children if c.dep_ in {'dobj', 'obj'}] if antecedent_root else []
                if antecedent_root and antecedent_root.lemma_.lower() == mechanism['action_predicate'] and len(objects) == 1:
                    # Display/identity expansion only. The direct effect keeps
                    # its verbatim remnant and the conditional keeps its scope.
                    action['intervention'] = ' '.join((branch_copy + ' ' + antecedent_root.lemma_ + ' ' + _nominal(objects[0])).split())
                    antecedent_host = _clause_for(text, antecedent)
                    action['clause_ids'] = list(dict.fromkeys(action['clause_ids'] + [antecedent_host['clause_id']]))
                    if provenance is not None:
                        provenance.append({'atom_id': 'action:' + action['action_id'], 'atom_kind': 'action',
                            'origin': 'STRUCTURALLY_DERIVED', 'clause_ids': [host['clause_id'], antecedent_host['clause_id']],
                            'evidence': [branch_copy, antecedent],
                            'explanation': 'Negative auxiliary VP ellipsis expands the unique paired intervention; source effect remains verbatim.'})
        retired.add(direct['party_id'])
        direct['party_id'] = instrument
        action['recipient_party_ids'] = [instrument]
        middle = next((e for e in owned if e['source_effect_ids'] == [direct['effect_id']]
                       and e['polarity'] == 'NEUTRAL'), None)
        if not middle:
            continue
        retired.add(middle['party_id'])
        if action['action_id'] in matched:
            middle.update(party_id=process, outcome=mechanism['transition'],
                predicate=mechanism['predicate'], source_proposition=mechanism['transition'],
                effect_kind='PHYSICAL_STATE', quantities=[],
                clause_ids=[mechanism_host['clause_id']],
                derivation_explanation='Explicit instrument relative clause supplies the transition on the selected intervention branch.')
            action['clause_ids'] = list(dict.fromkeys(action['clause_ids']+[mechanism_host['clause_id']]))
            for link in world['causal_links']:
                if link['target_id'] == middle['effect_id']:
                    link['clause_ids'] = [mechanism_host['clause_id']]
                    if link['source_id'] == direct['effect_id'] and link.get('link_relation', link.get('relation')) in {'ENABLES', 'CAUSES'}:
                        link['link_relation' if 'link_relation' in link else 'relation'] = 'CAUSES'
                        if provenance is not None:
                            provenance.append({'atom_id': 'relation:' + str(world['causal_links'].index(link)),
                                'atom_kind': 'relation', 'origin': 'STRUCTURALLY_DERIVED',
                                'clause_ids': link['clause_ids'], 'evidence': [mechanism['sentence']],
                                'explanation': 'The explicit instrument mechanism connects its operation to the stated redirection; not merely branch membership.'})
            # Only the source-bound intervention mechanism is causal here.
            # Do not upgrade arbitrary conditional associations or omission
            # dependencies just because their outcomes share a branch.
            for link_index, link in enumerate(world['causal_links']):
                target = next((e for e in owned if e['effect_id'] == link['target_id']), None)
                if (link['source_id'] != middle['effect_id'] or link.get('link_relation', link.get('relation')) not in {'ENABLES', 'CAUSES'}
                        or not target or target['polarity'] not in {'ADVERSE', 'BENEFICIAL'}
                        or not (_process_subject(target['outcome'], mechanism['process']) == 'explicit'
                                or _coordinated_process_outcome(clauses, target, mechanism['process']))):
                    continue
                link['link_relation' if 'link_relation' in link else 'relation'] = 'CAUSES'
                link['clause_ids'] = list(dict.fromkeys([mechanism_host['clause_id']] + target['clause_ids']))
                if provenance is not None:
                    provenance.append({'atom_id': 'relation:' + str(link_index), 'atom_kind': 'relation',
                        'origin': 'STRUCTURALLY_DERIVED', 'clause_ids': link['clause_ids'],
                        'evidence': [mechanism['sentence'], target['source_proposition']],
                        'explanation': 'The stated redirection and same-process conditional outcome bind the intervention mechanism to its consequence; no outcome is used as a means to its sibling.'})
        else:
            # Reuse the explicit outcome event as a process perspective, with
            # its human harm kept separately. No invented continuation/route.
            harms = [e for e in owned if e['polarity'] == 'ADVERSE']
            if len(harms) == 1:
                harm = harms[0]
                subject = _process_subject(harm['outcome'], mechanism['process'])
                antecedents = [e for e in world['effects'] if e['action_id'] in matched
                    and e['polarity'] == 'ADVERSE'
                    and _process_subject(e['outcome'], mechanism['process']) == 'explicit']
                if subject != 'explicit' and not (subject == 'pronoun' and len(antecedents) == 1):
                    continue
                middle.update(party_id=process, outcome=harm['source_proposition'],
                    source_proposition=harm['source_proposition'], predicate=harm['predicate'],
                    effect_kind='OTHER', quantities=harm.get('quantities', []),
                    modality=harm['modality'], condition_ids=harm.get('condition_ids', []),
                    likelihood_qualifiers=harm.get('likelihood_qualifiers', []),
                    clause_ids=harm['clause_ids'],
                    derivation_explanation='Source-stipulated process event; the separate human-bearer effect records its welfare consequence.')
                continuation = omission_continuation(text, mechanism['process'], harm['outcome'])
                if continuation:
                    from blueprint_allocation_invariants import group_quantity
                    scene_host = _clause_for(text, continuation['evidence'])
                    middle.update(outcome=continuation['transition'],
                        source_proposition=continuation['transition'], predicate=continuation['predicate'],
                        effect_kind='OTHER', quantities=group_quantity(continuation['transition']),
                        clause_ids=list(dict.fromkeys(harm['clause_ids'] + [scene_host['clause_id']])),
                        derivation_explanation='The stated default motion and same-patient omission outcome jointly support this branch continuation; no new route is invented.')
                    action['clause_ids'] = list(dict.fromkeys(action['clause_ids'] + middle['clause_ids']))
                    if provenance is not None:
                        provenance.append({'atom_id': 'effect:' + middle['effect_id'], 'atom_kind': 'effect',
                            'origin': 'STRUCTURALLY_DERIVED', 'clause_ids': middle['clause_ids'],
                            'evidence': [continuation['evidence'], harm['source_proposition'], condition],
                            'explanation': middle['derivation_explanation']})
                if provenance is not None and not continuation:
                    provenance.append({'atom_id': 'effect:' + middle['effect_id'], 'atom_kind': 'effect',
                        'origin': 'STRUCTURALLY_DERIVED', 'clause_ids': list(dict.fromkeys(
                            harm['clause_ids'] + [cid for e in antecedents for cid in e['clause_ids']])),
                        'evidence': [harm['source_proposition']] + [e['source_proposition'] for e in antecedents],
                        'explanation': 'Separate process-agent and human-patient perspectives on the stipulated event; it inherits the named process from the paired branch.'})
                for link in world['causal_links']:
                    if link['target_id'] == middle['effect_id']:
                        link.update(clause_ids=harm['clause_ids'], modality=harm['modality'],
                                    condition_ids=harm.get('condition_ids', []))
    in_use = {e['party_id'] for e in world['effects']} | {
        a['actor_party_id'] for a in world['actions']} | {instrument,process}
    parties[:] = [p for p in parties if p['party_id'] not in retired or p['party_id'] in in_use]
    # Retain relevant scene statements as source assertions, including full
    # can/cannot wording. An ASSERTED claim of inability is not an actual stop.
    selected = []
    process_head = next(t.lemma_.lower() for t in doc if t.text == mechanism['process'].split()[-1])
    for sentence in doc.sents:
        root = sentence.root
        if '?' in sentence.text or any(t.lower_ in {'if', 'unless'} for t in sentence):
            continue
        if sentence.text == mechanism['sentence'] or (
            root.dep_ == 'ROOT' and root.lemma_.lower() in {'head', 'move', 'travel', 'approach', 'stop'}
            and any(t.lemma_.lower() == process_head for t in sentence)):
            selected.append(sentence.text.strip())
    used = {p['proposition_id'] for p in world.get('propositions', [])}
    for sentence in selected:
        if any(p.get('predication') == sentence for p in world.get('propositions', [])):
            continue
        number = len(used)+1
        while 'PR'+str(number) in used:
            number += 1
        ident = 'PR'+str(number); used.add(ident)
        world.setdefault('propositions', []).append({'proposition_id': ident,
            'predication': sentence, 'polarity': 'POSITIVE', 'status': 'ASSERTED',
            'clause_ids': [c['clause_id'] for c in clauses if c['text'] == sentence]})
        # Locations stay OTHER rather than being mistaken for welfare bearers.
        for token in doc:
            if token.sent.text.strip() == sentence and token.dep_ == 'pobj' and token.head.lower_ in {'on', 'onto', 'into'}:
                _party(text, parties, index, _nominal(token), role='location', construction='physical_scene')
    return _resolve_scene_parties(text, world, provenance)
