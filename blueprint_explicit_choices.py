"""Bounded source constructions: binary question + local modal outcome.

No scenario names or rule vocabulary select a topology. Each association needs
an explicit action anaphor or an explicit duration/delay reference in the source.
"""
import re

from z10_world_model_adapter import segment_source_clauses


def extract_explicit_choices(text):
    from parsing_game_S import get_nlp
    from blueprint_deontic_rules import _nominal
    doc = get_nlp()(text)
    clauses = segment_source_clauses(text)

    def copy(tokens):
        tokens = [t for t in tokens if not t.is_punct and not t.is_space]
        return text[min(t.idx for t in tokens):max(t.idx + len(t.text) for t in tokens)] if tokens else ''

    def evidence(span):
        hits = [c['clause_id'] for c in clauses if span in c['text']]
        if hits or not span:
            return hits
        start = text.find(span)
        return [c['clause_id'] for c in clauses
                if start >= 0 and c['start'] < start + len(span) and c['end'] > start]

    def modal_risk(token):
        return (any(t.lower_ in {'may', 'might', 'could'} for t in token.children if t.dep_ == 'aux')
                and not any(t.dep_ == 'neg' for t in token.children))

    def same_object(token, expected):
        objects = [t for t in token.children if t.dep_ in {'dobj', 'obj'}]
        def name(value):
            return re.sub(r'^(?:a|an|the)\s+', '', ' '.join(value.casefold().split()))
        return len(objects) == 1 and name(_nominal(text, objects[0])) == name(expected)

    questions = [s for s in doc.sents if '?' in s.text]
    if len(questions) != 1:
        return None
    question = questions[0]
    root = question.root
    ors = [t for t in root.children if t.dep_ == 'cc' and t.lower_ == 'or']
    subjects = [t for t in root.children if t.dep_ == 'nsubj']
    if len(ors) != 1 or len(subjects) != 1 or subjects[0].pos_ != 'PROPN':
        return None
    other = [t for t in root.children if t.dep_ == 'conj' and t.idx > ors[0].idx]
    if len(other) != 1 or root.pos_ != 'VERB' or other[0].pos_ != 'VERB':
        return None
    if any(t.dep_ == 'neg' for head in (root, other[0]) for t in head.children):
        return None
    if any(t.pos_ == 'VERB' and t != root and t.idx < ors[0].idx for t in root.subtree):
        return None
    if any(t.pos_ == 'VERB' and t != other[0] and t.lemma_ != 'wait' for t in other[0].subtree):
        return None
    actor = _nominal(text, subjects[0])
    branches = []
    for number, action in enumerate([root, other[0]]):
        action_tokens = [t for t in action.subtree if t.idx >= action.idx
                         and (number or t.idx < ors[0].idx)]
        label = copy(action_tokens)
        objects = [t for t in action.children if t.dep_ in {'dobj', 'obj'}]
        if len(objects) != 1:
            return None
        branch = {'actor': actor, 'action': label, 'predicate': action.lemma_,
                  'process': _nominal(text, objects[0]), 'action_clause_ids': evidence(label),
                  'outcomes': [], 'duration': None}
        # A local 'doing so may ...' explicitly refers back to this action.
        for antecedent in doc:
            if antecedent.sent == question or antecedent.lemma_ != action.lemma_:
                continue
            people = [t for t in antecedent.children if t.dep_ == 'nsubj']
            if (antecedent.dep_ != 'ROOT' or len(people) != 1
                    or any(t.lower_ in {'if', 'unless', 'whether'} for t in antecedent.sent)
                    or ' '.join(_nominal(text, people[0]).split()).casefold() != ' '.join(actor.split()).casefold()
                    or not same_object(antecedent, branch['process'])
                    or not any(t.lower_ in {'can', 'may', 'could'} for t in antecedent.children)
                    or any(t.dep_ == 'neg' for t in antecedent.children)):
                continue
            for risk in antecedent.children:
                anaphors = [t for t in risk.children if t.dep_ == 'csubj'
                            and t.lemma_ == 'do' and any(c.lower_ == 'so' for c in t.children)]
                patients = [t for t in risk.children if t.dep_ in {'dobj', 'obj'}]
                if risk.dep_ != 'conj' or not modal_risk(risk) or len(anaphors) != 1 or len(patients) != 1:
                    continue
                span = copy(risk.subtree)
                if any(t.lower_ == 'or' for t in risk.subtree):
                    continue
                branch['outcomes'].append({
                    'text': span, 'patient': _nominal(text, patients[0]),
                    'predicate': risk.lemma_, 'clause_ids': evidence(span),
                    'likelihood': next(t.text for t in risk.children if t.dep_ == 'aux'),
                    'binding': 'explicit_action_anaphor', 'evidence': antecedent.sent.text.strip(),
                })
                antecedent_object = next(t for t in antecedent.children if t.dep_ in {'dobj', 'obj'})
                parts = [antecedent] + list(antecedent_object.subtree) + [
                    t for t in antecedent.children if t.dep_ == 'advmod']
                branch['source_action'] = copy(parts)
                branch['source_action_clause_ids'] = evidence(branch['source_action'])
        # 'Doing X takes N minutes; Y may die during that delay' supplies an
        # explicit duration association, not proof that X directly causes death.
        for duration in doc:
            if duration.lemma_ != 'take' or duration.sent == question:
                continue
            acts = [t for t in duration.children if t.dep_ == 'csubj' and t.lemma_ == action.lemma_]
            amounts = [t for t in duration.children if t.dep_ in {'dobj', 'obj'}
                       and t.lemma_ in {'second', 'minute', 'hour', 'day'}
                       and any(c.dep_ == 'nummod' for c in t.children)]
            if (len(acts) != 1 or len(amounts) != 1 or not same_object(acts[0], branch['process'])
                    or any(t.dep_ in {'neg', 'aux'} for t in duration.children)
                    or any(t.dep_ in {'ccomp', 'xcomp'} for t in duration.ancestors)):
                continue
            semi = [t for t in duration.sent if t.text == ';' and t.idx > duration.idx]
            if len(semi) != 1:
                continue
            duration_text = copy([t for t in duration.sent if acts[0].idx <= t.idx < semi[0].idx])
            risks = [t for t in duration.sent if t.idx > semi[0].idx and modal_risk(t)
                     and any(c.dep_ == 'prep' and c.lower_ == 'during'
                             and any(d.lemma_ == 'delay' and any(a.lower_ == 'that' for a in d.children)
                                     for d in c.children) for c in t.children)]
            if len(risks) != 1:
                continue
            risk = risks[0]
            patients = [t for t in risk.children if t.dep_ == 'nsubj']
            if len(patients) != 1 or any(t.pos_ == 'PRON' for t in patients):
                continue
            span = copy([t for t in risk.subtree if t.idx > semi[0].idx])
            branch['duration'] = {'text': duration_text, 'quantity': _nominal(text, amounts[0]),
                                  'clause_ids': evidence(duration_text), 'action': copy(acts[0].subtree)}
            branch['outcomes'].append({
                'text': span, 'patient': _nominal(text, patients[0]), 'predicate': risk.lemma_,
                'clause_ids': evidence(span), 'binding': 'explicit_duration_anaphor',
                'likelihood': next(t.text for t in risk.children if t.dep_ == 'aux'),
                'evidence': duration.sent.text.strip(),
            })
        branches.append(branch)
    return {'producer': 'source_dependency_choice_outcomes', 'question': question.text.strip(),
            'branches': branches, 'complete': all(len(b['outcomes']) == 1 for b in branches),
            'exclusivity': 'UNKNOWN', 'exhaustive': False}


def build_explicit_choice_world(text, inventory):
    from blueprint_cloze_chooser import _effect, _link, _party, _world, _outcome_reading
    parties, actions, effects, links = [], [], [], []
    counter = {'n': 0}
    actor = _party(text, parties, counter, inventory['branches'][0]['actor'], role='actor')
    for i, branch in enumerate(inventory['branches']):
        action_id = f'A{i}'
        direct_id = f'E{len(effects)+1}'
        process = _party(text, parties, counter, branch['process'], role='process', construction='physical_state')
        outcome = branch['outcomes'][0]
        patient = _party(text, parties, counter, outcome['patient'], role='bearer')
        intervention = branch.get('source_action') or branch['action']
        direct_text = (branch['duration']['action'] if branch['duration'] else intervention)
        direct_clauses = (branch['duration']['clause_ids'] if branch['duration']
                          else branch['source_action_clause_ids'])
        action_host = {'clause_id': direct_clauses[0], 'text': direct_text}
        direct = _effect(direct_id, action_id, process, direct_text, branch['predicate'],
                         'NEUTRAL', 'DIRECT', 'INTERVENTION', 'CERTAIN', action_host, [], [], source=direct_text)
        effects.append(direct)
        parent_id = direct_id
        branch_effects = [direct_id]
        if branch['duration']:
            duration = branch['duration']
            parent_id = f'E{len(effects)+1}'
            host = {'clause_id': duration['clause_ids'][0], 'text': duration['text']}
            delay = _effect(parent_id, action_id, process, duration['text'], 'take',
                            'NEUTRAL', 'DOWNSTREAM', 'OTHER', 'CERTAIN', host,
                            [direct_id], [], source=duration['text'])
            delay['quantities'] = [duration['quantity']]
            delay['temporal_qualifiers'] = [duration['quantity']]
            effects.append(delay)
            branch_effects.append(parent_id)
            links.append(_link(action_id, direct_id, parent_id, 'CERTAIN', host))
        outcome_id = f'E{len(effects)+1}'
        reading = _outcome_reading(outcome['text'])
        # Exposure to a source-described hazard is a risk, not an invented injury.
        if outcome['predicate'] == 'expose':
            adverse = bool(re.search(r'\b(?:hazardous|toxic|poisonous|harmful|dangerous)\b', outcome['text'], re.I))
            reading.update(predicate='expose', kind='OTHER', polarity='ADVERSE' if adverse else 'NEUTRAL')
        elif reading['predicate'] == 'outcome':
            reading['predicate'] = outcome['predicate']
        host = {'clause_id': outcome['clause_ids'][0], 'text': outcome['text']}
        effect = _effect(outcome_id, action_id, patient, outcome['text'], reading['predicate'],
                         reading['polarity'], 'DOWNSTREAM', reading['kind'], 'POSSIBLE', host,
                         [parent_id], [outcome['likelihood']], source=outcome['text'])
        # The parent association is specified; uncertainty belongs to the
        # outcome event. There is no separate stochastic gate in this text.
        # Do not invent a condition that points back to its own welfare outcome.
        link = _link(action_id, parent_id, outcome_id, 'CERTAIN', host)
        effects.append(effect)
        branch_effects.append(outcome_id)
        links.append(link)
        actions.append({'action_id': action_id, 'intervention': intervention,
                        'actor_party_id': actor, 'recipient_party_ids': [process],
                        'effect_ids': branch_effects,
                        'clause_ids': list(dict.fromkeys(branch['action_clause_ids'] + outcome['clause_ids']))})
    return _world(parties, actions, effects, links)
