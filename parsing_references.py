"""Bounded document reference proposals. Never merges mentions or selects facts."""
import json
from pathlib import Path


def add_reference_candidates(doc, package, mentions, candidate, question):
    policy = json.loads((Path(__file__).parent / 'resources/T_reference_policy.json').read_text())
    package['producer']['resources'].append({k: policy[k] for k in ('id', 'version', 'description')})
    nodes = {n['id']: n for n in package['nodes']}
    sentence_ids = {token.i: index for index, sent in enumerate(doc.sents) for token in sent}

    def animacy(token):
        if token.ent_type_ == 'PERSON' or token.lemma_.lower() in policy['human_lemmas']:
            return 'human'
        if token.lemma_.lower() in policy['nonhuman_lemmas']:
            return 'nonhuman'
        return 'unknown'

    def number(token):
        # Singular they and collective readings remain open.
        if token.lower_ in {'they', 'them'} or any(c.dep_ == 'conj' for c in token.children):
            return None
        values = token.morph.get('Number')
        return values[0] if len(values) == 1 else None

    def quantities(token):
        return tuple(c.lower_ for c in token.children if c.dep_ == 'nummod')

    for index in sorted(mentions):
        token = doc[index]
        pronoun = token.lower_ in policy['pronouns'] and token.pos_ == 'PRON'
        definite = any(c.lower_ == 'the' and c.dep_ == 'det' for c in token.children)
        proper = token.pos_ == 'PROPN'
        if not (pronoun or definite or proper):
            continue
        ref = mentions[index]
        evidence = nodes[ref]['evidence_ids']
        options = []
        for earlier in sorted(mentions):
            if earlier >= index:
                break
            antecedent = doc[earlier]
            distance = sentence_ids[index] - sentence_ids[earlier]
            if distance > policy['max_sentence_distance'] or antecedent.pos_ not in {'NOUN', 'PROPN'}:
                continue
            if not pronoun and antecedent.lemma_.lower() != token.lemma_.lower():
                continue
            if proper and nodes[mentions[earlier]]['label'].casefold() != nodes[ref]['label'].casefold():
                continue
            if number(token) and number(antecedent) and number(token) != number(antecedent):
                continue
            if quantities(token) and quantities(antecedent) and quantities(token) != quantities(antecedent):
                continue
            expected = policy['pronouns'].get(token.lower_, 'unknown') if pronoun else animacy(token)
            actual = animacy(antecedent)
            if expected != 'unknown' and actual != 'unknown' and expected != actual:
                continue
            # Personal object pronouns cannot simply co-refer with a local subject.
            # Full binding, reflexives, cataphora and discourse entities are deferred.
            if pronoun and token.head == antecedent.head and antecedent.dep_ in {'nsubj', 'nsubjpass'}:
                continue
            score = 0.35 + 0.2 / (1 + distance)
            score += 0.15 if expected != 'unknown' and actual == expected else 0.0
            score += 0.1 if antecedent.dep_ in {'nsubj', 'nsubjpass'} else 0.0
            score += 0.2 if not pronoun else 0.0
            item = candidate('SAME_REFERENT', dict(mention_a=ref, mention_b=mentions[earlier]), None,
                             list(dict.fromkeys(evidence + nodes[mentions[earlier]]['evidence_ids'])))
            item['provenance'] = [dict(producer='reference_candidates', version='T1',
                method='bounded_number_animacy_recency_role_ranking', resource_ids=[policy['id']])]
            item['assessment'] = dict(status='unresolved', score=dict(value=score,
                kind='uncalibrated_score', source=policy['id'], calibration_id=None))
            options.append(item)
        # First occurrences of names are not themselves unresolved anaphora.
        if proper and not options:
            continue
        options.sort(key=lambda c: (-c['assessment']['score']['value'], c['id']))
        if options:
            top = options[0]['assessment']['score']['value']
            runner = options[1]['assessment']['score']['value'] if len(options) > 1 else 0.0
            if top >= policy['minimum_preference_score'] and top - runner >= policy['minimum_preference_margin']:
                options[0]['assessment']['status'] = 'preferred'
        # Two antecedent mentions may themselves co-refer. These alternatives
        # are therefore NOT declared mutually exclusive identity facts.
        if options:
            package['choice_sets'].append(dict(id=f"choice{len(package['choice_sets'])}",
                kind='interpretation', candidate_ids=[c['id'] for c in options],
                selection_rule='any_subset', exhaustive=False, evidence_ids=evidence))
        blockers = [c['id'] for c in package['candidates']
                    if c['type'] == 'PARTICIPANT' and c['arguments'].get('mention') == ref]
        question('reference', evidence, options,
                 'Which earlier mention, if any, has the same referent? Ranked candidates are not identity decisions.', blockers)
