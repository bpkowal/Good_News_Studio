"""Bounded, observational construction cues; never a semantic completeness claim."""
import parsing_game_S as s
import parsing_game_Z5 as z5

GENERIC_GAP = 'Review unrepresented semantics and scope, including attribution, choice and entailment.'
DESCRIPTIONS = {
    'stripping': 'Stripping: identify the missing predicate, contrast role, unchanged participants and negation scope.',
    'verb_phrase_ellipsis': 'Verb-phrase ellipsis: the auxiliary/remnant lacks an exported predicate and participant reconstruction; resolve its antecedent and local scope.',
    'gapping': 'Possible gapping: identify the omitted predicate and align both remaining participants with their antecedent roles.',
    'sluicing': 'Possible sluicing: recover the question content and antecedent of the wh-remnant; do not treat it as a complete ordinary object.',
}


def construction_sites(doc):
    sites = []
    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_space and not t.is_punct]
        for index, token in enumerate(tokens):
            later = tokens[index + 1:]
            if token.lower_ == 'not' and index and tokens[index - 1].lower_ in {'but', 'and'}:
                if later and later[0].lower_ != 'only' and not any(t.pos_ in {'VERB', 'AUX'} for t in later):
                    sites.append(('stripping', [token] + later, sent))
            if (token.pos_ in {'AUX', 'VERB'} and token.lemma_ in {'do', 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must'}
                    and not any(c.pos_ == 'VERB' for c in token.children)
                    and (token.head == token or token.dep_ in {'conj', 'advcl'})
                    and not any(t.pos_ in {'VERB', 'AUX', 'NOUN', 'PROPN'} for t in later)):
                sites.append(('verb_phrase_ellipsis', [token] + later, sent))
            if token.lower_ in {'and', 'but'} and later and not any(t.pos_ in {'VERB', 'AUX'} for t in later):
                heads = [t for t in later if t.pos_ in {'NOUN', 'PROPN', 'PRON'} and t.dep_ not in {'compound', 'poss'}]
                nominal_pair = len(heads) >= 2 or (len(later) >= 2 and later[0].pos_ == 'PROPN'
                    and later[1].pos_ == 'NOUN' and later[0].dep_ == 'compound')
                if nominal_pair and not any(t.lower_ == 'not' for t in later):
                    sites.append(('gapping', [token] + later, sent))
                elif len(heads) == 1 and later[-1].lower_ in {'too', 'either'}:
                    sites.append(('stripping', [token] + later, sent))
        if (tokens and tokens[-1].lower_ in {'who', 'what', 'where', 'when', 'why', 'how'}
                and any(t.lemma_ in {'know', 'wonder', 'ask', 'remember'} for t in tokens[:-1])):
            sites.append(('sluicing', [tokens[-1]], sent))
    return sites


def add_construction_questions(text, package):
    doc = s.get_nlp()(text)
    handled = set()
    for kind, tokens, sent in construction_sites(doc):
        support = [z5._add_evidence(package, text, tokens)]
        # Narrow the generic notice while keeping partial coverage and all
        # source evidence. A cue is a review request, not an asserted analysis.
        generic = next((q for q in package['open_questions'] if q['question'] == GENERIC_GAP
                        and any(e['id'] in q['evidence_ids'] and e['start'] == sent.start_char
                                and e['end'] == sent.end_char for e in package['evidence'])), None)
        if generic is not None:
            generic.update(evidence_ids=support, question=DESCRIPTIONS[kind])
        else:
            package['open_questions'].append(dict(id=z5._next_id(package['open_questions'], 'q'),
                kind='missing_representation', evidence_ids=support, candidate_ids=[],
                question=DESCRIPTIONS[kind], blocking_for=[]))
        handled.add(kind)
    if handled:
        package['coverage']['limitations'].append('construction_cues_are_bounded_and_do_not_exhaust_semantic_gaps')
