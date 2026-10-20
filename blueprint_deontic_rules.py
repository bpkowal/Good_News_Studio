"""Source-aligned bundles for explicit institutional rule complements.

This is a narrow producer, not a normative adjudicator or coreference resolver.
Scopes remain inside each copied governed proposition in world schema 1.4.
"""
import re

from z10_world_model_adapter import segment_source_clauses


_AUTHORITY = {'rule', 'policy', 'law', 'code', 'protocol', 'regulation',
              'guideline', 'contract', 'order'}
_FORCE = {'require': 'obligation', 'oblige': 'obligation',
          'allow': 'permission', 'permit': 'permission',
          'forbid': 'prohibition', 'prohibit': 'prohibition'}


def _nominal(text, token):
    parts = [token]
    for child in token.children:
        if child.dep_ in {'det', 'amod', 'compound', 'poss', 'nummod'}:
            parts.extend(child.subtree)
    return text[min(t.idx for t in parts):max(t.idx + len(t.text) for t in parts)]


def source_rule_bundles(text):
    """Keep an operator with its own complement; never zip independent lists.

    Only matrix institutional rule statements with a verbal complement are
    covered here. Mechanistic 'allows', numeric 'requires', attributed rules,
    questions and negations of an obligation are not positive rule assertions.
    """
    from parsing_game_S import get_nlp
    doc = get_nlp()(text)
    clauses = segment_source_clauses(text)
    bundles = []
    for operator in doc:
        force = _FORCE.get(operator.lemma_.lower())
        if not force or operator.dep_ != 'ROOT' or '?' in operator.sent.text:
            continue
        if any(t.dep_ == 'neg' for t in operator.children):
            continue
        authorities = [t for t in operator.children if t.dep_ == 'nsubj'
                       and t.lemma_.lower() in _AUTHORITY]
        complements = [t for t in operator.children if t.dep_ in {'xcomp', 'ccomp'}
                       and t.pos_ == 'VERB']
        if len(authorities) != 1 or len(complements) != 1:
            continue
        content = complements[0]
        tokens = [t for t in content.subtree if not t.is_punct and not t.is_space
                  and not (t.dep_ == 'aux' and t.lower_ == 'to')]
        start = min(t.idx for t in tokens)
        end = max(t.idx + len(t.text) for t in tokens)
        governed = text[start:end]
        # A second coordinated rule cannot safely share this operator bundle.
        if any(t.dep_ == 'conj' and t.lemma_.lower() in _FORCE for t in tokens):
            continue
        bearers = [t for t in operator.children if t.dep_ in {'dobj', 'obj'}]
        local_subjects = [t for t in content.children if t.dep_ in {'nsubj', 'nsubjpass'}]
        bearer = _nominal(text, local_subjects[0]) if len(local_subjects) == 1 else (
            _nominal(text, bearers[0]) if len(bearers) == 1 else '')
        clause_ids = [row['clause_id'] for row in clauses
                      if row['start'] < end and row['end'] > operator.idx]
        scopes = []
        for token in tokens:
            if token.dep_ != 'mark' or token.lower_ not in {'while', 'after', 'before', 'if', 'unless'}:
                continue
            scoped = [t for t in token.head.subtree if not t.is_punct and not t.is_space]
            lo, hi = min(t.idx for t in scoped), max(t.idx + len(t.text) for t in scoped)
            scopes.append({'marker': token.text, 'text': text[lo:hi], 'start': lo, 'end': hi})
        bundles.append({
            'operator': operator.text, 'force': force, 'governed_action': governed,
            'content_polarity': 'negative' if any(t.dep_ == 'neg' for t in content.children) else 'positive',
            'bearer': bearer, 'authority': _nominal(text, authorities[0]),
            'operator_start': operator.idx, 'operator_end': operator.idx + len(operator.text),
            'content_start': start, 'content_end': end, 'clause_ids': clause_ids,
            'scope': scopes, 'origin': 'SOURCE_ASSERTED',
            'producer': 'source_dependency_rule_complement',
            'unresolved_references': [t.text for t in tokens if t.pos_ == 'PRON'],
        })
    return bundles


def rule_force(words):
    """Classify this operator span, never an unrelated modal elsewhere."""
    if re.search(r"\b(?:does|do|did)\s+not\s+(?:require|allow|permit)\b|\bnot\s+(?:required|obligated)\b", words, re.I):
        return 'unresolved'
    if re.search(r'\b(?:must not|may not|shall not|not allowed|not permitted|forbid\w*|prohibit\w*)\b', words, re.I):
        return 'prohibition'
    if re.search(r'\b(?:must|shall|should|require[sd]?|obligat\w*|oblige[sd]?)\b', words, re.I):
        return 'obligation'
    if re.search(r'\b(?:may|allow(?:s|ed)?|permit\w*)\b', words, re.I):
        return 'permission'
    return 'unresolved'
