"""Lossless recognition cards from an already admitted world, never new events.

The native integration calls supported_records only after restoring admission.
Binding these cards is exact: source rules cannot license an unconditional event
or arbitrary paraphrases merely because they share participants or vocabulary.
"""
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import re


def explicit_probability(attributes):
    """Known probability is distinct from certain occurrence; return a unit mass."""
    if attributes.get('modality') == 'CERTAIN':
        return 'CERTAIN'
    if attributes.get('modality') != 'PROBABILISTIC':
        return 'UNKNOWN'
    # Overall and local probabilities must not be conflated or multiplied.
    if attributes.get('overall_likelihood_qualifiers'):
        return 'UNKNOWN'
    masses = set()
    for qualifier in attributes.get('likelihood_qualifiers') or []:
        match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:chance|probability)?\s*', str(qualifier), re.I)
        if not match:
            return 'UNKNOWN'
        try:
            percentage = Decimal(match[1])
        except InvalidOperation:
            return 'UNKNOWN'
        if not 0 <= percentage <= 100:
            return 'UNKNOWN'
        with localcontext() as context:
            context.prec = max(28, len(match[1]) + 3)
            masses.add(percentage / 100)
    if len(masses) != 1:
        return 'UNKNOWN'
    # Decimal masses also avoid the native parser's ambiguous <=1 percent case.
    mass = format(next(iter(masses)), 'f')
    return mass if len(mass) <= 24 else 'UNKNOWN'


def normalized(text):
    return ' '.join(str(text).casefold().split()).rstrip('. ')


def survivor_count_projection(attributes, outcome):
    """Count one explicitly named person; never assign one unit of welfare."""
    label = str(attributes.get('party_label', ''))
    if (attributes.get('party_kind') != 'PERSON' or not attributes.get('party_id')
            or not re.fullmatch(r'[A-Z][a-z]+(?: [A-Z][a-z]+)*', label)
            or normalized(attributes.get('affected_subject', label)) != normalized(label)):
        return {}
    match = re.fullmatch(re.escape(label) + r' has a (\d+(?:\.\d+)?)% chance of survival\.?', str(outcome))
    if not match or explicit_probability(attributes) == 'UNKNOWN':
        return {}
    return {'endpoint': 'SURVIVOR_COUNT', 'amount': 1, 'party_id': attributes['party_id'],
            'basis': 'ADMITTED_SINGLE_PERSON_IDENTITY', 'source_outcome': outcome,
            'welfare_magnitude': 'UNKNOWN', 'duration': 'UNKNOWN'}


def supported_records(world, scenario):
    """Project quantity, constraint and relation evidence, keeping its full scope.

    Only admission-backed endpoints and verbatim source excerpts are eligible.
    Resource cards use admitted RESOURCE identities and their own source evidence;
    source quantity alone never establishes exclusivity. Source rules must be
    backed by a same-action link and a SOURCE_STIPULATED_CAUSAL target.
    """
    effects = {e['effect_id']: e for e in world.get('effects', [])}
    cards = {}

    def evidence(rows):
        return [p for p in rows if isinstance(p, dict) and p.get('excerpt')
                and p['excerpt'] in scenario and p.get('clause_id')]

    def add(kind, claim, proof, source, aliases=()):
        identity = kind + '|' + normalized(claim)
        ident = 'PROP:WORLD:RECOGNITION:' + hashlib.sha256(identity.encode()).hexdigest()[:16]
        card = cards.setdefault(ident, {'proposition_id': ident, 'kind': kind,
            'claim': claim, 'aliases': list(aliases), 'support_ids': [],
            'source_evidence': [], 'scope': 'SOURCE_RULE' if kind == 'RULE' else 'SOURCE_CONSTRAINT'})
        card['support_ids'] = sorted(set(card['support_ids']) | set(proof))
        for row in source:
            if row not in card['source_evidence']:
                card['source_evidence'].append(dict(row))

    # Admitted scene propositions carry whole source statements, including
    # can/cannot and relative-clause wording. Recognize that statement verbatim;
    # do not promote its embedded modal content into an occurrence effect.
    for proposition in world.get('propositions', []):
        claim = proposition.get('predication', '')
        if proposition.get('status') != 'ASSERTED' or not claim or not proposition.get('clause_ids'):
            continue
        start = scenario.find(claim)
        if start < 0 or (scenario[:start].rstrip() and scenario[:start].rstrip()[-1] not in '.!?'):
            continue
        end = start + len(claim)
        if claim[-1] not in '.!?' or (end < len(scenario) and not scenario[end].isspace()):
            continue
        proof = 'SOURCE_SPAN:' + hashlib.sha256(f'{start}:{end}:{claim}'.encode()).hexdigest()[:16]
        add('SCENE_ASSERTION', claim, [proposition['proposition_id'], proof],
            [{'clause_id': proof, 'excerpt': claim, 'start': start, 'end': end}])
        card = next(c for c in cards.values() if c['kind'] == 'SCENE_ASSERTION' and c['claim'] == claim)
        card.update(scope='SOURCE_SCENE_STATEMENT', embedded_occurrence='NOT_INFERRED')

    # Inventory is independent of transfer/choice topology. A resource's own
    # provenance supports its stock statement even when nobody transfers it.
    for party in world.get('parties', []):
        if party.get('kind') != 'RESOURCE':
            continue
        for row in evidence(party.get('provenance', [])):
            text = row['excerpt']
            quantity = re.fullmatch(r'(.+?)\s+(?:has|have)\s+((?:\d+|one|two|three|four|five)\s+.+?)\.?', text, re.I)
            if not quantity or re.search(r'\b(if|might|may|not|reports?|said|says?|claims?|believes?|perhaps)\b|[\"“”]', text, re.I):
                continue
            resource = quantity[2].rstrip('.')
            if normalized(party.get('label')) == normalized(resource):
                add('QUANTITY', text, [party['party_id']], [row], [resource])

    for effect in effects.values():
        if effect.get('derivation_operation') != 'EXCLUSIVE_ALLOCATION_COMPLEMENT':
            continue
        parents = [effects.get(i) for i in effect.get('source_effect_ids', [])]
        if len(parents) != 1 or not parents[0] or parents[0].get('effect_kind') != 'RESOURCE_TRANSFER':
            continue
        if parents[0].get('action_id') != effect.get('action_id'):
            continue
        proof = [effect['effect_id'], parents[0]['effect_id']]
        sources = evidence(effect.get('provenance', []))
        for row in sources:
            text = row['excerpt']
            if re.search(r'\bbut not both\b', text, re.I):
                add('EXCLUSIVITY', text, proof, [row])

    for link in world.get('causal_links', []):
        source = effects.get(link.get('source_id'))
        target = effects.get(link.get('target_id'))
        if not source or not target or source.get('action_id') != target.get('action_id'):
            continue
        if target.get('derivation_operation') != 'SOURCE_STIPULATED_CAUSAL':
            continue
        if source['effect_id'] not in target.get('source_effect_ids', []):
            continue
        for row in evidence(link.get('provenance', [])):
            # Only explicit rule wording; preserve modal/negative wording intact.
            if re.search(r'\bif\b|\bwho\b.+\b(?:will|may|might)\b', row['excerpt'], re.I):
                add('RULE', row['excerpt'], [source['effect_id'], target['effect_id']], [row])
    # Stated commitment occurrence is distinct from performance of its content.
    # Recognize only standalone affirmative past-tense source statements with
    # admitted named participants; quotes, attribution and conditional promises
    # remain outside this deliberately bounded construction.
    persons = {p['label']: p['party_id'] for p in world.get('parties', [])
               if p.get('kind') == 'PERSON' and re.fullmatch(r'[A-Z][a-z]+(?: [A-Z][a-z]+)*', p.get('label', ''))}
    for match in re.finditer(r'(?:^|(?<=\.))\s*([^.!?]+[.!?]?)', scenario):
        sentence = match[1].strip()
        for actor, actor_id in persons.items():
            for recipient, recipient_id in persons.items():
                if re.fullmatch(re.escape(actor)+r' promised '+re.escape(recipient)+r' (?:that .+|to .+)\.', sentence):
                    start = scenario.find(sentence, match.start())
                    proof = 'SOURCE_SPAN:' + hashlib.sha256(f'{start}:{start+len(sentence)}:{sentence}'.encode()).hexdigest()[:16]
                    add('COMMITMENT_OCCURRENCE', sentence, [actor_id, recipient_id, proof],
                        [{'clause_id': proof, 'excerpt': sentence, 'start': start, 'end': start+len(sentence)}])
                    card = next(c for c in cards.values() if c['claim'] == sentence)
                    card.update(scope='SOURCE_COMMITMENT_OCCURRENCE', content_occurrence='NOT_ESTABLISHED',
                                normative_force='NOT_ASSESSED')
    return list(cards.values())


def component_support(claim, cards):
    """Expose supported parts without licensing a compound conclusion.

    This is bookkeeping, not a derivation engine. Even when all components are
    recognized, a 'so' assertion still needs its own supported relation.
    """
    parts = [p.strip(' ()') for p in re.split(r';|\s*\(so\s+|\s+so\s+', claim, flags=re.I) if p.strip(' ()')]
    result = []
    for part in parts:
        matches = [c['proposition_id'] for c in cards
                   if normalized(part) in {normalized(c['claim']), *(normalized(a) for a in c.get('aliases', []))}]
        fragments = []
        if not matches and not re.search(r'\b(?:not|no|never|may|might|if|denies?)\b', part, re.I):
            for card in cards:
                if card.get('kind') != 'QUANTITY':
                    continue
                for alias in card.get('aliases', []):
                    quantity = re.fullmatch(r'(.+?) of (.+)', alias)
                    if not quantity:
                        continue
                    # Recover a referenced quantity inside a strengthened stock
                    # claim, without licensing 'only', ownership, or modifiers
                    # such as 'life-saving'. Unknown wording stays explicit.
                    pattern = r'\b' + re.escape(quantity[1]) + r'\s+of\s+(?:the\s+)?(?:[\w-]+\s+){0,2}' + re.escape(quantity[2]) + r'\b'
                    if re.search(pattern, part, re.I):
                        fragments.append({'claim': alias, 'support_ids': [card['proposition_id']]})
        result.append({'claim': part, 'support_ids': matches,
                       'status': 'SUPPORTED_COMPONENT' if matches else
                                 'PARTIALLY_SUPPORTED_COMPONENT' if fragments else 'UNRESOLVED_COMPONENT',
                       'supported_fragments': fragments})
    return result
