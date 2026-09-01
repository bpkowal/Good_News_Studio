"""Domain-neutral party and outcome vocabulary for canonical action records.

The older role extractor recognised only the oxygen-grid scenario's patients and
refugees, so every other dilemma produced empty structured fields. Nothing here
names a domain: the party vocabulary is derived from the scenario under analysis
and then used as a closed vocabulary by the role validators.

Division of labour:

* Interpretation (which party benefits, which is harmed) belongs to the model.
* Preservation (cardinality, provenance, no invented parties) belongs here.

Deterministic obligations are deliberately *sound rather than complete*. A
binding is only asserted for unambiguous prose such as "the three elderly
drown". Hedged or probabilistic constructions ("only a 60% chance of surviving")
are left to the interpreting model, because forcing an obligation there would
manufacture false repair verdicts on exactly the language that carries the
moral weight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "PartyMention",
    "party_identity",
    "parties_compatible",
    "extract_party_registry",
    "merge_party_mentions",
    "RoleBinding",
    "relational_role_bindings",
    "GroundedEffect",
    "extract_grounded_effects",
    "RoleIssue",
    "ROLE_FIELDS",
    "ROLE_UNRESOLVED",
    "DIRECTION_ADVERSE",
    "DIRECTION_BENEFICIAL",
    "RELATION_DOWNSTREAM_BENEFIT",
    "RELATION_FOREGONE_BENEFIT",
    "PARTY_KIND_FUTURE_POPULATION",
    "resolve_party",
    "validate_role_assignment",
    "completeness_from_role_issues",
]

ROLE_FIELDS = ("beneficiaries", "harmed")


_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "hundred": "100", "thousand": "1000",
    "single": "1", "lone": "1", "sole": "1",
}

# Tokens too generic to identify a group on their own. "three lives saved" and
# "three elderly survivors" are the same people, and "future life" must not bind
# to either, so a weak token can never anchor a match by itself.
_WEAK_TOKENS = frozenset({
    "life", "person", "individual", "human", "soul", "body", "one",
    "victim", "casualty", "party", "group", "member",
})

# Verb forms that end in "s" and would otherwise be mistaken for plural heads.
_NON_NOUN_HEADS = frozenset({
    "has", "is", "was", "does", "goes", "gives", "offers", "carries", "secures",
    "means", "makes", "takes", "leaves", "remains", "involves", "requires",
    "needs", "gets", "puts", "runs", "holds", "keeps", "lets", "says", "seems",
    "appears", "becomes", "comes", "faces", "loses", "wins", "cost", "costs",
    "this", "thus", "its", "his", "hers", "ours", "yours", "theirs", "us",
})

# Units of measure are never parties, even when they appear as "10 minutes".
_MEASURE_NOUNS = frozenset({
    "minute", "minutes", "hour", "hours", "second", "seconds", "day", "days",
    "week", "weeks", "month", "months", "year", "years", "decade", "decades",
    "percent", "percentage", "point", "points", "degree", "degrees",
    "mile", "miles", "meter", "meters", "metre", "metres", "kilometer",
    "kilometers", "dollar", "dollars", "euro", "euros", "unit", "units",
    "trip", "trips", "time", "times", "chance", "chances", "odds",
    "step", "steps", "stage", "stages", "option", "options", "action",
    "actions", "policy", "policies", "hous", "houses",
})

_PARTY_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "from", "with", "for",
    "and", "or", "but", "that", "this", "these", "those", "their", "its",
    "his", "her", "them", "they", "who", "whom", "which", "while", "all",
    "only", "other", "another", "remaining", "immediate", "potential",
    "guaranteed", "future", "delayed", "current", "further", "more", "most",
})

# Domain-neutral outcome polarity. Extended by scenario-specific verbs only
# through the model-proposed roles, never by editing this table per dilemma.
_ADVERSE_OUTCOME = re.compile(
    r"\b(?:die|dies|died|dying|death|deaths|dead|deceased|"
    r"drown|drowns|drowned|drowning|perish|perishes|perished|perishing|"
    r"kill|kills|killed|killing|lethal|fatality|fatalities|fatal|"
    r"sacrific(?:e|es|ed|ing)|suffocat\w*|starv\w*|freez(?:e|es|ing)\s+to\s+death)\b",
    re.IGNORECASE,
)
_BENEFICIAL_OUTCOME = re.compile(
    r"\b(?:sav(?:e|es|ed|ing)|surviv(?:e|es|ed|ing|al)|preserv(?:e|es|ed|ing)|"
    r"protect(?:s|ed|ing)?|spar(?:e|es|ed|ing)|alive|unharmed|"
    r"honor(?:s|ed|ing)?|honour(?:s|ed|ing)?|respect(?:s|ed|ing)?|"
    r"uphold(?:s|ing)?|upheld|safeguard(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
# "a right against lethal harm" names a harm that is prevented, not inflicted.
# Without this the averted outcome would be scored as the outcome itself.
_AVERTED = re.compile(
    r"\b(?:against|without|prevent(?:s|ed|ing)?|avoid(?:s|ed|ing)?|"
    r"protect(?:s|ed|ing)?\s+from|free\s+from|spared?\s+from|no|not|never)\s+"
    # A conjunction starts a new predicate, so the negation no longer reaches
    # the outcome: "receives no treatment and dies" states a death, not an
    # averted one.
    r"(?:(?!and\b|or\b|but\b|then\b|while\b)[\w-]+\s+){0,2}$",
    re.IGNORECASE,
)
# Participles that introduce a consequence clause. A party's name never runs
# past one of these, so they close the noun phrase being collected.
_CONSEQUENCE_CONNECTIVES = frozenset({
    "causing", "cause", "causes", "caused", "leaving", "leaves", "resulting",
    "results", "triggering", "triggers", "ensuring", "ensures", "guaranteeing",
    "guarantees", "allowing", "allows", "offering", "offers", "securing",
    "secures", "risking", "risks", "denying", "denies", "preventing",
    "prevents", "forcing", "forces", "making", "makes", "stabilizing",
    "stabilizes", "yielding", "producing", "creating", "meaning",
})
# Interventions are how an agent acts, not what happens to a party. They can
# mark a direct object as a beneficiary, but a noun phrase sitting next to one
# is not thereby a moral patient: "a single airlift trip" names equipment usage.
_RESCUE_INTERVENTION = re.compile(
    r"\b(?:rescu(?:e|es|ed|ing)|airlift(?:s|ed|ing)?|evacuat(?:e|es|ed|ing)|"
    r"extract(?:s|ed|ing)?|treat(?:s|ed|ing)?|stabiliz(?:e|es|ed|ing)|"
    r"stabilis(?:e|es|ed|ing)|shelter(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
# Any of these between a party and an outcome verb makes the relation hedged,
# so no deterministic obligation is asserted.
_HEDGE_MARKER = re.compile(
    r"\b(?:chance|chances|probability|probabilit\w*|risk|risks|odds|likel\w*|"
    r"may|might|could|possibly|potential\w*|percent|uncertain\w*|if|unless)\b"
    r"|%",
    re.IGNORECASE,
)

_AGE_FORM = re.compile(r"\b(\d{1,3})[-\s]year[-\s]old\b", re.IGNORECASE)
# Capture the count plus a candidate noun phrase; the head is resolved in Python
# because the last word of the run is the head only after verbs, prepositions
# and relativisers have been trimmed off the tail.
_COUNTED_GROUP = re.compile(
    r"\b(?P<count>\d{1,6}|"
    + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    + r")\s+(?P<phrase>(?:[A-Za-z][\w-]*,?\s+){0,4}[A-Za-z][\w-]*)\b",
    re.IGNORECASE,
)
# Words that terminate a noun phrase: everything after them belongs to the
# predicate, not to the party being named.
_PHRASE_TERMINATORS = frozenset({
    "are", "is", "was", "were", "be", "been", "being", "will", "would", "can",
    "could", "shall", "should", "may", "might", "must", "has", "have", "had",
    "do", "does", "did", "who", "whom", "whose", "which", "that", "in", "on",
    "at", "to", "from", "with", "for", "by", "of", "off", "onto", "into",
    "over", "under", "within", "before", "after", "while", "unless", "until",
    "and", "or", "but", "if", "then", "than", "as", "so", "though", "although",
    "trapped", "standing", "waiting", "left", "remaining", "stuck", "located",
})
_AGED_PARTY = re.compile(
    r"\b(?:the\s+|a\s+|an\s+)?(?P<age>\d{1,3}[-\s]year[-\s]old)"
    r"(?P<phrase>(?:\s+[A-Za-z][\w-]*){0,3})",
    re.IGNORECASE,
)

# Resources the agent allocates. They are counted like people ("a single dose",
# "one ventilator") but are never moral patients, and once candidate actions
# contribute outcome context they would otherwise be pulled into the registry.
_INSTRUMENT_NOUNS = frozenset({
    "dose", "doses", "dosage", "drug", "drugs", "medication", "medicine",
    "treatment", "treatments", "vaccine", "vaccines", "antiviral", "antivirals",
    "antibody", "antibodies", "ventilator", "ventilators", "machine", "machines",
    "device", "devices", "equipment", "kit", "kits", "bed", "beds", "seat",
    "seats", "slot", "slots", "ration", "rations", "vehicle", "vehicles",
    "drone", "drones", "battery", "batteries", "ticket", "tickets",
})

# Heads that name a slot in the problem statement rather than a party, so
# "Option 1" and "Location B" never become moral patients.
_NON_PARTY_DESIGNATORS = frozenset({
    "option", "location", "action", "policy", "plan", "phase", "step", "case",
    "site", "area", "zone", "room", "table", "figure", "exhibit", "appendix",
    "section", "clause", "scenario", "choice", "route", "path", "level", "tier",
    "class", "type", "model", "version", "chapter", "part", "stage", "round",
    "day", "week", "month", "year", "building", "floor", "lane", "track",
})

# "Patient A", "Subject 1", "Group B": a common way for dilemmas to name the
# parties, and invisible to both the counted-group and aged-individual patterns.
_LABELLED_PARTY = re.compile(
    r"\b(?P<noun>[A-Z][a-z]{2,})\s+(?P<tag>[A-Z]|\d{1,2})\b(?![\w-])",
)


def _singularize(token: str) -> str:
    text = token.casefold()
    irregular = {
        "children": "child", "people": "person", "men": "man", "women": "woman",
        "lives": "life", "wives": "wife", "persons": "person",
    }
    if text in irregular:
        return irregular[text]
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    if text.endswith("ses") or text.endswith("xes") or text.endswith("zes"):
        return text[:-2]
    if text.endswith("s") and not text.endswith("ss") and len(text) > 3:
        return text[:-1]
    return text


def _normalize_count(raw: str) -> str | None:
    text = str(raw or "").strip().casefold().replace(",", "")
    if not text:
        return None
    if text.isdigit():
        return str(int(text))
    return _NUMBER_WORDS.get(text)


# How a mention's cardinality was arrived at. Normalizing "a 12-year-old child"
# to one person is useful internally, but that 1 is derived, not stated by the
# scenario. Recording the difference stops the validator from later treating its
# own normalization artifact as source evidence.
COUNT_EXPLICIT = "EXPLICIT"
COUNT_DERIVED_SINGLETON = "DERIVED_SINGLETON"
COUNT_UNSPECIFIED = "UNSPECIFIED"


@dataclass(frozen=True, slots=True)
class PartyMention:
    """A group of moral patients named by the scenario under analysis."""

    label: str
    tokens: frozenset[str] = frozenset()
    count: str | None = None
    count_status: str = "UNSPECIFIED"
    source_span: str = ""
    clause_ids: tuple[str, ...] = ()
    # Normalized "patient a" style tag. Two parties with different designators
    # are different people however much of their description they share.
    designator: str = ""

    @property
    def count_is_explicit(self) -> bool:
        return self.count_status == COUNT_EXPLICIT

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "count_status": self.count_status,
            "source_span": self.source_span,
            "clause_ids": list(self.clause_ids),
        }


def party_identity(label: str) -> tuple[frozenset[str], str | None]:
    """Return distinctive identity tokens and explicit cardinality for a label.

    Ages are identity tokens, never counts: "the 12-year-old child" is one
    person whose distinguishing feature is the age, so it must not be read as a
    group of twelve.
    """
    text = " ".join(str(label or "").split())
    if not text:
        return frozenset(), None

    tokens: set[str] = set()
    count: str | None = None

    ages = _AGE_FORM.findall(text)
    for age in ages:
        tokens.add(f"{age}-year-old")
    scrubbed = _AGE_FORM.sub(" ", text)

    for raw in re.findall(r"[A-Za-z][\w-]*|\d+", scrubbed):
        lowered = raw.casefold()
        if lowered in _PARTY_STOPWORDS:
            continue
        numeric = _normalize_count(lowered)
        if numeric is not None and count is None:
            count = numeric
            continue
        if numeric is not None:
            continue
        singular = _singularize(lowered)
        if singular in _MEASURE_NOUNS:
            continue
        tokens.add(singular)

    return frozenset(tokens), count


def _non_party_head(word: str) -> bool:
    singular = _singularize(word)
    return (
        singular in _MEASURE_NOUNS
        or singular in _INSTRUMENT_NOUNS
        or singular in _NON_PARTY_DESIGNATORS
    )


def party_designator(label: str) -> str:
    """Return a normalized "patient a" tag for labels that carry one."""
    for match in _LABELLED_PARTY.finditer(str(label or "")):
        noun = match.group("noun")
        if _non_party_head(noun):
            continue
        return f"{noun.casefold()} {match.group('tag').casefold()}"
    return ""


def parties_compatible(left: str | PartyMention, right: str | PartyMention) -> bool:
    """Whether two mentions denote the same group.

    Compatibility needs a shared distinctive token and non-conflicting counts,
    so "the three elderly on the rooftop" matches "three elderly survivors"
    while "3 patients" does not match "16 refugees".
    """
    left_tokens, left_count = (
        (left.tokens, left.count) if isinstance(left, PartyMention)
        else party_identity(left)
    )
    right_tokens, right_count = (
        (right.tokens, right.count) if isinstance(right, PartyMention)
        else party_identity(right)
    )
    left_tag = (
        left.designator if isinstance(left, PartyMention) else party_designator(left)
    )
    right_tag = (
        right.designator if isinstance(right, PartyMention)
        else party_designator(right)
    )
    # An explicit tag settles identity on its own: "Patient A" and "Patient B"
    # share every descriptive word they have and are still different people.
    if left_tag and right_tag:
        return left_tag == right_tag
    if not left_tokens or not right_tokens:
        return False
    if left_count and right_count and left_count != right_count:
        return False
    shared = left_tokens & right_tokens
    if shared - _WEAK_TOKENS:
        return True
    # Only weak tokens in common: accept solely when the cardinality agrees,
    # which is what makes "three lives saved" the same group as "three elderly".
    return bool(shared) and bool(left_count) and left_count == right_count


def _label_quality(label: str) -> tuple[int, int, int]:
    tokens, count = party_identity(label)
    return (len(tokens - _WEAK_TOKENS), 1 if count else 0, len(tokens))


def merge_party_mentions(mentions: Iterable[PartyMention]) -> list[PartyMention]:
    """Collapse compatible mentions, keeping the most informative label."""
    merged: list[PartyMention] = []
    for mention in mentions:
        for index, existing in enumerate(merged):
            if not parties_compatible(existing, mention):
                continue
            clause_ids = tuple(dict.fromkeys((*existing.clause_ids, *mention.clause_ids)))
            better = (
                mention
                if _label_quality(mention.label) > _label_quality(existing.label)
                else existing
            )
            # A count the scenario states outranks one this module derived,
            # whichever mention happened to be seen first.
            stated = next(
                (item for item in (existing, mention) if item.count_is_explicit),
                None,
            )
            merged[index] = PartyMention(
                label=better.label,
                # Descriptions accumulate: an alias-linked mention must not lose
                # the words that let a differently-worded action still match it.
                tokens=existing.tokens | mention.tokens,
                count=stated.count if stated else (existing.count or mention.count),
                count_status=stated.count_status if stated else existing.count_status,
                source_span=(stated or existing).source_span or mention.source_span,
                clause_ids=clause_ids,
                designator=existing.designator or mention.designator,
            )
            break
        else:
            merged.append(mention)
    return merged


def _trim_noun_phrase(words: Sequence[str]) -> list[str]:
    """Cut a candidate noun phrase at the first word that ends it.

    Everything after a terminator, a consequence connective or an outcome verb
    belongs to the predicate rather than to the party being named.
    """
    trimmed: list[str] = []
    for raw in words:
        word = raw.strip(",")
        lowered = word.casefold()
        if not word:
            break
        if lowered in _PHRASE_TERMINATORS or lowered in _NON_NOUN_HEADS:
            break
        if lowered in _CONSEQUENCE_CONNECTIVES:
            break
        if trimmed and (
            _ADVERSE_OUTCOME.fullmatch(lowered)
            or _BENEFICIAL_OUTCOME.fullmatch(lowered)
            or _RESCUE_INTERVENTION.fullmatch(lowered)
        ):
            break
        trimmed.append(word)
    # Only filler is stripped. A unit or resource head is left in place so the
    # caller can reject the phrase outright rather than silently falling back
    # to a modifier, which is how "a single airlift trip" became a party.
    while trimmed and trimmed[-1].casefold() in _PARTY_STOPWORDS:
        trimmed.pop()
    return trimmed


def extract_party_registry(
    scenario: str,
    clauses: Sequence[dict[str, str]] | None = None,
    outcome_context: Sequence[str] = (),
) -> list[PartyMention]:
    """Best-effort deterministic party registry for a scenario.

    This is the offline fallback and the check against model-proposed registries;
    it is intentionally conservative, admitting only counted groups and aged
    individuals rather than guessing at arbitrary noun phrases.
    """
    spans: list[tuple[str, str]] = []
    if clauses:
        spans = [
            (str(clause.get("clause_id", "")), str(clause.get("text", "")))
            for clause in clauses
        ]
    else:
        spans = [("", " ".join(str(scenario or "").split()))]

    found: list[PartyMention] = []
    for clause_id, text in spans:
        for match in _LABELLED_PARTY.finditer(text):
            noun = match.group("noun")
            if _non_party_head(noun):
                continue
            label = f"{noun} {match.group('tag')}"
            found.append(PartyMention(
                label=label,
                tokens=frozenset({_singularize(noun)}),
                count="1",
                count_status=COUNT_DERIVED_SINGLETON,
                source_span=label,
                clause_ids=(clause_id,) if clause_id else (),
                designator=party_designator(label),
            ))
        for match in _AGED_PARTY.finditer(text):
            head_words = _trim_noun_phrase(match.group("phrase").split())
            # "the 40-year-old vehicle" is not a person; keep the age alone
            # rather than naming a party after a resource.
            if head_words and _non_party_head(head_words[-1]):
                head_words = []
            label = " ".join([match.group("age"), *head_words])
            tokens, count = party_identity(label)
            if tokens:
                found.append(PartyMention(
                    label=label,
                    tokens=tokens,
                    count=count or "1",
                    count_status=(
                        COUNT_EXPLICIT if count else COUNT_DERIVED_SINGLETON
                    ),
                    source_span=" ".join(match.group(0).split()),
                    clause_ids=(clause_id,) if clause_id else (),
                ))
        for match in _COUNTED_GROUP.finditer(text):
            words = [word.strip(",") for word in match.group("phrase").split()]
            # "within 10 minutes, offering ..." counts minutes, not people.
            if words and _singularize(words[0]) in _MEASURE_NOUNS:
                continue
            trimmed = _trim_noun_phrase(words)
            # A phrase whose head names a unit or a resource counts things, not
            # people. Trimming the head away and keeping the modifier would turn
            # "a single airlift trip" into a party named "single airlift".
            if not trimmed or _non_party_head(trimmed[-1]):
                continue
            if _normalize_count(trimmed[-1]) is not None:
                continue
            modifiers = [
                word for word in trimmed[:-1]
                if word.casefold() not in _PARTY_STOPWORDS
                and _singularize(word) not in _MEASURE_NOUNS
            ]
            label = " ".join([match.group("count"), *modifiers, trimmed[-1]])
            tokens, count = party_identity(label)
            if tokens:
                found.append(PartyMention(
                    label=label,
                    tokens=tokens,
                    count=count,
                    count_status=(
                        COUNT_EXPLICIT if count else COUNT_UNSPECIFIED
                    ),
                    source_span=" ".join(match.group(0).split()),
                    clause_ids=(clause_id,) if clause_id else (),
                ))
    # A scenario may introduce a group ("three elderly trapped on a rooftop")
    # without stating its fate until the candidate actions do. Scanning those
    # too keeps such a group in the registry, while still requiring that the
    # scenario be where the group was named.
    found = _link_designator_aliases(found, spans)
    context = " ".join([
        " ".join(str(scenario or "").split()) or " ".join(text for _, text in spans),
        *(" ".join(str(text).split()) for text in outcome_context),
    ])
    qualified = _outcome_participants(merge_party_mentions(found), context)
    return _absorb_weak_duplicates(qualified)


def _outcome_participants(
    mentions: Sequence[PartyMention],
    scenario: str,
) -> list[PartyMention]:
    """Keep only groups the scenario places in an outcome relation.

    Equipment and logistics ("one ventilator", "a single airlift trip") are
    counted noun phrases too, so cardinality alone cannot identify a moral
    patient. Requiring the group to share a sentence with a benefit or harm
    outcome is domain-neutral and keeps instruments out of the registry.
    """
    sentences = [
        sentence for sentence in re.split(r"(?<=[.!?])\s+", scenario)
        if sentence.strip()
    ]
    kept: list[PartyMention] = []
    for mention in mentions:
        anchors = mention.tokens - _WEAK_TOKENS or mention.tokens
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(token) for token in anchors) + r")\w*",
            re.IGNORECASE,
        )
        for sentence in sentences:
            if not pattern.search(sentence):
                continue
            if _ADVERSE_OUTCOME.search(sentence) or _BENEFICIAL_OUTCOME.search(sentence):
                kept.append(mention)
                break
    return kept


_COPULA_INTRODUCTION = re.compile(
    r"\b(?P<noun>[A-Z][a-z]{2,})\s+(?P<tag>[A-Z]|\d{1,2})\b\s+(?:is|was|are|were)\b",
)


def _link_designator_aliases(
    mentions: Sequence[PartyMention],
    spans: Sequence[tuple[str, str]],
) -> list[PartyMention]:
    """Fold a party's description into the tag that introduces it.

    Scenarios routinely write "Patient A is an 80-year-old retired virologist"
    and then refer to that person either way. Without linking the two mentions,
    an action naming only the tag binds to nothing and the person's description
    becomes a second, phantom party.
    """
    described: dict[str, list[PartyMention]] = {}
    for clause_id, text in spans:
        intro = _COPULA_INTRODUCTION.search(text)
        if not intro or _non_party_head(intro.group("noun")):
            continue
        tag = f"{intro.group('noun').casefold()} {intro.group('tag').casefold()}"
        for mention in mentions:
            if mention.designator or clause_id not in mention.clause_ids:
                continue
            described.setdefault(tag, []).append(mention)

    if not described:
        return list(mentions)

    absorbed: set[int] = set()
    linked: list[PartyMention] = []
    for mention in mentions:
        aliases = described.get(mention.designator or "", [])
        if not mention.designator or not aliases:
            continue
        tokens = set(mention.tokens)
        clause_ids = list(mention.clause_ids)
        for alias in aliases:
            absorbed.add(id(alias))
            tokens |= alias.tokens
            clause_ids.extend(alias.clause_ids)
        richest = max(aliases, key=lambda item: len(item.label))
        linked.append(PartyMention(
            label=f"{mention.label} ({richest.label})",
            tokens=frozenset(tokens),
            count=mention.count,
            count_status=mention.count_status,
            source_span=richest.source_span or mention.source_span,
            clause_ids=tuple(dict.fromkeys(clause_ids)),
            designator=mention.designator,
        ))
        absorbed.add(id(mention))

    return [*linked, *(m for m in mentions if id(m) not in absorbed)]


def _absorb_weak_duplicates(mentions: Sequence[PartyMention]) -> list[PartyMention]:
    """Drop generic restatements such as "three lives" beside "three elderly".

    A mention built only from weak tokens is a paraphrase of a concrete group
    whenever some concrete group shares its cardinality, and keeping both would
    let "future life" bind to a party that the prose never placed in that role.
    """
    strong_counts = {
        mention.count for mention in mentions
        if mention.count and (mention.tokens - _WEAK_TOKENS)
    }
    return [
        mention for mention in mentions
        if (mention.tokens - _WEAK_TOKENS) or mention.count not in strong_counts
    ]


# A party whose outcome the prose hedges is neither harmed nor spared as a
# matter of record. Dropping it silently was the old behaviour and it lost the
# party altogether; asserting a role would claim a certainty the source denies.
# UNRESOLVED is the third status that lets the party stay visible with its
# epistemic condition attached.
ROLE_UNRESOLVED = "unresolved"

DIRECTION_ADVERSE = "ADVERSE"
DIRECTION_BENEFICIAL = "BENEFICIAL"

_STATED_PROBABILITY = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")


@dataclass(frozen=True, slots=True)
class RoleBinding:
    """An outcome relation between an action and a party.

    ``field`` is one of ``beneficiaries``, ``harmed`` or ``unresolved``. For an
    unresolved binding ``direction`` records which way the prose leans and
    ``probability`` preserves any quantity it stated, without either being
    promoted to an assertion that the outcome occurs.
    """

    field: str
    party: PartyMention
    evidence: str = ""
    clause_ids: tuple[str, ...] = field(default_factory=tuple)
    direction: str = ""
    probability: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "party": self.party.label,
            "status": self.field,
            "direction": self.direction,
            "probability": self.probability,
            "evidence": self.evidence,
            "clause_ids": list(self.clause_ids),
        }


# Downstream and cascade effects are not direct intervention roles. A is the
# recipient of the dose; the people who would be uninfected if A lives are a
# different kind of affected party. Putting them in `beneficiaries` would erase
# that distinction, so they live on a surrounding grounded-effects layer.
RELATION_DOWNSTREAM_BENEFIT = "DOWNSTREAM_BENEFIT"
RELATION_FOREGONE_BENEFIT = "FOREGONE_BENEFIT"
PARTY_KIND_FUTURE_POPULATION = "FUTURE_POPULATION"
MODALITY_CERTAIN = "CERTAIN"
MODALITY_CONDITIONAL_ON_DIRECT_SURVIVAL = (
    "CONDITIONAL_ON_DIRECT_BENEFICIARY_SURVIVAL"
)
MODALITY_HEDGED = "HEDGED"

_FUTURE_COLLECTIVE = re.compile(
    r"\b(?:the\s+)?(?:broader\s+population|future\s+populations?|"
    r"future\s+generations?|society)\b",
    re.IGNORECASE,
)
_SCALE_FUTURE_OUTCOME = re.compile(
    r"(?P<outcome>(?:tens\s+of\s+)?(?:thousands|millions|hundreds)\s+of\s+"
    r"future\s+(?:infections?|deaths?|lives?|casualt(?:y|ies))"
    r"(?:\s+and\s+(?:subsequent\s+)?(?:infections?|deaths?|lives?))?)",
    re.IGNORECASE,
)
_PREVENTED_SCALE_FUTURE = re.compile(
    r"\b(?:prevent(?:s|ed|ing)?|avert(?:s|ed|ing)?|avoid(?:s|ed|ing)?)\s+"
    r"(?:tens\s+of\s+)?(?:thousands|millions|hundreds)\s+of\s+future\b",
    re.IGNORECASE,
)
_CASCADE_BENEFIT = re.compile(
    r"\b(?:downstream(?:\s+chain\s+reaction)?|chain\s+reaction|"
    r"cascad(?:e|ing)\s+future\s+benefits?|"
    r"large-scale\s+life-saving|"
    r"universal\s+vaccine|"
    r"life-saving\s+(?:medical\s+)?outcomes?\s+for)\b",
    re.IGNORECASE,
)
_FOREGONE_BENEFIT_CUE = re.compile(
    r"\b(?:forfeit(?:s|ing)?|forego(?:es|ing)?|foregoing|"
    r"lost\s+opportunity|"
    r"opportunity\b[\s\S]{0,120}?\b(?:is|are)\s+lost|"
    r"benefits?\b[\s\S]{0,40}?\b(?:is|are)\s+lost)\b",
    re.IGNORECASE,
)
# Remaining life of a direct recipient is not a distinct future population.
_OWN_FUTURE_LIFE = re.compile(
    r"\b(?:future\s+life(?:-?years?)?|remaining\s+life\s+expectancy|"
    r"decades\s+of\s+(?:potential\s+)?future\s+life|"
    r"aggregate\s+future\s+life-years?|"
    r"full,?\s+healthy\s+natural\s+life)\b",
    re.IGNORECASE,
)
_HEALTH_OUTCOME = re.compile(
    r"\b(?:infections?|deaths?|vaccine|life-saving|lives?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GroundedEffect:
    """An affected-party relation that is not a direct intervention role.

    Direct roles stay on the canonical action (`beneficiaries`, `harmed`).
    This object records a downstream or foregone consequence for a party the
    action does not itself treat, with the modality and provenance needed to
    keep that consequence from collapsing into those direct fields.
    """

    party: str
    party_kind: str
    outcome: str
    relation: str
    modality: str
    condition: str = ""
    provenance: tuple[str, ...] = ()
    clause_ids: tuple[str, ...] = ()
    dimension: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "party": self.party,
            "party_kind": self.party_kind,
            "outcome": self.outcome,
            "relation": self.relation,
            "modality": self.modality,
            "condition": self.condition,
            "provenance": list(self.provenance),
            "clause_ids": list(self.clause_ids),
            "dimension": self.dimension,
        }


def _clause_rows(clauses: Sequence[Any] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, clause in enumerate(clauses or ()):
        if isinstance(clause, dict):
            clause_id = str(clause.get("clause_id") or f"C{index}")
            text = " ".join(str(clause.get("text", "")).split())
        else:
            clause_id = f"C{index}"
            text = " ".join(str(clause).split())
        if text:
            rows.append({"clause_id": clause_id, "text": text})
    return rows


def _direct_role_labels(
    action_text: str,
    registry: Sequence[PartyMention] | None,
    direct_beneficiaries: Sequence[str],
    direct_harmed: Sequence[str],
    unresolved: Sequence[str],
) -> tuple[str, ...]:
    labels = [
        " ".join(str(label).split())
        for label in (*direct_beneficiaries, *direct_harmed, *unresolved)
        if " ".join(str(label).split())
    ]
    if labels:
        return tuple(dict.fromkeys(labels))
    parties = list(registry) if registry is not None else extract_party_registry(
        action_text, [{"clause_id": "A", "text": action_text}],
    )
    return tuple(dict.fromkeys(
        binding.party.label
        for binding in relational_role_bindings(action_text, parties)
    ))


def _label_is_direct(label: str, direct: Sequence[str]) -> bool:
    return any(parties_compatible(label, existing) for existing in direct if existing)


def _cascade_outcome(text: str, relation: str) -> str:
    scale = _SCALE_FUTURE_OUTCOME.search(text)
    if scale:
        outcome = " ".join(scale.group("outcome").split())
        if relation == RELATION_DOWNSTREAM_BENEFIT:
            return f"{outcome} prevented"
        if relation == RELATION_FOREGONE_BENEFIT:
            return f"{outcome} not prevented"
        return outcome
    cascade = _CASCADE_BENEFIT.search(text)
    if cascade:
        span = " ".join(cascade.group(0).split())
        if relation == RELATION_FOREGONE_BENEFIT:
            return f"{span} foregone"
        return span
    if relation == RELATION_FOREGONE_BENEFIT:
        lost = re.search(
            r"opportunity\b[\s\S]{0,120}?\b(?:is|are)\s+lost",
            text,
            re.IGNORECASE,
        )
        if lost:
            return " ".join(lost.group(0).split())
        return "cascading future benefits foregone"
    return "downstream benefit"


def _effect_dimension(outcome: str) -> str:
    if _HEALTH_OUTCOME.search(outcome):
        return "BASIC_SECURITY"
    return "OTHER"


def _matching_clause_ids(
    rows: Sequence[dict[str, str]], needles: Sequence[str],
) -> tuple[str, ...]:
    lowered_needles = [
        " ".join(str(needle).split()).casefold()
        for needle in needles
        if " ".join(str(needle).split())
    ]
    found: list[str] = []
    for row in rows:
        text = row["text"].casefold()
        if any(needle in text for needle in lowered_needles):
            found.append(row["clause_id"])
    return tuple(dict.fromkeys(found))


def extract_grounded_effects(
    action_text: str,
    *,
    clauses: Sequence[Any] | None = None,
    registry: Sequence[PartyMention] | None = None,
    direct_beneficiaries: Sequence[str] = (),
    direct_harmed: Sequence[str] = (),
    unresolved: Sequence[str] = (),
) -> tuple[GroundedEffect, ...]:
    """Recover downstream/foregone affected-party relations from action prose.

    Sound rather than complete: only constructions that name a cascade or a
    forfeited population-scale benefit, and never a party already bound as a
    direct beneficiary, harmed party, or unresolved subject.
    """
    text = " ".join(str(action_text or "").split())
    if not text:
        return ()
    direct = _direct_role_labels(
        text, registry, direct_beneficiaries, direct_harmed, unresolved,
    )
    has_cascade = bool(
        _PREVENTED_SCALE_FUTURE.search(text)
        or _SCALE_FUTURE_OUTCOME.search(text)
        or _CASCADE_BENEFIT.search(text)
    )
    has_foregone = bool(_FOREGONE_BENEFIT_CUE.search(text))
    if not has_cascade and not has_foregone:
        return ()
    # Life-years of a named recipient are not a distinct future population,
    # even when the prose says "future life".
    if _OWN_FUTURE_LIFE.search(text) and not (
        _PREVENTED_SCALE_FUTURE.search(text)
        or _FUTURE_COLLECTIVE.search(text)
        or _CASCADE_BENEFIT.search(text)
    ):
        return ()
    party = "future population"
    if _label_is_direct(party, direct):
        return ()
    if has_foregone:
        relation = RELATION_FOREGONE_BENEFIT
        modality = MODALITY_CERTAIN
        condition = ""
    else:
        relation = RELATION_DOWNSTREAM_BENEFIT
        if direct:
            modality = MODALITY_CONDITIONAL_ON_DIRECT_SURVIVAL
            condition = direct[0]
        else:
            modality = MODALITY_CERTAIN
            condition = ""
        if _HEDGE_MARKER.search(text) and re.search(
            r"\b(?:chance|probability|percent|%)\b", text, re.IGNORECASE,
        ):
            modality = MODALITY_HEDGED
    outcome = _cascade_outcome(text, relation)
    rows = _clause_rows(clauses)
    needles = [outcome, party, "future infections", "broader population"]
    if relation == RELATION_DOWNSTREAM_BENEFIT:
        needles.extend(["downstream", "prevent tens of thousands"])
    else:
        needles.extend(["foregoing", "forego", "opportunity", "benefits to society"])
    clause_ids = _matching_clause_ids(rows, needles)
    provenance = tuple(
        f"{row['clause_id']}: {row['text']}"
        for row in rows
        if row["clause_id"] in set(clause_ids)
    )
    return (GroundedEffect(
        party=party,
        party_kind=PARTY_KIND_FUTURE_POPULATION,
        outcome=outcome,
        relation=relation,
        modality=modality,
        condition=condition,
        provenance=provenance,
        clause_ids=clause_ids,
        dimension=_effect_dimension(outcome),
    ),)


def _party_spans(text: str, party: PartyMention) -> list[tuple[int, int]]:
    # A weak token may accompany a match but never anchor one on its own.
    anchors = party.tokens - _WEAK_TOKENS or party.tokens
    alternatives: list[str] = []
    if party.designator:
        noun, _, tag = party.designator.partition(" ")
        # The bare noun is ambiguous between tagged parties, so only the full
        # "Patient A" phrase may anchor a match for the shared word.
        anchors = anchors - {_singularize(noun), noun}
        alternatives.append(rf"{re.escape(noun)}\s+{re.escape(tag)}\b")
    alternatives.extend(
        re.escape(token) + r"\w*"
        for token in sorted(anchors, key=len, reverse=True)
    )
    if not alternatives:
        return []
    pattern = re.compile(r"\b(?:" + "|".join(alternatives) + r")", re.IGNORECASE)
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _segment_bounds(text: str, position: int) -> tuple[int, int]:
    start = max(
        (text.rfind(mark, 0, position) for mark in (".", ";", " while ", " but ", " though ")),
        default=-1,
    )
    end_candidates = [
        index for index in (
            text.find(".", position), text.find(";", position),
            text.find(" while ", position), text.find(" but ", position),
        ) if index != -1
    ]
    return (start + 1 if start != -1 else 0, min(end_candidates) if end_candidates else len(text))


def relational_role_bindings(
    action_text: str,
    registry: Sequence[PartyMention],
) -> list[RoleBinding]:
    """Extract each party's outcome relation to an action, including unsettled ones.

    A party gets exactly one status per action. A definite statement always
    outranks a hedged one, so "the three elderly drown" is recorded as harm even
    if an earlier clause speculated about their chances. When every mention is
    hedged the party is returned as UNRESOLVED rather than dropped, which keeps
    it visible to deliberation without asserting an outcome the source withheld.
    """
    text = " ".join(str(action_text or "").split())
    # party tokens -> (is_definite, binding). The first definite reading wins;
    # until one appears the first hedged reading is held as a fallback.
    resolved: dict[frozenset[str], tuple[bool, RoleBinding]] = {}

    for party in registry:
        for start, end in _party_spans(text, party):
            existing = resolved.get(party.tokens)
            if existing is not None and existing[0]:
                break

            left, right = _segment_bounds(text, start)
            window = text[left:right]
            relative = start - left

            adverse = [
                match for match in _ADVERSE_OUTCOME.finditer(window)
                if not _AVERTED.search(window[:match.start()])
            ]
            beneficial = list(_BENEFICIAL_OUTCOME.finditer(window))
            beneficial += [
                match for match in _RESCUE_INTERVENTION.finditer(window)
                if match.end() <= relative
            ]
            if not adverse and not beneficial:
                continue

            def _nearest(matches: list[re.Match[str]]) -> re.Match[str] | None:
                if not matches:
                    return None
                return min(matches, key=lambda m: abs(m.start() - relative))

            best_adverse = _nearest(adverse)
            best_beneficial = _nearest(beneficial)
            if best_adverse and best_beneficial:
                adverse_wins = (
                    abs(best_adverse.start() - relative)
                    <= abs(best_beneficial.start() - relative)
                )
                chosen = best_adverse if adverse_wins else best_beneficial
                role = "harmed" if adverse_wins else "beneficiaries"
            elif best_adverse:
                chosen, role = best_adverse, "harmed"
            elif best_beneficial:
                chosen, role = best_beneficial, "beneficiaries"
            else:
                continue

            span = window[min(relative, chosen.start()): max(end - left, chosen.end())]
            evidence = " ".join(span.split())
            hedged = bool(_HEDGE_MARKER.search(span))

            if hedged:
                if existing is not None:
                    continue
                stated = _STATED_PROBABILITY.search(window)
                resolved[party.tokens] = (False, RoleBinding(
                    field=ROLE_UNRESOLVED,
                    party=party,
                    evidence=evidence,
                    clause_ids=party.clause_ids,
                    direction=(
                        DIRECTION_ADVERSE if role == "harmed"
                        else DIRECTION_BENEFICIAL
                    ),
                    probability=f"{stated.group(1)}%" if stated else "",
                ))
                continue

            resolved[party.tokens] = (True, RoleBinding(
                field=role,
                party=party,
                evidence=evidence,
                clause_ids=party.clause_ids,
                direction=(
                    DIRECTION_ADVERSE if role == "harmed"
                    else DIRECTION_BENEFICIAL
                ),
            ))
            break

    return [binding for _definite, binding in resolved.values()]


# --- Role assignment validation ------------------------------------------------
#
# Severity separates two different verdicts. REPAIR means the structured record
# disagrees with the prose and must not be admitted. NORMALIZE means the roles
# are right but a detail was flattened, which is recoverable without discarding
# the record.

_SEVERITY_REPAIR = "REPAIR"
_SEVERITY_NORMALIZE = "NORMALIZE"

_DEBRIS_MARKER = re.compile(r":COUNT\b|:[A-Z]{3,}\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RoleIssue:
    """A single disagreement between structured roles and canonical prose."""

    code: str
    severity: str
    message: str

    def __str__(self) -> str:  # keeps existing string-oriented callers working
        return self.message


def resolve_party(
    label: str,
    registry: Sequence[PartyMention],
) -> PartyMention | None:
    """Return the registry party a role label refers to, if any."""
    for party in registry:
        if parties_compatible(party, label):
            return party
    return None


def _party_by_identity_only(
    label: str,
    registry: Sequence[PartyMention],
) -> PartyMention | None:
    """Find a party matching on identity tokens while ignoring cardinality.

    Used to tell "this group is not in the scenario" apart from "this is the
    right group with the wrong number", which are different errors to fix.
    """
    tokens, _ = party_identity(label)
    distinctive = tokens - _WEAK_TOKENS
    if not distinctive:
        return None
    for party in registry:
        if (party.tokens - _WEAK_TOKENS) & distinctive:
            return party
    return None


def _explicit_count(label: str) -> str | None:
    return party_identity(label)[1]


def validate_role_assignment(
    canonical_action: str,
    *,
    beneficiaries: Sequence[str] = (),
    harmed: Sequence[str] = (),
    unresolved: Sequence[str] = (),
    registry: Sequence[PartyMention] = (),
    provenance: dict[str, Sequence[str]] | None = None,
    known_clause_ids: Sequence[str] = (),
    require_provenance: bool = False,
    registry_is_authoritative: bool = False,
) -> tuple[RoleIssue, ...]:
    """Check structured role fields against the scenario registry and prose.

    Every check is domain-neutral: parties come from the scenario, and outcome
    polarity comes from a general lexicon. The prose scan is independent of
    whatever produced the role fields, so an extractor that silently returns
    nothing cannot also silently satisfy its own validation.
    """
    action = " ".join(str(canonical_action or "").split())
    assigned = {
        "beneficiaries": [str(value) for value in beneficiaries if str(value).strip()],
        "harmed": [str(value) for value in harmed if str(value).strip()],
        ROLE_UNRESOLVED: [str(value) for value in unresolved if str(value).strip()],
    }
    issues: list[RoleIssue] = []

    def _add(code: str, severity: str, message: str) -> None:
        if not any(existing.message == message for existing in issues):
            issues.append(RoleIssue(code=code, severity=severity, message=message))

    # Prose that reports outcomes but yields no parties means the registry stage
    # found nothing to validate against, and validating roles against nothing is
    # how empty records passed before. How hard to fail depends on the source:
    # an authoritative registry returning nothing is a genuine contradiction,
    # whereas the conservative fallback extractor missing an uncounted group is
    # a gap to report rather than grounds for rejecting the action.
    if not registry:
        if _ADVERSE_OUTCOME.search(action) or _BENEFICIAL_OUTCOME.search(action):
            _add(
                "REGISTRY_EMPTY",
                _SEVERITY_REPAIR if registry_is_authoritative else _SEVERITY_NORMALIZE,
                "no scenario parties were identified for an action that states outcomes",
            )
        return tuple(issues)

    for field in (*ROLE_FIELDS, ROLE_UNRESOLVED):
        for label in assigned[field]:
            if _DEBRIS_MARKER.search(label):
                _add(
                    "ROLE_DEBRIS", _SEVERITY_REPAIR,
                    f"{field} contains compiler debris rather than a party: {label}",
                )
                continue
            if resolve_party(label, registry) is not None:
                continue
            near = _party_by_identity_only(label, registry)
            if near is not None and near.count_is_explicit:
                _add(
                    "CARDINALITY_CONFLICT", _SEVERITY_REPAIR,
                    f"{field} states a different quantity for {near.label}: "
                    f"{label} contradicts the scenario's {near.count}"
                    + (f" in \"{near.source_span}\"" if near.source_span else ""),
                )
            else:
                _add(
                    "PARTY_NOT_IN_SCENARIO", _SEVERITY_REPAIR,
                    f"{field} names a party absent from the scenario: {label}",
                )

    bindings = relational_role_bindings(action, registry)

    for binding in (b for b in bindings if b.field == ROLE_UNRESOLVED):
        asserted = [
            f"{role}={label}"
            for role in ROLE_FIELDS
            for label in assigned[role]
            if parties_compatible(binding.party, label)
        ]
        if asserted:
            _add(
                "UNRESOLVED_ASSERTED_AS_CERTAIN", _SEVERITY_REPAIR,
                f"{binding.party.label} is recorded as {', '.join(asserted)} but the "
                f"action only states a hedged outcome: {binding.evidence}",
            )
        elif not any(
            parties_compatible(binding.party, label)
            for label in assigned[ROLE_UNRESOLVED]
        ):
            _add(
                "UNRESOLVED_DROPPED", _SEVERITY_REPAIR,
                f"{binding.party.label} is affected with an unsettled outcome but "
                f"appears in no role: {binding.evidence}",
            )

    for binding in (b for b in bindings if b.field != ROLE_UNRESOLVED):
        opposite = "harmed" if binding.field == "beneficiaries" else "beneficiaries"
        matches = [
            label for label in assigned[binding.field]
            if parties_compatible(binding.party, label)
        ]
        contradicting = [
            label for label in assigned[opposite]
            if parties_compatible(binding.party, label)
        ]
        if contradicting and not matches:
            _add(
                "ROLE_CONTRADICTS_PROSE", _SEVERITY_REPAIR,
                f"{opposite} lists {binding.party.label} but the action states "
                f"the opposite: {binding.evidence}",
            )
            continue
        if not matches:
            _add(
                "ROLE_MISSING", _SEVERITY_REPAIR,
                f"{binding.field} omits {binding.party.label} despite the action "
                f"stating: {binding.evidence}",
            )
            continue
        # Only a quantity the scenario states can be "lost". A derived singleton
        # (an individual identified by age) carries a count this module supplied,
        # and demanding a label repeat it would turn normalization into evidence.
        if not binding.party.count_is_explicit:
            continue
        # A label carrying a contradictory number never reaches here, because
        # conflicting cardinality already blocks the compatibility match above.
        if not any(_explicit_count(label) for label in matches):
            _add(
                "CARDINALITY_LOST", _SEVERITY_NORMALIZE,
                f"{binding.field} lost the explicit count for {binding.party.label}"
                + (f" (stated as \"{binding.party.source_span}\")"
                   if binding.party.source_span else ""),
            )

    for field in ROLE_FIELDS:
        if assigned[field]:
            continue
        if any(binding.field == field for binding in bindings):
            _add(
                "ROLE_FIELD_EMPTY", _SEVERITY_REPAIR,
                f"{field} is empty although the action states {field} outcomes",
            )

    for label in assigned["harmed"]:
        for other in assigned["beneficiaries"]:
            if parties_compatible(label, other):
                _add(
                    "ROLE_OVERLAP", _SEVERITY_REPAIR,
                    f"party appears as both harmed and beneficiary: {label}",
                )

    if provenance is not None:
        valid_ids = set(known_clause_ids)
        for field in (*ROLE_FIELDS, ROLE_UNRESOLVED):
            for label in assigned[field]:
                cited = [str(value) for value in (provenance.get(label) or ())]
                if not cited:
                    if require_provenance:
                        _add(
                            "PROVENANCE_MISSING", _SEVERITY_REPAIR,
                            f"{field} entry {label} cites no source clause",
                        )
                    continue
                unknown = [value for value in cited if valid_ids and value not in valid_ids]
                if unknown:
                    _add(
                        "PROVENANCE_UNKNOWN", _SEVERITY_REPAIR,
                        f"{field} entry {label} cites unknown clauses: "
                        + ", ".join(sorted(unknown)),
                    )

    return tuple(issues)


def completeness_from_role_issues(issues: Sequence[RoleIssue]) -> str:
    """Map role issues onto a record completeness status."""
    if not issues:
        return ""
    if any(issue.severity == _SEVERITY_REPAIR for issue in issues):
        return "NEEDS_REPAIR"
    return "COMPLETE_WITH_NORMALIZATION"
