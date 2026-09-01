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
]


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
    r"protect(?:s|ed|ing)?|spar(?:e|es|ed|ing)|alive|unharmed)\b",
    re.IGNORECASE,
)
# Interventions are how an agent acts, not what happens to a party. They can
# mark a direct object as a beneficiary, but a noun phrase sitting next to one
# is not thereby a moral patient: "a single airlift trip" names equipment usage.
_RESCUE_INTERVENTION = re.compile(
    r"\b(?:rescu(?:e|es|ed|ing)|airlift(?:s|ed|ing)?|evacuat(?:e|es|ed|ing)|"
    r"extract(?:s|ed|ing)?|treat(?:s|ed|ing)?)\b",
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
    r"(?:\s+(?P<head>[A-Za-z][\w-]*))?",
    re.IGNORECASE,
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


@dataclass(frozen=True, slots=True)
class PartyMention:
    """A group of moral patients named by the scenario under analysis."""

    label: str
    tokens: frozenset[str] = frozenset()
    count: str | None = None
    clause_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
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
            better = mention if _label_quality(mention.label) > _label_quality(existing.label) else existing
            merged[index] = PartyMention(
                label=better.label,
                tokens=better.tokens or existing.tokens,
                count=existing.count or mention.count,
                clause_ids=clause_ids,
            )
            break
        else:
            merged.append(mention)
    return merged


def extract_party_registry(
    scenario: str,
    clauses: Sequence[dict[str, str]] | None = None,
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
        for match in _AGED_PARTY.finditer(text):
            head = (match.group("head") or "").strip()
            if head and _singularize(head) in _MEASURE_NOUNS:
                head = ""
            label = " ".join(part for part in (match.group("age"), head) if part)
            tokens, count = party_identity(label)
            if tokens:
                found.append(PartyMention(
                    label=label,
                    tokens=tokens,
                    count=count or "1",
                    clause_ids=(clause_id,) if clause_id else (),
                ))
        for match in _COUNTED_GROUP.finditer(text):
            words = [word.strip(",") for word in match.group("phrase").split()]
            # "within 10 minutes, offering ..." counts minutes, not people.
            if words and _singularize(words[0]) in _MEASURE_NOUNS:
                continue
            trimmed: list[str] = []
            for word in words:
                lowered = word.casefold()
                if lowered in _PHRASE_TERMINATORS or lowered in _NON_NOUN_HEADS:
                    break
                trimmed.append(word)
            while trimmed and (
                _singularize(trimmed[-1]) in _MEASURE_NOUNS
                or trimmed[-1].casefold() in _PARTY_STOPWORDS
            ):
                trimmed.pop()
            if not trimmed or _normalize_count(trimmed[-1]) is not None:
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
                    clause_ids=(clause_id,) if clause_id else (),
                ))
    qualified = _outcome_participants(
        merge_party_mentions(found),
        " ".join(str(scenario or "").split()) or " ".join(text for _, text in spans),
    )
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


@dataclass(frozen=True, slots=True)
class RoleBinding:
    """An unambiguous outcome relation between an action and a party."""

    field: str
    party: PartyMention
    evidence: str = ""
    clause_ids: tuple[str, ...] = field(default_factory=tuple)


def _party_spans(text: str, party: PartyMention) -> list[tuple[int, int]]:
    # A weak token may accompany a match but never anchor one on its own.
    anchors = party.tokens - _WEAK_TOKENS or party.tokens
    if not anchors:
        return []
    alternation = "|".join(
        re.escape(token) for token in sorted(anchors, key=len, reverse=True)
    )
    pattern = re.compile(rf"\b(?:{alternation})\w*", re.IGNORECASE)
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
    """Extract only unambiguous beneficiary/harmed relations from action prose.

    Hedged relations are skipped on purpose; see the module docstring.
    """
    text = " ".join(str(action_text or "").split())
    bindings: list[RoleBinding] = []
    claimed: set[tuple[str, frozenset[str]]] = set()

    for party in registry:
        for start, end in _party_spans(text, party):
            left, right = _segment_bounds(text, start)
            window = text[left:right]
            relative = start - left

            adverse = list(_ADVERSE_OUTCOME.finditer(window))
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
            chosen, role = None, ""
            if best_adverse and best_beneficial:
                if abs(best_adverse.start() - relative) <= abs(best_beneficial.start() - relative):
                    chosen, role = best_adverse, "harmed"
                else:
                    chosen, role = best_beneficial, "beneficiaries"
            elif best_adverse:
                chosen, role = best_adverse, "harmed"
            elif best_beneficial:
                chosen, role = best_beneficial, "beneficiaries"
            if chosen is None:
                continue

            span = window[min(relative, chosen.start()): max(end - left, chosen.end())]
            if _HEDGE_MARKER.search(span):
                continue

            token = (role, party.tokens)
            if token in claimed:
                continue
            claimed.add(token)
            bindings.append(RoleBinding(
                field=role,
                party=party,
                evidence=" ".join(span.split()),
                clause_ids=party.clause_ids,
            ))
    return bindings
