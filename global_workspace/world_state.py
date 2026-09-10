"""Typed, framework-neutral world state for ethical scenarios.

The objects in this module describe who acts, what each action does, and what
effects the scenario attributes to it.  They deliberately contain no ethical
evaluation.  A single admitted model is projected into canonical action
records and the semantic graph so those two views cannot independently infer
contradictory facts from the same prose.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Sequence

from .world_validation import (
    WorldModelValidationError,
    validation_issues_from_messages,
)


DIRECTNESSES = {"DIRECT", "DOWNSTREAM", "FOREGONE", "INSTITUTIONAL"}
EFFECT_KINDS = {
    "INTERVENTION", "RESOURCE_TRANSFER", "CAPABILITY_CHANGE", "PHYSICAL_STATE",
    "HEALTH_OUTCOME", "WELFARE_OUTCOME", "INSTITUTIONAL_OUTCOME",
    "OPPORTUNITY_LOSS", "OTHER",
}
MODALITIES = {
    "CERTAIN", "STIPULATED_CONDITIONAL", "PROBABILISTIC", "POSSIBLE", "UNKNOWN",
}
POLARITIES = {"BENEFICIAL", "ADVERSE", "NEUTRAL", "UNRESOLVED", "FOREGONE"}
COUNTERFACTUAL_RELATIONS = {
    "FOREGOES_ALTERNATIVE_EFFECT", "PRECLUDES_ALTERNATIVE_EFFECT",
    "REPLACES_ALTERNATIVE_EFFECT",
}
# Cross-action foreclosure relations that may arrive on causal_links in older
# traces or live repairs. They are migrated onto counterfactual_links so they
# never become actual-world mechanisms.
_FORECLOSURE_LINK_RELATIONS = {
    "FOREGOES": "FOREGOES_ALTERNATIVE_EFFECT",
    "FOREGOES_ALTERNATIVE_EFFECT": "FOREGOES_ALTERNATIVE_EFFECT",
    "PRECLUDES": "PRECLUDES_ALTERNATIVE_EFFECT",
    "PRECLUDES_ALTERNATIVE_EFFECT": "PRECLUDES_ALTERNATIVE_EFFECT",
    "REPLACES": "REPLACES_ALTERNATIVE_EFFECT",
    "REPLACES_ALTERNATIVE_EFFECT": "REPLACES_ALTERNATIVE_EFFECT",
}
ADMISSION_STATUSES = {
    "COMMITTED", "COMMITTED_WITH_UNCERTAINTY",
    "USER_ACCEPTED_WITH_QUARANTINE", "ABANDONED_CONTRADICTORY_WORLD_STATE",
}


def _clean(value: Any, limit: int = 240) -> str:
    return " ".join(str(value or "").split())[:limit]


# Hyphenated ages are labels, not effect magnitudes. Quantities are closed-class
# spans copied from sources: numerals, English cardinals, collective nouns, and
# optional duration/percent units. Vague comparatives (few/many) are never counts.
_AGE_LABEL = re.compile(r"\b\d+-year-old\b", re.IGNORECASE)
_ACTION_SOURCE_ID = re.compile(r"^A\d+$", re.IGNORECASE)
_COMPARISON_CUE = re.compile(
    r"\b(?:must\s+choose|choose\s+between|chooses?\s+between|"
    r"between\s+.+\s+or\s+|either\s+.+\s+or\s+"
    r"|option\s+(?:1|2|a|b)\b.*\boption\s+(?:1|2|a|b)\b"
    r"|action\s+a\d+.*action\s+a\d+)\b",
    re.IGNORECASE | re.DOTALL,
)
_DURATION_UNIT = (
    r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|percent|%)"
)
# Longer phrases first. Prefix regex and core-stripping both compile from this
# tuple so a bound hedge cannot be visible to one check and invisible to the other.
_QUANTITY_BOUND_PREFIXES = (
    "as many as",
    "no more than",
    "not more than",
    "more than",
    "fewer than",
    "less than",
    "at least",
    "at most",
    "up to",
    "over",
    "about",
    "approximately",
    "nearly",
    "almost",
)


def _bound_prefix_regex() -> str:
    ordered = sorted(
        _QUANTITY_BOUND_PREFIXES,
        key=lambda phrase: (-len(phrase.split()), -len(phrase)),
    )
    return "|".join(
        r"\s+".join(re.escape(part) for part in phrase.split())
        for phrase in ordered
    )


_QUANTITY_PREFIX = r"(?:" + _bound_prefix_regex() + r")\s+"
_QUANTITY_CORE_PREFIX = re.compile(
    r"^(?:" + _bound_prefix_regex() + r")\s+",
    re.IGNORECASE,
)
_NUMBER_WORD = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
_SMALL_CARDINAL = r"(?:" + _NUMBER_WORD + r")"
_TENS_WORD = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_ONES_WORD = r"(?:one|two|three|four|five|six|seven|eight|nine)"
_COMPOUND_CARDINAL = _TENS_WORD + r"[-\s]+" + _ONES_WORD
_ARTICLE_OR_CARDINAL = r"(?:a|an|" + _NUMBER_WORD + r"|" + _COMPOUND_CARDINAL + r")"
_NUMBER_SCALE = r"(?:hundred|thousand|million|billion)"
_POPULATION_NOUN = (
    r"(?:people|persons?|patients?|residents?|infants?|children|adults|"
    r"workers?|staff|personnel|officers?|nurses?|guards?|medics?|"
    r"firefighters?|families|households?|communities|students?|animals?)"
)
_COLLECTIVE_QUANTITY = (
    r"(?:dozen|score|tens|hundreds|thousands|dozens|millions|billions)"
    r"(?:\s+of\s+(?:tens\s+of\s+)?(?:thousands|millions|billions))?"
)
# Longer compounds are listed first so "over five hundred" wins over "five"
# and "twenty-five patients" wins over the nested "five".
# Articles count only in "a hundred"/"an thousand"-style compounds, never as
# a bare cardinality before a population noun ("a dozen residents").
# Hyphen is a number-word joiner, not a boundary, so "five" in "twenty-five"
# is not a population count.
_EXPLICIT_QUANTITY = re.compile(
    r"(?<![\w.-])(?:"
    r"(?:" + _QUANTITY_PREFIX + r")?" + _ARTICLE_OR_CARDINAL + r"[-\s]+"
    + _NUMBER_SCALE + r"(?:[-\s]+" + _NUMBER_SCALE + r")?"
    r"|(?:" + _QUANTITY_PREFIX + r")?" + _COMPOUND_CARDINAL +
    r"|(?:" + _QUANTITY_PREFIX + r")?\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s*" + _DURATION_UNIT + r")?"
    r"|" + _SMALL_CARDINAL + r"\s+" + _DURATION_UNIT +
    r"|" + _COLLECTIVE_QUANTITY +
    r"|" + _SMALL_CARDINAL + r"(?![-\s]+(?:" + _NUMBER_SCALE + r"|" + _ONES_WORD + r"))"
    r"(?=\s+(?:[a-z-]+\s+){0,2}" + _POPULATION_NOUN + r")"
    r")(?![\w.-])",
    re.IGNORECASE,
)
# (canonical, regex fragment, high_confidence, negated_chance).
# Qualifier, high-confidence compact-role, and negated-survival binding all
# compile from this tuple so a hedge cannot be visible to one check and
# invisible to the others. Longer phrases sort first at compile time.
_LIKELIHOOD_HEDGES = (
    ("almost no chance", r"almost\s+no\s+chance", True, True),
    ("virtually no chance", r"virtually\s+no\s+chance", True, True),
    ("no chance", r"no\s+chance", True, True),
    ("little chance", r"little\s+chance", False, True),
    ("slim chance", r"slim\s+chance", False, True),
    ("a chance", r"a\s+chance", False, False),
    ("remote chance", r"remote\s+chance", False, False),
    ("some chance", r"some\s+chance", False, False),
    ("near-certain", r"near[- ]certain", True, False),
    ("almost certain", r"almost\s+certain", True, False),
    ("virtually certain", r"virtually\s+certain", True, False),
    ("nearly certain", r"nearly\s+certain", True, False),
    ("highly likely", r"highly\s+likely", False, False),
    ("chance", r"chance", False, False),
    ("could", r"could", False, False),
    ("might", r"might", False, False),
    ("may", r"may", False, False),
    ("probably", r"probably", False, False),
    ("possibly", r"possibly", False, False),
    ("likely", r"likely", False, False),
    ("unlikely", r"unlikely", False, False),
    ("possible", r"possible", False, False),
    ("uncertain", r"uncertain", False, False),
    ("unknown", r"unknown", False, False),
)

# Percent + chance/risk is a likelihood phrase, not a population count.
# Quantity extraction and likelihood extraction both consult this tuple.
_PERCENT_CHANCE_CUES = (
    "chance", "risk", "probability", "odds", "likelihood",
)


def _percent_chance_regex() -> str:
    cues = "|".join(re.escape(cue) for cue in _PERCENT_CHANCE_CUES)
    return (
        r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent)\s*(?:" + cues + r")"
    )


_AT_RISK_MAGNITUDES = (
    "low", "moderate", "high", "severe", "elevated", "serious",
)


def _at_risk_situation_regex() -> str:
    magnitudes = "|".join(re.escape(item) for item in _AT_RISK_MAGNITUDES)
    return (
        r"at\s+(?:(?:very\s+)?(?:" + magnitudes + r")\s+)?risk"
    )


_PERCENT_CHANCE_TAIL = re.compile(
    r"^\s*(?:%|percent)?\s*(?:"
    + "|".join(re.escape(cue) for cue in _PERCENT_CHANCE_CUES)
    + r")\b",
    re.IGNORECASE,
)


def _ordered_likelihood_hedges() -> tuple[tuple[str, str, bool, bool], ...]:
    return tuple(sorted(
        _LIKELIHOOD_HEDGES,
        key=lambda row: (-len(row[0].split()), -len(row[0])),
    ))


def _likelihood_alternation(predicate) -> str:
    return "|".join(
        row[1] for row in _ordered_likelihood_hedges() if predicate(row)
    )


_LIKELIHOOD_QUALIFIER = re.compile(
    r"\b(?:" + _percent_chance_regex() + r"|"
    + _at_risk_situation_regex() + r"|"
    + _likelihood_alternation(lambda _row: True) + r")\b",
    re.IGNORECASE,
)
_AT_RISK_SITUATION = re.compile(
    r"\b" + _at_risk_situation_regex() + r"\b",
    re.IGNORECASE,
)
_EXPOSURE_OUTCOME = re.compile(
    r"\b(?:at\s+(?:(?:very\s+)?(?:low|moderate|high|severe|elevated|serious)\s+)?"
    r"risk|exposed|exposure)\b",
    re.IGNORECASE,
)
_HIGH_CONFIDENCE_LIKELIHOOD = re.compile(
    r"\b(?:" + _likelihood_alternation(lambda row: row[2]) + r")\b",
    re.IGNORECASE,
)
_NEGATED_CHANCE = re.compile(
    r"\b(?:" + _likelihood_alternation(lambda row: row[3]) + r")\b",
    re.IGNORECASE,
)
_SCOPE_QUALIFIER = re.compile(
    r"\b(?:widespread|citywide|systemwide|nationwide|regional|localized|"
    r"broader|limited|narrow)\b", re.IGNORECASE,
)
_TEMPORAL_QUALIFIER = re.compile(
    r"\b(?:immediate|immediately|imminent|near[- ]term|short[- ]term|"
    r"long[- ]term|prolonged|ongoing|future)\b", re.IGNORECASE,
)
_CHAINED_OUTCOME = re.compile(
    r"\b(?:thereby|thus|which\s+(?:causes?|leads?|enables?|prevents?)|"
    r"resulting\s+in|leading\s+to|enabling\s+.+\s+to|"
    r"sustaining\s+(?:people|persons?|patients?|residents?|workers?|families|communities))\b",
    re.IGNORECASE,
)
_HUMAN_OUTCOME = re.compile(
    r"\b(?:death|die|dies|drown|drowns|drowned|drowning|surviv|health|"
    r"medical|injur|hunger|thirst|well-?being|"
    r"sustain(?:s|ing)?\s+(?:people|persons?|patients?|residents?|"
    r"workers?|families|communities))\b",
    re.IGNORECASE,
)
_NONHUMAN_PARTY_KINDS = {
    "ORGANIZATION", "INSTITUTION", "AUTOMATED_SYSTEM",
    "AUTOMATED_DECISION_SYSTEM", "RESOURCE", "INFRASTRUCTURE",
}


def classify_clause_role(text: str) -> str:
    """Structural role of a source span. Domain-neutral; no dilemma vocabulary."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return "FACT"
    if cleaned.endswith("?"):
        return "INTERROGATIVE"
    if _COMPARISON_CUE.search(cleaned):
        return "COMPARISON"
    return "FACT"


def is_supporting_source(ref: SourceRef) -> bool:
    """True when a provenance ref can carry an effect claim, not just frame it.

    Confirmed action text (A0, A1, …) and FACT clauses support claims.
    Comparison/interrogative clauses may appear as context only.
    """
    if _ACTION_SOURCE_ID.match(ref.clause_id or ""):
        return True
    return classify_clause_role(ref.excerpt) == "FACT"


_CARDINAL_MAGNITUDE = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1_000_000, "billion": 1_000_000_000,
}


def quantity_magnitude(span: str) -> int | None:
    """Best-effort cardinal for comparing party totals to assigned subsets."""
    text = str(span or "").casefold().replace(",", "")
    digits = re.findall(r"\d+", text)
    if digits:
        return int(digits[0])
    total = 0
    matched = False
    last_scale = 1
    for word in re.findall(r"[a-z]+", text):
        value = _CARDINAL_MAGNITUDE.get(word)
        if value is None:
            continue
        matched = True
        if value >= 100:
            total = (total or 1) * value
            last_scale = value
        else:
            total += value * last_scale if last_scale > 1 and total and value < last_scale else value
    return total if matched else None


def explicit_quantity_spans(text: str) -> tuple[str, ...]:
    """Return numerical and scale phrases copied from source wording.

    Deliberately incomplete: no invented probabilities/QALYs, no age labels,
    no bare duration units, and no vague comparatives such as 'few' or 'many'.
    Nested spans collapse to the longest source phrase, so 'over five hundred'
    is not also recorded as 'five hundred' or 'five', and 'as many as three
    hundred' or 'at most three hundred' is not also recorded as 'three hundred'.
    A percent that modifies chance, risk, probability, odds, or likelihood is
    a likelihood qualifier, not a quantity: '30% chance' is not also '30%'.
    """
    stripped = _AGE_LABEL.sub(" ", str(text or ""))
    matches = [
        (match.start(), match.end(), " ".join(match.group(0).split()))
        for match in _EXPLICIT_QUANTITY.finditer(stripped)
        if match.group(0).strip()
    ]
    kept: list[str] = []
    seen: set[str] = set()
    for start, end, span in matches:
        contained = any(
            not (other_start == start and other_end == end)
            and other_start <= start and end <= other_end
            for other_start, other_end, _span in matches
        )
        if contained:
            continue
        if _quantity_span_is_chance_percent(span, stripped):
            continue
        key = span.casefold()
        if key not in seen:
            seen.add(key)
            kept.append(span)
    return tuple(kept)


def _quantity_span_is_chance_percent(span: str, text: str) -> bool:
    """True when a numeral is the magnitude of a chance/risk phrase.

    Includes glued source '10%chance', so '10' and '10%' are not party counts.
    """
    match = re.search(re.escape(span), str(text or ""), re.IGNORECASE)
    if match is None:
        return False
    return bool(_PERCENT_CHANCE_TAIL.match(str(text)[match.end():]))


def _nested_recorded_quantities(quantities: Sequence[str]) -> tuple[str, ...]:
    """Shorter recorded spans that are already covered by a longer sibling."""
    cleaned = [str(value).strip() for value in quantities if str(value).strip()]
    nested: list[str] = []
    for short in cleaned:
        for long in cleaned:
            if short.casefold() == long.casefold():
                continue
            if re.search(
                rf"(?<![\w.]){re.escape(short)}(?![\w.])", long, re.IGNORECASE,
            ):
                nested.append(short)
                break
    return tuple(dict.fromkeys(nested))


def _explicit_qualifier_spans(text: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    """Copy source qualifier phrases; nested matches collapse to the longest."""
    matches = [
        (match.start(), match.end(), " ".join(match.group(0).split()))
        for match in pattern.finditer(str(text or ""))
        if match.group(0).strip()
    ]
    kept: list[str] = []
    seen: set[str] = set()
    for start, end, span in matches:
        contained = any(
            not (other_start == start and other_end == end)
            and other_start <= start and end <= other_end
            for other_start, other_end, _span in matches
        )
        if contained:
            continue
        key = span.casefold()
        if key not in seen:
            seen.add(key)
            kept.append(span)
    return tuple(kept)


@dataclass(frozen=True, slots=True)
class QualifierSpan:
    """One source qualifier, with the literal offsets and the folded identity."""

    literal: str
    canonical: str
    start: int
    end: int


def canonical_likelihood_span(span: str) -> str:
    """Whitespace-fold a likelihood phrase without stripping a nested hedge.

    Idempotent. Extractor, qualifier binding, and the unhedged-indicative gate
    all treat this string as the identity of the hedge: 'almost no chance',
    not 'chance'. A source that omits the space in '20%chance' still
    canonicalizes to '20% chance'. The literal and its offsets live on
    QualifierSpan; this function does not replace them.
    """
    folded = " ".join(str(span or "").split())
    folded = re.sub(
        r"(%)(?=(?:chance|risk|probability|odds|likelihood)\b)",
        r"% ",
        folded,
        flags=re.IGNORECASE,
    )
    folded = re.sub(
        r"(percent)(?=(?:chance|risk|probability|odds|likelihood)\b)",
        r"\1 ",
        folded,
        flags=re.IGNORECASE,
    )
    return " ".join(folded.split())


def _likelihood_identity_key(span: str) -> str:
    return canonical_likelihood_span(span).casefold()


def _unique_likelihood_spans(spans: Iterable[Any]) -> tuple[str, ...]:
    """Collapse equivalent chance phrases, including glued vs spaced percents."""
    kept: list[str] = []
    have: set[str] = set()
    for value in spans:
        cleaned = _clean(value, 80)
        if not cleaned:
            continue
        key = _likelihood_identity_key(cleaned)
        if key not in have:
            kept.append(cleaned)
            have.add(key)
    return tuple(kept)


def explicit_likelihood_span_records(text: str) -> tuple[QualifierSpan, ...]:
    """Literal source matches, including glued '20%chance', with offsets."""
    blob = str(text or "")
    matches = [
        match for match in _LIKELIHOOD_QUALIFIER.finditer(blob)
        if match.group(0).strip()
    ]
    kept: list[QualifierSpan] = []
    for match in matches:
        contained = any(
            not (other.start() == match.start() and other.end() == match.end())
            and other.start() <= match.start() and match.end() <= other.end()
            for other in matches
        )
        if contained:
            continue
        literal = match.group(0)
        kept.append(QualifierSpan(
            literal=literal,
            canonical=canonical_likelihood_span(literal),
            start=match.start(),
            end=match.end(),
        ))
    return tuple(kept)


def explicit_likelihood_spans(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        record.canonical for record in explicit_likelihood_span_records(text)
    ))


def explicit_scope_spans(text: str) -> tuple[str, ...]:
    return _explicit_qualifier_spans(text, _SCOPE_QUALIFIER)


def explicit_temporal_spans(text: str) -> tuple[str, ...]:
    return _explicit_qualifier_spans(text, _TEMPORAL_QUALIFIER)


def _qualifier_channels() -> tuple[tuple[str, str, Any], ...]:
    """Likelihood, scope, and time share one grammatical-head binder.

    Parse fill/strip, admission omit-checks, and the hedged-source gate all
    iterate this tuple. Do not bind one channel with a private overlap rule.
    """
    return (
        ("likelihood", "likelihood_qualifiers", explicit_likelihood_spans),
        ("scope", "scope_qualifiers", explicit_scope_spans),
        ("temporal", "temporal_qualifiers", explicit_temporal_spans),
    )


def _provenance_text(effect: WorldEffect) -> str:
    """Text that may ground a claim. Generated outcome sentences are excluded."""
    return " ".join(ref.excerpt for ref in effect.provenance if ref.excerpt)


def _normalize_quantity_key(span: str) -> str:
    return re.sub(r"[\s-]+", " ", str(span or "").casefold()).strip()


def canonical_quantity_span(span: str) -> str:
    """Whitespace-fold a quantity phrase without stripping a bound prefix.

    Idempotent. Extractor, completeness, and nested-span checks all treat this
    string as the thing to record: 'as many as three hundred', not 'three hundred'.
    """
    folded = " ".join(str(span or "").split())
    if not folded:
        return ""
    match = _QUANTITY_CORE_PREFIX.match(folded)
    if match is None:
        return folded
    prefix = " ".join(match.group(0).split())
    core = folded[match.end():].strip()
    if not core:
        return prefix
    return f"{prefix} {core}"


def _quantity_core(span: str) -> str:
    """Hyphen/space-folded count, stripping over/more-than style prefixes."""
    return _QUANTITY_CORE_PREFIX.sub("", _normalize_quantity_key(span)).strip()


def _quantities_equivalent(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left.casefold() == right.casefold():
        return True
    core = _quantity_core(left)
    return bool(core) and core == _quantity_core(right)


def _quantity_grounded_in_text(span: str, text: str) -> bool:
    """True when a recorded span is the same count as a source quantity phrase."""
    blob = str(text or "")
    if span and span.casefold() in blob.casefold():
        return True
    return any(
        _quantities_equivalent(span, candidate)
        for candidate in explicit_quantity_spans(blob)
    )


def _party_licenses_quantity(party: WorldParty | None, span: str) -> bool:
    """Anaphoric clauses may repeat a party's recorded count without the numeral.

    The count must already live on that party and in that party's provenance.
    Another party that shares the clause cannot inherit it.
    """
    if party is None or not span:
        return False
    licensed = [
        recorded for recorded in party.quantities
        if _quantities_equivalent(span, recorded)
    ]
    if not licensed:
        return False
    party_text = " ".join(ref.excerpt for ref in party.provenance if ref.excerpt)
    return any(_quantity_grounded_in_text(recorded, party_text) for recorded in licensed)


_PARTY_MATCH_STOPWORDS = {
    "affected", "city", "group", "people", "person", "population", "relying",
    "the", "their", "those",
}
_PARTY_GENERIC_NOUNS = {
    "community", "communities", "group", "household", "households",
    "people", "person", "persons", "population", "resident", "residents",
}
_PARTY_QUANTITY_KINDS = {
    "COMMUNITY", "GROUP", "HOUSEHOLD", "HUMAN_GROUP",
    "POPULATION", "POPULATION_GROUP",
}
_EFFECT_MATCH_STOPWORDS = {
    "affected", "cause", "caused", "causes", "effect", "failure", "outcome",
    "prevent", "prevented", "prevents", "risk", "the", "their", "would",
}
_QUALIFIER_HEAD_STOPWORDS = _EFFECT_MATCH_STOPWORDS | {
    "also", "face", "faces", "facing", "from", "into", "less", "more",
    "onto", "such", "than", "that", "them", "then", "these", "they",
    "this", "those", "very", "with",
}
_QUALIFIER_CLAUSE_BOUNDARY = re.compile(
    r"[;.]|,\s*(?:and|but|or|while|whereas)\b|\b(?:and|but|or|while|whereas)\b",
    re.IGNORECASE,
)
_QUALIFIER_LEADING_COORDINATOR = re.compile(
    r"^(?:and|but|or|while|whereas)\b[, ]*",
    re.IGNORECASE,
)
_CLAUSE_COMPLEMENT_START = re.compile(
    r"^(?:that|the|of|to)\b",
    re.IGNORECASE,
)
_COMPLEMENT_RESULT_TAIL = re.compile(r",\s+[A-Za-z]+ing\b")
_LIGHT_COMPLEMENT_VERBS = frozenset({
    "be", "been", "being", "get", "gets", "getting", "got",
})
_UNINFLECTED_COMPLEMENT_VERBS = frozenset({
    "block", "choke", "collapse", "die", "drown", "fail", "flood", "foul",
})
_DEATH_WORDS = frozenset({
    "death", "die", "died", "dies", "dying",
    "drown", "drowned", "drowning", "drowns",
})
_SURVIVAL_WORDS = frozenset({
    "escape", "escaped", "escapes", "escaping",
    "survival", "survive", "survived", "survives", "surviving",
})
_QUALIFIER_EQUIVALENTS = (
    _DEATH_WORDS,
    frozenset({"harm", "harmed", "harming", "harms"}),
    frozenset({"injure", "injured", "injuries", "injuring", "injury"}),
    _SURVIVAL_WORDS,
    frozenset({"held", "hold", "holding", "holds"}),
    frozenset({"contaminate", "contaminated", "contaminates", "contaminating"}),
)
_PROCESS_EFFECT_KINDS = {
    "CAPABILITY_CHANGE", "INTERVENTION", "OTHER", "PHYSICAL_STATE",
    "RESOURCE_TRANSFER",
}
_WELFARE_EFFECT_KINDS = {
    "HEALTH_OUTCOME", "INSTITUTIONAL_OUTCOME", "WELFARE_OUTCOME",
}
_CONDITION_STOPWORDS = _EFFECT_MATCH_STOPWORDS | {
    "also", "are", "been", "being", "continue", "continued", "continues",
    "do", "does", "if", "into", "is", "its", "provided", "remain",
    "remaining", "remains", "still", "that", "the", "them", "these",
    "this", "those", "unless", "was", "were", "when", "whenever",
    "whether", "with",
}
_SOURCE_HEDGE_CUE = re.compile(
    r"\b(?:if|unless|provided\s+that|depending|in\s+case|whether)\b",
    re.IGNORECASE,
)


def _match_words(text: str, *, stopwords: set[str]) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(word) > 2 and word not in stopwords
    }


def _ordered_content_words(text: str, *, stopwords: set[str]) -> list[str]:
    return [
        word for word in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text).casefold())
        if len(word) > 2 and word not in stopwords
    ]


def _token_is_closed_qualifier(word: str) -> bool:
    return bool(
        explicit_likelihood_spans(word)
        or explicit_scope_spans(word)
        or explicit_temporal_spans(word)
    )


def _quantity_local_window(text: str, quantity: str) -> str:
    """Quantity plus the following noun phrase, stopping before the next clause."""
    match = re.search(re.escape(quantity), str(text or ""), re.IGNORECASE)
    if match is None:
        return ""
    tail = str(text)[match.end():]
    tail = re.split(
        r"[,;.]|\b(?:and|but|or|while|whereas)\b",
        tail, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    return str(text)[match.start():match.end()] + tail


def _party_quantity_score(party: WorldParty, window: str) -> int:
    """Prefer distinctive label tokens; generic crowd nouns are weak ties."""
    label_tokens = {
        word for word in re.findall(r"[a-z0-9]+", party.label.casefold())
        if len(word) > 2 and word not in _PARTY_MATCH_STOPWORDS
    }
    window_tokens = {
        word for word in re.findall(r"[a-z0-9]+", window.casefold())
        if len(word) > 2 and word not in _PARTY_MATCH_STOPWORDS
    }
    distinctive = (label_tokens - _PARTY_GENERIC_NOUNS) & (
        window_tokens - _PARTY_GENERIC_NOUNS
    )
    generic = (label_tokens & _PARTY_GENERIC_NOUNS) & window_tokens
    return 10 * len(distinctive) + len(generic)


def assigned_party_quantities(
    parties: Sequence[WorldParty],
) -> dict[str, tuple[str, ...]]:
    """Bind each provenance quantity span to at most one uniquely matching party."""
    population = [
        party for party in parties if party.kind in _PARTY_QUANTITY_KINDS
    ]
    assigned: dict[str, list[str]] = {party.party_id: [] for party in parties}
    seen: set[tuple[str, str]] = set()
    for party in population:
        for ref in party.provenance:
            text = ref.excerpt
            for quantity in explicit_quantity_spans(text):
                job = (text, quantity.casefold())
                if job in seen:
                    continue
                seen.add(job)
                window = _quantity_local_window(text, quantity)
                scored = [
                    (_party_quantity_score(other, window), other.party_id)
                    for other in population
                    if any(item.excerpt == text for item in other.provenance)
                ]
                if not scored:
                    continue
                best = max(score for score, _party_id in scored)
                winners = [
                    party_id for score, party_id in scored if score == best
                ]
                if best > 0 and len(winners) == 1:
                    assigned[winners[0]].append(quantity)
    return {
        party_id: tuple(dict.fromkeys(values))
        for party_id, values in assigned.items()
    }


_ASSIGNMENT_CUE = re.compile(
    r"\b(?:assign(?:ed|s|ment)?|allocat(?:e|ed|es|ion)|"
    r"stay|stays|staying|assist(?:s|ed|ing)?|"
    r"remain(?:s|ing)?(?:\s+to\s+assist)?)\b",
    re.IGNORECASE,
)


def assignment_owned_quantities(
    effects: Sequence[WorldEffect],
) -> dict[str, tuple[str, ...]]:
    """Quantities that a DIRECT assignment/assist row owns for its party."""
    owned: dict[str, list[str]] = {}
    for effect in effects:
        if effect.directness != "DIRECT":
            continue
        blob = str(effect.outcome or "").replace("_", " ")
        if not _ASSIGNMENT_CUE.search(blob):
            continue
        for quantity in effect.quantities:
            cleaned = str(quantity).strip()
            if cleaned:
                owned.setdefault(effect.party_id, []).append(cleaned)
    return {
        party_id: tuple(dict.fromkeys(values))
        for party_id, values in owned.items()
    }


def _is_smaller_assignment_quantity(
    quantity: str,
    party_quantities: Sequence[str],
    owned: Sequence[str],
) -> bool:
    if quantity.casefold() not in {item.casefold() for item in owned}:
        return False
    magnitude = quantity_magnitude(quantity)
    if magnitude is None:
        return False
    siblings = [
        quantity_magnitude(item) for item in party_quantities
        if item.casefold() != quantity.casefold()
    ]
    return any(other is not None and other > magnitude for other in siblings)


def strip_assignment_subgroups_from_parties(
    parties: Sequence[WorldParty],
    effects: Sequence[WorldEffect],
) -> tuple[WorldParty, ...]:
    """Keep the party total; leave the assigned subset on the assignment effect.

    'fifty firefighters' stays the party. 'five firefighters' stays the
    DIRECT assignment quantity. Compact roles read the effect quantity.
    """
    owned = assignment_owned_quantities(effects)
    updated: list[WorldParty] = []
    for party in parties:
        kept = [
            quantity for quantity in party.quantities
            if not _is_smaller_assignment_quantity(
                quantity, party.quantities, owned.get(party.party_id, ()),
            )
        ]
        updated.append(replace(party, quantities=tuple(kept)))
    return tuple(updated)


def closed_class_party_quantities(
    parties: Sequence[WorldParty],
) -> tuple[WorldParty, ...]:
    """Keep each party's uniquely assigned spans; drop sibling leaks; fill gaps.

    Unique assignment needs the full party list. Do not ask a singleton
    party to classify every span in its provenance. Parse fill/strip and
    the schema 1.2 omit-check share assigned_party_quantities.
    """
    expected = assigned_party_quantities(parties)
    return tuple(
        replace(party, quantities=expected.get(party.party_id, ()))
        for party in parties
    )


def _expand_equivalent_words(words: set[str]) -> set[str]:
    expanded = set(words)
    for group in _QUALIFIER_EQUIVALENTS:
        if words & group:
            expanded |= set(group)
    return expanded


def _qualifier_modified_words(text: str, start: int, end: int) -> set[str]:
    """Content words the qualifier modifies: following head, else preceding NP."""
    after = _QUALIFIER_LEADING_COORDINATOR.sub("", str(text)[end:].strip())
    after_chunk = _QUALIFIER_CLAUSE_BOUNDARY.split(after, maxsplit=1)[0]
    after_words = _match_words(after_chunk, stopwords=_QUALIFIER_HEAD_STOPWORDS)
    if after_words:
        return after_words
    before = str(text)[:start]
    parts = _QUALIFIER_CLAUSE_BOUNDARY.split(before)
    before_chunk = parts[-1] if parts else before
    return _match_words(before_chunk, stopwords=_QUALIFIER_HEAD_STOPWORDS)


def _looks_like_complement_predicate(word: str) -> bool:
    """True for a verbal complement head, not the object noun of an SVO clause."""
    folded = str(word or "").casefold()
    if not folded or folded in _LIGHT_COMPLEMENT_VERBS:
        return False
    if folded in _UNINFLECTED_COMPLEMENT_VERBS:
        return True
    return folded.endswith(("ing", "ied", "ed", "es"))


def _clause_complement_focus(after_chunk: str, ordered: list[str]) -> set[str]:
    """Predicate of a chance/that/of/to complement, not the first following noun.

    'a 20% chance the device fails' modifies fails, not device.
    'a 25% chance that residue chokes the catchment, inundating the works'
    modifies chokes, not catchment or the result participle.
    'a 10% chance of being blocked' modifies blocked, not being.
    Attributive hedges ('near-certain death') do not use this path.
    """
    chunk = str(after_chunk or "")
    split = _COMPLEMENT_RESULT_TAIL.split(chunk, maxsplit=1)
    if split:
        chunk = split[0]
        ordered = _ordered_content_words(
            chunk, stopwords=_QUALIFIER_HEAD_STOPWORDS,
        )
    match = _CLAUSE_COMPLEMENT_START.match(chunk.strip())
    if match is None:
        return set()
    leader = match.group(0).casefold()
    content = [word for word in ordered if not _token_is_closed_qualifier(word)]
    if not content:
        return set()
    if leader in {"the", "that"}:
        verbal = [
            word for word in content if _looks_like_complement_predicate(word)
        ]
        focus = verbal[-1] if verbal else content[-1]
    else:
        focus = next(
            (word for word in content if word not in _LIGHT_COMPLEMENT_VERBS),
            content[-1],
        )
    return _expand_equivalent_words({focus})


def _qualifier_focus_words(text: str, start: int, end: int) -> set[str]:
    """The modified head word, skipping stacked closed-class qualifiers.

    'immediate and prolonged harm' both modify harm, not the sibling
    adjective. 'immediate holding the facility' modifies holding, not every
    row that mentions the facility. A chance/risk complement binds to the
    clause predicate so a verb-only outcome still matches.
    """
    after = _QUALIFIER_LEADING_COORDINATOR.sub("", str(text)[end:].strip())
    after_chunk = _QUALIFIER_CLAUSE_BOUNDARY.split(after, maxsplit=1)[0]
    ordered = _ordered_content_words(
        after_chunk, stopwords=_QUALIFIER_HEAD_STOPWORDS,
    )
    from_after = bool(ordered)
    if not ordered:
        before = str(text)[:start]
        parts = _QUALIFIER_CLAUSE_BOUNDARY.split(before)
        before_chunk = parts[-1] if parts else before
        ordered = list(reversed(_ordered_content_words(
            before_chunk, stopwords=_QUALIFIER_HEAD_STOPWORDS,
        )))
    if from_after:
        complement = _clause_complement_focus(after_chunk, ordered)
        if complement:
            return complement
    for word in ordered:
        if _token_is_closed_qualifier(word):
            continue
        return _expand_equivalent_words({word})
    return set()


def _head_is_human_outcome(words: set[str]) -> bool:
    expanded = _expand_equivalent_words(words)
    blob = " ".join(sorted(words))
    return bool(_HUMAN_OUTCOME.search(blob)) or bool(
        expanded & _DEATH_WORDS or expanded & _SURVIVAL_WORDS
    )


def _effect_is_process_row(effect: WorldEffect) -> bool:
    if effect.effect_kind in _WELFARE_EFFECT_KINDS:
        return False
    if _HUMAN_OUTCOME.search(effect.outcome):
        return False
    return effect.effect_kind in _PROCESS_EFFECT_KINDS


def _qualifier_span_match(text: str, qualifier: str) -> re.Match[str] | None:
    match = re.search(re.escape(qualifier), text, re.IGNORECASE)
    if match is None:
        relaxed = re.escape(qualifier)
        relaxed = re.sub(
            r"(%|percent)\\\s+",
            r"\1\\s*",
            relaxed,
            flags=re.IGNORECASE,
        )
        relaxed = relaxed.replace(r"\ ", r"[\s-]+")
        match = re.search(relaxed, text, re.IGNORECASE)
    return match


def _qualifier_grounded_in_text(span: str, text: str) -> bool:
    """True when this span is the literal or the canonical of a source match.

    A glued source '20%chance' grounds both that literal (at its offsets) and
    the canonical '20% chance'. A spaced source does not ground a glued form
    that never appeared.
    """
    if not span:
        return False
    blob = str(text or "")
    if span.casefold() in blob.casefold():
        return True
    if _qualifier_span_match(blob, span) is not None:
        return True
    folded = span.casefold()
    for record in explicit_likelihood_span_records(blob):
        if blob[record.start:record.end] != record.literal:
            continue
        if folded == record.literal.casefold():
            return True
        if folded == record.canonical.casefold():
            return True
    return False


def _folded_outcome_text(outcome: str) -> str:
    return re.sub(r"[_-]+", " ", str(outcome or "")).casefold()


def _outcome_is_exposure_state(outcome: str) -> bool:
    return bool(_EXPOSURE_OUTCOME.search(_folded_outcome_text(outcome)))


def _is_risk_situation_span(span: str) -> bool:
    return bool(_AT_RISK_SITUATION.fullmatch(canonical_likelihood_span(span).strip()))


def _risk_span_has_harm_complement(text: str, start: int, end: int) -> bool:
    after = _QUALIFIER_LEADING_COORDINATOR.sub("", str(text)[end:].strip())
    after_chunk = _QUALIFIER_CLAUSE_BOUNDARY.split(after, maxsplit=1)[0]
    if _CLAUSE_COMPLEMENT_START.match(after_chunk.strip()) is None:
        return False
    return _head_is_human_outcome(_qualifier_focus_words(text, start, end))


def _risk_situation_licenses_chance(effect: WorldEffect, span: str) -> bool:
    for ref in effect.provenance:
        match = _qualifier_span_match(ref.excerpt, span)
        if match is not None and _risk_span_has_harm_complement(
            ref.excerpt, match.start(), match.end(),
        ):
            return True
    return False


def _should_type_risk_situation_certain(effect: WorldEffect) -> bool:
    """Occupying an at-risk situation is stipulated CERTAIN, not a chance event.

    'at high risk of dying' still licenses a non-certain harm row.
    """
    if not _outcome_is_exposure_state(effect.outcome):
        return False
    spans = [
        span for span in effect.likelihood_qualifiers
        if _is_risk_situation_span(span)
    ]
    if not spans:
        return False
    return not any(
        _risk_situation_licenses_chance(effect, span) for span in spans
    )


def _chance_hedges_on_effect(effect: WorldEffect) -> tuple[str, ...]:
    """Likelihood hedges that are not an at-risk situation without a harm complement."""
    return tuple(
        span for span in effect.likelihood_qualifiers
        if span.strip() and not (
            _is_risk_situation_span(span)
            and not _risk_situation_licenses_chance(effect, span)
        )
    )


def _chance_hedge_blocks_certain(effect: WorldEffect) -> bool:
    """True when a chance hedge forbids leaving this row typed CERTAIN.

    Occupying an at-risk situation stays CERTAIN. 'almost certain' death does not.
    """
    if _should_type_risk_situation_certain(effect):
        return False
    return bool(_chance_hedges_on_effect(effect))


def _qualifier_binds_to_effect(
    effect: WorldEffect,
    text: str,
    start: int,
    end: int,
    *,
    qualifier: str = "",
    extractor: Any = None,
) -> bool:
    """True when this span's grammatical head is this effect's outcome.

    A human-outcome head does not bind to a non-welfare process row even if a
    crowd noun is shared. Parse, admission, and the hedged-source gate share
    this decision; do not reimplement overlap against a character window.
    An 'at risk' situation span binds to an exposure/AT_RISK outcome unless
    a harm complement ('of dying') makes it a chance hedge on that harm.
    Deverbal nouns share stems with participles: blockage/BLOCKED,
    failure/FAILS.
    """
    if (
        extractor is explicit_likelihood_spans
        and _is_risk_situation_span(qualifier)
        and not _risk_span_has_harm_complement(text, start, end)
    ):
        return (
            _outcome_is_exposure_state(effect.outcome)
            and not _effect_is_process_row(effect)
        )
    effect_words = _expand_equivalent_words(
        _match_words(effect.outcome, stopwords=_EFFECT_MATCH_STOPWORDS)
    )
    head = _expand_equivalent_words(
        _qualifier_modified_words(text, start, end)
    )
    focus = _qualifier_focus_words(text, start, end)
    effect_stems = _binder_stems(effect_words)
    focus_stems = _binder_stems(focus)
    overlap = bool(effect_stems & focus_stems)
    negated_survival_to_death = (
        extractor is explicit_likelihood_spans
        and bool(qualifier)
        and _NEGATED_CHANCE.search(qualifier)
        and bool(head & _SURVIVAL_WORDS)
        and bool(effect_words & _DEATH_WORDS)
        and not _effect_is_process_row(effect)
    )
    if not overlap and not negated_survival_to_death:
        return False
    if _head_is_human_outcome(head) and _effect_is_process_row(effect):
        return False
    return True


def effect_expected_qualifiers(
    effect: WorldEffect, extractor: Any,
) -> tuple[str, ...]:
    """Bind source qualifiers to the outcome they modify, not sibling rows.

    Token overlap anywhere in a ±28-character window was enough to copy
    "near-certain death" onto a same-clause "power lost" row. Attachment now
    uses the qualifier's grammatical head: the first modified word for
    attributive hedges, the clause predicate for chance/that/of/to
    complements. A human-outcome head does not bind to a non-welfare process
    row even if a crowd noun is shared.

    Parse fill/strip, schema 1.2 omit-checks, and `_supporting_source_is_hedged`
    all call this. Downstream graph, ledger, and compact copy the resulting
    tuples; they must not re-bind from prose.
    """
    found: list[str] = []
    for ref in effect.provenance:
        text = ref.excerpt
        if extractor is explicit_likelihood_spans:
            attachments = [
                (record.start, record.end, record.canonical)
                for record in explicit_likelihood_span_records(text)
            ]
        else:
            attachments = []
            for qualifier in extractor(text):
                match = _qualifier_span_match(text, qualifier)
                if match is not None:
                    attachments.append((match.start(), match.end(), qualifier))
        for start, end, qualifier in attachments:
            if _qualifier_binds_to_effect(
                effect, text, start, end,
                qualifier=qualifier, extractor=extractor,
            ):
                found.append(qualifier)
    return tuple(dict.fromkeys(found))


_effect_expected_qualifiers = effect_expected_qualifiers


@dataclass(frozen=True, slots=True)
class SourceRef:
    clause_id: str
    excerpt: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorldParty:
    party_id: str
    label: str
    kind: str = "OTHER"
    provenance: tuple[SourceRef, ...] = ()
    quantities: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorldCondition:
    condition_id: str
    description: str
    value_status: str = "UNKNOWN"
    decision_relevance: str = "MATERIAL"
    provenance: tuple[SourceRef, ...] = ()
    event_effect_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorldEffect:
    effect_id: str
    action_id: str
    party_id: str
    outcome: str
    relation: str
    polarity: str
    directness: str
    modality: str
    effect_kind: str = "OTHER"
    condition_ids: tuple[str, ...] = ()
    quantities: tuple[str, ...] = ()
    provenance: tuple[SourceRef, ...] = ()
    likelihood_qualifiers: tuple[str, ...] = ()
    scope_qualifiers: tuple[str, ...] = ()
    temporal_qualifiers: tuple[str, ...] = ()
    condition_join: str = ""
    overall_likelihood_qualifiers: tuple[str, ...] = ()
    source_proposition: str = ""
    source_effect_ids: tuple[str, ...] = ()
    derivation_operation: str = "UNSPECIFIED"
    derivation_explanation: str = ""
    derivation_assumptions: tuple[str, ...] = ()
    outcome_type_transformation: str = "PRESERVED"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CausalLink:
    source_id: str
    relation: str
    target_id: str
    modality: str
    condition_ids: tuple[str, ...] = ()
    provenance: tuple[SourceRef, ...] = ()
    # Added after the original positional fields for stored/test compatibility.
    # Schema-version 1.1+ models must populate it explicitly.
    action_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CounterfactualLink:
    """A typed comparison between effects belonging to alternative actions.

    This is not a causal edge. ``source_effect_id`` is the foregone/precluded
    effect of the chosen action; ``alternative_effect_id`` is the effect that
    occurs under the mutually exclusive alternative.
    """

    action_id: str
    source_effect_id: str
    relation: str
    alternative_action_id: str
    alternative_effect_id: str
    modality: str
    condition_ids: tuple[str, ...] = ()
    provenance: tuple[SourceRef, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_redundant_link_gates(
    links: Sequence[CausalLink],
    effects: Sequence[WorldEffect],
) -> tuple[CausalLink, ...]:
    """Remove only gates already owned by a conditional target effect.

    A link's modality describes the causal relationship.  Whether its target
    is activated belongs to the target effect.  Models sometimes duplicate the
    target's gate on an otherwise CERTAIN incoming link; that representation is
    both redundant and rejected by the schema.  Removing the duplicate does
    not weaken or invent a proposition because the same condition remains on
    the target.  Conditions unique to a link are left untouched for validation
    or semantic repair.
    """
    effect_by_id = {effect.effect_id: effect for effect in effects}
    normalized: list[CausalLink] = []
    for link in links:
        target = effect_by_id.get(link.target_id)
        duplicated = (
            link.modality == "CERTAIN"
            and bool(link.condition_ids)
            and target is not None
            and target.modality != "CERTAIN"
            and set(link.condition_ids).issubset(set(target.condition_ids))
        )
        normalized.append(
            replace(link, condition_ids=()) if duplicated else link
        )
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class WorldAction:
    action_id: str
    intervention: str
    actor_party_id: str = ""
    recipient_party_ids: tuple[str, ...] = ()
    effect_ids: tuple[str, ...] = ()
    provenance: tuple[SourceRef, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuarantinedEffect:
    effect_id: str
    action_id: str
    party_id: str
    contradiction_type: str
    conflicting_effect_ids: tuple[str, ...]
    source_clause_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorldStateAdmission:
    status: str = "COMMITTED"
    admitted_effect_ids: tuple[str, ...] = ()
    quarantined_effects: tuple[QuarantinedEffect, ...] = ()
    user_override: bool = False
    override_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScenarioWorldModel:
    parties: tuple[WorldParty, ...]
    actions: tuple[WorldAction, ...]
    effects: tuple[WorldEffect, ...]
    conditions: tuple[WorldCondition, ...] = ()
    causal_links: tuple[CausalLink, ...] = ()
    counterfactual_links: tuple[CounterfactualLink, ...] = ()
    admission: WorldStateAdmission = field(default_factory=WorldStateAdmission)
    schema_version: str = "1.0"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effects_for(self, action_id: str) -> tuple[WorldEffect, ...]:
        admitted = set(self.admission.admitted_effect_ids)
        filter_admission = bool(admitted) or self.admission.status in {
            "USER_ACCEPTED_WITH_QUARANTINE",
            "ABANDONED_CONTRADICTORY_WORLD_STATE",
        }
        return tuple(
            effect for effect in self.effects
            if effect.action_id == action_id
            and (not filter_admission or effect.effect_id in admitted)
        )


_ROLE_WELFARE_KINDS = {"HEALTH_OUTCOME", "WELFARE_OUTCOME"}
_ROLE_LIBERTY_KINDS = {"INSTITUTIONAL_OUTCOME"}
_ROLE_PHYSICAL_WELFARE_KINDS = {"PHYSICAL_STATE"}
# Named patients of framing/killing. Compact welfare roles use the broader
# bearer set below; do not add GROUP/POPULATION or completeness will demand a
# juridical effect on every mentioned crowd.
_ROLE_PERSON_KINDS = {
    "HUMAN", "HUMAN_GROUP", "PERSON", "PATIENT", "PATIENT_GROUP",
}
_ROLE_WELFARE_BEARING_KINDS = {
    "HUMAN", "HUMAN_GROUP", "PERSON", "PATIENT", "PATIENT_GROUP",
    "GROUP", "HOUSEHOLD", "COMMUNITY", "POPULATION", "POPULATION_GROUP",
}
_FALSE_ATTRIBUTION = re.compile(
    r"\b(?:falsely\s+accus\w*|false(?:ly)?\s+(?:charg\w*|convict\w*)|"
    r"convict(?:s|ed|ing)?\s+(?:an\s+|the\s+)?innocent)\b",
    re.IGNORECASE,
)
_FRAME_VERB = re.compile(r"\bfram(?:e|es|ed|ing)\b", re.IGNORECASE)
_EXECUTE_VERB = re.compile(r"\bexecut(?:e|ed|es|ing|ion)\b", re.IGNORECASE)
_REFUSED_HARM = re.compile(
    r"\brefus\w+\b[\s\S]{0,120}\b(?:fram(?:e|ed|ing)|execut|kill)|"
    r"\b(?:not|never|without)\s+(?:framing|executing|killing)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ProjectedActionRoles:
    beneficiaries: tuple[str, ...] = ()
    harmed: tuple[str, ...] = ()
    at_risk: tuple[str, ...] = ()
    conditionally_benefited: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()


def _party_bears_welfare(party: WorldParty | None) -> bool:
    return party is not None and party.kind in _ROLE_WELFARE_BEARING_KINDS


def counts_as_actual_welfare(
    *,
    polarity: str,
    directness: str,
    effect_kind: str,
    party_kind: str = "",
) -> bool:
    """True when a row is an actual harm/benefit, not a counterfactual dual."""
    if directness == "FOREGONE" or polarity in {"FOREGONE", "NEUTRAL"}:
        return False
    if effect_kind in _ROLE_WELFARE_KINDS:
        return True
    if party_kind not in _ROLE_WELFARE_BEARING_KINDS:
        return False
    if effect_kind in _ROLE_LIBERTY_KINDS:
        return True
    if effect_kind in _ROLE_PHYSICAL_WELFARE_KINDS:
        return True
    if effect_kind == "INTERVENTION" and polarity in {"BENEFICIAL", "ADVERSE"}:
        return True
    return False


def _party_has_actual_welfare_kind(
    model: ScenarioWorldModel, party_id: str,
) -> bool:
    """True when any action already records a real welfare/health row on party."""
    return any(
        item.party_id == party_id
        and item.effect_kind in _ROLE_WELFARE_KINDS
        and item.directness != "FOREGONE"
        and item.polarity not in {"FOREGONE", "NEUTRAL"}
        for item in model.effects
    )


def _effect_counts_for_roles(
    effect: WorldEffect,
    party: WorldParty | None,
    model: ScenarioWorldModel | None = None,
) -> bool:
    """True when an admitted effect should populate compact harm/benefit roles.

    Foregone counterfactuals stay out so they are not double-counted against
    the actual outcome. Neutral actor-performances are not welfare claims.
    Crowd use/remain process states are not welfare claims. Directness is not
    required: stipulated downstream deaths and survivals are the point of
    these roles. Facilities and institutions are intermediate process-bearers,
    not compact harmed/beneficiary parties, unless the same party already
    bears an actual welfare or health row somewhere in the model.
    """
    if _crowd_mediated_process(effect, party):
        return False
    if counts_as_actual_welfare(
        polarity=effect.polarity,
        directness=effect.directness,
        effect_kind=effect.effect_kind,
        party_kind=party.kind if party is not None else "",
    ):
        return True
    if (
        model is None
        or party is None
        or effect.directness == "FOREGONE"
        or effect.polarity not in {"BENEFICIAL", "ADVERSE"}
        or effect.effect_kind not in _ROLE_PHYSICAL_WELFARE_KINDS
    ):
        return False
    return _party_has_actual_welfare_kind(model, party.party_id)


def utilitarian_omits_foregone_dual(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> bool:
    """True when FOREGONE restates a swap already encoded by an actual outcome.

    Opportunity cost is the right utilitarian row when choosing this action
    merely forgoes another action's benefit on party P. It is a double count
    when this action also has its own admitted harm or benefit on P: that
    actual row already is the welfare consequence of the choice. Overlays on
    intermediate process-bearers (no compact welfare party) stay on the graph
    for completeness and are not a second welfare term.
    """
    if effect.directness != "FOREGONE" and effect.polarity != "FOREGONE":
        return False
    party = next(
        (item for item in model.parties if item.party_id == effect.party_id),
        None,
    )
    if any(
        other.effect_id != effect.effect_id
        and other.party_id == effect.party_id
        and _effect_counts_for_roles(other, party, model)
        for other in model.effects_for(effect.action_id)
    ):
        return True
    return not (
        _party_bears_welfare(party) or (
            party is not None
            and _party_has_actual_welfare_kind(model, party.party_id)
        )
    )


def modality_is_settled(
    modality: str,
    likelihood_qualifiers: Sequence[str] = (),
) -> bool:
    """True for CERTAIN, or near-certain probabilistic hedges.

    Semantic status is read before confidence: a STIPULATED_CONDITIONAL row
    stays unsettled even when the source marks the gated outcome near-certain.
    High-confidence includes almost-no-chance / no-chance, the same class as
    near-certain. Weaker likelihoods (a chance, likely, possible, unknown)
    are not settled. ENABLES-gated POSSIBLE rows stay unsettled.
    """
    normalized = str(modality or "").strip().upper()
    if normalized == "CERTAIN":
        return True
    if normalized == "STIPULATED_CONDITIONAL":
        return False
    if normalized != "PROBABILISTIC":
        return False
    blob = " ".join(str(item) for item in likelihood_qualifiers)
    return bool(_HIGH_CONFIDENCE_LIKELIHOOD.search(blob))


_CHANCE_MODALITIES = frozenset({
    "PROBABILISTIC", "POSSIBLE", "UNKNOWN", "STIPULATED_CONDITIONAL",
})


def _chance_modality_is_settled(
    modality: str,
    likelihood_qualifiers: Sequence[str] = (),
) -> bool:
    normalized = str(modality or "").strip().upper()
    if normalized not in _CHANCE_MODALITIES:
        return True
    return modality_is_settled(normalized, likelihood_qualifiers)


def counts_as_obtained_outcome(
    *,
    polarity: str,
    modality: str,
    likelihood_qualifiers: Sequence[str] = (),
) -> bool:
    """True when BENEFICIAL or ADVERSE welfare is obtained, not a chance of it.

    A POSSIBLE, PROBABILISTIC, UNKNOWN, or STIPULATED_CONDITIONAL row is not
    IMPROVES or WORSENS with lower confidence. CERTAIN rows remain obtained.
    Near-certain PROBABILISTIC hedges still count via modality_is_settled.
    A near-certain conditional remains conditional.
    """
    if str(polarity or "").upper() not in {"BENEFICIAL", "ADVERSE"}:
        return False
    return _chance_modality_is_settled(modality, likelihood_qualifiers)


def counts_as_settled_adverse(
    *,
    polarity: str,
    modality: str,
    likelihood_qualifiers: Sequence[str] = (),
) -> bool:
    """True when ADVERSE structure is obtained cost, not an unsettled chance.

    Extra PROBABILISTIC harms are not more certain cost than a smaller
    CERTAIN harm set.
    """
    if str(polarity or "").upper() != "ADVERSE":
        return False
    return counts_as_obtained_outcome(
        polarity=polarity,
        modality=modality,
        likelihood_qualifiers=likelihood_qualifiers,
    )


_AVERTED_RISK_PROCESS_PARTY_KINDS = frozenset({"PROCESS"})
_AVERTED_RISK_PROCESS_EFFECT_KINDS = frozenset({
    "PHYSICAL_STATE", "OTHER", "INSTITUTIONAL_OUTCOME",
})
_AVERTED_RISK_CAUSAL_RELATIONS = frozenset({
    "CAUSES", "ENABLES", "PREVENTS", "INCREASES", "DECREASES", "DISABLES",
})


def is_averted_risk_not_obtained_benefit(
    *,
    polarity: str,
    directness: str,
    modality: str,
    likelihood_qualifiers: Sequence[str] = (),
    effect_kind: str,
    party_kind: str,
    ancestor_direct_same_party: bool,
    opposed_unsettled_adverse: bool,
) -> bool:
    """True when a CERTAIN BENEFICIAL row only encodes an averted unsettled harm.

    Distinct from CHANCE_IS_NOT_AN_OUTCOME: this row is already typed CERTAIN.
    Distinct from FOREGONE_IS_NOT_OBTAINED: polarity is actual BENEFICIAL.
    DIRECT acts on the same party remain obtained. Chance may still occupy
    at_risk, conditionally_benefited, expected-value, or reversal boundaries.
    """
    if str(polarity or "").upper() != "BENEFICIAL":
        return False
    if str(directness or "").upper() in {"FOREGONE", "DIRECT"}:
        return False
    if not counts_as_obtained_outcome(
        polarity=polarity,
        modality=modality,
        likelihood_qualifiers=likelihood_qualifiers,
    ):
        return False
    kind = str(effect_kind or "").upper()
    bearer = str(party_kind or "").upper()
    welfare = (
        kind in _ROLE_WELFARE_KINDS
        and bearer in _ROLE_WELFARE_BEARING_KINDS
    )
    process_row = (
        bearer in _AVERTED_RISK_PROCESS_PARTY_KINDS
        and kind in _AVERTED_RISK_PROCESS_EFFECT_KINDS
    )
    if not welfare and not process_row:
        return False
    if ancestor_direct_same_party:
        return False
    return opposed_unsettled_adverse


def _has_same_party_direct_ancestor(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> bool:
    by_id = {item.effect_id: item for item in model.effects}
    seen: set[str] = set()
    stack = [effect.effect_id]
    while stack:
        current_id = stack.pop()
        if current_id in seen:
            continue
        seen.add(current_id)
        current = by_id.get(current_id)
        if (
            current is not None
            and current.effect_id != effect.effect_id
            and current.directness == "DIRECT"
            and current.party_id == effect.party_id
        ):
            return True
        for link in model.causal_links:
            if link.action_id != effect.action_id or link.target_id != current_id:
                continue
            stack.append(link.source_id)
    return False


def _has_opposed_unsettled_adverse(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> bool:
    for other in model.effects:
        if other.action_id == effect.action_id:
            continue
        if other.party_id != effect.party_id:
            continue
        if str(other.directness or "").upper() == "FOREGONE":
            continue
        if str(other.polarity or "").upper() != "ADVERSE":
            continue
        if counts_as_obtained_outcome(
            polarity=other.polarity,
            modality=other.modality,
            likelihood_qualifiers=other.likelihood_qualifiers,
        ):
            continue
        return True
    return False


def _same_action_ancestor_party_ids(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> set[str]:
    """Party IDs on the same-action causal ancestry of this effect."""
    by_id = {item.effect_id: item for item in model.effects}
    seen: set[str] = set()
    parties: set[str] = set()
    stack = [effect.effect_id]
    while stack:
        current_id = stack.pop()
        if current_id in seen:
            continue
        seen.add(current_id)
        current = by_id.get(current_id)
        if current is None:
            continue
        if current.effect_id != effect.effect_id and current.party_id:
            parties.add(current.party_id)
        for link in model.causal_links:
            if link.action_id != effect.action_id or link.target_id != current_id:
                continue
            if not _link_parents_target(link):
                continue
            stack.append(link.source_id)
    return parties


def _projected_averted_conditional_roles(
    model: ScenarioWorldModel,
    action_id: str,
    *,
    assigned: dict[str, tuple[str, ...]],
    already: set[str],
) -> tuple[tuple[str, str, str], ...]:
    """Crowd labels averted by CERTAIN intermediate protection on this action.

    If party P has unsettled ADVERSE under another action, and this action
    certainly benefits an intermediate bearer that is a same-action ancestor
    of that harm, P is conditionally_benefited here. Do not invent a welfare
    row or treat the averted crowd as an obtained beneficiary.
    """
    party_by_id = {party.party_id: party for party in model.parties}
    protected: set[str] = set()
    protectors: dict[str, str] = {}
    for effect in model.effects_for(action_id):
        party = party_by_id.get(effect.party_id)
        if party is None or party.kind not in _INTERMEDIATE_BEARER_KINDS:
            continue
        if str(effect.directness or "").upper() == "FOREGONE":
            continue
        if str(effect.polarity or "").upper() != "BENEFICIAL":
            continue
        if not counts_as_obtained_outcome(
            polarity=effect.polarity,
            modality=effect.modality,
            likelihood_qualifiers=effect.likelihood_qualifiers,
        ):
            continue
        protected.add(party.party_id)
        protectors.setdefault(party.party_id, effect.effect_id)
    if not protected:
        return ()
    rows: list[tuple[str, str, str]] = []
    seen_labels: set[str] = set()
    for other in model.effects:
        if other.action_id == action_id:
            continue
        if str(other.directness or "").upper() == "FOREGONE":
            continue
        if str(other.polarity or "").upper() != "ADVERSE":
            continue
        victim = party_by_id.get(other.party_id)
        if victim is None or not _party_bears_welfare(victim):
            continue
        if not _effect_counts_for_roles(other, victim, model):
            continue
        if counts_as_obtained_outcome(
            polarity=other.polarity,
            modality=other.modality,
            likelihood_qualifiers=other.likelihood_qualifiers,
        ):
            continue
        ancestors = _same_action_ancestor_party_ids(other, model)
        overlap = ancestors & protected
        if not overlap:
            continue
        label = _compact_role_label(
            other, victim,
            assignment_quantities=assigned.get(victim.party_id, ()),
        )
        if not label or label in already or label in seen_labels:
            continue
        seen_labels.add(label)
        protector = protectors[next(iter(overlap))]
        rows.append((
            label,
            other.effect_id,
            (
                f"CERTAIN intermediate protection {protector} averts "
                f"opposed unsettled {other.effect_id}; compact "
                f"conditionally_benefited, not obtained benefit"
            ),
        ))
    return tuple(rows)


def is_averted_risk_not_obtained_benefit_effect(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> bool:
    party = next(
        (item for item in model.parties if item.party_id == effect.party_id),
        None,
    )
    return is_averted_risk_not_obtained_benefit(
        polarity=effect.polarity,
        directness=effect.directness,
        modality=effect.modality,
        likelihood_qualifiers=effect.likelihood_qualifiers,
        effect_kind=effect.effect_kind,
        party_kind=party.kind if party is not None else "",
        ancestor_direct_same_party=_has_same_party_direct_ancestor(effect, model),
        opposed_unsettled_adverse=_has_opposed_unsettled_adverse(effect, model),
    )


def _consequence_action_id(graph: Any, consequence_id: str) -> str:
    for edge in getattr(graph, "edges", ()):
        if edge.relation != "HAS_CONSEQUENCE" or edge.target != consequence_id:
            continue
        owner = graph.nodes.get(edge.source)
        if owner is None:
            continue
        return str(owner.attributes.get("canonical_action_id", owner.id))
    return ""


def _graph_has_same_party_direct_ancestor(
    graph: Any, consequence: Any, party_id: str,
) -> bool:
    seen: set[str] = set()
    stack = [consequence.id]
    while stack:
        current_id = stack.pop()
        if current_id in seen:
            continue
        seen.add(current_id)
        current = graph.nodes.get(current_id)
        if (
            current is not None
            and current.id != consequence.id
            and current.kind == "CONSEQUENCE"
            and str(current.attributes.get("directness", "")).upper() == "DIRECT"
            and str(current.attributes.get("party_id", "")) == party_id
        ):
            return True
        for edge in graph.edges:
            if (
                edge.target == current_id
                and edge.relation in _AVERTED_RISK_CAUSAL_RELATIONS
            ):
                stack.append(edge.source)
    return False


def _graph_has_opposed_unsettled_adverse(
    graph: Any, action_id: str, party_id: str,
) -> bool:
    for edge in graph.edges:
        if edge.relation != "HAS_CONSEQUENCE":
            continue
        owner = graph.nodes.get(edge.source)
        other = graph.nodes.get(edge.target)
        if owner is None or other is None or other.kind != "CONSEQUENCE":
            continue
        owner_id = str(owner.attributes.get("canonical_action_id", owner.id))
        if owner_id == action_id:
            continue
        if str(other.attributes.get("party_id", "")) != party_id:
            continue
        if str(other.attributes.get("directness", "")).upper() == "FOREGONE":
            continue
        if str(other.attributes.get("polarity", "")).upper() != "ADVERSE":
            continue
        if counts_as_obtained_outcome(
            polarity=str(other.attributes.get("polarity", "")),
            modality=str(other.attributes.get("modality", "")),
            likelihood_qualifiers=other.attributes.get("likelihood_qualifiers", ()),
        ):
            continue
        return True
    return False


def is_averted_risk_not_obtained_benefit_consequence(graph: Any, consequence: Any) -> bool:
    attrs = getattr(consequence, "attributes", {}) or {}
    party_id = str(attrs.get("party_id", "")).strip()
    action_id = _consequence_action_id(graph, consequence.id)
    if not party_id or not action_id:
        return False
    return is_averted_risk_not_obtained_benefit(
        polarity=str(attrs.get("polarity", "")),
        directness=str(attrs.get("directness", "")),
        modality=str(attrs.get("modality", "")),
        likelihood_qualifiers=attrs.get("likelihood_qualifiers", ()),
        effect_kind=str(attrs.get("effect_kind", "")),
        party_kind=str(attrs.get("party_kind", "")),
        ancestor_direct_same_party=_graph_has_same_party_direct_ancestor(
            graph, consequence, party_id,
        ),
        opposed_unsettled_adverse=_graph_has_opposed_unsettled_adverse(
            graph, action_id, party_id,
        ),
    )


def _occupies_risk_situation(effect: WorldEffect) -> bool:
    """True when the row is occupying exposure, not an obtained welfare harm.

    Parser CERTAIN for 'at moderate risk' stays. Compact still routes it to
    at_risk: the situation is stipulated, the harm is not.
    """
    if _should_type_risk_situation_certain(effect):
        return True
    if not _outcome_is_exposure_state(effect.outcome):
        return False
    return not any(
        _risk_situation_licenses_chance(effect, span)
        for span in effect.likelihood_qualifiers
    )


def _effect_gated_by_referenced_event(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> bool:
    pointed = {
        condition.condition_id
        for condition in model.conditions
        if str(condition.event_effect_id or "").strip()
    }
    return any(condition_id in pointed for condition_id in effect.condition_ids)


def _compact_role_is_settled(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> bool:
    """Obtained welfare only. Status is read before confidence.

    STIPULATED_CONDITIONAL never occupies harmed/beneficiary slots, even with
    a near-certain hedge. Event-referenced gates stay conditional. Occupying
    an at-risk situation is at_risk, not obtained harm. High-confidence
    PROBABILISTIC hedges may still settle. Weaker likelihoods stay unresolved.
    """
    if effect.polarity == "UNRESOLVED":
        return False
    if str(effect.modality or "").strip().upper() == "STIPULATED_CONDITIONAL":
        return False
    if _effect_gated_by_referenced_event(effect, model):
        return False
    if _occupies_risk_situation(effect):
        return False
    return modality_is_settled(effect.modality, effect.likelihood_qualifiers)


@dataclass(frozen=True, slots=True)
class CompactRoleExplanation:
    action_id: str
    bucket: str
    label: str
    effect_id: str
    reason: str


_LEADING_QUANTITY = re.compile(
    r"^(?:(?:over|about|approximately|nearly|almost|at\s+least|at\s+most|"
    r"as\s+many\s+as)\s+)?(?:\d+(?:,\d{3})*(?:\.\d+)?|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"[-\s]+(?:one|two|three|four|five|six|seven|eight|nine)"
    r")\s+",
    re.IGNORECASE,
)


def _compact_role_label(
    effect: WorldEffect,
    party: WorldParty,
    *,
    assignment_quantities: Sequence[str] = (),
) -> str:
    """Prefer an assigned subset quantity over the party's total count.

    'five firefighters' is the compact party when the effect owns five and
    the party total is fifty. If the risk row omits the subset, a DIRECT
    assignment quantity that still appears in this row's provenance is used.
    Ordinary totals that already sit on the party stay off the compact label.
    """
    quantity = next((item for item in effect.quantities if str(item).strip()), "")
    if not quantity:
        blob = " ".join((effect.outcome, _provenance_text(effect))).casefold()
        quantity = next(
            (
                item for item in assignment_quantities
                if str(item).strip() and str(item).casefold() in blob
            ),
            "",
        )
    label = party.label.strip()
    if not quantity:
        return label
    effect_mag = quantity_magnitude(quantity)
    if effect_mag is None:
        return label
    party_mags = [quantity_magnitude(item) for item in party.quantities]
    if not any(other is not None and other > effect_mag for other in party_mags):
        return label
    if quantity.casefold() in label.casefold() and label.casefold().startswith(
        quantity.casefold()
    ):
        return label
    head = _LEADING_QUANTITY.sub("", label).strip() or label
    if head.casefold().startswith(quantity.casefold()):
        return head
    return f"{quantity} {head}".strip()


def _compact_role_reason(
    effect: WorldEffect,
    model: ScenarioWorldModel,
    *,
    settled: bool,
    bucket: str,
) -> str:
    parts: list[str] = [
        f"{effect.modality} {effect.effect_kind} {effect.polarity}",
    ]
    if effect.quantities:
        parts.append("affected quantity " + ", ".join(effect.quantities))
    if _occupies_risk_situation(effect):
        parts.append("occupies an at-risk situation, not obtained harm")
    if _effect_gated_by_referenced_event(effect, model):
        gates = []
        pointed = {
            condition.condition_id: condition for condition in model.conditions
        }
        for cond_id in effect.condition_ids:
            condition = pointed.get(cond_id)
            if condition is None or not condition.event_effect_id:
                continue
            gates.append(f"{cond_id}->{condition.event_effect_id}")
        if gates:
            parts.append("gated by " + ", ".join(gates))
        if effect.condition_join:
            parts.append("join " + effect.condition_join)
        parts.append("conditional, not obtained")
    if str(effect.modality or "").upper() == "STIPULATED_CONDITIONAL":
        parts.append("STIPULATED_CONDITIONAL never settles")
    if settled:
        parts.append(f"compact {bucket}")
    else:
        parts.append(f"compact {bucket} because the outcome is not obtained")
    return "; ".join(parts)


def project_world_action_roles(
    model: ScenarioWorldModel, action_id: str,
) -> ProjectedActionRoles:
    """Derive compact roles from admitted health, welfare, and liberty effects."""
    party_by_id = {party.party_id: party for party in model.parties}
    assigned = assignment_owned_quantities(model.effects)
    beneficiaries: list[str] = []
    harmed: list[str] = []
    at_risk: list[str] = []
    conditionally_benefited: list[str] = []
    for effect in model.effects_for(action_id):
        party = party_by_id.get(effect.party_id)
        if party is None or not _effect_counts_for_roles(effect, party, model):
            continue
        label = _compact_role_label(
            effect, party, assignment_quantities=assigned.get(party.party_id, ()),
        )
        settled = _compact_role_is_settled(effect, model)
        if settled and is_averted_risk_not_obtained_benefit_effect(effect, model):
            settled = False
        if not settled:
            if effect.polarity == "ADVERSE" and label not in at_risk:
                at_risk.append(label)
            elif (
                effect.polarity == "BENEFICIAL"
                and label not in conditionally_benefited
            ):
                conditionally_benefited.append(label)
            continue
        if effect.polarity == "BENEFICIAL":
            if label not in beneficiaries and label not in harmed:
                beneficiaries.append(label)
        elif effect.polarity == "ADVERSE":
            if label not in harmed:
                harmed.append(label)
                if label in beneficiaries:
                    beneficiaries.remove(label)
    settled = set(beneficiaries) | set(harmed)
    at_risk = [label for label in at_risk if label not in settled]
    conditionally_benefited = [
        label for label in conditionally_benefited if label not in settled
    ]
    already = set(beneficiaries) | set(harmed) | set(conditionally_benefited)
    for label, _effect_id, _reason in _projected_averted_conditional_roles(
        model, action_id, assigned=assigned, already=already,
    ):
        if label not in conditionally_benefited and label not in settled:
            conditionally_benefited.append(label)
    unresolved = tuple(dict.fromkeys((*at_risk, *conditionally_benefited)))
    return ProjectedActionRoles(
        beneficiaries=tuple(beneficiaries),
        harmed=tuple(harmed),
        at_risk=tuple(at_risk),
        conditionally_benefited=tuple(conditionally_benefited),
        unresolved=unresolved,
    )


def explain_compact_role_assignments(
    model: ScenarioWorldModel, action_id: str,
) -> tuple[CompactRoleExplanation, ...]:
    """Why each compact role label was assigned, including subgroup quantities."""
    party_by_id = {party.party_id: party for party in model.parties}
    assigned = assignment_owned_quantities(model.effects)
    rows: list[CompactRoleExplanation] = []
    seen: set[tuple[str, str, str]] = set()
    for effect in model.effects_for(action_id):
        party = party_by_id.get(effect.party_id)
        if party is None or not _effect_counts_for_roles(effect, party, model):
            continue
        label = _compact_role_label(
            effect, party, assignment_quantities=assigned.get(party.party_id, ()),
        )
        settled = _compact_role_is_settled(effect, model)
        if settled and is_averted_risk_not_obtained_benefit_effect(effect, model):
            settled = False
        if not settled:
            bucket = (
                "at_risk" if effect.polarity == "ADVERSE"
                else "conditionally_benefited" if effect.polarity == "BENEFICIAL"
                else ""
            )
        elif effect.polarity == "BENEFICIAL":
            bucket = "beneficiaries"
        elif effect.polarity == "ADVERSE":
            bucket = "harmed"
        else:
            bucket = ""
        if not bucket:
            continue
        key = (bucket, label, effect.effect_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(CompactRoleExplanation(
            action_id=action_id,
            bucket=bucket,
            label=label,
            effect_id=effect.effect_id,
            reason=_compact_role_reason(
                effect, model, settled=settled, bucket=bucket,
            ),
        ))
    already = {row.label for row in rows}
    for label, effect_id, reason in _projected_averted_conditional_roles(
        model, action_id, assigned=assigned, already=already,
    ):
        key = ("conditionally_benefited", label, effect_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(CompactRoleExplanation(
            action_id=action_id,
            bucket="conditionally_benefited",
            label=label,
            effect_id=effect_id,
            reason=reason,
        ))
    return tuple(rows)


def _refs(clause_ids: Iterable[Any], lookup: dict[str, str]) -> tuple[SourceRef, ...]:
    return tuple(
        SourceRef(clause_id=clause_id, excerpt=lookup.get(clause_id, ""))
        for clause_id in dict.fromkeys(_clean(value, 32) for value in clause_ids)
        if clause_id in lookup
    )


def _first_clean(row: dict[str, Any], *keys: str, limit: int = 64) -> str:
    for key in keys:
        value = _clean(row.get(key), limit)
        if value:
            return value
    return ""


def _effect_outcome_and_predicate(row: dict[str, Any]) -> tuple[str, str]:
    """Read the atomic event text. Empty predicate is a typing hole, not a fact."""
    outcome = _first_clean(
        row, "outcome", "event", "description", limit=240,
    )
    predicate = _first_clean(
        row, "predicate", "relation", "effect_relation", limit=64,
    ).upper()
    if outcome and not predicate:
        predicate = "EXPERIENCES"
    return outcome, predicate


def _likelihood_fill_values(effect: WorldEffect) -> tuple[str, ...]:
    """Keep recorded hedges that match a bound span; fill from the literal."""
    provenance_text = _provenance_text(effect)
    bound: list[QualifierSpan] = []
    for ref in effect.provenance:
        text = ref.excerpt
        for record in explicit_likelihood_span_records(text):
            if _qualifier_binds_to_effect(
                effect, text, record.start, record.end,
                qualifier=record.canonical, extractor=explicit_likelihood_spans,
            ):
                bound.append(record)
    expected_keys = {record.canonical.casefold() for record in bound}
    recorded = [
        value for value in effect.likelihood_qualifiers
        if _qualifier_grounded_in_text(value, provenance_text)
        and _likelihood_identity_key(value) in expected_keys
    ]
    kept: list[str] = []
    have: set[str] = set()
    for value in recorded:
        key = _likelihood_identity_key(value)
        if key not in have:
            kept.append(value)
            have.add(key)
    for record in bound:
        key = _likelihood_identity_key(record.canonical)
        if key not in have:
            kept.append(record.literal)
            have.add(key)
    return _unique_likelihood_spans(kept)


def _closed_class_qualifiers(effect: WorldEffect) -> WorldEffect:
    """Copy source-bound qualifiers and drop invented or sibling-bound ones.

    Qualifiers are closed-class spans. The parser can fill and strip them
    without inventing outcomes, parties, or causal structure. A span that
    appears in provenance but binds to a sibling head is not kept: graph,
    ledger, and compact then see the bound set without a second binder.
    Likelihood fill keeps a recorded form that matches the bound identity
    and otherwise copies the literal source span, not only the canonical.
    """
    provenance_text = _provenance_text(effect)
    updates: dict[str, tuple[str, ...]] = {}
    for _kind, field, extractor in _qualifier_channels():
        if field == "likelihood_qualifiers":
            updates[field] = _likelihood_fill_values(effect)
            continue
        recorded = [
            value for value in getattr(effect, field)
            if _qualifier_grounded_in_text(value, provenance_text)
        ]
        expected = list(effect_expected_qualifiers(effect, extractor))
        expected_keys = {item.casefold() for item in expected}
        kept = [value for value in recorded if value.casefold() in expected_keys]
        have = {item.casefold() for item in kept}
        for value in expected:
            if value.casefold() not in have:
                kept.append(value)
                have.add(value.casefold())
        updates[field] = tuple(dict.fromkeys(kept))
    filled = replace(effect, **updates)
    filled = _partition_overall_and_conditional_likelihood(filled)
    if _should_type_risk_situation_certain(filled):
        return replace(filled, modality="CERTAIN", condition_ids=())
    if filled.modality == "CERTAIN" and _chance_hedge_blocks_certain(filled):
        blob = " ".join(_chance_hedges_on_effect(filled))
        modality = (
            "PROBABILISTIC" if _HIGH_CONFIDENCE_LIKELIHOOD.search(blob) else "POSSIBLE"
        )
        return replace(filled, modality=modality, condition_ids=())
    return filled


def _is_overall_characterization_span(span: str) -> bool:
    """True for an unconditional characterization, not a gated severity."""
    folded = " ".join(str(span or "").split()).casefold()
    if folded in {"unlikely", "likely"}:
        return True
    return bool(re.search(r"\bdeaths?\s+unlikely\b", folded))


def _is_conditional_severity_span(span: str) -> bool:
    return bool(_HIGH_CONFIDENCE_LIKELIHOOD.search(str(span or "")))


def _partition_overall_and_conditional_likelihood(
    effect: WorldEffect,
) -> WorldEffect:
    """Keep UNLIKELY overall beside NEAR_CERTAIN-if-gate; do not overwrite."""
    overall = list(effect.overall_likelihood_qualifiers)
    conditional: list[str] = []
    has_overall = False
    has_conditional = False
    for span in effect.likelihood_qualifiers:
        if _is_overall_characterization_span(span):
            overall.append(span)
            has_overall = True
        else:
            conditional.append(span)
            if _is_conditional_severity_span(span):
                has_conditional = True
    if not (has_overall and (has_conditional or effect.condition_ids)):
        return effect
    return replace(
        effect,
        likelihood_qualifiers=_unique_likelihood_spans(conditional),
        overall_likelihood_qualifiers=_unique_likelihood_spans(overall),
    )


_PROTECTIVE_WALLS = re.compile(
    r"(?:(?:[a-z]+[-_ ]?)?resistant|protective|reinforced)[-_ ]walls?",
    re.IGNORECASE,
)
_OVERALL_DEATH_HEDGE = re.compile(
    r"\bdeaths?\s+unlikely\b|\bunlikely\s+to\s+(?:die|drown|perish)\b",
    re.IGNORECASE,
)
_REMAIN_OUTCOME = re.compile(
    r"(?<![a-z])(?:remain|remains|remaining|stay|stays|staying)(?![a-z])",
    re.IGNORECASE,
)
_DEATH_OUTCOME = re.compile(
    r"(?<![a-z])(?:death|die|dies|dying|killed|perish)(?![a-z])",
    re.IGNORECASE,
)


def source_plan_label(action: WorldAction) -> str:
    """Source Plan A/B label when a cited clause names the plan."""
    for ref in action.provenance:
        match = re.search(r"\bPlan\s+([A-Z])\b", ref.excerpt or "", re.IGNORECASE)
        if match:
            return f"Plan {match.group(1).upper()}"
    return ""


def recorded_quantity_payload(spans: Sequence[str]) -> list[dict[str, Any]]:
    """Raw source span plus canonical numeral, when the span has a magnitude."""
    payload: list[dict[str, Any]] = []
    for span in spans:
        raw = str(span).strip()
        if not raw:
            continue
        payload.append({"raw": raw, "canonical": quantity_magnitude(raw)})
    return payload


def _overall_death_hedge_clauses(
    lookup: dict[str, str],
) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for clause_id, text in lookup.items():
        if _ACTION_SOURCE_ID.match(clause_id):
            continue
        if not _OVERALL_DEATH_HEDGE.search(text):
            continue
        spans = [
            span for span in explicit_likelihood_spans(text)
            if _is_overall_characterization_span(span)
        ]
        hedge = spans[0] if spans else "unlikely"
        found.append((clause_id, text, hedge))
    return found


def _protective_wall_clauses(lookup: dict[str, str]) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for clause_id, text in lookup.items():
        if _ACTION_SOURCE_ID.match(clause_id):
            continue
        match = _PROTECTIVE_WALLS.search(text)
        if match is None:
            continue
        found.append((clause_id, text, match.group(0)))
    return found


def _gated_remain_deaths(
    effects: Sequence[WorldEffect],
    conditions: Sequence[WorldCondition],
) -> tuple[WorldEffect, ...]:
    pointed = {
        condition.condition_id
        for condition in conditions
        if str(condition.event_effect_id or "").strip()
    }
    remain_actions = {
        effect.action_id for effect in effects
        if _REMAIN_OUTCOME.search(effect.outcome)
        and effect.directness != "FOREGONE"
    }
    return tuple(
        effect for effect in effects
        if effect.action_id in remain_actions
        and effect.directness == "DOWNSTREAM"
        and effect.polarity == "ADVERSE"
        and effect.effect_kind in {"HEALTH_OUTCOME", "WELFARE_OUTCOME"}
        and _DEATH_OUTCOME.search(effect.outcome)
        and any(cond_id in pointed for cond_id in effect.condition_ids)
    )


def _protective_facility_party(
    parties: Sequence[WorldParty],
    effects: Sequence[WorldEffect],
    remain_action_id: str,
) -> WorldParty | None:
    facilities = [party for party in parties if party.kind == "FACILITY"]
    if not facilities:
        return None
    failing = {
        effect.party_id for effect in effects
        if effect.action_id == remain_action_id
        and str(effect.modality or "").upper() in {"PROBABILISTIC", "POSSIBLE"}
    }
    stable = [party for party in facilities if party.party_id not in failing]
    return (stable or facilities)[0]


def _fill_protective_overall_context(
    effects: Sequence[WorldEffect],
    parties: Sequence[WorldParty],
    conditions: Sequence[WorldCondition],
    lookup: dict[str, str],
) -> tuple[WorldEffect, ...]:
    """CERTAIN protective walls plus overall-unlikely on gated death.

    Does not add a second compact harm or benefit. Does not parent death
    from the walls or copy overall unlikely onto the gated severity.
    """
    deaths = _gated_remain_deaths(effects, conditions)
    hedges = _overall_death_hedge_clauses(lookup)
    walls = _protective_wall_clauses(lookup)
    updated = list(effects)
    by_id = {effect.effect_id: index for index, effect in enumerate(updated)}
    for death in deaths:
        if death.effect_id not in by_id:
            continue
        current = updated[by_id[death.effect_id]]
        added_overall = list(current.overall_likelihood_qualifiers)
        added_refs = list(current.provenance)
        cited = {ref.clause_id for ref in added_refs}
        for clause_id, text, hedge in hedges:
            added_overall.append(hedge)
            if clause_id not in cited:
                added_refs.append(SourceRef(clause_id, text))
                cited.add(clause_id)
        if added_overall != list(current.overall_likelihood_qualifiers):
            updated[by_id[death.effect_id]] = replace(
                current,
                overall_likelihood_qualifiers=_unique_likelihood_spans(added_overall),
                provenance=tuple(added_refs),
            )
    existing_ids = {effect.effect_id for effect in updated}
    cited_wall_clauses = {
        ref.clause_id
        for effect in updated
        if effect.effect_kind == "PHYSICAL_STATE"
        and effect.directness != "FOREGONE"
        and _PROTECTIVE_WALLS.search(effect.outcome)
        for ref in effect.provenance
    }
    for clause_id, text, outcome in walls:
        if clause_id in cited_wall_clauses:
            continue
        if not deaths:
            continue
        host = deaths[0]
        facility = _protective_facility_party(parties, updated, host.action_id)
        if facility is None:
            continue
        effect_id = f"{host.action_id}_WALLS"
        suffix = 0
        while effect_id in existing_ids:
            suffix += 1
            effect_id = f"{host.action_id}_WALLS{suffix}"
        updated.append(WorldEffect(
            effect_id,
            host.action_id,
            facility.party_id,
            outcome,
            "STATE_CHANGE",
            "NEUTRAL",
            "DOWNSTREAM",
            "CERTAIN",
            "PHYSICAL_STATE",
            provenance=(SourceRef(clause_id, text),),
        ))
        existing_ids.add(effect_id)
        cited_wall_clauses.add(clause_id)
    return tuple(updated)


def _overall_conditional_likelihood_errors(model: ScenarioWorldModel) -> list[str]:
    """UNLIKELY overall must not overwrite NEAR_CERTAIN if-failure, or vice versa."""
    errors: list[str] = []
    lookup = {
        ref.clause_id: ref.excerpt
        for effect in model.effects
        for ref in effect.provenance
        if ref.clause_id and ref.excerpt
    }
    for action in model.actions:
        for ref in action.provenance:
            if ref.clause_id and ref.excerpt:
                lookup.setdefault(ref.clause_id, ref.excerpt)
    deaths = _gated_remain_deaths(model.effects, model.conditions)
    hedge_clauses = _overall_death_hedge_clauses(lookup)
    wall_clauses = _protective_wall_clauses(lookup)
    for effect in model.effects:
        likelihood = effect.likelihood_qualifiers
        overall = effect.overall_likelihood_qualifiers
        has_overall = any(_is_overall_characterization_span(span) for span in likelihood)
        has_conditional = any(_is_conditional_severity_span(span) for span in likelihood)
        if has_overall and has_conditional:
            errors.append(
                f"{effect.effect_id} mixes overall characterization {likelihood} "
                "with conditional severity on likelihood_qualifiers; keep UNLIKELY "
                "overall distinct from NEAR_CERTAIN if the gate occurs"
            )
        if (
            any(_is_overall_characterization_span(span) for span in overall)
            and any(_is_overall_characterization_span(span) for span in likelihood)
        ):
            errors.append(
                f"{effect.effect_id} copied overall likelihood {likelihood} onto "
                "the gated severity; keep overall_likelihood_qualifiers distinct"
            )
    for death in deaths:
        if hedge_clauses and not any(
            _is_overall_characterization_span(span)
            for span in death.overall_likelihood_qualifiers
        ):
            errors.append(
                f"{death.effect_id} omits the overall death characterization "
                f"{hedge_clauses[0][2]!r}; do not overwrite the gated "
                f"{list(death.likelihood_qualifiers)} severity"
            )
        for other in model.effects:
            if other.effect_id == death.effect_id:
                continue
            if other.action_id != death.action_id or other.party_id != death.party_id:
                continue
            if other.directness == "FOREGONE":
                continue
            if other.polarity != "BENEFICIAL":
                continue
            if other.effect_kind not in {"HEALTH_OUTCOME", "WELFARE_OUTCOME"}:
                continue
            if not (
                _DEATH_OUTCOME.search(other.outcome)
                or any(_is_overall_characterization_span(span) for span in other.likelihood_qualifiers)
            ):
                continue
            errors.append(
                f"{other.effect_id} double-counts the overall death characterization "
                f"as a separate compact outcome from gated {death.effect_id}; keep "
                "UNLIKELY overall on the gated death, not a second harm or benefit"
            )
    if deaths and wall_clauses:
        cited = {
            ref.clause_id
            for effect in model.effects
            if effect.effect_kind == "PHYSICAL_STATE"
            and effect.polarity == "NEUTRAL"
            and effect.directness != "FOREGONE"
            and str(effect.modality or "").upper() == "CERTAIN"
            and _PROTECTIVE_WALLS.search(effect.outcome)
            for ref in effect.provenance
        }
        missing = [clause_id for clause_id, _text, _out in wall_clauses if clause_id not in cited]
        if missing:
            errors.append(
                f"{deaths[0].action_id} omits CERTAIN protective facility context "
                f"for {missing}; record resistant/protective walls as NEUTRAL "
                "PHYSICAL_STATE, not a second death outcome"
            )
    return errors


def _normalized_directness_and_kind(row: dict[str, Any]) -> tuple[str, str]:
    """FOREGONE is a typing of opportunity loss, not a second fact to invent."""
    directness = _clean(row.get("directness"), 32).upper()
    kind = _clean(row.get("effect_kind"), 48).upper() or "OTHER"
    if directness == "FOREGONE":
        kind = "OPPORTUNITY_LOSS"
    return directness, kind


def _foreclosure_source_effect_id(
    *,
    source_id: str,
    source_action: str,
    target_id: str,
    effects: Sequence[WorldEffect],
    effect_by_id: dict[str, WorldEffect],
) -> str:
    """Resolve a cross-action foreclosure edge onto a FOREGONE source effect."""
    source_effect = effect_by_id.get(source_id)
    if source_effect is not None and source_effect.directness == "FOREGONE":
        return source_id
    alternative_effect = effect_by_id.get(target_id)
    candidates = [
        effect for effect in effects
        if effect.action_id == source_action
        and effect.directness == "FOREGONE"
        and alternative_effect is not None
        and effect.party_id == alternative_effect.party_id
    ]
    return candidates[0].effect_id if len(candidates) == 1 else ""


def parse_world_model(
    raw: Any,
    *,
    clauses: Sequence[dict[str, str]],
    action_ids: Sequence[str],
    action_texts: dict[str, str] | None = None,
    require_completeness: bool = True,
) -> ScenarioWorldModel:
    """Parse grounding claims, deriving redundant indexes without altering facts."""
    if not isinstance(raw, dict):
        raise ValueError("world_model must be an object")
    schema_version = _clean(raw.get("schema_version"), 16) or "1.0"
    lookup = {
        _clean(row.get("clause_id"), 32): _clean(row.get("text"), 1000)
        for row in clauses if isinstance(row, dict)
    }
    for action_id, text in (action_texts or {}).items():
        key = _clean(action_id, 16).upper()
        if key:
            lookup[key] = _clean(text, 1000)
    parties = tuple(WorldParty(
        party_id=_clean(row.get("party_id"), 64).upper(),
        label=_clean(row.get("label"), 160),
        kind=_clean(row.get("kind"), 48).upper() or "OTHER",
        quantities=tuple(dict.fromkeys(
            _clean(value, 80) for value in row.get("quantities", [])
        )),
        provenance=_refs(row.get("clause_ids", []), lookup),
    ) for row in raw.get("parties", []) if isinstance(row, dict))
    if schema_version != "1.0":
        parties = closed_class_party_quantities(parties)
    actions = tuple(WorldAction(
        action_id=_clean(row.get("action_id"), 16).upper(),
        intervention=_clean(row.get("intervention"), 240),
        actor_party_id=_clean(row.get("actor_party_id"), 64).upper(),
        recipient_party_ids=tuple(
            dict.fromkeys(_clean(value, 64).upper() for value in row.get("recipient_party_ids", []))
        ),
        effect_ids=tuple(
            dict.fromkeys(_clean(value, 80) for value in row.get("effect_ids", []))
        ),
        provenance=_refs(row.get("clause_ids", []), lookup),
    ) for row in raw.get("actions", []) if isinstance(row, dict))
    conditions = tuple(WorldCondition(
        condition_id=_clean(row.get("condition_id"), 80).upper(),
        description=_clean(row.get("description"), 240),
        value_status=_clean(row.get("value_status"), 48).upper() or "UNKNOWN",
        decision_relevance=_clean(row.get("decision_relevance"), 48).upper() or "MATERIAL",
        provenance=_refs(row.get("clause_ids", []), lookup),
        event_effect_id=_clean(row.get("event_effect_id"), 80),
    ) for row in raw.get("conditions", []) if isinstance(row, dict))
    parsed_effects: list[WorldEffect] = []
    for row in raw.get("effects", []):
        if not isinstance(row, dict):
            continue
        directness, effect_kind = _normalized_directness_and_kind(row)
        outcome, predicate = _effect_outcome_and_predicate(row)
        effect = WorldEffect(
            effect_id=_clean(row.get("effect_id"), 80),
            action_id=_clean(row.get("action_id"), 16).upper(),
            party_id=_clean(row.get("party_id"), 64).upper(),
            outcome=outcome,
            relation=predicate,
            polarity=_clean(row.get("polarity"), 32).upper(),
            directness=directness,
            modality=_clean(row.get("modality"), 48).upper(),
            effect_kind=effect_kind,
            condition_ids=tuple(
                dict.fromkeys(_clean(value, 80).upper() for value in row.get("condition_ids", []))
            ),
            quantities=tuple(
                dict.fromkeys(_clean(value, 80) for value in row.get("quantities", []))
            ),
            likelihood_qualifiers=_unique_likelihood_spans(
                row.get("likelihood_qualifiers", [])
            ),
            scope_qualifiers=tuple(dict.fromkeys(
                _clean(value, 80) for value in row.get("scope_qualifiers", [])
            )),
            temporal_qualifiers=tuple(dict.fromkeys(
                _clean(value, 80) for value in row.get("temporal_qualifiers", [])
            )),
            condition_join=_clean(row.get("condition_join"), 8).upper(),
            overall_likelihood_qualifiers=_unique_likelihood_spans(
                row.get("overall_likelihood_qualifiers", [])
            ),
            source_proposition=_clean(row.get("source_proposition"), 500),
            source_effect_ids=tuple(dict.fromkeys(
                _clean(value, 80)
                for value in row.get("source_effect_ids", [])
                if _clean(value, 80)
            )),
            derivation_operation=(
                _clean(row.get("derivation_operation"), 48).upper()
                or "UNSPECIFIED"
            ),
            derivation_explanation=_clean(
                row.get("derivation_explanation"), 500,
            ),
            derivation_assumptions=tuple(dict.fromkeys(
                _clean(value, 240)
                for value in row.get("derivation_assumptions", [])
                if _clean(value, 240)
            )),
            outcome_type_transformation=(
                _clean(row.get("outcome_type_transformation"), 48).upper()
                or "PRESERVED"
            ),
            provenance=_refs(row.get("clause_ids", []), lookup),
        )
        if schema_version != "1.0":
            effect = _closed_class_qualifiers(effect)
        parsed_effects.append(effect)
    effects = tuple(parsed_effects)
    if schema_version != "1.0":
        conditions = compile_event_condition_bindings(conditions, effects)
        effects = normalize_event_probability_ownership(effects, conditions)
        effects = _fill_protective_overall_context(
            effects, parties, conditions, lookup,
        )
        parties = strip_assignment_subgroups_from_parties(parties, effects)
    # ``effect.action_id`` owns the relationship. ``action.effect_ids`` is only
    # a redundant lookup index, so derive it transactionally instead of letting
    # stale model-authored bookkeeping reject an otherwise coherent world model.
    effects_by_action: dict[str, list[str]] = {action_id: [] for action_id in action_ids}
    for effect in effects:
        effects_by_action.setdefault(effect.action_id, []).append(effect.effect_id)
    actions = tuple(
        replace(action, effect_ids=tuple(effects_by_action.get(action.action_id, ())))
        for action in actions
    )
    effect_by_id = {effect.effect_id: effect for effect in effects}
    parsed_links: list[CausalLink] = []
    migrated_counterfactuals: list[CounterfactualLink] = []
    for row in raw.get("causal_links", []):
        if not isinstance(row, dict):
            continue
        source_id = _clean(row.get("source_id"), 80)
        target_id = _clean(row.get("target_id"), 80)
        relation = _first_clean(row, "link_relation", "relation", limit=64).upper() or "CAUSES"
        source_action = (
            source_id if source_id in action_ids
            else effect_by_id[source_id].action_id if source_id in effect_by_id
            else ""
        )
        target_action = (
            target_id if target_id in action_ids
            else effect_by_id[target_id].action_id if target_id in effect_by_id
            else ""
        )
        refs = _refs(row.get("clause_ids", []), lookup)
        condition_values = tuple(dict.fromkeys(
            _clean(value, 80).upper() for value in row.get("condition_ids", [])
        ))
        modality = _clean(row.get("modality"), 48).upper()
        explicit_action = _clean(row.get("action_id"), 16).upper()
        # Cross-action foreclosure never belongs on the actual-world causal
        # graph. Schema 1.0 stored FOREGOES there; live 1.2 repairs sometimes
        # still do. Migrate when a FOREGONE source can be resolved. Leave
        # CAUSES/ENABLES/PREVENTS as causal_links so they still fail the
        # action-boundary check instead of being silently rewritten.
        if source_action and target_action and source_action != target_action:
            mapped = _FORECLOSURE_LINK_RELATIONS.get(relation)
            source_effect_id = (
                _foreclosure_source_effect_id(
                    source_id=source_id,
                    source_action=source_action,
                    target_id=target_id,
                    effects=effects,
                    effect_by_id=effect_by_id,
                )
                if mapped else ""
            )
            if mapped and (
                source_effect_id
                or (schema_version == "1.0" and relation == "FOREGOES")
            ):
                migrated_counterfactuals.append(CounterfactualLink(
                    action_id=source_action,
                    source_effect_id=source_effect_id,
                    relation=mapped,
                    alternative_action_id=target_action,
                    alternative_effect_id=target_id,
                    modality=modality,
                    condition_ids=condition_values,
                    provenance=refs,
                ))
                continue
            parsed_links.append(CausalLink(
                action_id=explicit_action or source_action,
                source_id=source_id, relation=relation, target_id=target_id,
                modality=modality, condition_ids=condition_values,
                provenance=refs,
            ))
            continue
        parsed_links.append(CausalLink(
            action_id=explicit_action or source_action or target_action,
            source_id=source_id, relation=relation, target_id=target_id,
            modality=modality, condition_ids=condition_values,
            provenance=refs,
        ))
    counterfactuals = [*migrated_counterfactuals]
    counterfactuals.extend(CounterfactualLink(
        action_id=_clean(row.get("action_id"), 16).upper(),
        source_effect_id=_clean(row.get("source_effect_id"), 80),
        relation=_first_clean(row, "counterfactual_relation", "relation", limit=64).upper(),
        alternative_action_id=_clean(row.get("alternative_action_id"), 16).upper(),
        alternative_effect_id=_clean(row.get("alternative_effect_id"), 80),
        modality=_clean(row.get("modality"), 48).upper(),
        condition_ids=tuple(dict.fromkeys(
            _clean(value, 80).upper() for value in row.get("condition_ids", [])
        )),
        provenance=_refs(row.get("clause_ids", []), lookup),
    ) for row in raw.get("counterfactual_links", []) if isinstance(row, dict))
    if schema_version != "1.0":
        parsed_links = list(normalize_redundant_link_gates(parsed_links, effects))
    model = ScenarioWorldModel(
        parties=parties, actions=actions, effects=effects,
        conditions=conditions, causal_links=tuple(parsed_links),
        counterfactual_links=tuple(counterfactuals),
        admission=WorldStateAdmission(
            status="COMMITTED",
            admitted_effect_ids=tuple(effect.effect_id for effect in effects),
        ),
        schema_version=schema_version,
    )
    if schema_version != "1.0":
        model = compile_chance_gated_world(model)
    errors, _ = validate_world_model(model, action_ids=action_ids)
    if require_completeness:
        errors.extend(validate_world_completeness(model, action_ids=action_ids))
    if errors:
        raise WorldModelValidationError(
            errors,
            validation_issues_from_messages(errors),
        )
    return model


def _opposed(left: WorldEffect, right: WorldEffect) -> bool:
    if left.polarity == right.polarity:
        return False
    if "NEUTRAL" in {left.polarity, right.polarity}:
        return False
    terminal = {"DIES", "SURVIVES", "KILLED", "SAVED", "DEATH", "SURVIVAL"}
    relations = {left.relation, right.relation}
    return bool(relations & terminal) or {
        left.polarity, right.polarity,
    } == {"BENEFICIAL", "ADVERSE"} and left.outcome.casefold() == right.outcome.casefold()


def validate_world_model(
    model: ScenarioWorldModel,
    *,
    action_ids: Sequence[str],
) -> tuple[list[str], list[tuple[str, ...]]]:
    """Return validation errors and sets of mutually contradictory effects."""
    errors: list[str] = []
    expected_actions = set(action_ids)
    party_ids = {party.party_id for party in model.parties if party.party_id}
    condition_ids = {condition.condition_id for condition in model.conditions}
    effect_ids = [effect.effect_id for effect in model.effects]
    if len(party_ids) != len(model.parties):
        errors.append("parties require unique non-empty party_id values")
    for party in model.parties:
        provenance_text = " ".join(
            ref.excerpt for ref in party.provenance if ref.excerpt
        ).casefold()
        for quantity in party.quantities:
            if quantity.casefold() not in provenance_text:
                errors.append(
                    f"{party.party_id} quantity {quantity!r} is not stated in its provenance"
                )
    if model.schema_version in {"1.2", "1.3"}:
        expected_by_party = assigned_party_quantities(model.parties)
        owned_by_party = assignment_owned_quantities(model.effects)
        for party in model.parties:
            recorded = {value.casefold() for value in party.quantities}
            omitted = [
                value for value in expected_by_party.get(party.party_id, ())
                if value.casefold() not in recorded
                and not _is_smaller_assignment_quantity(
                    value,
                    expected_by_party.get(party.party_id, ()),
                    owned_by_party.get(party.party_id, ()),
                )
            ]
            if omitted:
                errors.append(
                    f"{party.party_id} omits source-grounded population quantities: {omitted}"
                )
    if {action.action_id for action in model.actions} != expected_actions:
        errors.append("typed world actions must cover every canonical action exactly once")
    if len(set(effect_ids)) != len(effect_ids) or any(not value for value in effect_ids):
        errors.append("effects require unique non-empty effect_id values")
    for action in model.actions:
        if not action.intervention:
            errors.append(f"{action.action_id} lacks a neutral intervention")
        if not action.actor_party_id or action.actor_party_id not in party_ids:
            errors.append(f"{action.action_id} lacks a known actor")
        unknown_recipients = set(action.recipient_party_ids) - party_ids
        if unknown_recipients:
            errors.append(f"{action.action_id} cites unknown recipients: {sorted(unknown_recipients)}")
        if not action.provenance:
            errors.append(f"{action.action_id} lacks source provenance")
        actual_effects = {
            effect.effect_id for effect in model.effects
            if effect.action_id == action.action_id
        }
        if set(action.effect_ids) != actual_effects:
            errors.append(
                f"{action.action_id} effect_ids do not exactly match its typed effects"
            )
        if model.schema_version != "1.0":
            for recipient_id in action.recipient_party_ids:
                if not any(
                    effect.action_id == action.action_id
                    and effect.party_id == recipient_id
                    and effect.directness == "DIRECT"
                    and effect.effect_kind in _RECIPIENT_DIRECT_ACT_KINDS
                    for effect in model.effects
                ):
                    errors.append(
                        f"{action.action_id} recipient {recipient_id} lacks an atomic "
                        "DIRECT INTERVENTION, RESOURCE_TRANSFER, or "
                        "INSTITUTIONAL_OUTCOME effect"
                    )
    for effect in model.effects:
        prefix = effect.effect_id or "effect"
        if effect.action_id not in expected_actions:
            errors.append(f"{prefix} cites unknown action {effect.action_id}")
        if effect.party_id not in party_ids:
            errors.append(f"{prefix} cites unknown party {effect.party_id}")
        if not effect.outcome or not effect.relation:
            errors.append(f"{prefix} lacks outcome or relation")
        if effect.directness not in DIRECTNESSES:
            errors.append(f"{prefix} has invalid directness {effect.directness}")
        if effect.modality not in MODALITIES:
            errors.append(f"{prefix} has invalid modality {effect.modality}")
        if effect.polarity not in POLARITIES:
            errors.append(f"{prefix} has invalid polarity {effect.polarity}")
        if effect.effect_kind not in EFFECT_KINDS:
            errors.append(f"{prefix} has invalid effect_kind {effect.effect_kind}")
        if model.schema_version != "1.0":
            if (
                effect.effect_kind in {"INTERVENTION", "RESOURCE_TRANSFER"}
                and effect.directness != "DIRECT"
            ):
                errors.append(
                    f"{prefix} is an immediate {effect.effect_kind} effect so its "
                    "directness must be DIRECT. If this row is later conduct "
                    "(assist, remain, use a road) rather than the actor's act on "
                    "a recipient, change effect_kind and keep DOWNSTREAM; do not "
                    "add a later crowd as a recipient to make it DIRECT. If this "
                    "is the actor assigning or allocating that party, mark DIRECT "
                    "and name them as recipient of the assignment, distinct from "
                    "later assist or exposure"
                )
            if (
                effect.effect_kind == "INTERVENTION"
                and effect.directness == "DIRECT"
                and effect.polarity not in {"NEUTRAL", "FOREGONE"}
            ):
                bearer = next(
                    (item for item in model.parties if item.party_id == effect.party_id),
                    None,
                )
                if not _party_bears_welfare(bearer):
                    kind = bearer.kind if bearer is not None else "unknown"
                    errors.append(
                        f"{prefix} is a DIRECT INTERVENTION on non-welfare-bearing "
                        f"{kind} {effect.party_id}; polarity must be NEUTRAL. "
                        "Record BENEFICIAL or ADVERSE on the health, welfare, or "
                        "liberty row, not the system act"
                    )
            if (
                effect.effect_kind == "OPPORTUNITY_LOSS"
                and effect.directness != "FOREGONE"
            ):
                errors.append(
                    f"{prefix} is an OPPORTUNITY_LOSS so its directness must be FOREGONE"
                )
            if (
                effect.directness == "FOREGONE"
                and effect.effect_kind != "OPPORTUNITY_LOSS"
            ):
                errors.append(
                    f"{prefix} is FOREGONE so its effect_kind must be OPPORTUNITY_LOSS"
                )
        if model.schema_version != "1.0" and _CHAINED_OUTCOME.search(effect.outcome):
            errors.append(
                f"{prefix} combines multiple causal stages in one outcome; "
                "split resource/intermediate/human effects and connect them causally"
            )
        party = next((item for item in model.parties if item.party_id == effect.party_id), None)
        if (
            model.schema_version != "1.0"
            and
            party is not None
            and party.kind in _NONHUMAN_PARTY_KINDS
            and _HUMAN_OUTCOME.search(effect.outcome)
        ):
            errors.append(
                f"{prefix} assigns a human outcome to {party.kind} {party.party_id}; "
                "create a distinct affected-person or population party"
            )
        if effect.directness == "FOREGONE" and effect.polarity != "FOREGONE":
            errors.append(
                f"{prefix} is a foregone effect so its polarity must be FOREGONE, "
                f"not {effect.polarity}"
            )
        if effect.polarity == "FOREGONE" and effect.directness != "FOREGONE":
            errors.append(
                f"{prefix} uses polarity FOREGONE but directness is {effect.directness}"
            )
        if effect.modality != "CERTAIN" and not effect.condition_ids:
            independent_event = (
                model.schema_version != "1.0"
                and _stochastic_process_role(
                    effect,
                    next(
                        (
                            action for action in model.actions
                            if action.action_id == effect.action_id
                        ),
                        None,
                    ),
                    party,
                    {item.party_id: item for item in model.parties},
                ) == "INDEPENDENT"
            )
            if not (
                (
                    model.schema_version != "1.0"
                    and effect.likelihood_qualifiers
                )
                or independent_event
            ):
                errors.append(f"{prefix} is {effect.modality} but has no condition")
        if effect.modality == "CERTAIN" and effect.condition_ids:
            errors.append(
                f"{prefix} is CERTAIN but lists conditions; CERTAIN effects "
                "must not carry condition_ids"
            )
        if (
            model.schema_version != "1.0"
            and effect.modality == "CERTAIN"
            and _chance_hedge_blocks_certain(effect)
        ):
            errors.append(
                f"{prefix} is CERTAIN but likelihood_qualifiers contain a chance "
                "hedge; type PROBABILISTIC or POSSIBLE. CERTAIN is for unhedged "
                "indicatives and at-risk situations"
            )
        unknown_conditions = set(effect.condition_ids) - condition_ids
        if unknown_conditions:
            errors.append(f"{prefix} cites unknown conditions: {sorted(unknown_conditions)}")
        if not effect.provenance:
            errors.append(f"{prefix} lacks source provenance")
        elif not any(is_supporting_source(ref) for ref in effect.provenance):
            errors.append(
                f"{prefix} cites only comparison/interrogative context; "
                "effects need a FACT clause or confirmed action text as support"
            )
        # Quantities belong to this effect, not the union of every cited clause.
        # A harvest row that also cites a dehydration clause must copy 'hundreds'
        # if the outcome uses it; it must not be forced to list 'dozen'.
        # A party-recorded population count may ground the same count on that
        # party's effects when an anaphoric clause omits the numeral.
        provenance_text = _provenance_text(effect)
        claimed = explicit_quantity_spans(effect.outcome)
        recorded = {item.casefold() for item in effect.quantities}

        def quantity_is_grounded(span: str) -> bool:
            return _quantity_grounded_in_text(span, provenance_text) or (
                _party_licenses_quantity(party, span)
            )

        ungrounded = [span for span in claimed if not quantity_is_grounded(span)]
        if ungrounded:
            errors.append(
                f"{prefix} outcome uses quantities absent from provenance: "
                f"{ungrounded} (outcome text alone cannot ground a quantity)"
            )
        omitted = [
            span for span in claimed
            if quantity_is_grounded(span) and span.casefold() not in recorded
        ]
        if omitted:
            errors.append(
                f"{prefix} omits quantities stated in its outcome and provenance: "
                f"{omitted}"
            )
        for quantity in effect.quantities:
            if not quantity_is_grounded(quantity):
                errors.append(
                    f"{prefix} quantity {quantity!r} is not stated in its "
                    "provenance (outcome text alone cannot ground a quantity)"
                )
        qualifier_fields = _qualifier_channels()
        for qualifier_kind, field, extractor in qualifier_fields:
            recorded_values = getattr(effect, field)
            if field == "likelihood_qualifiers":
                recorded_qualifiers = {
                    _likelihood_identity_key(value) for value in recorded_values
                }
                recorded_qualifiers.update(
                    _likelihood_identity_key(value)
                    for value in effect.overall_likelihood_qualifiers
                )
            else:
                recorded_qualifiers = {value.casefold() for value in recorded_values}
            for value in recorded_values:
                if not _qualifier_grounded_in_text(value, provenance_text):
                    errors.append(
                        f"{prefix} {qualifier_kind} qualifier {value!r} is not stated "
                        "in its provenance"
                    )
            if model.schema_version in {"1.2", "1.3"}:
                required = list(effect_expected_qualifiers(effect, extractor))
                missing = [
                    value for value in required
                    if (
                        _likelihood_identity_key(value)
                        if field == "likelihood_qualifiers"
                        else value.casefold()
                    ) not in recorded_qualifiers
                ]
                if missing:
                    errors.append(
                        f"{prefix} omits source-grounded {qualifier_kind} qualifiers: "
                        f"{missing}"
                    )
    valid_link_nodes = expected_actions | set(effect_ids)
    if model.schema_version == "1.0":
        valid_link_nodes |= condition_ids
    effect_by_id = {effect.effect_id: effect for effect in model.effects}
    effect_owner = {effect_id: effect.action_id for effect_id, effect in effect_by_id.items()}
    for index, link in enumerate(model.causal_links):
        prefix = f"causal_link[{index}]"
        if link.source_id not in valid_link_nodes or link.target_id not in valid_link_nodes:
            errors.append(f"{prefix} cites an unknown causal endpoint")
        if link.modality not in MODALITIES:
            errors.append(f"{prefix} has invalid modality {link.modality}")
        if set(link.condition_ids) - condition_ids:
            errors.append(f"{prefix} cites unknown conditions")
        if link.modality != "CERTAIN" and not link.condition_ids:
            errors.append(f"{prefix} is conditional but has no condition")
        if link.modality == "CERTAIN" and link.condition_ids:
            errors.append(
                f"{prefix} is CERTAIN but lists conditions; CERTAIN links "
                "must not carry condition_ids"
            )
        if not link.provenance:
            errors.append(f"{prefix} lacks source provenance")
        endpoint_owners = {
            endpoint if endpoint in expected_actions else effect_owner.get(endpoint, "")
            for endpoint in (link.source_id, link.target_id)
        } - {""}
        owning_action = link.action_id or (
            next(iter(endpoint_owners)) if len(endpoint_owners) == 1 else ""
        )
        if owning_action not in expected_actions:
            errors.append(f"{prefix} lacks a valid owning action_id")
        if endpoint_owners != {owning_action}:
            errors.append(
                f"{prefix} crosses action boundaries {sorted(endpoint_owners)}; "
                "use counterfactual_links for alternative-action comparisons"
            )
        for endpoint_id, endpoint_label in (
            (link.source_id, "source"),
            (link.target_id, "target"),
        ):
            endpoint = effect_by_id.get(endpoint_id)
            if endpoint is not None and endpoint.directness == "FOREGONE":
                errors.append(
                    f"{prefix} {endpoint_label} is a FOREGONE effect; keep "
                    "opportunity-loss comparisons on counterfactual_links so "
                    "the actual-world causal chain stays action-local"
                )
        if (
            model.schema_version != "1.0"
            and link.relation == "PREVENTS"
        ):
            source = effect_by_id.get(link.source_id)
            target = effect_by_id.get(link.target_id)
            if (
                source is not None
                and target is not None
                and source.directness != "FOREGONE"
                and target.directness != "FOREGONE"
                and source.polarity in {"BENEFICIAL", "ADVERSE"}
                and source.polarity == target.polarity
            ):
                errors.append(
                    f"{prefix} uses PREVENTS between actual {source.polarity} "
                    f"effects {source.effect_id} and {target.effect_id}; both "
                    "obtain under this action, so the parent produces the child. "
                    "Change only link_relation to CAUSES, ENABLES, or ACCELERATES; "
                    "keep the same endpoints. Averted opposites belong on FOREGONE "
                    "counterfactual_links."
                )
    for index, link in enumerate(model.counterfactual_links):
        prefix = f"counterfactual_link[{index}]"
        source = effect_by_id.get(link.source_effect_id)
        alternative = effect_by_id.get(link.alternative_effect_id)
        if link.action_id not in expected_actions:
            errors.append(f"{prefix} cites unknown action {link.action_id}")
        if link.alternative_action_id not in expected_actions:
            errors.append(
                f"{prefix} cites unknown alternative action {link.alternative_action_id}"
            )
        if link.action_id == link.alternative_action_id:
            errors.append(f"{prefix} must compare distinct actions")
        if link.relation not in COUNTERFACTUAL_RELATIONS:
            errors.append(f"{prefix} has invalid relation {link.relation}")
        if source is None or source.action_id != link.action_id:
            errors.append(f"{prefix} source effect does not belong to its action")
        elif source.directness != "FOREGONE":
            errors.append(f"{prefix} source effect must be typed FOREGONE")
        if alternative is None or alternative.action_id != link.alternative_action_id:
            errors.append(
                f"{prefix} alternative effect does not belong to the alternative action"
            )
        if source is not None and alternative is not None and source.party_id != alternative.party_id:
            errors.append(f"{prefix} compares effects on different parties")
        if link.modality not in MODALITIES:
            errors.append(f"{prefix} has invalid modality {link.modality}")
        if link.modality != "CERTAIN" and not link.condition_ids:
            errors.append(f"{prefix} is conditional but has no condition")
        if link.modality == "CERTAIN" and link.condition_ids:
            errors.append(
                f"{prefix} is CERTAIN but lists conditions; CERTAIN links "
                "must not carry condition_ids"
            )
        if set(link.condition_ids) - condition_ids:
            errors.append(f"{prefix} cites unknown conditions")
        if not link.provenance:
            errors.append(f"{prefix} lacks source provenance")
    if model.schema_version == "1.3":
        errors.extend(validate_effect_source_bindings(model))
    contradictions: list[tuple[str, ...]] = []
    direct = [effect for effect in model.effects if effect.directness == "DIRECT"]
    for index, left in enumerate(direct):
        conflict = [left.effect_id]
        for right in direct[index + 1:]:
            if (
                left.action_id == right.action_id
                and left.party_id == right.party_id
                and left.modality == right.modality == "CERTAIN"
                and _opposed(left, right)
            ):
                conflict.append(right.effect_id)
        if len(conflict) > 1 and not any(set(conflict) <= set(row) for row in contradictions):
            contradictions.append(tuple(conflict))
    return list(dict.fromkeys(errors)), contradictions


def _action_source_text(action: WorldAction) -> str:
    return " ".join(
        part for part in (
            action.intervention,
            *(ref.excerpt for ref in action.provenance if ref.excerpt),
        ) if part
    )


def _source_states_false_attribution(text: str) -> bool:
    blob = str(text or "")
    if _FALSE_ATTRIBUTION.search(blob):
        return True
    return bool(_FRAME_VERB.search(blob) and _EXECUTE_VERB.search(blob))


def _mentioned_non_actor_humans(
    action: WorldAction, parties: Sequence[WorldParty],
) -> tuple[WorldParty, ...]:
    blob = _action_source_text(action).casefold()
    found: list[WorldParty] = []
    for party in parties:
        if party.kind not in _ROLE_PERSON_KINDS:
            continue
        if party.party_id == action.actor_party_id:
            continue
        label = party.label.casefold()
        if label and label in blob:
            found.append(party)
    return tuple(found)


_SKIP_INTERMEDIATE_PARENT_KINDS = {
    "INTERVENTION", "RESOURCE_TRANSFER", "HEALTH_OUTCOME", "WELFARE_OUTCOME",
    "INSTITUTIONAL_OUTCOME",
}
# A riot, spillway, channel, or institution can carry INSTITUTIONAL_OUTCOME or
# INTERVENTION as the intermediate stage. These kinds match non-welfare
# recipients. The skip list is for another person's act, framing, or bodily
# outcome posing as that stage.
_INTERMEDIATE_BEARER_KINDS = {
    "PROCESS", "FACILITY", "INSTITUTION", "ORGANIZATION",
    "SYSTEM", "AUTOMATED_SYSTEM", "INFRASTRUCTURE", "RESOURCE",
}
_INTERMEDIATE_STATE_KINDS = {
    "PHYSICAL_STATE", "OTHER", "INSTITUTIONAL_OUTCOME",
}
_RECIPIENT_DIRECT_ACT_KINDS = {
    "INTERVENTION", "RESOURCE_TRANSFER", "INSTITUTIONAL_OUTCOME",
}
_RESOURCE_TRANSFER_CUE = re.compile(
    r"\b(?:send|sends|sent|sending|dispatch|dispatched|deliver|delivered|"
    r"transfer|transferred|allocate|allocated)\b",
    re.IGNORECASE,
)
_RESOURCE_OBJECT_TRANSFER_CUE = re.compile(
    r"\b(?:send|sends|sent|sending|dispatch|dispatched|deliver|delivered|"
    r"transfer|transferred)\b",
    re.IGNORECASE,
)


def _link_parents_target(link: CausalLink) -> bool:
    """True when the link is an ordinary causal parent, not a negative constraint."""
    return str(link.relation or "CAUSES").upper() != "DOES_NOT_INCREASE"


def _downstream_health_parents(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> tuple[WorldEffect, ...]:
    by_id = {item.effect_id: item for item in model.effects}
    parents: list[WorldEffect] = []
    for link in model.causal_links:
        if link.action_id != effect.action_id or link.target_id != effect.effect_id:
            continue
        if not _link_parents_target(link):
            continue
        parent = by_id.get(link.source_id)
        if parent is not None:
            parents.append(parent)
    return tuple(parents)


def _ancestry_reaches_direct_act(
    effect: WorldEffect,
    action: WorldAction,
    model: ScenarioWorldModel,
) -> bool:
    """True when a within-action causal walk hits a DIRECT actor/recipient act.

    Counterfactual links are ignored. The walk may pass through intermediate
    process states; what must not happen is a terminal welfare outcome whose
    only ancestors are other unconnected process or population states.
    """
    by_id = {item.effect_id: item for item in model.effects}
    allowed = {action.actor_party_id, *action.recipient_party_ids}
    seen: set[str] = set()
    stack = [effect.effect_id]
    while stack:
        current_id = stack.pop()
        if current_id in seen:
            continue
        seen.add(current_id)
        if current_id == action.action_id:
            return True
        current = by_id.get(current_id)
        if (
            current is not None
            and current.effect_id != effect.effect_id
            and current.directness == "DIRECT"
            and current.party_id in allowed
        ):
            return True
        for link in model.causal_links:
            if link.action_id != effect.action_id or link.target_id != current_id:
                continue
            if not _link_parents_target(link):
                continue
            stack.append(link.source_id)
    return False


def _process_is_exogenous(
    effect: WorldEffect,
    party: WorldParty | None,
    action: WorldAction | None = None,
    party_by_id: dict[str, WorldParty] | None = None,
) -> bool:
    """True when a stochastic intermediate is not source-attributed to the act.

    Hedged does not mean exogenous. Action-caused stochastic processes
    (the intervention may cause a collapse) may be traversed. Independent
    and ambiguous rows may not.
    """
    role = _stochastic_process_role(effect, action, party, party_by_id)
    return role in {"INDEPENDENT", "AMBIGUOUS"}


def _human_outcome_reaches_direct_act(
    effect: WorldEffect,
    action: WorldAction,
    model: ScenarioWorldModel,
    party_by_id: dict[str, WorldParty],
) -> bool:
    """True when a path to DIRECT does not invent causation.

    Independent background events and ambiguous associations are not a
    mediated path. An explicitly action-caused stochastic process is.
    """
    by_id = {item.effect_id: item for item in model.effects}
    allowed = {action.actor_party_id, *action.recipient_party_ids}

    def walk(current_id: str, seen: frozenset[str]) -> bool:
        if current_id in seen:
            return False
        nxt = seen | {current_id}
        current = by_id.get(current_id)
        if current is not None and current.effect_id != effect.effect_id:
            if current.directness == "DIRECT" and current.party_id in allowed:
                return True
            bearer = party_by_id.get(current.party_id)
            if _process_is_exogenous(current, bearer, action, party_by_id):
                return False
        for link in model.causal_links:
            if link.action_id != effect.action_id or link.target_id != current_id:
                continue
            if not _link_parents_target(link):
                continue
            if walk(link.source_id, nxt):
                return True
        return False

    return walk(effect.effect_id, frozenset())


_CROWD_PROCESS_STATE = re.compile(
    r"\b(?:use|uses|using|used|remain|remains|remaining|occupy|occupies|"
    r"occupying|occupied|travel|travels|traveling|travelling|stay|stays|"
    r"staying|move|moved|moving|evacuat\w*|assist\w*)\b",
    re.IGNORECASE,
)


def _crowd_mediated_process(
    effect: WorldEffect, party: WorldParty | None,
) -> bool:
    """True when a crowd row is use/remain/occupy, not a harm or welfare event.

    That row is the action-mediated process. RESOURCE_TRANSFER on a different
    recipient may parent it. Trapped, death, and escape still need ancestry
    through that process, not through the other party's body.
    """
    if party is None or not _party_bears_welfare(party):
        return False
    if effect.effect_kind not in {"PHYSICAL_STATE", "OTHER"}:
        return False
    outcome = _folded_outcome_text(effect.outcome)
    if _HUMAN_OUTCOME.search(outcome):
        return False
    return bool(_CROWD_PROCESS_STATE.search(outcome))


def _requires_act_ancestry(effect: WorldEffect, party: WorldParty | None) -> bool:
    if effect.directness != "DOWNSTREAM":
        return False
    if _crowd_mediated_process(effect, party):
        return False
    if effect.effect_kind in _ROLE_WELFARE_KINDS:
        return True
    return (
        effect.effect_kind == "PHYSICAL_STATE"
        and _party_bears_welfare(party)
    )


def _parent_is_foreign_act_or_body(
    parent: WorldEffect,
    child: WorldEffect,
    party_by_id: dict[str, WorldParty],
) -> bool:
    """True when the parent is another party's act or body, not a process state.

    INSTITUTIONAL_OUTCOME on a person (frame, execute) is not a valid
    immediate parent of someone else's health. The same kind on a PROCESS,
    FACILITY, INSTITUTION, INFRASTRUCTURE, or RESOURCE is the intermediate
    the completeness gate asked for.
    """
    if parent.party_id == child.party_id:
        return False
    if parent.effect_kind not in _SKIP_INTERMEDIATE_PARENT_KINDS:
        return False
    bearer = party_by_id.get(parent.party_id)
    if bearer is not None and bearer.kind in _INTERMEDIATE_BEARER_KINDS:
        return False
    return True


def _party_mentioned_in_text(party: WorldParty, text: str) -> bool:
    blob = str(text or "").casefold()
    label = party.label.casefold().strip()
    if label and label in blob:
        return True
    tokens = {
        word for word in re.findall(r"[a-z0-9]+", label)
        if len(word) > 2 and word not in _PARTY_MATCH_STOPWORDS
    }
    distinctive = tokens - _PARTY_GENERIC_NOUNS
    return bool(distinctive) and all(token in blob for token in distinctive)


def _unused_cited_intermediate_parties(
    effect: WorldEffect,
    action: WorldAction,
    model: ScenarioWorldModel,
    party_by_id: dict[str, WorldParty],
) -> tuple[WorldParty, ...]:
    """Intermediate bearers named in the child's source with no process-state row."""
    source = " ".join(
        [ref.excerpt for ref in effect.provenance if ref.excerpt]
        + [_action_source_text(action)]
    )
    used = {
        other.party_id
        for other in model.effects
        if other.action_id == effect.action_id
        and other.effect_id != effect.effect_id
        and other.directness != "FOREGONE"
        and other.effect_kind in _INTERMEDIATE_STATE_KINDS
        and party_by_id.get(other.party_id) is not None
        and party_by_id[other.party_id].kind in _INTERMEDIATE_BEARER_KINDS
    }
    found: list[WorldParty] = []
    for party in model.parties:
        if party.kind not in _INTERMEDIATE_BEARER_KINDS:
            continue
        if party.party_id == action.actor_party_id:
            continue
        if party.party_id in used:
            continue
        if _party_mentioned_in_text(party, source):
            found.append(party)
    return tuple(found)


def _same_clause_process_parent_ids(
    effect: WorldEffect,
    action: WorldAction,
    model: ScenarioWorldModel,
    party_by_id: dict[str, WorldParty],
) -> tuple[str, ...]:
    """Process rows that share provenance with the child, not every process."""
    child_clauses = {ref.clause_id for ref in effect.provenance if ref.clause_id}
    found: list[str] = []
    for other in model.effects:
        if other.action_id != effect.action_id or other.effect_id == effect.effect_id:
            continue
        if other.directness == "FOREGONE" or other.polarity == "FOREGONE":
            continue
        if other.effect_kind not in _INTERMEDIATE_STATE_KINDS:
            continue
        bearer = party_by_id.get(other.party_id)
        if bearer is None or bearer.kind not in _INTERMEDIATE_BEARER_KINDS:
            continue
        if _process_is_exogenous(other, bearer, action, party_by_id):
            continue
        other_clauses = {ref.clause_id for ref in other.provenance if ref.clause_id}
        if child_clauses & other_clauses:
            found.append(other.effect_id)
    return tuple(found)


def _foreign_parent_error(
    effect: WorldEffect,
    skipped_parents: Sequence[WorldEffect],
    action: WorldAction,
    model: ScenarioWorldModel,
    party_by_id: dict[str, WorldParty],
) -> str:
    parent_ids = ", ".join(parent.effect_id for parent in skipped_parents)
    message = (
        f"{effect.effect_id} is caused directly by another party's act "
        f"or bodily outcome ({parent_ids}); insert a PROCESS, FACILITY, "
        "INSTITUTION, INFRASTRUCTURE, or RESOURCE state as the immediate "
        "parent. A person's framing, execution, or death is not that "
        f"intermediate. Do not delete {effect.effect_id} or drop its "
        "causal_link without a replacement parent"
    )
    unused = _unused_cited_intermediate_parties(
        effect, action, model, party_by_id,
    )
    if unused:
        named = ", ".join(
            f"{party.party_id} ({party.kind} {party.label})" for party in unused
        )
        return (
            message + f"; the source also names {named}, a plausible missing "
            f"process-state parent for {effect.effect_id} — confirm that "
            "reading from the source; an unused facility or resource is not "
            "automatically the correct intermediary"
        )
    same_clause = _same_clause_process_parent_ids(
        effect, action, model, party_by_id,
    )
    if same_clause:
        return (
            message + "; reuse the source-named facility or resource process "
            f"already in this clause ({', '.join(same_clause)}) as the immediate "
            "parent, not an unrelated stochastic process"
        )
    return message


def _effect_counts_for_foregone_overlays(
    effect: WorldEffect,
    party: WorldParty | None,
    model: ScenarioWorldModel,
) -> bool:
    """True when a row can participate in an opposed-welfare FOREGONE pair.

    Compact role-counting rows are the usual pair. Opposed PHYSICAL states on
    the same intermediate bearer (facility, resource, process) are the same
    stipulated swap even when those states are not compact welfare parties.
    """
    if _effect_counts_for_roles(effect, party, model):
        return True
    if (
        party is None
        or effect.directness == "FOREGONE"
        or effect.polarity not in {"BENEFICIAL", "ADVERSE"}
        or party.kind not in _INTERMEDIATE_BEARER_KINDS
        or effect.effect_kind not in _INTERMEDIATE_STATE_KINDS
    ):
        return False
    return True


def _actual_nonrecipient_role_effects(
    model: ScenarioWorldModel,
) -> dict[tuple[str, str], tuple[WorldEffect, ...]]:
    """Actual harm/benefit rows on parties the action does not itself treat.

    Recipient and actor effects stay on the within-action chain. FOREGONE
    overlays belong on the surrounding mutually exclusive welfare swap.
    """
    party_by_id = {party.party_id: party for party in model.parties}
    action_by_id = {action.action_id: action for action in model.actions}
    grouped: dict[tuple[str, str], list[WorldEffect]] = {}
    for effect in model.effects:
        if effect.directness == "FOREGONE" or effect.polarity == "FOREGONE":
            continue
        party = party_by_id.get(effect.party_id)
        if not _effect_counts_for_foregone_overlays(effect, party, model):
            continue
        action = action_by_id.get(effect.action_id)
        if action is None:
            continue
        if effect.party_id == action.actor_party_id:
            continue
        if effect.party_id in action.recipient_party_ids:
            if effect.directness == "DIRECT":
                continue
            bearer = party_by_id.get(effect.party_id)
            if bearer is not None and bearer.kind in _ROLE_PERSON_KINDS:
                continue
        grouped.setdefault((effect.party_id, effect.action_id), []).append(effect)
    return {key: tuple(value) for key, value in grouped.items()}


def _has_counterfactual_foreclosure(
    model: ScenarioWorldModel,
    *,
    action_id: str,
    party_id: str,
    alternative_action_id: str,
    alternative_effect_ids: set[str],
) -> bool:
    owned_foregone = {
        effect.effect_id
        for effect in model.effects
        if effect.action_id == action_id
        and effect.party_id == party_id
        and effect.directness == "FOREGONE"
    }
    if not owned_foregone:
        return False
    return any(
        link.action_id == action_id
        and link.source_effect_id in owned_foregone
        and link.alternative_action_id == alternative_action_id
        and link.alternative_effect_id in alternative_effect_ids
        for link in model.counterfactual_links
    )


def _foreclosure_completeness_errors(model: ScenarioWorldModel) -> list[str]:
    """Require FOREGONE overlays for opposed welfare, including downstream recipient rows."""
    grouped = _actual_nonrecipient_role_effects(model)
    action_ids = [action.action_id for action in model.actions]
    errors: list[str] = []
    for party_id in sorted({party for party, _ in grouped}):
        for index, left_id in enumerate(action_ids):
            for right_id in action_ids[index + 1:]:
                left = grouped.get((party_id, left_id), ())
                right = grouped.get((party_id, right_id), ())
                left_polarities = {item.polarity for item in left}
                right_polarities = {item.polarity for item in right}
                opposed = (
                    "BENEFICIAL" in left_polarities
                    and "ADVERSE" in right_polarities
                ) or (
                    "ADVERSE" in left_polarities
                    and "BENEFICIAL" in right_polarities
                )
                if not opposed:
                    continue
                left_ids = {item.effect_id for item in left}
                right_ids = {item.effect_id for item in right}
                if not _has_counterfactual_foreclosure(
                    model,
                    action_id=left_id,
                    party_id=party_id,
                    alternative_action_id=right_id,
                    alternative_effect_ids=right_ids,
                ):
                    errors.append(
                        f"{left_id} and {right_id} stipulate opposed welfare on "
                        f"{party_id}, but {left_id} has no FOREGONE "
                        f"effect with a counterfactual_link to {right_id}'s actual "
                        "effect; do not put that comparison on causal_links"
                    )
                if not _has_counterfactual_foreclosure(
                    model,
                    action_id=right_id,
                    party_id=party_id,
                    alternative_action_id=left_id,
                    alternative_effect_ids=left_ids,
                ):
                    errors.append(
                        f"{left_id} and {right_id} stipulate opposed welfare on "
                        f"{party_id}, but {right_id} has no FOREGONE "
                        f"effect with a counterfactual_link to {left_id}'s actual "
                        "effect; do not put that comparison on causal_links"
                    )
    return errors


def _stem_word(word: str) -> str:
    for suffix in ("ing", "ied", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            stem = word[: -len(suffix)]
            if suffix == "ies":
                stem += "y"
            return stem
    return word


_DEVERBAL_NOUN_SUFFIXES = ("age", "ure")


def _binder_stems(words: set[str]) -> set[str]:
    """Qualifier-head overlap, including deverbal nouns vs participles.

    'chance of blockage' must bind to BLOCKED. 'chance of failure' must bind
    to FAILS. Do not change `_stem_word`; independent-event classification
    uses that stemmer on whole sentences.
    """
    stems: set[str] = set()
    for word in words:
        stems.add(word)
        stems.add(_stem_word(word))
        for suffix in _DEVERBAL_NOUN_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                root = word[: -len(suffix)]
                stems.add(root)
                stems.add(_stem_word(root))
    return stems


_CAUSE_ATTRIBUTION = re.compile(
    r"\b(?:may|might|could|can|would|will)\s+"
    r"(?:cause|lead\s+to|result\s+in|bring\s+about)\b"
    r"|"
    r"\b(?:causes|causing|caused|leads\s+to|led\s+to|results\s+in|"
    r"resulted\s+in|brings\s+about|brought\s+about)\b",
    re.IGNORECASE,
)
_BACKGROUND_CHANCE = re.compile(
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent)\s*chance\b",
    re.IGNORECASE,
)


def _source_sentences(text: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in re.split(r"[.!?]+", str(text or ""))
        if part.strip()
    )


def _stemmed_content(text: str) -> set[str]:
    return {
        _stem_word(word)
        for word in re.findall(r"[a-z0-9]+", str(text or "").casefold())
        if len(word) > 2
    }


def _shares_stems(sentence: str, anchors: set[str]) -> bool:
    if not anchors:
        return False
    return bool(_stemmed_content(sentence) & {_stem_word(a) for a in anchors})


def _is_stochastic_intermediate(
    effect: WorldEffect, party: WorldParty | None,
) -> bool:
    if party is None or party.kind not in _INTERMEDIATE_BEARER_KINDS:
        return False
    if effect.directness != "DOWNSTREAM":
        return False
    if effect.effect_kind not in {"PHYSICAL_STATE", "OTHER"}:
        return False
    if str(effect.modality or "").upper() == "STIPULATED_CONDITIONAL":
        return False
    return bool(effect.likelihood_qualifiers) or effect.modality != "CERTAIN"


def _action_anchor_stems(
    action: WorldAction, party_by_id: dict[str, WorldParty],
) -> set[str]:
    blob = action.intervention
    for party_id in action.recipient_party_ids:
        party = party_by_id.get(party_id)
        if party is not None:
            blob += " " + party.label
    return _stemmed_content(blob)


def _effect_anchor_stems(effect: WorldEffect, party: WorldParty | None) -> set[str]:
    blob = effect.outcome
    if party is not None:
        blob += " " + party.label
    return _stemmed_content(blob)


def _causation_source_blob(effect: WorldEffect, action: WorldAction | None) -> str:
    parts = []
    if action is not None:
        parts.append(_action_source_text(action))
    parts.extend(ref.excerpt for ref in effect.provenance if ref.excerpt)
    return " ".join(part for part in parts if part)


def _source_action_causes_process(
    effect: WorldEffect,
    action: WorldAction | None,
    party: WorldParty | None,
    party_by_id: dict[str, WorldParty] | None,
) -> bool:
    if action is None:
        return False
    parties = party_by_id or {}
    blob = _causation_source_blob(effect, action)
    action_stems = _action_anchor_stems(action, parties)
    effect_stems = _effect_anchor_stems(effect, party)
    recipient = (
        party is not None
        and party.party_id in action.recipient_party_ids
    )
    for sentence in _source_sentences(blob):
        if _CAUSE_ATTRIBUTION.search(sentence) is None:
            continue
        has_action = _shares_stems(sentence, action_stems)
        has_effect = _shares_stems(sentence, effect_stems)
        if not has_effect and recipient:
            has_effect = bool(re.search(r"\b(?:it|its|they|them)\b", sentence, re.I))
        if has_action and has_effect:
            return True
    return False


def _source_stipulates_background(
    effect: WorldEffect,
    action: WorldAction | None,
    party: WorldParty | None,
    party_by_id: dict[str, WorldParty] | None,
) -> bool:
    blob = _causation_source_blob(effect, action)
    effect_stems = _effect_anchor_stems(effect, party)
    action_stems = (
        _action_anchor_stems(action, party_by_id or {}) if action is not None else set()
    )
    for sentence in _source_sentences(blob):
        if not _shares_stems(sentence, effect_stems):
            continue
        caused_here = (
            _CAUSE_ATTRIBUTION.search(sentence) is not None
            and _shares_stems(sentence, action_stems)
        )
        if caused_here:
            continue
        if _BACKGROUND_CHANCE.search(sentence) is not None:
            return True
        if action is not None and not _shares_stems(sentence, action_stems):
            return True
    return False


def _stochastic_process_role(
    effect: WorldEffect,
    action: WorldAction | None,
    party: WorldParty | None,
    party_by_id: dict[str, WorldParty] | None,
) -> str | None:
    """Classify a stochastic intermediate from the source, not from hedges.

    ACTION_CAUSED: the source attributes the process to the intervention.
    INDEPENDENT: a background chance/event the source does not attribute.
    AMBIGUOUS: associated with the action but neither attributed nor
    stipulated as background; do not invent a causal link.
    """
    if not _is_stochastic_intermediate(effect, party):
        return None
    if _source_action_causes_process(effect, action, party, party_by_id):
        return "ACTION_CAUSED"
    if _source_stipulates_background(effect, action, party, party_by_id):
        return "INDEPENDENT"
    return "AMBIGUOUS"


def _has_direct_intervention_parent(
    effect: WorldEffect,
    action: WorldAction,
    model: ScenarioWorldModel,
) -> bool:
    allowed = {action.actor_party_id, *action.recipient_party_ids}
    return any(
        parent.directness == "DIRECT" and parent.party_id in allowed
        for parent in _downstream_health_parents(effect, model)
    )


def _human_outcome_ancestry_error(
    effect: WorldEffect,
    action: WorldAction,
    model: ScenarioWorldModel,
    party_by_id: dict[str, WorldParty],
    parents: Sequence[WorldEffect],
) -> str:
    roles = []
    for parent in parents:
        bearer = party_by_id.get(parent.party_id)
        role = _stochastic_process_role(parent, action, bearer, party_by_id)
        if role is not None:
            roles.append((parent, role))
    action_caused_unlinked = any(
        role == "ACTION_CAUSED"
        and not _ancestry_reaches_direct_act(parent, action, model)
        for parent, role in roles
    )
    independent = any(role == "INDEPENDENT" for _parent, role in roles)
    if action_caused_unlinked:
        return (
            f"{effect.effect_id} is a downstream human outcome whose causal "
            "ancestry never reaches a DIRECT act on the actor or a named "
            "recipient; the source attributes a stochastic process to this "
            "action, so connect that process from the DIRECT intervention"
        )
    if _ancestry_reaches_direct_act(effect, action, model):
        if independent:
            return (
                f"{effect.effect_id} is a downstream human outcome whose "
                "only path to a DIRECT act passes through an independent "
                "stochastic background event; connect it through an "
                "action-mediated process. Do not parent the independent "
                "background event from the intervention to satisfy ancestry"
            )
        return (
            f"{effect.effect_id} is a downstream human outcome whose "
            "only path to a DIRECT act passes through a process the source "
            "does not attribute to the intervention; do not invent that "
            "causal link. Generate the missing action-mediated branch"
        )
    extra = (
        " An independent stochastic condition must not be parented "
        "from the intervention to satisfy this"
        if independent else
        " If the source does not attribute a stochastic process to the "
        "intervention, do not invent that causal link; generate the "
        "missing mediated branch"
    )
    return (
        f"{effect.effect_id} is a downstream human outcome whose causal "
        "ancestry never reaches a DIRECT act on the actor or a named "
        "recipient; connect this action's intervention to the "
        "action-mediated process that produces the outcome." + extra
    )


def _condition_content_stems(text: str) -> set[str]:
    return {
        _stem_word(word)
        for word in _match_words(text, stopwords=_CONDITION_STOPWORDS)
    }


def _condition_restates_outcome(description: str, outcome: str) -> bool:
    """True when an if-clause adds no unknown beyond a parent outcome."""
    condition = _condition_content_stems(description)
    parent = _condition_content_stems(outcome)
    if not condition or not parent:
        return False
    if parent <= condition:
        return True
    if condition <= parent and len(condition) >= 2:
        return True
    shared = condition & parent
    union = condition | parent
    distinctive = {word for word in shared if len(word) > 4}
    return bool(distinctive) and len(shared) / len(union) >= 0.6


# Schema predicate tags are not the source verb. Grounders often put IS /
# STATE_CHANGE in outcome or relation and the real verb in the other field.
_SCHEMA_PREDICATE_TAGS = frozenset({
    "ACTS", "DIES", "EXPERIENCES", "IS", "PERFORMS", "STATE_CHANGE",
    "STATE_REMAINS", "SUBJECT_TO", "SURVIVES", "WELFARE_LOSS",
    "WELFARE_PRESERVED",
})
_SCHEMA_PREDICATE_STEMS = frozenset({
    "act", "change", "die", "experienc", "experiences", "is", "loss",
    "perform", "preserv", "remain", "state", "subject", "surviv",
    "survive", "welfare",
})
_IDENTITY_STOPWORDS = frozenset({
    "after", "against", "before", "for", "from", "into", "onto", "over",
    "under",
})
_OUTCOME_PP_TAIL = re.compile(
    r"\b(?:from|of|for|against|onto|into)\s+\S.*$",
    re.IGNORECASE,
)
_BENEFICIAL_IDENTITY_STEMS = frozenset({
    "avert", "escape", "guard", "preserv", "protect", "retain", "safe",
    "save", "spare", "surviv",
})
_ADVERSE_IDENTITY_STEMS = frozenset({
    "block", "choke", "contamin", "contaminat", "destroy", "die", "fail",
    "flood", "foul", "harm", "inund", "injur", "kill", "lose", "ruin",
    "submerg",
})
_PROPOSITION_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
_CHANCE_HEDGE_PREFIX = re.compile(
    r"^\s*(?:there\s+is\s+)?(?:a\s+)?\d+(?:,\d{3})*(?:\.\d+)?\s*"
    r"(?:%|percent)\s*chance\s+(?:that\s+)?",
    re.IGNORECASE,
)


def _identity_head_text(effect: WorldEffect) -> str:
    """Outcome verb, ignoring schema tags and a trailing from/of complement."""
    outcome = _OUTCOME_PP_TAIL.sub("", str(effect.outcome or "")).strip()
    relation = str(effect.relation or "").strip()
    if relation.upper() in _SCHEMA_PREDICATE_TAGS:
        relation = ""
    if outcome.casefold() in {"", "is", "are"}:
        return relation
    return f"{outcome} {relation}".strip()


def _deverbal_identity_stems(stems: set[str]) -> set[str]:
    """Expand protect/protection and contaminate/contamination pairs."""
    expanded = set(_binder_stems(stems))
    for word in list(stems | expanded):
        for suffix in ("tion", "sion", "ment", "ance", "ence"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[: -len(suffix)]
                expanded.add(root)
                expanded.add(_stem_word(root))
                if root.endswith("c"):
                    expanded.add(root + "t")
    return expanded


def _effect_identity_stems(
    effect: WorldEffect,
    party: WorldParty | None = None,
) -> set[str]:
    """Content stems that name this effect: outcome plus non-tag predicate."""
    stems = _condition_content_stems(_identity_head_text(effect))
    stems -= _IDENTITY_STOPWORDS
    stems -= _SCHEMA_PREDICATE_STEMS
    stems -= {_stem_word(token) for token in _SCHEMA_PREDICATE_STEMS}
    stems -= {_stem_word(word) for word in _PARTY_GENERIC_NOUNS}
    if party is not None:
        stems -= _condition_content_stems(party.label)
    return stems


def _condition_identifies_event(
    description: str,
    effect: WorldEffect,
    party: WorldParty | None = None,
) -> bool:
    """True when an event-referenced if-clause names that event's verb.

    Dummy copula outcomes (IS) have no content stems. Identify the event
    from the stated predicate or source proposition instead of requiring
    the description to restates 'IS'.
    """
    description_stems = _condition_content_stems(description)
    event_stems = _effect_identity_stems(effect, party)
    event_stems |= _condition_content_stems(effect.source_proposition)
    event_stems -= _condition_content_stems("chance percent % however")
    if not description_stems or not event_stems:
        return False
    identity = _effect_identity_stems(effect, party)
    if identity and identity <= description_stems:
        return True
    return bool(description_stems & event_stems)


def canonical_probability_event_effects(
    effects: Sequence[WorldEffect],
) -> tuple[WorldEffect, ...]:
    """Return actual non-welfare effects that canonically own uncertainty."""
    return tuple(
        effect for effect in effects
        if effect.directness != "FOREGONE"
        and effect.effect_kind not in _ROLE_WELFARE_KINDS
        and (
            effect.modality in {"PROBABILISTIC", "POSSIBLE", "UNKNOWN"}
            or bool(effect.likelihood_qualifiers)
        )
    )


def compile_event_condition_bindings(
    conditions: Sequence[WorldCondition],
    effects: Sequence[WorldEffect],
) -> tuple[WorldCondition, ...]:
    """Bind an unbound event alias when exactly one branch-local event matches.

    The source proposition and its probability stay on the event effect.  A
    condition is only a gate referencing that event.  Ambiguous or unmatched
    descriptions remain untouched so validation can fail closed rather than
    guessing.
    """
    events = canonical_probability_event_effects(effects)
    compiled: list[WorldCondition] = []
    for condition in conditions:
        if condition.event_effect_id:
            compiled.append(condition)
            continue
        consumer_actions = {
            effect.action_id
            for effect in effects
            if (
                condition.condition_id in effect.condition_ids
                and effect.directness != "FOREGONE"
            )
        }
        matches = [
            event for event in events
            if (
                not consumer_actions or event.action_id in consumer_actions
            )
            and _condition_restates_outcome(
                condition.description, event.outcome,
            )
        ]
        if len(matches) == 1:
            compiled.append(replace(
                condition,
                event_effect_id=matches[0].effect_id,
                provenance=condition.provenance or matches[0].provenance,
            ))
        else:
            compiled.append(condition)
    return tuple(compiled)


def normalize_event_probability_ownership(
    effects: Sequence[WorldEffect],
    conditions: Sequence[WorldCondition],
) -> tuple[WorldEffect, ...]:
    """Keep each event probability on its event, never on gated descendants."""
    effect_by_id = {effect.effect_id: effect for effect in effects}
    condition_by_id = {
        condition.condition_id: condition for condition in conditions
    }
    normalized: list[WorldEffect] = []
    for effect in effects:
        event_ids = {
            condition_by_id[condition_id].event_effect_id
            for condition_id in effect.condition_ids
            if condition_id in condition_by_id
            and condition_by_id[condition_id].event_effect_id
            and condition_by_id[condition_id].event_effect_id != effect.effect_id
        }
        events = [
            effect_by_id[event_id]
            for event_id in event_ids if event_id in effect_by_id
        ]
        if not events:
            normalized.append(effect)
            continue
        event_probability_keys = {
            _likelihood_identity_key(span)
            for event in events for span in event.likelihood_qualifiers
        }
        retained_likelihood = tuple(
            span for span in effect.likelihood_qualifiers
            if _likelihood_identity_key(span) not in event_probability_keys
        )
        modality = effect.modality
        if modality in {"PROBABILISTIC", "POSSIBLE", "UNKNOWN"}:
            modality = "STIPULATED_CONDITIONAL"
        normalized.append(replace(
            effect,
            modality=modality,
            likelihood_qualifiers=retained_likelihood,
        ))
    return tuple(normalized)


def _allocate_prefixed_id(prefix: str, existing: set[str]) -> str:
    nums = [
        int(match.group(1))
        for identifier in existing
        for match in [re.fullmatch(rf"{re.escape(prefix)}(\d+)", identifier)]
        if match
    ]
    next_n = (max(nums) + 1) if nums else 0
    candidate = f"{prefix}{next_n}"
    while candidate in existing:
        next_n += 1
        candidate = f"{prefix}{next_n}"
    return candidate


def _reindex_world_action_effects(model: ScenarioWorldModel) -> ScenarioWorldModel:
    by_action: dict[str, list[str]] = {
        action.action_id: [] for action in model.actions
    }
    for effect in model.effects:
        by_action.setdefault(effect.action_id, []).append(effect.effect_id)
    actions = tuple(
        replace(action, effect_ids=tuple(by_action.get(action.action_id, ())))
        for action in model.actions
    )
    return replace(
        model,
        actions=actions,
        admission=replace(
            model.admission,
            admitted_effect_ids=tuple(
                effect.effect_id for effect in model.effects
            ),
        ),
    )


def _action_direct_effect(
    model: ScenarioWorldModel, action_id: str,
) -> WorldEffect | None:
    for effect in model.effects:
        if effect.action_id == action_id and effect.directness == "DIRECT":
            return effect
    return None


def _event_gate_description(effect: WorldEffect) -> str:
    outcome = str(effect.outcome or "").strip()
    if outcome and outcome.casefold() not in {"is", "are"}:
        return outcome
    proposition = _CHANCE_HEDGE_PREFIX.sub(
        "", str(effect.source_proposition or "").strip(),
    )
    return proposition or outcome


def _independent_event_ids(model: ScenarioWorldModel) -> set[str]:
    party_by_id = {party.party_id: party for party in model.parties}
    action_by_id = {action.action_id: action for action in model.actions}
    found: set[str] = set()
    for effect in model.effects:
        role = _stochastic_process_role(
            effect,
            action_by_id.get(effect.action_id),
            party_by_id.get(effect.party_id),
            party_by_id,
        )
        if role == "INDEPENDENT":
            found.add(effect.effect_id)
    return found


def compile_independent_event_gates(
    model: ScenarioWorldModel,
) -> ScenarioWorldModel:
    """Rewrite independent-event parenting into an event-referenced gate.

    Live grounders often hang the chance event off the intervention and then
    parent the flood/harm chain from that event. The independent-event rule
    still rejects that graph. This compiler detaches the event, reparents
    its children onto the action's DIRECT act, and synthesizes a condition
    whose event_effect_id is the chance event.
    """
    independent_ids = _independent_event_ids(model)
    if not independent_ids:
        return model
    effect_by_id = {effect.effect_id: effect for effect in model.effects}
    conditions = list(model.conditions)
    condition_ids = {condition.condition_id for condition in conditions}
    existing_by_event = {
        condition.event_effect_id: condition.condition_id
        for condition in conditions
        if condition.event_effect_id
    }
    gate_for_event: dict[str, str] = dict(existing_by_event)
    updated_effects = {
        effect.effect_id: effect for effect in model.effects
    }
    kept_links: list[CausalLink] = []
    for link in model.causal_links:
        if not _link_parents_target(link):
            kept_links.append(link)
            continue
        if link.target_id in independent_ids:
            child = updated_effects.get(link.target_id)
            if child is not None and child.source_effect_ids:
                updated_effects[child.effect_id] = replace(
                    child, source_effect_ids=(),
                )
            continue
        if link.source_id not in independent_ids:
            kept_links.append(link)
            continue
        event = updated_effects.get(link.source_id)
        child = updated_effects.get(link.target_id)
        if event is None or child is None:
            continue
        other_parents = [
            other.source_id
            for other in model.causal_links
            if (
                other.target_id == child.effect_id
                and _link_parents_target(other)
                and other.source_id != event.effect_id
                and other.source_id in effect_by_id
            )
        ]
        direct = _action_direct_effect(model, child.action_id)
        parent_id = other_parents[0] if other_parents else (
            direct.effect_id if direct is not None else ""
        )
        if not parent_id:
            continue
        cond_id = gate_for_event.get(event.effect_id)
        if not cond_id:
            cond_id = _allocate_prefixed_id("COND", condition_ids)
            condition_ids.add(cond_id)
            gate_for_event[event.effect_id] = cond_id
            conditions.append(WorldCondition(
                cond_id,
                _event_gate_description(event),
                provenance=event.provenance or child.provenance,
                event_effect_id=event.effect_id,
            ))
        if not other_parents:
            kept_links.append(CausalLink(
                parent_id,
                "CAUSES",
                child.effect_id,
                "CERTAIN",
                (),
                link.provenance or child.provenance,
                child.action_id,
            ))
        gates = tuple(dict.fromkeys((*child.condition_ids, cond_id)))
        modality = child.modality
        if modality in {"CERTAIN", "POSSIBLE", "PROBABILISTIC", "UNKNOWN"}:
            modality = "STIPULATED_CONDITIONAL"
        updated_effects[child.effect_id] = replace(
            child,
            condition_ids=gates,
            modality=modality,
            source_effect_ids=(parent_id,),
        )
    for effect in tuple(updated_effects.values()):
        if effect.effect_id in independent_ids:
            continue
        parents = [
            link.source_id
            for link in kept_links
            if link.target_id == effect.effect_id and _link_parents_target(link)
        ]
        inherited = [
            gate_for_event[parent_id]
            for parent_id in parents
            if parent_id in gate_for_event
        ]
        inherited.extend(
            cond_id
            for parent_id in parents
            for cond_id in updated_effects.get(parent_id, effect).condition_ids
            if parent_id in updated_effects
        )
        if not inherited:
            continue
        gates = tuple(dict.fromkeys((*effect.condition_ids, *inherited)))
        if gates == effect.condition_ids:
            continue
        modality = effect.modality
        if modality == "CERTAIN":
            modality = "STIPULATED_CONDITIONAL"
        updated_effects[effect.effect_id] = replace(
            effect, condition_ids=gates, modality=modality,
        )
    return replace(
        model,
        effects=tuple(updated_effects[effect.effect_id] for effect in model.effects),
        conditions=tuple(conditions),
        causal_links=tuple(kept_links),
    )


def compile_gated_link_certainty(
    model: ScenarioWorldModel,
) -> ScenarioWorldModel:
    """Incoming links are CERTAIN when the target already owns the gate."""
    effect_by_id = {effect.effect_id: effect for effect in model.effects}
    links: list[CausalLink] = []
    for link in model.causal_links:
        target = effect_by_id.get(link.target_id)
        if (
            link.modality != "CERTAIN"
            and not link.condition_ids
            and target is not None
            and target.condition_ids
        ):
            links.append(replace(link, modality="CERTAIN"))
        else:
            links.append(link)
    return replace(model, causal_links=tuple(links))


def compile_missing_parent_links(
    model: ScenarioWorldModel,
) -> ScenarioWorldModel:
    """Materialize same-action source_effect_ids that lack a causal_link."""
    independent_ids = _independent_event_ids(model)
    effect_by_id = {effect.effect_id: effect for effect in model.effects}
    existing = {
        (link.source_id, link.target_id)
        for link in model.causal_links
        if _link_parents_target(link)
    }
    added: list[CausalLink] = []
    for effect in model.effects:
        if effect.directness == "FOREGONE":
            continue
        for parent_id in effect.source_effect_ids:
            parent = effect_by_id.get(parent_id)
            if parent is None or parent.action_id != effect.action_id:
                continue
            if parent_id in independent_ids:
                continue
            if (parent_id, effect.effect_id) in existing:
                continue
            added.append(CausalLink(
                parent_id,
                "CAUSES",
                effect.effect_id,
                "CERTAIN",
                (),
                effect.provenance,
                effect.action_id,
            ))
            existing.add((parent_id, effect.effect_id))
    if not added:
        return model
    return replace(model, causal_links=(*model.causal_links, *added))


def _snap_ellipsis_span(proposition: str, excerpts: Sequence[str]) -> str:
    parts = _PROPOSITION_ELLIPSIS.split(str(proposition or ""), maxsplit=1)
    if len(parts) != 2:
        return ""
    left, right = parts[0].strip(), parts[1].strip()
    if not left or not right:
        return ""
    for excerpt in excerpts:
        folded = excerpt.casefold()
        start = folded.find(left.casefold())
        if start < 0:
            continue
        end = folded.find(right.casefold(), start + len(left))
        if end < 0:
            continue
        return excerpt[start:end + len(right)]
    return ""


def _proposition_is_exact_span(effect: WorldEffect) -> bool:
    proposition = _normalized_proposition_text(effect.source_proposition)
    if not proposition:
        return False
    return any(
        proposition in _normalized_proposition_text(ref.excerpt)
        for ref in effect.provenance if ref.excerpt
    )


def compile_source_proposition_spans(
    model: ScenarioWorldModel,
) -> ScenarioWorldModel:
    """Replace ellipsis paraphrases with the exact provenance span they omit."""
    updated: list[WorldEffect] = []
    changed = False
    for effect in model.effects:
        excerpts = [ref.excerpt for ref in effect.provenance if ref.excerpt]
        if not excerpts or _proposition_is_exact_span(effect):
            updated.append(effect)
            continue
        snapped = _snap_ellipsis_span(effect.source_proposition, excerpts)
        if not snapped:
            updated.append(effect)
            continue
        updated.append(replace(effect, source_proposition=snapped))
        changed = True
    if not changed:
        return model
    return replace(model, effects=tuple(updated))


def compile_grounded_quantities(
    model: ScenarioWorldModel,
) -> ScenarioWorldModel:
    """Drop effect quantities that neither provenance nor the party licenses."""
    party_by_id = {party.party_id: party for party in model.parties}
    updated: list[WorldEffect] = []
    changed = False
    for effect in model.effects:
        party = party_by_id.get(effect.party_id)
        provenance_text = _provenance_text(effect)
        kept = tuple(
            quantity for quantity in effect.quantities
            if (
                _quantity_grounded_in_text(quantity, provenance_text)
                or _party_licenses_quantity(party, quantity)
            )
        )
        if kept != effect.quantities:
            updated.append(replace(effect, quantities=kept))
            changed = True
        else:
            updated.append(effect)
    if not changed:
        return model
    return replace(model, effects=tuple(updated))


def compile_counterfactual_source_bindings(
    model: ScenarioWorldModel,
) -> ScenarioWorldModel:
    """FOREGONE source_effect_ids are the linked alternative actual effect."""
    alternatives = {
        link.source_effect_id: link.alternative_effect_id
        for link in model.counterfactual_links
        if link.source_effect_id and link.alternative_effect_id
    }
    updated: list[WorldEffect] = []
    changed = False
    for effect in model.effects:
        alternative = alternatives.get(effect.effect_id)
        if (
            effect.directness != "FOREGONE"
            or effect.derivation_operation != "COUNTERFACTUAL_PROJECTION"
            or not alternative
        ):
            updated.append(effect)
            continue
        if effect.source_effect_ids == (alternative,):
            updated.append(effect)
            continue
        updated.append(replace(effect, source_effect_ids=(alternative,)))
        changed = True
    if not changed:
        return model
    return replace(model, effects=tuple(updated))


def compile_foregone_overlays(
    model: ScenarioWorldModel,
) -> ScenarioWorldModel:
    """Add the missing FOREGONE duals for opposed stipulated welfare."""
    grouped = _actual_nonrecipient_role_effects(model)
    action_ids = [action.action_id for action in model.actions]
    working = model
    effects = list(model.effects)
    links = list(model.counterfactual_links)
    existing_ids = {effect.effect_id for effect in effects}
    added = False
    for party_id in sorted({party for party, _ in grouped}):
        for index, left_id in enumerate(action_ids):
            for right_id in action_ids[index + 1:]:
                left_effects = grouped.get((party_id, left_id), ())
                right_effects = grouped.get((party_id, right_id), ())
                left_polarities = {item.polarity for item in left_effects}
                right_polarities = {item.polarity for item in right_effects}
                opposed = (
                    "BENEFICIAL" in left_polarities
                    and "ADVERSE" in right_polarities
                ) or (
                    "ADVERSE" in left_polarities
                    and "BENEFICIAL" in right_polarities
                )
                if not opposed:
                    continue
                pairs = (
                    (left_id, right_id, right_effects),
                    (right_id, left_id, left_effects),
                )
                for action_id, alternative_id, alternatives in pairs:
                    if _has_counterfactual_foreclosure(
                        working,
                        action_id=action_id,
                        party_id=party_id,
                        alternative_action_id=alternative_id,
                        alternative_effect_ids={
                            item.effect_id for item in alternatives
                        },
                    ):
                        continue
                    if not alternatives:
                        continue
                    preferred = next(
                        (
                            item for item in alternatives
                            if item.effect_kind in _ROLE_WELFARE_KINDS
                        ),
                        alternatives[0],
                    )
                    overlay_id = _allocate_prefixed_id("F", existing_ids)
                    existing_ids.add(overlay_id)
                    effects.append(WorldEffect(
                        overlay_id,
                        action_id,
                        party_id,
                        preferred.outcome,
                        preferred.relation,
                        "FOREGONE",
                        "FOREGONE",
                        preferred.modality,
                        "OPPORTUNITY_LOSS",
                        condition_ids=preferred.condition_ids,
                        quantities=preferred.quantities,
                        provenance=preferred.provenance,
                        likelihood_qualifiers=preferred.likelihood_qualifiers,
                        scope_qualifiers=preferred.scope_qualifiers,
                        temporal_qualifiers=preferred.temporal_qualifiers,
                        source_proposition=preferred.source_proposition,
                        source_effect_ids=(preferred.effect_id,),
                        derivation_operation="COUNTERFACTUAL_PROJECTION",
                        derivation_explanation=(
                            f"Counterfactual dual of {preferred.effect_id} "
                            f"under {alternative_id}"
                        ),
                    ))
                    links.append(CounterfactualLink(
                        action_id,
                        overlay_id,
                        "FOREGOES_ALTERNATIVE_EFFECT",
                        alternative_id,
                        preferred.effect_id,
                        "CERTAIN",
                        (),
                        preferred.provenance,
                    ))
                    working = replace(
                        working,
                        effects=tuple(effects),
                        counterfactual_links=tuple(links),
                    )
                    added = True
    if not added:
        return model
    return working


def compile_omit_unbound_derived_effects(
    model: ScenarioWorldModel,
) -> ScenarioWorldModel:
    """Omit SOURCE_STIPULATED_CAUSAL rows the cited span does not state."""
    party_by_id = {party.party_id: party for party in model.parties}
    independent_ids = _independent_event_ids(model)
    dropped: set[str] = set()
    changed = True
    while changed:
        changed = False
        for effect in model.effects:
            if effect.effect_id in dropped:
                continue
            if effect.directness in {"DIRECT", "FOREGONE"}:
                continue
            if effect.effect_id in independent_ids:
                continue
            if (
                model.schema_version != "1.3"
                or effect.derivation_operation != "SOURCE_STIPULATED_CAUSAL"
                or not effect.source_proposition
            ):
                continue
            supports = (
                _proposition_is_exact_span(effect)
                and _source_proposition_supports_outcome(
                    effect, party_by_id.get(effect.party_id),
                )
            )
            orphaned = bool(
                effect.source_effect_ids
                and set(effect.source_effect_ids) <= dropped
            )
            if supports and not orphaned:
                continue
            dropped.add(effect.effect_id)
            changed = True
        for link in model.counterfactual_links:
            if (
                link.alternative_effect_id in dropped
                and link.source_effect_id not in dropped
            ):
                dropped.add(link.source_effect_id)
                changed = True
    if not dropped:
        return model
    effects = tuple(
        effect for effect in model.effects if effect.effect_id not in dropped
    )
    links = tuple(
        link for link in model.causal_links
        if link.source_id not in dropped and link.target_id not in dropped
    )
    counterfactuals = tuple(
        link for link in model.counterfactual_links
        if (
            link.source_effect_id not in dropped
            and link.alternative_effect_id not in dropped
        )
    )
    remaining = {effect.effect_id for effect in effects}
    conditions = tuple(
        condition for condition in model.conditions
        if (
            not condition.event_effect_id
            or condition.event_effect_id in remaining
        )
    )
    return replace(
        model,
        effects=effects,
        causal_links=links,
        counterfactual_links=counterfactuals,
        conditions=conditions,
    )


def compile_chance_gated_world(
    model: ScenarioWorldModel,
) -> ScenarioWorldModel:
    """Deterministic repairs that do not invent source facts."""
    compiled = compile_independent_event_gates(model)
    compiled = compile_gated_link_certainty(compiled)
    compiled = compile_missing_parent_links(compiled)
    compiled = compile_source_proposition_spans(compiled)
    compiled = compile_grounded_quantities(compiled)
    compiled = compile_omit_unbound_derived_effects(compiled)
    compiled = compile_counterfactual_source_bindings(compiled)
    compiled = compile_foregone_overlays(compiled)
    compiled = replace(
        compiled,
        conditions=compile_event_condition_bindings(
            compiled.conditions, compiled.effects,
        ),
    )
    compiled = replace(
        compiled,
        effects=normalize_event_probability_ownership(
            compiled.effects, compiled.conditions,
        ),
        causal_links=normalize_redundant_link_gates(
            compiled.causal_links, compiled.effects,
        ),
    )
    return _reindex_world_action_effects(compiled)


_WORLD_DERIVATION_OPERATIONS = {
    "DIRECT_COPY", "SOURCE_STIPULATED_CAUSAL", "STRUCTURAL_ABSTRACTION",
    "COUNTERFACTUAL_PROJECTION",
}


def _normalized_proposition_text(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _source_proposition_supports_outcome(
    effect: WorldEffect,
    party: WorldParty | None = None,
) -> bool:
    outcome_stems = _deverbal_identity_stems(_effect_identity_stems(effect, party))
    proposition_stems = _deverbal_identity_stems(
        _condition_content_stems(effect.source_proposition)
    )
    if outcome_stems and proposition_stems:
        shared = outcome_stems & proposition_stems
        if shared and len(shared) / len(outcome_stems) >= 0.5:
            return True
    if party is None or not proposition_stems:
        return False
    party_stems = _condition_content_stems(party.label)
    if not (party_stems & proposition_stems):
        return False
    polarity = str(effect.polarity or "").upper()
    verbs = proposition_stems | outcome_stems
    if polarity in {"BENEFICIAL", "FOREGONE"} and verbs & _BENEFICIAL_IDENTITY_STEMS:
        return True
    if polarity in {"ADVERSE", "FOREGONE"} and verbs & _ADVERSE_IDENTITY_STEMS:
        return True
    return False


def validate_effect_source_bindings(model: ScenarioWorldModel) -> list[str]:
    """Validate schema-1.3 effect-level proposition and derivation bindings."""
    effect_by_id = {effect.effect_id: effect for effect in model.effects}
    party_by_id = {party.party_id: party for party in model.parties}
    action_by_id = {action.action_id: action for action in model.actions}
    errors: list[str] = []
    for effect in model.effects:
        prefix = effect.effect_id
        proposition = _normalized_proposition_text(effect.source_proposition)
        source_texts = [
            _normalized_proposition_text(ref.excerpt)
            for ref in effect.provenance if ref.excerpt
        ]
        if not proposition:
            errors.append(f"{prefix} lacks a bound source_proposition")
        elif not any(proposition in source for source in source_texts):
            errors.append(
                f"{prefix} source_proposition is not an exact span of its provenance"
            )
        elif (
            effect.derivation_operation != "STRUCTURAL_ABSTRACTION"
            and not _source_proposition_supports_outcome(
                effect, party_by_id.get(effect.party_id),
            )
        ):
            errors.append(
                f"{prefix} source_proposition does not state its normalized outcome; "
                "bind the explicit source proposition or omit the derived world effect"
            )
        operation = effect.derivation_operation
        if operation not in _WORLD_DERIVATION_OPERATIONS:
            errors.append(
                f"{prefix} has inadmissible derivation_operation {operation!r}"
            )
        if effect.derivation_assumptions:
            errors.append(
                f"{prefix} depends on unestablished derivation assumptions; "
                "quarantine it as a hypothesis instead of a world effect"
            )
        if effect.outcome_type_transformation != "PRESERVED":
            errors.append(
                f"{prefix} changes source outcome type to "
                f"{effect.outcome_type_transformation}; quarantine the transformation"
            )
        unknown_sources = set(effect.source_effect_ids) - set(effect_by_id)
        if unknown_sources:
            errors.append(
                f"{prefix} derives from unknown source effects: {sorted(unknown_sources)}"
            )
        if operation == "DIRECT_COPY" and effect.source_effect_ids:
            errors.append(
                f"{prefix} is DIRECT_COPY but lists source_effect_ids"
            )
        if operation == "SOURCE_STIPULATED_CAUSAL":
            independent_event = (
                _stochastic_process_role(
                    effect,
                    action_by_id.get(effect.action_id),
                    party_by_id.get(effect.party_id),
                    party_by_id,
                ) == "INDEPENDENT"
            )
            if not effect.source_effect_ids and not independent_event:
                errors.append(
                    f"{prefix} is SOURCE_STIPULATED_CAUSAL but lists no source_effect_ids"
                )
            if not effect.derivation_explanation:
                errors.append(
                    f"{prefix} is SOURCE_STIPULATED_CAUSAL but lacks a derivation explanation"
                )
            immediate_parents = {
                link.source_id for link in model.causal_links
                if link.action_id == effect.action_id
                and link.target_id == effect.effect_id
                and _link_parents_target(link)
            }
            invalid_parents = set(effect.source_effect_ids) - immediate_parents
            if invalid_parents:
                errors.append(
                    f"{prefix} source_effect_ids are not same-action immediate parents: "
                    f"{sorted(invalid_parents)}"
                )
        if operation == "STRUCTURAL_ABSTRACTION":
            proposition_stems = _condition_content_stems(
                effect.source_proposition,
            )
            outcome_stems = _condition_content_stems(effect.outcome)
            immediate_parents = {
                link.source_id for link in model.causal_links
                if link.action_id == effect.action_id
                and link.target_id == effect.effect_id
                and _link_parents_target(link)
            }
            if (
                effect.effect_kind not in _PROCESS_EFFECT_KINDS
                or effect.polarity != "NEUTRAL"
                or effect.directness == "FOREGONE"
            ):
                errors.append(
                    f"{prefix} STRUCTURAL_ABSTRACTION must be an actual NEUTRAL "
                    "process or physical-state effect"
                )
            if not outcome_stems & proposition_stems:
                errors.append(
                    f"{prefix} STRUCTURAL_ABSTRACTION has no lexical anchor in "
                    "its source proposition"
                )
            if not effect.source_effect_ids:
                errors.append(
                    f"{prefix} STRUCTURAL_ABSTRACTION lists no source_effect_ids"
                )
            invalid_parents = set(effect.source_effect_ids) - immediate_parents
            if invalid_parents:
                errors.append(
                    f"{prefix} structural source_effect_ids are not same-action "
                    f"immediate parents: {sorted(invalid_parents)}"
                )
        if operation == "COUNTERFACTUAL_PROJECTION":
            matching = {
                link.alternative_effect_id
                for link in model.counterfactual_links
                if link.source_effect_id == effect.effect_id
                and link.action_id == effect.action_id
            }
            if effect.directness != "FOREGONE":
                errors.append(
                    f"{prefix} uses COUNTERFACTUAL_PROJECTION but is not FOREGONE"
                )
            if not effect.source_effect_ids or not set(effect.source_effect_ids) <= matching:
                errors.append(
                    f"{prefix} counterfactual source_effect_ids do not match its "
                    "alternative_effect_id"
                )
        elif effect.directness == "FOREGONE":
            errors.append(
                f"{prefix} is FOREGONE so derivation_operation must be "
                "COUNTERFACTUAL_PROJECTION"
            )
    return errors


def _immediate_causal_parents(
    model: ScenarioWorldModel, effect_id: str,
) -> tuple[WorldEffect, ...]:
    by_id = {effect.effect_id: effect for effect in model.effects}
    return tuple(
        by_id[link.source_id]
        for link in model.causal_links
        if (
            link.target_id == effect_id
            and link.source_id in by_id
            and _link_parents_target(link)
        )
    )


def _supporting_source_is_hedged(effect: WorldEffect) -> bool:
    """True when a hedge modifies this outcome, not a sibling in the same clause.

    Occupying an at-risk situation is not a chance hedge. 'at high risk of
    dying' is, because the complement names a distinct harm event.
    """
    chance_spans = [
        span for span in effect.likelihood_qualifiers
        if not (
            _is_risk_situation_span(span)
            and not _risk_situation_licenses_chance(effect, span)
        )
    ]
    if chance_spans:
        return True
    expected = effect_expected_qualifiers(effect, explicit_likelihood_spans)
    if any(
        not (
            _is_risk_situation_span(span)
            and not _risk_situation_licenses_chance(effect, span)
        )
        for span in expected
    ):
        return True
    for ref in effect.provenance:
        text = ref.excerpt or ""
        for match in _SOURCE_HEDGE_CUE.finditer(text):
            if _qualifier_binds_to_effect(effect, text, match.start(), match.end()):
                return True
    return False


def _independent_condition_errors(model: ScenarioWorldModel) -> list[str]:
    """Reject tautological ifs and unhedged POSSIBLE rows.

    A condition may not restate an immediate causal parent. POSSIBLE/UNKNOWN
    rows whose supporting source is an unhedged indicative must be typed
    CERTAIN, or must copy a source if/unless/likelihood hedge. STIPULATED
    conditionals such as 'if the attempt works' after CERTAIN 'attempt made'
    remain legal.
    """
    conditions = {
        condition.condition_id: condition for condition in model.conditions
    }
    errors: list[str] = []
    flagged: set[str] = set()
    for effect in model.effects:
        if effect.modality == "CERTAIN":
            continue
        parents = _immediate_causal_parents(model, effect.effect_id)
        for cond_id in effect.condition_ids:
            condition = conditions.get(cond_id)
            if condition is None:
                continue
            for parent in parents:
                if not _condition_restates_outcome(
                    condition.description, parent.outcome,
                ):
                    continue
                errors.append(
                    f"{effect.effect_id} condition {cond_id} restates immediate "
                    f"parent {parent.effect_id} ({parent.outcome!r}); drop the "
                    "condition and type CERTAIN, or keep a non-certain modality "
                    "only with a source hedge that is not this parent restated"
                )
                flagged.add(effect.effect_id)
                break
        if (
            effect.effect_id not in flagged
            and effect.modality in {"POSSIBLE", "UNKNOWN"}
            and not _supporting_source_is_hedged(effect)
        ):
            errors.append(
                f"{effect.effect_id} is {effect.modality} but its supporting "
                "source is an unhedged indicative; type CERTAIN, or copy a "
                "source if/unless/likelihood hedge onto an independent condition"
            )
            flagged.add(effect.effect_id)
    for index, link in enumerate(model.causal_links):
        if link.modality == "CERTAIN" or not link.condition_ids:
            continue
        parents = [
            effect for effect in model.effects if effect.effect_id == link.source_id
        ]
        if not parents:
            continue
        parent = parents[0]
        for cond_id in link.condition_ids:
            condition = conditions.get(cond_id)
            if condition is None:
                continue
            if _condition_restates_outcome(condition.description, parent.outcome):
                errors.append(
                    f"causal_link[{index}] condition {cond_id} restates source "
                    f"{parent.effect_id} ({parent.outcome!r}); make the link "
                    "CERTAIN or use an independent condition"
                )
                break
    return errors


def _event_referenced_condition_errors(model: ScenarioWorldModel) -> list[str]:
    """Independent events gate via condition.event_effect_id, not a second parent.

    Probability stays on the referenced event. The action-mediated path still
    reaches the outcome. A free-text condition may not stand in for an existing
    event. Event-gated human outcomes are STIPULATED_CONDITIONAL.
    """
    effect_by_id = {effect.effect_id: effect for effect in model.effects}
    party_by_id = {party.party_id: party for party in model.parties}
    action_by_id = {action.action_id: action for action in model.actions}
    errors: list[str] = []
    for condition in model.conditions:
        # Conditions are branch-local.  A lexical match in another action (and
        # especially in a derived FOREGONE mirror) is not an alternative event
        # identity for this condition.  Without this scope, counterfactual
        # projections can make an otherwise valid actual-world event look
        # ambiguously referenced.
        condition_action_ids = {
            effect.action_id
            for effect in model.effects
            if (
                condition.condition_id in effect.condition_ids
                and effect.directness != "FOREGONE"
            )
        }
        condition_action_ids.update(
            link.action_id
            for link in model.causal_links
            if condition.condition_id in link.condition_ids and link.action_id
        )
        event_id = str(condition.event_effect_id or "").strip()
        if not event_id:
            for effect in model.effects:
                if effect.directness == "FOREGONE":
                    continue
                if condition_action_ids and effect.action_id not in condition_action_ids:
                    continue
                action = action_by_id.get(effect.action_id)
                if action is None:
                    continue
                role = _stochastic_process_role(
                    effect, action, party_by_id.get(effect.party_id), party_by_id,
                )
                if role != "INDEPENDENT":
                    continue
                if not _condition_identifies_event(
                    condition.description, effect, party_by_id.get(effect.party_id),
                ):
                    continue
                errors.append(
                    f"{condition.condition_id} restates existing event "
                    f"{effect.effect_id} ({effect.outcome!r}) as free text; "
                    f"set event_effect_id to {effect.effect_id} instead of a "
                    "disconnected condition description"
                )
            continue
        event = effect_by_id.get(event_id)
        if event is None:
            errors.append(
                f"{condition.condition_id} event_effect_id {event_id} is unknown"
            )
            continue
        event_party = party_by_id.get(event.party_id)
        if not _condition_identifies_event(
            condition.description, event, event_party,
        ):
            errors.append(
                f"{condition.condition_id} description does not identify "
                f"referenced event {event_id} ({event.outcome!r})"
            )
        if event.directness == "FOREGONE" or event.effect_kind == "OPPORTUNITY_LOSS":
            errors.append(
                f"{condition.condition_id} references FOREGONE overlay {event_id}; "
                "event_effect_id must point at an actual probabilistic event"
            )
            continue
        if event.effect_kind in _ROLE_WELFARE_KINDS or _party_bears_welfare(event_party):
            errors.append(
                f"{condition.condition_id} references human welfare outcome "
                f"{event_id}; point at the independent process event, not the "
                "gated harm or benefit"
            )
            continue
        event_modality = str(event.modality or "").upper()
        if event_modality == "CERTAIN":
            errors.append(
                f"{condition.condition_id} references {event_id} which is CERTAIN; "
                "event_effect_id must point at a probabilistic event and keep "
                "that event's chance hedge"
            )
            continue
        if (
            not event.likelihood_qualifiers
            and event_modality not in {"PROBABILISTIC", "POSSIBLE"}
        ):
            errors.append(
                f"{condition.condition_id} references {event_id} but that event "
                "has no retained probability; keep the chance hedge on the "
                "referenced event"
            )
        if not event.provenance:
            errors.append(
                f"{condition.condition_id} references {event_id} but that event "
                "has no source provenance; keep the cited clauses on the event"
            )
        for effect in model.effects:
            if effect.effect_id == event_id:
                continue
            if effect.directness == "FOREGONE":
                continue
            if effect.action_id != event.action_id:
                continue
            if _condition_restates_outcome(condition.description, effect.outcome):
                errors.append(
                    f"{condition.condition_id} description restates "
                    f"{effect.effect_id} but event_effect_id is {event_id}"
                )
        cited = any(
            condition.condition_id in effect.condition_ids
            for effect in model.effects
        ) or any(
            condition.condition_id in link.condition_ids
            for link in model.causal_links
        )
        if not cited:
            errors.append(
                f"{condition.condition_id} references {event_id} but no effect "
                "or link cites it as a gate"
            )
        children = [
            link.target_id
            for link in model.causal_links
            if link.source_id == event_id and _link_parents_target(link)
        ]
        if children:
            errors.append(
                f"{event_id} is referenced by {condition.condition_id} and must "
                "not also be an ordinary causal parent of "
                f"{sorted(set(children))}; gate the action-mediated path with "
                "this condition"
            )
    for link in model.causal_links:
        if not _link_parents_target(link):
            continue
        source = effect_by_id.get(link.source_id)
        target = effect_by_id.get(link.target_id)
        if source is None or target is None:
            continue
        action = action_by_id.get(source.action_id)
        if action is None:
            continue
        role = _stochastic_process_role(
            source, action, party_by_id.get(source.party_id), party_by_id,
        )
        if role != "INDEPENDENT":
            continue
        errors.append(
            f"{source.effect_id} is an independent stochastic event; do not "
            f"parent {target.effect_id} from it. Gate the action-mediated path "
            "with a condition whose event_effect_id is "
            f"{source.effect_id}"
        )
    for effect in model.effects:
        if not _effect_gated_by_referenced_event(effect, model):
            continue
        if str(effect.modality or "").upper() == "STIPULATED_CONDITIONAL":
            continue
        if not (
            effect.effect_kind in _ROLE_WELFARE_KINDS
            or (
                effect.effect_kind == "PHYSICAL_STATE"
                and _party_bears_welfare(party_by_id.get(effect.party_id))
            )
        ):
            continue
        errors.append(
            f"{effect.effect_id} is gated by an event-referenced condition so "
            "its modality must be STIPULATED_CONDITIONAL; keep the probability "
            "on the referenced event"
        )
    pointed = {
        condition.condition_id: condition
        for condition in model.conditions
        if str(condition.event_effect_id or "").strip()
    }
    for effect in model.effects:
        join = str(effect.condition_join or "").upper()
        if join and join not in {"AND", "OR"}:
            errors.append(
                f"{effect.effect_id} condition_join must be AND or OR, not {join!r}"
            )
        if len(effect.condition_ids) > 1 and join not in {"AND", "OR"}:
            errors.append(
                f"{effect.effect_id} is gated by multiple conditions so "
                "condition_join must be AND or OR"
            )
        gate_events = [
            effect_by_id[pointed[cond_id].event_effect_id]
            for cond_id in effect.condition_ids
            if cond_id in pointed
            and pointed[cond_id].event_effect_id in effect_by_id
        ]
        if not gate_events:
            continue
        if str(effect.modality or "").upper() == "CERTAIN":
            errors.append(
                f"{effect.effect_id} is gated by an event-referenced condition "
                "so it must not be CERTAIN; keep the event probability on the "
                "referenced event and compose it once"
            )
        event_keys = {
            _likelihood_identity_key(span)
            for event in gate_events
            for span in event.likelihood_qualifiers
        }
        copied = [
            span for span in effect.likelihood_qualifiers
            if _likelihood_identity_key(span) in event_keys
        ]
        if copied:
            errors.append(
                f"{effect.effect_id} copies the gate probability {copied} from "
                "the referenced event; keep that chance on the event and do not "
                "multiply it onto the gated outcome"
            )
    return errors


def _does_not_increase_errors(model: ScenarioWorldModel) -> list[str]:
    """DOES_NOT_INCREASE names this action, the risk event, and the baseline.

    The baseline is the retained chance on the independent event. The link
    does not mean the risk cannot occur.
    """
    effect_by_id = {effect.effect_id: effect for effect in model.effects}
    party_by_id = {party.party_id: party for party in model.parties}
    action_by_id = {action.action_id: action for action in model.actions}
    errors: list[str] = []
    for index, link in enumerate(model.causal_links):
        if str(link.relation or "").upper() != "DOES_NOT_INCREASE":
            continue
        source = effect_by_id.get(link.source_id)
        target = effect_by_id.get(link.target_id)
        action = action_by_id.get(link.action_id or (source.action_id if source else ""))
        prefix = f"causal_link[{index}] DOES_NOT_INCREASE"
        if source is None or target is None or action is None:
            errors.append(
                f"{prefix} must identify this action's DIRECT act and the "
                "affected risk event"
            )
            continue
        allowed = {action.actor_party_id, *action.recipient_party_ids}
        if source.directness != "DIRECT" or source.party_id not in allowed:
            errors.append(
                f"{prefix} must start from {action.action_id}'s DIRECT act, "
                f"not {source.effect_id}"
            )
        target_party = party_by_id.get(target.party_id)
        if target.directness == "FOREGONE" or target.effect_kind == "OPPORTUNITY_LOSS":
            errors.append(
                f"{prefix} target {target.effect_id} is a FOREGONE overlay, "
                "not the affected risk event"
            )
            continue
        if target.effect_kind in _ROLE_WELFARE_KINDS or (
            _party_bears_welfare(target_party)
            and not (
                target_party is not None
                and target_party.kind in _INTERMEDIATE_BEARER_KINDS
            )
        ):
            errors.append(
                f"{prefix} target {target.effect_id} must be the affected "
                "risk event, not a human welfare outcome"
            )
        if str(target.modality or "").upper() == "CERTAIN" and not target.likelihood_qualifiers:
            errors.append(
                f"{prefix} must not imply that {target.effect_id} cannot occur; "
                "keep the baseline chance on the risk event"
            )
        if not target.likelihood_qualifiers:
            errors.append(
                f"{prefix} needs the baseline comparison: keep the source chance "
                f"hedge on risk event {target.effect_id}"
            )
        role = _stochastic_process_role(
            target, action, target_party, party_by_id,
        )
        if role == "ACTION_CAUSED":
            errors.append(
                f"{prefix} target {target.effect_id} is action-caused, not an "
                "independent background risk"
            )
    return errors


def _crowd_process_polarity_errors(model: ScenarioWorldModel) -> list[str]:
    """Use/remain process states are NEUTRAL; harm lives on later welfare rows."""
    party_by_id = {party.party_id: party for party in model.parties}
    errors: list[str] = []
    for effect in model.effects:
        party = party_by_id.get(effect.party_id)
        if not _crowd_mediated_process(effect, party):
            continue
        if effect.polarity in {"NEUTRAL", "FOREGONE"}:
            continue
        errors.append(
            f"{effect.effect_id} is a crowd process state so polarity must be "
            "NEUTRAL; put ADVERSE or BENEFICIAL on trapping, exposure, injury, "
            "or death"
        )
    return errors


def _action_cites_direct_effect_clauses(model: ScenarioWorldModel) -> list[str]:
    """Each action must cite FACT clauses of its own DIRECT effects."""
    errors: list[str] = []
    for action in model.actions:
        cited = {ref.clause_id for ref in action.provenance}
        missing: list[str] = []
        for effect in model.effects:
            if effect.action_id != action.action_id or effect.directness != "DIRECT":
                continue
            for ref in effect.provenance:
                if not is_supporting_source(ref):
                    continue
                if _ACTION_SOURCE_ID.match(ref.clause_id or ""):
                    continue
                if ref.clause_id not in cited and ref.clause_id not in missing:
                    missing.append(ref.clause_id)
        if missing:
            errors.append(
                f"{action.action_id} omits source clauses of its DIRECT effects: "
                f"{missing}; cite every FACT clause that states an owned "
                "assignment or transfer, including a named subgroup"
            )
    return errors


def _human_transfer_kind_errors(model: ScenarioWorldModel) -> list[str]:
    """A sent resource received by a crowd is RESOURCE_TRANSFER, not INTERVENTION."""
    party_by_id = {party.party_id: party for party in model.parties}
    resource_labels = [
        party.label.casefold()
        for party in model.parties
        if party.kind == "RESOURCE" and party.label.strip()
    ]
    if not resource_labels:
        return []
    errors: list[str] = []
    for action in model.actions:
        owned = [
            effect for effect in model.effects
            if effect.action_id == action.action_id
        ]
        blob = " ".join([
            action.intervention,
            *(effect.outcome for effect in owned if effect.directness == "DIRECT"),
        ])
        if not _RESOURCE_OBJECT_TRANSFER_CUE.search(blob):
            continue
        if not any(label and label in blob.casefold() for label in resource_labels):
            continue
        for effect in owned:
            if effect.directness != "DIRECT":
                continue
            if effect.party_id not in action.recipient_party_ids:
                continue
            if effect.effect_kind != "INTERVENTION":
                continue
            if not _party_bears_welfare(party_by_id.get(effect.party_id)):
                continue
            errors.append(
                f"{effect.effect_id} is the receipt of a transferred resource so "
                "effect_kind must be RESOURCE_TRANSFER, not INTERVENTION"
            )
    return errors


def _transfer_resource_recipient_errors(model: ScenarioWorldModel) -> list[str]:
    """A sent or dispatched RESOURCE is the transferred object, not a recipient."""
    party_by_id = {party.party_id: party for party in model.parties}
    errors: list[str] = []
    for action in model.actions:
        owned = [
            effect for effect in model.effects
            if effect.action_id == action.action_id
        ]
        for recipient_id in action.recipient_party_ids:
            party = party_by_id.get(recipient_id)
            if party is None or party.kind != "RESOURCE":
                continue
            blob = " ".join([
                action.intervention,
                *(
                    effect.outcome
                    for effect in owned
                    if effect.party_id == recipient_id
                    and effect.directness == "DIRECT"
                ),
            ])
            if not _RESOURCE_TRANSFER_CUE.search(blob):
                continue
            receivers = [
                effect.party_id
                for effect in owned
                if (
                    effect.party_id not in action.recipient_party_ids
                    and effect.directness == "DOWNSTREAM"
                    and effect.polarity == "BENEFICIAL"
                    and _party_bears_welfare(party_by_id.get(effect.party_id))
                )
            ]
            named = f"; {receivers[0]} is the receiving group" if receivers else ""
            errors.append(
                f"{action.action_id} names transferred resource {recipient_id} "
                "as a recipient; the person or crowd who receives that resource "
                "should be the recipient, with a DIRECT RESOURCE_TRANSFER. Keep "
                f"the resource as a party, not a recipient{named}"
            )
    return errors


def validate_world_completeness(
    model: ScenarioWorldModel,
    *,
    action_ids: Sequence[str],
) -> list[str]:
    """Require source-grounded causal stages, not only terminal physical outcomes.

    Structural identity still lives in ``validate_world_model``. These checks
    reject models that skip a named patient, a named false-attribution, or the
    intermediate process that stands between an actor's intervention and a
    different party's downstream health outcome. They also reject a DIRECT
    effect on a party the action neither is nor targets, a downstream
    welfare outcome whose causal ancestry never reaches that DIRECT act, and
    a mutually exclusive non-recipient welfare swap with no FOREGONE
    counterfactual overlay. Schema 1.1+ also rejects a condition that restates
    an immediate causal parent, POSSIBLE/UNKNOWN rows whose source is an
    unhedged indicative, and a transferred RESOURCE named as a recipient.
    """
    del action_ids
    errors: list[str] = []
    party_by_id = {party.party_id: party for party in model.parties}
    for party in model.parties:
        nested = _nested_recorded_quantities(party.quantities)
        if nested:
            errors.append(
                f"{party.party_id} records nested quantity spans {list(nested)}; "
                "keep only the longest source phrase"
            )
    for effect in model.effects:
        nested = _nested_recorded_quantities(effect.quantities)
        if nested:
            errors.append(
                f"{effect.effect_id} records nested quantity spans {list(nested)}; "
                "keep only the longest source phrase"
            )
    for action in model.actions:
        source = _action_source_text(action)
        owned = [effect for effect in model.effects if effect.action_id == action.action_id]
        allowed_direct_parties = {action.actor_party_id, *action.recipient_party_ids}
        typed_topology = model.schema_version != "1.0"
        if _source_states_false_attribution(source):
            refused = bool(_REFUSED_HARM.search(source))
            wanted_polarities = {"BENEFICIAL", "NEUTRAL"} if refused else {"ADVERSE"}
            has_juridical = any(
                effect.effect_kind == "INSTITUTIONAL_OUTCOME"
                and effect.directness == "DIRECT"
                and effect.polarity in wanted_polarities
                and party_by_id.get(effect.party_id) is not None
                and party_by_id[effect.party_id].kind in _ROLE_PERSON_KINDS
                for effect in owned
            )
            if not has_juridical:
                if refused:
                    errors.append(
                        f"{action.action_id} source refuses framing or false "
                        "attribution but has no DIRECT BENEFICIAL or NEUTRAL "
                        "INSTITUTIONAL_OUTCOME on a human party"
                    )
                else:
                    errors.append(
                        f"{action.action_id} source states framing or false "
                        "attribution but has no DIRECT ADVERSE "
                        "INSTITUTIONAL_OUTCOME on a human party"
                    )
        if _REFUSED_HARM.search(source):
            for party in _mentioned_non_actor_humans(action, model.parties):
                has_patient = any(
                    effect.party_id == party.party_id
                    and effect.directness != "FOREGONE"
                    and effect.polarity in {"BENEFICIAL", "NEUTRAL"}
                    for effect in owned
                )
                if not has_patient:
                    errors.append(
                        f"{action.action_id} omits a patient-status effect on "
                        f"{party.party_id} ({party.label}); a refused framing or "
                        "killing still changes that party's state"
                    )
                if party.party_id not in action.recipient_party_ids:
                    errors.append(
                        f"{action.action_id} recipient list omits {party.party_id}, "
                        "the patient of the refused intervention"
                    )
        for effect in owned:
            if (
                typed_topology
                and effect.directness not in {"DOWNSTREAM", "FOREGONE"}
                and effect.party_id not in allowed_direct_parties
            ):
                errors.append(
                    f"{effect.effect_id} is {effect.directness} on "
                    f"{effect.party_id}, who is neither the actor nor a named "
                    "recipient; non-recipient process and population states "
                    "are DOWNSTREAM or FOREGONE. If this row is later conduct "
                    "(assist, remain, use a road), recast it DOWNSTREAM with a "
                    "non-INTERVENTION kind; do not add that party as a recipient "
                    "to make the conduct DIRECT. If this row is the actor "
                    "assigning or allocating that party, they may be a recipient "
                    "of a separate DIRECT assignment, not of this later conduct"
                )
            if not _requires_act_ancestry(effect, party_by_id.get(effect.party_id)):
                continue
            parents = _downstream_health_parents(effect, model)
            if not parents:
                errors.append(
                    f"{effect.effect_id} is a downstream human outcome with no "
                    "causal parent; connect it through the source-named process. "
                    f"Do not drop {effect.effect_id}; insert or reuse a PROCESS, "
                    "FACILITY, INSTITUTION, INFRASTRUCTURE, or RESOURCE state "
                    "as its parent"
                )
                continue
            skipped_parents = [
                parent for parent in parents
                if (
                    _parent_is_foreign_act_or_body(parent, effect, party_by_id)
                    if typed_topology else (
                        parent.party_id != effect.party_id
                        and parent.effect_kind == "INTERVENTION"
                    )
                )
            ]
            if skipped_parents and len(skipped_parents) == len(parents):
                errors.append(
                    _foreign_parent_error(
                        effect, skipped_parents, action, model, party_by_id,
                    )
                )
            if typed_topology and not _human_outcome_reaches_direct_act(
                effect, action, model, party_by_id,
            ):
                errors.append(
                    _human_outcome_ancestry_error(
                        effect, action, model, party_by_id, parents,
                    )
                )
        if typed_topology:
            for owned_effect in owned:
                bearer = party_by_id.get(owned_effect.party_id)
                role = _stochastic_process_role(
                    owned_effect, action, bearer, party_by_id,
                )
                if role is None:
                    continue
                has_direct_parent = _has_direct_intervention_parent(
                    owned_effect, action, model,
                )
                reaches = _ancestry_reaches_direct_act(
                    owned_effect, action, model,
                )
                if role == "ACTION_CAUSED" and not reaches:
                    errors.append(
                        f"{owned_effect.effect_id} is an action-caused "
                        "stochastic process with no inbound path from the "
                        "DIRECT intervention"
                    )
                if role == "INDEPENDENT" and has_direct_parent:
                    errors.append(
                        f"{owned_effect.effect_id} is an independent "
                        "background event; Do not parent it from the "
                        "intervention"
                    )
                if role == "AMBIGUOUS" and has_direct_parent:
                    errors.append(
                        f"{owned_effect.effect_id} is not source-attributed "
                        "to the intervention; do not invent that causal "
                        "link. Generate the missing action-mediated branch"
                    )
    if model.schema_version != "1.0":
        errors.extend(_foreclosure_completeness_errors(model))
        errors.extend(_independent_condition_errors(model))
        errors.extend(_event_referenced_condition_errors(model))
        errors.extend(_does_not_increase_errors(model))
        errors.extend(_overall_conditional_likelihood_errors(model))
        errors.extend(_crowd_process_polarity_errors(model))
        errors.extend(_action_cites_direct_effect_clauses(model))
        errors.extend(_human_transfer_kind_errors(model))
        errors.extend(_transfer_resource_recipient_errors(model))
    return list(dict.fromkeys(errors))


def quarantine_contradictions(
    model: ScenarioWorldModel,
    contradictions: Sequence[Sequence[str]],
    *,
    user_override: bool,
) -> ScenarioWorldModel:
    conflict_ids = {effect_id for group in contradictions for effect_id in group}
    by_id = {effect.effect_id: effect for effect in model.effects}
    quarantined: list[QuarantinedEffect] = []
    for group in contradictions:
        group_ids = tuple(dict.fromkeys(str(value) for value in group))
        for effect_id in group_ids:
            effect = by_id.get(effect_id)
            if effect is None:
                continue
            quarantined.append(QuarantinedEffect(
                effect_id=effect.effect_id,
                action_id=effect.action_id,
                party_id=effect.party_id,
                contradiction_type="CONTRADICTORY_DIRECT_EFFECTS",
                conflicting_effect_ids=tuple(value for value in group_ids if value != effect_id),
                source_clause_ids=tuple(ref.clause_id for ref in effect.provenance),
            ))
    admission = WorldStateAdmission(
        status=(
            "USER_ACCEPTED_WITH_QUARANTINE"
            if user_override else "ABANDONED_CONTRADICTORY_WORLD_STATE"
        ),
        admitted_effect_ids=tuple(
            effect.effect_id for effect in model.effects if effect.effect_id not in conflict_ids
        ) if user_override else (),
        quarantined_effects=tuple(quarantined),
        user_override=user_override,
        override_reason=(
            "user elected to continue without unresolved contradictory direct effects"
            if user_override else "user abandoned run after unresolved direct-effect contradictions"
        ),
    )
    return ScenarioWorldModel(
        parties=model.parties, actions=model.actions, effects=model.effects,
        conditions=model.conditions, causal_links=model.causal_links,
        counterfactual_links=model.counterfactual_links,
        admission=admission, schema_version=model.schema_version,
    )


def world_model_from_dict(data: Any) -> ScenarioWorldModel | None:
    """Restore a serialized, previously validated world model."""
    if not isinstance(data, dict) or not data.get("parties"):
        return None
    clauses: dict[str, str] = {}
    for collection in (
        data.get("parties", []), data.get("actions", []), data.get("effects", []),
        data.get("conditions", []), data.get("causal_links", []),
        data.get("counterfactual_links", []),
    ):
        for row in collection if isinstance(collection, (list, tuple)) else []:
            for ref in row.get("provenance", []) if isinstance(row, dict) else []:
                if isinstance(ref, dict) and ref.get("clause_id"):
                    clauses[str(ref["clause_id"])] = str(ref.get("excerpt", ""))
    compact = {
        "parties": [{**row, "clause_ids": [r.get("clause_id") for r in row.get("provenance", [])]} for row in data.get("parties", [])],
        "actions": [{**row, "clause_ids": [r.get("clause_id") for r in row.get("provenance", [])]} for row in data.get("actions", [])],
        "effects": [{**row, "clause_ids": [r.get("clause_id") for r in row.get("provenance", [])]} for row in data.get("effects", [])],
        "conditions": [{**row, "clause_ids": [r.get("clause_id") for r in row.get("provenance", [])]} for row in data.get("conditions", [])],
        "causal_links": [{**row, "clause_ids": [r.get("clause_id") for r in row.get("provenance", [])]} for row in data.get("causal_links", [])],
        "counterfactual_links": [{**row, "clause_ids": [r.get("clause_id") for r in row.get("provenance", [])]} for row in data.get("counterfactual_links", [])],
        "schema_version": str(data.get("schema_version", "1.0")),
    }
    action_ids = [str(row.get("action_id", "")) for row in data.get("actions", [])]
    model = parse_world_model(
        compact,
        clauses=[{"clause_id": key, "text": value} for key, value in clauses.items()],
        action_ids=action_ids,
        require_completeness=False,
    )
    admission_raw = data.get("admission", {})
    quarantined = tuple(QuarantinedEffect(**row) for row in admission_raw.get("quarantined_effects", []))
    admission = WorldStateAdmission(
        status=str(admission_raw.get("status", "COMMITTED")),
        admitted_effect_ids=tuple(admission_raw.get("admitted_effect_ids", [])),
        quarantined_effects=quarantined,
        user_override=bool(admission_raw.get("user_override", False)),
        override_reason=str(admission_raw.get("override_reason", "")),
    )
    return ScenarioWorldModel(
        parties=model.parties, actions=model.actions, effects=model.effects,
        conditions=model.conditions, causal_links=model.causal_links,
        counterfactual_links=model.counterfactual_links,
        admission=admission, schema_version=str(data.get("schema_version", "1.0")),
    )


def _rows_with_clause_ids(rows: Sequence[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {key: value for key, value in row.items() if key != "provenance"}
        if "clause_ids" not in item:
            item["clause_ids"] = [
                ref.get("clause_id") for ref in row.get("provenance", [])
                if isinstance(ref, dict) and ref.get("clause_id")
            ]
        converted.append(item)
    return converted


def world_model_as_parse_payload(model: ScenarioWorldModel) -> dict[str, Any]:
    """Serialize a committed model back into the parser's clause_id shape."""
    data = model.as_dict()
    return {
        "schema_version": str(data.get("schema_version") or model.schema_version),
        "parties": _rows_with_clause_ids(data.get("parties", [])),
        "actions": _rows_with_clause_ids(data.get("actions", [])),
        "effects": _rows_with_clause_ids(data.get("effects", [])),
        "conditions": _rows_with_clause_ids(data.get("conditions", [])),
        "causal_links": _rows_with_clause_ids(data.get("causal_links", [])),
        "counterfactual_links": _rows_with_clause_ids(
            data.get("counterfactual_links", []),
        ),
    }


def compact_committed_world(model: Any) -> dict[str, Any]:
    """Cycle-stable factual projection of an admitted world model.

    Specialists see this as committed_world. It is scenario evidence, not a
    framework position. Hypotheses never appear here.
    """
    typed = model if isinstance(model, ScenarioWorldModel) else world_model_from_dict(model)
    if typed is None:
        return {}
    admitted = set(typed.admission.admitted_effect_ids)
    filter_admission = bool(admitted) or typed.admission.status in {
        "USER_ACCEPTED_WITH_QUARANTINE",
        "ABANDONED_CONTRADICTORY_WORLD_STATE",
    }
    effects = [
        effect for effect in typed.effects
        if not filter_admission or effect.effect_id in admitted
    ]
    admitted_ids = {effect.effect_id for effect in effects}
    return {
        "schema_version": typed.schema_version,
        "admission_status": typed.admission.status,
        "parties": [
            {
                "party_id": party.party_id,
                "label": party.label,
                "kind": party.kind,
                "quantities": recorded_quantity_payload(party.quantities),
            }
            for party in typed.parties
        ],
        "actions": [
            {
                "action_id": action.action_id,
                "intervention": action.intervention,
                "source_label": source_plan_label(action),
            }
            for action in typed.actions
        ],
        "effects": [
            {
                "effect_id": effect.effect_id,
                "action_id": effect.action_id,
                "party_id": effect.party_id,
                "directness": effect.directness,
                "effect_kind": effect.effect_kind,
                "polarity": effect.polarity,
                "modality": effect.modality,
                "outcome": effect.outcome,
                "condition_ids": list(effect.condition_ids),
                "likelihood_qualifiers": list(effect.likelihood_qualifiers),
                "overall_likelihood_qualifiers": list(
                    effect.overall_likelihood_qualifiers
                ),
                "quantities": recorded_quantity_payload(effect.quantities),
                **({
                    "source_binding": {
                        "source_proposition": effect.source_proposition,
                        "source_effect_ids": list(effect.source_effect_ids),
                        "derivation_operation": effect.derivation_operation,
                        "derivation_explanation": effect.derivation_explanation,
                        "derivation_assumptions": list(effect.derivation_assumptions),
                        "outcome_type_transformation": effect.outcome_type_transformation,
                    },
                } if typed.schema_version == "1.3" else {}),
            }
            for effect in effects
        ],
        "causal_links": [
            {
                "action_id": link.action_id,
                "source_id": link.source_id,
                "relation": link.relation,
                "target_id": link.target_id,
                "modality": link.modality,
            }
            for link in typed.causal_links
            if (
                (link.source_id not in {item.effect_id for item in typed.effects}
                 or link.source_id in admitted_ids)
                and (link.target_id not in {item.effect_id for item in typed.effects}
                     or link.target_id in admitted_ids)
            )
        ],
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "description": condition.description,
                "event_effect_id": condition.event_effect_id,
                "value_status": condition.value_status,
            }
            for condition in typed.conditions
        ],
        "counterfactual_links": [
            {
                "action_id": link.action_id,
                "source_effect_id": link.source_effect_id,
                "relation": link.relation,
                "alternative_action_id": link.alternative_action_id,
                "alternative_effect_id": link.alternative_effect_id,
                "modality": link.modality,
            }
            for link in typed.counterfactual_links
            if (
                link.source_effect_id in admitted_ids
                and link.alternative_effect_id in admitted_ids
            )
        ],
    }


_EXTENSION_ID_FIELDS = {
    "parties": "party_id",
    "effects": "effect_id",
    "conditions": "condition_id",
}


def admit_world_model_extension(
    base: ScenarioWorldModel,
    extension: dict[str, Any],
    *,
    clauses: Sequence[dict[str, str]],
    action_ids: Sequence[str],
    action_texts: dict[str, str] | None = None,
) -> ScenarioWorldModel:
    """Admit additional parties, effects, or links into a committed world model.

    IDs are append-only. The merged model is re-validated, including completeness.
    On failure this raises ValueError and the caller must keep ``base``.
    """
    if not isinstance(extension, dict):
        raise ValueError("world-model extension must be an object")
    payload = world_model_as_parse_payload(base)
    for key, id_field in _EXTENSION_ID_FIELDS.items():
        extra = extension.get(key) or []
        if extra and not isinstance(extra, list):
            raise ValueError(f"extension {key} must be an array")
        existing = {
            _clean(row.get(id_field), 80)
            for row in payload.get(key, [])
            if isinstance(row, dict)
        }
        for row in extra if isinstance(extra, list) else []:
            if not isinstance(row, dict):
                continue
            row_id = _clean(row.get(id_field), 80)
            if not row_id:
                raise ValueError(f"extension {key} row is missing {id_field}")
            if row_id in existing:
                raise ValueError(
                    f"extension reuses {id_field} {row_id}; committed "
                    "world-model ids are append-only"
                )
            payload.setdefault(key, []).append(row)
            existing.add(row_id)
    for key in ("causal_links", "counterfactual_links"):
        extra = extension.get(key) or []
        if extra and not isinstance(extra, list):
            raise ValueError(f"extension {key} must be an array")
        payload.setdefault(key, []).extend(
            row for row in extra if isinstance(row, dict)
        )
    return parse_world_model(
        payload,
        clauses=clauses,
        action_ids=action_ids,
        action_texts=action_texts,
        require_completeness=True,
    )
