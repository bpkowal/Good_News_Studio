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
_QUANTITY_PREFIX = (
    r"(?:over|more\s+than|at\s+least|up\s+to|fewer\s+than|less\s+than|"
    r"about|approximately|nearly|almost)\s+"
)
_NUMBER_WORD = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
_SMALL_CARDINAL = r"(?:" + _NUMBER_WORD + r")"
_ARTICLE_OR_CARDINAL = r"(?:a|an|" + _NUMBER_WORD + r")"
_NUMBER_SCALE = r"(?:hundred|thousand|million|billion)"
_POPULATION_NOUN = (
    r"(?:people|persons?|patients?|residents?|infants?|children|adults|"
    r"workers?|families|households?|communities|students?|animals?)"
)
_COLLECTIVE_QUANTITY = (
    r"(?:dozen|score|tens|hundreds|thousands|dozens|millions|billions)"
    r"(?:\s+of\s+(?:tens\s+of\s+)?(?:thousands|millions|billions))?"
)
# Longer compounds are listed first so "over five hundred" wins over "five".
# Articles count only in "a hundred"/"an thousand"-style compounds, never as
# a bare cardinality before a population noun ("a dozen residents").
_EXPLICIT_QUANTITY = re.compile(
    r"(?<![\w.])(?:"
    r"(?:" + _QUANTITY_PREFIX + r")?" + _ARTICLE_OR_CARDINAL + r"[-\s]+"
    + _NUMBER_SCALE + r"(?:[-\s]+" + _NUMBER_SCALE + r")?"
    r"|(?:" + _QUANTITY_PREFIX + r")?\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s*" + _DURATION_UNIT + r")?"
    r"|" + _SMALL_CARDINAL + r"\s+" + _DURATION_UNIT +
    r"|" + _COLLECTIVE_QUANTITY +
    r"|" + _SMALL_CARDINAL + r"(?![-\s]+" + _NUMBER_SCALE + r")"
    r"(?=\s+(?:[a-z-]+\s+){0,2}" + _POPULATION_NOUN + r")"
    r")(?![\w.])",
    re.IGNORECASE,
)
_LIKELIHOOD_QUALIFIER = re.compile(
    r"\b(?:near[- ]certain|almost certain|virtually certain|highly likely|"
    r"likely|unlikely|possible|uncertain|unknown)\b", re.IGNORECASE,
)
_HIGH_CONFIDENCE_LIKELIHOOD = re.compile(
    r"\b(?:near[- ]certain|almost certain|virtually certain)\b",
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
    r"\b(?:death|die|dies|surviv|health|medical|injur|hunger|thirst|"
    r"well-?being|sustain(?:s|ing)?\s+(?:people|persons?|patients?|residents?|"
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


def explicit_quantity_spans(text: str) -> tuple[str, ...]:
    """Return numerical and scale phrases copied from source wording.

    Deliberately incomplete: no invented probabilities/QALYs, no age labels,
    no bare duration units, and no vague comparatives such as 'few' or 'many'.
    Nested spans collapse to the longest source phrase, so 'over five hundred'
    is not also recorded as 'five hundred' or 'five'.
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
        key = span.casefold()
        if key not in seen:
            seen.add(key)
            kept.append(span)
    return tuple(kept)


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
    found: list[str] = []
    for match in pattern.finditer(str(text or "")):
        span = " ".join(match.group(0).split())
        if span and span.casefold() not in {item.casefold() for item in found}:
            found.append(span)
    return tuple(found)


def explicit_likelihood_spans(text: str) -> tuple[str, ...]:
    return _explicit_qualifier_spans(text, _LIKELIHOOD_QUALIFIER)


def explicit_scope_spans(text: str) -> tuple[str, ...]:
    return _explicit_qualifier_spans(text, _SCOPE_QUALIFIER)


def explicit_temporal_spans(text: str) -> tuple[str, ...]:
    return _explicit_qualifier_spans(text, _TEMPORAL_QUALIFIER)


def _provenance_text(effect: WorldEffect) -> str:
    """Text that may ground a claim. Generated outcome sentences are excluded."""
    return " ".join(ref.excerpt for ref in effect.provenance if ref.excerpt)


_PARTY_MATCH_STOPWORDS = {
    "affected", "city", "group", "people", "person", "population", "relying",
    "the", "their", "those",
}
_EFFECT_MATCH_STOPWORDS = {
    "affected", "cause", "caused", "causes", "effect", "failure", "outcome",
    "prevent", "prevented", "prevents", "risk", "the", "their", "would",
}


def _match_words(text: str, *, stopwords: set[str]) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(word) > 2 and word not in stopwords
    }


def _party_expected_quantities(party: WorldParty) -> tuple[str, ...]:
    """Find source cardinalities locally modifying a population party."""
    if party.kind not in {"POPULATION", "GROUP", "HOUSEHOLD", "COMMUNITY"}:
        return ()
    party_words = {
        word for word in re.findall(r"[a-z0-9]+", party.label.casefold())
        if len(word) > 2 and word not in _PARTY_MATCH_STOPWORDS
    }
    found: list[str] = []
    for ref in party.provenance:
        text = ref.excerpt
        for quantity in explicit_quantity_spans(text):
            match = re.search(re.escape(quantity), text, re.IGNORECASE)
            if match is None:
                continue
            tail = text[match.end():match.end() + 90]
            tail = re.split(
                r"[,;.]|\b(?:and|but|or|while|whereas)\b",
                tail, maxsplit=1, flags=re.IGNORECASE,
            )[0]
            local = text[match.start():match.end()] + tail
            local_words = set(re.findall(r"[a-z0-9]+", local.casefold()))
            if party_words & local_words:
                found.append(quantity)
    return tuple(dict.fromkeys(found))


def _effect_expected_qualifiers(
    effect: WorldEffect, extractor: Any,
) -> tuple[str, ...]:
    """Bind explicit source qualifiers by local overlap with an atomic effect."""
    effect_words = _match_words(effect.outcome, stopwords=_EFFECT_MATCH_STOPWORDS)
    found: list[str] = []
    for ref in effect.provenance:
        text = ref.excerpt
        for qualifier in extractor(text):
            match = re.search(re.escape(qualifier), text, re.IGNORECASE)
            if match is None:
                continue
            boundary = r"[;.]|,\s*(?:and|but|or|while|whereas)\b"
            before = text[max(0, match.start() - 28):match.start()]
            after = text[match.end():match.end() + 28]
            before_parts = re.split(boundary, before, flags=re.IGNORECASE)
            after_parts = re.split(
                boundary, after, maxsplit=1, flags=re.IGNORECASE,
            )
            local = before_parts[-1] + qualifier + after_parts[0]
            local_words = _match_words(local, stopwords=_EFFECT_MATCH_STOPWORDS)
            if effect_words & local_words:
                found.append(qualifier)
    return tuple(dict.fromkeys(found))


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


def _effect_counts_for_roles(effect: WorldEffect, party: WorldParty | None) -> bool:
    """True when an admitted effect should populate compact harm/benefit roles.

    Foregone counterfactuals stay out so they are not double-counted against
    the actual outcome. Neutral actor-performances are not welfare claims.
    Directness is not required: stipulated downstream deaths and survivals
    are the point of these roles. Facilities and institutions are intermediate
    process-bearers, not compact harmed/beneficiary parties.
    """
    return counts_as_actual_welfare(
        polarity=effect.polarity,
        directness=effect.directness,
        effect_kind=effect.effect_kind,
        party_kind=party.kind if party is not None else "",
    )


def utilitarian_omits_foregone_dual(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> bool:
    """True when FOREGONE restates a swap already encoded by an actual outcome.

    Opportunity cost is the right utilitarian row when choosing this action
    merely forgoes another action's benefit on party P. It is a double count
    when this action also has its own admitted harm or benefit on P: that
    actual row already is the welfare consequence of the choice.
    """
    if effect.directness != "FOREGONE" and effect.polarity != "FOREGONE":
        return False
    party = next(
        (item for item in model.parties if item.party_id == effect.party_id),
        None,
    )
    return any(
        other.effect_id != effect.effect_id
        and other.party_id == effect.party_id
        and _effect_counts_for_roles(other, party)
        for other in model.effects_for(effect.action_id)
    )


def _high_confidence_likelihood(effect: WorldEffect) -> bool:
    """True when source-copied likelihood is in the closed high-confidence class."""
    blob = " ".join(effect.likelihood_qualifiers)
    return bool(_HIGH_CONFIDENCE_LIKELIHOOD.search(blob))


def _compact_role_is_settled(effect: WorldEffect) -> bool:
    """CERTAIN, or probabilistic/conditional harm the source marks near-certain.

    Weaker likelihoods (likely, possible, unknown) stay in unresolved rather
    than occupying compact harmed/beneficiary slots.
    """
    if effect.polarity == "UNRESOLVED":
        return False
    if effect.modality == "CERTAIN":
        return True
    if effect.modality not in {"PROBABILISTIC", "STIPULATED_CONDITIONAL"}:
        return False
    return _high_confidence_likelihood(effect)


def project_world_action_roles(
    model: ScenarioWorldModel, action_id: str,
) -> ProjectedActionRoles:
    """Derive compact roles from admitted health, welfare, and liberty effects."""
    party_by_id = {party.party_id: party for party in model.parties}
    beneficiaries: list[str] = []
    harmed: list[str] = []
    unresolved: list[str] = []
    for effect in model.effects_for(action_id):
        party = party_by_id.get(effect.party_id)
        if party is None or not _effect_counts_for_roles(effect, party):
            continue
        label = party.label
        if not _compact_role_is_settled(effect):
            if label not in unresolved:
                unresolved.append(label)
            continue
        if effect.polarity == "BENEFICIAL":
            if label not in beneficiaries and label not in harmed:
                beneficiaries.append(label)
        elif effect.polarity == "ADVERSE":
            if label not in harmed:
                harmed.append(label)
                if label in beneficiaries:
                    beneficiaries.remove(label)
    unresolved = [
        label for label in unresolved
        if label not in beneficiaries and label not in harmed
    ]
    return ProjectedActionRoles(
        beneficiaries=tuple(beneficiaries),
        harmed=tuple(harmed),
        unresolved=tuple(unresolved),
    )


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


def _closed_class_qualifiers(effect: WorldEffect) -> WorldEffect:
    """Copy source-bound qualifiers and drop invented ones.

    Qualifiers are closed-class spans. The parser can fill and strip them
    without inventing outcomes, parties, or causal structure.
    """
    provenance_text = _provenance_text(effect).casefold()

    def stated(values: tuple[str, ...]) -> list[str]:
        return [value for value in values if value.casefold() in provenance_text]

    likelihood = stated(effect.likelihood_qualifiers)
    scope = stated(effect.scope_qualifiers)
    temporal = stated(effect.temporal_qualifiers)
    for extractor, bucket in (
        (explicit_likelihood_spans, likelihood),
        (explicit_scope_spans, scope),
        (explicit_temporal_spans, temporal),
    ):
        have = {item.casefold() for item in bucket}
        for value in _effect_expected_qualifiers(effect, extractor):
            if value.casefold() not in have:
                bucket.append(value)
                have.add(value.casefold())
    return replace(
        effect,
        likelihood_qualifiers=tuple(dict.fromkeys(likelihood)),
        scope_qualifiers=tuple(dict.fromkeys(scope)),
        temporal_qualifiers=tuple(dict.fromkeys(temporal)),
    )


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
            likelihood_qualifiers=tuple(dict.fromkeys(
                _clean(value, 80) for value in row.get("likelihood_qualifiers", [])
            )),
            scope_qualifiers=tuple(dict.fromkeys(
                _clean(value, 80) for value in row.get("scope_qualifiers", [])
            )),
            temporal_qualifiers=tuple(dict.fromkeys(
                _clean(value, 80) for value in row.get("temporal_qualifiers", [])
            )),
            provenance=_refs(row.get("clause_ids", []), lookup),
        )
        if schema_version != "1.0":
            effect = _closed_class_qualifiers(effect)
        parsed_effects.append(effect)
    effects = tuple(parsed_effects)
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
    errors, _ = validate_world_model(model, action_ids=action_ids)
    if require_completeness:
        errors.extend(validate_world_completeness(model, action_ids=action_ids))
    if errors:
        raise ValueError("; ".join(errors))
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
        if model.schema_version == "1.2":
            recorded = {value.casefold() for value in party.quantities}
            omitted = [
                value for value in _party_expected_quantities(party)
                if value.casefold() not in recorded
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
                    "directness must be DIRECT"
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
            errors.append(f"{prefix} is {effect.modality} but has no condition")
        if effect.modality == "CERTAIN" and effect.condition_ids:
            errors.append(
                f"{prefix} is CERTAIN but lists conditions; CERTAIN effects "
                "must not carry condition_ids"
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
        provenance_text = _provenance_text(effect).casefold()
        claimed = explicit_quantity_spans(effect.outcome)
        recorded = {item.casefold() for item in effect.quantities}
        ungrounded = [
            span for span in claimed if span.casefold() not in provenance_text
        ]
        if ungrounded:
            errors.append(
                f"{prefix} outcome uses quantities absent from provenance: "
                f"{ungrounded} (outcome text alone cannot ground a quantity)"
            )
        omitted = [
            span for span in claimed
            if span.casefold() in provenance_text and span.casefold() not in recorded
        ]
        if omitted:
            errors.append(
                f"{prefix} omits quantities stated in its outcome and provenance: "
                f"{omitted}"
            )
        for quantity in effect.quantities:
            if quantity.casefold() not in provenance_text:
                errors.append(
                    f"{prefix} quantity {quantity!r} is not stated in its "
                    "provenance (outcome text alone cannot ground a quantity)"
                )
        qualifier_fields = (
            ("likelihood", effect.likelihood_qualifiers, explicit_likelihood_spans),
            ("scope", effect.scope_qualifiers, explicit_scope_spans),
            ("temporal", effect.temporal_qualifiers, explicit_temporal_spans),
        )
        for qualifier_kind, recorded_values, extractor in qualifier_fields:
            recorded_qualifiers = {value.casefold() for value in recorded_values}
            for value in recorded_values:
                if value.casefold() not in provenance_text:
                    errors.append(
                        f"{prefix} {qualifier_kind} qualifier {value!r} is not stated "
                        "in its provenance"
                    )
            if model.schema_version == "1.2":
                required = list(_effect_expected_qualifiers(effect, extractor))
                missing = [
                    value for value in required
                    if value.casefold() not in recorded_qualifiers
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
# A riot, spillway, or institution can carry INSTITUTIONAL_OUTCOME or
# INTERVENTION as the intermediate stage. The skip list is for another
# person's act, framing, or bodily outcome posing as that stage.
_INTERMEDIATE_BEARER_KINDS = {
    "PROCESS", "FACILITY", "INSTITUTION", "ORGANIZATION",
    "SYSTEM", "AUTOMATED_SYSTEM",
}
_RECIPIENT_DIRECT_ACT_KINDS = {
    "INTERVENTION", "RESOURCE_TRANSFER", "INSTITUTIONAL_OUTCOME",
}


def _downstream_health_parents(
    effect: WorldEffect, model: ScenarioWorldModel,
) -> tuple[WorldEffect, ...]:
    by_id = {item.effect_id: item for item in model.effects}
    parents: list[WorldEffect] = []
    for link in model.causal_links:
        if link.action_id != effect.action_id or link.target_id != effect.effect_id:
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
            stack.append(link.source_id)
    return False


def _requires_act_ancestry(effect: WorldEffect, party: WorldParty | None) -> bool:
    if effect.directness != "DOWNSTREAM":
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
    immediate parent of someone else's health. The same kind on a PROCESS
    or INSTITUTION is the intermediate the completeness gate asked for.
    """
    if parent.party_id == child.party_id:
        return False
    if parent.effect_kind not in _SKIP_INTERMEDIATE_PARENT_KINDS:
        return False
    bearer = party_by_id.get(parent.party_id)
    if bearer is not None and bearer.kind in _INTERMEDIATE_BEARER_KINDS:
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
        if not _effect_counts_for_roles(effect, party):
            continue
        action = action_by_id.get(effect.action_id)
        if action is None:
            continue
        if effect.party_id in {action.actor_party_id, *action.recipient_party_ids}:
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
    """Require FOREGONE overlays for opposed non-recipient welfare, not new chains."""
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
                        f"non-recipient {party_id}, but {left_id} has no FOREGONE "
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
                        f"non-recipient {party_id}, but {right_id} has no FOREGONE "
                        f"effect with a counterfactual_link to {left_id}'s actual "
                        "effect; do not put that comparison on causal_links"
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
    counterfactual overlay.
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
                    "are DOWNSTREAM or FOREGONE"
                )
            if not _requires_act_ancestry(effect, party_by_id.get(effect.party_id)):
                continue
            parents = _downstream_health_parents(effect, model)
            if not parents:
                errors.append(
                    f"{effect.effect_id} is a downstream human outcome with no "
                    "causal parent; connect it through the source-named process"
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
                parent_ids = ", ".join(parent.effect_id for parent in skipped_parents)
                errors.append(
                    f"{effect.effect_id} is caused directly by another party's act "
                    f"or bodily outcome ({parent_ids}); insert a PROCESS, FACILITY, "
                    "or INSTITUTION state as the immediate parent. A person's "
                    "framing, execution, or death is not that intermediate"
                )
            if typed_topology and not _ancestry_reaches_direct_act(effect, action, model):
                errors.append(
                    f"{effect.effect_id} is a downstream human outcome whose causal "
                    "ancestry never reaches a DIRECT act on the actor or a named "
                    "recipient; connect this action's intervention to the intermediate "
                    "process that produces the outcome"
                )
    if model.schema_version != "1.0":
        errors.extend(_foreclosure_completeness_errors(model))
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
            {"party_id": party.party_id, "label": party.label, "kind": party.kind}
            for party in typed.parties
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
