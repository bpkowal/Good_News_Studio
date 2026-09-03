"""Typed, framework-neutral world state for ethical scenarios.

The objects in this module describe who acts, what each action does, and what
effects the scenario attributes to it.  They deliberately contain no ethical
evaluation.  A single admitted model is projected into canonical action
records and the semantic graph so those two views cannot independently infer
contradictory facts from the same prose.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
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
_EXPLICIT_QUANTITY = re.compile(
    r"(?<![\w.])(?:"
    r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*" + _DURATION_UNIT + r")?"
    # Bare "one crew" / "two pipelines" are not population magnitudes. Written
    # cardinals count with a duration or recognized affected-population noun;
    # collective nouns (dozen/score) count on their own.
    r"|(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s+" + _DURATION_UNIT +
    r"|(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"(?=\s+(?:[a-z-]+\s+){0,2}(?:people|persons?|patients?|residents?|"
    r"infants?|children|adults|workers?|families|households?|communities|"
    r"students?|animals?))"
    r"|(?:dozen|score|tens|hundreds|thousands|dozens|millions|billions)"
    r"(?:\s+of\s+(?:tens\s+of\s+)?(?:thousands|millions|billions))?"
    r")(?![\w.])",
    re.IGNORECASE,
)
_LIKELIHOOD_QUALIFIER = re.compile(
    r"\b(?:near[- ]certain|almost certain|highly likely|likely|unlikely|"
    r"possible|uncertain|unknown)\b", re.IGNORECASE,
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
    """
    stripped = _AGE_LABEL.sub(" ", str(text or ""))
    found: list[str] = []
    for match in _EXPLICIT_QUANTITY.finditer(stripped):
        span = " ".join(match.group(0).split())
        if span and span.casefold() not in {item.casefold() for item in found}:
            found.append(span)
    return tuple(found)


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
            boundary = r"[;.]|\b(?:and|but|or|while|whereas)\b"
            before = text[max(0, match.start() - 90):match.start()]
            after = text[match.end():match.end() + 90]
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


def _refs(clause_ids: Iterable[Any], lookup: dict[str, str]) -> tuple[SourceRef, ...]:
    return tuple(
        SourceRef(clause_id=clause_id, excerpt=lookup.get(clause_id, ""))
        for clause_id in dict.fromkeys(_clean(value, 32) for value in clause_ids)
        if clause_id in lookup
    )


def parse_world_model(
    raw: Any,
    *,
    clauses: Sequence[dict[str, str]],
    action_ids: Sequence[str],
    action_texts: dict[str, str] | None = None,
) -> ScenarioWorldModel:
    """Parse the compact grounding payload without silently repairing it."""
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
    effects = tuple(WorldEffect(
        effect_id=_clean(row.get("effect_id"), 80),
        action_id=_clean(row.get("action_id"), 16).upper(),
        party_id=_clean(row.get("party_id"), 64).upper(),
        outcome=_clean(row.get("outcome"), 240),
        relation=_clean(row.get("relation"), 64).upper(),
        polarity=_clean(row.get("polarity"), 32).upper(),
        directness=_clean(row.get("directness"), 32).upper(),
        modality=_clean(row.get("modality"), 48).upper(),
        effect_kind=_clean(row.get("effect_kind"), 48).upper() or "OTHER",
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
    ) for row in raw.get("effects", []) if isinstance(row, dict))
    effect_by_id = {effect.effect_id: effect for effect in effects}
    parsed_links: list[CausalLink] = []
    migrated_counterfactuals: list[CounterfactualLink] = []
    for row in raw.get("causal_links", []):
        if not isinstance(row, dict):
            continue
        source_id = _clean(row.get("source_id"), 80)
        target_id = _clean(row.get("target_id"), 80)
        relation = _clean(row.get("relation"), 64).upper()
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
        # Compatibility normalization for traces emitted before counterfactual
        # links had their own schema. A cross-action FOREGOES edge is matched to
        # the chosen action's explicit foregone effect for the same party.
        if source_action and target_action and source_action != target_action:
            if schema_version != "1.0" or relation != "FOREGOES":
                parsed_links.append(CausalLink(
                    action_id=explicit_action or source_action,
                    source_id=source_id, relation=relation, target_id=target_id,
                    modality=modality, condition_ids=condition_values,
                    provenance=refs,
                ))
                continue
            alternative_effect = effect_by_id.get(target_id)
            candidates = [
                effect for effect in effects
                if effect.action_id == source_action
                and effect.directness == "FOREGONE"
                and alternative_effect is not None
                and effect.party_id == alternative_effect.party_id
            ]
            migrated_counterfactuals.append(CounterfactualLink(
                action_id=source_action,
                source_effect_id=candidates[0].effect_id if len(candidates) == 1 else "",
                relation="FOREGOES_ALTERNATIVE_EFFECT",
                alternative_action_id=target_action,
                alternative_effect_id=target_id,
                modality=modality,
                condition_ids=condition_values,
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
        relation=_clean(row.get("relation"), 64).upper(),
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
    if errors:
        raise ValueError("; ".join(errors))
    return model


def _opposed(left: WorldEffect, right: WorldEffect) -> bool:
    if left.polarity == right.polarity:
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
                    and effect.effect_kind in {"INTERVENTION", "RESOURCE_TRANSFER"}
                    for effect in model.effects
                ):
                    errors.append(
                        f"{action.action_id} recipient {recipient_id} lacks an atomic "
                        "atomic DIRECT INTERVENTION or RESOURCE_TRANSFER effect"
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
