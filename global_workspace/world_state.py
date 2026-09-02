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
MODALITIES = {
    "CERTAIN", "STIPULATED_CONDITIONAL", "PROBABILISTIC", "POSSIBLE", "UNKNOWN",
}
POLARITIES = {"BENEFICIAL", "ADVERSE", "NEUTRAL", "UNRESOLVED", "FOREGONE"}
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
    # Bare "one crew" / "two pipelines" are not magnitudes. Cardinals count only
    # with an explicit unit; collective nouns (dozen/score) count on their own.
    r"|(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s+" + _DURATION_UNIT +
    r"|(?:dozen|score|tens|hundreds|thousands|dozens|millions|billions)"
    r"(?:\s+of\s+(?:tens\s+of\s+)?(?:thousands|millions|billions))?"
    r")(?![\w.])",
    re.IGNORECASE,
)


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


def _provenance_text(effect: WorldEffect) -> str:
    """Text that may ground a claim. Generated outcome sentences are excluded."""
    return " ".join(ref.excerpt for ref in effect.provenance if ref.excerpt)


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
    condition_ids: tuple[str, ...] = ()
    quantities: tuple[str, ...] = ()
    provenance: tuple[SourceRef, ...] = ()

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
        condition_ids=tuple(
            dict.fromkeys(_clean(value, 80).upper() for value in row.get("condition_ids", []))
        ),
        quantities=tuple(
            dict.fromkeys(_clean(value, 80) for value in row.get("quantities", []))
        ),
        provenance=_refs(row.get("clause_ids", []), lookup),
    ) for row in raw.get("effects", []) if isinstance(row, dict))
    links = tuple(CausalLink(
        source_id=_clean(row.get("source_id"), 80),
        relation=_clean(row.get("relation"), 64).upper(),
        target_id=_clean(row.get("target_id"), 80),
        modality=_clean(row.get("modality"), 48).upper(),
        condition_ids=tuple(
            dict.fromkeys(_clean(value, 80).upper() for value in row.get("condition_ids", []))
        ),
        provenance=_refs(row.get("clause_ids", []), lookup),
    ) for row in raw.get("causal_links", []) if isinstance(row, dict))
    model = ScenarioWorldModel(
        parties=parties, actions=actions, effects=effects,
        conditions=conditions, causal_links=links,
        admission=WorldStateAdmission(
            status="COMMITTED",
            admitted_effect_ids=tuple(effect.effect_id for effect in effects),
        ),
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
    valid_link_nodes = expected_actions | set(effect_ids) | condition_ids
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
        admission=admission, schema_version=str(data.get("schema_version", "1.0")),
    )
