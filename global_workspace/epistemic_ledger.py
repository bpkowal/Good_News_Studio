"""Authoritative proposition identities and monotone epistemic status.

World facts seed the ledger. Delegates may cite them or introduce hypotheses,
but repetition and cross-framework reuse only increase attention, never status.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any, Iterable

from .scenario_semantics import project_grounded_action_effects
from .semantic_graph import SemanticGraph
from .world_state import explicit_quantity_spans


EPISTEMIC_STATUS_RANK = {
    "REJECTED": 0,
    "HYPOTHETICAL": 1,
    "UNRESOLVED": 2,
    "DERIVED": 3,
    "ESTABLISHED": 4,
}
DECISION_CRITICAL_CAP_STATUSES = {"REJECTED", "UNRESOLVED"}
HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP = 0.70
EPISTEMIC_TYPES = {
    "WORLD_ESTABLISHED",
    "UNKNOWN_PARAMETER",
    "HYPOTHESIS",
    "FRAMEWORK_DERIVED",
}
_COMPOSITION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
    "is", "of", "on", "or", "that", "the", "to", "with",
}
_BIND_STOPWORDS = _COMPOSITION_STOPWORDS | {
    "be", "been", "being", "can", "could", "did", "do", "does", "had", "has",
    "have", "it", "its", "may", "might", "shall", "their", "them", "then",
    "this", "those", "which", "who", "whom", "will", "would", "into", "also",
    "than", "when", "while", "because", "via", "per", "if", "so", "thus",
    "thereby", "hence", "about", "after", "before", "across", "over", "more",
    "plus", "claim", "effect", "outcome", "result", "results", "resulting",
    "lead", "leads", "leading", "cause", "causes", "causing", "allow",
    "allows", "allowing", "refuse", "refuses", "refusing", "refusal",
    "remain", "remains", "remaining", "stay", "stays", "carry", "carried",
    "immediately", "immediate", "predictably", "affected", "subject",
    "magnitude", "qualifier", "action",
}
_STRENGTHENING_WORDS = {
    "always", "never", "only", "exclusive", "exclusively", "unique",
    "inevitable", "irreversible", "guaranteed", "must", "fatal", "fatality",
    "fatalities", "mortality", "certain", "certainly", "prove", "proves",
    "organ",
}
_CARDINAL_VALUES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1_000_000, "billion": 1_000_000_000,
}
_OUTCOME_FAMILIES = (
    frozenset({
        "kill", "killed", "killing", "die", "dies", "died", "dying", "dead",
        "death", "deaths", "execute", "executed", "executes", "executing",
        "execution", "murder", "slay", "fatal", "fatality", "fatalities",
        "mortality",
    }),
    frozenset({
        "survive", "survives", "survived", "survival", "live", "lives",
        "alive", "spare", "spared", "save", "saves", "saved", "saving",
        "preserve", "preserves", "preserved", "preserving",
    }),
    frozenset({
        "frame", "framed", "frames", "framing", "accuse", "accused",
        "accuses", "accusing", "accusation", "convict", "convicted",
        "convicting",
    }),
    frozenset({
        "halt", "halts", "halted", "halting", "stop", "stops", "stopped",
        "stopping", "quell", "quells", "quelled", "quelling",
    }),
    frozenset({
        "continue", "continues", "continued", "continuing", "spread",
        "spreads", "spreading", "ongoing", "escalate", "escalates",
        "escalating",
    }),
    frozenset({
        "prevent", "prevents", "prevented", "preventing", "avert", "averts",
        "averted",
    }),
    frozenset({
        "injur", "injury", "injuries", "injured", "harm", "harms", "harmed",
        "damage", "damaged",
    }),
)
_CAUSAL_RELATIONS = {
    "CAUSES", "ENABLES", "INCREASES", "PREVENTS", "DISABLES", "DECREASES",
}


@dataclass(slots=True)
class PropositionRecord:
    proposition_id: str
    claim: str
    proposition_type: str
    epistemic_status: str
    support_ids: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    introduced_by: str = "SYSTEM"
    mention_count: int = 1
    decision_critical_mentions: int = 0
    epistemic_type: str = ""
    aliases: list[str] = field(default_factory=list)
    action_id: str = ""
    outcome: str = ""
    polarity: str = ""
    party_labels: list[str] = field(default_factory=list)
    quantities: list[str] = field(default_factory=list)
    effect_kind: str = ""
    context_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(prefix: str, value: str) -> str:
    normalized = " ".join(str(value).casefold().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"PROP:{prefix}:{digest}"


def seed_proposition_ledger(graph: SemanticGraph) -> dict[str, PropositionRecord]:
    """Create established descriptive propositions from admitted world effects."""
    ledger: dict[str, PropositionRecord] = {}
    seen_consequences: set[str] = set()
    seen_party_quantities: set[tuple[str, str]] = set()
    for effect in project_grounded_action_effects(graph):
        if effect.consequence_id in seen_consequences:
            continue
        seen_consequences.add(effect.consequence_id)
        consequence = graph.nodes.get(effect.consequence_id)
        if consequence is None:
            continue
        world_effect_id = str(consequence.attributes.get("world_effect_id", "")).strip()
        proposition_id = (
            f"PROP:WORLD:{world_effect_id}"
            if world_effect_id else _stable_id("GROUND", effect.consequence_id)
        )
        claim_parts = [consequence.label]
        affected_subject = " ".join(effect.affected_subject.split())
        if affected_subject and affected_subject.casefold() != "affected constituency":
            claim_parts.append(f"affected subject: {affected_subject}")
        qualifier = " ".join(effect.magnitude_or_qualifier.split())
        if qualifier and qualifier.upper() != "STATED":
            qualifier_words = set(qualifier.casefold().split())
            subject_words = set(affected_subject.casefold().split())
            if not qualifier_words <= subject_words:
                claim_parts.append(f"magnitude or qualifier: {qualifier}")
        ledger[proposition_id] = PropositionRecord(
            proposition_id=proposition_id,
            claim="; ".join(claim_parts),
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            support_ids=[world_effect_id or effect.consequence_id],
            introduced_by="WORLD_MODEL",
            epistemic_type="WORLD_ESTABLISHED",
            action_id=effect.action_id,
            outcome=consequence.label,
            polarity=str(consequence.attributes.get("polarity", "")).upper(),
            party_labels=[affected_subject] if affected_subject else [],
            quantities=list(dict.fromkeys([
                *list(effect.affected_subject_quantities),
                *[
                    str(value).strip()
                    for value in consequence.attributes.get("quantities", [])
                    if str(value).strip()
                ],
            ])),
            effect_kind=str(consequence.attributes.get("effect_kind", "")).upper(),
        )
        party_id = (
            effect.affected_subject_node_ids[0].removeprefix("PARTY:")
            if effect.affected_subject_node_ids else effect.affected_subject
        )
        for quantity in effect.affected_subject_quantities:
            quantity_key = (party_id, quantity.casefold())
            if quantity_key in seen_party_quantities:
                continue
            seen_party_quantities.add(quantity_key)
            subject = effect.affected_subject
            atomic_claim = (
                subject if quantity.casefold() in subject.casefold()
                else f"{quantity} {subject}"
            )
            quantity_digest = hashlib.sha256(
                f"{party_id}|{quantity.casefold()}".encode("utf-8")
            ).hexdigest()[:12]
            atomic_id = f"PROP:WORLD:PARTY:{party_id}:QUANTITY:{quantity_digest}"
            ledger[atomic_id] = PropositionRecord(
                proposition_id=atomic_id,
                claim=atomic_claim,
                proposition_type="DESCRIPTIVE",
                epistemic_status="ESTABLISHED",
                support_ids=[party_id, quantity],
                introduced_by="WORLD_MODEL",
                epistemic_type="WORLD_ESTABLISHED",
                action_id=effect.action_id,
                party_labels=[subject] if subject else [],
                quantities=[quantity],
            )
        qualifier_groups = (
            ("LIKELIHOOD", effect.likelihood_qualifiers),
            ("SCOPE", effect.scope_qualifiers),
            ("TEMPORAL", effect.temporal_qualifiers),
        )
        for qualifier_kind, qualifiers in qualifier_groups:
            for index, qualifier in enumerate(qualifiers):
                atomic_id = (
                    f"PROP:WORLD:{world_effect_id or effect.effect_id}:"
                    f"{qualifier_kind}:{index}"
                )
                ledger[atomic_id] = PropositionRecord(
                    proposition_id=atomic_id,
                    claim=f"{qualifier} — {consequence.label}",
                    proposition_type="DESCRIPTIVE",
                    epistemic_status="ESTABLISHED",
                    support_ids=[world_effect_id or effect.effect_id, qualifier],
                    introduced_by="WORLD_MODEL",
                    epistemic_type="WORLD_ESTABLISHED",
                    action_id=effect.action_id,
                    outcome=consequence.label,
                    party_labels=[affected_subject] if affected_subject else [],
                )
    _annotate_binding_context(ledger, graph)
    _seed_unknown_parameters(ledger, graph)
    return ledger


def ledger_projection(
    ledger: dict[str, PropositionRecord],
) -> list[dict[str, Any]]:
    return [ledger[key].to_dict() for key in sorted(ledger)]


def _normalized_claim(value: str) -> str:
    return " ".join(str(value).casefold().split()).strip(" .")


def _claim_is_covered(claim: str, authoritative_claim: str) -> bool:
    """Accept exact canonical atoms and lossless projections of bundled display text."""
    normalized = _normalized_claim(claim)
    if normalized == _normalized_claim(authoritative_claim):
        return True
    components = [
        _normalized_claim(value) for value in str(authoritative_claim).split(";")
        if _normalized_claim(value)
    ]
    return normalized in components


def _content_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", str(text).casefold())
        if word not in _BIND_STOPWORDS and len(word) > 1
    }


def _stem(word: str) -> str:
    token = str(word or "").casefold()
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[:-len(suffix)]
    return token


def _stems(text: str) -> set[str]:
    return {_stem(word) for word in _content_words(text)}


def _has_negation(text: str) -> bool:
    blob = str(text or "").casefold()
    return bool(re.search(
        r"\b(?:not|never|without)\b|\bun(?:executed|framed|harmed)\b",
        blob,
    ))


def _implied_polarity(text: str, families: set[int]) -> str:
    negated = _has_negation(text)
    if 0 in families:
        return "BENEFICIAL" if negated else "ADVERSE"
    if 1 in families or 3 in families or 5 in families:
        return "BENEFICIAL"
    if 2 in families:
        return "BENEFICIAL" if negated else "ADVERSE"
    if 4 in families or 6 in families:
        return "ADVERSE"
    return ""


def _outcome_families(text: str) -> set[int]:
    words = set(re.findall(r"[a-z0-9]+", str(text).casefold()))
    return {
        index for index, family in enumerate(_OUTCOME_FAMILIES)
        if words & family
    }


def _quantity_value(span: str) -> tuple[int, bool] | None:
    text = " ".join(str(span or "").casefold().split())
    if not text:
        return None
    lower_bound = bool(
        re.search(r"\b(?:over|more than|at least|plus)\b", text)
        or text.endswith("+")
    )
    digits = re.findall(r"\d+", text.replace(",", ""))
    if digits:
        return int(digits[0]), lower_bound
    total = 0
    last_scale = 1
    matched = False
    for word in re.findall(r"[a-z]+", text):
        if word not in _CARDINAL_VALUES:
            continue
        matched = True
        value = _CARDINAL_VALUES[word]
        if value >= 100:
            total = (total or 1) * value
            last_scale = value
        else:
            total += value * last_scale if last_scale > 1 and total and value < last_scale else value
    if not matched:
        return None
    return total, lower_bound


def _quantity_keys(text: str) -> set[tuple[int, bool]]:
    keys: set[tuple[int, bool]] = set()
    blob = str(text or "")
    for span in explicit_quantity_spans(blob):
        parsed = _quantity_value(span)
        if parsed is not None:
            keys.add(parsed)
    for match in re.finditer(r"\b(\d+)\s*\+", blob):
        keys.add((int(match.group(1)), True))
    parsed = _quantity_value(blob)
    if parsed is not None and not keys:
        keys.add(parsed)
    return keys


def _quantities_compatible(claim_keys: set[tuple[int, bool]], record_keys: set[tuple[int, bool]]) -> bool:
    if not claim_keys:
        return True
    if not record_keys:
        return False
    claim_values = {value for value, _flag in claim_keys}
    record_values = {value for value, _flag in record_keys}
    return bool(claim_values & record_values)


def _record_from_mapping(row: Any) -> PropositionRecord | None:
    if isinstance(row, PropositionRecord):
        return row
    if not isinstance(row, dict) or not row.get("proposition_id"):
        return None
    return PropositionRecord(
        proposition_id=str(row.get("proposition_id", "")),
        claim=str(row.get("claim", "")),
        proposition_type=str(row.get("proposition_type", "")),
        epistemic_status=str(row.get("epistemic_status", "")),
        support_ids=list(row.get("support_ids", []) or []),
        derived_from=list(row.get("derived_from", []) or []),
        introduced_by=str(row.get("introduced_by", "")),
        mention_count=int(row.get("mention_count", 1) or 1),
        decision_critical_mentions=int(row.get("decision_critical_mentions", 0) or 0),
        epistemic_type=str(row.get("epistemic_type", "")),
        aliases=list(row.get("aliases", []) or []),
        action_id=str(row.get("action_id", "")),
        outcome=str(row.get("outcome", "")),
        polarity=str(row.get("polarity", "")),
        party_labels=[str(value) for value in row.get("party_labels", []) or []],
        quantities=[str(value) for value in row.get("quantities", []) or []],
        effect_kind=str(row.get("effect_kind", "")),
        context_terms=[str(value) for value in row.get("context_terms", []) or []],
    )


def _is_world_established(record: PropositionRecord) -> bool:
    if record.epistemic_type == "WORLD_ESTABLISHED":
        return True
    if record.proposition_id.startswith("PROP:WORLD:"):
        return record.epistemic_status in {"ESTABLISHED", "DERIVED"}
    return (
        record.epistemic_status in {"ESTABLISHED", "DERIVED"}
        and record.proposition_type == "DESCRIPTIVE"
    )


def _remember_alias(record: PropositionRecord, claim: str) -> None:
    cleaned = " ".join(str(claim).split())[:240]
    if not cleaned:
        return
    if _normalized_claim(cleaned) in {
        _normalized_claim(record.claim),
        *(_normalized_claim(item) for item in record.aliases),
    }:
        return
    record.aliases = [*record.aliases, cleaned][:8]


def _incoming_edges(graph: SemanticGraph, node_id: str, relations: set[str]) -> list[Any]:
    return [
        edge for edge in graph.edges
        if edge.target == node_id and edge.relation in relations
    ]


def _collect_context_terms(graph: SemanticGraph, node_id: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = {node_id}
    stack = [node_id]
    while stack:
        current = stack.pop()
        node = graph.nodes.get(current)
        if node is not None and node.label:
            terms.append(node.label)
        for edge in graph.outgoing(current, "HAS_INTERVENTION"):
            target = graph.nodes.get(edge.target)
            if target is not None:
                terms.append(target.label)
        for edge in _incoming_edges(
            graph, current, _CAUSAL_RELATIONS | {"HAS_CONSEQUENCE"},
        ):
            if edge.source in seen:
                continue
            seen.add(edge.source)
            stack.append(edge.source)
            source = graph.nodes.get(edge.source)
            if source is not None and source.label:
                terms.append(source.label)
    return list(dict.fromkeys(term for term in terms if term))


def _annotate_binding_context(
    ledger: dict[str, PropositionRecord], graph: SemanticGraph,
) -> None:
    consequence_by_effect: dict[str, str] = {}
    for node in graph.nodes.values():
        world_effect_id = str(node.attributes.get("world_effect_id", "")).strip()
        if node.kind == "CONSEQUENCE" and world_effect_id:
            consequence_by_effect[world_effect_id] = node.id
    for record in ledger.values():
        if not _is_world_established(record):
            continue
        node_id = ""
        for support_id in record.support_ids:
            node_id = consequence_by_effect.get(str(support_id), "")
            if node_id:
                break
            if str(support_id) in graph.nodes:
                node_id = str(support_id)
                break
        if not node_id and record.action_id in graph.nodes:
            node_id = record.action_id
        if not node_id:
            continue
        record.context_terms = _collect_context_terms(graph, node_id)


def _seed_unknown_parameters(
    ledger: dict[str, PropositionRecord], graph: SemanticGraph,
) -> None:
    for node in graph.nodes.values():
        if node.kind != "CONDITION":
            continue
        if str(node.attributes.get("value_status", "")).upper() != "UNKNOWN":
            continue
        proposition_id = f"PROP:WORLD:CONDITION:{node.id}"
        if proposition_id in ledger:
            continue
        ledger[proposition_id] = PropositionRecord(
            proposition_id=proposition_id,
            claim=node.label,
            proposition_type="DESCRIPTIVE",
            epistemic_status="UNRESOLVED",
            support_ids=[node.id],
            introduced_by="WORLD_MODEL",
            epistemic_type="UNKNOWN_PARAMETER",
        )


def _party_mentions(claim: str, ledger: dict[str, PropositionRecord]) -> set[str]:
    blob = _normalized_claim(claim)
    mentions: set[str] = set()
    for record in ledger.values():
        for label in record.party_labels:
            token = _normalized_claim(label)
            if token and token in blob:
                mentions.add(token)
            for word in _content_words(label):
                if len(word) > 3 and word in blob:
                    mentions.add(word)
    return mentions


def _semantic_match(
    claim: str,
    record: PropositionRecord,
    *,
    ledger: dict[str, PropositionRecord],
) -> int | None:
    if record.polarity == "FOREGONE" and not re.search(
        r"\b(?:forego|forgo|foregone|forgone|instead|missed)\b", claim, re.I,
    ):
        return None
    if any(
        _claim_is_covered(claim, item)
        for item in (record.claim, *record.aliases)
    ):
        return 100
    outcome = record.outcome or record.claim.split(";")[0]
    claim_families = _outcome_families(claim)
    record_families = _outcome_families(" ".join((outcome, record.claim)))
    families = claim_families & record_families
    if claim_families and not families:
        return None
    implied = _implied_polarity(claim, families)
    if (
        implied
        and record.polarity in {"BENEFICIAL", "ADVERSE"}
        and implied != record.polarity
    ):
        return None
    claim_quantities = _quantity_keys(claim)
    record_quantities = _quantity_keys(" ".join((
        " ".join(record.quantities), record.claim, " ".join(record.party_labels),
    )))
    if claim_quantities and not _quantities_compatible(claim_quantities, record_quantities):
        return None
    mentioned_parties = _party_mentions(claim, ledger)
    record_party_words = {
        word
        for label in record.party_labels
        for word in _content_words(label)
    } | _content_words(" ".join(record.party_labels))
    if mentioned_parties and record_party_words:
        if not (mentioned_parties & record_party_words) and not any(
            label in _normalized_claim(claim) for label in (
                _normalized_claim(item) for item in record.party_labels
            )
        ):
            return None
    if not families and not (claim_quantities and record_quantities):
        return None
    context_words = _content_words(" ".join((
        record.claim, record.outcome, " ".join(record.party_labels),
        " ".join(record.context_terms), " ".join(record.quantities),
        " ".join(record.aliases),
    )))
    family_words = {
        word
        for index in (
            families
            | _outcome_families(" ".join((outcome, record.claim, " ".join(record.context_terms))))
        )
        for word in _OUTCOME_FAMILIES[index]
    }
    extra = _content_words(claim) - context_words - family_words
    extra -= {str(value) for value, _flag in claim_quantities}
    extra -= {word for word in extra if word.isdigit()}
    if extra & _STRENGTHENING_WORDS:
        return None
    extra_families = _outcome_families(" ".join(extra)) - _outcome_families(
        " ".join((outcome, record.claim, " ".join(record.context_terms)))
    )
    if extra_families:
        return None
    leftover = extra - _STRENGTHENING_WORDS
    if leftover:
        return None
    score = 10 * len(families) + 5 * int(bool(claim_quantities & record_quantities))
    score += 15 * len(_stems(claim) & _stems(outcome))
    if implied and implied == record.polarity:
        score += 12
    if record.polarity == "NEUTRAL":
        score -= 6
    if re.search(r":(?:TEMPORAL|LIKELIHOOD|SCOPE):", record.proposition_id):
        score -= 20
    if mentioned_parties & record_party_words:
        score += 8
    if record.action_id and record.action_id.casefold() in claim.casefold():
        score += 4
    return score if score > 0 else None


def resolve_proposition(
    ledger: dict[str, PropositionRecord],
    claim: str,
    *,
    preferred: str = "",
) -> str:
    """Return a canonical proposition ID for a paraphrase, or empty if none."""
    cleaned = " ".join(str(claim).split())[:240]
    if not cleaned or cleaned.upper() == "NONE":
        return ""
    preferred_record = ledger.get(preferred)
    if preferred_record is not None and _claim_is_covered(cleaned, preferred_record.claim):
        return preferred
    for proposition_id, record in ledger.items():
        if _claim_is_covered(cleaned, record.claim):
            return proposition_id
        if any(_claim_is_covered(cleaned, alias) for alias in record.aliases):
            return proposition_id
    ranked: list[tuple[int, str]] = []
    world_ids = [
        proposition_id for proposition_id, record in ledger.items()
        if _is_world_established(record)
    ]
    hypothesis_ids = [
        proposition_id for proposition_id, record in ledger.items()
        if record.epistemic_type == "HYPOTHESIS"
        or record.proposition_type == "HYPOTHESIS"
    ]
    for proposition_id in (*world_ids, *hypothesis_ids):
        record = ledger[proposition_id]
        score = _semantic_match(cleaned, record, ledger=ledger)
        if score is None:
            continue
        ranked.append((score, proposition_id))
    if not ranked:
        return ""
    ranked.sort(reverse=True)
    best_score, best_id = ranked[0]
    if preferred_record is not None:
        preferred_score = _semantic_match(cleaned, preferred_record, ledger=ledger)
        if preferred_score is not None:
            return preferred
    ties = [item_id for score, item_id in ranked if score == best_score]
    if len(ties) > 1:
        world_ties = [item_id for item_id in ties if item_id in world_ids]
        if len(world_ties) == 1:
            return world_ties[0]
        return ""
    return best_id


def claim_matches_established(
    claim: str, ledger: dict[str, Any] | dict[str, PropositionRecord],
) -> bool:
    """True when claim restates an established or derived ledger atom."""
    records: dict[str, PropositionRecord] = {}
    for key, value in ledger.items():
        record = _record_from_mapping(value)
        if record is not None:
            records[str(key)] = record
    bound = resolve_proposition(records, claim)
    if not bound:
        return False
    return records[bound].epistemic_status in {"ESTABLISHED", "DERIVED"}


def register_hypothesis(
    ledger: dict[str, PropositionRecord],
    claim: str,
    *,
    specialist: str,
    derived_from: Iterable[str] = (),
    decision_critical: bool = False,
) -> str:
    """Register or mention a hypothesis without ever promoting its status."""
    cleaned = " ".join(str(claim).split())[:240]
    if not cleaned or cleaned.upper() == "NONE":
        return ""
    known_dependencies = [
        value for value in dict.fromkeys(str(item) for item in derived_from)
        if value in ledger
    ]
    bound = resolve_proposition(ledger, cleaned)
    if bound:
        existing = ledger[bound]
        _remember_alias(existing, cleaned)
        existing.mention_count += 1
        existing.decision_critical_mentions += 1 if decision_critical else 0
        existing.derived_from = list(dict.fromkeys([
            *existing.derived_from, *known_dependencies,
        ]))
        return bound
    proposition_id = _stable_id("HYPOTHESIS", cleaned)
    existing = ledger.get(proposition_id)
    if existing is None:
        ledger[proposition_id] = PropositionRecord(
            proposition_id=proposition_id,
            claim=cleaned,
            proposition_type="HYPOTHESIS",
            epistemic_status="HYPOTHETICAL",
            derived_from=known_dependencies,
            introduced_by=specialist,
            decision_critical_mentions=1 if decision_critical else 0,
            epistemic_type="HYPOTHESIS",
        )
    else:
        existing.mention_count += 1
        existing.decision_critical_mentions += 1 if decision_critical else 0
        existing.derived_from = list(dict.fromkeys([
            *existing.derived_from, *known_dependencies,
        ]))
        # Deliberately no status mutation: recurrence is attention, not evidence.
    return proposition_id


def register_derived_proposition(
    ledger: dict[str, PropositionRecord],
    claim: str,
    *,
    specialist: str,
    derived_from: Iterable[str],
    decision_critical: bool = False,
) -> str:
    """Register a transparent composition without promoting any dependency."""
    cleaned = " ".join(str(claim).split())[:240]
    dependencies = [
        value for value in dict.fromkeys(str(item) for item in derived_from)
        if value in ledger
    ]
    if not cleaned:
        return ""
    dependency_statuses_are_authoritative = all(
        ledger[value].epistemic_status in {"ESTABLISHED", "DERIVED"}
        for value in dependencies
    )
    dependency_words = {
        word
        for value in dependencies
        for word in re.findall(r"[a-z0-9]+", ledger[value].claim.casefold())
        if word not in _COMPOSITION_STOPWORDS
    }
    claim_words = {
        word for word in re.findall(r"[a-z0-9]+", cleaned.casefold())
        if word not in _COMPOSITION_STOPWORDS
    }
    # A model may propose a transparent composition, but only this deterministic
    # coverage check can admit it as DERIVED. A single proposition should be
    # cited directly; new vocabulary remains a hypothesis.
    transparent_composition = (
        len(dependencies) >= 2
        and dependency_statuses_are_authoritative
        and claim_words <= dependency_words
    )
    if not transparent_composition:
        return register_hypothesis(
            ledger, cleaned, specialist=specialist,
            derived_from=dependencies,
            decision_critical=decision_critical,
        )
    proposition_id = _stable_id("DERIVED", cleaned)
    existing = ledger.get(proposition_id)
    if existing is None:
        ledger[proposition_id] = PropositionRecord(
            proposition_id=proposition_id,
            claim=cleaned,
            proposition_type="DESCRIPTIVE",
            epistemic_status="DERIVED",
            derived_from=dependencies,
            introduced_by=specialist,
            decision_critical_mentions=1 if decision_critical else 0,
            epistemic_type="WORLD_ESTABLISHED",
        )
    else:
        existing.mention_count += 1
        existing.decision_critical_mentions += 1 if decision_critical else 0
        existing.derived_from = list(dict.fromkeys([
            *existing.derived_from, *dependencies,
        ]))
    return proposition_id


def register_framework_derived_proposition(
    ledger: dict[str, PropositionRecord],
    claim: str,
    *,
    specialist: str,
    derived_from: Iterable[str] = (),
    decision_critical: bool = False,
) -> str:
    """Register a framework-native relation without minting an empirical hypothesis."""
    cleaned = " ".join(str(claim).split())[:240]
    if not cleaned:
        return ""
    known_dependencies = [
        value for value in dict.fromkeys(str(item) for item in derived_from)
        if value in ledger
    ]
    bound = resolve_proposition(ledger, cleaned)
    if bound:
        existing = ledger[bound]
        _remember_alias(existing, cleaned)
        existing.mention_count += 1
        existing.decision_critical_mentions += 1 if decision_critical else 0
        existing.derived_from = list(dict.fromkeys([
            *existing.derived_from, *known_dependencies,
        ]))
        return bound
    proposition_id = _stable_id("FRAMEWORK", cleaned)
    existing = ledger.get(proposition_id)
    if existing is None:
        ledger[proposition_id] = PropositionRecord(
            proposition_id=proposition_id,
            claim=cleaned,
            proposition_type="NORMATIVE",
            epistemic_status="DERIVED",
            derived_from=known_dependencies,
            introduced_by=specialist,
            decision_critical_mentions=1 if decision_critical else 0,
            epistemic_type="FRAMEWORK_DERIVED",
        )
    else:
        existing.mention_count += 1
        existing.decision_critical_mentions += 1 if decision_critical else 0
        existing.derived_from = list(dict.fromkeys([
            *existing.derived_from, *known_dependencies,
        ]))
    return proposition_id


def weakest_status(
    ledger: dict[str, PropositionRecord], proposition_ids: Iterable[str],
) -> str:
    records = [ledger.get(str(value)) for value in proposition_ids]
    records = [record for record in records if record is not None]
    if not records:
        return "ESTABLISHED"
    return min(
        (record.epistemic_status for record in records),
        key=lambda status: EPISTEMIC_STATUS_RANK.get(status, 0),
    )


def _record_is_hypothesis(record: PropositionRecord | None) -> bool:
    if record is None:
        return False
    if record.epistemic_type == "HYPOTHESIS":
        return True
    return record.epistemic_status == "HYPOTHETICAL"


def _record_blocks_closed_world(record: PropositionRecord | None) -> bool:
    """Unknown admitted parameters and rejected claims can unsettle the ranking.

    A HYPOTHESIS cannot: it is a reversal boundary, not a closed-world score input.
    """
    if record is None or _record_is_hypothesis(record):
        return False
    if record.epistemic_type == "FRAMEWORK_DERIVED":
        return False
    return record.epistemic_status in DECISION_CRITICAL_CAP_STATUSES


def _hypothesis_reversal_text(records: list[PropositionRecord]) -> str:
    claims = [
        " ".join(record.claim.split())
        for record in records
        if " ".join(record.claim.split())
    ]
    if not claims:
        return ""
    joined = "; ".join(claims[:2])
    return (
        "Reverse the admitted ranking if this unestablished consequence is "
        f"verified and exceeds the admitted welfare margin: {joined}"
    )[:180]


def _attach_hypothesis_boundary(candidate: Any, records: list[PropositionRecord]) -> None:
    text = _hypothesis_reversal_text(records)
    if not text:
        return
    current_ft = " ".join(str(getattr(candidate, "factual_reversal_threshold", "") or "").split())
    if not current_ft or current_ft.casefold() == "none":
        candidate.factual_reversal_threshold = text
    unresolved = str(getattr(candidate, "unresolved", "NONE") or "NONE").upper()
    if unresolved in {"NONE", "VERIFY_FACTS"}:
        candidate.unresolved = "DECISION_BOUNDARY"
    question = (
        "Would verifying this unestablished consequence reverse the admitted "
        f"ranking: {records[0].claim}?"
    )[:240]
    questions = [
        " ".join(str(item).split())
        for item in list(getattr(candidate, "framework_specific_open_questions", []) or [])
        if " ".join(str(item).split())
    ]
    if question not in questions:
        candidate.framework_specific_open_questions = [*questions, question][:3]
    if not str(getattr(candidate, "investigative_claim", "") or "").strip():
        candidate.investigative_claim = question


def _restore_closed_world_utilitarian_ranking(candidate: Any) -> None:
    if str(getattr(candidate, "specialist", "")).casefold() != "utilitarian":
        return
    from .local_specialists import (
        align_action_scores_to_leader,
        closed_world_utilitarian_leader,
    )

    scores = dict(getattr(candidate, "action_scores", {}) or {})
    actions = list(scores)
    leader = closed_world_utilitarian_leader(
        actions,
        dict(getattr(candidate, "utilitarian_consequence_table", {}) or {}),
        dict(getattr(candidate, "expected_value_estimates", {}) or {}),
    )
    if leader is None or leader not in scores:
        return
    aligned = align_action_scores_to_leader(scores, leader)
    candidate.action_scores = aligned
    candidate.recommended_action = leader
    ordered = sorted(aligned.values(), reverse=True)
    candidate.preference_strength = max(
        0.0,
        min(1.0, ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]),
    )
    candidate.friction = candidate.preference_strength
    candidate.utilitarian_decision_depends_on_unknown = False
    candidate.comparison_complete = True
    notes = list(getattr(candidate, "epistemic_binding_notes", []) or [])
    note = (
        "Closed-world ranking uses admitted consequences only; unestablished "
        "hypotheses remain reversal boundaries."
    )
    candidate.epistemic_binding_notes = list(dict.fromkeys([*notes, note]))[:12]


def _apply_closed_world_incompleteness_cap(candidate: Any) -> None:
    if str(getattr(candidate, "assumption_status", "")).upper() != "NORMATIVELY_CONTESTED":
        candidate.assumption_status = "UNDERDETERMINED"
    if str(getattr(candidate, "unresolved", "NONE")).upper() in {"NONE", "DECISION_BOUNDARY"}:
        candidate.unresolved = "VERIFY_FACTS"
    candidate.selection_status = "PROVISIONAL"
    candidate.comparison_complete = False
    candidate.evidence_sufficient_for_action = False
    candidate.epistemic_confidence = min(
        float(getattr(candidate, "epistemic_confidence", 1.0)), 0.50,
    )
    candidate.confidence = candidate.epistemic_confidence


def _apply_candidate_authority_cap(
    ledger: dict[str, PropositionRecord], candidate: Any,
) -> None:
    status = weakest_status(ledger, candidate.decision_critical_proposition_ids)
    candidate.weakest_decision_critical_status = status
    candidate.decision_critical_dependency_claims = [
        ledger[value].claim for value in candidate.decision_critical_proposition_ids
        if value in ledger
    ][:4]
    records = [
        ledger[value]
        for value in candidate.decision_critical_proposition_ids
        if value in ledger
    ]
    hypothesis_records = [record for record in records if _record_is_hypothesis(record)]
    blocking_records = [record for record in records if _record_blocks_closed_world(record)]
    if hypothesis_records:
        _attach_hypothesis_boundary(candidate, hypothesis_records)
        candidate.epistemic_confidence = min(
            float(getattr(candidate, "epistemic_confidence", 1.0)),
            HYPOTHESIS_OPEN_WORLD_CONFIDENCE_CAP,
        )
        candidate.confidence = min(
            float(getattr(candidate, "confidence", candidate.epistemic_confidence)),
            candidate.epistemic_confidence,
        )
        _restore_closed_world_utilitarian_ranking(candidate)
    if blocking_records:
        _apply_closed_world_incompleteness_cap(candidate)


def attach_candidate_dependencies(
    ledger: dict[str, PropositionRecord], candidate: Any,
) -> None:
    """Resolve candidate references and impose the decision-critical status cap."""
    submitted_cited = list(dict.fromkeys(
        str(item) for item in getattr(candidate, "supporting_proposition_ids", [])
    ))
    submitted_critical = list(dict.fromkeys(
        str(item) for item in getattr(candidate, "decision_critical_proposition_ids", [])
    ))
    unknown = (set(submitted_cited) | set(submitted_critical)) - set(ledger)
    if unknown:
        candidate.schema_valid = False
        candidate.validation_errors = list(dict.fromkeys([
            *list(getattr(candidate, "validation_errors", []) or []),
            f"candidate cites unknown proposition IDs: {sorted(unknown)}",
        ]))
        return
    cited = [
        value for value in dict.fromkeys(
            submitted_cited
        ) if value in ledger
    ]
    critical = [
        value for value in dict.fromkeys(
            submitted_critical
        ) if value in ledger
    ]
    registered_hypotheses: set[str] = set()
    registered_claims: dict[str, str] = {}
    processed_premises: set[tuple[str, str, bool]] = set()
    binding_notes = list(getattr(candidate, "epistemic_binding_notes", []) or [])
    for premise in list(getattr(candidate, "material_empirical_claims", []) or []):
        if not isinstance(premise, dict):
            continue
        claim = " ".join(str(premise.get("claim", "")).split())[:240]
        basis = str(premise.get("proposition_id", "HYPOTHESIS")).strip()
        decision_critical = premise.get("decision_critical") is True
        premise_key = (_normalized_claim(claim), basis, decision_critical)
        if premise_key in processed_premises:
            continue
        processed_premises.add(premise_key)
        authoritative = ledger.get(basis)
        bound = resolve_proposition(
            ledger, claim, preferred=basis if authoritative is not None else "",
        )
        if bound:
            record = ledger[bound]
            _remember_alias(record, claim)
            record.mention_count += 1
            if decision_critical:
                record.decision_critical_mentions += 1
            cited.append(bound)
            if decision_critical:
                critical.append(bound)
            premise["proposition_id"] = bound
            premise["canonical_proposition"] = bound
            registered_hypotheses.add(bound)
            registered_claims[_normalized_claim(claim)] = bound
            if bound != basis and authoritative is not None:
                binding_notes.append(
                    f"Premise rebound from {basis} to canonical {bound}."
                )
            continue
        derived_from = [basis] if authoritative is not None else cited
        existing_ids = set(ledger)
        hypothesis_id = register_hypothesis(
            ledger, claim,
            specialist=str(getattr(candidate, "specialist", "unknown")),
            derived_from=derived_from,
            decision_critical=decision_critical,
        )
        if not hypothesis_id:
            continue
        registered_hypotheses.add(hypothesis_id)
        registered_claims[_normalized_claim(claim)] = hypothesis_id
        cited.append(hypothesis_id)
        if decision_critical:
            critical.append(hypothesis_id)
        premise["proposition_id"] = hypothesis_id
        premise["canonical_proposition"] = hypothesis_id
        if hypothesis_id not in existing_ids and authoritative is not None:
            binding_notes.append(
                f"Premise strengthened {basis}; reclassified as {hypothesis_id}."
            )
    speculative = str(getattr(candidate, "speculative_claim", "") or "").strip()
    tier = str(getattr(candidate, "evidence_calibration_tier", "") or "").upper()
    if speculative and speculative.upper() != "NONE":
        hypothesis_id = registered_claims.get(_normalized_claim(speculative), "")
        critical_speculation = tier in {"DECISION_CRITICAL", "REMOTE"}
        if not hypothesis_id:
            hypothesis_id = register_hypothesis(
                ledger, speculative,
                specialist=str(getattr(candidate, "specialist", "unknown")),
                derived_from=cited,
                decision_critical=critical_speculation,
            )
            registered_hypotheses.add(hypothesis_id)
        elif critical_speculation and hypothesis_id not in critical:
            ledger[hypothesis_id].decision_critical_mentions += 1
        cited.append(hypothesis_id)
        if critical_speculation:
            critical.append(hypothesis_id)
    candidate.supporting_proposition_ids = list(dict.fromkeys(cited))
    candidate.decision_critical_proposition_ids = list(dict.fromkeys(critical))
    candidate.epistemic_binding_notes = list(dict.fromkeys(binding_notes))[:12]
    for proposition_id in candidate.supporting_proposition_ids:
        if proposition_id in registered_hypotheses:
            continue
        ledger[proposition_id].mention_count += 1
        if proposition_id in candidate.decision_critical_proposition_ids:
            ledger[proposition_id].decision_critical_mentions += 1
    _apply_candidate_authority_cap(ledger, candidate)


def apply_side_premise_audit(
    ledger: dict[str, PropositionRecord],
    candidates: Iterable[Any],
    audit: dict[str, Any],
) -> None:
    """Attach independent audit findings and conservatively handle audit failure."""
    candidate_list = [
        candidate for candidate in candidates
        if bool(getattr(candidate, "schema_valid", True))
    ]
    by_specialist = {
        str(getattr(candidate, "specialist", "")): candidate
        for candidate in candidate_list
    }
    status = str((audit or {}).get("status", "UNAVAILABLE")).strip().upper()
    if status not in {"PASSED", "FINDINGS", "UNAVAILABLE"}:
        status = "UNAVAILABLE"
    if status == "UNAVAILABLE":
        error = " ".join(str((audit or {}).get("error", "")).split())[:220]
        for candidate in candidate_list:
            candidate.side_premise_audit_status = "UNAVAILABLE"
            candidate.weakest_decision_critical_status = "UNRESOLVED"
            candidate.decision_critical_dependency_claims = list(dict.fromkeys([
                *candidate.decision_critical_dependency_claims,
                "independent empirical-premise coverage remains unverified",
            ]))[:4]
            if str(getattr(candidate, "assumption_status", "")).upper() != "NORMATIVELY_CONTESTED":
                candidate.assumption_status = "UNDERDETERMINED"
            if str(getattr(candidate, "unresolved", "NONE")).upper() == "NONE":
                candidate.unresolved = "VERIFY_FACTS"
            candidate.selection_status = "PROVISIONAL"
            candidate.comparison_complete = False
            candidate.evidence_sufficient_for_action = False
            candidate.epistemic_confidence = min(candidate.epistemic_confidence, 0.50)
            candidate.confidence = candidate.epistemic_confidence
            candidate.epistemic_binding_notes = list(dict.fromkeys([
                *candidate.epistemic_binding_notes,
                f"Independent side-premise audit unavailable: {error or 'unknown error'}",
            ]))[:12]
        return

    findings_by_specialist: dict[str, list[dict[str, Any]]] = {}
    for raw in list((audit or {}).get("findings", []) or []):
        if not isinstance(raw, dict):
            continue
        specialist = str(raw.get("specialist", ""))
        candidate = by_specialist.get(specialist)
        if candidate is None:
            continue
        claim = " ".join(str(raw.get("claim", "")).split())[:240]
        if not claim:
            continue
        critical = raw.get("decision_critical") is True
        binding = str(raw.get("binding", "NEW_HYPOTHESIS"))
        derived_from = [
            str(value) for value in raw.get("derived_from", [])
            if str(value) in ledger
        ] if isinstance(raw.get("derived_from", []), list) else []
        already_cited = set(candidate.supporting_proposition_ids)
        already_critical = set(candidate.decision_critical_proposition_ids)
        covered_id = resolve_proposition(ledger, claim)
        if covered_id:
            binding = covered_id
            _remember_alias(ledger[covered_id], claim)
        if binding in ledger:
            proposition_id = binding
            if proposition_id not in already_cited:
                ledger[proposition_id].mention_count += 1
            if critical and proposition_id not in already_critical:
                ledger[proposition_id].decision_critical_mentions += 1
        elif binding == "DERIVED_ESTABLISHED":
            proposition_id = register_derived_proposition(
                ledger, claim, specialist=specialist,
                derived_from=derived_from,
                decision_critical=critical,
            )
        else:
            proposition_id = register_hypothesis(
                ledger, claim, specialist=specialist,
                derived_from=derived_from or candidate.supporting_proposition_ids,
                decision_critical=critical,
            )
        candidate.supporting_proposition_ids = list(dict.fromkeys([
            *candidate.supporting_proposition_ids, proposition_id,
        ]))
        if critical:
            candidate.decision_critical_proposition_ids = list(dict.fromkeys([
                *candidate.decision_critical_proposition_ids, proposition_id,
            ]))
        normalized = {
            "claim": claim,
            "proposition_id": proposition_id,
            "binding": binding,
            "derived_from": derived_from,
            "decision_critical": critical,
            "source_field": " ".join(str(raw.get("source_field", "")).split())[:80],
            "reason": " ".join(str(raw.get("reason", "")).split())[:180],
        }
        findings_by_specialist.setdefault(specialist, []).append(normalized)

    for candidate in candidate_list:
        findings = findings_by_specialist.get(candidate.specialist, [])
        candidate.side_premise_audit_status = "FINDINGS" if findings else "PASSED"
        candidate.side_premise_audit_findings = findings[:12]
        _apply_candidate_authority_cap(ledger, candidate)


def focus_proposition_ids(
    ledger: dict[str, PropositionRecord], candidates: Iterable[Any], *, limit: int = 4,
) -> tuple[str, ...]:
    """Rank unresolved propositions for attention without promoting them."""
    agents_by_id: dict[str, set[str]] = {}
    for candidate in candidates:
        if not bool(getattr(candidate, "schema_valid", True)):
            continue
        agent = str(getattr(candidate, "specialist", "unknown"))
        for proposition_id in getattr(candidate, "decision_critical_proposition_ids", []):
            record = ledger.get(str(proposition_id))
            if record is None or record.epistemic_status not in {
                "REJECTED", "HYPOTHETICAL", "UNRESOLVED",
            }:
                continue
            agents_by_id.setdefault(record.proposition_id, set()).add(agent)
    ranked = sorted(
        agents_by_id,
        key=lambda proposition_id: (
            len(agents_by_id[proposition_id]),
            ledger[proposition_id].decision_critical_mentions,
            ledger[proposition_id].mention_count,
            proposition_id,
        ),
        reverse=True,
    )
    return tuple(ranked[:max(0, limit)])


def shared_unresolved_dependency_projection(
    ledger: dict[str, PropositionRecord], candidates: Iterable[Any],
) -> list[dict[str, Any]]:
    """Expose correlated support resting on the same unresolved proposition."""
    agents_by_id: dict[str, set[str]] = {}
    for candidate in candidates:
        if not bool(getattr(candidate, "schema_valid", True)):
            continue
        for proposition_id in getattr(candidate, "decision_critical_proposition_ids", []):
            record = ledger.get(str(proposition_id))
            if record is None or record.epistemic_status not in {
                "REJECTED", "HYPOTHETICAL", "UNRESOLVED",
            }:
                continue
            agents_by_id.setdefault(record.proposition_id, set()).add(
                str(getattr(candidate, "specialist", "unknown"))
            )
    rows = [{
        "proposition_id": proposition_id,
        "claim": ledger[proposition_id].claim,
        "epistemic_status": ledger[proposition_id].epistemic_status,
        "dependent_specialists": sorted(agents),
        "dependent_specialist_count": len(agents),
        "mention_count": ledger[proposition_id].mention_count,
    } for proposition_id, agents in agents_by_id.items()]
    return sorted(
        rows,
        key=lambda row: (
            int(row["dependent_specialist_count"]), int(row["mention_count"]),
            str(row["proposition_id"]),
        ),
        reverse=True,
    )
