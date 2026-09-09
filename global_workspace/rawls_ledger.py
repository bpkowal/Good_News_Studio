"""Transactional Rawlsian comparative-position ledger.

Delegate prose is a proposal, not graph state.  This module binds Rawlsian
position claims to canonical action and target nodes, preserves grounded mixed
comparative states, normalizes grounded tradeoff claims to MIXED, weakens
unsupported directional claims to UNCERTAIN, and commits the normalized ledger
atomically.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from .graph_transactions import GraphTransactionRecord, SemanticGraphStore
from .scenario_semantics import query_grounded_action_effects
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode, merge_graphs, validate_graph


RawlsEffect = Literal["IMPROVES", "PRESERVES", "WORSENS", "MIXED", "UNCERTAIN"]
RawlsDimension = Literal[
    "BASIC_LIBERTY", "OPPORTUNITY", "INCOME_WEALTH", "POWERS_OFFICES",
    "SELF_RESPECT", "BASIC_INTEREST_SECURITY", "OTHER_PRIMARY_GOOD", "UNKNOWN",
]
RawlsBasicLibertyKind = Literal[
    "POLITICAL_LIBERTY", "SPEECH_ASSEMBLY", "CONSCIENCE_THOUGHT",
    "PERSONAL_FREEDOM_INTEGRITY", "PERSONAL_PROPERTY", "RULE_OF_LAW",
    "NOT_APPLICABLE", "UNRESOLVED",
]
RawlsInstitutionalRelation = Literal[
    "DIRECT_BASIC_STRUCTURE_RULE", "DIRECT_COERCIVE_RESTRICTION",
    "FAIR_VALUE_PRECONDITION", "MATERIAL_PRECONDITION",
    "NATURAL_CONTINGENCY", "NOT_APPLICABLE", "UNRESOLVED",
]


class RawlsPositionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: str = Field(pattern=r"^A\d+$")
    subject: str = Field(
        min_length=2,
        max_length=100,
        validation_alias=AliasChoices("subject", "group", "s"),
    )
    subject_kind: Literal[
        "INDIVIDUAL", "GROUP", "INSTITUTION", "FUTURE_POPULATION",
        "CONSTITUENCY", "UNKNOWN",
    ] = Field(
        default="UNKNOWN",
        validation_alias=AliasChoices("subject_kind", "sk"),
    )
    principle_basis: str = Field(
        default="",
        min_length=0,
        max_length=120,
        validation_alias=AliasChoices("principle_basis", "pb"),
    )
    dimension: RawlsDimension = Field(
        validation_alias=AliasChoices("dimension", "d"),
    )
    additional_dimensions: list[RawlsDimension] = Field(
        default_factory=list,
        validation_alias=AliasChoices("additional_dimensions", "ad", "secondary_dimensions"),
    )
    basic_liberty_kind: RawlsBasicLibertyKind = "UNRESOLVED"
    institutional_relation: RawlsInstitutionalRelation = "UNRESOLVED"
    effect: RawlsEffect
    compared_to_action_id: str = Field(pattern=r"^A\d+$")
    evidence_basis: Literal["ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"]
    reason: str = Field(min_length=4, max_length=180)

    @model_validator(mode="after")
    def distinct_comparison(self):
        if self.action_id == self.compared_to_action_id:
            raise ValueError("Rawlsian position must compare distinct actions")
        extras = [
            str(dimension).strip().upper()
            for dimension in self.additional_dimensions
        ]
        if len(extras) != len(set(extras)):
            raise ValueError("Rawlsian position repeats an additional dimension")
        if self.dimension in extras:
            raise ValueError("Rawlsian position cannot repeat its primary dimension as additional")
        if self.dimension == "BASIC_LIBERTY":
            if self.basic_liberty_kind == "NOT_APPLICABLE":
                raise ValueError("basic-liberty position must classify the liberty")
            if self.institutional_relation == "NOT_APPLICABLE":
                raise ValueError("basic-liberty position must classify its institutional relation")
        elif self.basic_liberty_kind not in {"NOT_APPLICABLE", "UNRESOLVED"}:
            raise ValueError("non-liberty position cannot assert a basic-liberty kind")
        self.additional_dimensions = extras
        return self


class RawlsLedgerProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ranking_basis: Literal[
        "LEXICAL_BASIC_LIBERTY", "FAIR_EQUALITY_OPPORTUNITY",
        "MAXIMIN_PRIMARY_GOODS", "DIFFERENCE_PRINCIPLE",
        "BASIC_INTEREST_SECURITY", "ORIGINAL_POSITION_PUBLIC_RULE", "UNRESOLVED",
    ]
    ranking_classification_justification: str = Field(min_length=8, max_length=240)
    lexical_priority_justification: str = Field(min_length=8, max_length=240)
    liberty_status: dict[str, Literal[
        "SATISFIED", "INFRINGED", "CONFLICTED", "UNKNOWN", "NOT_APPLICABLE",
    ]]
    positions: list[RawlsPositionProposal] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def unique_actions(self):
        identities = [
            (
                position.action_id,
                position.dimension,
                _canonical_group(position.subject),
            )
            for position in self.positions
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Rawlsian ledger repeats an action-dimension-subject position")
        liberty_positions = [
            position for position in self.positions
            if position.dimension == "BASIC_LIBERTY"
        ]
        if self.ranking_basis == "LEXICAL_BASIC_LIBERTY":
            if not liberty_positions:
                raise ValueError("lexical priority requires a classified basic-liberty position")
            if any(
                position.basic_liberty_kind == "UNRESOLVED"
                or position.institutional_relation not in {
                    "DIRECT_BASIC_STRUCTURE_RULE", "DIRECT_COERCIVE_RESTRICTION",
                }
                for position in liberty_positions
            ):
                raise ValueError(
                    "lexical priority requires a resolved liberty classification and direct institutional relation"
                )
        return self


_IGNORED_GROUP_WORDS = {
    "group", "groups", "people", "person", "persons", "affected", "least",
    "advantaged", "disadvantaged", "worse", "off", "stakeholder", "stakeholders",
    "under", "action", "option", "those", "their", "with", "without",
}

_SUBJECT_TOKEN_ALIASES = {
    "team": "worker",
    "teams": "worker",
    "staff": "worker",
    "employee": "worker",
    "employees": "worker",
    "crew": "worker",
    "personnel": "worker",
    "worker": "worker",
    "workers": "worker",
    "floor": "worker",
    "fulfillment": "worker",
    "customer": "customer",
    "customers": "customer",
    "client": "customer",
    "clients": "customer",
    "user": "user",
    "users": "user",
    "resident": "resident",
    "residents": "resident",
    "patient": "patient",
    "patients": "patient",
    "school": "student",
    "schools": "student",
    "student": "student",
    "students": "student",
}


def _words(text: str) -> set[str]:
    return {
        _SUBJECT_TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(token) >= 3 and token not in _IGNORED_GROUP_WORDS
    }


def _subject_kind(subject: str) -> str:
    lowered = " ".join(str(subject).split()).casefold()
    if any(token in lowered for token in ("future population", "future generation", "descendant")):
        return "FUTURE_POPULATION"
    if any(token in lowered for token in ("institution", "board", "agency", "government", "system")):
        return "INSTITUTION"
    if any(token in lowered for token in ("constituency", "community", "stakeholder", "public")):
        return "CONSTITUENCY"
    if any(token in lowered for token in ("individual", "person", "resident", "patient", "worker", "family", "user", "citizen")):
        return "INDIVIDUAL"
    if any(token in lowered for token in ("group", "people", "persons")):
        return "GROUP"
    return "UNKNOWN"


def _resolve_action(graph: SemanticGraph, reference: str) -> SemanticNode | None:
    value = str(reference).strip()
    matches = [
        node for node in graph.nodes.values()
        if node.kind == "ACTION" and value in {
            node.id,
            node.label,
            str(node.attributes.get("canonical_action_id", "")),
            str(node.attributes.get("semantic_action_key", "")),
        }
    ]
    return matches[0] if len(matches) == 1 else None


def _action_consequences(
    graph: SemanticGraph, action_id: str,
) -> list[tuple[SemanticNode, list[SemanticNode]]]:
    values: list[tuple[SemanticNode, list[SemanticNode]]] = []
    for edge in graph.outgoing(action_id, "HAS_CONSEQUENCE"):
        consequence = graph.nodes.get(edge.target)
        if consequence is None or consequence.kind != "CONSEQUENCE":
            continue
        # A framework may inspect shared scenario structure, but another
        # delegate's interpretation is not scenario evidence. This prevents a
        # Utilitarian consequence proposal from silently becoming Rawlsian fact.
        if consequence.attributes.get("framework"):
            continue
        targets = [
            graph.nodes[target_edge.target]
            for target_edge in graph.outgoing(consequence.id, "AFFECTS")
            if target_edge.target in graph.nodes
            and graph.nodes[target_edge.target].kind == "TARGET"
        ]
        values.append((consequence, targets))
    return values


def _action_targets(graph: SemanticGraph, action_id: str) -> list[SemanticNode]:
    targets = [
        graph.nodes[edge.target]
        for edge in graph.outgoing(action_id, "TARGETS")
        if edge.target in graph.nodes and graph.nodes[edge.target].kind == "TARGET"
    ]
    for _, affected in _action_consequences(graph, action_id):
        targets.extend(affected)
    return list({target.id: target for target in targets}.values())


def _subject_matches(subject: str, targets: list[SemanticNode]) -> bool:
    subject_words = _words(subject)
    return bool(subject_words and any(subject_words & _words(target.label) for target in targets))


def _action_graph_can_bind_group(
    position: RawlsPositionProposal,
    own_consequences: list[SemanticNode],
    rival_consequences: list[SemanticNode],
) -> bool:
    if position.evidence_basis not in {"ACTION_GRAPH", "SCENARIO"}:
        return False
    # If the graph already contains concrete action-linked targets and the
    # delegate is comparing one action against another on that basis, we allow
    # the graph to bind the abstract Rawlsian group label instead of requiring
    # a literal lexical overlap with the scenario's wording.
    return bool(own_consequences or rival_consequences)


def _has_foreign_framework_consequences(graph: SemanticGraph, action_id: str) -> bool:
    for edge in graph.outgoing(action_id, "HAS_CONSEQUENCE"):
        consequence = graph.nodes.get(edge.target)
        if consequence is None or consequence.kind != "CONSEQUENCE":
            continue
        framework = str(consequence.attributes.get("framework", "")).strip().upper()
        if framework and framework != "RAWLSIAN":
            return True
    return False


def _comparison_action_targets(
    own_action_targets: list[SemanticNode],
    rival_action_targets: list[SemanticNode],
) -> list[SemanticNode]:
    """Collect the action-graph targets visible across both sides of a comparison.

    Rawlsian grounding is comparative: a position can be justified by a target
    that is explicit on the rival side even when the delegate's own action text
    only names the action mechanism. Using the union keeps world-state grounding
    available without letting unrelated framework notes block binding.
    """
    return list({
        target.id: target for target in (*own_action_targets, *rival_action_targets)
    }.values())


def _grounded_consequences(
    graph: SemanticGraph, action_id: str, subject: str,
) -> tuple[list[SemanticNode], list[SemanticNode]]:
    effects = query_grounded_action_effects(
        graph, action_id, affected_subject=subject,
    )
    if not effects:
        # Conservative compatibility fallback for abstract constituency labels
        # such as "hospital patients" whose action consequence names only the
        # institution ("hospital"). Canonical effects remain the first route.
        subject_words = _words(subject)
        consequences: list[SemanticNode] = []
        targets: list[SemanticNode] = []
        for consequence, affected_targets in _action_consequences(graph, action_id):
            matched = [
                target for target in affected_targets
                if subject_words & _words(target.label)
            ]
            if matched:
                consequences.append(consequence)
                targets.extend(matched)
        return consequences, list({target.id: target for target in targets}.values())
    consequences = [
        graph.nodes[effect.consequence_id]
        for effect in effects if effect.consequence_id in graph.nodes
    ]
    target_ids = {
        target_id for effect in effects
        for target_id in effect.affected_subject_node_ids
    }
    targets = [graph.nodes[target_id] for target_id in target_ids if target_id in graph.nodes]
    if not targets:
        subject_words = _words(subject)
        for consequence in consequences:
            targets.extend(
                graph.nodes[edge.target]
                for edge in graph.outgoing(consequence.id, "AFFECTS")
                if edge.target in graph.nodes
                and subject_words & _words(graph.nodes[edge.target].label)
            )
    return consequences, list({target.id: target for target in targets}.values())


_DIMENSION_TERMS: dict[str, set[str]] = {
    "BASIC_LIBERTY": {
        "autonomy", "bodily", "choice", "consent", "detain", "detention",
        "integrity", "liberty", "movement", "occupational", "privacy",
        "right", "rights", "speech", "vote", "worship",
    },
    "INCOME_WEALTH": {
        "aid", "asset", "assistance", "economic", "funding", "income",
        "material", "money", "property", "redistribution", "resource",
        "resources", "subsidy", "wage", "wealth",
    },
    "OPPORTUNITY": {
        "access", "career", "education", "employment", "office",
        "opportunity", "school", "training",
    },
    "POWERS_OFFICES": {"authority", "office", "power", "representation"},
    "SELF_RESPECT": {"dignity", "humiliation", "respect", "status"},
    "BASIC_INTEREST_SECURITY": {
        "food", "health", "housing", "medical", "safety", "security",
        "shelter", "survival", "water",
    },
}


def _dimension_compatible(node: SemanticNode, dimension: str) -> bool:
    """Require evidence to concern the position's Rawlsian dimension.

    Shared scenario facts remain framework-neutral, but an income benefit must
    not silently prove a liberty improvement (or vice versa).
    """
    if dimension not in {"BASIC_LIBERTY", "INCOME_WEALTH"}:
        return True
    attributes = node.attributes
    evidence_text = " ".join((
        node.label,
        str(attributes.get("predicate", "")),
        str(attributes.get("affected_resource", "")),
        " ".join(map(str, attributes.get("affected_resources", []) or [])),
        str(attributes.get("protected_interest", "")),
        " ".join(map(str, attributes.get("protected_interests", []) or [])),
        str(attributes.get("source_text", "")),
    )).casefold()
    words = set(re.findall(r"[a-z0-9]+", evidence_text))
    liberty_signal = bool(words & _DIMENSION_TERMS["BASIC_LIBERTY"])
    material_signal = bool(words & _DIMENSION_TERMS["INCOME_WEALTH"])
    if dimension == "BASIC_LIBERTY":
        return liberty_signal or not material_signal
    return material_signal or not liberty_signal


def _unambiguous_dimension_subject(
    pairs: list[tuple[SemanticNode, list[SemanticNode]]],
    dimension: str,
) -> tuple[str, list[SemanticNode]] | None:
    """Resolve one constituency explicitly attached to compatible evidence.

    This is deliberately conservative: correction is allowed only when all
    affected-subject targets form one lexically connected identity cluster.
    Protected interests and affected resources can never become the subject.
    """
    targets = list({
        target.id: target
        for consequence, affected_targets in pairs
        if _dimension_compatible(consequence, dimension)
        for target in affected_targets
        if target.attributes.get("semantic_role") == "AFFECTED_SUBJECT"
    }.values())
    if not targets:
        return None
    clusters: list[list[SemanticNode]] = []
    for target in targets:
        target_words = _words(target.label)
        matching = [
            cluster for cluster in clusters
            if any(target_words & _words(member.label) for member in cluster)
        ]
        if not matching:
            clusters.append([target])
            continue
        merged = [target]
        for cluster in matching:
            merged.extend(cluster)
            clusters.remove(cluster)
        clusters.append(merged)
    if len(clusters) != 1:
        return None
    cluster = clusters[0]
    canonical = min(
        (_canonical_group(target.label) for target in cluster),
        key=lambda label: (len(_words(label)), len(label), label),
    )
    return canonical, cluster


def _direction_supported(
    effect: str,
    own: list[SemanticNode],
    rival: list[SemanticNode],
) -> bool:
    if effect == "UNCERTAIN":
        return True
    own_polarities = {str(node.attributes.get("polarity", "")) for node in own}
    rival_polarities = {str(node.attributes.get("polarity", "")) for node in rival}
    own_predicates = {node.label.casefold() for node in own}
    if effect == "IMPROVES":
        return bool(
            ("BENEFICIAL" in own_polarities and "BENEFICIAL" not in rival_polarities)
            or ("ADVERSE" in rival_polarities and "ADVERSE" not in own_polarities)
        )
    if effect == "WORSENS":
        return bool(
            ("ADVERSE" in own_polarities and "ADVERSE" not in rival_polarities)
            or ("BENEFICIAL" in rival_polarities and "BENEFICIAL" not in own_polarities)
        )
    if effect == "PRESERVES":
        return bool(
            own_predicates & {"preserve", "preserve_life", "protect"}
            or "ADVERSE" in rival_polarities
        )
    if effect == "FOREGOES":
        return "FOREGONE" in own_polarities
    if effect == "MIXED":
        return bool(own or rival)
    return False


def _mixed_tradeoff_supported(
    own: list[SemanticNode],
    rival: list[SemanticNode],
) -> bool:
    """Detect whether the grounded comparison is genuinely tradeoff-shaped.

    We preserve tradeoff structure when either side carries both beneficial and
    adverse consequences on the grounded comparison target, because that is the
    Rawlsian situation where the model should be allowed to say "mixed" rather
    than being forced into a premature single-direction label.
    """
    own_polarities = {str(node.attributes.get("polarity", "")) for node in own}
    rival_polarities = {str(node.attributes.get("polarity", "")) for node in rival}
    return bool(
        ("BENEFICIAL" in own_polarities and "ADVERSE" in own_polarities)
        or ("BENEFICIAL" in rival_polarities and "ADVERSE" in rival_polarities)
    )


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _canonical_group(group: str) -> str:
    """Give semantically identical casing/spacing one stable graph label."""
    return " ".join(str(group).casefold().split()).strip(" ,.;:")


def apply_rawls_ledger_transaction(
    store: SemanticGraphStore,
    proposal: dict[str, Any],
    *,
    cycle: int,
    specialist: str,
    allowed_actions: tuple[str, ...],
) -> GraphTransactionRecord:
    """Normalize and commit a complete Rawlsian ledger as one transaction."""
    raw = dict(proposal) if isinstance(proposal, dict) else {"raw": proposal}
    try:
        validated = RawlsLedgerProposal.model_validate(proposal)
    except ValidationError as exc:
        errors = [
            f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        ]
        record = GraphTransactionRecord(
            cycle, specialist, "RAWLS_POSITION_LEDGER", "REJECTED", raw,
            errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    allowed_nodes = {
        node.id: node
        for action in allowed_actions
        if (node := _resolve_action(store.graph, action)) is not None
    }
    submitted_ids = {position.action_id for position in validated.positions}
    allowed_ids = {
        str(node.attributes.get("canonical_action_id", node.id))
        for node in allowed_nodes.values()
    }
    errors: list[str] = []
    if submitted_ids != allowed_ids:
        errors.append("Rawlsian ledger must cover exactly the canonical action set")
    if set(validated.liberty_status) != allowed_ids:
        errors.append("Rawlsian liberty status must cover exactly the canonical action set")

    normalized: list[dict[str, Any]] = []
    delta = SemanticGraph()
    assessment_ids: set[str] = set()
    for position in validated.positions:
        action = _resolve_action(store.graph, position.action_id)
        rival = _resolve_action(store.graph, position.compared_to_action_id)
        if action is None or rival is None:
            errors.append(f"Rawlsian position {position.action_id} references an unknown action")
            continue
        own_pairs = _action_consequences(store.graph, action.id)
        rival_pairs = _action_consequences(store.graph, rival.id)
        dimension_subject = _unambiguous_dimension_subject(
            [*own_pairs, *rival_pairs], position.dimension,
        )
        submitted_subject = position.subject
        effective_subject = submitted_subject
        dimension_subject_repaired = False
        if (
            dimension_subject is not None
            and not _subject_matches(submitted_subject, dimension_subject[1])
        ):
            effective_subject = dimension_subject[0]
            dimension_subject_repaired = True
        own_consequences, own_targets = _grounded_consequences(
            store.graph, action.id, effective_subject
        )
        rival_consequences, rival_targets = _grounded_consequences(
            store.graph, rival.id, effective_subject
        )
        own_action_targets = _action_targets(store.graph, action.id)
        rival_action_targets = _action_targets(store.graph, rival.id)
        comparison_action_targets = _comparison_action_targets(
            own_action_targets, rival_action_targets
        )
        foreign_framework_evidence = (
            _has_foreign_framework_consequences(store.graph, action.id)
            or _has_foreign_framework_consequences(store.graph, rival.id)
        )
        grounded_targets = list({
            target.id: target for target in (*own_targets, *rival_targets)
        }.values())
        action_graph_targets = comparison_action_targets
        action_graph_binding = False
        if not grounded_targets and action_graph_targets:
            grounded_targets = action_graph_targets[:]
            action_graph_binding = True
        # Do not copy every action consequence onto a subject that those
        # consequences do not affect. That would transfer another party's
        # effect across the comparison.
        # Binding requires a matching non-framework consequence target or a
        # separately typed scenario/projection target. Raw lexical action
        # objects and foreign-framework scopes cannot select the Rawls subject.
        typed_action_targets = [
            target for target in comparison_action_targets
            if not target.attributes.get("framework")
            and (
                target.attributes.get("semantic_role") == "AFFECTED_SUBJECT"
                or target.attributes.get("burden_evidence")
                or target.attributes.get("projection_kind")
            )
        ]
        subject_bound_to_action = bool(
            own_targets
            or rival_targets
            or _subject_matches(effective_subject, typed_action_targets)
        )
        subject_grounded = bool(
            grounded_targets
            and subject_bound_to_action
            and position.evidence_basis in {"ACTION_GRAPH", "SCENARIO"}
        )
        dimension_own = [
            consequence for consequence in own_consequences
            if _dimension_compatible(consequence, position.dimension)
        ]
        dimension_rival = [
            consequence for consequence in rival_consequences
            if _dimension_compatible(consequence, position.dimension)
        ]
        direction_supported = _direction_supported(
            position.effect, dimension_own, dimension_rival
        )
        mixed_tradeoff_supported = _mixed_tradeoff_supported(
            dimension_own, dimension_rival
        )
        committed_effect = position.effect
        epistemic_status = "GROUNDED"
        if position.effect == "MIXED":
            if not subject_grounded:
                committed_effect = "UNCERTAIN"
                epistemic_status = "UNSUPPORTED_MIXED_COMPARISON"
                errors.append(
                    f"{position.action_id} {position.effect} lacked action-bound subject grounding; committed as UNCERTAIN"
                )
            else:
                committed_effect = "MIXED"
                epistemic_status = "MIXED_COMPARISON"
        elif position.effect != "UNCERTAIN" and not subject_grounded:
            committed_effect = "UNCERTAIN"
            epistemic_status = "UNSUPPORTED_DIRECTION"
            errors.append(
                f"{position.action_id} {position.effect} lacked action-bound subject grounding; committed as UNCERTAIN"
            )
        elif position.effect != "UNCERTAIN" and subject_grounded:
            if mixed_tradeoff_supported:
                committed_effect = "MIXED"
                epistemic_status = "MIXED_COMPARISON"
            elif not direction_supported:
                committed_effect = "UNCERTAIN"
                epistemic_status = "UNSUPPORTED_DIRECTION"
                errors.append(
                    f"{position.action_id} {position.effect} lacked directional support; committed as UNCERTAIN"
                )
        elif position.effect == "UNCERTAIN":
            epistemic_status = "EXPLICIT_UNCERTAINTY"

        subject_label = _canonical_group(effective_subject)
        subject_id = _stable_id("RAWLS_SUBJECT", subject_label)
        subject_kind = (
            position.subject_kind
            if position.subject_kind != "UNKNOWN"
            else _subject_kind(effective_subject)
        )
        dimension_id = f"RAWLS_VALUE:{position.dimension}"
        assessment_identity = "|".join((
            action.id, position.dimension, subject_label,
        ))
        assessment_id = (
            f"RAWLS_POSITION:{specialist}:{action.id}:"
            f"{_stable_id('DIMENSION_SUBJECT', assessment_identity).rsplit(':', 1)[-1]}"
        )
        assessment_ids.add(assessment_id)
        provenance = (f"delegate:{specialist}", f"cycle:{cycle}")
        binding_mode = (
            "DIMENSION_TARGET"
            if dimension_subject_repaired
            else (
                "ACTION_GRAPH" if action_graph_binding
                else ("LEXICAL" if subject_bound_to_action else "UNRESOLVED")
            )
        )
        delta.add_node(SemanticNode(
            subject_id, "TARGET", subject_label, provenance,
            {
                "framework": "RAWLSIAN",
                "moral_subject_kind": subject_kind,
                "grounded_target_node_ids": [target.id for target in grounded_targets],
                "epistemic_status": "GROUNDED" if grounded_targets else "UNRESOLVED_TARGET",
                "subject_binding_mode": binding_mode,
                "group_binding_mode": binding_mode,
            },
        ))
        delta.add_node(SemanticNode(
            dimension_id, "VALUE", position.dimension, ("rawlsian_primary_goods",),
            {"framework": "RAWLSIAN"},
        ))
        delta.add_node(SemanticNode(
            assessment_id,
            "ASSESSMENT",
            f"Rawlsian position for {position.action_id}",
            provenance,
            {
                "framework": "RAWLSIAN",
                "specialist": specialist,
                "cycle": cycle,
                "canonical_action_id": position.action_id,
                "compared_to_action_id": position.compared_to_action_id,
                "effect": committed_effect,
                "proposed_effect": position.effect,
                "subject": subject_label,
                "proposed_subject": _canonical_group(submitted_subject),
                "dimension_subject_repaired": dimension_subject_repaired,
                "affected_subject": subject_label,
                "subject_kind": subject_kind,
                "dimension": position.dimension,
                "basic_liberty_kind": position.basic_liberty_kind,
                "institutional_relation": position.institutional_relation,
                "additional_dimensions": list(position.additional_dimensions),
                "dimension_bundle": [position.dimension, *position.additional_dimensions],
                "comparative_effect": committed_effect,
                "group_node_id": subject_id,
                "subject_node_id": subject_id,
                "dimension_node_id": dimension_id,
                "subject_binding_mode": binding_mode,
                "group_binding_mode": binding_mode,
                "evidence_basis": position.evidence_basis,
                "epistemic_status": epistemic_status,
                "reason": position.reason,
                "ranking_basis": validated.ranking_basis,
                "ranking_classification_justification": validated.ranking_classification_justification,
                "lexical_priority_justification": validated.lexical_priority_justification,
                "liberty_status": validated.liberty_status.get(position.action_id, "UNKNOWN"),
                "subject_selection_status": (
                    "BOUND_TO_ACTION" if subject_bound_to_action else "UNRESOLVED_SUBJECT_SELECTION"
                ),
                "group_selection_status": (
                    "BOUND_TO_ACTION" if subject_bound_to_action else "UNRESOLVED_GROUP_SELECTION"
                ),
            },
        ))
        delta.add_edge(SemanticEdge(
            action.id, "HAS_ASSESSMENT", assessment_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id,
            {
            "IMPROVES": "IMPROVES_POSITION",
            "PRESERVES": "PRESERVES_POSITION",
            "WORSENS": "WORSENS_POSITION",
            "MIXED": "MIXED_POSITION",
            "UNCERTAIN": "POSITION_UNCERTAIN",
        }[committed_effect],
            subject_id,
            justification=position.reason,
            provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, "ASSESSES", dimension_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, "COMPARES_TO", rival.id, provenance=provenance,
        ))
        for evidence in [*own_consequences, *rival_consequences, *grounded_targets]:
            delta.add_edge(SemanticEdge(
                assessment_id, "SUPPORTED_BY", evidence.id, provenance=provenance,
            ))
        normalized.append({
            **position.model_dump(),
            "effect": committed_effect,
            "epistemic_status": epistemic_status,
            "group_node_id": subject_id,
            "subject_node_id": subject_id,
            "affected_subject": subject_label,
            "proposed_subject": _canonical_group(submitted_subject),
            "dimension_subject_repaired": dimension_subject_repaired,
            "subject_kind": subject_kind,
            "comparative_effect": committed_effect,
            "additional_dimensions": list(position.additional_dimensions),
            "dimension_bundle": [position.dimension, *position.additional_dimensions],
            "subject_binding_mode": binding_mode,
            "group_binding_mode": binding_mode,
            "evidence_node_ids": [
                node.id for node in [*own_consequences, *rival_consequences, *grounded_targets]
            ],
            "ranking_basis": validated.ranking_basis,
            "ranking_classification_justification": validated.ranking_classification_justification,
            "lexical_priority_justification": validated.lexical_priority_justification,
            "liberty_status": validated.liberty_status.get(position.action_id, "UNKNOWN"),
            "subject_selection_status": (
                "BOUND_TO_ACTION" if subject_bound_to_action else "UNRESOLVED_SUBJECT_SELECTION"
            ),
            "group_selection_status": (
                "BOUND_TO_ACTION" if subject_bound_to_action else "UNRESOLVED_GROUP_SELECTION"
            ),
        })

    hard_errors = [error for error in errors if "committed as UNCERTAIN" not in error]
    if hard_errors or len(normalized) != len(validated.positions):
        record = GraphTransactionRecord(
            cycle, specialist, "RAWLS_POSITION_LEDGER", "REJECTED", raw,
            list(dict.fromkeys(errors)), previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    # Replace the complete active ledger for this specialist while
    # retaining target/value nodes and all unrelated semantic state.
    previous_assessment_ids = {
        node.id for node in store.graph.nodes.values()
        if node.kind == "ASSESSMENT"
        and node.attributes.get("framework") == "RAWLSIAN"
        and node.attributes.get("specialist") == specialist
    }
    replaced_assessment_ids = assessment_ids | previous_assessment_ids
    base = SemanticGraph(
        nodes={
            node_id: node for node_id, node in store.graph.nodes.items()
            if node_id not in replaced_assessment_ids
        },
        edges=[
            edge for edge in store.graph.edges
            if edge.source not in replaced_assessment_ids
            and edge.target not in replaced_assessment_ids
        ],
    )
    try:
        prospective = merge_graphs([base, delta])
    except ValueError as exc:
        record = GraphTransactionRecord(
            cycle, specialist, "RAWLS_POSITION_LEDGER", "REJECTED", raw,
            [f"graph merge rejected: {exc}"],
            previous_state_preserved=True,
            retryable=True,
        )
        store.transactions.append(record)
        return record
    graph_validation = validate_graph(prospective)
    if not graph_validation.valid:
        record = GraphTransactionRecord(
            cycle, specialist, "RAWLS_POSITION_LEDGER", "REJECTED", raw,
            graph_validation.errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    store.graph = prospective
    status = "COMMITTED_WITH_UNCERTAINTY" if errors else "COMMITTED"
    record = GraphTransactionRecord(
        cycle,
        specialist,
        "RAWLS_POSITION_LEDGER",
        status,
        {"submitted": raw, "committed_positions": normalized},
        list(dict.fromkeys(errors)),
        previous_state_preserved=False,
    )
    store.transactions.append(record)
    return record


def committed_rawls_positions(graph: SemanticGraph) -> list[dict[str, Any]]:
    """Read the active Rawlsian assessments from graph objects only."""
    values = []
    for node in graph.nodes.values():
        if node.kind != "ASSESSMENT" or node.attributes.get("framework") != "RAWLSIAN":
            continue
        values.append({
            "assessment_node_id": node.id,
            **dict(node.attributes),
            "provenance": list(node.provenance),
        })
    return sorted(values, key=lambda value: (
        str(value.get("canonical_action_id", "")),
        str(value.get("dimension", "")),
        str(value.get("subject", "")),
    ))
