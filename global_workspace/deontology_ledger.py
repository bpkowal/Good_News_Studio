"""Transactional Deontological duty, right, and permission ledger."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .graph_transactions import GraphTransactionRecord, SemanticGraphStore
from .scenario_semantics import query_grounded_action_effects
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode, merge_graphs, validate_graph


class DutyAssessmentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: str = Field(pattern=r"^A\d+$")
    verdict: Literal["REQUIRED", "PERMISSIBLE", "PROHIBITED", "CONFLICTED"]
    norm_kind: Literal[
        "DUTY", "RIGHT", "AUTONOMY", "UNIVERSAL_LAW", "RESPECT_PERSONS",
        "OTHER", "UNKNOWN",
    ]
    norm: str = Field(min_length=3, max_length=100)
    relation: Literal["SATISFIES", "CONSISTENT", "VIOLATES", "CONFLICTS", "UNCERTAIN"]
    duty_bearer: str = Field(min_length=1, max_length=80)
    protected_party: str = Field(min_length=1, max_length=100)
    competing_norm: str = Field(min_length=1, max_length=100)
    competing_norm_kind: Literal[
        "DUTY", "RIGHT", "AUTONOMY", "UNIVERSAL_LAW", "RESPECT_PERSONS",
        "OTHER", "UNKNOWN",
    ] = "UNKNOWN"
    competing_relation: Literal[
        "SATISFIES", "CONSISTENT", "VIOLATES", "CONFLICTS", "UNCERTAIN",
    ] = "UNCERTAIN"
    competing_protected_party: str = Field(default="unspecified party", min_length=1, max_length=100)
    competing_reason: str = Field(default="competing norm remains unresolved", min_length=4, max_length=180)
    governing_norm: Literal["PRIMARY", "COMPETING", "UNRESOLVED"] = "PRIMARY"
    priority_basis: Literal[
        "UNIVERSAL_LAW", "RESPECT_PERSONS", "PERFECT_DUTY",
        "RIGHTFUL_COERCION", "AUTONOMY", "UNRESOLVED",
    ] = "UNRESOLVED"
    priority_rule: str = Field(default="priority remains unresolved", min_length=4, max_length=180)
    protected_standing: Literal[
        "EXTERNAL_FREEDOM", "AUTONOMY", "EQUAL_JURIDICAL_STATUS",
        "BODILY_INTEGRITY", "CONSENT", "CONTRACTUAL_RIGHT",
        "SPECIAL_OBLIGATION", "OTHER", "UNKNOWN",
    ] = "UNKNOWN"
    competing_protected_standing: Literal[
        "EXTERNAL_FREEDOM", "AUTONOMY", "EQUAL_JURIDICAL_STATUS",
        "BODILY_INTEGRITY", "CONSENT", "CONTRACTUAL_RIGHT",
        "SPECIAL_OBLIGATION", "OTHER", "UNKNOWN",
    ] = "UNKNOWN"
    coercion_kind: Literal[
        "NONE", "PUBLIC", "PRIVATE", "INSTITUTIONAL", "INTERPERSONAL", "UNKNOWN",
    ] = "UNKNOWN"
    coercive_actor: str = Field(default="NONE", min_length=1, max_length=100)
    coerced_party: str = Field(default="NONE", min_length=1, max_length=100)
    public_justification: str = Field(default="unresolved", min_length=4, max_length=180)
    reciprocity_status: Literal["SATISFIED", "FAILED", "CONTESTED", "UNKNOWN"] = "UNKNOWN"
    necessity_status: Literal[
        "NECESSARY", "LESS_RESTRICTIVE_ROUTE_AVAILABLE", "CONTESTED", "UNKNOWN",
    ] = "UNKNOWN"
    authorization_status: Literal[
        "JUSTIFIED", "UNJUSTIFIED", "CONTESTED", "UNKNOWN", "NOT_APPLICABLE",
    ] = "UNKNOWN"
    derivation: Literal[
        "UNIVERSAL_LAW", "RECIPROCAL_EXTERNAL_FREEDOM", "RESPECT_PERSONS",
        "PERFECT_DUTY", "RIGHTFUL_PUBLIC_COERCION", "CONSENT", "UNRESOLVED",
    ] = "UNRESOLVED"
    resolution_status: Literal["RESOLVED", "CONTESTED", "UNKNOWN"] = "UNKNOWN"
    evidence_basis: Literal["ACTION_GRAPH", "SCENARIO", "FRAMEWORK_ONLY", "UNKNOWN"]
    reason: str = Field(min_length=4, max_length=180)

    @model_validator(mode="after")
    def adjudication_form_is_coherent(self):
        if self.coercion_kind == "NONE" and self.authorization_status != "NOT_APPLICABLE":
            raise ValueError("non-coercive action must use NOT_APPLICABLE authorization")
        if self.coercion_kind != "NONE" and self.authorization_status == "NOT_APPLICABLE":
            raise ValueError("coercive action must adjudicate authorization")
        if self.resolution_status == "RESOLVED" and self.derivation == "UNRESOLVED":
            raise ValueError("resolved adjudication requires a deontological derivation")
        if self.verdict == "CONFLICTED" and self.resolution_status == "RESOLVED":
            raise ValueError("conflicted verdict cannot claim resolved adjudication")
        borrowed = re.search(
            r"\b(?:fair equality of opportunity|difference principle|least[- ]advantaged)\b",
            " ".join((self.norm, self.competing_norm, self.priority_rule)), re.I,
        )
        kantian_bridge = re.search(
            r"\b(?:external freedom|autonom\w*|juridical|universal\w*|persons?|"
            r"consent|rightful coercion)\b",
            " ".join((
                self.norm, self.competing_norm, self.priority_rule,
                self.public_justification,
            )), re.I,
        )
        if borrowed and not kantian_bridge:
            raise ValueError("CROSS_FRAMEWORK_CONCEPT_UNTRANSLATED")
        return self


class DeontologicalLedgerProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    assessments: list[DutyAssessmentProposal] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def unique_actions(self):
        action_ids = [item.action_id for item in self.assessments]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Deontological ledger repeats an action")
        return self


@dataclass(frozen=True, slots=True)
class AdjudicationCalibration:
    assessment: DutyAssessmentProposal
    errors: tuple[str, ...] = ()
    support_node_ids: tuple[str, ...] = ()

    @property
    def calibrated(self) -> bool:
        return not self.errors


def _resolve_action(graph: SemanticGraph, reference: str) -> SemanticNode | None:
    matches = [
        node for node in graph.nodes.values()
        if node.kind == "ACTION" and str(reference).strip() in {
            node.id, node.label,
            str(node.attributes.get("canonical_action_id", "")),
            str(node.attributes.get("semantic_action_key", "")),
        }
    ]
    return matches[0] if len(matches) == 1 else None


def _words(text: str) -> set[str]:
    ignored = {"person", "people", "party", "actor", "agent", "the", "and", "for"}
    return {
        value for value in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(value) >= 3 and value not in ignored
    }


def _action_parties(graph: SemanticGraph, action_id: str) -> list[SemanticNode]:
    parties: list[SemanticNode] = []
    for edge in graph.outgoing(action_id):
        target = graph.nodes.get(edge.target)
        if target is not None and target.kind in {"TARGET", "ACTOR"}:
            parties.append(target)
        if target is not None and target.kind == "CONSEQUENCE":
            parties.extend(
                graph.nodes[subedge.target]
                for subedge in graph.outgoing(target.id, "AFFECTS")
                if subedge.target in graph.nodes
                and graph.nodes[subedge.target].kind == "TARGET"
            )
    return list({party.id: party for party in parties}.values())


def _matching_parties(
    graph: SemanticGraph, action_id: str, text: str,
) -> list[SemanticNode]:
    words = _words(text)
    parties = _action_parties(graph, action_id)
    effects = query_grounded_action_effects(
        graph, action_id, affected_subject=text,
    )
    grounded_ids = {
        node_id for effect in effects for node_id in effect.affected_subject_node_ids
    }
    grounded_labels = {
        effect.affected_subject.casefold() for effect in effects
    }
    canonical_matches = [
        party for party in parties
        if party.id in grounded_ids or party.label.casefold() in grounded_labels
    ]
    if canonical_matches:
        return list({party.id: party for party in canonical_matches}.values())
    return [party for party in parties if words & _words(party.label)]


def _clause_grounded_parties(
    graph: SemanticGraph, action_id: str, text: str,
) -> list[SemanticNode]:
    """Ground a named party in the scenario clauses this action cites.

    An action's identity graph only keeps the targets its own predicate
    governs, so a party the scenario names as affected without being acted on
    directly - a third party framed by the override, for instance - has no
    target node to match. The cited clause is still current-run evidence, so
    admit it only when every substantive word of the party description appears
    in one asserted clause. Loose overlap would let "innocent bystander" bind
    to a clause that merely says "innocent".
    """
    words = _words(text)
    if not words:
        return []
    return [
        clause for edge in graph.outgoing(action_id, "GROUNDED_IN")
        if (clause := graph.nodes.get(edge.target)) is not None
        and clause.kind == "EVIDENCE"
        and clause.attributes.get("evidence_role") == "SCENARIO_ASSERTION"
        and words <= _words(clause.label)
    ]


def _party_grounding(
    graph: SemanticGraph, action_id: str, text: str,
) -> tuple[list[SemanticNode], str]:
    """Resolve a party against the action identity graph, then its clauses."""
    identity_matches = _matching_parties(graph, action_id, text)
    if identity_matches:
        return identity_matches, "ACTION_IDENTITY"
    clause_matches = _clause_grounded_parties(graph, action_id, text)
    if clause_matches:
        return clause_matches, "SCENARIO_CLAUSE"
    return [], "NONE"


def _stable_id(prefix: str, value: str) -> str:
    normalized = " ".join(str(value).casefold().split()).strip(" ,.;:")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


_COERCION_MECHANISM = re.compile(
    r"\b(?:abolish|ban|compel|compulsor\w*|detain|forbid|force|illegal|mandat\w*|"
    r"monopol\w*|prohibit|require|restrict|seize|without consent)\b", re.I,
)
_NECESSITY_GROUND = re.compile(
    r"\b(?:only means|only way|no less restrictive|no other means|cannot otherwise|"
    r"necessary condition|strictly necessary|without any alternative)\b", re.I,
)
_RECIPROCAL_GROUND = re.compile(
    r"\b(?:compatible external freedom|equal external freedom|reciprocal public rule|"
    r"same public rule|freedom of each|freedom for both)\b", re.I,
)
_PRIORITY_SLOGAN = re.compile(
    r"\b(?:justice|equality)\b.{0,40}\b(?:override|outweigh|trump)s?\b.{0,40}"
    r"\b(?:libert\w*|freedom|autonom\w*)\b", re.I,
)


def _graph_supporting_nodes(graph: SemanticGraph, pattern: re.Pattern[str]) -> list[str]:
    return [
        node.id for node in graph.nodes.values()
        if node.attributes.get("framework") is None and pattern.search(node.label)
    ]


def _party_named(explanation: str, party: str) -> bool:
    return bool(_words(explanation) & _words(party))


def _coercion_path_support(
    graph: SemanticGraph,
    action: SemanticNode,
    item: DutyAssessmentProposal,
) -> tuple[bool, list[str]]:
    actor_matches, _actor_basis = _party_grounding(graph, action.id, item.coercive_actor)
    party_matches, _party_basis = _party_grounding(graph, action.id, item.coerced_party)
    adverse = query_grounded_action_effects(
        graph, item.action_id, affected_subject=item.coerced_party,
        direction="ADVERSE",
    )
    mechanism_ids = [action.id] if _COERCION_MECHANISM.search(action.label) else []
    mechanism_ids.extend(
        effect.consequence_id for effect in adverse
        if (node := graph.nodes.get(effect.consequence_id)) is not None
        and _COERCION_MECHANISM.search(node.label)
    )
    support = [
        *(node.id for node in actor_matches),
        *mechanism_ids,
        *(effect.consequence_id for effect in adverse),
        *(node.id for node in party_matches),
    ]
    return bool(actor_matches and mechanism_ids and adverse and party_matches), list(
        dict.fromkeys(support)
    )


def calibrate_deontological_adjudication(
    graph: SemanticGraph,
    action: SemanticNode,
    item: DutyAssessmentProposal,
) -> AdjudicationCalibration:
    """Downgrade normative conclusions whose decisive premises lack support.

    The calibration never decides which duty should win. It only prevents a
    delegate from presenting necessity, reciprocity, authorization, or a
    coercive relation as established when the current graph does not establish
    the required premise.
    """
    errors: list[str] = []
    support: list[str] = []
    updates: dict[str, Any] = {}

    if _PRIORITY_SLOGAN.search(" ".join((item.priority_rule, item.reason))):
        errors.append("priority slogan does not supply a Kantian derivation")
        updates["derivation"] = "UNRESOLVED"

    if item.coercion_kind != "NONE":
        path_supported, path_support = _coercion_path_support(graph, action, item)
        support.extend(path_support)
        if not path_supported:
            errors.append(
                "coercion claim lacks actor-to-mechanism-to-restriction-to-party grounding"
            )

        necessity_support = _graph_supporting_nodes(graph, _NECESSITY_GROUND)
        if item.necessity_status == "NECESSARY" and not necessity_support:
            errors.append("necessity lacks grounded less-restrictive-route evidence")
            updates["necessity_status"] = "CONTESTED"
        support.extend(necessity_support)

        reciprocity_supported = bool(
            _RECIPROCAL_GROUND.search(item.public_justification)
            and _party_named(item.public_justification, item.coerced_party)
            and _party_named(item.public_justification, item.protected_party)
        )
        if item.reciprocity_status == "SATISFIED" and not reciprocity_supported:
            errors.append("reciprocity lacks a two-party compatible-freedom derivation")
            updates["reciprocity_status"] = "CONTESTED"

        effective_necessity = updates.get("necessity_status", item.necessity_status)
        effective_reciprocity = updates.get("reciprocity_status", item.reciprocity_status)
        if item.authorization_status == "JUSTIFIED" and (
            not path_supported
            or effective_necessity != "NECESSARY"
            or effective_reciprocity != "SATISFIED"
        ):
            errors.append("authorization asserted without grounded necessity and reciprocity")
            updates["authorization_status"] = "CONTESTED"

    effective_derivation = updates.get("derivation", item.derivation)
    effective_authorization = updates.get(
        "authorization_status", item.authorization_status,
    )
    if item.resolution_status == "RESOLVED" and (
        effective_derivation == "UNRESOLVED"
        or (item.coercion_kind != "NONE" and effective_authorization != "JUSTIFIED")
        or errors
    ):
        errors.append("resolved adjudication rests on unsupported decisive premises")

    if errors:
        updates.update({
            "verdict": "CONFLICTED",
            "governing_norm": "UNRESOLVED",
            "priority_basis": "UNRESOLVED",
            "resolution_status": "CONTESTED",
        })
        if item.coercion_kind != "NONE" and "authorization_status" not in updates:
            updates["authorization_status"] = "CONTESTED"
    calibrated = item.model_copy(update=updates)
    return AdjudicationCalibration(
        assessment=calibrated,
        errors=tuple(dict.fromkeys(errors)),
        support_node_ids=tuple(dict.fromkeys(support)),
    )


def latest_assessment_by_action(
    assessments: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index the most recently committed adjudication for each action.

    A revised adjudication commits as a new node rather than overwriting the
    old one, so the projection can hold several entries per action. Reading the
    first match would let a superseded verdict govern the rendered claim.
    """
    latest: dict[str, dict[str, Any]] = {}
    for item in assessments:
        action_id = str(item.get("canonical_action_id", ""))
        if not action_id:
            continue
        current = latest.get(action_id)
        if current is None or int(item.get("cycle", 0) or 0) >= int(
            current.get("cycle", 0) or 0
        ):
            latest[action_id] = item
    return latest


def _unresolved_premises(assessment: dict[str, Any]) -> list[str]:
    """Name the premises the calibration found unestablished."""
    errors = [str(value) for value in assessment.get("calibration_errors", [])]
    unresolved: list[str] = []
    if any("necessity" in error for error in errors):
        unresolved.append("whether a less restrictive route can secure the protected claim")
    if any("reciprocity" in error for error in errors):
        unresolved.append("whether the same public rule preserves compatible external freedom")
    if any("coercion claim" in error for error in errors):
        unresolved.append("whether the asserted coercive relationship is grounded")
    if any("authorization" in error for error in errors):
        unresolved.append(
            "whether the restriction is authorized by a rule both parties could will"
        )
    if any("priority slogan" in error for error in errors):
        unresolved.append(
            "whether the priority follows from a Kantian derivation rather than a maxim slogan"
        )
    if not unresolved:
        unresolved.append("which claim governs under a universal public rule")
    return list(dict.fromkeys(unresolved))


# Illustrative attenuation for provisional Kantian leanings. Tunable later;
# the architectural invariant is that provisionality reduces policy weight
# without zeroing directional information or investigative attention.
PROVISIONAL_LEANING_POLICY_FACTOR = 0.45
CONFLICTED_NO_LEANING_POLICY_FACTOR = 0.0
# Preference gap below which an incomplete adjudication contributes no lean.
_MIN_LEANING_PREFERENCE = 0.10


@dataclass(frozen=True)
class DeontologicalAuthorityProfile:
    """Separate policy, attention, and governing authority for Kantian claims.

    Resolved arguments may govern. Unadjudicated but consequential strict-duty
    conflicts may interrupt (investigative broadcast) and may lean for search,
    but may not supply a final justificatory rule.
    """

    adjudication_status: str
    broadcast_authority: str
    governing_eligible: bool
    policy_weight_factor: float
    investigative_claim: str
    rationale: str
    decision_rule: str
    internal_conflicts: tuple[str, ...]
    open_questions: tuple[str, ...]


def render_deontological_adjudication(
    assessment: dict[str, Any],
    rival_assessments: Sequence[dict[str, Any]] = (),
) -> tuple[str, str, list[str], list[str]]:
    """Render the public reason and framework-local tension from committed state.

    A Kantian preference is comparative: "publishing is required" carries its
    force only if suppression is genuinely prohibited. So the rival actions'
    committed adjudications constrain this one. When the selected action's own
    adjudication is clean but a rival's prohibition rests on premises the
    calibration rejected, the priority is not established and must not be
    reported as a settled rule.
    """
    profile = classify_deontological_authority(assessment, rival_assessments)
    return (
        profile.rationale,
        profile.decision_rule,
        list(profile.internal_conflicts),
        list(profile.open_questions),
    )


def classify_deontological_authority(
    assessment: dict[str, Any],
    rival_assessments: Sequence[dict[str, Any]] = (),
    *,
    recommended_action: str = "",
    preference_strength: float = 1.0,
) -> DeontologicalAuthorityProfile:
    """Classify Kantian claim authority without laundering leanings into rules.

    Three levels:
    - CONFLICTED_NO_LEANING: incomplete adjudication, no directional preference
    - PROVISIONAL_LEANING: incomplete adjudication with a comparative lean
    - ADJUDICATED_SUPPORTS: comparative priority actually established
    """
    primary_party = str(assessment.get("protected_party", "the protected party"))
    competing_party = str(
        assessment.get("competing_protected_party", "the competing claimant")
    )
    primary_claim = str(assessment.get("norm", "a protected claim"))
    competing_claim = str(assessment.get("competing_norm", "a competing claim"))
    conflict = (
        f"{primary_party}: {primary_claim} ↔ {competing_party}: {competing_claim}"
    )
    leaning = (
        " ".join(str(recommended_action).split())
        if recommended_action and float(preference_strength) >= _MIN_LEANING_PREFERENCE
        else ""
    )

    if str(assessment.get("resolution_status", "UNKNOWN")) != "RESOLVED":
        actor = str(assessment.get("coercive_actor", "the acting authority"))
        coerced = str(assessment.get("coerced_party", "the restricted party"))
        unresolved = _unresolved_premises(assessment)
        rationale = (
            f"Deontology identifies competing claims between {primary_party}'s "
            f"{primary_claim} and {competing_party}'s {competing_claim}. "
            f"{actor}'s restriction of {coerced} is not yet justified because "
            + "; and ".join(unresolved)
            + ". The Kantian judgment remains contested."
        )
        return _incomplete_authority(
            conflict=conflict,
            unresolved=unresolved,
            rationale=rationale,
            leaning=leaning,
            decision_rule_prefix="Keep the action contested until ",
        )

    contested_rivals = [
        item for item in rival_assessments
        if str(item.get("resolution_status", "UNKNOWN")) != "RESOLVED"
    ]
    derivation = str(assessment.get("derivation", "UNRESOLVED")).lower().replace("_", " ")
    if contested_rivals:
        unresolved = list(dict.fromkeys(
            premise
            for item in contested_rivals
            for premise in _unresolved_premises(item)
        ))
        rival_ids = ", ".join(sorted(
            str(item.get("canonical_action_id", "the rival action"))
            for item in contested_rivals
        ))
        rival_conflicts = [conflict]
        for item in contested_rivals:
            rival_claim = str(item.get("norm", "a protected claim"))
            rival_id = str(item.get("canonical_action_id", "the rival action"))
            for premise in _unresolved_premises(item):
                rival_conflicts.append(
                    f"{rival_id} prohibited by {rival_claim} ↔ {premise}"
                )
        rationale = (
            f"Deontology ranks {primary_party}'s {primary_claim} above "
            f"{competing_party}'s {competing_claim} through {derivation}, but that "
            f"priority holds only if {rival_ids} is genuinely prohibited, and the "
            f"prohibition rests on unestablished premises: "
            + "; and ".join(unresolved)
            + ". The comparative Kantian judgment remains contested."
        )
        return _incomplete_authority(
            conflict=conflict,
            unresolved=unresolved,
            rationale=rationale,
            leaning=leaning,
            decision_rule_prefix="Treat the duty priority as unestablished until ",
            extra_conflicts=rival_conflicts[1:],
        )

    return DeontologicalAuthorityProfile(
        adjudication_status="ADJUDICATED_SUPPORTS",
        broadcast_authority="GOVERNING_CANDIDATE",
        governing_eligible=True,
        policy_weight_factor=1.0,
        investigative_claim="",
        rationale=(
            f"Deontology resolves {primary_party}'s {primary_claim} against "
            f"{competing_party}'s {competing_claim} through {derivation}."
        ),
        decision_rule=str(assessment.get("priority_rule", "priority remains unresolved")),
        internal_conflicts=(),
        open_questions=(),
    )


def _incomplete_authority(
    *,
    conflict: str,
    unresolved: Sequence[str],
    rationale: str,
    leaning: str,
    decision_rule_prefix: str,
    extra_conflicts: Sequence[str] = (),
) -> DeontologicalAuthorityProfile:
    open_questions = tuple(dict.fromkeys(unresolved))
    conflicts = tuple(dict.fromkeys((conflict, *extra_conflicts)))
    decision_rule = decision_rule_prefix + "; and ".join(open_questions)
    if leaning:
        investigative = (
            f"UNRESOLVED DUTY CONFLICT: {conflict}. Current reasoning leans "
            f"{leaning}, but the competing strict claim has not been vindicated "
            "or defeated."
        )
        return DeontologicalAuthorityProfile(
            adjudication_status="PROVISIONAL_LEANING",
            broadcast_authority="INVESTIGATIVE",
            governing_eligible=False,
            policy_weight_factor=PROVISIONAL_LEANING_POLICY_FACTOR,
            investigative_claim=investigative,
            rationale=rationale,
            decision_rule=decision_rule,
            internal_conflicts=conflicts,
            open_questions=open_questions,
        )
    investigative = (
        f"UNRESOLVED DUTY CONFLICT: {conflict}. No comparative Kantian leaning "
        "is yet grounded."
    )
    return DeontologicalAuthorityProfile(
        adjudication_status="CONFLICTED_NO_LEANING",
        broadcast_authority="INVESTIGATIVE",
        governing_eligible=False,
        policy_weight_factor=CONFLICTED_NO_LEANING_POLICY_FACTOR,
        investigative_claim=investigative,
        rationale=rationale,
        decision_rule=decision_rule,
        internal_conflicts=conflicts,
        open_questions=open_questions,
    )


def apply_deontological_ledger_transaction(
    store: SemanticGraphStore,
    proposal: dict[str, Any],
    *,
    cycle: int,
    specialist: str,
    allowed_actions: tuple[str, ...],
) -> GraphTransactionRecord:
    raw = dict(proposal) if isinstance(proposal, dict) else {"raw": proposal}
    try:
        validated = DeontologicalLedgerProposal.model_validate(proposal)
    except ValidationError as exc:
        errors = [
            f"schema {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        ]
        record = GraphTransactionRecord(
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    allowed_ids = {
        str(node.attributes.get("canonical_action_id", node.id))
        for reference in allowed_actions
        if (node := _resolve_action(store.graph, reference)) is not None
    }
    submitted_ids = {item.action_id for item in validated.assessments}
    if submitted_ids != allowed_ids:
        record = GraphTransactionRecord(
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            ["Deontological ledger must cover exactly the canonical action set"],
            previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record

    verdict_relations = {
        "REQUIRED": {"SATISFIES"},
        "PERMISSIBLE": {"CONSISTENT", "SATISFIES"},
        "PROHIBITED": {"VIOLATES"},
        "CONFLICTED": {"CONFLICTS", "UNCERTAIN"},
    }
    relation_edges = {
        "SATISFIES": "SATISFIES_NORM",
        "CONSISTENT": "CONSISTENT_WITH_NORM",
        "VIOLATES": "VIOLATES_NORM",
        "CONFLICTS": "CONFLICTS_NORM",
        "UNCERTAIN": "NORM_UNCERTAIN",
    }
    warnings: list[str] = []
    delta = SemanticGraph()
    committed: list[dict[str, Any]] = []
    replace_ids = {
        node.id for node in store.graph.nodes.values()
        if node.kind in {"ASSESSMENT", "DECISION"}
        and node.attributes.get("framework") == "DEONTOLOGICAL"
        and node.attributes.get("specialist") == specialist
    }
    for proposed_item in validated.assessments:
        action = _resolve_action(store.graph, proposed_item.action_id)
        if action is None:
            warnings.append(f"unknown action {proposed_item.action_id}")
            continue
        calibration = calibrate_deontological_adjudication(
            store.graph, action, proposed_item,
        )
        item = calibration.assessment
        warnings.extend(
            f"{item.action_id} ADJUDICATION_CALIBRATION: {error}"
            for error in calibration.errors
        )
        committed_relation = item.relation
        committed_verdict = item.verdict
        epistemic_status = (
            "FRAMEWORK_INTERPRETATION"
            if item.evidence_basis == "FRAMEWORK_ONLY" else "PROPOSED"
        )
        governing_relation = (
            item.relation if item.governing_norm == "PRIMARY"
            else item.competing_relation if item.governing_norm == "COMPETING"
            else "UNCERTAIN"
        )
        if governing_relation not in verdict_relations[item.verdict]:
            committed_verdict = "UNCERTAIN"
            epistemic_status = "INTERNALLY_INCONSISTENT"
            warnings.append(
                f"{item.action_id} {item.verdict} conflicts with governing "
                f"{item.governing_norm} relation {governing_relation}; committed as UNCERTAIN"
            )
        party_label = " ".join(item.protected_party.casefold().split()).strip(" ,.;:")
        bearer_label = " ".join(item.duty_bearer.casefold().split()).strip(" ,.;:")
        norm_label = " ".join(item.norm.casefold().split()).strip(" ,.;:")
        matched, party_basis = _party_grounding(
            store.graph, action.id, item.protected_party,
        )
        competing_matched, _competing_basis = _party_grounding(
            store.graph, action.id, item.competing_protected_party,
        )
        if matched and epistemic_status == "PROPOSED":
            epistemic_status = (
                "GROUNDED_PARTY" if party_basis == "ACTION_IDENTITY"
                else "GROUNDED_PARTY_BY_SCENARIO_CLAUSE"
            )
        elif not matched:
            epistemic_status = "UNRESOLVED_PARTY"
            warnings.append(
                f"{item.action_id} primary protected party lacks current-run action grounding"
            )
        if not competing_matched:
            warnings.append(
                f"{item.action_id} competing protected party lacks current-run action grounding"
            )
        coercive_actor_matches = _party_grounding(
            store.graph, action.id, item.coercive_actor,
        )[0] if item.coercion_kind != "NONE" else []
        coerced_party_matches = _party_grounding(
            store.graph, action.id, item.coerced_party,
        )[0] if item.coercion_kind != "NONE" else []
        if item.coercion_kind != "NONE" and (
            not coercive_actor_matches or not coerced_party_matches
        ):
            warnings.append(
                f"{item.action_id} coercion terrain lacks current-run actor/party grounding"
            )
        party_id = _stable_id("DEON_PARTY", party_label)
        bearer_id = _stable_id("DEON_BEARER", bearer_label)
        norm_id = _stable_id("DEON_NORM", f"{item.norm_kind}:{norm_label}")
        competing_norm_label = " ".join(item.competing_norm.casefold().split()).strip(" ,.;:")
        competing_party_label = " ".join(
            item.competing_protected_party.casefold().split()
        ).strip(" ,.;:")
        competing_norm_id = _stable_id(
            "DEON_NORM", f"{item.competing_norm_kind}:{competing_norm_label}",
        )
        competing_party_id = _stable_id("DEON_PARTY", competing_party_label)
        assessment_id = f"DEON_ASSESSMENT:{specialist}:{action.id}"
        competing_assessment_id = f"{assessment_id}:COMPETING"
        verdict_id = f"DEON_VERDICT:{specialist}:{action.id}"
        provenance = (f"delegate:{specialist}", f"cycle:{cycle}")
        delta.add_node(SemanticNode(
            party_id, "TARGET", party_label, provenance,
            {
                "framework": "DEONTOLOGICAL",
                "grounded_node_ids": [p.id for p in matched],
                "grounding_basis": party_basis,
            },
        ))
        delta.add_node(SemanticNode(
            bearer_id, "ACTOR", bearer_label, provenance,
            {"framework": "DEONTOLOGICAL"},
        ))
        delta.add_node(SemanticNode(
            norm_id, "VALUE", norm_label, provenance,
            {"framework": "DEONTOLOGICAL", "norm_kind": item.norm_kind},
        ))
        delta.add_node(SemanticNode(
            assessment_id, "ASSESSMENT", f"Deontological status for {item.action_id}",
            provenance,
            {
                "framework": "DEONTOLOGICAL", "specialist": specialist,
                "assessment_role": "PRIMARY",
                "cycle": cycle, "canonical_action_id": item.action_id,
                "verdict": committed_verdict, "proposed_verdict": item.verdict,
                "relation": committed_relation, "proposed_relation": item.relation,
                "norm_kind": item.norm_kind, "norm_node_id": norm_id,
                "party_node_id": party_id, "bearer_node_id": bearer_id,
                "competing_norm": item.competing_norm,
                "competing_norm_kind": item.competing_norm_kind,
                "competing_relation": item.competing_relation,
                "competing_protected_party": competing_party_label,
                "competing_reason": item.competing_reason,
                "governing_norm": item.governing_norm,
                "priority_basis": item.priority_basis,
                "priority_rule": item.priority_rule,
                "protected_standing": item.protected_standing,
                "competing_protected_standing": item.competing_protected_standing,
                "coercion_kind": item.coercion_kind,
                "coercive_actor": item.coercive_actor,
                "coerced_party": item.coerced_party,
                "public_justification": item.public_justification,
                "reciprocity_status": item.reciprocity_status,
                "necessity_status": item.necessity_status,
                "authorization_status": item.authorization_status,
                "derivation": item.derivation,
                "resolution_status": item.resolution_status,
                "calibration_errors": list(calibration.errors),
                "calibration_support_node_ids": list(calibration.support_node_ids),
                "evidence_basis": item.evidence_basis,
                "epistemic_status": epistemic_status, "reason": item.reason,
            },
        ))
        delta.add_edge(SemanticEdge(
            action.id, "HAS_ASSESSMENT", assessment_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, relation_edges[committed_relation], norm_id,
            justification=item.reason, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, "AFFECTS", party_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            assessment_id, "HAS_ACTOR", bearer_id, provenance=provenance,
        ))
        delta.add_node(SemanticNode(
            competing_norm_id, "VALUE", competing_norm_label, provenance,
            {
                "framework": "DEONTOLOGICAL",
                "norm_kind": item.competing_norm_kind,
                "norm_role": "COMPETING",
            },
        ))
        delta.add_node(SemanticNode(
            competing_party_id, "TARGET", competing_party_label, provenance,
            {
                "framework": "DEONTOLOGICAL", "norm_role": "COMPETING",
                "grounded_node_ids": [party.id for party in competing_matched],
            },
        ))
        delta.add_node(SemanticNode(
            competing_assessment_id, "ASSESSMENT",
            f"Competing Deontological norm for {item.action_id}", provenance,
            {
                "framework": "DEONTOLOGICAL", "specialist": specialist,
                "assessment_role": "COMPETING", "cycle": cycle,
                "canonical_action_id": item.action_id,
                "verdict": "NORM_BURDEN", "proposed_verdict": "NORM_BURDEN",
                "relation": item.competing_relation,
                "proposed_relation": item.competing_relation,
                "norm_kind": item.competing_norm_kind,
                "norm_node_id": competing_norm_id,
                "party_node_id": competing_party_id,
                "bearer_node_id": bearer_id,
                "evidence_basis": item.evidence_basis,
                "epistemic_status": "FRAMEWORK_INTERPRETATION",
                "reason": item.competing_reason,
            },
        ))
        delta.add_edge(SemanticEdge(
            action.id, "HAS_ASSESSMENT", competing_assessment_id,
            provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            competing_assessment_id, relation_edges[item.competing_relation], competing_norm_id,
            justification=item.competing_reason, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            competing_assessment_id, "AFFECTS", competing_party_id,
            provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            competing_assessment_id, "HAS_ACTOR", bearer_id,
            provenance=provenance,
        ))
        delta.add_node(SemanticNode(
            verdict_id, "DECISION", f"Deontological verdict for {item.action_id}",
            provenance,
            {
                "framework": "DEONTOLOGICAL", "specialist": specialist,
                "cycle": cycle, "canonical_action_id": item.action_id,
                "verdict": committed_verdict, "proposed_verdict": item.verdict,
                "governing_norm": item.governing_norm,
                "priority_basis": item.priority_basis,
                "priority_rule": item.priority_rule,
                "derivation": item.derivation,
                "resolution_status": item.resolution_status,
                "coercion_kind": item.coercion_kind,
                "authorization_status": item.authorization_status,
                "epistemic_status": epistemic_status,
            },
        ))
        delta.add_edge(SemanticEdge(
            action.id, "HAS_VERDICT", verdict_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            verdict_id, "RESOLVES", assessment_id, provenance=provenance,
        ))
        delta.add_edge(SemanticEdge(
            verdict_id, "RESOLVES", competing_assessment_id, provenance=provenance,
        ))
        governing_norm_id = (
            norm_id if item.governing_norm == "PRIMARY"
            else competing_norm_id if item.governing_norm == "COMPETING"
            else ""
        )
        if governing_norm_id:
            delta.add_edge(SemanticEdge(
                verdict_id, "GOVERNED_BY", governing_norm_id,
                justification=item.priority_rule, provenance=provenance,
            ))
        for party in matched:
            delta.add_edge(SemanticEdge(
                assessment_id, "SUPPORTED_BY", party.id, provenance=provenance,
            ))
        for party in competing_matched:
            delta.add_edge(SemanticEdge(
                competing_assessment_id, "SUPPORTED_BY", party.id,
                provenance=provenance,
            ))
        committed.append({
            "assessment_node_id": assessment_id,
            "canonical_action_id": item.action_id,
            "verdict": committed_verdict,
            "proposed_verdict": item.verdict,
            "norm_kind": item.norm_kind,
            "norm": norm_label,
            "relation": committed_relation,
            "proposed_relation": item.relation,
            "duty_bearer": bearer_label,
            "protected_party": party_label,
            "competing_norm": item.competing_norm,
            "competing_norm_kind": item.competing_norm_kind,
            "competing_relation": item.competing_relation,
            "competing_protected_party": competing_party_label,
            "competing_reason": item.competing_reason,
            "governing_norm": item.governing_norm,
            "priority_basis": item.priority_basis,
            "priority_rule": item.priority_rule,
            "protected_standing": item.protected_standing,
            "competing_protected_standing": item.competing_protected_standing,
            "coercion_kind": item.coercion_kind,
            "coercive_actor": item.coercive_actor,
            "coerced_party": item.coerced_party,
            "public_justification": item.public_justification,
            "reciprocity_status": item.reciprocity_status,
            "necessity_status": item.necessity_status,
            "authorization_status": item.authorization_status,
            "derivation": item.derivation,
            "resolution_status": item.resolution_status,
            "calibration_errors": list(calibration.errors),
            "calibration_support_node_ids": list(calibration.support_node_ids),
            "evidence_basis": item.evidence_basis,
            "epistemic_status": epistemic_status,
            "reason": item.reason,
        })

    if len(committed) != len(validated.assessments):
        record = GraphTransactionRecord(
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            warnings or ["Deontological ledger could not bind every action"],
            previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record
    base = SemanticGraph(
        nodes={key: node for key, node in store.graph.nodes.items() if key not in replace_ids},
        edges=[
            edge for edge in store.graph.edges
            if edge.source not in replace_ids and edge.target not in replace_ids
        ],
    )
    try:
        prospective = merge_graphs([base, delta])
    except ValueError as exc:
        record = GraphTransactionRecord(
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            [f"graph merge rejected: {exc}"], previous_state_preserved=True,
            retryable=True,
        )
        store.transactions.append(record)
        return record
    validation = validate_graph(prospective)
    if not validation.valid:
        record = GraphTransactionRecord(
            cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER", "REJECTED", raw,
            validation.errors, previous_state_preserved=True, retryable=True,
        )
        store.transactions.append(record)
        return record
    store.graph = prospective
    record = GraphTransactionRecord(
        cycle, specialist, "DEONTOLOGICAL_DUTY_LEDGER",
        "COMMITTED_WITH_UNCERTAINTY" if warnings else "COMMITTED",
        {"submitted": raw, "committed_assessments": committed}, warnings,
        previous_state_preserved=False,
    )
    store.transactions.append(record)
    return record


def committed_deontological_assessments(graph: SemanticGraph) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for node in graph.nodes.values():
        if (
            node.kind != "ASSESSMENT"
            or node.attributes.get("framework") != "DEONTOLOGICAL"
            or node.attributes.get("assessment_role") == "COMPETING"
        ):
            continue
        attributes = dict(node.attributes)

        def resolved_label(reference_field: str, fallback: str = "") -> str:
            reference = str(attributes.get(reference_field, ""))
            target = graph.nodes.get(reference)
            return target.label if target is not None else fallback

        # Keep graph identity authoritative while also projecting the resolved
        # labels consumed by the framework renderer.  Previously the renderer
        # received only node IDs and silently fell back to "the protected
        # party" / "a protected claim", flattening a richer committed ledger.
        attributes.setdefault("protected_party", resolved_label("party_node_id"))
        attributes.setdefault("norm", resolved_label("norm_node_id"))
        attributes.setdefault("duty_bearer", resolved_label("bearer_node_id"))
        values.append({
            "assessment_node_id": node.id,
            **attributes,
            "provenance": list(node.provenance),
        })
    return sorted(values, key=lambda item: str(item.get("canonical_action_id", "")))
