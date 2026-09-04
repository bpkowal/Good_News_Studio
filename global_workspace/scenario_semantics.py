"""Deterministic compilation of explicit scenario facts into typed graph nodes.

This module deliberately recognizes *semantic classes*, not particular dilemmas.
It turns explicit observability language and the workspace action set into stable
identifiers before any LLM is asked to interpret their ethical significance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Sequence
import unicodedata

from .action_identity import (
    add_action_identity_subgraph,
    action_clause_looks_complete,
    compile_action_identity,
    graph_action_key,
)
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode
from .semantic_roles import (
    RELATION_DOWNSTREAM_BENEFIT,
    RELATION_FOREGONE_BENEFIT,
    extract_grounded_effects,
)


@dataclass(frozen=True, slots=True)
class GroundedActionEffect:
    effect_id: str
    action_id: str
    source_clause_id: str
    consequence_id: str
    affected_subject: str
    affected_subject_node_ids: tuple[str, ...]
    dimension: str
    direction: str
    magnitude_or_qualifier: str
    provenance: tuple[str, ...]
    confidence: float
    epistemic_status: str
    affected_subject_quantities: tuple[str, ...] = ()
    likelihood_qualifiers: tuple[str, ...] = ()
    scope_qualifiers: tuple[str, ...] = ()
    temporal_qualifiers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "action_id": self.action_id,
            "source_clause_id": self.source_clause_id,
            "consequence_id": self.consequence_id,
            "affected_subject": self.affected_subject,
            "affected_subject_node_ids": list(self.affected_subject_node_ids),
            "dimension": self.dimension,
            "direction": self.direction,
            "magnitude_or_qualifier": self.magnitude_or_qualifier,
            "provenance": list(self.provenance),
            "confidence": self.confidence,
            "epistemic_status": self.epistemic_status,
            "affected_subject_quantities": list(self.affected_subject_quantities),
            "likelihood_qualifiers": list(self.likelihood_qualifiers),
            "scope_qualifiers": list(self.scope_qualifiers),
            "temporal_qualifiers": list(self.temporal_qualifiers),
        }


_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_ACTION_LABEL = re.compile(
    r"(?<![A-Za-z0-9])\$?A\s*(?:_\s*)?(?:\{\s*)?"
    r"(?P<index>[0-9₀₁₂₃₄₅₆₇₈₉]+)(?:\s*\})?\$?(?![A-Za-z0-9])",
)

_CLAIM_BENEFIT = re.compile(
    r"\b(?:benefit\w*|improv\w*|rais\w*|provid\w*|suppl\w*|guarante\w*|"
    r"protect\w*|preserv\w*|support\w*|access|adequate|reliable)\b",
    re.IGNORECASE,
)
_CLAIM_ADVERSE = re.compile(
    r"\b(?:adverse|harm\w*|wors\w*|depriv\w*|leav\w*|lack\w*|without|"
    r"understaff\w*|underfund\w*|delay\w*|pollut\w*|poor)\b",
    re.IGNORECASE,
)
_GROUNDING_STOPWORDS = {
    "action", "case", "choose", "choosing", "directly", "most", "more",
    "less", "least", "than", "that", "this", "their", "them", "they",
    "with", "without", "from", "into", "under", "while", "whose", "all",
    "the", "and", "for", "but", "may", "its", "own",
}


def _grounding_words(text: str) -> set[str]:
    aliases = {
        "poorest": "disadvantaged", "poor": "disadvantaged",
        "impoverished": "disadvantaged", "vulnerable": "disadvantaged",
        "patients": "patient", "populations": "population",
        "communities": "community", "residents": "resident",
        "clinics": "clinic", "resources": "resource",
        "school": "student", "schools": "student",
        "student": "student", "students": "student",
        "family": "family", "families": "family",
    }
    return {
        aliases.get(word, word[:-1] if word.endswith("s") and len(word) > 4 else word)
        for word in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(word) >= 3 and word not in _GROUNDING_STOPWORDS
    }


def grounded_action_claim_evidence(
    graph: SemanticGraph | None,
    action_id: str,
    claim: str,
) -> list[str]:
    """Return scenario consequence IDs that support one factual action claim.

    This service is framework-neutral: it validates only action ownership,
    consequence direction, affected targets, and scenario provenance.
    """
    if graph is None or action_id not in graph.nodes:
        return []
    claim_words = _grounding_words(claim)
    claim_polarities = set()
    if _CLAIM_BENEFIT.search(claim):
        claim_polarities.add("BENEFICIAL")
    if _CLAIM_ADVERSE.search(claim):
        claim_polarities.add("ADVERSE")
    if not claim_words or not claim_polarities:
        return []
    supported: list[str] = []
    for polarity in claim_polarities:
        supported.extend(
            effect.consequence_id
            for effect in query_grounded_action_effects(
                graph, action_id, claim=claim, direction=polarity,
            )
        )
    return list(dict.fromkeys(supported))


_SHARED_DIMENSION_ALIASES = {
    "BASIC_LIBERTY": "LIBERTY_AUTONOMY",
    "LIBERTY_AUTONOMY": "LIBERTY_AUTONOMY",
    "INCOME_WEALTH": "MATERIAL_FLOOR",
    "MATERIAL_FLOOR": "MATERIAL_FLOOR",
    "OPPORTUNITY": "OPPORTUNITY_ACCESS",
    "OPPORTUNITY_ACCESS": "OPPORTUNITY_ACCESS",
    "BASIC_INTEREST_SECURITY": "BASIC_SECURITY",
    "BASIC_SECURITY": "BASIC_SECURITY",
    "POWERS_OFFICES": "INSTITUTIONAL_ACCESS",
    "SELF_RESPECT": "SOCIAL_STANDING",
    "OTHER_PRIMARY_GOOD": "OTHER",
}


def normalize_shared_dimension(dimension: object) -> str:
    """Translate framework-local dimension names into neutral shared facts."""
    normalized = str(dimension or "UNKNOWN").strip().upper()
    return _SHARED_DIMENSION_ALIASES.get(normalized, normalized)


_CONCRETE_PLANNING_DETAILS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("legislature", re.compile(r"\b(?:legislature|parliament|congress|senate)\b", re.I)),
    ("political coalition", re.compile(r"\b(?:coalition|opposition party|lawmakers?|legislators?)\b", re.I)),
    ("legislative procedure", re.compile(r"\b(?:bill|filibuster|veto|committee hearing|floor vote)\b", re.I)),
    ("judicial actor", re.compile(r"\b(?:court|judge|lawsuit|injunction|litigation)\b", re.I)),
    ("administrative actor", re.compile(r"\b(?:agency|regulator|ministry|department|bureau)\b", re.I)),
    ("external supplier", re.compile(r"\b(?:vendor|contractor|supplier|bank|insurer)\b", re.I)),
    ("invented disruption", re.compile(r"\b(?:strike|sabotage|protest|riot|election)\b", re.I)),
)


def classify_planning_failure_grounding(
    necessary_condition: str,
    failure_condition: str,
    scenario: str,
    actions: Sequence[str],
    provenance_note: str = "",
) -> tuple[str, list[str]]:
    """Classify planning branches without turning plausibility into provenance."""
    current_text = " ".join((scenario, *map(str, actions))).casefold()
    proposed_text = " ".join((necessary_condition, failure_condition)).casefold()
    unsupported = [
        label for label, pattern in _CONCRETE_PLANNING_DETAILS
        if pattern.search(proposed_text) and not pattern.search(current_text)
    ]
    if unsupported:
        return "REJECTED_UNGROUNDED_FAILURE_CONDITION", unsupported

    current_words = set(re.findall(r"[a-z]{4,}", current_text))
    proposed_words = set(re.findall(r"[a-z]{4,}", proposed_text))
    if len(current_words & proposed_words) >= 2:
        return "GROUNDED_FAILURE_CONDITION", []
    provenance_words = set(re.findall(r"[a-z]{4,}", provenance_note.casefold()))
    if current_words & proposed_words and len(current_words & provenance_words) >= 2:
        return "MECHANISM_DERIVED_FAILURE_CONDITION", []

    mechanism_terms: set[str] = set()
    if re.search(r"\b(?:opt[- ]?in|voluntar\w*|donat\w*|contribut\w*|fund\w*)\b", current_text):
        mechanism_terms.update({
            "voluntary", "participation", "participants", "uptake", "donation",
            "donations", "contribution", "contributions", "funding", "finance",
            "financing", "assistance", "sufficient", "insufficient",
        })
    if proposed_words & mechanism_terms:
        return "MECHANISM_DERIVED_FAILURE_CONDITION", []
    return "PLAUSIBLE_HYPOTHETICAL_FAILURE_CONDITION", []


_DIMENSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("LIBERTY_AUTONOMY", re.compile(
        r"\b(?:association|autonom\w*|bodily|choice|consent|detain\w*|freedom|integrity|libert\w*|"
        r"movement|occupational|privacy|rights?|speech|vote|worship)\b", re.I,
    )),
    ("OPPORTUNITY_ACCESS", re.compile(
        r"\b(?:access|career|education|employment|office|opportunit\w*|school|training|transit)\b", re.I,
    )),
    ("BASIC_SECURITY", re.compile(
        r"\b(?:food|health|housing|medical|safety|security|shelter|survival|water)\b", re.I,
    )),
    ("MATERIAL_FLOOR", re.compile(
        r"\b(?:aid|asset\w*|assistance|economic|funding|income|material|money|"
        r"poverty|poor|property|redistribut\w*|resource\w*|subsid\w*|wage\w*|wealth)\b", re.I,
    )),
)


def _grounded_effect_dimension(consequence: SemanticNode, targets: list[SemanticNode]) -> str:
    explicit = str(consequence.attributes.get("dimension") or "").strip()
    if explicit:
        return explicit
    protected = [
        target.label for target in targets
        if target.attributes.get("semantic_role") == "PROTECTED_INTEREST"
    ]
    if protected:
        return "LIBERTY_AUTONOMY"
    direct_text = " ".join([
        consequence.label,
        *map(str, consequence.attributes.get("targets", [])),
        *map(str, consequence.attributes.get("affected_resources", [])),
        *map(str, consequence.attributes.get("protected_interests", [])),
        *(target.label for target in targets),
    ])
    for dimension, pattern in _DIMENSION_PATTERNS:
        if pattern.search(direct_text):
            return dimension
    source_text = " ".join(map(str, consequence.provenance))
    for dimension, pattern in _DIMENSION_PATTERNS:
        if pattern.search(source_text):
            return dimension
    return "OTHER"


def _plausible_affected_subject(label: str) -> bool:
    return bool(re.search(
        r"\b(?:citizens?|communities|community|famil(?:y|ies)|households?|"
        r"patients?|people|persons?|populations?|residents?|schools?|students?|"
        r"workers?|employees?|owners?|affluent|elite|poor|impoverished|vulnerable|"
        r"children|adults|users?|customers?|farmers?)\b",
        str(label), re.IGNORECASE,
    ))


def project_grounded_action_effects(graph: SemanticGraph) -> list[GroundedActionEffect]:
    """Project canonical scenario facts before any framework interpretation."""
    effects: list[GroundedActionEffect] = []
    for action in graph.nodes.values():
        if action.kind != "ACTION":
            continue
        action_id = str(action.attributes.get("canonical_action_id", action.id))
        for edge in graph.outgoing(action.id, "HAS_CONSEQUENCE"):
            consequence = graph.nodes.get(edge.target)
            if consequence is None or consequence.kind != "CONSEQUENCE":
                continue
            if consequence.attributes.get("framework"):
                continue
            scenario_grounded = consequence.attributes.get("scenario_grounded") is True
            action_text_grounded = "deterministic_action_identity" in consequence.provenance
            if not scenario_grounded and not action_text_grounded:
                continue
            source_text = " ".join(map(str, consequence.provenance))
            # Introductory role/context clauses are not action consequences.
            # In particular, a phrase such as "child-welfare system" must not
            # become a generic welfare improvement for every action.
            welfare_evidence_text = re.sub(
                r"\b(?:child|social)[- ]welfare\b", "", source_text, flags=re.I,
            )
            if (
                consequence.label.casefold() == "welfare"
                and not re.search(
                    r"\b(?:wellbeing|well-being|welfare)\b", welfare_evidence_text, re.I,
                )
            ):
                continue
            targets = [
                graph.nodes[target_edge.target]
                for target_edge in graph.outgoing(consequence.id, "AFFECTS")
                if target_edge.target in graph.nodes
                and graph.nodes[target_edge.target].kind == "TARGET"
            ]
            dimension = _grounded_effect_dimension(consequence, targets)
            subject_targets = [
                target for target in targets
                if target.attributes.get("semantic_role") == "AFFECTED_SUBJECT"
            ]
            if not subject_targets:
                subject_targets = [
                    target for target in targets
                    if target.attributes.get("semantic_role") not in {
                        "AFFECTED_RESOURCE", "PROTECTED_INTEREST",
                    }
                    and _plausible_affected_subject(target.label)
                ]
            subject_labels = list(dict.fromkeys(
                target.label.strip() for target in subject_targets if target.label.strip()
            ))
            direct_effect_text = " ".join([
                consequence.label,
                *map(str, consequence.attributes.get("targets", [])),
                source_text,
            ])
            if dimension == "LIBERTY_AUTONOMY" and re.search(
                r"\b(?:parental|parents?|famil(?:y|ies))\b", direct_effect_text, re.I,
            ):
                subject_labels = ["parents/families"]
            if not subject_labels:
                subject_labels = [
                    str(value).strip()
                    for value in consequence.attributes.get("affected_subjects", [])
                    if str(value).strip()
                ]
            if not subject_labels:
                subject_labels = ["affected constituency"]
            polarity = str(consequence.attributes.get("polarity", "")).upper()
            relation = str(consequence.attributes.get("relation", "")).upper()
            predicate = consequence.label.casefold()
            direction = (
                "FOREGOES" if polarity == "FOREGONE" or relation == RELATION_FOREGONE_BENEFIT
                else "PRESERVES" if polarity == "BENEFICIAL" and predicate.startswith(("preserv", "protect"))
                else "IMPROVES" if polarity == "BENEFICIAL"
                else "WORSENS" if polarity == "ADVERSE"
                else "UNCERTAIN"
            )
            quantities = [
                str(value).strip()
                for value in consequence.attributes.get("quantities", [])
                if str(value).strip()
            ]
            qualifier = quantities[0] if quantities else str(
                consequence.attributes.get("probability", "STATED")
            ).upper()
            source_clause_id = str(consequence.attributes.get("source_clause_id", ""))
            for subject in subject_labels:
                digest = hashlib.sha256(
                    f"{consequence.id}|{dimension}|{subject.casefold()}".encode("utf-8")
                ).hexdigest()[:16]
                effects.append(GroundedActionEffect(
                    effect_id=(
                        str(consequence.attributes.get("world_effect_id", "")).strip()
                        or f"GROUNDED_EFFECT:{digest}"
                    ),
                    action_id=action_id,
                    source_clause_id=source_clause_id,
                    consequence_id=consequence.id,
                    affected_subject=subject,
                    affected_subject_node_ids=tuple(
                        target.id for target in subject_targets
                        if target.label.strip().casefold() == subject.casefold()
                    ),
                    dimension=dimension,
                    direction=direction,
                    magnitude_or_qualifier=qualifier or "STATED",
                    provenance=tuple(consequence.provenance),
                    confidence=0.95 if scenario_grounded else 0.82,
                    epistemic_status=(
                        "SCENARIO_GROUNDED" if scenario_grounded else "ACTION_TEXT_GROUNDED"
                    ),
                    affected_subject_quantities=tuple(dict.fromkeys(
                        str(value).strip() for value in [
                            *[
                                item
                                for target in subject_targets
                                if target.label.strip().casefold() == subject.casefold()
                                for item in target.attributes.get("quantities", [])
                            ],
                            *list(consequence.attributes.get("party_quantities", [])),
                        ] if str(value).strip()
                    )),
                    likelihood_qualifiers=tuple(dict.fromkeys(
                        str(value).strip()
                        for value in consequence.attributes.get("likelihood_qualifiers", [])
                        if str(value).strip()
                    )),
                    scope_qualifiers=tuple(dict.fromkeys(
                        str(value).strip()
                        for value in consequence.attributes.get("scope_qualifiers", [])
                        if str(value).strip()
                    )),
                    temporal_qualifiers=tuple(dict.fromkeys(
                        str(value).strip()
                        for value in consequence.attributes.get("temporal_qualifiers", [])
                        if str(value).strip()
                    )),
                ))
        if str(action.attributes.get("source_type", "")).upper() == "SYNTHESIS_PROPOSAL":
            proposal_text = str(action.attributes.get("proposal_text", action.label))
            existing_dimensions = {
                effect.dimension for effect in effects if effect.action_id == action_id
            }
            subject_match = re.search(
                r"\b(?:to|for)\s+(?P<subject>(?:the\s+)?(?:poorest|poor|impoverished|"
                r"vulnerable|low-income)?\s*(?:households?|families|communities|residents|people))\b",
                proposal_text, re.IGNORECASE,
            )
            material_subject = (
                subject_match.group("subject").strip() if subject_match else "affected constituency"
            )
            synthesis_specs: list[tuple[str, str, str, str]] = []
            if _DIMENSION_PATTERNS[3][1].search(proposal_text) or re.search(
                r"\b(?:donat\w*|fund\w*|poorest|need(?:y|iest))\b",
                proposal_text, re.IGNORECASE,
            ):
                material_direction = (
                    "IMPROVES" if re.search(
                        r"\b(?:aid|donat\w*|fund\w*|provide\w*|rais\w*|support\w*|subsid\w*)\b",
                        proposal_text, re.IGNORECASE,
                    ) else "UNCERTAIN"
                )
                synthesis_specs.append((
                    "MATERIAL_FLOOR", material_direction, material_subject,
                    "potential material effect stated by synthesis action",
                ))
            if re.search(
                r"\b(?:opt[- ]?in|voluntar\w*|with\s+consent|consensual)\b",
                proposal_text, re.IGNORECASE,
            ):
                synthesis_specs.append((
                    "LIBERTY_AUTONOMY", "PRESERVES", "participants",
                    "preserves voluntary participation",
                ))
            for dimension, direction, subject, claim in synthesis_specs:
                if dimension in existing_dimensions:
                    continue
                digest = hashlib.sha256(
                    f"{action_id}|synthesis|{dimension}|{subject}".encode("utf-8")
                ).hexdigest()[:16]
                effects.append(GroundedActionEffect(
                    effect_id=f"GROUNDED_EFFECT:{digest}",
                    action_id=action_id,
                    source_clause_id="SYNTHESIS_PROPOSAL",
                    # The admitted synthesis ACTION is the authoritative source
                    # node; no virtual consequence node is invented here.
                    consequence_id=action.id,
                    affected_subject=subject,
                    affected_subject_node_ids=(),
                    dimension=dimension,
                    direction=direction,
                    magnitude_or_qualifier="POTENTIAL",
                    provenance=(
                        "synthesis_proposal",
                        proposal_text,
                        claim,
                    ),
                    confidence=0.78,
                    epistemic_status="SYNTHESIS_GROUNDED",
                ))
    return sorted(effects, key=lambda effect: (
        effect.action_id, effect.dimension, effect.affected_subject, effect.consequence_id,
    ))


def query_grounded_action_effects(
    graph: SemanticGraph | None,
    action_id: str,
    *,
    claim: str = "",
    affected_subject: str = "",
    dimension: str = "",
    direction: str = "",
) -> list[GroundedActionEffect]:
    """Shared factual query used by every normative framework.

    The query never evaluates moral importance. It only filters canonical
    scenario effects by action ownership, subject, dimension, direction, and
    optional lexical claim overlap.
    """
    if graph is None:
        return []
    requested_direction = str(direction).strip().upper()
    direction_aliases = {
        "BENEFIT": {"IMPROVES", "PRESERVES"},
        "BENEFICIAL": {"IMPROVES", "PRESERVES"},
        "HARM": {"WORSENS"},
        "ADVERSE": {"WORSENS"},
    }
    allowed_directions = direction_aliases.get(
        requested_direction, {requested_direction} if requested_direction else set()
    )
    claim_words = _grounding_words(claim)
    subject_words = _grounding_words(affected_subject)
    ranked: list[tuple[int, GroundedActionEffect]] = []
    projected = project_grounded_action_effects(graph)
    action_effects = [effect for effect in projected if effect.action_id == str(action_id).strip()]
    if any(effect.epistemic_status == "SCENARIO_GROUNDED" for effect in action_effects):
        action_effects = [
            effect for effect in action_effects
            if effect.epistemic_status == "SCENARIO_GROUNDED"
        ]
    for effect in action_effects:
        if dimension and effect.dimension != normalize_shared_dimension(dimension):
            continue
        if allowed_directions and effect.direction not in allowed_directions:
            continue
        if subject_words and not (subject_words & _grounding_words(effect.affected_subject)):
            continue
        consequence = graph.nodes.get(effect.consequence_id)
        evidence_text = " ".join((
            consequence.label if consequence is not None else "",
            effect.affected_subject,
            effect.dimension,
            " ".join(map(str, consequence.attributes.get("targets", [])))
            if consequence is not None else "",
            " ".join(map(str, consequence.attributes.get("affected_resources", [])))
            if consequence is not None else "",
        ))
        overlap = len(claim_words & _grounding_words(evidence_text))
        if claim_words and overlap == 0:
            continue
        ranked.append((overlap, effect))
    return [effect for _score, effect in sorted(
        ranked,
        key=lambda item: (-item[0], item[1].consequence_id, item[1].affected_subject),
    )]


def segment_scenario_clauses(scenario: str) -> list[dict[str, str]]:
    """Create stable, auditable source spans for action-grounding proposals."""
    normalized = " ".join(normalize_action_labels(str(scenario)).split())
    parts = [
        part.strip(" ,;:")
        for part in re.split(r"(?<=[.!?;])\s+", normalized)
        if part.strip(" ,;:")
    ]
    clauses: list[str] = []
    label_pattern = re.compile(r"\b(?:policy|action|option)\s+(?:A|B|A\d+)\b", re.I)
    for part in parts:
        if part.rstrip().endswith("?"):
            clauses.append(part)
            continue
        matches = list(label_pattern.finditer(part))
        if len(matches) <= 1:
            clauses.append(part)
            continue
        prefix = part[:matches[0].start()].strip(" ,;:")
        if prefix:
            clauses.append(prefix)
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(part)
            clause = part[match.start():end].strip(" ,;:")
            if clause:
                clauses.append(clause)
    return [
        {"clause_id": f"C{index}", "text": clause}
        for index, clause in enumerate(dict.fromkeys(clauses))
        if len(re.findall(r"[A-Za-z0-9]+", clause)) >= 3
    ]


def _typed_consequence_roles(
    text: str, predicate: str,
) -> tuple[list[str], list[str]]:
    """Recover explicit affected-subject/resource spans from deprivation syntax."""
    if predicate != "deprivation":
        return [], []
    normalized = " ".join(str(text).split()).strip(" ,;:.")
    patterns = (
        re.compile(
            r"\bleav(?:e|es|ing|t)\s+(?P<subject>.+?)\s+"
            r"(?:with\s+(?:only\s+|(?:a\s+)?(?:lower|reduced|inadequate|insufficient)\s+)|without\s+)"
            r"(?P<resource>.+?)(?=$|[.;])",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bleav(?:e|es|ing|t)\s+(?P<subject>.+?)\s+in\s+"
            r"(?P<resource>.+?)(?=$|[.;])",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdepriv(?:e|es|ing|ed)\s+(?P<subject>.+?)\s+of\s+"
            r"(?P<resource>.+?)(?=$|[.;])",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(normalized)
        if not match:
            continue
        subject = match.group("subject").strip(" ,;:")
        resource = match.group("resource").strip(" ,;:")
        # Stop at a new contrast/action clause if punctuation was omitted.
        resource = re.split(
            r"\b(?:but|while|whereas|although|however)\b", resource,
            maxsplit=1, flags=re.IGNORECASE,
        )[0].strip(" ,;:")
        if subject and resource:
            return [subject], [resource]
    return [], []


def _typed_liberty_roles(
    text: str, predicate: str,
) -> tuple[list[str], list[str]]:
    """Recover rights-bearing subjects and protected interests from explicit text."""
    normalized = " ".join(str(text).split()).strip(" ,;:.")
    liberty_predicates = {
        "detain", "compel", "liberty_infringement", "preserve", "prevent",
    }
    interests = [
        " ".join(match.group(0).casefold().split())
        for match in re.finditer(
            r"\b(?:bodily\s+(?:autonomy|integrity)|individual\s+rights?|"
            r"legal\s+rights?|personal\s+libert\w*|physical\s+liberty|"
            r"occupational\s+choice|free\s+choice\s+of\s+(?:career|association)|"
            r"freedom\s+of\s+movement|privacy)\b",
            normalized,
            re.IGNORECASE,
        )
    ] if predicate in liberty_predicates else []
    subjects: list[str] = []
    if predicate in {"detain", "compel"}:
        match = re.search(
            r"\b(?:detain\w*|confin\w*|compel\w*|forc\w*)\s+"
            r"(?:and\s+us(?:e|es|ed|ing)\s+)?(?P<subject>.+?)\s+"
            r"(?:for|into|to|without)\b",
            normalized,
            re.IGNORECASE,
        )
        if match:
            subject = match.group("subject").strip(" ,;:")
            subject = re.sub(
                r"^(?:a|an|the)\s+(?:small\s+)?(?:group\s+of\s+)?",
                "", subject, flags=re.IGNORECASE,
            ).strip()
            if subject:
                subjects.append(subject)
        if re.search(r"\b(?:involuntary|forced|without\s+consent)\b", normalized, re.I):
            interests.append("bodily autonomy")
    elif predicate == "liberty_infringement":
        match = re.search(
            r"\bstrip(?:s|ped|ping)?\s+(?P<subject>.+?)\s+of\s+",
            normalized,
            re.IGNORECASE,
        )
        if match:
            subject = match.group("subject").strip(" ,;:")
            if subject:
                subjects.append(subject)
    return list(dict.fromkeys(subjects)), list(dict.fromkeys(interests))


def _attach_grounded_action_sources(
    graph: SemanticGraph,
    action_source_groundings: dict[str, dict[str, object]],
    *,
    attach_consequences: bool = True,
) -> None:
    """Attach cited scenario clauses and their generic consequence identities."""
    for action_id, grounding in action_source_groundings.items():
        if action_id not in graph.nodes or not isinstance(grounding, dict):
            continue
        clauses = grounding.get("clauses", [])
        if not isinstance(clauses, list):
            continue
        for source_index, clause in enumerate(clauses):
            if not isinstance(clause, dict):
                continue
            clause_id = str(clause.get("clause_id", f"C{source_index}"))
            text = " ".join(str(clause.get("text", "")).split())
            if not text:
                continue
            source_id = f"ACTION_SOURCE:{action_id}:{clause_id}"
            provenance = (f"scenario_clause:{clause_id}", text)
            is_question_context = text.rstrip().endswith("?")
            graph.add_node(SemanticNode(
                source_id, "EVIDENCE", text, provenance,
                {
                    "clause_id": clause_id,
                    "grounding_status": "MODEL_MAPPED_VALIDATED",
                    "evidence_role": (
                        "DECISION_QUESTION_CONTEXT"
                        if is_question_context else "SCENARIO_ASSERTION"
                    ),
                },
            ))
            graph.add_edge(SemanticEdge(
                action_id, "GROUNDED_IN", source_id, provenance=provenance,
            ))
            if not attach_consequences:
                continue
            # A choice question can restate both alternatives and remains useful
            # mapping provenance, but it does not assert that either consequence
            # follows from the action to which the model happened to attach it.
            if is_question_context:
                continue
            identity = compile_action_identity(text)
            for consequence_index, consequence in enumerate(identity.consequences):
                affected_subjects, affected_resources = _typed_consequence_roles(
                    text, consequence.predicate,
                )
                liberty_subjects, protected_interests = _typed_liberty_roles(
                    text, consequence.predicate,
                )
                affected_subjects = list(dict.fromkeys([
                    *affected_subjects, *liberty_subjects,
                ]))
                consequence_id = (
                    f"{action_id}:SCENARIO_CONSEQUENCE:{clause_id}:{consequence_index}"
                )
                graph.add_node(SemanticNode(
                    consequence_id, "CONSEQUENCE", consequence.predicate, provenance,
                    {
                        "polarity": consequence.polarity,
                        "targets": list(consequence.targets),
                        "quantities": list(consequence.quantities),
                        "probability": consequence.probability,
                        "scenario_grounded": True,
                        "source_clause_id": clause_id,
                        "affected_subjects": affected_subjects,
                        "affected_resources": affected_resources,
                        "protected_interests": protected_interests,
                    },
                ))
                graph.add_edge(SemanticEdge(
                    action_id, "HAS_CONSEQUENCE", consequence_id, provenance=provenance,
                ))
                graph.add_edge(SemanticEdge(
                    consequence_id, "SUPPORTED_BY", source_id, provenance=provenance,
                ))
                for target_index, target in enumerate(consequence.targets):
                    target_id = f"{consequence_id}:TARGET:{target_index}"
                    graph.add_node(SemanticNode(
                        target_id, "TARGET", target, provenance,
                        {"source_clause_id": clause_id},
                    ))
                    graph.add_edge(SemanticEdge(
                        consequence_id, "AFFECTS", target_id, provenance=provenance,
                    ))
                for role, values in (
                    ("AFFECTED_SUBJECT", affected_subjects),
                    ("AFFECTED_RESOURCE", affected_resources),
                    ("PROTECTED_INTEREST", protected_interests),
                ):
                    for role_index, value in enumerate(values):
                        target_id = f"{consequence_id}:{role}:{role_index}"
                        graph.add_node(SemanticNode(
                            target_id, "TARGET", value, provenance,
                            {"source_clause_id": clause_id, "semantic_role": role},
                        ))
                        graph.add_edge(SemanticEdge(
                            consequence_id, "AFFECTS", target_id, provenance=provenance,
                        ))


def normalize_action_labels(text: str) -> str:
    """Canonicalize common surface forms such as ``$A_0$`` and ``A₀``.

    Canonicalization happens before parsing so every downstream component uses
    the same stable action identifiers. It changes labels only; it does not infer
    an action, reorder alternatives, or alter their descriptions.
    """
    source = str(text)

    def replace(match: re.Match[str]) -> str:
        index = match.group("index").translate(_SUBSCRIPT_DIGITS)
        return f"A{index}"

    return _ACTION_LABEL.sub(replace, source)


def canonical_action_text(action: str) -> str:
    """Return a label-insensitive representation of one physical action.

    This is intentionally a lexical identity rather than a claim that arbitrary
    paraphrases are semantically equivalent.  It is sufficient for permutation
    tests in which the same action descriptions are presented under different
    A0/A1 labels, and it keeps presentation labels out of evaluator state.
    """
    text = unicodedata.normalize("NFKC", normalize_action_labels(str(action)))
    text = re.sub(
        r"^\s*(?:(?:action|option)\s+)?A\d+\s*[:.)\]-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # References to another run-local label are provenance, not action identity.
    text = re.sub(r"\bA\d+\b", "ACTION_REF", text, flags=re.IGNORECASE)
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def semantic_action_key(action: str) -> str:
    """Stable graph key, with a conservative lexical fallback when necessary."""
    return graph_action_key(action)


def resolved_semantic_action_keys(actions: Sequence[str]) -> list[str]:
    """Return graph keys, conservatively splitting within-set collisions.

    A graph abstraction may intentionally equate close paraphrases. If two
    distinct alternatives in the *same* decision collapse to one signature,
    however, treating them as one node would be unsafe. In that case an opaque
    lexical discriminator preserves both physical branches.
    """
    values = [str(action) for action in actions]
    keys = [semantic_action_key(action) for action in values]
    counts = {key: keys.count(key) for key in set(keys)}
    resolved = []
    for action, key in zip(values, keys):
        if counts[key] <= 1:
            resolved.append(key)
            continue
        lexical = canonical_action_text(action)
        suffix = hashlib.sha256(lexical.encode("utf-8")).hexdigest()[:12]
        resolved.append(f"{key}:lex:{suffix}")
    return resolved


def canonicalize_action_order(actions: Sequence[str]) -> list[str]:
    """Assign internal positions independently of source/presentation order.

    Presentation labels remain in ``source_action_legend``. The hash order is
    deliberately opaque: unlike alphabetical sorting, it does not systematically
    put particular verbs, stakeholders, or moral language into A0.
    """
    values = [str(action) for action in actions]
    resolved = resolved_semantic_action_keys(values)
    return [
        action for _, action in sorted(
            zip(resolved, values),
            key=lambda item: (item[0], canonical_action_text(item[1])),
        )
    ]


def _source_to_canonical_bindings(
    source_action_legend: dict[str, str],
    canonical_actions: Sequence[str],
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Resolve source action IDs without using their presentation positions."""
    canonical = list(canonical_actions)
    bindings: dict[str, str] = {}
    bases: dict[str, str] = {}
    for source_id, action in source_action_legend.items():
        exact_matches = [
            index for index, candidate in enumerate(canonical)
            if canonical_action_text(candidate) == canonical_action_text(action)
        ]
        graph_matches = [
            index for index, candidate in enumerate(canonical)
            if semantic_action_key(candidate) == semantic_action_key(action)
        ]
        matches = exact_matches or graph_matches
        if len(matches) != 1:
            return None
        bindings[source_id] = f"A{matches[0]}"
        bases[source_id] = (
            "EXACT_ACTION_IDENTITY" if exact_matches else "SEMANTIC_ACTION_IDENTITY"
        )
    if len(set(bindings.values())) != len(bindings):
        return None
    return bindings, bases


def build_presentation_action_mapping(
    scenario: str,
    source_action_legend: dict[str, str],
    canonical_actions: Sequence[str],
    *,
    source_labels_explicit: bool,
) -> list[dict[str, object]]:
    """Build passive display provenance for source labels and canonical IDs.

    This metadata is intentionally not part of the canonical scenario or any
    specialist input. It explains an already-completed normalization without
    allowing presentation order to influence deliberation.
    """
    resolved = _source_to_canonical_bindings(
        source_action_legend, canonical_actions,
    )
    if resolved is None:
        return []
    bindings, bases = resolved
    canonical = list(canonical_actions)
    source_ids = sorted(
        source_action_legend,
        key=lambda value: (
            (0, int(value[1:])) if value[1:].isdigit() else (1, value)
        ),
    )
    alpha_option = all(
        re.search(rf"\boption\s+{letter}\b", scenario, re.IGNORECASE)
        for letter in ("A", "B")
    )
    alpha_action = all(
        re.search(rf"\baction\s+{letter}\b", scenario, re.IGNORECASE)
        for letter in ("A", "B")
    )

    mapping: list[dict[str, object]] = []
    for source_position, source_id in enumerate(source_ids):
        if not source_labels_explicit:
            source_label = f"Presented option {source_position + 1}"
        elif alpha_option and source_position < 26:
            source_label = f"Original Option {chr(ord('A') + source_position)}"
        elif alpha_action and source_position < 26:
            source_label = f"Original Action {chr(ord('A') + source_position)}"
        else:
            source_label = f"Original {source_id}"
        canonical_id = bindings[source_id]
        canonical_index = int(canonical_id[1:])
        mapping.append({
            "source_label": source_label,
            "source_position": source_position,
            "source_action": source_action_legend[source_id],
            "canonical_action_id": canonical_id,
            "canonical_action": canonical[canonical_index],
            "mapping_basis": bases[source_id],
        })
    return mapping


def canonicalize_deliberation_scenario(
    scenario: str,
    source_action_legend: dict[str, str],
    canonical_actions: Sequence[str],
) -> str:
    """Rewrite a labeled binary dilemma into stable internal action order.

    The user's presentation order remains available separately.  This text is
    the authoritative semantic input consumed by agents and graph compilers, so
    source labels cannot conflict with the stable internal A0/A1 assignment.
    Shared prefix/suffix facts are retained and their numeric label references
    are remapped transactionally.
    """
    canonical = list(canonical_actions)
    if len(canonical) != 2 or set(source_action_legend) != {"A0", "A1"}:
        return str(scenario)

    resolved = _source_to_canonical_bindings(source_action_legend, canonical)
    if resolved is None:
        return str(scenario)
    source_to_canonical, _ = resolved

    text = " ".join(normalize_action_labels(str(scenario)).split())

    def remap_references(value: str) -> str:
        # Placeholders make swaps simultaneous: A0->A1 cannot then be changed
        # again by the A1->A0 replacement.
        placeholders = {"A0": "__SOURCE_ZERO__", "A1": "__SOURCE_ONE__"}
        remapped = value
        for source_id, letter in (("A0", "A"), ("A1", "B")):
            remapped = re.sub(
                rf"\b(?:action|option)\s+{letter}\b",
                placeholders[source_id],
                remapped,
                flags=re.IGNORECASE,
            )
        for source_id, placeholder in placeholders.items():
            remapped = re.sub(
                rf"(?<![A-Za-z0-9]){source_id}(?![A-Za-z0-9])",
                placeholder,
                remapped,
                flags=re.IGNORECASE,
            )
        for source_id, placeholder in placeholders.items():
            remapped = remapped.replace(placeholder, source_to_canonical[source_id])
        return remapped

    # Locate the two explicit source labels. Alphabetic A/B labels are treated
    # as presentation aliases for source A0/A1; numeric labels are already
    # normalized above.
    alpha_a = re.search(r"\b(?:action|option)\s+A\s*:", text, re.IGNORECASE)
    alpha_b = re.search(r"\b(?:action|option)\s+B\s*:", text, re.IGNORECASE)
    if alpha_a and alpha_b:
        label_matches = {"A0": alpha_a, "A1": alpha_b}
    else:
        numeric_matches = {
            source_id: re.search(
                rf"\b(?:(?:action|option)\s+)?{source_id}\b\s*[:.)\]-]?",
                text,
                re.IGNORECASE,
            )
            for source_id in ("A0", "A1")
        }
        if not all(numeric_matches.values()):
            # We can still make label references directionally correct, but do
            # not pretend an unlabeled prose problem has been losslessly split.
            return remap_references(text)
        label_matches = numeric_matches

    spans: list[tuple[int, int]] = []
    folded = text.casefold()
    for source_id, label_match in label_matches.items():
        assert label_match is not None
        action_text = source_action_legend[source_id].strip(" ,;:.?")
        start = folded.find(action_text.casefold(), label_match.end())
        if start < 0:
            return remap_references(text)
        spans.append((label_match.start(), start + len(action_text)))

    block_start = min(start for start, _ in spans)
    block_end = max(end for _, end in spans)
    prefix = remap_references(text[:block_start].strip(" ,;:.?"))
    suffix = remap_references(text[block_end:].strip(" ,;:.?"))
    parts = [part for part in (
        prefix,
        "Authoritative action mapping (presentation order is irrelevant):",
        *(f"Action A{index}: {action}" for index, action in enumerate(canonical)),
        suffix,
    ) if part]
    return "\n".join(parts)


_VISIBILITY_PATTERNS = (
    re.compile(
        r"(?P<evidence>[^.;!?]{0,140}\b(?:unmonitored|unlogged|unrecorded|"
        r"untracked|off[- ]telemetry|not\s+(?:be\s+)?(?:monitored|logged|recorded|"
        r"tracked)|(?:will|would|do|does)\s+not\s+register)\b[^.;!?]{0,100})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<evidence>[^.;!?]{0,140}\b(?:lack|lacks|lacking|without)\b[^.;!?]{0,30}"
        r"\b(?:telemetry|monitoring|records?|reporting|measurement|data)\b[^.;!?]{0,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<evidence>[^.;!?]{0,140}\b(?:excluded|omitted|missing)\s+from\b[^.;!?]{0,50}"
        r"\b(?:telemetry|monitoring|records?|registry|reporting|measurement|data)\b"
        r"[^.;!?]{0,80})",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True, slots=True)
class ObservabilityFact:
    node_id: str
    target_label: str
    telemetry_visibility: float
    evidence: str
    relation: str = "EXPLICIT_TELEMETRY_ABSENCE"
    affected_action_node_id: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionObstacleFact:
    node_id: str
    affected_action_node_id: str
    evidence: str
    relation: str = "IMPAIRS_EXECUTION"


@dataclass(frozen=True, slots=True)
class ActionBurdenFact:
    affected_action_node_id: str
    evidence: str


_BURDEN_ROLE_PROJECTIONS = (
    (
        re.compile(
            r"\b(?:worker|workers|staff|team|employee|employees|crew|personnel|floor|"
            r"fulfillment|warehouse|shift)\b",
            re.I,
        ),
        "workers",
    ),
    (
        re.compile(
            r"\b(?:patient|patients|medical|medicine|medication|drug|drugs|"
            r"treatment|therapy|dose|doses|prescription|pharma|pharmaceutical)\b",
            re.I,
        ),
        "patients",
    ),
    (
        re.compile(r"\b(?:customer|customers|client|clients|buyer|orders?)\b", re.I),
        "customers",
    ),
    (
        re.compile(r"\b(?:patient|patients|resident|residents|user|users|citizen|citizens)\b", re.I),
        "affected people",
    ),
    (
        re.compile(
            r"\b(?:career|job|jobs|license|licence|standing|reputation|"
            r"blacklist\w*|fire\w*|firing|dismiss\w*|retaliat\w*|lawsuit\w*|"
            r"legal|assets?|savings?|income|livelihood)\b",
            re.I,
        ),
        "professional standing",
    ),
    (
        re.compile(r"\b(?:family|families|spouse|children|child|dependents?)\b", re.I),
        "family",
    ),
    (
        re.compile(r"\b(?:regulator\w*|authority|authorities|agency|board|oversight)\b", re.I),
        "regulators",
    ),
)


def _burden_target_label(evidence: str) -> str:
    cleaned = " ".join(str(evidence).split()).strip(" ,;:.")
    lowered = cleaned.casefold()
    labels: list[str] = []
    for pattern, label in _BURDEN_ROLE_PROJECTIONS:
        if pattern.search(lowered):
            labels.append(label)
    if labels:
        # Keep a compact multi-token label so downstream subject matching can
        # bind to the salient constituency without inventing a finer-grained
        # role than the evidence supports.
        return " ".join(dict.fromkeys(labels))
    return cleaned[:120] or "affected subject"


_BURDEN_MARKER = re.compile(
    r"\b(?:kill(?:s|ing|ed)?|death(?:s)?|die|dies|fatalit(?:y|ies)|casualt(?:y|ies)|"
    r"harm(?:s|ed|ful)?|injur(?:y|ies)|suffer(?:s|ing)?|depriv(?:e|es|ed|ation)|"
    r"displac(?:e|es|ed|ement)|coerc(?:e|ion|ive)|sacrific(?:e|es|ed)|"
    r"bottleneck(?:s|ed|ing)?|backlog(?:s|ged|ging)?|delay(?:s|ed|ing)?|"
    r"deadline(?:s)?|miss(?:es|ed|ing)?|exhaust(?:s|ed|ing|ion)?|"
    r"overload(?:s|ed|ing)?|throughput|complianc(?:e|es)?|fulfillment|"
    r"well[- ]being|collapse(?:s|d|ing)?|rush|risk(?:s|ed|ing)?|"
    r"danger(?:s|ous)?|threat(?:s|en(?:s|ed|ing)?)?|retaliat\w*|blacklist\w*|"
    r"firing|fire(?:s|d|ing)?|dismiss\w*|terminate\w*|lawsuit\w*|"
    r"career|reputation|license|livelihood|safety|medical|medication|drug|"
    r"patient(?:s)?|public\s+safety)\b",
    re.IGNORECASE,
)


def _shared_burden_clause(clause: str) -> bool:
    """Detect a scenario-global burden that should be available to every action.

    This is intentionally conservative: it is used when the scenario states a
    central hazard or protected population but no single option is explicitly
    named in the clause.
    """
    lowered = clause.casefold()
    has_hazard = bool(re.search(
        r"\b(?:risk(?:s|ed|ing)?|danger(?:s|ous)?|threat(?:s|en(?:s|ed|ing)?)?|"
        r"harm(?:s|ed|ful)?|death(?:s)?|fatalit(?:y|ies)|collapse(?:s|d|ing)?|"
        r"retaliat\w*|blacklist\w*|firing|fire(?:s|d|ing)?|lawsuit\w*|"
        r"medical|medication|drug|cardiac|safety|public\s+safety)\b",
        lowered,
        re.IGNORECASE,
    ))
    has_medical_subject = bool(re.search(
        r"\b(?:patient(?:s)?|medical|medication|drug|drugs|cardiac|health|"
        r"clinical|pharma|pharmaceutical|hospital|clinic)\b",
        lowered,
        re.IGNORECASE,
    ))
    has_protected_subject = bool(re.search(
        r"\b(?:patient(?:s)?|resident(?:s)?|worker(?:s)?|family|families|"
        r"customer(?:s)?|client(?:s)?|user(?:s)?|citizen(?:s)?|public|community)\b",
        lowered,
        re.IGNORECASE,
    ))
    return has_hazard and (has_protected_subject or has_medical_subject)


def compile_action_burdens(scenario: str, actions: Sequence[str]) -> list[ActionBurdenFact]:
    """Identify explicit burdens assigned by each option without comparing them."""
    normalized_scenario = normalize_action_labels(scenario)
    clauses = [
        " ".join(part.split()).strip()
        for part in re.split(r"[.;!?]", normalized_scenario)
    ]
    facts: list[ActionBurdenFact] = []
    for clause in clauses:
        if not clause or not _BURDEN_MARKER.search(clause):
            continue
        clause_words = set(re.findall(r"[a-z0-9]+", clause.casefold()))
        scored_actions: list[tuple[int, int]] = []
        for index, action in enumerate(actions):
            explicit_id = bool(re.search(rf"\bA{index}\b", clause, re.IGNORECASE))
            action_words = {
                word for word in re.findall(r"[a-z0-9]+", action.casefold())
                if len(word) >= 4 and word not in {"action", "option", "instead"}
            }
            if explicit_id:
                facts.append(ActionBurdenFact(action_node_id(index), clause))
                continue
            overlap = len(action_words & clause_words)
            if overlap > 0:
                scored_actions.append((index, overlap))
        if scored_actions:
            best_score = max(score for _, score in scored_actions)
            for index, score in scored_actions:
                if score == best_score:
                    facts.append(ActionBurdenFact(action_node_id(index), clause))
        elif _shared_burden_clause(clause):
            for index, _action in enumerate(actions):
                facts.append(ActionBurdenFact(action_node_id(index), clause))
    return facts


_EXECUTION_MARKER = re.compile(
    r"\b(?:cannot|can't|unable|unavailable|inaccessible|depends? on|"
    r"only if|before|deadline|limited time|access|control|authority|capacity|"
    r"resource|implement|execute|actuat|deploy|reach|refus(?:e|es|ed|al)|"
    r"within\s+\d+\s+(?:seconds?|minutes?|hours?|days?)|"
    r"requires?\b.{0,50}\b(?:access|control|authority|capacity|resource|"
    r"permission|consent|safeguard|equipment|personnel))\w*\b",
    re.IGNORECASE,
)


def compile_execution_obstacles(
    scenario: str, actions: Sequence[str]
) -> list[ExecutionObstacleFact]:
    """Compile only clauses that connect an execution constraint to an action.

    Generic danger words (risk, failure, emergency) are intentionally insufficient:
    they describe stakes, not necessarily an obstacle to carrying out an option.
    """
    normalized_scenario = normalize_action_labels(scenario)
    clauses = [
        " ".join(part.split()).strip()
        for part in re.split(r"[.;!?]", normalized_scenario)
    ]
    facts: list[ExecutionObstacleFact] = []
    for clause in clauses:
        if not clause or not _EXECUTION_MARKER.search(clause):
            continue
        clause_words = set(re.findall(r"[a-z0-9]+", clause.casefold()))
        for index, action in enumerate(actions):
            node_id = action_node_id(index)
            explicit_id = bool(re.search(rf"\bA{index}\b", clause, re.IGNORECASE))
            action_words = {
                word for word in re.findall(r"[a-z0-9]+", action.casefold())
                if len(word) >= 4 and word not in {"action", "option", "instead"}
            }
            lexical_link = bool(action_words and action_words & clause_words)
            if explicit_id or lexical_link:
                facts.append(ExecutionObstacleFact(
                    f"EXECUTION_OBSTACLE_{len(facts)}", node_id, clause
                ))
    return facts


def action_node_id(index: int) -> str:
    return f"A{index}"


def action_legend(actions: Sequence[str]) -> dict[str, str]:
    return {action_node_id(index): action for index, action in enumerate(actions)}


def _target_label(evidence: str) -> str:
    """Keep a readable target phrase without pretending to solve coreference."""
    cleaned = " ".join(evidence.split()).strip(" ,:-")
    # The typed visibility value is the invariant; this label is explanatory.
    return cleaned[:120] or "low-observability target"


def compile_observability_facts(scenario: str) -> list[ObservabilityFact]:
    """Map explicit absence-of-observation language to visibility zero.

    This does not infer casualty counts, moral priority, or a cause such as social
    neglect. It records only what the scenario explicitly says: the target is absent
    from the relevant telemetry/recording channel.
    """
    scenario = normalize_action_labels(scenario)
    matches: list[tuple[int, str]] = []
    for pattern in _VISIBILITY_PATTERNS:
        for match in pattern.finditer(scenario):
            evidence = " ".join(match.group("evidence").split()).strip(" ,:-")
            if evidence:
                matches.append((match.start(), evidence))
    facts: list[ObservabilityFact] = []
    seen: set[str] = set()
    for _, evidence in sorted(matches):
        key = evidence.casefold()
        if key in seen:
            continue
        seen.add(key)
        action_match = re.search(r"\bA(?P<index>\d+)\b", evidence, re.IGNORECASE)
        affected_action = (
            f"A{action_match.group('index')}" if action_match is not None else ""
        )
        facts.append(ObservabilityFact(
            node_id=f"TARGET_VISIBILITY_{len(facts)}",
            target_label=_target_label(evidence),
            telemetry_visibility=0.0,
            evidence=evidence,
            affected_action_node_id=affected_action,
        ))
    return facts


def attach_grounded_world_effects(
    graph: SemanticGraph,
    action_id: str,
    effects: Sequence[Any],
) -> None:
    """Surround an ACTION node with cascade/foregone affected-party relations.

    Direct intervention roles stay on the canonical action. These nodes are the
    world-state layer that lets structured agents see a future population as
    affected without treating it as Patient A.
    """
    if action_id not in graph.nodes:
        return
    for index, effect in enumerate(effects):
        party = str(getattr(effect, "party", "") or "").strip()
        if not party:
            continue
        relation = str(getattr(effect, "relation", "") or "").upper()
        outcome = str(getattr(effect, "outcome", "") or relation).strip()
        modality = str(getattr(effect, "modality", "") or "")
        condition = str(getattr(effect, "condition", "") or "")
        party_kind = str(getattr(effect, "party_kind", "") or "")
        dimension = str(getattr(effect, "dimension", "") or "")
        clause_ids = tuple(getattr(effect, "clause_ids", ()) or ())
        provenance = tuple(getattr(effect, "provenance", ()) or ())
        if not provenance:
            provenance = ("grounded_world_effect",)
        polarity = (
            "BENEFICIAL" if relation == RELATION_DOWNSTREAM_BENEFIT
            else "FOREGONE" if relation == RELATION_FOREGONE_BENEFIT
            else "UNCERTAIN"
        )
        consequence_id = f"{action_id}:GROUNDED_EFFECT:{index}"
        graph.add_node(SemanticNode(
            consequence_id, "CONSEQUENCE", outcome, provenance,
            {
                "polarity": polarity,
                "relation": relation,
                "modality": modality,
                "condition": condition,
                "party_kind": party_kind,
                "dimension": dimension,
                "targets": [party],
                "affected_subjects": [party],
                "scenario_grounded": bool(clause_ids),
                "source_clause_id": clause_ids[0] if clause_ids else "",
                "effect_layer": "GROUNDED_WORLD",
            },
        ))
        graph.add_edge(SemanticEdge(
            action_id, "HAS_CONSEQUENCE", consequence_id, provenance=provenance,
        ))
        target_id = f"{consequence_id}:AFFECTED_SUBJECT:0"
        graph.add_node(SemanticNode(
            target_id, "TARGET", party, provenance,
            {
                "semantic_role": "AFFECTED_SUBJECT",
                "party_kind": party_kind,
                "source_clause_id": clause_ids[0] if clause_ids else "",
            },
        ))
        graph.add_edge(SemanticEdge(
            consequence_id, "AFFECTS", target_id, provenance=provenance,
        ))


def _graph_has_edge(
    graph: SemanticGraph, source: str, relation: str, target: str,
) -> bool:
    return any(
        edge.source == source and edge.relation == relation and edge.target == target
        for edge in graph.edges
    )


def _add_unique_edge(graph: SemanticGraph, edge: SemanticEdge) -> None:
    if _graph_has_edge(graph, edge.source, edge.relation, edge.target):
        return
    graph.add_edge(edge)


def attach_typed_world_model(graph: SemanticGraph, model: Any) -> None:
    """Compile an admitted typed world model without re-reading source prose.

    Re-applying the same model, or applying an approved extension of it, is
    idempotent: existing consequence nodes are reused and duplicate edges are
    not added.
    """
    party_by_id = {party.party_id: party for party in model.parties}
    admitted = set(model.admission.admitted_effect_ids)
    filter_admission = bool(admitted) or model.admission.status in {
        "USER_ACCEPTED_WITH_QUARANTINE",
        "ABANDONED_CONTRADICTORY_WORLD_STATE",
    }
    for action in model.actions:
        if action.action_id not in graph.nodes:
            continue
        provenance = tuple(
            f"scenario_clause:{ref.clause_id}" for ref in action.provenance
        ) or ("typed_world_model",)
        intervention_id = f"{action.action_id}:INTERVENTION"
        graph.add_node(SemanticNode(
            intervention_id, "INTERVENTION", action.intervention, provenance,
            {"world_state_typed": True},
        ))
        _add_unique_edge(graph, SemanticEdge(
            action.action_id, "HAS_INTERVENTION", intervention_id,
            provenance=provenance,
        ))
        actor = party_by_id.get(action.actor_party_id)
        if actor is not None:
            actor_id = f"PARTY:{actor.party_id}"
            if actor_id not in graph.nodes:
                graph.add_node(SemanticNode(
                    actor_id, "ACTOR", actor.label, provenance,
                    {"party_id": actor.party_id, "party_kind": actor.kind,
                     "quantities": list(actor.quantities), "world_state_typed": True},
                ))
            _add_unique_edge(graph, SemanticEdge(
                action.action_id, "HAS_ACTOR", actor_id, provenance=provenance,
            ))
        for recipient_id in action.recipient_party_ids:
            recipient = party_by_id.get(recipient_id)
            if recipient is None:
                continue
            target_id = f"PARTY:{recipient.party_id}"
            if target_id not in graph.nodes:
                graph.add_node(SemanticNode(
                    target_id, "TARGET", recipient.label, provenance,
                    {"party_id": recipient.party_id, "party_kind": recipient.kind,
                     "quantities": list(recipient.quantities), "world_state_typed": True},
                ))
            _add_unique_edge(graph, SemanticEdge(
                action.action_id, "TARGETS", target_id, provenance=provenance,
            ))
    for condition in model.conditions:
        provenance = tuple(
            f"scenario_clause:{ref.clause_id}" for ref in condition.provenance
        ) or ("typed_world_model",)
        graph.add_node(SemanticNode(
            condition.condition_id, "CONDITION", condition.description, provenance,
            {
                "value_status": condition.value_status,
                "decision_relevance": condition.decision_relevance,
                "source_clause_ids": [ref.clause_id for ref in condition.provenance],
                "world_state_typed": True,
            },
        ))
    effect_node_ids: dict[str, str] = {}
    for effect in model.effects:
        if filter_admission and effect.effect_id not in admitted:
            continue
        if effect.action_id not in graph.nodes:
            continue
        party = party_by_id.get(effect.party_id)
        if party is None:
            continue
        provenance = tuple(
            f"scenario_clause:{ref.clause_id}" for ref in effect.provenance
        ) or ("typed_world_model",)
        consequence_id = f"{effect.action_id}:WORLD_EFFECT:{effect.effect_id}"
        effect_node_ids[effect.effect_id] = consequence_id
        graph.add_node(SemanticNode(
            consequence_id, "CONSEQUENCE", effect.outcome, provenance,
            {
                "polarity": effect.polarity,
                "relation": effect.relation,
                "directness": effect.directness,
                "effect_kind": effect.effect_kind,
                "modality": effect.modality,
                "condition_ids": list(effect.condition_ids),
                "quantities": list(effect.quantities),
                "likelihood_qualifiers": list(effect.likelihood_qualifiers),
                "scope_qualifiers": list(effect.scope_qualifiers),
                "temporal_qualifiers": list(effect.temporal_qualifiers),
                "targets": [party.label],
                "affected_subjects": [party.label],
                "party_id": party.party_id,
                "party_kind": party.kind,
                "party_quantities": list(party.quantities),
                "scenario_grounded": True,
                "source_clause_id": (
                    effect.provenance[0].clause_id if effect.provenance else ""
                ),
                "source_clause_ids": [ref.clause_id for ref in effect.provenance],
                "effect_layer": "TYPED_WORLD",
                "world_effect_id": effect.effect_id,
            },
        ))
        _add_unique_edge(graph, SemanticEdge(
            effect.action_id, "HAS_CONSEQUENCE", consequence_id,
            provenance=provenance,
        ))
        target_id = f"PARTY:{party.party_id}"
        if target_id not in graph.nodes:
            graph.add_node(SemanticNode(
                target_id, "TARGET", party.label,
                tuple(f"scenario_clause:{ref.clause_id}" for ref in party.provenance)
                or provenance,
                {
                    "semantic_role": "AFFECTED_SUBJECT",
                    "party_id": party.party_id,
                    "party_kind": party.kind,
                    "quantities": list(party.quantities),
                    "world_state_typed": True,
                },
            ))
        _add_unique_edge(graph, SemanticEdge(
            consequence_id, "AFFECTS", target_id, provenance=provenance,
        ))
        for condition_id in effect.condition_ids:
            if condition_id in graph.nodes:
                _add_unique_edge(graph, SemanticEdge(
                    consequence_id, "CONDITIONAL_ON", condition_id,
                    provenance=provenance,
                ))
        for ref in effect.provenance:
            evidence_id = f"ACTION_SOURCE:{effect.action_id}:{ref.clause_id}"
            if evidence_id in graph.nodes:
                _add_unique_edge(graph, SemanticEdge(
                    consequence_id, "SUPPORTED_BY", evidence_id,
                    provenance=provenance,
                ))
    for index, link in enumerate(model.causal_links):
        source = effect_node_ids.get(link.source_id, link.source_id)
        target = effect_node_ids.get(link.target_id, link.target_id)
        if source not in graph.nodes or target not in graph.nodes:
            continue
        provenance = tuple(
            f"scenario_clause:{ref.clause_id}" for ref in link.provenance
        ) or ("typed_world_model",)
        relation = {
            "MAY_CAUSE": "CAUSES", "MIGHT_CAUSE": "CAUSES",
            "MAY_ENABLE": "ENABLES", "ACCELERATES": "INCREASES",
            "FOREGOES": "DISABLES",
        }.get(link.relation, link.relation)
        from .semantic_graph import EDGE_RELATIONS
        if relation not in EDGE_RELATIONS:
            relation = "CAUSES"
        _add_unique_edge(graph, SemanticEdge(
            source, relation, target,
            justification=(
                f"typed_relation={link.relation}; modality={link.modality}"
            ), provenance=provenance,
        ))
    for link in model.counterfactual_links:
        source = effect_node_ids.get(link.source_effect_id)
        alternative = effect_node_ids.get(link.alternative_effect_id)
        if source not in graph.nodes or alternative not in graph.nodes:
            continue
        provenance = tuple(
            f"scenario_clause:{ref.clause_id}" for ref in link.provenance
        ) or ("typed_world_model",)
        relation = {
            "FOREGOES_ALTERNATIVE_EFFECT": "COUNTERFACTUALLY_FOREGOES",
            "PRECLUDES_ALTERNATIVE_EFFECT": "COUNTERFACTUALLY_PRECLUDES",
            "REPLACES_ALTERNATIVE_EFFECT": "COUNTERFACTUALLY_REPLACES",
        }.get(link.relation, "COUNTERFACTUALLY_FOREGOES")
        _add_unique_edge(graph, SemanticEdge(
            source, relation, alternative,
            justification=f"modality={link.modality}", provenance=provenance,
        ))


def compile_scenario_graph(
    scenario: str,
    actions: Sequence[str],
    action_source_groundings: dict[str, dict[str, object]] | None = None,
    world_model: dict[str, Any] | None = None,
) -> SemanticGraph:
    graph = SemanticGraph()
    incomplete_actions = [
        f"A{index}: {canonical_action_text(action)[:120]}"
        for index, action in enumerate(actions)
        if not action_clause_looks_complete(action)
    ]
    if incomplete_actions and not world_model:
        raise ValueError(
            "scenario graph admission rejected incomplete or truncated action clause(s): "
            + "; ".join(incomplete_actions)
        )
    resolved_keys = resolved_semantic_action_keys(actions)
    scenario_clauses = segment_scenario_clauses(scenario)
    for (node_id, action), resolved_key in zip(
        action_legend(actions).items(), resolved_keys,
    ):
        identity = compile_action_identity(action)
        graph.add_node(SemanticNode(
            node_id, "ACTION", action, ("scenario_action_set",),
            {
                "canonical_action_id": node_id,
                "semantic_action_key": resolved_key,
                "canonical_action_text": canonical_action_text(action),
                "action_identity_basis": identity.basis,
                "action_identity_signature": identity.signature(),
            },
        ))
        if not world_model:
            add_action_identity_subgraph(graph, node_id, identity)
        if not world_model:
            attach_grounded_world_effects(
                graph,
                node_id,
                extract_grounded_effects(action, clauses=scenario_clauses),
            )
    _attach_grounded_action_sources(
        graph, action_source_groundings or {},
        attach_consequences=not bool(world_model),
    )
    if world_model:
        from .world_state import world_model_from_dict
        typed = world_model_from_dict(world_model)
        if typed is not None:
            attach_typed_world_model(graph, typed)
            expected_effect_ids = {
                effect.effect_id
                for action in typed.actions
                for effect in typed.effects_for(action.action_id)
            }
            compiled_effect_ids = {
                str(node.attributes.get("world_effect_id"))
                for node in graph.nodes.values()
                if node.kind == "CONSEQUENCE"
                and node.attributes.get("effect_layer") == "TYPED_WORLD"
            }
            if compiled_effect_ids != expected_effect_ids:
                raise ValueError(
                    "typed world projection mismatch: semantic graph effects differ "
                    "from the admitted world model"
                )
    for fact in compile_observability_facts(scenario):
        graph.add_node(SemanticNode(
            fact.node_id, "TARGET", fact.target_label, (fact.evidence,),
            {
                "telemetry_visibility": fact.telemetry_visibility,
                "visibility_relation": fact.relation,
            },
        ))
        estimate_id = f"HARM_ESTIMATE_{fact.node_id}"
        estimate_label = (
            f"estimated harm of {fact.affected_action_node_id}"
            if fact.affected_action_node_id else "estimated harm to low-observability target"
        )
        graph.add_node(SemanticNode(
            estimate_id, "METRIC", estimate_label, (fact.evidence,),
            {
                "bias_direction": "DOWNWARD",
                "magnitude_status": "UNKNOWN",
                "epistemic_status": "STATED_MISSINGNESS",
            },
        ))
        graph.add_edge(SemanticEdge(
            fact.node_id, "BIASES_ESTIMATE", estimate_id,
            justification="telemetry_visibility=0.0",
            provenance=(fact.evidence,),
        ))
        if fact.affected_action_node_id in graph.nodes:
            graph.add_edge(SemanticEdge(
                fact.affected_action_node_id, "HAS_METRIC", estimate_id,
                provenance=(fact.evidence,),
            ))
    for fact in compile_execution_obstacles(scenario, actions):
        graph.add_node(SemanticNode(
            fact.node_id, "CONDITION", fact.evidence, (fact.evidence,),
            {"condition_type": "EXECUTION_OBSTACLE"},
        ))
        graph.add_edge(SemanticEdge(
            fact.node_id, fact.relation, fact.affected_action_node_id,
            provenance=(fact.evidence,),
        ))
    for fact in compile_action_burdens(scenario, actions):
        burden_id = f"BURDEN_{fact.affected_action_node_id}_{len(graph.nodes)}"
        burden_label = _burden_target_label(fact.evidence)
        graph.add_node(SemanticNode(
            burden_id, "TARGET", burden_label, (fact.evidence,),
            {
                "burden_evidence": fact.evidence,
                "burden_projection": burden_label,
            },
        ))
        graph.add_edge(SemanticEdge(
            fact.affected_action_node_id, "TARGETS", burden_id,
            provenance=(fact.evidence,),
        ))
    return graph
