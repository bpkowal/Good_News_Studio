"""Deterministic compilation of explicit scenario facts into typed graph nodes.

This module deliberately recognizes *semantic classes*, not particular dilemmas.
It turns explicit observability language and the workspace action set into stable
identifiers before any LLM is asked to interpret their ethical significance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Sequence
import unicodedata

from .action_identity import (
    add_action_identity_subgraph,
    compile_action_identity,
    graph_action_key,
)
from .semantic_graph import SemanticEdge, SemanticGraph, SemanticNode


_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_ACTION_LABEL = re.compile(
    r"(?<![A-Za-z0-9])\$?A\s*(?:_\s*)?(?:\{\s*)?"
    r"(?P<index>[0-9₀₁₂₃₄₅₆₇₈₉]+)(?:\s*\})?\$?(?![A-Za-z0-9])",
)


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

    source_to_canonical: dict[str, str] = {}
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
            return str(scenario)
        source_to_canonical[source_id] = f"A{matches[0]}"
    if len(set(source_to_canonical.values())) != 2:
        return str(scenario)

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


_BURDEN_MARKER = re.compile(
    r"\b(?:kill(?:s|ing|ed)?|death(?:s)?|die|dies|fatalit(?:y|ies)|casualt(?:y|ies)|"
    r"harm(?:s|ed|ful)?|injur(?:y|ies)|suffer(?:s|ing)?|depriv(?:e|es|ed|ation)|"
    r"displac(?:e|es|ed|ement)|coerc(?:e|ion|ive)|sacrific(?:e|es|ed))\b",
    re.IGNORECASE,
)


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
        for index, action in enumerate(actions):
            explicit_id = bool(re.search(rf"\bA{index}\b", clause, re.IGNORECASE))
            action_words = {
                word for word in re.findall(r"[a-z0-9]+", action.casefold())
                if len(word) >= 4 and word not in {"action", "option", "instead"}
            }
            if explicit_id or (action_words and action_words & clause_words):
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


def compile_scenario_graph(scenario: str, actions: Sequence[str]) -> SemanticGraph:
    graph = SemanticGraph()
    resolved_keys = resolved_semantic_action_keys(actions)
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
        add_action_identity_subgraph(graph, node_id, identity)
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
    return graph
