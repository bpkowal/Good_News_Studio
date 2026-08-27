from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .models import VisibilityAssessment
from .scenario_semantics import compile_observability_facts
from .structured_io import call_json_llm, extract_json, strict_number


_VISIBILITY_MECHANISM_ANCHORS = (
    "telemetry", "monitor", "monitored", "monitoring", "logged", "record",
    "records", "report", "reporting", "registry", "register", "registered",
    "registration", "data", "measurement", "measurements", "signal", "signals",
    "tracked", "tracking", "observe", "observed", "observation", "observations",
    "outcome",
)

_VISIBILITY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "with", "without", "in",
    "on", "at", "by", "from", "this", "that", "these", "those", "their", "its",
    "is", "are", "was", "were", "be", "been", "being", "less", "more", "most",
    "some", "any", "can", "could", "should", "would", "may", "might", "will",
    "into", "through", "under", "over", "because", "throughout", "per", "each",
}

_STRUCTURAL_VISIBILITY_PATTERNS = (
    re.compile(
        r"\b(?:dominat\w*|control\w*|capture\w*|retaliat\w*|backlash|"
        r"punish\w*|silenc\w*|chill\w*|coerc\w*|blackmail\w*|"
        r"power\s+and\s+politic\w*|economy\s+and\s+politic\w*|"
        r"legal\s+campaign|seize\s+(?:your|their|his|her|the)?\s*assets?|"
        r"bankrupt\w*|destroy\s+your\s+career|career\s+destroy\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:impoverish\w*|vulnerab\w*|rural\s+community|community|residents?|"
        r"town|settlement|workers?|famil\w*)\b",
        re.IGNORECASE,
    ),
)


def _mechanism_has_visibility_anchor(mechanism: str) -> bool:
    normalized = mechanism.casefold()
    return any(anchor in normalized for anchor in _VISIBILITY_MECHANISM_ANCHORS)


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) >= 3 and token not in _VISIBILITY_STOPWORDS
    }


def _has_structural_visibility_cue(scenario: str) -> bool:
    normalized = " ".join(str(scenario).split())
    return any(pattern.search(normalized) for pattern in _STRUCTURAL_VISIBILITY_PATTERNS)


def assess_visibility(
    llm: Any,
    scenario: str,
    actions: Sequence[str],
    max_tokens: int = 160,
) -> VisibilityAssessment:
    """Audit endogenous missingness without adding another normative vote."""
    action_ids = [f"A{index}" for index in range(len(actions))]
    legend = dict(zip(action_ids, actions))
    observability_facts = compile_observability_facts(scenario)
    typed_facts = [
        {
            "node_id": fact.node_id,
            "target_label": fact.target_label,
            "telemetry_visibility": fact.telemetry_visibility,
            "relation": fact.relation,
            "evidence": fact.evidence,
            "affected_action_node_id": fact.affected_action_node_id,
        }
        for fact in observability_facts
    ]
    schema = {
        "type": "object",
        "properties": {
            "lo": {"type": "boolean"},
            "en": {"type": "boolean"},
            "g": {"type": "string", "maxLength": 100},
            "m": {"type": "string", "maxLength": 180},
            "q": {"type": "string", "maxLength": 180},
            "p": {
                "type": "object",
                "properties": {
                    action_id: {"type": "number", "minimum": 0.65, "maximum": 1.0}
                    for action_id in action_ids
                },
                "required": action_ids,
                "additionalProperties": False,
            },
        },
        "required": ["lo", "en", "g", "m", "q", "p"],
        "additionalProperties": False,
    }
    prompt = f"""You are a non-voting visibility auditor in an ethical workspace.
Scenario: {' '.join(scenario.split())[:900]}
Actions: {json.dumps(legend)}
Typed observability facts: {json.dumps(typed_facts, sort_keys=True)}

Determine whether relevant people or harms have low observability, and whether that
missingness is endogenous to structural exclusion: unequal infrastructure, access,
measurement, representation, institutional attention, or ability to signal. Ordinary
uncertainty, random sensor failure, and merely unspecified facts are not endogenous.

Return JSON only. lo=low observability; en=endogenous structural cause; g=affected
group; m=the causal visibility mechanism; q=a concise scenario-grounded fact supporting
that mechanism (an exact quote when practical, otherwise a faithful paraphrase). p gives
an epistemic-confidence multiplier for every action. Use 1.0 unless that action's
apparent advantage relies on treating missing observations as evidence of absent people,
absent harm, or lower need. Use 0.65-0.95 only to weaken that unsupported evidential
advantage. Do not express moral preference, redistribute votes, infer hidden casualties,
or penalize an action merely because its evidence is quantitative. If either lo or en
is false, every multiplier must be 1.0.
If you cannot ground the mechanism in the typed observability facts or the scenario
text, set the mechanism aside as hypothetical rather than inventing a sociological
story. A grounded visibility audit should rest on the typed fact, not on a fresh
explanation that the scenario never states.
Explicit institutional dominance, retaliation, or coercive economic/political power
over a vulnerable community may support an EXTERNAL_GENERALIZATION even when the
scenario does not spell out the entire signaling failure.
"""
    try:
        output = call_json_llm(
            llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = extract_json(raw)
        # Explicit off-telemetry language is compiled before interpretation. The
        # model may explain its relevance, but cannot erase the typed fact merely
        # because the prompt did not also narrate its societal implications.
        low = bool(data.get("lo")) or bool(observability_facts)
        endogenous = bool(data.get("en")) or bool(observability_facts)
        group = " ".join(str(data.get("g", "")).split())
        mechanism = " ".join(str(data.get("m", "")).split())
        evidence = " ".join(str(data.get("q", "")).split())
        raw_multipliers = data.get("p")
        if not isinstance(raw_multipliers, dict) or set(raw_multipliers) != set(action_ids):
            raise ValueError("visibility multipliers must cover every action")
        multipliers = {
            action: strict_number(raw_multipliers[action_id], f"p.{action_id}")
            for action_id, action in zip(action_ids, actions)
        }
        if any(value < 0.65 for value in multipliers.values()):
            raise ValueError("visibility confidence multiplier cannot be below 0.65")
        scenario_normalized = " ".join(scenario.split()).casefold()
        typed_evidence = {
            str(fact.get("evidence", "")).casefold()
            for fact in typed_facts
            if str(fact.get("evidence", "")).strip()
        }
        mechanism_tokens = _meaningful_tokens(mechanism)
        evidence_tokens = _meaningful_tokens(evidence)
        typed_mechanism_support = any(
            mechanism_tokens & (
                _meaningful_tokens(str(fact.get("evidence", "")))
                | _meaningful_tokens(str(fact.get("target_label", "")))
            )
            for fact in typed_facts
        ) or bool(mechanism_tokens & evidence_tokens)
        structural_visibility_cue = _has_structural_visibility_cue(scenario)
        visibility_support_words = {
            "signal", "signals", "report", "reports", "reporting", "attention",
            "visibility", "monitor", "monitored", "monitoring", "institutional",
            "institution", "exclude", "excluded", "suppress", "suppresses",
            "silence", "silenced", "silencing",
        }
        mechanism_provenance = (
            "SCENARIO_GROUNDED"
            if _mechanism_has_visibility_anchor(mechanism)
            and (
                mechanism.casefold() in scenario_normalized
                or mechanism.casefold() in typed_evidence
                or typed_mechanism_support
            )
            else "EXTERNAL_GENERALIZATION"
            if _mechanism_has_visibility_anchor(mechanism)
            and structural_visibility_cue
            and bool(mechanism_tokens & visibility_support_words)
            else "HYPOTHETICAL"
        )
        if low and endogenous:
            if len(group.split()) < 1 or len(mechanism.split()) < 3:
                raise ValueError("endogenous visibility finding needs a group and mechanism")
            if len(evidence.split()) < 3:
                raise ValueError("visibility mechanism lacks scenario-grounded evidence")
            # A compiled zero-visibility TargetNode is itself the provenance for
            # the observability claim. Generated prose need not reproduce its
            # source clause or infer an additional social narrative.
            typed_grounding = observability_facts[0] if observability_facts else None
            if (
                mechanism_provenance == "SCENARIO_GROUNDED"
                and evidence.casefold() not in scenario_normalized
                and typed_grounding is None
            ):
                supported, reason = _verify_visibility_grounding(
                    llm, scenario, group, mechanism, evidence, max_tokens=max_tokens
                )
                if not supported:
                    raise ValueError(
                        "visibility mechanism failed semantic grounding"
                        + (f": {reason}" if reason else "")
                    )
            if mechanism_provenance not in {"SCENARIO_GROUNDED", "EXTERNAL_GENERALIZATION"}:
                return VisibilityAssessment(
                    low, endogenous, group, mechanism, evidence,
                    {action: 1.0 for action in actions}, activated=False,
                    typed_facts=typed_facts,
                    mechanism_provenance=mechanism_provenance,
                )
            if not any(value < 0.999 for value in multipliers.values()):
                raise ValueError("activated visibility audit identifies no confidence penalty")
            return VisibilityAssessment(
                low, endogenous, group, mechanism, evidence, multipliers,
                activated=True, typed_facts=typed_facts,
                mechanism_provenance=mechanism_provenance,
            )
        if any(value < 0.999 for value in multipliers.values()):
            raise ValueError("non-endogenous uncertainty cannot receive a visibility penalty")
        return VisibilityAssessment(
            low, endogenous, group, mechanism, evidence,
            {action: 1.0 for action in actions}, activated=False,
            typed_facts=typed_facts,
            mechanism_provenance=mechanism_provenance,
        )
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
        return VisibilityAssessment(
            False, False, "", "", "",
            {action: 1.0 for action in actions}, activated=False,
            valid=False, error=str(error), typed_facts=typed_facts,
        )


def _verify_visibility_grounding(
    llm: Any,
    scenario: str,
    group: str,
    mechanism: str,
    evidence: str,
    *,
    max_tokens: int,
) -> tuple[bool, str]:
    schema = {
        "type": "object",
        "properties": {
            "s": {"type": "boolean"},
            "r": {"type": "string", "maxLength": 160},
        },
        "required": ["s", "r"],
        "additionalProperties": False,
    }
    prompt = f"""Verify grounding only; do not make an ethical recommendation.
Scenario: {' '.join(scenario.split())[:1000]}
Affected group: {group}
Proposed endogenous visibility mechanism: {mechanism}
Proposed supporting fact: {evidence}

Return JSON with s=true only if the scenario explicitly states or strongly entails
both the supporting fact and the claimed administrative, institutional,
infrastructural, representational, or signaling mechanism. Mere uncertainty,
unregistered status without a visibility consequence, and invented causal details
must return false. r briefly identifies the scenario fact or the unsupported leap.
"""
    try:
        output = call_json_llm(
            llm, prompt, max_tokens=max(96, max_tokens), temperature=0.0, schema=schema
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = extract_json(raw)
        return bool(data.get("s")), " ".join(str(data.get("r", "")).split())
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
        return False, f"grounding verifier unavailable: {error}"
