from __future__ import annotations

import json
from typing import Any, Sequence

from .models import VisibilityAssessment
from .scenario_semantics import compile_observability_facts
from .structured_io import call_json_llm, extract_json, strict_number


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
        if low and endogenous:
            if len(group.split()) < 1 or len(mechanism.split()) < 3:
                raise ValueError("endogenous visibility finding needs a group and mechanism")
            if len(evidence.split()) < 3:
                raise ValueError("visibility mechanism lacks scenario-grounded evidence")
            # A compiled zero-visibility TargetNode is itself the provenance for
            # the observability claim. Generated prose need not reproduce its
            # source clause or infer an additional social narrative.
            typed_grounding = observability_facts[0] if observability_facts else None
            if evidence.casefold() not in scenario_normalized and typed_grounding is None:
                supported, reason = _verify_visibility_grounding(
                    llm, scenario, group, mechanism, evidence, max_tokens=max_tokens
                )
                if not supported:
                    raise ValueError(
                        "visibility mechanism failed semantic grounding"
                        + (f": {reason}" if reason else "")
                    )
            if not any(value < 0.999 for value in multipliers.values()):
                raise ValueError("activated visibility audit identifies no confidence penalty")
            return VisibilityAssessment(
                low, endogenous, group, mechanism, evidence, multipliers,
                activated=True, typed_facts=typed_facts,
            )
        if any(value < 0.999 for value in multipliers.values()):
            raise ValueError("non-endogenous uncertainty cannot receive a visibility penalty")
        return VisibilityAssessment(
            low, endogenous, group, mechanism, evidence,
            {action: 1.0 for action in actions}, activated=False,
            typed_facts=typed_facts,
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
