from __future__ import annotations

import json
from typing import Any, Sequence

from .models import AutonomyAssessment
from .structured_io import call_json_llm, extract_json


def assess_autonomy_and_coercion(
    llm: Any,
    scenario: str,
    actions: Sequence[str],
    max_tokens: int = 180,
) -> AutonomyAssessment:
    """Tag coercion and test its factual justification without casting a vote."""
    action_ids = [f"A{index}" for index in range(len(actions))]
    legend = dict(zip(action_ids, actions))
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "object",
                "properties": {
                    key: {
                        "type": "string",
                        "enum": ["NONE", "COERCIVE", "RIGHTS_INTRUSION", "COVENANT_BREACH"],
                    }
                    for key in action_ids
                },
                "required": action_ids,
                "additionalProperties": False,
            },
            "threshold": {
                "type": "object",
                "properties": {key: {"type": "boolean"} for key in action_ids},
                "required": action_ids,
                "additionalProperties": False,
            },
            "evidence": {
                "type": "object",
                "properties": {key: {"type": "string", "maxLength": 180} for key in action_ids},
                "required": action_ids,
                "additionalProperties": False,
            },
            "voluntary": {"type": "string", "maxLength": 200},
        },
        "required": ["tags", "threshold", "evidence", "voluntary"],
        "additionalProperties": False,
    }
    prompt = f"""You are a non-voting Autonomy & Coercion auditor.
Scenario: {' '.join(scenario.split())[:1100]}
Actions: {json.dumps(legend)}

Tag each action. COERCIVE means intentional force, confinement, compelled treatment,
seizure, or overriding a competent refusal. RIGHTS_INTRUSION means a direct invasion
of bodily, property, speech, privacy, or association rights without the stronger force
label. COVENANT_BREACH means breaking an explicit agreement or entrusted commitment.
Ordinary costs, persuasion, criticism, refusal to assist, and harmful side effects are
not automatically coercion.

For each tagged action, threshold=true only when the scenario itself establishes all
three: imminent harm, catastrophic or comparably grave physical harm, and harm to third
parties. Do not infer hidden victims, assume an intervention works, or use consensus as
evidence. Evidence must identify the scenario-grounded facts; use NONE for untagged
actions. voluntary names the most relevant voluntary or less-coercive alternative whose
effectiveness remains unresolved, or NONE if the scenario expressly excludes one.
Return JSON only and make no ethical recommendation.
"""
    try:
        output = call_json_llm(
            llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = extract_json(raw)
        tags_raw = data.get("tags")
        threshold_raw = data.get("threshold")
        evidence_raw = data.get("evidence")
        if not all(isinstance(value, dict) for value in (tags_raw, threshold_raw, evidence_raw)):
            raise ValueError("autonomy audit mappings are required")
        if any(set(value) != set(action_ids) for value in (tags_raw, threshold_raw, evidence_raw)):
            raise ValueError("autonomy audit must cover every action")
        tags = {action: str(tags_raw[key]).upper() for key, action in zip(action_ids, actions)}
        thresholds = {action: bool(threshold_raw[key]) for key, action in zip(action_ids, actions)}
        evidence = {
            action: " ".join(str(evidence_raw[key]).split())
            for key, action in zip(action_ids, actions)
        }
        for action in actions:
            if tags[action] == "NONE" and thresholds[action]:
                raise ValueError("untagged action cannot satisfy coercion exception")
            if tags[action] != "NONE" and len(evidence[action].split()) < 3:
                raise ValueError("tagged action requires scenario-grounded evidence")
        voluntary = " ".join(str(data.get("voluntary", "NONE")).split()) or "NONE"
        return AutonomyAssessment(tags, thresholds, evidence, voluntary)
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
        return AutonomyAssessment(
            {action: "NONE" for action in actions},
            {action: False for action in actions},
            {action: "" for action in actions},
            "NONE", valid=False, error=str(error),
        )
