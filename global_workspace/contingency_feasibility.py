"""Independent non-voting verifier for contingency fallback executability."""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import ContingencyFeasibilityAssessment, FailureCondition
from .structured_io import call_json_llm, extract_json


class FallbackFeasibilityClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: Literal["A0", "A1"]
    status: Literal["AVAILABLE", "UNAVAILABLE", "UNKNOWN"]
    basis: Literal["SCENARIO_STRUCTURE", "FAILURE_SCOPE", "INSUFFICIENT"]
    reason: str = Field(min_length=8, max_length=120)


class FeasibilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    fallback_assessments: list[FallbackFeasibilityClaim] = Field(
        alias="fa", min_length=2, max_length=2
    )
    shared_failure: bool = Field(alias="sf")


def verify_contingency_feasibility(
    llm: Any,
    scenario: str,
    condition: FailureCondition,
    max_tokens: int = 96,
) -> ContingencyFeasibilityAssessment:
    """Assess physical availability without making an ethical recommendation."""
    schema = {
        "type": "object",
        "properties": {
            "fa": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "action_id": {"type": "string", "enum": ["A0", "A1"]},
                        "status": {
                            "type": "string",
                            "enum": ["AVAILABLE", "UNAVAILABLE", "UNKNOWN"],
                        },
                        "basis": {
                            "type": "string",
                            "enum": ["SCENARIO_STRUCTURE", "FAILURE_SCOPE", "INSUFFICIENT"],
                        },
                        "reason": {"type": "string", "minLength": 8, "maxLength": 120},
                    },
                    "required": ["action_id", "status", "basis", "reason"],
                    "additionalProperties": False,
                },
            },
            "sf": {"type": "boolean"},
        },
        "required": ["fa", "sf"],
        "additionalProperties": False,
    }
    legend = {f"A{index}": action for index, action in enumerate(condition.fallback_actions)}
    prompt = f"""[INST]
You are a non-voting implementation-feasibility verifier. Do not make an ethical
recommendation and do not compare moral desirability.
Scenario: {' '.join(scenario.split())[:800]}
Synthesis: {condition.synthesis_action}
Required predicate: {condition.predicate_label}
Assume that predicate is FALSE. Direct effect: {condition.failure_condition}
Original fallbacks: {json.dumps(legend)}
For A0 and A1 separately, decide only whether the actor can still physically execute
that action after the stated failure. AVAILABLE means executable, not safe or moral.
Use UNKNOWN when the scenario and failure scope do not establish executability.
basis is SCENARIO_STRUCTURE, FAILURE_SCOPE, or INSUFFICIENT. sf=true when the failure
removes a capability shared by the synthesis and either fallback. Return JSON only.
[/INST]"""
    try:
        output = call_json_llm(
            llm, prompt, max_tokens=max(72, min(max_tokens, 96)),
            temperature=0.0, schema=schema, call_kind="auxiliary",
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        response = FeasibilityResponse.model_validate(extract_json(raw))
        by_id = {item.action_id: item for item in response.fallback_assessments}
        errors = []
        if set(by_id) != {"A0", "A1"}:
            errors.append("verifier must classify A0 and A1 exactly once")
        statuses = {key: item.status for key, item in by_id.items()}
        reasons = {key: item.reason for key, item in by_id.items()}
        bases = {key: item.basis for key, item in by_id.items()}
        approved = (
            not errors
            and not response.shared_failure
            and all(statuses.get(action_id) == "AVAILABLE" for action_id in ("A0", "A1"))
            and all(bases.get(action_id) != "INSUFFICIENT" for action_id in ("A0", "A1"))
        )
        if response.shared_failure:
            errors.append(
                "shared failure: the failure removes a capability used by the synthesis and at least one fallback; "
                "the original fallbacks may still remain individually available"
            )
        for action_id in ("A0", "A1"):
            if statuses.get(action_id, "UNKNOWN") != "AVAILABLE":
                errors.append(
                    f"{action_id} availability is {statuses.get(action_id, 'UNKNOWN')}"
                )
            if bases.get(action_id, "INSUFFICIENT") == "INSUFFICIENT":
                errors.append(f"{action_id} availability has insufficient evidence")
        return ContingencyFeasibilityAssessment(
            condition.synthesis_action,
            condition.predicate_label,
            statuses,
            reasons,
            bases,
            response.shared_failure,
            valid=not (set(by_id) != {"A0", "A1"}),
            approved=approved,
            error="; ".join(dict.fromkeys(errors)),
        )
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        return ContingencyFeasibilityAssessment(
            condition.synthesis_action,
            condition.predicate_label,
            {}, {}, {}, False, valid=False, approved=False,
            error=f"independent feasibility verification unavailable: {exc}",
        )
