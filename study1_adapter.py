"""Matched structured solo-response adapter for Study 1."""
from __future__ import annotations

import json
from typing import Any, Sequence


SOLO_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {"type": ["integer", "null"]},
        "status": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "framework_state": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "choice": {"type": ["integer", "null"]},
                    "status": {"type": "string"},
                },
                "required": ["name", "choice", "status"],
                "additionalProperties": False,
            },
        },
        "constraints": {"type": "array", "items": {"type": "string"}},
        "propositions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["choice", "status", "confidence", "framework_state", "constraints", "propositions"],
    "additionalProperties": False,
}


def build_structured_solo_prompt(question: str, actions: Sequence[str]) -> str:
    """Request state fields without asking the solo model to rate answer quality."""
    return f"""Analyze this ethical scenario and return JSON only.
Do not judge or score the quality of your answer. Record only the response state
you would have produced for this scenario. `choice` is the zero-based action index,
or null if unresolved. `status` is your own state label (for example RESOLVED or
UNRESOLVED). `confidence` is epistemic confidence in that state, not answer quality.
`framework_state` may be empty; do not imitate a Parliament or invent framework
votes. `constraints` are the considerations that changed or governed your state.
`propositions` are short factual premises you actually used; do not add facts.

SCENARIO: {question}
ACTIONS: {json.dumps(list(actions))}

Return exactly an object matching this shape:
{{"choice":null,"status":"UNRESOLVED","confidence":0.0,
"framework_state":[],"constraints":[],"propositions":[]}}"""


__all__ = ("SOLO_RESPONSE_SCHEMA", "build_structured_solo_prompt")
