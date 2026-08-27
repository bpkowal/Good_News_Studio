from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .structured_io import ModelCallBudgetExceeded, call_json_llm, extract_json


_COMPARATIVE_RANKING = re.compile(
    r"\b(?:better\s+off|worse\s+off|"
    r"more\s+than|less\s+than|greater\s+than|fewer\s+than|"
    r"outweigh\w*|exceed\w*|surpass\w*|dominat\w*|trump\w*|override\w*|"
    r"prefer\w*|rank\w*)\b",
    re.IGNORECASE,
)

_RELATIONAL_PREMISE = re.compile(
    r"\b(?:third\s+party|authorized\s+internal\s+agent|internal\s+agent|"
    r"authorized\s+agent|authorization\s+scope|consent\s+scope|privacy\s+scope|promise\s+scope|"
    r"least[- ]advantaged|most\s+vulnerable|better\s+off|worse\s+off|"
    r"materially\s+better\s+off|materially\s+worse\s+off|organizational\s+boundary|"
    r"role\s+relation|status)\b",
    re.IGNORECASE,
)

def _comparative_claim_errors(
    actions: Sequence[str], landscape_cases: dict[str, str]
) -> list[str]:
    """Identify explicit comparative claims that need an action-to-target fact.

    This is only an admission trigger for semantic verification. It does not
    decide from vocabulary alone whether the claim is true.
    """
    errors: list[str] = []
    for index, action in enumerate(actions):
        case = landscape_cases.get(action, "")
        comparative_fragments = [
            fragment.strip()
            for fragment in re.split(r"(?<=[.!?;])\s+|\s+[—–-]\s+", case)
            if _COMPARATIVE_RANKING.search(fragment)
        ]
        if not comparative_fragments:
            continue
        comparative_text = " ".join(comparative_fragments)
        if _RELATIONAL_PREMISE.search(comparative_text):
            errors.append(
                f"UNRESOLVED_RELATIONAL_PREMISE: case for A{index} makes an unverified "
                f"comparative beneficiary claim (needs a resolved relational premise; "
                f"operator={comparative_fragments[0][:80]})"
            )
        else:
            errors.append(
                f"COMPARATIVE_MAGNITUDE: case for A{index} makes an unverified comparative "
                f"beneficiary claim (operator={comparative_fragments[0][:100]})"
            )
    return errors


def verify_landscape_alignment(
    llm: Any,
    actions: Sequence[str],
    landscape_cases: dict[str, str],
    suspected_errors: Sequence[str],
    *,
    scenario: str = "",
    max_tokens: int = 128,
) -> list[str]:
    """Verify action mapping and action-to-beneficiary grounding semantically."""
    mapping_errors = [
        error for error in suspected_errors
        if "tracks the opposing action but not its own action" in error
    ]
    comparative_errors = _comparative_claim_errors(actions, landscape_cases) if scenario else []
    other_errors = [
        error for error in suspected_errors
        if error not in mapping_errors and error not in comparative_errors
    ]
    if not mapping_errors and not comparative_errors:
        return list(suspected_errors)
    action_ids = [f"A{index}" for index in range(len(actions))]
    schema = {
        "type": "object",
        "properties": {
            "aligned": {
                "type": "object",
                "properties": {key: {"type": "boolean"} for key in action_ids},
                "required": action_ids,
                "additionalProperties": False,
            },
            "reason": {
                "type": "object",
                "properties": {key: {"type": "string", "maxLength": 140} for key in action_ids},
                "required": action_ids,
                "additionalProperties": False,
            },
            "comparative_grounded": {
                "type": "object",
                "properties": {key: {"type": "boolean"} for key in action_ids},
                "required": action_ids,
                "additionalProperties": False,
            },
            "fact_node": {
                "type": "object",
                "properties": {
                    key: {
                        "type": "string",
                        "enum": [*action_ids, "SCENARIO", "MULTIPLE_ACTIONS", "NONE"],
                    }
                    for key in action_ids
                },
                "required": action_ids,
                "additionalProperties": False,
            },
        },
        "required": ["aligned", "reason", "comparative_grounded", "fact_node"],
        "additionalProperties": False,
    }
    payload = {
        key: {"action": action, "case": landscape_cases.get(action, "")}
        for key, action in zip(action_ids, actions)
    }
    prompt = f"""Verify causal and beneficiary grounding only; do not rank the actions.
Committed scenario: {' '.join(scenario.split())[:900] or 'NONE'}
Action-case pairs: {json.dumps(payload)}

For each ID, aligned=true only if its case describes a mechanism, consequence, harm,
benefit, affected party, or value that follows from choosing ITS OWN action. A case may
compare the rival afterward and may paraphrase or use synonyms. Shared outcomes in a
symmetric trade-off are allowed when the case explains how its own action affects them.
aligned=false when the case actually describes choosing the opposing action, swaps the
actions' consequences, or gives no identifiable consequence of its own action.
For any case that explicitly ranks outcomes, says one side is better/worse off,
or uses comparative language like "outweighs", "exceeds", "more than", "less than",
"least advantaged", or "most vulnerable", set comparative_grounded=true only when
a committed scenario fact establishes the target's treatment under this action
relative to the rival. Plain descriptive consequences are allowed and should stay
grounded even if they mention benefits, harms, privacy, or welcome effects.
"No additional aid" does not mean "made worse," and "unchanged" does not mean
"protected" unless the other action explicitly worsens that same target. A
procedural value such as non-interference may support a procedural argument but
cannot by itself establish that a population is materially better off. fact_node
identifies the supporting ActionNode, MULTIPLE_ACTIONS, SCENARIO, or NONE. For
cases without such a ranking claim, return comparative_grounded=true and
fact_node=NONE.
For each flagged case, explicitly answer: which committed fact makes the claimed
target better or worse off under this action than its rival? If none, use false/NONE.
Return JSON only.
"""
    try:
        output = call_json_llm(
            llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema,
            call_kind="auxiliary", cache=True,
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = extract_json(raw)
        aligned = data.get("aligned")
        reasons = data.get("reason")
        comparative_grounded = data.get("comparative_grounded", {})
        fact_nodes = data.get("fact_node", {})
        if not isinstance(aligned, dict) or set(aligned) != set(action_ids):
            raise ValueError("semantic landscape verdict must cover every action")
        if not isinstance(reasons, dict) or set(reasons) != set(action_ids):
            raise ValueError("semantic landscape reasons must cover every action")
        if comparative_errors and (
            not isinstance(comparative_grounded, dict)
            or set(comparative_grounded) != set(action_ids)
            or not isinstance(fact_nodes, dict)
            or set(fact_nodes) != set(action_ids)
        ):
            raise ValueError("comparative grounding verdict must cover every action")
        confirmed = []
        for index, key in enumerate(action_ids):
            lexical_error = f"case for A{index} tracks the opposing action but not its own action"
            if lexical_error in mapping_errors and aligned[key] is not True:
                reason = " ".join(str(reasons[key]).split())
                confirmed.append(
                    lexical_error + (f" ({reason})" if reason else "")
                )
            comparative_error_prefixes = (
                f"COMPARATIVE_MAGNITUDE: case for A{index} makes an unverified comparative "
                f"beneficiary claim",
                f"UNRESOLVED_RELATIONAL_PREMISE: case for A{index} makes an unverified "
                f"comparative beneficiary claim",
            )
            matched_comparative_error = next(
                (
                    error for error in comparative_errors
                    if any(error.startswith(prefix) for prefix in comparative_error_prefixes)
                ),
                "",
            )
            if (
                bool(matched_comparative_error)
                and (
                    comparative_grounded.get(key) is not True
                    or fact_nodes.get(key) in {None, "NONE"}
                )
            ):
                reason = " ".join(str(reasons[key]).split())
                confirmed.append(
                    matched_comparative_error + (f" ({reason})" if reason else "")
                )
        return [*other_errors, *confirmed]
    except (ModelCallBudgetExceeded, ValueError, json.JSONDecodeError, KeyError, TypeError):
        # A failed verifier must not erase a conservative lexical warning.
        return list(suspected_errors)
