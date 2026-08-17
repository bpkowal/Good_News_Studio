from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .structured_io import ModelCallBudgetExceeded, call_json_llm, extract_json


_COMPARATIVE_BENEFICIARY = re.compile(
    r"\b(?:(?:protect|benefit|safeguard|improv|abandon|neglect|disadvantag)\w*|"
    r"better\s+off|worse\s+off|"
    r"least[- ]advantaged|most\s+vulnerable|fair\s+claim)\b",
    re.IGNORECASE,
)

_RELATION_FAMILIES = {
    "POSITIVE": re.compile(
        r"\b(?:(?:protect|benefit|safeguard|improv|rescu)\w*|"
        r"save|saves|saved|saving)\b", re.I
    ),
    "ADVERSE": re.compile(r"\b(?:abandon|neglect|disadvantag)\w*\b|\bworse\s+off\b", re.I),
    "PRIORITY": re.compile(
        r"\b(?:least[- ]advantaged|most\s+vulnerable|fair\s+claim|better\s+off)\b",
        re.I,
    ),
}
_GROUNDING_WORDS_IGNORED = {
    "action", "option", "protect", "protects", "protected", "protecting",
    "benefit", "benefits", "benefited", "benefiting", "safeguard", "safeguards",
    "improve", "improves", "improved", "abandon", "abandons", "abandoned",
    "abandonment", "neglect", "neglects", "neglected", "disadvantage",
    "disadvantages", "disadvantaged", "better", "worse", "least", "most",
    "advantaged", "fair", "claim", "with", "without", "while",
    "save", "saves", "saved", "saving", "rescue", "rescues", "rescued",
    "that", "this", "their", "from", "under", "than", "into", "through",
}


def _relation_families(text: str) -> set[str]:
    return {
        family for family, pattern in _RELATION_FAMILIES.items()
        if pattern.search(text)
    }


def _content_words(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) >= 4 and token not in _GROUNDING_WORDS_IGNORED
    }


def _explicitly_grounded_in_action(action: str, claim: str) -> bool:
    """Fast-path literal action facts before spending an auxiliary model call."""
    claim_relations = _relation_families(claim)
    action_relations = _relation_families(action)
    if not claim_relations or not (claim_relations & action_relations):
        return False
    # Matching a relation word alone is insufficient: the action and claim must
    # also identify at least one common target/mechanism anchor.
    return bool(_content_words(action) & _content_words(claim))


def _comparative_claim_errors(
    actions: Sequence[str], landscape_cases: dict[str, str]
) -> list[str]:
    """Identify beneficiary comparisons that need an action-to-target fact.

    This is only an admission trigger for semantic verification. It does not
    decide from vocabulary alone whether the claim is true.
    """
    return [
        f"case for A{index} makes an unverified comparative beneficiary claim"
        for index, action in enumerate(actions)
        if (
            _COMPARATIVE_BENEFICIARY.search(landscape_cases.get(action, ""))
            and not _explicitly_grounded_in_action(
                action, landscape_cases.get(action, "")
            )
        )
    ]


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
For any case claiming that its action protects, benefits, safeguards, improves,
abandons, neglects, disadvantages, or is better/worse for a person or group, set
comparative_grounded=true only when a committed scenario fact establishes the
target's treatment under that action relative to the rival. "No additional aid"
does not mean "made worse," and "unchanged" does not mean "protected" unless the
other action explicitly worsens that same target. A procedural value such as
non-interference may support a procedural argument but cannot by itself establish
that a population is materially better off. fact_node identifies the supporting
ActionNode, MULTIPLE_ACTIONS, SCENARIO, or NONE. For cases without such a claim, return
comparative_grounded=true and fact_node=NONE.
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
            comparative_error = (
                f"case for A{index} makes an unverified comparative beneficiary claim"
            )
            if (
                comparative_error in comparative_errors
                and (
                    comparative_grounded.get(key) is not True
                    or fact_nodes.get(key) in {None, "NONE"}
                )
            ):
                reason = " ".join(str(reasons[key]).split())
                confirmed.append(
                    comparative_error + (f" ({reason})" if reason else "")
                )
        return [*other_errors, *confirmed]
    except (ModelCallBudgetExceeded, ValueError, json.JSONDecodeError, KeyError, TypeError):
        # A failed verifier must not erase a conservative lexical warning.
        return list(suspected_errors)
