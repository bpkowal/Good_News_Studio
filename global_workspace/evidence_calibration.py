from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .structured_io import ModelCallBudgetExceeded, call_json_llm, extract_json


@dataclass(frozen=True, slots=True)
class EvidenceCalibration:
    tier: str
    direction_retention: float
    epistemic_cap: float
    requires_verification: bool
    reason: str = ""
    valid: bool = True


_TIERS = {
    "ENTAILED": (1.00, 0.85, False),
    "BOUNDED_BACKGROUND": (0.80, 0.70, True),
    "DECISION_CRITICAL": (0.55, 0.50, True),
    "REMOTE": (0.35, 0.35, True),
}


def default_evidence_calibration(reason: str = "calibration unavailable") -> EvidenceCalibration:
    retention, cap, verify = _TIERS["DECISION_CRITICAL"]
    return EvidenceCalibration(
        "DECISION_CRITICAL", retention, cap, verify, reason, valid=False
    )


def calibrate_speculative_claim(
    llm: Any,
    scenario: str,
    actions: Sequence[str],
    claim: str,
    rationale: str,
    *,
    max_tokens: int = 128,
) -> EvidenceCalibration:
    """Classify evidential distance without evaluating the moral recommendation."""
    schema = {
        "type": "object",
        "properties": {
            "tier": {"type": "string", "enum": list(_TIERS)},
            "reason": {"type": "string", "maxLength": 180},
        },
        "required": ["tier", "reason"],
        "additionalProperties": False,
    }
    prompt = f"""You are an evidence calibrator, not an ethical voter.
Scenario: {' '.join(scenario.split())[:1000]}
Actions: {json.dumps(list(actions))}
Claim: {' '.join(claim.split())[:240]}
Reason containing the claim: {' '.join(rationale.split())[:240]}

Classify only how far the claim goes beyond the scenario:
- ENTAILED: explicitly stated or strongly entailed by the scenario's causal setup.
- BOUNDED_BACKGROUND: an ordinary, stable causal premise that adds no new actors,
  quantities, extreme outcomes, exclusivity, or special implementation success.
- DECISION_CRITICAL: an unstated premise whose truth could reverse the action ranking,
  including claims that an option is uniquely effective or feasible.
- REMOTE: a weakly grounded chain, invented event, hidden population, or dramatic
  extrapolation remote from the stated facts.

Separate a supported core from an unsupported qualifier. For example, a scenario may
entail that containment reduces exposure while not entailing that one containment
method is uniquely effective. Do not use moral desirability, consensus, or severity
of the stakes as evidence. Return JSON only.
"""
    try:
        output = call_json_llm(
            llm, prompt, max_tokens=max_tokens, temperature=0.0, schema=schema,
            call_kind="auxiliary", cache=True,
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = extract_json(raw)
        tier = str(data.get("tier", "")).strip().upper()
        if tier not in _TIERS:
            raise ValueError("unknown evidence calibration tier")
        reason = " ".join(str(data.get("reason", "")).split())
        if len(reason.split()) < 3:
            raise ValueError("evidence calibration needs a reason")
        retention, cap, verify = _TIERS[tier]
        return EvidenceCalibration(tier, retention, cap, verify, reason)
    except (ModelCallBudgetExceeded, ValueError, json.JSONDecodeError, KeyError, TypeError) as error:
        return default_evidence_calibration(str(error))
