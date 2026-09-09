"""Independent audit for empirical premises omitted from candidate bindings.

The audit is semantic rather than vocabulary-based: ordinary paraphrase is allowed,
while added outcomes, severity, probability, actors, mechanisms, or exclusivity must
be bound to an existing proposition or admitted as an unestablished hypothesis.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from .models import CandidateChunk
from .structured_io import call_json_llm, extract_json


_DECISION_FIELDS = (
    "rationale", "decision_rule", "landscape_cases", "landscape_decisive_axis",
    "landscape_tiebreaker", "factual_reversal_threshold", "reversal_condition",
    "unsupported_assumption", "framework_application", "framework_action_map",
    "utilitarian_consequence_table", "rawls_position_proposal",
    "deontological_ledger_proposal", "virtue_character_proposal",
    "care_ledger_proposal", "care_relational_map",
)


def _compact_value(value: Any, *, limit: int = 1800) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())[:limit]
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, limit=500)
            for key, item in list(value.items())[:16]
        }
    if isinstance(value, (list, tuple)):
        return [_compact_value(item, limit=500) for item in list(value)[:16]]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return " ".join(str(value).split())[:limit]


def candidate_audit_packet(candidate: CandidateChunk) -> dict[str, Any]:
    return {
        "specialist": candidate.specialist,
        "declared_supporting_proposition_ids": list(
            candidate.supporting_proposition_ids
        ),
        "declared_decision_critical_proposition_ids": list(
            candidate.decision_critical_proposition_ids
        ),
        "declared_empirical_premises": list(candidate.material_empirical_claims),
        "decision_fields": {
            field: _compact_value(getattr(candidate, field, None))
            for field in _DECISION_FIELDS
            if getattr(candidate, field, None) not in (None, "", {}, [])
        },
    }


def audit_side_premises(
    llm: Any,
    proposition_ledger: Sequence[dict[str, Any]],
    candidates: Iterable[CandidateChunk],
    *,
    max_tokens: int = 700,
) -> dict[str, Any]:
    """Find material empirical specificity missing from declared dependencies."""
    valid_candidates = [candidate for candidate in candidates if candidate.schema_valid]
    specialists = list(dict.fromkeys(candidate.specialist for candidate in valid_candidates))
    if not valid_candidates:
        return {"status": "PASSED", "findings": [], "error": ""}
    proposition_ids = [
        str(row.get("proposition_id", ""))
        for row in proposition_ledger if str(row.get("proposition_id", ""))
    ]
    schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array", "minItems": 0, "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "specialist": {"type": "string", "enum": specialists},
                        "claim": {"type": "string", "minLength": 4, "maxLength": 240},
                        "binding": {
                            "type": "string",
                            "enum": [
                                *proposition_ids,
                                "DERIVED_ESTABLISHED",
                                "NEW_HYPOTHESIS",
                                "FRAMEWORK_DERIVED",
                            ],
                        },
                        "derived_from": {
                            "type": "array", "minItems": 0, "maxItems": 4,
                            "items": {"type": "string", "enum": proposition_ids},
                        },
                        "decision_critical": {"type": "boolean"},
                        "source_field": {
                            "type": "string", "minLength": 2, "maxLength": 80,
                        },
                        "reason": {"type": "string", "minLength": 4, "maxLength": 180},
                    },
                    "required": [
                        "specialist", "claim", "binding", "derived_from",
                        "decision_critical", "source_field", "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["findings"],
        "additionalProperties": False,
    }
    prompt = f"""You are an independent empirical-premise auditor, not an ethical voter.
Authoritative proposition ledger:
{json.dumps(list(proposition_ledger), ensure_ascii=False, sort_keys=True)}

Candidate decision records:
{json.dumps([candidate_audit_packet(c) for c in valid_candidates], ensure_ascii=False, sort_keys=True)}

Find every MATERIAL empirical premise used in a decision-bearing field that is not
covered by the candidate's declared proposition bindings. Audit semantic content,
not keywords. An ordinary paraphrase that preserves the scope and strength of an
authoritative claim is covered. A phrase adds empirical specificity when it adds
an outcome, severity, probability, affected actor, causal mechanism, exclusivity,
comparative magnitude, timing, or certainty that the bound proposition does not
establish. Normative classifications such as duty, compassion, fairness, virtue,
basic-good priority, or moral urgency are not empirical additions by themselves.

ATOMIZATION CONTRACT: each finding must contain only the smallest independently
assessable empirical addition. If candidate wording combines established content
with an unsupported addition, do not repeat the whole sentence as a hypothesis:
report only the unsupported atom. For example, when a ledger establishes acute
dehydration but not fatality, the finding is the fatality or mortality-risk claim,
not "the residents are acutely dehydrated and in mortal peril." Quantities,
affected populations, and time horizons written in an authoritative proposition
are established content and must not be reported as hypotheses.

A specific outcome is allowed when it is established or explicitly treated as an
unestablished condition. The error is silently using added specificity as though it
were established. When an existing HYPOTHETICAL, UNRESOLVED, or WORLD_ESTABLISHED proposition expresses
the added premise, set binding to that proposition ID even if wording differs. Use
NEW_HYPOTHESIS only when no existing proposition is semantically equivalent, and
cite the closest supporting propositions in derived_from. A restatement of an
admitted world effect, causal link, or counterfactual foreclosure is not a new
hypothesis. Use FRAMEWORK_DERIVED, not NEW_HYPOTHESIS, for framework-native
normative relations (doing/allowing, means, duty of care, least-advantaged
priority, practical wisdom) that add no descriptive outcome. A claim that denies,
reopens, or treats as avoidable an admitted CERTAIN world effect is still a
descriptive NEW_HYPOTHESIS: report it so the system can quarantine that candidate.
decision_critical=true
when changing or removing the premise could materially weaken, reverse, or remove
the candidate's stated ranking or normative classification. Do not infer factual
authority from repetition, agreement, salience, or moral importance.

Use DERIVED_ESTABLISHED only when the claim is a transparent conjunction or direct
restatement of two or more ESTABLISHED/DERIVED ledger propositions and adds no new
actor, outcome, mechanism, severity, probability, magnitude, timing, or certainty.
List every composing proposition in derived_from. A causal inference not already
expressed by those propositions is not transparent composition and remains a
NEW_HYPOTHESIS.

Return JSON only. Return an empty findings list when every material empirical
premise is already covered.
"""
    try:
        output = call_json_llm(
            llm, prompt, max_tokens=max(256, max_tokens), temperature=0.0,
            schema=schema, call_kind="epistemic_audit", cache=True,
        )
        raw = output["choices"][0]["text"] if isinstance(output, dict) else str(output)
        data = extract_json(raw)
        findings = data.get("findings", [])
        if not isinstance(findings, list):
            raise ValueError("side-premise findings must be a list")
        cleaned = [dict(item) for item in findings if isinstance(item, dict)]
        return {
            "status": "FINDINGS" if cleaned else "PASSED",
            "findings": cleaned,
            "error": "",
        }
    except Exception as error:
        return {
            "status": "UNAVAILABLE", "findings": [],
            "error": f"{type(error).__name__}: {error}"[:240],
        }
