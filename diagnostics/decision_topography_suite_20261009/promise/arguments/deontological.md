# Deontological — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

The agent assesses Ada’s duty “perfect duty to keep promises” toward cara. Ada is the duty-bearer; cara is the protected party. The competing obligation being assessed is Ada’s “imperfect duty of beneficence” toward ben. For Ada’s option “Ada gives the medicine to Ben”, the ledger records conflicted. The agent’s stated reason is: breaks a promise. For Ada’s option “Ada gives the medicine to Cara”, the ledger records conflicted. The agent’s stated reason is: keeps the promise. The ledger retains unresolved issues; this does not establish that one duty overrides the other.

Presentation basis: READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: prefer “Ada gives the medicine to Cara” because perfect duty of promise-keeping outranks imperfect beneficence
- Recorded interpretive commitment: beneficence numbers are secondary to promise duty
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- A1 ADJUDICATION_CALIBRATION: perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis
- A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- what makes the positive duty perfect rather than imperfect
- cara: perfect duty to keep promises ↔ ben: imperfect duty of beneficence

## Conditions for reconsidering

- The agent reports a switch from “Ada gives the medicine to Cara” to “Ada gives the medicine to Ben”. Condition: would reverse if promise-keeping not obligatory. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Provisionally: Deontology identifies competing claims between cara's perfect duty to keep promises and ben's imperfect duty of beneficence. NONE's restriction of NONE is not yet justified because what makes the positive duty perfect rather ",
  "framework_application": "NONE",
  "decision_rule": "Provisionally: Keep the action contested until what makes the positive duty perfect rather than imperfect",
  "landscape_cases": {
    "Ada gives the medicine to Ben": "best case: saves Ben, meets beneficence",
    "Ada gives the medicine to Cara": "best case: keeps promise, respects perfect duty"
  },
  "proposed_framework_state": {
    "constraint": "DUTY",
    "action_scores": {
      "Ada gives the medicine to Ben": 0.20749999999999996,
      "Ada gives the medicine to Cara": 0.7925
    },
    "recommended_action": "Ada gives the medicine to Cara",
    "rationale": "Deontology identifies competing claims between cara's perfect duty to keep promises and ben's imperfect duty of beneficence. NONE's restriction of NONE is not yet justified because what makes the positive duty perfect rather than imperfect. The Kantian judgment remains contested.",
    "decision_rule": "Keep the action contested until what makes the positive duty perfect rather than imperfect",
    "framework_action_map": {
      "Ada gives the medicine to Ben": "PROHIBITED: violates perfect duty of promise-keeping despite beneficence",
      "Ada gives the medicine to Cara": "REQUIRED: fulfills perfect duty of promise-keeping; imperfect beneficence yields"
    },
    "framework_numerical_role": "SECONDARY",
    "framework_numerical_justification": "beneficence numbers are secondary to promise duty",
    "action_admissibility": {
      "Ada gives the medicine to Ben": "REJECTED",
      "Ada gives the medicine to Cara": "REQUIRED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "Ada gives the medicine to Cara",
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "would reverse if promise-keeping not obligatory",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "DUTY",
          "norm": "perfect duty to keep promises",
          "relation": "VIOLATES",
          "duty_bearer": "Ada",
          "protected_party": "Cara",
          "competing_norm": "imperfect duty of beneficence",
          "competing_norm_kind": "DUTY",
          "competing_relation": "SATISFIES",
          "competing_protected_party": "Ben",
          "competing_reason": "saving Ben advances beneficence",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relationship beyond promise",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "perfect duties override imperfect duties",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "EXTERNAL_FREEDOM",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "cannot universalize promise-breaking",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "breaks a promise"
        },
        {
          "action_id": "A1",
          "verdict": "REQUIRED",
          "norm_kind": "DUTY",
          "norm": "perfect duty to keep promises",
          "relation": "SATISFIES",
          "duty_bearer": "Ada",
          "protected_party": "Cara",
          "competing_norm": "imperfect duty of beneficence",
          "competing_norm_kind": "DUTY",
          "competing_relation": "VIOLATES",
          "competing_protected_party": "Ben",
          "competing_reason": "allows Ben to die",
          "duty_type": "PERFECT_POSITIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special obligation to Ben",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "perfect duties override imperfect duties",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "EXTERNAL_FREEDOM",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "promise-keeping can be universal law",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "keeps the promise"
        }
      ]
    },
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": ""
  },
  "framework_vote_reason": "selected action has non-ranking duty verdict CONFLICTED",
  "investigative_claim": "UNRESOLVED DUTY CONFLICT: cara: perfect duty to keep promises ↔ ben: imperfect duty of beneficence. Current reasoning leans Ada gives the medicine to Cara, but the competing strict claim has not been vindicated or defeated.",
  "committed_native_ledger": {
    "ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER",
    "transaction_status": "COMMITTED_WITH_UNCERTAINTY",
    "records": [
      {
        "assessment_node_id": "DEON_ASSESSMENT:deontological:A0",
        "framework": "DEONTOLOGICAL",
        "specialist": "deontological",
        "assessment_role": "PRIMARY",
        "cycle": 1,
        "canonical_action_id": "A0",
        "verdict": "CONFLICTED",
        "proposed_verdict": "CONFLICTED",
        "relation": "VIOLATES",
        "proposed_relation": "VIOLATES",
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:c18fe86a51e5ca4f",
        "party_node_id": "DEON_PARTY:f61f5c5e22749aeb",
        "bearer_node_id": "DEON_BEARER:fdee430d40bd57de",
        "competing_norm": "imperfect duty of beneficence",
        "competing_norm_kind": "DUTY",
        "competing_relation": "SATISFIES",
        "competing_protected_party": "ben",
        "competing_reason": "saving Ben advances beneficence",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "ALLOWING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special relationship beyond promise",
        "means_relation": "NO_INSTRUMENTALIZATION",
        "governing_norm": "UNRESOLVED",
        "priority_basis": "UNRESOLVED",
        "priority_rule": "perfect duties override imperfect duties",
        "protected_standing": "CONTRACTUAL_RIGHT",
        "competing_protected_standing": "EXTERNAL_FREEDOM",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "cannot universalize promise-breaking",
        "reciprocity_status": "FAILED",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "UNIVERSAL_LAW",
        "resolution_status": "CONTESTED",
        "calibration_errors": [
          "omission was classified as a perfect negative-duty violation without a separate basis",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "calibration_issues": [
          {
            "kind": "OMISSION_PERFECT_NEGATIVE_VIOLATION",
            "message": "omission was classified as a perfect negative-duty violation without a separate basis",
            "status": "ACTIVE"
          },
          {
            "kind": "CALIBRATION",
            "message": "resolved adjudication rests on unsupported decisive premises",
            "status": "ACTIVE"
          }
        ],
        "calibration_support_node_ids": [],
        "evidence_basis": "FRAMEWORK_ONLY",
        "epistemic_status": "FRAMEWORK_INTERPRETATION",
        "reason": "breaks a promise",
        "protected_party": "cara",
        "norm": "perfect duty to keep promises",
        "duty_bearer": "ada",
        "provenance": [
          "delegate:deontological",
          "cycle:1"
        ]
      },
      {
        "assessment_node_id": "DEON_ASSESSMENT:deontological:A1",
        "framework": "DEONTOLOGICAL",
        "specialist": "deontological",
        "assessment_role": "PRIMARY",
        "cycle": 1,
        "canonical_action_id": "A1",
        "verdict": "CONFLICTED",
        "proposed_verdict": "CONFLICTED",
        "relation": "SATISFIES",
        "proposed_relation": "SATISFIES",
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:c18fe86a51e5ca4f",
        "party_node_id": "DEON_PARTY:f61f5c5e22749aeb",
        "bearer_node_id": "DEON_BEARER:fdee430d40bd57de",
        "competing_norm": "imperfect duty of beneficence",
        "competing_norm_kind": "DUTY",
        "competing_relation": "VIOLATES",
        "competing_protected_party": "ben",
        "competing_reason": "allows Ben to die",
        "duty_type": "PERFECT_POSITIVE",
        "harm_relation": "ALLOWING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special obligation to Ben",
        "means_relation": "NO_INSTRUMENTALIZATION",
        "governing_norm": "UNRESOLVED",
        "priority_basis": "UNRESOLVED",
        "priority_rule": "perfect duties override imperfect duties",
        "protected_standing": "CONTRACTUAL_RIGHT",
        "competing_protected_standing": "EXTERNAL_FREEDOM",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "promise-keeping can be universal law",
        "reciprocity_status": "SATISFIED",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "UNIVERSAL_LAW",
        "resolution_status": "CONTESTED",
        "calibration_errors": [
          "perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "calibration_issues": [
          {
            "kind": "PERFECT_POSITIVE_LACKS_BASIS",
            "message": "perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis",
            "status": "ACTIVE"
          },
          {
            "kind": "CALIBRATION",
            "message": "resolved adjudication rests on unsupported decisive premises",
            "status": "ACTIVE"
          }
        ],
        "calibration_support_node_ids": [],
        "evidence_basis": "FRAMEWORK_ONLY",
        "epistemic_status": "FRAMEWORK_INTERPRETATION",
        "reason": "keeps the promise",
        "protected_party": "cara",
        "norm": "perfect duty to keep promises",
        "duty_bearer": "ada",
        "provenance": [
          "delegate:deontological",
          "cycle:1"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "would reverse if promise-keeping not obligatory",
  "factual_reversal_threshold": "NONE",
  "model_argument_fields": {
    "rationale": "Promise is a perfect duty",
    "framework_application": "NONE",
    "decision_rule": "prefer A1 because perfect duty of promise-keeping outranks imperfect beneficence",
    "landscape_cases": {
      "A0": "best case: saves Ben, meets beneficence",
      "A1": "best case: keeps promise, respects perfect duty"
    },
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "would reverse if promise-keeping not obligatory"
  }
}
```
