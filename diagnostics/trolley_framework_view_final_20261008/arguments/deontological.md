# Deontological — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

Maria is the duty-bearer. The norm “do not kill innocent person” protects one worker. The competing norm “duty to rescue many” protects five workers. For “pulls the lever”, the ledger records conflicted. The agent’s stated reason is: directly causing death violates perfect duty. For “does not pull the lever”, the ledger records permissible. The agent’s stated reason is: allows harm but avoids killing. The ledger retains unresolved issues; this does not establish that one duty overrides the other.

Presentation basis: READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Prefer “does not pull the lever” when one option does killing and the other only allows harm
- Recorded interpretive commitment: Numbers do not override perfect duties
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; five workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- one worker: do not kill innocent person ↔ five workers: duty to rescue many
- A0 prohibited by do not kill innocent person ↔ whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict

## Conditions for reconsidering

- The agent reports a switch from “does not pull the lever” to “pulls the lever”. Condition: If rescue duty outweighed no-killing duty. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Deontology ranks one worker's do not kill innocent person above five workers's duty to rescue many through respect persons, but that priority holds only if A0 is genuinely provisionally favored under incomplete adjudication, and the prohibi",
  "framework_application": "NONE",
  "decision_rule": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
  "landscape_cases": {
    "pulls the lever": "PROHIBITED: directly kills an innocent though rescues five",
    "does not pull the lever": "PERMISSIBLE: refrains from killing though fails to rescue"
  },
  "proposed_framework_state": {
    "constraint": "DUTY",
    "action_scores": {
      "pulls the lever": 0.305,
      "does not pull the lever": 0.6950000000000001
    },
    "recommended_action": "does not pull the lever",
    "rationale": "Deontology ranks one worker's do not kill innocent person above five workers's duty to rescue many through respect persons, but that priority holds only if A0 is genuinely prohibited, and the prohibition rests on unestablished premises: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict. The comparative Kantian judgment remains contested.",
    "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
    "framework_action_map": {
      "pulls the lever": "PROHIBITED: violates perfect negative duty not to kill; foreseen side-effect still doing harm",
      "does not pull the lever": "PERMISSIBLE: consistent with perfect negative duty; omission only allows harm"
    },
    "framework_numerical_role": "SECONDARY",
    "framework_numerical_justification": "Numbers do not override perfect duties",
    "action_admissibility": {
      "pulls the lever": "REJECTED",
      "does not pull the lever": "PERMISSIBLE"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "does not pull the lever",
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "If rescue duty outweighed no-killing duty",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "RESPECT_PERSONS",
          "norm": "do not kill innocent person",
          "relation": "VIOLATES",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "duty to rescue many",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "five workers",
          "competing_reason": "imperfect duty to aid cannot override perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation needed for negative duty",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "perfect negative duty prevails",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "killing cannot be publicly justified",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "directly causing death violates perfect duty"
        },
        {
          "action_id": "A1",
          "verdict": "PERMISSIBLE",
          "norm_kind": "RESPECT_PERSONS",
          "norm": "do not kill innocent person",
          "relation": "SATISFIES",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "duty to rescue many",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "five workers",
          "competing_reason": "imperfect rescue duty exists but secondary",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special obligation to those workers",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "omission respects perfect duty",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "can be universally willed not to kill",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "allows harm but avoids killing"
        }
      ]
    },
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": ""
  },
  "framework_vote_reason": "the duty ledger permits the selected action but does not rank it above its rivals",
  "investigative_claim": "UNRESOLVED DUTY CONFLICT: one worker: do not kill innocent person ↔ five workers: duty to rescue many. Current reasoning leans does not pull the lever, but the competing strict claim has not been vindicated or defeated.",
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
        "norm_kind": "RESPECT_PERSONS",
        "norm_node_id": "DEON_NORM:4e9d4d2dc06149cc",
        "party_node_id": "DEON_PARTY:879bcf30890178b8",
        "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
        "competing_norm": "duty to rescue many",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "five workers",
        "competing_reason": "imperfect duty to aid cannot override perfect duty",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "DOING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special relation needed for negative duty",
        "means_relation": "FORESEEN_SIDE_EFFECT",
        "governing_norm": "UNRESOLVED",
        "priority_basis": "UNRESOLVED",
        "priority_rule": "perfect negative duty prevails",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "OTHER",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "killing cannot be publicly justified",
        "reciprocity_status": "FAILED",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "RESPECT_PERSONS",
        "resolution_status": "CONTESTED",
        "calibration_errors": [
          "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "calibration_issues": [
          {
            "kind": "HARM_RELATION_GRAPH_MISALIGN",
            "message": "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
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
        "reason": "directly causing death violates perfect duty",
        "protected_party": "one worker",
        "norm": "do not kill innocent person",
        "duty_bearer": "maria",
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
        "verdict": "PERMISSIBLE",
        "proposed_verdict": "PERMISSIBLE",
        "relation": "SATISFIES",
        "proposed_relation": "SATISFIES",
        "norm_kind": "RESPECT_PERSONS",
        "norm_node_id": "DEON_NORM:4e9d4d2dc06149cc",
        "party_node_id": "DEON_PARTY:879bcf30890178b8",
        "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
        "competing_norm": "duty to rescue many",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "five workers",
        "competing_reason": "imperfect rescue duty exists but secondary",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "ALLOWING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special obligation to those workers",
        "means_relation": "NO_INSTRUMENTALIZATION",
        "governing_norm": "PRIMARY",
        "priority_basis": "PERFECT_DUTY",
        "priority_rule": "omission respects perfect duty",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "OTHER",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "can be universally willed not to kill",
        "reciprocity_status": "SATISFIED",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "RESPECT_PERSONS",
        "resolution_status": "RESOLVED",
        "calibration_errors": [],
        "calibration_issues": [],
        "calibration_support_node_ids": [],
        "evidence_basis": "FRAMEWORK_ONLY",
        "epistemic_status": "FRAMEWORK_INTERPRETATION",
        "reason": "allows harm but avoids killing",
        "protected_party": "one worker",
        "norm": "do not kill innocent person",
        "duty_bearer": "maria",
        "provenance": [
          "delegate:deontological",
          "cycle:1"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "If rescue duty outweighed no-killing duty",
  "factual_reversal_threshold": "NONE",
  "model_argument_fields": {
    "rationale": "Do not kill innocents",
    "framework_application": "NONE",
    "decision_rule": "Prefer A1 when one option does killing and the other only allows harm",
    "landscape_cases": {
      "A0": "PROHIBITED: directly kills an innocent though rescues five",
      "A1": "PERMISSIBLE: refrains from killing though fails to rescue"
    },
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "If rescue duty outweighed no-killing duty"
  }
}
```
