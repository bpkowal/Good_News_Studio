# Deontological — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

The agent assesses Maria’s duty “do not intentionally kill” toward one worker. Maria is the duty-bearer; one worker is the protected party. The competing obligation being assessed is Maria’s “imperfect duty to aid” toward five workers. For Maria’s option “pulls the lever”, the ledger records conflicted. The agent’s stated reason is: kills innocent as means. The agent assesses Maria’s duty “refrain from killing innocents” toward one worker. Maria is the duty-bearer; one worker is the protected party. The competing obligation being assessed is Maria’s “imperfect duty to rescue” toward five workers. For Maria’s option “does not”, the ledger records permissible. The agent’s stated reason is: does not kill. The ledger retains unresolved issues; this does not establish that one duty overrides the other.

Presentation basis: READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: prefer “does not” when “pulls the lever” breaches a perfect duty
- Recorded interpretive commitment: Number saved cannot override perfect duty
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; five workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- A0 ADJUDICATION_CALIBRATION: intended-as-means classification lacks an in-action causal path
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- whether an in-action causal path makes the burden instrumental to the chosen end
- one worker: refrain from killing innocents ↔ five workers: imperfect duty to rescue
- A0 prohibited by do not intentionally kill ↔ whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- A0 prohibited by do not intentionally kill ↔ whether an in-action causal path makes the burden instrumental to the chosen end

## Conditions for reconsidering

- The agent reports a switch from “does not” to “pulls the lever”. Condition: If perfect duty could be outweighed. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Deontology ranks one worker's refrain from killing innocents above five workers's imperfect duty to rescue through respect persons, but that priority holds only if A0 is genuinely provisionally favored under incomplete adjudication, and the",
  "framework_application": "NONE",
  "decision_rule": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden inst",
  "landscape_cases": {
    "pulls the lever": "PROHIBITED: intentionally kills an innocent, violating a perfect negative duty not to kill or use a",
    "does not": "PERMISSIBLE: refrains from killing; omission allows harm but does not breach any perfect duty"
  },
  "proposed_framework_state": {
    "constraint": "DUTY",
    "action_scores": {
      "pulls the lever": 0.305,
      "does not": 0.6950000000000001
    },
    "recommended_action": "does not",
    "rationale": "Deontology ranks one worker's refrain from killing innocents above five workers's imperfect duty to rescue through respect persons, but that priority holds only if A0 is genuinely prohibited, and the prohibition rests on unestablished premises: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden instrumental to the chosen end. The comparative Kantian judgment remains contested.",
    "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden instrumental to the chosen end",
    "framework_action_map": {
      "pulls the lever": "PROHIBITED: violates perfect negative duty not to intentionally kill and treats the lone worker merely as a means.",
      "does not": "PERMISSIBLE: omission allows harm but does not violate any perfect negative duty; imperfect rescue duty cannot override."
    },
    "framework_numerical_role": "SECONDARY",
    "framework_numerical_justification": "Number saved cannot override perfect duty",
    "action_admissibility": {
      "pulls the lever": "REJECTED",
      "does not": "PERMISSIBLE"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "does not",
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "If perfect duty could be outweighed",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "RESPECT_PERSONS",
          "norm": "Do not intentionally kill",
          "relation": "VIOLATES",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "imperfect duty to aid",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "five workers",
          "competing_reason": "saving five is only an imperfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relationship",
          "means_relation": "INTENDED_AS_MEANS",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "perfect negative duty prevails",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No public law can license killing one innocent",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "kills innocent as means"
        },
        {
          "action_id": "A1",
          "verdict": "PERMISSIBLE",
          "norm_kind": "DUTY",
          "norm": "Refrain from killing innocents",
          "relation": "CONSISTENT",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "imperfect duty to rescue",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "five workers",
          "competing_reason": "rescue is imperfect",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special obligation to five",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "perfect duty satisfied",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No coercive killing; omission is legally tolerable",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "does not kill"
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
  "investigative_claim": "UNRESOLVED DUTY CONFLICT: one worker: refrain from killing innocents ↔ five workers: imperfect duty to rescue. Current reasoning leans does not, but the competing strict claim has not been vindicated or defeated.",
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
        "norm_node_id": "DEON_NORM:05d52a19b110e908",
        "party_node_id": "DEON_PARTY:879bcf30890178b8",
        "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
        "competing_norm": "imperfect duty to aid",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "five workers",
        "competing_reason": "saving five is only an imperfect duty",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "DOING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special relationship",
        "means_relation": "INTENDED_AS_MEANS",
        "governing_norm": "UNRESOLVED",
        "priority_basis": "UNRESOLVED",
        "priority_rule": "perfect negative duty prevails",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "OTHER",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "No public law can license killing one innocent",
        "reciprocity_status": "FAILED",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "RESPECT_PERSONS",
        "resolution_status": "CONTESTED",
        "calibration_errors": [
          "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
          "intended-as-means classification lacks an in-action causal path",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "calibration_issues": [
          {
            "kind": "HARM_RELATION_GRAPH_MISALIGN",
            "message": "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
            "status": "ACTIVE"
          },
          {
            "kind": "INTENDED_AS_MEANS_LACKS_PATH",
            "message": "intended-as-means classification lacks an in-action causal path",
            "status": "ACTIVE"
          },
          {
            "kind": "CALIBRATION",
            "message": "resolved adjudication rests on unsupported decisive premises",
            "status": "ACTIVE"
          }
        ],
        "calibration_support_node_ids": [],
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "GROUNDED_PARTY",
        "reason": "kills innocent as means",
        "protected_party": "one worker",
        "norm": "do not intentionally kill",
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
        "relation": "CONSISTENT",
        "proposed_relation": "CONSISTENT",
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:32339e170b4286d9",
        "party_node_id": "DEON_PARTY:879bcf30890178b8",
        "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
        "competing_norm": "imperfect duty to rescue",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "five workers",
        "competing_reason": "rescue is imperfect",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "ALLOWING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special obligation to five",
        "means_relation": "NO_INSTRUMENTALIZATION",
        "governing_norm": "PRIMARY",
        "priority_basis": "RESPECT_PERSONS",
        "priority_rule": "perfect duty satisfied",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "OTHER",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "No coercive killing; omission is legally tolerable",
        "reciprocity_status": "SATISFIED",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "RESPECT_PERSONS",
        "resolution_status": "RESOLVED",
        "calibration_errors": [],
        "calibration_issues": [],
        "calibration_support_node_ids": [],
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "GROUNDED_PARTY",
        "reason": "does not kill",
        "protected_party": "one worker",
        "norm": "refrain from killing innocents",
        "duty_bearer": "maria",
        "provenance": [
          "delegate:deontological",
          "cycle:1"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "If perfect duty could be outweighed",
  "factual_reversal_threshold": "NONE",
  "model_argument_fields": {
    "rationale": "Killing violates perfect duty",
    "framework_application": "NONE",
    "decision_rule": "prefer A1 when A0 breaches a perfect duty",
    "landscape_cases": {
      "A0": "PROHIBITED: intentionally kills an innocent, violating a perfect negative duty not to kill or use a ",
      "A1": "PERMISSIBLE: refrains from killing; omission allows harm but does not breach any perfect duty"
    },
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "If perfect duty could be outweighed"
  }
}
```
