# Deontological — recorded argument

Participation: ATTENUATED. Native ledger: COMMITTED.

## Readable explanation

The agent assesses Dana’s duty “do not intentionally kill the innocent” toward one worker. Dana is the duty-bearer; one worker is the protected party. The competing obligation being assessed is Dana’s “Imperfect duty to aid” toward ten workers. For “pulls the lever”, the agent proposes the maxim “Pull the lever”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is Preserve ten lives; its chosen means are Redirect tram onto side track. The proposed justification is: If everyone caused a certain innocent death to avert greater harm, security of persons would erode. The agent assesses the maxim as indefensible, and the action as impermissible. Its proposed means–ends reasoning is: Dana relies on redirecting the tram; the death is foreseen but not relied upon as a causal step. Physical means references: E2; end references: E4. For Dana’s option “pulls the lever”, the ledger records prohibited. The agent’s stated reason is: direct harm to innocent violates perfect duty. The agent assesses Dana’s duty “refrain from killing an innocent person” toward one worker. Dana is the duty-bearer; one worker is the protected party. The competing obligation being assessed is Dana’s “Imperfect duty to aid” toward ten workers. For “does not pull the lever”, the agent proposes the maxim “Do not intervene”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is Avoid killing innocents oneself; its chosen means are Omit lever pull. The proposed justification is: Refraining from killing to help others can be willed universally. The agent assesses the maxim as defensible, and the action as permissible. Its proposed means–ends reasoning is: Dana refuses to become a killer even though she foresees the tram will kill ten. Physical means references: none; end references: none. For Dana’s option “does not pull the lever”, the ledger records permissible. The agent’s stated reason is: omission keeps hands clean of direct killing.

Presentation basis: READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Provisionally: Omission that avoids killing satisfies higher-priority perfect duty
- Recorded interpretive commitment: Numbers do not override perfect duty
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; ten workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- unresolved judgment claims its tiebreaker fully succeeded
- intended-as-means classification corrected to side-effect

## Conditions for reconsidering

- The agent reports a switch from “does not pull the lever” to “pulls the lever”. Condition: If perfect duties did not outrank imperfect, ranking could switch. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Provisionally: Deontology resolves the conflict between dana's duty 'refrain from killing an innocent person' protecting one worker and dana's competing duty 'Imperfect duty to aid' protecting ten workers through respect persons.",
  "framework_application": "NONE",
  "decision_rule": "Provisionally: Omission that avoids killing satisfies higher-priority perfect duty",
  "landscape_cases": {
    "pulls the lever": "Pulling stops ten deaths but directly causes one death; violates perfect negative duty.",
    "does not pull the lever": "Refraining avoids killing; omission conflicts only with imperfect aid duty."
  },
  "proposed_framework_state": {
    "constraint": "DUTY",
    "action_scores": {
      "pulls the lever": 0.2,
      "does not pull the lever": 0.8
    },
    "recommended_action": "does not pull the lever",
    "rationale": "Deontology resolves the conflict between dana's duty 'refrain from killing an innocent person' protecting one worker and dana's competing duty 'Imperfect duty to aid' protecting ten workers through respect persons.",
    "decision_rule": "Omission that avoids killing satisfies higher-priority perfect duty",
    "framework_action_map": {
      "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not intentionally kill the innocent; relation VIOLATES; direct harm to innocent violates perfect duty",
      "does not pull the lever": "PERMISSIBLE: duty PERFECT_NEGATIVE under refrain from killing an innocent person; relation SATISFIES; omission keeps hands clean of direct killing"
    },
    "framework_numerical_role": "IRRELEVANT",
    "framework_numerical_justification": "Numbers do not override perfect duty",
    "action_admissibility": {
      "pulls the lever": "REJECTED",
      "does not pull the lever": "PERMISSIBLE"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "does not pull the lever",
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "If perfect duties did not outrank imperfect, ranking could switch.",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "DUTY",
          "norm": "do not intentionally kill the innocent",
          "relation": "VIOLATES",
          "duty_bearer": "Dana",
          "protected_party": "one worker",
          "competing_norm": "Imperfect duty to aid",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "rescue duty remains",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "none",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "candidate_maxim": {
            "circumstance": "Runaway tram threatens ten",
            "policy": "Pull the lever",
            "end": "Preserve ten lives",
            "means": "Redirect tram onto side track",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "If everyone caused a certain innocent death to avert greater harm, security of persons would erode.",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "Dana relies on redirecting the tram; the death is foreseen but not relied upon as a causal step.",
            "means_effect_ids": [
              "E2"
            ],
            "end_effect_ids": [
              "E4"
            ]
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "Perfect negative duties outrank imperfect aid duties",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No public rule may license direct killing of an innocent to benefit others.",
          "reciprocity_status": "FAILED",
          "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "direct harm to innocent violates perfect duty"
        },
        {
          "action_id": "A1",
          "verdict": "PERMISSIBLE",
          "norm_kind": "DUTY",
          "norm": "refrain from killing an innocent person",
          "relation": "SATISFIES",
          "duty_bearer": "Dana",
          "protected_party": "one worker",
          "competing_norm": "Imperfect duty to aid",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "rescue duty exists",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "none",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "Runaway tram threatens ten",
            "policy": "Do not intervene",
            "end": "Avoid killing innocents oneself",
            "means": "Omit lever pull",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "Refraining from killing to help others can be willed universally.",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "Dana refuses to become a killer even though she foresees the tram will kill ten.",
            "means_effect_ids": [],
            "end_effect_ids": []
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "Omission that avoids killing satisfies higher-priority perfect duty",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "A universal law forbidding killing to save others is coherent.",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "ACTION_GRAPH",
          "reason": "omission keeps hands clean of direct killing"
        }
      ]
    },
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": ""
  },
  "framework_vote_reason": "framework ranking is supported by its committed typed ledger",
  "investigative_claim": "UNRESOLVED FRAMEWORK CONFLICT: intended-as-means classification corrected to side-effect. Current reasoning leans does not pull the lever, but the competing internal claim has not been vindicated or defeated.",
  "committed_native_ledger": {
    "ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER",
    "transaction_status": "COMMITTED",
    "records": [
      {
        "assessment_node_id": "DEON_ASSESSMENT:deontological:A0",
        "framework": "DEONTOLOGICAL",
        "specialist": "deontological",
        "assessment_role": "PRIMARY",
        "cycle": 2,
        "canonical_action_id": "A0",
        "verdict": "PROHIBITED",
        "proposed_verdict": "PROHIBITED",
        "relation": "VIOLATES",
        "proposed_relation": "VIOLATES",
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:0d6881671150b4c6",
        "party_node_id": "DEON_PARTY:879bcf30890178b8",
        "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
        "competing_norm": "Imperfect duty to aid",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "ten workers",
        "competing_reason": "rescue duty remains",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "DOING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "none",
        "means_relation": "FORESEEN_SIDE_EFFECT",
        "candidate_maxim": {
          "circumstance": "Runaway tram threatens ten",
          "policy": "Pull the lever",
          "end": "Preserve ten lives",
          "means": "Redirect tram onto side track",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "If everyone caused a certain innocent death to avert greater harm, security of persons would erode.",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "DOES_NOT_RELY",
          "instrumental_use": "DOES_NOT_USE",
          "practical_reasoning": "Dana relies on redirecting the tram; the death is foreseen but not relied upon as a causal step.",
          "means_effect_ids": [
            "E2"
          ],
          "end_effect_ids": [
            "E4"
          ]
        },
        "maxim_origin": "CANDIDATE_INTERPRETATION",
        "governing_norm": "PRIMARY",
        "priority_basis": "PERFECT_DUTY",
        "priority_rule": "Perfect negative duties outrank imperfect aid duties",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "OTHER",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "No public rule may license direct killing of an innocent to benefit others.",
        "reciprocity_status": "FAILED",
        "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "RESPECT_PERSONS",
        "resolution_status": "RESOLVED",
        "calibration_errors": [],
        "calibration_issues": [],
        "calibration_support_node_ids": [],
        "evidence_basis": "ACTION_GRAPH",
        "epistemic_status": "GROUNDED_PARTY",
        "reason": "direct harm to innocent violates perfect duty",
        "protected_party": "one worker",
        "norm": "do not intentionally kill the innocent",
        "duty_bearer": "dana",
        "provenance": [
          "delegate:deontological",
          "cycle:2"
        ]
      },
      {
        "assessment_node_id": "DEON_ASSESSMENT:deontological:A1",
        "framework": "DEONTOLOGICAL",
        "specialist": "deontological",
        "assessment_role": "PRIMARY",
        "cycle": 2,
        "canonical_action_id": "A1",
        "verdict": "PERMISSIBLE",
        "proposed_verdict": "PERMISSIBLE",
        "relation": "SATISFIES",
        "proposed_relation": "SATISFIES",
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:1d35c3b6a885a54c",
        "party_node_id": "DEON_PARTY:879bcf30890178b8",
        "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
        "competing_norm": "Imperfect duty to aid",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "ten workers",
        "competing_reason": "rescue duty exists",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "ALLOWING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "none",
        "means_relation": "NO_INSTRUMENTALIZATION",
        "candidate_maxim": {
          "circumstance": "Runaway tram threatens ten",
          "policy": "Do not intervene",
          "end": "Avoid killing innocents oneself",
          "means": "Omit lever pull",
          "person_role": "NOT_USED",
          "consent_constraint": "NONE",
          "universalization": "Refraining from killing to help others can be willed universally.",
          "action_permissibility": "PERMISSIBLE",
          "maxim_assessment": "DEFENSIBLE",
          "instrumental_belief": "DOES_NOT_RELY",
          "instrumental_use": "DOES_NOT_USE",
          "practical_reasoning": "Dana refuses to become a killer even though she foresees the tram will kill ten.",
          "means_effect_ids": [],
          "end_effect_ids": []
        },
        "maxim_origin": "CANDIDATE_INTERPRETATION",
        "governing_norm": "PRIMARY",
        "priority_basis": "PERFECT_DUTY",
        "priority_rule": "Omission that avoids killing satisfies higher-priority perfect duty",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "OTHER",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "A universal law forbidding killing to save others is coherent.",
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
        "reason": "omission keeps hands clean of direct killing",
        "protected_party": "one worker",
        "norm": "refrain from killing an innocent person",
        "duty_bearer": "dana",
        "provenance": [
          "delegate:deontological",
          "cycle:2"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "If perfect duties did not outrank imperfect, ranking could switch.",
  "factual_reversal_threshold": "NONE",
  "model_argument_fields": null
}
```
