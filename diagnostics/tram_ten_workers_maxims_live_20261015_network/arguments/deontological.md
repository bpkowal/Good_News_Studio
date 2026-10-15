# Deontological — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

The agent assesses Dana’s duty “do not kill an innocent person” toward side-track worker. Dana is the duty-bearer; side-track worker is the protected party. The competing obligation being assessed is Dana’s “imperfect duty to rescue” toward ten workers. For “pulls the lever”, the agent proposes the maxim “Pull lever to divert tram”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is Save ten; its chosen means are Tram strikes side-track worker. The proposed justification is: Cannot will rule that permits killing innocents as side-effect rescue. The agent assesses the maxim as indefensible, and the action as impermissible. For Dana’s option “pulls the lever”, the ledger records conflicted. The agent’s stated reason is: Intervention causes death of protected party.. The agent assesses Dana’s duty “do not kill an innocent person” toward all workers. Dana is the duty-bearer; all workers is the protected party. The competing obligation being assessed is Dana’s “imperfect duty to rescue” toward ten workers. For “does not pull the lever”, the agent proposes the maxim “Refrain from diverting tram”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is Avoid killing; its chosen means are Take no action affecting tram path. The proposed justification is: Can will omission when proposed action is killing. The agent assesses the maxim as defensible, and the action as permissible. For Dana’s option “does not pull the lever”, the ledger records permissible. The agent’s stated reason is: No act that kills; harm permitted is not agent-caused. The ledger retains unresolved issues; this does not establish that one duty overrides the other.

Presentation basis: READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- Recorded interpretive commitment: Headcount does not override perfect duty
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; ten workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- dana's duty 'do not kill an innocent person' protecting all workers ↔ dana's competing duty 'imperfect duty to rescue' protecting ten workers
- A0 prohibited by do not kill an innocent person ↔ whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict

## Conditions for reconsidering

- The agent reports a switch from “does not pull the lever” to “pulls the lever”. Condition: If imperfect rescue could override perfect negative, preference would switch. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Deontology provisionally prioritizes dana's duty 'do not kill an innocent person' protecting all workers over dana's competing duty 'imperfect duty to rescue' protecting ten workers through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict. The comparative Kantian judgment remains contested.",
  "framework_application": "NONE",
  "decision_rule": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
  "landscape_cases": {
    "pulls the lever": "PROHIBITED: act intervenes to kill an innocent, violating the perfect negative duty not to kill; rescue benefit is only an imperfect duty and cannot override.",
    "does not pull the lever": "PERMISSIBLE: omission avoids directly killing anyone; while ten die, that harm is allowed rather than done, so the perfect duty is not violated though the imperfect rescue duty remains unsatisfied."
  },
  "proposed_framework_state": {
    "constraint": "DUTY",
    "action_scores": {
      "pulls the lever": 0.24,
      "does not pull the lever": 0.76
    },
    "recommended_action": "does not pull the lever",
    "rationale": "Deontology provisionally prioritizes dana's duty 'do not kill an innocent person' protecting all workers over dana's competing duty 'imperfect duty to rescue' protecting ten workers through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict. The comparative Kantian judgment remains contested.",
    "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
    "framework_action_map": {
      "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill an innocent person; relation VIOLATES; Intervention causes death of protected party.",
      "does not pull the lever": "PERMISSIBLE: duty PERFECT_NEGATIVE under do not kill an innocent person; relation CONSISTENT; No act that kills; harm permitted is not agent-caused"
    },
    "framework_numerical_role": "IRRELEVANT",
    "framework_numerical_justification": "Headcount does not override perfect duty",
    "action_admissibility": {
      "pulls the lever": "REJECTED",
      "does not pull the lever": "PERMISSIBLE"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "does not pull the lever",
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "If imperfect rescue could override perfect negative, preference would switch",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "DUTY",
          "norm": "do not kill an innocent person",
          "relation": "VIOLATES",
          "duty_bearer": "Dana",
          "protected_party": "side-track worker",
          "competing_norm": "imperfect duty to rescue",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "Rescue duty opposes but is lexically weaker",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation needed",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "candidate_maxim": {
            "circumstance": "Runaway tram endangers workers",
            "policy": "Pull lever to divert tram",
            "end": "Save ten",
            "means": "Tram strikes side-track worker",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "Cannot will rule that permits killing innocents as side-effect rescue",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE"
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "perfect negative outweighs imperfect positive",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "Dana",
          "coerced_party": "NONE",
          "public_justification": "No universal law could will intentional killing of an innocent",
          "reciprocity_status": "FAILED",
          "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Intervention causes death of protected party."
        },
        {
          "action_id": "A1",
          "verdict": "PERMISSIBLE",
          "norm_kind": "DUTY",
          "norm": "do not kill an innocent person",
          "relation": "CONSISTENT",
          "duty_bearer": "Dana",
          "protected_party": "all workers",
          "competing_norm": "imperfect duty to rescue",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "Omission fails rescue but avoids killing",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "Runaway tram endangers workers",
            "policy": "Refrain from diverting tram",
            "end": "Avoid killing",
            "means": "Take no action affecting tram path",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "Can will omission when proposed action is killing",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE"
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "Avoiding direct killing prevails",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "Dana",
          "coerced_party": "NONE",
          "public_justification": "Universal permission to refrain from killing remains coherent",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "No act that kills; harm permitted is not agent-caused"
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
  "investigative_claim": "UNRESOLVED DUTY CONFLICT: dana's duty 'do not kill an innocent person' protecting all workers ↔ dana's competing duty 'imperfect duty to rescue' protecting ten workers. Current reasoning leans does not pull the lever, but the competing strict claim has not been vindicated or defeated.",
  "committed_native_ledger": {
    "ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER",
    "transaction_status": "COMMITTED_WITH_UNCERTAINTY",
    "records": [
      {
        "assessment_node_id": "DEON_ASSESSMENT:deontological:A0",
        "framework": "DEONTOLOGICAL",
        "specialist": "deontological",
        "assessment_role": "PRIMARY",
        "cycle": 2,
        "canonical_action_id": "A0",
        "verdict": "CONFLICTED",
        "proposed_verdict": "CONFLICTED",
        "relation": "VIOLATES",
        "proposed_relation": "VIOLATES",
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:cb883d685732cde9",
        "party_node_id": "DEON_PARTY:6ea3692afd533bef",
        "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
        "competing_norm": "imperfect duty to rescue",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "ten workers",
        "competing_reason": "Rescue duty opposes but is lexically weaker",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "DOING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special relation needed",
        "means_relation": "FORESEEN_SIDE_EFFECT",
        "candidate_maxim": {
          "circumstance": "Runaway tram endangers workers",
          "policy": "Pull lever to divert tram",
          "end": "Save ten",
          "means": "Tram strikes side-track worker",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "Cannot will rule that permits killing innocents as side-effect rescue",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE"
        },
        "maxim_origin": "CANDIDATE_INTERPRETATION",
        "governing_norm": "UNRESOLVED",
        "priority_basis": "UNRESOLVED",
        "priority_rule": "perfect negative outweighs imperfect positive",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "OTHER",
        "coercion_kind": "NONE",
        "coercive_actor": "Dana",
        "coerced_party": "NONE",
        "public_justification": "No universal law could will intentional killing of an innocent",
        "reciprocity_status": "FAILED",
        "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
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
        "reason": "Intervention causes death of protected party.",
        "protected_party": "side-track worker",
        "norm": "do not kill an innocent person",
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
        "relation": "CONSISTENT",
        "proposed_relation": "CONSISTENT",
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:cb883d685732cde9",
        "party_node_id": "DEON_PARTY:adabb06acfe5ddc4",
        "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
        "competing_norm": "imperfect duty to rescue",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "ten workers",
        "competing_reason": "Omission fails rescue but avoids killing",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "ALLOWING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special relation",
        "means_relation": "NO_INSTRUMENTALIZATION",
        "candidate_maxim": {
          "circumstance": "Runaway tram endangers workers",
          "policy": "Refrain from diverting tram",
          "end": "Avoid killing",
          "means": "Take no action affecting tram path",
          "person_role": "NOT_USED",
          "consent_constraint": "NONE",
          "universalization": "Can will omission when proposed action is killing",
          "action_permissibility": "PERMISSIBLE",
          "maxim_assessment": "DEFENSIBLE"
        },
        "maxim_origin": "CANDIDATE_INTERPRETATION",
        "governing_norm": "PRIMARY",
        "priority_basis": "PERFECT_DUTY",
        "priority_rule": "Avoiding direct killing prevails",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "OTHER",
        "coercion_kind": "NONE",
        "coercive_actor": "Dana",
        "coerced_party": "NONE",
        "public_justification": "Universal permission to refrain from killing remains coherent",
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
        "reason": "No act that kills; harm permitted is not agent-caused",
        "protected_party": "all workers",
        "norm": "do not kill an innocent person",
        "duty_bearer": "dana",
        "provenance": [
          "delegate:deontological",
          "cycle:2"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "If imperfect rescue could override perfect negative, preference would switch",
  "factual_reversal_threshold": "NONE",
  "model_argument_fields": null
}
```
