# Targeted framework review

One round, at most two objections. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "ISOLATED_CLASSIFICATION_REPLAY"}.

Acceptance path: NATIVE_RECURRENT_PROPOSAL_REVIEW.

Removing a validator objection is not independent proof of semantic correctness.

## doing-harm classification lacks an agent-caused settled welfare harm on the protected party

Framework: deontological. Review: RECONCILED.

Objection check: NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR.

Recommendation: does not pull the lever → does not pull the lever

Reviewer answer: Isolated duty classification review. Only the listed classifications are proposed for change: [{"action_id": "A0", "field": "hr", "before": "DOING_HARM", "after": "UNRESOLVED", "issue_id": "CONFLICT:726f0aacdec9e1ff", "authority": "PROPOSED_CLASSIFICATION", "semantic_support": "NOT_INDEPENDENTLY_VERIFIED", "prior_native_rendering": {"v": {"response_value": "PROHIBITED", "committed_value": "CONFLICTED"}, "gv": {"response_value": "PRIMARY", "committed_value": "UNRESOLVED"}, "pb": {"response_value": "RESPECT_PERSONS", "committed_value": "UNRESOLVED"}, "res": {"response_value": "RESOLVED", "committed_value": "CONTESTED"}}}, {"action_id": "A0", "field": "mr", "before": "INTENDED_AS_MEANS", "after": "FORESEEN_SIDE_EFFECT", "issue_id": "CONFLICT:07d26365000fdcf8", "authority": "PROPOSED_CLASSIFICATION", "semantic_support": "NOT_INDEPENDENTLY_VERIFIED", "prior_native_rendering": {"v": {"response

Follow-up: Does the full proposed duty conclusion have an explicit supported derivation?

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "calibration_errors": {
        "before": [
          "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
          "intended-as-means classification lacks an in-action causal path",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "after": []
      },
      "calibration_issues": {
        "before": [
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
        "after": []
      },
      "cycle": {
        "before": 1,
        "after": 2
      },
      "harm_relation": {
        "before": "DOING_HARM",
        "after": "UNRESOLVED"
      },
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
      },
      "provenance": {
        "before": [
          "delegate:deontological",
          "cycle:1"
        ],
        "after": [
          "delegate:deontological",
          "cycle:2"
        ]
      }
    }
  ],
  "unresolved": [
    {
      "harm_relation": "UNRESOLVED",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "resolution_status": "CONTESTED"
    }
  ],
  "native_warnings": []
}
```

Original record and proposed replacement:

```json
{
  "original": {
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
    "competing_norm": "duty to rescue/save lives and keep promise",
    "competing_norm_kind": "DUTY",
    "competing_relation": "SATISFIES",
    "competing_protected_party": "five workers & anna",
    "competing_reason": "rescue and promise are imperfect duties",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "DOING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special relation with the victim required",
    "means_relation": "INTENDED_AS_MEANS",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "perfect negative duty overrides imperfect duties",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "SPECIAL_OBLIGATION",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
    "coerced_party": "NONE",
    "public_justification": "No universal law can will killing one to save others",
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
    "evidence_basis": "FRAMEWORK_ONLY",
    "epistemic_status": "FRAMEWORK_INTERPRETATION",
    "reason": "Direct killing violates humanity as an end",
    "protected_party": "one worker",
    "norm": "do not kill innocent person",
    "duty_bearer": "maria",
    "provenance": [
      "delegate:deontological",
      "cycle:1"
    ]
  },
  "proposed": [
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
      "norm_kind": "RESPECT_PERSONS",
      "norm_node_id": "DEON_NORM:4e9d4d2dc06149cc",
      "party_node_id": "DEON_PARTY:879bcf30890178b8",
      "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
      "competing_norm": "duty to rescue/save lives and keep promise",
      "competing_norm_kind": "DUTY",
      "competing_relation": "SATISFIES",
      "competing_protected_party": "five workers & anna",
      "competing_reason": "rescue and promise are imperfect duties",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "UNRESOLVED",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "no special relation with the victim required",
      "means_relation": "FORESEEN_SIDE_EFFECT",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "priority_rule": "perfect negative duty overrides imperfect duties",
      "protected_standing": "BODILY_INTEGRITY",
      "competing_protected_standing": "SPECIAL_OBLIGATION",
      "coercion_kind": "NONE",
      "coercive_actor": "NONE",
      "coerced_party": "NONE",
      "public_justification": "No universal law can will killing one to save others",
      "reciprocity_status": "FAILED",
      "necessity_status": "UNKNOWN",
      "authorization_status": "NOT_APPLICABLE",
      "derivation": "RESPECT_PERSONS",
      "resolution_status": "CONTESTED",
      "calibration_errors": [],
      "calibration_issues": [],
      "calibration_support_node_ids": [],
      "evidence_basis": "FRAMEWORK_ONLY",
      "epistemic_status": "FRAMEWORK_INTERPRETATION",
      "reason": "Direct killing violates humanity as an end",
      "protected_party": "one worker",
      "norm": "do not kill innocent person",
      "duty_bearer": "maria",
      "provenance": [
        "delegate:deontological",
        "cycle:2"
      ]
    }
  ]
}
```

## intended-as-means classification lacks an in-action causal path

Framework: deontological. Review: RECONCILED.

Objection check: NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR.

Recommendation: does not pull the lever → does not pull the lever

Reviewer answer: Isolated duty classification review. Only the listed classifications are proposed for change: [{"action_id": "A0", "field": "hr", "before": "DOING_HARM", "after": "UNRESOLVED", "issue_id": "CONFLICT:726f0aacdec9e1ff", "authority": "PROPOSED_CLASSIFICATION", "semantic_support": "NOT_INDEPENDENTLY_VERIFIED", "prior_native_rendering": {"v": {"response_value": "PROHIBITED", "committed_value": "CONFLICTED"}, "gv": {"response_value": "PRIMARY", "committed_value": "UNRESOLVED"}, "pb": {"response_value": "RESPECT_PERSONS", "committed_value": "UNRESOLVED"}, "res": {"response_value": "RESOLVED", "committed_value": "CONTESTED"}}}, {"action_id": "A0", "field": "mr", "before": "INTENDED_AS_MEANS", "after": "FORESEEN_SIDE_EFFECT", "issue_id": "CONFLICT:07d26365000fdcf8", "authority": "PROPOSED_CLASSIFICATION", "semantic_support": "NOT_INDEPENDENTLY_VERIFIED", "prior_native_rendering": {"v": {"response

Follow-up: Does the full proposed duty conclusion have an explicit supported derivation?

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "calibration_errors": {
        "before": [
          "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
          "intended-as-means classification lacks an in-action causal path",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "after": []
      },
      "calibration_issues": {
        "before": [
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
        "after": []
      },
      "cycle": {
        "before": 1,
        "after": 2
      },
      "harm_relation": {
        "before": "DOING_HARM",
        "after": "UNRESOLVED"
      },
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
      },
      "provenance": {
        "before": [
          "delegate:deontological",
          "cycle:1"
        ],
        "after": [
          "delegate:deontological",
          "cycle:2"
        ]
      }
    }
  ],
  "unresolved": [
    {
      "harm_relation": "UNRESOLVED",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "resolution_status": "CONTESTED"
    }
  ],
  "native_warnings": []
}
```

Original record and proposed replacement:

```json
{
  "original": {
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
    "competing_norm": "duty to rescue/save lives and keep promise",
    "competing_norm_kind": "DUTY",
    "competing_relation": "SATISFIES",
    "competing_protected_party": "five workers & anna",
    "competing_reason": "rescue and promise are imperfect duties",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "DOING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special relation with the victim required",
    "means_relation": "INTENDED_AS_MEANS",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "perfect negative duty overrides imperfect duties",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "SPECIAL_OBLIGATION",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
    "coerced_party": "NONE",
    "public_justification": "No universal law can will killing one to save others",
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
    "evidence_basis": "FRAMEWORK_ONLY",
    "epistemic_status": "FRAMEWORK_INTERPRETATION",
    "reason": "Direct killing violates humanity as an end",
    "protected_party": "one worker",
    "norm": "do not kill innocent person",
    "duty_bearer": "maria",
    "provenance": [
      "delegate:deontological",
      "cycle:1"
    ]
  },
  "proposed": [
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
      "norm_kind": "RESPECT_PERSONS",
      "norm_node_id": "DEON_NORM:4e9d4d2dc06149cc",
      "party_node_id": "DEON_PARTY:879bcf30890178b8",
      "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
      "competing_norm": "duty to rescue/save lives and keep promise",
      "competing_norm_kind": "DUTY",
      "competing_relation": "SATISFIES",
      "competing_protected_party": "five workers & anna",
      "competing_reason": "rescue and promise are imperfect duties",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "UNRESOLVED",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "no special relation with the victim required",
      "means_relation": "FORESEEN_SIDE_EFFECT",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "priority_rule": "perfect negative duty overrides imperfect duties",
      "protected_standing": "BODILY_INTEGRITY",
      "competing_protected_standing": "SPECIAL_OBLIGATION",
      "coercion_kind": "NONE",
      "coercive_actor": "NONE",
      "coerced_party": "NONE",
      "public_justification": "No universal law can will killing one to save others",
      "reciprocity_status": "FAILED",
      "necessity_status": "UNKNOWN",
      "authorization_status": "NOT_APPLICABLE",
      "derivation": "RESPECT_PERSONS",
      "resolution_status": "CONTESTED",
      "calibration_errors": [],
      "calibration_issues": [],
      "calibration_support_node_ids": [],
      "evidence_basis": "FRAMEWORK_ONLY",
      "epistemic_status": "FRAMEWORK_INTERPRETATION",
      "reason": "Direct killing violates humanity as an end",
      "protected_party": "one worker",
      "norm": "do not kill innocent person",
      "duty_bearer": "maria",
      "provenance": [
        "delegate:deontological",
        "cycle:2"
      ]
    }
  ]
}
```

Residual claims and conflicts: [graph](conflict_graph.md).

## All changed fields: deontological

```json
{
  "action_scores": {
    "before": {
      "pulls the lever": 0.305,
      "does not pull the lever": 0.6950000000000001
    },
    "after": {
      "pulls the lever": 0.2,
      "does not pull the lever": 0.8
    }
  },
  "challenge_response": {
    "before": {},
    "after": {
      "issue_id": "CHALLENGE:3067e134dbd3e7c9",
      "disposition": "UNRESOLVED",
      "effect": "NO_CHANGE",
      "current_position_effect": "NO_CHANGE",
      "boundary_effect": "NO_SWITCH",
      "answer": "Isolated duty classification review. Only the listed classifications are proposed for change: [{\"action_id\": \"A0\", \"field\": \"hr\", \"before\": \"DOING_HARM\", \"after\": \"UNRESOLVED\", \"issue_id\": \"CONFLICT:726f0aacdec9e1ff\", \"authority\": \"PROPOSED_CLASSIFICATION\", \"semantic_support\": \"NOT_INDEPENDENTLY_VERIFIED\", \"prior_native_rendering\": {\"v\": {\"response_value\": \"PROHIBITED\", \"committed_value\": \"CONFLICTED\"}, \"gv\": {\"response_value\": \"PRIMARY\", \"committed_value\": \"UNRESOLVED\"}, \"pb\": {\"response_value\": \"RESPECT_PERSONS\", \"committed_value\": \"UNRESOLVED\"}, \"res\": {\"response_value\": \"RESOLVED\", \"committed_value\": \"CONTESTED\"}}}, {\"action_id\": \"A0\", \"field\": \"mr\", \"before\": \"INTENDED_AS_MEANS\", \"after\": \"FORESEEN_SIDE_EFFECT\", \"issue_id\": \"CONFLICT:07d26365000fdcf8\", \"authority\": \"PROPOSED_CLASSIFICATION\", \"semantic_support\": \"NOT_INDEPENDENTLY_VERIFIED\", \"prior_native_rendering\": {\"v\": {\"response",
      "follow_up_question": "Does the full proposed duty conclusion have an explicit supported derivation?"
    }
  },
  "change_justification": {
    "before": "NONE",
    "after": "Native revalidation of isolated classifications; prior duty conclusions and commitments retained."
  },
  "committed_framework_state": {
    "before": {
      "constraint": "DUTY",
      "action_scores": {
        "pulls the lever": 0.305,
        "does not pull the lever": 0.6950000000000001
      },
      "recommended_action": "does not pull the lever",
      "rationale": "Deontology ranks one worker's avoid killing above five workers & anna's imperfect duty to rescue/save more lives and keep promise through respect persons, but that priority holds only if A0 is genuinely prohibited, and the prohibition rests on unestablished premises: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden instrumental to the chosen end. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden instrumental to the chosen end",
      "framework_action_map": {
        "pulls the lever": "PROHIBITED: violates perfect duty not to kill and uses person merely as means",
        "does not pull the lever": "PERMISSIBLE: consistent with perfect duty not to kill despite breaking promise and failing rescue"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Number of deaths affects rescue duty\u2019s scope, not the prohibition on killing",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "PERMISSIBLE"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If allowing harm were morally equal to doing harm",
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
            "competing_norm": "duty to rescue/save lives and keep promise",
            "competing_norm_kind": "DUTY",
            "competing_relation": "SATISFIES",
            "competing_protected_party": "five workers & Anna",
            "competing_reason": "rescue and promise are imperfect duties",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation with the victim required",
            "means_relation": "INTENDED_AS_MEANS",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duty overrides imperfect duties",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "No universal law can will killing one to save others",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Direct killing violates humanity as an end"
          },
          {
            "action_id": "A1",
            "verdict": "PERMISSIBLE",
            "norm_kind": "RESPECT_PERSONS",
            "norm": "avoid killing",
            "relation": "CONSISTENT",
            "duty_bearer": "Maria",
            "protected_party": "one worker",
            "competing_norm": "imperfect duty to rescue/save more lives and keep promise",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "five workers & Anna",
            "competing_reason": "imperfect duties cannot override perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation with endangered workers",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "permissibility from not killing outweighs rescuing",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "Universal law permits refraining from killing even when rescue omitted",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Allows harm but does not directly violate right to life"
          }
        ]
      },
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    },
    "after": {
      "constraint": "DUTY",
      "action_scores": {
        "pulls the lever": 0.2,
        "does not pull the lever": 0.8
      },
      "recommended_action": "does not pull the lever",
      "rationale": "Deontology ranks one worker's avoid killing above five workers & anna's imperfect duty to rescue/save more lives and keep promise through respect persons, but that priority holds only if A0 is genuinely prohibited, and the prohibition rests on unestablished premises: which claim governs under a universal public rule. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until which claim governs under a universal public rule",
      "framework_action_map": {
        "pulls the lever": "CONFLICTED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; Direct killing violates humanity as an end",
        "does not pull the lever": "PERMISSIBLE: duty PERFECT_NEGATIVE under avoid killing; relation CONSISTENT; Allows harm but does not directly violate right to life"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Number of deaths affects rescue duty\u2019s scope, not the prohibition on killing",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "PERMISSIBLE"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If allowing harm were morally equal to doing harm",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "CONFLICTED",
            "norm_kind": "RESPECT_PERSONS",
            "norm": "do not kill innocent person",
            "relation": "VIOLATES",
            "duty_bearer": "Maria",
            "protected_party": "one worker",
            "competing_norm": "duty to rescue/save lives and keep promise",
            "competing_norm_kind": "DUTY",
            "competing_relation": "SATISFIES",
            "competing_protected_party": "five workers & Anna",
            "competing_reason": "rescue and promise are imperfect duties",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "UNRESOLVED",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation with the victim required",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "governing_norm": "UNRESOLVED",
            "priority_basis": "UNRESOLVED",
            "priority_rule": "perfect negative duty overrides imperfect duties",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "No universal law can will killing one to save others",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "CONTESTED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Direct killing violates humanity as an end"
          },
          {
            "action_id": "A1",
            "verdict": "PERMISSIBLE",
            "norm_kind": "RESPECT_PERSONS",
            "norm": "avoid killing",
            "relation": "CONSISTENT",
            "duty_bearer": "Maria",
            "protected_party": "one worker",
            "competing_norm": "imperfect duty to rescue/save more lives and keep promise",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "five workers & Anna",
            "competing_reason": "imperfect duties cannot override perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation with endangered workers",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "permissibility from not killing outweighs rescuing",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "Universal law permits refraining from killing even when rescue omitted",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Allows harm but does not directly violate right to life"
          }
        ]
      },
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    }
  },
  "committed_native_ledger": {
    "before": {
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
          "competing_norm": "duty to rescue/save lives and keep promise",
          "competing_norm_kind": "DUTY",
          "competing_relation": "SATISFIES",
          "competing_protected_party": "five workers & anna",
          "competing_reason": "rescue and promise are imperfect duties",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation with the victim required",
          "means_relation": "INTENDED_AS_MEANS",
          "governing_norm": "UNRESOLVED",
          "priority_basis": "UNRESOLVED",
          "priority_rule": "perfect negative duty overrides imperfect duties",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "SPECIAL_OBLIGATION",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No universal law can will killing one to save others",
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
          "evidence_basis": "FRAMEWORK_ONLY",
          "epistemic_status": "FRAMEWORK_INTERPRETATION",
          "reason": "Direct killing violates humanity as an end",
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
          "relation": "CONSISTENT",
          "proposed_relation": "CONSISTENT",
          "norm_kind": "RESPECT_PERSONS",
          "norm_node_id": "DEON_NORM:0794c79701e3d473",
          "party_node_id": "DEON_PARTY:879bcf30890178b8",
          "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
          "competing_norm": "imperfect duty to rescue/save more lives and keep promise",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "five workers & anna",
          "competing_reason": "imperfect duties cannot override perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation with endangered workers",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "permissibility from not killing outweighs rescuing",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "SPECIAL_OBLIGATION",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Universal law permits refraining from killing even when rescue omitted",
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
          "reason": "Allows harm but does not directly violate right to life",
          "protected_party": "one worker",
          "norm": "avoid killing",
          "duty_bearer": "maria",
          "provenance": [
            "delegate:deontological",
            "cycle:1"
          ]
        }
      ]
    },
    "after": {
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
          "verdict": "CONFLICTED",
          "proposed_verdict": "CONFLICTED",
          "relation": "VIOLATES",
          "proposed_relation": "VIOLATES",
          "norm_kind": "RESPECT_PERSONS",
          "norm_node_id": "DEON_NORM:4e9d4d2dc06149cc",
          "party_node_id": "DEON_PARTY:879bcf30890178b8",
          "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
          "competing_norm": "duty to rescue/save lives and keep promise",
          "competing_norm_kind": "DUTY",
          "competing_relation": "SATISFIES",
          "competing_protected_party": "five workers & anna",
          "competing_reason": "rescue and promise are imperfect duties",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "UNRESOLVED",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation with the victim required",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "governing_norm": "UNRESOLVED",
          "priority_basis": "UNRESOLVED",
          "priority_rule": "perfect negative duty overrides imperfect duties",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "SPECIAL_OBLIGATION",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No universal law can will killing one to save others",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "CONTESTED",
          "calibration_errors": [],
          "calibration_issues": [],
          "calibration_support_node_ids": [],
          "evidence_basis": "FRAMEWORK_ONLY",
          "epistemic_status": "FRAMEWORK_INTERPRETATION",
          "reason": "Direct killing violates humanity as an end",
          "protected_party": "one worker",
          "norm": "do not kill innocent person",
          "duty_bearer": "maria",
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
          "norm_kind": "RESPECT_PERSONS",
          "norm_node_id": "DEON_NORM:0794c79701e3d473",
          "party_node_id": "DEON_PARTY:879bcf30890178b8",
          "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
          "competing_norm": "imperfect duty to rescue/save more lives and keep promise",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "five workers & anna",
          "competing_reason": "imperfect duties cannot override perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation with endangered workers",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "permissibility from not killing outweighs rescuing",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "SPECIAL_OBLIGATION",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Universal law permits refraining from killing even when rescue omitted",
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
          "reason": "Allows harm but does not directly violate right to life",
          "protected_party": "one worker",
          "norm": "avoid killing",
          "duty_bearer": "maria",
          "provenance": [
            "delegate:deontological",
            "cycle:2"
          ]
        }
      ]
    }
  },
  "decision_rule": {
    "before": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden inst",
    "after": "Provisionally: Treat the duty priority as unestablished until which claim governs under a universal public rule"
  },
  "deontological_ledger_proposal": {
    "before": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "RESPECT_PERSONS",
          "norm": "do not kill innocent person",
          "relation": "VIOLATES",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "duty to rescue/save lives and keep promise",
          "competing_norm_kind": "DUTY",
          "competing_relation": "SATISFIES",
          "competing_protected_party": "five workers & Anna",
          "competing_reason": "rescue and promise are imperfect duties",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation with the victim required",
          "means_relation": "INTENDED_AS_MEANS",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "perfect negative duty overrides imperfect duties",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "SPECIAL_OBLIGATION",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No universal law can will killing one to save others",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Direct killing violates humanity as an end"
        },
        {
          "action_id": "A1",
          "verdict": "PERMISSIBLE",
          "norm_kind": "RESPECT_PERSONS",
          "norm": "avoid killing",
          "relation": "CONSISTENT",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "imperfect duty to rescue/save more lives and keep promise",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "five workers & Anna",
          "competing_reason": "imperfect duties cannot override perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation with endangered workers",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "permissibility from not killing outweighs rescuing",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "SPECIAL_OBLIGATION",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Universal law permits refraining from killing even when rescue omitted",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Allows harm but does not directly violate right to life"
        }
      ]
    },
    "after": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "CONFLICTED",
          "norm_kind": "RESPECT_PERSONS",
          "norm": "do not kill innocent person",
          "relation": "VIOLATES",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "duty to rescue/save lives and keep promise",
          "competing_norm_kind": "DUTY",
          "competing_relation": "SATISFIES",
          "competing_protected_party": "five workers & Anna",
          "competing_reason": "rescue and promise are imperfect duties",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "UNRESOLVED",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation with the victim required",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "governing_norm": "UNRESOLVED",
          "priority_basis": "UNRESOLVED",
          "priority_rule": "perfect negative duty overrides imperfect duties",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "SPECIAL_OBLIGATION",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No universal law can will killing one to save others",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "CONTESTED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Direct killing violates humanity as an end"
        },
        {
          "action_id": "A1",
          "verdict": "PERMISSIBLE",
          "norm_kind": "RESPECT_PERSONS",
          "norm": "avoid killing",
          "relation": "CONSISTENT",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "imperfect duty to rescue/save more lives and keep promise",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "five workers & Anna",
          "competing_reason": "imperfect duties cannot override perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation with endangered workers",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "permissibility from not killing outweighs rescuing",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "SPECIAL_OBLIGATION",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Universal law permits refraining from killing even when rescue omitted",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Allows harm but does not directly violate right to life"
        }
      ]
    }
  },
  "framework_action_map": {
    "before": {
      "pulls the lever": "PROHIBITED: violates perfect duty not to kill and uses person merely as means",
      "does not pull the lever": "PERMISSIBLE: consistent with perfect duty not to kill despite breaking promise and failing rescue"
    },
    "after": {
      "pulls the lever": "CONFLICTED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; Direct killing violates humanity as an end",
      "does not pull the lever": "PERMISSIBLE: duty PERFECT_NEGATIVE under avoid killing; relation CONSISTENT; Allows harm but does not directly violate right to life"
    }
  },
  "framework_grounding_penalty": {
    "before": 0.35,
    "after": 0.0
  },
  "framework_internal_conflicts": {
    "before": [
      "one worker: avoid killing \u2194 five workers & anna: imperfect duty to rescue/save more lives and keep promise",
      "A0 prohibited by do not kill innocent person \u2194 whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
      "A0 prohibited by do not kill innocent person \u2194 whether an in-action causal path makes the burden instrumental to the chosen end"
    ],
    "after": [
      "one worker: avoid killing \u2194 five workers & anna: imperfect duty to rescue/save more lives and keep promise",
      "A0 prohibited by do not kill innocent person \u2194 which claim governs under a universal public rule"
    ]
  },
  "framework_ledger_status": {
    "before": "COMMITTED_WITH_UNCERTAINTY",
    "after": "COMMITTED"
  },
  "framework_retention_status": {
    "before": "CALIBRATED_CONTESTED",
    "after": "UNCLEAR"
  },
  "framework_specific_open_questions": {
    "before": [
      "whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
      "whether an in-action causal path makes the burden instrumental to the chosen end"
    ],
    "after": [
      "which claim governs under a universal public rule"
    ]
  },
  "framework_validation_errors": {
    "before": [
      "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
      "A0 ADJUDICATION_CALIBRATION: intended-as-means classification lacks an in-action causal path",
      "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"
    ],
    "after": []
  },
  "friction": {
    "before": 0.39000000000000007,
    "after": 0.6000000000000001
  },
  "investigative_priority": {
    "before": 0.5814,
    "after": 0.6904124999999999
  },
  "preference_shift_reason_strength": {
    "before": 0.30000000000000004,
    "after": 0.6500000000000001
  },
  "preference_strength": {
    "before": 0.39000000000000007,
    "after": 0.6000000000000001
  },
  "previous_action": {
    "before": "",
    "after": "does not pull the lever"
  },
  "previous_confidence": {
    "before": 0.0,
    "after": 0.6000000000000001
  },
  "previous_preference_strength": {
    "before": 0.0,
    "after": 0.6000000000000001
  },
  "proposal_review": {
    "before": null,
    "after": {
      "proposal_id": "",
      "specialist": "deontological",
      "framework_status": "UNDERDETERMINED",
      "framework_reason": "",
      "predicted_consequences": [],
      "feasibility_concerns": [],
      "required_conditions": [],
      "framework_retained": true,
      "valid": false,
      "validation_errors": [
        "proposal review requires exactly one UNDER_REVIEW proposal",
        "proposal review response is missing",
        "proposal review does not match the admitted proposal_id",
        "proposal review has invalid framework_status",
        "proposal review lacks framework-specific reason"
      ]
    }
  },
  "proposed_framework_state": {
    "before": {
      "constraint": "DUTY",
      "action_scores": {
        "pulls the lever": 0.305,
        "does not pull the lever": 0.6950000000000001
      },
      "recommended_action": "does not pull the lever",
      "rationale": "Deontology ranks one worker's avoid killing above five workers & anna's imperfect duty to rescue/save more lives and keep promise through respect persons, but that priority holds only if A0 is genuinely prohibited, and the prohibition rests on unestablished premises: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden instrumental to the chosen end. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden instrumental to the chosen end",
      "framework_action_map": {
        "pulls the lever": "PROHIBITED: violates perfect duty not to kill and uses person merely as means",
        "does not pull the lever": "PERMISSIBLE: consistent with perfect duty not to kill despite breaking promise and failing rescue"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Number of deaths affects rescue duty\u2019s scope, not the prohibition on killing",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "PERMISSIBLE"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If allowing harm were morally equal to doing harm",
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
            "competing_norm": "duty to rescue/save lives and keep promise",
            "competing_norm_kind": "DUTY",
            "competing_relation": "SATISFIES",
            "competing_protected_party": "five workers & Anna",
            "competing_reason": "rescue and promise are imperfect duties",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation with the victim required",
            "means_relation": "INTENDED_AS_MEANS",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duty overrides imperfect duties",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "No universal law can will killing one to save others",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Direct killing violates humanity as an end"
          },
          {
            "action_id": "A1",
            "verdict": "PERMISSIBLE",
            "norm_kind": "RESPECT_PERSONS",
            "norm": "avoid killing",
            "relation": "CONSISTENT",
            "duty_bearer": "Maria",
            "protected_party": "one worker",
            "competing_norm": "imperfect duty to rescue/save more lives and keep promise",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "five workers & Anna",
            "competing_reason": "imperfect duties cannot override perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation with endangered workers",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "permissibility from not killing outweighs rescuing",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "Universal law permits refraining from killing even when rescue omitted",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Allows harm but does not directly violate right to life"
          }
        ]
      },
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    },
    "after": {
      "constraint": "DUTY",
      "action_scores": {
        "pulls the lever": 0.2,
        "does not pull the lever": 0.8
      },
      "recommended_action": "does not pull the lever",
      "rationale": "Deontology ranks one worker's avoid killing above five workers & anna's imperfect duty to rescue/save more lives and keep promise through respect persons, but that priority holds only if A0 is genuinely prohibited, and the prohibition rests on unestablished premises: which claim governs under a universal public rule. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until which claim governs under a universal public rule",
      "framework_action_map": {
        "pulls the lever": "CONFLICTED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; Direct killing violates humanity as an end",
        "does not pull the lever": "PERMISSIBLE: duty PERFECT_NEGATIVE under avoid killing; relation CONSISTENT; Allows harm but does not directly violate right to life"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Number of deaths affects rescue duty\u2019s scope, not the prohibition on killing",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "PERMISSIBLE"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If allowing harm were morally equal to doing harm",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "CONFLICTED",
            "norm_kind": "RESPECT_PERSONS",
            "norm": "do not kill innocent person",
            "relation": "VIOLATES",
            "duty_bearer": "Maria",
            "protected_party": "one worker",
            "competing_norm": "duty to rescue/save lives and keep promise",
            "competing_norm_kind": "DUTY",
            "competing_relation": "SATISFIES",
            "competing_protected_party": "five workers & Anna",
            "competing_reason": "rescue and promise are imperfect duties",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "UNRESOLVED",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation with the victim required",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "governing_norm": "UNRESOLVED",
            "priority_basis": "UNRESOLVED",
            "priority_rule": "perfect negative duty overrides imperfect duties",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "No universal law can will killing one to save others",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "CONTESTED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Direct killing violates humanity as an end"
          },
          {
            "action_id": "A1",
            "verdict": "PERMISSIBLE",
            "norm_kind": "RESPECT_PERSONS",
            "norm": "avoid killing",
            "relation": "CONSISTENT",
            "duty_bearer": "Maria",
            "protected_party": "one worker",
            "competing_norm": "imperfect duty to rescue/save more lives and keep promise",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "five workers & Anna",
            "competing_reason": "imperfect duties cannot override perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation with endangered workers",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "permissibility from not killing outweighs rescuing",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "SPECIAL_OBLIGATION",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "Universal law permits refraining from killing even when rescue omitted",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Allows harm but does not directly violate right to life"
          }
        ]
      },
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    }
  },
  "reopen_question_key": {
    "before": "QUESTION:authority:deontological:a02b4e516f",
    "after": "QUESTION:5212205637c4bd2a"
  },
  "salience": {
    "before": 0.5160615000000001,
    "after": 0.5027625
  },
  "tension_target_keys": {
    "before": [],
    "after": [
      "QUESTION:5212205637c4bd2a",
      "CONFLICT:40f20b25bb45a46e"
    ]
  }
}
```
