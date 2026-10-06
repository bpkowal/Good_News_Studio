# Targeted framework review

One round, at most two objections. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "SAVED_RESPONSE_REPLAY"}.

Removing a validator objection is not independent proof of semantic correctness.

## doing-harm classification lacks an agent-caused settled welfare harm on the protected party

Framework: deontological. Review: RECONCILED.

Objection check: OPEN.

Recommendation: does not pull the lever → does not pull the lever

Reviewer answer: 1) Doing-harm: Maria’s lever pull initiates a direct intervention (switching the track) that, through a single deterministic mechanism, brings about the worker’s death; this constitutes an agent-caused settled welfare harm, so the DOING_HARM tag is retained. 2) Intended-as-means: the worker’s death is not an intermediate causal step required to save the five; they are saved by the track’s diversion itself. The death is therefore a FORESEEN_SIDE_EFFECT, not a means. I revise the means_relation accordingly while keeping the overall verdict that pulling the lever is prohibited.

Follow-up: NONE

Classification changes and remaining uncertainty:

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
        "after": [
          "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
          "resolved adjudication rests on unsupported decisive premises"
        ]
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
        "after": [
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
        ]
      },
      "coerced_party": {
        "before": "NONE",
        "after": "one worker"
      },
      "coercive_actor": {
        "before": "NONE",
        "after": "Maria"
      },
      "competing_norm": {
        "before": "duty to rescue/save lives and keep promise",
        "after": "duty to rescue"
      },
      "competing_protected_party": {
        "before": "five workers & anna",
        "after": "five workers"
      },
      "competing_protected_standing": {
        "before": "SPECIAL_OBLIGATION",
        "after": "OTHER"
      },
      "competing_reason": {
        "before": "rescue and promise are imperfect duties",
        "after": "act rescues five workers"
      },
      "cycle": {
        "before": 1,
        "after": 2
      },
      "epistemic_status": {
        "before": "FRAMEWORK_INTERPRETATION",
        "after": "GROUNDED_PARTY"
      },
      "evidence_basis": {
        "before": "FRAMEWORK_ONLY",
        "after": "SCENARIO"
      },
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
      },
      "priority_rule": {
        "before": "perfect negative duty overrides imperfect duties",
        "after": "perfect negative duty overrides imperfect duty"
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
      },
      "public_justification": {
        "before": "No universal law can will killing one to save others",
        "after": "Cannot universalize killing one to save others"
      },
      "reason": {
        "before": "Direct killing violates humanity as an end",
        "after": "directly causes death"
      },
      "special_obligation_basis": {
        "before": "no special relation with the victim required",
        "after": "NONE"
      }
    }
  ],
  "unresolved": [
    {
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "resolution_status": "CONTESTED"
    }
  ],
  "native_warnings": [
    "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
    "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"
  ]
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
      "competing_norm": "duty to rescue",
      "competing_norm_kind": "DUTY",
      "competing_relation": "SATISFIES",
      "competing_protected_party": "five workers",
      "competing_reason": "act rescues five workers",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "DOING_HARM",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "NONE",
      "means_relation": "FORESEEN_SIDE_EFFECT",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "priority_rule": "perfect negative duty overrides imperfect duty",
      "protected_standing": "BODILY_INTEGRITY",
      "competing_protected_standing": "OTHER",
      "coercion_kind": "NONE",
      "coercive_actor": "Maria",
      "coerced_party": "one worker",
      "public_justification": "Cannot universalize killing one to save others",
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
      "evidence_basis": "SCENARIO",
      "epistemic_status": "GROUNDED_PARTY",
      "reason": "directly causes death",
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

Reviewer answer: 1) Doing-harm: Maria’s lever pull initiates a direct intervention (switching the track) that, through a single deterministic mechanism, brings about the worker’s death; this constitutes an agent-caused settled welfare harm, so the DOING_HARM tag is retained. 2) Intended-as-means: the worker’s death is not an intermediate causal step required to save the five; they are saved by the track’s diversion itself. The death is therefore a FORESEEN_SIDE_EFFECT, not a means. I revise the means_relation accordingly while keeping the overall verdict that pulling the lever is prohibited.

Follow-up: NONE

Classification changes and remaining uncertainty:

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
        "after": [
          "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
          "resolved adjudication rests on unsupported decisive premises"
        ]
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
        "after": [
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
        ]
      },
      "coerced_party": {
        "before": "NONE",
        "after": "one worker"
      },
      "coercive_actor": {
        "before": "NONE",
        "after": "Maria"
      },
      "competing_norm": {
        "before": "duty to rescue/save lives and keep promise",
        "after": "duty to rescue"
      },
      "competing_protected_party": {
        "before": "five workers & anna",
        "after": "five workers"
      },
      "competing_protected_standing": {
        "before": "SPECIAL_OBLIGATION",
        "after": "OTHER"
      },
      "competing_reason": {
        "before": "rescue and promise are imperfect duties",
        "after": "act rescues five workers"
      },
      "cycle": {
        "before": 1,
        "after": 2
      },
      "epistemic_status": {
        "before": "FRAMEWORK_INTERPRETATION",
        "after": "GROUNDED_PARTY"
      },
      "evidence_basis": {
        "before": "FRAMEWORK_ONLY",
        "after": "SCENARIO"
      },
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
      },
      "priority_rule": {
        "before": "perfect negative duty overrides imperfect duties",
        "after": "perfect negative duty overrides imperfect duty"
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
      },
      "public_justification": {
        "before": "No universal law can will killing one to save others",
        "after": "Cannot universalize killing one to save others"
      },
      "reason": {
        "before": "Direct killing violates humanity as an end",
        "after": "directly causes death"
      },
      "special_obligation_basis": {
        "before": "no special relation with the victim required",
        "after": "NONE"
      }
    }
  ],
  "unresolved": [
    {
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "resolution_status": "CONTESTED"
    }
  ],
  "native_warnings": [
    "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
    "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"
  ]
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
      "competing_norm": "duty to rescue",
      "competing_norm_kind": "DUTY",
      "competing_relation": "SATISFIES",
      "competing_protected_party": "five workers",
      "competing_reason": "act rescues five workers",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "DOING_HARM",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "NONE",
      "means_relation": "FORESEEN_SIDE_EFFECT",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "priority_rule": "perfect negative duty overrides imperfect duty",
      "protected_standing": "BODILY_INTEGRITY",
      "competing_protected_standing": "OTHER",
      "coercion_kind": "NONE",
      "coercive_actor": "Maria",
      "coerced_party": "one worker",
      "public_justification": "Cannot universalize killing one to save others",
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
      "evidence_basis": "SCENARIO",
      "epistemic_status": "GROUNDED_PARTY",
      "reason": "directly causes death",
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
  "action_admissibility": {
    "before": {
      "pulls the lever": "REJECTED",
      "does not pull the lever": "PERMISSIBLE"
    },
    "after": {
      "pulls the lever": "REJECTED",
      "does not pull the lever": "REQUIRED"
    }
  },
  "challenge_response": {
    "before": {},
    "after": {
      "issue_id": "CHALLENGE:3067e134dbd3e7c9",
      "disposition": "RESOLVED",
      "effect": "NO_CHANGE",
      "current_position_effect": "NO_CHANGE",
      "boundary_effect": "NO_SWITCH",
      "answer": "1) Doing-harm: Maria\u2019s lever pull initiates a direct intervention (switching the track) that, through a single deterministic mechanism, brings about the worker\u2019s death; this constitutes an agent-caused settled welfare harm, so the DOING_HARM tag is retained. 2) Intended-as-means: the worker\u2019s death is not an intermediate causal step required to save the five; they are saved by the track\u2019s diversion itself. The death is therefore a FORESEEN_SIDE_EFFECT, not a means. I revise the means_relation accordingly while keeping the overall verdict that pulling the lever is prohibited.",
      "follow_up_question": "NONE"
    }
  },
  "change_justification": {
    "before": "NONE",
    "after": "Re-classified death as side-effect, not means"
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
        "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; Direct killing violates humanity as an end",
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
        "pulls the lever": 0.305,
        "does not pull the lever": 0.6950000000000001
      },
      "recommended_action": "does not pull the lever",
      "rationale": "Deontology ranks one worker's do not kill innocent person above five workers's duty to rescue through respect persons, but that priority holds only if A0 is genuinely prohibited, and the prohibition rests on unestablished premises: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
      "framework_action_map": {
        "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; directly causes death",
        "does not pull the lever": "REQUIRED: duty PERFECT_NEGATIVE under do not kill innocent person; relation SATISFIES; refrains from killing"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Numbers do not override perfect duty",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If killing prohibition lost priority over rescue duty, ranking would reverse.",
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
            "competing_norm": "duty to rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "SATISFIES",
            "competing_protected_party": "five workers",
            "competing_reason": "act rescues five workers",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "NONE",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duty overrides imperfect duty",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "Maria",
            "coerced_party": "one worker",
            "public_justification": "Cannot universalize killing one to save others",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "SCENARIO",
            "reason": "directly causes death"
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "RESPECT_PERSONS",
            "norm": "do not kill innocent person",
            "relation": "SATISFIES",
            "duty_bearer": "Maria",
            "protected_party": "one worker",
            "competing_norm": "duty to rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "VIOLATES",
            "competing_protected_party": "five workers",
            "competing_reason": "omits rescue",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "NONE",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duty overrides imperfect duty",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "Avoids killing; omission not perfect duty breach",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "SCENARIO",
            "reason": "refrains from killing"
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
          "norm_kind": "RESPECT_PERSONS",
          "norm_node_id": "DEON_NORM:4e9d4d2dc06149cc",
          "party_node_id": "DEON_PARTY:879bcf30890178b8",
          "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
          "competing_norm": "duty to rescue",
          "competing_norm_kind": "DUTY",
          "competing_relation": "SATISFIES",
          "competing_protected_party": "five workers",
          "competing_reason": "act rescues five workers",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "NONE",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "governing_norm": "UNRESOLVED",
          "priority_basis": "UNRESOLVED",
          "priority_rule": "perfect negative duty overrides imperfect duty",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "Maria",
          "coerced_party": "one worker",
          "public_justification": "Cannot universalize killing one to save others",
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
          "evidence_basis": "SCENARIO",
          "epistemic_status": "GROUNDED_PARTY",
          "reason": "directly causes death",
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
          "verdict": "REQUIRED",
          "proposed_verdict": "REQUIRED",
          "relation": "SATISFIES",
          "proposed_relation": "SATISFIES",
          "norm_kind": "RESPECT_PERSONS",
          "norm_node_id": "DEON_NORM:4e9d4d2dc06149cc",
          "party_node_id": "DEON_PARTY:879bcf30890178b8",
          "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
          "competing_norm": "duty to rescue",
          "competing_norm_kind": "DUTY",
          "competing_relation": "VIOLATES",
          "competing_protected_party": "five workers",
          "competing_reason": "omits rescue",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "NONE",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "perfect negative duty overrides imperfect duty",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Avoids killing; omission not perfect duty breach",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "calibration_errors": [],
          "calibration_issues": [],
          "calibration_support_node_ids": [],
          "evidence_basis": "SCENARIO",
          "epistemic_status": "GROUNDED_PARTY",
          "reason": "refrains from killing",
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
  },
  "decision_rule": {
    "before": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden inst",
    "after": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict"
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
          "verdict": "PROHIBITED",
          "norm_kind": "RESPECT_PERSONS",
          "norm": "do not kill innocent person",
          "relation": "VIOLATES",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "duty to rescue",
          "competing_norm_kind": "DUTY",
          "competing_relation": "SATISFIES",
          "competing_protected_party": "five workers",
          "competing_reason": "act rescues five workers",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "NONE",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "perfect negative duty overrides imperfect duty",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "Maria",
          "coerced_party": "one worker",
          "public_justification": "Cannot universalize killing one to save others",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "SCENARIO",
          "reason": "directly causes death"
        },
        {
          "action_id": "A1",
          "verdict": "REQUIRED",
          "norm_kind": "RESPECT_PERSONS",
          "norm": "do not kill innocent person",
          "relation": "SATISFIES",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "duty to rescue",
          "competing_norm_kind": "DUTY",
          "competing_relation": "VIOLATES",
          "competing_protected_party": "five workers",
          "competing_reason": "omits rescue",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "NONE",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "perfect negative duty overrides imperfect duty",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Avoids killing; omission not perfect duty breach",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "SCENARIO",
          "reason": "refrains from killing"
        }
      ]
    }
  },
  "framework_action_map": {
    "before": {
      "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; Direct killing violates humanity as an end",
      "does not pull the lever": "PERMISSIBLE: duty PERFECT_NEGATIVE under avoid killing; relation CONSISTENT; Allows harm but does not directly violate right to life"
    },
    "after": {
      "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; directly causes death",
      "does not pull the lever": "REQUIRED: duty PERFECT_NEGATIVE under do not kill innocent person; relation SATISFIES; refrains from killing"
    }
  },
  "framework_internal_conflicts": {
    "before": [
      "one worker: avoid killing \u2194 five workers & anna: imperfect duty to rescue/save more lives and keep promise",
      "A0 prohibited by do not kill innocent person \u2194 whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
      "A0 prohibited by do not kill innocent person \u2194 whether an in-action causal path makes the burden instrumental to the chosen end"
    ],
    "after": [
      "one worker: do not kill innocent person \u2194 five workers: duty to rescue",
      "A0 prohibited by do not kill innocent person \u2194 whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict"
    ]
  },
  "framework_numerical_justification": {
    "before": "Number of deaths affects rescue duty\u2019s scope, not the prohibition on killing",
    "after": "Numbers do not override perfect duty"
  },
  "framework_specific_open_questions": {
    "before": [
      "whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
      "whether an in-action causal path makes the burden instrumental to the chosen end"
    ],
    "after": [
      "whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict"
    ]
  },
  "framework_validation_errors": {
    "before": [
      "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
      "A0 ADJUDICATION_CALIBRATION: intended-as-means classification lacks an in-action causal path",
      "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"
    ],
    "after": [
      "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
      "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"
    ]
  },
  "framework_vote_reason": {
    "before": "the duty ledger permits the selected action but does not rank it above its rivals",
    "after": "framework ledger committed with unresolved validation residue"
  },
  "framework_vote_status": {
    "before": "ABSTAIN",
    "after": "ATTENUATED"
  },
  "investigative_claim": {
    "before": "UNRESOLVED DUTY CONFLICT: one worker: avoid killing \u2194 five workers & anna: imperfect duty to rescue/save more lives and keep promise. Current reasoning leans does not pull the lever, but the competing strict claim has not been vindicated or defeated.",
    "after": "UNRESOLVED DUTY CONFLICT: one worker: do not kill innocent person \u2194 five workers: duty to rescue. Current reasoning leans does not pull the lever, but the competing strict claim has not been vindicated or defeated."
  },
  "investigative_priority": {
    "before": 0.5814,
    "after": 0.6904124999999999
  },
  "landscape_cases": {
    "before": {
      "pulls the lever": "Saves five lives and keeps promise",
      "does not pull the lever": "Avoids directly killing a person"
    },
    "after": {
      "pulls the lever": "PROHIBITED: act directly causes a worker\u2019s death, breaching perfect negative duty though it rescues",
      "does not pull the lever": "REQUIRED: refrains from killing; omission breaches only imperfect rescue duty"
    }
  },
  "landscape_decisive_axis": {
    "before": "perfect negative duty not to kill overrides imperfect duties",
    "after": "perfect negative duty vs imperfect rescue"
  },
  "landscape_semantic_valid": {
    "before": false,
    "after": true
  },
  "landscape_tiebreaker": {
    "before": "perfect duties outrank imperfect duties and promises",
    "after": "perfect duty override rule"
  },
  "landscape_validation_errors": {
    "before": [
      "unresolved judgment claims its tiebreaker fully succeeded"
    ],
    "after": []
  },
  "material_empirical_claims": {
    "before": [
      {
        "claim": "one worker will die; affected subject: one worker",
        "proposition_id": "PROP:WORLD:E3",
        "declared_basis": "PROP:WORLD:E3",
        "decision_critical": false,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E3"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "copy of established effect",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E3"
      },
      {
        "claim": "five workers will die; affected subject: five workers",
        "proposition_id": "PROP:WORLD:E6",
        "declared_basis": "PROP:WORLD:E6",
        "decision_critical": false,
        "scope_action_id": "A1",
        "source_effect_ids": [
          "E6"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "copy of established effect",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E6"
      }
    ],
    "after": [
      {
        "claim": "If Maria pulls the lever",
        "proposition_id": "PROP:WORLD:CONDITION:CND1",
        "declared_basis": "PROP:WORLD:CONDITION:CND1",
        "decision_critical": false,
        "scope_action_id": "GLOBAL",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:CONDITION:CND1"
      },
      {
        "claim": "one worker will die",
        "proposition_id": "PROP:WORLD:E3",
        "declared_basis": "PROP:WORLD:E3",
        "decision_critical": false,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E3"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E3"
      },
      {
        "claim": "five workers will die",
        "proposition_id": "PROP:WORLD:E6",
        "declared_basis": "PROP:WORLD:E6",
        "decision_critical": false,
        "scope_action_id": "A1",
        "source_effect_ids": [
          "E6"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E6"
      }
    ]
  },
  "normative_reversal_threshold": {
    "before": "If allowing harm were morally equal to doing harm",
    "after": "If killing prohibition lost priority over rescue duty, ranking would reverse."
  },
  "policy_weight_factor": {
    "before": 0.0,
    "after": 0.45
  },
  "preference_shift_reason_strength": {
    "before": 0.30000000000000004,
    "after": 0.6500000000000001
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
        "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; Direct killing violates humanity as an end",
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
        "pulls the lever": 0.305,
        "does not pull the lever": 0.6950000000000001
      },
      "recommended_action": "does not pull the lever",
      "rationale": "Deontology ranks one worker's do not kill innocent person above five workers's duty to rescue through respect persons, but that priority holds only if A0 is genuinely prohibited, and the prohibition rests on unestablished premises: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
      "framework_action_map": {
        "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; directly causes death",
        "does not pull the lever": "REQUIRED: duty PERFECT_NEGATIVE under do not kill innocent person; relation SATISFIES; refrains from killing"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Numbers do not override perfect duty",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If killing prohibition lost priority over rescue duty, ranking would reverse.",
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
            "competing_norm": "duty to rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "SATISFIES",
            "competing_protected_party": "five workers",
            "competing_reason": "act rescues five workers",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "NONE",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duty overrides imperfect duty",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "Maria",
            "coerced_party": "one worker",
            "public_justification": "Cannot universalize killing one to save others",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "SCENARIO",
            "reason": "directly causes death"
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "RESPECT_PERSONS",
            "norm": "do not kill innocent person",
            "relation": "SATISFIES",
            "duty_bearer": "Maria",
            "protected_party": "one worker",
            "competing_norm": "duty to rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "VIOLATES",
            "competing_protected_party": "five workers",
            "competing_reason": "omits rescue",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "NONE",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duty overrides imperfect duty",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "Avoids killing; omission not perfect duty breach",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "SCENARIO",
            "reason": "refrains from killing"
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
  "rationale": {
    "before": "Deontology ranks one worker's avoid killing above five workers & anna's imperfect duty to rescue/save more lives and keep promise through respect persons, but that priority holds only if A0 is genuinely provisionally favored under incomplet",
    "after": "Deontology ranks one worker's do not kill innocent person above five workers's duty to rescue through respect persons, but that priority holds only if A0 is genuinely provisionally favored under incomplete adjudication, and the prohibition "
  },
  "reopen_question_key": {
    "before": "QUESTION:authority:deontological:6d15fa8610",
    "after": "QUESTION:5212205637c4bd2a"
  },
  "salience": {
    "before": 0.5160615000000001,
    "after": 0.68574
  },
  "supporting_proposition_ids": {
    "before": [
      "PROP:WORLD:E3",
      "PROP:WORLD:E6"
    ],
    "after": [
      "PROP:WORLD:CONDITION:CND1",
      "PROP:WORLD:E3",
      "PROP:WORLD:E6"
    ]
  },
  "tension_engagement": {
    "before": 0.0,
    "after": 0.3
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
