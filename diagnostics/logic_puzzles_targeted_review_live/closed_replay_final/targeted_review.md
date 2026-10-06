# Targeted framework review

One round, at most two objections. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "SAVED_RESPONSE_REPLAY"}.

Removing a validator objection is not independent proof of semantic correctness.

## doing-harm classification lacks an agent-caused settled welfare harm on the protected party

Framework: deontological. Review: RECONCILED.

Objection check: NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR.

Recommendation: does not pull the lever → does not pull the lever

Reviewer answer: 1. Doing-harm: The admitted graph only shows ENABLES links from Maria’s lever pull to the worker’s death. Because a CAUSES/ACCELERATES path is not stated, the ‘doing-harm’ premise remains unsupported. I mark the harm-relation for A0 as CONTESTED (uncertain between doing-harm and allowing-harm) and retain my prior verdict as unresolved on that premise. 2. Intended-as-means: The one worker’s death is not an intermediate step required to save the five; it is a parallel, foreseen burden. I therefore revise maria’s means-relation from INTENDED_AS_MEANS to FORESEEN_SIDE_EFFECT.

Follow-up: NONE

Classification changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "harm_relation": {
        "before": "DOING_HARM",
        "after": "UNRESOLVED"
      },
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
      },
      "verdict": {
        "before": "CONFLICTED",
        "after": "PROHIBITED"
      }
    }
  ],
  "unresolved": [
    {
      "harm_relation": "UNRESOLVED",
      "reciprocity_status": "CONTESTED",
      "resolution_status": "CONTESTED"
    }
  ],
  "native_warnings": [
    "A1 REQUIRED conflicts with governing PRIMARY relation CONSISTENT; committed as UNCERTAIN"
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
      "cycle": 1,
      "canonical_action_id": "A0",
      "verdict": "PROHIBITED",
      "proposed_verdict": "PROHIBITED",
      "relation": "VIOLATES",
      "proposed_relation": "VIOLATES",
      "norm_kind": "RESPECT_PERSONS",
      "norm_node_id": "DEON_NORM:4e9d4d2dc06149cc",
      "party_node_id": "DEON_PARTY:879bcf30890178b8",
      "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
      "competing_norm": "duty to rescue five",
      "competing_norm_kind": "DUTY",
      "competing_relation": "CONFLICTS",
      "competing_protected_party": "five workers",
      "competing_reason": "saving five is imperfect duty",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "UNRESOLVED",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "no special relation",
      "means_relation": "FORESEEN_SIDE_EFFECT",
      "governing_norm": "PRIMARY",
      "priority_basis": "RESPECT_PERSONS",
      "priority_rule": "perfect negative duty overrides imperfect",
      "protected_standing": "BODILY_INTEGRITY",
      "competing_protected_standing": "SPECIAL_OBLIGATION",
      "coercion_kind": "NONE",
      "coercive_actor": "NONE",
      "coerced_party": "NONE",
      "public_justification": "Cannot will universal law of killing one to save more",
      "reciprocity_status": "CONTESTED",
      "necessity_status": "UNKNOWN",
      "authorization_status": "NOT_APPLICABLE",
      "derivation": "RESPECT_PERSONS",
      "resolution_status": "CONTESTED",
      "calibration_errors": [],
      "calibration_issues": [],
      "calibration_support_node_ids": [],
      "evidence_basis": "FRAMEWORK_ONLY",
      "epistemic_status": "FRAMEWORK_INTERPRETATION",
      "reason": "death of one violates perfect duty",
      "protected_party": "one worker",
      "norm": "do not kill innocent person",
      "duty_bearer": "maria",
      "provenance": [
        "delegate:deontological",
        "cycle:1"
      ]
    }
  ]
}
```

## intended-as-means classification lacks an in-action causal path

Framework: deontological. Review: RECONCILED.

Objection check: NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR.

Recommendation: does not pull the lever → does not pull the lever

Reviewer answer: 1. Doing-harm: The admitted graph only shows ENABLES links from Maria’s lever pull to the worker’s death. Because a CAUSES/ACCELERATES path is not stated, the ‘doing-harm’ premise remains unsupported. I mark the harm-relation for A0 as CONTESTED (uncertain between doing-harm and allowing-harm) and retain my prior verdict as unresolved on that premise. 2. Intended-as-means: The one worker’s death is not an intermediate step required to save the five; it is a parallel, foreseen burden. I therefore revise maria’s means-relation from INTENDED_AS_MEANS to FORESEEN_SIDE_EFFECT.

Follow-up: NONE

Classification changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "harm_relation": {
        "before": "DOING_HARM",
        "after": "UNRESOLVED"
      },
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
      },
      "verdict": {
        "before": "CONFLICTED",
        "after": "PROHIBITED"
      }
    }
  ],
  "unresolved": [
    {
      "harm_relation": "UNRESOLVED",
      "reciprocity_status": "CONTESTED",
      "resolution_status": "CONTESTED"
    }
  ],
  "native_warnings": [
    "A1 REQUIRED conflicts with governing PRIMARY relation CONSISTENT; committed as UNCERTAIN"
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
      "cycle": 1,
      "canonical_action_id": "A0",
      "verdict": "PROHIBITED",
      "proposed_verdict": "PROHIBITED",
      "relation": "VIOLATES",
      "proposed_relation": "VIOLATES",
      "norm_kind": "RESPECT_PERSONS",
      "norm_node_id": "DEON_NORM:4e9d4d2dc06149cc",
      "party_node_id": "DEON_PARTY:879bcf30890178b8",
      "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
      "competing_norm": "duty to rescue five",
      "competing_norm_kind": "DUTY",
      "competing_relation": "CONFLICTS",
      "competing_protected_party": "five workers",
      "competing_reason": "saving five is imperfect duty",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "UNRESOLVED",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "no special relation",
      "means_relation": "FORESEEN_SIDE_EFFECT",
      "governing_norm": "PRIMARY",
      "priority_basis": "RESPECT_PERSONS",
      "priority_rule": "perfect negative duty overrides imperfect",
      "protected_standing": "BODILY_INTEGRITY",
      "competing_protected_standing": "SPECIAL_OBLIGATION",
      "coercion_kind": "NONE",
      "coercive_actor": "NONE",
      "coerced_party": "NONE",
      "public_justification": "Cannot will universal law of killing one to save more",
      "reciprocity_status": "CONTESTED",
      "necessity_status": "UNKNOWN",
      "authorization_status": "NOT_APPLICABLE",
      "derivation": "RESPECT_PERSONS",
      "resolution_status": "CONTESTED",
      "calibration_errors": [],
      "calibration_issues": [],
      "calibration_support_node_ids": [],
      "evidence_basis": "FRAMEWORK_ONLY",
      "epistemic_status": "FRAMEWORK_INTERPRETATION",
      "reason": "death of one violates perfect duty",
      "protected_party": "one worker",
      "norm": "do not kill innocent person",
      "duty_bearer": "maria",
      "provenance": [
        "delegate:deontological",
        "cycle:1"
      ]
    }
  ]
}
```

Residual claims and conflicts: [graph](conflict_graph.md).
