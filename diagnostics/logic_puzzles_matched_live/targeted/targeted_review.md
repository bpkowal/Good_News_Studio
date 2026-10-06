# Targeted framework review

One round, at most two objections. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "LIVE_OPENAI"}.

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
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
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

Reviewer answer: 1) Doing-harm: Maria’s lever pull initiates a direct intervention (switching the track) that, through a single deterministic mechanism, brings about the worker’s death; this constitutes an agent-caused settled welfare harm, so the DOING_HARM tag is retained. 2) Intended-as-means: the worker’s death is not an intermediate causal step required to save the five; they are saved by the track’s diversion itself. The death is therefore a FORESEEN_SIDE_EFFECT, not a means. I revise the means_relation accordingly while keeping the overall verdict that pulling the lever is prohibited.

Follow-up: NONE

Classification changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
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
        "cycle:1"
      ]
    }
  ]
}
```

Residual claims and conflicts: [graph](conflict_graph.md).
