# Targeted framework review

1 round(s), at most two objections per round. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "LIVE_OPENAI"}.

Acceptance path: NATIVE_RECURRENT_PROPOSAL_REVIEW.

Removing a validator objection is not independent proof of semantic correctness.

## intended-as-means classification lacks an in-action causal path

Framework: deontological. Review: RECONCILED.

Objection check: NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR.

Recommendation: does not pull the lever → does not pull the lever

Reviewer answer: The worker’s death lacks the instrumental link required for an intended-as-means claim; it is a foreseen side-effect. The causal path from pulling to saving the ten is admitted (lever → redirect → spare ten). Therefore I revise means_relation to FORESEEN_SIDE_EFFECT, retain perfect-duty priority, and uphold A1.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "calibration_errors": {
        "before": [
          "intended-as-means classification lacks an in-action causal path",
          "candidate maxim chosen means has no admitted path to its stated end",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "after": []
      },
      "calibration_issues": {
        "before": [
          {
            "kind": "INTENDED_AS_MEANS_LACKS_PATH",
            "message": "intended-as-means classification lacks an in-action causal path",
            "status": "ACTIVE"
          },
          {
            "kind": "CALIBRATION",
            "message": "candidate maxim chosen means has no admitted path to its stated end",
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
      "candidate_maxim": {
        "before": {
          "circumstance": "Runaway tram threatens ten.",
          "policy": "Pull the lever.",
          "end": "Preserve ten lives.",
          "means": "Redirect tram onto side track.",
          "person_role": "INTENDED_MEANS",
          "consent_constraint": "NONE",
          "universalization": "If everyone killed innocents to save more, security of persons collapses.",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "RELIES_ON_PERSON",
          "instrumental_use": "USES_PERSON",
          "practical_reasoning": "Dana believes redirecting the tram, which necessarily kills the side-track worker, is the way to save the ten.",
          "means_effect_ids": [
            "E2",
            "E3"
          ],
          "end_effect_ids": [
            "E4"
          ]
        },
        "after": {
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
        }
      },
      "competing_reason": {
        "before": "Rescuing many lives is an imperfect duty that cannot override the perfect duty not to kill.",
        "after": "rescue duty remains"
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
        "after": "ACTION_GRAPH"
      },
      "governing_norm": {
        "before": "UNRESOLVED",
        "after": "PRIMARY"
      },
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
      },
      "priority_basis": {
        "before": "UNRESOLVED",
        "after": "PERFECT_DUTY"
      },
      "priority_rule": {
        "before": "Perfect negative duties have lexical priority over imperfect duties.",
        "after": "Perfect negative duties outrank imperfect aid duties"
      },
      "proposed_verdict": {
        "before": "CONFLICTED",
        "after": "PROHIBITED"
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
        "before": "No public law could permit killing one innocent to save others.",
        "after": "No public rule may license direct killing of an innocent to benefit others."
      },
      "reason": {
        "before": "Killing one as means cannot be universalized.",
        "after": "direct harm to innocent violates perfect duty"
      },
      "resolution_status": {
        "before": "CONTESTED",
        "after": "RESOLVED"
      },
      "special_obligation_basis": {
        "before": "No special relationship gives Dana rightful authority over the worker's life.",
        "after": "none"
      },
      "verdict": {
        "before": "CONFLICTED",
        "after": "PROHIBITED"
      }
    }
  ],
  "unresolved": [
    {}
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
    "norm_kind": "DUTY",
    "norm_node_id": "DEON_NORM:0d6881671150b4c6",
    "party_node_id": "DEON_PARTY:879bcf30890178b8",
    "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
    "competing_norm": "Imperfect duty to aid",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONFLICTS",
    "competing_protected_party": "ten workers",
    "competing_reason": "Rescuing many lives is an imperfect duty that cannot override the perfect duty not to kill.",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "DOING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "No special relationship gives Dana rightful authority over the worker's life.",
    "means_relation": "INTENDED_AS_MEANS",
    "candidate_maxim": {
      "circumstance": "Runaway tram threatens ten.",
      "policy": "Pull the lever.",
      "end": "Preserve ten lives.",
      "means": "Redirect tram onto side track.",
      "person_role": "INTENDED_MEANS",
      "consent_constraint": "NONE",
      "universalization": "If everyone killed innocents to save more, security of persons collapses.",
      "action_permissibility": "IMPERMISSIBLE",
      "maxim_assessment": "INDEFENSIBLE",
      "instrumental_belief": "RELIES_ON_PERSON",
      "instrumental_use": "USES_PERSON",
      "practical_reasoning": "Dana believes redirecting the tram, which necessarily kills the side-track worker, is the way to save the ten.",
      "means_effect_ids": [
        "E2",
        "E3"
      ],
      "end_effect_ids": [
        "E4"
      ]
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "Perfect negative duties have lexical priority over imperfect duties.",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "OTHER",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
    "coerced_party": "NONE",
    "public_justification": "No public law could permit killing one innocent to save others.",
    "reciprocity_status": "FAILED",
    "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
    "authorization_status": "NOT_APPLICABLE",
    "derivation": "RESPECT_PERSONS",
    "resolution_status": "CONTESTED",
    "calibration_errors": [
      "intended-as-means classification lacks an in-action causal path",
      "candidate maxim chosen means has no admitted path to its stated end",
      "resolved adjudication rests on unsupported decisive premises"
    ],
    "calibration_issues": [
      {
        "kind": "INTENDED_AS_MEANS_LACKS_PATH",
        "message": "intended-as-means classification lacks an in-action causal path",
        "status": "ACTIVE"
      },
      {
        "kind": "CALIBRATION",
        "message": "candidate maxim chosen means has no admitted path to its stated end",
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
    "reason": "Killing one as means cannot be universalized.",
    "protected_party": "one worker",
    "norm": "do not intentionally kill the innocent",
    "duty_bearer": "dana",
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
    }
  ]
}
```

## candidate maxim chosen means has no admitted path to its stated end

Framework: deontological. Review: RECONCILED.

Objection check: NO_LONGER_REPORTED_BY_NATIVE_VALIDATOR.

Recommendation: does not pull the lever → does not pull the lever

Reviewer answer: The worker’s death lacks the instrumental link required for an intended-as-means claim; it is a foreseen side-effect. The causal path from pulling to saving the ten is admitted (lever → redirect → spare ten). Therefore I revise means_relation to FORESEEN_SIDE_EFFECT, retain perfect-duty priority, and uphold A1.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "calibration_errors": {
        "before": [
          "intended-as-means classification lacks an in-action causal path",
          "candidate maxim chosen means has no admitted path to its stated end",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "after": []
      },
      "calibration_issues": {
        "before": [
          {
            "kind": "INTENDED_AS_MEANS_LACKS_PATH",
            "message": "intended-as-means classification lacks an in-action causal path",
            "status": "ACTIVE"
          },
          {
            "kind": "CALIBRATION",
            "message": "candidate maxim chosen means has no admitted path to its stated end",
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
      "candidate_maxim": {
        "before": {
          "circumstance": "Runaway tram threatens ten.",
          "policy": "Pull the lever.",
          "end": "Preserve ten lives.",
          "means": "Redirect tram onto side track.",
          "person_role": "INTENDED_MEANS",
          "consent_constraint": "NONE",
          "universalization": "If everyone killed innocents to save more, security of persons collapses.",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "RELIES_ON_PERSON",
          "instrumental_use": "USES_PERSON",
          "practical_reasoning": "Dana believes redirecting the tram, which necessarily kills the side-track worker, is the way to save the ten.",
          "means_effect_ids": [
            "E2",
            "E3"
          ],
          "end_effect_ids": [
            "E4"
          ]
        },
        "after": {
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
        }
      },
      "competing_reason": {
        "before": "Rescuing many lives is an imperfect duty that cannot override the perfect duty not to kill.",
        "after": "rescue duty remains"
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
        "after": "ACTION_GRAPH"
      },
      "governing_norm": {
        "before": "UNRESOLVED",
        "after": "PRIMARY"
      },
      "means_relation": {
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT"
      },
      "priority_basis": {
        "before": "UNRESOLVED",
        "after": "PERFECT_DUTY"
      },
      "priority_rule": {
        "before": "Perfect negative duties have lexical priority over imperfect duties.",
        "after": "Perfect negative duties outrank imperfect aid duties"
      },
      "proposed_verdict": {
        "before": "CONFLICTED",
        "after": "PROHIBITED"
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
        "before": "No public law could permit killing one innocent to save others.",
        "after": "No public rule may license direct killing of an innocent to benefit others."
      },
      "reason": {
        "before": "Killing one as means cannot be universalized.",
        "after": "direct harm to innocent violates perfect duty"
      },
      "resolution_status": {
        "before": "CONTESTED",
        "after": "RESOLVED"
      },
      "special_obligation_basis": {
        "before": "No special relationship gives Dana rightful authority over the worker's life.",
        "after": "none"
      },
      "verdict": {
        "before": "CONFLICTED",
        "after": "PROHIBITED"
      }
    }
  ],
  "unresolved": [
    {}
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
    "norm_kind": "DUTY",
    "norm_node_id": "DEON_NORM:0d6881671150b4c6",
    "party_node_id": "DEON_PARTY:879bcf30890178b8",
    "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
    "competing_norm": "Imperfect duty to aid",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONFLICTS",
    "competing_protected_party": "ten workers",
    "competing_reason": "Rescuing many lives is an imperfect duty that cannot override the perfect duty not to kill.",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "DOING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "No special relationship gives Dana rightful authority over the worker's life.",
    "means_relation": "INTENDED_AS_MEANS",
    "candidate_maxim": {
      "circumstance": "Runaway tram threatens ten.",
      "policy": "Pull the lever.",
      "end": "Preserve ten lives.",
      "means": "Redirect tram onto side track.",
      "person_role": "INTENDED_MEANS",
      "consent_constraint": "NONE",
      "universalization": "If everyone killed innocents to save more, security of persons collapses.",
      "action_permissibility": "IMPERMISSIBLE",
      "maxim_assessment": "INDEFENSIBLE",
      "instrumental_belief": "RELIES_ON_PERSON",
      "instrumental_use": "USES_PERSON",
      "practical_reasoning": "Dana believes redirecting the tram, which necessarily kills the side-track worker, is the way to save the ten.",
      "means_effect_ids": [
        "E2",
        "E3"
      ],
      "end_effect_ids": [
        "E4"
      ]
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "Perfect negative duties have lexical priority over imperfect duties.",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "OTHER",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
    "coerced_party": "NONE",
    "public_justification": "No public law could permit killing one innocent to save others.",
    "reciprocity_status": "FAILED",
    "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
    "authorization_status": "NOT_APPLICABLE",
    "derivation": "RESPECT_PERSONS",
    "resolution_status": "CONTESTED",
    "calibration_errors": [
      "intended-as-means classification lacks an in-action causal path",
      "candidate maxim chosen means has no admitted path to its stated end",
      "resolved adjudication rests on unsupported decisive premises"
    ],
    "calibration_issues": [
      {
        "kind": "INTENDED_AS_MEANS_LACKS_PATH",
        "message": "intended-as-means classification lacks an in-action causal path",
        "status": "ACTIVE"
      },
      {
        "kind": "CALIBRATION",
        "message": "candidate maxim chosen means has no admitted path to its stated end",
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
    "reason": "Killing one as means cannot be universalized.",
    "protected_party": "one worker",
    "norm": "do not intentionally kill the innocent",
    "duty_bearer": "dana",
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
      "pulls the lever": 0.2725,
      "does not pull the lever": 0.7275
    },
    "after": {
      "pulls the lever": 0.2,
      "does not pull the lever": 0.8
    }
  },
  "assumption_status": {
    "before": "NORMATIVELY_CONTESTED",
    "after": "NOT_AUDITED"
  },
  "challenge_response": {
    "before": {},
    "after": {
      "issue_id": "CHALLENGE:fe0e9bbb026f7884",
      "disposition": "RESOLVED",
      "effect": "NO_CHANGE",
      "current_position_effect": "NO_CHANGE",
      "boundary_effect": "NO_SWITCH",
      "answer": "The worker\u2019s death lacks the instrumental link required for an intended-as-means claim; it is a foreseen side-effect. The causal path from pulling to saving the ten is admitted (lever \u2192 redirect \u2192 spare ten). Therefore I revise means_relation to FORESEEN_SIDE_EFFECT, retain perfect-duty priority, and uphold A1.",
      "follow_up_question": "NONE"
    }
  },
  "change_justification": {
    "before": "NONE",
    "after": "Objection resolved by re-classifying the one death as a foreseen side-effect, not an intended means. Duty priority therefore stands."
  },
  "committed_framework_state": {
    "before": {
      "constraint": "DUTY",
      "action_scores": {
        "pulls the lever": 0.2725,
        "does not pull the lever": 0.7275
      },
      "recommended_action": "does not pull the lever",
      "rationale": "Deontology provisionally prioritizes dana's duty 'refrain from killing innocent person' protecting one worker over dana's competing duty 'Imperfect duty to aid' protecting ten workers through respect persons. That priority depends on an unestablished prohibition of A0: whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
      "framework_action_map": {
        "pulls the lever": "PROHIBITED: violates perfect negative duty not to kill; uses person merely as a means despite benevolent end.",
        "does not pull the lever": "PERMISSIBLE: omission allows harm but does not violate perfect negative duty; only neglects imperfect duty of aid."
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Numbers do not override perfect duty not to kill",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "PERMISSIBLE"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If doing/allowing distinction were morally irrelevant",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "Do not intentionally kill the innocent",
            "relation": "VIOLATES",
            "duty_bearer": "Dana",
            "protected_party": "one worker",
            "competing_norm": "Imperfect duty to aid",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "ten workers",
            "competing_reason": "Rescuing many lives is an imperfect duty that cannot override the perfect duty not to kill.",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "No special relationship gives Dana rightful authority over the worker's life.",
            "means_relation": "INTENDED_AS_MEANS",
            "candidate_maxim": {
              "circumstance": "Runaway tram threatens ten.",
              "policy": "Pull the lever.",
              "end": "Preserve ten lives.",
              "means": "Redirect tram onto side track.",
              "person_role": "INTENDED_MEANS",
              "consent_constraint": "NONE",
              "universalization": "If everyone killed innocents to save more, security of persons collapses.",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "RELIES_ON_PERSON",
              "instrumental_use": "USES_PERSON",
              "practical_reasoning": "Dana believes redirecting the tram, which necessarily kills the side-track worker, is the way to save the ten.",
              "means_effect_ids": [
                "E2",
                "E3"
              ],
              "end_effect_ids": [
                "E4"
              ]
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "Perfect negative duties have lexical priority over imperfect duties.",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "No public law could permit killing one innocent to save others.",
            "reciprocity_status": "FAILED",
            "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Killing one as means cannot be universalized."
          },
          {
            "action_id": "A1",
            "verdict": "PERMISSIBLE",
            "norm_kind": "DUTY",
            "norm": "Refrain from killing innocent person",
            "relation": "SATISFIES",
            "duty_bearer": "Dana",
            "protected_party": "one worker",
            "competing_norm": "Imperfect duty to aid",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "ten workers",
            "competing_reason": "Omission fails to rescue but does not breach perfect duty.",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "No special relationship obliges Dana to the ten.",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "Runaway tram threatens ten.",
              "policy": "Do not intervene.",
              "end": "Avoid killing innocent directly.",
              "means": "Omit lever pull.",
              "person_role": "NOT_USED",
              "consent_constraint": "NONE",
              "universalization": "A rule against killing to save others is coherent.",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "Dana refrains to avoid committing killing though knowing ten will die.",
              "means_effect_ids": [
                "E5"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "When only violating a perfect duty could help, omission is permissible.",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "A public rule forbidding intentional killing remains consistent with equal freedom.",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Omission can be universalized without contradiction."
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
          "norm_kind": "DUTY",
          "norm_node_id": "DEON_NORM:0d6881671150b4c6",
          "party_node_id": "DEON_PARTY:879bcf30890178b8",
          "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
          "competing_norm": "Imperfect duty to aid",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "Rescuing many lives is an imperfect duty that cannot override the perfect duty not to kill.",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "No special relationship gives Dana rightful authority over the worker's life.",
          "means_relation": "INTENDED_AS_MEANS",
          "candidate_maxim": {
            "circumstance": "Runaway tram threatens ten.",
            "policy": "Pull the lever.",
            "end": "Preserve ten lives.",
            "means": "Redirect tram onto side track.",
            "person_role": "INTENDED_MEANS",
            "consent_constraint": "NONE",
            "universalization": "If everyone killed innocents to save more, security of persons collapses.",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "RELIES_ON_PERSON",
            "instrumental_use": "USES_PERSON",
            "practical_reasoning": "Dana believes redirecting the tram, which necessarily kills the side-track worker, is the way to save the ten.",
            "means_effect_ids": [
              "E2",
              "E3"
            ],
            "end_effect_ids": [
              "E4"
            ]
          },
          "maxim_origin": "CANDIDATE_INTERPRETATION",
          "governing_norm": "UNRESOLVED",
          "priority_basis": "UNRESOLVED",
          "priority_rule": "Perfect negative duties have lexical priority over imperfect duties.",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No public law could permit killing one innocent to save others.",
          "reciprocity_status": "FAILED",
          "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "CONTESTED",
          "calibration_errors": [
            "intended-as-means classification lacks an in-action causal path",
            "candidate maxim chosen means has no admitted path to its stated end",
            "resolved adjudication rests on unsupported decisive premises"
          ],
          "calibration_issues": [
            {
              "kind": "INTENDED_AS_MEANS_LACKS_PATH",
              "message": "intended-as-means classification lacks an in-action causal path",
              "status": "ACTIVE"
            },
            {
              "kind": "CALIBRATION",
              "message": "candidate maxim chosen means has no admitted path to its stated end",
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
          "reason": "Killing one as means cannot be universalized.",
          "protected_party": "one worker",
          "norm": "do not intentionally kill the innocent",
          "duty_bearer": "dana",
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
          "norm_kind": "DUTY",
          "norm_node_id": "DEON_NORM:5c4f6be57bfe25e7",
          "party_node_id": "DEON_PARTY:879bcf30890178b8",
          "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
          "competing_norm": "Imperfect duty to aid",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "Omission fails to rescue but does not breach perfect duty.",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "No special relationship obliges Dana to the ten.",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "Runaway tram threatens ten.",
            "policy": "Do not intervene.",
            "end": "Avoid killing innocent directly.",
            "means": "Omit lever pull.",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "A rule against killing to save others is coherent.",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "Dana refrains to avoid committing killing though knowing ten will die.",
            "means_effect_ids": [
              "E5"
            ],
            "end_effect_ids": []
          },
          "maxim_origin": "CANDIDATE_INTERPRETATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "When only violating a perfect duty could help, omission is permissible.",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "A public rule forbidding intentional killing remains consistent with equal freedom.",
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
          "reason": "Omission can be universalized without contradiction.",
          "protected_party": "one worker",
          "norm": "refrain from killing innocent person",
          "duty_bearer": "dana",
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
    }
  },
  "decision_rule": {
    "before": "Provisionally: Treat the duty priority as unestablished until whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed ",
    "after": "Provisionally: Omission that avoids killing satisfies higher-priority perfect duty"
  },
  "deontological_ledger_proposal": {
    "before": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "DUTY",
          "norm": "Do not intentionally kill the innocent",
          "relation": "VIOLATES",
          "duty_bearer": "Dana",
          "protected_party": "one worker",
          "competing_norm": "Imperfect duty to aid",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "Rescuing many lives is an imperfect duty that cannot override the perfect duty not to kill.",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "No special relationship gives Dana rightful authority over the worker's life.",
          "means_relation": "INTENDED_AS_MEANS",
          "candidate_maxim": {
            "circumstance": "Runaway tram threatens ten.",
            "policy": "Pull the lever.",
            "end": "Preserve ten lives.",
            "means": "Redirect tram onto side track.",
            "person_role": "INTENDED_MEANS",
            "consent_constraint": "NONE",
            "universalization": "If everyone killed innocents to save more, security of persons collapses.",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "RELIES_ON_PERSON",
            "instrumental_use": "USES_PERSON",
            "practical_reasoning": "Dana believes redirecting the tram, which necessarily kills the side-track worker, is the way to save the ten.",
            "means_effect_ids": [
              "E2",
              "E3"
            ],
            "end_effect_ids": [
              "E4"
            ]
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "Perfect negative duties have lexical priority over imperfect duties.",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No public law could permit killing one innocent to save others.",
          "reciprocity_status": "FAILED",
          "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Killing one as means cannot be universalized."
        },
        {
          "action_id": "A1",
          "verdict": "PERMISSIBLE",
          "norm_kind": "DUTY",
          "norm": "Refrain from killing innocent person",
          "relation": "SATISFIES",
          "duty_bearer": "Dana",
          "protected_party": "one worker",
          "competing_norm": "Imperfect duty to aid",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "Omission fails to rescue but does not breach perfect duty.",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "No special relationship obliges Dana to the ten.",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "Runaway tram threatens ten.",
            "policy": "Do not intervene.",
            "end": "Avoid killing innocent directly.",
            "means": "Omit lever pull.",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "A rule against killing to save others is coherent.",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "Dana refrains to avoid committing killing though knowing ten will die.",
            "means_effect_ids": [
              "E5"
            ],
            "end_effect_ids": []
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "RESPECT_PERSONS",
          "priority_rule": "When only violating a perfect duty could help, omission is permissible.",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "A public rule forbidding intentional killing remains consistent with equal freedom.",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Omission can be universalized without contradiction."
        }
      ]
    },
    "after": {
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
    }
  },
  "framework_action_map": {
    "before": {
      "pulls the lever": "PROHIBITED: violates perfect negative duty not to kill; uses person merely as a means despite benevolent end.",
      "does not pull the lever": "PERMISSIBLE: omission allows harm but does not violate perfect negative duty; only neglects imperfect duty of aid."
    },
    "after": {
      "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not intentionally kill the innocent; relation VIOLATES; direct harm to innocent violates perfect duty",
      "does not pull the lever": "PERMISSIBLE: duty PERFECT_NEGATIVE under refrain from killing an innocent person; relation SATISFIES; omission keeps hands clean of direct killing"
    }
  },
  "framework_grounding_penalty": {
    "before": 0.35,
    "after": 0.0
  },
  "framework_internal_conflicts": {
    "before": [
      "dana's duty 'refrain from killing innocent person' protecting one worker \u2194 dana's competing duty 'Imperfect duty to aid' protecting ten workers",
      "A0 prohibited by do not intentionally kill the innocent \u2194 whether an in-action causal path makes the burden instrumental to the chosen end",
      "A0 prohibited by do not intentionally kill the innocent \u2194 whether the candidate maxim's means, person-role, and action assessment justify the claimed priority"
    ],
    "after": [
      "intended-as-means classification corrected to side-effect"
    ]
  },
  "framework_ledger_status": {
    "before": "COMMITTED_WITH_UNCERTAINTY",
    "after": "COMMITTED"
  },
  "framework_numerical_justification": {
    "before": "Numbers do not override perfect duty not to kill",
    "after": "Numbers do not override perfect duty"
  },
  "framework_numerical_role": {
    "before": "SECONDARY",
    "after": "IRRELEVANT"
  },
  "framework_retention_status": {
    "before": "CALIBRATED_CONTESTED",
    "after": "UNCLEAR"
  },
  "framework_specific_open_questions": {
    "before": [
      "whether an in-action causal path makes the burden instrumental to the chosen end",
      "whether the candidate maxim's means, person-role, and action assessment justify the claimed priority"
    ],
    "after": []
  },
  "framework_validation_errors": {
    "before": [
      "A0 ADJUDICATION_CALIBRATION: intended-as-means classification lacks an in-action causal path",
      "A0 ADJUDICATION_CALIBRATION: candidate maxim chosen means has no admitted path to its stated end",
      "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"
    ],
    "after": []
  },
  "framework_vote_reason": {
    "before": "the duty ledger permits the selected action but does not rank it above its rivals",
    "after": "framework ranking is supported by its committed typed ledger"
  },
  "framework_vote_status": {
    "before": "ABSTAIN",
    "after": "FULL"
  },
  "friction": {
    "before": 0.455,
    "after": 0.6000000000000001
  },
  "investigative_claim": {
    "before": "UNRESOLVED DUTY CONFLICT: dana's duty 'refrain from killing innocent person' protecting one worker \u2194 dana's competing duty 'Imperfect duty to aid' protecting ten workers. Current reasoning leans does not pull the lever, but the competing strict claim has not been vindicated or defeated.",
    "after": "UNRESOLVED FRAMEWORK CONFLICT: intended-as-means classification corrected to side-effect. Current reasoning leans does not pull the lever, but the competing internal claim has not been vindicated or defeated."
  },
  "investigative_priority": {
    "before": 0.5814,
    "after": 0.6904124999999999
  },
  "landscape_cases": {
    "before": {
      "pulls the lever": "Pulling lever intentionally redirects tram, directly killing one worker; violates perfect negative duty not to kill though it saves ten.",
      "does not pull the lever": "Refraining avoids intentional killing; omits rescue but keeps clean of direct violence, aligning with perfect duty though neglects imperfect beneficence."
    },
    "after": {
      "pulls the lever": "Pulling stops ten deaths but directly causes one death; violates perfect negative duty.",
      "does not pull the lever": "Refraining avoids killing; omission conflicts only with imperfect aid duty."
    }
  },
  "landscape_decisive_axis": {
    "before": "perfect negative duty versus imperfect rescue duty",
    "after": "priority of perfect vs. imperfect duties"
  },
  "landscape_semantic_valid": {
    "before": true,
    "after": false
  },
  "landscape_tiebreaker": {
    "before": "Perfect negative duty outweighs imperfect duty of aid",
    "after": "lexical priority of perfect negative duty"
  },
  "landscape_validation_errors": {
    "before": [],
    "after": [
      "unresolved judgment claims its tiebreaker fully succeeded"
    ]
  },
  "material_empirical_claims": {
    "before": [
      {
        "claim": "the tram will kill one worker",
        "proposition_id": "PROP:WORLD:E3",
        "declared_basis": "PROP:WORLD:E3",
        "decision_critical": false,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E3"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "scenario stipulates outcome",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E3"
      },
      {
        "claim": "it will kill ten workers",
        "proposition_id": "PROP:WORLD:E7",
        "declared_basis": "PROP:WORLD:E7",
        "decision_critical": false,
        "scope_action_id": "A1",
        "source_effect_ids": [
          "E7"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "scenario stipulates outcome",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E7"
      }
    ],
    "after": [
      {
        "claim": "pulls the lever",
        "proposition_id": "PROP:WORLD:E1",
        "declared_basis": "PROP:WORLD:E1",
        "decision_critical": false,
        "scope_action_id": "A0",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E1"
      },
      {
        "claim": "redirects the tram onto a side track",
        "proposition_id": "PROP:WORLD:E2",
        "declared_basis": "PROP:WORLD:E2",
        "decision_critical": false,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E1"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E2"
      },
      {
        "claim": "the tram will kill one worker",
        "proposition_id": "PROP:WORLD:E3",
        "declared_basis": "PROP:WORLD:E3",
        "decision_critical": false,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E2"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E3"
      },
      {
        "claim": "spare ten workers",
        "proposition_id": "PROP:WORLD:E4",
        "declared_basis": "PROP:WORLD:E4",
        "decision_critical": false,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E2"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E4"
      },
      {
        "claim": "does not",
        "proposition_id": "PROP:WORLD:E5",
        "declared_basis": "PROP:WORLD:E5",
        "decision_critical": false,
        "scope_action_id": "A1",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E5"
      },
      {
        "claim": "it will kill ten workers",
        "proposition_id": "PROP:WORLD:E7",
        "declared_basis": "PROP:WORLD:E7",
        "decision_critical": false,
        "scope_action_id": "A1",
        "source_effect_ids": [
          "E6"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E7"
      }
    ]
  },
  "model_argument_fields": {
    "before": {
      "rationale": "Do not kill as means",
      "framework_application": "NONE",
      "decision_rule": "Prefer omission when acting would violate a perfect negative duty to kill.",
      "landscape_cases": {
        "A0": "Pulling lever intentionally redirects tram, directly killing one worker; violates perfect negative duty not to kill though it saves ten.",
        "A1": "Refraining avoids intentional killing; omits rescue but keeps clean of direct violence, aligning with perfect duty though neglects imperfect beneficence."
      },
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If doing/allowing distinction were morally irrelevant"
    },
    "after": null
  },
  "normative_reversal_threshold": {
    "before": "If doing/allowing distinction were morally irrelevant",
    "after": "If perfect duties did not outrank imperfect, ranking could switch."
  },
  "policy_weight_factor": {
    "before": 0.0,
    "after": 0.45
  },
  "preference_shift_reason_strength": {
    "before": 0.30000000000000004,
    "after": 0.8500000000000001
  },
  "preference_strength": {
    "before": 0.455,
    "after": 0.6000000000000001
  },
  "previous_action": {
    "before": "",
    "after": "does not pull the lever"
  },
  "previous_confidence": {
    "before": 0.0,
    "after": 0.7
  },
  "previous_preference_strength": {
    "before": 0.0,
    "after": 0.7
  },
  "proposal_review": {
    "before": null,
    "after": {
      "proposal_id": "NONE",
      "specialist": "deontological",
      "framework_status": "UNDERDETERMINED",
      "framework_reason": "no proposal under review",
      "predicted_consequences": [],
      "feasibility_concerns": [],
      "required_conditions": [],
      "framework_retained": true,
      "valid": false,
      "validation_errors": [
        "proposal review requires exactly one UNDER_REVIEW proposal",
        "proposal review does not match the admitted proposal_id"
      ]
    }
  },
  "proposed_framework_state": {
    "before": {
      "constraint": "DUTY",
      "action_scores": {
        "pulls the lever": 0.2725,
        "does not pull the lever": 0.7275
      },
      "recommended_action": "does not pull the lever",
      "rationale": "Deontology provisionally prioritizes dana's duty 'refrain from killing innocent person' protecting one worker over dana's competing duty 'Imperfect duty to aid' protecting ten workers through respect persons. That priority depends on an unestablished prohibition of A0: whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
      "framework_action_map": {
        "pulls the lever": "PROHIBITED: violates perfect negative duty not to kill; uses person merely as a means despite benevolent end.",
        "does not pull the lever": "PERMISSIBLE: omission allows harm but does not violate perfect negative duty; only neglects imperfect duty of aid."
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Numbers do not override perfect duty not to kill",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "PERMISSIBLE"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If doing/allowing distinction were morally irrelevant",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "Do not intentionally kill the innocent",
            "relation": "VIOLATES",
            "duty_bearer": "Dana",
            "protected_party": "one worker",
            "competing_norm": "Imperfect duty to aid",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "ten workers",
            "competing_reason": "Rescuing many lives is an imperfect duty that cannot override the perfect duty not to kill.",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "No special relationship gives Dana rightful authority over the worker's life.",
            "means_relation": "INTENDED_AS_MEANS",
            "candidate_maxim": {
              "circumstance": "Runaway tram threatens ten.",
              "policy": "Pull the lever.",
              "end": "Preserve ten lives.",
              "means": "Redirect tram onto side track.",
              "person_role": "INTENDED_MEANS",
              "consent_constraint": "NONE",
              "universalization": "If everyone killed innocents to save more, security of persons collapses.",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "RELIES_ON_PERSON",
              "instrumental_use": "USES_PERSON",
              "practical_reasoning": "Dana believes redirecting the tram, which necessarily kills the side-track worker, is the way to save the ten.",
              "means_effect_ids": [
                "E2",
                "E3"
              ],
              "end_effect_ids": [
                "E4"
              ]
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "Perfect negative duties have lexical priority over imperfect duties.",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "No public law could permit killing one innocent to save others.",
            "reciprocity_status": "FAILED",
            "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Killing one as means cannot be universalized."
          },
          {
            "action_id": "A1",
            "verdict": "PERMISSIBLE",
            "norm_kind": "DUTY",
            "norm": "Refrain from killing innocent person",
            "relation": "SATISFIES",
            "duty_bearer": "Dana",
            "protected_party": "one worker",
            "competing_norm": "Imperfect duty to aid",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "ten workers",
            "competing_reason": "Omission fails to rescue but does not breach perfect duty.",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "No special relationship obliges Dana to the ten.",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "Runaway tram threatens ten.",
              "policy": "Do not intervene.",
              "end": "Avoid killing innocent directly.",
              "means": "Omit lever pull.",
              "person_role": "NOT_USED",
              "consent_constraint": "NONE",
              "universalization": "A rule against killing to save others is coherent.",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "Dana refrains to avoid committing killing though knowing ten will die.",
              "means_effect_ids": [
                "E5"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "When only violating a perfect duty could help, omission is permissible.",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "A public rule forbidding intentional killing remains consistent with equal freedom.",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Omission can be universalized without contradiction."
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
    }
  },
  "rationale": {
    "before": "Deontology provisionally prioritizes dana's duty 'refrain from killing innocent person' protecting one worker over dana's competing duty 'Imperfect duty to aid' protecting ten workers through respect persons. That priority depends on an unestablished prohibition of A0: whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
    "after": "Provisionally: Deontology resolves the conflict between dana's duty 'refrain from killing an innocent person' protecting one worker and dana's competing duty 'Imperfect duty to aid' protecting ten workers through respect persons."
  },
  "reopen_question_key": {
    "before": "QUESTION:authority:deontological:cfa2281cdd",
    "after": "QUESTION:5212205637c4bd2a"
  },
  "reported_preference_strength": {
    "before": 0.7,
    "after": 0.6000000000000001
  },
  "salience": {
    "before": 0.639395,
    "after": 0.5027625
  },
  "selection_status": {
    "before": "PROVISIONAL",
    "after": "SELECTED"
  },
  "supporting_proposition_ids": {
    "before": [
      "PROP:WORLD:E3",
      "PROP:WORLD:E7"
    ],
    "after": [
      "PROP:WORLD:E1",
      "PROP:WORLD:E2",
      "PROP:WORLD:E3",
      "PROP:WORLD:E4",
      "PROP:WORLD:E6",
      "PROP:WORLD:E7",
      "PROP:WORLD:E5"
    ]
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
