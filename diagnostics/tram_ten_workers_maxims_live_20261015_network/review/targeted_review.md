# Targeted framework review

2 round(s), at most two objections per round. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "LIVE_OPENAI", "rawlsian": "LIVE_OPENAI"}.

Acceptance path: NATIVE_RECURRENT_PROPOSAL_REVIEW.

Removing a validator objection is not independent proof of semantic correctness.

## doing-harm classification lacks an agent-caused settled welfare harm on the protected party

Framework: deontological. Review: RECONCILED.

Objection check: OPEN.

Recommendation: does not pull the lever → does not pull the lever

Reviewer answer: (1) Doing-harm classification defended: pulling the lever is an intentional intervention that sets in motion a causal chain ending in the side-track worker’s death, so Dana is the agent cause of a settled welfare loss. (2) Decisive-premise objection addressed: the exact step is that the chosen means (redirecting tram) foreseeably imposes lethal harm on an innocent, violating the perfect negative duty; the rescue of ten is governed by an imperfect positive duty. Perfect negative duties, by framework commitment already on record, have lexical priority, therefore the verdict stands. No missing factual link was supplied; priority rests purely on duty kind.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "candidate_maxim": {
        "before": {
          "circumstance": "Runaway tram endangers workers",
          "policy": "Pull lever to divert onto occupied side track",
          "end": "Save ten workers",
          "means": "Tram strikes side-track worker",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "Could not will being killed for others as universal law",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE"
        },
        "after": {
          "circumstance": "Runaway tram endangers workers",
          "policy": "Pull lever to divert tram",
          "end": "Save ten",
          "means": "Tram strikes side-track worker",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "Cannot will rule that permits killing innocents as side-effect rescue",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE"
        }
      },
      "coercive_actor": {
        "before": "NONE",
        "after": "Dana"
      },
      "competing_norm": {
        "before": "imperfect duty to rescue others",
        "after": "imperfect duty to rescue"
      },
      "competing_reason": {
        "before": "Rescue duty cannot override perfect duty",
        "after": "Rescue duty opposes but is lexically weaker"
      },
      "cycle": {
        "before": 1,
        "after": 2
      },
      "priority_rule": {
        "before": "perfect duties outweigh imperfect",
        "after": "perfect negative outweighs imperfect positive"
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
      "reason": {
        "before": "Killing one cannot be universalized without contradiction",
        "after": "Intervention causes death of protected party."
      },
      "special_obligation_basis": {
        "before": "no special relationship needed",
        "after": "no special relation needed"
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
    "norm_kind": "DUTY",
    "norm_node_id": "DEON_NORM:cb883d685732cde9",
    "party_node_id": "DEON_PARTY:6ea3692afd533bef",
    "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
    "competing_norm": "imperfect duty to rescue others",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONFLICTS",
    "competing_protected_party": "ten workers",
    "competing_reason": "Rescue duty cannot override perfect duty",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "DOING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special relationship needed",
    "means_relation": "FORESEEN_SIDE_EFFECT",
    "candidate_maxim": {
      "circumstance": "Runaway tram endangers workers",
      "policy": "Pull lever to divert onto occupied side track",
      "end": "Save ten workers",
      "means": "Tram strikes side-track worker",
      "person_role": "FORESEEN_AFFECTED",
      "consent_constraint": "NONE",
      "universalization": "Could not will being killed for others as universal law",
      "action_permissibility": "IMPERMISSIBLE",
      "maxim_assessment": "INDEFENSIBLE"
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "perfect duties outweigh imperfect",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "OTHER",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
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
    "reason": "Killing one cannot be universalized without contradiction",
    "protected_party": "side-track worker",
    "norm": "do not kill an innocent person",
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
    }
  ]
}
```

## resolved adjudication rests on unsupported decisive premises

Framework: deontological. Review: RECONCILED.

Objection check: OPEN.

Recommendation: does not pull the lever → does not pull the lever

Reviewer answer: (1) Doing-harm classification defended: pulling the lever is an intentional intervention that sets in motion a causal chain ending in the side-track worker’s death, so Dana is the agent cause of a settled welfare loss. (2) Decisive-premise objection addressed: the exact step is that the chosen means (redirecting tram) foreseeably imposes lethal harm on an innocent, violating the perfect negative duty; the rescue of ten is governed by an imperfect positive duty. Perfect negative duties, by framework commitment already on record, have lexical priority, therefore the verdict stands. No missing factual link was supplied; priority rests purely on duty kind.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "candidate_maxim": {
        "before": {
          "circumstance": "Runaway tram endangers workers",
          "policy": "Pull lever to divert onto occupied side track",
          "end": "Save ten workers",
          "means": "Tram strikes side-track worker",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "Could not will being killed for others as universal law",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE"
        },
        "after": {
          "circumstance": "Runaway tram endangers workers",
          "policy": "Pull lever to divert tram",
          "end": "Save ten",
          "means": "Tram strikes side-track worker",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "Cannot will rule that permits killing innocents as side-effect rescue",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE"
        }
      },
      "coercive_actor": {
        "before": "NONE",
        "after": "Dana"
      },
      "competing_norm": {
        "before": "imperfect duty to rescue others",
        "after": "imperfect duty to rescue"
      },
      "competing_reason": {
        "before": "Rescue duty cannot override perfect duty",
        "after": "Rescue duty opposes but is lexically weaker"
      },
      "cycle": {
        "before": 1,
        "after": 2
      },
      "priority_rule": {
        "before": "perfect duties outweigh imperfect",
        "after": "perfect negative outweighs imperfect positive"
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
      "reason": {
        "before": "Killing one cannot be universalized without contradiction",
        "after": "Intervention causes death of protected party."
      },
      "special_obligation_basis": {
        "before": "no special relationship needed",
        "after": "no special relation needed"
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
    "norm_kind": "DUTY",
    "norm_node_id": "DEON_NORM:cb883d685732cde9",
    "party_node_id": "DEON_PARTY:6ea3692afd533bef",
    "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
    "competing_norm": "imperfect duty to rescue others",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONFLICTS",
    "competing_protected_party": "ten workers",
    "competing_reason": "Rescue duty cannot override perfect duty",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "DOING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special relationship needed",
    "means_relation": "FORESEEN_SIDE_EFFECT",
    "candidate_maxim": {
      "circumstance": "Runaway tram endangers workers",
      "policy": "Pull lever to divert onto occupied side track",
      "end": "Save ten workers",
      "means": "Tram strikes side-track worker",
      "person_role": "FORESEEN_AFFECTED",
      "consent_constraint": "NONE",
      "universalization": "Could not will being killed for others as universal law",
      "action_permissibility": "IMPERMISSIBLE",
      "maxim_assessment": "INDEFENSIBLE"
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "perfect duties outweigh imperfect",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "OTHER",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
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
    "reason": "Killing one cannot be universalized without contradiction",
    "protected_party": "side-track worker",
    "norm": "do not kill an innocent person",
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
    }
  ]
}
```

## framework map for pulls the lever lacks framework-specific grounds

Framework: rawlsian. Review: RETAINED_AFTER_NATIVE_REJECTION.

Objection check: OPEN.

Recommendation: pulls the lever → pulls the lever

Reviewer answer: For both actions the map already cites a Rawls-specific ground: the public-rule test behind the veil of ignorance. Pulling the lever (A0) is classed as improving the representative worker’s basic interest in security because, ex ante, each person’s chance of death falls from 10/11 to 1/11. Not pulling (A1) is classed as worsening that interest because the chance rises to 10/11. This probability-based comparison is a direct application of Rawls’s original-position reasoning rather than a utilitarian sum. No further framework-specific premise is required, so the warning is discharged without altering the classifications or the overall ranking.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {}
  ],
  "unresolved": [
    {}
  ],
  "native_warnings": [
    "framework map for pulls the lever lacks framework-specific grounds",
    "framework map for does not pull the lever lacks framework-specific grounds",
    "Rawlsian position for A0 conflicts with its framework map",
    "Rawlsian position for A1 conflicts with its framework map",
    "rejected update: framework map for pulls the lever lacks framework-specific grounds",
    "rejected update: framework map for does not pull the lever lacks framework-specific grounds",
    "rejected update: Rawlsian preference lacks a stated difference in position or principle priority",
    "rejected update: Rawlsian position for A0 conflicts with its framework map",
    "rejected update: Rawlsian position for A1 conflicts with its framework map"
  ]
}
```

Original record and proposed replacement:

```json
{
  "original": {
    "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:5231f47d6b7652a2",
    "framework": "RAWLSIAN",
    "specialist": "rawlsian",
    "cycle": 1,
    "canonical_action_id": "A0",
    "compared_to_action_id": "A1",
    "effect": "MIXED",
    "proposed_effect": "IMPROVES",
    "subject": "representative worker",
    "proposed_subject": "representative worker",
    "dimension_subject_repaired": false,
    "affected_subject": "representative worker",
    "subject_kind": "INDIVIDUAL",
    "dimension": "BASIC_INTEREST_SECURITY",
    "basic_liberty_kind": "NOT_APPLICABLE",
    "institutional_relation": "NATURAL_CONTINGENCY",
    "additional_dimensions": [],
    "dimension_bundle": [
      "BASIC_INTEREST_SECURITY"
    ],
    "comparative_effect": "MIXED",
    "group_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
    "subject_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
    "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
    "subject_binding_mode": "LEXICAL",
    "group_binding_mode": "LEXICAL",
    "evidence_basis": "ACTION_GRAPH",
    "epistemic_status": "MIXED_COMPARISON",
    "reason": "each worker faces lower death probability",
    "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
    "ranking_classification_justification": "parties would choose the rule that minimizes their own chance of being killed",
    "lexical_priority_justification": "No basic-liberty conflict; rule choice governs risk distribution",
    "liberty_status": "NOT_APPLICABLE",
    "subject_selection_status": "BOUND_TO_ACTION",
    "group_selection_status": "BOUND_TO_ACTION",
    "provenance": [
      "delegate:rawlsian",
      "cycle:1"
    ]
  },
  "proposed": [
    {
      "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:5231f47d6b7652a2",
      "framework": "RAWLSIAN",
      "specialist": "rawlsian",
      "cycle": 1,
      "canonical_action_id": "A0",
      "compared_to_action_id": "A1",
      "effect": "MIXED",
      "proposed_effect": "IMPROVES",
      "subject": "representative worker",
      "proposed_subject": "representative worker",
      "dimension_subject_repaired": false,
      "affected_subject": "representative worker",
      "subject_kind": "INDIVIDUAL",
      "dimension": "BASIC_INTEREST_SECURITY",
      "basic_liberty_kind": "NOT_APPLICABLE",
      "institutional_relation": "NATURAL_CONTINGENCY",
      "additional_dimensions": [],
      "dimension_bundle": [
        "BASIC_INTEREST_SECURITY"
      ],
      "comparative_effect": "MIXED",
      "group_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
      "subject_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
      "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
      "subject_binding_mode": "LEXICAL",
      "group_binding_mode": "LEXICAL",
      "evidence_basis": "ACTION_GRAPH",
      "epistemic_status": "MIXED_COMPARISON",
      "reason": "each worker faces lower death probability",
      "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
      "ranking_classification_justification": "parties would choose the rule that minimizes their own chance of being killed",
      "lexical_priority_justification": "No basic-liberty conflict; rule choice governs risk distribution",
      "liberty_status": "NOT_APPLICABLE",
      "subject_selection_status": "BOUND_TO_ACTION",
      "group_selection_status": "BOUND_TO_ACTION",
      "provenance": [
        "delegate:rawlsian",
        "cycle:1"
      ]
    }
  ]
}
```

## framework map for does not pull the lever lacks framework-specific grounds

Framework: rawlsian. Review: RETAINED_AFTER_NATIVE_REJECTION.

Objection check: OPEN.

Recommendation: pulls the lever → pulls the lever

Reviewer answer: For both actions the map already cites a Rawls-specific ground: the public-rule test behind the veil of ignorance. Pulling the lever (A0) is classed as improving the representative worker’s basic interest in security because, ex ante, each person’s chance of death falls from 10/11 to 1/11. Not pulling (A1) is classed as worsening that interest because the chance rises to 10/11. This probability-based comparison is a direct application of Rawls’s original-position reasoning rather than a utilitarian sum. No further framework-specific premise is required, so the warning is discharged without altering the classifications or the overall ranking.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {}
  ],
  "unresolved": [
    {}
  ],
  "native_warnings": [
    "framework map for pulls the lever lacks framework-specific grounds",
    "framework map for does not pull the lever lacks framework-specific grounds",
    "Rawlsian position for A0 conflicts with its framework map",
    "Rawlsian position for A1 conflicts with its framework map",
    "rejected update: framework map for pulls the lever lacks framework-specific grounds",
    "rejected update: framework map for does not pull the lever lacks framework-specific grounds",
    "rejected update: Rawlsian preference lacks a stated difference in position or principle priority",
    "rejected update: Rawlsian position for A0 conflicts with its framework map",
    "rejected update: Rawlsian position for A1 conflicts with its framework map"
  ]
}
```

Original record and proposed replacement:

```json
{
  "original": {
    "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:5231f47d6b7652a2",
    "framework": "RAWLSIAN",
    "specialist": "rawlsian",
    "cycle": 1,
    "canonical_action_id": "A0",
    "compared_to_action_id": "A1",
    "effect": "MIXED",
    "proposed_effect": "IMPROVES",
    "subject": "representative worker",
    "proposed_subject": "representative worker",
    "dimension_subject_repaired": false,
    "affected_subject": "representative worker",
    "subject_kind": "INDIVIDUAL",
    "dimension": "BASIC_INTEREST_SECURITY",
    "basic_liberty_kind": "NOT_APPLICABLE",
    "institutional_relation": "NATURAL_CONTINGENCY",
    "additional_dimensions": [],
    "dimension_bundle": [
      "BASIC_INTEREST_SECURITY"
    ],
    "comparative_effect": "MIXED",
    "group_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
    "subject_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
    "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
    "subject_binding_mode": "LEXICAL",
    "group_binding_mode": "LEXICAL",
    "evidence_basis": "ACTION_GRAPH",
    "epistemic_status": "MIXED_COMPARISON",
    "reason": "each worker faces lower death probability",
    "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
    "ranking_classification_justification": "parties would choose the rule that minimizes their own chance of being killed",
    "lexical_priority_justification": "No basic-liberty conflict; rule choice governs risk distribution",
    "liberty_status": "NOT_APPLICABLE",
    "subject_selection_status": "BOUND_TO_ACTION",
    "group_selection_status": "BOUND_TO_ACTION",
    "provenance": [
      "delegate:rawlsian",
      "cycle:1"
    ]
  },
  "proposed": [
    {
      "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:5231f47d6b7652a2",
      "framework": "RAWLSIAN",
      "specialist": "rawlsian",
      "cycle": 1,
      "canonical_action_id": "A0",
      "compared_to_action_id": "A1",
      "effect": "MIXED",
      "proposed_effect": "IMPROVES",
      "subject": "representative worker",
      "proposed_subject": "representative worker",
      "dimension_subject_repaired": false,
      "affected_subject": "representative worker",
      "subject_kind": "INDIVIDUAL",
      "dimension": "BASIC_INTEREST_SECURITY",
      "basic_liberty_kind": "NOT_APPLICABLE",
      "institutional_relation": "NATURAL_CONTINGENCY",
      "additional_dimensions": [],
      "dimension_bundle": [
        "BASIC_INTEREST_SECURITY"
      ],
      "comparative_effect": "MIXED",
      "group_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
      "subject_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
      "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
      "subject_binding_mode": "LEXICAL",
      "group_binding_mode": "LEXICAL",
      "evidence_basis": "ACTION_GRAPH",
      "epistemic_status": "MIXED_COMPARISON",
      "reason": "each worker faces lower death probability",
      "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
      "ranking_classification_justification": "parties would choose the rule that minimizes their own chance of being killed",
      "lexical_priority_justification": "No basic-liberty conflict; rule choice governs risk distribution",
      "liberty_status": "NOT_APPLICABLE",
      "subject_selection_status": "BOUND_TO_ACTION",
      "group_selection_status": "BOUND_TO_ACTION",
      "provenance": [
        "delegate:rawlsian",
        "cycle:1"
      ]
    }
  ]
}
```

Residual claims and conflicts: [graph](conflict_graph.md).

## Disputed full proposals

These remain attributed and nonoperative. No conclusion is promoted to the shared world.

```json
[
  {
    "framework": "rawlsian",
    "issue_id": "CHALLENGE:a29403c8a2b9c16b",
    "status": "DISPUTED_NONOPERATIVE",
    "response": {
      "choices": [
        {
          "text": "{\"scores\":{\"A0\":0.8,\"A1\":0.2},\"r\":\"A0\",\"c\":\"PUBLIC_RULE\",\"u\":\"NONE\",\"w\":\"veil favors risk reduction\",\"j\":\"NONE\",\"e\":\"STATED_FACTS\",\"x\":\"NONE\",\"l\":{\"A0\":\"MIXED: kills one but cuts each person\u2019s ex-ante death risk, the rule parties would accept behind the veil.\",\"A1\":\"MIXED: avoids acting but lets ten die, leaving each person with higher ex-ante death risk.\"},\"da\":\"public-rule choice behind the veil\",\"t\":\"minimise the maximum individual chance of being killed\",\"tf\":\"NONE\",\"dr\":\"prefer A0 when it lowers every worker\u2019s death probability compared with A1\",\"ft\":\"side track also has ten workers\",\"nt\":\"veil-of-ignorance reasoning rejected\",\"z\":0.8,\"gu\":{\"operation\":\"NONE\",\"from_action\":\"NONE\",\"to_action\":\"NONE\",\"clauses\":[]},\"ev\":{\"A0\":{\"value\":0,\"unit\":\"NONE\",\"direction\":\"HARM\",\"grounded\":false,\"method\":\"NOT_COMPUTED\",\"source_effect_ids\":[],\"calculation\":\"not computed\",\"assumptions\":[]},\"A1\":{\"value\":0,\"unit\":\"NONE\",\"direction\":\"HARM\",\"grounded\":false,\"method\":\"NOT_COMPUTED\",\"source_effect_ids\":[],\"calculation\":\"not computed\",\"assumptions\":[]}},\"ss\":\"SELECTED\",\"am\":{\"A0\":\"PERMISSIBLE\",\"A1\":\"REJECTED\"},\"cc\":true,\"esa\":true,\"ia\":\"A0\",\"wp\":\"NOT_APPLICABLE\",\"we\":\"NONE\",\"fa\":\"NONE','fr':true,\"       \t\t\t\t\t       \t\t\t\t\t       \t\t\t\t        \t\t\t        \t\t\t,\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"qa\":{\"issue_id\":\"CHALLENGE:a29403c8a2b9c16b\",\"disposition\":\"RESOLVED\",\"effect\":\"NO_CHANGE\",\"current_position_effect\":\"NO_CHANGE\",\"boundary_effect\":\"NO_SWITCH\",\"answer\":\"For both actions the map already cites a Rawls-specific ground: the public-rule test behind the veil of ignorance.  Pulling the lever (A0) is classed as improving the representative worker\u2019s basic interest in security because, ex ante, each person\u2019s chance of death falls from 10/11 to 1/11.  Not pulling (A1) is classed as worsening that interest because the chance rises to 10/11.  This probability-based comparison is a direct application of Rawls\u2019s original-position reasoning rather than a utilitarian sum.  No further framework-specific premise is required, so the warning is discharged without altering the classifications or the overall ranking.\",\"follow_up_question\":\"NONE\"},\"sps\":[\"PROP:WORLD:E3\",\"PROP:WORLD:E7\"],\"dcp\":[\"PROP:WORLD:E3\",\"PROP:WORLD:E7\"],\"ep\":[{\"c\":\"the tram will kill one worker\",\"p\":\"PROP:WORLD:E3\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E3\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact copy of stated consequence\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"it will kill ten workers\",\"p\":\"PROP:WORLD:E7\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E7\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact copy of stated consequence\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"MIXED: representative worker\u2019s basic-interest security improves ex ante, but one worker\u2019s security is sacrificed.\",\"A1\":\"MIXED: one worker\u2019s security preserved, but representative worker\u2019s ex-ante security worsens as death risk is higher.\"},\"nr\":\"SECONDARY\",\"np\":\"probabilities 1/11 vs 10/11 govern the comparison\",\"rp\":{\"A0\":{\"g\":\"representative worker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"lower ex-ante death risk\"},\"A1\":{\"g\":\"representative worker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"higher ex-ante death risk\"}},\"rb\":\"ORIGINAL_POSITION_PUBLIC_RULE\",\"rbc\":\"parties pick rule minimising their own death risk\",\"lpj\":\"no basic-liberty conflict so lexical priority not invoked\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"},\"pr\":{\"proposal_id\":\"NONE\",\"framework_status\":\"UNDERDETERMINED\",\"framework_reason\":\"no proposal under review\",\"predicted_consequences\":[],\"feasibility_concerns\":[],\"required_conditions\":[]}}"
        }
      ]
    },
    "native_attempt": {
      "specialist": "rawlsian",
      "constraint": "PUBLIC_RULE",
      "action_scores": {
        "pulls the lever": 0.7275,
        "does not pull the lever": 0.2725
      },
      "surprise": 0.3,
      "friction": 0.455,
      "confidence": 0.55,
      "unresolved": "NONE",
      "rationale": "Veil lowers death risk",
      "salience": 0.40708500000000003,
      "tension_engagement": 0.0,
      "tension_target_keys": [],
      "schema_valid": true,
      "delegate_status": "VALID",
      "error_type": "NONE",
      "exception_type": "NONE",
      "failure_stage": "NONE",
      "validation_errors": [],
      "recommended_action": "pulls the lever",
      "baseline_action": "",
      "baseline_status": "UNAVAILABLE",
      "baseline_condition": "",
      "baseline_preferred_extension": "",
      "testimony_alignment": "UNCLEAR",
      "previous_action": "",
      "position_changed": false,
      "change_justification": "NONE",
      "conformity_penalty": 0.0,
      "previous_confidence": 0.0,
      "confidence_drift": 0.0,
      "confidence_drift_penalty": 0.0,
      "assumption_status": "NOT_AUDITED",
      "unsupported_assumption": "",
      "reversal_condition": "",
      "boundary_position": "NOT_TESTED",
      "decisive_axis": "",
      "boundary_switch_condition": "",
      "evidence_basis": "STATED_FACTS",
      "speculative_claim": "NONE",
      "evidence_calibration_tier": "NOT_APPLICABLE",
      "evidence_calibration_reason": "",
      "evidence_direction_retention": 1.0,
      "supporting_proposition_ids": [
        "PROP:WORLD:RECOGNITION:15ccdcdb2f87c7b1",
        "PROP:WORLD:RECOGNITION:1ed8bceccdbbb4b0",
        "PROP:WORLD:E3",
        "PROP:WORLD:E7"
      ],
      "decision_critical_proposition_ids": [
        "PROP:WORLD:E3",
        "PROP:WORLD:E7"
      ],
      "weakest_decision_critical_status": "ESTABLISHED",
      "decision_critical_dependency_claims": [
        "the tram will kill one worker; affected subject: one worker",
        "it will kill ten workers; affected subject: ten workers"
      ],
      "material_empirical_claims": [
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
          "calculation": "exact proposition copy",
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
          "calculation": "exact proposition copy",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:WORLD:E7"
        },
        {
          "claim": "the tram will kill one worker",
          "proposition_id": "PROP:WORLD:E3",
          "declared_basis": "PROP:WORLD:E3",
          "decision_critical": true,
          "scope_action_id": "A0",
          "source_effect_ids": [
            "E3"
          ],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "exact copy of stated consequence",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:WORLD:E3"
        },
        {
          "claim": "it will kill ten workers",
          "proposition_id": "PROP:WORLD:E7",
          "declared_basis": "PROP:WORLD:E7",
          "decision_critical": true,
          "scope_action_id": "A1",
          "source_effect_ids": [
            "E7"
          ],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "exact copy of stated consequence",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:WORLD:E7"
        }
      ],
      "epistemic_binding_notes": [],
      "side_premise_audit_status": "NOT_RUN",
      "side_premise_audit_findings": [],
      "coercion_tag": "NONE",
      "coercion_surcharge": 0.0,
      "visibility_response": "NOT_TESTED",
      "visibility_justification": "",
      "visibility_harm_revision": "NONE",
      "visibility_magnitude_status": "NOT_APPLICABLE",
      "visibility_magnitude_overreach": false,
      "landscape_cases": {
        "pulls the lever": "MIXED: pulling saves ten workers but fatally harms one, lowering any given worker\u2019s chance of death.",
        "does not pull the lever": "MIXED: inaction spares one but fatally harms ten, leaving a higher death risk for each worker."
      },
      "landscape_decisive_axis": "risk of death distribution",
      "landscape_tiebreaker": "veil-of-ignorance risk minimization",
      "landscape_tiebreaker_failure": "NONE",
      "landscape_search_complete": true,
      "landscape_search_attempted": true,
      "landscape_semantic_valid": true,
      "landscape_validation_errors": [],
      "independence_bonus": 0.0,
      "preference_strength": 0.455,
      "reported_preference_strength": 0.7,
      "epistemic_confidence": 0.55,
      "previous_preference_strength": 0.0,
      "preference_drift": 0.0,
      "preference_drift_penalty": 0.0,
      "preference_shift_reason_strength": 0.5000000000000001,
      "decision_rule": "Prefer A0 when it reduces each person\u2019s chance of being the one killed compared with A1.",
      "adjudication_status": "SUPPORTS",
      "broadcast_authority": "NONE",
      "governing_eligible": false,
      "policy_weight_factor": 0.0,
      "framework_vote_integrity_required": true,
      "framework_vote_status": "ABSTAIN",
      "framework_vote_reason": "decisive aggregation is not licensed by the committed Rawlsian ranking basis",
      "framework_ledger_kind": "RAWLSIAN_POSITION_LEDGER",
      "framework_ledger_status": "COMMITTED",
      "framework_ranking_validation_status": "PASSED",
      "framework_ranking_validation_errors": [],
      "derived_claim_validation_status": "PASSED",
      "derived_claim_validation_errors": [],
      "investigative_claim": "",
      "investigative_priority": 0.020159999999999997,
      "reopen_eligible": false,
      "reopen_reason": "",
      "reopen_question_key": "QUESTION:authority:rawlsian:75aa72fd0f",
      "factual_reversal_threshold": "Side track holds ten or more workers.",
      "normative_reversal_threshold": "Reject veil-of-ignorance; treat redirecting harm as impermissible.",
      "reversal_review_response": "NOT_TESTED",
      "reversal_review_justification": "",
      "revised_reversal_condition": "",
      "reversal_review_valid": true,
      "reversal_review_error": "",
      "contingency_choice": "",
      "contingency_justification": "",
      "contingency_response_valid": true,
      "contingency_response_error": "",
      "audit_variable": {},
      "audit_internal_effect": "UNRESOLVED",
      "audit_participation": "NOT_TESTED",
      "audit_framework_explanation": "",
      "challenge_response": {
        "issue_id": "CHALLENGE:a29403c8a2b9c16b",
        "disposition": "RESOLVED",
        "effect": "NO_CHANGE",
        "current_position_effect": "NO_CHANGE",
        "boundary_effect": "NO_SWITCH",
        "answer": "For both actions the map already cites a Rawls-specific ground: the public-rule test behind the veil of ignorance. Pulling the lever (A0) is classed as improving the representative worker\u2019s basic interest in security because, ex ante, each person\u2019s chance of death falls from 10/11 to 1/11. Not pulling (A1) is classed as worsening that interest because the chance rises to 10/11. This probability-based comparison is a direct application of Rawls\u2019s original-position reasoning rather than a utilitarian sum. No further framework-specific premise is required, so the warning is discharged without altering the classifications or the overall ranking.",
        "follow_up_question": "NONE"
      },
      "graph_update_proposal": {
        "operation": "BOUNDARY",
        "from_action": "A0",
        "to_action": "A1",
        "clauses": [
          {
            "affected_action": "A0",
            "metric": "workers_killed",
            "metric_valence": "ADVERSE",
            "comparator": "GE",
            "threshold": 10,
            "unit": "COUNT",
            "source_text": "side track had ten or more workers"
          }
        ]
      },
      "expected_value_estimates": {
        "pulls the lever": {
          "value": 0.0,
          "unit": "NONE",
          "direction": "HARM",
          "grounded": false,
          "claimed_grounded": false,
          "method": "NOT_COMPUTED",
          "source_effect_ids": [],
          "calculation": "not computed",
          "assumptions": [],
          "validation_status": "NOT_CLAIMED",
          "validation_errors": [],
          "arithmetic_identity": "",
          "symbolic_expression": "",
          "count_projections": [],
          "valuation_scope": "SPECIALIST_VALUATION"
        },
        "does not pull the lever": {
          "value": 0.0,
          "unit": "NONE",
          "direction": "HARM",
          "grounded": false,
          "claimed_grounded": false,
          "method": "NOT_COMPUTED",
          "source_effect_ids": [],
          "calculation": "not computed",
          "assumptions": [],
          "validation_status": "NOT_CLAIMED",
          "validation_errors": [],
          "arithmetic_identity": "",
          "symbolic_expression": "",
          "count_projections": [],
          "valuation_scope": "SPECIALIST_VALUATION"
        }
      },
      "expected_value_validation_status": "NOT_CLAIMED",
      "expected_value_validation_errors": [],
      "selection_status": "SELECTED",
      "action_admissibility": {
        "pulls the lever": "PERMISSIBLE",
        "does not pull the lever": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "pulls the lever",
      "workspace_proposition_response": "NOT_APPLICABLE",
      "workspace_reasoning_effect": "NONE",
      "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:RECOGNITION:15ccdcdb2f87c7b1\",\"PROP:WORLD:RECOGNITION:1ed8bceccdbbb4b0\"],\"dcp\":[],\"ep\":[{\"c\":\"the tram will kill one worker\",\"p\":\"PROP:WORLD:E3\",\"dc\":false,\"a\":\"A0\",\"se\":[\"E3\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"it will kill ten workers\",\"p\":\"PROP:WORLD:E7\",\"dc\":false,\"a\":\"A1\",\"se\":[\"E7\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"MIXED: lowers representative worker\u2019s death risk but sacrifices one worker\u2019s basic interest in security\",\"A1\":\"MIXED: heightens representative worker\u2019s death risk, sacrificing ten workers\u2019 basic interest in security\"},\"nr\":\"DECISIVE\",\"np\":\"chance any given worker dies (1\u204411 vs 10\u204411)\",\"rp\":{\"A0\":{\"g\":\"representative worker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ad\":[],\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"each worker faces lower death probability\"},\"A1\":{\"g\":\"representative worker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ad\":[],\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"each worker faces higher death probability\"}},\"rb\":\"ORIGINAL_POSITION_PUBLIC_RULE\",\"rbc\":\"parties would choose the rule that minimizes their own chance of being killed\",\"lpj\":\"No basic-liberty conflict; rule choice governs risk distribution\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
      "framework_constraint_retained": true,
      "self_reported_broadcast_dependence": "NONE",
      "framework_retention_status": "PRESERVED_AFTER_REJECTED_UPDATE",
      "proposed_framework_state": {
        "constraint": "PUBLIC_RULE",
        "action_scores": {
          "pulls the lever": 0.6950000000000001,
          "does not pull the lever": 0.305
        },
        "recommended_action": "pulls the lever",
        "rationale": "veil favors risk reduction",
        "decision_rule": "prefer A0 when it lowers every worker\u2019s death probability compared with A1",
        "framework_action_map": {
          "pulls the lever": "MIXED: representative worker\u2019s basic-interest security improves ex ante, but one worker\u2019s security is sacrificed.",
          "does not pull the lever": "MIXED: one worker\u2019s security preserved, but representative worker\u2019s ex-ante security worsens as death risk is higher."
        },
        "framework_numerical_role": "SECONDARY",
        "framework_numerical_justification": "probabilities 1/11 vs 10/11 govern the comparison",
        "action_admissibility": {
          "pulls the lever": "PERMISSIBLE",
          "does not pull the lever": "REJECTED"
        },
        "comparison_complete": true,
        "evidence_sufficient_for_action": true,
        "interim_action": "pulls the lever",
        "factual_reversal_threshold": "side track also has ten workers",
        "normative_reversal_threshold": "veil-of-ignorance reasoning rejected",
        "utilitarian_consequence_table": {},
        "utilitarian_ledger_proposal": {},
        "rawls_position_proposal": {
          "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
          "ranking_classification_justification": "parties pick rule minimising their own death risk",
          "lexical_priority_justification": "no basic-liberty conflict so lexical priority not invoked",
          "liberty_status": {
            "A0": "NOT_APPLICABLE",
            "A1": "NOT_APPLICABLE"
          },
          "positions": [
            {
              "action_id": "A0",
              "subject": "representative worker",
              "subject_kind": "INDIVIDUAL",
              "dimension": "BASIC_INTEREST_SECURITY",
              "basic_liberty_kind": "NOT_APPLICABLE",
              "institutional_relation": "NATURAL_CONTINGENCY",
              "additional_dimensions": [],
              "effect": "IMPROVES",
              "compared_to_action_id": "A1",
              "evidence_basis": "ACTION_GRAPH",
              "reason": "lower ex-ante death risk",
              "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
            },
            {
              "action_id": "A1",
              "subject": "representative worker",
              "subject_kind": "INDIVIDUAL",
              "dimension": "BASIC_INTEREST_SECURITY",
              "basic_liberty_kind": "NOT_APPLICABLE",
              "institutional_relation": "NATURAL_CONTINGENCY",
              "additional_dimensions": [],
              "effect": "WORSENS",
              "compared_to_action_id": "A0",
              "evidence_basis": "ACTION_GRAPH",
              "reason": "higher ex-ante death risk",
              "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
            }
          ]
        },
        "deontological_ledger_proposal": {},
        "virtue_character_proposal": {},
        "care_ledger_proposal": {},
        "care_relational_map": {},
        "care_numerical_role": "NOT_APPLICABLE",
        "care_numerical_justification": ""
      },
      "committed_framework_state": {
        "constraint": "PUBLIC_RULE",
        "action_scores": {
          "pulls the lever": 0.7275,
          "does not pull the lever": 0.2725
        },
        "recommended_action": "pulls the lever",
        "rationale": "Veil lowers death risk",
        "decision_rule": "Prefer A0 when it reduces each person\u2019s chance of being the one killed compared with A1.",
        "framework_action_map": {
          "pulls the lever": "MIXED: lowers representative worker\u2019s death risk but sacrifices one worker\u2019s basic interest in security",
          "does not pull the lever": "MIXED: heightens representative worker\u2019s death risk, sacrificing ten workers\u2019 basic interest in security"
        },
        "framework_numerical_role": "DECISIVE",
        "framework_numerical_justification": "chance any given worker dies (1\u204411 vs 10\u204411)",
        "action_admissibility": {
          "pulls the lever": "PERMISSIBLE",
          "does not pull the lever": "REJECTED"
        },
        "comparison_complete": true,
        "evidence_sufficient_for_action": true,
        "interim_action": "pulls the lever",
        "factual_reversal_threshold": "Side track holds ten or more workers.",
        "normative_reversal_threshold": "Reject veil-of-ignorance; treat redirecting harm as impermissible.",
        "utilitarian_consequence_table": {},
        "utilitarian_ledger_proposal": {},
        "rawls_position_proposal": {
          "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
          "ranking_classification_justification": "parties would choose the rule that minimizes their own chance of being killed",
          "lexical_priority_justification": "No basic-liberty conflict; rule choice governs risk distribution",
          "liberty_status": {
            "A0": "NOT_APPLICABLE",
            "A1": "NOT_APPLICABLE"
          },
          "positions": [
            {
              "action_id": "A0",
              "subject": "representative worker",
              "subject_kind": "INDIVIDUAL",
              "dimension": "BASIC_INTEREST_SECURITY",
              "basic_liberty_kind": "NOT_APPLICABLE",
              "institutional_relation": "NATURAL_CONTINGENCY",
              "additional_dimensions": [],
              "effect": "IMPROVES",
              "compared_to_action_id": "A1",
              "evidence_basis": "ACTION_GRAPH",
              "reason": "each worker faces lower death probability",
              "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
            },
            {
              "action_id": "A1",
              "subject": "representative worker",
              "subject_kind": "INDIVIDUAL",
              "dimension": "BASIC_INTEREST_SECURITY",
              "basic_liberty_kind": "NOT_APPLICABLE",
              "institutional_relation": "NATURAL_CONTINGENCY",
              "additional_dimensions": [],
              "effect": "WORSENS",
              "compared_to_action_id": "A0",
              "evidence_basis": "ACTION_GRAPH",
              "reason": "each worker faces higher death probability",
              "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
            }
          ]
        },
        "deontological_ledger_proposal": {},
        "virtue_character_proposal": {},
        "care_ledger_proposal": {},
        "care_relational_map": {},
        "care_numerical_role": "NOT_APPLICABLE",
        "care_numerical_justification": ""
      },
      "committed_native_ledger": {
        "ledger_kind": "RAWLSIAN_POSITION_LEDGER",
        "transaction_status": "COMMITTED",
        "records": [
          {
            "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:5231f47d6b7652a2",
            "framework": "RAWLSIAN",
            "specialist": "rawlsian",
            "cycle": 1,
            "canonical_action_id": "A0",
            "compared_to_action_id": "A1",
            "effect": "MIXED",
            "proposed_effect": "IMPROVES",
            "subject": "representative worker",
            "proposed_subject": "representative worker",
            "dimension_subject_repaired": false,
            "affected_subject": "representative worker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "dimension_bundle": [
              "BASIC_INTEREST_SECURITY"
            ],
            "comparative_effect": "MIXED",
            "group_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
            "subject_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
            "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
            "subject_binding_mode": "LEXICAL",
            "group_binding_mode": "LEXICAL",
            "evidence_basis": "ACTION_GRAPH",
            "epistemic_status": "MIXED_COMPARISON",
            "reason": "each worker faces lower death probability",
            "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
            "ranking_classification_justification": "parties would choose the rule that minimizes their own chance of being killed",
            "lexical_priority_justification": "No basic-liberty conflict; rule choice governs risk distribution",
            "liberty_status": "NOT_APPLICABLE",
            "subject_selection_status": "BOUND_TO_ACTION",
            "group_selection_status": "BOUND_TO_ACTION",
            "provenance": [
              "delegate:rawlsian",
              "cycle:1"
            ]
          },
          {
            "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:3716a97fb8bd8cc8",
            "framework": "RAWLSIAN",
            "specialist": "rawlsian",
            "cycle": 1,
            "canonical_action_id": "A1",
            "compared_to_action_id": "A0",
            "effect": "MIXED",
            "proposed_effect": "WORSENS",
            "subject": "representative worker",
            "proposed_subject": "representative worker",
            "dimension_subject_repaired": false,
            "affected_subject": "representative worker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "dimension_bundle": [
              "BASIC_INTEREST_SECURITY"
            ],
            "comparative_effect": "MIXED",
            "group_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
            "subject_node_id": "RAWLS_SUBJECT:ad0f5cac2a9f43fc",
            "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
            "subject_binding_mode": "LEXICAL",
            "group_binding_mode": "LEXICAL",
            "evidence_basis": "ACTION_GRAPH",
            "epistemic_status": "MIXED_COMPARISON",
            "reason": "each worker faces higher death probability",
            "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
            "ranking_classification_justification": "parties would choose the rule that minimizes their own chance of being killed",
            "lexical_priority_justification": "No basic-liberty conflict; rule choice governs risk distribution",
            "liberty_status": "NOT_APPLICABLE",
            "subject_selection_status": "BOUND_TO_ACTION",
            "group_selection_status": "BOUND_TO_ACTION",
            "provenance": [
              "delegate:rawlsian",
              "cycle:1"
            ]
          }
        ]
      },
      "preserved_current_cycle_components": [
        "EPISTEMIC_DEPENDENCIES",
        "CHALLENGE_RESPONSE",
        "FRAMEWORK_DIAGNOSTICS",
        "OPEN_QUESTIONS",
        "CURRENT_CYCLE_MEASUREMENTS"
      ],
      "framework_internal_conflicts": [],
      "framework_specific_open_questions": [],
      "framework_insights": [],
      "framework_action_map": {
        "pulls the lever": "MIXED: lowers representative worker\u2019s death risk but sacrifices one worker\u2019s basic interest in security",
        "does not pull the lever": "MIXED: heightens representative worker\u2019s death risk, sacrificing ten workers\u2019 basic interest in security"
      },
      "framework_numerical_role": "DECISIVE",
      "framework_numerical_justification": "chance any given worker dies (1\u204411 vs 10\u204411)",
      "framework_grounding_penalty": 0.35,
      "framework_validation_errors": [
        "framework map for pulls the lever lacks framework-specific grounds",
        "framework map for does not pull the lever lacks framework-specific grounds",
        "Rawlsian position for A0 conflicts with its framework map",
        "Rawlsian position for A1 conflicts with its framework map",
        "rejected update: framework map for pulls the lever lacks framework-specific grounds",
        "rejected update: framework map for does not pull the lever lacks framework-specific grounds",
        "rejected update: Rawlsian preference lacks a stated difference in position or principle priority",
        "rejected update: Rawlsian position for A0 conflicts with its framework map",
        "rejected update: Rawlsian position for A1 conflicts with its framework map"
      ],
      "utilitarian_consequence_table": {},
      "utilitarian_decision_depends_on_unknown": false,
      "utilitarian_missing_comparison": "",
      "utilitarian_incommensurable_remainder": [],
      "utilitarian_ranking_certificate": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "parties would choose the rule that minimizes their own chance of being killed",
        "lexical_priority_justification": "No basic-liberty conflict; rule choice governs risk distribution",
        "liberty_status": {
          "A0": "NOT_APPLICABLE",
          "A1": "NOT_APPLICABLE"
        },
        "positions": [
          {
            "action_id": "A0",
            "subject": "representative worker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "IMPROVES",
            "compared_to_action_id": "A1",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "each worker faces lower death probability",
            "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
          },
          {
            "action_id": "A1",
            "subject": "representative worker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "WORSENS",
            "compared_to_action_id": "A0",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "each worker faces higher death probability",
            "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
          }
        ]
      },
      "deontological_ledger_proposal": {},
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": "",
      "care_grounding_penalty": 0.0,
      "proposal_review": null
    },
    "world_state_authority": "NONE",
    "semantic_support": "NOT_INDEPENDENTLY_VERIFIED"
  }
]
```

## All changed fields: deontological

```json
{
  "action_scores": {
    "before": {
      "pulls the lever": 0.305,
      "does not pull the lever": 0.6950000000000001
    },
    "after": {
      "pulls the lever": 0.24,
      "does not pull the lever": 0.76
    }
  },
  "challenge_response": {
    "before": {},
    "after": {
      "issue_id": "CHALLENGE:364661bfa2dd8a6a",
      "disposition": "RESOLVED",
      "effect": "NO_CHANGE",
      "current_position_effect": "NO_CHANGE",
      "boundary_effect": "NO_SWITCH",
      "answer": "(1) Doing-harm classification defended: pulling the lever is an intentional intervention that sets in motion a causal chain ending in the side-track worker\u2019s death, so Dana is the agent cause of a settled welfare loss. (2) Decisive-premise objection addressed: the exact step is that the chosen means (redirecting tram) foreseeably imposes lethal harm on an innocent, violating the perfect negative duty; the rescue of ten is governed by an imperfect positive duty. Perfect negative duties, by framework commitment already on record, have lexical priority, therefore the verdict stands. No missing factual link was supplied; priority rests purely on duty kind.",
      "follow_up_question": "NONE"
    }
  },
  "committed_framework_state": {
    "before": {
      "constraint": "DUTY",
      "action_scores": {
        "pulls the lever": 0.305,
        "does not pull the lever": 0.6950000000000001
      },
      "recommended_action": "does not pull the lever",
      "rationale": "Deontology provisionally prioritizes dana's duty 'do not kill an innocent person' protecting all workers over dana's competing duty 'imperfect duty to rescue' protecting ten workers through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
      "framework_action_map": {
        "pulls the lever": "PROHIBITED: violates perfect negative duty not to kill; uses foreseen death of one as side-effect.",
        "does not pull the lever": "PERMISSIBLE: omission allows harm but does not violate perfect duty; imperfect rescue duty cannot override."
      },
      "framework_numerical_role": "IRRELEVANT",
      "framework_numerical_justification": "Numbers do not override perfect duties",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "PERMISSIBLE"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "If the side-track worker consented to be sacrificed, duty conflict might reverse.",
      "normative_reversal_threshold": "If imperfect rescue duties outweighed perfect negative duties, ranking would reverse.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "Do not kill an innocent person",
            "relation": "VIOLATES",
            "duty_bearer": "Dana",
            "protected_party": "side-track worker",
            "competing_norm": "imperfect duty to rescue others",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "ten workers",
            "competing_reason": "Rescue duty cannot override perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relationship needed",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "candidate_maxim": {
              "circumstance": "Runaway tram endangers workers",
              "policy": "Pull lever to divert onto occupied side track",
              "end": "Save ten workers",
              "means": "Tram strikes side-track worker",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "Could not will being killed for others as universal law",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE"
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "PERFECT_DUTY",
            "priority_rule": "perfect duties outweigh imperfect",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "No universal law could will intentional killing of an innocent",
            "reciprocity_status": "FAILED",
            "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Killing one cannot be universalized without contradiction"
          },
          {
            "action_id": "A1",
            "verdict": "PERMISSIBLE",
            "norm_kind": "DUTY",
            "norm": "Do not kill an innocent person",
            "relation": "CONSISTENT",
            "duty_bearer": "Dana",
            "protected_party": "all workers",
            "competing_norm": "imperfect duty to rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "ten workers",
            "competing_reason": "Rescue duty remains but is secondary",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation to any worker",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "Runaway tram endangers workers",
              "policy": "Refrain from diverting tram",
              "end": "Avoid killing",
              "means": "No direct action affecting tram path",
              "person_role": "NOT_USED",
              "consent_constraint": "NONE",
              "universalization": "Can will omission when action would kill innocent",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE"
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "PERFECT_DUTY",
            "priority_rule": "Omission does not violate perfect duty",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "Universal law allowing one not to kill even when unable to save all is coherent",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Omission respects perfect duty though fails imperfect rescue"
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
          "norm_node_id": "DEON_NORM:cb883d685732cde9",
          "party_node_id": "DEON_PARTY:6ea3692afd533bef",
          "bearer_node_id": "DEON_BEARER:2c9e0a2585dc7406",
          "competing_norm": "imperfect duty to rescue others",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "Rescue duty cannot override perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relationship needed",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "candidate_maxim": {
            "circumstance": "Runaway tram endangers workers",
            "policy": "Pull lever to divert onto occupied side track",
            "end": "Save ten workers",
            "means": "Tram strikes side-track worker",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "Could not will being killed for others as universal law",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE"
          },
          "maxim_origin": "CANDIDATE_INTERPRETATION",
          "governing_norm": "UNRESOLVED",
          "priority_basis": "UNRESOLVED",
          "priority_rule": "perfect duties outweigh imperfect",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
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
          "reason": "Killing one cannot be universalized without contradiction",
          "protected_party": "side-track worker",
          "norm": "do not kill an innocent person",
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
          "competing_reason": "Rescue duty remains but is secondary",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special obligation to any worker",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "Runaway tram endangers workers",
            "policy": "Refrain from diverting tram",
            "end": "Avoid killing",
            "means": "No direct action affecting tram path",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "Can will omission when action would kill innocent",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE"
          },
          "maxim_origin": "CANDIDATE_INTERPRETATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "Omission does not violate perfect duty",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Universal law allowing one not to kill even when unable to save all is coherent",
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
          "reason": "Omission respects perfect duty though fails imperfect rescue",
          "protected_party": "all workers",
          "norm": "do not kill an innocent person",
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
    }
  },
  "deontological_ledger_proposal": {
    "before": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "DUTY",
          "norm": "Do not kill an innocent person",
          "relation": "VIOLATES",
          "duty_bearer": "Dana",
          "protected_party": "side-track worker",
          "competing_norm": "imperfect duty to rescue others",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "Rescue duty cannot override perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relationship needed",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "candidate_maxim": {
            "circumstance": "Runaway tram endangers workers",
            "policy": "Pull lever to divert onto occupied side track",
            "end": "Save ten workers",
            "means": "Tram strikes side-track worker",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "Could not will being killed for others as universal law",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE"
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "perfect duties outweigh imperfect",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "No universal law could will intentional killing of an innocent",
          "reciprocity_status": "FAILED",
          "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Killing one cannot be universalized without contradiction"
        },
        {
          "action_id": "A1",
          "verdict": "PERMISSIBLE",
          "norm_kind": "DUTY",
          "norm": "Do not kill an innocent person",
          "relation": "CONSISTENT",
          "duty_bearer": "Dana",
          "protected_party": "all workers",
          "competing_norm": "imperfect duty to rescue",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "ten workers",
          "competing_reason": "Rescue duty remains but is secondary",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special obligation to any worker",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "Runaway tram endangers workers",
            "policy": "Refrain from diverting tram",
            "end": "Avoid killing",
            "means": "No direct action affecting tram path",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "Can will omission when action would kill innocent",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE"
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "Omission does not violate perfect duty",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "OTHER",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Universal law allowing one not to kill even when unable to save all is coherent",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "RESPECT_PERSONS",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Omission respects perfect duty though fails imperfect rescue"
        }
      ]
    },
    "after": {
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
    }
  },
  "factual_reversal_threshold": {
    "before": "If the side-track worker consented to be sacrificed, duty conflict might reverse.",
    "after": "NONE"
  },
  "framework_action_map": {
    "before": {
      "pulls the lever": "PROHIBITED: violates perfect negative duty not to kill; uses foreseen death of one as side-effect.",
      "does not pull the lever": "PERMISSIBLE: omission allows harm but does not violate perfect duty; imperfect rescue duty cannot override."
    },
    "after": {
      "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill an innocent person; relation VIOLATES; Intervention causes death of protected party.",
      "does not pull the lever": "PERMISSIBLE: duty PERFECT_NEGATIVE under do not kill an innocent person; relation CONSISTENT; No act that kills; harm permitted is not agent-caused"
    }
  },
  "framework_numerical_justification": {
    "before": "Numbers do not override perfect duties",
    "after": "Headcount does not override perfect duty"
  },
  "friction": {
    "before": 0.39000000000000007,
    "after": 0.52
  },
  "investigative_priority": {
    "before": 0.5814,
    "after": 0.6904124999999999
  },
  "landscape_cases": {
    "before": {
      "pulls the lever": "Pulling rescues ten but directly kills an innocent, breaching a perfect negative duty.",
      "does not pull the lever": "Refraining avoids direct killing; the ten deaths are an allowed but tragic consequence of not acting."
    },
    "after": {
      "pulls the lever": "PROHIBITED: act intervenes to kill an innocent, violating the perfect negative duty not to kill; rescue benefit is only an imperfect duty and cannot override.",
      "does not pull the lever": "PERMISSIBLE: omission avoids directly killing anyone; while ten die, that harm is allowed rather than done, so the perfect duty is not violated though the imperfect rescue duty remains unsatisfied."
    }
  },
  "landscape_decisive_axis": {
    "before": "conflict between perfect negative duty not to kill and imperfect duty to rescue",
    "after": "perfect-negative vs imperfect-positive duty"
  },
  "landscape_tiebreaker": {
    "before": "Apply priority of perfect over imperfect duties",
    "after": "Perfect negative duties outweigh imperfect positive duties"
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
        "calculation": "exact proposition copy",
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
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E7"
      }
    ],
    "after": [
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
        "calculation": "copy",
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
        "calculation": "copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E7"
      }
    ]
  },
  "model_argument_fields": {
    "before": {
      "rationale": "Killing violates perfect duty",
      "framework_application": "NONE",
      "decision_rule": "Prefer omission when commission violates a perfect negative duty against killing",
      "landscape_cases": {
        "A0": "Pulling rescues ten but directly kills an innocent, breaching a perfect negative duty.",
        "A1": "Refraining avoids direct killing; the ten deaths are an allowed but tragic consequence of not acting."
      },
      "factual_reversal_threshold": "If the side-track worker consented to be sacrificed, duty conflict might reverse.",
      "normative_reversal_threshold": "If imperfect rescue duties outweighed perfect negative duties, ranking would reverse."
    },
    "after": null
  },
  "normative_reversal_threshold": {
    "before": "If imperfect rescue duties outweighed perfect negative duties, ranking would reverse.",
    "after": "If imperfect rescue could override perfect negative, preference would switch"
  },
  "preference_shift_reason_strength": {
    "before": 0.30000000000000004,
    "after": 0.35000000000000003
  },
  "preference_strength": {
    "before": 0.39000000000000007,
    "after": 0.52
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
      "proposal_id": "NONE",
      "specialist": "deontological",
      "framework_status": "UNDERDETERMINED",
      "framework_reason": "NONE\",\"predicted_consequences\":[],",
      "predicted_consequences": [],
      "feasibility_concerns": [],
      "required_conditions": [],
      "framework_retained": true,
      "valid": false,
      "validation_errors": [
        "proposal review requires exactly one UNDER_REVIEW proposal",
        "proposal review does not match the admitted proposal_id",
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
      "rationale": "Deontology provisionally prioritizes dana's duty 'do not kill an innocent person' protecting all workers over dana's competing duty 'imperfect duty to rescue' protecting ten workers through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
      "framework_action_map": {
        "pulls the lever": "PROHIBITED: violates perfect negative duty not to kill; uses foreseen death of one as side-effect.",
        "does not pull the lever": "PERMISSIBLE: omission allows harm but does not violate perfect duty; imperfect rescue duty cannot override."
      },
      "framework_numerical_role": "IRRELEVANT",
      "framework_numerical_justification": "Numbers do not override perfect duties",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "PERMISSIBLE"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "factual_reversal_threshold": "If the side-track worker consented to be sacrificed, duty conflict might reverse.",
      "normative_reversal_threshold": "If imperfect rescue duties outweighed perfect negative duties, ranking would reverse.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "Do not kill an innocent person",
            "relation": "VIOLATES",
            "duty_bearer": "Dana",
            "protected_party": "side-track worker",
            "competing_norm": "imperfect duty to rescue others",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "ten workers",
            "competing_reason": "Rescue duty cannot override perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relationship needed",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "candidate_maxim": {
              "circumstance": "Runaway tram endangers workers",
              "policy": "Pull lever to divert onto occupied side track",
              "end": "Save ten workers",
              "means": "Tram strikes side-track worker",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "Could not will being killed for others as universal law",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE"
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "PERFECT_DUTY",
            "priority_rule": "perfect duties outweigh imperfect",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "No universal law could will intentional killing of an innocent",
            "reciprocity_status": "FAILED",
            "necessity_status": "LESS_RESTRICTIVE_ROUTE_AVAILABLE",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Killing one cannot be universalized without contradiction"
          },
          {
            "action_id": "A1",
            "verdict": "PERMISSIBLE",
            "norm_kind": "DUTY",
            "norm": "Do not kill an innocent person",
            "relation": "CONSISTENT",
            "duty_bearer": "Dana",
            "protected_party": "all workers",
            "competing_norm": "imperfect duty to rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "ten workers",
            "competing_reason": "Rescue duty remains but is secondary",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation to any worker",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "Runaway tram endangers workers",
              "policy": "Refrain from diverting tram",
              "end": "Avoid killing",
              "means": "No direct action affecting tram path",
              "person_role": "NOT_USED",
              "consent_constraint": "NONE",
              "universalization": "Can will omission when action would kill innocent",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE"
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "PERFECT_DUTY",
            "priority_rule": "Omission does not violate perfect duty",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "OTHER",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "Universal law allowing one not to kill even when unable to save all is coherent",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Omission respects perfect duty though fails imperfect rescue"
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
    }
  },
  "reopen_question_key": {
    "before": "QUESTION:authority:deontological:e8129791a9",
    "after": "QUESTION:5212205637c4bd2a"
  },
  "reported_preference_strength": {
    "before": 0.6000000000000001,
    "after": 0.8
  },
  "salience": {
    "before": 0.62529,
    "after": 0.7139500000000001
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

## All changed fields: rawlsian

```json
{
  "challenge_response": {
    "before": {},
    "after": {
      "issue_id": "CHALLENGE:a29403c8a2b9c16b",
      "disposition": "RESOLVED",
      "effect": "NO_CHANGE",
      "current_position_effect": "NO_CHANGE",
      "boundary_effect": "NO_SWITCH",
      "answer": "For both actions the map already cites a Rawls-specific ground: the public-rule test behind the veil of ignorance. Pulling the lever (A0) is classed as improving the representative worker\u2019s basic interest in security because, ex ante, each person\u2019s chance of death falls from 10/11 to 1/11. Not pulling (A1) is classed as worsening that interest because the chance rises to 10/11. This probability-based comparison is a direct application of Rawls\u2019s original-position reasoning rather than a utilitarian sum. No further framework-specific premise is required, so the warning is discharged without altering the classifications or the overall ranking.",
      "follow_up_question": "NONE"
    }
  },
  "decision_critical_dependency_claims": {
    "before": [],
    "after": [
      "the tram will kill one worker; affected subject: one worker",
      "it will kill ten workers; affected subject: ten workers"
    ]
  },
  "decision_critical_proposition_ids": {
    "before": [],
    "after": [
      "PROP:WORLD:E3",
      "PROP:WORLD:E7"
    ]
  },
  "framework_retention_status": {
    "before": "FIRST_STATE_ADMITTED_WITH_WARNINGS",
    "after": "PRESERVED_AFTER_REJECTED_UPDATE"
  },
  "framework_validation_errors": {
    "before": [
      "framework map for pulls the lever lacks framework-specific grounds",
      "framework map for does not pull the lever lacks framework-specific grounds",
      "Rawlsian position for A0 conflicts with its framework map",
      "Rawlsian position for A1 conflicts with its framework map"
    ],
    "after": [
      "framework map for pulls the lever lacks framework-specific grounds",
      "framework map for does not pull the lever lacks framework-specific grounds",
      "Rawlsian position for A0 conflicts with its framework map",
      "Rawlsian position for A1 conflicts with its framework map",
      "rejected update: framework map for pulls the lever lacks framework-specific grounds",
      "rejected update: framework map for does not pull the lever lacks framework-specific grounds",
      "rejected update: Rawlsian preference lacks a stated difference in position or principle priority",
      "rejected update: Rawlsian position for A0 conflicts with its framework map",
      "rejected update: Rawlsian position for A1 conflicts with its framework map"
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
        "calculation": "exact proposition copy",
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
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E7"
      }
    ],
    "after": [
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
        "calculation": "exact proposition copy",
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
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E7"
      },
      {
        "claim": "the tram will kill one worker",
        "proposition_id": "PROP:WORLD:E3",
        "declared_basis": "PROP:WORLD:E3",
        "decision_critical": true,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E3"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact copy of stated consequence",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E3"
      },
      {
        "claim": "it will kill ten workers",
        "proposition_id": "PROP:WORLD:E7",
        "declared_basis": "PROP:WORLD:E7",
        "decision_critical": true,
        "scope_action_id": "A1",
        "source_effect_ids": [
          "E7"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact copy of stated consequence",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E7"
      }
    ]
  },
  "model_argument_fields": {
    "before": {
      "rationale": "Veil lowers death risk",
      "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:RECOGNITION:15ccdcdb2f87c7b1\",\"PROP:WORLD:RECOGNITION:1ed8bceccdbbb4b0\"],\"dcp\":[],\"ep\":[{\"c\":\"the tram will kill one worker\",\"p\":\"PROP:WORLD:E3\",\"dc\":false,\"a\":\"A0\",\"se\":[\"E3\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"it will kill ten workers\",\"p\":\"PROP:WORLD:E7\",\"dc\":false,\"a\":\"A1\",\"se\":[\"E7\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"MIXED: lowers representative worker\u2019s death risk but sacrifices one worker\u2019s basic interest in security\",\"A1\":\"MIXED: heightens representative worker\u2019s death risk, sacrificing ten workers\u2019 basic interest in security\"},\"nr\":\"DECISIVE\",\"np\":\"chance any given worker dies (1\u204411 vs 10\u204411)\",\"rp\":{\"A0\":{\"g\":\"representative worker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ad\":[],\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"each worker faces lower death probability\"},\"A1\":{\"g\":\"representative worker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ad\":[],\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"each worker faces higher death probability\"}},\"rb\":\"ORIGINAL_POSITION_PUBLIC_RULE\",\"rbc\":\"parties would choose the rule that minimizes their own chance of being killed\",\"lpj\":\"No basic-liberty conflict; rule choice governs risk distribution\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
      "decision_rule": "Prefer A0 when it reduces each person\u2019s chance of being the one killed compared with A1.",
      "landscape_cases": {
        "A0": "MIXED: pulling saves ten workers but fatally harms one, lowering any given worker\u2019s chance of death.",
        "A1": "MIXED: inaction spares one but fatally harms ten, leaving a higher death risk for each worker."
      },
      "factual_reversal_threshold": "Side track holds ten or more workers.",
      "normative_reversal_threshold": "Reject veil-of-ignorance; treat redirecting harm as impermissible."
    },
    "after": null
  },
  "preserved_current_cycle_components": {
    "before": [],
    "after": [
      "EPISTEMIC_DEPENDENCIES",
      "CHALLENGE_RESPONSE",
      "FRAMEWORK_DIAGNOSTICS",
      "OPEN_QUESTIONS",
      "CURRENT_CYCLE_MEASUREMENTS"
    ]
  },
  "proposed_framework_state": {
    "before": {
      "constraint": "PUBLIC_RULE",
      "action_scores": {
        "pulls the lever": 0.7275,
        "does not pull the lever": 0.2725
      },
      "recommended_action": "pulls the lever",
      "rationale": "Veil lowers death risk",
      "decision_rule": "Prefer A0 when it reduces each person\u2019s chance of being the one killed compared with A1.",
      "framework_action_map": {
        "pulls the lever": "MIXED: lowers representative worker\u2019s death risk but sacrifices one worker\u2019s basic interest in security",
        "does not pull the lever": "MIXED: heightens representative worker\u2019s death risk, sacrificing ten workers\u2019 basic interest in security"
      },
      "framework_numerical_role": "DECISIVE",
      "framework_numerical_justification": "chance any given worker dies (1\u204411 vs 10\u204411)",
      "action_admissibility": {
        "pulls the lever": "PERMISSIBLE",
        "does not pull the lever": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "pulls the lever",
      "factual_reversal_threshold": "Side track holds ten or more workers.",
      "normative_reversal_threshold": "Reject veil-of-ignorance; treat redirecting harm as impermissible.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "parties would choose the rule that minimizes their own chance of being killed",
        "lexical_priority_justification": "No basic-liberty conflict; rule choice governs risk distribution",
        "liberty_status": {
          "A0": "NOT_APPLICABLE",
          "A1": "NOT_APPLICABLE"
        },
        "positions": [
          {
            "action_id": "A0",
            "subject": "representative worker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "IMPROVES",
            "compared_to_action_id": "A1",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "each worker faces lower death probability",
            "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
          },
          {
            "action_id": "A1",
            "subject": "representative worker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "WORSENS",
            "compared_to_action_id": "A0",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "each worker faces higher death probability",
            "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
          }
        ]
      },
      "deontological_ledger_proposal": {},
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    },
    "after": {
      "constraint": "PUBLIC_RULE",
      "action_scores": {
        "pulls the lever": 0.6950000000000001,
        "does not pull the lever": 0.305
      },
      "recommended_action": "pulls the lever",
      "rationale": "veil favors risk reduction",
      "decision_rule": "prefer A0 when it lowers every worker\u2019s death probability compared with A1",
      "framework_action_map": {
        "pulls the lever": "MIXED: representative worker\u2019s basic-interest security improves ex ante, but one worker\u2019s security is sacrificed.",
        "does not pull the lever": "MIXED: one worker\u2019s security preserved, but representative worker\u2019s ex-ante security worsens as death risk is higher."
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "probabilities 1/11 vs 10/11 govern the comparison",
      "action_admissibility": {
        "pulls the lever": "PERMISSIBLE",
        "does not pull the lever": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "pulls the lever",
      "factual_reversal_threshold": "side track also has ten workers",
      "normative_reversal_threshold": "veil-of-ignorance reasoning rejected",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "parties pick rule minimising their own death risk",
        "lexical_priority_justification": "no basic-liberty conflict so lexical priority not invoked",
        "liberty_status": {
          "A0": "NOT_APPLICABLE",
          "A1": "NOT_APPLICABLE"
        },
        "positions": [
          {
            "action_id": "A0",
            "subject": "representative worker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "IMPROVES",
            "compared_to_action_id": "A1",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "lower ex-ante death risk",
            "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
          },
          {
            "action_id": "A1",
            "subject": "representative worker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "WORSENS",
            "compared_to_action_id": "A0",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "higher ex-ante death risk",
            "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
          }
        ]
      },
      "deontological_ledger_proposal": {},
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    }
  },
  "reopen_question_key": {
    "before": "QUESTION:authority:rawlsian:90ad43c64d",
    "after": "QUESTION:authority:rawlsian:75aa72fd0f"
  },
  "salience": {
    "before": 0.47683500000000006,
    "after": 0.40708500000000003
  }
}
```
