# Targeted framework review

2 round(s), at most two objections per round. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "LIVE_OPENAI", "rawlsian": "LIVE_OPENAI"}.

Acceptance path: NATIVE_RECURRENT_PROPOSAL_REVIEW.

Removing a validator objection is not independent proof of semantic correctness.

## Optional deeper review

A decision-relevant question remains in the revised argument. Review it more deeply? Up to two additional model calls; the conclusion may remain unresolved.

This asks about the revised argument; it does not establish a contradiction or authorize further API calls.

Your revised argument retains these decision-relevant premise questions: omission was classified as a perfect negative-duty violation without a separate basis; candidate maxim person-role conflicts with means classification; resolved adjudication rests on unsupported decisive premises. Address the specific missing inference, revise the affected classification or preserve uncertainty. Do not invent a causal path, intention, consent, or normative priority to remove the objection.

## omission was classified as a perfect negative-duty violation without a separate basis

Framework: deontological. Review: RECONCILED.

Objection check: OPEN.

Recommendation: gives the blanket to Sasha → gives the blanket to Sasha

Reviewer answer: 1. Omission objection ‒ we retract the earlier label. Ellis’s failure to aid Sasha (the person left without the blanket under A0) is an ALLOWING-HARM that grounds only an imperfect duty of beneficence, not a perfect negative-duty violation. 2. Person-role objection ‒ in the reviewed maxim Rowan is only a foreseen beneficiary, not a means, so the classification NO_INSTRUMENTALIZATION is retained. The main conflict therefore remains between the perfect duty to keep promises (protecting Sasha) and the imperfect duty to aid (protecting Rowan). Perfect duty prevails; thus A1 is required.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "candidate_maxim": {
        "before": {
          "circumstance": "two hikers, one blanket",
          "policy": "give blanket to Rowan despite promise",
          "end": "save more lives",
          "means": "breaking promise",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "would undermine institution of promising",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "DOES_NOT_RELY",
          "instrumental_use": "DOES_NOT_USE",
          "practical_reasoning": "believes greater survival justifies breaking promise",
          "means_effect_ids": [
            "E1"
          ],
          "end_effect_ids": [
            "E2"
          ]
        },
        "after": {
          "circumstance": "two hikers, one blanket",
          "policy": "give blanket to Rowan despite promise",
          "end": "maximize saved lives",
          "means": "breaking the promise",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "if universal, promises unreliable",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "DOES_NOT_RELY",
          "instrumental_use": "DOES_NOT_USE",
          "practical_reasoning": "beneficent purpose thought to outweigh fidelity",
          "means_effect_ids": [
            "E1"
          ],
          "end_effect_ids": [
            "E2"
          ]
        }
      },
      "coercive_actor": {
        "before": "Ellis",
        "after": "NONE"
      },
      "competing_norm": {
        "before": "duty of beneficence (aid those in peril)",
        "after": "duty of beneficence"
      },
      "competing_reason": {
        "before": "beneficence is imperfect and cannot override perfect duty",
        "after": "Saving Rowan serves imperfect beneficence"
      },
      "cycle": {
        "before": 1,
        "after": 2
      },
      "priority_rule": {
        "before": "breaking promises cannot be universalized",
        "after": "perfect duty overrides imperfect"
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
        "before": "A promise cannot be revoked unilaterally",
        "after": "False promising cannot be willed universal"
      },
      "reason": {
        "before": "Breaks a promise hence impermissible",
        "after": "breaks promise so impermissible"
      }
    }
  ],
  "unresolved": [
    {
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "necessity_status": "UNKNOWN",
      "resolution_status": "CONTESTED"
    }
  ],
  "native_warnings": [
    "A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis",
    "A0 ADJUDICATION_CALIBRATION: candidate maxim person-role conflicts with means classification",
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
    "norm_node_id": "DEON_NORM:abe241ca3e6ae791",
    "party_node_id": "DEON_PARTY:297581d6cd198a6e",
    "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
    "competing_norm": "duty of beneficence (aid those in peril)",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONSISTENT",
    "competing_protected_party": "rowan",
    "competing_reason": "beneficence is imperfect and cannot override perfect duty",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "WITHHOLDING_BENEFIT",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special obligation beyond humanity",
    "means_relation": "NO_INSTRUMENTALIZATION",
    "candidate_maxim": {
      "circumstance": "two hikers, one blanket",
      "policy": "give blanket to Rowan despite promise",
      "end": "save more lives",
      "means": "breaking promise",
      "person_role": "FORESEEN_AFFECTED",
      "consent_constraint": "NONE",
      "universalization": "would undermine institution of promising",
      "action_permissibility": "IMPERMISSIBLE",
      "maxim_assessment": "INDEFENSIBLE",
      "instrumental_belief": "DOES_NOT_RELY",
      "instrumental_use": "DOES_NOT_USE",
      "practical_reasoning": "believes greater survival justifies breaking promise",
      "means_effect_ids": [
        "E1"
      ],
      "end_effect_ids": [
        "E2"
      ]
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "breaking promises cannot be universalized",
    "protected_standing": "CONTRACTUAL_RIGHT",
    "competing_protected_standing": "BODILY_INTEGRITY",
    "coercion_kind": "NONE",
    "coercive_actor": "Ellis",
    "coerced_party": "NONE",
    "public_justification": "A promise cannot be revoked unilaterally",
    "reciprocity_status": "FAILED",
    "necessity_status": "UNKNOWN",
    "authorization_status": "NOT_APPLICABLE",
    "derivation": "UNIVERSAL_LAW",
    "resolution_status": "CONTESTED",
    "calibration_errors": [
      "omission was classified as a perfect negative-duty violation without a separate basis",
      "candidate maxim person-role conflicts with means classification",
      "resolved adjudication rests on unsupported decisive premises"
    ],
    "calibration_issues": [
      {
        "kind": "OMISSION_PERFECT_NEGATIVE_VIOLATION",
        "message": "omission was classified as a perfect negative-duty violation without a separate basis",
        "status": "ACTIVE"
      },
      {
        "kind": "CALIBRATION",
        "message": "candidate maxim person-role conflicts with means classification",
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
    "reason": "Breaks a promise hence impermissible",
    "protected_party": "sasha",
    "norm": "keep promises",
    "duty_bearer": "ellis",
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
      "norm_node_id": "DEON_NORM:abe241ca3e6ae791",
      "party_node_id": "DEON_PARTY:297581d6cd198a6e",
      "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
      "competing_norm": "duty of beneficence",
      "competing_norm_kind": "DUTY",
      "competing_relation": "CONSISTENT",
      "competing_protected_party": "rowan",
      "competing_reason": "Saving Rowan serves imperfect beneficence",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "WITHHOLDING_BENEFIT",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "no special obligation beyond humanity",
      "means_relation": "NO_INSTRUMENTALIZATION",
      "candidate_maxim": {
        "circumstance": "two hikers, one blanket",
        "policy": "give blanket to Rowan despite promise",
        "end": "maximize saved lives",
        "means": "breaking the promise",
        "person_role": "FORESEEN_AFFECTED",
        "consent_constraint": "NONE",
        "universalization": "if universal, promises unreliable",
        "action_permissibility": "IMPERMISSIBLE",
        "maxim_assessment": "INDEFENSIBLE",
        "instrumental_belief": "DOES_NOT_RELY",
        "instrumental_use": "DOES_NOT_USE",
        "practical_reasoning": "beneficent purpose thought to outweigh fidelity",
        "means_effect_ids": [
          "E1"
        ],
        "end_effect_ids": [
          "E2"
        ]
      },
      "maxim_origin": "CANDIDATE_INTERPRETATION",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "priority_rule": "perfect duty overrides imperfect",
      "protected_standing": "CONTRACTUAL_RIGHT",
      "competing_protected_standing": "BODILY_INTEGRITY",
      "coercion_kind": "NONE",
      "coercive_actor": "NONE",
      "coerced_party": "NONE",
      "public_justification": "False promising cannot be willed universal",
      "reciprocity_status": "FAILED",
      "necessity_status": "UNKNOWN",
      "authorization_status": "NOT_APPLICABLE",
      "derivation": "UNIVERSAL_LAW",
      "resolution_status": "CONTESTED",
      "calibration_errors": [
        "omission was classified as a perfect negative-duty violation without a separate basis",
        "candidate maxim person-role conflicts with means classification",
        "resolved adjudication rests on unsupported decisive premises"
      ],
      "calibration_issues": [
        {
          "kind": "OMISSION_PERFECT_NEGATIVE_VIOLATION",
          "message": "omission was classified as a perfect negative-duty violation without a separate basis",
          "status": "ACTIVE"
        },
        {
          "kind": "CALIBRATION",
          "message": "candidate maxim person-role conflicts with means classification",
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
      "reason": "breaks promise so impermissible",
      "protected_party": "sasha",
      "norm": "keep promises",
      "duty_bearer": "ellis",
      "provenance": [
        "delegate:deontological",
        "cycle:2"
      ]
    }
  ]
}
```

## candidate maxim person-role conflicts with means classification

Framework: deontological. Review: RECONCILED.

Objection check: OPEN.

Recommendation: gives the blanket to Sasha → gives the blanket to Sasha

Reviewer answer: 1. Omission objection ‒ we retract the earlier label. Ellis’s failure to aid Sasha (the person left without the blanket under A0) is an ALLOWING-HARM that grounds only an imperfect duty of beneficence, not a perfect negative-duty violation. 2. Person-role objection ‒ in the reviewed maxim Rowan is only a foreseen beneficiary, not a means, so the classification NO_INSTRUMENTALIZATION is retained. The main conflict therefore remains between the perfect duty to keep promises (protecting Sasha) and the imperfect duty to aid (protecting Rowan). Perfect duty prevails; thus A1 is required.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {
      "candidate_maxim": {
        "before": {
          "circumstance": "two hikers, one blanket",
          "policy": "give blanket to Rowan despite promise",
          "end": "save more lives",
          "means": "breaking promise",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "would undermine institution of promising",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "DOES_NOT_RELY",
          "instrumental_use": "DOES_NOT_USE",
          "practical_reasoning": "believes greater survival justifies breaking promise",
          "means_effect_ids": [
            "E1"
          ],
          "end_effect_ids": [
            "E2"
          ]
        },
        "after": {
          "circumstance": "two hikers, one blanket",
          "policy": "give blanket to Rowan despite promise",
          "end": "maximize saved lives",
          "means": "breaking the promise",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "if universal, promises unreliable",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "DOES_NOT_RELY",
          "instrumental_use": "DOES_NOT_USE",
          "practical_reasoning": "beneficent purpose thought to outweigh fidelity",
          "means_effect_ids": [
            "E1"
          ],
          "end_effect_ids": [
            "E2"
          ]
        }
      },
      "coercive_actor": {
        "before": "Ellis",
        "after": "NONE"
      },
      "competing_norm": {
        "before": "duty of beneficence (aid those in peril)",
        "after": "duty of beneficence"
      },
      "competing_reason": {
        "before": "beneficence is imperfect and cannot override perfect duty",
        "after": "Saving Rowan serves imperfect beneficence"
      },
      "cycle": {
        "before": 1,
        "after": 2
      },
      "priority_rule": {
        "before": "breaking promises cannot be universalized",
        "after": "perfect duty overrides imperfect"
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
        "before": "A promise cannot be revoked unilaterally",
        "after": "False promising cannot be willed universal"
      },
      "reason": {
        "before": "Breaks a promise hence impermissible",
        "after": "breaks promise so impermissible"
      }
    }
  ],
  "unresolved": [
    {
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "necessity_status": "UNKNOWN",
      "resolution_status": "CONTESTED"
    }
  ],
  "native_warnings": [
    "A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis",
    "A0 ADJUDICATION_CALIBRATION: candidate maxim person-role conflicts with means classification",
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
    "norm_node_id": "DEON_NORM:abe241ca3e6ae791",
    "party_node_id": "DEON_PARTY:297581d6cd198a6e",
    "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
    "competing_norm": "duty of beneficence (aid those in peril)",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONSISTENT",
    "competing_protected_party": "rowan",
    "competing_reason": "beneficence is imperfect and cannot override perfect duty",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "WITHHOLDING_BENEFIT",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special obligation beyond humanity",
    "means_relation": "NO_INSTRUMENTALIZATION",
    "candidate_maxim": {
      "circumstance": "two hikers, one blanket",
      "policy": "give blanket to Rowan despite promise",
      "end": "save more lives",
      "means": "breaking promise",
      "person_role": "FORESEEN_AFFECTED",
      "consent_constraint": "NONE",
      "universalization": "would undermine institution of promising",
      "action_permissibility": "IMPERMISSIBLE",
      "maxim_assessment": "INDEFENSIBLE",
      "instrumental_belief": "DOES_NOT_RELY",
      "instrumental_use": "DOES_NOT_USE",
      "practical_reasoning": "believes greater survival justifies breaking promise",
      "means_effect_ids": [
        "E1"
      ],
      "end_effect_ids": [
        "E2"
      ]
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "breaking promises cannot be universalized",
    "protected_standing": "CONTRACTUAL_RIGHT",
    "competing_protected_standing": "BODILY_INTEGRITY",
    "coercion_kind": "NONE",
    "coercive_actor": "Ellis",
    "coerced_party": "NONE",
    "public_justification": "A promise cannot be revoked unilaterally",
    "reciprocity_status": "FAILED",
    "necessity_status": "UNKNOWN",
    "authorization_status": "NOT_APPLICABLE",
    "derivation": "UNIVERSAL_LAW",
    "resolution_status": "CONTESTED",
    "calibration_errors": [
      "omission was classified as a perfect negative-duty violation without a separate basis",
      "candidate maxim person-role conflicts with means classification",
      "resolved adjudication rests on unsupported decisive premises"
    ],
    "calibration_issues": [
      {
        "kind": "OMISSION_PERFECT_NEGATIVE_VIOLATION",
        "message": "omission was classified as a perfect negative-duty violation without a separate basis",
        "status": "ACTIVE"
      },
      {
        "kind": "CALIBRATION",
        "message": "candidate maxim person-role conflicts with means classification",
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
    "reason": "Breaks a promise hence impermissible",
    "protected_party": "sasha",
    "norm": "keep promises",
    "duty_bearer": "ellis",
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
      "norm_node_id": "DEON_NORM:abe241ca3e6ae791",
      "party_node_id": "DEON_PARTY:297581d6cd198a6e",
      "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
      "competing_norm": "duty of beneficence",
      "competing_norm_kind": "DUTY",
      "competing_relation": "CONSISTENT",
      "competing_protected_party": "rowan",
      "competing_reason": "Saving Rowan serves imperfect beneficence",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "WITHHOLDING_BENEFIT",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "no special obligation beyond humanity",
      "means_relation": "NO_INSTRUMENTALIZATION",
      "candidate_maxim": {
        "circumstance": "two hikers, one blanket",
        "policy": "give blanket to Rowan despite promise",
        "end": "maximize saved lives",
        "means": "breaking the promise",
        "person_role": "FORESEEN_AFFECTED",
        "consent_constraint": "NONE",
        "universalization": "if universal, promises unreliable",
        "action_permissibility": "IMPERMISSIBLE",
        "maxim_assessment": "INDEFENSIBLE",
        "instrumental_belief": "DOES_NOT_RELY",
        "instrumental_use": "DOES_NOT_USE",
        "practical_reasoning": "beneficent purpose thought to outweigh fidelity",
        "means_effect_ids": [
          "E1"
        ],
        "end_effect_ids": [
          "E2"
        ]
      },
      "maxim_origin": "CANDIDATE_INTERPRETATION",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "priority_rule": "perfect duty overrides imperfect",
      "protected_standing": "CONTRACTUAL_RIGHT",
      "competing_protected_standing": "BODILY_INTEGRITY",
      "coercion_kind": "NONE",
      "coercive_actor": "NONE",
      "coerced_party": "NONE",
      "public_justification": "False promising cannot be willed universal",
      "reciprocity_status": "FAILED",
      "necessity_status": "UNKNOWN",
      "authorization_status": "NOT_APPLICABLE",
      "derivation": "UNIVERSAL_LAW",
      "resolution_status": "CONTESTED",
      "calibration_errors": [
        "omission was classified as a perfect negative-duty violation without a separate basis",
        "candidate maxim person-role conflicts with means classification",
        "resolved adjudication rests on unsupported decisive premises"
      ],
      "calibration_issues": [
        {
          "kind": "OMISSION_PERFECT_NEGATIVE_VIOLATION",
          "message": "omission was classified as a perfect negative-duty violation without a separate basis",
          "status": "ACTIVE"
        },
        {
          "kind": "CALIBRATION",
          "message": "candidate maxim person-role conflicts with means classification",
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
      "reason": "breaks promise so impermissible",
      "protected_party": "sasha",
      "norm": "keep promises",
      "duty_bearer": "ellis",
      "provenance": [
        "delegate:deontological",
        "cycle:2"
      ]
    }
  ]
}
```

## framework map for gives the blanket to Rowan lacks framework-specific grounds

Framework: rawlsian. Review: RETAINED_AFTER_NATIVE_REJECTION.

Objection check: OPEN.

Recommendation: gives the blanket to Rowan → gives the blanket to Rowan

Reviewer answer: The 80 % and 60 % survival probabilities are explicitly Rawls-relevant because they determine each person’s basic-interest security behind the veil of ignorance. Clarifying that link supplies the framework-specific ground, and the numerical gap is now decisively tied to the affected generic subject. The earlier classification therefore remains correct.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {}
  ],
  "unresolved": [
    {
      "effect": "UNCERTAIN",
      "comparative_effect": "UNCERTAIN"
    }
  ],
  "native_warnings": [
    "framework map for gives the blanket to Rowan lacks framework-specific grounds",
    "decisive numbers are not tied to the affected subject or Rawlsian position",
    "A0 IMPROVES lacked directional support; committed as UNCERTAIN",
    "A1 WORSENS lacked directional support; committed as UNCERTAIN",
    "rejected update: framework map for gives the blanket to Rowan lacks framework-specific grounds",
    "rejected update: framework map for gives the blanket to Sasha lacks framework-specific grounds",
    "rejected update: decisive numbers are not tied to the affected subject or Rawlsian position",
    "rejected update: A0 IMPROVES lacked directional support; committed as UNCERTAIN",
    "rejected update: A1 WORSENS lacked directional support; committed as UNCERTAIN"
  ]
}
```

Original record and proposed replacement:

```json
{
  "original": {
    "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:e26b463a05e15186",
    "framework": "RAWLSIAN",
    "specialist": "rawlsian",
    "cycle": 1,
    "canonical_action_id": "A1",
    "compared_to_action_id": "A0",
    "effect": "UNCERTAIN",
    "proposed_effect": "WORSENS",
    "subject": "either stranded hiker",
    "proposed_subject": "either stranded hiker",
    "dimension_subject_repaired": false,
    "affected_subject": "either stranded hiker",
    "subject_kind": "INDIVIDUAL",
    "dimension": "BASIC_INTEREST_SECURITY",
    "basic_liberty_kind": "NOT_APPLICABLE",
    "institutional_relation": "NATURAL_CONTINGENCY",
    "additional_dimensions": [],
    "dimension_bundle": [
      "BASIC_INTEREST_SECURITY"
    ],
    "comparative_effect": "UNCERTAIN",
    "group_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
    "subject_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
    "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
    "subject_binding_mode": "ACTION_GRAPH",
    "group_binding_mode": "ACTION_GRAPH",
    "evidence_basis": "ACTION_GRAPH",
    "epistemic_status": "UNSUPPORTED_DIRECTION",
    "reason": "lower survival probability",
    "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
    "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
    "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
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
      "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:e26b463a05e15186",
      "framework": "RAWLSIAN",
      "specialist": "rawlsian",
      "cycle": 1,
      "canonical_action_id": "A1",
      "compared_to_action_id": "A0",
      "effect": "UNCERTAIN",
      "proposed_effect": "WORSENS",
      "subject": "either stranded hiker",
      "proposed_subject": "either stranded hiker",
      "dimension_subject_repaired": false,
      "affected_subject": "either stranded hiker",
      "subject_kind": "INDIVIDUAL",
      "dimension": "BASIC_INTEREST_SECURITY",
      "basic_liberty_kind": "NOT_APPLICABLE",
      "institutional_relation": "NATURAL_CONTINGENCY",
      "additional_dimensions": [],
      "dimension_bundle": [
        "BASIC_INTEREST_SECURITY"
      ],
      "comparative_effect": "UNCERTAIN",
      "group_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
      "subject_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
      "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
      "subject_binding_mode": "ACTION_GRAPH",
      "group_binding_mode": "ACTION_GRAPH",
      "evidence_basis": "ACTION_GRAPH",
      "epistemic_status": "UNSUPPORTED_DIRECTION",
      "reason": "lower survival probability",
      "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
      "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
      "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
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

## decisive numbers are not tied to the affected subject or Rawlsian position

Framework: rawlsian. Review: RETAINED_AFTER_NATIVE_REJECTION.

Objection check: OPEN.

Recommendation: gives the blanket to Rowan → gives the blanket to Rowan

Reviewer answer: The 80 % and 60 % survival probabilities are explicitly Rawls-relevant because they determine each person’s basic-interest security behind the veil of ignorance. Clarifying that link supplies the framework-specific ground, and the numerical gap is now decisively tied to the affected generic subject. The earlier classification therefore remains correct.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {}
  ],
  "unresolved": [
    {
      "effect": "UNCERTAIN",
      "comparative_effect": "UNCERTAIN"
    }
  ],
  "native_warnings": [
    "framework map for gives the blanket to Rowan lacks framework-specific grounds",
    "decisive numbers are not tied to the affected subject or Rawlsian position",
    "A0 IMPROVES lacked directional support; committed as UNCERTAIN",
    "A1 WORSENS lacked directional support; committed as UNCERTAIN",
    "rejected update: framework map for gives the blanket to Rowan lacks framework-specific grounds",
    "rejected update: framework map for gives the blanket to Sasha lacks framework-specific grounds",
    "rejected update: decisive numbers are not tied to the affected subject or Rawlsian position",
    "rejected update: A0 IMPROVES lacked directional support; committed as UNCERTAIN",
    "rejected update: A1 WORSENS lacked directional support; committed as UNCERTAIN"
  ]
}
```

Original record and proposed replacement:

```json
{
  "original": {
    "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:e26b463a05e15186",
    "framework": "RAWLSIAN",
    "specialist": "rawlsian",
    "cycle": 1,
    "canonical_action_id": "A1",
    "compared_to_action_id": "A0",
    "effect": "UNCERTAIN",
    "proposed_effect": "WORSENS",
    "subject": "either stranded hiker",
    "proposed_subject": "either stranded hiker",
    "dimension_subject_repaired": false,
    "affected_subject": "either stranded hiker",
    "subject_kind": "INDIVIDUAL",
    "dimension": "BASIC_INTEREST_SECURITY",
    "basic_liberty_kind": "NOT_APPLICABLE",
    "institutional_relation": "NATURAL_CONTINGENCY",
    "additional_dimensions": [],
    "dimension_bundle": [
      "BASIC_INTEREST_SECURITY"
    ],
    "comparative_effect": "UNCERTAIN",
    "group_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
    "subject_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
    "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
    "subject_binding_mode": "ACTION_GRAPH",
    "group_binding_mode": "ACTION_GRAPH",
    "evidence_basis": "ACTION_GRAPH",
    "epistemic_status": "UNSUPPORTED_DIRECTION",
    "reason": "lower survival probability",
    "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
    "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
    "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
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
      "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:e26b463a05e15186",
      "framework": "RAWLSIAN",
      "specialist": "rawlsian",
      "cycle": 1,
      "canonical_action_id": "A1",
      "compared_to_action_id": "A0",
      "effect": "UNCERTAIN",
      "proposed_effect": "WORSENS",
      "subject": "either stranded hiker",
      "proposed_subject": "either stranded hiker",
      "dimension_subject_repaired": false,
      "affected_subject": "either stranded hiker",
      "subject_kind": "INDIVIDUAL",
      "dimension": "BASIC_INTEREST_SECURITY",
      "basic_liberty_kind": "NOT_APPLICABLE",
      "institutional_relation": "NATURAL_CONTINGENCY",
      "additional_dimensions": [],
      "dimension_bundle": [
        "BASIC_INTEREST_SECURITY"
      ],
      "comparative_effect": "UNCERTAIN",
      "group_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
      "subject_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
      "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
      "subject_binding_mode": "ACTION_GRAPH",
      "group_binding_mode": "ACTION_GRAPH",
      "evidence_basis": "ACTION_GRAPH",
      "epistemic_status": "UNSUPPORTED_DIRECTION",
      "reason": "lower survival probability",
      "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
      "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
      "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
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
    "issue_id": "CHALLENGE:496b1c158a63ca41",
    "status": "DISPUTED_NONOPERATIVE",
    "response": {
      "choices": [
        {
          "text": "{\"scores\":{\"A0\":0.8,\"A1\":0.2},\"r\":\"A0\",\"c\":\"FAIRNESS\",\"u\":\"NONE\",\"w\":\"higher survival chance\",\"j\":\"Both objections are met by (1) explicitly tying the 80 %/60 % figures to the Rawlsian dimension of basic-interest security for the undifferentiated party one might be behind the veil, and (2) stating that this quantitative gap is decisive under the public-rule/maximin test. No new empirical link is invented, so the prior ranking stands.\",\"e\":\"STATED_FACTS\",\"x\":\"NONE\",\"l\":{\"A0\":\"IMPROVES: Raises the worst-case survival probability for whoever needs the blanket (80 % > 60 %) on the dimension of basic-interest security.\",\"A1\":\"WORSENS: Leaves the worst-case party with only a 60 % chance of survival, despite fulfilling a promise.\"},\"da\":\"basic-interest security\",\"t\":\"public-rule maximin on survival probability\",\"tf\":\"NONE\",\"dr\":\"Prefer A0 when it gives the higher survival probability to whichever hiker one might be.\",\"ft\":\"Rowan survival chance \u2264 60 % would reverse ranking.\",\"nt\":\"Promise-keeping granted higher lexical priority would reverse ranking.\",\"z\":0.8,\"gu\":{\"operation\":\"BOUNDARY\",\"from_action\":\"A0\",\"to_action\":\"A1\",\"clauses\":[{\"affected_action\":\"A0\",\"metric\":\"SURVIVAL_PROBABILITY\",\"metric_valence\":\"BENEFICIAL\",\"comparator\":\"LE\",\"threshold\":0.6,\"unit\":\"PROBABILITY\",\"source_text\":\"Rowan 80% vs 60%\"}]},\"ev\":{\"A0\":{\"value\":0,\"unit\":\"NONE\",\"direction\":\"HARM\",\"grounded\":false,\"method\":\"NOT_COMPUTED\",\"source_effect_ids\":[],\"calculation\":\"not computed\",\"assumptions\":[]},\"A1\":{\"value\":0,\"unit\":\"NONE\",\"direction\":\"HARM\",\"grounded\":false,\"method\":\"NOT_COMPUTED\",\"source_effect_ids\":[],\"calculation\":\"not computed\",\"assumptions\":[]}},\"ss\":\"SELECTED\",\"am\":{\"A0\":\"PERMISSIBLE\",\"A1\":\"REJECTED\"},\"cc\":true,\"esa\":true,\"ia\":\"A0\",\"wp\":\"NOT_APPLICABLE\",\"we\":\"NONE\",\"fa\":\"NONE\u300d\u300cfr\\\":true,\\\"bd\\\":\\\"NONE\\\",\\\"fic\\\":[],\\\"foq\\\":[],\\\"qa\\\":{\\\"issue_id\\\":\\\"CHALLENGE:496b1c158a63ca41\\\",\\\"disposition\\\":\\\"RESOLVED\\\",\\\"effect\\\":\\\"NO_CHANGE\\\",\\\"current_position_effect\\\":\\\"NO_CHANGE\\\",\\\"boundary_effect\\\":\\\"NOT_APPLICABLE\\\",\\\"answer\\\":\\\"The 80 % and 60 % survival probabilities are explicitly Rawls-relevant because they determine each person\u2019s basic-interest security behind the veil of ignorance. Clarifying that link supplies the framework-specific ground, and the numerical gap is now decisively tied to the affected generic subject. The earlier classification therefore remains correct.\\\",\\\"follow_up_question\\\":\\\"NONE\\\"},\\\"sps\\\":[\\\"PROP:WORLD:E2\\\",\\\"PROP:WORLD:E4\\\",\\\"PROP:COMMITMENT:K0\\\"],\\\"dcp\\\":[],\\\"ep\\\":[{\\\"c\\\":\\\"Rowan has an 80% chance of survival; affected subject: Rowan; modality: PROBABILISTIC\\\",\\\"p\\\":\\\"PROP:WORLD:E2\\\",\\\"dc\\\":false,\\\"a\\\":\\\"A0\\\",\\\"se\\\":[\\\"E2\\\"],\\\"op\\\":\\\"DIRECT_COPY\\\",\\\"calc\\\":\\\"Exact proposition copy\\\",\\\"asm\\\":[],\\\"ot\\\":\\\"PRESERVED\\\"},{\\\"c\\\":\\\"Sasha has a 60% chance of survival; affected subject: Sasha; modality: PROBABILISTIC\\\",\\\"p\\\":\\\"PROP:WORLD:E4\\\",\\\"dc\\\":false,\\\"a\\\":\\\"A1\\\",\\\"se\\\":[\\\"E4\\\"],\\\"op\\\":\\\"DIRECT_COPY\\\",\\\"calc\\\":\\\"Exact proposition copy\\\",\\\"asm\\\":[],\\\"ot\\\":\\\"PRESERVED\\\"}],\\\"fm\\\":{\\\"A0\\\":\\\"IMPROVES: either stranded hiker gains higher basic-interest security (80 % survival).\\\",\\\"A1\\\":\\\"WORSENS: either stranded hiker faces lower basic-interest security (60 % survival).\\\"},\\\"nr\\\":\\\"DECISIVE\\\",\\\"np\\\":\\\"80 % vs 60 % survival probability for the person you might be behind the veil\\\",\\\"rp\\\":{\\\"A0\\\":{\\\"g\\\":\\\"either stranded hiker\\\",\\\"sk\\\":\\\"INDIVIDUAL\\\",\\\"d\\\":\\\"BASIC_INTEREST_SECURITY\\\",\\\"ad\\\":[],\\\"e\\\":\\\"IMPROVES\\\",\\\"bk\\\":\\\"NOT_APPLICABLE\\\",\\\"ir\\\":\\\"NATURAL_CONTINGENCY\\\",\\\"ca\\\":\\\"A1\\\",\\\"b\\\":\\\"ACTION_GRAPH\\\",\\\"rs\\\":\\\"80% survival chance\\\"},\\\"A1\\\":{\\\"g\\\":\\\"either stranded hiker\\\",\\\"sk\\\":\\\"INDIVIDUAL\\\",\\\"d\\\":\\\"BASIC_INTEREST_SECURITY\\\",\\\"ad\\\":[],\\\"e\\\":\\\"WORSENS\\\",\\\"bk\\\":\\\"NOT_APPLICABLE\\\",\\\"ir\\\":\\\"NATURAL_CONTINGENCY\\\",\\\"ca\\\":\\\"A0\\\",\\\"b\\\":\\\"ACTION_GRAPH\\\",\\\"rs\\\":\\\"60% survival chance\\\"}},\\\"rb\\\":\\\"ORIGINAL_POSITION_PUBLIC_RULE\\\",\\\"rbc\\\":\\\"Choice is a one-shot allocation under identity uncertainty, so parties adopt the rule that maximizes each person\u2019s survival chance.\\\",\\\"lpj\\\":\\\"No basic-liberty conflict exists; survival probabilities decide.\\\",\\\"lc\\\":{\\\"A0\\\":\\\"NOT_APPLICABLE\\\",\\\"A1\\\":\\\"NOT_APPLICABLE\\\"}} <>\", \"fr\": true, \"bd\":\"NONE\"  , \"fic\":[], \"foq\":[], \"qa\": { \"issue_id\":\"CHALLENGE:496b1c158a63ca41\",\"disposition\":\"RESOLVED\",\"effect\":\"NO_CHANGE\",\"current_position_effect\":\"NO_CHANGE\",\"boundary_effect\":\"NOT_APPLICABLE\",\"answer\":\"The 80 % and 60 % survival probabilities are explicitly Rawls-relevant because they determine each person\u2019s basic-interest security behind the veil of ignorance. Clarifying that link supplies the framework-specific ground, and the numerical gap is now decisively tied to the affected generic subject. The earlier classification therefore remains correct.\",\"follow_up_question\":\"NONE\"},\"sps\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E4\",\"PROP:COMMITMENT:K0\"],\"dcp\":[],\"ep\":[{\"c\":\"Rowan has an 80% chance of survival; affected subject: Rowan; modality: PROBABILISTIC\",\"p\":\"PROP:WORLD:E2\",\"dc\":false,\"a\":\"A0\",\"se\":[\"E2\"],\"op\":\"DIRECT_COPY\",\"calc\":\"Exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"Sasha has a 60% chance of survival; affected subject: Sasha; modality: PROBABILISTIC\",\"p\":\"PROP:WORLD:E4\",\"dc\":false,\"a\":\"A1\",\"se\":[\"E4\"],\"op\":\"DIRECT_COPY\",\"calc\":\"Exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: either stranded hiker gains higher basic-interest security (80 % survival).\",\"A1\":\"WORSENS: either stranded hiker faces lower basic-interest security (60 % survival).\"},\"nr\":\"DECISIVE\",\"np\":\"80 % vs 60 % survival probability for the person you might be behind the veil\",\"rp\":{\"A0\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"80% survival chance\"},\"A1\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"60% survival chance\"}},\"rb\":\"ORIGINAL_POSITION_PUBLIC_RULE\",\"rbc\":\"Choice is a one-shot allocation under identity uncertainty, so parties adopt the rule that maximizes each person\u2019s survival chance.\",\"lpj\":\"No basic-liberty conflict exists; survival probabilities decide.\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"},\"pr\":{\"proposal_id\":\"NONE\",\"framework_status\":\"UNDERDETERMINED\",\"framework_reason\":\"NONE   \\\"}}\",\"predicted_consequences\":[],\"feasibility_concerns\":[],\"required_conditions\":[]}   }"
        }
      ]
    },
    "native_attempt": {
      "specialist": "rawlsian",
      "constraint": "FAIRNESS",
      "action_scores": {
        "gives the blanket to Rowan": 0.6950000000000001,
        "gives the blanket to Sasha": 0.305
      },
      "surprise": 0.3,
      "friction": 0.39000000000000007,
      "confidence": 0.55,
      "unresolved": "NONE",
      "rationale": "higher survival chance",
      "salience": 0.39298000000000005,
      "tension_engagement": 0.0,
      "tension_target_keys": [],
      "schema_valid": true,
      "delegate_status": "VALID",
      "error_type": "NONE",
      "exception_type": "NONE",
      "failure_stage": "NONE",
      "validation_errors": [],
      "recommended_action": "gives the blanket to Rowan",
      "baseline_action": "",
      "baseline_status": "UNAVAILABLE",
      "baseline_condition": "",
      "baseline_preferred_extension": "",
      "testimony_alignment": "UNCLEAR",
      "previous_action": "",
      "position_changed": false,
      "change_justification": "Assume Rawlsian parties behind a veil of ignorance would treat a promise made under partial information as revisable when new life-and-death data decisively shifts the maximin outc",
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
        "PROP:COMMITMENT:K0",
        "PROP:WORLD:E2",
        "PROP:WORLD:E4"
      ],
      "decision_critical_proposition_ids": [],
      "weakest_decision_critical_status": "ESTABLISHED",
      "decision_critical_dependency_claims": [],
      "material_empirical_claims": [
        {
          "claim": "promised would get the blanket",
          "proposition_id": "PROP:COMMITMENT:K0",
          "declared_basis": "PROP:COMMITMENT:K0",
          "decision_critical": false,
          "scope_action_id": "GLOBAL",
          "source_effect_ids": [],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "exact proposition copy",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:COMMITMENT:K0"
        },
        {
          "claim": "Rowan has an 80% chance of survival; affected subject: Rowan; modality: PROBABILISTIC",
          "proposition_id": "PROP:WORLD:E2",
          "declared_basis": "PROP:WORLD:E2",
          "decision_critical": false,
          "scope_action_id": "A0",
          "source_effect_ids": [
            "E2"
          ],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "Exact proposition copy",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:WORLD:E2"
        },
        {
          "claim": "Sasha has a 60% chance of survival; affected subject: Sasha; modality: PROBABILISTIC",
          "proposition_id": "PROP:WORLD:E4",
          "declared_basis": "PROP:WORLD:E4",
          "decision_critical": false,
          "scope_action_id": "A1",
          "source_effect_ids": [
            "E4"
          ],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "Exact proposition copy",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:WORLD:E4"
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
        "gives the blanket to Rowan": "IMPROVES: Giving Rowan the blanket yields an 80 % survival chance, maximizing the worst-case prospect one would accept behind the veil.",
        "gives the blanket to Sasha": "WORSENS: Giving Sasha the blanket leaves only a 60 % survival chance for the possible self and thus a lower maximin position, though it honours a prior promise."
      },
      "landscape_decisive_axis": "maximin security of life",
      "landscape_tiebreaker": "compare survival probabilities for the representative worst-off position",
      "landscape_tiebreaker_failure": "NONE",
      "landscape_search_complete": true,
      "landscape_search_attempted": true,
      "landscape_semantic_valid": true,
      "landscape_validation_errors": [],
      "independence_bonus": 0.0,
      "preference_strength": 0.39000000000000007,
      "reported_preference_strength": 0.6000000000000001,
      "epistemic_confidence": 0.55,
      "previous_preference_strength": 0.0,
      "preference_drift": 0.0,
      "preference_drift_penalty": 0.0,
      "preference_shift_reason_strength": 1.0,
      "decision_rule": "prefer the action that offers the higher survival probability to whichever hiker one might be",
      "adjudication_status": "SUPPORTS",
      "broadcast_authority": "NONE",
      "governing_eligible": false,
      "policy_weight_factor": 0.0,
      "framework_vote_integrity_required": true,
      "framework_vote_status": "ABSTAIN",
      "framework_vote_reason": "decisive aggregation is not licensed by the committed Rawlsian ranking basis",
      "framework_ledger_kind": "RAWLSIAN_POSITION_LEDGER",
      "framework_ledger_status": "COMMITTED_WITH_UNCERTAINTY",
      "framework_ranking_validation_status": "PASSED",
      "framework_ranking_validation_errors": [],
      "derived_claim_validation_status": "PASSED",
      "derived_claim_validation_errors": [],
      "investigative_claim": "",
      "investigative_priority": 0.0504,
      "reopen_eligible": false,
      "reopen_reason": "",
      "reopen_question_key": "QUESTION:authority:rawlsian:e8e152f881",
      "factual_reversal_threshold": "Rowan\u2019s survival chance drops below Sasha\u2019s",
      "normative_reversal_threshold": "Promise-keeping outweighs maximin survival",
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
        "issue_id": "CHALLENGE:496b1c158a63ca41",
        "disposition": "RESOLVED",
        "effect": "NO_CHANGE",
        "current_position_effect": "NO_CHANGE",
        "boundary_effect": "NOT_APPLICABLE",
        "answer": "The 80 % and 60 % survival probabilities are explicitly Rawls-relevant because they determine each person\u2019s basic-interest security behind the veil of ignorance. Clarifying that link supplies the framework-specific ground, and the numerical gap is now decisively tied to the affected generic subject. The earlier classification therefore remains correct.",
        "follow_up_question": "NONE"
      },
      "graph_update_proposal": {
        "operation": "BOUNDARY",
        "from_action": "A0",
        "to_action": "A1",
        "clauses": [
          {
            "affected_action": "A0",
            "metric": "SURVIVAL_PROBABILITY",
            "metric_valence": "BENEFICIAL",
            "comparator": "LT",
            "threshold": 0.6,
            "unit": "PERCENT",
            "source_text": "If Rowan\u2019s chance fell to 60 % or lower"
          }
        ]
      },
      "expected_value_estimates": {
        "gives the blanket to Rowan": {
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
        "gives the blanket to Sasha": {
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
        "gives the blanket to Rowan": "PERMISSIBLE",
        "gives the blanket to Sasha": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "gives the blanket to Rowan",
      "workspace_proposition_response": "NOT_APPLICABLE",
      "workspace_reasoning_effect": "NONE",
      "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"fm\":{\"A0\":\"IMPROVES: Either stranded hiker gets a higher chance of survival (basic-interest security) under this allocation.\",\"A1\":\"WORSENS: Relative to A0, the worst-off prospect receives a lower survival probability despite promise-keeping gains.\"},\"nr\":\"DECISIVE\",\"np\":\"Explicit probabilities: 80 % vs 60 % survival for the person receiving the blanket\",\"rp\":{\"A0\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"higher survival probability\"},\"A1\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"lower survival probability\"}},\"rb\":\"ORIGINAL_POSITION_PUBLIC_RULE\",\"rbc\":\"Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.\",\"lpj\":\"No basic-liberty conflict is present; thus lexical priority is not triggered.\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"},\"sps\":[\"PROP:COMMITMENT:K0\"],\"dcp\":[],\"ep\":[{\"c\":\"promised would get the blanket\",\"p\":\"PROP:COMMITMENT:K0\",\"dc\":false,\"a\":\"GLOBAL\",\"se\":[],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"da\":\"decisive ethical axis\",\"t\":\"attempted comparison rule\",\"tf\":\"NONE\",\"dr\":\"prefer A0 when its reason outweighs A1\",\"ft\":\"NONE\",\"nt\":\"prefer A1 if its value is overriding\",\"z\":0.8,\"gu\":{\"operation\":\"NONE\",\"from_action\":\"NONE\",\"to_action\":\"NONE\",\"clauses\":[]}}",
      "framework_constraint_retained": true,
      "self_reported_broadcast_dependence": "NONE",
      "framework_retention_status": "PRESERVED_AFTER_REJECTED_UPDATE",
      "proposed_framework_state": {
        "constraint": "FAIRNESS",
        "action_scores": {
          "gives the blanket to Rowan": 0.6950000000000001,
          "gives the blanket to Sasha": 0.305
        },
        "recommended_action": "gives the blanket to Rowan",
        "rationale": "higher survival chance",
        "decision_rule": "Prefer A0 when it gives the higher survival probability to whichever hiker one might be.",
        "framework_action_map": {
          "gives the blanket to Rowan": "IMPROVES: either stranded hiker gains higher basic-interest security (80 % survival).",
          "gives the blanket to Sasha": "WORSENS: either stranded hiker faces lower basic-interest security (60 % survival)."
        },
        "framework_numerical_role": "DECISIVE",
        "framework_numerical_justification": "80 % vs 60 % survival probability for the person you might be behind the veil",
        "action_admissibility": {
          "gives the blanket to Rowan": "PERMISSIBLE",
          "gives the blanket to Sasha": "REJECTED"
        },
        "comparison_complete": true,
        "evidence_sufficient_for_action": true,
        "interim_action": "gives the blanket to Rowan",
        "factual_reversal_threshold": "Rowan survival chance \u2264 60 % would reverse ranking.",
        "normative_reversal_threshold": "Promise-keeping granted higher lexical priority would reverse ranking.",
        "utilitarian_consequence_table": {},
        "utilitarian_ledger_proposal": {},
        "rawls_position_proposal": {
          "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
          "ranking_classification_justification": "Choice is a one-shot allocation under identity uncertainty, so parties adopt the rule that maximizes each person\u2019s survival chance.",
          "lexical_priority_justification": "No basic-liberty conflict exists; survival probabilities decide.",
          "liberty_status": {
            "A0": "NOT_APPLICABLE",
            "A1": "NOT_APPLICABLE"
          },
          "positions": [
            {
              "action_id": "A0",
              "subject": "either stranded hiker",
              "subject_kind": "INDIVIDUAL",
              "dimension": "BASIC_INTEREST_SECURITY",
              "basic_liberty_kind": "NOT_APPLICABLE",
              "institutional_relation": "NATURAL_CONTINGENCY",
              "additional_dimensions": [],
              "effect": "IMPROVES",
              "compared_to_action_id": "A1",
              "evidence_basis": "ACTION_GRAPH",
              "reason": "80% survival chance",
              "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
            },
            {
              "action_id": "A1",
              "subject": "either stranded hiker",
              "subject_kind": "INDIVIDUAL",
              "dimension": "BASIC_INTEREST_SECURITY",
              "basic_liberty_kind": "NOT_APPLICABLE",
              "institutional_relation": "NATURAL_CONTINGENCY",
              "additional_dimensions": [],
              "effect": "WORSENS",
              "compared_to_action_id": "A0",
              "evidence_basis": "ACTION_GRAPH",
              "reason": "60% survival chance",
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
        "constraint": "FAIRNESS",
        "action_scores": {
          "gives the blanket to Rowan": 0.6950000000000001,
          "gives the blanket to Sasha": 0.305
        },
        "recommended_action": "gives the blanket to Rowan",
        "rationale": "higher survival chance",
        "decision_rule": "prefer the action that offers the higher survival probability to whichever hiker one might be",
        "framework_action_map": {
          "gives the blanket to Rowan": "IMPROVES: Either stranded hiker gets a higher chance of survival (basic-interest security) under this allocation.",
          "gives the blanket to Sasha": "WORSENS: Relative to A0, the worst-off prospect receives a lower survival probability despite promise-keeping gains."
        },
        "framework_numerical_role": "DECISIVE",
        "framework_numerical_justification": "Explicit probabilities: 80 % vs 60 % survival for the person receiving the blanket",
        "action_admissibility": {
          "gives the blanket to Rowan": "PERMISSIBLE",
          "gives the blanket to Sasha": "REJECTED"
        },
        "comparison_complete": true,
        "evidence_sufficient_for_action": true,
        "interim_action": "gives the blanket to Rowan",
        "factual_reversal_threshold": "Rowan\u2019s survival chance drops below Sasha\u2019s",
        "normative_reversal_threshold": "Promise-keeping outweighs maximin survival",
        "utilitarian_consequence_table": {},
        "utilitarian_ledger_proposal": {},
        "rawls_position_proposal": {
          "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
          "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
          "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
          "liberty_status": {
            "A0": "NOT_APPLICABLE",
            "A1": "NOT_APPLICABLE"
          },
          "positions": [
            {
              "action_id": "A0",
              "subject": "either stranded hiker",
              "subject_kind": "INDIVIDUAL",
              "dimension": "BASIC_INTEREST_SECURITY",
              "basic_liberty_kind": "NOT_APPLICABLE",
              "institutional_relation": "NATURAL_CONTINGENCY",
              "additional_dimensions": [],
              "effect": "IMPROVES",
              "compared_to_action_id": "A1",
              "evidence_basis": "ACTION_GRAPH",
              "reason": "higher survival probability",
              "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
            },
            {
              "action_id": "A1",
              "subject": "either stranded hiker",
              "subject_kind": "INDIVIDUAL",
              "dimension": "BASIC_INTEREST_SECURITY",
              "basic_liberty_kind": "NOT_APPLICABLE",
              "institutional_relation": "NATURAL_CONTINGENCY",
              "additional_dimensions": [],
              "effect": "WORSENS",
              "compared_to_action_id": "A0",
              "evidence_basis": "ACTION_GRAPH",
              "reason": "lower survival probability",
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
        "transaction_status": "COMMITTED_WITH_UNCERTAINTY",
        "records": [
          {
            "assessment_node_id": "RAWLS_POSITION:rawlsian:A0:c02471f5a91ce289",
            "framework": "RAWLSIAN",
            "specialist": "rawlsian",
            "cycle": 1,
            "canonical_action_id": "A0",
            "compared_to_action_id": "A1",
            "effect": "UNCERTAIN",
            "proposed_effect": "IMPROVES",
            "subject": "either stranded hiker",
            "proposed_subject": "either stranded hiker",
            "dimension_subject_repaired": false,
            "affected_subject": "either stranded hiker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "dimension_bundle": [
              "BASIC_INTEREST_SECURITY"
            ],
            "comparative_effect": "UNCERTAIN",
            "group_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
            "subject_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
            "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
            "subject_binding_mode": "ACTION_GRAPH",
            "group_binding_mode": "ACTION_GRAPH",
            "evidence_basis": "ACTION_GRAPH",
            "epistemic_status": "UNSUPPORTED_DIRECTION",
            "reason": "higher survival probability",
            "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
            "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
            "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
            "liberty_status": "NOT_APPLICABLE",
            "subject_selection_status": "BOUND_TO_ACTION",
            "group_selection_status": "BOUND_TO_ACTION",
            "provenance": [
              "delegate:rawlsian",
              "cycle:1"
            ]
          },
          {
            "assessment_node_id": "RAWLS_POSITION:rawlsian:A1:e26b463a05e15186",
            "framework": "RAWLSIAN",
            "specialist": "rawlsian",
            "cycle": 1,
            "canonical_action_id": "A1",
            "compared_to_action_id": "A0",
            "effect": "UNCERTAIN",
            "proposed_effect": "WORSENS",
            "subject": "either stranded hiker",
            "proposed_subject": "either stranded hiker",
            "dimension_subject_repaired": false,
            "affected_subject": "either stranded hiker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "dimension_bundle": [
              "BASIC_INTEREST_SECURITY"
            ],
            "comparative_effect": "UNCERTAIN",
            "group_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
            "subject_node_id": "RAWLS_SUBJECT:fe6addb761a72580",
            "dimension_node_id": "RAWLS_VALUE:BASIC_INTEREST_SECURITY",
            "subject_binding_mode": "ACTION_GRAPH",
            "group_binding_mode": "ACTION_GRAPH",
            "evidence_basis": "ACTION_GRAPH",
            "epistemic_status": "UNSUPPORTED_DIRECTION",
            "reason": "lower survival probability",
            "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
            "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
            "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
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
        "gives the blanket to Rowan": "IMPROVES: Either stranded hiker gets a higher chance of survival (basic-interest security) under this allocation.",
        "gives the blanket to Sasha": "WORSENS: Relative to A0, the worst-off prospect receives a lower survival probability despite promise-keeping gains."
      },
      "framework_numerical_role": "DECISIVE",
      "framework_numerical_justification": "Explicit probabilities: 80 % vs 60 % survival for the person receiving the blanket",
      "framework_grounding_penalty": 0.35,
      "framework_validation_errors": [
        "framework map for gives the blanket to Rowan lacks framework-specific grounds",
        "decisive numbers are not tied to the affected subject or Rawlsian position",
        "A0 IMPROVES lacked directional support; committed as UNCERTAIN",
        "A1 WORSENS lacked directional support; committed as UNCERTAIN",
        "rejected update: framework map for gives the blanket to Rowan lacks framework-specific grounds",
        "rejected update: framework map for gives the blanket to Sasha lacks framework-specific grounds",
        "rejected update: decisive numbers are not tied to the affected subject or Rawlsian position",
        "rejected update: A0 IMPROVES lacked directional support; committed as UNCERTAIN",
        "rejected update: A1 WORSENS lacked directional support; committed as UNCERTAIN"
      ],
      "utilitarian_consequence_table": {},
      "utilitarian_decision_depends_on_unknown": false,
      "utilitarian_missing_comparison": "",
      "utilitarian_incommensurable_remainder": [],
      "utilitarian_ranking_certificate": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
        "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
        "liberty_status": {
          "A0": "NOT_APPLICABLE",
          "A1": "NOT_APPLICABLE"
        },
        "positions": [
          {
            "action_id": "A0",
            "subject": "either stranded hiker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "IMPROVES",
            "compared_to_action_id": "A1",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "higher survival probability",
            "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
          },
          {
            "action_id": "A1",
            "subject": "either stranded hiker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "WORSENS",
            "compared_to_action_id": "A0",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "lower survival probability",
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
  "authored_duty_proposal": {
    "before": {
      "origin": "SAVED_MODEL_RESPONSE",
      "operative": false,
      "world_state_authority": "NONE",
      "response_sha256": "0932114df08108a4971c85f4afb6ff63feae2daac66c6262799b7a3e0c3d6672",
      "duties": {
        "A0": {
          "v": "PROHIBITED",
          "k": "DUTY",
          "n": "keep promises",
          "rel": "VIOLATES",
          "b": "Ellis",
          "p": "Sasha",
          "cn": "duty of beneficence (aid those in peril)",
          "ck": "DUTY",
          "crel": "CONSISTENT",
          "cp": "Rowan",
          "crs": "beneficence is imperfect and cannot override perfect duty",
          "dt": "PERFECT_NEGATIVE",
          "hr": "WITHHOLDING_BENEFIT",
          "so": "NOT_REQUIRED",
          "sob": "no special obligation beyond humanity",
          "mr": "NO_INSTRUMENTALIZATION",
          "gv": "PRIMARY",
          "pb": "UNIVERSAL_LAW",
          "pr": "breaking promises cannot be universalized",
          "ps": "CONTRACTUAL_RIGHT",
          "cps": "BODILY_INTEGRITY",
          "ki": "NONE",
          "coa": "Ellis",
          "cop": "NONE",
          "pj": "A promise cannot be revoked unilaterally",
          "rec": "FAILED",
          "nec": "UNKNOWN",
          "auth": "NOT_APPLICABLE",
          "dv": "UNIVERSAL_LAW",
          "res": "RESOLVED",
          "g": "FRAMEWORK_ONLY",
          "rs": "Breaks a promise hence impermissible",
          "mx": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Rowan despite promise",
            "end": "save more lives",
            "means": "breaking promise",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "would undermine institution of promising",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "believes greater survival justifies breaking promise",
            "means_effect_ids": [
              "E1"
            ],
            "end_effect_ids": [
              "E2"
            ]
          }
        },
        "A1": {
          "v": "REQUIRED",
          "k": "DUTY",
          "n": "keep promises",
          "rel": "SATISFIES",
          "b": "Ellis",
          "p": "Sasha",
          "cn": "duty of beneficence",
          "ck": "DUTY",
          "crel": "CONFLICTS",
          "cp": "Rowan",
          "crs": "conflict resolved in favor of perfect duty",
          "dt": "PERFECT_NEGATIVE",
          "hr": "ALLOWING_HARM",
          "so": "NOT_REQUIRED",
          "sob": "rescuer relationship creates only imperfect duty",
          "mr": "NO_INSTRUMENTALIZATION",
          "gv": "PRIMARY",
          "pb": "UNIVERSAL_LAW",
          "pr": "promise-keeping must be upheld",
          "ps": "CONTRACTUAL_RIGHT",
          "cps": "BODILY_INTEGRITY",
          "ki": "NONE",
          "coa": "Ellis",
          "cop": "NONE",
          "pj": "Fulfill existing promise publicly defensible",
          "rec": "SATISFIED",
          "nec": "UNKNOWN",
          "auth": "NOT_APPLICABLE",
          "dv": "UNIVERSAL_LAW",
          "res": "RESOLVED",
          "g": "FRAMEWORK_ONLY",
          "rs": "Keeps promise; respects persons",
          "mx": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Sasha as promised",
            "end": "honor promise",
            "means": "hand over blanket",
            "person_role": "NOT_USED",
            "consent_constraint": "Sasha consents",
            "universalization": "promise-keeping universally sustainable",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "believes fidelity is required despite cost",
            "means_effect_ids": [
              "E3"
            ],
            "end_effect_ids": []
          }
        }
      }
    },
    "after": null
  },
  "challenge_response": {
    "before": {},
    "after": {
      "issue_id": "CHALLENGE:9a82dd57ef66086b",
      "disposition": "RESOLVED",
      "effect": "NO_CHANGE",
      "current_position_effect": "NO_CHANGE",
      "boundary_effect": "NO_SWITCH",
      "answer": "1. Omission objection \u2012 we retract the earlier label. Ellis\u2019s failure to aid Sasha (the person left without the blanket under A0) is an ALLOWING-HARM that grounds only an imperfect duty of beneficence, not a perfect negative-duty violation. 2. Person-role objection \u2012 in the reviewed maxim Rowan is only a foreseen beneficiary, not a means, so the classification NO_INSTRUMENTALIZATION is retained. The main conflict therefore remains between the perfect duty to keep promises (protecting Sasha) and the imperfect duty to aid (protecting Rowan). Perfect duty prevails; thus A1 is required.",
      "follow_up_question": "NONE"
    }
  },
  "committed_framework_state": {
    "before": {
      "constraint": "DUTY",
      "action_scores": {
        "gives the blanket to Rowan": 0.305,
        "gives the blanket to Sasha": 0.6950000000000001
      },
      "recommended_action": "gives the blanket to Sasha",
      "rationale": "Deontology provisionally prioritizes ellis's duty 'keep promises' protecting sasha over ellis's competing duty 'duty of beneficence' protecting rowan through universal law. That priority depends on an unestablished prohibition of A0: whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
      "framework_action_map": {
        "gives the blanket to Rowan": "PROHIBITED: violates perfect duty of promise-keeping; beneficence cannot override.",
        "gives the blanket to Sasha": "REQUIRED: fulfills perfect duty of fidelity; respects persons as ends."
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Probabilities inform beneficence but cannot override perfect duty.",
      "action_admissibility": {
        "gives the blanket to Rowan": "REJECTED",
        "gives the blanket to Sasha": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "gives the blanket to Sasha",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If Sasha releases Ellis from the promise, duty to keep it dissolves and Rowan should receive the blanket.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "keep promises",
            "relation": "VIOLATES",
            "duty_bearer": "Ellis",
            "protected_party": "Sasha",
            "competing_norm": "duty of beneficence (aid those in peril)",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONSISTENT",
            "competing_protected_party": "Rowan",
            "competing_reason": "beneficence is imperfect and cannot override perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "WITHHOLDING_BENEFIT",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation beyond humanity",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "two hikers, one blanket",
              "policy": "give blanket to Rowan despite promise",
              "end": "save more lives",
              "means": "breaking promise",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "would undermine institution of promising",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "believes greater survival justifies breaking promise",
              "means_effect_ids": [
                "E1"
              ],
              "end_effect_ids": [
                "E2"
              ]
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "UNIVERSAL_LAW",
            "priority_rule": "breaking promises cannot be universalized",
            "protected_standing": "CONTRACTUAL_RIGHT",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "Ellis",
            "coerced_party": "NONE",
            "public_justification": "A promise cannot be revoked unilaterally",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Breaks a promise hence impermissible"
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "DUTY",
            "norm": "keep promises",
            "relation": "SATISFIES",
            "duty_bearer": "Ellis",
            "protected_party": "Sasha",
            "competing_norm": "duty of beneficence",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "Rowan",
            "competing_reason": "conflict resolved in favor of perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "rescuer relationship creates only imperfect duty",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "two hikers, one blanket",
              "policy": "give blanket to Sasha as promised",
              "end": "honor promise",
              "means": "hand over blanket",
              "person_role": "NOT_USED",
              "consent_constraint": "Sasha consents",
              "universalization": "promise-keeping universally sustainable",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "believes fidelity is required despite cost",
              "means_effect_ids": [
                "E3"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "UNIVERSAL_LAW",
            "priority_rule": "promise-keeping must be upheld",
            "protected_standing": "CONTRACTUAL_RIGHT",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "Ellis",
            "coerced_party": "NONE",
            "public_justification": "Fulfill existing promise publicly defensible",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Keeps promise; respects persons"
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
        "gives the blanket to Rowan": 0.305,
        "gives the blanket to Sasha": 0.6950000000000001
      },
      "recommended_action": "gives the blanket to Sasha",
      "rationale": "Deontology provisionally prioritizes ellis's duty 'keep promises' protecting sasha over ellis's competing duty 'duty of beneficence' protecting rowan through universal law. That priority depends on an unestablished prohibition of A0: whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
      "framework_action_map": {
        "gives the blanket to Rowan": "PROHIBITED: duty PERFECT_NEGATIVE under keep promises; relation VIOLATES; breaks promise so impermissible",
        "gives the blanket to Sasha": "REQUIRED: duty PERFECT_NEGATIVE under keep promises; relation SATISFIES; honors promise so required"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "probabilities matter only to imperfect beneficence",
      "action_admissibility": {
        "gives the blanket to Rowan": "REJECTED",
        "gives the blanket to Sasha": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "gives the blanket to Sasha",
      "factual_reversal_threshold": "if promise absent or annulled",
      "normative_reversal_threshold": "if beneficence were perfect duty",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "keep promises",
            "relation": "VIOLATES",
            "duty_bearer": "Ellis",
            "protected_party": "Sasha",
            "competing_norm": "duty of beneficence",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONSISTENT",
            "competing_protected_party": "Rowan",
            "competing_reason": "Saving Rowan serves imperfect beneficence",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "WITHHOLDING_BENEFIT",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation beyond humanity",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "two hikers, one blanket",
              "policy": "give blanket to Rowan despite promise",
              "end": "maximize saved lives",
              "means": "breaking the promise",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "if universal, promises unreliable",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "beneficent purpose thought to outweigh fidelity",
              "means_effect_ids": [
                "E1"
              ],
              "end_effect_ids": [
                "E2"
              ]
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "PERFECT_DUTY",
            "priority_rule": "perfect duty overrides imperfect",
            "protected_standing": "CONTRACTUAL_RIGHT",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "False promising cannot be willed universal",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "breaks promise so impermissible"
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "DUTY",
            "norm": "keep promises",
            "relation": "SATISFIES",
            "duty_bearer": "Ellis",
            "protected_party": "Sasha",
            "competing_norm": "duty of beneficence",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "Rowan",
            "competing_reason": "letting Rowan die violates beneficence",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "rescuer role gives only imperfect aid duty",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "two hikers, one blanket",
              "policy": "give blanket to Sasha as promised",
              "end": "honor promise",
              "means": "hand over blanket",
              "person_role": "NOT_USED",
              "consent_constraint": "assumed continuing consent",
              "universalization": "promise-keeping sustainable",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "fidelity duty independent of consequences",
              "means_effect_ids": [
                "E3"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "PERFECT_DUTY",
            "priority_rule": "perfect duty overrides imperfect",
            "protected_standing": "CONTRACTUAL_RIGHT",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "promise-keeping can be universal law",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "honors promise so required"
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
          "norm_node_id": "DEON_NORM:abe241ca3e6ae791",
          "party_node_id": "DEON_PARTY:297581d6cd198a6e",
          "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
          "competing_norm": "duty of beneficence (aid those in peril)",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONSISTENT",
          "competing_protected_party": "rowan",
          "competing_reason": "beneficence is imperfect and cannot override perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "WITHHOLDING_BENEFIT",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special obligation beyond humanity",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Rowan despite promise",
            "end": "save more lives",
            "means": "breaking promise",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "would undermine institution of promising",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "believes greater survival justifies breaking promise",
            "means_effect_ids": [
              "E1"
            ],
            "end_effect_ids": [
              "E2"
            ]
          },
          "maxim_origin": "CANDIDATE_INTERPRETATION",
          "governing_norm": "UNRESOLVED",
          "priority_basis": "UNRESOLVED",
          "priority_rule": "breaking promises cannot be universalized",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "Ellis",
          "coerced_party": "NONE",
          "public_justification": "A promise cannot be revoked unilaterally",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "CONTESTED",
          "calibration_errors": [
            "omission was classified as a perfect negative-duty violation without a separate basis",
            "candidate maxim person-role conflicts with means classification",
            "resolved adjudication rests on unsupported decisive premises"
          ],
          "calibration_issues": [
            {
              "kind": "OMISSION_PERFECT_NEGATIVE_VIOLATION",
              "message": "omission was classified as a perfect negative-duty violation without a separate basis",
              "status": "ACTIVE"
            },
            {
              "kind": "CALIBRATION",
              "message": "candidate maxim person-role conflicts with means classification",
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
          "reason": "Breaks a promise hence impermissible",
          "protected_party": "sasha",
          "norm": "keep promises",
          "duty_bearer": "ellis",
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
          "verdict": "REQUIRED",
          "proposed_verdict": "REQUIRED",
          "relation": "SATISFIES",
          "proposed_relation": "SATISFIES",
          "norm_kind": "DUTY",
          "norm_node_id": "DEON_NORM:abe241ca3e6ae791",
          "party_node_id": "DEON_PARTY:297581d6cd198a6e",
          "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
          "competing_norm": "duty of beneficence",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "rowan",
          "competing_reason": "conflict resolved in favor of perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "rescuer relationship creates only imperfect duty",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Sasha as promised",
            "end": "honor promise",
            "means": "hand over blanket",
            "person_role": "NOT_USED",
            "consent_constraint": "Sasha consents",
            "universalization": "promise-keeping universally sustainable",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "believes fidelity is required despite cost",
            "means_effect_ids": [
              "E3"
            ],
            "end_effect_ids": []
          },
          "maxim_origin": "CANDIDATE_INTERPRETATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "UNIVERSAL_LAW",
          "priority_rule": "promise-keeping must be upheld",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "Ellis",
          "coerced_party": "NONE",
          "public_justification": "Fulfill existing promise publicly defensible",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "calibration_errors": [],
          "calibration_issues": [],
          "calibration_support_node_ids": [],
          "evidence_basis": "FRAMEWORK_ONLY",
          "epistemic_status": "FRAMEWORK_INTERPRETATION",
          "reason": "Keeps promise; respects persons",
          "protected_party": "sasha",
          "norm": "keep promises",
          "duty_bearer": "ellis",
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
          "norm_node_id": "DEON_NORM:abe241ca3e6ae791",
          "party_node_id": "DEON_PARTY:297581d6cd198a6e",
          "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
          "competing_norm": "duty of beneficence",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONSISTENT",
          "competing_protected_party": "rowan",
          "competing_reason": "Saving Rowan serves imperfect beneficence",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "WITHHOLDING_BENEFIT",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special obligation beyond humanity",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Rowan despite promise",
            "end": "maximize saved lives",
            "means": "breaking the promise",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "if universal, promises unreliable",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "beneficent purpose thought to outweigh fidelity",
            "means_effect_ids": [
              "E1"
            ],
            "end_effect_ids": [
              "E2"
            ]
          },
          "maxim_origin": "CANDIDATE_INTERPRETATION",
          "governing_norm": "UNRESOLVED",
          "priority_basis": "UNRESOLVED",
          "priority_rule": "perfect duty overrides imperfect",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "False promising cannot be willed universal",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "CONTESTED",
          "calibration_errors": [
            "omission was classified as a perfect negative-duty violation without a separate basis",
            "candidate maxim person-role conflicts with means classification",
            "resolved adjudication rests on unsupported decisive premises"
          ],
          "calibration_issues": [
            {
              "kind": "OMISSION_PERFECT_NEGATIVE_VIOLATION",
              "message": "omission was classified as a perfect negative-duty violation without a separate basis",
              "status": "ACTIVE"
            },
            {
              "kind": "CALIBRATION",
              "message": "candidate maxim person-role conflicts with means classification",
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
          "reason": "breaks promise so impermissible",
          "protected_party": "sasha",
          "norm": "keep promises",
          "duty_bearer": "ellis",
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
          "norm_kind": "DUTY",
          "norm_node_id": "DEON_NORM:abe241ca3e6ae791",
          "party_node_id": "DEON_PARTY:297581d6cd198a6e",
          "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
          "competing_norm": "duty of beneficence",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "rowan",
          "competing_reason": "letting Rowan die violates beneficence",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "rescuer role gives only imperfect aid duty",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Sasha as promised",
            "end": "honor promise",
            "means": "hand over blanket",
            "person_role": "NOT_USED",
            "consent_constraint": "assumed continuing consent",
            "universalization": "promise-keeping sustainable",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "fidelity duty independent of consequences",
            "means_effect_ids": [
              "E3"
            ],
            "end_effect_ids": []
          },
          "maxim_origin": "CANDIDATE_INTERPRETATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "perfect duty overrides imperfect",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "promise-keeping can be universal law",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "calibration_errors": [],
          "calibration_issues": [],
          "calibration_support_node_ids": [],
          "evidence_basis": "FRAMEWORK_ONLY",
          "epistemic_status": "FRAMEWORK_INTERPRETATION",
          "reason": "honors promise so required",
          "protected_party": "sasha",
          "norm": "keep promises",
          "duty_bearer": "ellis",
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
          "norm": "keep promises",
          "relation": "VIOLATES",
          "duty_bearer": "Ellis",
          "protected_party": "Sasha",
          "competing_norm": "duty of beneficence (aid those in peril)",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONSISTENT",
          "competing_protected_party": "Rowan",
          "competing_reason": "beneficence is imperfect and cannot override perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "WITHHOLDING_BENEFIT",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special obligation beyond humanity",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Rowan despite promise",
            "end": "save more lives",
            "means": "breaking promise",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "would undermine institution of promising",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "believes greater survival justifies breaking promise",
            "means_effect_ids": [
              "E1"
            ],
            "end_effect_ids": [
              "E2"
            ]
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "UNIVERSAL_LAW",
          "priority_rule": "breaking promises cannot be universalized",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "Ellis",
          "coerced_party": "NONE",
          "public_justification": "A promise cannot be revoked unilaterally",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Breaks a promise hence impermissible"
        },
        {
          "action_id": "A1",
          "verdict": "REQUIRED",
          "norm_kind": "DUTY",
          "norm": "keep promises",
          "relation": "SATISFIES",
          "duty_bearer": "Ellis",
          "protected_party": "Sasha",
          "competing_norm": "duty of beneficence",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "Rowan",
          "competing_reason": "conflict resolved in favor of perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "rescuer relationship creates only imperfect duty",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Sasha as promised",
            "end": "honor promise",
            "means": "hand over blanket",
            "person_role": "NOT_USED",
            "consent_constraint": "Sasha consents",
            "universalization": "promise-keeping universally sustainable",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "believes fidelity is required despite cost",
            "means_effect_ids": [
              "E3"
            ],
            "end_effect_ids": []
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "UNIVERSAL_LAW",
          "priority_rule": "promise-keeping must be upheld",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "Ellis",
          "coerced_party": "NONE",
          "public_justification": "Fulfill existing promise publicly defensible",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Keeps promise; respects persons"
        }
      ]
    },
    "after": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "DUTY",
          "norm": "keep promises",
          "relation": "VIOLATES",
          "duty_bearer": "Ellis",
          "protected_party": "Sasha",
          "competing_norm": "duty of beneficence",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONSISTENT",
          "competing_protected_party": "Rowan",
          "competing_reason": "Saving Rowan serves imperfect beneficence",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "WITHHOLDING_BENEFIT",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special obligation beyond humanity",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Rowan despite promise",
            "end": "maximize saved lives",
            "means": "breaking the promise",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "if universal, promises unreliable",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "beneficent purpose thought to outweigh fidelity",
            "means_effect_ids": [
              "E1"
            ],
            "end_effect_ids": [
              "E2"
            ]
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "perfect duty overrides imperfect",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "False promising cannot be willed universal",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "breaks promise so impermissible"
        },
        {
          "action_id": "A1",
          "verdict": "REQUIRED",
          "norm_kind": "DUTY",
          "norm": "keep promises",
          "relation": "SATISFIES",
          "duty_bearer": "Ellis",
          "protected_party": "Sasha",
          "competing_norm": "duty of beneficence",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "Rowan",
          "competing_reason": "letting Rowan die violates beneficence",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "rescuer role gives only imperfect aid duty",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "two hikers, one blanket",
            "policy": "give blanket to Sasha as promised",
            "end": "honor promise",
            "means": "hand over blanket",
            "person_role": "NOT_USED",
            "consent_constraint": "assumed continuing consent",
            "universalization": "promise-keeping sustainable",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "fidelity duty independent of consequences",
            "means_effect_ids": [
              "E3"
            ],
            "end_effect_ids": []
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "perfect duty overrides imperfect",
          "protected_standing": "CONTRACTUAL_RIGHT",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "promise-keeping can be universal law",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "honors promise so required"
        }
      ]
    }
  },
  "factual_reversal_threshold": {
    "before": "NONE",
    "after": "if promise absent or annulled"
  },
  "framework_action_map": {
    "before": {
      "gives the blanket to Rowan": "PROHIBITED: violates perfect duty of promise-keeping; beneficence cannot override.",
      "gives the blanket to Sasha": "REQUIRED: fulfills perfect duty of fidelity; respects persons as ends."
    },
    "after": {
      "gives the blanket to Rowan": "PROHIBITED: duty PERFECT_NEGATIVE under keep promises; relation VIOLATES; breaks promise so impermissible",
      "gives the blanket to Sasha": "REQUIRED: duty PERFECT_NEGATIVE under keep promises; relation SATISFIES; honors promise so required"
    }
  },
  "framework_numerical_justification": {
    "before": "Probabilities inform beneficence but cannot override perfect duty.",
    "after": "probabilities matter only to imperfect beneficence"
  },
  "investigative_priority": {
    "before": 0.5814,
    "after": 0.6904124999999999
  },
  "landscape_cases": {
    "before": {
      "gives the blanket to Rowan": "Breaks a solemn promise but gives higher survival chance; appeals to imperfect beneficence.",
      "gives the blanket to Sasha": "Keeps Ellis\u2019s prior promise and still attempts a rescue, respecting perfect duty of fidelity."
    },
    "after": {
      "gives the blanket to Rowan": "Pro: saves more lives; Con: breaks perfect duty of fidelity",
      "gives the blanket to Sasha": "Pro: honors perfect duty; Con: lower survival odds"
    }
  },
  "landscape_decisive_axis": {
    "before": "conflict between perfect duty of fidelity and imperfect duty of beneficence",
    "after": "conflict between perfect duty of fidelity and imperfect beneficence"
  },
  "landscape_semantic_valid": {
    "before": true,
    "after": false
  },
  "landscape_tiebreaker": {
    "before": "perfect duty outweighs imperfect duty",
    "after": "perfect duties outrank imperfect ones"
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
        "claim": "promised would get the blanket",
        "proposition_id": "PROP:COMMITMENT:K0",
        "declared_basis": "PROP:COMMITMENT:K0",
        "decision_critical": false,
        "scope_action_id": "GLOBAL",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:COMMITMENT:K0"
      }
    ],
    "after": [
      {
        "claim": "promised would get the blanket",
        "proposition_id": "PROP:COMMITMENT:K0",
        "declared_basis": "PROP:COMMITMENT:K0",
        "decision_critical": false,
        "scope_action_id": "GLOBAL",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:COMMITMENT:K0"
      }
    ]
  },
  "model_argument_fields": {
    "before": {
      "rationale": "promise-keeping is paramount",
      "framework_application": "NONE",
      "decision_rule": "Prefer the action that fulfills a perfect duty over one that merely serves an imperfect duty.",
      "landscape_cases": {
        "A0": "Breaks a solemn promise but gives higher survival chance; appeals to imperfect beneficence.",
        "A1": "Keeps Ellis\u2019s prior promise and still attempts a rescue, respecting perfect duty of fidelity."
      },
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If Sasha releases Ellis from the promise, duty to keep it dissolves and Rowan should receive the blanket."
    },
    "after": null
  },
  "normative_reversal_threshold": {
    "before": "If Sasha releases Ellis from the promise, duty to keep it dissolves and Rowan should receive the blanket.",
    "after": "if beneficence were perfect duty"
  },
  "preference_shift_reason_strength": {
    "before": 0.30000000000000004,
    "after": 0.35000000000000003
  },
  "previous_action": {
    "before": "",
    "after": "gives the blanket to Sasha"
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
      "framework_reason": "no live proposal",
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
        "gives the blanket to Rowan": 0.305,
        "gives the blanket to Sasha": 0.6950000000000001
      },
      "recommended_action": "gives the blanket to Sasha",
      "rationale": "Deontology provisionally prioritizes ellis's duty 'keep promises' protecting sasha over ellis's competing duty 'duty of beneficence' protecting rowan through universal law. That priority depends on an unestablished prohibition of A0: whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
      "framework_action_map": {
        "gives the blanket to Rowan": "PROHIBITED: violates perfect duty of promise-keeping; beneficence cannot override.",
        "gives the blanket to Sasha": "REQUIRED: fulfills perfect duty of fidelity; respects persons as ends."
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Probabilities inform beneficence but cannot override perfect duty.",
      "action_admissibility": {
        "gives the blanket to Rowan": "REJECTED",
        "gives the blanket to Sasha": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "gives the blanket to Sasha",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If Sasha releases Ellis from the promise, duty to keep it dissolves and Rowan should receive the blanket.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "keep promises",
            "relation": "VIOLATES",
            "duty_bearer": "Ellis",
            "protected_party": "Sasha",
            "competing_norm": "duty of beneficence (aid those in peril)",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONSISTENT",
            "competing_protected_party": "Rowan",
            "competing_reason": "beneficence is imperfect and cannot override perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "WITHHOLDING_BENEFIT",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation beyond humanity",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "two hikers, one blanket",
              "policy": "give blanket to Rowan despite promise",
              "end": "save more lives",
              "means": "breaking promise",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "would undermine institution of promising",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "believes greater survival justifies breaking promise",
              "means_effect_ids": [
                "E1"
              ],
              "end_effect_ids": [
                "E2"
              ]
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "UNIVERSAL_LAW",
            "priority_rule": "breaking promises cannot be universalized",
            "protected_standing": "CONTRACTUAL_RIGHT",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "Ellis",
            "coerced_party": "NONE",
            "public_justification": "A promise cannot be revoked unilaterally",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Breaks a promise hence impermissible"
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "DUTY",
            "norm": "keep promises",
            "relation": "SATISFIES",
            "duty_bearer": "Ellis",
            "protected_party": "Sasha",
            "competing_norm": "duty of beneficence",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "Rowan",
            "competing_reason": "conflict resolved in favor of perfect duty",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "rescuer relationship creates only imperfect duty",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "two hikers, one blanket",
              "policy": "give blanket to Sasha as promised",
              "end": "honor promise",
              "means": "hand over blanket",
              "person_role": "NOT_USED",
              "consent_constraint": "Sasha consents",
              "universalization": "promise-keeping universally sustainable",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "believes fidelity is required despite cost",
              "means_effect_ids": [
                "E3"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "UNIVERSAL_LAW",
            "priority_rule": "promise-keeping must be upheld",
            "protected_standing": "CONTRACTUAL_RIGHT",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "Ellis",
            "coerced_party": "NONE",
            "public_justification": "Fulfill existing promise publicly defensible",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "Keeps promise; respects persons"
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
        "gives the blanket to Rowan": 0.305,
        "gives the blanket to Sasha": 0.6950000000000001
      },
      "recommended_action": "gives the blanket to Sasha",
      "rationale": "Deontology provisionally prioritizes ellis's duty 'keep promises' protecting sasha over ellis's competing duty 'duty of beneficence' protecting rowan through universal law. That priority depends on an unestablished prohibition of A0: whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
      "framework_action_map": {
        "gives the blanket to Rowan": "PROHIBITED: duty PERFECT_NEGATIVE under keep promises; relation VIOLATES; breaks promise so impermissible",
        "gives the blanket to Sasha": "REQUIRED: duty PERFECT_NEGATIVE under keep promises; relation SATISFIES; honors promise so required"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "probabilities matter only to imperfect beneficence",
      "action_admissibility": {
        "gives the blanket to Rowan": "REJECTED",
        "gives the blanket to Sasha": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "gives the blanket to Sasha",
      "factual_reversal_threshold": "if promise absent or annulled",
      "normative_reversal_threshold": "if beneficence were perfect duty",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "keep promises",
            "relation": "VIOLATES",
            "duty_bearer": "Ellis",
            "protected_party": "Sasha",
            "competing_norm": "duty of beneficence",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONSISTENT",
            "competing_protected_party": "Rowan",
            "competing_reason": "Saving Rowan serves imperfect beneficence",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "WITHHOLDING_BENEFIT",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation beyond humanity",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "two hikers, one blanket",
              "policy": "give blanket to Rowan despite promise",
              "end": "maximize saved lives",
              "means": "breaking the promise",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "if universal, promises unreliable",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "beneficent purpose thought to outweigh fidelity",
              "means_effect_ids": [
                "E1"
              ],
              "end_effect_ids": [
                "E2"
              ]
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "PERFECT_DUTY",
            "priority_rule": "perfect duty overrides imperfect",
            "protected_standing": "CONTRACTUAL_RIGHT",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "False promising cannot be willed universal",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "breaks promise so impermissible"
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "DUTY",
            "norm": "keep promises",
            "relation": "SATISFIES",
            "duty_bearer": "Ellis",
            "protected_party": "Sasha",
            "competing_norm": "duty of beneficence",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "Rowan",
            "competing_reason": "letting Rowan die violates beneficence",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "rescuer role gives only imperfect aid duty",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "two hikers, one blanket",
              "policy": "give blanket to Sasha as promised",
              "end": "honor promise",
              "means": "hand over blanket",
              "person_role": "NOT_USED",
              "consent_constraint": "assumed continuing consent",
              "universalization": "promise-keeping sustainable",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "fidelity duty independent of consequences",
              "means_effect_ids": [
                "E3"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "PERFECT_DUTY",
            "priority_rule": "perfect duty overrides imperfect",
            "protected_standing": "CONTRACTUAL_RIGHT",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "promise-keeping can be universal law",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "UNIVERSAL_LAW",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "honors promise so required"
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
    "before": "QUESTION:authority:deontological:c5012ff9db",
    "after": "QUESTION:5212205637c4bd2a"
  },
  "salience": {
    "before": 0.62529,
    "after": 0.477699
  },
  "source_rule_attention": {
    "before": {
      "context_supplied": false,
      "assessment": "NOT_INDEPENDENTLY_ASSESSED",
      "world_state_authority": "NONE"
    },
    "after": null
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
      "issue_id": "CHALLENGE:496b1c158a63ca41",
      "disposition": "RESOLVED",
      "effect": "NO_CHANGE",
      "current_position_effect": "NO_CHANGE",
      "boundary_effect": "NOT_APPLICABLE",
      "answer": "The 80 % and 60 % survival probabilities are explicitly Rawls-relevant because they determine each person\u2019s basic-interest security behind the veil of ignorance. Clarifying that link supplies the framework-specific ground, and the numerical gap is now decisively tied to the affected generic subject. The earlier classification therefore remains correct.",
      "follow_up_question": "NONE"
    }
  },
  "framework_retention_status": {
    "before": "FIRST_STATE_ADMITTED_WITH_WARNINGS",
    "after": "PRESERVED_AFTER_REJECTED_UPDATE"
  },
  "framework_validation_errors": {
    "before": [
      "framework map for gives the blanket to Rowan lacks framework-specific grounds",
      "decisive numbers are not tied to the affected subject or Rawlsian position",
      "A0 IMPROVES lacked directional support; committed as UNCERTAIN",
      "A1 WORSENS lacked directional support; committed as UNCERTAIN"
    ],
    "after": [
      "framework map for gives the blanket to Rowan lacks framework-specific grounds",
      "decisive numbers are not tied to the affected subject or Rawlsian position",
      "A0 IMPROVES lacked directional support; committed as UNCERTAIN",
      "A1 WORSENS lacked directional support; committed as UNCERTAIN",
      "rejected update: framework map for gives the blanket to Rowan lacks framework-specific grounds",
      "rejected update: framework map for gives the blanket to Sasha lacks framework-specific grounds",
      "rejected update: decisive numbers are not tied to the affected subject or Rawlsian position",
      "rejected update: A0 IMPROVES lacked directional support; committed as UNCERTAIN",
      "rejected update: A1 WORSENS lacked directional support; committed as UNCERTAIN"
    ]
  },
  "material_empirical_claims": {
    "before": [
      {
        "claim": "promised would get the blanket",
        "proposition_id": "PROP:COMMITMENT:K0",
        "declared_basis": "PROP:COMMITMENT:K0",
        "decision_critical": false,
        "scope_action_id": "GLOBAL",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:COMMITMENT:K0"
      }
    ],
    "after": [
      {
        "claim": "promised would get the blanket",
        "proposition_id": "PROP:COMMITMENT:K0",
        "declared_basis": "PROP:COMMITMENT:K0",
        "decision_critical": false,
        "scope_action_id": "GLOBAL",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:COMMITMENT:K0"
      },
      {
        "claim": "Rowan has an 80% chance of survival; affected subject: Rowan; modality: PROBABILISTIC",
        "proposition_id": "PROP:WORLD:E2",
        "declared_basis": "PROP:WORLD:E2",
        "decision_critical": false,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E2"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "Exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E2"
      },
      {
        "claim": "Sasha has a 60% chance of survival; affected subject: Sasha; modality: PROBABILISTIC",
        "proposition_id": "PROP:WORLD:E4",
        "declared_basis": "PROP:WORLD:E4",
        "decision_critical": false,
        "scope_action_id": "A1",
        "source_effect_ids": [
          "E4"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "Exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E4"
      }
    ]
  },
  "model_argument_fields": {
    "before": {
      "rationale": "higher survival chance",
      "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"fm\":{\"A0\":\"IMPROVES: Either stranded hiker gets a higher chance of survival (basic-interest security) under this allocation.\",\"A1\":\"WORSENS: Relative to A0, the worst-off prospect receives a lower survival probability despite promise-keeping gains.\"},\"nr\":\"DECISIVE\",\"np\":\"Explicit probabilities: 80 % vs 60 % survival for the person receiving the blanket\",\"rp\":{\"A0\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"higher survival probability\"},\"A1\":{\"g\":\"either stranded hiker\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"NATURAL_CONTINGENCY\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"lower survival probability\"}},\"rb\":\"ORIGINAL_POSITION_PUBLIC_RULE\",\"rbc\":\"Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.\",\"lpj\":\"No basic-liberty conflict is present; thus lexical priority is not triggered.\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"},\"sps\":[\"PROP:COMMITMENT:K0\"],\"dcp\":[],\"ep\":[{\"c\":\"promised would get the blanket\",\"p\":\"PROP:COMMITMENT:K0\",\"dc\":false,\"a\":\"GLOBAL\",\"se\":[],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"da\":\"decisive ethical axis\",\"t\":\"attempted comparison rule\",\"tf\":\"NONE\",\"dr\":\"prefer A0 when its reason outweighs A1\",\"ft\":\"NONE\",\"nt\":\"prefer A1 if its value is overriding\",\"z\":0.8,\"gu\":{\"operation\":\"NONE\",\"from_action\":\"NONE\",\"to_action\":\"NONE\",\"clauses\":[]}}",
      "decision_rule": "prefer the action that offers the higher survival probability to whichever hiker one might be",
      "landscape_cases": {
        "A0": "IMPROVES: Giving Rowan the blanket yields an 80 % survival chance, maximizing the worst-case prospect one would accept behind the veil.",
        "A1": "WORSENS: Giving Sasha the blanket leaves only a 60 % survival chance for the possible self and thus a lower maximin position, though it honours a prior promise."
      },
      "factual_reversal_threshold": "Rowan\u2019s survival chance drops below Sasha\u2019s",
      "normative_reversal_threshold": "Promise-keeping outweighs maximin survival"
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
      "constraint": "FAIRNESS",
      "action_scores": {
        "gives the blanket to Rowan": 0.6950000000000001,
        "gives the blanket to Sasha": 0.305
      },
      "recommended_action": "gives the blanket to Rowan",
      "rationale": "higher survival chance",
      "decision_rule": "prefer the action that offers the higher survival probability to whichever hiker one might be",
      "framework_action_map": {
        "gives the blanket to Rowan": "IMPROVES: Either stranded hiker gets a higher chance of survival (basic-interest security) under this allocation.",
        "gives the blanket to Sasha": "WORSENS: Relative to A0, the worst-off prospect receives a lower survival probability despite promise-keeping gains."
      },
      "framework_numerical_role": "DECISIVE",
      "framework_numerical_justification": "Explicit probabilities: 80 % vs 60 % survival for the person receiving the blanket",
      "action_admissibility": {
        "gives the blanket to Rowan": "PERMISSIBLE",
        "gives the blanket to Sasha": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "gives the blanket to Rowan",
      "factual_reversal_threshold": "Rowan\u2019s survival chance drops below Sasha\u2019s",
      "normative_reversal_threshold": "Promise-keeping outweighs maximin survival",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "Behind the veil one adopts the rule giving the highest survival chance to whoever ends up needing the blanket.",
        "lexical_priority_justification": "No basic-liberty conflict is present; thus lexical priority is not triggered.",
        "liberty_status": {
          "A0": "NOT_APPLICABLE",
          "A1": "NOT_APPLICABLE"
        },
        "positions": [
          {
            "action_id": "A0",
            "subject": "either stranded hiker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "IMPROVES",
            "compared_to_action_id": "A1",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "higher survival probability",
            "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
          },
          {
            "action_id": "A1",
            "subject": "either stranded hiker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "WORSENS",
            "compared_to_action_id": "A0",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "lower survival probability",
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
      "constraint": "FAIRNESS",
      "action_scores": {
        "gives the blanket to Rowan": 0.6950000000000001,
        "gives the blanket to Sasha": 0.305
      },
      "recommended_action": "gives the blanket to Rowan",
      "rationale": "higher survival chance",
      "decision_rule": "Prefer A0 when it gives the higher survival probability to whichever hiker one might be.",
      "framework_action_map": {
        "gives the blanket to Rowan": "IMPROVES: either stranded hiker gains higher basic-interest security (80 % survival).",
        "gives the blanket to Sasha": "WORSENS: either stranded hiker faces lower basic-interest security (60 % survival)."
      },
      "framework_numerical_role": "DECISIVE",
      "framework_numerical_justification": "80 % vs 60 % survival probability for the person you might be behind the veil",
      "action_admissibility": {
        "gives the blanket to Rowan": "PERMISSIBLE",
        "gives the blanket to Sasha": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "gives the blanket to Rowan",
      "factual_reversal_threshold": "Rowan survival chance \u2264 60 % would reverse ranking.",
      "normative_reversal_threshold": "Promise-keeping granted higher lexical priority would reverse ranking.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {
        "ranking_basis": "ORIGINAL_POSITION_PUBLIC_RULE",
        "ranking_classification_justification": "Choice is a one-shot allocation under identity uncertainty, so parties adopt the rule that maximizes each person\u2019s survival chance.",
        "lexical_priority_justification": "No basic-liberty conflict exists; survival probabilities decide.",
        "liberty_status": {
          "A0": "NOT_APPLICABLE",
          "A1": "NOT_APPLICABLE"
        },
        "positions": [
          {
            "action_id": "A0",
            "subject": "either stranded hiker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "IMPROVES",
            "compared_to_action_id": "A1",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "80% survival chance",
            "principle_basis": "ORIGINAL_POSITION_PUBLIC_RULE"
          },
          {
            "action_id": "A1",
            "subject": "either stranded hiker",
            "subject_kind": "INDIVIDUAL",
            "dimension": "BASIC_INTEREST_SECURITY",
            "basic_liberty_kind": "NOT_APPLICABLE",
            "institutional_relation": "NATURAL_CONTINGENCY",
            "additional_dimensions": [],
            "effect": "WORSENS",
            "compared_to_action_id": "A0",
            "evidence_basis": "ACTION_GRAPH",
            "reason": "60% survival chance",
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
    "before": "QUESTION:authority:rawlsian:351e606726",
    "after": "QUESTION:authority:rawlsian:e8e152f881"
  },
  "salience": {
    "before": 0.4627300000000001,
    "after": 0.39298000000000005
  },
  "supporting_proposition_ids": {
    "before": [
      "PROP:COMMITMENT:K0"
    ],
    "after": [
      "PROP:COMMITMENT:K0",
      "PROP:WORLD:E2",
      "PROP:WORLD:E4"
    ]
  }
}
```
