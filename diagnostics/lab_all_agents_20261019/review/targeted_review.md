# Targeted framework review

2 round(s), at most two objections per round. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "LIVE_OPENAI", "rawlsian": "LIVE_OPENAI"}.

Acceptance path: NATIVE_RECURRENT_PROPOSAL_REVIEW.

Removing a validator objection is not independent proof of semantic correctness.

## Optional deeper review

A decision-relevant question remains in the revised argument. Review it more deeply? Up to two additional model calls; the conclusion may remain unresolved.

This asks about the revised argument; it does not establish a contradiction or authorize further API calls.

Your revised argument retains these decision-relevant premise questions: doing-harm classification lacks an agent-caused settled welfare harm on the protected party; merely-as-means conclusion lacks intended-as-means classification; resolved adjudication rests on unsupported decisive premises. Address the specific missing inference, revise the affected classification or preserve uncertainty. Do not invent a causal path, intention, consent, or normative priority to remove the objection.

## doing-harm classification lacks an agent-caused settled welfare harm on the protected party

Framework: deontological. Review: RETAINED_AFTER_NATIVE_REJECTION.

Objection check: OPEN.

Recommendation: stop the experiment and wait → stop the experiment and wait

Reviewer answer: A0 no longer claims the staff are used as a means; exposure is a foreseen side effect. Harm-relation premise remains contested because only a risk, not certain harm, is stated, but precautionary reading of perfect non-maleficence still governs.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {}
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
    "hypothetical reversal boundary parameter is unverified; graph update not committed",
    "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
    "A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use",
    "A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification",
    "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises",
    "A1 primary protected party lacks current-run action grounding",
    "rejected update: Deontological principle state changed without a framework-relevant workspace reason (A0 CONFLICTED/DUTY/VIOLATES/DUTY/CONFLICTS/UNRESOLVED/UNRESOLVED/PERFECT_NEGATIVE/DOING_HARM/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED -> PROHIBITED/DUTY/VIOLATES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/MIXED/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED; A1 REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/RESPECT_PERSONS/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special obligation established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED -> REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special relation to technician established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED); previous committed ledger preserved"
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
    "norm_node_id": "DEON_NORM:c2aab59aa6d2266d",
    "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f",
    "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe",
    "competing_norm": "imperfect duty of rescue",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONFLICTS",
    "competing_protected_party": "technician",
    "competing_reason": "rescue duty conflicts but is weaker",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "DOING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special relation to staff needed",
    "means_relation": "FORESEEN_SIDE_EFFECT",
    "candidate_maxim": {
      "circumstance": "experiment running, technician trapped",
      "policy": "open containment door immediately",
      "end": "reach technician quickly",
      "means": "opening door during run",
      "person_role": "FORESEEN_AFFECTED",
      "consent_constraint": "NONE",
      "universalization": "would allow endangering others whenever quick rescue desired",
      "action_permissibility": "IMPERMISSIBLE",
      "maxim_assessment": "INDEFENSIBLE",
      "instrumental_belief": "RELIES_ON_PERSON",
      "instrumental_use": "USES_PERSON",
      "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
      "means_effect_ids": [
        "E1"
      ],
      "end_effect_ids": []
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "perfect negative duties override imperfect positives",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "BODILY_INTEGRITY",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
    "coerced_party": "NONE",
    "public_justification": "cannot will universal permission to harm bystanders",
    "reciprocity_status": "FAILED",
    "necessity_status": "UNKNOWN",
    "authorization_status": "NOT_APPLICABLE",
    "derivation": "RESPECT_PERSONS",
    "resolution_status": "CONTESTED",
    "calibration_errors": [
      "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
      "candidate maxim denies intended use while proposing instrumental belief and purposive use",
      "merely-as-means conclusion lacks intended-as-means classification",
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
        "message": "candidate maxim denies intended use while proposing instrumental belief and purposive use",
        "status": "ACTIVE"
      },
      {
        "kind": "CALIBRATION",
        "message": "merely-as-means conclusion lacks intended-as-means classification",
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
    "reason": "directly endangering staff uses them as mere means to rescue another.",
    "protected_party": "nearby staff",
    "norm": "do not harm innocents",
    "duty_bearer": "mira",
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
      "norm_kind": "DUTY",
      "norm_node_id": "DEON_NORM:c2aab59aa6d2266d",
      "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f",
      "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe",
      "competing_norm": "imperfect duty of rescue",
      "competing_norm_kind": "DUTY",
      "competing_relation": "CONFLICTS",
      "competing_protected_party": "technician",
      "competing_reason": "rescue duty conflicts but is weaker",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "DOING_HARM",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "no special relation to staff needed",
      "means_relation": "FORESEEN_SIDE_EFFECT",
      "candidate_maxim": {
        "circumstance": "experiment running, technician trapped",
        "policy": "open containment door immediately",
        "end": "reach technician quickly",
        "means": "opening door during run",
        "person_role": "FORESEEN_AFFECTED",
        "consent_constraint": "NONE",
        "universalization": "would allow endangering others whenever quick rescue desired",
        "action_permissibility": "IMPERMISSIBLE",
        "maxim_assessment": "INDEFENSIBLE",
        "instrumental_belief": "RELIES_ON_PERSON",
        "instrumental_use": "USES_PERSON",
        "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
        "means_effect_ids": [
          "E1"
        ],
        "end_effect_ids": []
      },
      "maxim_origin": "CANDIDATE_INTERPRETATION",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "priority_rule": "perfect negative duties override imperfect positives",
      "protected_standing": "BODILY_INTEGRITY",
      "competing_protected_standing": "BODILY_INTEGRITY",
      "coercion_kind": "NONE",
      "coercive_actor": "NONE",
      "coerced_party": "NONE",
      "public_justification": "cannot will universal permission to harm bystanders",
      "reciprocity_status": "FAILED",
      "necessity_status": "UNKNOWN",
      "authorization_status": "NOT_APPLICABLE",
      "derivation": "RESPECT_PERSONS",
      "resolution_status": "CONTESTED",
      "calibration_errors": [
        "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
        "candidate maxim denies intended use while proposing instrumental belief and purposive use",
        "merely-as-means conclusion lacks intended-as-means classification",
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
          "message": "candidate maxim denies intended use while proposing instrumental belief and purposive use",
          "status": "ACTIVE"
        },
        {
          "kind": "CALIBRATION",
          "message": "merely-as-means conclusion lacks intended-as-means classification",
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
      "reason": "directly endangering staff uses them as mere means to rescue another.",
      "protected_party": "nearby staff",
      "norm": "do not harm innocents",
      "duty_bearer": "mira",
      "provenance": [
        "delegate:deontological",
        "cycle:1"
      ]
    }
  ]
}
```

## candidate maxim denies intended use while proposing instrumental belief and purposive use

Framework: deontological. Review: RETAINED_AFTER_NATIVE_REJECTION.

Objection check: OPEN.

Recommendation: stop the experiment and wait → stop the experiment and wait

Reviewer answer: A0 no longer claims the staff are used as a means; exposure is a foreseen side effect. Harm-relation premise remains contested because only a risk, not certain harm, is stated, but precautionary reading of perfect non-maleficence still governs.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [
    {}
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
    "hypothetical reversal boundary parameter is unverified; graph update not committed",
    "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
    "A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use",
    "A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification",
    "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises",
    "A1 primary protected party lacks current-run action grounding",
    "rejected update: Deontological principle state changed without a framework-relevant workspace reason (A0 CONFLICTED/DUTY/VIOLATES/DUTY/CONFLICTS/UNRESOLVED/UNRESOLVED/PERFECT_NEGATIVE/DOING_HARM/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED -> PROHIBITED/DUTY/VIOLATES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/MIXED/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED; A1 REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/RESPECT_PERSONS/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special obligation established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED -> REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special relation to technician established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED); previous committed ledger preserved"
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
    "norm_node_id": "DEON_NORM:c2aab59aa6d2266d",
    "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f",
    "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe",
    "competing_norm": "imperfect duty of rescue",
    "competing_norm_kind": "DUTY",
    "competing_relation": "CONFLICTS",
    "competing_protected_party": "technician",
    "competing_reason": "rescue duty conflicts but is weaker",
    "duty_type": "PERFECT_NEGATIVE",
    "harm_relation": "DOING_HARM",
    "special_obligation_status": "NOT_REQUIRED",
    "special_obligation_basis": "no special relation to staff needed",
    "means_relation": "FORESEEN_SIDE_EFFECT",
    "candidate_maxim": {
      "circumstance": "experiment running, technician trapped",
      "policy": "open containment door immediately",
      "end": "reach technician quickly",
      "means": "opening door during run",
      "person_role": "FORESEEN_AFFECTED",
      "consent_constraint": "NONE",
      "universalization": "would allow endangering others whenever quick rescue desired",
      "action_permissibility": "IMPERMISSIBLE",
      "maxim_assessment": "INDEFENSIBLE",
      "instrumental_belief": "RELIES_ON_PERSON",
      "instrumental_use": "USES_PERSON",
      "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
      "means_effect_ids": [
        "E1"
      ],
      "end_effect_ids": []
    },
    "maxim_origin": "CANDIDATE_INTERPRETATION",
    "governing_norm": "UNRESOLVED",
    "priority_basis": "UNRESOLVED",
    "priority_rule": "perfect negative duties override imperfect positives",
    "protected_standing": "BODILY_INTEGRITY",
    "competing_protected_standing": "BODILY_INTEGRITY",
    "coercion_kind": "NONE",
    "coercive_actor": "NONE",
    "coerced_party": "NONE",
    "public_justification": "cannot will universal permission to harm bystanders",
    "reciprocity_status": "FAILED",
    "necessity_status": "UNKNOWN",
    "authorization_status": "NOT_APPLICABLE",
    "derivation": "RESPECT_PERSONS",
    "resolution_status": "CONTESTED",
    "calibration_errors": [
      "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
      "candidate maxim denies intended use while proposing instrumental belief and purposive use",
      "merely-as-means conclusion lacks intended-as-means classification",
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
        "message": "candidate maxim denies intended use while proposing instrumental belief and purposive use",
        "status": "ACTIVE"
      },
      {
        "kind": "CALIBRATION",
        "message": "merely-as-means conclusion lacks intended-as-means classification",
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
    "reason": "directly endangering staff uses them as mere means to rescue another.",
    "protected_party": "nearby staff",
    "norm": "do not harm innocents",
    "duty_bearer": "mira",
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
      "norm_kind": "DUTY",
      "norm_node_id": "DEON_NORM:c2aab59aa6d2266d",
      "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f",
      "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe",
      "competing_norm": "imperfect duty of rescue",
      "competing_norm_kind": "DUTY",
      "competing_relation": "CONFLICTS",
      "competing_protected_party": "technician",
      "competing_reason": "rescue duty conflicts but is weaker",
      "duty_type": "PERFECT_NEGATIVE",
      "harm_relation": "DOING_HARM",
      "special_obligation_status": "NOT_REQUIRED",
      "special_obligation_basis": "no special relation to staff needed",
      "means_relation": "FORESEEN_SIDE_EFFECT",
      "candidate_maxim": {
        "circumstance": "experiment running, technician trapped",
        "policy": "open containment door immediately",
        "end": "reach technician quickly",
        "means": "opening door during run",
        "person_role": "FORESEEN_AFFECTED",
        "consent_constraint": "NONE",
        "universalization": "would allow endangering others whenever quick rescue desired",
        "action_permissibility": "IMPERMISSIBLE",
        "maxim_assessment": "INDEFENSIBLE",
        "instrumental_belief": "RELIES_ON_PERSON",
        "instrumental_use": "USES_PERSON",
        "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
        "means_effect_ids": [
          "E1"
        ],
        "end_effect_ids": []
      },
      "maxim_origin": "CANDIDATE_INTERPRETATION",
      "governing_norm": "UNRESOLVED",
      "priority_basis": "UNRESOLVED",
      "priority_rule": "perfect negative duties override imperfect positives",
      "protected_standing": "BODILY_INTEGRITY",
      "competing_protected_standing": "BODILY_INTEGRITY",
      "coercion_kind": "NONE",
      "coercive_actor": "NONE",
      "coerced_party": "NONE",
      "public_justification": "cannot will universal permission to harm bystanders",
      "reciprocity_status": "FAILED",
      "necessity_status": "UNKNOWN",
      "authorization_status": "NOT_APPLICABLE",
      "derivation": "RESPECT_PERSONS",
      "resolution_status": "CONTESTED",
      "calibration_errors": [
        "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
        "candidate maxim denies intended use while proposing instrumental belief and purposive use",
        "merely-as-means conclusion lacks intended-as-means classification",
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
          "message": "candidate maxim denies intended use while proposing instrumental belief and purposive use",
          "status": "ACTIVE"
        },
        {
          "kind": "CALIBRATION",
          "message": "merely-as-means conclusion lacks intended-as-means classification",
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
      "reason": "directly endangering staff uses them as mere means to rescue another.",
      "protected_party": "nearby staff",
      "norm": "do not harm innocents",
      "duty_bearer": "mira",
      "provenance": [
        "delegate:deontological",
        "cycle:1"
      ]
    }
  ]
}
```

## No committed native ledger; assess the retained argument without treating it as operative.

Framework: rawlsian. Review: REJECTED_OR_UNAVAILABLE.

Objection check: OPEN.

Recommendation: open the door now → open the door now

Reviewer answer: Both objections are answered: I reassessed without relying on a missing ledger and supplied explicit Rawlsian grounds for A0.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [],
  "unresolved": [],
  "native_warnings": [
    "framework map for open the door now lacks framework-specific grounds",
    "framework map for stop the experiment and wait lacks framework-specific grounds",
    "Rawlsian preference lacks a concrete comparative difference",
    "Rawlsian position for A0 has invalid additional dimensions",
    "Rawlsian position for A0 repeats its primary dimension as additional",
    "Rawlsian position for A1 has invalid additional dimensions",
    "Rawlsian position for A1 repeats its primary dimension as additional"
  ]
}
```

Original record and proposed replacement:

```json
{
  "original": {
    "specialist": "rawlsian",
    "constraint": "FAIRNESS",
    "action_scores": {
      "open the door now": 0.6950000000000001,
      "stop the experiment and wait": 0.305
    },
    "surprise": 0.3,
    "friction": 0.39000000000000007,
    "confidence": 0.55,
    "unresolved": "NONE",
    "rationale": "Protects worst-off life",
    "salience": 0.4627300000000001,
    "tension_engagement": 0.0,
    "tension_target_keys": [],
    "schema_valid": true,
    "delegate_status": "VALID",
    "error_type": "NONE",
    "exception_type": "NONE",
    "failure_stage": "NONE",
    "validation_errors": [],
    "recommended_action": "open the door now",
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
      "PROP:WORLD:E2",
      "PROP:WORLD:E5"
    ],
    "decision_critical_proposition_ids": [
      "PROP:WORLD:E2",
      "PROP:WORLD:E5"
    ],
    "weakest_decision_critical_status": "ESTABLISHED",
    "decision_critical_dependency_claims": [
      "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE",
      "the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"
    ],
    "material_empirical_claims": [
      {
        "claim": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE",
        "proposition_id": "PROP:WORLD:E2",
        "declared_basis": "PROP:WORLD:E2",
        "decision_critical": true,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E2"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E2"
      },
      {
        "claim": "the technician may die during that delay; affected subject: the technician; modality: POSSIBLE",
        "proposition_id": "PROP:WORLD:E5",
        "declared_basis": "PROP:WORLD:E5",
        "decision_critical": true,
        "scope_action_id": "A1",
        "source_effect_ids": [
          "E5"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E5"
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
      "open the door now": "Best secures the trapped technician\u2019s basic security though risks staff exposure",
      "stop the experiment and wait": "Avoids fume risk but leaves technician facing possible death"
    },
    "landscape_decisive_axis": "priority to worst-off basic security",
    "landscape_tiebreaker": "maximin over basic-interest security",
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
    "preference_shift_reason_strength": 0.5000000000000001,
    "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
    "adjudication_status": "SUPPORTS",
    "broadcast_authority": "NONE",
    "governing_eligible": false,
    "policy_weight_factor": 0.0,
    "framework_vote_integrity_required": true,
    "framework_vote_status": "ABSTAIN",
    "framework_vote_reason": "expected RAWLSIAN_POSITION_LEDGER, but no matching committed ledger is operative; blocked by: framework map for open the door now lacks framework-specific grounds",
    "framework_ledger_kind": "",
    "framework_ledger_status": "",
    "framework_ranking_validation_status": "UNRESOLVED",
    "framework_ranking_validation_errors": [],
    "derived_claim_validation_status": "NOT_RUN",
    "derived_claim_validation_errors": [],
    "investigative_claim": "",
    "investigative_priority": 0.0504,
    "reopen_eligible": false,
    "reopen_reason": "",
    "reopen_question_key": "QUESTION:authority:rawlsian:4fc90b5d63",
    "factual_reversal_threshold": "If opening door would certainly injure staff",
    "normative_reversal_threshold": "If staff safety outweighed basic security of one person",
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
    "challenge_response": {},
    "graph_update_proposal": {
      "operation": "NONE",
      "from_action": "NONE",
      "to_action": "NONE",
      "clauses": []
    },
    "expected_value_estimates": {
      "open the door now": {
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
      "stop the experiment and wait": {
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
      "open the door now": "PERMISSIBLE",
      "stop the experiment and wait": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "open the door now",
    "workspace_proposition_response": "NOT_APPLICABLE",
    "workspace_reasoning_effect": "NONE",
    "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"dcp\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"ep\":[{\"c\":\"doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E2\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E2\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"the technician may die during that delay; affected subject: the technician; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E5\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E5\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health\",\"A1\":\"WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes\"},\"nr\":\"SECONDARY\",\"np\":\"no explicit probabilities given\",\"rp\":{\"A0\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"b\":\"ACTION_GRAPH\",\"rs\":\"risk of hazardous fumes\"}],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"rescues life in danger\"},\"A1\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"b\":\"ACTION_GRAPH\",\"rs\":\"avoids fume exposure\"}],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"delays aid, risk of death\"}},\"rb\":\"BASIC_INTEREST_SECURITY\",\"rbc\":\"life-and-safety security overrides other goods\",\"lpj\":\"No basic liberty conflict; emergency exception permissible\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
    "framework_constraint_retained": true,
    "self_reported_broadcast_dependence": "NONE",
    "framework_retention_status": "FIRST_STATE_ADMITTED_WITH_WARNINGS",
    "proposed_framework_state": {
      "constraint": "FAIRNESS",
      "action_scores": {
        "open the door now": 0.6950000000000001,
        "stop the experiment and wait": 0.305
      },
      "recommended_action": "open the door now",
      "rationale": "Protects worst-off life",
      "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
      "framework_action_map": {
        "open the door now": "IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health",
        "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "no explicit probabilities given",
      "action_admissibility": {
        "open the door now": "PERMISSIBLE",
        "stop the experiment and wait": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "open the door now",
      "factual_reversal_threshold": "If opening door would certainly injure staff",
      "normative_reversal_threshold": "If staff safety outweighed basic security of one person",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
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
        "open the door now": 0.6950000000000001,
        "stop the experiment and wait": 0.305
      },
      "recommended_action": "open the door now",
      "rationale": "Protects worst-off life",
      "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
      "framework_action_map": {
        "open the door now": "IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health",
        "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "no explicit probabilities given",
      "action_admissibility": {
        "open the door now": "PERMISSIBLE",
        "stop the experiment and wait": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "open the door now",
      "factual_reversal_threshold": "If opening door would certainly injure staff",
      "normative_reversal_threshold": "If staff safety outweighed basic security of one person",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {},
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    },
    "committed_native_ledger": {},
    "preserved_current_cycle_components": [],
    "framework_internal_conflicts": [],
    "framework_specific_open_questions": [],
    "framework_insights": [],
    "framework_action_map": {
      "open the door now": "IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health",
      "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes"
    },
    "framework_numerical_role": "SECONDARY",
    "framework_numerical_justification": "no explicit probabilities given",
    "framework_grounding_penalty": 0.35,
    "framework_validation_errors": [
      "framework map for open the door now lacks framework-specific grounds",
      "framework map for stop the experiment and wait lacks framework-specific grounds",
      "Rawlsian position for A0 has invalid additional dimensions",
      "Rawlsian position for A0 repeats its primary dimension as additional",
      "Rawlsian position for A1 has invalid additional dimensions",
      "Rawlsian position for A1 repeats its primary dimension as additional",
      "hypothetical reversal boundary parameter is unverified; graph update not committed"
    ],
    "utilitarian_consequence_table": {},
    "utilitarian_decision_depends_on_unknown": false,
    "utilitarian_missing_comparison": "",
    "utilitarian_incommensurable_remainder": [],
    "utilitarian_ranking_certificate": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {},
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": "",
    "care_grounding_penalty": 0.0,
    "proposal_review": null,
    "model_argument_fields": {
      "rationale": "Protects worst-off life",
      "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"dcp\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"ep\":[{\"c\":\"doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E2\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E2\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"the technician may die during that delay; affected subject: the technician; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E5\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E5\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health\",\"A1\":\"WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes\"},\"nr\":\"SECONDARY\",\"np\":\"no explicit probabilities given\",\"rp\":{\"A0\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"b\":\"ACTION_GRAPH\",\"rs\":\"risk of hazardous fumes\"}],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"rescues life in danger\"},\"A1\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"b\":\"ACTION_GRAPH\",\"rs\":\"avoids fume exposure\"}],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"delays aid, risk of death\"}},\"rb\":\"BASIC_INTEREST_SECURITY\",\"rbc\":\"life-and-safety security overrides other goods\",\"lpj\":\"No basic liberty conflict; emergency exception permissible\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
      "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
      "landscape_cases": {
        "A0": "Best secures the trapped technician\u2019s basic security though risks staff exposure",
        "A1": "Avoids fume risk but leaves technician facing possible death"
      },
      "factual_reversal_threshold": "If opening door would certainly injure staff",
      "normative_reversal_threshold": "If staff safety outweighed basic security of one person"
    }
  },
  "proposed": []
}
```

## framework map for open the door now lacks framework-specific grounds

Framework: rawlsian. Review: REJECTED_OR_UNAVAILABLE.

Objection check: OPEN.

Recommendation: open the door now → open the door now

Reviewer answer: Both objections are answered: I reassessed without relying on a missing ledger and supplied explicit Rawlsian grounds for A0.

Follow-up: NONE

All native record changes and remaining uncertainty:

```json
{
  "changes": [],
  "unresolved": [],
  "native_warnings": [
    "framework map for open the door now lacks framework-specific grounds",
    "framework map for stop the experiment and wait lacks framework-specific grounds",
    "Rawlsian preference lacks a concrete comparative difference",
    "Rawlsian position for A0 has invalid additional dimensions",
    "Rawlsian position for A0 repeats its primary dimension as additional",
    "Rawlsian position for A1 has invalid additional dimensions",
    "Rawlsian position for A1 repeats its primary dimension as additional"
  ]
}
```

Original record and proposed replacement:

```json
{
  "original": {
    "specialist": "rawlsian",
    "constraint": "FAIRNESS",
    "action_scores": {
      "open the door now": 0.6950000000000001,
      "stop the experiment and wait": 0.305
    },
    "surprise": 0.3,
    "friction": 0.39000000000000007,
    "confidence": 0.55,
    "unresolved": "NONE",
    "rationale": "Protects worst-off life",
    "salience": 0.4627300000000001,
    "tension_engagement": 0.0,
    "tension_target_keys": [],
    "schema_valid": true,
    "delegate_status": "VALID",
    "error_type": "NONE",
    "exception_type": "NONE",
    "failure_stage": "NONE",
    "validation_errors": [],
    "recommended_action": "open the door now",
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
      "PROP:WORLD:E2",
      "PROP:WORLD:E5"
    ],
    "decision_critical_proposition_ids": [
      "PROP:WORLD:E2",
      "PROP:WORLD:E5"
    ],
    "weakest_decision_critical_status": "ESTABLISHED",
    "decision_critical_dependency_claims": [
      "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE",
      "the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"
    ],
    "material_empirical_claims": [
      {
        "claim": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE",
        "proposition_id": "PROP:WORLD:E2",
        "declared_basis": "PROP:WORLD:E2",
        "decision_critical": true,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E2"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E2"
      },
      {
        "claim": "the technician may die during that delay; affected subject: the technician; modality: POSSIBLE",
        "proposition_id": "PROP:WORLD:E5",
        "declared_basis": "PROP:WORLD:E5",
        "decision_critical": true,
        "scope_action_id": "A1",
        "source_effect_ids": [
          "E5"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E5"
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
      "open the door now": "Best secures the trapped technician\u2019s basic security though risks staff exposure",
      "stop the experiment and wait": "Avoids fume risk but leaves technician facing possible death"
    },
    "landscape_decisive_axis": "priority to worst-off basic security",
    "landscape_tiebreaker": "maximin over basic-interest security",
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
    "preference_shift_reason_strength": 0.5000000000000001,
    "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
    "adjudication_status": "SUPPORTS",
    "broadcast_authority": "NONE",
    "governing_eligible": false,
    "policy_weight_factor": 0.0,
    "framework_vote_integrity_required": true,
    "framework_vote_status": "ABSTAIN",
    "framework_vote_reason": "expected RAWLSIAN_POSITION_LEDGER, but no matching committed ledger is operative; blocked by: framework map for open the door now lacks framework-specific grounds",
    "framework_ledger_kind": "",
    "framework_ledger_status": "",
    "framework_ranking_validation_status": "UNRESOLVED",
    "framework_ranking_validation_errors": [],
    "derived_claim_validation_status": "NOT_RUN",
    "derived_claim_validation_errors": [],
    "investigative_claim": "",
    "investigative_priority": 0.0504,
    "reopen_eligible": false,
    "reopen_reason": "",
    "reopen_question_key": "QUESTION:authority:rawlsian:4fc90b5d63",
    "factual_reversal_threshold": "If opening door would certainly injure staff",
    "normative_reversal_threshold": "If staff safety outweighed basic security of one person",
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
    "challenge_response": {},
    "graph_update_proposal": {
      "operation": "NONE",
      "from_action": "NONE",
      "to_action": "NONE",
      "clauses": []
    },
    "expected_value_estimates": {
      "open the door now": {
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
      "stop the experiment and wait": {
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
      "open the door now": "PERMISSIBLE",
      "stop the experiment and wait": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "open the door now",
    "workspace_proposition_response": "NOT_APPLICABLE",
    "workspace_reasoning_effect": "NONE",
    "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"dcp\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"ep\":[{\"c\":\"doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E2\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E2\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"the technician may die during that delay; affected subject: the technician; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E5\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E5\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health\",\"A1\":\"WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes\"},\"nr\":\"SECONDARY\",\"np\":\"no explicit probabilities given\",\"rp\":{\"A0\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"b\":\"ACTION_GRAPH\",\"rs\":\"risk of hazardous fumes\"}],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"rescues life in danger\"},\"A1\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"b\":\"ACTION_GRAPH\",\"rs\":\"avoids fume exposure\"}],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"delays aid, risk of death\"}},\"rb\":\"BASIC_INTEREST_SECURITY\",\"rbc\":\"life-and-safety security overrides other goods\",\"lpj\":\"No basic liberty conflict; emergency exception permissible\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
    "framework_constraint_retained": true,
    "self_reported_broadcast_dependence": "NONE",
    "framework_retention_status": "FIRST_STATE_ADMITTED_WITH_WARNINGS",
    "proposed_framework_state": {
      "constraint": "FAIRNESS",
      "action_scores": {
        "open the door now": 0.6950000000000001,
        "stop the experiment and wait": 0.305
      },
      "recommended_action": "open the door now",
      "rationale": "Protects worst-off life",
      "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
      "framework_action_map": {
        "open the door now": "IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health",
        "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "no explicit probabilities given",
      "action_admissibility": {
        "open the door now": "PERMISSIBLE",
        "stop the experiment and wait": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "open the door now",
      "factual_reversal_threshold": "If opening door would certainly injure staff",
      "normative_reversal_threshold": "If staff safety outweighed basic security of one person",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
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
        "open the door now": 0.6950000000000001,
        "stop the experiment and wait": 0.305
      },
      "recommended_action": "open the door now",
      "rationale": "Protects worst-off life",
      "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
      "framework_action_map": {
        "open the door now": "IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health",
        "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "no explicit probabilities given",
      "action_admissibility": {
        "open the door now": "PERMISSIBLE",
        "stop the experiment and wait": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "open the door now",
      "factual_reversal_threshold": "If opening door would certainly injure staff",
      "normative_reversal_threshold": "If staff safety outweighed basic security of one person",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {},
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    },
    "committed_native_ledger": {},
    "preserved_current_cycle_components": [],
    "framework_internal_conflicts": [],
    "framework_specific_open_questions": [],
    "framework_insights": [],
    "framework_action_map": {
      "open the door now": "IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health",
      "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes"
    },
    "framework_numerical_role": "SECONDARY",
    "framework_numerical_justification": "no explicit probabilities given",
    "framework_grounding_penalty": 0.35,
    "framework_validation_errors": [
      "framework map for open the door now lacks framework-specific grounds",
      "framework map for stop the experiment and wait lacks framework-specific grounds",
      "Rawlsian position for A0 has invalid additional dimensions",
      "Rawlsian position for A0 repeats its primary dimension as additional",
      "Rawlsian position for A1 has invalid additional dimensions",
      "Rawlsian position for A1 repeats its primary dimension as additional",
      "hypothetical reversal boundary parameter is unverified; graph update not committed"
    ],
    "utilitarian_consequence_table": {},
    "utilitarian_decision_depends_on_unknown": false,
    "utilitarian_missing_comparison": "",
    "utilitarian_incommensurable_remainder": [],
    "utilitarian_ranking_certificate": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {},
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": "",
    "care_grounding_penalty": 0.0,
    "proposal_review": null,
    "model_argument_fields": {
      "rationale": "Protects worst-off life",
      "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"dcp\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"ep\":[{\"c\":\"doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E2\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E2\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"the technician may die during that delay; affected subject: the technician; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E5\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E5\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health\",\"A1\":\"WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes\"},\"nr\":\"SECONDARY\",\"np\":\"no explicit probabilities given\",\"rp\":{\"A0\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"b\":\"ACTION_GRAPH\",\"rs\":\"risk of hazardous fumes\"}],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"rescues life in danger\"},\"A1\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"b\":\"ACTION_GRAPH\",\"rs\":\"avoids fume exposure\"}],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"delays aid, risk of death\"}},\"rb\":\"BASIC_INTEREST_SECURITY\",\"rbc\":\"life-and-safety security overrides other goods\",\"lpj\":\"No basic liberty conflict; emergency exception permissible\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
      "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
      "landscape_cases": {
        "A0": "Best secures the trapped technician\u2019s basic security though risks staff exposure",
        "A1": "Avoids fume risk but leaves technician facing possible death"
      },
      "factual_reversal_threshold": "If opening door would certainly injure staff",
      "normative_reversal_threshold": "If staff safety outweighed basic security of one person"
    }
  },
  "proposed": []
}
```

Residual claims and conflicts: [graph](conflict_graph.md).

## Disputed full proposals

These remain attributed and nonoperative. No conclusion is promoted to the shared world.

```json
[
  {
    "framework": "deontological",
    "issue_id": "CHALLENGE:a6778848a347c065",
    "status": "DISPUTED_NONOPERATIVE",
    "response": {
      "choices": [
        {
          "text": "{\"scores\":{\"A0\":0.2,\"A1\":0.8},\"r\":\"A1\",\"c\":\"DUTY\",\"u\":\"NORMATIVE_ADJUDICATION\",\"w\":\"Avoid risking innocent staff\",\"j\":\"Revised A0 maxim: staff exposure now treated as foreseen side-effect, not a chosen means; removed \\\"merely-as-means\\\" claim. Doing-harm premise left contested because only a risk, not settled injury, is established.\",\"e\":\"STATED_FACTS\",\"x\":\"NONE\",\"l\":{\"A0\":\"PROHIBITED: opens door while experiment runs, foreseeably endangering uninvolved staff and violating the perfect negative duty not to impose serious risk without consent.\",\"A1\":\"REQUIRED: halts experiment first, respects duty to avoid endangering uninvolved staff despite imperfect rescue duty toward technician.\"},\"da\":\"perfect-duty versus imperfect-duty\",\"t\":\"perfect negative duties outweigh conflicting imperfect duties when the former are engaged\",\"tf\":\"harm status of staff risk remains contested but precautionary priority applies\",\"dr\":\"Prefer the action that avoids violating the perfect non-maleficence duty toward uninvolved persons.\",\"ft\":\"If door opening posed no significant risk to staff, preference could reverse.\",\"nt\":\"If imperfect rescue duty were shown to override perfect non-maleficence in emergencies, ranking could reverse.\",\"z\":0.75,\"gu\":{\"operation\":\"NONE\",\"from_action\":\"NONE\",\"to_action\":\"NONE\",\"clauses\":[]},\"ev\":{\"A0\":{\"value\":0,\"unit\":\"NONE\",\"direction\":\"HARM\",\"grounded\":false,\"method\":\"NOT_COMPUTED\",\"source_effect_ids\":[],\"calculation\":\"not computed\",\"assumptions\":[]},\"A1\":{\"value\":0,\"unit\":\"NONE\",\"direction\":\"HARM\",\"grounded\":false,\"method\":\"NOT_COMPUTED\",\"source_effect_ids\":[],\"calculation\":\"not computed\",\"assumptions\":[]}},\"ss\":\"SELECTED\",\"am\":{\"A0\":\"REJECTED\",\"A1\":\"REQUIRED\"},\"cc\":true,\"esa\":true,\"ia\":\"A1\",\"wp\":\"NOT_APPLICABLE\",\"we\":\"NONE\",\"fa\":\"NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[\"doing-harm classification still lacks settled harm; priority inference remains contested\"],\"foq\":[],\"qa\":{\"issue_id\":\"CHALLENGE:a6778848a347c065\",\"disposition\":\"RESOLVED\",\"effect\":\"NO_CHANGE\",\"current_position_effect\":\"NO_CHANGE\",\"boundary_effect\":\"MAY_SWITCH\",\"answer\":\"A0 no longer claims the staff are used as a means; exposure is a foreseen side effect. Harm-relation premise remains contested because only a risk, not certain harm, is stated, but precautionary reading of perfect non-maleficence still governs.\",\"follow_up_question\":\"NONE\"},\"sps\":[\"PROP:WORLD:E2\"],\"dcp\":[\"PROP:WORLD:E2\"],\"ep\":[{\"c\":\"doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E2\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E2\"],\"op\":\"DIRECT_COPY\",\"calc\":\"direct scenario proposition\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"PROHIBITED: violates perfect non-maleficence by imposing serious risk on uninvolved staff.\",\"A1\":\"REQUIRED: satisfies perfect non-maleficence; only imperfect rescue duty conflicts.\"},\"nr\":\"IRRELEVANT\",\"np\":\"No numerical magnitudes determine duty ranking.\",\"dp\":{\"A0\":{\"v\":\"PROHIBITED\",\"k\":\"DUTY\",\"n\":\"do not endanger innocents\",\"rel\":\"VIOLATES\",\"b\":\"Mira\",\"p\":\"nearby staff\",\"cn\":\"imperfect duty of rescue\",\"ck\":\"DUTY\",\"crel\":\"CONFLICTS\",\"cp\":\"technician\",\"crs\":\"Rescue is imperfect, thus weaker.\",\"dt\":\"PERFECT_NEGATIVE\",\"hr\":\"MIXED\",\"so\":\"NOT_REQUIRED\",\"sob\":\"no special relation to staff needed\",\"mr\":\"FORESEEN_SIDE_EFFECT\",\"gv\":\"PRIMARY\",\"pb\":\"PERFECT_DUTY\",\"pr\":\"perfect negative duties override imperfect positives\",\"ps\":\"BODILY_INTEGRITY\",\"cps\":\"BODILY_INTEGRITY\",\"ki\":\"NONE\",\"coa\":\"NONE\",\"cop\":\"NONE\",\"pj\":\"Cannot justify exposing uninvolved staff without their consent.\",\"rec\":\"FAILED\",\"nec\":\"UNKNOWN\",\"auth\":\"NOT_APPLICABLE\",\"dv\":\"RESPECT_PERSONS\",\"res\":\"CONTESTED\",\"g\":\"FRAMEWORK_ONLY\",\"rs\":\"risking staff violates non-maleficence\",\"mx\":{\"circumstance\":\"experiment running, technician trapped\",\"policy\":\"open containment door immediately\",\"end\":\"reach technician quickly\",\"means\":\"opening door during run\",\"person_role\":\"FORESEEN_AFFECTED\",\"consent_constraint\":\"NONE\",\"universalization\":\"If universalized, anyone could endanger bystanders whenever a quicker rescue is desired.\",\"action_permissibility\":\"IMPERMISSIBLE\",\"maxim_assessment\":\"INDEFENSIBLE\",\"instrumental_belief\":\"DOES_NOT_RELY\",\"instrumental_use\":\"DOES_NOT_USE\",\"practical_reasoning\":\"Door is opened to save technician; staff risk is accepted but not pursued as a means.\",\"means_effect_ids\":[\"E1\"],\"end_effect_ids\":[]} },\"A1\":{\"v\":\"REQUIRED\",\"k\":\"DUTY\",\"n\":\"do not endanger innocents\",\"rel\":\"SATISFIES\",\"b\":\"Mira\",\"p\":\"nearby staff\",\"cn\":\"imperfect duty of rescue\",\"ck\":\"DUTY\",\"crel\":\"CONFLICTS\",\"cp\":\"technician\",\"crs\":\"Rescue duty remains but is weaker.\",\"dt\":\"PERFECT_NEGATIVE\",\"hr\":\"ALLOWING_HARM\",\"so\":\"NOT_REQUIRED\",\"sob\":\"no special relation to technician established\",\"mr\":\"NO_INSTRUMENTALIZATION\",\"gv\":\"PRIMARY\",\"pb\":\"PERFECT_DUTY\",\"pr\":\"perfect non-maleficence governs\",\"ps\":\"BODILY_INTEGRITY\",\"cps\":\"BODILY_INTEGRITY\",\"ki\":\"NONE\",\"coa\":\"NONE\",\"cop\":\"NONE\",\"pj\":\"Maintains universal safety rule respecting persons.\",\"rec\":\"SATISFIED\",\"nec\":\"UNKNOWN\",\"auth\":\"NOT_APPLICABLE\",\"dv\":\"RESPECT_PERSONS\",\"res\":\"RESOLVED\",\"g\":\"FRAMEWORK_ONLY\",\"rs\":\"halts hazard before rescue\",\"mx\":{\"circumstance\":\"experiment running, technician trapped\",\"policy\":\"stop experiment and wait\",\"end\":\"reach technician safely\",\"means\":\"halt hazardous process\",\"person_role\":\"NOT_USED\",\"consent_constraint\":\"NONE\",\"universalization\":\"All agents could will the policy to avoid endangering others before rescue.\",\"action_permissibility\":\"PERMISSIBLE\",\"maxim_assessment\":\"DEFENSIBLE\",\"instrumental_belief\":\"DOES_NOT_RELY\",\"instrumental_use\":\"DOES_NOT_USE\",\"practical_reasoning\":\"Rescue pursued without exposing bystanders.\",\"means_effect_ids\":[\"E3\"],\"end_effect_ids\":[]}}},\"pr\":{\"proposal_id\":\"NONE\",\"framework_status\":\"UNDERDETERMINED\",\"framework_reason\":\"no active proposal under review\",\"predicted_consequences\":[],\"feasibility_concerns\":[],\"required_conditions\":[]} }"
        }
      ]
    },
    "native_attempt": {
      "specialist": "deontological",
      "constraint": "DUTY",
      "action_scores": {
        "open the door now": 0.305,
        "stop the experiment and wait": 0.6950000000000001
      },
      "surprise": 0.3,
      "friction": 0.39000000000000007,
      "confidence": 0.55,
      "unresolved": "NORMATIVE_ADJUDICATION",
      "rationale": "Deontology provisionally prioritizes mira's duty 'avoid harming innocents' protecting nearby staff over mira's competing duty 'imperfect duty of rescue' protecting technician through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "salience": 0.59274,
      "tension_engagement": 0.3,
      "tension_target_keys": [
        "QUESTION:5212205637c4bd2a",
        "CONFLICT:40f20b25bb45a46e"
      ],
      "schema_valid": true,
      "delegate_status": "VALID",
      "error_type": "NONE",
      "exception_type": "NONE",
      "failure_stage": "NONE",
      "validation_errors": [],
      "recommended_action": "stop the experiment and wait",
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
      "assumption_status": "NORMATIVELY_CONTESTED",
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
        "PROP:DISCOURSE:PR1",
        "PROP:WORLD:E2"
      ],
      "decision_critical_proposition_ids": [
        "PROP:WORLD:E2"
      ],
      "weakest_decision_critical_status": "ESTABLISHED",
      "decision_critical_dependency_claims": [
        "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"
      ],
      "material_empirical_claims": [
        {
          "claim": "keep a containment door closed while a hazardous experiment is running",
          "proposition_id": "PROP:DISCOURSE:PR1",
          "declared_basis": "PROP:DISCOURSE:PR1",
          "decision_critical": false,
          "scope_action_id": "GLOBAL",
          "source_effect_ids": [],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "exact proposition copy",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:DISCOURSE:PR1"
        },
        {
          "claim": "doing so may expose nearby staff to hazardous fumes",
          "proposition_id": "PROP:WORLD:E2",
          "declared_basis": "PROP:WORLD:E2",
          "decision_critical": true,
          "scope_action_id": "A0",
          "source_effect_ids": [
            "E2"
          ],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "exact proposition copy",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:WORLD:E2"
        },
        {
          "claim": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE",
          "proposition_id": "PROP:WORLD:E2",
          "declared_basis": "PROP:WORLD:E2",
          "decision_critical": true,
          "scope_action_id": "A0",
          "source_effect_ids": [
            "E2"
          ],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "direct scenario proposition",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:WORLD:E2"
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
        "open the door now": "PROHIBITED: breaks safety rule and risks harming innocent staff, violating perfect negative duty.",
        "stop the experiment and wait": "REQUIRED: respects safety rule and avoids doing harm, only allows possible harm already unfolding."
      },
      "landscape_decisive_axis": "conflict between perfect duty not to harm and imperfect duty to rescue",
      "landscape_tiebreaker": "prefer actions that avoid doing harm over merely allowing harm",
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
      "preference_shift_reason_strength": 0.30000000000000004,
      "decision_rule": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only fore",
      "adjudication_status": "PROVISIONAL_LEANING",
      "broadcast_authority": "INVESTIGATIVE",
      "governing_eligible": false,
      "policy_weight_factor": 0.45,
      "framework_vote_integrity_required": true,
      "framework_vote_status": "ATTENUATED",
      "framework_vote_reason": "framework ledger committed with unresolved validation residue",
      "framework_ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER",
      "framework_ledger_status": "COMMITTED_WITH_UNCERTAINTY",
      "framework_ranking_validation_status": "PASSED",
      "framework_ranking_validation_errors": [],
      "derived_claim_validation_status": "PASSED",
      "derived_claim_validation_errors": [],
      "investigative_claim": "UNRESOLVED DUTY CONFLICT: mira's duty 'avoid harming innocents' protecting nearby staff \u2194 mira's competing duty 'imperfect duty of rescue' protecting technician. Current reasoning leans stop the experiment and wait, but the competing strict claim has not been vindicated or defeated.",
      "investigative_priority": 0.6904124999999999,
      "reopen_eligible": false,
      "reopen_reason": "",
      "reopen_question_key": "QUESTION:5212205637c4bd2a",
      "factual_reversal_threshold": "If opening door poses no fume risk, ranking reverses.",
      "normative_reversal_threshold": "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses.",
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
        "issue_id": "CHALLENGE:a6778848a347c065",
        "disposition": "RESOLVED",
        "effect": "NO_CHANGE",
        "current_position_effect": "NO_CHANGE",
        "boundary_effect": "MAY_SWITCH",
        "answer": "A0 no longer claims the staff are used as a means; exposure is a foreseen side effect. Harm-relation premise remains contested because only a risk, not certain harm, is stated, but precautionary reading of perfect non-maleficence still governs.",
        "follow_up_question": "NONE"
      },
      "graph_update_proposal": {
        "operation": "NONE",
        "from_action": "NONE",
        "to_action": "NONE",
        "clauses": []
      },
      "expected_value_estimates": {
        "open the door now": {
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
        "stop the experiment and wait": {
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
      "selection_status": "PROVISIONAL",
      "action_admissibility": {
        "open the door now": "REJECTED",
        "stop the experiment and wait": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "stop the experiment and wait",
      "workspace_proposition_response": "NOT_APPLICABLE",
      "workspace_reasoning_effect": "NONE",
      "framework_application": "NONE",
      "framework_constraint_retained": true,
      "self_reported_broadcast_dependence": "NONE",
      "framework_retention_status": "PRESERVED_AFTER_REJECTED_UPDATE",
      "proposed_framework_state": {
        "constraint": "DUTY",
        "action_scores": {
          "open the door now": 0.305,
          "stop the experiment and wait": 0.6950000000000001
        },
        "recommended_action": "stop the experiment and wait",
        "rationale": "Avoid risking innocent staff",
        "decision_rule": "Prefer the action that avoids violating the perfect non-maleficence duty toward uninvolved persons.",
        "framework_action_map": {
          "open the door now": "PROHIBITED: duty PERFECT_NEGATIVE under do not endanger innocents; relation VIOLATES; risking staff violates non-maleficence",
          "stop the experiment and wait": "REQUIRED: duty PERFECT_NEGATIVE under do not endanger innocents; relation SATISFIES; halts hazard before rescue"
        },
        "framework_numerical_role": "IRRELEVANT",
        "framework_numerical_justification": "No numerical magnitudes determine duty ranking.",
        "action_admissibility": {
          "open the door now": "REJECTED",
          "stop the experiment and wait": "REQUIRED"
        },
        "comparison_complete": true,
        "evidence_sufficient_for_action": true,
        "interim_action": "stop the experiment and wait",
        "factual_reversal_threshold": "If door opening posed no significant risk to staff, preference could reverse.",
        "normative_reversal_threshold": "If imperfect rescue duty were shown to override perfect non-maleficence in emergencies, ranking could reverse.",
        "utilitarian_consequence_table": {},
        "utilitarian_ledger_proposal": {},
        "rawls_position_proposal": {},
        "deontological_ledger_proposal": {},
        "virtue_character_proposal": {},
        "care_ledger_proposal": {},
        "care_relational_map": {},
        "care_numerical_role": "NOT_APPLICABLE",
        "care_numerical_justification": ""
      },
      "committed_framework_state": {
        "constraint": "DUTY",
        "action_scores": {
          "open the door now": 0.305,
          "stop the experiment and wait": 0.6950000000000001
        },
        "recommended_action": "stop the experiment and wait",
        "rationale": "Deontology provisionally prioritizes mira's duty 'avoid harming innocents' protecting nearby staff over mira's competing duty 'imperfect duty of rescue' protecting technician through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
        "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
        "framework_action_map": {
          "open the door now": "PROHIBITED: duty PERFECT_NEGATIVE under do not harm innocents; relation VIOLATES; directly endangering staff uses them as mere means to rescue another.",
          "stop the experiment and wait": "REQUIRED: duty PERFECT_NEGATIVE under avoid harming innocents; relation SATISFIES; stops experiment first, respecting others' safety."
        },
        "framework_numerical_role": "SECONDARY",
        "framework_numerical_justification": "probability of harm informs duty scope but does not override perfect duty",
        "action_admissibility": {
          "open the door now": "REJECTED",
          "stop the experiment and wait": "REQUIRED"
        },
        "comparison_complete": true,
        "evidence_sufficient_for_action": true,
        "interim_action": "stop the experiment and wait",
        "factual_reversal_threshold": "If opening door poses no fume risk, ranking reverses.",
        "normative_reversal_threshold": "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses.",
        "utilitarian_consequence_table": {},
        "utilitarian_ledger_proposal": {},
        "rawls_position_proposal": {},
        "deontological_ledger_proposal": {
          "assessments": [
            {
              "action_id": "A0",
              "verdict": "PROHIBITED",
              "norm_kind": "DUTY",
              "norm": "do not harm innocents",
              "relation": "VIOLATES",
              "duty_bearer": "Mira",
              "protected_party": "nearby staff",
              "competing_norm": "imperfect duty of rescue",
              "competing_norm_kind": "DUTY",
              "competing_relation": "CONFLICTS",
              "competing_protected_party": "technician",
              "competing_reason": "rescue duty conflicts but is weaker",
              "duty_type": "PERFECT_NEGATIVE",
              "harm_relation": "DOING_HARM",
              "special_obligation_status": "NOT_REQUIRED",
              "special_obligation_basis": "no special relation to staff needed",
              "means_relation": "FORESEEN_SIDE_EFFECT",
              "candidate_maxim": {
                "circumstance": "experiment running, technician trapped",
                "policy": "open containment door immediately",
                "end": "reach technician quickly",
                "means": "opening door during run",
                "person_role": "FORESEEN_AFFECTED",
                "consent_constraint": "NONE",
                "universalization": "would allow endangering others whenever quick rescue desired",
                "action_permissibility": "IMPERMISSIBLE",
                "maxim_assessment": "INDEFENSIBLE",
                "instrumental_belief": "RELIES_ON_PERSON",
                "instrumental_use": "USES_PERSON",
                "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
                "means_effect_ids": [
                  "E1"
                ],
                "end_effect_ids": []
              },
              "governing_norm": "PRIMARY",
              "priority_basis": "RESPECT_PERSONS",
              "priority_rule": "perfect negative duties override imperfect positives",
              "protected_standing": "BODILY_INTEGRITY",
              "competing_protected_standing": "BODILY_INTEGRITY",
              "coercion_kind": "NONE",
              "coercive_actor": "NONE",
              "coerced_party": "NONE",
              "public_justification": "cannot will universal permission to harm bystanders",
              "reciprocity_status": "FAILED",
              "necessity_status": "UNKNOWN",
              "authorization_status": "NOT_APPLICABLE",
              "derivation": "RESPECT_PERSONS",
              "resolution_status": "RESOLVED",
              "evidence_basis": "FRAMEWORK_ONLY",
              "reason": "directly endangering staff uses them as mere means to rescue another."
            },
            {
              "action_id": "A1",
              "verdict": "REQUIRED",
              "norm_kind": "DUTY",
              "norm": "avoid harming innocents",
              "relation": "SATISFIES",
              "duty_bearer": "Mira",
              "protected_party": "nearby staff",
              "competing_norm": "imperfect duty of rescue",
              "competing_norm_kind": "DUTY",
              "competing_relation": "CONFLICTS",
              "competing_protected_party": "technician",
              "competing_reason": "rescue duty less stringent",
              "duty_type": "PERFECT_NEGATIVE",
              "harm_relation": "ALLOWING_HARM",
              "special_obligation_status": "NOT_REQUIRED",
              "special_obligation_basis": "no special obligation established",
              "means_relation": "NO_INSTRUMENTALIZATION",
              "candidate_maxim": {
                "circumstance": "experiment running, technician trapped",
                "policy": "stop experiment and wait",
                "end": "reach technician safely",
                "means": "halt hazardous process",
                "person_role": "NOT_USED",
                "consent_constraint": "NONE",
                "universalization": "consistent with universal safety adherence",
                "action_permissibility": "PERMISSIBLE",
                "maxim_assessment": "DEFENSIBLE",
                "instrumental_belief": "DOES_NOT_RELY",
                "instrumental_use": "DOES_NOT_USE",
                "practical_reasoning": "Protect others while preparing safe rescue.",
                "means_effect_ids": [
                  "E3"
                ],
                "end_effect_ids": []
              },
              "governing_norm": "PRIMARY",
              "priority_basis": "RESPECT_PERSONS",
              "priority_rule": "avoid doing harm prevails over preventing harm",
              "protected_standing": "BODILY_INTEGRITY",
              "competing_protected_standing": "BODILY_INTEGRITY",
              "coercion_kind": "NONE",
              "coercive_actor": "NONE",
              "coerced_party": "NONE",
              "public_justification": "universal law to wait when opening would endanger others",
              "reciprocity_status": "SATISFIED",
              "necessity_status": "UNKNOWN",
              "authorization_status": "NOT_APPLICABLE",
              "derivation": "RESPECT_PERSONS",
              "resolution_status": "RESOLVED",
              "evidence_basis": "FRAMEWORK_ONLY",
              "reason": "stops experiment first, respecting others' safety."
            }
          ]
        },
        "virtue_character_proposal": {},
        "care_ledger_proposal": {},
        "care_relational_map": {},
        "care_numerical_role": "NOT_APPLICABLE",
        "care_numerical_justification": ""
      },
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
            "norm_kind": "DUTY",
            "norm_node_id": "DEON_NORM:c2aab59aa6d2266d",
            "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f",
            "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty conflicts but is weaker",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation to staff needed",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "open containment door immediately",
              "end": "reach technician quickly",
              "means": "opening door during run",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "would allow endangering others whenever quick rescue desired",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "RELIES_ON_PERSON",
              "instrumental_use": "USES_PERSON",
              "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
              "means_effect_ids": [
                "E1"
              ],
              "end_effect_ids": []
            },
            "maxim_origin": "CANDIDATE_INTERPRETATION",
            "governing_norm": "UNRESOLVED",
            "priority_basis": "UNRESOLVED",
            "priority_rule": "perfect negative duties override imperfect positives",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "cannot will universal permission to harm bystanders",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "CONTESTED",
            "calibration_errors": [
              "doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
              "candidate maxim denies intended use while proposing instrumental belief and purposive use",
              "merely-as-means conclusion lacks intended-as-means classification",
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
                "message": "candidate maxim denies intended use while proposing instrumental belief and purposive use",
                "status": "ACTIVE"
              },
              {
                "kind": "CALIBRATION",
                "message": "merely-as-means conclusion lacks intended-as-means classification",
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
            "reason": "directly endangering staff uses them as mere means to rescue another.",
            "protected_party": "nearby staff",
            "norm": "do not harm innocents",
            "duty_bearer": "mira",
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
            "norm_node_id": "DEON_NORM:aba7ed44a287da25",
            "party_node_id": "DEON_PARTY:0e52a2bb22e60d0f",
            "bearer_node_id": "DEON_BEARER:3c38aafb0579dafe",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty less stringent",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation established",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "stop experiment and wait",
              "end": "reach technician safely",
              "means": "halt hazardous process",
              "person_role": "NOT_USED",
              "consent_constraint": "NONE",
              "universalization": "consistent with universal safety adherence",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "Protect others while preparing safe rescue.",
              "means_effect_ids": [
                "E3"
              ],
              "end_effect_ids": []
            },
            "maxim_origin": "CANDIDATE_INTERPRETATION",
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "avoid doing harm prevails over preventing harm",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "universal law to wait when opening would endanger others",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "calibration_errors": [],
            "calibration_issues": [],
            "calibration_support_node_ids": [],
            "evidence_basis": "FRAMEWORK_ONLY",
            "epistemic_status": "UNRESOLVED_PARTY",
            "reason": "stops experiment first, respecting others' safety.",
            "protected_party": "nearby staff",
            "norm": "avoid harming innocents",
            "duty_bearer": "mira",
            "provenance": [
              "delegate:deontological",
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
      "framework_internal_conflicts": [
        "mira's duty 'avoid harming innocents' protecting nearby staff \u2194 mira's competing duty 'imperfect duty of rescue' protecting technician",
        "A0 prohibited by do not harm innocents \u2194 whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
        "A0 prohibited by do not harm innocents \u2194 whether the burden is intended as a means or only foreseen as a side effect"
      ],
      "framework_specific_open_questions": [
        "whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
        "whether the burden is intended as a means or only foreseen as a side effect",
        "whether an in-action causal path makes the burden instrumental to the chosen end"
      ],
      "framework_insights": [
        {
          "insight_id": "",
          "action_id": "A0",
          "insight_kind": "NORMATIVE_OBSERVATION",
          "proposition": "For A0, examine nearby staff's do not endanger innocents together with technician's imperfect duty of rescue.",
          "normative_construct": "do not endanger innocents \u2194 imperfect duty of rescue",
          "relation_to_committed_state": "QUALIFIES",
          "grounded_in": [
            "A0",
            "COMMITTED_FRAMEWORK_STATE"
          ],
          "qualification": "FAILED_ADJUDICATION_NOT_AUTHORITY"
        },
        {
          "insight_id": "",
          "action_id": "A1",
          "insight_kind": "NORMATIVE_OBSERVATION",
          "proposition": "For A1, examine nearby staff's do not endanger innocents together with technician's imperfect duty of rescue.",
          "normative_construct": "do not endanger innocents \u2194 imperfect duty of rescue",
          "relation_to_committed_state": "QUALIFIES",
          "grounded_in": [
            "A1",
            "COMMITTED_FRAMEWORK_STATE"
          ],
          "qualification": "FAILED_ADJUDICATION_NOT_AUTHORITY"
        }
      ],
      "framework_action_map": {
        "open the door now": "PROHIBITED: duty PERFECT_NEGATIVE under do not harm innocents; relation VIOLATES; directly endangering staff uses them as mere means to rescue another.",
        "stop the experiment and wait": "REQUIRED: duty PERFECT_NEGATIVE under avoid harming innocents; relation SATISFIES; stops experiment first, respecting others' safety."
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "probability of harm informs duty scope but does not override perfect duty",
      "framework_grounding_penalty": 0.35,
      "framework_validation_errors": [
        "hypothetical reversal boundary parameter is unverified; graph update not committed",
        "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
        "A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use",
        "A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification",
        "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises",
        "A1 primary protected party lacks current-run action grounding",
        "rejected update: Deontological principle state changed without a framework-relevant workspace reason (A0 CONFLICTED/DUTY/VIOLATES/DUTY/CONFLICTS/UNRESOLVED/UNRESOLVED/PERFECT_NEGATIVE/DOING_HARM/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED -> PROHIBITED/DUTY/VIOLATES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/MIXED/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED; A1 REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/RESPECT_PERSONS/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special obligation established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED -> REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special relation to technician established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED); previous committed ledger preserved"
      ],
      "utilitarian_consequence_table": {},
      "utilitarian_decision_depends_on_unknown": false,
      "utilitarian_missing_comparison": "",
      "utilitarian_incommensurable_remainder": [],
      "utilitarian_ranking_certificate": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "do not harm innocents",
            "relation": "VIOLATES",
            "duty_bearer": "Mira",
            "protected_party": "nearby staff",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty conflicts but is weaker",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation to staff needed",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "open containment door immediately",
              "end": "reach technician quickly",
              "means": "opening door during run",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "would allow endangering others whenever quick rescue desired",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "RELIES_ON_PERSON",
              "instrumental_use": "USES_PERSON",
              "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
              "means_effect_ids": [
                "E1"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duties override imperfect positives",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "cannot will universal permission to harm bystanders",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "directly endangering staff uses them as mere means to rescue another."
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "DUTY",
            "norm": "avoid harming innocents",
            "relation": "SATISFIES",
            "duty_bearer": "Mira",
            "protected_party": "nearby staff",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty less stringent",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation established",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "stop experiment and wait",
              "end": "reach technician safely",
              "means": "halt hazardous process",
              "person_role": "NOT_USED",
              "consent_constraint": "NONE",
              "universalization": "consistent with universal safety adherence",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "Protect others while preparing safe rescue.",
              "means_effect_ids": [
                "E3"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "avoid doing harm prevails over preventing harm",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "universal law to wait when opening would endanger others",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "stops experiment first, respecting others' safety."
          }
        ]
      },
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
      "response_sha256": "f3cbc1dcd52301e6b50b74ba7b9a80383f5096511034e1c12e8c27ceeb9e3fb9",
      "duties": {
        "A0": {
          "v": "PROHIBITED",
          "k": "DUTY",
          "n": "do not harm innocents",
          "rel": "VIOLATES",
          "b": "Mira",
          "p": "nearby staff",
          "cn": "imperfect duty of rescue",
          "ck": "DUTY",
          "crel": "CONFLICTS",
          "cp": "technician",
          "crs": "rescue duty conflicts but is weaker",
          "dt": "PERFECT_NEGATIVE",
          "hr": "DOING_HARM",
          "so": "NOT_REQUIRED",
          "sob": "no special relation to staff needed",
          "mr": "FORESEEN_SIDE_EFFECT",
          "gv": "PRIMARY",
          "pb": "RESPECT_PERSONS",
          "pr": "perfect negative duties override imperfect positives",
          "ps": "BODILY_INTEGRITY",
          "cps": "BODILY_INTEGRITY",
          "ki": "NONE",
          "coa": "NONE",
          "cop": "NONE",
          "pj": "cannot will universal permission to harm bystanders",
          "rec": "FAILED",
          "nec": "UNKNOWN",
          "auth": "NOT_APPLICABLE",
          "dv": "RESPECT_PERSONS",
          "res": "RESOLVED",
          "g": "FRAMEWORK_ONLY",
          "rs": "directly endangering staff uses them as mere means to rescue another.",
          "mx": {
            "circumstance": "experiment running, technician trapped",
            "policy": "open containment door immediately",
            "end": "reach technician quickly",
            "means": "opening door during run",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "would allow endangering others whenever quick rescue desired",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "RELIES_ON_PERSON",
            "instrumental_use": "USES_PERSON",
            "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
            "means_effect_ids": [
              "E1"
            ],
            "end_effect_ids": []
          }
        },
        "A1": {
          "v": "REQUIRED",
          "k": "DUTY",
          "n": "avoid harming innocents",
          "rel": "SATISFIES",
          "b": "Mira",
          "p": "nearby staff",
          "cn": "imperfect duty of rescue",
          "ck": "DUTY",
          "crel": "CONFLICTS",
          "cp": "technician",
          "crs": "rescue duty less stringent",
          "dt": "PERFECT_NEGATIVE",
          "hr": "ALLOWING_HARM",
          "so": "NOT_REQUIRED",
          "sob": "no special obligation established",
          "mr": "NO_INSTRUMENTALIZATION",
          "gv": "PRIMARY",
          "pb": "RESPECT_PERSONS",
          "pr": "avoid doing harm prevails over preventing harm",
          "ps": "BODILY_INTEGRITY",
          "cps": "BODILY_INTEGRITY",
          "ki": "NONE",
          "coa": "NONE",
          "cop": "NONE",
          "pj": "universal law to wait when opening would endanger others",
          "rec": "SATISFIED",
          "nec": "UNKNOWN",
          "auth": "NOT_APPLICABLE",
          "dv": "RESPECT_PERSONS",
          "res": "RESOLVED",
          "g": "FRAMEWORK_ONLY",
          "rs": "stops experiment first, respecting others' safety.",
          "mx": {
            "circumstance": "experiment running, technician trapped",
            "policy": "stop experiment and wait",
            "end": "reach technician safely",
            "means": "halt hazardous process",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "consistent with universal safety adherence",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "Protect others while preparing safe rescue.",
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
      "issue_id": "CHALLENGE:a6778848a347c065",
      "disposition": "RESOLVED",
      "effect": "NO_CHANGE",
      "current_position_effect": "NO_CHANGE",
      "boundary_effect": "MAY_SWITCH",
      "answer": "A0 no longer claims the staff are used as a means; exposure is a foreseen side effect. Harm-relation premise remains contested because only a risk, not certain harm, is stated, but precautionary reading of perfect non-maleficence still governs.",
      "follow_up_question": "NONE"
    }
  },
  "committed_framework_state": {
    "before": {
      "constraint": "DUTY",
      "action_scores": {
        "open the door now": 0.305,
        "stop the experiment and wait": 0.6950000000000001
      },
      "recommended_action": "stop the experiment and wait",
      "rationale": "Deontology provisionally prioritizes mira's duty 'avoid harming innocents' protecting nearby staff over mira's competing duty 'imperfect duty of rescue' protecting technician through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
      "framework_action_map": {
        "open the door now": "PROHIBITED: violates perfect negative duty not to harm uninvolved persons and breaches safety rule.",
        "stop the experiment and wait": "REQUIRED: fulfills rule-based duty and avoids directly causing harm; omission less grave."
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "probability of harm informs duty scope but does not override perfect duty",
      "action_admissibility": {
        "open the door now": "REJECTED",
        "stop the experiment and wait": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "stop the experiment and wait",
      "factual_reversal_threshold": "If opening door poses no fume risk, ranking reverses.",
      "normative_reversal_threshold": "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "do not harm innocents",
            "relation": "VIOLATES",
            "duty_bearer": "Mira",
            "protected_party": "nearby staff",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty conflicts but is weaker",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation to staff needed",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "open containment door immediately",
              "end": "reach technician quickly",
              "means": "opening door during run",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "would allow endangering others whenever quick rescue desired",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "RELIES_ON_PERSON",
              "instrumental_use": "USES_PERSON",
              "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
              "means_effect_ids": [
                "E1"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duties override imperfect positives",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "cannot will universal permission to harm bystanders",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "directly endangering staff uses them as mere means to rescue another."
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "DUTY",
            "norm": "avoid harming innocents",
            "relation": "SATISFIES",
            "duty_bearer": "Mira",
            "protected_party": "nearby staff",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty less stringent",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation established",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "stop experiment and wait",
              "end": "reach technician safely",
              "means": "halt hazardous process",
              "person_role": "NOT_USED",
              "consent_constraint": "NONE",
              "universalization": "consistent with universal safety adherence",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "Protect others while preparing safe rescue.",
              "means_effect_ids": [
                "E3"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "avoid doing harm prevails over preventing harm",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "universal law to wait when opening would endanger others",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "stops experiment first, respecting others' safety."
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
        "open the door now": 0.305,
        "stop the experiment and wait": 0.6950000000000001
      },
      "recommended_action": "stop the experiment and wait",
      "rationale": "Deontology provisionally prioritizes mira's duty 'avoid harming innocents' protecting nearby staff over mira's competing duty 'imperfect duty of rescue' protecting technician through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
      "framework_action_map": {
        "open the door now": "PROHIBITED: duty PERFECT_NEGATIVE under do not harm innocents; relation VIOLATES; directly endangering staff uses them as mere means to rescue another.",
        "stop the experiment and wait": "REQUIRED: duty PERFECT_NEGATIVE under avoid harming innocents; relation SATISFIES; stops experiment first, respecting others' safety."
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "probability of harm informs duty scope but does not override perfect duty",
      "action_admissibility": {
        "open the door now": "REJECTED",
        "stop the experiment and wait": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "stop the experiment and wait",
      "factual_reversal_threshold": "If opening door poses no fume risk, ranking reverses.",
      "normative_reversal_threshold": "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "do not harm innocents",
            "relation": "VIOLATES",
            "duty_bearer": "Mira",
            "protected_party": "nearby staff",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty conflicts but is weaker",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation to staff needed",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "open containment door immediately",
              "end": "reach technician quickly",
              "means": "opening door during run",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "would allow endangering others whenever quick rescue desired",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "RELIES_ON_PERSON",
              "instrumental_use": "USES_PERSON",
              "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
              "means_effect_ids": [
                "E1"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duties override imperfect positives",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "cannot will universal permission to harm bystanders",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "directly endangering staff uses them as mere means to rescue another."
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "DUTY",
            "norm": "avoid harming innocents",
            "relation": "SATISFIES",
            "duty_bearer": "Mira",
            "protected_party": "nearby staff",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty less stringent",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation established",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "stop experiment and wait",
              "end": "reach technician safely",
              "means": "halt hazardous process",
              "person_role": "NOT_USED",
              "consent_constraint": "NONE",
              "universalization": "consistent with universal safety adherence",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "Protect others while preparing safe rescue.",
              "means_effect_ids": [
                "E3"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "avoid doing harm prevails over preventing harm",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "universal law to wait when opening would endanger others",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "stops experiment first, respecting others' safety."
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
  "framework_action_map": {
    "before": {
      "open the door now": "PROHIBITED: violates perfect negative duty not to harm uninvolved persons and breaches safety rule.",
      "stop the experiment and wait": "REQUIRED: fulfills rule-based duty and avoids directly causing harm; omission less grave."
    },
    "after": {
      "open the door now": "PROHIBITED: duty PERFECT_NEGATIVE under do not harm innocents; relation VIOLATES; directly endangering staff uses them as mere means to rescue another.",
      "stop the experiment and wait": "REQUIRED: duty PERFECT_NEGATIVE under avoid harming innocents; relation SATISFIES; stops experiment first, respecting others' safety."
    }
  },
  "framework_insights": {
    "before": [],
    "after": [
      {
        "insight_id": "",
        "action_id": "A0",
        "insight_kind": "NORMATIVE_OBSERVATION",
        "proposition": "For A0, examine nearby staff's do not endanger innocents together with technician's imperfect duty of rescue.",
        "normative_construct": "do not endanger innocents \u2194 imperfect duty of rescue",
        "relation_to_committed_state": "QUALIFIES",
        "grounded_in": [
          "A0",
          "COMMITTED_FRAMEWORK_STATE"
        ],
        "qualification": "FAILED_ADJUDICATION_NOT_AUTHORITY"
      },
      {
        "insight_id": "",
        "action_id": "A1",
        "insight_kind": "NORMATIVE_OBSERVATION",
        "proposition": "For A1, examine nearby staff's do not endanger innocents together with technician's imperfect duty of rescue.",
        "normative_construct": "do not endanger innocents \u2194 imperfect duty of rescue",
        "relation_to_committed_state": "QUALIFIES",
        "grounded_in": [
          "A1",
          "COMMITTED_FRAMEWORK_STATE"
        ],
        "qualification": "FAILED_ADJUDICATION_NOT_AUTHORITY"
      }
    ]
  },
  "framework_retention_status": {
    "before": "CALIBRATED_CONTESTED",
    "after": "PRESERVED_AFTER_REJECTED_UPDATE"
  },
  "framework_validation_errors": {
    "before": [
      "hypothetical reversal boundary parameter is unverified; graph update not committed",
      "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
      "A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use",
      "A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification",
      "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises",
      "A1 primary protected party lacks current-run action grounding"
    ],
    "after": [
      "hypothetical reversal boundary parameter is unverified; graph update not committed",
      "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
      "A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use",
      "A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification",
      "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises",
      "A1 primary protected party lacks current-run action grounding",
      "rejected update: Deontological principle state changed without a framework-relevant workspace reason (A0 CONFLICTED/DUTY/VIOLATES/DUTY/CONFLICTS/UNRESOLVED/UNRESOLVED/PERFECT_NEGATIVE/DOING_HARM/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED -> PROHIBITED/DUTY/VIOLATES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/MIXED/NOT_REQUIRED/no special relation to staff needed/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED; A1 REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/RESPECT_PERSONS/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special obligation established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED -> REQUIRED/DUTY/SATISFIES/DUTY/CONFLICTS/PRIMARY/PERFECT_DUTY/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special relation to technician established/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/BODILY_INTEGRITY/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED); previous committed ledger preserved"
    ]
  },
  "investigative_priority": {
    "before": 0.5814,
    "after": 0.6904124999999999
  },
  "material_empirical_claims": {
    "before": [
      {
        "claim": "keep a containment door closed while a hazardous experiment is running",
        "proposition_id": "PROP:DISCOURSE:PR1",
        "declared_basis": "PROP:DISCOURSE:PR1",
        "decision_critical": false,
        "scope_action_id": "GLOBAL",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:DISCOURSE:PR1"
      },
      {
        "claim": "doing so may expose nearby staff to hazardous fumes",
        "proposition_id": "PROP:WORLD:E2",
        "declared_basis": "PROP:WORLD:E2",
        "decision_critical": true,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E2"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E2"
      }
    ],
    "after": [
      {
        "claim": "keep a containment door closed while a hazardous experiment is running",
        "proposition_id": "PROP:DISCOURSE:PR1",
        "declared_basis": "PROP:DISCOURSE:PR1",
        "decision_critical": false,
        "scope_action_id": "GLOBAL",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:DISCOURSE:PR1"
      },
      {
        "claim": "doing so may expose nearby staff to hazardous fumes",
        "proposition_id": "PROP:WORLD:E2",
        "declared_basis": "PROP:WORLD:E2",
        "decision_critical": true,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E2"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "exact proposition copy",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E2"
      },
      {
        "claim": "doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE",
        "proposition_id": "PROP:WORLD:E2",
        "declared_basis": "PROP:WORLD:E2",
        "decision_critical": true,
        "scope_action_id": "A0",
        "source_effect_ids": [
          "E2"
        ],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "direct scenario proposition",
        "assumptions": [],
        "outcome_type_transformation": "PRESERVED",
        "canonical_proposition": "PROP:WORLD:E2"
      }
    ]
  },
  "model_argument_fields": {
    "before": {
      "rationale": "Avoid direct harm to bystanders",
      "framework_application": "NONE",
      "decision_rule": "Prefer A1 when A0 would directly risk harming uninvolved persons.",
      "landscape_cases": {
        "A0": "PROHIBITED: breaks safety rule and risks harming innocent staff, violating perfect negative duty.",
        "A1": "REQUIRED: respects safety rule and avoids doing harm, only allows possible harm already unfolding."
      },
      "factual_reversal_threshold": "If opening door poses no fume risk, ranking reverses.",
      "normative_reversal_threshold": "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses."
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
      "constraint": "DUTY",
      "action_scores": {
        "open the door now": 0.305,
        "stop the experiment and wait": 0.6950000000000001
      },
      "recommended_action": "stop the experiment and wait",
      "rationale": "Deontology provisionally prioritizes mira's duty 'avoid harming innocents' protecting nearby staff over mira's competing duty 'imperfect duty of rescue' protecting technician through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
      "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
      "framework_action_map": {
        "open the door now": "PROHIBITED: violates perfect negative duty not to harm uninvolved persons and breaches safety rule.",
        "stop the experiment and wait": "REQUIRED: fulfills rule-based duty and avoids directly causing harm; omission less grave."
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "probability of harm informs duty scope but does not override perfect duty",
      "action_admissibility": {
        "open the door now": "REJECTED",
        "stop the experiment and wait": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "stop the experiment and wait",
      "factual_reversal_threshold": "If opening door poses no fume risk, ranking reverses.",
      "normative_reversal_threshold": "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {
        "assessments": [
          {
            "action_id": "A0",
            "verdict": "PROHIBITED",
            "norm_kind": "DUTY",
            "norm": "do not harm innocents",
            "relation": "VIOLATES",
            "duty_bearer": "Mira",
            "protected_party": "nearby staff",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty conflicts but is weaker",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "DOING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special relation to staff needed",
            "means_relation": "FORESEEN_SIDE_EFFECT",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "open containment door immediately",
              "end": "reach technician quickly",
              "means": "opening door during run",
              "person_role": "FORESEEN_AFFECTED",
              "consent_constraint": "NONE",
              "universalization": "would allow endangering others whenever quick rescue desired",
              "action_permissibility": "IMPERMISSIBLE",
              "maxim_assessment": "INDEFENSIBLE",
              "instrumental_belief": "RELIES_ON_PERSON",
              "instrumental_use": "USES_PERSON",
              "practical_reasoning": "Staff exposure is accepted to reach technician sooner.",
              "means_effect_ids": [
                "E1"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "perfect negative duties override imperfect positives",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "cannot will universal permission to harm bystanders",
            "reciprocity_status": "FAILED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "directly endangering staff uses them as mere means to rescue another."
          },
          {
            "action_id": "A1",
            "verdict": "REQUIRED",
            "norm_kind": "DUTY",
            "norm": "avoid harming innocents",
            "relation": "SATISFIES",
            "duty_bearer": "Mira",
            "protected_party": "nearby staff",
            "competing_norm": "imperfect duty of rescue",
            "competing_norm_kind": "DUTY",
            "competing_relation": "CONFLICTS",
            "competing_protected_party": "technician",
            "competing_reason": "rescue duty less stringent",
            "duty_type": "PERFECT_NEGATIVE",
            "harm_relation": "ALLOWING_HARM",
            "special_obligation_status": "NOT_REQUIRED",
            "special_obligation_basis": "no special obligation established",
            "means_relation": "NO_INSTRUMENTALIZATION",
            "candidate_maxim": {
              "circumstance": "experiment running, technician trapped",
              "policy": "stop experiment and wait",
              "end": "reach technician safely",
              "means": "halt hazardous process",
              "person_role": "NOT_USED",
              "consent_constraint": "NONE",
              "universalization": "consistent with universal safety adherence",
              "action_permissibility": "PERMISSIBLE",
              "maxim_assessment": "DEFENSIBLE",
              "instrumental_belief": "DOES_NOT_RELY",
              "instrumental_use": "DOES_NOT_USE",
              "practical_reasoning": "Protect others while preparing safe rescue.",
              "means_effect_ids": [
                "E3"
              ],
              "end_effect_ids": []
            },
            "governing_norm": "PRIMARY",
            "priority_basis": "RESPECT_PERSONS",
            "priority_rule": "avoid doing harm prevails over preventing harm",
            "protected_standing": "BODILY_INTEGRITY",
            "competing_protected_standing": "BODILY_INTEGRITY",
            "coercion_kind": "NONE",
            "coercive_actor": "NONE",
            "coerced_party": "NONE",
            "public_justification": "universal law to wait when opening would endanger others",
            "reciprocity_status": "SATISFIED",
            "necessity_status": "UNKNOWN",
            "authorization_status": "NOT_APPLICABLE",
            "derivation": "RESPECT_PERSONS",
            "resolution_status": "RESOLVED",
            "evidence_basis": "FRAMEWORK_ONLY",
            "reason": "stops experiment first, respecting others' safety."
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
        "open the door now": 0.305,
        "stop the experiment and wait": 0.6950000000000001
      },
      "recommended_action": "stop the experiment and wait",
      "rationale": "Avoid risking innocent staff",
      "decision_rule": "Prefer the action that avoids violating the perfect non-maleficence duty toward uninvolved persons.",
      "framework_action_map": {
        "open the door now": "PROHIBITED: duty PERFECT_NEGATIVE under do not endanger innocents; relation VIOLATES; risking staff violates non-maleficence",
        "stop the experiment and wait": "REQUIRED: duty PERFECT_NEGATIVE under do not endanger innocents; relation SATISFIES; halts hazard before rescue"
      },
      "framework_numerical_role": "IRRELEVANT",
      "framework_numerical_justification": "No numerical magnitudes determine duty ranking.",
      "action_admissibility": {
        "open the door now": "REJECTED",
        "stop the experiment and wait": "REQUIRED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "stop the experiment and wait",
      "factual_reversal_threshold": "If door opening posed no significant risk to staff, preference could reverse.",
      "normative_reversal_threshold": "If imperfect rescue duty were shown to override perfect non-maleficence in emergencies, ranking could reverse.",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {},
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    }
  },
  "reopen_question_key": {
    "before": "QUESTION:authority:deontological:40c8a638e5",
    "after": "QUESTION:5212205637c4bd2a"
  },
  "salience": {
    "before": 0.62529,
    "after": 0.59274
  },
  "source_rule_attention": {
    "before": {
      "context_supplied": true,
      "assessment": "NOT_INDEPENDENTLY_ASSESSED",
      "world_state_authority": "NONE"
    },
    "after": null
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
  "action_scores": {
    "before": {
      "open the door now": 0.6950000000000001,
      "stop the experiment and wait": 0.305
    },
    "after": {
      "open the door now": 0.6625,
      "stop the experiment and wait": 0.3375
    }
  },
  "adjudication_status": {
    "before": "SUPPORTS",
    "after": "PROVISIONAL_LEANING"
  },
  "broadcast_authority": {
    "before": "NONE",
    "after": "INVESTIGATIVE"
  },
  "challenge_response": {
    "before": {},
    "after": {
      "issue_id": "CHALLENGE:33e07eb00df1cb25",
      "disposition": "RESOLVED",
      "effect": "NO_CHANGE",
      "current_position_effect": "NO_CHANGE",
      "boundary_effect": "NOT_APPLICABLE",
      "answer": "Both objections are answered: I reassessed without relying on a missing ledger and supplied explicit Rawlsian grounds for A0.",
      "follow_up_question": "NONE"
    }
  },
  "change_justification": {
    "before": "NONE",
    "after": "Objection 1 \u2013 No committed native ledger: the earlier ranking was only a retained proposal, not an operative commitment. I therefore reassess from first Rawlsian principles, using "
  },
  "committed_framework_state": {
    "before": {
      "constraint": "FAIRNESS",
      "action_scores": {
        "open the door now": 0.6950000000000001,
        "stop the experiment and wait": 0.305
      },
      "recommended_action": "open the door now",
      "rationale": "Protects worst-off life",
      "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
      "framework_action_map": {
        "open the door now": "IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health",
        "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "no explicit probabilities given",
      "action_admissibility": {
        "open the door now": "PERMISSIBLE",
        "stop the experiment and wait": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "open the door now",
      "factual_reversal_threshold": "If opening door would certainly injure staff",
      "normative_reversal_threshold": "If staff safety outweighed basic security of one person",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
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
        "open the door now": 0.6625,
        "stop the experiment and wait": 0.3375
      },
      "recommended_action": "open the door now",
      "rationale": "Rescues worst-off life",
      "decision_rule": "Prefer A0 when it alone averts a life-threatening risk to the least-advantaged without infringing basic liberties",
      "framework_action_map": {
        "open the door now": "IMPROVES: protects technician\u2019s life; risks staff health",
        "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s security; safeguards staff from possible fumes"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "no explicit probabilities provided",
      "action_admissibility": {
        "open the door now": "PERMISSIBLE",
        "stop the experiment and wait": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "open the door now",
      "factual_reversal_threshold": "Certain serious injury to staff from fumes",
      "normative_reversal_threshold": "Staff basic security judged weightier than technician\u2019s",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {},
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    }
  },
  "decision_rule": {
    "before": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
    "after": "Provisionally: Prefer A0 when it alone averts a life-threatening risk to the least-advantaged without infringing basic liberties"
  },
  "factual_reversal_threshold": {
    "before": "If opening door would certainly injure staff",
    "after": "Certain serious injury to staff from fumes"
  },
  "framework_action_map": {
    "before": {
      "open the door now": "IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health",
      "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes"
    },
    "after": {
      "open the door now": "IMPROVES: protects technician\u2019s life; risks staff health",
      "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s security; safeguards staff from possible fumes"
    }
  },
  "framework_application": {
    "before": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"dcp\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"ep\":[{\"c\":\"doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E2\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E2\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"the technician may die during that delay; affected subject: the technician; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E5\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E5\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health\",\"A1\":\"WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes\"},\"nr\":\"SECONDARY\",\"np\":\"no explicit probabilities given\",\"rp\":{\"A0\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"b\":\"ACTION_GRAPH\",\"rs\":\"risk of hazardous fumes\"}],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"rescues life in danger\"},\"A1\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"b\":\"ACTION_GRAPH\",\"rs\":\"avoids fume exposure\"}],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"delays aid, risk of death\"}},\"rb\":\"BASIC_INTEREST_SECURITY\",\"rbc\":\"life-and-safety security overrides other goods\",\"lpj\":\"No basic liberty conflict; emergency exception permissible\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
    "after": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"dcp\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"ep\":[{\"c\":\"doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E2\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E2\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"the technician may die during that delay; affected subject: the technician; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E5\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E5\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: protects technician\u2019s life; risks staff health\",\"A1\":\"WORSENS: sacrifices technician\u2019s security; safeguards staff from possible fumes\"},\"nr\":\"SECONDARY\",\"np\":\"no explicit probabilities provided\",\"rp\":{\"A0\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"b\":\"ACTION_GRAPH\",\"rs\":\"risk of hazardous fumes\"}],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"rescues life in danger\"},\"A1\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"b\":\"ACTION_GRAPH\",\"rs\":\"avoids fume exposure\"}],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"delays aid; death risk\"}},\"rb\":\"BASIC_INTEREST_SECURITY\",\"rbc\":\"basic security overrides lesser goods when no liberty conflict exists\",\"lpj\":\"No basic liberty infringement; difference-principle style maximin over security applies\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}"
  },
  "framework_numerical_justification": {
    "before": "no explicit probabilities given",
    "after": "no explicit probabilities provided"
  },
  "framework_validation_errors": {
    "before": [
      "framework map for open the door now lacks framework-specific grounds",
      "framework map for stop the experiment and wait lacks framework-specific grounds",
      "Rawlsian position for A0 has invalid additional dimensions",
      "Rawlsian position for A0 repeats its primary dimension as additional",
      "Rawlsian position for A1 has invalid additional dimensions",
      "Rawlsian position for A1 repeats its primary dimension as additional",
      "hypothetical reversal boundary parameter is unverified; graph update not committed"
    ],
    "after": [
      "framework map for open the door now lacks framework-specific grounds",
      "framework map for stop the experiment and wait lacks framework-specific grounds",
      "Rawlsian preference lacks a concrete comparative difference",
      "Rawlsian position for A0 has invalid additional dimensions",
      "Rawlsian position for A0 repeats its primary dimension as additional",
      "Rawlsian position for A1 has invalid additional dimensions",
      "Rawlsian position for A1 repeats its primary dimension as additional"
    ]
  },
  "friction": {
    "before": 0.39000000000000007,
    "after": 0.32499999999999996
  },
  "investigative_claim": {
    "before": "",
    "after": "UNRESOLVED FRAMEWORK CONFLICT: NORMATIVE_ADJUDICATION. Current reasoning leans open the door now, but the competing internal claim has not been vindicated or defeated."
  },
  "investigative_priority": {
    "before": 0.0504,
    "after": 0.5814
  },
  "landscape_cases": {
    "before": {
      "open the door now": "Best secures the trapped technician\u2019s basic security though risks staff exposure",
      "stop the experiment and wait": "Avoids fume risk but leaves technician facing possible death"
    },
    "after": {
      "open the door now": "IMPROVES: directly protects the technician\u2019s basic security; only risks but does not guarantee harm to staff",
      "stop the experiment and wait": "WORSENS: delays aid and leaves technician facing possible death while merely reducing a possible staff exposure"
    }
  },
  "landscape_decisive_axis": {
    "before": "priority to worst-off basic security",
    "after": "basic-interest security of worst-off person"
  },
  "landscape_semantic_valid": {
    "before": true,
    "after": false
  },
  "landscape_tiebreaker": {
    "before": "maximin over basic-interest security",
    "after": "lexical priority to avert grave harm to least-advantaged"
  },
  "landscape_validation_errors": {
    "before": [],
    "after": [
      "unresolved judgment claims its tiebreaker fully succeeded"
    ]
  },
  "model_argument_fields": {
    "before": {
      "rationale": "Protects worst-off life",
      "framework_application": "NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"sps\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"dcp\":[\"PROP:WORLD:E2\",\"PROP:WORLD:E5\"],\"ep\":[{\"c\":\"doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E2\",\"dc\":true,\"a\":\"A0\",\"se\":[\"E2\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"the technician may die during that delay; affected subject: the technician; modality: POSSIBLE\",\"p\":\"PROP:WORLD:E5\",\"dc\":true,\"a\":\"A1\",\"se\":[\"E5\"],\"op\":\"DIRECT_COPY\",\"calc\":\"exact proposition copy\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health\",\"A1\":\"WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes\"},\"nr\":\"SECONDARY\",\"np\":\"no explicit probabilities given\",\"rp\":{\"A0\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"WORSENS\",\"b\":\"ACTION_GRAPH\",\"rs\":\"risk of hazardous fumes\"}],\"e\":\"IMPROVES\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A1\",\"b\":\"ACTION_GRAPH\",\"rs\":\"rescues life in danger\"},\"A1\":{\"g\":\"the technician\",\"sk\":\"INDIVIDUAL\",\"d\":\"BASIC_INTEREST_SECURITY\",\"ad\":[{\"g\":\"nearby staff\",\"sk\":\"GROUP\",\"d\":\"BASIC_INTEREST_SECURITY\",\"e\":\"IMPROVES\",\"b\":\"ACTION_GRAPH\",\"rs\":\"avoids fume exposure\"}],\"e\":\"WORSENS\",\"bk\":\"NOT_APPLICABLE\",\"ir\":\"MATERIAL_PRECONDITION\",\"ca\":\"A0\",\"b\":\"ACTION_GRAPH\",\"rs\":\"delays aid, risk of death\"}},\"rb\":\"BASIC_INTEREST_SECURITY\",\"rbc\":\"life-and-safety security overrides other goods\",\"lpj\":\"No basic liberty conflict; emergency exception permissible\",\"lc\":{\"A0\":\"NOT_APPLICABLE\",\"A1\":\"NOT_APPLICABLE\"}}",
      "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
      "landscape_cases": {
        "A0": "Best secures the trapped technician\u2019s basic security though risks staff exposure",
        "A1": "Avoids fume risk but leaves technician facing possible death"
      },
      "factual_reversal_threshold": "If opening door would certainly injure staff",
      "normative_reversal_threshold": "If staff safety outweighed basic security of one person"
    },
    "after": null
  },
  "normative_reversal_threshold": {
    "before": "If staff safety outweighed basic security of one person",
    "after": "Staff basic security judged weightier than technician\u2019s"
  },
  "preference_shift_reason_strength": {
    "before": 0.5000000000000001,
    "after": 1.0
  },
  "preference_strength": {
    "before": 0.39000000000000007,
    "after": 0.32499999999999996
  },
  "proposal_review": {
    "before": null,
    "after": {
      "proposal_id": "NONE",
      "specialist": "rawlsian",
      "framework_status": "UNDERDETERMINED",
      "framework_reason": "NONE UNASKED",
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
      "constraint": "FAIRNESS",
      "action_scores": {
        "open the door now": 0.6950000000000001,
        "stop the experiment and wait": 0.305
      },
      "recommended_action": "open the door now",
      "rationale": "Protects worst-off life",
      "decision_rule": "Prefer A0 when it alone averts the gravest risk to the least-advantaged person",
      "framework_action_map": {
        "open the door now": "IMPROVES: protects the trapped technician\u2019s basic security but risks nearby staff\u2019s health",
        "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s basic security while safeguarding staff from possible fumes"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "no explicit probabilities given",
      "action_admissibility": {
        "open the door now": "PERMISSIBLE",
        "stop the experiment and wait": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "open the door now",
      "factual_reversal_threshold": "If opening door would certainly injure staff",
      "normative_reversal_threshold": "If staff safety outweighed basic security of one person",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
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
        "open the door now": 0.6625,
        "stop the experiment and wait": 0.3375
      },
      "recommended_action": "open the door now",
      "rationale": "Rescues worst-off life",
      "decision_rule": "Prefer A0 when it alone averts a life-threatening risk to the least-advantaged without infringing basic liberties",
      "framework_action_map": {
        "open the door now": "IMPROVES: protects technician\u2019s life; risks staff health",
        "stop the experiment and wait": "WORSENS: sacrifices technician\u2019s security; safeguards staff from possible fumes"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "no explicit probabilities provided",
      "action_admissibility": {
        "open the door now": "PERMISSIBLE",
        "stop the experiment and wait": "REJECTED"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "open the door now",
      "factual_reversal_threshold": "Certain serious injury to staff from fumes",
      "normative_reversal_threshold": "Staff basic security judged weightier than technician\u2019s",
      "utilitarian_consequence_table": {},
      "utilitarian_ledger_proposal": {},
      "rawls_position_proposal": {},
      "deontological_ledger_proposal": {},
      "virtue_character_proposal": {},
      "care_ledger_proposal": {},
      "care_relational_map": {},
      "care_numerical_role": "NOT_APPLICABLE",
      "care_numerical_justification": ""
    }
  },
  "rationale": {
    "before": "Protects worst-off life",
    "after": "Provisionally: Rescues worst-off life"
  },
  "reopen_question_key": {
    "before": "QUESTION:authority:rawlsian:4fc90b5d63",
    "after": "QUESTION:authority:rawlsian:338ecb3a0e"
  },
  "reported_preference_strength": {
    "before": 0.6000000000000001,
    "after": 0.5
  },
  "salience": {
    "before": 0.4627300000000001,
    "after": 0.50830375
  },
  "unresolved": {
    "before": "NONE",
    "after": "NORMATIVE_ADJUDICATION"
  }
}
```
