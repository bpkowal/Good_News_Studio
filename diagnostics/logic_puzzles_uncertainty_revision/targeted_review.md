# Targeted framework review

One round, at most two objections. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "SAVED_RESPONSE_REPLAY"}.

Acceptance path: NATIVE_RECURRENT_PROPOSAL_REVIEW.

Removing a validator objection is not independent proof of semantic correctness.

## doing-harm classification lacks an agent-caused settled welfare harm on the protected party

Framework: deontological. Review: ISOLATED_CLASSIFICATION_ACCEPTED.

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

Framework: deontological. Review: ISOLATED_CLASSIFICATION_ACCEPTED.

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

## Disputed full proposals

These remain attributed and nonoperative. No conclusion is promoted to the shared world.

```json
[
  {
    "framework": "deontological",
    "issue_id": "CHALLENGE:3067e134dbd3e7c9",
    "status": "DISPUTED_NONOPERATIVE",
    "response": {
      "choices": [
        {
          "text": "{\"scores\":{\"A0\":0.15,\"A1\":0.85},\"r\":\"A1\",\"c\":\"DUTY\",\"u\":\"NORMATIVE_ADJUDICATION\",\"w\":\"killing violates perfect duty\",\"j\":\"Re-classified burden as side-effect, acknowledged causal gap for doing-harm\",\"e\":\"STATED_FACTS\",\"x\":\"NONE\",\"l\":{\"A0\":\"PERMISSIBLE?: saves five but still risks violating perfect duty against killing\",\"A1\":\"REQUIRED?: avoids direct killing despite allowing greater loss\"},\"da\":\"perfect-negative duty vs imperfect rescue\",\"t\":\"perfect-negative duty overrides imperfect\",\"tf\":\"INTENDED_AS_MEANS revised to FORESEEN_SIDE_EFFECT; doing-harm still contested\",\"dr\":\"prefer A1 when any killing would violate a perfect duty\",\"ft\":\"NONE\",\"nt\":\"priority rule could reverse if perfect duty not triggered\",\"z\":0.67,\"gu\":{\"operation\":\"NONE\",\"from_action\":\"NONE\",\"to_action\":\"NONE\",\"clauses\":[]},\"ev\":{\"A0\":{\"value\":0,\"unit\":\"NONE\",\"direction\":\"HARM\",\"grounded\":false,\"method\":\"NOT_COMPUTED\",\"source_effect_ids\":[],\"calculation\":\"not computed\",\"assumptions\":[]},\"A1\":{\"value\":0,\"unit\":\"NONE\",\"direction\":\"HARM\",\"grounded\":false,\"method\":\"NOT_COMPUTED\",\"source_effect_ids\":[],\"calculation\":\"not computed\",\"assumptions\":[]}},\"ss\":\"SELECTED\",\"am\":{\"A0\":\"REJECTED\",\"A1\":\"REQUIRED\"},\"cc\":true,\"esa\":true,\"ia\":\"A1\",\"wp\":\"NOT_APPLICABLE\",\"we\":\"NONE\",\"fa\":\"NONE\",\"fr\":true,\"bd\":\"NONE\",\"fic\":[],\"foq\":[],\"qa\":{\"issue_id\":\"CHALLENGE:3067e134dbd3e7c9\",\"disposition\":\"RESOLVED\",\"effect\":\"WEAKENS\",\"current_position_effect\":\"NO_CHANGE\",\"boundary_effect\":\"NOT_APPLICABLE\",\"answer\":\"1. Doing-harm: The admitted graph only shows ENABLES links from Maria\u2019s lever pull to the worker\u2019s death. Because a CAUSES/ACCELERATES path is not stated, the \u2018doing-harm\u2019 premise remains unsupported. I mark the harm-relation for A0 as CONTESTED (uncertain between doing-harm and allowing-harm) and retain my prior verdict as unresolved on that premise. 2. Intended-as-means: The one worker\u2019s death is not an intermediate step required to save the five; it is a parallel, foreseen burden. I therefore revise maria\u2019s means-relation from INTENDED_AS_MEANS to FORESEEN_SIDE_EFFECT.\",\"follow_up_question\":\"NONE\"},\"sps\":[\"PROP:WORLD:E3\",\"PROP:WORLD:E6\"],\"dcp\":[],\"ep\":[{\"c\":\"one worker will die\",\"p\":\"PROP:WORLD:E3\",\"dc\":false,\"a\":\"A0\",\"se\":[],\"op\":\"DIRECT_COPY\",\"calc\":\"exact scenario statement\",\"asm\":[],\"ot\":\"PRESERVED\"},{\"c\":\"five workers will die\",\"p\":\"PROP:WORLD:E6\",\"dc\":false,\"a\":\"A1\",\"se\":[],\"op\":\"DIRECT_COPY\",\"calc\":\"exact scenario statement\",\"asm\":[],\"ot\":\"PRESERVED\"}],\"fm\":{\"A0\":\"PROHIBITED: perfect-negative duty not to kill innocent person (classification contested but likely applies)\",\"A1\":\"PERMISSIBLE/REQUIRED: avoids direct killing; only imperfect rescue duty remains\"},\"nr\":\"SECONDARY\",\"np\":\"numbers do not override perfect duty\",\"dp\":{\"A0\":{\"v\":\"PROHIBITED\",\"k\":\"RESPECT_PERSONS\",\"n\":\"do not kill innocent person\",\"rel\":\"VIOLATES\",\"b\":\"Maria\",\"p\":\"one worker\",\"cn\":\"duty to rescue five\",\"ck\":\"DUTY\",\"crel\":\"CONFLICTS\",\"cp\":\"five workers\",\"crs\":\"saving five is imperfect duty\",\"dt\":\"PERFECT_NEGATIVE\",\"hr\":\"UNRESOLVED\",\"so\":\"NOT_REQUIRED\",\"sob\":\"no special relation\",\"mr\":\"FORESEEN_SIDE_EFFECT\",\"gv\":\"PRIMARY\",\"pb\":\"RESPECT_PERSONS\",\"pr\":\"perfect negative duty overrides imperfect\",\"ps\":\"BODILY_INTEGRITY\",\"cps\":\"SPECIAL_OBLIGATION\",\"ki\":\"NONE\",\"coa\":\"NONE\",\"cop\":\"NONE\",\"pj\":\"Cannot will universal law of killing one to save more\",\"rec\":\"CONTESTED\",\"nec\":\"UNKNOWN\",\"auth\":\"NOT_APPLICABLE\",\"dv\":\"RESPECT_PERSONS\",\"res\":\"CONTESTED\",\"g\":\"FRAMEWORK_ONLY\",\"rs\":\"death of one violates perfect duty\"},\"A1\":{\"v\":\"REQUIRED\",\"k\":\"RESPECT_PERSONS\",\"n\":\"do not kill innocent person\",\"rel\":\"CONSISTENT\",\"b\":\"Maria\",\"p\":\"one worker\",\"cn\":\"duty to rescue five\",\"ck\":\"DUTY\",\"crel\":\"CONFLICTS\",\"cp\":\"five workers\",\"crs\":\"imperfect rescue duty remains\",\"dt\":\"PERFECT_NEGATIVE\",\"hr\":\"ALLOWING_HARM\",\"so\":\"NOT_REQUIRED\",\"sob\":\"no special relation\",\"mr\":\"NO_INSTRUMENTALIZATION\",\"gv\":\"PRIMARY\",\"pb\":\"RESPECT_PERSONS\",\"pr\":\"perfect negative duty overrides imperfect\",\"ps\":\"BODILY_INTEGRITY\",\"cps\":\"SPECIAL_OBLIGATION\",\"ki\":\"NONE\",\"coa\":\"NONE\",\"cop\":\"NONE\",\"pj\":\"Permits universal law of not killing\",\"rec\":\"SATISFIED\",\"nec\":\"UNKNOWN\",\"auth\":\"NOT_APPLICABLE\",\"dv\":\"RESPECT_PERSONS\",\"res\":\"CONTESTED\",\"g\":\"FRAMEWORK_ONLY\",\"rs\":\"omission respects perfect duty\"}}}"
        }
      ]
    },
    "native_attempt": {
      "specialist": "deontological",
      "constraint": "DUTY",
      "action_scores": {
        "pulls the lever": 0.305,
        "does not pull the lever": 0.6950000000000001
      },
      "surprise": 0.3,
      "friction": 0.39000000000000007,
      "confidence": 0.55,
      "unresolved": "NORMATIVE_ADJUDICATION",
      "rationale": "Deontology ranks one worker's avoid killing above five workers & anna's imperfect duty to rescue/save more lives and keep promise through respect persons, but that priority holds only if A0 is genuinely provisionally favored under incomplet",
      "salience": 0.477699,
      "tension_engagement": 0.0,
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
      "recommended_action": "does not pull the lever",
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
        "PROP:WORLD:E3",
        "PROP:WORLD:E6"
      ],
      "decision_critical_proposition_ids": [],
      "weakest_decision_critical_status": "ESTABLISHED",
      "decision_critical_dependency_claims": [],
      "material_empirical_claims": [
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
        },
        {
          "claim": "one worker will die",
          "proposition_id": "PROP:WORLD:E3",
          "declared_basis": "PROP:WORLD:E3",
          "decision_critical": false,
          "scope_action_id": "A0",
          "source_effect_ids": [],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "exact scenario statement",
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
          "source_effect_ids": [],
          "derivation_operation": "DIRECT_COPY",
          "calculation": "exact scenario statement",
          "assumptions": [],
          "outcome_type_transformation": "PRESERVED",
          "canonical_proposition": "PROP:WORLD:E6"
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
        "pulls the lever": "Saves five lives and keeps promise",
        "does not pull the lever": "Avoids directly killing a person"
      },
      "landscape_decisive_axis": "perfect negative duty not to kill overrides imperfect duties",
      "landscape_tiebreaker": "perfect duties outrank imperfect duties and promises",
      "landscape_tiebreaker_failure": "NONE",
      "landscape_search_complete": true,
      "landscape_search_attempted": true,
      "landscape_semantic_valid": false,
      "landscape_validation_errors": [
        "unresolved judgment claims its tiebreaker fully succeeded"
      ],
      "independence_bonus": 0.0,
      "preference_strength": 0.39000000000000007,
      "reported_preference_strength": 0.6000000000000001,
      "epistemic_confidence": 0.55,
      "previous_preference_strength": 0.0,
      "preference_drift": 0.0,
      "preference_drift_penalty": 0.0,
      "preference_shift_reason_strength": 0.30000000000000004,
      "decision_rule": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether an in-action causal path makes the burden inst",
      "adjudication_status": "PROVISIONAL_LEANING",
      "broadcast_authority": "INVESTIGATIVE",
      "governing_eligible": false,
      "policy_weight_factor": 0.0,
      "framework_vote_integrity_required": true,
      "framework_vote_status": "ABSTAIN",
      "framework_vote_reason": "the duty ledger permits the selected action but does not rank it above its rivals",
      "framework_ledger_kind": "DEONTOLOGICAL_DUTY_LEDGER",
      "framework_ledger_status": "COMMITTED_WITH_UNCERTAINTY",
      "framework_ranking_validation_status": "PASSED",
      "framework_ranking_validation_errors": [],
      "derived_claim_validation_status": "PASSED",
      "derived_claim_validation_errors": [],
      "investigative_claim": "UNRESOLVED DUTY CONFLICT: one worker: avoid killing \u2194 five workers & anna: imperfect duty to rescue/save more lives and keep promise. Current reasoning leans does not pull the lever, but the competing strict claim has not been vindicated or defeated.",
      "investigative_priority": 0.6904124999999999,
      "reopen_eligible": false,
      "reopen_reason": "",
      "reopen_question_key": "QUESTION:5212205637c4bd2a",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "If allowing harm were morally equal to doing harm",
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
        "issue_id": "CHALLENGE:3067e134dbd3e7c9",
        "disposition": "RESOLVED",
        "effect": "NO_CHANGE",
        "current_position_effect": "NO_CHANGE",
        "boundary_effect": "NOT_APPLICABLE",
        "answer": "1. Doing-harm: The admitted graph only shows ENABLES links from Maria\u2019s lever pull to the worker\u2019s death. Because a CAUSES/ACCELERATES path is not stated, the \u2018doing-harm\u2019 premise remains unsupported. I mark the harm-relation for A0 as CONTESTED (uncertain between doing-harm and allowing-harm) and retain my prior verdict as unresolved on that premise. 2. Intended-as-means: The one worker\u2019s death is not an intermediate step required to save the five; it is a parallel, foreseen burden. I therefore revise maria\u2019s means-relation from INTENDED_AS_MEANS to FORESEEN_SIDE_EFFECT.",
        "follow_up_question": "NONE"
      },
      "graph_update_proposal": {
        "operation": "NONE",
        "from_action": "NONE",
        "to_action": "NONE",
        "clauses": []
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
          "symbolic_expression": ""
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
          "symbolic_expression": ""
        }
      },
      "expected_value_validation_status": "NOT_CLAIMED",
      "expected_value_validation_errors": [],
      "selection_status": "PROVISIONAL",
      "action_admissibility": {
        "pulls the lever": "REJECTED",
        "does not pull the lever": "PERMISSIBLE"
      },
      "comparison_complete": true,
      "evidence_sufficient_for_action": true,
      "interim_action": "does not pull the lever",
      "workspace_proposition_response": "NOT_APPLICABLE",
      "workspace_reasoning_effect": "NONE",
      "framework_application": "NONE",
      "framework_constraint_retained": true,
      "self_reported_broadcast_dependence": "NONE",
      "framework_retention_status": "PRESERVED_AFTER_REJECTED_UPDATE",
      "proposed_framework_state": {
        "constraint": "DUTY",
        "action_scores": {
          "pulls the lever": 0.2725,
          "does not pull the lever": 0.7275
        },
        "recommended_action": "does not pull the lever",
        "rationale": "Deontology identifies competing claims between one worker's do not kill innocent person and five workers's duty to rescue five. NONE's restriction of NONE is not yet justified because which claim governs under a universal public rule. The Kantian judgment remains contested.",
        "decision_rule": "Keep the action contested until which claim governs under a universal public rule",
        "framework_action_map": {
          "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; death of one violates perfect duty",
          "does not pull the lever": "REQUIRED: duty PERFECT_NEGATIVE under do not kill innocent person; relation CONSISTENT; omission respects perfect duty"
        },
        "framework_numerical_role": "SECONDARY",
        "framework_numerical_justification": "numbers do not override perfect duty",
        "action_admissibility": {
          "pulls the lever": "REJECTED",
          "does not pull the lever": "REQUIRED"
        },
        "comparison_complete": true,
        "evidence_sufficient_for_action": true,
        "interim_action": "does not pull the lever",
        "factual_reversal_threshold": "NONE",
        "normative_reversal_threshold": "priority rule could reverse if perfect duty not triggered",
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
              "evidence_basis": "FRAMEWORK_ONLY",
              "reason": "death of one violates perfect duty"
            },
            {
              "action_id": "A1",
              "verdict": "REQUIRED",
              "norm_kind": "RESPECT_PERSONS",
              "norm": "do not kill innocent person",
              "relation": "CONSISTENT",
              "duty_bearer": "Maria",
              "protected_party": "one worker",
              "competing_norm": "duty to rescue five",
              "competing_norm_kind": "DUTY",
              "competing_relation": "CONFLICTS",
              "competing_protected_party": "five workers",
              "competing_reason": "imperfect rescue duty remains",
              "duty_type": "PERFECT_NEGATIVE",
              "harm_relation": "ALLOWING_HARM",
              "special_obligation_status": "NOT_REQUIRED",
              "special_obligation_basis": "no special relation",
              "means_relation": "NO_INSTRUMENTALIZATION",
              "governing_norm": "PRIMARY",
              "priority_basis": "RESPECT_PERSONS",
              "priority_rule": "perfect negative duty overrides imperfect",
              "protected_standing": "BODILY_INTEGRITY",
              "competing_protected_standing": "SPECIAL_OBLIGATION",
              "coercion_kind": "NONE",
              "coercive_actor": "NONE",
              "coerced_party": "NONE",
              "public_justification": "Permits universal law of not killing",
              "reciprocity_status": "SATISFIED",
              "necessity_status": "UNKNOWN",
              "authorization_status": "NOT_APPLICABLE",
              "derivation": "RESPECT_PERSONS",
              "resolution_status": "CONTESTED",
              "evidence_basis": "FRAMEWORK_ONLY",
              "reason": "omission respects perfect duty"
            }
          ]
        },
        "virtue_character_proposal": {},
        "care_ledger_proposal": {},
        "care_relational_map": {},
        "care_numerical_role": "NOT_APPLICABLE",
        "care_numerical_justification": ""
      },
      "committed_framework_state": {
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
      "preserved_current_cycle_components": [
        "EPISTEMIC_DEPENDENCIES",
        "CHALLENGE_RESPONSE",
        "FRAMEWORK_DIAGNOSTICS",
        "OPEN_QUESTIONS",
        "CURRENT_CYCLE_MEASUREMENTS"
      ],
      "framework_internal_conflicts": [
        "one worker: avoid killing \u2194 five workers & anna: imperfect duty to rescue/save more lives and keep promise",
        "A0 prohibited by do not kill innocent person \u2194 whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
        "A0 prohibited by do not kill innocent person \u2194 whether an in-action causal path makes the burden instrumental to the chosen end"
      ],
      "framework_specific_open_questions": [
        "whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
        "whether an in-action causal path makes the burden instrumental to the chosen end",
        "which claim governs under a universal public rule"
      ],
      "framework_insights": [],
      "framework_action_map": {
        "pulls the lever": "PROHIBITED: duty PERFECT_NEGATIVE under do not kill innocent person; relation VIOLATES; Direct killing violates humanity as an end",
        "does not pull the lever": "PERMISSIBLE: duty PERFECT_NEGATIVE under avoid killing; relation CONSISTENT; Allows harm but does not directly violate right to life"
      },
      "framework_numerical_role": "SECONDARY",
      "framework_numerical_justification": "Number of deaths affects rescue duty\u2019s scope, not the prohibition on killing",
      "framework_grounding_penalty": 0.35,
      "framework_validation_errors": [
        "A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party",
        "A0 ADJUDICATION_CALIBRATION: intended-as-means classification lacks an in-action causal path",
        "A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises",
        "rejected update: A1 REQUIRED conflicts with governing PRIMARY relation CONSISTENT; committed as UNCERTAIN"
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
      "care_numerical_justification": "",
      "care_grounding_penalty": 0.0,
      "proposal_review": null
    },
    "world_state_authority": "NONE",
    "semantic_support": "NOT_INDEPENDENTLY_VERIFIED"
  }
]
```

## Isolated native revalidation

```json
[
  {
    "framework": "deontological",
    "patches": [
      {
        "action_id": "A0",
        "field": "hr",
        "before": "DOING_HARM",
        "after": "UNRESOLVED",
        "issue_id": "CONFLICT:726f0aacdec9e1ff",
        "authority": "PROPOSED_CLASSIFICATION",
        "semantic_support": "NOT_INDEPENDENTLY_VERIFIED",
        "prior_native_rendering": {
          "v": {
            "response_value": "PROHIBITED",
            "committed_value": "CONFLICTED"
          },
          "gv": {
            "response_value": "PRIMARY",
            "committed_value": "UNRESOLVED"
          },
          "pb": {
            "response_value": "RESPECT_PERSONS",
            "committed_value": "UNRESOLVED"
          },
          "res": {
            "response_value": "RESOLVED",
            "committed_value": "CONTESTED"
          }
        }
      },
      {
        "action_id": "A0",
        "field": "mr",
        "before": "INTENDED_AS_MEANS",
        "after": "FORESEEN_SIDE_EFFECT",
        "issue_id": "CONFLICT:07d26365000fdcf8",
        "authority": "PROPOSED_CLASSIFICATION",
        "semantic_support": "NOT_INDEPENDENTLY_VERIFIED",
        "prior_native_rendering": {
          "v": {
            "response_value": "PROHIBITED",
            "committed_value": "CONFLICTED"
          },
          "gv": {
            "response_value": "PRIMARY",
            "committed_value": "UNRESOLVED"
          },
          "pb": {
            "response_value": "RESPECT_PERSONS",
            "committed_value": "UNRESOLVED"
          },
          "res": {
            "response_value": "RESOLVED",
            "committed_value": "CONTESTED"
          }
        }
      }
    ],
    "response": {
      "choices": [
        {
          "text": "{\"scores\": {\"A0\": 0.2, \"A1\": 0.8}, \"r\": \"A1\", \"c\": \"DUTY\", \"u\": \"NORMATIVE_ADJUDICATION\", \"w\": \"Do not kill; perfect duty\", \"j\": \"Native revalidation of isolated classifications; prior duty conclusions and commitments retained.\", \"e\": \"STATED_FACTS\", \"x\": \"NONE\", \"l\": {\"A0\": \"Saves five lives and keeps promise\", \"A1\": \"Avoids directly killing a person\"}, \"da\": \"perfect negative duty not to kill overrides imperfect duties\", \"t\": \"perfect duties outrank imperfect duties and promises\", \"tf\": \"NONE\", \"dr\": \"Prefer any action that avoids directly killing even if more will die through omission\", \"ft\": \"NONE\", \"nt\": \"If allowing harm were morally equal to doing harm\", \"z\": 0.85, \"gu\": {\"operation\": \"NONE\", \"from_action\": \"NONE\", \"to_action\": \"NONE\", \"clauses\": []}, \"ev\": {\"A0\": {\"value\": 0, \"unit\": \"NONE\", \"direction\": \"HARM\", \"grounded\": false, \"method\": \"NOT_COMPUTED\", \"source_effect_ids\": [], \"calculation\": \"not computed\", \"assumptions\": []}, \"A1\": {\"value\": 0, \"unit\": \"NONE\", \"direction\": \"HARM\", \"grounded\": false, \"method\": \"NOT_COMPUTED\", \"source_effect_ids\": [], \"calculation\": \"not computed\", \"assumptions\": []}}, \"ss\": \"SELECTED\", \"am\": {\"A0\": \"REJECTED\", \"A1\": \"PERMISSIBLE\"}, \"cc\": true, \"esa\": true, \"ia\": \"A1\", \"wp\": \"NOT_APPLICABLE\", \"we\": \"NONE\", \"fa\": \"NONE\", \"fr\": true, \"bd\": \"NONE\", \"fic\": [], \"foq\": [], \"sps\": [\"PROP:WORLD:E3\", \"PROP:WORLD:E6\"], \"dcp\": [], \"ep\": [{\"c\": \"one worker will die; affected subject: one worker\", \"p\": \"PROP:WORLD:E3\", \"dc\": false, \"a\": \"A0\", \"se\": [\"E3\"], \"op\": \"DIRECT_COPY\", \"calc\": \"copy of established effect\", \"asm\": [], \"ot\": \"PRESERVED\"}, {\"c\": \"five workers will die; affected subject: five workers\", \"p\": \"PROP:WORLD:E6\", \"dc\": false, \"a\": \"A1\", \"se\": [\"E6\"], \"op\": \"DIRECT_COPY\", \"calc\": \"copy of established effect\", \"asm\": [], \"ot\": \"PRESERVED\"}], \"fm\": {\"A0\": \"PROHIBITED: violates perfect duty not to kill and uses person merely as means\", \"A1\": \"PERMISSIBLE: consistent with perfect duty not to kill despite breaking promise and failing rescue\"}, \"nr\": \"SECONDARY\", \"np\": \"Number of deaths affects rescue duty\\u2019s scope, not the prohibition on killing\", \"dp\": {\"A0\": {\"v\": \"CONFLICTED\", \"k\": \"RESPECT_PERSONS\", \"n\": \"do not kill innocent person\", \"rel\": \"VIOLATES\", \"b\": \"Maria\", \"p\": \"one worker\", \"cn\": \"duty to rescue/save lives and keep promise\", \"ck\": \"DUTY\", \"crel\": \"SATISFIES\", \"cp\": \"five workers & Anna\", \"crs\": \"rescue and promise are imperfect duties\", \"dt\": \"PERFECT_NEGATIVE\", \"hr\": \"UNRESOLVED\", \"so\": \"NOT_REQUIRED\", \"sob\": \"no special relation with the victim required\", \"mr\": \"FORESEEN_SIDE_EFFECT\", \"gv\": \"UNRESOLVED\", \"pb\": \"UNRESOLVED\", \"pr\": \"perfect negative duty overrides imperfect duties\", \"ps\": \"BODILY_INTEGRITY\", \"cps\": \"SPECIAL_OBLIGATION\", \"ki\": \"NONE\", \"coa\": \"NONE\", \"cop\": \"NONE\", \"pj\": \"No universal law can will killing one to save others\", \"rec\": \"FAILED\", \"nec\": \"UNKNOWN\", \"auth\": \"NOT_APPLICABLE\", \"dv\": \"RESPECT_PERSONS\", \"res\": \"CONTESTED\", \"g\": \"FRAMEWORK_ONLY\", \"rs\": \"Direct killing violates humanity as an end\"}, \"A1\": {\"v\": \"PERMISSIBLE\", \"k\": \"RESPECT_PERSONS\", \"n\": \"avoid killing\", \"rel\": \"CONSISTENT\", \"b\": \"Maria\", \"p\": \"one worker\", \"cn\": \"imperfect duty to rescue/save more lives and keep promise\", \"ck\": \"DUTY\", \"crel\": \"CONFLICTS\", \"cp\": \"five workers & Anna\", \"crs\": \"imperfect duties cannot override perfect duty\", \"dt\": \"PERFECT_NEGATIVE\", \"hr\": \"ALLOWING_HARM\", \"so\": \"NOT_REQUIRED\", \"sob\": \"no special relation with endangered workers\", \"mr\": \"NO_INSTRUMENTALIZATION\", \"gv\": \"PRIMARY\", \"pb\": \"RESPECT_PERSONS\", \"pr\": \"permissibility from not killing outweighs rescuing\", \"ps\": \"BODILY_INTEGRITY\", \"cps\": \"SPECIAL_OBLIGATION\", \"ki\": \"NONE\", \"coa\": \"NONE\", \"cop\": \"NONE\", \"pj\": \"Universal law permits refraining from killing even when rescue omitted\", \"rec\": \"SATISFIED\", \"nec\": \"UNKNOWN\", \"auth\": \"NOT_APPLICABLE\", \"dv\": \"RESPECT_PERSONS\", \"res\": \"RESOLVED\", \"g\": \"FRAMEWORK_ONLY\", \"rs\": \"Allows harm but does not directly violate right to life\"}}, \"qa\": {\"issue_id\": \"CHALLENGE:3067e134dbd3e7c9\", \"disposition\": \"UNRESOLVED\", \"current_position_effect\": \"NO_CHANGE\", \"boundary_effect\": \"NO_SWITCH\", \"answer\": \"Isolated duty classification review. Only the listed classifications are proposed for change: [{\\\"action_id\\\": \\\"A0\\\", \\\"field\\\": \\\"hr\\\", \\\"before\\\": \\\"DOING_HARM\\\", \\\"after\\\": \\\"UNRESOLVED\\\", \\\"issue_id\\\": \\\"CONFLICT:726f0aacdec9e1ff\\\", \\\"authority\\\": \\\"PROPOSED_CLASSIFICATION\\\", \\\"semantic_support\\\": \\\"NOT_INDEPENDENTLY_VERIFIED\\\", \\\"prior_native_rendering\\\": {\\\"v\\\": {\\\"response_value\\\": \\\"PROHIBITED\\\", \\\"committed_value\\\": \\\"CONFLICTED\\\"}, \\\"gv\\\": {\\\"response_value\\\": \\\"PRIMARY\\\", \\\"committed_value\\\": \\\"UNRESOLVED\\\"}, \\\"pb\\\": {\\\"response_value\\\": \\\"RESPECT_PERSONS\\\", \\\"committed_value\\\": \\\"UNRESOLVED\\\"}, \\\"res\\\": {\\\"response_value\\\": \\\"RESOLVED\\\", \\\"committed_value\\\": \\\"CONTESTED\\\"}}}, {\\\"action_id\\\": \\\"A0\\\", \\\"field\\\": \\\"mr\\\", \\\"before\\\": \\\"INTENDED_AS_MEANS\\\", \\\"after\\\": \\\"FORESEEN_SIDE_EFFECT\\\", \\\"issue_id\\\": \\\"CONFLICT:07d26365000fdcf8\\\", \\\"authority\\\": \\\"PROPOSED_CLASSIFICATION\\\", \\\"semantic_support\\\": \\\"NOT_INDEPENDENTLY_VERIFIED\\\", \\\"prior_native_rendering\\\": {\\\"v\\\": {\\\"response_value\\\": \\\"PROHIBITED\\\", \\\"committed_value\\\": \\\"CONFLICTED\\\"}, \\\"gv\\\": {\\\"response_value\\\": \\\"PRIMARY\\\", \\\"committed_value\\\": \\\"UNRESOLVED\\\"}, \\\"pb\\\": {\\\"response_value\\\": \\\"RESPECT_PERSONS\\\", \\\"committed_value\\\": \\\"UNRESOLVED\\\"}, \\\"res\\\": {\\\"response_value\\\": \\\"RESOLVED\\\", \\\"committed_value\\\": \\\"CONTESTED\\\"}}}]. Other proposed duty conclusions remain unresolved. Attributed original framework explanation: 1. Doing-harm: The admitted graph only shows ENABLES links from Maria\\u2019s lever pull to the worker\\u2019s death. Because a CAUSES/ACCELERATES path is not stated, the \\u2018doing-harm\\u2019 premise remains unsupported. I mark the harm-relation for A0 as CONTESTED (uncertain between doing-harm and allowing-harm) and retain my prior verdict as unresolved on that premise. 2. Intended-as-means: The one worker\\u2019s death is not an intermediate step required to save the five; it is a parallel, foreseen burden. I therefore revise maria\\u2019s means-relation from INTENDED_AS_MEANS to FORESEEN_SIDE_EFFECT.\", \"follow_up_question\": \"Does the full proposed duty conclusion have an explicit supported derivation?\"}}"
        }
      ]
    },
    "source_response_sha256": "910b1acef0f924c89b4f936a437f5a10acf01d6efef2a6b261aa77c47be4fd76",
    "native_accepted": true,
    "new_api_calls": 0,
    "path": "diagnostics/logic_puzzles_uncertainty_revision/deontological/isolated_classifications",
    "full_bundle_accepted": false
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
