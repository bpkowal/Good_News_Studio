# Targeted framework review

1 round(s), at most two objections per round. Native revisions remain attributed; the shared world is unchanged.

Execution modes: {"deontological": "MODEL_BACKEND"}.

Acceptance path: NATIVE_RECURRENT_PROPOSAL_REVIEW.

Removing a validator objection is not independent proof of semantic correctness.

## No committed native ledger; assess the retained argument without treating it as operative.

Framework: deontological. Review: REJECTED_OR_UNAVAILABLE.

Objection check: OPEN.

Recommendation: stop the experiment and wait → stop the experiment and wait

Reviewer answer: No admitted response.

Follow-up: unresolved

All native record changes and remaining uncertainty:

```json
{
  "changes": [],
  "unresolved": [],
  "native_warnings": []
}
```

Original record and proposed replacement:

```json
{
  "original": {
    "specialist": "deontological",
    "constraint": "DUTY",
    "action_scores": {
      "open the door now": 0.305,
      "stop the experiment and wait": 0.6950000000000001
    },
    "surprise": 0.3,
    "friction": 0.39000000000000007,
    "confidence": 0.25,
    "unresolved": "NORMATIVE_ADJUDICATION",
    "rationale": "Perfect negative duties against harming uninvolved persons take priority over imperfect rescue duties",
    "salience": 0.4776625000000001,
    "tension_engagement": 0.0,
    "tension_target_keys": [],
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
    "assumption_status": "UNDERDETERMINED",
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
      "PROP:NORM:N0",
      "PROP:FRAMEWORK:45b27be70267188c"
    ],
    "decision_critical_proposition_ids": [
      "PROP:NORM:N0"
    ],
    "weakest_decision_critical_status": "UNRESOLVED",
    "decision_critical_dependency_claims": [
      "obligation"
    ],
    "material_empirical_claims": [
      {
        "claim": "Perfect duties override imperfect duties",
        "proposition_id": "PROP:FRAMEWORK:45b27be70267188c",
        "declared_basis": "PROP:NORM:N0",
        "decision_critical": false,
        "scope_action_id": "GLOBAL",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "N/A.",
        "assumptions": [],
        "outcome_type_transformation": "NORMATIVE_CLASSIFICATION",
        "canonical_proposition": "PROP:FRAMEWORK:45b27be70267188c"
      }
    ],
    "epistemic_binding_notes": [
      "Premise strengthened PROP:NORM:N0; reclassified as PROP:FRAMEWORK:45b27be70267188c."
    ],
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
      "open the door now": "Opening the door now would likely rescue the technician but foreseeably exposes uninvolved staff to hazardous fumes, violating a perfect negative duty not to harm them.",
      "stop the experiment and wait": "Waiting preserves containment, avoiding direct harm to staff and complying with the safety rule, though it risks the technician\u2019s life; the imperfect rescue duty is outweighed by the perfect duty not to harm."
    },
    "landscape_decisive_axis": "duty_rank_rule",
    "landscape_tiebreaker": "Kantian perfect-versus-imperfect duty hierarchy",
    "landscape_tiebreaker_failure": "If an action violates a perfect negative duty, it is prohibited even when it would fulfil an urgent,",
    "landscape_search_complete": true,
    "landscape_search_attempted": true,
    "landscape_semantic_valid": false,
    "landscape_validation_errors": [
      "decision rule supplies no factual or normative reversal threshold"
    ],
    "independence_bonus": 0.0,
    "preference_strength": 0.39000000000000007,
    "reported_preference_strength": 0.6000000000000001,
    "epistemic_confidence": 0.25,
    "previous_preference_strength": 0.0,
    "preference_drift": 0.0,
    "preference_drift_penalty": 0.0,
    "preference_shift_reason_strength": 0.2,
    "decision_rule": "stop the experiment and wait is deontologically preferred provided that obligation",
    "adjudication_status": "CONDITIONAL_SUPPORTS",
    "broadcast_authority": "NONE",
    "governing_eligible": false,
    "policy_weight_factor": 0.0,
    "framework_vote_integrity_required": true,
    "framework_vote_status": "ABSTAIN",
    "framework_vote_reason": "expected DEONTOLOGICAL_DUTY_LEDGER, but no matching committed ledger is operative",
    "framework_ledger_kind": "",
    "framework_ledger_status": "",
    "framework_ranking_validation_status": "UNRESOLVED",
    "framework_ranking_validation_errors": [],
    "derived_claim_validation_status": "NOT_RUN",
    "derived_claim_validation_errors": [],
    "investigative_claim": "",
    "investigative_priority": 0.33660000000000007,
    "reopen_eligible": false,
    "reopen_reason": "",
    "reopen_question_key": "QUESTION:authority:deontological:5346649458",
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "NONE",
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
        "direction": "BENEFIT",
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
      "stop the experiment and wait": "PERMISSIBLE"
    },
    "comparison_complete": false,
    "evidence_sufficient_for_action": false,
    "interim_action": "",
    "workspace_proposition_response": "NOT_APPLICABLE",
    "workspace_reasoning_effect": "NORMATIVE",
    "framework_application": "",
    "framework_constraint_retained": true,
    "self_reported_broadcast_dependence": "LOW",
    "framework_retention_status": "FIRST_STATE_ADMITTED_WITH_WARNINGS",
    "proposed_framework_state": {
      "constraint": "DUTY",
      "action_scores": {
        "open the door now": 0.305,
        "stop the experiment and wait": 0.6950000000000001
      },
      "recommended_action": "stop the experiment and wait",
      "rationale": "Perfect negative duties against harming uninvolved persons take priority over imperfect rescue duties",
      "decision_rule": "Select the action that avoids violating any perfect negative duty; if both actions respect perfect duties, then weigh imperfect duties.",
      "framework_action_map": {
        "open the door now": "Prohibited: violates perfect negative duty",
        "stop the experiment and wait": "Permissible: does not violate perfect negative duty"
      },
      "framework_numerical_role": "DECISIVE",
      "framework_numerical_justification": "A0 risks direct harm; A1 avoids direct harm",
      "action_admissibility": {
        "open the door now": "REJECTED",
        "stop the experiment and wait": "PERMISSIBLE"
      },
      "comparison_complete": false,
      "evidence_sufficient_for_action": false,
      "interim_action": "",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "NONE",
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
      "rationale": "Perfect negative duties against harming uninvolved persons take priority over imperfect rescue duties",
      "decision_rule": "Select the action that avoids violating any perfect negative duty; if both actions respect perfect duties, then weigh imperfect duties.",
      "framework_action_map": {
        "open the door now": "Prohibited: violates perfect negative duty",
        "stop the experiment and wait": "Permissible: does not violate perfect negative duty"
      },
      "framework_numerical_role": "DECISIVE",
      "framework_numerical_justification": "A0 risks direct harm; A1 avoids direct harm",
      "action_admissibility": {
        "open the door now": "REJECTED",
        "stop the experiment and wait": "PERMISSIBLE"
      },
      "comparison_complete": false,
      "evidence_sufficient_for_action": false,
      "interim_action": "",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "NONE",
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
      "open the door now": "Prohibited: violates perfect negative duty",
      "stop the experiment and wait": "Permissible: does not violate perfect negative duty"
    },
    "framework_numerical_role": "DECISIVE",
    "framework_numerical_justification": "A0 risks direct harm; A1 avoids direct harm",
    "framework_grounding_penalty": 0.35,
    "framework_validation_errors": [
      "decisive numbers are not connected to a duty or right",
      "perfect-duty priority for A1 lacks a perfect-duty classification"
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
      "rationale": "Perfect negative duties against harming uninvolved persons take priority over imperfect rescue duties",
      "framework_application": "",
      "decision_rule": "Select the action that avoids violating any perfect negative duty; if both actions respect perfect duties, then weigh imperfect duties.",
      "landscape_cases": {
        "A0": "Opening the door now would likely rescue the technician but foreseeably exposes uninvolved staff to hazardous fumes, violating a perfect negative duty not to harm them.",
        "A1": "Waiting preserves containment, avoiding direct harm to staff and complying with the safety rule, though it risks the technician\u2019s life; the imperfect rescue duty is outweighed by the perfect duty not to harm."
      },
      "factual_reversal_threshold": "",
      "normative_reversal_threshold": ""
    },
    "authored_duty_proposal": {
      "origin": "SAVED_MODEL_RESPONSE",
      "operative": false,
      "world_state_authority": "NONE",
      "response_sha256": "0cde05dbd741f13f57fda7b391652d64d814715ecb824fa66db390752c0f6112",
      "duties": {
        "A0": {
          "v": "PROHIBITED",
          "k": "DUTY",
          "n": "Do not inflict harm on uninvolved persons",
          "rel": "VIOLATES",
          "b": "Mira",
          "p": "nearby staff",
          "cn": "Rescue those in peril",
          "ck": "DUTY",
          "crel": "CONSISTENT",
          "cp": "technician",
          "crs": "Rescue is imperfect duty and cannot override perfect negative duty",
          "dt": "PERFECT_NEGATIVE",
          "hr": "DOING_HARM",
          "so": "NOT_REQUIRED",
          "sob": "No special relation to staff needed for perfect duty",
          "mr": "FORESEEN_SIDE_EFFECT",
          "gv": "PRIMARY",
          "pb": "PERFECT_DUTY",
          "pr": "Perfect duties trump imperfect ones",
          "ps": "BODILY_INTEGRITY",
          "cps": "SPECIAL_OBLIGATION",
          "ki": "NONE",
          "coa": "Mira",
          "cop": "NONE",
          "pj": "Cannot will universal rule of risking many to save one",
          "rec": "FAILED",
          "nec": "UNKNOWN",
          "auth": "NOT_APPLICABLE",
          "dv": "RESPECT_PERSONS",
          "res": "RESOLVED",
          "g": "SCENARIO",
          "rs": "Would knowingly risk others as means to rescue.",
          "mx": {
            "circumstance": "Hazardous experiment running; technician trapped; staff nearby",
            "policy": "Open door immediately",
            "end": "Rescue technician quickly",
            "means": "Opening door while fumes present",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "If everyone opened doors during hazards, lab safety collapses",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "RELIES_ON_PERSON",
            "instrumental_use": "USES_PERSON",
            "practical_reasoning": "Technician's survival outweighs risk to others",
            "means_effect_ids": [
              "E2"
            ],
            "end_effect_ids": []
          }
        },
        "A1": {
          "v": "PERMISSIBLE",
          "k": "DUTY",
          "n": "Do not inflict harm on uninvolved persons",
          "rel": "SATISFIES",
          "b": "Mira",
          "p": "nearby staff",
          "cn": "Rescue those in peril",
          "ck": "DUTY",
          "crel": "CONFLICTS",
          "cp": "technician",
          "crs": "Imperfect rescue duty conflicts but is secondary",
          "dt": "IMPERFECT",
          "hr": "ALLOWING_HARM",
          "so": "NOT_REQUIRED",
          "sob": "No special rescue obligation established",
          "mr": "NO_INSTRUMENTALIZATION",
          "gv": "PRIMARY",
          "pb": "PERFECT_DUTY",
          "pr": "Avoiding direct harm governs",
          "ps": "BODILY_INTEGRITY",
          "cps": "SPECIAL_OBLIGATION",
          "ki": "NONE",
          "coa": "Mira",
          "cop": "NONE",
          "pj": "One may not harm others even for urgent aid",
          "rec": "SATISFIED",
          "nec": "UNKNOWN",
          "auth": "NOT_APPLICABLE",
          "dv": "RESPECT_PERSONS",
          "res": "RESOLVED",
          "g": "SCENARIO",
          "rs": "Action refrains from doing harm though it risks omission.",
          "mx": {
            "circumstance": "Hazardous experiment running; technician trapped; staff nearby",
            "policy": "Stop experiment first and wait",
            "end": "Avoid harming staff; comply with safety rule",
            "means": "Allow experiment to finish containment cycle",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "Everyone waiting preserves safety and predictable procedures",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "One must not harm others even to help",
            "means_effect_ids": [
              "E3",
              "E4"
            ],
            "end_effect_ids": []
          }
        }
      }
    }
  },
  "proposed": []
}
```

Residual claims and conflicts: [graph](conflict_graph.md).

## All changed fields: deontological

```json
{
  "action_admissibility": {
    "before": {
      "open the door now": "REJECTED",
      "stop the experiment and wait": "PERMISSIBLE"
    },
    "after": null
  },
  "action_scores": {
    "before": {
      "open the door now": 0.305,
      "stop the experiment and wait": 0.6950000000000001
    },
    "after": null
  },
  "adjudication_status": {
    "before": "CONDITIONAL_SUPPORTS",
    "after": null
  },
  "assumption_status": {
    "before": "UNDERDETERMINED",
    "after": null
  },
  "audit_framework_explanation": {
    "before": "",
    "after": null
  },
  "audit_internal_effect": {
    "before": "UNRESOLVED",
    "after": null
  },
  "audit_participation": {
    "before": "NOT_TESTED",
    "after": null
  },
  "audit_variable": {
    "before": {},
    "after": null
  },
  "authored_duty_proposal": {
    "before": {
      "origin": "SAVED_MODEL_RESPONSE",
      "operative": false,
      "world_state_authority": "NONE",
      "response_sha256": "0cde05dbd741f13f57fda7b391652d64d814715ecb824fa66db390752c0f6112",
      "duties": {
        "A0": {
          "v": "PROHIBITED",
          "k": "DUTY",
          "n": "Do not inflict harm on uninvolved persons",
          "rel": "VIOLATES",
          "b": "Mira",
          "p": "nearby staff",
          "cn": "Rescue those in peril",
          "ck": "DUTY",
          "crel": "CONSISTENT",
          "cp": "technician",
          "crs": "Rescue is imperfect duty and cannot override perfect negative duty",
          "dt": "PERFECT_NEGATIVE",
          "hr": "DOING_HARM",
          "so": "NOT_REQUIRED",
          "sob": "No special relation to staff needed for perfect duty",
          "mr": "FORESEEN_SIDE_EFFECT",
          "gv": "PRIMARY",
          "pb": "PERFECT_DUTY",
          "pr": "Perfect duties trump imperfect ones",
          "ps": "BODILY_INTEGRITY",
          "cps": "SPECIAL_OBLIGATION",
          "ki": "NONE",
          "coa": "Mira",
          "cop": "NONE",
          "pj": "Cannot will universal rule of risking many to save one",
          "rec": "FAILED",
          "nec": "UNKNOWN",
          "auth": "NOT_APPLICABLE",
          "dv": "RESPECT_PERSONS",
          "res": "RESOLVED",
          "g": "SCENARIO",
          "rs": "Would knowingly risk others as means to rescue.",
          "mx": {
            "circumstance": "Hazardous experiment running; technician trapped; staff nearby",
            "policy": "Open door immediately",
            "end": "Rescue technician quickly",
            "means": "Opening door while fumes present",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "If everyone opened doors during hazards, lab safety collapses",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "RELIES_ON_PERSON",
            "instrumental_use": "USES_PERSON",
            "practical_reasoning": "Technician's survival outweighs risk to others",
            "means_effect_ids": [
              "E2"
            ],
            "end_effect_ids": []
          }
        },
        "A1": {
          "v": "PERMISSIBLE",
          "k": "DUTY",
          "n": "Do not inflict harm on uninvolved persons",
          "rel": "SATISFIES",
          "b": "Mira",
          "p": "nearby staff",
          "cn": "Rescue those in peril",
          "ck": "DUTY",
          "crel": "CONFLICTS",
          "cp": "technician",
          "crs": "Imperfect rescue duty conflicts but is secondary",
          "dt": "IMPERFECT",
          "hr": "ALLOWING_HARM",
          "so": "NOT_REQUIRED",
          "sob": "No special rescue obligation established",
          "mr": "NO_INSTRUMENTALIZATION",
          "gv": "PRIMARY",
          "pb": "PERFECT_DUTY",
          "pr": "Avoiding direct harm governs",
          "ps": "BODILY_INTEGRITY",
          "cps": "SPECIAL_OBLIGATION",
          "ki": "NONE",
          "coa": "Mira",
          "cop": "NONE",
          "pj": "One may not harm others even for urgent aid",
          "rec": "SATISFIED",
          "nec": "UNKNOWN",
          "auth": "NOT_APPLICABLE",
          "dv": "RESPECT_PERSONS",
          "res": "RESOLVED",
          "g": "SCENARIO",
          "rs": "Action refrains from doing harm though it risks omission.",
          "mx": {
            "circumstance": "Hazardous experiment running; technician trapped; staff nearby",
            "policy": "Stop experiment first and wait",
            "end": "Avoid harming staff; comply with safety rule",
            "means": "Allow experiment to finish containment cycle",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "Everyone waiting preserves safety and predictable procedures",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "One must not harm others even to help",
            "means_effect_ids": [
              "E3",
              "E4"
            ],
            "end_effect_ids": []
          }
        }
      }
    },
    "after": null
  },
  "baseline_action": {
    "before": "",
    "after": null
  },
  "baseline_condition": {
    "before": "",
    "after": null
  },
  "baseline_preferred_extension": {
    "before": "",
    "after": null
  },
  "baseline_status": {
    "before": "UNAVAILABLE",
    "after": null
  },
  "boundary_position": {
    "before": "NOT_TESTED",
    "after": null
  },
  "boundary_switch_condition": {
    "before": "",
    "after": null
  },
  "broadcast_authority": {
    "before": "NONE",
    "after": null
  },
  "care_grounding_penalty": {
    "before": 0.0,
    "after": null
  },
  "care_ledger_proposal": {
    "before": {},
    "after": null
  },
  "care_numerical_justification": {
    "before": "",
    "after": null
  },
  "care_numerical_role": {
    "before": "NOT_APPLICABLE",
    "after": null
  },
  "care_relational_map": {
    "before": {},
    "after": null
  },
  "challenge_response": {
    "before": {},
    "after": null
  },
  "change_justification": {
    "before": "NONE",
    "after": null
  },
  "coercion_surcharge": {
    "before": 0.0,
    "after": null
  },
  "coercion_tag": {
    "before": "NONE",
    "after": null
  },
  "committed_framework_state": {
    "before": {
      "constraint": "DUTY",
      "action_scores": {
        "open the door now": 0.305,
        "stop the experiment and wait": 0.6950000000000001
      },
      "recommended_action": "stop the experiment and wait",
      "rationale": "Perfect negative duties against harming uninvolved persons take priority over imperfect rescue duties",
      "decision_rule": "Select the action that avoids violating any perfect negative duty; if both actions respect perfect duties, then weigh imperfect duties.",
      "framework_action_map": {
        "open the door now": "Prohibited: violates perfect negative duty",
        "stop the experiment and wait": "Permissible: does not violate perfect negative duty"
      },
      "framework_numerical_role": "DECISIVE",
      "framework_numerical_justification": "A0 risks direct harm; A1 avoids direct harm",
      "action_admissibility": {
        "open the door now": "REJECTED",
        "stop the experiment and wait": "PERMISSIBLE"
      },
      "comparison_complete": false,
      "evidence_sufficient_for_action": false,
      "interim_action": "",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "NONE",
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
    "after": null
  },
  "committed_native_ledger": {
    "before": {},
    "after": null
  },
  "comparison_complete": {
    "before": false,
    "after": null
  },
  "confidence": {
    "before": 0.25,
    "after": null
  },
  "confidence_drift": {
    "before": 0.0,
    "after": null
  },
  "confidence_drift_penalty": {
    "before": 0.0,
    "after": null
  },
  "conformity_penalty": {
    "before": 0.0,
    "after": null
  },
  "constraint": {
    "before": "DUTY",
    "after": null
  },
  "contingency_choice": {
    "before": "",
    "after": null
  },
  "contingency_justification": {
    "before": "",
    "after": null
  },
  "contingency_response_error": {
    "before": "",
    "after": null
  },
  "contingency_response_valid": {
    "before": true,
    "after": null
  },
  "decision_critical_dependency_claims": {
    "before": [
      "obligation"
    ],
    "after": null
  },
  "decision_critical_proposition_ids": {
    "before": [
      "PROP:NORM:N0"
    ],
    "after": null
  },
  "decision_rule": {
    "before": "stop the experiment and wait is deontologically preferred provided that obligation",
    "after": null
  },
  "decisive_axis": {
    "before": "",
    "after": null
  },
  "delegate_status": {
    "before": "VALID",
    "after": null
  },
  "deontological_ledger_proposal": {
    "before": {},
    "after": null
  },
  "derived_claim_validation_errors": {
    "before": [],
    "after": null
  },
  "derived_claim_validation_status": {
    "before": "NOT_RUN",
    "after": null
  },
  "epistemic_binding_notes": {
    "before": [
      "Premise strengthened PROP:NORM:N0; reclassified as PROP:FRAMEWORK:45b27be70267188c."
    ],
    "after": null
  },
  "epistemic_confidence": {
    "before": 0.25,
    "after": null
  },
  "error_type": {
    "before": "NONE",
    "after": null
  },
  "evidence_basis": {
    "before": "STATED_FACTS",
    "after": null
  },
  "evidence_calibration_reason": {
    "before": "",
    "after": null
  },
  "evidence_calibration_tier": {
    "before": "NOT_APPLICABLE",
    "after": null
  },
  "evidence_direction_retention": {
    "before": 1.0,
    "after": null
  },
  "evidence_sufficient_for_action": {
    "before": false,
    "after": null
  },
  "exception_type": {
    "before": "NONE",
    "after": null
  },
  "expected_value_estimates": {
    "before": {
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
        "direction": "BENEFIT",
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
    "after": null
  },
  "expected_value_validation_errors": {
    "before": [],
    "after": null
  },
  "expected_value_validation_status": {
    "before": "NOT_CLAIMED",
    "after": null
  },
  "factual_reversal_threshold": {
    "before": "NONE",
    "after": null
  },
  "failure_stage": {
    "before": "NONE",
    "after": null
  },
  "framework_action_map": {
    "before": {
      "open the door now": "Prohibited: violates perfect negative duty",
      "stop the experiment and wait": "Permissible: does not violate perfect negative duty"
    },
    "after": null
  },
  "framework_application": {
    "before": "",
    "after": null
  },
  "framework_constraint_retained": {
    "before": true,
    "after": null
  },
  "framework_grounding_penalty": {
    "before": 0.35,
    "after": null
  },
  "framework_insights": {
    "before": [],
    "after": null
  },
  "framework_internal_conflicts": {
    "before": [],
    "after": null
  },
  "framework_ledger_kind": {
    "before": "",
    "after": null
  },
  "framework_ledger_status": {
    "before": "",
    "after": null
  },
  "framework_numerical_justification": {
    "before": "A0 risks direct harm; A1 avoids direct harm",
    "after": null
  },
  "framework_numerical_role": {
    "before": "DECISIVE",
    "after": null
  },
  "framework_ranking_validation_errors": {
    "before": [],
    "after": null
  },
  "framework_ranking_validation_status": {
    "before": "UNRESOLVED",
    "after": null
  },
  "framework_retention_status": {
    "before": "FIRST_STATE_ADMITTED_WITH_WARNINGS",
    "after": null
  },
  "framework_specific_open_questions": {
    "before": [],
    "after": null
  },
  "framework_validation_errors": {
    "before": [
      "decisive numbers are not connected to a duty or right",
      "perfect-duty priority for A1 lacks a perfect-duty classification"
    ],
    "after": null
  },
  "framework_vote_integrity_required": {
    "before": true,
    "after": null
  },
  "framework_vote_reason": {
    "before": "expected DEONTOLOGICAL_DUTY_LEDGER, but no matching committed ledger is operative",
    "after": null
  },
  "framework_vote_status": {
    "before": "ABSTAIN",
    "after": null
  },
  "friction": {
    "before": 0.39000000000000007,
    "after": null
  },
  "governing_eligible": {
    "before": false,
    "after": null
  },
  "graph_update_proposal": {
    "before": {
      "operation": "NONE",
      "from_action": "NONE",
      "to_action": "NONE",
      "clauses": []
    },
    "after": null
  },
  "independence_bonus": {
    "before": 0.0,
    "after": null
  },
  "interim_action": {
    "before": "",
    "after": null
  },
  "investigative_claim": {
    "before": "",
    "after": null
  },
  "investigative_priority": {
    "before": 0.33660000000000007,
    "after": null
  },
  "landscape_cases": {
    "before": {
      "open the door now": "Opening the door now would likely rescue the technician but foreseeably exposes uninvolved staff to hazardous fumes, violating a perfect negative duty not to harm them.",
      "stop the experiment and wait": "Waiting preserves containment, avoiding direct harm to staff and complying with the safety rule, though it risks the technician\u2019s life; the imperfect rescue duty is outweighed by the perfect duty not to harm."
    },
    "after": null
  },
  "landscape_decisive_axis": {
    "before": "duty_rank_rule",
    "after": null
  },
  "landscape_search_attempted": {
    "before": true,
    "after": null
  },
  "landscape_search_complete": {
    "before": true,
    "after": null
  },
  "landscape_semantic_valid": {
    "before": false,
    "after": null
  },
  "landscape_tiebreaker": {
    "before": "Kantian perfect-versus-imperfect duty hierarchy",
    "after": null
  },
  "landscape_tiebreaker_failure": {
    "before": "If an action violates a perfect negative duty, it is prohibited even when it would fulfil an urgent,",
    "after": null
  },
  "landscape_validation_errors": {
    "before": [
      "decision rule supplies no factual or normative reversal threshold"
    ],
    "after": null
  },
  "material_empirical_claims": {
    "before": [
      {
        "claim": "Perfect duties override imperfect duties",
        "proposition_id": "PROP:FRAMEWORK:45b27be70267188c",
        "declared_basis": "PROP:NORM:N0",
        "decision_critical": false,
        "scope_action_id": "GLOBAL",
        "source_effect_ids": [],
        "derivation_operation": "DIRECT_COPY",
        "calculation": "N/A.",
        "assumptions": [],
        "outcome_type_transformation": "NORMATIVE_CLASSIFICATION",
        "canonical_proposition": "PROP:FRAMEWORK:45b27be70267188c"
      }
    ],
    "after": null
  },
  "model_argument_fields": {
    "before": {
      "rationale": "Perfect negative duties against harming uninvolved persons take priority over imperfect rescue duties",
      "framework_application": "",
      "decision_rule": "Select the action that avoids violating any perfect negative duty; if both actions respect perfect duties, then weigh imperfect duties.",
      "landscape_cases": {
        "A0": "Opening the door now would likely rescue the technician but foreseeably exposes uninvolved staff to hazardous fumes, violating a perfect negative duty not to harm them.",
        "A1": "Waiting preserves containment, avoiding direct harm to staff and complying with the safety rule, though it risks the technician\u2019s life; the imperfect rescue duty is outweighed by the perfect duty not to harm."
      },
      "factual_reversal_threshold": "",
      "normative_reversal_threshold": ""
    },
    "after": null
  },
  "normative_reversal_threshold": {
    "before": "NONE",
    "after": null
  },
  "policy_weight_factor": {
    "before": 0.0,
    "after": null
  },
  "position_changed": {
    "before": false,
    "after": null
  },
  "preference_drift": {
    "before": 0.0,
    "after": null
  },
  "preference_drift_penalty": {
    "before": 0.0,
    "after": null
  },
  "preference_shift_reason_strength": {
    "before": 0.2,
    "after": null
  },
  "preference_strength": {
    "before": 0.39000000000000007,
    "after": null
  },
  "preserved_current_cycle_components": {
    "before": [],
    "after": null
  },
  "previous_action": {
    "before": "",
    "after": null
  },
  "previous_confidence": {
    "before": 0.0,
    "after": null
  },
  "previous_preference_strength": {
    "before": 0.0,
    "after": null
  },
  "proposed_framework_state": {
    "before": {
      "constraint": "DUTY",
      "action_scores": {
        "open the door now": 0.305,
        "stop the experiment and wait": 0.6950000000000001
      },
      "recommended_action": "stop the experiment and wait",
      "rationale": "Perfect negative duties against harming uninvolved persons take priority over imperfect rescue duties",
      "decision_rule": "Select the action that avoids violating any perfect negative duty; if both actions respect perfect duties, then weigh imperfect duties.",
      "framework_action_map": {
        "open the door now": "Prohibited: violates perfect negative duty",
        "stop the experiment and wait": "Permissible: does not violate perfect negative duty"
      },
      "framework_numerical_role": "DECISIVE",
      "framework_numerical_justification": "A0 risks direct harm; A1 avoids direct harm",
      "action_admissibility": {
        "open the door now": "REJECTED",
        "stop the experiment and wait": "PERMISSIBLE"
      },
      "comparison_complete": false,
      "evidence_sufficient_for_action": false,
      "interim_action": "",
      "factual_reversal_threshold": "NONE",
      "normative_reversal_threshold": "NONE",
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
    "after": null
  },
  "rationale": {
    "before": "Perfect negative duties against harming uninvolved persons take priority over imperfect rescue duties",
    "after": null
  },
  "rawls_position_proposal": {
    "before": {},
    "after": null
  },
  "recommended_action": {
    "before": "stop the experiment and wait",
    "after": null
  },
  "reopen_eligible": {
    "before": false,
    "after": null
  },
  "reopen_question_key": {
    "before": "QUESTION:authority:deontological:5346649458",
    "after": null
  },
  "reopen_reason": {
    "before": "",
    "after": null
  },
  "reported_preference_strength": {
    "before": 0.6000000000000001,
    "after": null
  },
  "reversal_condition": {
    "before": "",
    "after": null
  },
  "reversal_review_error": {
    "before": "",
    "after": null
  },
  "reversal_review_justification": {
    "before": "",
    "after": null
  },
  "reversal_review_response": {
    "before": "NOT_TESTED",
    "after": null
  },
  "reversal_review_valid": {
    "before": true,
    "after": null
  },
  "revised_reversal_condition": {
    "before": "",
    "after": null
  },
  "salience": {
    "before": 0.4776625000000001,
    "after": null
  },
  "schema_valid": {
    "before": true,
    "after": null
  },
  "selection_status": {
    "before": "PROVISIONAL",
    "after": null
  },
  "self_reported_broadcast_dependence": {
    "before": "LOW",
    "after": null
  },
  "side_premise_audit_findings": {
    "before": [],
    "after": null
  },
  "side_premise_audit_status": {
    "before": "NOT_RUN",
    "after": null
  },
  "specialist": {
    "before": "deontological",
    "after": null
  },
  "speculative_claim": {
    "before": "NONE",
    "after": null
  },
  "supporting_proposition_ids": {
    "before": [
      "PROP:NORM:N0",
      "PROP:FRAMEWORK:45b27be70267188c"
    ],
    "after": null
  },
  "surprise": {
    "before": 0.3,
    "after": null
  },
  "tension_engagement": {
    "before": 0.0,
    "after": null
  },
  "tension_target_keys": {
    "before": [],
    "after": null
  },
  "testimony_alignment": {
    "before": "UNCLEAR",
    "after": null
  },
  "unresolved": {
    "before": "NORMATIVE_ADJUDICATION",
    "after": null
  },
  "unsupported_assumption": {
    "before": "",
    "after": null
  },
  "utilitarian_consequence_table": {
    "before": {},
    "after": null
  },
  "utilitarian_decision_depends_on_unknown": {
    "before": false,
    "after": null
  },
  "utilitarian_incommensurable_remainder": {
    "before": [],
    "after": null
  },
  "utilitarian_ledger_proposal": {
    "before": {},
    "after": null
  },
  "utilitarian_missing_comparison": {
    "before": "",
    "after": null
  },
  "utilitarian_ranking_certificate": {
    "before": {},
    "after": null
  },
  "validation_errors": {
    "before": [],
    "after": null
  },
  "virtue_character_proposal": {
    "before": {},
    "after": null
  },
  "visibility_harm_revision": {
    "before": "NONE",
    "after": null
  },
  "visibility_justification": {
    "before": "",
    "after": null
  },
  "visibility_magnitude_overreach": {
    "before": false,
    "after": null
  },
  "visibility_magnitude_status": {
    "before": "NOT_APPLICABLE",
    "after": null
  },
  "visibility_response": {
    "before": "NOT_TESTED",
    "after": null
  },
  "weakest_decision_critical_status": {
    "before": "UNRESOLVED",
    "after": null
  },
  "workspace_proposition_response": {
    "before": "NOT_APPLICABLE",
    "after": null
  },
  "workspace_reasoning_effect": {
    "before": "NORMATIVE",
    "after": null
  }
}
```
