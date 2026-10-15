# Deontological — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

The agent assesses Maria’s duty “do not kill innocent persons” toward one worker. Maria is the duty-bearer; one worker is the protected party. The competing obligation being assessed is Maria’s “imperfect duty to rescue” toward five workers. For Maria’s option “pulls the lever”, the ledger records conflicted. The agent’s stated reason is: Killing one as a means/side-effect breaches universal rule. The agent assesses Maria’s duty “imperfect duty to aid others” toward five workers. Maria is the duty-bearer; five workers is the protected party. The competing obligation being assessed is Maria’s “perfect duty not to kill” toward one worker. For Maria’s option “does not pull the lever”, the ledger records permissible. The agent’s stated reason is: Omission respects perfect duty hierarchy. The ledger retains unresolved issues; this does not establish that one duty overrides the other.

Presentation basis: READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Prefer the action that avoids violating a perfect negative duty even when an imperfect rescue duty remains unmet.
- Recorded interpretive commitment: Numbers do not override duty categories
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; five workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- maria's duty 'imperfect duty to aid others' protecting five workers ↔ maria's competing duty 'perfect duty not to kill' protecting one worker
- A0 prohibited by do not kill innocent persons ↔ whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict

## Conditions for reconsidering

- The agent reports a switch from “does not pull the lever” to “pulls the lever”. Condition: If imperfect rescue duties were equal or higher priority than perfect negative duties, ranking would reverse. This is an agent-reported condition, not an independently established threshold.
- The agent reports a switch from “does not pull the lever” to “pulls the lever”. Condition: If the side-track worker could consent or be unharmed, ranking would reverse. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Deontology provisionally prioritizes maria's duty 'imperfect duty to aid others' protecting five workers over maria's competing duty 'perfect duty not to kill' protecting one worker through universal law. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict. The comparative Kantian judgment remains contested.",
  "framework_application": "NONE",
  "decision_rule": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
  "landscape_cases": {
    "pulls the lever": "PROHIBITED: Pulling the lever directly initiates lethal force against an innocent worker, violating the perfect negative duty not to kill.",
    "does not pull the lever": "PERMISSIBLE: Refraining leaves Maria sadly allowing harm, but she does not directly violate the perfect duty; the rescue duty is only imperfect."
  },
  "proposed_framework_state": {
    "constraint": "DUTY",
    "action_scores": {
      "pulls the lever": 0.305,
      "does not pull the lever": 0.6950000000000001
    },
    "recommended_action": "does not pull the lever",
    "rationale": "Deontology provisionally prioritizes maria's duty 'imperfect duty to aid others' protecting five workers over maria's competing duty 'perfect duty not to kill' protecting one worker through universal law. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict. The comparative Kantian judgment remains contested.",
    "decision_rule": "Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict",
    "framework_action_map": {
      "pulls the lever": "PROHIBITED: violates perfect negative duty not to kill; uses lethal action against an innocent person.",
      "does not pull the lever": "PERMISSIBLE: allows harm but does not directly violate any perfect duty; only fails an imperfect rescue duty."
    },
    "framework_numerical_role": "SECONDARY",
    "framework_numerical_justification": "Numbers do not override duty categories",
    "action_admissibility": {
      "pulls the lever": "REJECTED",
      "does not pull the lever": "PERMISSIBLE"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "does not pull the lever",
    "factual_reversal_threshold": "If the side-track worker could consent or be unharmed, ranking would reverse.",
    "normative_reversal_threshold": "If imperfect rescue duties were equal or higher priority than perfect negative duties, ranking would reverse.",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "DUTY",
          "norm": "do not kill innocent persons",
          "relation": "VIOLATES",
          "duty_bearer": "Maria",
          "protected_party": "one worker",
          "competing_norm": "imperfect duty to rescue",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "five workers",
          "competing_reason": "saving more lives is an imperfect duty, conflicted with perfect duty",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "DOING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relation needed for perfect duty",
          "means_relation": "FORESEEN_SIDE_EFFECT",
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "perfect negative duties have priority",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "EXTERNAL_FREEDOM",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Cannot universalise intentional killing to save others",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Killing one as a means/side-effect breaches universal rule"
        },
        {
          "action_id": "A1",
          "verdict": "PERMISSIBLE",
          "norm_kind": "DUTY",
          "norm": "imperfect duty to aid others",
          "relation": "CONSISTENT",
          "duty_bearer": "Maria",
          "protected_party": "five workers",
          "competing_norm": "perfect duty not to kill",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONSISTENT",
          "competing_protected_party": "one worker",
          "competing_reason": "no perfect duty violated",
          "duty_type": "IMPERFECT",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "no special relationship established",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "governing_norm": "PRIMARY",
          "priority_basis": "AUTONOMY",
          "priority_rule": "omission does not breach perfect duty",
          "protected_standing": "BODILY_INTEGRITY",
          "competing_protected_standing": "EXTERNAL_FREEDOM",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Universal law can permit inaction where acting would kill",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Omission respects perfect duty hierarchy"
        }
      ]
    },
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": ""
  },
  "framework_vote_reason": "the duty ledger permits the selected action but does not rank it above its rivals",
  "investigative_claim": "UNRESOLVED DUTY CONFLICT: maria's duty 'imperfect duty to aid others' protecting five workers ↔ maria's competing duty 'perfect duty not to kill' protecting one worker. Current reasoning leans does not pull the lever, but the competing strict claim has not been vindicated or defeated.",
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
        "norm_node_id": "DEON_NORM:35d2199789086808",
        "party_node_id": "DEON_PARTY:879bcf30890178b8",
        "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
        "competing_norm": "imperfect duty to rescue",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "five workers",
        "competing_reason": "saving more lives is an imperfect duty, conflicted with perfect duty",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "DOING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special relation needed for perfect duty",
        "means_relation": "FORESEEN_SIDE_EFFECT",
        "governing_norm": "UNRESOLVED",
        "priority_basis": "UNRESOLVED",
        "priority_rule": "perfect negative duties have priority",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "EXTERNAL_FREEDOM",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "Cannot universalise intentional killing to save others",
        "reciprocity_status": "FAILED",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "UNIVERSAL_LAW",
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
        "reason": "Killing one as a means/side-effect breaches universal rule",
        "protected_party": "one worker",
        "norm": "do not kill innocent persons",
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
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:1b1672077c245673",
        "party_node_id": "DEON_PARTY:e099130fb57c35fc",
        "bearer_node_id": "DEON_BEARER:94aec9fbed989ece",
        "competing_norm": "perfect duty not to kill",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONSISTENT",
        "competing_protected_party": "one worker",
        "competing_reason": "no perfect duty violated",
        "duty_type": "IMPERFECT",
        "harm_relation": "ALLOWING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "no special relationship established",
        "means_relation": "NO_INSTRUMENTALIZATION",
        "governing_norm": "PRIMARY",
        "priority_basis": "AUTONOMY",
        "priority_rule": "omission does not breach perfect duty",
        "protected_standing": "BODILY_INTEGRITY",
        "competing_protected_standing": "EXTERNAL_FREEDOM",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "Universal law can permit inaction where acting would kill",
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
        "reason": "Omission respects perfect duty hierarchy",
        "protected_party": "five workers",
        "norm": "imperfect duty to aid others",
        "duty_bearer": "maria",
        "provenance": [
          "delegate:deontological",
          "cycle:1"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "If imperfect rescue duties were equal or higher priority than perfect negative duties, ranking would reverse.",
  "factual_reversal_threshold": "If the side-track worker could consent or be unharmed, ranking would reverse.",
  "model_argument_fields": {
    "rationale": "Perfect duty not to kill",
    "framework_application": "NONE",
    "decision_rule": "Prefer the action that avoids violating a perfect negative duty even when an imperfect rescue duty remains unmet.",
    "landscape_cases": {
      "A0": "PROHIBITED: Pulling the lever directly initiates lethal force against an innocent worker, violating the perfect negative duty not to kill.",
      "A1": "PERMISSIBLE: Refraining leaves Maria sadly allowing harm, but she does not directly violate the perfect duty; the rescue duty is only imperfect."
    },
    "factual_reversal_threshold": "If the side-track worker could consent or be unharmed, ranking would reverse.",
    "normative_reversal_threshold": "If imperfect rescue duties were equal or higher priority than perfect negative duties, ranking would reverse."
  }
}
```
