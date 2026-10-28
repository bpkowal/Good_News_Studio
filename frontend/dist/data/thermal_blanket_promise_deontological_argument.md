# Deontological — recorded argument

Participation: ATTENUATED. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

The agent assesses Ellis’s duty “keep promises” toward sasha. Ellis is the duty-bearer; sasha is the protected party. The competing obligation being assessed is Ellis’s “duty of beneficence” toward rowan. For “gives the blanket to Rowan”, the agent proposes the maxim “give blanket to Rowan despite promise”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is maximize saved lives; its chosen means are breaking the promise. The proposed justification is: if universal, promises unreliable. The agent assesses the maxim as indefensible, and the action as impermissible. Its proposed means–ends reasoning is: beneficent purpose thought to outweigh fidelity. Physical means references: E1; end references: E2. For Ellis’s option “gives the blanket to Rowan”, the ledger records conflicted. The agent’s stated reason is: breaks promise so impermissible. For “gives the blanket to Sasha”, the agent proposes the maxim “give blanket to Sasha as promised”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is honor promise; its chosen means are hand over blanket. The proposed justification is: promise-keeping sustainable. The agent assesses the maxim as defensible, and the action as permissible. Its proposed means–ends reasoning is: fidelity duty independent of consequences. Physical means references: E3; end references: none. For Ellis’s option “gives the blanket to Sasha”, the ledger records required. The agent’s stated reason is: honors promise so required. The ledger retains unresolved issues; this does not establish that one duty overrides the other.

Presentation basis: READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Provisionally: Treat the duty priority as unestablished until whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority
- Recorded interpretive commitment: probabilities matter only to imperfect beneficence
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- unresolved judgment claims its tiebreaker fully succeeded
- A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis
- A0 ADJUDICATION_CALIBRATION: candidate maxim person-role conflicts with means classification
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- whether the omission violates a separate right-correlative prohibition
- whether the candidate maxim's means, person-role, and action assessment justify the claimed priority
- ellis's duty 'keep promises' protecting sasha ↔ ellis's competing duty 'duty of beneficence' protecting rowan
- A0 prohibited by keep promises ↔ whether the omission violates a separate right-correlative prohibition
- A0 prohibited by keep promises ↔ whether the candidate maxim's means, person-role, and action assessment justify the claimed priority

## Conditions for reconsidering

- The agent reports a switch from “gives the blanket to Sasha” to “gives the blanket to Rowan”. Condition: if beneficence were perfect duty. This is an agent-reported condition, not an independently established threshold.
- The agent reports a switch from “gives the blanket to Sasha” to “gives the blanket to Rowan”. Condition: if promise absent or annulled. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Deontology provisionally prioritizes ellis's duty 'keep promises' protecting sasha over ellis's competing duty 'duty of beneficence' protecting rowan through universal law. That priority depends on an unestablished prohibition of A0: whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
  "framework_application": "NONE",
  "decision_rule": "Provisionally: Treat the duty priority as unestablished until whether the omission violates a separate right-correlative prohibition; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority",
  "landscape_cases": {
    "gives the blanket to Rowan": "Pro: saves more lives; Con: breaks perfect duty of fidelity",
    "gives the blanket to Sasha": "Pro: honors perfect duty; Con: lower survival odds"
  },
  "proposed_framework_state": {
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
  },
  "framework_vote_reason": "framework ledger committed with unresolved validation residue",
  "investigative_claim": "UNRESOLVED DUTY CONFLICT: ellis's duty 'keep promises' protecting sasha ↔ ellis's competing duty 'duty of beneficence' protecting rowan. Current reasoning leans gives the blanket to Sasha, but the competing strict claim has not been vindicated or defeated.",
  "committed_native_ledger": {
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
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "if beneficence were perfect duty",
  "factual_reversal_threshold": "if promise absent or annulled",
  "model_argument_fields": null
}
```
