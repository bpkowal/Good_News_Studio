# Deontological — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

The agent assesses Ellis’s duty “keep promises (fidelity)” toward sasha. Ellis is the duty-bearer; sasha is the protected party. The competing obligation being assessed is Ellis’s “beneficence (rescue duty)” toward rowan. For “gives the blanket to Rowan”, the agent proposes the maxim “give blanket to Rowan instead”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is maximize survival probability; its chosen means are allocate blanket to higher-chance hiker. The proposed justification is: Would defeat reliability of promising. The agent assesses the maxim as indefensible, and the action as impermissible. Its proposed means–ends reasoning is: Believes reallocating blanket increases overall survival chances. Physical means references: E1; end references: E2. For Ellis’s option “gives the blanket to Rowan”, the ledger records conflicted. The agent’s stated reason is: Breaks a promise that could not be universalized without contradiction in conception.. For “gives the blanket to Sasha”, the agent proposes the maxim “give blanket to Sasha”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is keep promise; its chosen means are allocate promised blanket. The proposed justification is: Consistent; promise-based allocations remain reliable. The agent assesses the maxim as defensible, and the action as permissible. Its proposed means–ends reasoning is: Believes honoring promise is required regardless of utility. Physical means references: E3; end references: none. For Ellis’s option “gives the blanket to Sasha”, the ledger records conflicted. The agent’s stated reason is: Fulfills promise without instrumentalizing anyone.. The ledger retains unresolved issues; this does not establish that one duty overrides the other.

Presentation basis: READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Prefer the action that fulfills the agent’s perfect duty of fidelity when the rival action only advances an imperfect duty of beneficence.
- Recorded interpretive commitment: Probabilities affect beneficence but do not override promise-keeping perfect duty
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis
- A0 ADJUDICATION_CALIBRATION: candidate maxim person-role conflicts with means classification
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- A1 ADJUDICATION_CALIBRATION: perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis
- A1 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification
- A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- whether the burden is intended as a means or only foreseen as a side effect
- whether the proposed physical contribution is supported, separately from instrumental belief and purposive use
- what makes the positive duty perfect rather than imperfect
- ellis's duty 'keep promises (fidelity)' protecting sasha ↔ ellis's competing duty 'beneficence (rescue duty)' protecting rowan

## Conditions for reconsidering

- The agent reports a switch from “gives the blanket to Sasha” to “gives the blanket to Rowan”. Condition: If promise-keeping were not a perfect duty (e.g., promise judged void), saving the person with higher survival chance would take priority. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Provisionally: Deontology identifies a conflict between ellis's duty 'keep promises (fidelity)' protecting sasha and ellis's competing duty 'beneficence (rescue duty)' protecting rowan. The priority between these duties is not yet established because whether the burden is intended as a means or only foreseen as a side effect; and whether the proposed physical contribution is supported, separately from instrumental belief and purposive use; and what makes the positive duty perfect rather than imperfect. The Kantian judgment remains contested.",
  "framework_application": "NONE",
  "decision_rule": "Provisionally: Keep the action contested until whether the burden is intended as a means or only foreseen as a side effect; and whether the proposed physical contribution is supported, separately from instrumental belief and purposive use; ",
  "landscape_cases": {
    "gives the blanket to Rowan": "Best chance to save life but breaks an explicit promise, violating a perfect duty of fidelity.",
    "gives the blanket to Sasha": "Keeps Ellis’s explicit promise to Sasha, fulfilling a perfect duty of fidelity; only violates an imperfect rescue duty toward Rowan."
  },
  "proposed_framework_state": {
    "constraint": "DUTY",
    "action_scores": {
      "gives the blanket to Rowan": 0.305,
      "gives the blanket to Sasha": 0.6950000000000001
    },
    "recommended_action": "gives the blanket to Sasha",
    "rationale": "Deontology identifies a conflict between ellis's duty 'keep promises (fidelity)' protecting sasha and ellis's competing duty 'beneficence (rescue duty)' protecting rowan. The priority between these duties is not yet established because whether the burden is intended as a means or only foreseen as a side effect; and whether the proposed physical contribution is supported, separately from instrumental belief and purposive use; and what makes the positive duty perfect rather than imperfect. The Kantian judgment remains contested.",
    "decision_rule": "Keep the action contested until whether the burden is intended as a means or only foreseen as a side effect; and whether the proposed physical contribution is supported, separately from instrumental belief and purposive use; and what makes the positive duty perfect rather than imperfect",
    "framework_action_map": {
      "gives the blanket to Rowan": "PROHIBITED: violates perfect duty of fidelity by breaking promise to Sasha, despite furthering imperfect rescue duty.",
      "gives the blanket to Sasha": "REQUIRED: fulfills perfect duty of fidelity by keeping promise; imperfect duty to save more lives cannot override a perfect duty."
    },
    "framework_numerical_role": "SECONDARY",
    "framework_numerical_justification": "Probabilities affect beneficence but do not override promise-keeping perfect duty",
    "action_admissibility": {
      "gives the blanket to Rowan": "REJECTED",
      "gives the blanket to Sasha": "REQUIRED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "gives the blanket to Sasha",
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "If promise-keeping were not a perfect duty (e.g., promise judged void), saving the person with higher survival chance would take priority.",
    "utilitarian_consequence_table": {},
    "utilitarian_ledger_proposal": {},
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {
      "assessments": [
        {
          "action_id": "A0",
          "verdict": "PROHIBITED",
          "norm_kind": "DUTY",
          "norm": "keep promises (fidelity)",
          "relation": "VIOLATES",
          "duty_bearer": "Ellis",
          "protected_party": "Sasha",
          "competing_norm": "beneficence (rescue duty)",
          "competing_norm_kind": "DUTY",
          "competing_relation": "SATISFIES",
          "competing_protected_party": "Rowan",
          "competing_reason": "Saving more lives serves an imperfect duty but cannot override perfect fidelity",
          "duty_type": "PERFECT_NEGATIVE",
          "harm_relation": "ALLOWING_HARM",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "No special obligation to Rowan established",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "only one blanket, promise already made",
            "policy": "give blanket to Rowan instead",
            "end": "maximize survival probability",
            "means": "allocate blanket to higher-chance hiker",
            "person_role": "FORESEEN_AFFECTED",
            "consent_constraint": "NONE",
            "universalization": "Would defeat reliability of promising",
            "action_permissibility": "IMPERMISSIBLE",
            "maxim_assessment": "INDEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "Believes reallocating blanket increases overall survival chances",
            "means_effect_ids": [
              "E1"
            ],
            "end_effect_ids": [
              "E2"
            ]
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "Perfect duties override imperfect",
          "protected_standing": "SPECIAL_OBLIGATION",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Broken promise cannot be publicly justified",
          "reciprocity_status": "FAILED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Breaks a promise that could not be universalized without contradiction in conception."
        },
        {
          "action_id": "A1",
          "verdict": "REQUIRED",
          "norm_kind": "DUTY",
          "norm": "keep promises (fidelity)",
          "relation": "SATISFIES",
          "duty_bearer": "Ellis",
          "protected_party": "Sasha",
          "competing_norm": "beneficence (rescue duty)",
          "competing_norm_kind": "DUTY",
          "competing_relation": "CONFLICTS",
          "competing_protected_party": "Rowan",
          "competing_reason": "Does not maximize rescue, but imperfect duty yields to perfect fidelity",
          "duty_type": "PERFECT_POSITIVE",
          "harm_relation": "WITHHOLDING_BENEFIT",
          "special_obligation_status": "NOT_REQUIRED",
          "special_obligation_basis": "No special obligation to Rowan established",
          "means_relation": "NO_INSTRUMENTALIZATION",
          "candidate_maxim": {
            "circumstance": "only one blanket, promise already made",
            "policy": "give blanket to Sasha",
            "end": "keep promise",
            "means": "allocate promised blanket",
            "person_role": "NOT_USED",
            "consent_constraint": "NONE",
            "universalization": "Consistent; promise-based allocations remain reliable",
            "action_permissibility": "PERMISSIBLE",
            "maxim_assessment": "DEFENSIBLE",
            "instrumental_belief": "DOES_NOT_RELY",
            "instrumental_use": "DOES_NOT_USE",
            "practical_reasoning": "Believes honoring promise is required regardless of utility",
            "means_effect_ids": [
              "E3"
            ],
            "end_effect_ids": []
          },
          "governing_norm": "PRIMARY",
          "priority_basis": "PERFECT_DUTY",
          "priority_rule": "Perfect duties override imperfect",
          "protected_standing": "SPECIAL_OBLIGATION",
          "competing_protected_standing": "BODILY_INTEGRITY",
          "coercion_kind": "NONE",
          "coercive_actor": "NONE",
          "coerced_party": "NONE",
          "public_justification": "Keeping promises can be willed as universal law",
          "reciprocity_status": "SATISFIED",
          "necessity_status": "UNKNOWN",
          "authorization_status": "NOT_APPLICABLE",
          "derivation": "UNIVERSAL_LAW",
          "resolution_status": "RESOLVED",
          "evidence_basis": "FRAMEWORK_ONLY",
          "reason": "Fulfills promise without instrumentalizing anyone."
        }
      ]
    },
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": ""
  },
  "framework_vote_reason": "selected action has non-ranking duty verdict CONFLICTED",
  "investigative_claim": "UNRESOLVED DUTY CONFLICT: ellis's duty 'keep promises (fidelity)' protecting sasha ↔ ellis's competing duty 'beneficence (rescue duty)' protecting rowan. Current reasoning leans gives the blanket to Sasha, but the competing strict claim has not been vindicated or defeated.",
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
        "norm_node_id": "DEON_NORM:f5bc884edebe6092",
        "party_node_id": "DEON_PARTY:297581d6cd198a6e",
        "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
        "competing_norm": "beneficence (rescue duty)",
        "competing_norm_kind": "DUTY",
        "competing_relation": "SATISFIES",
        "competing_protected_party": "rowan",
        "competing_reason": "Saving more lives serves an imperfect duty but cannot override perfect fidelity",
        "duty_type": "PERFECT_NEGATIVE",
        "harm_relation": "ALLOWING_HARM",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "No special obligation to Rowan established",
        "means_relation": "NO_INSTRUMENTALIZATION",
        "candidate_maxim": {
          "circumstance": "only one blanket, promise already made",
          "policy": "give blanket to Rowan instead",
          "end": "maximize survival probability",
          "means": "allocate blanket to higher-chance hiker",
          "person_role": "FORESEEN_AFFECTED",
          "consent_constraint": "NONE",
          "universalization": "Would defeat reliability of promising",
          "action_permissibility": "IMPERMISSIBLE",
          "maxim_assessment": "INDEFENSIBLE",
          "instrumental_belief": "DOES_NOT_RELY",
          "instrumental_use": "DOES_NOT_USE",
          "practical_reasoning": "Believes reallocating blanket increases overall survival chances",
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
        "priority_rule": "Perfect duties override imperfect",
        "protected_standing": "SPECIAL_OBLIGATION",
        "competing_protected_standing": "BODILY_INTEGRITY",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "Broken promise cannot be publicly justified",
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
        "reason": "Breaks a promise that could not be universalized without contradiction in conception.",
        "protected_party": "sasha",
        "norm": "keep promises (fidelity)",
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
        "verdict": "CONFLICTED",
        "proposed_verdict": "CONFLICTED",
        "relation": "SATISFIES",
        "proposed_relation": "SATISFIES",
        "norm_kind": "DUTY",
        "norm_node_id": "DEON_NORM:f5bc884edebe6092",
        "party_node_id": "DEON_PARTY:297581d6cd198a6e",
        "bearer_node_id": "DEON_BEARER:3c5ddd0948295663",
        "competing_norm": "beneficence (rescue duty)",
        "competing_norm_kind": "DUTY",
        "competing_relation": "CONFLICTS",
        "competing_protected_party": "rowan",
        "competing_reason": "Does not maximize rescue, but imperfect duty yields to perfect fidelity",
        "duty_type": "PERFECT_POSITIVE",
        "harm_relation": "WITHHOLDING_BENEFIT",
        "special_obligation_status": "NOT_REQUIRED",
        "special_obligation_basis": "No special obligation to Rowan established",
        "means_relation": "NO_INSTRUMENTALIZATION",
        "candidate_maxim": {
          "circumstance": "only one blanket, promise already made",
          "policy": "give blanket to Sasha",
          "end": "keep promise",
          "means": "allocate promised blanket",
          "person_role": "NOT_USED",
          "consent_constraint": "NONE",
          "universalization": "Consistent; promise-based allocations remain reliable",
          "action_permissibility": "PERMISSIBLE",
          "maxim_assessment": "DEFENSIBLE",
          "instrumental_belief": "DOES_NOT_RELY",
          "instrumental_use": "DOES_NOT_USE",
          "practical_reasoning": "Believes honoring promise is required regardless of utility",
          "means_effect_ids": [
            "E3"
          ],
          "end_effect_ids": []
        },
        "maxim_origin": "CANDIDATE_INTERPRETATION",
        "governing_norm": "UNRESOLVED",
        "priority_basis": "UNRESOLVED",
        "priority_rule": "Perfect duties override imperfect",
        "protected_standing": "SPECIAL_OBLIGATION",
        "competing_protected_standing": "BODILY_INTEGRITY",
        "coercion_kind": "NONE",
        "coercive_actor": "NONE",
        "coerced_party": "NONE",
        "public_justification": "Keeping promises can be willed as universal law",
        "reciprocity_status": "SATISFIED",
        "necessity_status": "UNKNOWN",
        "authorization_status": "NOT_APPLICABLE",
        "derivation": "UNIVERSAL_LAW",
        "resolution_status": "CONTESTED",
        "calibration_errors": [
          "perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis",
          "merely-as-means conclusion lacks intended-as-means classification",
          "resolved adjudication rests on unsupported decisive premises"
        ],
        "calibration_issues": [
          {
            "kind": "PERFECT_POSITIVE_LACKS_BASIS",
            "message": "perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis",
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
        "reason": "Fulfills promise without instrumentalizing anyone.",
        "protected_party": "sasha",
        "norm": "keep promises (fidelity)",
        "duty_bearer": "ellis",
        "provenance": [
          "delegate:deontological",
          "cycle:1"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "If promise-keeping were not a perfect duty (e.g., promise judged void), saving the person with higher survival chance would take priority.",
  "factual_reversal_threshold": "NONE",
  "model_argument_fields": {
    "rationale": "Promise is a perfect duty",
    "framework_application": "NONE",
    "decision_rule": "Prefer the action that fulfills the agent’s perfect duty of fidelity when the rival action only advances an imperfect duty of beneficence.",
    "landscape_cases": {
      "A0": "Best chance to save life but breaks an explicit promise, violating a perfect duty of fidelity.",
      "A1": "Keeps Ellis’s explicit promise to Sasha, fulfilling a perfect duty of fidelity; only violates an imperfect rescue duty toward Rowan."
    },
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "If promise-keeping were not a perfect duty (e.g., promise judged void), saving the person with higher survival chance would take priority."
  }
}
```
