# Deontological — recorded argument

Participation: ATTENUATED. Native ledger: COMMITTED_WITH_UNCERTAINTY.

## Readable explanation

The agent assesses Mira’s duty “do not harm innocents” toward nearby staff. Mira is the duty-bearer; nearby staff is the protected party. The competing obligation being assessed is Mira’s “imperfect duty of rescue” toward technician. For “open the door now”, the agent proposes the maxim “open containment door immediately”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is reach technician quickly; its chosen means are opening door during run. The proposed justification is: would allow endangering others whenever quick rescue desired. The agent assesses the maxim as indefensible, and the action as impermissible. Its proposed means–ends reasoning is: Staff exposure is accepted to reach technician sooner. Physical means references: E1; end references: none. For Mira’s option “open the door now”, the ledger records conflicted. The agent’s stated reason is: directly endangering staff uses them as mere means to rescue another.. The agent assesses Mira’s duty “avoid harming innocents” toward nearby staff. Mira is the duty-bearer; nearby staff is the protected party. The competing obligation being assessed is Mira’s “imperfect duty of rescue” toward technician. For “stop the experiment and wait”, the agent proposes the maxim “stop experiment and wait”. This is an interpretation of possible practical reasoning, not an established intention. Its purpose is reach technician safely; its chosen means are halt hazardous process. The proposed justification is: consistent with universal safety adherence. The agent assesses the maxim as defensible, and the action as permissible. Its proposed means–ends reasoning is: Protect others while preparing safe rescue. Physical means references: E3; end references: none. For Mira’s option “stop the experiment and wait”, the ledger records required. The agent’s stated reason is: stops experiment first, respecting others' safety.. The ledger retains unresolved issues; this does not establish that one duty overrides the other.

Presentation basis: READABLE_PROJECTION_OF_NATIVE_DUTY_LEDGER.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Prefer “stop the experiment and wait” when “open the door now” would directly risk harming uninvolved persons.
- Recorded interpretive commitment: probability of harm informs duty scope but does not override perfect duty
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- hypothetical reversal boundary parameter is unverified; graph update not committed
- A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use
- A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- A1 primary protected party lacks current-run action grounding
- whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- whether the burden is intended as a means or only foreseen as a side effect
- whether an in-action causal path makes the burden instrumental to the chosen end
- mira's duty 'avoid harming innocents' protecting nearby staff ↔ mira's competing duty 'imperfect duty of rescue' protecting technician
- A0 prohibited by do not harm innocents ↔ whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict
- A0 prohibited by do not harm innocents ↔ whether the burden is intended as a means or only foreseen as a side effect

## Conditions for reconsidering

- The agent reports a switch from “stop the experiment and wait” to “open the door now”. Condition: If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses. This is an agent-reported condition, not an independently established threshold.
- The agent reports a switch from “stop the experiment and wait” to “open the door now”. Condition: If opening door poses no fume risk, ranking reverses. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Deontology provisionally prioritizes mira's duty 'avoid harming innocents' protecting nearby staff over mira's competing duty 'imperfect duty of rescue' protecting technician through respect persons. That priority depends on an unestablished prohibition of A0: whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only foreseen as a side effect; and whether an in-action causal path makes the burden instrumental to the chosen end; and whether the candidate maxim's means, person-role, and action assessment justify the claimed priority. The comparative Kantian judgment remains contested.",
  "framework_application": "NONE",
  "decision_rule": "Provisionally: Treat the duty priority as unestablished until whether the harm-relation follows the admitted graph's causal topology rather than a relabel chosen to fit the verdict; and whether the burden is intended as a means or only fore",
  "landscape_cases": {
    "open the door now": "PROHIBITED: breaks safety rule and risks harming innocent staff, violating perfect negative duty.",
    "stop the experiment and wait": "REQUIRED: respects safety rule and avoids doing harm, only allows possible harm already unfolding."
  },
  "proposed_framework_state": {
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
  "framework_vote_reason": "framework ledger committed with unresolved validation residue",
  "investigative_claim": "UNRESOLVED DUTY CONFLICT: mira's duty 'avoid harming innocents' protecting nearby staff ↔ mira's competing duty 'imperfect duty of rescue' protecting technician. Current reasoning leans stop the experiment and wait, but the competing strict claim has not been vindicated or defeated.",
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
  "unsupported_assumption": "",
  "normative_reversal_threshold": "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses.",
  "factual_reversal_threshold": "If opening door poses no fume risk, ranking reverses.",
  "model_argument_fields": {
    "rationale": "Avoid direct harm to bystanders",
    "framework_application": "NONE",
    "decision_rule": "Prefer A1 when A0 would directly risk harming uninvolved persons.",
    "landscape_cases": {
      "A0": "PROHIBITED: breaks safety rule and risks harming innocent staff, violating perfect negative duty.",
      "A1": "REQUIRED: respects safety rule and avoids doing harm, only allows possible harm already unfolding."
    },
    "factual_reversal_threshold": "If opening door poses no fume risk, ranking reverses.",
    "normative_reversal_threshold": "If imperfect rescue duty outweighs perfect non-harm duty, ranking reverses."
  }
}
```
