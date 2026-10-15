# Utilitarian — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED.

## Readable explanation

Framework did not select a live action. UNRESOLVED FRAMEWORK CONFLICT: live competing derivations. No comparative leaning is yet grounded. Interim working action recorded without becoming a ranking: Ada gives the medicine to Ben. Max expected survival. Unresolved contestation: prefer “Ada gives the medicine to Ben” when it yields greater expected survivors than “Ada gives the medicine to Cara”. Recorded comparison — “Ada gives the medicine to Ben”: Gives Ben a 95% survival chance; maximises expected lives saved.; “Ada gives the medicine to Cara”: Keeps Ada’s promise to Cara; still offers small 5% chance of saving one life. Unadmitted magnitude cannot decide the ranking; the claim remains a reversal boundary.

Presentation basis: RECORDED_ARGUMENT_FIELDS.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: prefer “Ada gives the medicine to Ben” when it yields greater expected survivors than “Ada gives the medicine to Cara”
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.

## Conditions for reconsidering


## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Unresolved contestation: Max expected survival",
  "framework_application": "NONE",
  "decision_rule": "Unresolved contestation: prefer A0 when it yields greater expected survivors than A1",
  "landscape_cases": {
    "Ada gives the medicine to Ben": "Gives Ben a 95% survival chance; maximises expected lives saved.",
    "Ada gives the medicine to Cara": "Keeps Ada’s promise to Cara; still offers small 5% chance of saving one life."
  },
  "proposed_framework_state": {
    "constraint": "IMMINENT_HARM",
    "action_scores": {
      "Ada gives the medicine to Ben": 0.5,
      "Ada gives the medicine to Cara": 0.5
    },
    "recommended_action": "",
    "rationale": "Max expected survival",
    "decision_rule": "prefer A0 when it yields greater expected survivors than A1",
    "framework_action_map": {},
    "framework_numerical_role": "NOT_APPLICABLE",
    "framework_numerical_justification": "",
    "action_admissibility": {
      "Ada gives the medicine to Ben": "PERMISSIBLE",
      "Ada gives the medicine to Cara": "REJECTED"
    },
    "comparison_complete": false,
    "evidence_sufficient_for_action": false,
    "interim_action": "Ada gives the medicine to Ben",
    "factual_reversal_threshold": "Ben survival probability falls to ≤5%",
    "normative_reversal_threshold": "Promise-keeping given overriding moral weight",
    "utilitarian_consequence_table": {
      "Ada gives the medicine to Ben": [
        {
          "effect_id": "E1",
          "outcome": "Ada gives the medicine to Ben",
          "scope": "Ben",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "LOW",
          "valuation_reason": "transfer itself neutral utilitarian value"
        },
        {
          "effect_id": "E2",
          "outcome": "Ben has a 95% chance of survival",
          "scope": "Ben",
          "direction": "UNKNOWN",
          "polarity": "BENEFICIAL",
          "probability": "0.95",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "HIGH",
          "valuation_reason": "95% chance to save Ben's life"
        },
        {
          "effect_id": "E5",
          "outcome": "does not receive one dose of medicine",
          "scope": "Cara",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "one",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "HIGH",
          "valuation_reason": "Cara lacks medicine (leads to death)"
        },
        {
          "effect_id": "E7",
          "outcome": "will die",
          "scope": "Cara",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "HIGH",
          "valuation_reason": "Cara certainly dies"
        }
      ],
      "Ada gives the medicine to Cara": [
        {
          "effect_id": "E3",
          "outcome": "Ada gives the medicine to Cara",
          "scope": "Cara",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "LOW",
          "valuation_reason": "transfer itself neutral utilitarian value"
        },
        {
          "effect_id": "E4",
          "outcome": "Cara has a 5% chance of survival",
          "scope": "Cara",
          "direction": "UNKNOWN",
          "polarity": "BENEFICIAL",
          "probability": "0.05",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "HIGH",
          "valuation_reason": "5% chance to save Cara's life"
        },
        {
          "effect_id": "E6",
          "outcome": "does not receive one dose of medicine",
          "scope": "Ben",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "one",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "HIGH",
          "valuation_reason": "Ben lacks medicine (leads to death)"
        },
        {
          "effect_id": "E8",
          "outcome": "will die",
          "scope": "Ben",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "HIGH",
          "valuation_reason": "Ben certainly dies"
        }
      ]
    },
    "utilitarian_ledger_proposal": {
      "actions": [
        {
          "action_id": "A0",
          "valuations": [
            {
              "effect_id": "E1",
              "importance": "LOW",
              "reason": "transfer itself neutral utilitarian value"
            },
            {
              "effect_id": "E2",
              "importance": "HIGH",
              "reason": "95% chance to save Ben's life"
            },
            {
              "effect_id": "E5",
              "importance": "HIGH",
              "reason": "Cara lacks medicine (leads to death)"
            },
            {
              "effect_id": "E7",
              "importance": "HIGH",
              "reason": "Cara certainly dies"
            }
          ]
        },
        {
          "action_id": "A1",
          "valuations": [
            {
              "effect_id": "E3",
              "importance": "LOW",
              "reason": "transfer itself neutral utilitarian value"
            },
            {
              "effect_id": "E4",
              "importance": "HIGH",
              "reason": "5% chance to save Cara's life"
            },
            {
              "effect_id": "E6",
              "importance": "HIGH",
              "reason": "Ben lacks medicine (leads to death)"
            },
            {
              "effect_id": "E8",
              "importance": "HIGH",
              "reason": "Ben certainly dies"
            }
          ]
        }
      ]
    },
    "rawls_position_proposal": {},
    "deontological_ledger_proposal": {},
    "virtue_character_proposal": {},
    "care_ledger_proposal": {},
    "care_relational_map": {},
    "care_numerical_role": "NOT_APPLICABLE",
    "care_numerical_justification": ""
  },
  "framework_vote_reason": "framework did not select a live action",
  "investigative_claim": "UNRESOLVED FRAMEWORK CONFLICT: live competing derivations. No comparative leaning is yet grounded.",
  "committed_native_ledger": {
    "ledger_kind": "UTILITARIAN_CONSEQUENCE_LEDGER",
    "transaction_status": "COMMITTED",
    "records": [
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:22249aa61fd2",
        "outcome": "Ada gives the medicine to Ben",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "grounded_effect_id": "E1",
        "world_effect_id": "E1",
        "direction": "NEUTRAL",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "NEUTRAL",
        "grounded_world_polarity": "NEUTRAL",
        "probability": "CERTAIN",
        "modality": "CERTAIN",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "LOW",
        "valuation_reason": "transfer itself neutral utilitarian value",
        "scope_node_id": "UTIL_SCOPE:6700869c8ff7480e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E1"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:713a5bb2d170",
        "outcome": "Ben has a 95% chance of survival",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "grounded_effect_id": "E2",
        "world_effect_id": "E2",
        "direction": "BENEFIT",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "BENEFICIAL",
        "grounded_world_polarity": "BENEFICIAL",
        "probability": "0.95",
        "modality": "PROBABILISTIC",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "HIGH",
        "valuation_reason": "95% chance to save Ben's life",
        "scope_node_id": "UTIL_SCOPE:6700869c8ff7480e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E2"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:a29d84f0185e",
        "outcome": "will die",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "grounded_effect_id": "E7",
        "world_effect_id": "E7",
        "direction": "HARM",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "ADVERSE",
        "grounded_world_polarity": "ADVERSE",
        "probability": "CERTAIN",
        "modality": "CERTAIN",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "HIGH",
        "valuation_reason": "Cara certainly dies",
        "scope_node_id": "UTIL_SCOPE:f61f5c5e22749aeb",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E7"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:ef099fcece6e",
        "outcome": "does not receive one dose of medicine",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "grounded_effect_id": "E5",
        "world_effect_id": "E5",
        "direction": "HARM",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "ADVERSE",
        "grounded_world_polarity": "ADVERSE",
        "probability": "CERTAIN",
        "modality": "CERTAIN",
        "magnitude": "one",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "HIGH",
        "valuation_reason": "Cara lacks medicine (leads to death)",
        "scope_node_id": "UTIL_SCOPE:f61f5c5e22749aeb",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E5"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:183499aaeec2",
        "outcome": "Ada gives the medicine to Cara",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "grounded_effect_id": "E3",
        "world_effect_id": "E3",
        "direction": "NEUTRAL",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "NEUTRAL",
        "grounded_world_polarity": "NEUTRAL",
        "probability": "CERTAIN",
        "modality": "CERTAIN",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "LOW",
        "valuation_reason": "transfer itself neutral utilitarian value",
        "scope_node_id": "UTIL_SCOPE:f61f5c5e22749aeb",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E3"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:a8d0a1117e28",
        "outcome": "will die",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "grounded_effect_id": "E8",
        "world_effect_id": "E8",
        "direction": "HARM",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "ADVERSE",
        "grounded_world_polarity": "ADVERSE",
        "probability": "CERTAIN",
        "modality": "CERTAIN",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "HIGH",
        "valuation_reason": "Ben certainly dies",
        "scope_node_id": "UTIL_SCOPE:6700869c8ff7480e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E8"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:b073b98568c2",
        "outcome": "does not receive one dose of medicine",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "grounded_effect_id": "E6",
        "world_effect_id": "E6",
        "direction": "HARM",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "ADVERSE",
        "grounded_world_polarity": "ADVERSE",
        "probability": "CERTAIN",
        "modality": "CERTAIN",
        "magnitude": "one",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "HIGH",
        "valuation_reason": "Ben lacks medicine (leads to death)",
        "scope_node_id": "UTIL_SCOPE:6700869c8ff7480e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E6"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:de793ba19330",
        "outcome": "Cara has a 5% chance of survival",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "grounded_effect_id": "E4",
        "world_effect_id": "E4",
        "direction": "BENEFIT",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "BENEFICIAL",
        "grounded_world_polarity": "BENEFICIAL",
        "probability": "0.05",
        "modality": "PROBABILISTIC",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "HIGH",
        "valuation_reason": "5% chance to save Cara's life",
        "scope_node_id": "UTIL_SCOPE:f61f5c5e22749aeb",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E4"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "Promise-keeping given overriding moral weight",
  "factual_reversal_threshold": "Ben survival probability falls to ≤5%",
  "model_argument_fields": {
    "rationale": "Max expected survival",
    "framework_application": "NONE",
    "decision_rule": "prefer A0 when it yields greater expected survivors than A1",
    "landscape_cases": {
      "A0": "Gives Ben a 95% survival chance; maximises expected lives saved.",
      "A1": "Keeps Ada’s promise to Cara; still offers small 5% chance of saving one life."
    },
    "factual_reversal_threshold": "Ben survival probability falls to ≤5%",
    "normative_reversal_threshold": "Promise-keeping given overriding moral weight"
  }
}
```
