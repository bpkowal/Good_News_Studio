# Utilitarian — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED.

## Readable explanation

Framework did not select a live action. UNRESOLVED FRAMEWORK CONFLICT: live competing derivations. No comparative leaning is yet grounded. Interim working action recorded without becoming a ranking: gives the blanket to Rowan. higher expected survival. Unresolved contestation: prefer the action producing the larger expected number of survivors. Recorded comparison — “gives the blanket to Rowan”: Giving Rowan the blanket yields the highest expected number of lives (0.8 vs 0.6) and therefore maximises aggregate welfare.; “gives the blanket to Sasha”: Giving Sasha the blanket honours Ellis’s promise, but it saves fewer expected lives (0.6) and thus sacrifices overall welfare. Unadmitted magnitude cannot decide the ranking; the claim remains a reversal boundary.

Presentation basis: RECORDED_ARGUMENT_FIELDS.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: prefer the action producing the larger expected number of survivors
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- A0: numeric EV cannot be recomputed from its cited effects
- A1: numeric EV cannot be recomputed from its cited effects

## Conditions for reconsidering


## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Unresolved contestation: higher expected survival",
  "framework_application": "NONE",
  "decision_rule": "Unresolved contestation: prefer the action producing the larger expected number of survivors",
  "landscape_cases": {
    "gives the blanket to Rowan": "Giving Rowan the blanket yields the highest expected number of lives (0.8 vs 0.6) and therefore maximises aggregate welfare.",
    "gives the blanket to Sasha": "Giving Sasha the blanket honours Ellis’s promise, but it saves fewer expected lives (0.6) and thus sacrifices overall welfare."
  },
  "proposed_framework_state": {
    "constraint": "FEASIBILITY",
    "action_scores": {
      "gives the blanket to Rowan": 0.5,
      "gives the blanket to Sasha": 0.5
    },
    "recommended_action": "",
    "rationale": "higher expected survival",
    "decision_rule": "prefer the action producing the larger expected number of survivors",
    "framework_action_map": {},
    "framework_numerical_role": "NOT_APPLICABLE",
    "framework_numerical_justification": "",
    "action_admissibility": {
      "gives the blanket to Rowan": "PERMISSIBLE",
      "gives the blanket to Sasha": "REJECTED"
    },
    "comparison_complete": false,
    "evidence_sufficient_for_action": false,
    "interim_action": "gives the blanket to Rowan",
    "factual_reversal_threshold": "Rowan’s survival probability drops below Sasha’s (e.g. <60 %).",
    "normative_reversal_threshold": "promise keeping given decisive priority",
    "utilitarian_consequence_table": {
      "gives the blanket to Rowan": [
        {
          "effect_id": "E2",
          "outcome": "Rowan has an 80% chance of survival",
          "scope": "Rowan",
          "direction": "UNKNOWN",
          "polarity": "BENEFICIAL",
          "probability": "0.8",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "probabilistic life-saving benefit to Rowan"
        },
        {
          "effect_id": "E5",
          "outcome": "does not receive one thermal blanket",
          "scope": "Sasha",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "one",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "certain loss of blanket and life for Sasha"
        },
        {
          "effect_id": "E7",
          "outcome": "does not get the blanket",
          "scope": "Sasha",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "certain death for Sasha"
        },
        {
          "effect_id": "E1",
          "outcome": "gives the blanket to Rowan",
          "scope": "Rowan",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "LOW",
          "valuation_reason": "transfer itself neutral"
        }
      ],
      "gives the blanket to Sasha": [
        {
          "effect_id": "E4",
          "outcome": "Sasha has a 60% chance of survival",
          "scope": "Sasha",
          "direction": "UNKNOWN",
          "polarity": "BENEFICIAL",
          "probability": "0.6",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "probabilistic life-saving benefit to Sasha"
        },
        {
          "effect_id": "E6",
          "outcome": "does not receive one thermal blanket",
          "scope": "Rowan",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "one",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "certain loss of blanket and life for Rowan"
        },
        {
          "effect_id": "E8",
          "outcome": "does not get the blanket",
          "scope": "Rowan",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "certain death for Rowan"
        },
        {
          "effect_id": "E3",
          "outcome": "gives the blanket to Sasha",
          "scope": "Sasha",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "LOW",
          "valuation_reason": "transfer itself neutral"
        }
      ]
    },
    "utilitarian_ledger_proposal": {
      "actions": [
        {
          "action_id": "A0",
          "valuations": [
            {
              "effect_id": "E2",
              "importance": "CRITICAL",
              "reason": "probabilistic life-saving benefit to Rowan"
            },
            {
              "effect_id": "E5",
              "importance": "CRITICAL",
              "reason": "certain loss of blanket and life for Sasha"
            },
            {
              "effect_id": "E7",
              "importance": "CRITICAL",
              "reason": "certain death for Sasha"
            },
            {
              "effect_id": "E1",
              "importance": "LOW",
              "reason": "transfer itself neutral"
            }
          ]
        },
        {
          "action_id": "A1",
          "valuations": [
            {
              "effect_id": "E4",
              "importance": "CRITICAL",
              "reason": "probabilistic life-saving benefit to Sasha"
            },
            {
              "effect_id": "E6",
              "importance": "CRITICAL",
              "reason": "certain loss of blanket and life for Rowan"
            },
            {
              "effect_id": "E8",
              "importance": "CRITICAL",
              "reason": "certain death for Rowan"
            },
            {
              "effect_id": "E3",
              "importance": "LOW",
              "reason": "transfer itself neutral"
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
        "outcome": "gives the blanket to Rowan",
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
        "valuation_reason": "transfer itself neutral",
        "scope_node_id": "UTIL_SCOPE:4ce5019f09edebc7",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E1"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:713a5bb2d170",
        "outcome": "Rowan has an 80% chance of survival",
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
        "probability": "0.8",
        "modality": "PROBABILISTIC",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "CRITICAL",
        "valuation_reason": "probabilistic life-saving benefit to Rowan",
        "scope_node_id": "UTIL_SCOPE:4ce5019f09edebc7",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E2"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:a29d84f0185e",
        "outcome": "does not get the blanket",
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
        "importance": "CRITICAL",
        "valuation_reason": "certain death for Sasha",
        "scope_node_id": "UTIL_SCOPE:297581d6cd198a6e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E7"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:ef099fcece6e",
        "outcome": "does not receive one thermal blanket",
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
        "importance": "CRITICAL",
        "valuation_reason": "certain loss of blanket and life for Sasha",
        "scope_node_id": "UTIL_SCOPE:297581d6cd198a6e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E5"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:183499aaeec2",
        "outcome": "gives the blanket to Sasha",
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
        "valuation_reason": "transfer itself neutral",
        "scope_node_id": "UTIL_SCOPE:297581d6cd198a6e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E3"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:a8d0a1117e28",
        "outcome": "does not get the blanket",
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
        "importance": "CRITICAL",
        "valuation_reason": "certain death for Rowan",
        "scope_node_id": "UTIL_SCOPE:4ce5019f09edebc7",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E8"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:b073b98568c2",
        "outcome": "does not receive one thermal blanket",
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
        "importance": "CRITICAL",
        "valuation_reason": "certain loss of blanket and life for Rowan",
        "scope_node_id": "UTIL_SCOPE:4ce5019f09edebc7",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E6"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:de793ba19330",
        "outcome": "Sasha has a 60% chance of survival",
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
        "probability": "0.6",
        "modality": "PROBABILISTIC",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "CRITICAL",
        "valuation_reason": "probabilistic life-saving benefit to Sasha",
        "scope_node_id": "UTIL_SCOPE:297581d6cd198a6e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E4"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "promise keeping given decisive priority",
  "factual_reversal_threshold": "Rowan’s survival probability drops below Sasha’s (e.g. <60 %).",
  "model_argument_fields": {
    "rationale": "higher expected survival",
    "framework_application": "NONE",
    "decision_rule": "prefer the action producing the larger expected number of survivors",
    "landscape_cases": {
      "A0": "Giving Rowan the blanket yields the highest expected number of lives (0.8 vs 0.6) and therefore maximises aggregate welfare.",
      "A1": "Giving Sasha the blanket honours Ellis’s promise, but it saves fewer expected lives (0.6) and thus sacrifices overall welfare."
    },
    "factual_reversal_threshold": "Rowan’s survival probability drops below Sasha’s (e.g. <60 %).",
    "normative_reversal_threshold": "promise keeping given decisive priority"
  }
}
```
