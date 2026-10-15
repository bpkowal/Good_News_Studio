# Utilitarian — recorded argument

Participation: FULL. Native ledger: COMMITTED.

## Readable explanation

Minimises total deaths. Prefer the action that results in fewer fatalities. Recorded comparison — “pulls the lever”: Pulling sacrifices one but saves five, minimising deaths.; “does not”: Inaction avoids direct agency but lets five die, greater loss.

Presentation basis: RECORDED_ARGUMENT_FIELDS.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Prefer the action that results in fewer fatalities.
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; five workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.

## Conditions for reconsidering

- The agent reports a switch from “pulls the lever” to “does not”. Condition: If the side track had five or more workers, the ranking would reverse. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Minimises total deaths",
  "framework_application": "NONE",
  "decision_rule": "Prefer the action that results in fewer fatalities.",
  "landscape_cases": {
    "pulls the lever": "Pulling sacrifices one but saves five, minimising deaths.",
    "does not": "Inaction avoids direct agency but lets five die, greater loss."
  },
  "proposed_framework_state": {
    "constraint": "IMMINENT_HARM",
    "action_scores": {
      "pulls the lever": 0.95,
      "does not": 0.05
    },
    "recommended_action": "pulls the lever",
    "rationale": "Minimises total deaths",
    "decision_rule": "Prefer the action that results in fewer fatalities.",
    "framework_action_map": {},
    "framework_numerical_role": "NOT_APPLICABLE",
    "framework_numerical_justification": "",
    "action_admissibility": {
      "pulls the lever": "PERMISSIBLE",
      "does not": "REJECTED"
    },
    "comparison_complete": false,
    "evidence_sufficient_for_action": true,
    "interim_action": "pulls the lever",
    "factual_reversal_threshold": "If the side track had five or more workers, the ranking would reverse.",
    "normative_reversal_threshold": "NONE",
    "utilitarian_consequence_table": {
      "pulls the lever": [
        {
          "effect_id": "E3",
          "outcome": "the trolley will kill that worker",
          "scope": "one worker",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "one",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "certain loss of one life"
        },
        {
          "effect_id": "E4",
          "outcome": "spare the five",
          "scope": "five workers",
          "direction": "BENEFIT",
          "polarity": "BENEFICIAL",
          "probability": "CERTAIN",
          "magnitude": "five",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "certain saving of five lives"
        },
        {
          "effect_id": "E1",
          "outcome": "pulls the lever",
          "scope": "a lever",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "LOW",
          "valuation_reason": "neutral physical act"
        },
        {
          "effect_id": "E2",
          "outcome": "switches the trolley onto a side track",
          "scope": "the trolley",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "LOW",
          "valuation_reason": "neutral track switch"
        }
      ],
      "does not": [
        {
          "effect_id": "E7",
          "outcome": "it will kill the five",
          "scope": "five workers",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "five",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "certain loss of five lives"
        },
        {
          "effect_id": "E8",
          "outcome": "spare the one",
          "scope": "one worker",
          "direction": "BENEFIT",
          "polarity": "BENEFICIAL",
          "probability": "CERTAIN",
          "magnitude": "one",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "MEDIUM",
          "valuation_reason": "saving one life"
        },
        {
          "effect_id": "E5",
          "outcome": "does not",
          "scope": "a lever",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "LOW",
          "valuation_reason": "inaction itself neutral"
        },
        {
          "effect_id": "E6",
          "outcome": "does not",
          "scope": "a lever",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "LOW",
          "valuation_reason": "no switch occurs"
        }
      ]
    },
    "utilitarian_ledger_proposal": {
      "actions": [
        {
          "action_id": "A0",
          "valuations": [
            {
              "effect_id": "E3",
              "importance": "CRITICAL",
              "reason": "certain loss of one life"
            },
            {
              "effect_id": "E4",
              "importance": "CRITICAL",
              "reason": "certain saving of five lives"
            },
            {
              "effect_id": "E1",
              "importance": "LOW",
              "reason": "neutral physical act"
            },
            {
              "effect_id": "E2",
              "importance": "LOW",
              "reason": "neutral track switch"
            }
          ]
        },
        {
          "action_id": "A1",
          "valuations": [
            {
              "effect_id": "E7",
              "importance": "CRITICAL",
              "reason": "certain loss of five lives"
            },
            {
              "effect_id": "E8",
              "importance": "MEDIUM",
              "reason": "saving one life"
            },
            {
              "effect_id": "E5",
              "importance": "LOW",
              "reason": "inaction itself neutral"
            },
            {
              "effect_id": "E6",
              "importance": "LOW",
              "reason": "no switch occurs"
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
  "framework_vote_reason": "framework compared every live action but reports residual uncertainty",
  "investigative_claim": "",
  "committed_native_ledger": {
    "ledger_kind": "UTILITARIAN_CONSEQUENCE_LEDGER",
    "transaction_status": "COMMITTED",
    "records": [
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:183499aaeec2",
        "outcome": "the trolley will kill that worker",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "grounded_effect_id": "E3",
        "world_effect_id": "E3",
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
        "valuation_reason": "certain loss of one life",
        "scope_node_id": "UTIL_SCOPE:879bcf30890178b8",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E3"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:22249aa61fd2",
        "outcome": "pulls the lever",
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
        "valuation_reason": "neutral physical act",
        "scope_node_id": "UTIL_SCOPE:f7203884bca9e74c",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E1"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:713a5bb2d170",
        "outcome": "switches the trolley onto a side track",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "grounded_effect_id": "E2",
        "world_effect_id": "E2",
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
        "valuation_reason": "neutral track switch",
        "scope_node_id": "UTIL_SCOPE:182e607160b94be4",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E2"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:de793ba19330",
        "outcome": "spare the five",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A0",
        "grounded_effect_id": "E4",
        "world_effect_id": "E4",
        "direction": "BENEFIT",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "BENEFICIAL",
        "grounded_world_polarity": "BENEFICIAL",
        "probability": "CERTAIN",
        "modality": "CERTAIN",
        "magnitude": "five",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "CRITICAL",
        "valuation_reason": "certain saving of five lives",
        "scope_node_id": "UTIL_SCOPE:e099130fb57c35fc",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E4"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:a29d84f0185e",
        "outcome": "it will kill the five",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "grounded_effect_id": "E7",
        "world_effect_id": "E7",
        "direction": "HARM",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "ADVERSE",
        "grounded_world_polarity": "ADVERSE",
        "probability": "CERTAIN",
        "modality": "CERTAIN",
        "magnitude": "five",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "CRITICAL",
        "valuation_reason": "certain loss of five lives",
        "scope_node_id": "UTIL_SCOPE:e099130fb57c35fc",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E7"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:a8d0a1117e28",
        "outcome": "spare the one",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "grounded_effect_id": "E8",
        "world_effect_id": "E8",
        "direction": "BENEFIT",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "BENEFICIAL",
        "grounded_world_polarity": "BENEFICIAL",
        "probability": "CERTAIN",
        "modality": "CERTAIN",
        "magnitude": "one",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "MEDIUM",
        "valuation_reason": "saving one life",
        "scope_node_id": "UTIL_SCOPE:879bcf30890178b8",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E8"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:b073b98568c2",
        "outcome": "does not",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "grounded_effect_id": "E6",
        "world_effect_id": "E6",
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
        "valuation_reason": "no switch occurs",
        "scope_node_id": "UTIL_SCOPE:f7203884bca9e74c",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E6"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:ef099fcece6e",
        "outcome": "does not",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 1,
        "canonical_action_id": "A1",
        "grounded_effect_id": "E5",
        "world_effect_id": "E5",
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
        "valuation_reason": "inaction itself neutral",
        "scope_node_id": "UTIL_SCOPE:f7203884bca9e74c",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E5"
        ]
      }
    ]
  },
  "unsupported_assumption": "",
  "normative_reversal_threshold": "NONE",
  "factual_reversal_threshold": "If the side track had five or more workers, the ranking would reverse.",
  "model_argument_fields": {
    "rationale": "Minimises total deaths",
    "framework_application": "NONE",
    "decision_rule": "Prefer the action that results in fewer fatalities.",
    "landscape_cases": {
      "A0": "Pulling sacrifices one but saves five, minimising deaths.",
      "A1": "Inaction avoids direct agency but lets five die, greater loss."
    },
    "factual_reversal_threshold": "If the side track had five or more workers, the ranking would reverse.",
    "normative_reversal_threshold": "NONE"
  }
}
```
