# Utilitarian — recorded argument

Participation: FULL. Native ledger: COMMITTED.

## Readable explanation

Net four lives saved. Prefer action that yields greatest net lives saved. Recorded comparison — “pulls the lever”: Pulling diverts trolley, sacrifices one, saves five—maximises total welfare.; “does not pull the lever”: Inaction avoids direct agency but results in five deaths, far greater harm.

Presentation basis: RECORDED_ARGUMENT_FIELDS.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Prefer action that yields greatest net lives saved.
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; five workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.

## Conditions for reconsidering

- The agent reports a switch from “pulls the lever” to “does not pull the lever”. Condition: If killing is absolutely prohibited regardless of numbers. This is an agent-reported condition, not an independently established threshold.
- The agent reports a switch from “pulls the lever” to “does not pull the lever”. Condition: If side track held five or more workers instead of one. This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Net four lives saved",
  "framework_application": "NONE",
  "decision_rule": "Prefer action that yields greatest net lives saved.",
  "landscape_cases": {
    "pulls the lever": "Pulling diverts trolley, sacrifices one, saves five—maximises total welfare.",
    "does not pull the lever": "Inaction avoids direct agency but results in five deaths, far greater harm."
  },
  "proposed_framework_state": {
    "constraint": "IMMINENT_HARM",
    "action_scores": {
      "pulls the lever": 0.9,
      "does not pull the lever": 0.1
    },
    "recommended_action": "pulls the lever",
    "rationale": "Net four lives saved",
    "decision_rule": "Prefer action that yields greatest net lives saved.",
    "framework_action_map": {},
    "framework_numerical_role": "NOT_APPLICABLE",
    "framework_numerical_justification": "",
    "action_admissibility": {
      "pulls the lever": "REQUIRED",
      "does not pull the lever": "REJECTED"
    },
    "comparison_complete": false,
    "evidence_sufficient_for_action": true,
    "interim_action": "pulls the lever",
    "factual_reversal_threshold": "If side track held five or more workers instead of one.",
    "normative_reversal_threshold": "If killing is absolutely prohibited regardless of numbers.",
    "utilitarian_consequence_table": {
      "pulls the lever": [
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
          "valuation_reason": "neutral intervention"
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
          "valuation_reason": "neutral physical switch"
        },
        {
          "effect_id": "E3",
          "outcome": "the trolley will kill one worker",
          "scope": "one worker",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "one",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "HIGH",
          "valuation_reason": "one certain death"
        },
        {
          "effect_id": "E4",
          "outcome": "spare five workers",
          "scope": "five workers",
          "direction": "BENEFIT",
          "polarity": "BENEFICIAL",
          "probability": "CERTAIN",
          "magnitude": "five",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "five certain lives spared"
        }
      ],
      "does not pull the lever": [
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
          "valuation_reason": "neutral inaction"
        },
        {
          "effect_id": "E6",
          "outcome": "it will kill the five",
          "scope": "the trolley",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "five",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "LOW",
          "valuation_reason": "neutral process event"
        },
        {
          "effect_id": "E7",
          "outcome": "it will kill five workers",
          "scope": "five workers",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "five",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "five certain deaths"
        },
        {
          "effect_id": "E8",
          "outcome": "spare one worker",
          "scope": "one worker",
          "direction": "BENEFIT",
          "polarity": "BENEFICIAL",
          "probability": "CERTAIN",
          "magnitude": "one",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "HIGH",
          "valuation_reason": "one certain life spared"
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
              "reason": "neutral intervention"
            },
            {
              "effect_id": "E2",
              "importance": "LOW",
              "reason": "neutral physical switch"
            },
            {
              "effect_id": "E3",
              "importance": "HIGH",
              "reason": "one certain death"
            },
            {
              "effect_id": "E4",
              "importance": "CRITICAL",
              "reason": "five certain lives spared"
            }
          ]
        },
        {
          "action_id": "A1",
          "valuations": [
            {
              "effect_id": "E5",
              "importance": "LOW",
              "reason": "neutral inaction"
            },
            {
              "effect_id": "E6",
              "importance": "LOW",
              "reason": "neutral process event"
            },
            {
              "effect_id": "E7",
              "importance": "CRITICAL",
              "reason": "five certain deaths"
            },
            {
              "effect_id": "E8",
              "importance": "HIGH",
              "reason": "one certain life spared"
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
        "outcome": "the trolley will kill one worker",
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
        "importance": "HIGH",
        "valuation_reason": "one certain death",
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
        "valuation_reason": "neutral intervention",
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
        "valuation_reason": "neutral physical switch",
        "scope_node_id": "UTIL_SCOPE:182e607160b94be4",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E2"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:de793ba19330",
        "outcome": "spare five workers",
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
        "valuation_reason": "five certain lives spared",
        "scope_node_id": "UTIL_SCOPE:e099130fb57c35fc",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E4"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:a29d84f0185e",
        "outcome": "it will kill five workers",
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
        "valuation_reason": "five certain deaths",
        "scope_node_id": "UTIL_SCOPE:e099130fb57c35fc",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E7"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:a8d0a1117e28",
        "outcome": "spare one worker",
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
        "importance": "HIGH",
        "valuation_reason": "one certain life spared",
        "scope_node_id": "UTIL_SCOPE:879bcf30890178b8",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E8"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:b073b98568c2",
        "outcome": "it will kill the five",
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
        "magnitude": "five",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "LOW",
        "valuation_reason": "neutral process event",
        "scope_node_id": "UTIL_SCOPE:182e607160b94be4",
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
        "valuation_reason": "neutral inaction",
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
  "normative_reversal_threshold": "If killing is absolutely prohibited regardless of numbers.",
  "factual_reversal_threshold": "If side track held five or more workers instead of one.",
  "model_argument_fields": {
    "rationale": "Net four lives saved",
    "framework_application": "NONE",
    "decision_rule": "Prefer action that yields greatest net lives saved.",
    "landscape_cases": {
      "A0": "Pulling diverts trolley, sacrifices one, saves five—maximises total welfare.",
      "A1": "Inaction avoids direct agency but results in five deaths, far greater harm."
    },
    "factual_reversal_threshold": "If side track held five or more workers instead of one.",
    "normative_reversal_threshold": "If killing is absolutely prohibited regardless of numbers."
  }
}
```
