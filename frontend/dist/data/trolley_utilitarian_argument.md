# Utilitarian — recorded argument

Participation: ATTENUATED. Native ledger: COMMITTED.

## Readable explanation

Pulling minimizes deaths. Prefer “pulls the lever” when it results in fewer deaths than “does not pull the lever”. Recorded comparison — “pulls the lever”: Pulling diverts the tram, killing one but saving ten; net harm is lowest so total welfare highest.; “does not pull the lever”: Not pulling lets the tram continue, killing ten and sparing one; avoids direct intervention but causes far greater harm.

Presentation basis: RECORDED_ARGUMENT_FIELDS.

## Assumptions and interpretive commitments

- Recorded interpretive commitment: Prefer “pulls the lever” when it results in fewer deaths than “does not pull the lever”.
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The counts (one worker; ten workers) are stipulated by the scenario for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- case for A1 tracks the opposing action but not its own action
- numeric reversal threshold lacks a scenario-grounded quantity

## Conditions for reconsidering

- The agent reports a switch from “pulls the lever” to “does not pull the lever”. Condition: Absolute duty never to kill even to save more lives. This is an agent-reported condition, not an independently established threshold.
- The agent reports a switch from “pulls the lever” to “does not pull the lever”. Condition: Main-track workers drop below side-track workers (e.g., 0 vs 1). This is an agent-reported condition, not an independently established threshold.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "Pulling minimizes deaths",
  "framework_application": "NONE",
  "decision_rule": "Prefer A0 when it results in fewer deaths than A1.",
  "landscape_cases": {
    "pulls the lever": "Pulling diverts the tram, killing one but saving ten; net harm is lowest so total welfare highest.",
    "does not pull the lever": "Not pulling lets the tram continue, killing ten and sparing one; avoids direct intervention but causes far greater harm."
  },
  "proposed_framework_state": {
    "constraint": "IMMINENT_HARM",
    "action_scores": {
      "pulls the lever": 0.9,
      "does not pull the lever": 0.1
    },
    "recommended_action": "pulls the lever",
    "rationale": "Pulling minimizes deaths",
    "decision_rule": "Prefer A0 when it results in fewer deaths than A1.",
    "framework_action_map": {},
    "framework_numerical_role": "NOT_APPLICABLE",
    "framework_numerical_justification": "",
    "action_admissibility": {
      "pulls the lever": "PERMISSIBLE",
      "does not pull the lever": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": true,
    "interim_action": "pulls the lever",
    "factual_reversal_threshold": "NONE",
    "normative_reversal_threshold": "Absolute duty never to kill even to save more lives.",
    "utilitarian_consequence_table": {
      "pulls the lever": [
        {
          "effect_id": "E3",
          "outcome": "the tram will kill one worker",
          "scope": "one worker",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "one",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "one certain death: high negative utility"
        },
        {
          "effect_id": "E4",
          "outcome": "spare ten workers",
          "scope": "ten workers",
          "direction": "BENEFIT",
          "polarity": "BENEFICIAL",
          "probability": "CERTAIN",
          "magnitude": "ten",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "saving ten lives: very large positive utility"
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
          "importance": "NEGLIGIBLE",
          "valuation_reason": "mere lever pull has no welfare impact"
        },
        {
          "effect_id": "E2",
          "outcome": "redirects the tram onto a side track",
          "scope": "the tram",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "NEGLIGIBLE",
          "valuation_reason": "track change itself has no intrinsic welfare"
        }
      ],
      "does not pull the lever": [
        {
          "effect_id": "E7",
          "outcome": "it will kill ten workers",
          "scope": "ten workers",
          "direction": "HARM",
          "polarity": "ADVERSE",
          "probability": "CERTAIN",
          "magnitude": "ten",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "ten certain deaths: maximal negative utility"
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
          "valuation_reason": "one life spared: moderate positive utility"
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
          "importance": "NEGLIGIBLE",
          "valuation_reason": "inaction itself no intrinsic welfare"
        },
        {
          "effect_id": "E6",
          "outcome": "heading toward ten workers on the main track",
          "scope": "the tram",
          "direction": "UNKNOWN",
          "polarity": "NEUTRAL",
          "probability": "CERTAIN",
          "magnitude": "ten",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "NEGLIGIBLE",
          "valuation_reason": "tram motion direction no intrinsic welfare"
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
              "reason": "one certain death: high negative utility"
            },
            {
              "effect_id": "E4",
              "importance": "CRITICAL",
              "reason": "saving ten lives: very large positive utility"
            },
            {
              "effect_id": "E1",
              "importance": "NEGLIGIBLE",
              "reason": "mere lever pull has no welfare impact"
            },
            {
              "effect_id": "E2",
              "importance": "NEGLIGIBLE",
              "reason": "track change itself has no intrinsic welfare"
            }
          ]
        },
        {
          "action_id": "A1",
          "valuations": [
            {
              "effect_id": "E7",
              "importance": "CRITICAL",
              "reason": "ten certain deaths: maximal negative utility"
            },
            {
              "effect_id": "E8",
              "importance": "HIGH",
              "reason": "one life spared: moderate positive utility"
            },
            {
              "effect_id": "E5",
              "importance": "NEGLIGIBLE",
              "reason": "inaction itself no intrinsic welfare"
            },
            {
              "effect_id": "E6",
              "importance": "NEGLIGIBLE",
              "reason": "tram motion direction no intrinsic welfare"
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
  "framework_vote_reason": "framework ranking is supported by its committed typed ledger",
  "investigative_claim": "",
  "committed_native_ledger": {
    "ledger_kind": "UTILITARIAN_CONSEQUENCE_LEDGER",
    "transaction_status": "COMMITTED",
    "records": [
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:183499aaeec2",
        "outcome": "the tram will kill one worker",
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
        "valuation_reason": "one certain death: high negative utility",
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
        "importance": "NEGLIGIBLE",
        "valuation_reason": "mere lever pull has no welfare impact",
        "scope_node_id": "UTIL_SCOPE:f7203884bca9e74c",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E1"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:713a5bb2d170",
        "outcome": "redirects the tram onto a side track",
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
        "importance": "NEGLIGIBLE",
        "valuation_reason": "track change itself has no intrinsic welfare",
        "scope_node_id": "UTIL_SCOPE:6a5cf86dd55beb32",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E2"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:de793ba19330",
        "outcome": "spare ten workers",
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
        "magnitude": "ten",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "CRITICAL",
        "valuation_reason": "saving ten lives: very large positive utility",
        "scope_node_id": "UTIL_SCOPE:dd427dcfeb914347",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E4"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:a29d84f0185e",
        "outcome": "it will kill ten workers",
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
        "magnitude": "ten",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "CRITICAL",
        "valuation_reason": "ten certain deaths: maximal negative utility",
        "scope_node_id": "UTIL_SCOPE:dd427dcfeb914347",
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
        "valuation_reason": "one life spared: moderate positive utility",
        "scope_node_id": "UTIL_SCOPE:879bcf30890178b8",
        "provenance": [
          "delegate:utilitarian",
          "cycle:1",
          "grounded_effect:E8"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:b073b98568c2",
        "outcome": "heading toward ten workers on the main track",
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
        "magnitude": "ten",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "NEGLIGIBLE",
        "valuation_reason": "tram motion direction no intrinsic welfare",
        "scope_node_id": "UTIL_SCOPE:6a5cf86dd55beb32",
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
        "importance": "NEGLIGIBLE",
        "valuation_reason": "inaction itself no intrinsic welfare",
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
  "normative_reversal_threshold": "Absolute duty never to kill even to save more lives.",
  "factual_reversal_threshold": "NONE",
  "model_argument_fields": {
    "rationale": "Pulling minimizes deaths",
    "framework_application": "NONE",
    "decision_rule": "Prefer A0 when it results in fewer deaths than A1.",
    "landscape_cases": {
      "A0": "Pulling diverts the tram, killing one but saving ten; net harm is lowest so total welfare highest.",
      "A1": "Not pulling lets the tram continue, killing ten and sparing one; avoids direct intervention but causes far greater harm."
    },
    "factual_reversal_threshold": "Main-track workers drop below side-track workers (e.g., 0 vs 1).",
    "normative_reversal_threshold": "Absolute duty never to kill even to save more lives."
  }
}
```
