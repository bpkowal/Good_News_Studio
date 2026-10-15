# Utilitarian — recorded argument

Participation: ABSTAIN. Native ledger: COMMITTED.

## Readable explanation

The utilitarian agent starts with the chance of saving a life. Giving the dose to Ben offers a 95% chance of survival; giving it to Cara offers only 5%, while the untreated patient dies. That comparison favors Ben on expected survivors. It does not, by itself, settle every consequence that matters to welfare. The saved review also continues to question whether the one-dose restriction can support the recommendation, even though the source states it explicitly. The agent therefore records Ben as its working preference but abstains from an operative ranking.

Presentation basis: REVIEWED_PLAIN_LANGUAGE_SUMMARY_OF_SAVED_ARGUMENT.

## Assumptions and interpretive commitments

- Model reported assumption: Survival probabilities decisive for utilitarian calculus.
- Recorded interpretive commitment: Prefer the action whose expected survivor count is higher.
- Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

## Source and uncertainty

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.
- The survivor comparison is clear within the stipulated scenario. The broader welfare comparison remains incomplete, and the saved review retains an unresolved challenge to the scarcity premise.

## Original presentation and objections

```json
{
  "reason": "Utilitarian evidence is insufficient to rank actions. Would verifying this unestablished consequence reverse the admitted ranking: Only one dose of the life-saving medicine is available (so only one patient can receive it)? 95% beats 5% survival. Prefer the action whose expected survivor count is higher. Recorded comparison — Ada gives the medicine to Ben: One dose to Ben saves a life with 95 % chance; only Cara dies.; Ada gives the medicine to Cara: One dose to Cara makes slim 5 % rescue attempt while Ben certainly dies. Premise rebound from PROP:WORLD:E7 to canonical PROP:HYPOTHESIS:67d6bcaf133579a0. Dependency continuity preserved unresolved decision-critical premises from the prior operative framework state because the recommendation did not change. Unsupported hypothesis may raise investigative salience; it cannot uniquely govern.",
  "reason_authority": "RECORDED_ARGUMENT_FIELDS",
  "live_objections": [
    "dependency continuity preserved unresolved proposition IDs: PROP:HYPOTHESIS:b11eb479a9d1aeb0",
    "Would verifying this unestablished consequence reverse the admitted ranking: Only one dose of the life-saving medicine is available (so only one patient can receive it)?"
  ],
  "reversal_explanations": [
    "The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: Survival probabilities decisive for utilitarian calculus. This is an agent-reported condition, not an independently established threshold.",
    "The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: Cara’s survival chance exceeds Ben’s. This is an agent-reported condition, not an independently established threshold."
  ]
}
```

## Explanation provenance

```json
{
  "basis_sha256": "cc884bab1b3c813acdaf0a879f15dbce63fddd6e3399408d03add056229271ab",
  "support_fields": [
    "recorded_argument",
    "position",
    "preference",
    "ledger_status",
    "live_objections",
    "decision_boundaries",
    "scenario",
    "world"
  ],
  "reviewer": "Codex editorial review of saved arguments; no new framework model call"
}
```

## Conditions for reconsidering

- The agent says it would favor treating Cara if her survival chance exceeded Ben’s. This is its reported condition; the broader welfare comparison would still need support.

## Original recorded fields

These fields are preserved verbatim as data. They can include incomplete or malformed model text.

```json
{
  "rationale": "95% beats 5% survival",
  "framework_application": "NONE",
  "decision_rule": "Prefer the action whose expected survivor count is higher.",
  "landscape_cases": {
    "Ada gives the medicine to Ben": "One dose to Ben saves a life with 95 % chance; only Cara dies.",
    "Ada gives the medicine to Cara": "One dose to Cara makes slim 5 % rescue attempt while Ben certainly dies."
  },
  "proposed_framework_state": {
    "constraint": "UNCERTAINTY",
    "action_scores": {
      "Ada gives the medicine to Ben": 0.9,
      "Ada gives the medicine to Cara": 0.1
    },
    "recommended_action": "Ada gives the medicine to Ben",
    "rationale": "95% beats 5% survival",
    "decision_rule": "Prefer the action whose expected survivor count is higher.",
    "framework_action_map": {},
    "framework_numerical_role": "NOT_APPLICABLE",
    "framework_numerical_justification": "",
    "action_admissibility": {
      "Ada gives the medicine to Ben": "PERMISSIBLE",
      "Ada gives the medicine to Cara": "REJECTED"
    },
    "comparison_complete": true,
    "evidence_sufficient_for_action": false,
    "interim_action": "Ada gives the medicine to Ben",
    "factual_reversal_threshold": "Cara’s survival chance exceeds Ben’s",
    "normative_reversal_threshold": "NONE",
    "utilitarian_consequence_table": {
      "Ada gives the medicine to Ben": [
        {
          "effect_id": "E2",
          "outcome": "Ben has a 95% chance of survival",
          "scope": "Ben",
          "direction": "UNKNOWN",
          "polarity": "BENEFICIAL",
          "probability": "UNKNOWN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "high survival benefit"
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
          "importance": "CRITICAL",
          "valuation_reason": "certain death for Cara"
        },
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
          "importance": "NEGLIGIBLE",
          "valuation_reason": "mere transfer act"
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
          "valuation_reason": "foregone dose for Cara"
        }
      ],
      "Ada gives the medicine to Cara": [
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
          "importance": "CRITICAL",
          "valuation_reason": "certain death for Ben"
        },
        {
          "effect_id": "E4",
          "outcome": "Cara has a 5% chance of survival",
          "scope": "Cara",
          "direction": "UNKNOWN",
          "polarity": "BENEFICIAL",
          "probability": "UNKNOWN",
          "magnitude": "UNKNOWN",
          "duration": "UNKNOWN",
          "reversibility": "UNKNOWN",
          "support": "STATED",
          "importance": "CRITICAL",
          "valuation_reason": "small survival benefit"
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
          "valuation_reason": "foregone dose for Ben"
        },
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
          "importance": "NEGLIGIBLE",
          "valuation_reason": "mere transfer act"
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
              "reason": "high survival benefit"
            },
            {
              "effect_id": "E7",
              "importance": "CRITICAL",
              "reason": "certain death for Cara"
            },
            {
              "effect_id": "E1",
              "importance": "NEGLIGIBLE",
              "reason": "mere transfer act"
            },
            {
              "effect_id": "E5",
              "importance": "HIGH",
              "reason": "foregone dose for Cara"
            }
          ]
        },
        {
          "action_id": "A1",
          "valuations": [
            {
              "effect_id": "E8",
              "importance": "CRITICAL",
              "reason": "certain death for Ben"
            },
            {
              "effect_id": "E4",
              "importance": "CRITICAL",
              "reason": "small survival benefit"
            },
            {
              "effect_id": "E6",
              "importance": "HIGH",
              "reason": "foregone dose for Ben"
            },
            {
              "effect_id": "E3",
              "importance": "NEGLIGIBLE",
              "reason": "mere transfer act"
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
  "framework_vote_reason": "utilitarian evidence is insufficient to rank actions",
  "investigative_claim": "Would verifying this unestablished consequence reverse the admitted ranking: Only one dose of the life-saving medicine is available (so only one patient can receive it)?",
  "committed_native_ledger": {
    "ledger_kind": "UTILITARIAN_CONSEQUENCE_LEDGER",
    "transaction_status": "COMMITTED",
    "records": [
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:22249aa61fd2",
        "outcome": "Ada gives the medicine to Ben",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 3,
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
        "valuation_reason": "mere transfer act",
        "scope_node_id": "UTIL_SCOPE:6700869c8ff7480e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:3",
          "grounded_effect:E1"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:713a5bb2d170",
        "outcome": "Ben has a 95% chance of survival",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 3,
        "canonical_action_id": "A0",
        "grounded_effect_id": "E2",
        "world_effect_id": "E2",
        "direction": "BENEFIT",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "BENEFICIAL",
        "grounded_world_polarity": "BENEFICIAL",
        "probability": "UNKNOWN",
        "modality": "PROBABILISTIC",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "CRITICAL",
        "valuation_reason": "high survival benefit",
        "scope_node_id": "UTIL_SCOPE:6700869c8ff7480e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:3",
          "grounded_effect:E2"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:a29d84f0185e",
        "outcome": "will die",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 3,
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
        "valuation_reason": "certain death for Cara",
        "scope_node_id": "UTIL_SCOPE:f61f5c5e22749aeb",
        "provenance": [
          "delegate:utilitarian",
          "cycle:3",
          "grounded_effect:E7"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A0:ef099fcece6e",
        "outcome": "does not receive one dose of medicine",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 3,
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
        "valuation_reason": "foregone dose for Cara",
        "scope_node_id": "UTIL_SCOPE:f61f5c5e22749aeb",
        "provenance": [
          "delegate:utilitarian",
          "cycle:3",
          "grounded_effect:E5"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:183499aaeec2",
        "outcome": "Ada gives the medicine to Cara",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 3,
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
        "importance": "NEGLIGIBLE",
        "valuation_reason": "mere transfer act",
        "scope_node_id": "UTIL_SCOPE:f61f5c5e22749aeb",
        "provenance": [
          "delegate:utilitarian",
          "cycle:3",
          "grounded_effect:E3"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:a8d0a1117e28",
        "outcome": "will die",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 3,
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
        "valuation_reason": "certain death for Ben",
        "scope_node_id": "UTIL_SCOPE:6700869c8ff7480e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:3",
          "grounded_effect:E8"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:b073b98568c2",
        "outcome": "does not receive one dose of medicine",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 3,
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
        "valuation_reason": "foregone dose for Ben",
        "scope_node_id": "UTIL_SCOPE:6700869c8ff7480e",
        "provenance": [
          "delegate:utilitarian",
          "cycle:3",
          "grounded_effect:E6"
        ]
      },
      {
        "consequence_node_id": "UTIL_CONSEQUENCE:utilitarian:A1:de793ba19330",
        "outcome": "Cara has a 5% chance of survival",
        "framework": "UTILITARIAN",
        "specialist": "utilitarian",
        "cycle": 3,
        "canonical_action_id": "A1",
        "grounded_effect_id": "E4",
        "world_effect_id": "E4",
        "direction": "BENEFIT",
        "direction_source": "GROUNDED_WORLD_POLARITY",
        "polarity": "BENEFICIAL",
        "grounded_world_polarity": "BENEFICIAL",
        "probability": "UNKNOWN",
        "modality": "PROBABILISTIC",
        "magnitude": "UNKNOWN",
        "duration": "UNKNOWN",
        "reversibility": "UNKNOWN",
        "support": "STATED",
        "epistemic_status": "SCENARIO_GROUNDED",
        "importance": "CRITICAL",
        "valuation_reason": "small survival benefit",
        "scope_node_id": "UTIL_SCOPE:f61f5c5e22749aeb",
        "provenance": [
          "delegate:utilitarian",
          "cycle:3",
          "grounded_effect:E4"
        ]
      }
    ]
  },
  "unsupported_assumption": "Survival probabilities decisive for utilitarian calculus.",
  "normative_reversal_threshold": "NONE",
  "factual_reversal_threshold": "Cara’s survival chance exceeds Ben’s",
  "model_argument_fields": null
}
```
