# Admitted graph differences

Blueprint names do not establish semantic equivalence.

## difference_0: candidate_0 / candidate_1 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": [
        "one"
      ]
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_1: candidate_0 / candidate_2 — modal_operators

Which modal_operators reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "bearer_party_id": "parties:maria",
      "force": "ability",
      "governed_proposition_id": "pr1",
      "modal_id": "m0"
    }
  ]
}
```

## difference_2: candidate_0 / candidate_2 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": [
        "one"
      ]
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_3: candidate_0 / candidate_2 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "polarity": "positive",
      "predication": "can pull the lever",
      "proposition_id": "pr1",
      "status": "governed"
    }
  ]
}
```

## difference_4: candidate_0 / candidate_3 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
        "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f",
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "intervention": "does not pull the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    },
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
        "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7",
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    }
  ],
  "right": []
}
```

## difference_5: candidate_0 / candidate_3 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
      "target_id": "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f"
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03",
      "target_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
      "target_id": "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39",
      "target_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
    }
  ],
  "right": []
}
```

## difference_6: candidate_0 / candidate_3 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [
    {
      "decision_relevance": "material",
      "description": "if maria does not pull the lever",
      "operator": "if",
      "polarity": "negated",
      "value_status": "unknown"
    },
    {
      "decision_relevance": "material",
      "description": "if maria pulls the lever",
      "operator": "if",
      "polarity": "positive",
      "value_status": "unknown"
    }
  ],
  "right": []
}
```

## difference_7: candidate_0 / candidate_3 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "five workers will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:five workers",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
      ],
      "source_proposition": "five workers will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "one worker will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:one worker",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
      ],
      "source_proposition": "one worker will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    }
  ],
  "right": []
}
```

## difference_8: candidate_0 / candidate_3 — modal_operators

Which modal_operators reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "bearer_party_id": "parties:maria",
      "force": "ability",
      "governed_proposition_id": "pr1",
      "modal_id": "m0"
    }
  ]
}
```

## difference_9: candidate_0 / candidate_3 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": [
        "one"
      ]
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_10: candidate_0 / candidate_3 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "polarity": "positive",
      "predication": "can pull the lever",
      "proposition_id": "pr1",
      "status": "governed"
    }
  ]
}
```

## difference_11: candidate_0 / candidate_4 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": [
        "one"
      ]
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_12: candidate_0 / candidate_5 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": [
        "one"
      ]
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_13: candidate_1 / candidate_2 — modal_operators

Which modal_operators reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "bearer_party_id": "parties:maria",
      "force": "ability",
      "governed_proposition_id": "pr1",
      "modal_id": "m0"
    }
  ]
}
```

## difference_14: candidate_1 / candidate_2 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_15: candidate_1 / candidate_2 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "polarity": "positive",
      "predication": "can pull the lever",
      "proposition_id": "pr1",
      "status": "governed"
    }
  ]
}
```

## difference_16: candidate_1 / candidate_3 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
        "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f",
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "intervention": "does not pull the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    },
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
        "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7",
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    }
  ],
  "right": []
}
```

## difference_17: candidate_1 / candidate_3 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
      "target_id": "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f"
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03",
      "target_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
      "target_id": "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39",
      "target_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
    }
  ],
  "right": []
}
```

## difference_18: candidate_1 / candidate_3 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [
    {
      "decision_relevance": "material",
      "description": "if maria does not pull the lever",
      "operator": "if",
      "polarity": "negated",
      "value_status": "unknown"
    },
    {
      "decision_relevance": "material",
      "description": "if maria pulls the lever",
      "operator": "if",
      "polarity": "positive",
      "value_status": "unknown"
    }
  ],
  "right": []
}
```

## difference_19: candidate_1 / candidate_3 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "five workers will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:five workers",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
      ],
      "source_proposition": "five workers will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "one worker will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:one worker",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
      ],
      "source_proposition": "one worker will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    }
  ],
  "right": []
}
```

## difference_20: candidate_1 / candidate_3 — modal_operators

Which modal_operators reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "bearer_party_id": "parties:maria",
      "force": "ability",
      "governed_proposition_id": "pr1",
      "modal_id": "m0"
    }
  ]
}
```

## difference_21: candidate_1 / candidate_3 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_22: candidate_1 / candidate_3 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "polarity": "positive",
      "predication": "can pull the lever",
      "proposition_id": "pr1",
      "status": "governed"
    }
  ]
}
```

## difference_23: candidate_2 / candidate_3 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
        "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f",
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "intervention": "does not pull the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    },
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
        "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7",
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    }
  ],
  "right": []
}
```

## difference_24: candidate_2 / candidate_3 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
      "target_id": "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f"
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03",
      "target_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
      "target_id": "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39",
      "target_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
    }
  ],
  "right": []
}
```

## difference_25: candidate_2 / candidate_3 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [
    {
      "decision_relevance": "material",
      "description": "if maria does not pull the lever",
      "operator": "if",
      "polarity": "negated",
      "value_status": "unknown"
    },
    {
      "decision_relevance": "material",
      "description": "if maria pulls the lever",
      "operator": "if",
      "polarity": "positive",
      "value_status": "unknown"
    }
  ],
  "right": []
}
```

## difference_26: candidate_2 / candidate_3 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "five workers will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:five workers",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
      ],
      "source_proposition": "five workers will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "one worker will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:one worker",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
      ],
      "source_proposition": "one worker will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    }
  ],
  "right": []
}
```

## difference_27: candidate_2 / candidate_3 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_28: candidate_2 / candidate_4 — modal_operators

Which modal_operators reading is supported by the source?

```json
{
  "left": [
    {
      "bearer_party_id": "parties:maria",
      "force": "ability",
      "governed_proposition_id": "pr1",
      "modal_id": "m0"
    }
  ],
  "right": []
}
```

## difference_29: candidate_2 / candidate_4 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_30: candidate_2 / candidate_4 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [
    {
      "polarity": "positive",
      "predication": "can pull the lever",
      "proposition_id": "pr1",
      "status": "governed"
    }
  ],
  "right": []
}
```

## difference_31: candidate_2 / candidate_5 — modal_operators

Which modal_operators reading is supported by the source?

```json
{
  "left": [
    {
      "bearer_party_id": "parties:maria",
      "force": "ability",
      "governed_proposition_id": "pr1",
      "modal_id": "m0"
    }
  ],
  "right": []
}
```

## difference_32: candidate_2 / candidate_5 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_33: candidate_2 / candidate_5 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [
    {
      "polarity": "positive",
      "predication": "can pull the lever",
      "proposition_id": "pr1",
      "status": "governed"
    }
  ],
  "right": []
}
```

## difference_34: candidate_3 / candidate_4 — actions

Which actions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
        "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f",
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "intervention": "does not pull the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    },
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
        "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7",
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    }
  ]
}
```

## difference_35: candidate_3 / candidate_4 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
      "target_id": "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f"
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03",
      "target_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
      "target_id": "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39",
      "target_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
    }
  ]
}
```

## difference_36: candidate_3 / candidate_4 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "decision_relevance": "material",
      "description": "if maria does not pull the lever",
      "operator": "if",
      "polarity": "negated",
      "value_status": "unknown"
    },
    {
      "decision_relevance": "material",
      "description": "if maria pulls the lever",
      "operator": "if",
      "polarity": "positive",
      "value_status": "unknown"
    }
  ]
}
```

## difference_37: candidate_3 / candidate_4 — effects

Which effects reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "five workers will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:five workers",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
      ],
      "source_proposition": "five workers will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "one worker will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:one worker",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
      ],
      "source_proposition": "one worker will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    }
  ]
}
```

## difference_38: candidate_3 / candidate_4 — modal_operators

Which modal_operators reading is supported by the source?

```json
{
  "left": [
    {
      "bearer_party_id": "parties:maria",
      "force": "ability",
      "governed_proposition_id": "pr1",
      "modal_id": "m0"
    }
  ],
  "right": []
}
```

## difference_39: candidate_3 / candidate_4 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_40: candidate_3 / candidate_4 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [
    {
      "polarity": "positive",
      "predication": "can pull the lever",
      "proposition_id": "pr1",
      "status": "governed"
    }
  ],
  "right": []
}
```

## difference_41: candidate_3 / candidate_5 — actions

Which actions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
        "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f",
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "intervention": "does not pull the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    },
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
        "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7",
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    }
  ]
}
```

## difference_42: candidate_3 / candidate_5 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf",
      "target_id": "effect:1a8ab0752e9a537772c70946225798d6d667b66c446cf0ead026ccc51b3dbc7f"
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03",
      "target_id": "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
      "target_id": "effect:eb97811e060d44824d961716ff4e6aa6d942940b0be74f55f828f753bbcb1cb7"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39",
      "target_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
    }
  ]
}
```

## difference_43: candidate_3 / candidate_5 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "decision_relevance": "material",
      "description": "if maria does not pull the lever",
      "operator": "if",
      "polarity": "negated",
      "value_status": "unknown"
    },
    {
      "decision_relevance": "material",
      "description": "if maria pulls the lever",
      "operator": "if",
      "polarity": "positive",
      "value_status": "unknown"
    }
  ]
}
```

## difference_44: candidate_3 / candidate_5 — effects

Which effects reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "five workers will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:five workers",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:04694bbe1a8a3e25a8cbd0925e04e2a0171de3a8de0b12be6cdc0c758b2ad8bf"
      ],
      "source_proposition": "five workers will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not pull the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not pull the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:d4002e478b7c056757b6c342249ba0315fde11971eb9113716a6aa3f9fc4eb03"
      ],
      "source_proposition": "does not pull the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "one worker will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:one worker",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
      ],
      "source_proposition": "one worker will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "source_proposition": "pulls the lever",
      "temporal_qualifiers": []
    }
  ]
}
```

## difference_45: candidate_3 / candidate_5 — modal_operators

Which modal_operators reading is supported by the source?

```json
{
  "left": [
    {
      "bearer_party_id": "parties:maria",
      "force": "ability",
      "governed_proposition_id": "pr1",
      "modal_id": "m0"
    }
  ],
  "right": []
}
```

## difference_46: candidate_3 / candidate_5 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "human_group",
      "kind_origin": "structurally_derived",
      "label": "five workers",
      "quantities": [
        "five"
      ]
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "maria",
      "quantities": []
    },
    {
      "kind": "person",
      "kind_origin": "structurally_derived",
      "label": "one worker",
      "quantities": []
    },
    {
      "kind": "process",
      "kind_origin": "structurally_derived",
      "label": "the lever",
      "quantities": []
    }
  ]
}
```

## difference_47: candidate_3 / candidate_5 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [
    {
      "polarity": "positive",
      "predication": "can pull the lever",
      "proposition_id": "pr1",
      "status": "governed"
    }
  ],
  "right": []
}
```

