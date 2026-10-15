# Admitted graph differences

Blueprint names do not establish semantic equivalence.

## difference_0: candidate_0 / candidate_1 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0270c09ead24ebff597aea0d0bf3aac4711d3c160bab6d7f79facfeefaa08f0c",
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
        "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4",
        "effect:a304acfbebde490c3e80eb01bb60fc0990ef7720eaeb0217bd5987bf88dacc4b",
        "effect:dcca4481b9a4c3152299b35fb3fb5c9e88d3ea2834080e5d56b792c231b79e34",
        "effect:e1fb3e01def022a11d6ffa73c65e7457b13279a1875cbebab16b09c90fc72c77"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    }
  ],
  "right": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0270c09ead24ebff597aea0d0bf3aac4711d3c160bab6d7f79facfeefaa08f0c",
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
        "effect:a304acfbebde490c3e80eb01bb60fc0990ef7720eaeb0217bd5987bf88dacc4b",
        "effect:dcca4481b9a4c3152299b35fb3fb5c9e88d3ea2834080e5d56b792c231b79e34"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    }
  ]
}
```

## difference_1: candidate_0 / candidate_1 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
      "target_id": "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4",
      "target_id": "effect:e1fb3e01def022a11d6ffa73c65e7457b13279a1875cbebab16b09c90fc72c77"
    }
  ],
  "right": []
}
```

## difference_2: candidate_0 / candidate_1 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [
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

## difference_3: candidate_0 / candidate_1 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
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
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563"
      ],
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
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "the signal will activate",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the signal",
      "polarity": "neutral",
      "predicate": "activate",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4"
      ],
      "source_proposition": "if maria pulls the lever, the signal will activate",
      "temporal_qualifiers": []
    }
  ],
  "right": []
}
```

## difference_4: candidate_0 / candidate_1 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "process",
      "label": "the signal",
      "quantities": []
    }
  ],
  "right": []
}
```

## difference_5: candidate_0 / candidate_3 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0270c09ead24ebff597aea0d0bf3aac4711d3c160bab6d7f79facfeefaa08f0c",
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
        "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4",
        "effect:a304acfbebde490c3e80eb01bb60fc0990ef7720eaeb0217bd5987bf88dacc4b",
        "effect:dcca4481b9a4c3152299b35fb3fb5c9e88d3ea2834080e5d56b792c231b79e34",
        "effect:e1fb3e01def022a11d6ffa73c65e7457b13279a1875cbebab16b09c90fc72c77"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    }
  ],
  "right": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0270c09ead24ebff597aea0d0bf3aac4711d3c160bab6d7f79facfeefaa08f0c",
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
        "effect:a304acfbebde490c3e80eb01bb60fc0990ef7720eaeb0217bd5987bf88dacc4b",
        "effect:dcca4481b9a4c3152299b35fb3fb5c9e88d3ea2834080e5d56b792c231b79e34"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    }
  ]
}
```

## difference_6: candidate_0 / candidate_3 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
      "target_id": "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4",
      "target_id": "effect:e1fb3e01def022a11d6ffa73c65e7457b13279a1875cbebab16b09c90fc72c77"
    }
  ],
  "right": []
}
```

## difference_7: candidate_0 / candidate_3 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [
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

## difference_8: candidate_0 / candidate_3 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
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
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563"
      ],
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
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "the signal will activate",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the signal",
      "polarity": "neutral",
      "predicate": "activate",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4"
      ],
      "source_proposition": "if maria pulls the lever, the signal will activate",
      "temporal_qualifiers": []
    }
  ],
  "right": []
}
```

## difference_9: candidate_0 / candidate_3 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "process",
      "label": "the signal",
      "quantities": []
    }
  ],
  "right": []
}
```

## difference_10: candidate_1 / candidate_2 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0270c09ead24ebff597aea0d0bf3aac4711d3c160bab6d7f79facfeefaa08f0c",
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
        "effect:a304acfbebde490c3e80eb01bb60fc0990ef7720eaeb0217bd5987bf88dacc4b",
        "effect:dcca4481b9a4c3152299b35fb3fb5c9e88d3ea2834080e5d56b792c231b79e34"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    }
  ],
  "right": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0270c09ead24ebff597aea0d0bf3aac4711d3c160bab6d7f79facfeefaa08f0c",
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
        "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4",
        "effect:a304acfbebde490c3e80eb01bb60fc0990ef7720eaeb0217bd5987bf88dacc4b",
        "effect:dcca4481b9a4c3152299b35fb3fb5c9e88d3ea2834080e5d56b792c231b79e34",
        "effect:e1fb3e01def022a11d6ffa73c65e7457b13279a1875cbebab16b09c90fc72c77"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    }
  ]
}
```

## difference_11: candidate_1 / candidate_2 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
      "target_id": "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4",
      "target_id": "effect:e1fb3e01def022a11d6ffa73c65e7457b13279a1875cbebab16b09c90fc72c77"
    }
  ]
}
```

## difference_12: candidate_1 / candidate_2 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [],
  "right": [
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

## difference_13: candidate_1 / candidate_2 — effects

Which effects reading is supported by the source?

```json
{
  "left": [],
  "right": [
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
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563"
      ],
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
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "the signal will activate",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the signal",
      "polarity": "neutral",
      "predicate": "activate",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4"
      ],
      "source_proposition": "if maria pulls the lever, the signal will activate",
      "temporal_qualifiers": []
    }
  ]
}
```

## difference_14: candidate_1 / candidate_2 — parties

Which parties reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "kind": "process",
      "label": "the signal",
      "quantities": []
    }
  ]
}
```

## difference_15: candidate_2 / candidate_3 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0270c09ead24ebff597aea0d0bf3aac4711d3c160bab6d7f79facfeefaa08f0c",
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
        "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4",
        "effect:a304acfbebde490c3e80eb01bb60fc0990ef7720eaeb0217bd5987bf88dacc4b",
        "effect:dcca4481b9a4c3152299b35fb3fb5c9e88d3ea2834080e5d56b792c231b79e34",
        "effect:e1fb3e01def022a11d6ffa73c65e7457b13279a1875cbebab16b09c90fc72c77"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    }
  ],
  "right": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0270c09ead24ebff597aea0d0bf3aac4711d3c160bab6d7f79facfeefaa08f0c",
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
        "effect:a304acfbebde490c3e80eb01bb60fc0990ef7720eaeb0217bd5987bf88dacc4b",
        "effect:dcca4481b9a4c3152299b35fb3fb5c9e88d3ea2834080e5d56b792c231b79e34"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    }
  ]
}
```

## difference_16: candidate_2 / candidate_3 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563",
      "target_id": "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4",
      "target_id": "effect:e1fb3e01def022a11d6ffa73c65e7457b13279a1875cbebab16b09c90fc72c77"
    }
  ],
  "right": []
}
```

## difference_17: candidate_2 / candidate_3 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [
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

## difference_18: candidate_2 / candidate_3 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
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
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:179e8d16711605418767b073b77bd68a39354ad84c9415eea051ded1c8e0a563"
      ],
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
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "the signal will activate",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the signal",
      "polarity": "neutral",
      "predicate": "activate",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:97b772cbd00a2838a521e2abd3deb9db40d587710773616544ac4214e1d075a4"
      ],
      "source_proposition": "if maria pulls the lever, the signal will activate",
      "temporal_qualifiers": []
    }
  ],
  "right": []
}
```

## difference_19: candidate_2 / candidate_3 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "process",
      "label": "the signal",
      "quantities": []
    }
  ],
  "right": []
}
```

