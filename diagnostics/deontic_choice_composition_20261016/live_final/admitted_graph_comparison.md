# Admitted graph differences

Blueprint names do not establish semantic equivalence.

## difference_0: candidate_0 / candidate_1 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:mira",
      "effect_ids": [
        "effect:15f557db1cb4231f034eabc621edbeb3ed68a8119a2650ee468d32c3118195f4",
        "effect:ef1db5d8be68092121ca0b6b0e0723fa485b39db5f452668d0c058c987e6b2fb"
      ],
      "intervention": "open the door now",
      "recipient_party_ids": [
        "parties:the door"
      ]
    },
    {
      "actor_party_id": "parties:mira",
      "effect_ids": [
        "effect:6b51b5d3bfba70d755bba55cca247a318711d9462214f3840af57331668a132c",
        "effect:7fad10ef10a3c5a3ff760da96af06e9ab4ded9344c2c2242fe278b53d2e831df",
        "effect:a67e76663970937db4cf04215f77b938630685cef997099477a8594ad096a3d0"
      ],
      "intervention": "stop the experiment and wait",
      "recipient_party_ids": [
        "parties:the experiment"
      ]
    }
  ],
  "right": []
}
```

## difference_1: candidate_0 / candidate_1 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:open the door now",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:15f557db1cb4231f034eabc621edbeb3ed68a8119a2650ee468d32c3118195f4",
      "target_id": "effect:ef1db5d8be68092121ca0b6b0e0723fa485b39db5f452668d0c058c987e6b2fb"
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:6b51b5d3bfba70d755bba55cca247a318711d9462214f3840af57331668a132c",
      "target_id": "effect:7fad10ef10a3c5a3ff760da96af06e9ab4ded9344c2c2242fe278b53d2e831df"
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:a67e76663970937db4cf04215f77b938630685cef997099477a8594ad096a3d0",
      "target_id": "effect:6b51b5d3bfba70d755bba55cca247a318711d9462214f3840af57331668a132c"
    }
  ],
  "right": []
}
```

## difference_2: candidate_0 / candidate_1 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:open the door now",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "open the door now",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the door",
      "polarity": "neutral",
      "predicate": "open",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "open the door now",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:open the door now",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [
        "may"
      ],
      "modality": "possible",
      "outcome": "doing so may expose nearby staff to hazardous fumes",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:nearby staff",
      "polarity": "adverse",
      "predicate": "expose",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:15f557db1cb4231f034eabc621edbeb3ed68a8119a2650ee468d32c3118195f4"
      ],
      "source_proposition": "doing so may expose nearby staff to hazardous fumes",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "stopping the experiment",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the experiment",
      "polarity": "neutral",
      "predicate": "stop",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "stopping the experiment",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [
        "may"
      ],
      "modality": "possible",
      "outcome": "the technician may die during that delay",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the technician",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:6b51b5d3bfba70d755bba55cca247a318711d9462214f3840af57331668a132c"
      ],
      "source_proposition": "the technician may die during that delay",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "stopping the experiment first takes ten minutes",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the experiment",
      "polarity": "neutral",
      "predicate": "take",
      "quantities": [
        "ten minutes"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:a67e76663970937db4cf04215f77b938630685cef997099477a8594ad096a3d0"
      ],
      "source_proposition": "stopping the experiment first takes ten minutes",
      "temporal_qualifiers": [
        "ten minutes"
      ]
    }
  ],
  "right": []
}
```

## difference_3: candidate_0 / candidate_1 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "person",
      "label": "nearby staff",
      "quantities": []
    },
    {
      "kind": "person",
      "label": "the technician",
      "quantities": []
    },
    {
      "kind": "process",
      "label": "the door",
      "quantities": []
    },
    {
      "kind": "process",
      "label": "the experiment",
      "quantities": []
    }
  ],
  "right": []
}
```

## difference_4: candidate_1 / candidate_2 — actions

Which actions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "actor_party_id": "parties:mira",
      "effect_ids": [
        "effect:15f557db1cb4231f034eabc621edbeb3ed68a8119a2650ee468d32c3118195f4",
        "effect:ef1db5d8be68092121ca0b6b0e0723fa485b39db5f452668d0c058c987e6b2fb"
      ],
      "intervention": "open the door now",
      "recipient_party_ids": [
        "parties:the door"
      ]
    },
    {
      "actor_party_id": "parties:mira",
      "effect_ids": [
        "effect:6b51b5d3bfba70d755bba55cca247a318711d9462214f3840af57331668a132c",
        "effect:7fad10ef10a3c5a3ff760da96af06e9ab4ded9344c2c2242fe278b53d2e831df",
        "effect:a67e76663970937db4cf04215f77b938630685cef997099477a8594ad096a3d0"
      ],
      "intervention": "stop the experiment and wait",
      "recipient_party_ids": [
        "parties:the experiment"
      ]
    }
  ]
}
```

## difference_5: candidate_1 / candidate_2 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "action_id": "actions:open the door now",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:15f557db1cb4231f034eabc621edbeb3ed68a8119a2650ee468d32c3118195f4",
      "target_id": "effect:ef1db5d8be68092121ca0b6b0e0723fa485b39db5f452668d0c058c987e6b2fb"
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:6b51b5d3bfba70d755bba55cca247a318711d9462214f3840af57331668a132c",
      "target_id": "effect:7fad10ef10a3c5a3ff760da96af06e9ab4ded9344c2c2242fe278b53d2e831df"
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:a67e76663970937db4cf04215f77b938630685cef997099477a8594ad096a3d0",
      "target_id": "effect:6b51b5d3bfba70d755bba55cca247a318711d9462214f3840af57331668a132c"
    }
  ]
}
```

## difference_6: candidate_1 / candidate_2 — effects

Which effects reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "action_id": "actions:open the door now",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "open the door now",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the door",
      "polarity": "neutral",
      "predicate": "open",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "open the door now",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:open the door now",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [
        "may"
      ],
      "modality": "possible",
      "outcome": "doing so may expose nearby staff to hazardous fumes",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:nearby staff",
      "polarity": "adverse",
      "predicate": "expose",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:15f557db1cb4231f034eabc621edbeb3ed68a8119a2650ee468d32c3118195f4"
      ],
      "source_proposition": "doing so may expose nearby staff to hazardous fumes",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "stopping the experiment",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the experiment",
      "polarity": "neutral",
      "predicate": "stop",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "stopping the experiment",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [
        "may"
      ],
      "modality": "possible",
      "outcome": "the technician may die during that delay",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the technician",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:6b51b5d3bfba70d755bba55cca247a318711d9462214f3840af57331668a132c"
      ],
      "source_proposition": "the technician may die during that delay",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:stop the experiment and wait",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "stopping the experiment first takes ten minutes",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the experiment",
      "polarity": "neutral",
      "predicate": "take",
      "quantities": [
        "ten minutes"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:a67e76663970937db4cf04215f77b938630685cef997099477a8594ad096a3d0"
      ],
      "source_proposition": "stopping the experiment first takes ten minutes",
      "temporal_qualifiers": [
        "ten minutes"
      ]
    }
  ]
}
```

## difference_7: candidate_1 / candidate_2 — parties

Which parties reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "kind": "person",
      "label": "nearby staff",
      "quantities": []
    },
    {
      "kind": "person",
      "label": "the technician",
      "quantities": []
    },
    {
      "kind": "process",
      "label": "the door",
      "quantities": []
    },
    {
      "kind": "process",
      "label": "the experiment",
      "quantities": []
    }
  ]
}
```

