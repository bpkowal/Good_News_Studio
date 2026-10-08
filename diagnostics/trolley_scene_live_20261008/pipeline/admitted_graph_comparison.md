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
        "effect:0507cf607ae2b110f4866d2614c2c18660ca0fee7a40b326d0c75433a4d5ef3f",
        "effect:8894d2bea3d086e07acd203dda8916f4fabd62c7bba1d0013c130869c89c0441",
        "effect:94b922c59b0d2f3527e2c02da4d6d516c8d9870a373275d807627d6dc0bea21b",
        "effect:97bce75d7f60acf989fe994b088bd82616d337b59bfb7cd7afa04700891def8e"
      ],
      "intervention": "does not",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    },
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:2e6f8388d95a1c8d8d965fed54bbe474a0eec88e48b5fb97d1a91cccb2b33366",
        "effect:562558ee082065ac7621e22c88a1fefe1ae33c03aafd231d5b6adb993a96dc9d",
        "effect:5f0252bb3c81a8e1fe4a4dd27cfb9669997efdf96798f24a33dc5c0800215f57",
        "effect:96fe8a36e4eec195e7fcbaa57305d59e33d6d6cb8649b45f6a5e9e6717bb46d7"
      ],
      "intervention": "pull a lever that switches the trolley onto a side track",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    }
  ],
  "right": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0a56e867138f9aca3fc53abc2cedb3ad13e997f1b9ff0c81b82b1d07a2f4d11c",
        "effect:16fbcb232b61631aa0a32df3fd529419b5d6fae91189899e7a18ceadabf51a42",
        "effect:38e3345553a42e5c968e422b6e2b72cc02198be85488ffcda0f22c249005e433",
        "effect:db4f82db9143d53838454a8e3a99426038cfc76c37a8d8efb9875ed4c2380f67"
      ],
      "intervention": "she pulls the lever",
      "recipient_party_ids": [
        "parties:a lever"
      ]
    },
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:113c5fadc3ec93a8287193d51c5c35c2267eb05449e0f2fac7d2f7716d7ede15",
        "effect:13189b22bc09ce022f8e6950c2dbb257cd055c9ae4db4cf33e054e90acbe59ec",
        "effect:30a65594c0968d394f171eae54c9e7792689e527ea6a7e76db1dcf95240696e3",
        "effect:8ea4a6f8c52e248249aa1d7e6934c75c71e136ddde6bbbf4583f99fe7bdb59df"
      ],
      "intervention": "she does not",
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
      "action_id": "actions:does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:0507cf607ae2b110f4866d2614c2c18660ca0fee7a40b326d0c75433a4d5ef3f",
      "target_id": "effect:97bce75d7f60acf989fe994b088bd82616d337b59bfb7cd7afa04700891def8e"
    },
    {
      "action_id": "actions:does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:97bce75d7f60acf989fe994b088bd82616d337b59bfb7cd7afa04700891def8e",
      "target_id": "effect:8894d2bea3d086e07acd203dda8916f4fabd62c7bba1d0013c130869c89c0441"
    },
    {
      "action_id": "actions:does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:97bce75d7f60acf989fe994b088bd82616d337b59bfb7cd7afa04700891def8e",
      "target_id": "effect:94b922c59b0d2f3527e2c02da4d6d516c8d9870a373275d807627d6dc0bea21b"
    },
    {
      "action_id": "actions:pull a lever that switches the trolley onto a side track",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:2e6f8388d95a1c8d8d965fed54bbe474a0eec88e48b5fb97d1a91cccb2b33366",
      "target_id": "effect:562558ee082065ac7621e22c88a1fefe1ae33c03aafd231d5b6adb993a96dc9d"
    },
    {
      "action_id": "actions:pull a lever that switches the trolley onto a side track",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:2e6f8388d95a1c8d8d965fed54bbe474a0eec88e48b5fb97d1a91cccb2b33366",
      "target_id": "effect:96fe8a36e4eec195e7fcbaa57305d59e33d6d6cb8649b45f6a5e9e6717bb46d7"
    },
    {
      "action_id": "actions:pull a lever that switches the trolley onto a side track",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:5f0252bb3c81a8e1fe4a4dd27cfb9669997efdf96798f24a33dc5c0800215f57",
      "target_id": "effect:2e6f8388d95a1c8d8d965fed54bbe474a0eec88e48b5fb97d1a91cccb2b33366"
    }
  ],
  "right": [
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:113c5fadc3ec93a8287193d51c5c35c2267eb05449e0f2fac7d2f7716d7ede15",
      "target_id": "effect:13189b22bc09ce022f8e6950c2dbb257cd055c9ae4db4cf33e054e90acbe59ec"
    },
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:113c5fadc3ec93a8287193d51c5c35c2267eb05449e0f2fac7d2f7716d7ede15",
      "target_id": "effect:8ea4a6f8c52e248249aa1d7e6934c75c71e136ddde6bbbf4583f99fe7bdb59df"
    },
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:30a65594c0968d394f171eae54c9e7792689e527ea6a7e76db1dcf95240696e3",
      "target_id": "effect:113c5fadc3ec93a8287193d51c5c35c2267eb05449e0f2fac7d2f7716d7ede15"
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:0a56e867138f9aca3fc53abc2cedb3ad13e997f1b9ff0c81b82b1d07a2f4d11c",
      "target_id": "effect:16fbcb232b61631aa0a32df3fd529419b5d6fae91189899e7a18ceadabf51a42"
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:16fbcb232b61631aa0a32df3fd529419b5d6fae91189899e7a18ceadabf51a42",
      "target_id": "effect:38e3345553a42e5c968e422b6e2b72cc02198be85488ffcda0f22c249005e433"
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:16fbcb232b61631aa0a32df3fd529419b5d6fae91189899e7a18ceadabf51a42",
      "target_id": "effect:db4f82db9143d53838454a8e3a99426038cfc76c37a8d8efb9875ed4c2380f67"
    }
  ]
}
```

## difference_2: candidate_0 / candidate_1 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:does not",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "do",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "does not",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "it will kill the five",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:five workers",
      "polarity": "adverse",
      "predicate": "kill",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:97bce75d7f60acf989fe994b088bd82616d337b59bfb7cd7afa04700891def8e"
      ],
      "source_proposition": "it will kill the five",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "do",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:0507cf607ae2b110f4866d2614c2c18660ca0fee7a40b326d0c75433a4d5ef3f"
      ],
      "source_proposition": "does not",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:does not",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "spare the one",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:one worker",
      "polarity": "beneficial",
      "predicate": "spare",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:97bce75d7f60acf989fe994b088bd82616d337b59bfb7cd7afa04700891def8e"
      ],
      "source_proposition": "spare the one",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pull a lever that switches the trolley onto a side track",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "pull a lever that switches the trolley onto a side track",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "pull a lever that switches the trolley onto a side track",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pull a lever that switches the trolley onto a side track",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "the trolley will kill that worker",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:one worker",
      "polarity": "adverse",
      "predicate": "kill",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:2e6f8388d95a1c8d8d965fed54bbe474a0eec88e48b5fb97d1a91cccb2b33366"
      ],
      "source_proposition": "the trolley will kill that worker",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pull a lever that switches the trolley onto a side track",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "spare the five",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:five workers",
      "polarity": "beneficial",
      "predicate": "spare",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:2e6f8388d95a1c8d8d965fed54bbe474a0eec88e48b5fb97d1a91cccb2b33366"
      ],
      "source_proposition": "spare the five",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pull a lever that switches the trolley onto a side track",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "switches the trolley onto a side track",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the trolley",
      "polarity": "neutral",
      "predicate": "switch",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:5f0252bb3c81a8e1fe4a4dd27cfb9669997efdf96798f24a33dc5c0800215f57"
      ],
      "source_proposition": "switches the trolley onto a side track",
      "temporal_qualifiers": []
    }
  ],
  "right": [
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "she does not",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "do",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "she does not",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "it will kill the five",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:five workers",
      "polarity": "adverse",
      "predicate": "kill",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:113c5fadc3ec93a8287193d51c5c35c2267eb05449e0f2fac7d2f7716d7ede15"
      ],
      "source_proposition": "it will kill the five",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "she does not",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "do",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:30a65594c0968d394f171eae54c9e7792689e527ea6a7e76db1dcf95240696e3"
      ],
      "source_proposition": "she does not",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "spare the one",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:one worker",
      "polarity": "beneficial",
      "predicate": "spare",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:113c5fadc3ec93a8287193d51c5c35c2267eb05449e0f2fac7d2f7716d7ede15"
      ],
      "source_proposition": "spare the one",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "she pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:a lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "she pulls the lever",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "the trolley will kill that worker",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:one worker",
      "polarity": "adverse",
      "predicate": "kill",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:16fbcb232b61631aa0a32df3fd529419b5d6fae91189899e7a18ceadabf51a42"
      ],
      "source_proposition": "the trolley will kill that worker",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "spare the five",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:five workers",
      "polarity": "beneficial",
      "predicate": "spare",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:16fbcb232b61631aa0a32df3fd529419b5d6fae91189899e7a18ceadabf51a42"
      ],
      "source_proposition": "spare the five",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "physical_state",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "switches the trolley onto a side track",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the trolley",
      "polarity": "neutral",
      "predicate": "switch",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:0a56e867138f9aca3fc53abc2cedb3ad13e997f1b9ff0c81b82b1d07a2f4d11c"
      ],
      "source_proposition": "switches the trolley onto a side track",
      "temporal_qualifiers": []
    }
  ]
}
```

## difference_3: candidate_0 / candidate_1 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "process",
      "label": "switches the trolley",
      "quantities": []
    }
  ],
  "right": []
}
```

