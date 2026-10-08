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
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
        "effect:85d066740806cbf69a7d04cf1ae65fc7f5ac167120722b715ca67ac2c45c2c6b",
        "effect:bd1f76b811a66b85752f3d2639c647e62f580888f5be8a0000a5bc1cdaeeb287",
        "effect:fdc8e78d85274db9159ab15f29e3f7c5cea24bc8a0a2b43d1221311ca79dbc39"
      ],
      "intervention": "pulls the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    },
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:71d07cd000cabbf1f4b9c26b310ddaa36649cd4d678f81f900cfc346794cb789",
        "effect:7abeb2e53ed3d38c76e7573edf5d2e7a296ce5e1c0dc2f22f877bc50ce5edd08",
        "effect:d57968b79fbd3ef3c8953e80ed0a56f654a9a5260f1e7fb391494deff801fc97",
        "effect:de22ea404c74679e82567bd878b24464d9c9906ead34e1fff89aaf50776833b5"
      ],
      "intervention": "does not",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    }
  ],
  "right": [
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:0640ff55588a6ece0770d290ed8542d99cd1b9eb397cb8959d2aed83ab1a98b8",
        "effect:1d576ed6f67d420c63d628186cf0e366faf433e96e03eb6e3277b7f26d1ac111",
        "effect:2a9481f7d57aea45e28411a7c51df8ab42bde5324998377e5c12f23166427ee6",
        "effect:60872817221c4ee6513992849cecdef45885abf3006f5c07d01ef3047ea0d716"
      ],
      "intervention": "she pulls the lever",
      "recipient_party_ids": [
        "parties:the lever"
      ]
    },
    {
      "actor_party_id": "parties:maria",
      "effect_ids": [
        "effect:395395bf98f914481f47d8c112837792975c1ca41e174aa94679f70bc1b0904e",
        "effect:4cf8fae16ff7aaf90e28ae7b26e1a02481d5eaceed83d6ff68a1cf467439b032",
        "effect:8f654e3cae7e04ed1511d07a60ecfd5637120be78056aa5ec89364fc2670b29b",
        "effect:97bdb1b2f9b65d505a34bc6a3fc430bc35a45a4200269cc631a635622038aad4"
      ],
      "intervention": "she does not",
      "recipient_party_ids": [
        "parties:the lever"
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
      "source_id": "effect:d57968b79fbd3ef3c8953e80ed0a56f654a9a5260f1e7fb391494deff801fc97",
      "target_id": "effect:71d07cd000cabbf1f4b9c26b310ddaa36649cd4d678f81f900cfc346794cb789"
    },
    {
      "action_id": "actions:does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:d57968b79fbd3ef3c8953e80ed0a56f654a9a5260f1e7fb391494deff801fc97",
      "target_id": "effect:7abeb2e53ed3d38c76e7573edf5d2e7a296ce5e1c0dc2f22f877bc50ce5edd08"
    },
    {
      "action_id": "actions:does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:de22ea404c74679e82567bd878b24464d9c9906ead34e1fff89aaf50776833b5",
      "target_id": "effect:d57968b79fbd3ef3c8953e80ed0a56f654a9a5260f1e7fb391494deff801fc97"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
      "target_id": "effect:85d066740806cbf69a7d04cf1ae65fc7f5ac167120722b715ca67ac2c45c2c6b"
    },
    {
      "action_id": "actions:pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84",
      "target_id": "effect:bd1f76b811a66b85752f3d2639c647e62f580888f5be8a0000a5bc1cdaeeb287"
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
  "right": [
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:4cf8fae16ff7aaf90e28ae7b26e1a02481d5eaceed83d6ff68a1cf467439b032",
      "target_id": "effect:97bdb1b2f9b65d505a34bc6a3fc430bc35a45a4200269cc631a635622038aad4"
    },
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:97bdb1b2f9b65d505a34bc6a3fc430bc35a45a4200269cc631a635622038aad4",
      "target_id": "effect:395395bf98f914481f47d8c112837792975c1ca41e174aa94679f70bc1b0904e"
    },
    {
      "action_id": "actions:she does not",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:97bdb1b2f9b65d505a34bc6a3fc430bc35a45a4200269cc631a635622038aad4",
      "target_id": "effect:8f654e3cae7e04ed1511d07a60ecfd5637120be78056aa5ec89364fc2670b29b"
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:1d576ed6f67d420c63d628186cf0e366faf433e96e03eb6e3277b7f26d1ac111",
      "target_id": "effect:2a9481f7d57aea45e28411a7c51df8ab42bde5324998377e5c12f23166427ee6"
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:2a9481f7d57aea45e28411a7c51df8ab42bde5324998377e5c12f23166427ee6",
      "target_id": "effect:0640ff55588a6ece0770d290ed8542d99cd1b9eb397cb8959d2aed83ab1a98b8"
    },
    {
      "action_id": "actions:she pulls the lever",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:2a9481f7d57aea45e28411a7c51df8ab42bde5324998377e5c12f23166427ee6",
      "target_id": "effect:60872817221c4ee6513992849cecdef45885abf3006f5c07d01ef3047ea0d716"
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
      "party_id": "parties:the lever",
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
      "party_id": "parties:the five",
      "polarity": "adverse",
      "predicate": "kill",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:d57968b79fbd3ef3c8953e80ed0a56f654a9a5260f1e7fb391494deff801fc97"
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
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "do",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:de22ea404c74679e82567bd878b24464d9c9906ead34e1fff89aaf50776833b5"
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
      "party_id": "parties:the one",
      "polarity": "beneficial",
      "predicate": "spare",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:d57968b79fbd3ef3c8953e80ed0a56f654a9a5260f1e7fb391494deff801fc97"
      ],
      "source_proposition": "it will kill the five and spare the one",
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
      "outcome": "the trolley will kill that worker",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:that worker",
      "polarity": "adverse",
      "predicate": "kill",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
      ],
      "source_proposition": "the trolley will kill that worker",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:pulls the lever",
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
      "party_id": "parties:the five",
      "polarity": "beneficial",
      "predicate": "spare",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:0e37cdf0345f536d0f4da3bea66e547fe5493dd90cf1b437c550806ae757eb84"
      ],
      "source_proposition": "the trolley will kill that worker and spare the five",
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
      "party_id": "parties:the lever",
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
      "party_id": "parties:the five",
      "polarity": "adverse",
      "predicate": "kill",
      "quantities": [
        "five"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:97bdb1b2f9b65d505a34bc6a3fc430bc35a45a4200269cc631a635622038aad4"
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
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "do",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:4cf8fae16ff7aaf90e28ae7b26e1a02481d5eaceed83d6ff68a1cf467439b032"
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
      "party_id": "parties:the one",
      "polarity": "beneficial",
      "predicate": "spare",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:97bdb1b2f9b65d505a34bc6a3fc430bc35a45a4200269cc631a635622038aad4"
      ],
      "source_proposition": "it will kill the five and spare the one",
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
      "party_id": "parties:the lever",
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
      "party_id": "parties:that worker",
      "polarity": "adverse",
      "predicate": "kill",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:2a9481f7d57aea45e28411a7c51df8ab42bde5324998377e5c12f23166427ee6"
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
      "party_id": "parties:the five",
      "polarity": "beneficial",
      "predicate": "spare",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:2a9481f7d57aea45e28411a7c51df8ab42bde5324998377e5c12f23166427ee6"
      ],
      "source_proposition": "the trolley will kill that worker and spare the five",
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
      "outcome": "she pulls the lever",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:the lever",
      "polarity": "neutral",
      "predicate": "pull",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:1d576ed6f67d420c63d628186cf0e366faf433e96e03eb6e3277b7f26d1ac111"
      ],
      "source_proposition": "she pulls the lever",
      "temporal_qualifiers": []
    }
  ]
}
```

