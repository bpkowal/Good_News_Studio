# Admitted graph differences

Blueprint names do not establish semantic equivalence.

## difference_0: candidate_0 / candidate_1 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:rescuer ellis",
      "effect_ids": [
        "effect:379ea201d1cd731652aa153a6956399a5dbcde4fb80fc9a9c06e96d026cec2d6",
        "effect:c9d291c3600075cbae03db8d6062798a705f40b44e8a4d753d219123a21d0f1c",
        "effect:e0ed6d83c830b939341d0aeb5035a8a4a6cb86a97fe77ba2bdbf8350418f6ba4",
        "effect:f592754297e544973538b723c2419e96d2f7056a1fbdad74712cf5aa7481edc7"
      ],
      "intervention": "gives the blanket to sasha",
      "recipient_party_ids": [
        "parties:sasha"
      ]
    },
    {
      "actor_party_id": "parties:rescuer ellis",
      "effect_ids": [
        "effect:3ac83fd89f524d292b25a18398ddd2daaa55d97c57ea732909da301d7b23cb2f",
        "effect:7f9bceea2154ef7eae950660f4c3daeb8b01baaf699f17d5728d8d14fdfc5dab",
        "effect:dae7cf092009ee42c2a59ea1033ec612dc776d60a57e18802a520e7d218681f8",
        "effect:ef73c1db80739a46f44d548d68a0365176a409358ade9ae7923d9237089ea406"
      ],
      "intervention": "gives the blanket to rowan",
      "recipient_party_ids": [
        "parties:rowan"
      ]
    }
  ],
  "right": [
    {
      "actor_party_id": "parties:ellis",
      "effect_ids": [],
      "intervention": "promised",
      "recipient_party_ids": [
        "parties:sasha"
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
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:7f9bceea2154ef7eae950660f4c3daeb8b01baaf699f17d5728d8d14fdfc5dab",
      "target_id": "effect:3ac83fd89f524d292b25a18398ddd2daaa55d97c57ea732909da301d7b23cb2f"
    },
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:ef73c1db80739a46f44d548d68a0365176a409358ade9ae7923d9237089ea406",
      "target_id": "effect:dae7cf092009ee42c2a59ea1033ec612dc776d60a57e18802a520e7d218681f8"
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:379ea201d1cd731652aa153a6956399a5dbcde4fb80fc9a9c06e96d026cec2d6",
      "target_id": "effect:c9d291c3600075cbae03db8d6062798a705f40b44e8a4d753d219123a21d0f1c"
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:f592754297e544973538b723c2419e96d2f7056a1fbdad74712cf5aa7481edc7",
      "target_id": "effect:e0ed6d83c830b939341d0aeb5035a8a4a6cb86a97fe77ba2bdbf8350418f6ba4"
    }
  ],
  "right": []
}
```

## difference_2: candidate_0 / candidate_1 — commitments

Which commitments reading is supported by the source?

```json
{
  "left": [
    {
      "breach": "",
      "commitment_event": "promised",
      "commitment_id": "k0",
      "content_proposition_id": "propositions:would get the blanket",
      "promisee_party_id": "parties:sasha",
      "promisor_party_id": "parties:ellis",
      "reliance": ""
    }
  ],
  "right": [
    {
      "breach": "",
      "commitment_event": "promised",
      "commitment_id": "k0",
      "content_proposition_id": "propositions:sasha would get the blanket",
      "promisee_party_id": "parties:sasha",
      "promisor_party_id": "parties:ellis",
      "reliance": ""
    }
  ]
}
```

## difference_3: candidate_0 / candidate_1 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "resource_transfer",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "gives the blanket to rowan",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "gives the blanket to rowan",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "exclusive_allocation_complement",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not receive one thermal blanket",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "adverse",
      "predicate": "not_receives",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:7f9bceea2154ef7eae950660f4c3daeb8b01baaf699f17d5728d8d14fdfc5dab"
      ],
      "source_proposition": "a mountain rescue shelter has one thermal blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [
        "80% chance"
      ],
      "modality": "probabilistic",
      "outcome": "rowan has an 80% chance of survival",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "beneficial",
      "predicate": "survive",
      "quantities": [],
      "scope_qualifiers": [
        "unlicensed_atom:no_outcome_span"
      ],
      "source_effect_ids": [
        "effect:7f9bceea2154ef7eae950660f4c3daeb8b01baaf699f17d5728d8d14fdfc5dab"
      ],
      "source_proposition": "rowan has an 80% chance of survival",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not get the blanket",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [
        "unlicensed_atom:no_outcome_span"
      ],
      "source_effect_ids": [
        "effect:ef73c1db80739a46f44d548d68a0365176a409358ade9ae7923d9237089ea406"
      ],
      "source_proposition": "does not get the blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "resource_transfer",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "gives the blanket to sasha",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "gives the blanket to sasha",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "exclusive_allocation_complement",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not receive one thermal blanket",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "adverse",
      "predicate": "not_receives",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:379ea201d1cd731652aa153a6956399a5dbcde4fb80fc9a9c06e96d026cec2d6"
      ],
      "source_proposition": "a mountain rescue shelter has one thermal blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [
        "60% chance"
      ],
      "modality": "probabilistic",
      "outcome": "sasha has a 60% chance of survival",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "beneficial",
      "predicate": "survive",
      "quantities": [],
      "scope_qualifiers": [
        "unlicensed_atom:no_outcome_span"
      ],
      "source_effect_ids": [
        "effect:379ea201d1cd731652aa153a6956399a5dbcde4fb80fc9a9c06e96d026cec2d6"
      ],
      "source_proposition": "sasha has a 60% chance of survival",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not get the blanket",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [
        "unlicensed_atom:no_outcome_span"
      ],
      "source_effect_ids": [
        "effect:f592754297e544973538b723c2419e96d2f7056a1fbdad74712cf5aa7481edc7"
      ],
      "source_proposition": "does not get the blanket",
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
      "kind": "person",
      "label": "rescuer ellis",
      "quantities": []
    },
    {
      "kind": "person",
      "label": "rowan",
      "quantities": []
    },
    {
      "kind": "resource",
      "label": "one thermal blanket",
      "quantities": [
        "one"
      ]
    }
  ],
  "right": []
}
```

## difference_5: candidate_0 / candidate_1 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [
    {
      "polarity": "positive",
      "predication": "would get the blanket",
      "status": "committed_content"
    }
  ],
  "right": [
    {
      "polarity": "positive",
      "predication": "sasha would get the blanket",
      "status": "committed_content"
    }
  ]
}
```

## difference_6: candidate_0 / candidate_2 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:rescuer ellis",
      "effect_ids": [
        "effect:379ea201d1cd731652aa153a6956399a5dbcde4fb80fc9a9c06e96d026cec2d6",
        "effect:c9d291c3600075cbae03db8d6062798a705f40b44e8a4d753d219123a21d0f1c",
        "effect:e0ed6d83c830b939341d0aeb5035a8a4a6cb86a97fe77ba2bdbf8350418f6ba4",
        "effect:f592754297e544973538b723c2419e96d2f7056a1fbdad74712cf5aa7481edc7"
      ],
      "intervention": "gives the blanket to sasha",
      "recipient_party_ids": [
        "parties:sasha"
      ]
    },
    {
      "actor_party_id": "parties:rescuer ellis",
      "effect_ids": [
        "effect:3ac83fd89f524d292b25a18398ddd2daaa55d97c57ea732909da301d7b23cb2f",
        "effect:7f9bceea2154ef7eae950660f4c3daeb8b01baaf699f17d5728d8d14fdfc5dab",
        "effect:dae7cf092009ee42c2a59ea1033ec612dc776d60a57e18802a520e7d218681f8",
        "effect:ef73c1db80739a46f44d548d68a0365176a409358ade9ae7923d9237089ea406"
      ],
      "intervention": "gives the blanket to rowan",
      "recipient_party_ids": [
        "parties:rowan"
      ]
    }
  ],
  "right": [
    {
      "actor_party_id": "parties:ellis",
      "effect_ids": [
        "effect:2f7aba62fbfc6b2d5acae93960ca1b1feb210d78daea806b3775564d847881b7"
      ],
      "intervention": "gives the blanket to sasha",
      "recipient_party_ids": [
        "parties:a 60% chance"
      ]
    },
    {
      "actor_party_id": "parties:ellis",
      "effect_ids": [
        "effect:98ac2f237fc2a35faea0e14acd14862215b454ccd319ce120b6b12549c864a82"
      ],
      "intervention": "gives the blanket to rowan",
      "recipient_party_ids": [
        "parties:an 80% chance"
      ]
    }
  ]
}
```

## difference_7: candidate_0 / candidate_2 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:7f9bceea2154ef7eae950660f4c3daeb8b01baaf699f17d5728d8d14fdfc5dab",
      "target_id": "effect:3ac83fd89f524d292b25a18398ddd2daaa55d97c57ea732909da301d7b23cb2f"
    },
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:ef73c1db80739a46f44d548d68a0365176a409358ade9ae7923d9237089ea406",
      "target_id": "effect:dae7cf092009ee42c2a59ea1033ec612dc776d60a57e18802a520e7d218681f8"
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:379ea201d1cd731652aa153a6956399a5dbcde4fb80fc9a9c06e96d026cec2d6",
      "target_id": "effect:c9d291c3600075cbae03db8d6062798a705f40b44e8a4d753d219123a21d0f1c"
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:f592754297e544973538b723c2419e96d2f7056a1fbdad74712cf5aa7481edc7",
      "target_id": "effect:e0ed6d83c830b939341d0aeb5035a8a4a6cb86a97fe77ba2bdbf8350418f6ba4"
    }
  ],
  "right": []
}
```

## difference_8: candidate_0 / candidate_2 — commitments

Which commitments reading is supported by the source?

```json
{
  "left": [
    {
      "breach": "",
      "commitment_event": "promised",
      "commitment_id": "k0",
      "content_proposition_id": "propositions:would get the blanket",
      "promisee_party_id": "parties:sasha",
      "promisor_party_id": "parties:ellis",
      "reliance": ""
    }
  ],
  "right": []
}
```

## difference_9: candidate_0 / candidate_2 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "decision_relevance": "material",
      "description": "if ellis gives the blanket to rowan",
      "operator": "if",
      "polarity": "positive",
      "value_status": "unknown"
    },
    {
      "decision_relevance": "material",
      "description": "if ellis gives the blanket to sasha",
      "operator": "if",
      "polarity": "positive",
      "value_status": "unknown"
    }
  ]
}
```

## difference_10: candidate_0 / candidate_2 — effects

Which effects reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "resource_transfer",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "gives the blanket to rowan",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "gives the blanket to rowan",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "exclusive_allocation_complement",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not receive one thermal blanket",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "adverse",
      "predicate": "not_receives",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:7f9bceea2154ef7eae950660f4c3daeb8b01baaf699f17d5728d8d14fdfc5dab"
      ],
      "source_proposition": "a mountain rescue shelter has one thermal blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [
        "80% chance"
      ],
      "modality": "probabilistic",
      "outcome": "rowan has an 80% chance of survival",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "beneficial",
      "predicate": "survive",
      "quantities": [],
      "scope_qualifiers": [
        "unlicensed_atom:no_outcome_span"
      ],
      "source_effect_ids": [
        "effect:7f9bceea2154ef7eae950660f4c3daeb8b01baaf699f17d5728d8d14fdfc5dab"
      ],
      "source_proposition": "rowan has an 80% chance of survival",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not get the blanket",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [
        "unlicensed_atom:no_outcome_span"
      ],
      "source_effect_ids": [
        "effect:ef73c1db80739a46f44d548d68a0365176a409358ade9ae7923d9237089ea406"
      ],
      "source_proposition": "does not get the blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "resource_transfer",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "gives the blanket to sasha",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "gives the blanket to sasha",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "exclusive_allocation_complement",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not receive one thermal blanket",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "adverse",
      "predicate": "not_receives",
      "quantities": [
        "one"
      ],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:379ea201d1cd731652aa153a6956399a5dbcde4fb80fc9a9c06e96d026cec2d6"
      ],
      "source_proposition": "a mountain rescue shelter has one thermal blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [
        "60% chance"
      ],
      "modality": "probabilistic",
      "outcome": "sasha has a 60% chance of survival",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "beneficial",
      "predicate": "survive",
      "quantities": [],
      "scope_qualifiers": [
        "unlicensed_atom:no_outcome_span"
      ],
      "source_effect_ids": [
        "effect:379ea201d1cd731652aa153a6956399a5dbcde4fb80fc9a9c06e96d026cec2d6"
      ],
      "source_proposition": "sasha has a 60% chance of survival",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "other",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "does not get the blanket",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [
        "unlicensed_atom:no_outcome_span"
      ],
      "source_effect_ids": [
        "effect:f592754297e544973538b723c2419e96d2f7056a1fbdad74712cf5aa7481edc7"
      ],
      "source_proposition": "does not get the blanket",
      "temporal_qualifiers": []
    }
  ],
  "right": [
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "gives the blanket to rowan",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:an 80% chance",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "gives the blanket to rowan",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "gives the blanket to sasha",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:a 60% chance",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "gives the blanket to sasha",
      "temporal_qualifiers": []
    }
  ]
}
```

## difference_11: candidate_0 / candidate_2 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "person",
      "label": "rescuer ellis",
      "quantities": []
    },
    {
      "kind": "person",
      "label": "rowan",
      "quantities": []
    },
    {
      "kind": "person",
      "label": "sasha",
      "quantities": []
    },
    {
      "kind": "resource",
      "label": "one thermal blanket",
      "quantities": [
        "one"
      ]
    }
  ],
  "right": [
    {
      "kind": "person",
      "label": "a 60% chance",
      "quantities": []
    },
    {
      "kind": "person",
      "label": "an 80% chance",
      "quantities": []
    }
  ]
}
```

## difference_12: candidate_0 / candidate_2 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [
    {
      "polarity": "positive",
      "predication": "would get the blanket",
      "status": "committed_content"
    }
  ],
  "right": []
}
```

## difference_13: candidate_1 / candidate_2 — actions

Which actions reading is supported by the source?

```json
{
  "left": [
    {
      "actor_party_id": "parties:ellis",
      "effect_ids": [],
      "intervention": "promised",
      "recipient_party_ids": [
        "parties:sasha"
      ]
    }
  ],
  "right": [
    {
      "actor_party_id": "parties:ellis",
      "effect_ids": [
        "effect:2f7aba62fbfc6b2d5acae93960ca1b1feb210d78daea806b3775564d847881b7"
      ],
      "intervention": "gives the blanket to sasha",
      "recipient_party_ids": [
        "parties:a 60% chance"
      ]
    },
    {
      "actor_party_id": "parties:ellis",
      "effect_ids": [
        "effect:98ac2f237fc2a35faea0e14acd14862215b454ccd319ce120b6b12549c864a82"
      ],
      "intervention": "gives the blanket to rowan",
      "recipient_party_ids": [
        "parties:an 80% chance"
      ]
    }
  ]
}
```

## difference_14: candidate_1 / candidate_2 — commitments

Which commitments reading is supported by the source?

```json
{
  "left": [
    {
      "breach": "",
      "commitment_event": "promised",
      "commitment_id": "k0",
      "content_proposition_id": "propositions:sasha would get the blanket",
      "promisee_party_id": "parties:sasha",
      "promisor_party_id": "parties:ellis",
      "reliance": ""
    }
  ],
  "right": []
}
```

## difference_15: candidate_1 / candidate_2 — conditions

Which conditions reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "decision_relevance": "material",
      "description": "if ellis gives the blanket to rowan",
      "operator": "if",
      "polarity": "positive",
      "value_status": "unknown"
    },
    {
      "decision_relevance": "material",
      "description": "if ellis gives the blanket to sasha",
      "operator": "if",
      "polarity": "positive",
      "value_status": "unknown"
    }
  ]
}
```

## difference_16: candidate_1 / candidate_2 — effects

Which effects reading is supported by the source?

```json
{
  "left": [],
  "right": [
    {
      "action_id": "actions:gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "gives the blanket to rowan",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:an 80% chance",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "gives the blanket to rowan",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "intervention",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "gives the blanket to sasha",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:a 60% chance",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "gives the blanket to sasha",
      "temporal_qualifiers": []
    }
  ]
}
```

## difference_17: candidate_1 / candidate_2 — parties

Which parties reading is supported by the source?

```json
{
  "left": [
    {
      "kind": "person",
      "label": "sasha",
      "quantities": []
    }
  ],
  "right": [
    {
      "kind": "person",
      "label": "a 60% chance",
      "quantities": []
    },
    {
      "kind": "person",
      "label": "an 80% chance",
      "quantities": []
    }
  ]
}
```

## difference_18: candidate_1 / candidate_2 — propositions

Which propositions reading is supported by the source?

```json
{
  "left": [
    {
      "polarity": "positive",
      "predication": "sasha would get the blanket",
      "status": "committed_content"
    }
  ],
  "right": []
}
```

