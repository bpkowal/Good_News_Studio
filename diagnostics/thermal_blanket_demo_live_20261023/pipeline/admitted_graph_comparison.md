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
        "effect:ad22bb5bec49f4529c59fccd39aa21506f52a8f4797e892d2d7461bd45a2d992",
        "effect:bcc43d5151f6acd2bf2e3b8b4ac9efe881086237d8acdee2f901412fa89c3efb",
        "effect:ca7dda52485644a01619364816aa91e5d709e88a0d555c70cfe6172e018f1187",
        "effect:f1d151e1ec0dc6af5f670039c107273a36c43fec5062693833bc99d9c47d2e79"
      ],
      "intervention": "ellis gives the blanket to rowan",
      "recipient_party_ids": [
        "parties:rowan"
      ]
    },
    {
      "actor_party_id": "parties:rescuer ellis",
      "effect_ids": [
        "effect:c06b40a638cc5459920bf8e5eb476ee667776db268d334eb406b8871f714225b",
        "effect:c0951114e3be69c26ab12cd0bb8c626e16ff8b3824217a4f86435a745fb802f6",
        "effect:ca6d868e38c40699ad4bdb77ca9166418f134d64b4bdf2d824632d01ee81f57c",
        "effect:d0ad9afd04d6b6ee50beee23c579dc12f367a686ac3d3c50aecc40c6d608ea84"
      ],
      "intervention": "ellis gives the blanket to sasha",
      "recipient_party_ids": [
        "parties:sasha"
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
      "action_id": "actions:ellis gives the blanket to rowan",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:bcc43d5151f6acd2bf2e3b8b4ac9efe881086237d8acdee2f901412fa89c3efb",
      "target_id": "effect:f1d151e1ec0dc6af5f670039c107273a36c43fec5062693833bc99d9c47d2e79"
    },
    {
      "action_id": "actions:ellis gives the blanket to rowan",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:ca7dda52485644a01619364816aa91e5d709e88a0d555c70cfe6172e018f1187",
      "target_id": "effect:ad22bb5bec49f4529c59fccd39aa21506f52a8f4797e892d2d7461bd45a2d992"
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:c06b40a638cc5459920bf8e5eb476ee667776db268d334eb406b8871f714225b",
      "target_id": "effect:ca6d868e38c40699ad4bdb77ca9166418f134d64b4bdf2d824632d01ee81f57c"
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:c0951114e3be69c26ab12cd0bb8c626e16ff8b3824217a4f86435a745fb802f6",
      "target_id": "effect:d0ad9afd04d6b6ee50beee23c579dc12f367a686ac3d3c50aecc40c6d608ea84"
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
      "content_proposition_id": "propositions:that sasha would get the blanket",
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
      "action_id": "actions:ellis gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "resource_transfer",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "ellis gives the blanket to rowan",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "ellis gives the blanket to rowan",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to rowan",
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
        "effect:ca7dda52485644a01619364816aa91e5d709e88a0d555c70cfe6172e018f1187"
      ],
      "source_proposition": "a mountain rescue shelter has one thermal blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:bcc43d5151f6acd2bf2e3b8b4ac9efe881086237d8acdee2f901412fa89c3efb"
      ],
      "source_proposition": "will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to rowan",
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
        "effect:ca7dda52485644a01619364816aa91e5d709e88a0d555c70cfe6172e018f1187"
      ],
      "source_proposition": "rowan has an 80% chance of survival",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "resource_transfer",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "ellis gives the blanket to sasha",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "ellis gives the blanket to sasha",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
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
        "effect:c0951114e3be69c26ab12cd0bb8c626e16ff8b3824217a4f86435a745fb802f6"
      ],
      "source_proposition": "a mountain rescue shelter has one thermal blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:c06b40a638cc5459920bf8e5eb476ee667776db268d334eb406b8871f714225b"
      ],
      "source_proposition": "will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
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
        "effect:c0951114e3be69c26ab12cd0bb8c626e16ff8b3824217a4f86435a745fb802f6"
      ],
      "source_proposition": "sasha has a 60% chance of survival",
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
      "predication": "that sasha would get the blanket",
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
        "effect:ad22bb5bec49f4529c59fccd39aa21506f52a8f4797e892d2d7461bd45a2d992",
        "effect:bcc43d5151f6acd2bf2e3b8b4ac9efe881086237d8acdee2f901412fa89c3efb",
        "effect:ca7dda52485644a01619364816aa91e5d709e88a0d555c70cfe6172e018f1187",
        "effect:f1d151e1ec0dc6af5f670039c107273a36c43fec5062693833bc99d9c47d2e79"
      ],
      "intervention": "ellis gives the blanket to rowan",
      "recipient_party_ids": [
        "parties:rowan"
      ]
    },
    {
      "actor_party_id": "parties:rescuer ellis",
      "effect_ids": [
        "effect:c06b40a638cc5459920bf8e5eb476ee667776db268d334eb406b8871f714225b",
        "effect:c0951114e3be69c26ab12cd0bb8c626e16ff8b3824217a4f86435a745fb802f6",
        "effect:ca6d868e38c40699ad4bdb77ca9166418f134d64b4bdf2d824632d01ee81f57c",
        "effect:d0ad9afd04d6b6ee50beee23c579dc12f367a686ac3d3c50aecc40c6d608ea84"
      ],
      "intervention": "ellis gives the blanket to sasha",
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

## difference_7: candidate_0 / candidate_2 — causal_links

Which causal_links reading is supported by the source?

```json
{
  "left": [
    {
      "action_id": "actions:ellis gives the blanket to rowan",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:bcc43d5151f6acd2bf2e3b8b4ac9efe881086237d8acdee2f901412fa89c3efb",
      "target_id": "effect:f1d151e1ec0dc6af5f670039c107273a36c43fec5062693833bc99d9c47d2e79"
    },
    {
      "action_id": "actions:ellis gives the blanket to rowan",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:ca7dda52485644a01619364816aa91e5d709e88a0d555c70cfe6172e018f1187",
      "target_id": "effect:ad22bb5bec49f4529c59fccd39aa21506f52a8f4797e892d2d7461bd45a2d992"
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:c06b40a638cc5459920bf8e5eb476ee667776db268d334eb406b8871f714225b",
      "target_id": "effect:ca6d868e38c40699ad4bdb77ca9166418f134d64b4bdf2d824632d01ee81f57c"
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
      "condition_ids": [],
      "link_relation": "enables",
      "modality": "certain",
      "source_id": "effect:c0951114e3be69c26ab12cd0bb8c626e16ff8b3824217a4f86435a745fb802f6",
      "target_id": "effect:d0ad9afd04d6b6ee50beee23c579dc12f367a686ac3d3c50aecc40c6d608ea84"
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
      "action_id": "actions:ellis gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "resource_transfer",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "ellis gives the blanket to rowan",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "ellis gives the blanket to rowan",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to rowan",
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
        "effect:ca7dda52485644a01619364816aa91e5d709e88a0d555c70cfe6172e018f1187"
      ],
      "source_proposition": "a mountain rescue shelter has one thermal blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to rowan",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:bcc43d5151f6acd2bf2e3b8b4ac9efe881086237d8acdee2f901412fa89c3efb"
      ],
      "source_proposition": "will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to rowan",
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
        "effect:ca7dda52485644a01619364816aa91e5d709e88a0d555c70cfe6172e018f1187"
      ],
      "source_proposition": "rowan has an 80% chance of survival",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "direct_copy",
      "directness": "direct",
      "effect_kind": "resource_transfer",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "ellis gives the blanket to sasha",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:sasha",
      "polarity": "neutral",
      "predicate": "give",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [],
      "source_proposition": "ellis gives the blanket to sasha",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
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
        "effect:c0951114e3be69c26ab12cd0bb8c626e16ff8b3824217a4f86435a745fb802f6"
      ],
      "source_proposition": "a mountain rescue shelter has one thermal blanket",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
      "condition_ids": [],
      "condition_join": "and",
      "derivation_assumptions": [],
      "derivation_operation": "source_stipulated_causal",
      "directness": "downstream",
      "effect_kind": "health_outcome",
      "likelihood_qualifiers": [],
      "modality": "certain",
      "outcome": "will die",
      "outcome_type_transformation": "preserved",
      "overall_likelihood_qualifiers": [],
      "party_id": "parties:rowan",
      "polarity": "adverse",
      "predicate": "die",
      "quantities": [],
      "scope_qualifiers": [],
      "source_effect_ids": [
        "effect:c06b40a638cc5459920bf8e5eb476ee667776db268d334eb406b8871f714225b"
      ],
      "source_proposition": "will die",
      "temporal_qualifiers": []
    },
    {
      "action_id": "actions:ellis gives the blanket to sasha",
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
        "effect:c0951114e3be69c26ab12cd0bb8c626e16ff8b3824217a4f86435a745fb802f6"
      ],
      "source_proposition": "sasha has a 60% chance of survival",
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
      "content_proposition_id": "propositions:that sasha would get the blanket",
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
      "predication": "that sasha would get the blanket",
      "status": "committed_content"
    }
  ],
  "right": []
}
```

