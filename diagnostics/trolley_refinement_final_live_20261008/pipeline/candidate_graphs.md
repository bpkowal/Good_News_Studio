# Candidate world attempts

Chosen blueprint: **omission_harm**  
Chooser status: **FILLED**

## 1. rescue_contrast (alternative; baseline)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `first_saved`, `second_saved`, `rescue_exclusivity`, `first_rescue_action`, `second_rescue_action`, `scene`

World withheld:

- Required slots remain unfilled: first_saved, second_saved, rescue_exclusivity, first_rescue_action, second_rescue_action

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **63**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c0`: Which construction should preserve this PREDICATION reading?
- `c1`: Which construction should preserve this PREDICATION reading?
- `c2`: Which construction should preserve this PREDICATION reading?
- `c3`: Which construction should preserve this PREDICATION reading?
- `c6`: Which construction should preserve this PREDICATION reading?
- `c9`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this PREDICATION reading?
- `c11`: Which construction should preserve this PREDICATION reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this MODALITY reading?
- `c16`: Which construction should preserve this MODALITY reading?
- `c17`: Which construction should preserve this MODALITY reading?
- `c18`: Which construction should preserve this MODALITY reading?
- `c19`: Which construction should preserve this MODALITY reading?
- `c20`: Which construction should preserve this MODALITY reading?
- `c21`: Which construction should preserve this MODALITY reading?
- `c22`: Which construction should preserve this MODALITY reading?
- `c23`: Which construction should preserve this MODALITY reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c26`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c29`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this PARTICIPANT reading?
- `c33`: Which construction should preserve this PARTICIPANT reading?
- `c34`: Which construction should preserve this PARTICIPANT reading?
- `c36`: Which construction should preserve this PARTICIPANT reading?
- `c37`: Which construction should preserve this PARTICIPANT reading?
- `c38`: Which construction should preserve this PARTICIPANT reading?
- `c39`: Which construction should preserve this PARTICIPANT reading?
- `c40`: Which construction should preserve this PARTICIPANT reading?
- `c41`: Which construction should preserve this PARTICIPANT reading?
- `c42`: Which construction should preserve this PARTICIPANT reading?
- `c43`: Which construction should preserve this QUANTITY reading?
- `c44`: Which construction should preserve this QUANTITY reading?
- `c45`: Which construction should preserve this SAME_REFERENT reading?
- `c46`: Which construction should preserve this SAME_REFERENT reading?
- `c47`: Which construction should preserve this SAME_REFERENT reading?
- `c48`: Which construction should preserve this SAME_REFERENT reading?
- `c49`: Which construction should preserve this SAME_REFERENT reading?
- `c50`: Which construction should preserve this SAME_REFERENT reading?
- `c51`: Which construction should preserve this SAME_REFERENT reading?
- `c52`: Which construction should preserve this SAME_REFERENT reading?
- `c53`: Which construction should preserve this SAME_REFERENT reading?
- `c54`: Which construction should preserve this SAME_REFERENT reading?
- `c55`: Which construction should preserve this SAME_REFERENT reading?
- `c56`: Which construction should preserve this SAME_REFERENT reading?
- `c57`: Which construction should preserve this SAME_REFERENT reading?
- `c58`: Which construction should preserve this SAME_REFERENT reading?
- `c59`: Which construction should preserve this SAME_REFERENT reading?
- `c60`: Which construction should preserve this SAME_REFERENT reading?
- `c61`: Which construction should preserve this SAME_REFERENT reading?
- `c62`: Which construction should preserve this SAME_REFERENT reading?
- `c63`: Which construction should preserve this SAME_REFERENT reading?
- `c64`: Which construction should preserve this SAME_REFERENT reading?
- `c65`: Which construction should preserve this SAME_REFERENT reading?
- `c66`: Which construction should preserve this SAME_REFERENT reading?
- `c67`: Which construction should preserve this SAME_REFERENT reading?
- `c68`: Which construction should preserve this SAME_REFERENT reading?
- `c69`: Which construction should preserve this SAME_REFERENT reading?
- `c70`: Which construction should preserve this SAME_REFERENT reading?
- `c71`: Which construction should preserve this SAME_REFERENT reading?

## 2. omission_harm (selected; baseline)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `instrument`

### omission_harm candidate graph

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — one worker<br/>PERSON"]
    P4["P4 — five workers<br/>HUMAN_GROUP"]
    P6["P6 — a lever<br/>PROCESS"]
    P7["P7 — the trolley<br/>PROCESS"]
    P8["P8 — the main track<br/>OTHER"]
    P9["P9 — a side track<br/>OTHER"]
    A0["A0 — pulls the lever"]
    A1["A1 — does not pull the lever"]
    E1["E1 — pulls the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — switches the trolley onto a side track<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — the trolley will kill one worker<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E4["E4 — spare five workers<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E5["E5 — does not<br/>DIRECT · NEUTRAL · CERTAIN"]
    E6["E6 — it will kill the five<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E7["E7 — it will kill five workers<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E8["E8 — spare one worker<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    CND1{"CND1 — If she pulls the lever<br/>POSITIVE"}
    CND2{"CND2 — If she does not<br/>NEGATED"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P6
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    A0 -.->|"effect"| E4
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P6
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    A1 -.->|"effect"| E7
    A1 -.->|"effect"| E8
    E1 -.->|"affects"| P6
    E2 -.->|"affects"| P7
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P4
    E5 -.->|"affects"| P6
    E6 -.->|"affects"| P7
    E7 -.->|"affects"| P4
    E8 -.->|"affects"| P2
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E2 -->|"ENABLES"| E4
    E5 -->|"ENABLES"| E6
    E6 -->|"ENABLES"| E7
    E6 -->|"ENABLES"| E8
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
    PR1([" PR1 — A runaway trolley is heading toward five workers on the main track.<br/>proposition · ASSERTED "])
    PR2([" PR2 — Maria can pull a lever that switches the trolley onto a side track, where one worker stands.<br/>proposition · ASSERTED "])
    PR3([" PR3 — Maria cannot stop the trolley.<br/>proposition · ASSERTED "])
    P4 -.->|"MENTIONED_IN"| PR1
    P8 -.->|"MENTIONED_IN"| PR1
    P1 -.->|"MENTIONED_IN"| PR2
    P2 -.->|"MENTIONED_IN"| PR2
    P6 -.->|"MENTIONED_IN"| PR2
    P7 -.->|"MENTIONED_IN"| PR2
    P9 -.->|"MENTIONED_IN"| PR2
    P1 -.->|"MENTIONED_IN"| PR3
    P7 -.->|"MENTIONED_IN"| PR3
```

Native admission:

```json
{
  "blueprint_id": "omission_harm",
  "rank": 1,
  "variant": "baseline",
  "candidate_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/trolley_refinement_final_live_20261008/pipeline/admission_attempts/01_omission_harm/candidate.json",
  "overlay_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/trolley_refinement_final_live_20261008/pipeline/admission_attempts/01_omission_harm/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 8,
    "causal_links": 6,
    "frozen_trace_valid": true,
    "discourse_committed": false,
    "relent_workspace": {
      "established": [
        {
          "kind": "occurrence_effect",
          "effect_id": "E1",
          "outcome": "pulls the lever",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E2",
          "outcome": "switches the trolley onto a side track",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E3",
          "outcome": "the trolley will kill one worker",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E4",
          "outcome": "spare five workers",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E5",
          "outcome": "does not",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E6",
          "outcome": "it will kill the five",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E7",
          "outcome": "it will kill five workers",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E8",
          "outcome": "spare one worker",
          "predicate": null,
          "status": "established"
        }
      ],
      "not_established": [],
      "duty_ledger_citations": [],
      "ev_forbidden_proposition_ids": [],
      "factual_status_sources": [
        "occurrence_effects"
      ]
    },
    "fingerprint_sha256": "1385b6f34087b29e291f691b7714f90bc63bbf9123b63568adae2fa49ae65107"
  },
  "trace_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/trolley_refinement_final_live_20261008/pipeline/admission_attempts/01_omission_harm/frozen_world_trace.json"
}
```

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **63**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c0`: Which construction should preserve this PREDICATION reading?
- `c1`: Which construction should preserve this PREDICATION reading?
- `c2`: Which construction should preserve this PREDICATION reading?
- `c3`: Which construction should preserve this PREDICATION reading?
- `c6`: Which construction should preserve this PREDICATION reading?
- `c9`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this PREDICATION reading?
- `c11`: Which construction should preserve this PREDICATION reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this MODALITY reading?
- `c16`: Which construction should preserve this MODALITY reading?
- `c17`: Which construction should preserve this MODALITY reading?
- `c18`: Which construction should preserve this MODALITY reading?
- `c19`: Which construction should preserve this MODALITY reading?
- `c20`: Which construction should preserve this MODALITY reading?
- `c21`: Which construction should preserve this MODALITY reading?
- `c22`: Which construction should preserve this MODALITY reading?
- `c23`: Which construction should preserve this MODALITY reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c26`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c29`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this PARTICIPANT reading?
- `c33`: Which construction should preserve this PARTICIPANT reading?
- `c34`: Which construction should preserve this PARTICIPANT reading?
- `c36`: Which construction should preserve this PARTICIPANT reading?
- `c37`: Which construction should preserve this PARTICIPANT reading?
- `c38`: Which construction should preserve this PARTICIPANT reading?
- `c39`: Which construction should preserve this PARTICIPANT reading?
- `c40`: Which construction should preserve this PARTICIPANT reading?
- `c41`: Which construction should preserve this PARTICIPANT reading?
- `c42`: Which construction should preserve this PARTICIPANT reading?
- `c43`: Which construction should preserve this QUANTITY reading?
- `c44`: Which construction should preserve this QUANTITY reading?
- `c45`: Which construction should preserve this SAME_REFERENT reading?
- `c46`: Which construction should preserve this SAME_REFERENT reading?
- `c47`: Which construction should preserve this SAME_REFERENT reading?
- `c48`: Which construction should preserve this SAME_REFERENT reading?
- `c49`: Which construction should preserve this SAME_REFERENT reading?
- `c50`: Which construction should preserve this SAME_REFERENT reading?
- `c51`: Which construction should preserve this SAME_REFERENT reading?
- `c52`: Which construction should preserve this SAME_REFERENT reading?
- `c53`: Which construction should preserve this SAME_REFERENT reading?
- `c54`: Which construction should preserve this SAME_REFERENT reading?
- `c55`: Which construction should preserve this SAME_REFERENT reading?
- `c56`: Which construction should preserve this SAME_REFERENT reading?
- `c57`: Which construction should preserve this SAME_REFERENT reading?
- `c58`: Which construction should preserve this SAME_REFERENT reading?
- `c59`: Which construction should preserve this SAME_REFERENT reading?
- `c60`: Which construction should preserve this SAME_REFERENT reading?
- `c61`: Which construction should preserve this SAME_REFERENT reading?
- `c62`: Which construction should preserve this SAME_REFERENT reading?
- `c63`: Which construction should preserve this SAME_REFERENT reading?
- `c64`: Which construction should preserve this SAME_REFERENT reading?
- `c65`: Which construction should preserve this SAME_REFERENT reading?
- `c66`: Which construction should preserve this SAME_REFERENT reading?
- `c67`: Which construction should preserve this SAME_REFERENT reading?
- `c68`: Which construction should preserve this SAME_REFERENT reading?
- `c69`: Which construction should preserve this SAME_REFERENT reading?
- `c70`: Which construction should preserve this SAME_REFERENT reading?
- `c71`: Which construction should preserve this SAME_REFERENT reading?

## 3. conditional_outcome (alternative; baseline)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `chance`

### conditional_outcome candidate graph

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — one worker<br/>PERSON"]
    P3["P3 — the trolley<br/>PROCESS"]
    P4["P4 — five workers<br/>HUMAN_GROUP"]
    P6["P6 — a lever<br/>PROCESS"]
    P7["P7 — the main track<br/>OTHER"]
    P8["P8 — a side track<br/>OTHER"]
    A0["A0 — pulls the lever"]
    A1["A1 — does not pull the lever"]
    E1["E1 — pulls the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — switches the trolley onto a side track<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — the trolley will kill one worker<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E4["E4 — spare five workers<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E5["E5 — does not<br/>DIRECT · NEUTRAL · CERTAIN"]
    E6["E6 — it will kill the five<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E7["E7 — it will kill five workers<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E8["E8 — spare one worker<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    CND1{"CND1 — If she pulls the lever<br/>POSITIVE"}
    CND2{"CND2 — If she does not<br/>NEGATED"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P6
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    A0 -.->|"effect"| E4
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P6
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    A1 -.->|"effect"| E7
    A1 -.->|"effect"| E8
    E1 -.->|"affects"| P6
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P4
    E5 -.->|"affects"| P6
    E6 -.->|"affects"| P3
    E7 -.->|"affects"| P4
    E8 -.->|"affects"| P2
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E2 -->|"ENABLES"| E4
    E5 -->|"ENABLES"| E6
    E6 -->|"ENABLES"| E7
    E6 -->|"ENABLES"| E8
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
    PR1([" PR1 — A runaway trolley is heading toward five workers on the main track.<br/>proposition · ASSERTED "])
    PR2([" PR2 — Maria can pull a lever that switches the trolley onto a side track, where one worker stands.<br/>proposition · ASSERTED "])
    PR3([" PR3 — Maria cannot stop the trolley.<br/>proposition · ASSERTED "])
    P4 -.->|"MENTIONED_IN"| PR1
    P7 -.->|"MENTIONED_IN"| PR1
    P1 -.->|"MENTIONED_IN"| PR2
    P2 -.->|"MENTIONED_IN"| PR2
    P3 -.->|"MENTIONED_IN"| PR2
    P6 -.->|"MENTIONED_IN"| PR2
    P8 -.->|"MENTIONED_IN"| PR2
    P1 -.->|"MENTIONED_IN"| PR3
    P3 -.->|"MENTIONED_IN"| PR3
```

Native admission:

```json
{
  "blueprint_id": "conditional_outcome",
  "rank": 2,
  "variant": "baseline",
  "candidate_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/trolley_refinement_final_live_20261008/pipeline/admission_attempts/02_conditional_outcome/candidate.json",
  "overlay_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/trolley_refinement_final_live_20261008/pipeline/admission_attempts/02_conditional_outcome/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 8,
    "causal_links": 6,
    "frozen_trace_valid": true,
    "discourse_committed": false,
    "relent_workspace": {
      "established": [
        {
          "kind": "occurrence_effect",
          "effect_id": "E1",
          "outcome": "pulls the lever",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E2",
          "outcome": "switches the trolley onto a side track",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E3",
          "outcome": "the trolley will kill one worker",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E4",
          "outcome": "spare five workers",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E5",
          "outcome": "does not",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E6",
          "outcome": "it will kill the five",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E7",
          "outcome": "it will kill five workers",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E8",
          "outcome": "spare one worker",
          "predicate": null,
          "status": "established"
        }
      ],
      "not_established": [],
      "duty_ledger_citations": [],
      "ev_forbidden_proposition_ids": [],
      "factual_status_sources": [
        "occurrence_effects"
      ]
    },
    "fingerprint_sha256": "d79c072481ffb9bb2cd32b9dab8f21c7b4ef2e0183044e5b4490b3b16d39d8f7"
  },
  "trace_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/trolley_refinement_final_live_20261008/pipeline/admission_attempts/02_conditional_outcome/frozen_world_trace.json"
}
```

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **63**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c0`: Which construction should preserve this PREDICATION reading?
- `c1`: Which construction should preserve this PREDICATION reading?
- `c2`: Which construction should preserve this PREDICATION reading?
- `c3`: Which construction should preserve this PREDICATION reading?
- `c6`: Which construction should preserve this PREDICATION reading?
- `c9`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this PREDICATION reading?
- `c11`: Which construction should preserve this PREDICATION reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this MODALITY reading?
- `c16`: Which construction should preserve this MODALITY reading?
- `c17`: Which construction should preserve this MODALITY reading?
- `c18`: Which construction should preserve this MODALITY reading?
- `c19`: Which construction should preserve this MODALITY reading?
- `c20`: Which construction should preserve this MODALITY reading?
- `c21`: Which construction should preserve this MODALITY reading?
- `c22`: Which construction should preserve this MODALITY reading?
- `c23`: Which construction should preserve this MODALITY reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c26`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c29`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this PARTICIPANT reading?
- `c33`: Which construction should preserve this PARTICIPANT reading?
- `c34`: Which construction should preserve this PARTICIPANT reading?
- `c36`: Which construction should preserve this PARTICIPANT reading?
- `c37`: Which construction should preserve this PARTICIPANT reading?
- `c38`: Which construction should preserve this PARTICIPANT reading?
- `c39`: Which construction should preserve this PARTICIPANT reading?
- `c40`: Which construction should preserve this PARTICIPANT reading?
- `c41`: Which construction should preserve this PARTICIPANT reading?
- `c42`: Which construction should preserve this PARTICIPANT reading?
- `c43`: Which construction should preserve this QUANTITY reading?
- `c44`: Which construction should preserve this QUANTITY reading?
- `c45`: Which construction should preserve this SAME_REFERENT reading?
- `c46`: Which construction should preserve this SAME_REFERENT reading?
- `c47`: Which construction should preserve this SAME_REFERENT reading?
- `c48`: Which construction should preserve this SAME_REFERENT reading?
- `c49`: Which construction should preserve this SAME_REFERENT reading?
- `c50`: Which construction should preserve this SAME_REFERENT reading?
- `c51`: Which construction should preserve this SAME_REFERENT reading?
- `c52`: Which construction should preserve this SAME_REFERENT reading?
- `c53`: Which construction should preserve this SAME_REFERENT reading?
- `c54`: Which construction should preserve this SAME_REFERENT reading?
- `c55`: Which construction should preserve this SAME_REFERENT reading?
- `c56`: Which construction should preserve this SAME_REFERENT reading?
- `c57`: Which construction should preserve this SAME_REFERENT reading?
- `c58`: Which construction should preserve this SAME_REFERENT reading?
- `c59`: Which construction should preserve this SAME_REFERENT reading?
- `c60`: Which construction should preserve this SAME_REFERENT reading?
- `c61`: Which construction should preserve this SAME_REFERENT reading?
- `c62`: Which construction should preserve this SAME_REFERENT reading?
- `c63`: Which construction should preserve this SAME_REFERENT reading?
- `c64`: Which construction should preserve this SAME_REFERENT reading?
- `c65`: Which construction should preserve this SAME_REFERENT reading?
- `c66`: Which construction should preserve this SAME_REFERENT reading?
- `c67`: Which construction should preserve this SAME_REFERENT reading?
- `c68`: Which construction should preserve this SAME_REFERENT reading?
- `c69`: Which construction should preserve this SAME_REFERENT reading?
- `c70`: Which construction should preserve this SAME_REFERENT reading?
- `c71`: Which construction should preserve this SAME_REFERENT reading?

## 4. primitive_composition (alternative; composition)

Template status: **WITHHELD**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: none

World withheld:

- No supported conditional branch was extracted.

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **63**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c0`: Which construction should preserve this PREDICATION reading?
- `c1`: Which construction should preserve this PREDICATION reading?
- `c2`: Which construction should preserve this PREDICATION reading?
- `c3`: Which construction should preserve this PREDICATION reading?
- `c6`: Which construction should preserve this PREDICATION reading?
- `c9`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this PREDICATION reading?
- `c11`: Which construction should preserve this PREDICATION reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this MODALITY reading?
- `c16`: Which construction should preserve this MODALITY reading?
- `c17`: Which construction should preserve this MODALITY reading?
- `c18`: Which construction should preserve this MODALITY reading?
- `c19`: Which construction should preserve this MODALITY reading?
- `c20`: Which construction should preserve this MODALITY reading?
- `c21`: Which construction should preserve this MODALITY reading?
- `c22`: Which construction should preserve this MODALITY reading?
- `c23`: Which construction should preserve this MODALITY reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c26`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c29`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this PARTICIPANT reading?
- `c33`: Which construction should preserve this PARTICIPANT reading?
- `c34`: Which construction should preserve this PARTICIPANT reading?
- `c36`: Which construction should preserve this PARTICIPANT reading?
- `c37`: Which construction should preserve this PARTICIPANT reading?
- `c38`: Which construction should preserve this PARTICIPANT reading?
- `c39`: Which construction should preserve this PARTICIPANT reading?
- `c40`: Which construction should preserve this PARTICIPANT reading?
- `c41`: Which construction should preserve this PARTICIPANT reading?
- `c42`: Which construction should preserve this PARTICIPANT reading?
- `c43`: Which construction should preserve this QUANTITY reading?
- `c44`: Which construction should preserve this QUANTITY reading?
- `c45`: Which construction should preserve this SAME_REFERENT reading?
- `c46`: Which construction should preserve this SAME_REFERENT reading?
- `c47`: Which construction should preserve this SAME_REFERENT reading?
- `c48`: Which construction should preserve this SAME_REFERENT reading?
- `c49`: Which construction should preserve this SAME_REFERENT reading?
- `c50`: Which construction should preserve this SAME_REFERENT reading?
- `c51`: Which construction should preserve this SAME_REFERENT reading?
- `c52`: Which construction should preserve this SAME_REFERENT reading?
- `c53`: Which construction should preserve this SAME_REFERENT reading?
- `c54`: Which construction should preserve this SAME_REFERENT reading?
- `c55`: Which construction should preserve this SAME_REFERENT reading?
- `c56`: Which construction should preserve this SAME_REFERENT reading?
- `c57`: Which construction should preserve this SAME_REFERENT reading?
- `c58`: Which construction should preserve this SAME_REFERENT reading?
- `c59`: Which construction should preserve this SAME_REFERENT reading?
- `c60`: Which construction should preserve this SAME_REFERENT reading?
- `c61`: Which construction should preserve this SAME_REFERENT reading?
- `c62`: Which construction should preserve this SAME_REFERENT reading?
- `c63`: Which construction should preserve this SAME_REFERENT reading?
- `c64`: Which construction should preserve this SAME_REFERENT reading?
- `c65`: Which construction should preserve this SAME_REFERENT reading?
- `c66`: Which construction should preserve this SAME_REFERENT reading?
- `c67`: Which construction should preserve this SAME_REFERENT reading?
- `c68`: Which construction should preserve this SAME_REFERENT reading?
- `c69`: Which construction should preserve this SAME_REFERENT reading?
- `c70`: Which construction should preserve this SAME_REFERENT reading?
- `c71`: Which construction should preserve this SAME_REFERENT reading?
