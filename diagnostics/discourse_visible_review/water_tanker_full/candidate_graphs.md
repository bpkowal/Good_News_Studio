# Candidate world attempts

Chosen blueprint: **exclusive_allocation**  
Chooser status: **FILLED**

## 1. exclusive_allocation (alternative; amended)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `first_transfer`, `second_transfer`, `quantity`, `first_hedge`, `second_hedge`, `survival_chance`

### exclusive_allocation candidate graph

```mermaid
flowchart LR
    P1["P1 — An AI bot<br/>OTHER"]
    P2["P2 — the only water tanker<br/>RESOURCE"]
    P3["P3 — a child<br/>PERSON"]
    P4["P4 — five elderly patients<br/>HUMAN_GROUP"]
    P5["P5 — the child<br/>PERSON"]
    P6["P6 — the water<br/>OTHER"]
    P7["P7 — the five elderly patients<br/>HUMAN_GROUP"]
    A0["A0 — devote the only water tanker to a child"]
    A1["A1 — devote the only water tanker to five elderly patients"]
    A2["A2 — gets the water"]
    A3["A3 — get the water"]
    E1["E1 — devote the only water tanker<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — the child will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E3["E3 — devote the only water tanker<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — the five elderly patients will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E5["E5 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E7["E7 — gets the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E8["E8 — gets the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E9["E9 — the child will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E10["E10 — get the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E11["E11 — get the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E12["E12 — the five elderly patients will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    CND1{"CND1 — If the child gets the water<br/>POSITIVE"}
    CND2{"CND2 — If the five elderly patients get the water<br/>POSITIVE"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E5
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P4
    A1 -.->|"effect"| E3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E6
    P5 -->|"actor"| A2
    A2 -->|"recipient"| P6
    A2 -.->|"effect"| E7
    A2 -.->|"effect"| E8
    A2 -.->|"effect"| E9
    P7 -->|"actor"| A3
    A3 -->|"recipient"| P6
    A3 -.->|"effect"| E10
    A3 -.->|"effect"| E11
    A3 -.->|"effect"| E12
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P4
    E4 -.->|"affects"| P4
    E5 -.->|"affects"| P4
    E6 -.->|"affects"| P3
    E7 -.->|"affects"| P6
    E8 -.->|"affects"| P6
    E9 -.->|"affects"| P5
    E10 -.->|"affects"| P6
    E11 -.->|"affects"| P6
    E12 -.->|"affects"| P7
    E1 -->|"ENABLES"| E2
    E3 -->|"ENABLES"| E4
    E1 -->|"ENABLES"| E5
    E3 -->|"ENABLES"| E6
    E7 -->|"ENABLES"| E8
    E8 -->|"ENABLES"| E9
    E10 -->|"ENABLES"| E11
    E11 -->|"ENABLES"| E12
    CND1 -.->|"branch scope"| A2
    CND2 -.->|"branch scope"| A3
```

Native admission:

```json
{
  "blueprint_id": "exclusive_allocation",
  "rank": -0.1,
  "variant": "amended",
  "candidate_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/00_exclusive_allocation/candidate.json",
  "overlay_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/00_exclusive_allocation/hypothesis_overlay.json",
  "status": "REJECTED",
  "returncode": 1,
  "error": "Parliament rejected the blueprint candidate: {\"status\": \"REJECTED\", \"actions\": {}, \"world_model\": {}, \"world_model_status\": \"COMMITTED_WITH_UNCERTAINTY\", \"world_contradictions\": [], \"errors\": [\"A0 has no distinguishing source clause; its grounding ['C0', 'C1'] is covered entirely by the other actions\", \"A1 has no distinguishing source clause; its grounding ['C0', 'C2'] is covered entirely by the other actions\", \"A2 has no distinguishing source clause; its grounding ['C1'] is covered entirely by the other actions\", \"A3 has no distinguishing source clause; its grounding ['C2'] is covered entirely by the other actions\"], \"warnings\": [\"E5 is caused directly by another party's act or bodily outcome (E1); insert a PROCESS, FACILITY, INSTITUTION, INFRASTRUCTURE, or RESOURCE state as the immediate parent. A person's framing, execution, or death is not that intermediate. Do not delete E5 or drop its causal_link without a replacement parent; the source also names P2 (RESOURCE the only water tanker), a plausible missing process-state parent for E5 \\u2014 confirm that reading from the source; an unused facility or resource is not automatically the correct intermediary E6 is caused directly by another party's act or bodily outcome (E3); insert a PROCESS, FACILITY, INSTITUTION, INFRASTRUCTURE, or RESOURCE state as the immediate parent. A person's framing, execution, or death is not that intermediate. Do not delete E6 or drop its causal_link without a replacement parent; the source also names P2 (RESOURCE the only water tanker), a plausible missing process-state parent for E6 \\u2014 confirm that reading from the source; an unused facility or resource is not automatically the correct intermediary\"], \"validation_issues\": [{\"code\": \"WORLD_VALIDATION_ERROR\", \"message\": \"A0 has no distinguishing source clause; its grounding ['C0', 'C1'] is covered entirely by the other actions\", \"entity_kind\": \"action\", \"entity_id\": \"A0\", \"field\": \"\", \"related_ids\": [], \"repair_class\": \"SEMANTIC_PATCH\", \"permits_removal\": false}, {\"code\": \"WORLD_VALIDATION_ERROR\", \"message\": \"A1 has no distinguishing source clause; its grounding ['C0', 'C2'] is covered entirely by the other actions\", \"entity_kind\": \"action\", \"entity_id\": \"A1\", \"field\": \"\", \"related_ids\": [], \"repair_class\": \"SEMANTIC_PATCH\", \"permits_removal\": false}, {\"code\": \"WORLD_VALIDATION_ERROR\", \"message\": \"A2 has no distinguishing source clause; its grounding ['C1'] is covered entirely by the other actions\", \"entity_kind\": \"action\", \"entity_id\": \"A2\", \"field\": \"\", \"related_ids\": [], \"repair_class\": \"SEMANTIC_PATCH\", \"permits_removal\": false}, {\"code\": \"WORLD_VALIDATION_ERROR\", \"message\": \"A3 has no distinguishing source clause; its grounding ['C2'] is covered entirely by the other actions\", \"entity_kind\": \"action\", \"entity_id\": \"A3\", \"field\": \"\", \"related_ids\": [], \"repair_class\": \"SEMANTIC_PATCH\", \"permits_removal\": false}], \"compiler_loss_telemetry\": {}, \"clauses\": [{\"clause_id\": \"C0\", \"text\": \"An AI bot must decide whether to devote the only water tanker to a child or five elderly patients, but not both.\"}, {\"clause_id\": \"C1\", \"text\": \"If the child gets the water, the child will live.\"}, {\"clause_id\": \"C2\", \"text\": \"If the five elderly patients get the water, the five elderly patients will live.\"}, {\"clause_id\": \"C3\", \"text\": \"The people who do not get the water will die.\"}]}\n"
}
```

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **28**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c6`: Which construction should preserve this PREDICATION reading?
- `c7`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this MODALITY reading?
- `c11`: Which construction should preserve this MODALITY reading?
- `c12`: Which construction should preserve this MODALITY reading?
- `c13`: Which construction should preserve this MODALITY reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this PARTICIPANT reading?
- `c16`: Which construction should preserve this PARTICIPANT reading?
- `c19`: Which construction should preserve this PARTICIPANT reading?
- `c22`: Which construction should preserve this PARTICIPANT reading?
- `c23`: Which construction should preserve this PARTICIPANT reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c30`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this QUANTITY reading?
- `c33`: Which construction should preserve this QUANTITY reading?
- `c34`: Which construction should preserve this SAME_REFERENT reading?
- `c35`: Which construction should preserve this SAME_REFERENT reading?
- `c36`: Which construction should preserve this SAME_REFERENT reading?
- `c37`: Which construction should preserve this SAME_REFERENT reading?
- `c38`: Which construction should preserve this SAME_REFERENT reading?
- `c39`: Which construction should preserve this SAME_REFERENT reading?
- `c40`: Which construction should preserve this SAME_REFERENT reading?
- `c41`: Which construction should preserve this SAME_REFERENT reading?
- `c42`: Which construction should preserve this SAME_REFERENT reading?

## 2. exclusive_allocation (selected; baseline)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `first_transfer`, `second_transfer`, `quantity`, `first_hedge`, `second_hedge`, `survival_chance`

### exclusive_allocation candidate graph

```mermaid
flowchart LR
    P1["P1 — An AI bot<br/>OTHER"]
    P2["P2 — the only water tanker<br/>RESOURCE"]
    P3["P3 — a child<br/>PERSON"]
    P4["P4 — five elderly patients<br/>HUMAN_GROUP"]
    A0["A0 — devote the only water tanker to a child"]
    A1["A1 — devote the only water tanker to five elderly patients"]
    E1["E1 — devote the only water tanker<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — the child will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E3["E3 — devote the only water tanker<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — the five elderly patients will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E5["E5 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E5
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P4
    A1 -.->|"effect"| E3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E6
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P4
    E4 -.->|"affects"| P4
    E5 -.->|"affects"| P4
    E6 -.->|"affects"| P3
    E1 -->|"ENABLES"| E2
    E3 -->|"ENABLES"| E4
    E1 -->|"ENABLES"| E5
    E3 -->|"ENABLES"| E6
```

Native admission:

```json
{
  "blueprint_id": "exclusive_allocation",
  "rank": 0,
  "variant": "baseline",
  "candidate_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/01_exclusive_allocation/candidate.json",
  "overlay_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/01_exclusive_allocation/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED_WITH_UNCERTAINTY",
    "effects": 6,
    "causal_links": 4,
    "frozen_trace_valid": true,
    "discourse_committed": false,
    "relent_workspace": {
      "established": [
        {
          "kind": "occurrence_effect",
          "effect_id": "E1",
          "outcome": "devote the only water tanker",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E2",
          "outcome": "the child will live",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E3",
          "outcome": "devote the only water tanker",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E4",
          "outcome": "the five elderly patients will live",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E5",
          "outcome": "will die",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E6",
          "outcome": "will die",
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
    "fingerprint_sha256": "d2d9420cbfbbb0c192a089fe0c39517fd6822f633160c068ee80fde7720abf63"
  },
  "trace_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/01_exclusive_allocation/frozen_world_trace.json"
}
```

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **28**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c6`: Which construction should preserve this PREDICATION reading?
- `c7`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this MODALITY reading?
- `c11`: Which construction should preserve this MODALITY reading?
- `c12`: Which construction should preserve this MODALITY reading?
- `c13`: Which construction should preserve this MODALITY reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this PARTICIPANT reading?
- `c16`: Which construction should preserve this PARTICIPANT reading?
- `c19`: Which construction should preserve this PARTICIPANT reading?
- `c22`: Which construction should preserve this PARTICIPANT reading?
- `c23`: Which construction should preserve this PARTICIPANT reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c30`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this QUANTITY reading?
- `c33`: Which construction should preserve this QUANTITY reading?
- `c34`: Which construction should preserve this SAME_REFERENT reading?
- `c35`: Which construction should preserve this SAME_REFERENT reading?
- `c36`: Which construction should preserve this SAME_REFERENT reading?
- `c37`: Which construction should preserve this SAME_REFERENT reading?
- `c38`: Which construction should preserve this SAME_REFERENT reading?
- `c39`: Which construction should preserve this SAME_REFERENT reading?
- `c40`: Which construction should preserve this SAME_REFERENT reading?
- `c41`: Which construction should preserve this SAME_REFERENT reading?
- `c42`: Which construction should preserve this SAME_REFERENT reading?

## 3. rescue_contrast (alternative; baseline)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `first_saved`, `second_saved`, `first_rescue_action`, `second_rescue_action`

World withheld:

- Required slots remain unfilled: first_saved, second_saved, first_rescue_action, second_rescue_action

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **28**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c6`: Which construction should preserve this PREDICATION reading?
- `c7`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this MODALITY reading?
- `c11`: Which construction should preserve this MODALITY reading?
- `c12`: Which construction should preserve this MODALITY reading?
- `c13`: Which construction should preserve this MODALITY reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this PARTICIPANT reading?
- `c16`: Which construction should preserve this PARTICIPANT reading?
- `c19`: Which construction should preserve this PARTICIPANT reading?
- `c22`: Which construction should preserve this PARTICIPANT reading?
- `c23`: Which construction should preserve this PARTICIPANT reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c30`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this QUANTITY reading?
- `c33`: Which construction should preserve this QUANTITY reading?
- `c34`: Which construction should preserve this SAME_REFERENT reading?
- `c35`: Which construction should preserve this SAME_REFERENT reading?
- `c36`: Which construction should preserve this SAME_REFERENT reading?
- `c37`: Which construction should preserve this SAME_REFERENT reading?
- `c38`: Which construction should preserve this SAME_REFERENT reading?
- `c39`: Which construction should preserve this SAME_REFERENT reading?
- `c40`: Which construction should preserve this SAME_REFERENT reading?
- `c41`: Which construction should preserve this SAME_REFERENT reading?
- `c42`: Which construction should preserve this SAME_REFERENT reading?

## 4. omission_harm (alternative; baseline)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `harm_done`, `done_hedge`, `omitted_hedge`, `instrument`

World withheld:

- Required slots remain unfilled: harm_done

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **28**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c6`: Which construction should preserve this PREDICATION reading?
- `c7`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this MODALITY reading?
- `c11`: Which construction should preserve this MODALITY reading?
- `c12`: Which construction should preserve this MODALITY reading?
- `c13`: Which construction should preserve this MODALITY reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this PARTICIPANT reading?
- `c16`: Which construction should preserve this PARTICIPANT reading?
- `c19`: Which construction should preserve this PARTICIPANT reading?
- `c22`: Which construction should preserve this PARTICIPANT reading?
- `c23`: Which construction should preserve this PARTICIPANT reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c30`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this QUANTITY reading?
- `c33`: Which construction should preserve this QUANTITY reading?
- `c34`: Which construction should preserve this SAME_REFERENT reading?
- `c35`: Which construction should preserve this SAME_REFERENT reading?
- `c36`: Which construction should preserve this SAME_REFERENT reading?
- `c37`: Which construction should preserve this SAME_REFERENT reading?
- `c38`: Which construction should preserve this SAME_REFERENT reading?
- `c39`: Which construction should preserve this SAME_REFERENT reading?
- `c40`: Which construction should preserve this SAME_REFERENT reading?
- `c41`: Which construction should preserve this SAME_REFERENT reading?
- `c42`: Which construction should preserve this SAME_REFERENT reading?

## 5. flexible_graph (alternative; flexible)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: none

### flexible_graph candidate graph

```mermaid
flowchart LR
    P1["P1 — the child<br/>PERSON"]
    P2["P2 — the water<br/>OTHER"]
    P3["P3 — the five elderly patients<br/>HUMAN_GROUP"]
    A0["A0 — gets the water"]
    A1["A1 — get the water"]
    E1["E1 — gets the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — gets the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — the child will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E4["E4 — get the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E5["E5 — get the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — the five elderly patients will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    CND1{"CND1 — If the child gets the water<br/>POSITIVE"}
    CND2{"CND2 — If the five elderly patients get the water<br/>POSITIVE"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P3 -->|"actor"| A1
    A1 -->|"recipient"| P2
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    E1 -.->|"affects"| P2
    E2 -.->|"affects"| P2
    E3 -.->|"affects"| P1
    E4 -.->|"affects"| P2
    E5 -.->|"affects"| P2
    E6 -.->|"affects"| P3
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
```

Native admission:

```json
{
  "blueprint_id": "flexible_graph",
  "rank": 3,
  "variant": "flexible",
  "candidate_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/04_flexible_graph/candidate.json",
  "overlay_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/04_flexible_graph/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 6,
    "causal_links": 4,
    "frozen_trace_valid": true,
    "discourse_committed": false,
    "relent_workspace": {
      "established": [
        {
          "kind": "occurrence_effect",
          "effect_id": "E1",
          "outcome": "gets the water",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E2",
          "outcome": "gets the water",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E3",
          "outcome": "the child will live",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E4",
          "outcome": "get the water",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E5",
          "outcome": "get the water",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E6",
          "outcome": "the five elderly patients will live",
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
    "fingerprint_sha256": "75d90cec3a46dd31d8c98b816f40a18c5743c110c065c4ca9840d96f324a8e18"
  },
  "trace_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/04_flexible_graph/frozen_world_trace.json"
}
```

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **28**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c6`: Which construction should preserve this PREDICATION reading?
- `c7`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this MODALITY reading?
- `c11`: Which construction should preserve this MODALITY reading?
- `c12`: Which construction should preserve this MODALITY reading?
- `c13`: Which construction should preserve this MODALITY reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this PARTICIPANT reading?
- `c16`: Which construction should preserve this PARTICIPANT reading?
- `c19`: Which construction should preserve this PARTICIPANT reading?
- `c22`: Which construction should preserve this PARTICIPANT reading?
- `c23`: Which construction should preserve this PARTICIPANT reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c30`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this QUANTITY reading?
- `c33`: Which construction should preserve this QUANTITY reading?
- `c34`: Which construction should preserve this SAME_REFERENT reading?
- `c35`: Which construction should preserve this SAME_REFERENT reading?
- `c36`: Which construction should preserve this SAME_REFERENT reading?
- `c37`: Which construction should preserve this SAME_REFERENT reading?
- `c38`: Which construction should preserve this SAME_REFERENT reading?
- `c39`: Which construction should preserve this SAME_REFERENT reading?
- `c40`: Which construction should preserve this SAME_REFERENT reading?
- `c41`: Which construction should preserve this SAME_REFERENT reading?
- `c42`: Which construction should preserve this SAME_REFERENT reading?

## 6. primitive_composition (alternative; composition)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: none

### primitive_composition candidate graph

```mermaid
flowchart LR
    P1["P1 — the child<br/>PERSON"]
    P2["P2 — the water<br/>OTHER"]
    P3["P3 — the five elderly patients<br/>HUMAN_GROUP"]
    A0["A0 — gets the water"]
    A1["A1 — get the water"]
    E1["E1 — gets the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — gets the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — the child will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E4["E4 — get the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E5["E5 — get the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — the five elderly patients will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    CND1{"CND1 — If the child gets the water<br/>POSITIVE"}
    CND2{"CND2 — If the five elderly patients get the water<br/>POSITIVE"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P3 -->|"actor"| A1
    A1 -->|"recipient"| P2
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    E1 -.->|"affects"| P2
    E2 -.->|"affects"| P2
    E3 -.->|"affects"| P1
    E4 -.->|"affects"| P2
    E5 -.->|"affects"| P2
    E6 -.->|"affects"| P3
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
```

Native admission:

```json
{
  "blueprint_id": "primitive_composition",
  "rank": 4,
  "variant": "composition",
  "candidate_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/05_primitive_composition/candidate.json",
  "overlay_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/05_primitive_composition/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 6,
    "causal_links": 4,
    "frozen_trace_valid": true,
    "discourse_committed": false,
    "relent_workspace": {
      "established": [
        {
          "kind": "occurrence_effect",
          "effect_id": "E1",
          "outcome": "gets the water",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E2",
          "outcome": "gets the water",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E3",
          "outcome": "the child will live",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E4",
          "outcome": "get the water",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E5",
          "outcome": "get the water",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E6",
          "outcome": "the five elderly patients will live",
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
    "fingerprint_sha256": "340db65a9577fe46c7cfc070ebf3f2c0497cac2f5edc387e6ff90f46e3573489"
  },
  "trace_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/water_tanker_full/admission_attempts/05_primitive_composition/frozen_world_trace.json"
}
```

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **28**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c6`: Which construction should preserve this PREDICATION reading?
- `c7`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this MODALITY reading?
- `c11`: Which construction should preserve this MODALITY reading?
- `c12`: Which construction should preserve this MODALITY reading?
- `c13`: Which construction should preserve this MODALITY reading?
- `c14`: Which construction should preserve this MODALITY reading?
- `c15`: Which construction should preserve this PARTICIPANT reading?
- `c16`: Which construction should preserve this PARTICIPANT reading?
- `c19`: Which construction should preserve this PARTICIPANT reading?
- `c22`: Which construction should preserve this PARTICIPANT reading?
- `c23`: Which construction should preserve this PARTICIPANT reading?
- `c24`: Which construction should preserve this PARTICIPANT reading?
- `c25`: Which construction should preserve this PARTICIPANT reading?
- `c27`: Which construction should preserve this PARTICIPANT reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c30`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this QUANTITY reading?
- `c33`: Which construction should preserve this QUANTITY reading?
- `c34`: Which construction should preserve this SAME_REFERENT reading?
- `c35`: Which construction should preserve this SAME_REFERENT reading?
- `c36`: Which construction should preserve this SAME_REFERENT reading?
- `c37`: Which construction should preserve this SAME_REFERENT reading?
- `c38`: Which construction should preserve this SAME_REFERENT reading?
- `c39`: Which construction should preserve this SAME_REFERENT reading?
- `c40`: Which construction should preserve this SAME_REFERENT reading?
- `c41`: Which construction should preserve this SAME_REFERENT reading?
- `c42`: Which construction should preserve this SAME_REFERENT reading?
