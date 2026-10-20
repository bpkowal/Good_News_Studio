# Candidate world attempts

Chosen blueprint: **primitive_composition**  
Chooser status: **FILLED**

## 1. primitive_composition (selected; composition)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: none

### primitive_composition candidate graph

```mermaid
flowchart LR
    P1["P1 — Mira<br/>PERSON"]
    P2["P2 — the door<br/>PROCESS"]
    P3["P3 — nearby staff<br/>PERSON"]
    P4["P4 — the experiment<br/>PROCESS"]
    P5["P5 — the technician<br/>PERSON"]
    P6["P6 — A laboratory safety rule<br/>OTHER"]
    P7["P7 — The rule<br/>OTHER"]
    A0["A0 — open the door now"]
    A1["A1 — stop the experiment and wait"]
    E1["E1 — open the door now<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — doing so may expose nearby staff to hazardous fumes<br/>DOWNSTREAM · NEUTRAL · POSSIBLE"]
    E3["E3 — Stopping the experiment<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — Stopping the experiment first takes ten minutes<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E5["E5 — the technician may die during that delay<br/>DOWNSTREAM · ADVERSE · POSSIBLE"]
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P4
    A1 -.->|"effect"| E3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    E1 -.->|"affects"| P2
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P4
    E4 -.->|"affects"| P4
    E5 -.->|"affects"| P5
    E1 -->|"ENABLES"| E2
    E3 -->|"ENABLES"| E4
    E4 -->|"ENABLES"| E5
    PR1([" PR1 — keep a containment door closed while a hazardous experiment is running<br/>proposition · GOVERNED "])
    PR2([" PR2 — opening it only after the experiment has stopped<br/>proposition · GOVERNED "])
    N0([" N0 — obligation<br/>norm ·  "])
    N1([" N1 — permission<br/>norm ·  "])
    P1 -.->|"BEARER"| N0
    P6 -.->|"AUTHORITY"| N0
    N0 -.->|"GOVERNS"| PR1
    P7 -.->|"AUTHORITY"| N1
    N1 -.->|"GOVERNS"| PR2
```

Native admission:

```json
{
  "blueprint_id": "primitive_composition",
  "rank": 0,
  "variant": "composition",
  "candidate_path": "diagnostics/deontic_choice_composition_20261016/event_uncertainty/admission_attempts/00_primitive_composition/candidate.json",
  "overlay_path": "diagnostics/deontic_choice_composition_20261016/event_uncertainty/admission_attempts/00_primitive_composition/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 5,
    "causal_links": 3,
    "frozen_trace_valid": true,
    "discourse_committed": false,
    "relent_workspace": {
      "established": [
        {
          "kind": "occurrence_effect",
          "effect_id": "E1",
          "outcome": "open the door now",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E2",
          "outcome": "doing so may expose nearby staff to hazardous fumes",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E3",
          "outcome": "Stopping the experiment",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E4",
          "outcome": "Stopping the experiment first takes ten minutes",
          "predicate": null,
          "status": "established"
        },
        {
          "kind": "occurrence_effect",
          "effect_id": "E5",
          "outcome": "the technician may die during that delay",
          "predicate": null,
          "status": "established"
        }
      ],
      "not_established": [
        {
          "kind": "proposition",
          "proposition_id": "PR1",
          "predication": "keep a containment door closed while a hazardous experiment is running",
          "status": "GOVERNED"
        },
        {
          "kind": "proposition",
          "proposition_id": "PR2",
          "predication": "opening it only after the experiment has stopped",
          "status": "GOVERNED"
        },
        {
          "kind": "normative_proposition",
          "norm_id": "N0",
          "force": "obligation",
          "governed_proposition_id": "PR1",
          "status": "normative"
        },
        {
          "kind": "normative_proposition",
          "norm_id": "N1",
          "force": "permission",
          "governed_proposition_id": "PR2",
          "status": "normative"
        }
      ],
      "duty_ledger_citations": [
        {
          "kind": "norm",
          "norm_id": "N0"
        },
        {
          "kind": "norm",
          "norm_id": "N1"
        }
      ],
      "ev_forbidden_proposition_ids": [
        "PR1",
        "PR2"
      ],
      "factual_status_sources": [
        "occurrence_effects"
      ]
    },
    "fingerprint_sha256": "6cc85ac208c011aeefc32b9ff4f5c01ce3d02562a2d5c3b323278a1fd2985486"
  },
  "trace_path": "diagnostics/deontic_choice_composition_20261016/event_uncertainty/admission_attempts/00_primitive_composition/frozen_world_trace.json"
}
```

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **67**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c0`: Which construction should preserve this PREDICATION reading?
- `c1`: Which construction should preserve this PREDICATION reading?
- `c2`: Which construction should preserve this PREDICATION reading?
- `c3`: Which construction should preserve this PREDICATION reading?
- `c4`: Which construction should preserve this PREDICATION reading?
- `c5`: Which construction should preserve this PREDICATION reading?
- `c6`: Which construction should preserve this PREDICATION reading?
- `c7`: Which construction should preserve this PREDICATION reading?
- `c8`: Which construction should preserve this PREDICATION reading?
- `c9`: Which construction should preserve this PREDICATION reading?
- `c10`: Which construction should preserve this PREDICATION reading?
- `c11`: Which construction should preserve this PREDICATION reading?
- `c12`: Which construction should preserve this PREDICATION reading?
- `c13`: Which construction should preserve this PREDICATION reading?
- `c14`: Which construction should preserve this PREDICATION reading?
- `c15`: Which construction should preserve this PREDICATION reading?
- `c16`: Which construction should preserve this PREDICATION reading?
- `c17`: Which construction should preserve this PREDICATION reading?
- `c18`: Which construction should preserve this PREDICATION reading?
- `c19`: Which construction should preserve this MODALITY reading?
- `c20`: Which construction should preserve this MODALITY reading?
- `c21`: Which construction should preserve this MODALITY reading?
- `c22`: Which construction should preserve this MODALITY reading?
- `c23`: Which construction should preserve this MODALITY reading?
- `c24`: Which construction should preserve this MODALITY reading?
- `c25`: Which construction should preserve this MODALITY reading?
- `c26`: Which construction should preserve this MODALITY reading?
- `c27`: Which construction should preserve this MODALITY reading?
- `c28`: Which construction should preserve this PARTICIPANT reading?
- `c29`: Which construction should preserve this PARTICIPANT reading?
- `c30`: Which construction should preserve this PARTICIPANT reading?
- `c31`: Which construction should preserve this PARTICIPANT reading?
- `c32`: Which construction should preserve this PARTICIPANT reading?
- `c33`: Which construction should preserve this PARTICIPANT reading?
- `c34`: Which construction should preserve this PARTICIPANT reading?
- `c35`: Which construction should preserve this PARTICIPANT reading?
- `c36`: Which construction should preserve this PARTICIPANT reading?
- `c37`: Which construction should preserve this PARTICIPANT reading?
- `c38`: Which construction should preserve this PARTICIPANT reading?
- `c39`: Which construction should preserve this PARTICIPANT reading?
- `c40`: Which construction should preserve this PARTICIPANT reading?
- `c41`: Which construction should preserve this PARTICIPANT reading?
- `c42`: Which construction should preserve this PARTICIPANT reading?
- `c43`: Which construction should preserve this PARTICIPANT reading?
- `c44`: Which construction should preserve this PARTICIPANT reading?
- `c45`: Which construction should preserve this PARTICIPANT reading?
- `c46`: Which construction should preserve this PARTICIPANT reading?
- `c47`: Which construction should preserve this EVENT_LINK reading?
- `c48`: Which construction should preserve this EVENT_LINK reading?
- `c49`: Which construction should preserve this EVENT_LINK reading?
- `c50`: Which construction should preserve this PARTICIPANT reading?
- `c51`: Which construction should preserve this QUANTITY reading?
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
