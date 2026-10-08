# Candidate world attempts

Chosen blueprint: **disputed_report**  
Chooser status: **FILLED**

## 1. disputed_report (selected; baseline)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `competing_report`, `reliability`, `confirmation`

### disputed_report candidate graph

```mermaid
flowchart LR
    P1["P1 — Ada<br/>PERSON"]
    A0["A0 — claims"]
    P1 -->|"actor"| A0
    PR1([" PR1 — the water is safe<br/>proposition · ATTRIBUTED "])
    R0([" R0 — claims<br/>report ·  "])
    P1 -.->|"CLAIMS"| R0
    R0 -.->|"CONTENT"| PR1
    A0 -.->|"SPEECH_ACT"| R0
```

Native admission:

```json
{
  "blueprint_id": "disputed_report",
  "rank": 0,
  "variant": "baseline",
  "candidate_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/report/admission_attempts/00_disputed_report/candidate.json",
  "overlay_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/report/admission_attempts/00_disputed_report/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 0,
    "causal_links": 0,
    "frozen_trace_valid": false,
    "discourse_committed": true,
    "relent_workspace": {
      "established": [],
      "not_established": [
        {
          "kind": "proposition",
          "proposition_id": "PR1",
          "predication": "the water is safe",
          "status": "ATTRIBUTED"
        },
        {
          "kind": "report",
          "report_id": "R0",
          "speech_act": "claims",
          "content_proposition_id": "PR1",
          "status": "attributed"
        }
      ],
      "duty_ledger_citations": [],
      "ev_forbidden_proposition_ids": [
        "PR1"
      ],
      "factual_status_sources": [
        "occurrence_effects"
      ]
    },
    "fingerprint_sha256": null
  },
  "trace_path": "/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/discourse_visible_review/report/admission_attempts/00_disputed_report/frozen_world_trace.json"
}
```

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **4**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c0`: Which construction should preserve this PREDICATION reading?
- `c1`: Which construction should preserve this PREDICATION reading?
- `c2`: Which construction should preserve this PARTICIPANT reading?
- `c3`: Which construction should preserve this PARTICIPANT reading?

## 2. conditional_outcome (alternative; baseline)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `condition`, `bearer`, `outcome`, `second_condition`, `second_bearer`, `second_outcome`, `chance`

World withheld:

- Required slots remain unfilled: condition, bearer, outcome

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **4**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c0`: Which construction should preserve this PREDICATION reading?
- `c1`: Which construction should preserve this PREDICATION reading?
- `c2`: Which construction should preserve this PARTICIPANT reading?
- `c3`: Which construction should preserve this PARTICIPANT reading?

## 3. none (alternative; baseline)

Template status: **FILLED**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: none

World withheld:

- No listed blueprint is supported by the scenario.

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **4**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c0`: Which construction should preserve this PREDICATION reading?
- `c1`: Which construction should preserve this PREDICATION reading?
- `c2`: Which construction should preserve this PARTICIPANT reading?
- `c3`: Which construction should preserve this PARTICIPANT reading?

## 4. primitive_composition (alternative; composition)

Template status: **WITHHELD**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: none

World withheld:

- No supported conditional branch was extracted.

Source constructions retained outside the admitted world:

Parser proposals below preserve alternatives and local scope; they do not establish occurrence or a resolved interpretation.

Unconsumed parser candidates: **4**. These are visible mapping gaps, not admission failures.

```mermaid
flowchart LR
    gap{"Missing construction mappings"}
    source["Unconsumed source candidates"] -.-> gap
```

- `c0`: Which construction should preserve this PREDICATION reading?
- `c1`: Which construction should preserve this PREDICATION reading?
- `c2`: Which construction should preserve this PARTICIPANT reading?
- `c3`: Which construction should preserve this PARTICIPANT reading?
