# Candidate world attempts

Chosen blueprint: **exclusive_allocation**  
Chooser status: **FILLED**

## 1. exclusive_allocation (selected)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `exclusivity`, `survival_chance`, `nonreceipt`

### exclusive_allocation candidate graph

```mermaid
flowchart LR
    P1["P1 — Ada<br/>PERSON"]
    P2["P2 — one indivisible dose of medicine<br/>RESOURCE"]
    P3["P3 — Ben<br/>PERSON"]
    P4["P4 — Cara<br/>PERSON"]
    A0["A0 — Ben receives it"]
    A1["A1 — Cara receives it"]
    E1["E1 — Ben receives it<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — Ben will recover<br/>DOWNSTREAM · BENEFICIAL · POSSIBLE"]
    E3["E3 — Cara receives it<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — Cara will recover<br/>DOWNSTREAM · BENEFICIAL · POSSIBLE"]
    E5["E5 — does not receive one indivisible dose of medicine<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — does not receive one indivisible dose of medicine<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
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
```

## 2. rescue_contrast (alternative)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `first_saved`, `second_saved`, `rescue_exclusivity`, `first_rescue_action`, `second_rescue_action`, `first_harm`, `second_harm`, `scene`

World withheld:

- Required slots remain unfilled: first_saved, second_saved, rescue_exclusivity, first_rescue_action, second_rescue_action

## 3. conditional_outcome (alternative)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `chance`

### conditional_outcome candidate graph

```mermaid
flowchart LR
    P1["P1 — Ada<br/>PERSON"]
    P2["P2 — Ben<br/>PERSON"]
    P3["P3 — Cara<br/>PERSON"]
    A0["A0 — If Ben receives it"]
    A1["A1 — If Cara receives it"]
    E1["E1 — If Ben receives it<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — Ben will recover<br/>DOWNSTREAM · BENEFICIAL · POSSIBLE"]
    E3["E3 — If Cara receives it<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — Cara will recover<br/>DOWNSTREAM · BENEFICIAL · POSSIBLE"]
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P3
    A1 -.->|"effect"| E3
    A1 -.->|"effect"| E4
    E1 -.->|"affects"| P2
    E2 -.->|"affects"| P2
    E3 -.->|"affects"| P3
    E4 -.->|"affects"| P3
    E1 -->|"ENABLES"| E2
    E3 -->|"ENABLES"| E4
```
