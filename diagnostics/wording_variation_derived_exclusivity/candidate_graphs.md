# Candidate world attempts

Chosen blueprint: **exclusive_allocation**  
Chooser status: **WITHHELD**

## 1. exclusive_allocation (selected)

Template status: **FILLED**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `exclusivity`, `first_hedge`, `second_hedge`, `survival_chance`, `nonreceipt`

World withheld:

- Exclusivity is unspecified. The alternatives are not an exclusive set, so an averted-alternative effect is not licensed.

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
