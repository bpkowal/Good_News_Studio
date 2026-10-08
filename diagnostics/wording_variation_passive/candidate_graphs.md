# Candidate world attempts

Chosen blueprint: **exclusive_allocation**  
Chooser status: **FILLED**

## 1. exclusive_allocation (selected)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `quantity`, `first_hedge`, `second_hedge`, `nonreceipt`

### exclusive_allocation candidate graph

```mermaid
flowchart LR
    P1["P1 — Priya<br/>PERSON"]
    P2["P2 — serum<br/>RESOURCE"]
    P3["P3 — Malik<br/>PERSON"]
    P4["P4 — Rosa<br/>PERSON"]
    A0["A0 — Malik receives the serum"]
    A1["A1 — Rosa receives the serum"]
    E1["E1 — Malik receives the serum<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — he has a 70% chance of surviving<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E3["E3 — Rosa receives the serum<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — she has a 60% chance of surviving<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P4
    A1 -.->|"effect"| E3
    A1 -.->|"effect"| E4
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P4
    E4 -.->|"affects"| P4
    E1 -->|"ENABLES"| E2
    E3 -->|"ENABLES"| E4
```

## 2. rescue_contrast (alternative)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `first_saved`, `second_saved`, `first_rescue_action`, `second_rescue_action`, `first_benefit`, `second_benefit`

World withheld:

- Required slots remain unfilled: first_saved, second_saved, first_rescue_action, second_rescue_action, first_benefit, second_benefit

## 3. conditional_outcome (alternative)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `second_outcome`, `chance`

### conditional_outcome candidate graph

```mermaid
flowchart LR
    P1["P1 — Priya<br/>PERSON"]
    P2["P2 — Malik<br/>PERSON"]
    A0["A0 — If Malik receives the serum"]
    E1["E1 — If Malik receives the serum<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — he has a 70% chance of surviving<br/>DOWNSTREAM · BENEFICIAL · POSSIBLE"]
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    E1 -.->|"affects"| P2
    E2 -.->|"affects"| P2
    E1 -->|"ENABLES"| E2
```
