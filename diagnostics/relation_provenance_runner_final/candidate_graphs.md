# Candidate world attempts

Chosen blueprint: **exclusive_allocation**  
Chooser status: **FILLED**

## 1. exclusive_allocation (selected)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `first_hedge`, `second_hedge`, `survival_chance`

### exclusive_allocation candidate graph

```mermaid
flowchart LR
    P1["P1 — Ada<br/>PERSON"]
    P2["P2 — one dose of medicine<br/>RESOURCE"]
    P3["P3 — Ben<br/>PERSON"]
    P4["P4 — Cara<br/>PERSON"]
    A0["A0 — Ada gives the medicine to Ben"]
    A1["A1 — Ada gives the medicine to Cara"]
    E1["E1 — Ada gives the medicine to Ben<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — Ben has a 95% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E3["E3 — Ada gives the medicine to Cara<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — Cara has a 5% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E5["E5 — does not receive one dose of medicine<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — does not receive one dose of medicine<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E7["E7 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E8["E8 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E5
    A0 -.->|"effect"| E7
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P4
    A1 -.->|"effect"| E3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E6
    A1 -.->|"effect"| E8
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P4
    E4 -.->|"affects"| P4
    E5 -.->|"affects"| P4
    E6 -.->|"affects"| P3
    E7 -.->|"affects"| P4
    E8 -.->|"affects"| P3
    E1 -->|"ENABLES"| E2
    E3 -->|"ENABLES"| E4
    E5 -->|"ENABLES"| E7
    E6 -->|"ENABLES"| E8
```

## 2. rescue_contrast (alternative)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `first_saved`, `second_saved`, `first_rescue_action`, `second_rescue_action`, `first_benefit`, `second_benefit`

World withheld:

- Required slots remain unfilled: first_saved, second_saved, first_rescue_action, second_rescue_action, first_benefit, second_benefit

## 3. omission_harm (alternative)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `done`, `omitted`, `done_hedge`, `omitted_hedge`, `group_counts`, `instrument`

World withheld:

- Required slots remain unfilled: done, omitted
