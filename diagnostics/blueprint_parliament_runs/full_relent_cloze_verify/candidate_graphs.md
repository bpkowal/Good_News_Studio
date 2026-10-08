# Candidate world attempts

Chosen blueprint: **exclusive_allocation**  
Chooser status: **FILLED**

## 1. exclusive_allocation (selected)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `second_branch_sentence`

### exclusive_allocation candidate graph

```mermaid
flowchart LR
    P1["P1 — Dr. Rivera<br/>PERSON"]
    P2["P2 — one dose of antiviral<br/>RESOURCE"]
    P3["P3 — one child<br/>PERSON"]
    P4["P4 — three adult patients<br/>HUMAN_GROUP"]
    A0["A0 — gives the antiviral to the child"]
    A1["A1 — gives the antiviral to the three adult patients"]
    E1["E1 — gives the antiviral to the child<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — the child has an 80% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E3["E3 — gives the antiviral to the three adult patients<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — each of the three adult patients has a 60% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E5["E5 — does not receive one dose of antiviral<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — does not receive one dose of antiviral<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
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

## 3. conditional_outcome (alternative)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: none

### conditional_outcome candidate graph

```mermaid
flowchart LR
    P1["P1 — Dr. Rivera<br/>PERSON"]
    P2["P2 — the child<br/>PERSON"]
    P3["P3 — each<br/>PERSON"]
    A0["A0 — gives the antiviral to the child"]
    A1["A1 — gives the antiviral to the three adult patients"]
    E1["E1 — gives the antiviral to the child<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — the child has an 80% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E3["E3 — gives the antiviral to the three adult patients<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — each of the three adult patients has a 60% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    CND1{"CND1 — If Dr. Rivera gives the antiviral to the child<br/>POSITIVE"}
    CND2{"CND2 — If Dr. Rivera gives the antiviral to the three adult patients<br/>POSITIVE"}
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
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
```
