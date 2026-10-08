# Candidate world attempts

Chosen blueprint: **exclusive_allocation**  
Chooser status: **FILLED**

## 1. exclusive_allocation (selected)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `first_transfer`, `second_transfer`, `first_hedge`, `second_hedge`, `second_branch_sentence`

### exclusive_allocation candidate graph

```mermaid
flowchart LR
    P1["P1 — Dr. Rivera<br/>PERSON"]
    P2["P2 — one vial of antivenom<br/>RESOURCE"]
    P3["P3 — one child<br/>PERSON"]
    P4["P4 — three adult patients<br/>HUMAN_GROUP"]
    A0["A0 — gives the antivenom to the child"]
    A1["A1 — gives the antivenom to the three adult patients"]
    E1["E1 — gives the antivenom to the child<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — the child has an 80% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E3["E3 — gives the antivenom to the three adult patients<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — each of the three adult patients has a 60% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E5["E5 — does not receive one vial of antivenom<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — does not receive one vial of antivenom<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
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
    E1 -->|"CAUSES"| E2
    E3 -->|"CAUSES"| E4
    E5 -->|"CAUSES"| E7
    E6 -->|"CAUSES"| E8
```

## 2. rescue_contrast (alternative)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `first_saved`, `second_saved`, `first_rescue_action`, `second_rescue_action`, `first_benefit`, `first_harm`, `second_benefit`, `scene`

World withheld:

- Required slots remain unfilled: first_saved, second_saved, first_rescue_action, second_rescue_action, first_benefit, second_benefit

## 3. conditional_outcome (alternative)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `second_outcome`

### conditional_outcome candidate graph

```mermaid
flowchart LR
    P1["P1 — Dr. Rivera<br/>PERSON"]
    P2["P2 — the child<br/>PERSON"]
    A0["A0 — If Dr. Rivera gives the antivenom to the child"]
    E1["E1 — If Dr. Rivera gives the antivenom to the child<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — the child has an 80% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    E1 -.->|"affects"| P2
    E2 -.->|"affects"| P2
    E1 -->|"CAUSES"| E2
```
