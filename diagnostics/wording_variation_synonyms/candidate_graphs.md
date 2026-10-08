# Candidate world attempts

Chosen blueprint: **exclusive_allocation**  
Chooser status: **FILLED**

## 1. exclusive_allocation (selected)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `first_transfer`, `second_transfer`, `exclusivity`, `survival_chance`, `nonreceipt`

### exclusive_allocation candidate graph

```mermaid
flowchart LR
    P1["P1 — Dr. Lee<br/>PERSON"]
    P2["P2 — one vial of antidote<br/>RESOURCE"]
    P3["P3 — Ana<br/>PERSON"]
    P4["P4 — Bo<br/>PERSON"]
    A0["A0 — administer the antidote"]
    A1["A1 — administer the antidote"]
    E1["E1 — administer the antidote<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — Ana has an 85% chance of recovering if Dr. Lee administers it to Ana<br/>DOWNSTREAM · BENEFICIAL · POSSIBLE"]
    E3["E3 — administer the antidote<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — Bo has a 40% chance of recovering if Dr. Lee administers it to Bo<br/>DOWNSTREAM · BENEFICIAL · POSSIBLE"]
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
Missing slots: `first_saved`, `second_saved`, `rescue_exclusivity`, `first_rescue_action`, `second_rescue_action`, `first_benefit`, `second_benefit`

World withheld:

- Required slots remain unfilled: first_saved, second_saved, rescue_exclusivity, first_rescue_action, second_rescue_action, first_benefit, second_benefit

## 3. omission_harm (alternative)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `omitted`, `done_hedge`, `omitted_hedge`, `group_counts`, `instrument`

World withheld:

- Required slots remain unfilled: omitted
