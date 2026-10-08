# Candidate world attempts

Chosen blueprint: **omission_harm**  
Chooser status: **FILLED**

## 1. rescue_contrast (alternative)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `first_saved`, `second_saved`, `rescue_exclusivity`, `first_rescue_action`, `second_rescue_action`, `scene`

World withheld:

- Required slots remain unfilled: first_saved, second_saved, rescue_exclusivity, first_rescue_action, second_rescue_action

## 2. omission_harm (selected)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: none

### omission_harm candidate graph

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — One worker<br/>PERSON"]
    P3["P3 — the lever<br/>PROCESS"]
    P4["P4 — five workers<br/>HUMAN_GROUP"]
    A0["A0 — pulls the lever"]
    A1["A1 — does not pull the lever"]
    E1["E1 — pulls the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — pulls the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — one worker will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E4["E4 — does not pull the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E5["E5 — does not pull the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — five workers will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    CND1{"CND1 — If Maria pulls the lever<br/>POSITIVE"}
    CND2{"CND2 — If Maria does not pull the lever<br/>NEGATED"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P3
    E5 -.->|"affects"| P3
    E6 -.->|"affects"| P4
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
```

## 3. conditional_outcome (alternative)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `chance`

### conditional_outcome candidate graph

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — One worker<br/>PERSON"]
    P3["P3 — the lever<br/>PROCESS"]
    P4["P4 — Five workers<br/>HUMAN_GROUP"]
    A0["A0 — pulls the lever"]
    A1["A1 — does not pull the lever"]
    E1["E1 — pulls the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — pulls the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — one worker will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E4["E4 — does not pull the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E5["E5 — does not pull the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — five workers will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    CND1{"CND1 — If Maria pulls the lever<br/>POSITIVE"}
    CND2{"CND2 — If Maria does not pull the lever<br/>NEGATED"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P3
    E5 -.->|"affects"| P3
    E6 -.->|"affects"| P4
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
```
