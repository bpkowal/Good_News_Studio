# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — Rescuer Ellis<br/>PERSON"]
    P2["P2 — one thermal blanket<br/>RESOURCE"]
    P3["P3 — Rowan<br/>PERSON"]
    P4["P4 — Sasha<br/>PERSON"]
    P5["P5 — Ellis<br/>PERSON"]
    A0["A0 — gives the blanket to Rowan"]
    A1["A1 — gives the blanket to Sasha"]
    E1["E1 — gives the blanket to Rowan<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — Rowan has an 80% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E3["E3 — gives the blanket to Sasha<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — Sasha has a 60% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E5["E5 — does not receive one thermal blanket<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — does not receive one thermal blanket<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E7["E7 — does not get the blanket<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E8["E8 — does not get the blanket<br/>DOWNSTREAM · ADVERSE · CERTAIN"]

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
    E1 -->|"CAUSES"| E5
    E3 -->|"CAUSES"| E6
    PR1(["PR1 — would get the blanket<br/>proposition · COMMITTED_CONTENT"])
    K0(["K0 — promised<br/>commitment · "])
    P5 -.->|"PROMISOR"| K0
    K0 -.->|"PROMISEE"| P4
    K0 -.->|"CONTENT"| PR1
```

Admission: **COMMITTED**  
Admitted effects: `E1, E2, E3, E4, E5, E6, E7, E8`  
Conditions: **0**; Temporal links: **0**; Counterfactual links: **0**.

Discourse objects (not established effects):

- `PR1` (proposition): would get the blanket
- `K0` (commitment): promised
