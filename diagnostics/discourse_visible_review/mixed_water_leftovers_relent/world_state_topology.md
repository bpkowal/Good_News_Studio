# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — An AI bot<br/>OTHER"]
    P2["P2 — the only water tanker<br/>RESOURCE"]
    P3["P3 — the child<br/>PERSON"]
    P4["P4 — five elderly patients<br/>HUMAN_GROUP"]
    P5["P5 — a child<br/>PERSON"]
    A0["A0 — the child gets the water"]
    A1["A1 — the five elderly patients get the water"]
    E1["E1 — the child gets the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — the child will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E3["E3 — the five elderly patients get the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — the five elderly patients will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E5["E5 — The people who do not get the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — The people who do not get the water<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E7["E7 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E8["E8 — The people who do not get the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E9["E9 — The people who do not get the water<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E10["E10 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]

    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E5
    A0 -.->|"effect"| E6
    A0 -.->|"effect"| E7
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P4
    A1 -.->|"effect"| E3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E8
    A1 -.->|"effect"| E9
    A1 -.->|"effect"| E10
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P4
    E4 -.->|"affects"| P4
    E5 -.->|"affects"| P2
    E6 -.->|"affects"| P4
    E7 -.->|"affects"| P4
    E8 -.->|"affects"| P2
    E9 -.->|"affects"| P3
    E10 -.->|"affects"| P3
    E1 -->|"ENABLES"| E2
    E3 -->|"ENABLES"| E4
    E1 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    E6 -->|"ENABLES"| E7
    E3 -->|"ENABLES"| E8
    E8 -->|"ENABLES"| E9
    E9 -->|"ENABLES"| E10
    PR1(["PR1 — a child the only water tanker<br/>proposition · COMMITTED_CONTENT"])
    K0(["K0 — promised<br/>commitment · "])
    P1 -.->|"PROMISOR"| K0
    K0 -.->|"PROMISEE"| P5
    K0 -.->|"CONTENT"| PR1
```

Admission: **COMMITTED**  
Admitted effects: `E1, E2, E3, E4, E5, E6, E7, E8, E9, E10`  
Conditions: **0**; Temporal links: **0**; Counterfactual links: **0**.

Discourse objects (not established effects):

- `PR1` (proposition): a child the only water tanker
- `K0` (commitment): promised
