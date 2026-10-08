# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — An AI bot<br/>OTHER"]
    P2["P2 — the only water tanker<br/>RESOURCE"]
    P3["P3 — the child<br/>PERSON"]
    P4["P4 — five elderly patients<br/>HUMAN_GROUP"]
    P5["P5 — a child<br/>PERSON"]
    A0["A0 — devote the tanker to the child or five elderly patients, but not both to the child"]
    A1["A1 — devote the tanker to the child or five elderly patients, but not both to five elderly patients"]
    E1["E1 — devote the tanker to the child or five elderly patients, but not both<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — the child will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E3["E3 — devote the tanker to the child or five elderly patients, but not both<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — the five elderly patients will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E5["E5 — The people who do not get the water<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E7["E7 — The people who do not get the water<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E8["E8 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]

    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E5
    A0 -.->|"effect"| E6
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P4
    A1 -.->|"effect"| E3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E7
    A1 -.->|"effect"| E8
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P4
    E4 -.->|"affects"| P4
    E5 -.->|"affects"| P4
    E6 -.->|"affects"| P4
    E7 -.->|"affects"| P3
    E8 -.->|"affects"| P3
    E1 -->|"ENABLES"| E2
    E3 -->|"ENABLES"| E4
    E1 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    E3 -->|"ENABLES"| E7
    E7 -->|"ENABLES"| E8
    PR1(["PR1 — <br/>proposition · COMMITTED_CONTENT"])
    K0(["K0 — promised<br/>commitment · "])
    P1 -.->|"PROMISOR"| K0
    K0 -.->|"PROMISEE"| P5
    K0 -.->|"CONTENT"| PR1
```

Admission: **COMMITTED**  
Admitted effects: `E1, E2, E3, E4, E5, E6, E7, E8`  
Conditions: **0**; Temporal links: **0**; Counterfactual links: **0**.

Discourse objects (not established effects):

- `PR1` (proposition): 
- `K0` (commitment): promised
