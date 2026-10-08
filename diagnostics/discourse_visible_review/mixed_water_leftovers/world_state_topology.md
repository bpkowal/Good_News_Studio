# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — the child<br/>PERSON"]
    P2["P2 — the water<br/>OTHER"]
    P3["P3 — the five elderly patients<br/>HUMAN_GROUP"]
    A0["A0 — gets the water"]
    A1["A1 — get the water"]
    E1["E1 — gets the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — gets the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — the child will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E4["E4 — get the water<br/>DIRECT · NEUTRAL · CERTAIN"]
    E5["E5 — get the water<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — the five elderly patients will live<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    CND1{"CND1 — If the child gets the water<br/>POSITIVE"}
    CND2{"CND2 — If the five elderly patients get the water<br/>POSITIVE"}

    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P3 -->|"actor"| A1
    A1 -->|"recipient"| P2
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    E1 -.->|"affects"| P2
    E2 -.->|"affects"| P2
    E3 -.->|"affects"| P1
    E4 -.->|"affects"| P2
    E5 -.->|"affects"| P2
    E6 -.->|"affects"| P3
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
```

Admission: **COMMITTED**  
Admitted effects: `E1, E2, E3, E4, E5, E6`  
Conditions: **2**; Temporal links: **0**; Counterfactual links: **0**.
