# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — Ada<br/>PERSON"]
    P2["P2 — Ben<br/>PERSON"]
    A0["A0 — Ada promised Ben"]

    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    PR1(["PR1 — she would deliver the water<br/>proposition · COMMITTED_CONTENT"])
    K0(["K0 — Ada promised Ben<br/>commitment · "])
    P1 -.->|"PROMISOR"| K0
    K0 -.->|"PROMISEE"| P2
    K0 -.->|"CONTENT"| PR1
```

Admission: ****  
Admitted effects: ``  
Conditions: **0**; Temporal links: **0**; Counterfactual links: **0**.

Discourse objects (not established effects):

- `PR1` (proposition): she would deliver the water
- `K0` (commitment): Ada promised Ben
