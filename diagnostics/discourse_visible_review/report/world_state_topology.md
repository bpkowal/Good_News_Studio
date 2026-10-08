# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — Ada<br/>PERSON"]
    A0["A0 — claims"]

    P1 -->|"actor"| A0
    PR1(["PR1 — the water is safe<br/>proposition · ATTRIBUTED"])
    R0(["R0 — claims<br/>report · "])
    P1 -.->|"CLAIMS"| R0
    R0 -.->|"CONTENT"| PR1
    A0 -.->|"SPEECH_ACT"| R0
```

Admission: ****  
Admitted effects: ``  
Conditions: **0**; Temporal links: **0**; Counterfactual links: **0**.

Discourse objects (not established effects):

- `PR1` (proposition): the water is safe
- `R0` (report): claims
