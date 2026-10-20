# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — Mira<br/>PERSON"]
    P2["P2 — A laboratory safety rule<br/>OTHER"]
    P3["P3 — The rule<br/>OTHER"]

    PR1(["PR1 — keep a containment door closed while a hazardous experiment is running<br/>proposition · GOVERNED"])
    PR2(["PR2 — opening it only after the experiment has stopped<br/>proposition · GOVERNED"])
    N0(["N0 — obligation<br/>norm · "])
    N1(["N1 — permission<br/>norm · "])
    P1 -.->|"BEARER"| N0
    P2 -.->|"AUTHORITY"| N0
    N0 -.->|"GOVERNS"| PR1
    P3 -.->|"AUTHORITY"| N1
    N1 -.->|"GOVERNS"| PR2
```

Admission: **COMMITTED**  
Admitted effects: ``  
Conditions: **0**; Temporal links: **0**; Counterfactual links: **0**.

Discourse objects (not established effects):

- `PR1` (proposition): keep a containment door closed while a hazardous experiment is running
- `PR2` (proposition): opening it only after the experiment has stopped
- `N0` (norm): obligation
- `N1` (norm): permission
