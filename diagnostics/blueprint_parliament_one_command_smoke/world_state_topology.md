# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — Ada<br/>PERSON"]
    P2["P2 — Ben<br/>PERSON"]
    P3["P3 — Cara<br/>PERSON"]
    P4["P4 — the serum<br/>RESOURCE"]
    A0["A0 — give the serum to Ben"]
    A1["A1 — give the serum to Cara"]
    E1["E1 — receives serum<br/>DIRECT · BENEFICIAL · CERTAIN"]
    E2["E2 — does not receive serum<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E3["E3 — Ben recovers<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    E4["E4 — receives serum<br/>DIRECT · BENEFICIAL · CERTAIN"]
    E5["E5 — does not receive serum<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — Cara recovers<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]

    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    E1 -.->|"affects"| P2
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P3
    E5 -.->|"affects"| P2
    E6 -.->|"affects"| P3
    E1 -->|"CAUSES"| E3
    E4 -->|"CAUSES"| E6
    E1 -->|"CAUSES"| E2
    E4 -->|"CAUSES"| E5
```

Admission: **COMMITTED**  
Admitted effects: `E1, E2, E3, E4, E5, E6`  
Conditions: **0**; Temporal links: **0**; Counterfactual links: **0**.
