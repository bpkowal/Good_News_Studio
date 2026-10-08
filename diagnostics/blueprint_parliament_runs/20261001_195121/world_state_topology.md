# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — Ada<br/>PERSON"]
    P2["P2 — Ben<br/>PERSON"]
    P3["P3 — Cara<br/>PERSON"]
    P4["P4 — the medicine<br/>RESOURCE"]
    A0["A0 — give the medicine to Ben"]
    A1["A1 — give the medicine to Cara"]
    E1["E1 — receives medicine<br/>DIRECT · BENEFICIAL · CERTAIN"]
    E2["E2 — does not receive medicine<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E3["E3 — survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E4["E4 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E5["E5 — receives medicine<br/>DIRECT · BENEFICIAL · CERTAIN"]
    E6["E6 — does not receive medicine<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E7["E7 — survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E8["E8 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]

    P1 -->|"actor"| A0
    A0 -->|"recipient"| P2
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    A0 -.->|"effect"| E4
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P3
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    A1 -.->|"effect"| E7
    A1 -.->|"effect"| E8
    E1 -.->|"affects"| P2
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P3
    E5 -.->|"affects"| P3
    E6 -.->|"affects"| P2
    E7 -.->|"affects"| P3
    E8 -.->|"affects"| P2
    E1 -->|"CAUSES"| E3
    E2 -->|"CAUSES"| E4
    E5 -->|"CAUSES"| E7
    E6 -->|"CAUSES"| E8
    E1 -->|"CAUSES"| E2
    E5 -->|"CAUSES"| E6
```

Admission: **COMMITTED**  
Admitted effects: `E1, E2, E3, E4, E5, E6, E7, E8`  
Conditions: **0**; Temporal links: **0**; Counterfactual links: **0**.
