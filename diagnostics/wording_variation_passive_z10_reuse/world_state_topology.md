# Admitted world-state topology

```mermaid
flowchart LR
    P1["P1 — Priya<br/>PERSON"]
    P2["P2 — serum<br/>RESOURCE"]
    P3["P3 — Malik<br/>PERSON"]
    P4["P4 — Rosa<br/>PERSON"]
    A0["A0 — Malik receives the serum"]
    A1["A1 — Rosa receives the serum"]
    E1["E1 — Malik receives the serum<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — he has a 70% chance of surviving<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E3["E3 — Rosa receives the serum<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — she has a 60% chance of surviving<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E5["E5 — does not receive serum<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — does not receive serum<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E7["E7 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E8["E8 — will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]

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
```

Admission: **COMMITTED**  
Admitted effects: `E1, E2, E3, E4, E5, E6, E7, E8`  
Conditions: **0**; Temporal links: **0**; Counterfactual links: **0**.
