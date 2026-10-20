# Frozen laboratory world: readable projection

This view groups the existing effects by action. The full saved topology is in [world_state_topology.md](world_state_topology.md). Edges shown below exist in the supplied world; some duplicate action-to-effect and participant edges are omitted for readability. These are alternative branches, not events asserted to have occurred. CERTAIN effects are conditional on choosing their branch. ENABLES is the saved structural relation, not an established probability of the next effect.

```mermaid
flowchart TB
  M["P1: Mira"]
  subgraph Open["A0: Open the door now"]
    A0["Open now"] -->|HAS_EFFECT| E1["E1: Door opens — CERTAIN"]
    E1 -->|ENABLES| E2["E2: Hazardous fume exposure — POSSIBLE"]
    E2 -->|AFFECTS| S["P3: Nearby staff"]
  end
  subgraph Wait["A1: Stop the experiment and wait"]
    A1["Stop and wait"] -->|HAS_EFFECT| E3["E3: Experiment stops — CERTAIN"]
    E3 -->|ENABLES| E4["E4: Stopping takes ten minutes"]
    E4 -->|ENABLES| E5["E5: Death during delay — POSSIBLE"]
    E5 -->|AFFECTS| T["P5: Technician"]
  end
  M -->|ACTOR| A0
  M -->|ACTOR| A1
  subgraph Rules["Source rules — moral authority not established"]
    R0["P6: A laboratory safety rule"] -->|AUTHORITY| N0["N0: Obligation"]
    N0 -->|GOVERNS| PR1["PR1: Keep door closed while experiment runs"]
    R1["P7: The rule — separate source mention"] -->|AUTHORITY| N1["N1: Permission"]
    N1 -->|GOVERNS| PR2["PR2: Opening only after experiment stops"]
  end
  M -->|BEARER| N0
```

The graph supplies no numerical risk probabilities, fume-injury severity or guaranteed rescue effect. The technician's trapped state and urgent need also appear in the full scenario supplied to every agent. N1 has no resolved bearer; the two authority mentions were not silently merged. No norm-to-action compliance edge or actual rule violation is inferred in this display.
