# Typed trolley reader view

The saved trolley demo now opens with typed, directional arrows, organized into scene, pull, and omission branches. Entity, place, process, and effect have distinct display colors. Repeated nodes are display aliases retaining their original world IDs.

Open the running [viewer](http://127.0.0.1:7861/?run=trolley). Use the full-graph toggle to inspect the original admitted topology. Click reader nodes to inspect their original records and provenance.

## What changed

- Source-supported ON, OCCUPIED_BY, HEADING_TOWARD and ONTO relations expose places previously buried in scene prose.
- ACTOR, AFFECTS, UNDERGOES and ENABLES have explicit endpoint types. Pulling affects the lever; the trolley undergoes the switching process.
- RESULT connects an affected person or group to a saved effect. It does not assign causation or moral responsibility.
- Dashed arrows distinguish source readings and unresolved structure from projections of saved roles/links.
- Cloze version 0.13 shares optional evidence-linked physical_scene_cells: mover, instrument, destination, occupancy and branch transitions. The existing required blanks remain unchanged.

## Remaining gap

The saved omission branch contains a placeholder “does not” physical effect. The new view exposes a missing physical continuation instead of inventing a “stays on the main track” event. REMAINS_ON is not inferred; AWAY_FROM the side track would also be misleading. The next construction improvement should bind an explicitly supported default-route process to this branch, with its own provenance.

This is a reversible reader projection, not a new native world schema. Display roles do not change original native kinds. The exported frozen world exactly equals the saved report world; no new world was admitted and no agent/API calls were made.

## Verification

94 regression tests ran: 90 passed and four skipped. Coverage includes typed endpoints, original-world preservation, source offsets, negation/attribution, whitespace invariance, absent/ambiguous mechanisms, and shared Cloze cells. The browser rendered the view successfully; payload and renderer receipts matched. Git diff whitespace checks passed.

Artifacts: [typed records](typed_reader_view.json), [Mermaid source](diagram.mmd), [regression log](regression.log), [export verification](verification.json).

## Diagram

```mermaid
flowchart TB
  subgraph scene["Scene"]
    direction LR
    SC_P7["the trolley<br/>entity"]
    SC_P4["five workers<br/>entity"]
    SC_P8["the main track<br/>place"]
    SC_P2["one worker<br/>entity"]
    SC_P9["a side track<br/>place"]
  end
  subgraph branch_A0["If she pulls the lever"]
    direction LR
    A0["pulls the lever<br/>process"]
    E2["switches the trolley onto a side track<br/>process"]
    A0_P6["a lever<br/>entity"]
    A0_P7["the trolley<br/>entity"]
    A0_P9["a side track<br/>place"]
    A0_P2["one worker<br/>entity"]
    E3["the trolley will kill that worker<br/>effect"]
    A0_P4["five workers<br/>entity"]
    E4["spare the five<br/>effect"]
    A0_P8["the main track<br/>place"]
  end
  subgraph branch_A1["If she does not"]
    direction LR
    A1["Does not pull the lever<br/>process"]
    A1_GAP["Physical continuation not recorded<br/>gap"]
    A1_P4["five workers<br/>entity"]
    E7["it will kill the five<br/>effect"]
    A1_P8["the main track<br/>place"]
    A1_P2["one worker<br/>entity"]
    E8["spare the one<br/>effect"]
    A1_P9["a side track<br/>place"]
  end
  subgraph actor["Decision maker"]
    direction LR
    ACT_P1["Maria<br/>entity"]
  end
  SC_P7 -.->|"HEADING_TOWARD"| SC_P4
  SC_P4 -.->|"ON"| SC_P8
  SC_P2 -.->|"ON"| SC_P9
  ACT_P1 -->|"ACTOR"| A0
  A0 -->|"AFFECTS"| A0_P6
  A0 -->|"ENABLES"| E2
  A0_P7 -->|"UNDERGOES"| E2
  E2 -.->|"ONTO"| A0_P9
  A0_P2 -->|"RESULT"| E3
  A0_P9 -.->|"OCCUPIED_BY"| A0_P2
  A0_P4 -->|"RESULT"| E4
  A0_P8 -.->|"OCCUPIED_BY"| A0_P4
  ACT_P1 -->|"ACTOR"| A1
  A1 -.->|"MECHANISM_UNRESOLVED"| A1_GAP
  A1_P4 -->|"RESULT"| E7
  A1_P8 -.->|"OCCUPIED_BY"| A1_P4
  A1_GAP -.->|"BRANCH_OUTCOME"| E7
  A1_P2 -->|"RESULT"| E8
  A1_P9 -.->|"OCCUPIED_BY"| A1_P2
  A1_GAP -.->|"BRANCH_OUTCOME"| E8
  classDef sourceReading fill:#eef4fb,stroke:#46739a,color:#233b51
  classDef unresolved fill:#fff3db,stroke:#af7b26,stroke-dasharray:5 3,color:#60461e
  classDef reader_entity fill:#eef3fa,stroke:#64809e,color:#26332e
  class SC_P7,SC_P4,SC_P2,ACT_P1,A0_P6,A0_P7,A0_P2,A0_P4,A1_P4,A1_P2 reader_entity
  classDef reader_place fill:#e9f4ec,stroke:#60836a,color:#26332e
  class SC_P8,SC_P9,A0_P9,A0_P8,A1_P8,A1_P9 reader_place
  classDef reader_process fill:#fff6e4,stroke:#a18a56,color:#26332e
  class A0,E2,A1 reader_process
  classDef reader_effect fill:#f6edf5,stroke:#96738d,color:#26332e
  class E3,E4,E7,E8 reader_effect
  class A1_GAP unresolved
```
