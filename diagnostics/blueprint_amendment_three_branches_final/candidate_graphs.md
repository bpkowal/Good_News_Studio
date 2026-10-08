# Candidate world attempts

Chosen blueprint: **conditional_outcome**  
Chooser status: **FILLED**

## 1. conditional_outcome (selected; amended)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: none

### conditional_outcome candidate graph

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — one worker<br/>PERSON"]
    P3["P3 — the lever<br/>PROCESS"]
    P4["P4 — five workers<br/>HUMAN_GROUP"]
    P5["P5 — two workers<br/>HUMAN_GROUP"]
    P6["P6 — the button<br/>PROCESS"]
    A0["A0 — pulls the lever"]
    A1["A1 — does not pull the lever"]
    A2["A2 — presses the button"]
    E1["E1 — pulls the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — pulls the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — one worker will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E4["E4 — does not pull the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E5["E5 — does not pull the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — five workers will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E7["E7 — presses the button<br/>DIRECT · NEUTRAL · CERTAIN"]
    E8["E8 — presses the button<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E9["E9 — two workers will survive<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    CND1{"CND1 — If Maria pulls the lever<br/>POSITIVE"}
    CND2{"CND2 — If Maria does not pull the lever<br/>NEGATED"}
    CND3{"CND3 — If Maria presses the button<br/>POSITIVE"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    P1 -->|"actor"| A2
    A2 -->|"recipient"| P6
    A2 -.->|"effect"| E7
    A2 -.->|"effect"| E8
    A2 -.->|"effect"| E9
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P3
    E5 -.->|"affects"| P3
    E6 -.->|"affects"| P4
    E7 -.->|"affects"| P6
    E8 -.->|"affects"| P6
    E9 -.->|"affects"| P5
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    E7 -->|"ENABLES"| E8
    E8 -->|"ENABLES"| E9
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
    CND3 -.->|"branch scope"| A2
```

Native admission:

```json
{
  "blueprint_id": "conditional_outcome",
  "rank": -0.1,
  "candidate_path": "diagnostics/blueprint_amendment_three_branches_final/admission_attempts/00_conditional_outcome/candidate.json",
  "overlay_path": "diagnostics/blueprint_amendment_three_branches_final/admission_attempts/00_conditional_outcome/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 9,
    "causal_links": 6,
    "frozen_trace_valid": true,
    "fingerprint_sha256": "b3461c2d5330c6e966dc1487eb775451c976874090506ee6750d87273393ea2c"
  },
  "trace_path": "diagnostics/blueprint_amendment_three_branches_final/admission_attempts/00_conditional_outcome/frozen_world_trace.json",
  "variant": "amended"
}
```

## 2. conditional_outcome (alternative; baseline)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: none

### conditional_outcome candidate graph

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — one worker<br/>PERSON"]
    P3["P3 — the lever<br/>PROCESS"]
    P4["P4 — five workers<br/>HUMAN_GROUP"]
    A0["A0 — pulls the lever"]
    A1["A1 — does not pull the lever"]
    E1["E1 — pulls the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — pulls the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — one worker will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E4["E4 — does not pull the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E5["E5 — does not pull the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — five workers will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    CND1{"CND1 — If Maria pulls the lever<br/>POSITIVE"}
    CND2{"CND2 — If Maria does not pull the lever<br/>NEGATED"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P3
    E5 -.->|"affects"| P3
    E6 -.->|"affects"| P4
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
```

Native admission:

```json
{
  "blueprint_id": "conditional_outcome",
  "rank": 0,
  "candidate_path": "diagnostics/blueprint_amendment_three_branches_final/admission_attempts/01_conditional_outcome/candidate.json",
  "overlay_path": "diagnostics/blueprint_amendment_three_branches_final/admission_attempts/01_conditional_outcome/hypothesis_overlay.json",
  "status": "REJECTED",
  "returncode": 1,
  "error": "Traceback (most recent call last):\n  File \"/var/folders/n8/ljhm3wm554j1rvxbx9vdvw600000gn/T/tmp_i3bpzmw/admit_blueprint.py\", line 27, in <module>\n    parse_world_model(\n  File \"/private/tmp/parliament-smoke-614af0c/global_workspace/world_state.py\", line 3120, in parse_world_model\n    raise WorldModelValidationError(\nglobal_workspace.world_validation.WorldModelValidationError: A0 omits quantity-bearing consequence 'two workers will survive'; admit a matching effect that records source quantities ['two']\n",
  "variant": "baseline"
}
```

## 3. flexible_graph (alternative; flexible)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: none

### flexible_graph candidate graph

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — one worker<br/>PERSON"]
    P3["P3 — the lever<br/>PROCESS"]
    P4["P4 — five workers<br/>HUMAN_GROUP"]
    P5["P5 — two workers<br/>HUMAN_GROUP"]
    P6["P6 — the button<br/>PROCESS"]
    A0["A0 — pulls the lever"]
    A1["A1 — does not pull the lever"]
    A2["A2 — presses the button"]
    E1["E1 — pulls the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — pulls the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — one worker will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E4["E4 — does not pull the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E5["E5 — does not pull the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — five workers will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E7["E7 — presses the button<br/>DIRECT · NEUTRAL · CERTAIN"]
    E8["E8 — presses the button<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E9["E9 — two workers will survive<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    CND1{"CND1 — If Maria pulls the lever<br/>POSITIVE"}
    CND2{"CND2 — If Maria does not pull the lever<br/>NEGATED"}
    CND3{"CND3 — If Maria presses the button<br/>POSITIVE"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    P1 -->|"actor"| A2
    A2 -->|"recipient"| P6
    A2 -.->|"effect"| E7
    A2 -.->|"effect"| E8
    A2 -.->|"effect"| E9
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P3
    E5 -.->|"affects"| P3
    E6 -.->|"affects"| P4
    E7 -.->|"affects"| P6
    E8 -.->|"affects"| P6
    E9 -.->|"affects"| P5
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    E7 -->|"ENABLES"| E8
    E8 -->|"ENABLES"| E9
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
    CND3 -.->|"branch scope"| A2
```

Native admission:

```json
{
  "blueprint_id": "flexible_graph",
  "rank": 1,
  "candidate_path": "diagnostics/blueprint_amendment_three_branches_final/admission_attempts/02_flexible_graph/candidate.json",
  "overlay_path": "diagnostics/blueprint_amendment_three_branches_final/admission_attempts/02_flexible_graph/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 9,
    "causal_links": 6,
    "frozen_trace_valid": true,
    "fingerprint_sha256": "22ab5c2e95282722c0cbe121c1075751e3954cb6410480c4ab9606a78cdbea18"
  },
  "trace_path": "diagnostics/blueprint_amendment_three_branches_final/admission_attempts/02_flexible_graph/frozen_world_trace.json",
  "variant": "flexible"
}
```
