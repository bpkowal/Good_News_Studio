# Candidate world attempts

Chosen blueprint: **conditional_outcome**  
Chooser status: **FILLED**

## 1. conditional_outcome (selected)

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
    AV3["AV3 — averts alternative harm to one worker<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    AV6["AV6 — averts alternative harm to five workers<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    CND1{"CND1 — If Maria pulls the lever<br/>POSITIVE"}
    CND2{"CND2 — If Maria does not pull the lever<br/>NEGATED"}
    P1 -->|"actor"| A0
    A0 -->|"recipient"| P3
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    A0 -.->|"effect"| AV6
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P3
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    A1 -.->|"effect"| AV3
    E1 -.->|"affects"| P3
    E2 -.->|"affects"| P3
    E3 -.->|"affects"| P2
    E4 -.->|"affects"| P3
    E5 -.->|"affects"| P3
    E6 -.->|"affects"| P4
    AV3 -.->|"affects"| P2
    AV6 -.->|"affects"| P4
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
```

Hypothesis overlay (not submitted as world effects):

- `AV3`: averts alternative harm to one worker; unresolved_derivation_assumptions, unsupported_outcome_transformation, hypothesized_averted_harm
- `AV6`: averts alternative harm to five workers; unresolved_derivation_assumptions, unsupported_outcome_transformation, hypothesized_averted_harm

### Supported core for admission

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
  "candidate_path": "diagnostics/blueprint_amendment_three_branches/admission_attempts/01_conditional_outcome/candidate.json",
  "overlay_path": "diagnostics/blueprint_amendment_three_branches/admission_attempts/01_conditional_outcome/hypothesis_overlay.json",
  "status": "REJECTED",
  "returncode": 1,
  "error": "Traceback (most recent call last):\n  File \"/var/folders/n8/ljhm3wm554j1rvxbx9vdvw600000gn/T/tmpwr5f_wuh/admit_blueprint.py\", line 27, in <module>\n    parse_world_model(\n  File \"/private/tmp/parliament-smoke-614af0c/global_workspace/world_state.py\", line 3120, in parse_world_model\n    raise WorldModelValidationError(\nglobal_workspace.world_validation.WorldModelValidationError: A0 omits quantity-bearing consequence 'two workers will survive'; admit a matching effect that records source quantities ['two']\n"
}
```

## 0.9. conditional_outcome (alternative)

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
    A3["A3 — presses the button"]
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
    P1 -->|"actor"| A3
    A3 -->|"recipient"| P6
    A3 -.->|"effect"| E7
    A3 -.->|"effect"| E8
    A3 -.->|"effect"| E9
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
    CND3 -.->|"branch scope"| A3
```

Native admission:

```json
{
  "blueprint_id": "conditional_outcome",
  "rank": -0.1,
  "candidate_path": "diagnostics/blueprint_amendment_three_branches/admission_attempts/00_conditional_outcome/candidate.json",
  "overlay_path": "diagnostics/blueprint_amendment_three_branches/admission_attempts/00_conditional_outcome/hypothesis_overlay.json",
  "status": "REJECTED",
  "returncode": 1,
  "error": "Traceback (most recent call last):\n  File \"/var/folders/n8/ljhm3wm554j1rvxbx9vdvw600000gn/T/tmpwr5f_wuh/admit_blueprint.py\", line 27, in <module>\n    parse_world_model(\n  File \"/private/tmp/parliament-smoke-614af0c/global_workspace/world_state.py\", line 3120, in parse_world_model\n    raise WorldModelValidationError(\nglobal_workspace.world_validation.WorldModelValidationError: typed world actions must cover every canonical action exactly once; E7 cites unknown action A3; E8 cites unknown action A3; E9 cites unknown action A3; causal_link[4] lacks a valid owning action_id; causal_link[5] lacks a valid owning action_id\n"
}
```

## 2. flexible_graph (alternative)

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
    A2["A2 — does not pull the lever"]
    A3["A3 — presses the button"]
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
    P1 -->|"actor"| A2
    A2 -->|"recipient"| P3
    A2 -.->|"effect"| E4
    A2 -.->|"effect"| E5
    A2 -.->|"effect"| E6
    P1 -->|"actor"| A3
    A3 -->|"recipient"| P6
    A3 -.->|"effect"| E7
    A3 -.->|"effect"| E8
    A3 -.->|"effect"| E9
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
    CND2 -.->|"branch scope"| A2
    CND3 -.->|"branch scope"| A3
```

Native admission:

```json
{
  "blueprint_id": "flexible_graph",
  "rank": 1,
  "candidate_path": "diagnostics/blueprint_amendment_three_branches/admission_attempts/02_flexible_graph/candidate.json",
  "overlay_path": "diagnostics/blueprint_amendment_three_branches/admission_attempts/02_flexible_graph/hypothesis_overlay.json",
  "status": "REJECTED",
  "returncode": 1,
  "error": "Traceback (most recent call last):\n  File \"/var/folders/n8/ljhm3wm554j1rvxbx9vdvw600000gn/T/tmpwr5f_wuh/admit_blueprint.py\", line 27, in <module>\n    parse_world_model(\n  File \"/private/tmp/parliament-smoke-614af0c/global_workspace/world_state.py\", line 3120, in parse_world_model\n    raise WorldModelValidationError(\nglobal_workspace.world_validation.WorldModelValidationError: typed world actions must cover every canonical action exactly once; E7 cites unknown action A3; E8 cites unknown action A3; E9 cites unknown action A3; causal_link[4] lacks a valid owning action_id; causal_link[5] lacks a valid owning action_id\n"
}
```
