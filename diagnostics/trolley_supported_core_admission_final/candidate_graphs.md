# Candidate world attempts

Chosen blueprint: **omission_harm**  
Chooser status: **FILLED**

## 1. rescue_contrast (alternative)

Template status: **PARTIAL**  
Proposal status: **WITHHELD**  
Contract valid: **true**  
Missing slots: `first_saved`, `second_saved`, `rescue_exclusivity`, `first_rescue_action`, `second_rescue_action`, `first_harm`, `second_benefit`, `scene`

World withheld:

- Required slots remain unfilled: first_saved, second_saved, rescue_exclusivity, first_rescue_action, second_rescue_action, second_benefit

## 2. omission_harm (selected)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: none

### omission_harm candidate graph

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — One worker<br/>PERSON"]
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
    AV3["AV3 — averts alternative harm to One worker<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
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

- `AV3`: averts alternative harm to One worker; unresolved_derivation_assumptions, unsupported_outcome_transformation, hypothesized_averted_harm
- `AV6`: averts alternative harm to five workers; unresolved_derivation_assumptions, unsupported_outcome_transformation, hypothesized_averted_harm

### Supported core for admission

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — One worker<br/>PERSON"]
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
  "blueprint_id": "omission_harm",
  "rank": 1,
  "candidate_path": "diagnostics/trolley_supported_core_admission_final/admission_attempts/01_omission_harm/candidate.json",
  "overlay_path": "diagnostics/trolley_supported_core_admission_final/admission_attempts/01_omission_harm/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 6,
    "causal_links": 4,
    "frozen_trace_valid": true,
    "fingerprint_sha256": "ba13eb0eb78e08f96a55153a7339379c37009b68123275a54b953fca8bea71c8"
  },
  "trace_path": "diagnostics/trolley_supported_core_admission_final/admission_attempts/01_omission_harm/frozen_world_trace.json"
}
```

## 3. conditional_outcome (alternative)

Template status: **FILLED**  
Proposal status: **FILLED**  
Contract valid: **true**  
Missing slots: `chance`

### conditional_outcome candidate graph

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — One worker<br/>PERSON"]
    P3["P3 — the lever<br/>PROCESS"]
    P4["P4 — Five workers<br/>HUMAN_GROUP"]
    A0["A0 — pulls the lever"]
    A1["A1 — does not pull the lever"]
    E1["E1 — pulls the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — pulls the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E3["E3 — one worker will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E4["E4 — does not pull the lever<br/>DIRECT · NEUTRAL · CERTAIN"]
    E5["E5 — does not pull the lever<br/>DOWNSTREAM · NEUTRAL · CERTAIN"]
    E6["E6 — five workers will die<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    AV3["AV3 — averts alternative harm to One worker<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
    AV6["AV6 — averts alternative harm to Five workers<br/>DOWNSTREAM · BENEFICIAL · CERTAIN"]
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

- `AV3`: averts alternative harm to One worker; unresolved_derivation_assumptions, unsupported_outcome_transformation, hypothesized_averted_harm
- `AV6`: averts alternative harm to Five workers; unresolved_derivation_assumptions, unsupported_outcome_transformation, hypothesized_averted_harm

### Supported core for admission

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — One worker<br/>PERSON"]
    P3["P3 — the lever<br/>PROCESS"]
    P4["P4 — Five workers<br/>HUMAN_GROUP"]
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
  "rank": 2,
  "candidate_path": "diagnostics/trolley_supported_core_admission_final/admission_attempts/02_conditional_outcome/candidate.json",
  "overlay_path": "diagnostics/trolley_supported_core_admission_final/admission_attempts/02_conditional_outcome/hypothesis_overlay.json",
  "status": "ADMITTED",
  "admission": {
    "status": "COMMITTED",
    "world_model_status": "COMMITTED",
    "effects": 6,
    "causal_links": 4,
    "frozen_trace_valid": true,
    "fingerprint_sha256": "2fe4885ca8e22fd3aa390dac4e98a9e272320bb8be4ea3ca0d0838cb40664078"
  },
  "trace_path": "diagnostics/trolley_supported_core_admission_final/admission_attempts/02_conditional_outcome/frozen_world_trace.json"
}
```
