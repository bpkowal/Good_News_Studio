# Parliament demo

Status: **INTERPRETIVE_BEAM** (epistemic_abstention).

Native judgment status: NOT_RUN.

## Scenario

Five workers are on the track. One worker is on the side track. Maria can pull the lever, but not the brake. If Maria pulls the lever, one worker will die. If Maria does not pull the lever, five workers will die.

## What Parliament thinks is happening

No unique frozen world is reported.

## World-state diagram

No unique world was frozen. The following are competing admitted interpretations; **none was sent to agents**.

### omission_harm — candidate only; not sent to agents

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

### conditional_outcome — candidate only; not sent to agents

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

### ability_permission — candidate only; not sent to agents

```mermaid
flowchart LR
    P1["P1 — Maria<br/>PERSON"]
    P2["P2 — the lever<br/>PERSON"]
    P3["P3 — one worker<br/>PERSON"]
    P4["P4 — the lever<br/>PROCESS"]
    P5["P5 — five workers<br/>HUMAN_GROUP"]
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
    A0 -->|"recipient"| P4
    A0 -.->|"effect"| E1
    A0 -.->|"effect"| E2
    A0 -.->|"effect"| E3
    P1 -->|"actor"| A1
    A1 -->|"recipient"| P4
    A1 -.->|"effect"| E4
    A1 -.->|"effect"| E5
    A1 -.->|"effect"| E6
    E1 -.->|"affects"| P4
    E2 -.->|"affects"| P4
    E3 -.->|"affects"| P3
    E4 -.->|"affects"| P4
    E5 -.->|"affects"| P4
    E6 -.->|"affects"| P5
    E1 -->|"ENABLES"| E2
    E2 -->|"ENABLES"| E3
    E4 -->|"ENABLES"| E5
    E5 -->|"ENABLES"| E6
    CND1 -.->|"branch scope"| A0
    CND2 -.->|"branch scope"| A1
    PR1([" PR1 — can pull the lever<br/>proposition · GOVERNED "])
    M0([" M0 — ability<br/>modal ·  "])
    P1 -.->|"BEARER"| M0
    M0 -.->|"GOVERNS"| PR1
```


## What is grounded

Admitted source copies and structurally derived records follow. Exact excerpts establish traceability; they do not independently prove every interpretation.


## What remains uncertain

No additional framework uncertainty was recorded; this is not proof of completeness.

## Framework views

## Agreement and disagreement

No operative preference recorded.

Framework-local normative arguments and empirical dependencies are shown above; no automatic causal classification is claimed.


## Decision hinges

No explicit reversal hinge was recorded. The report does not fabricate one.

## What Parliament did not decide

- This report does not adjudicate a governing moral verdict.
- Ledger admission does not prove the argument or supply missing welfare units.
- Unresolved objections remain live; reported reversal thresholds are not source facts.

## Pipeline stop and inspection

Competing admitted worlds are not separated or invariant.

Missing slots: none recorded

Competing admitted interpretations: omission_harm: pulls the lever, does not pull the lever; conditional_outcome: pulls the lever, does not pull the lever; ability_permission: pulls the lever, does not pull the lever

Inspection artifacts (saved replay references its original records):

- [candidate_graphs.md](/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/live_pipeline_latest_20261006/trolley/candidate_graphs.md)
- [candidate_attempts.json](/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/live_pipeline_latest_20261006/trolley/candidate_attempts.json)
- [question_assessment.json](/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/live_pipeline_latest_20261006/trolley/question_assessment.json)
- [world_beam.json](/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/live_pipeline_latest_20261006/trolley/world_beam.json)
- [preparation_manifest.json](/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/live_pipeline_latest_20261006/trolley/preparation_manifest.json)
