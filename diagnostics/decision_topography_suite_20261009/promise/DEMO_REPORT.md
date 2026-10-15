# Parliament demo

Status: **FRAMEWORK_PARTIAL** (recorded_pipeline_outcome).

Native judgment status: NOT_ADJUDICATED.

## Scenario

A clinic has one dose of medicine. Ada can give the medicine to Ben or Cara, but not both. If Ada gives the medicine to Ben, Ben has a 95% chance of survival. If Ada gives the medicine to Cara, Cara has a 5% chance of survival. The patient who does not get the medicine will die. Ada promised Cara that she would give Cara the medicine.

## What Parliament thinks is happening

Actions: Ada gives the medicine to Ben; Ada gives the medicine to Cara

Parties: Ada, one dose of medicine, Ben, Cara

These are alternative action branches and their stipulated outcomes, not claims that both actions occurred.

- Under A0, Ben: Ada gives the medicine to Ben (CERTAIN)
- Under A0, Ben: Ben has a 95% chance of survival (PROBABILISTIC)
- Under A1, Cara: Ada gives the medicine to Cara (CERTAIN)
- Under A1, Cara: Cara has a 5% chance of survival (PROBABILISTIC)
- Under A0, Cara: does not receive one dose of medicine (CERTAIN)
- Under A1, Ben: does not receive one dose of medicine (CERTAIN)
- Under A0, Cara: will die (CERTAIN)
- Under A1, Ben: will die (CERTAIN)
- Ada gives the medicine to Ben → ENABLES → Ben has a 95% chance of survival
- Ada gives the medicine to Cara → ENABLES → Cara has a 5% chance of survival
- does not receive one dose of medicine → ENABLES → will die
- does not receive one dose of medicine → ENABLES → will die
- Ada gives the medicine to Ben → CAUSES → does not receive one dose of medicine
- Ada gives the medicine to Cara → CAUSES → does not receive one dose of medicine

## World-state diagram

### Frozen world supplied to framework assessment

```mermaid
flowchart LR
    P1["P1 — Ada<br/>PERSON"]
    P2["P2 — one dose of medicine<br/>RESOURCE"]
    P3["P3 — Ben<br/>PERSON"]
    P4["P4 — Cara<br/>PERSON"]
    A0["A0 — Ada gives the medicine to Ben"]
    A1["A1 — Ada gives the medicine to Cara"]
    E1["E1 — Ada gives the medicine to Ben<br/>DIRECT · NEUTRAL · CERTAIN"]
    E2["E2 — Ben has a 95% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E3["E3 — Ada gives the medicine to Cara<br/>DIRECT · NEUTRAL · CERTAIN"]
    E4["E4 — Cara has a 5% chance of survival<br/>DOWNSTREAM · BENEFICIAL · PROBABILISTIC"]
    E5["E5 — does not receive one dose of medicine<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
    E6["E6 — does not receive one dose of medicine<br/>DOWNSTREAM · ADVERSE · CERTAIN"]
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


## What is grounded

Admitted source copies and structurally derived records follow. Exact excerpts establish traceability; they do not independently prove every interpretation.

- Ada gives the medicine to Ben — DIRECT_COPY; source: “If Ada gives the medicine to Ben, Ben has a 95% chance of survival.”
- Ben has a 95% chance of survival — SOURCE_STIPULATED_CAUSAL; source: “If Ada gives the medicine to Ben, Ben has a 95% chance of survival.”
- Ada gives the medicine to Cara — DIRECT_COPY; source: “If Ada gives the medicine to Cara, Cara has a 5% chance of survival.”
- Cara has a 5% chance of survival — SOURCE_STIPULATED_CAUSAL; source: “If Ada gives the medicine to Cara, Cara has a 5% chance of survival.”
- does not receive one dose of medicine — EXCLUSIVE_ALLOCATION_COMPLEMENT; source: “A clinic has one dose of medicine.; Ada can give the medicine to Ben or Cara, but not both.; If Ada gives the medicine to Ben, Ben has a 95% chance of survival.”
- does not receive one dose of medicine — EXCLUSIVE_ALLOCATION_COMPLEMENT; source: “A clinic has one dose of medicine.; Ada can give the medicine to Ben or Cara, but not both.; If Ada gives the medicine to Cara, Cara has a 5% chance of survival.”
- will die — SOURCE_STIPULATED_CAUSAL; source: “The patient who does not get the medicine will die.”
- will die — SOURCE_STIPULATED_CAUSAL; source: “The patient who does not get the medicine will die.”
- Stipulated source claim: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC
- Stipulated source claim: 95% chance — Ben has a 95% chance of survival
- Stipulated source claim: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC
- Stipulated source claim: 5% chance — Cara has a 5% chance of survival
- Stipulated source claim: A clinic has one dose of medicine.
- Stipulated source claim: Ada promised Cara that she would give Cara the medicine.
- Stipulated source claim: Ada can give the medicine to Ben or Cara, but not both.
- Stipulated source claim: The patient who does not get the medicine will die.
- Stipulated source claim: If Ada gives the medicine to Ben, Ben has a 95% chance of survival.
- Stipulated source claim: If Ada gives the medicine to Cara, Cara has a 5% chance of survival.

## What remains uncertain

- The utilitarian ledger contains unknown welfare magnitude or duration; survivor arithmetic does not settle total welfare.
- A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis
- A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- A1 ADJUDICATION_CALIBRATION: perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis
- A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises
- what makes the positive duty perfect rather than imperfect
- cara: perfect duty to keep promises ↔ ben: imperfect duty of beneficence
- unresolved Care ranking must remain provisional

## Framework views

### utilitarian

Position: **ABSTAIN**. Recorded preference: withheld (not necessarily operative).

Framework did not select a live action. UNRESOLVED FRAMEWORK CONFLICT: live competing derivations. No comparative leaning is yet grounded. Interim working action recorded without becoming a ranking: Ada gives the medicine to Ben. Max expected survival. Unresolved contestation: prefer “Ada gives the medicine to Ben” when it yields greater expected survivors than “Ada gives the medicine to Cara”. Recorded comparison — “Ada gives the medicine to Ben”: Gives Ben a 95% survival chance; maximises expected lives saved.; “Ada gives the medicine to Cara”: Keeps Ada’s promise to Cara; still offers small 5% chance of saving one life. Unadmitted magnitude cannot decide the ranking; the claim remains a reversal boundary.

[Full recorded explanation (Markdown)](arguments/utilitarian.md).

Assumptions / interpretive commitments: Recorded interpretive commitment: prefer “Ada gives the medicine to Ben” when it yields greater expected survivors than “Ada gives the medicine to Cara”; Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.

Dependencies: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC; Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC

Live objection: None separately recorded.

Abstention boundary (no ranking to reverse): Promise-keeping given overriding moral weight; Ben survival probability falls to ≤5%

### deontological

Position: **ABSTAIN**. Recorded preference: Ada gives the medicine to Cara (not necessarily operative).

The agent assesses Ada’s duty “perfect duty to keep promises” toward cara. Ada is the duty-bearer; cara is the protected party. The competing obligation being assessed is Ada’s “imperfect duty of beneficence” toward ben. For Ada’s option “Ada gives the medicine to Ben”, the ledger records conflicted. The agent’s stated reason is: breaks a promise. For Ada’s option “Ada gives the medicine to Cara”, the ledger records conflicted. The agent’s stated reason is: keeps the promise. The ledger retains unresolved issues; this does not establish that one duty overrides the other.

[Full recorded explanation (Markdown)](arguments/deontological.md).

Assumptions / interpretive commitments: Recorded interpretive commitment: prefer “Ada gives the medicine to Cara” because perfect duty of promise-keeping outranks imperfect beneficence; Recorded interpretive commitment: beneficence numbers are secondary to promise duty; Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.

Dependencies: Ada promised Cara that she would give Cara the medicine.

Live objection: A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis; A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises; A1 ADJUDICATION_CALIBRATION: perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis; A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises; what makes the positive duty perfect rather than imperfect; cara: perfect duty to keep promises ↔ ben: imperfect duty of beneficence

Would reverse the recorded preference: The agent reports a switch from “Ada gives the medicine to Cara” to “Ada gives the medicine to Ben”. Condition: would reverse if promise-keeping not obligatory. This is an agent-reported condition, not an independently established threshold.

### care

Position: **ABSTAIN**. Recorded preference: Ada gives the medicine to Ben (not necessarily operative).

Expected CARE_RELATIONSHIP_LEDGER, but no matching committed ledger is operative. competence saves life. prefer “Ada gives the medicine to Ben” when it far better meets the same urgent need

[Full recorded explanation (Markdown)](arguments/care.md).

Assumptions / interpretive commitments: Recorded interpretive commitment: prefer “Ada gives the medicine to Ben” when it far better meets the same urgent need; Recorded interpretive commitment: probability difference informs competence and survival likelihood; Run assumption: The stated scenario facts are accepted as givens for this analysis; their accuracy outside this scenario has not been verified.

The scenario’s stated facts are treated as givens for this run. No separate record says whether they were observed, reported by someone, or supplied as simulation facts. This is a provenance limitation, not a claim that the stated counts are missing.

Dependencies: Not separately recorded.

Live objection: unresolved Care ranking must remain provisional

Would reverse the recorded preference: The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: promise‐keeping prioritized over competence. This is an agent-reported condition, not an independently established threshold.; The agent reports a switch from “Ada gives the medicine to Ben” to “Ada gives the medicine to Cara”. Condition: Cara’s survival chance rises near Ben’s. This is an agent-reported condition, not an independently established threshold.

## Agreement and disagreement

Recorded preferences differ.

Framework-local normative arguments and empirical dependencies are shown above; no automatic causal classification is claimed.

- deontological: normative dependency — would reverse if promise-keeping not obligatory
- care: normative dependency — promise‐keeping prioritized over competence

## Decision hinges

- utilitarian: Promise-keeping given overriding moral weight (abstention boundary, no ranking to reverse; `cycles/0/candidates/0/normative_reversal_threshold`).
- deontological: would reverse if promise-keeping not obligatory (would reverse a recorded preference; `cycles/0/candidates/1/normative_reversal_threshold`).
- care: promise‐keeping prioritized over competence (would reverse a recorded preference; `cycles/0/candidates/2/normative_reversal_threshold`).

## What Parliament did not decide

- This report does not adjudicate a governing moral verdict.
- Ledger admission does not prove the argument or supply missing welfare units.
- Unresolved objections remain live; reported reversal thresholds are not source facts.

## Topography

Plotted coordinates are jackknife 95% intervals of Decisions-API probabilities. Each replicate ablates one contiguous tenth of the Parliament packet text. The marker is the jackknife mean. A refused or missing replicate is omitted, not zero. Attractors require a unique maximum whose interval sits entirely above the others on that axis.

Recorded action loci only. Missing axes are omitted; UNKNOWN is not plotted as zero.

- A0 Ada gives the medicine to Ben — care: 0.541 [0.265, 0.817]; welfare: 1.000 [1.000, 1.000]; obligation: 0.411 [0.054, 0.768]
- A1 Ada gives the medicine to Cara — care: 0.268 [0.033, 0.503]; welfare: 0.000 [0.000, 0.000]; obligation: 0.526 [0.390, 0.662]

Degenerate: care_welfare_probability_copy
Attractors: none

- Plotted axes are Decisions probabilities over the Parliament packet, not ledger ordinals.
- Care and welfare probability rankings coincide; this is marked degenerate, not an attractor.
- Axis maxima use non-overlapping jackknife intervals; overlapping intervals do not yield a unique maximum.
- Most hinging packet tenth: A0 obligation fold 2 (Δ=0.091). Omitting that slice moved the leave-one-out estimate the most.

## Pipeline stop and inspection

FRAMEWORK_PARTIAL

Missing slots: none recorded

Competing admitted interpretations: none recorded

Inspection artifacts (saved replay references its original records):

- [framework_generation.json](/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/argument_quality_pilot_fixes_20261006/competing_promise/standard/framework_generation.json)
- [conflict_graph.md](/Users/benjaminkowal/Documents/Python/Python Coding/RL_Algorithms/diagnostics/argument_quality_pilot_fixes_20261006/competing_promise/standard/conflict_graph.md)
