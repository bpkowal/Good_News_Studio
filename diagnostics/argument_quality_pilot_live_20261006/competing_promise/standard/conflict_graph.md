# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: Ada gives the medicine to Ben"]
    n1["WORLD_EFFECT: Ben has a 95% chance of survival"]
    n2["WORLD_EFFECT: Ada gives the medicine to Cara"]
    n3["WORLD_EFFECT: Cara has a 5% chance of survival"]
    n4["WORLD_EFFECT: does not receive one dose of medicine"]
    n5["WORLD_EFFECT: does not receive one dose of medicine"]
    n6["WORLD_EFFECT: will die"]
    n7["WORLD_EFFECT: will die"]
    n8["PROPOSITION: Ada gives the medicine to Ben; affected subject: Ben"]
    n9["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n10["PROPOSITION: 95% chance — Ben has a 95% chance of survival"]
    n11["PROPOSITION: Ada gives the medicine to Cara; affected subject: Cara"]
    n12["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n13["PROPOSITION: 5% chance — Cara has a 5% chance of survival"]
    n14["PROPOSITION: does not receive one dose of medicine; affected subject: Cara; magnitude or qualifier: one"]
    n15["PROPOSITION: does not receive one dose of medicine; affected subject: Ben; magnitude or qualifier: one"]
    n16["PROPOSITION: will die; affected subject: Cara"]
    n17["PROPOSITION: will die; affected subject: Ben"]
    n18["PROPOSITION: A clinic has one dose of medicine."]
    n19["PROPOSITION: Ada can give the medicine to Ben or Cara, but not both."]
    n20["PROPOSITION: The patient who does not get the medicine will die."]
    n21["PROPOSITION: If Ada gives the medicine to Ben, Ben has a 95% chance of survival."]
    n22["PROPOSITION: If Ada gives the medicine to Cara, Cara has a 5% chance of survival."]
    n23["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n24["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n25["PROPOSITION: Ada promised Cara that she would give Cara the medicine."]
    n26["PROPOSITION: Ada gives the medicine to Ben; affected subject: Ben"]
    n27["PROPOSITION: Ada gives the medicine to Cara; affected subject: Cara"]
    n28["PROPOSITION: Cara are entrusted to the decision maker"]
    n29["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n30["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n31["CLAIM: utilitarian: A0 assessment"]
    n32["CLAIM: utilitarian: A0 assessment"]
    n33["CLAIM: utilitarian: A0 assessment"]
    n34["CLAIM: utilitarian: A0 assessment"]
    n35["CLAIM: utilitarian: A1 assessment"]
    n36["CLAIM: utilitarian: A1 assessment"]
    n37["CLAIM: utilitarian: A1 assessment"]
    n38["CLAIM: utilitarian: A1 assessment"]
    n39["CLAIM: deontological: A0 CONFLICTED"]
    n40["FRAMEWORK_COMMITMENT: keep promises"]
    n41["FRAMEWORK_COMMITMENT: perfect duty has priority"]
    n42["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n43["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n44["CLAIM: deontological: A1 CONFLICTED"]
    n45["FRAMEWORK_COMMITMENT: perfect duty governs"]
    n46["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n47["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n48["UNRESOLVED_CONFLICT: cara: keep promises ↔ ben: beneficence (rescue)"]
    n49["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n50["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n51["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n52["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n53["CLAIM: care: A0 RESPONSIVE"]
    n54["FRAMEWORK_COMMITMENT: Control creates direct responsibility to meet survival need effectively"]
    n55["FRAMEWORK_COMMITMENT: RESPONSIVE_FEASIBILITY"]
    n56["CLAIM: care: A1 MIXED"]
    n57["FRAMEWORK_COMMITMENT: Promise creates relational obligation"]
    n58["UNRESOLVED_CONFLICT: numeric reversal threshold lacks a scenario-grounded quantity"]
    n59["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n60["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n39 -->|"DEPENDENCY"| n25
    n40 -->|"SUPPORT"| n39
    n41 -->|"SUPPORT"| n39
    n42 -->|"OBJECTION"| n39
    n43 -->|"OBJECTION"| n39
    n44 -->|"DEPENDENCY"| n25
    n40 -->|"SUPPORT"| n44
    n45 -->|"SUPPORT"| n44
    n46 -->|"OBJECTION"| n44
    n47 -->|"OBJECTION"| n44
    n0 -->|"SUPPORT"| n53
    n1 -->|"SUPPORT"| n53
    n54 -->|"SUPPORT"| n53
    n55 -->|"SUPPORT"| n53
    n2 -->|"SUPPORT"| n56
    n3 -->|"SUPPORT"| n56
    n57 -->|"SUPPORT"| n56
    n55 -->|"SUPPORT"| n56
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
