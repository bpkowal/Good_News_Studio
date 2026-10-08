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
    n19["PROPOSITION: Ada promised Cara that she would give Cara the medicine."]
    n20["PROPOSITION: Ada can give the medicine to Ben or Cara, but not both."]
    n21["PROPOSITION: The patient who does not get the medicine will die."]
    n22["PROPOSITION: If Ada gives the medicine to Ben, Ben has a 95% chance of survival."]
    n23["PROPOSITION: If Ada gives the medicine to Cara, Cara has a 5% chance of survival."]
    n24["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n25["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n26["PROPOSITION: The patient who does not get the medicine will die."]
    n27["PROPOSITION: Ada promised Cara that she would give Cara the medicine."]
    n28["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n29["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n30["PROPOSITION: Ada promised Cara that she would give Cara the medicine."]
    n31["CLAIM: utilitarian: A0 assessment"]
    n32["CLAIM: utilitarian: A0 assessment"]
    n33["CLAIM: utilitarian: A0 assessment"]
    n34["CLAIM: utilitarian: A0 assessment"]
    n35["CLAIM: utilitarian: A1 assessment"]
    n36["CLAIM: utilitarian: A1 assessment"]
    n37["CLAIM: utilitarian: A1 assessment"]
    n38["CLAIM: utilitarian: A1 assessment"]
    n39["CLAIM: deontological: A0 CONFLICTED"]
    n40["FRAMEWORK_COMMITMENT: perfect duty to keep promises"]
    n41["FRAMEWORK_COMMITMENT: perfect duties override imperfect duties"]
    n42["UNRESOLVED_CONFLICT: omission was classified as a perfect negative-duty violation without a separate basis"]
    n43["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n44["CLAIM: deontological: A1 CONFLICTED"]
    n45["UNRESOLVED_CONFLICT: perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis"]
    n46["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n47["UNRESOLVED_CONFLICT: cara: perfect duty to keep promises ↔ ben: imperfect duty of beneficence"]
    n48["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis"]
    n49["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n50["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis"]
    n51["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n31 -->|"DEPENDENCY"| n24
    n31 -->|"DEPENDENCY"| n25
    n32 -->|"DEPENDENCY"| n24
    n32 -->|"DEPENDENCY"| n25
    n33 -->|"DEPENDENCY"| n24
    n33 -->|"DEPENDENCY"| n25
    n34 -->|"DEPENDENCY"| n24
    n34 -->|"DEPENDENCY"| n25
    n35 -->|"DEPENDENCY"| n24
    n35 -->|"DEPENDENCY"| n25
    n36 -->|"DEPENDENCY"| n24
    n36 -->|"DEPENDENCY"| n25
    n37 -->|"DEPENDENCY"| n24
    n37 -->|"DEPENDENCY"| n25
    n38 -->|"DEPENDENCY"| n24
    n38 -->|"DEPENDENCY"| n25
    n39 -->|"DEPENDENCY"| n27
    n40 -->|"SUPPORT"| n39
    n41 -->|"SUPPORT"| n39
    n42 -->|"OBJECTION"| n39
    n43 -->|"OBJECTION"| n39
    n44 -->|"DEPENDENCY"| n27
    n40 -->|"SUPPORT"| n44
    n41 -->|"SUPPORT"| n44
    n45 -->|"OBJECTION"| n44
    n46 -->|"OBJECTION"| n44
```

Proposed review queue (at most two issues):

- omission was classified as a perfect negative-duty violation without a separate basis
- resolved adjudication rests on unsupported decisive premises
