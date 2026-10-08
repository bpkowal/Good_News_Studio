# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: Ada gives the medicine to Ben"]
    n1["WORLD_EFFECT: Ben has a 5% chance of survival"]
    n2["WORLD_EFFECT: Ada gives the medicine to Cara"]
    n3["WORLD_EFFECT: Cara has a 95% chance of survival"]
    n4["WORLD_EFFECT: does not receive one dose of medicine"]
    n5["WORLD_EFFECT: does not receive one dose of medicine"]
    n6["WORLD_EFFECT: will die"]
    n7["WORLD_EFFECT: will die"]
    n8["PROPOSITION: Ada gives the medicine to Ben; affected subject: Ben"]
    n9["PROPOSITION: Ben has a 5% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n10["PROPOSITION: 5% chance — Ben has a 5% chance of survival"]
    n11["PROPOSITION: Ada gives the medicine to Cara; affected subject: Cara"]
    n12["PROPOSITION: Cara has a 95% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n13["PROPOSITION: 95% chance — Cara has a 95% chance of survival"]
    n14["PROPOSITION: does not receive one dose of medicine; affected subject: Cara; magnitude or qualifier: one"]
    n15["PROPOSITION: does not receive one dose of medicine; affected subject: Ben; magnitude or qualifier: one"]
    n16["PROPOSITION: will die; affected subject: Cara"]
    n17["PROPOSITION: will die; affected subject: Ben"]
    n18["PROPOSITION: A clinic has one dose of medicine."]
    n19["PROPOSITION: If Ada gives the medicine to Cara, Cara has a 95% chance of survival."]
    n20["PROPOSITION: Ada can give the medicine to Ben or Cara, but not both."]
    n21["PROPOSITION: If Ada gives the medicine to Ben, Ben has a 5% chance of survival."]
    n22["PROPOSITION: The patient who does not get the medicine will die."]
    n23["PROPOSITION: Ben has a 5% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n24["PROPOSITION: Cara has a 95% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n25["PROPOSITION: will die; affected subject: Cara"]
    n26["PROPOSITION: will die; affected subject: Ben"]
    n27["PROPOSITION: Ben has a 5% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n28["PROPOSITION: Cara has a 95% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n29["PROPOSITION: will die; affected subject: Cara"]
    n30["PROPOSITION: will die; affected subject: Ben"]
    n31["PROPOSITION: The patient who does not get the medicine will die."]
    n32["PROPOSITION: Ben has a 5% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n33["PROPOSITION: Cara has a 95% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n34["PROPOSITION: Ada can give the medicine to Ben or Cara, but not both."]
    n35["PROPOSITION: The patient who does not get the medicine will die."]
    n36["CLAIM: utilitarian: A0 assessment"]
    n37["CLAIM: utilitarian: A0 assessment"]
    n38["CLAIM: utilitarian: A0 assessment"]
    n39["CLAIM: utilitarian: A0 assessment"]
    n40["CLAIM: utilitarian: A1 assessment"]
    n41["CLAIM: utilitarian: A1 assessment"]
    n42["CLAIM: utilitarian: A1 assessment"]
    n43["CLAIM: utilitarian: A1 assessment"]
    n44["CLAIM: deontological: A0 UNCERTAIN"]
    n45["FRAMEWORK_COMMITMENT: imperfect duty of beneficence"]
    n46["FRAMEWORK_COMMITMENT: aid duty guides when feasible"]
    n47["CLAIM: deontological: A1 REQUIRED"]
    n48["FRAMEWORK_COMMITMENT: must do most effective aid"]
    n49["UNRESOLVED_CONFLICT: A0 PERMISSIBLE conflicts with governing COMPETING relation VIOLATES; committed as UNCERTAIN"]
    n50["CLAIM: care: A0 MIXED"]
    n51["FRAMEWORK_COMMITMENT: controller of scarce life-saving resource owes response"]
    n52["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n53["CLAIM: care: A1 RESPONSIVE"]
    n54["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n55["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n36 -->|"DEPENDENCY"| n23
    n36 -->|"DEPENDENCY"| n24
    n37 -->|"DEPENDENCY"| n23
    n37 -->|"DEPENDENCY"| n24
    n38 -->|"DEPENDENCY"| n23
    n38 -->|"DEPENDENCY"| n24
    n39 -->|"DEPENDENCY"| n23
    n39 -->|"DEPENDENCY"| n24
    n40 -->|"DEPENDENCY"| n23
    n40 -->|"DEPENDENCY"| n24
    n41 -->|"DEPENDENCY"| n23
    n41 -->|"DEPENDENCY"| n24
    n42 -->|"DEPENDENCY"| n23
    n42 -->|"DEPENDENCY"| n24
    n43 -->|"DEPENDENCY"| n23
    n43 -->|"DEPENDENCY"| n24
    n45 -->|"SUPPORT"| n44
    n46 -->|"SUPPORT"| n44
    n45 -->|"SUPPORT"| n47
    n48 -->|"SUPPORT"| n47
    n0 -->|"SUPPORT"| n50
    n1 -->|"SUPPORT"| n50
    n50 -->|"DEPENDENCY"| n32
    n50 -->|"DEPENDENCY"| n33
    n51 -->|"SUPPORT"| n50
    n52 -->|"SUPPORT"| n50
    n2 -->|"SUPPORT"| n53
    n3 -->|"SUPPORT"| n53
    n53 -->|"DEPENDENCY"| n32
    n53 -->|"DEPENDENCY"| n33
    n51 -->|"SUPPORT"| n53
    n52 -->|"SUPPORT"| n53
```

Proposed review queue (at most two issues):

- A0 PERMISSIBLE conflicts with governing COMPETING relation VIOLATES; committed as UNCERTAIN
- A0 competing Care claim remains contested
