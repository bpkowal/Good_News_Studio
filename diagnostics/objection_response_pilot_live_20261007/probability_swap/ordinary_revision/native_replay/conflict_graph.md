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
    n25["PROPOSITION: Ben has a 5% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n26["PROPOSITION: Cara has a 95% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n27["PROPOSITION: will die; affected subject: Cara"]
    n28["PROPOSITION: will die; affected subject: Ben"]
    n29["PROPOSITION: Ben has a 5% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n30["PROPOSITION: Cara has a 95% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n31["CLAIM: utilitarian: A0 assessment"]
    n32["CLAIM: utilitarian: A0 assessment"]
    n33["CLAIM: utilitarian: A0 assessment"]
    n34["CLAIM: utilitarian: A0 assessment"]
    n35["CLAIM: utilitarian: A1 assessment"]
    n36["CLAIM: utilitarian: A1 assessment"]
    n37["CLAIM: utilitarian: A1 assessment"]
    n38["CLAIM: utilitarian: A1 assessment"]
    n39["CLAIM: deontological: A0 PROHIBITED"]
    n40["FRAMEWORK_COMMITMENT: imperfect duty of beneficence"]
    n41["FRAMEWORK_COMMITMENT: must aid as effectively as possible"]
    n42["CLAIM: deontological: A1 REQUIRED"]
    n43["CLAIM: care: A0 NEGLECTFUL"]
    n44["FRAMEWORK_COMMITMENT: control creates duty"]
    n45["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n46["CLAIM: care: A1 RESPONSIVE"]
    n40 -->|"SUPPORT"| n39
    n41 -->|"SUPPORT"| n39
    n40 -->|"SUPPORT"| n42
    n41 -->|"SUPPORT"| n42
    n4 -->|"SUPPORT"| n43
    n6 -->|"SUPPORT"| n43
    n43 -->|"DEPENDENCY"| n29
    n43 -->|"DEPENDENCY"| n30
    n44 -->|"SUPPORT"| n43
    n45 -->|"SUPPORT"| n43
    n2 -->|"SUPPORT"| n46
    n3 -->|"SUPPORT"| n46
    n46 -->|"DEPENDENCY"| n29
    n46 -->|"DEPENDENCY"| n30
    n44 -->|"SUPPORT"| n46
    n45 -->|"SUPPORT"| n46
```

Proposed review queue (at most two issues):

