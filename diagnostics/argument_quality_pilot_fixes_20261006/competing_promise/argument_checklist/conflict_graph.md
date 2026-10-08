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
    n26["PROPOSITION: will die; affected subject: Cara"]
    n27["PROPOSITION: will die; affected subject: Ben"]
    n28["PROPOSITION: Ada promised Cara that she would give Cara the medicine."]
    n29["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n30["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n31["PROPOSITION: Ada promised Cara that she would give Cara the medicine."]
    n32["CLAIM: utilitarian: A0 assessment"]
    n33["CLAIM: utilitarian: A0 assessment"]
    n34["CLAIM: utilitarian: A0 assessment"]
    n35["CLAIM: utilitarian: A0 assessment"]
    n36["CLAIM: utilitarian: A1 assessment"]
    n37["CLAIM: utilitarian: A1 assessment"]
    n38["CLAIM: utilitarian: A1 assessment"]
    n39["CLAIM: utilitarian: A1 assessment"]
    n40["CLAIM: deontological: A0 PROHIBITED"]
    n41["FRAMEWORK_COMMITMENT: keep promises"]
    n42["FRAMEWORK_COMMITMENT: perfect duties override imperfect"]
    n43["CLAIM: deontological: A1 REQUIRED"]
    n40 -->|"DEPENDENCY"| n28
    n41 -->|"SUPPORT"| n40
    n42 -->|"SUPPORT"| n40
    n43 -->|"DEPENDENCY"| n28
    n41 -->|"SUPPORT"| n43
    n42 -->|"SUPPORT"| n43
```

Proposed review queue (at most two issues):

