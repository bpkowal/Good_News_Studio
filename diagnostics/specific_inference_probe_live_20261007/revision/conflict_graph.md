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
    n24["PROPOSITION: Assume broken-promise harm &lt; 0.9 life-equivalents"]
    n25["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n26["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n27["PROPOSITION: Ada promised Cara that she would give Cara the medicine."]
    n28["CLAIM: utilitarian: A0 assessment"]
    n29["CLAIM: utilitarian: A0 assessment"]
    n30["CLAIM: utilitarian: A0 assessment"]
    n31["CLAIM: utilitarian: A0 assessment"]
    n32["CLAIM: utilitarian: A1 assessment"]
    n33["CLAIM: utilitarian: A1 assessment"]
    n34["CLAIM: utilitarian: A1 assessment"]
    n35["CLAIM: utilitarian: A1 assessment"]
    n36["UNRESOLVED_CONFLICT: numeric reversal threshold lacks a scenario-grounded quantity"]
    n28 -->|"DEPENDENCY"| n25
    n28 -->|"DEPENDENCY"| n26
    n28 -->|"DEPENDENCY"| n24
    n29 -->|"DEPENDENCY"| n25
    n29 -->|"DEPENDENCY"| n26
    n29 -->|"DEPENDENCY"| n24
    n30 -->|"DEPENDENCY"| n25
    n30 -->|"DEPENDENCY"| n26
    n30 -->|"DEPENDENCY"| n24
    n31 -->|"DEPENDENCY"| n25
    n31 -->|"DEPENDENCY"| n26
    n31 -->|"DEPENDENCY"| n24
    n32 -->|"DEPENDENCY"| n25
    n32 -->|"DEPENDENCY"| n26
    n32 -->|"DEPENDENCY"| n24
    n33 -->|"DEPENDENCY"| n25
    n33 -->|"DEPENDENCY"| n26
    n33 -->|"DEPENDENCY"| n24
    n34 -->|"DEPENDENCY"| n25
    n34 -->|"DEPENDENCY"| n26
    n34 -->|"DEPENDENCY"| n24
    n35 -->|"DEPENDENCY"| n25
    n35 -->|"DEPENDENCY"| n26
    n35 -->|"DEPENDENCY"| n24
```

Proposed review queue (at most two issues):

- numeric reversal threshold lacks a scenario-grounded quantity
