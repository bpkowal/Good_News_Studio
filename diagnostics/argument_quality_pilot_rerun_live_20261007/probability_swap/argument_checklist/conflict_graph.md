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
    n27["PROPOSITION: The patient who does not get the medicine will die."]
    n28["PROPOSITION: Ben has a 5% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n29["PROPOSITION: Cara has a 95% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n30["PROPOSITION: Cara (1) are entrusted to the decision maker"]
    n31["PROPOSITION: Ben (1) are entrusted to the decision maker"]
    n32["PROPOSITION: Ben has a 5% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n33["PROPOSITION: 5% chance — Ben has a 5% chance of survival"]
    n34["PROPOSITION: Cara has a 95% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n35["PROPOSITION: 95% chance — Cara has a 95% chance of survival"]
    n36["CLAIM: utilitarian: A0 assessment"]
    n37["CLAIM: utilitarian: A0 assessment"]
    n38["CLAIM: utilitarian: A0 assessment"]
    n39["CLAIM: utilitarian: A0 assessment"]
    n40["CLAIM: utilitarian: A1 assessment"]
    n41["CLAIM: utilitarian: A1 assessment"]
    n42["CLAIM: utilitarian: A1 assessment"]
    n43["CLAIM: utilitarian: A1 assessment"]
    n44["CLAIM: deontological: A0 PROHIBITED"]
    n45["FRAMEWORK_COMMITMENT: imperfect duty of beneficence"]
    n46["FRAMEWORK_COMMITMENT: reasonable aid must not be withheld when easily given"]
    n47["CLAIM: deontological: A1 REQUIRED"]
    n48["FRAMEWORK_COMMITMENT: aid others effectively when possible"]
    n49["CLAIM: care: A0 MIXED"]
    n50["FRAMEWORK_COMMITMENT: control creates responsibility"]
    n51["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n52["CLAIM: care: A1 RESPONSIVE"]
    n53["UNRESOLVED_CONFLICT: decisive counts lack relational comparability"]
    n54["UNRESOLVED_CONFLICT: A0 Care relation claims scenario grounding without a matching action effect"]
    n55["UNRESOLVED_CONFLICT: A1 Care relation claims scenario grounding without a matching action effect"]
    n45 -->|"SUPPORT"| n44
    n46 -->|"SUPPORT"| n44
    n45 -->|"SUPPORT"| n47
    n48 -->|"SUPPORT"| n47
    n49 -->|"DEPENDENCY"| n33
    n49 -->|"DEPENDENCY"| n35
    n49 -->|"DEPENDENCY"| n32
    n49 -->|"DEPENDENCY"| n34
    n50 -->|"SUPPORT"| n49
    n51 -->|"SUPPORT"| n49
    n52 -->|"DEPENDENCY"| n33
    n52 -->|"DEPENDENCY"| n35
    n52 -->|"DEPENDENCY"| n32
    n52 -->|"DEPENDENCY"| n34
    n50 -->|"SUPPORT"| n52
    n51 -->|"SUPPORT"| n52
```

Proposed review queue (at most two issues):

- decisive counts lack relational comparability
- A0 Care relation claims scenario grounding without a matching action effect
