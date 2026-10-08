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
    n25["PROPOSITION: If Ada gives the medicine to Ben, Ben has a 95% chance of survival."]
    n26["PROPOSITION: If Ada gives the medicine to Cara, Cara has a 5% chance of survival."]
    n27["PROPOSITION: Cara are entrusted to the decision maker"]
    n28["PROPOSITION: Ben are entrusted to the decision maker"]
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
    n39["CLAIM: deontological: A0 REQUIRED"]
    n40["FRAMEWORK_COMMITMENT: imperfect duty of beneficence"]
    n41["FRAMEWORK_COMMITMENT: more effective rescue required"]
    n42["CLAIM: deontological: A1 PROHIBITED"]
    n43["FRAMEWORK_COMMITMENT: maxim wastes scarce lifesaving means"]
    n44["UNRESOLVED_CONFLICT: numeric reversal threshold lacks a scenario-grounded quantity"]
    n45["UNRESOLVED_CONFLICT: A0 competing protected party lacks current-run action grounding"]
    n46["UNRESOLVED_CONFLICT: A1 competing protected party lacks current-run action grounding"]
    n47["CLAIM: care: A0 RESPONSIVE"]
    n48["FRAMEWORK_COMMITMENT: control creates direct responsibility"]
    n49["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n50["CLAIM: care: A1 NEGLECTFUL"]
    n51["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n52["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n40 -->|"SUPPORT"| n39
    n41 -->|"SUPPORT"| n39
    n40 -->|"SUPPORT"| n42
    n43 -->|"SUPPORT"| n42
    n0 -->|"SUPPORT"| n47
    n1 -->|"SUPPORT"| n47
    n47 -->|"DEPENDENCY"| n29
    n47 -->|"DEPENDENCY"| n30
    n48 -->|"SUPPORT"| n47
    n49 -->|"SUPPORT"| n47
    n2 -->|"SUPPORT"| n50
    n3 -->|"SUPPORT"| n50
    n50 -->|"DEPENDENCY"| n29
    n50 -->|"DEPENDENCY"| n30
    n48 -->|"SUPPORT"| n50
    n49 -->|"SUPPORT"| n50
```

Proposed review queue (at most two issues):

- numeric reversal threshold lacks a scenario-grounded quantity
- A0 competing protected party lacks current-run action grounding
