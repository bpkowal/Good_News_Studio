# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: If Ada gives the medicine to Ben,"]
    n1["WORLD_EFFECT: Ben has a 95% chance of survival"]
    n2["WORLD_EFFECT: If Ada gives the medicine to Cara,"]
    n3["WORLD_EFFECT: Cara has a 5% chance of survival"]
    n4["WORLD_EFFECT: does not receive one dose of medicine"]
    n5["WORLD_EFFECT: does not receive one dose of medicine"]
    n6["WORLD_EFFECT: will die"]
    n7["WORLD_EFFECT: will die"]
    n8["PROPOSITION: If Ada gives the medicine to Ben,; affected subject: Ben"]
    n9["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n10["PROPOSITION: 95% chance — Ben has a 95% chance of survival"]
    n11["PROPOSITION: If Ada gives the medicine to Cara,; affected subject: Cara"]
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
    n25["PROPOSITION: will die; affected subject: Cara"]
    n26["PROPOSITION: will die; affected subject: Ben"]
    n27["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n28["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n29["PROPOSITION: Ada can give the medicine to Ben or Cara, but not both."]
    n30["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n31["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n32["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n33["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n34["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n35["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n36["PROPOSED_REVISION: care: disputed full revision; not operative"]
    n37["CLAIM: utilitarian: A0 assessment"]
    n38["CLAIM: utilitarian: A0 assessment"]
    n39["CLAIM: utilitarian: A0 assessment"]
    n40["CLAIM: utilitarian: A0 assessment"]
    n41["CLAIM: utilitarian: A1 assessment"]
    n42["CLAIM: utilitarian: A1 assessment"]
    n43["CLAIM: utilitarian: A1 assessment"]
    n44["CLAIM: utilitarian: A1 assessment"]
    n45["CLAIM: care: A0 RESPONSIVE"]
    n46["FRAMEWORK_COMMITMENT: Control creates responsibility to respond competently"]
    n47["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n48["CLAIM: care: A1 MIXED"]
    n49["FRAMEWORK_COMMITMENT: Control creates responsibility to respond"]
    n50["UNRESOLVED_CONFLICT: decisive counts lack relational comparability"]
    n51["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n52["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n53["CLAIM: virtue: A0 EXEMPLIFIES"]
    n54["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n55["CLAIM: virtue: A1 MIXED"]
    n56["CLAIM: rawlsian: A0 assessment"]
    n57["FRAMEWORK_COMMITMENT: ORIGINAL_POSITION_PUBLIC_RULE"]
    n58["CLAIM: rawlsian: A1 assessment"]
    n59["UNRESOLVED_CONFLICT: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n60["UNRESOLVED_CONFLICT: A0 IMPROVES lacked directional support; committed as UNCERTAIN"]
    n61["UNRESOLVED_CONFLICT: A1 WORSENS lacked directional support; committed as UNCERTAIN"]
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
    n44 -->|"DEPENDENCY"| n23
    n44 -->|"DEPENDENCY"| n24
    n0 -->|"SUPPORT"| n45
    n1 -->|"SUPPORT"| n45
    n45 -->|"DEPENDENCY"| n30
    n45 -->|"DEPENDENCY"| n31
    n46 -->|"SUPPORT"| n45
    n47 -->|"SUPPORT"| n45
    n2 -->|"SUPPORT"| n48
    n3 -->|"SUPPORT"| n48
    n48 -->|"DEPENDENCY"| n30
    n48 -->|"DEPENDENCY"| n31
    n49 -->|"SUPPORT"| n48
    n47 -->|"SUPPORT"| n48
    n54 -->|"SUPPORT"| n53
    n54 -->|"SUPPORT"| n55
    n57 -->|"SUPPORT"| n56
    n57 -->|"SUPPORT"| n58
```

Proposed review queue (at most two issues):

- decisive counts lack relational comparability
- A0 competing Care claim remains contested
