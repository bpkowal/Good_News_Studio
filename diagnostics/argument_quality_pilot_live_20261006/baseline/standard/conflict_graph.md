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
    n25["PROPOSITION: will die; affected subject: Cara"]
    n26["PROPOSITION: will die; affected subject: Ben"]
    n27["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n28["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n29["PROPOSITION: Ben are entrusted to the decision maker"]
    n30["PROPOSITION: Ada gives the medicine to Ben; affected subject: Ben"]
    n31["PROPOSITION: Ben has a 95% chance of survival; affected subject: Ben; modality: PROBABILISTIC"]
    n32["PROPOSITION: Cara has a 5% chance of survival; affected subject: Cara; modality: PROBABILISTIC"]
    n33["CLAIM: utilitarian: A0 assessment"]
    n34["CLAIM: utilitarian: A0 assessment"]
    n35["CLAIM: utilitarian: A0 assessment"]
    n36["CLAIM: utilitarian: A0 assessment"]
    n37["CLAIM: utilitarian: A1 assessment"]
    n38["CLAIM: utilitarian: A1 assessment"]
    n39["CLAIM: utilitarian: A1 assessment"]
    n40["CLAIM: utilitarian: A1 assessment"]
    n41["CLAIM: deontological: A0 REQUIRED"]
    n42["FRAMEWORK_COMMITMENT: imperfect duty of beneficence/rescue"]
    n43["FRAMEWORK_COMMITMENT: When only one can be treated, choose more effective rescue"]
    n44["CLAIM: deontological: A1 CONFLICTED"]
    n45["FRAMEWORK_COMMITMENT: Choosing far less effective rescue violates beneficence"]
    n46["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n47["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n48["UNRESOLVED_CONFLICT: ben: imperfect duty of beneficence/rescue ↔ cara: beneficence toward Cara"]
    n49["UNRESOLVED_CONFLICT: A1 prohibited by imperfect duty of beneficence/rescue ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n50["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n51["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n52["CLAIM: care: A0 RESPONSIVE"]
    n53["FRAMEWORK_COMMITMENT: control establishes responsibility"]
    n54["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n55["CLAIM: care: A1 NEGLECTFUL"]
    n56["FRAMEWORK_COMMITMENT: responsibility ignored"]
    n57["UNRESOLVED_CONFLICT: decisive counts lack relational comparability"]
    n58["UNRESOLVED_CONFLICT: numeric reversal threshold lacks a scenario-grounded quantity"]
    n59["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n60["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n42 -->|"SUPPORT"| n41
    n43 -->|"SUPPORT"| n41
    n42 -->|"SUPPORT"| n44
    n45 -->|"SUPPORT"| n44
    n46 -->|"OBJECTION"| n44
    n47 -->|"OBJECTION"| n44
    n0 -->|"SUPPORT"| n52
    n1 -->|"SUPPORT"| n52
    n52 -->|"DEPENDENCY"| n31
    n52 -->|"DEPENDENCY"| n32
    n53 -->|"SUPPORT"| n52
    n54 -->|"SUPPORT"| n52
    n5 -->|"SUPPORT"| n55
    n7 -->|"SUPPORT"| n55
    n55 -->|"DEPENDENCY"| n31
    n55 -->|"DEPENDENCY"| n32
    n56 -->|"SUPPORT"| n55
    n54 -->|"SUPPORT"| n55
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
