# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: gives the blanket to Rowan"]
    n1["WORLD_EFFECT: Rowan has an 80% chance of survival"]
    n2["WORLD_EFFECT: gives the blanket to Sasha"]
    n3["WORLD_EFFECT: Sasha has a 60% chance of survival"]
    n4["WORLD_EFFECT: does not receive one thermal blanket"]
    n5["WORLD_EFFECT: does not receive one thermal blanket"]
    n6["WORLD_EFFECT: does not get the blanket"]
    n7["WORLD_EFFECT: does not get the blanket"]
    n8["PROPOSITION: promised would get the blanket"]
    n9["PROPOSITION: would get the blanket"]
    n10["PROPOSITION: gives the blanket to Rowan; affected subject: Rowan"]
    n11["PROPOSITION: Rowan has an 80% chance of survival; affected subject: Rowan; modality: PROBABILISTIC"]
    n12["PROPOSITION: 80% chance — Rowan has an 80% chance of survival"]
    n13["PROPOSITION: gives the blanket to Sasha; affected subject: Sasha"]
    n14["PROPOSITION: Sasha has a 60% chance of survival; affected subject: Sasha; modality: PROBABILISTIC"]
    n15["PROPOSITION: 60% chance — Sasha has a 60% chance of survival"]
    n16["PROPOSITION: does not receive one thermal blanket; affected subject: Sasha; magnitude or qualifier: one"]
    n17["PROPOSITION: does not receive one thermal blanket; affected subject: Rowan; magnitude or qualifier: one"]
    n18["PROPOSITION: does not get the blanket; affected subject: Sasha"]
    n19["PROPOSITION: does not get the blanket; affected subject: Rowan"]
    n20["PROPOSITION: Rescuer Ellis can give the blanket to Rowan or Sasha, but not both."]
    n21["PROPOSITION: A mountain rescue shelter has one thermal blanket."]
    n22["PROPOSITION: If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival."]
    n23["PROPOSITION: If Ellis gives the blanket to Sasha, Sasha has a 60% chance of survival."]
    n24["PROPOSITION: The hiker who does not get the blanket will die."]
    n25["PROPOSITION: Rowan has an 80% chance of survival; affected subject: Rowan; modality: PROBABILISTIC"]
    n26["PROPOSITION: Sasha has a 60% chance of survival; affected subject: Sasha; modality: PROBABILISTIC"]
    n27["PROPOSITION: promised would get the blanket"]
    n28["PROPOSITION: promised would get the blanket"]
    n29["PROPOSITION: promised would get the blanket"]
    n30["PROPOSITION: Rowan has an 80% chance of survival; affected subject: Rowan; modality: PROBABILISTIC"]
    n31["PROPOSITION: Sasha has a 60% chance of survival; affected subject: Sasha; modality: PROBABILISTIC"]
    n32["CLAIM: utilitarian: A0 assessment"]
    n33["CLAIM: utilitarian: A0 assessment"]
    n34["CLAIM: utilitarian: A0 assessment"]
    n35["CLAIM: utilitarian: A0 assessment"]
    n36["CLAIM: utilitarian: A1 assessment"]
    n37["CLAIM: utilitarian: A1 assessment"]
    n38["CLAIM: utilitarian: A1 assessment"]
    n39["CLAIM: utilitarian: A1 assessment"]
    n40["CLAIM: deontological: A0 CONFLICTED"]
    n41["FRAMEWORK_COMMITMENT: keep promises (fidelity)"]
    n42["FRAMEWORK_COMMITMENT: Perfect duties override imperfect"]
    n43["UNRESOLVED_CONFLICT: omission was classified as a perfect negative-duty violation without a separate basis"]
    n44["UNRESOLVED_CONFLICT: candidate maxim person-role conflicts with means classification"]
    n45["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n46["CLAIM: deontological: A1 CONFLICTED"]
    n47["UNRESOLVED_CONFLICT: perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis"]
    n48["UNRESOLVED_CONFLICT: merely-as-means conclusion lacks intended-as-means classification"]
    n49["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n50["UNRESOLVED_CONFLICT: ellis&#x27;s duty &#x27;keep promises (fidelity)&#x27; protecting sasha ↔ ellis&#x27;s competing duty &#x27;beneficence (rescue duty)&#x27; protecting rowan"]
    n51["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis"]
    n52["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: candidate maxim person-role conflicts with means classification"]
    n53["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n54["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: perfect positive duty lacks a right-correlative, undertaking, or grounded emergency basis"]
    n55["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification"]
    n56["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n57["PROPOSED_ARGUMENT: care: argument without committed ledger"]
    n58["UNRESOLVED_CONFLICT: No committed native ledger; assess the retained argument without treating it as operative."]
    n59["UNRESOLVED_CONFLICT: unresolved Care ranking must remain provisional"]
    n60["CLAIM: virtue: A0 EXEMPLIFIES"]
    n61["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n62["CLAIM: virtue: A1 MIXED"]
    n32 -->|"DEPENDENCY"| n25
    n32 -->|"DEPENDENCY"| n26
    n33 -->|"DEPENDENCY"| n25
    n33 -->|"DEPENDENCY"| n26
    n34 -->|"DEPENDENCY"| n25
    n34 -->|"DEPENDENCY"| n26
    n35 -->|"DEPENDENCY"| n25
    n35 -->|"DEPENDENCY"| n26
    n36 -->|"DEPENDENCY"| n25
    n36 -->|"DEPENDENCY"| n26
    n37 -->|"DEPENDENCY"| n25
    n37 -->|"DEPENDENCY"| n26
    n38 -->|"DEPENDENCY"| n25
    n38 -->|"DEPENDENCY"| n26
    n39 -->|"DEPENDENCY"| n25
    n39 -->|"DEPENDENCY"| n26
    n41 -->|"SUPPORT"| n40
    n42 -->|"SUPPORT"| n40
    n43 -->|"OBJECTION"| n40
    n44 -->|"OBJECTION"| n40
    n45 -->|"OBJECTION"| n40
    n41 -->|"SUPPORT"| n46
    n42 -->|"SUPPORT"| n46
    n47 -->|"OBJECTION"| n46
    n48 -->|"OBJECTION"| n46
    n49 -->|"OBJECTION"| n46
    n58 -->|"OBJECTION"| n57
    n59 -->|"OBJECTION"| n57
    n1 -->|"SUPPORT"| n60
    n0 -->|"SUPPORT"| n60
    n4 -->|"SUPPORT"| n60
    n6 -->|"SUPPORT"| n60
    n60 -->|"DEPENDENCY"| n29
    n61 -->|"SUPPORT"| n60
    n2 -->|"SUPPORT"| n62
    n5 -->|"SUPPORT"| n62
    n7 -->|"SUPPORT"| n62
    n62 -->|"DEPENDENCY"| n29
    n61 -->|"SUPPORT"| n62
```

Proposed review queue (at most two issues):

- omission was classified as a perfect negative-duty violation without a separate basis
- candidate maxim person-role conflicts with means classification
