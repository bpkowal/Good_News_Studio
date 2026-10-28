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
    n26["PROPOSITION: 80% chance — Rowan has an 80% chance of survival"]
    n27["PROPOSITION: Sasha has a 60% chance of survival; affected subject: Sasha; modality: PROBABILISTIC"]
    n28["PROPOSITION: 60% chance — Sasha has a 60% chance of survival"]
    n29["PROPOSITION: If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival."]
    n30["PROPOSITION: If Ellis gives the blanket to Sasha, Sasha has a 60% chance of survival."]
    n31["PROPOSITION: promised would get the blanket"]
    n32["PROPOSITION: promised would get the blanket"]
    n33["PROPOSITION: promised would get the blanket"]
    n34["PROPOSITION: promised would get the blanket"]
    n35["PROPOSED_REVISION: rawlsian: disputed full revision; not operative"]
    n36["CLAIM: utilitarian: A0 assessment"]
    n37["CLAIM: utilitarian: A0 assessment"]
    n38["CLAIM: utilitarian: A0 assessment"]
    n39["CLAIM: utilitarian: A0 assessment"]
    n40["CLAIM: utilitarian: A1 assessment"]
    n41["CLAIM: utilitarian: A1 assessment"]
    n42["CLAIM: utilitarian: A1 assessment"]
    n43["CLAIM: utilitarian: A1 assessment"]
    n44["CLAIM: deontological: A0 CONFLICTED"]
    n45["FRAMEWORK_COMMITMENT: keep promises"]
    n46["FRAMEWORK_COMMITMENT: perfect duty overrides imperfect"]
    n47["UNRESOLVED_CONFLICT: omission was classified as a perfect negative-duty violation without a separate basis"]
    n48["UNRESOLVED_CONFLICT: candidate maxim person-role conflicts with means classification"]
    n49["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n50["CLAIM: deontological: A1 REQUIRED"]
    n51["UNRESOLVED_CONFLICT: ellis&#x27;s duty &#x27;keep promises&#x27; protecting sasha ↔ ellis&#x27;s competing duty &#x27;duty of beneficence&#x27; protecting rowan"]
    n52["UNRESOLVED_CONFLICT: A0 prohibited by keep promises ↔ whether the omission violates a separate right-correlative prohibition"]
    n53["UNRESOLVED_CONFLICT: A0 prohibited by keep promises ↔ whether the candidate maxim&#x27;s means, person-role, and action assessment justify the claimed priority"]
    n54["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: omission was classified as a perfect negative-duty violation without a separate basis"]
    n55["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: candidate maxim person-role conflicts with means classification"]
    n56["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n57["PROPOSED_ARGUMENT: care: argument without committed ledger"]
    n58["UNRESOLVED_CONFLICT: No committed native ledger; assess the retained argument without treating it as operative."]
    n59["UNRESOLVED_CONFLICT: unresolved Care ranking must remain provisional"]
    n60["CLAIM: virtue: A0 EXEMPLIFIES"]
    n61["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n62["CLAIM: virtue: A1 MIXED"]
    n63["CLAIM: rawlsian: A0 assessment"]
    n64["FRAMEWORK_COMMITMENT: ORIGINAL_POSITION_PUBLIC_RULE"]
    n65["CLAIM: rawlsian: A1 assessment"]
    n66["UNRESOLVED_CONFLICT: framework map for gives the blanket to Rowan lacks framework-specific grounds"]
    n67["UNRESOLVED_CONFLICT: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n68["UNRESOLVED_CONFLICT: A0 IMPROVES lacked directional support; committed as UNCERTAIN"]
    n69["UNRESOLVED_CONFLICT: A1 WORSENS lacked directional support; committed as UNCERTAIN"]
    n36 -->|"DEPENDENCY"| n26
    n36 -->|"DEPENDENCY"| n28
    n36 -->|"DEPENDENCY"| n29
    n36 -->|"DEPENDENCY"| n30
    n37 -->|"DEPENDENCY"| n26
    n37 -->|"DEPENDENCY"| n28
    n37 -->|"DEPENDENCY"| n29
    n37 -->|"DEPENDENCY"| n30
    n38 -->|"DEPENDENCY"| n26
    n38 -->|"DEPENDENCY"| n28
    n38 -->|"DEPENDENCY"| n29
    n38 -->|"DEPENDENCY"| n30
    n39 -->|"DEPENDENCY"| n26
    n39 -->|"DEPENDENCY"| n28
    n39 -->|"DEPENDENCY"| n29
    n39 -->|"DEPENDENCY"| n30
    n40 -->|"DEPENDENCY"| n26
    n40 -->|"DEPENDENCY"| n28
    n40 -->|"DEPENDENCY"| n29
    n40 -->|"DEPENDENCY"| n30
    n41 -->|"DEPENDENCY"| n26
    n41 -->|"DEPENDENCY"| n28
    n41 -->|"DEPENDENCY"| n29
    n41 -->|"DEPENDENCY"| n30
    n42 -->|"DEPENDENCY"| n26
    n42 -->|"DEPENDENCY"| n28
    n42 -->|"DEPENDENCY"| n29
    n42 -->|"DEPENDENCY"| n30
    n43 -->|"DEPENDENCY"| n26
    n43 -->|"DEPENDENCY"| n28
    n43 -->|"DEPENDENCY"| n29
    n43 -->|"DEPENDENCY"| n30
    n45 -->|"SUPPORT"| n44
    n46 -->|"SUPPORT"| n44
    n47 -->|"OBJECTION"| n44
    n48 -->|"OBJECTION"| n44
    n49 -->|"OBJECTION"| n44
    n45 -->|"SUPPORT"| n50
    n46 -->|"SUPPORT"| n50
    n58 -->|"OBJECTION"| n57
    n59 -->|"OBJECTION"| n57
    n61 -->|"SUPPORT"| n60
    n61 -->|"SUPPORT"| n62
    n64 -->|"SUPPORT"| n63
    n64 -->|"SUPPORT"| n65
```

Proposed review queue (at most two issues):

- omission was classified as a perfect negative-duty violation without a separate basis
- candidate maxim person-role conflicts with means classification
