# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: pulls the lever"]
    n1["WORLD_EFFECT: pulls the lever"]
    n2["WORLD_EFFECT: one worker will die"]
    n3["WORLD_EFFECT: does not pull the lever"]
    n4["WORLD_EFFECT: does not pull the lever"]
    n5["WORLD_EFFECT: five workers will die"]
    n6["PROPOSITION: If Maria pulls the lever"]
    n7["PROPOSITION: If Maria does not pull the lever"]
    n8["PROPOSITION: pulls the lever; affected subject: the lever"]
    n9["PROPOSITION: pulls the lever; affected subject: the lever"]
    n10["PROPOSITION: one worker will die; affected subject: one worker"]
    n11["PROPOSITION: does not pull the lever; affected subject: the lever"]
    n12["PROPOSITION: does not pull the lever; affected subject: the lever"]
    n13["PROPOSITION: five workers will die; affected subject: five workers"]
    n14["PROPOSITION: five workers"]
    n15["PROPOSITION: one worker will die; affected subject: one worker"]
    n16["PROPOSITION: five workers will die; affected subject: five workers"]
    n17["PROPOSITION: one worker will die; affected subject: one worker"]
    n18["PROPOSITION: five workers will die; affected subject: five workers"]
    n19["PROPOSITION: If Maria pulls the lever"]
    n20["PROPOSITION: If Maria does not pull the lever"]
    n21["PROPOSITION: one worker will die; affected subject: one worker"]
    n22["PROPOSITION: five workers will die; affected subject: five workers"]
    n23["PROPOSITION: one worker will die; affected subject: one worker"]
    n24["PROPOSITION: five workers will die; affected subject: five workers"]
    n25["CLAIM: utilitarian: A0 assessment"]
    n26["CLAIM: utilitarian: A0 assessment"]
    n27["CLAIM: utilitarian: A0 assessment"]
    n28["CLAIM: utilitarian: A1 assessment"]
    n29["CLAIM: utilitarian: A1 assessment"]
    n30["CLAIM: utilitarian: A1 assessment"]
    n31["CLAIM: deontological: A0 CONFLICTED"]
    n32["FRAMEWORK_COMMITMENT: do not kill innocent person"]
    n33["FRAMEWORK_COMMITMENT: perfect negative duty overrides imperfect duties"]
    n34["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n35["UNRESOLVED_CONFLICT: intended-as-means classification lacks an in-action causal path"]
    n36["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n37["CLAIM: deontological: A1 PERMISSIBLE"]
    n38["FRAMEWORK_COMMITMENT: avoid killing"]
    n39["FRAMEWORK_COMMITMENT: permissibility from not killing outweighs rescuing"]
    n40["UNRESOLVED_CONFLICT: one worker: avoid killing ↔ five workers &amp; anna: imperfect duty to rescue/save more lives and keep promise"]
    n41["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocent person ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n42["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocent person ↔ whether an in-action causal path makes the burden instrumental to the chosen end"]
    n43["CLAIM: care: A0 RESPONSIVE"]
    n44["FRAMEWORK_COMMITMENT: control of lever creates responsibility"]
    n45["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n46["CLAIM: care: A1 NEGLECTFUL"]
    n47["FRAMEWORK_COMMITMENT: ignoring control fails responsibility"]
    n48["CLAIM: virtue: A0 EXEMPLIFIES"]
    n49["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n50["CLAIM: virtue: A1 UNDERMINES"]
    n51["CLAIM: rawlsian: A0 assessment"]
    n52["FRAMEWORK_COMMITMENT: MAXIMIN_PRIMARY_GOODS"]
    n53["CLAIM: rawlsian: A1 assessment"]
    n25 -->|"DEPENDENCY"| n15
    n25 -->|"DEPENDENCY"| n16
    n26 -->|"DEPENDENCY"| n15
    n26 -->|"DEPENDENCY"| n16
    n27 -->|"DEPENDENCY"| n15
    n27 -->|"DEPENDENCY"| n16
    n28 -->|"DEPENDENCY"| n15
    n28 -->|"DEPENDENCY"| n16
    n29 -->|"DEPENDENCY"| n15
    n29 -->|"DEPENDENCY"| n16
    n30 -->|"DEPENDENCY"| n15
    n30 -->|"DEPENDENCY"| n16
    n32 -->|"SUPPORT"| n31
    n33 -->|"SUPPORT"| n31
    n34 -->|"OBJECTION"| n31
    n35 -->|"OBJECTION"| n31
    n36 -->|"OBJECTION"| n31
    n38 -->|"SUPPORT"| n37
    n39 -->|"SUPPORT"| n37
    n44 -->|"SUPPORT"| n43
    n45 -->|"SUPPORT"| n43
    n5 -->|"SUPPORT"| n46
    n47 -->|"SUPPORT"| n46
    n45 -->|"SUPPORT"| n46
    n48 -->|"DEPENDENCY"| n21
    n48 -->|"DEPENDENCY"| n22
    n49 -->|"SUPPORT"| n48
    n5 -->|"SUPPORT"| n50
    n50 -->|"DEPENDENCY"| n21
    n50 -->|"DEPENDENCY"| n22
    n49 -->|"SUPPORT"| n50
    n52 -->|"SUPPORT"| n51
    n52 -->|"SUPPORT"| n53
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- intended-as-means classification lacks an in-action causal path
