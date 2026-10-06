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
    n25["PROPOSED_REVISION: deontological: disputed full revision; not operative"]
    n26["CLAIM: utilitarian: A0 assessment"]
    n27["CLAIM: utilitarian: A0 assessment"]
    n28["CLAIM: utilitarian: A0 assessment"]
    n29["CLAIM: utilitarian: A1 assessment"]
    n30["CLAIM: utilitarian: A1 assessment"]
    n31["CLAIM: utilitarian: A1 assessment"]
    n32["UNRESOLVED_CONFLICT: numeric reversal threshold lacks a scenario-grounded quantity"]
    n33["CLAIM: deontological: A0 CONFLICTED"]
    n34["FRAMEWORK_COMMITMENT: do not kill innocent person"]
    n35["FRAMEWORK_COMMITMENT: perfect negative duty overrides imperfect duties"]
    n36["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n37["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n38["CLAIM: deontological: A1 PERMISSIBLE"]
    n39["FRAMEWORK_COMMITMENT: avoid killing"]
    n40["FRAMEWORK_COMMITMENT: permissibility from not killing outweighs rescuing"]
    n41["UNRESOLVED_CONFLICT: one worker: avoid killing ↔ five workers &amp; anna: imperfect duty to rescue/save more lives and keep promise"]
    n42["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocent person ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n43["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n44["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n45["CLAIM: care: A0 RESPONSIVE"]
    n46["FRAMEWORK_COMMITMENT: control of lever creates responsibility"]
    n47["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n48["CLAIM: care: A1 NEGLECTFUL"]
    n49["FRAMEWORK_COMMITMENT: ignoring control fails responsibility"]
    n50["UNRESOLVED_CONFLICT: A0 Care relation claims scenario grounding without a matching action effect"]
    n51["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n52["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n53["CLAIM: virtue: A0 EXEMPLIFIES"]
    n54["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n55["CLAIM: virtue: A1 UNDERMINES"]
    n56["CLAIM: rawlsian: A0 assessment"]
    n57["FRAMEWORK_COMMITMENT: MAXIMIN_PRIMARY_GOODS"]
    n58["CLAIM: rawlsian: A1 assessment"]
    n59["UNRESOLVED_CONFLICT: framework map for does not pull the lever lacks framework-specific grounds"]
    n60["UNRESOLVED_CONFLICT: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n61["UNRESOLVED_CONFLICT: A0 IMPROVES lacked directional support; committed as UNCERTAIN"]
    n62["UNRESOLVED_CONFLICT: A1 WORSENS lacked directional support; committed as UNCERTAIN"]
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
    n31 -->|"DEPENDENCY"| n15
    n31 -->|"DEPENDENCY"| n16
    n34 -->|"SUPPORT"| n33
    n35 -->|"SUPPORT"| n33
    n36 -->|"OBJECTION"| n33
    n37 -->|"OBJECTION"| n33
    n39 -->|"SUPPORT"| n38
    n40 -->|"SUPPORT"| n38
    n46 -->|"SUPPORT"| n45
    n47 -->|"SUPPORT"| n45
    n5 -->|"SUPPORT"| n48
    n49 -->|"SUPPORT"| n48
    n47 -->|"SUPPORT"| n48
    n53 -->|"DEPENDENCY"| n21
    n53 -->|"DEPENDENCY"| n22
    n54 -->|"SUPPORT"| n53
    n5 -->|"SUPPORT"| n55
    n55 -->|"DEPENDENCY"| n21
    n55 -->|"DEPENDENCY"| n22
    n54 -->|"SUPPORT"| n55
    n57 -->|"SUPPORT"| n56
    n57 -->|"SUPPORT"| n58
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
