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
    n6["PROPOSITION: five deaths is worse than one"]
    n7["PROPOSITION: If Maria pulls the lever"]
    n8["PROPOSITION: If Maria does not pull the lever"]
    n9["PROPOSITION: pulls the lever; affected subject: the lever"]
    n10["PROPOSITION: pulls the lever; affected subject: the lever"]
    n11["PROPOSITION: one worker will die; affected subject: one worker"]
    n12["PROPOSITION: does not pull the lever; affected subject: the lever"]
    n13["PROPOSITION: does not pull the lever; affected subject: the lever"]
    n14["PROPOSITION: five workers will die; affected subject: five workers"]
    n15["PROPOSITION: five workers"]
    n16["CLAIM: utilitarian: A0 assessment"]
    n17["CLAIM: utilitarian: A0 assessment"]
    n18["CLAIM: utilitarian: A0 assessment"]
    n19["CLAIM: utilitarian: A1 assessment"]
    n20["CLAIM: utilitarian: A1 assessment"]
    n21["CLAIM: utilitarian: A1 assessment"]
    n22["CLAIM: deontological: A0 CONFLICTED"]
    n23["FRAMEWORK_COMMITMENT: do not kill innocent person"]
    n24["FRAMEWORK_COMMITMENT: perfect negative duty overrides imperfect duties"]
    n25["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n26["UNRESOLVED_CONFLICT: intended-as-means classification lacks an in-action causal path"]
    n27["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n28["CLAIM: deontological: A1 PERMISSIBLE"]
    n29["FRAMEWORK_COMMITMENT: avoid killing"]
    n30["FRAMEWORK_COMMITMENT: permissibility from not killing outweighs rescuing"]
    n31["UNRESOLVED_CONFLICT: one worker: avoid killing ↔ five workers &amp; anna: imperfect duty to rescue/save more lives and keep promise"]
    n32["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocent person ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n33["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocent person ↔ whether an in-action causal path makes the burden instrumental to the chosen end"]
    n34["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n35["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: intended-as-means classification lacks an in-action causal path"]
    n36["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n37["UNRESOLVED_CONFLICT: rejected update: Deontological principle state changed without a framework-relevant workspace reason (A0 CONFLICTED/RESPECT_PERSONS/VIOLATES/DUTY/SATISFIES/UNRESOLVED/UNRESOLVED/PERFECT_NEGATIVE/DOING_HARM/NOT_REQUIRED/no special relation with the victim required/INTENDED_AS_MEANS/BODILY_INTEGRITY/SPECIAL_OBLIGATION/NONE/NOT_APPLICABLE/RESPECT_PERSONS/CONTESTED -&gt; PROHIBITED/DUTY/VIOLATES/DUTY/CONFLICTS/PRIMARY/RESPECT_PERSONS/PERFECT_NEGATIVE/DOING_HARM/NOT_REQUIRED/NONE/FORESEEN_SIDE_EFFECT/BODILY_INTEGRITY/OTHER/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED; A1 PERMISSIBLE/RESPECT_PERSONS/CONSISTENT/DUTY/CONFLICTS/PRIMARY/RESPECT_PERSONS/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/no special relation with endangered workers/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/SPECIAL_OBLIGATION/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED -&gt; PERMISSIBLE/DUTY/CONSISTENT/DUTY/CONFLICTS/PRIMARY/RESPECT_PERSONS/PERFECT_NEGATIVE/ALLOWING_HARM/NOT_REQUIRED/NONE/NO_INSTRUMENTALIZATION/BODILY_INTEGRITY/OTHER/NONE/NOT_APPLICABLE/RESPECT_PERSONS/RESOLVED); previous committed ledger preserved"]
    n38["CLAIM: care: A0 RESPONSIVE"]
    n39["FRAMEWORK_COMMITMENT: control of lever creates responsibility"]
    n40["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n41["CLAIM: care: A1 NEGLECTFUL"]
    n42["FRAMEWORK_COMMITMENT: ignoring control fails responsibility"]
    n43["UNRESOLVED_CONFLICT: A0 Care relation claims scenario grounding without a matching action effect"]
    n44["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n45["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n46["UNRESOLVED_CONFLICT: rejected update: A0 Care relation claims scenario grounding without a matching action effect"]
    n47["UNRESOLVED_CONFLICT: rejected update: A0 competing Care claim remains contested"]
    n48["UNRESOLVED_CONFLICT: rejected update: A1 competing Care claim remains contested"]
    n49["CLAIM: virtue: A0 EXEMPLIFIES"]
    n50["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n51["CLAIM: virtue: A1 UNDERMINES"]
    n52["CLAIM: rawlsian: A0 assessment"]
    n53["FRAMEWORK_COMMITMENT: MAXIMIN_PRIMARY_GOODS"]
    n54["CLAIM: rawlsian: A1 assessment"]
    n55["UNRESOLVED_CONFLICT: framework map for does not pull the lever lacks framework-specific grounds"]
    n56["UNRESOLVED_CONFLICT: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n57["UNRESOLVED_CONFLICT: A0 IMPROVES lacked directional support; committed as UNCERTAIN"]
    n58["UNRESOLVED_CONFLICT: A1 WORSENS lacked directional support; committed as UNCERTAIN"]
    n59["UNRESOLVED_CONFLICT: rejected update: framework map for pulls the lever lacks framework-specific grounds"]
    n60["UNRESOLVED_CONFLICT: rejected update: framework map for does not pull the lever lacks framework-specific grounds"]
    n61["UNRESOLVED_CONFLICT: rejected update: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n62["UNRESOLVED_CONFLICT: rejected update: A0 IMPROVES lacked directional support; committed as UNCERTAIN"]
    n63["UNRESOLVED_CONFLICT: rejected update: A1 WORSENS lacked directional support; committed as UNCERTAIN"]
    n23 -->|"SUPPORT"| n22
    n24 -->|"SUPPORT"| n22
    n25 -->|"OBJECTION"| n22
    n26 -->|"OBJECTION"| n22
    n27 -->|"OBJECTION"| n22
    n29 -->|"SUPPORT"| n28
    n30 -->|"SUPPORT"| n28
    n39 -->|"SUPPORT"| n38
    n40 -->|"SUPPORT"| n38
    n5 -->|"SUPPORT"| n41
    n42 -->|"SUPPORT"| n41
    n40 -->|"SUPPORT"| n41
    n50 -->|"SUPPORT"| n49
    n50 -->|"SUPPORT"| n51
    n52 -->|"DEPENDENCY"| n11
    n52 -->|"DEPENDENCY"| n14
    n53 -->|"SUPPORT"| n52
    n54 -->|"DEPENDENCY"| n11
    n54 -->|"DEPENDENCY"| n14
    n53 -->|"SUPPORT"| n54
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- intended-as-means classification lacks an in-action causal path
