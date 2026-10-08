# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: pulls the lever"]
    n1["WORLD_EFFECT: switches the trolley onto a side track"]
    n2["WORLD_EFFECT: the trolley will kill one worker"]
    n3["WORLD_EFFECT: spare five workers"]
    n4["WORLD_EFFECT: does not"]
    n5["WORLD_EFFECT: it will kill the five"]
    n6["WORLD_EFFECT: it will kill five workers"]
    n7["WORLD_EFFECT: spare one worker"]
    n8["PROPOSITION: If she pulls the lever"]
    n9["PROPOSITION: If she does not"]
    n10["PROPOSITION: pulls the lever; affected subject: a lever"]
    n11["PROPOSITION: switches the trolley onto a side track; affected subject: the trolley"]
    n12["PROPOSITION: the trolley will kill one worker; affected subject: one worker"]
    n13["PROPOSITION: spare five workers; affected subject: five workers"]
    n14["PROPOSITION: does not; affected subject: a lever"]
    n15["PROPOSITION: it will kill the five; affected subject: the trolley; magnitude or qualifier: five"]
    n16["PROPOSITION: it will kill five workers; affected subject: five workers"]
    n17["PROPOSITION: spare one worker; affected subject: one worker"]
    n18["PROPOSITION: five workers"]
    n19["PROPOSITION: If she does not, it will kill the five and spare the one."]
    n20["PROPOSITION: If she pulls the lever, the trolley will kill that worker and spare the five."]
    n21["PROPOSITION: Maria can pull a lever that switches the trolley onto a side track, where one worker stands."]
    n22["PROPOSITION: Maria cannot stop the trolley."]
    n23["PROPOSITION: A runaway trolley is heading toward five workers on the main track."]
    n24["PROPOSITION: the trolley will kill one worker; affected subject: one worker"]
    n25["PROPOSITION: spare five workers; affected subject: five workers"]
    n26["PROPOSITION: it will kill five workers; affected subject: five workers"]
    n27["PROPOSITION: spare one worker; affected subject: one worker"]
    n28["PROPOSITION: the trolley will kill one worker; affected subject: one worker"]
    n29["PROPOSITION: spare five workers; affected subject: five workers"]
    n30["PROPOSITION: it will kill five workers; affected subject: five workers"]
    n31["PROPOSITION: If she pulls the lever"]
    n32["PROPOSITION: If she does not"]
    n33["PROPOSITION: If she pulls the lever, the trolley will kill that worker and spare the five."]
    n34["PROPOSITION: If she pulls the lever"]
    n35["CLAIM: utilitarian: A0 assessment"]
    n36["CLAIM: utilitarian: A0 assessment"]
    n37["CLAIM: utilitarian: A0 assessment"]
    n38["CLAIM: utilitarian: A0 assessment"]
    n39["CLAIM: utilitarian: A1 assessment"]
    n40["CLAIM: utilitarian: A1 assessment"]
    n41["CLAIM: utilitarian: A1 assessment"]
    n42["CLAIM: utilitarian: A1 assessment"]
    n43["CLAIM: deontological: A0 CONFLICTED"]
    n44["FRAMEWORK_COMMITMENT: do not kill innocent person"]
    n45["FRAMEWORK_COMMITMENT: perfect negative duty prevails"]
    n46["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n47["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n48["CLAIM: deontological: A1 PERMISSIBLE"]
    n49["FRAMEWORK_COMMITMENT: omission respects perfect duty"]
    n50["UNRESOLVED_CONFLICT: one worker: do not kill innocent person ↔ five workers: duty to rescue many"]
    n51["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocent person ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n52["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n53["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n54["CLAIM: care: A0 RESPONSIVE"]
    n55["FRAMEWORK_COMMITMENT: control imposes responsibility to protect dependent lives"]
    n56["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n57["CLAIM: care: A1 NEGLECTFUL"]
    n58["FRAMEWORK_COMMITMENT: withholding action abandons dependents"]
    n59["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n60["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n61["CLAIM: virtue: A0 EXEMPLIFIES"]
    n62["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n63["CLAIM: virtue: A1 MIXED"]
    n64["CLAIM: rawlsian: A0 assessment"]
    n65["FRAMEWORK_COMMITMENT: ORIGINAL_POSITION_PUBLIC_RULE"]
    n66["CLAIM: rawlsian: A1 assessment"]
    n67["UNRESOLVED_CONFLICT: framework map for pulls the lever lacks framework-specific grounds"]
    n68["UNRESOLVED_CONFLICT: framework map for does not pull the lever lacks framework-specific grounds"]
    n69["UNRESOLVED_CONFLICT: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n70["UNRESOLVED_CONFLICT: decisive numerical role cites no explicit quantity"]
    n71["UNRESOLVED_CONFLICT: Rawlsian preference lacks a stated difference in position or principle priority"]
    n72["UNRESOLVED_CONFLICT: Rawlsian position for A0 conflicts with its framework map"]
    n73["UNRESOLVED_CONFLICT: Rawlsian position for A1 conflicts with its framework map"]
    n44 -->|"SUPPORT"| n43
    n45 -->|"SUPPORT"| n43
    n46 -->|"OBJECTION"| n43
    n47 -->|"OBJECTION"| n43
    n44 -->|"SUPPORT"| n48
    n49 -->|"SUPPORT"| n48
    n3 -->|"SUPPORT"| n54
    n55 -->|"SUPPORT"| n54
    n56 -->|"SUPPORT"| n54
    n6 -->|"SUPPORT"| n57
    n58 -->|"SUPPORT"| n57
    n56 -->|"SUPPORT"| n57
    n62 -->|"SUPPORT"| n61
    n5 -->|"SUPPORT"| n63
    n6 -->|"SUPPORT"| n63
    n62 -->|"SUPPORT"| n63
    n65 -->|"SUPPORT"| n64
    n65 -->|"SUPPORT"| n66
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
