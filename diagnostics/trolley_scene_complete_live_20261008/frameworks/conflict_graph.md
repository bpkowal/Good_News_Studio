# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: pulls the lever"]
    n1["WORLD_EFFECT: switches the trolley onto a side track"]
    n2["WORLD_EFFECT: the trolley will kill that worker"]
    n3["WORLD_EFFECT: spare the five"]
    n4["WORLD_EFFECT: does not"]
    n5["WORLD_EFFECT: does not"]
    n6["WORLD_EFFECT: it will kill the five"]
    n7["WORLD_EFFECT: spare the one"]
    n8["PROPOSITION: If she pulls the lever"]
    n9["PROPOSITION: If she does not"]
    n10["PROPOSITION: pulls the lever; affected subject: a lever"]
    n11["PROPOSITION: switches the trolley onto a side track; affected subject: the trolley"]
    n12["PROPOSITION: the trolley will kill that worker; affected subject: one worker"]
    n13["PROPOSITION: spare the five; affected subject: five workers"]
    n14["PROPOSITION: does not; affected subject: a lever"]
    n15["PROPOSITION: does not; affected subject: a lever"]
    n16["PROPOSITION: it will kill the five; affected subject: five workers"]
    n17["PROPOSITION: spare the one; affected subject: one worker"]
    n18["PROPOSITION: five workers"]
    n19["PROPOSITION: If she does not, it will kill the five and spare the one."]
    n20["PROPOSITION: If she pulls the lever, the trolley will kill that worker and spare the five."]
    n21["PROPOSITION: Maria can pull a lever that switches the trolley onto a side track, where one worker stands."]
    n22["PROPOSITION: Maria cannot stop the trolley."]
    n23["PROPOSITION: A runaway trolley is heading toward five workers on the main track."]
    n24["PROPOSITION: the trolley will kill that worker; affected subject: one worker"]
    n25["PROPOSITION: spare the five; affected subject: five workers"]
    n26["PROPOSITION: it will kill the five; affected subject: five workers"]
    n27["PROPOSITION: the trolley will kill that worker; affected subject: one worker"]
    n28["PROPOSITION: it will kill the five; affected subject: five workers"]
    n29["PROPOSITION: five workers are entrusted to the decision maker"]
    n30["PROPOSITION: If she pulls the lever"]
    n31["PROPOSITION: If she does not, it will kill the five and spare the one."]
    n32["PROPOSITION: If she pulls the lever, the trolley will kill that worker and spare the five."]
    n33["PROPOSITION: If she pulls the lever"]
    n34["PROPOSITION: the trolley will kill that worker; affected subject: one worker"]
    n35["PROPOSITION: it will kill the five; affected subject: five workers"]
    n36["CLAIM: utilitarian: A0 assessment"]
    n37["CLAIM: utilitarian: A0 assessment"]
    n38["CLAIM: utilitarian: A0 assessment"]
    n39["CLAIM: utilitarian: A0 assessment"]
    n40["CLAIM: utilitarian: A1 assessment"]
    n41["CLAIM: utilitarian: A1 assessment"]
    n42["CLAIM: utilitarian: A1 assessment"]
    n43["CLAIM: utilitarian: A1 assessment"]
    n44["CLAIM: deontological: A0 CONFLICTED"]
    n45["FRAMEWORK_COMMITMENT: do not intentionally kill"]
    n46["FRAMEWORK_COMMITMENT: perfect negative duty prevails"]
    n47["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n48["UNRESOLVED_CONFLICT: intended-as-means classification lacks an in-action causal path"]
    n49["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n50["CLAIM: deontological: A1 PERMISSIBLE"]
    n51["FRAMEWORK_COMMITMENT: refrain from killing innocents"]
    n52["FRAMEWORK_COMMITMENT: perfect duty satisfied"]
    n53["UNRESOLVED_CONFLICT: one worker: refrain from killing innocents ↔ five workers: imperfect duty to rescue"]
    n54["UNRESOLVED_CONFLICT: A0 prohibited by do not intentionally kill ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n55["UNRESOLVED_CONFLICT: A0 prohibited by do not intentionally kill ↔ whether an in-action causal path makes the burden instrumental to the chosen end"]
    n56["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n57["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: intended-as-means classification lacks an in-action causal path"]
    n58["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n59["CLAIM: care: A0 RESPONSIVE"]
    n60["FRAMEWORK_COMMITMENT: exclusive situational control imposes responsibility"]
    n61["FRAMEWORK_COMMITMENT: ENTRUSTED_RESPONSIBILITY"]
    n62["CLAIM: care: A1 NEGLECTFUL"]
    n63["FRAMEWORK_COMMITMENT: ignores responsibility despite control"]
    n64["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n65["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n66["CLAIM: virtue: A0 EXEMPLIFIES"]
    n67["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n68["CLAIM: virtue: A1 UNDERMINES"]
    n69["CLAIM: rawlsian: A0 assessment"]
    n70["FRAMEWORK_COMMITMENT: ORIGINAL_POSITION_PUBLIC_RULE"]
    n71["CLAIM: rawlsian: A1 assessment"]
    n72["UNRESOLVED_CONFLICT: framework map for pulls the lever lacks framework-specific grounds"]
    n73["UNRESOLVED_CONFLICT: framework map for does not lacks framework-specific grounds"]
    n74["UNRESOLVED_CONFLICT: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n75["UNRESOLVED_CONFLICT: Rawlsian position for A0 conflicts with its framework map"]
    n76["UNRESOLVED_CONFLICT: numeric reversal threshold lacks a scenario-grounded quantity"]
    n45 -->|"SUPPORT"| n44
    n46 -->|"SUPPORT"| n44
    n47 -->|"OBJECTION"| n44
    n48 -->|"OBJECTION"| n44
    n49 -->|"OBJECTION"| n44
    n51 -->|"SUPPORT"| n50
    n52 -->|"SUPPORT"| n50
    n3 -->|"SUPPORT"| n59
    n59 -->|"DEPENDENCY"| n29
    n60 -->|"SUPPORT"| n59
    n61 -->|"SUPPORT"| n59
    n6 -->|"SUPPORT"| n62
    n62 -->|"DEPENDENCY"| n29
    n63 -->|"SUPPORT"| n62
    n61 -->|"SUPPORT"| n62
    n67 -->|"SUPPORT"| n66
    n67 -->|"SUPPORT"| n68
    n70 -->|"SUPPORT"| n69
    n70 -->|"SUPPORT"| n71
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- intended-as-means classification lacks an in-action causal path
