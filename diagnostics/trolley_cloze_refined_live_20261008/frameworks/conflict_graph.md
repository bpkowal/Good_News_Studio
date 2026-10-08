# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: pulls the lever"]
    n1["WORLD_EFFECT: switches the trolley onto a side track"]
    n2["WORLD_EFFECT: the trolley will kill that worker"]
    n3["WORLD_EFFECT: spare the five"]
    n4["WORLD_EFFECT: does not"]
    n5["WORLD_EFFECT: it will kill the five"]
    n6["WORLD_EFFECT: it will kill the five"]
    n7["WORLD_EFFECT: spare the one"]
    n8["PROPOSITION: If she pulls the lever"]
    n9["PROPOSITION: If she does not"]
    n10["PROPOSITION: pulls the lever; affected subject: a lever"]
    n11["PROPOSITION: switches the trolley onto a side track; affected subject: the trolley"]
    n12["PROPOSITION: the trolley will kill that worker; affected subject: one worker"]
    n13["PROPOSITION: spare the five; affected subject: five workers"]
    n14["PROPOSITION: does not; affected subject: a lever"]
    n15["PROPOSITION: it will kill the five; affected subject: the trolley; magnitude or qualifier: five"]
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
    n27["PROPOSITION: spare the one; affected subject: one worker"]
    n28["PROPOSITION: the trolley will kill that worker; affected subject: one worker"]
    n29["PROPOSITION: If she pulls the lever"]
    n30["PROPOSITION: If she does not"]
    n31["PROPOSITION: If she pulls the lever"]
    n32["PROPOSITION: If she pulls the lever"]
    n33["PROPOSITION: If she does not"]
    n34["CLAIM: utilitarian: A0 assessment"]
    n35["CLAIM: utilitarian: A0 assessment"]
    n36["CLAIM: utilitarian: A0 assessment"]
    n37["CLAIM: utilitarian: A0 assessment"]
    n38["CLAIM: utilitarian: A1 assessment"]
    n39["CLAIM: utilitarian: A1 assessment"]
    n40["CLAIM: utilitarian: A1 assessment"]
    n41["CLAIM: utilitarian: A1 assessment"]
    n42["CLAIM: deontological: A0 CONFLICTED"]
    n43["FRAMEWORK_COMMITMENT: do not kill innocents"]
    n44["FRAMEWORK_COMMITMENT: perfect negative duties outweigh imperfect ones"]
    n45["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n46["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n47["CLAIM: deontological: A1 PERMISSIBLE"]
    n48["FRAMEWORK_COMMITMENT: refrain from killing"]
    n49["FRAMEWORK_COMMITMENT: respecting perfect duty sufficient for permissibility"]
    n50["UNRESOLVED_CONFLICT: one worker: refrain from killing ↔ five workers: imperfect duty to rescue"]
    n51["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocents ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n52["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n53["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n54["CLAIM: care: A0 RESPONSIVE"]
    n55["FRAMEWORK_COMMITMENT: control confers responsibility to respond"]
    n56["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n57["CLAIM: care: A1 MIXED"]
    n58["FRAMEWORK_COMMITMENT: control links her to his safety"]
    n59["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n60["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n61["CLAIM: virtue: A0 EXEMPLIFIES"]
    n62["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n63["CLAIM: virtue: A1 UNDERMINES"]
    n64["CLAIM: rawlsian: A0 assessment"]
    n65["FRAMEWORK_COMMITMENT: ORIGINAL_POSITION_PUBLIC_RULE"]
    n66["CLAIM: rawlsian: A1 assessment"]
    n67["UNRESOLVED_CONFLICT: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n34 -->|"DEPENDENCY"| n24
    n34 -->|"DEPENDENCY"| n26
    n35 -->|"DEPENDENCY"| n24
    n35 -->|"DEPENDENCY"| n26
    n36 -->|"DEPENDENCY"| n24
    n36 -->|"DEPENDENCY"| n26
    n37 -->|"DEPENDENCY"| n24
    n37 -->|"DEPENDENCY"| n26
    n38 -->|"DEPENDENCY"| n24
    n38 -->|"DEPENDENCY"| n26
    n39 -->|"DEPENDENCY"| n24
    n39 -->|"DEPENDENCY"| n26
    n40 -->|"DEPENDENCY"| n24
    n40 -->|"DEPENDENCY"| n26
    n41 -->|"DEPENDENCY"| n24
    n41 -->|"DEPENDENCY"| n26
    n43 -->|"SUPPORT"| n42
    n44 -->|"SUPPORT"| n42
    n45 -->|"OBJECTION"| n42
    n46 -->|"OBJECTION"| n42
    n48 -->|"SUPPORT"| n47
    n49 -->|"SUPPORT"| n47
    n3 -->|"SUPPORT"| n54
    n55 -->|"SUPPORT"| n54
    n56 -->|"SUPPORT"| n54
    n7 -->|"SUPPORT"| n57
    n58 -->|"SUPPORT"| n57
    n56 -->|"SUPPORT"| n57
    n62 -->|"SUPPORT"| n61
    n62 -->|"SUPPORT"| n63
    n65 -->|"SUPPORT"| n64
    n65 -->|"SUPPORT"| n66
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
