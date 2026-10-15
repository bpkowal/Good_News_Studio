# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: pulls the lever"]
    n1["WORLD_EFFECT: redirects the tram onto a side track"]
    n2["WORLD_EFFECT: the tram will kill one worker"]
    n3["WORLD_EFFECT: spare ten workers"]
    n4["WORLD_EFFECT: does not"]
    n5["WORLD_EFFECT: heading toward ten workers on the main track"]
    n6["WORLD_EFFECT: it will kill ten workers"]
    n7["WORLD_EFFECT: spare one worker"]
    n8["PROPOSITION: If she pulls the lever"]
    n9["PROPOSITION: If she does not"]
    n10["PROPOSITION: pulls the lever; affected subject: a lever"]
    n11["PROPOSITION: redirects the tram onto a side track; affected subject: the tram"]
    n12["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n13["PROPOSITION: spare ten workers; affected subject: ten workers"]
    n14["PROPOSITION: does not; affected subject: a lever"]
    n15["PROPOSITION: heading toward ten workers on the main track; affected subject: the tram; magnitude or qualifier: ten"]
    n16["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n17["PROPOSITION: spare one worker; affected subject: one worker"]
    n18["PROPOSITION: ten workers"]
    n19["PROPOSITION: Dana cannot stop the tram."]
    n20["PROPOSITION: If she pulls the lever, the tram will kill that worker and spare the ten."]
    n21["PROPOSITION: If she does not, it will kill the ten and spare the one."]
    n22["PROPOSITION: Dana can pull a lever that redirects the tram onto a side track, where one worker stands."]
    n23["PROPOSITION: A runaway tram is heading toward ten workers on the main track."]
    n24["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n25["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n26["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n27["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n28["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n29["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n30["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n31["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n32["PROPOSITION: If she pulls the lever, the tram will kill that worker and spare the ten."]
    n33["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n34["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n35["PROPOSITION: If she pulls the lever, the tram will kill that worker and spare the ten."]
    n36["PROPOSITION: If she does not, it will kill the ten and spare the one."]
    n37["CLAIM: utilitarian: A0 assessment"]
    n38["CLAIM: utilitarian: A0 assessment"]
    n39["CLAIM: utilitarian: A0 assessment"]
    n40["CLAIM: utilitarian: A0 assessment"]
    n41["CLAIM: utilitarian: A1 assessment"]
    n42["CLAIM: utilitarian: A1 assessment"]
    n43["CLAIM: utilitarian: A1 assessment"]
    n44["CLAIM: utilitarian: A1 assessment"]
    n45["CLAIM: deontological: A0 CONFLICTED"]
    n46["FRAMEWORK_COMMITMENT: do not kill an innocent person"]
    n47["FRAMEWORK_COMMITMENT: perfect duties outweigh imperfect"]
    n48["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n49["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n50["CLAIM: deontological: A1 PERMISSIBLE"]
    n51["FRAMEWORK_COMMITMENT: Omission does not violate perfect duty"]
    n52["UNRESOLVED_CONFLICT: dana&#x27;s duty &#x27;do not kill an innocent person&#x27; protecting all workers ↔ dana&#x27;s competing duty &#x27;imperfect duty to rescue&#x27; protecting ten workers"]
    n53["UNRESOLVED_CONFLICT: A0 prohibited by do not kill an innocent person ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n54["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n55["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n56["CLAIM: care: A0 RESPONSIVE"]
    n57["FRAMEWORK_COMMITMENT: shared moral community and unique control create responsibility"]
    n58["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n59["CLAIM: care: A1 NEGLECTFUL"]
    n60["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n61["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n62["CLAIM: virtue: A0 EXEMPLIFIES"]
    n63["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n64["CLAIM: virtue: A1 UNDERMINES"]
    n65["CLAIM: rawlsian: A0 assessment"]
    n66["FRAMEWORK_COMMITMENT: ORIGINAL_POSITION_PUBLIC_RULE"]
    n67["CLAIM: rawlsian: A1 assessment"]
    n68["UNRESOLVED_CONFLICT: framework map for pulls the lever lacks framework-specific grounds"]
    n69["UNRESOLVED_CONFLICT: framework map for does not pull the lever lacks framework-specific grounds"]
    n70["UNRESOLVED_CONFLICT: Rawlsian position for A0 conflicts with its framework map"]
    n71["UNRESOLVED_CONFLICT: Rawlsian position for A1 conflicts with its framework map"]
    n37 -->|"DEPENDENCY"| n24
    n37 -->|"DEPENDENCY"| n25
    n38 -->|"DEPENDENCY"| n24
    n38 -->|"DEPENDENCY"| n25
    n39 -->|"DEPENDENCY"| n24
    n39 -->|"DEPENDENCY"| n25
    n40 -->|"DEPENDENCY"| n24
    n40 -->|"DEPENDENCY"| n25
    n41 -->|"DEPENDENCY"| n24
    n41 -->|"DEPENDENCY"| n25
    n42 -->|"DEPENDENCY"| n24
    n42 -->|"DEPENDENCY"| n25
    n43 -->|"DEPENDENCY"| n24
    n43 -->|"DEPENDENCY"| n25
    n44 -->|"DEPENDENCY"| n24
    n44 -->|"DEPENDENCY"| n25
    n46 -->|"SUPPORT"| n45
    n47 -->|"SUPPORT"| n45
    n48 -->|"OBJECTION"| n45
    n49 -->|"OBJECTION"| n45
    n46 -->|"SUPPORT"| n50
    n51 -->|"SUPPORT"| n50
    n3 -->|"SUPPORT"| n56
    n57 -->|"SUPPORT"| n56
    n58 -->|"SUPPORT"| n56
    n6 -->|"SUPPORT"| n59
    n57 -->|"SUPPORT"| n59
    n58 -->|"SUPPORT"| n59
    n1 -->|"SUPPORT"| n62
    n2 -->|"SUPPORT"| n62
    n63 -->|"SUPPORT"| n62
    n5 -->|"SUPPORT"| n64
    n63 -->|"SUPPORT"| n64
    n66 -->|"SUPPORT"| n65
    n66 -->|"SUPPORT"| n67
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
