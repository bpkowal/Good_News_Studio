# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: pulls the lever"]
    n1["WORLD_EFFECT: switches the trolley onto a side track"]
    n2["WORLD_EFFECT: the trolley will kill one worker"]
    n3["WORLD_EFFECT: spare five workers"]
    n4["WORLD_EFFECT: does not"]
    n5["WORLD_EFFECT: heading toward five workers on the main track"]
    n6["WORLD_EFFECT: it will kill five workers"]
    n7["WORLD_EFFECT: spare one worker"]
    n8["PROPOSITION: If she pulls the lever"]
    n9["PROPOSITION: If she does not"]
    n10["PROPOSITION: pulls the lever; affected subject: a lever"]
    n11["PROPOSITION: switches the trolley onto a side track; affected subject: the trolley"]
    n12["PROPOSITION: the trolley will kill one worker; affected subject: one worker"]
    n13["PROPOSITION: spare five workers; affected subject: five workers"]
    n14["PROPOSITION: does not; affected subject: a lever"]
    n15["PROPOSITION: heading toward five workers on the main track; affected subject: the trolley; magnitude or qualifier: five"]
    n16["PROPOSITION: it will kill five workers; affected subject: five workers"]
    n17["PROPOSITION: spare one worker; affected subject: one worker"]
    n18["PROPOSITION: five workers"]
    n19["PROPOSITION: If she does not, it will kill the five and spare the one."]
    n20["PROPOSITION: If she pulls the lever, the trolley will kill that worker and spare the five."]
    n21["PROPOSITION: Maria can pull a lever that switches the trolley onto a side track, where one worker stands."]
    n22["PROPOSITION: Maria cannot stop the trolley."]
    n23["PROPOSITION: A runaway trolley is heading toward five workers on the main track."]
    n24["PROPOSITION: the trolley will kill one worker; affected subject: one worker"]
    n25["PROPOSITION: it will kill five workers; affected subject: five workers"]
    n26["PROPOSITION: the trolley will kill one worker; affected subject: one worker"]
    n27["PROPOSITION: it will kill five workers; affected subject: five workers"]
    n28["CLAIM: utilitarian: A0 assessment"]
    n29["CLAIM: utilitarian: A0 assessment"]
    n30["CLAIM: utilitarian: A0 assessment"]
    n31["CLAIM: utilitarian: A0 assessment"]
    n32["CLAIM: utilitarian: A1 assessment"]
    n33["CLAIM: utilitarian: A1 assessment"]
    n34["CLAIM: utilitarian: A1 assessment"]
    n35["CLAIM: utilitarian: A1 assessment"]
    n36["CLAIM: deontological: A0 CONFLICTED"]
    n37["FRAMEWORK_COMMITMENT: do not kill innocent persons"]
    n38["FRAMEWORK_COMMITMENT: perfect negative duties have priority"]
    n39["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n40["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n41["CLAIM: deontological: A1 PERMISSIBLE"]
    n42["FRAMEWORK_COMMITMENT: imperfect duty to aid others"]
    n43["FRAMEWORK_COMMITMENT: omission does not breach perfect duty"]
    n44["UNRESOLVED_CONFLICT: maria&#x27;s duty &#x27;imperfect duty to aid others&#x27; protecting five workers ↔ maria&#x27;s competing duty &#x27;perfect duty not to kill&#x27; protecting one worker"]
    n45["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocent persons ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n46["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n47["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n36 -->|"DEPENDENCY"| n26
    n36 -->|"DEPENDENCY"| n27
    n37 -->|"SUPPORT"| n36
    n38 -->|"SUPPORT"| n36
    n39 -->|"OBJECTION"| n36
    n40 -->|"OBJECTION"| n36
    n41 -->|"DEPENDENCY"| n26
    n41 -->|"DEPENDENCY"| n27
    n42 -->|"SUPPORT"| n41
    n43 -->|"SUPPORT"| n41
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
