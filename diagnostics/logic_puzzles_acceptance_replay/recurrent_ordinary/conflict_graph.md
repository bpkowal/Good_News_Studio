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
    n15["CLAIM: deontological: A0 CONFLICTED"]
    n16["FRAMEWORK_COMMITMENT: do not kill innocent person"]
    n17["FRAMEWORK_COMMITMENT: perfect negative duty overrides imperfect duties"]
    n18["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n19["UNRESOLVED_CONFLICT: intended-as-means classification lacks an in-action causal path"]
    n20["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n21["CLAIM: deontological: A1 PERMISSIBLE"]
    n22["FRAMEWORK_COMMITMENT: avoid killing"]
    n23["FRAMEWORK_COMMITMENT: permissibility from not killing outweighs rescuing"]
    n24["UNRESOLVED_CONFLICT: one worker: avoid killing ↔ five workers &amp; anna: imperfect duty to rescue/save more lives and keep promise"]
    n25["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocent person ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n26["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocent person ↔ whether an in-action causal path makes the burden instrumental to the chosen end"]
    n27["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n28["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: intended-as-means classification lacks an in-action causal path"]
    n29["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n16 -->|"SUPPORT"| n15
    n17 -->|"SUPPORT"| n15
    n18 -->|"OBJECTION"| n15
    n19 -->|"OBJECTION"| n15
    n20 -->|"OBJECTION"| n15
    n22 -->|"SUPPORT"| n21
    n23 -->|"SUPPORT"| n21
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- intended-as-means classification lacks an in-action causal path
