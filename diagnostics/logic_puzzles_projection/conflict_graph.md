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
    n6["PROPOSITION: Pulling the lever will save the five workers (they will survive)"]
    n7["PROPOSITION: If Maria pulls the lever"]
    n8["PROPOSITION: If Maria does not pull the lever"]
    n9["PROPOSITION: pulls the lever; affected subject: the lever"]
    n10["PROPOSITION: pulls the lever; affected subject: the lever"]
    n11["PROPOSITION: one worker will die; affected subject: one worker"]
    n12["PROPOSITION: does not pull the lever; affected subject: the lever"]
    n13["PROPOSITION: does not pull the lever; affected subject: the lever"]
    n14["PROPOSITION: five workers will die; affected subject: five workers"]
    n15["PROPOSITION: five workers"]
    n16["CLAIM: deontological: A0 CONFLICTED"]
    n17["FRAMEWORK_COMMITMENT: do not kill innocents"]
    n18["FRAMEWORK_COMMITMENT: perfect negative duties override imperfect"]
    n19["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n20["UNRESOLVED_CONFLICT: intended-as-means classification lacks an in-action causal path"]
    n21["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n22["CLAIM: deontological: A1 REQUIRED"]
    n23["UNRESOLVED_CONFLICT: one worker: do not kill innocents ↔ five workers: duty of rescue"]
    n24["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocents ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n25["UNRESOLVED_CONFLICT: A0 prohibited by do not kill innocents ↔ whether an in-action causal path makes the burden instrumental to the chosen end"]
    n26["CLAIM: care: A0 MIXED"]
    n27["FRAMEWORK_COMMITMENT: creating lethal risk generates responsibility for after-care"]
    n28["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n29["CLAIM: care: A1 NEGLECTFUL"]
    n30["FRAMEWORK_COMMITMENT: control of lever makes Maria responsible to respond"]
    n16 -->|"DEPENDENCY"| n11
    n16 -->|"DEPENDENCY"| n14
    n17 -->|"SUPPORT"| n16
    n18 -->|"SUPPORT"| n16
    n19 -->|"OBJECTION"| n16
    n20 -->|"OBJECTION"| n16
    n21 -->|"OBJECTION"| n16
    n22 -->|"DEPENDENCY"| n11
    n22 -->|"DEPENDENCY"| n14
    n17 -->|"SUPPORT"| n22
    n18 -->|"SUPPORT"| n22
    n2 -->|"SUPPORT"| n26
    n26 -->|"DEPENDENCY"| n11
    n26 -->|"DEPENDENCY"| n14
    n26 -->|"DEPENDENCY"| n6
    n27 -->|"SUPPORT"| n26
    n28 -->|"SUPPORT"| n26
    n5 -->|"SUPPORT"| n29
    n29 -->|"DEPENDENCY"| n11
    n29 -->|"DEPENDENCY"| n14
    n29 -->|"DEPENDENCY"| n6
    n30 -->|"SUPPORT"| n29
    n28 -->|"SUPPORT"| n29
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- intended-as-means classification lacks an in-action causal path
