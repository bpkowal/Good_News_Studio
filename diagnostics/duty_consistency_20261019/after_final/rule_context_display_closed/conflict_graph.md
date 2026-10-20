# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: open the door now"]
    n1["WORLD_EFFECT: doing so may expose nearby staff to hazardous fumes"]
    n2["WORLD_EFFECT: Stopping the experiment"]
    n3["WORLD_EFFECT: Stopping the experiment first takes ten minutes"]
    n4["WORLD_EFFECT: the technician may die during that delay"]
    n5["PROPOSITION: keep a containment door closed while a hazardous experiment is running"]
    n6["PROPOSITION: opening it only after the experiment has stopped"]
    n7["PROPOSITION: obligation"]
    n8["PROPOSITION: permission"]
    n9["PROPOSITION: open the door now; affected subject: the door"]
    n10["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n11["PROPOSITION: may — doing so may expose nearby staff to hazardous fumes"]
    n12["PROPOSITION: Stopping the experiment; affected subject: the experiment"]
    n13["PROPOSITION: Stopping the experiment first takes ten minutes; affected subject: the experiment; magnitude or qualifier: ten minutes"]
    n14["PROPOSITION: the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"]
    n15["PROPOSITION: may — the technician may die during that delay"]
    n16["PROPOSITION: Opening is allowed only after the experiment stops (N1)"]
    n17["PROPOSITION: Mira must keep the door closed while the experiment is running (N0)"]
    n18["PROPOSITION: obligation"]
    n19["PROPOSITION: permission"]
    n20["CLAIM: deontological: A0 CONFLICTED"]
    n21["FRAMEWORK_COMMITMENT: safety containment duty"]
    n22["FRAMEWORK_COMMITMENT: perfect duty not to harm overrides"]
    n23["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n24["UNRESOLVED_CONFLICT: candidate maxim denies intended use while proposing instrumental belief and purposive use"]
    n25["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n26["CLAIM: deontological: A1 CONFLICTED"]
    n27["FRAMEWORK_COMMITMENT: perfect duty governs"]
    n28["UNRESOLVED_CONFLICT: revised duty classification is retained but requires independent review"]
    n29["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n30["UNRESOLVED_CONFLICT: mira&#x27;s duty &#x27;safety containment duty&#x27; protecting nearby staff ↔ mira&#x27;s competing duty &#x27;imperfect rescue duty&#x27; protecting technician"]
    n31["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n32["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use"]
    n33["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n34["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: revised duty classification is retained but requires independent review"]
    n35["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n36["UNRESOLVED_CONFLICT: A1 primary protected party lacks current-run action grounding"]
    n20 -->|"DEPENDENCY"| n18
    n20 -->|"DEPENDENCY"| n19
    n21 -->|"SUPPORT"| n20
    n22 -->|"SUPPORT"| n20
    n23 -->|"OBJECTION"| n20
    n24 -->|"OBJECTION"| n20
    n25 -->|"OBJECTION"| n20
    n26 -->|"DEPENDENCY"| n18
    n26 -->|"DEPENDENCY"| n19
    n21 -->|"SUPPORT"| n26
    n27 -->|"SUPPORT"| n26
    n28 -->|"OBJECTION"| n26
    n29 -->|"OBJECTION"| n26
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- candidate maxim denies intended use while proposing instrumental belief and purposive use
