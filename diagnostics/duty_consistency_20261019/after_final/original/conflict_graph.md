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
    n16["PROPOSITION: Perfect duties override imperfect duties"]
    n17["PROPOSITION: obligation"]
    n18["CLAIM: deontological: A0 CONFLICTED"]
    n19["FRAMEWORK_COMMITMENT: do not inflict harm on uninvolved persons"]
    n20["FRAMEWORK_COMMITMENT: Perfect duties trump imperfect ones"]
    n21["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n22["UNRESOLVED_CONFLICT: candidate maxim denies intended use while proposing instrumental belief and purposive use"]
    n23["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n24["CLAIM: deontological: A1 CONFLICTED"]
    n25["FRAMEWORK_COMMITMENT: Avoiding direct harm governs"]
    n26["UNRESOLVED_CONFLICT: perfect-duty priority lacks a perfect-duty classification of the primary governing norm"]
    n27["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n28["UNRESOLVED_CONFLICT: mira&#x27;s duty &#x27;do not inflict harm on uninvolved persons&#x27; protecting nearby staff ↔ mira&#x27;s competing duty &#x27;Rescue those in peril&#x27; protecting technician"]
    n29["UNRESOLVED_CONFLICT: decisive numbers are not connected to a duty or right"]
    n30["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n31["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use"]
    n32["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n33["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: perfect-duty priority lacks a perfect-duty classification of the primary governing norm"]
    n34["UNRESOLVED_CONFLICT: A1 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n35["UNRESOLVED_CONFLICT: A1 primary protected party lacks current-run action grounding"]
    n18 -->|"DEPENDENCY"| n17
    n19 -->|"SUPPORT"| n18
    n20 -->|"SUPPORT"| n18
    n21 -->|"OBJECTION"| n18
    n22 -->|"OBJECTION"| n18
    n23 -->|"OBJECTION"| n18
    n24 -->|"DEPENDENCY"| n17
    n19 -->|"SUPPORT"| n24
    n25 -->|"SUPPORT"| n24
    n26 -->|"OBJECTION"| n24
    n27 -->|"OBJECTION"| n24
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- candidate maxim denies intended use while proposing instrumental belief and purposive use
