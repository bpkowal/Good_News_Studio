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
    n16["PROPOSITION: keep a containment door closed while a hazardous experiment is running"]
    n17["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n18["PROPOSITION: the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"]
    n19["CLAIM: deontological: A0 CONFLICTED"]
    n20["FRAMEWORK_COMMITMENT: do not directly harm uninvolved persons"]
    n21["FRAMEWORK_COMMITMENT: perfect negative duties outweigh imperfect positives"]
    n22["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n23["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n24["CLAIM: deontological: A1 PERMISSIBLE"]
    n25["FRAMEWORK_COMMITMENT: avoid directly harming others"]
    n26["FRAMEWORK_COMMITMENT: avoiding direct harm overrides aiding"]
    n27["UNRESOLVED_CONFLICT: mira&#x27;s duty &#x27;avoid directly harming others&#x27; protecting nearby staff ↔ mira&#x27;s competing duty &#x27;imperfect duty to rescue&#x27; protecting technician"]
    n28["UNRESOLVED_CONFLICT: A0 prohibited by do not directly harm uninvolved persons ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n29["UNRESOLVED_CONFLICT: hypothetical reversal boundary parameter is unverified; graph update not committed"]
    n30["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n31["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n32["UNRESOLVED_CONFLICT: A1 primary protected party lacks current-run action grounding"]
    n19 -->|"DEPENDENCY"| n17
    n19 -->|"DEPENDENCY"| n18
    n20 -->|"SUPPORT"| n19
    n21 -->|"SUPPORT"| n19
    n22 -->|"OBJECTION"| n19
    n23 -->|"OBJECTION"| n19
    n24 -->|"DEPENDENCY"| n17
    n24 -->|"DEPENDENCY"| n18
    n25 -->|"SUPPORT"| n24
    n26 -->|"SUPPORT"| n24
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
