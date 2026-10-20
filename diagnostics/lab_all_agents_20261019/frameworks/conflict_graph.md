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
    n16["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n17["PROPOSITION: the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"]
    n18["PROPOSITION: keep a containment door closed while a hazardous experiment is running"]
    n19["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n20["PROPOSITION: the trapped technician are entrusted to the decision maker"]
    n21["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n22["PROPOSITION: the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"]
    n23["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n24["PROPOSITION: the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"]
    n25["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n26["PROPOSITION: the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"]
    n27["CLAIM: utilitarian: A0 assessment"]
    n28["CLAIM: utilitarian: A0 assessment"]
    n29["CLAIM: utilitarian: A1 assessment"]
    n30["CLAIM: utilitarian: A1 assessment"]
    n31["CLAIM: utilitarian: A1 assessment"]
    n32["CLAIM: deontological: A0 CONFLICTED"]
    n33["FRAMEWORK_COMMITMENT: do not harm innocents"]
    n34["FRAMEWORK_COMMITMENT: perfect negative duties override imperfect positives"]
    n35["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n36["UNRESOLVED_CONFLICT: candidate maxim denies intended use while proposing instrumental belief and purposive use"]
    n37["UNRESOLVED_CONFLICT: merely-as-means conclusion lacks intended-as-means classification"]
    n38["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n39["CLAIM: deontological: A1 REQUIRED"]
    n40["FRAMEWORK_COMMITMENT: avoid harming innocents"]
    n41["FRAMEWORK_COMMITMENT: avoid doing harm prevails over preventing harm"]
    n42["UNRESOLVED_CONFLICT: mira&#x27;s duty &#x27;avoid harming innocents&#x27; protecting nearby staff ↔ mira&#x27;s competing duty &#x27;imperfect duty of rescue&#x27; protecting technician"]
    n43["UNRESOLVED_CONFLICT: A0 prohibited by do not harm innocents ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n44["UNRESOLVED_CONFLICT: A0 prohibited by do not harm innocents ↔ whether the burden is intended as a means or only foreseen as a side effect"]
    n45["UNRESOLVED_CONFLICT: hypothetical reversal boundary parameter is unverified; graph update not committed"]
    n46["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n47["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use"]
    n48["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification"]
    n49["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n50["UNRESOLVED_CONFLICT: A1 primary protected party lacks current-run action grounding"]
    n51["CLAIM: care: A0 RESPONSIVE"]
    n52["FRAMEWORK_COMMITMENT: Control over access creates responsibility to assist"]
    n53["FRAMEWORK_COMMITMENT: ENTRUSTED_RESPONSIBILITY"]
    n54["CLAIM: care: A1 NEGLECTFUL"]
    n55["FRAMEWORK_COMMITMENT: Same responsibility but unmet"]
    n56["UNRESOLVED_CONFLICT: hypothetical reversal boundary parameter is unverified; graph update not committed"]
    n57["UNRESOLVED_CONFLICT: A0 Care relation claims scenario grounding without a matching action effect"]
    n58["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n59["UNRESOLVED_CONFLICT: A1 Care relation claims scenario grounding without a matching action effect"]
    n60["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n61["CLAIM: virtue: A0 EXEMPLIFIES"]
    n62["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n63["CLAIM: virtue: A1 MIXED"]
    n64["UNRESOLVED_CONFLICT: hypothetical reversal boundary parameter is unverified; graph update not committed"]
    n65["PROPOSED_ARGUMENT: rawlsian: argument without committed ledger"]
    n66["UNRESOLVED_CONFLICT: No committed native ledger; assess the retained argument without treating it as operative."]
    n67["UNRESOLVED_CONFLICT: framework map for open the door now lacks framework-specific grounds"]
    n68["UNRESOLVED_CONFLICT: framework map for stop the experiment and wait lacks framework-specific grounds"]
    n69["UNRESOLVED_CONFLICT: Rawlsian position for A0 has invalid additional dimensions"]
    n70["UNRESOLVED_CONFLICT: Rawlsian position for A0 repeats its primary dimension as additional"]
    n71["UNRESOLVED_CONFLICT: Rawlsian position for A1 has invalid additional dimensions"]
    n72["UNRESOLVED_CONFLICT: Rawlsian position for A1 repeats its primary dimension as additional"]
    n73["UNRESOLVED_CONFLICT: hypothetical reversal boundary parameter is unverified; graph update not committed"]
    n27 -->|"DEPENDENCY"| n16
    n27 -->|"DEPENDENCY"| n17
    n28 -->|"DEPENDENCY"| n16
    n28 -->|"DEPENDENCY"| n17
    n29 -->|"DEPENDENCY"| n16
    n29 -->|"DEPENDENCY"| n17
    n30 -->|"DEPENDENCY"| n16
    n30 -->|"DEPENDENCY"| n17
    n31 -->|"DEPENDENCY"| n16
    n31 -->|"DEPENDENCY"| n17
    n32 -->|"DEPENDENCY"| n19
    n33 -->|"SUPPORT"| n32
    n34 -->|"SUPPORT"| n32
    n35 -->|"OBJECTION"| n32
    n36 -->|"OBJECTION"| n32
    n37 -->|"OBJECTION"| n32
    n38 -->|"OBJECTION"| n32
    n39 -->|"DEPENDENCY"| n19
    n40 -->|"SUPPORT"| n39
    n41 -->|"SUPPORT"| n39
    n51 -->|"DEPENDENCY"| n21
    n51 -->|"DEPENDENCY"| n22
    n51 -->|"DEPENDENCY"| n20
    n52 -->|"SUPPORT"| n51
    n53 -->|"SUPPORT"| n51
    n54 -->|"DEPENDENCY"| n21
    n54 -->|"DEPENDENCY"| n22
    n54 -->|"DEPENDENCY"| n20
    n55 -->|"SUPPORT"| n54
    n53 -->|"SUPPORT"| n54
    n61 -->|"DEPENDENCY"| n24
    n62 -->|"SUPPORT"| n61
    n4 -->|"SUPPORT"| n63
    n63 -->|"DEPENDENCY"| n24
    n62 -->|"SUPPORT"| n63
    n66 -->|"OBJECTION"| n65
    n67 -->|"OBJECTION"| n65
    n68 -->|"OBJECTION"| n65
    n69 -->|"OBJECTION"| n65
    n70 -->|"OBJECTION"| n65
    n71 -->|"OBJECTION"| n65
    n72 -->|"OBJECTION"| n65
    n73 -->|"OBJECTION"| n65
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- candidate maxim denies intended use while proposing instrumental belief and purposive use
