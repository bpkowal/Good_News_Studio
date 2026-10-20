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
    n27["PROPOSED_REVISION: deontological: disputed full revision; not operative"]
    n28["CLAIM: utilitarian: A0 assessment"]
    n29["CLAIM: utilitarian: A0 assessment"]
    n30["CLAIM: utilitarian: A1 assessment"]
    n31["CLAIM: utilitarian: A1 assessment"]
    n32["CLAIM: utilitarian: A1 assessment"]
    n33["CLAIM: deontological: A0 CONFLICTED"]
    n34["FRAMEWORK_COMMITMENT: do not harm innocents"]
    n35["FRAMEWORK_COMMITMENT: perfect negative duties override imperfect positives"]
    n36["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n37["UNRESOLVED_CONFLICT: candidate maxim denies intended use while proposing instrumental belief and purposive use"]
    n38["UNRESOLVED_CONFLICT: merely-as-means conclusion lacks intended-as-means classification"]
    n39["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n40["CLAIM: deontological: A1 REQUIRED"]
    n41["FRAMEWORK_COMMITMENT: avoid harming innocents"]
    n42["FRAMEWORK_COMMITMENT: avoid doing harm prevails over preventing harm"]
    n43["UNRESOLVED_CONFLICT: mira&#x27;s duty &#x27;avoid harming innocents&#x27; protecting nearby staff ↔ mira&#x27;s competing duty &#x27;imperfect duty of rescue&#x27; protecting technician"]
    n44["UNRESOLVED_CONFLICT: A0 prohibited by do not harm innocents ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n45["UNRESOLVED_CONFLICT: A0 prohibited by do not harm innocents ↔ whether the burden is intended as a means or only foreseen as a side effect"]
    n46["UNRESOLVED_CONFLICT: hypothetical reversal boundary parameter is unverified; graph update not committed"]
    n47["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n48["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: candidate maxim denies intended use while proposing instrumental belief and purposive use"]
    n49["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: merely-as-means conclusion lacks intended-as-means classification"]
    n50["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n51["UNRESOLVED_CONFLICT: A1 primary protected party lacks current-run action grounding"]
    n52["CLAIM: care: A0 RESPONSIVE"]
    n53["FRAMEWORK_COMMITMENT: Control over access creates responsibility to assist"]
    n54["FRAMEWORK_COMMITMENT: ENTRUSTED_RESPONSIBILITY"]
    n55["CLAIM: care: A1 NEGLECTFUL"]
    n56["FRAMEWORK_COMMITMENT: Same responsibility but unmet"]
    n57["UNRESOLVED_CONFLICT: hypothetical reversal boundary parameter is unverified; graph update not committed"]
    n58["UNRESOLVED_CONFLICT: A0 Care relation claims scenario grounding without a matching action effect"]
    n59["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n60["UNRESOLVED_CONFLICT: A1 Care relation claims scenario grounding without a matching action effect"]
    n61["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n62["CLAIM: virtue: A0 EXEMPLIFIES"]
    n63["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n64["CLAIM: virtue: A1 MIXED"]
    n65["UNRESOLVED_CONFLICT: hypothetical reversal boundary parameter is unverified; graph update not committed"]
    n66["PROPOSED_ARGUMENT: rawlsian: argument without committed ledger"]
    n67["UNRESOLVED_CONFLICT: No committed native ledger; assess the retained argument without treating it as operative."]
    n68["UNRESOLVED_CONFLICT: framework map for open the door now lacks framework-specific grounds"]
    n69["UNRESOLVED_CONFLICT: framework map for stop the experiment and wait lacks framework-specific grounds"]
    n70["UNRESOLVED_CONFLICT: Rawlsian position for A0 has invalid additional dimensions"]
    n71["UNRESOLVED_CONFLICT: Rawlsian position for A0 repeats its primary dimension as additional"]
    n72["UNRESOLVED_CONFLICT: Rawlsian position for A1 has invalid additional dimensions"]
    n73["UNRESOLVED_CONFLICT: Rawlsian position for A1 repeats its primary dimension as additional"]
    n74["UNRESOLVED_CONFLICT: hypothetical reversal boundary parameter is unverified; graph update not committed"]
    n28 -->|"DEPENDENCY"| n16
    n28 -->|"DEPENDENCY"| n17
    n29 -->|"DEPENDENCY"| n16
    n29 -->|"DEPENDENCY"| n17
    n30 -->|"DEPENDENCY"| n16
    n30 -->|"DEPENDENCY"| n17
    n31 -->|"DEPENDENCY"| n16
    n31 -->|"DEPENDENCY"| n17
    n32 -->|"DEPENDENCY"| n16
    n32 -->|"DEPENDENCY"| n17
    n33 -->|"DEPENDENCY"| n19
    n34 -->|"SUPPORT"| n33
    n35 -->|"SUPPORT"| n33
    n36 -->|"OBJECTION"| n33
    n37 -->|"OBJECTION"| n33
    n38 -->|"OBJECTION"| n33
    n39 -->|"OBJECTION"| n33
    n40 -->|"DEPENDENCY"| n19
    n41 -->|"SUPPORT"| n40
    n42 -->|"SUPPORT"| n40
    n52 -->|"DEPENDENCY"| n21
    n52 -->|"DEPENDENCY"| n22
    n52 -->|"DEPENDENCY"| n20
    n53 -->|"SUPPORT"| n52
    n54 -->|"SUPPORT"| n52
    n55 -->|"DEPENDENCY"| n21
    n55 -->|"DEPENDENCY"| n22
    n55 -->|"DEPENDENCY"| n20
    n56 -->|"SUPPORT"| n55
    n54 -->|"SUPPORT"| n55
    n62 -->|"DEPENDENCY"| n24
    n63 -->|"SUPPORT"| n62
    n4 -->|"SUPPORT"| n64
    n64 -->|"DEPENDENCY"| n24
    n63 -->|"SUPPORT"| n64
    n67 -->|"OBJECTION"| n66
    n68 -->|"OBJECTION"| n66
    n69 -->|"OBJECTION"| n66
    n70 -->|"OBJECTION"| n66
    n71 -->|"OBJECTION"| n66
    n72 -->|"OBJECTION"| n66
    n73 -->|"OBJECTION"| n66
    n74 -->|"OBJECTION"| n66
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- candidate maxim denies intended use while proposing instrumental belief and purposive use
