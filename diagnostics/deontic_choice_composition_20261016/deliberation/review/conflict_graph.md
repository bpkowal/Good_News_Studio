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
    n18["PROPOSITION: Perfect duties override imperfect duties"]
    n19["PROPOSITION: obligation"]
    n20["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n21["PROPOSITION: the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"]
    n22["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n23["PROPOSITION: the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"]
    n24["PROPOSITION: doing so may expose nearby staff to hazardous fumes; affected subject: nearby staff; modality: POSSIBLE"]
    n25["PROPOSITION: the technician may die during that delay; affected subject: the technician; modality: POSSIBLE"]
    n26["PROPOSED_REVISION: utilitarian: disputed full revision; not operative"]
    n27["PROPOSED_REVISION: care: disputed full revision; not operative"]
    n28["CLAIM: utilitarian: A0 assessment"]
    n29["CLAIM: utilitarian: A0 assessment"]
    n30["CLAIM: utilitarian: A1 assessment"]
    n31["CLAIM: utilitarian: A1 assessment"]
    n32["CLAIM: utilitarian: A1 assessment"]
    n33["UNRESOLVED_CONFLICT: utilitarian rationale imports Rawlsian normative language"]
    n34["CLAIM: care: A0 RESPONSIVE"]
    n35["FRAMEWORK_COMMITMENT: Control of hazardous setup creates responsibility for occupant’s safety"]
    n36["FRAMEWORK_COMMITMENT: ACUTE_DEPENDENCY"]
    n37["CLAIM: care: A1 NEGLECTFUL"]
    n38["FRAMEWORK_COMMITMENT: Responsibility exists but is postponed"]
    n39["UNRESOLVED_CONFLICT: A0 Care relation claims scenario grounding without a matching action effect"]
    n40["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n41["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n42["CLAIM: virtue: A0 EXEMPLIFIES"]
    n43["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n44["CLAIM: virtue: A1 MIXED"]
    n45["UNRESOLVED_CONFLICT: If fumes are certainly fatal or highly likely to harm many, A0 becomes reckless"]
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
    n34 -->|"DEPENDENCY"| n20
    n34 -->|"DEPENDENCY"| n21
    n35 -->|"SUPPORT"| n34
    n36 -->|"SUPPORT"| n34
    n4 -->|"SUPPORT"| n37
    n37 -->|"DEPENDENCY"| n20
    n37 -->|"DEPENDENCY"| n21
    n38 -->|"SUPPORT"| n37
    n36 -->|"SUPPORT"| n37
    n42 -->|"DEPENDENCY"| n22
    n42 -->|"DEPENDENCY"| n23
    n43 -->|"SUPPORT"| n42
    n44 -->|"DEPENDENCY"| n22
    n44 -->|"DEPENDENCY"| n23
    n43 -->|"SUPPORT"| n44
```

Proposed review queue (at most two issues):

- If fumes are certainly fatal or highly likely to harm many, A0 becomes reckless
- utilitarian rationale imports Rawlsian normative language
