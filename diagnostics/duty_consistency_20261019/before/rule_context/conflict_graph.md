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
    n16["PROPOSED_ARGUMENT: deontological: argument without committed ledger"]
    n17["UNRESOLVED_CONFLICT: No committed native ledger; assess the retained argument without treating it as operative."]
    n18["UNRESOLVED_CONFLICT: initial=DUTY_CONSISTENCY_REPAIR: {&quot;A1&quot;: {&quot;dt&quot;: &quot;IMPERFECT&quot;, &quot;pb&quot;: &quot;PERFECT_DUTY&quot;, &quot;v&quot;: &quot;PERMISSIBLE&quot;, &quot;gv&quot;: &quot;PRIMARY&quot;, &quot;res&quot;: &quot;RESOLVED&quot;}}; repair=Duty repair changed the original duty classification"]
    n17 -->|"OBJECTION"| n16
    n18 -->|"OBJECTION"| n16
```

Proposed review queue (at most two issues):

- No committed native ledger; assess the retained argument without treating it as operative.
- initial=DUTY_CONSISTENCY_REPAIR: {"A1": {"dt": "IMPERFECT", "pb": "PERFECT_DUTY", "v": "PERMISSIBLE", "gv": "PRIMARY", "res": "RESOLVED"}}; repair=Duty repair changed the original duty classification
