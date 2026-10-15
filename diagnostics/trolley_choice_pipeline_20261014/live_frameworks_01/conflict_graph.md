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
```

Proposed review queue (at most two issues):

