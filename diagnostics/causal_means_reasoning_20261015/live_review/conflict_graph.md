# Logic_Puzzles claim/conflict projection

Read-only projection. Native records remain attributed; evidence links do not establish whole-record correctness.

```mermaid
flowchart LR
    n0["WORLD_EFFECT: pulls the lever"]
    n1["WORLD_EFFECT: redirects the tram onto a side track"]
    n2["WORLD_EFFECT: the tram will kill one worker"]
    n3["WORLD_EFFECT: spare ten workers"]
    n4["WORLD_EFFECT: does not"]
    n5["WORLD_EFFECT: heading toward ten workers on the main track"]
    n6["WORLD_EFFECT: it will kill ten workers"]
    n7["WORLD_EFFECT: spare one worker"]
    n8["PROPOSITION: If she pulls the lever"]
    n9["PROPOSITION: If she does not"]
    n10["PROPOSITION: pulls the lever; affected subject: a lever"]
    n11["PROPOSITION: redirects the tram onto a side track; affected subject: the tram"]
    n12["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n13["PROPOSITION: spare ten workers; affected subject: ten workers"]
    n14["PROPOSITION: does not; affected subject: a lever"]
    n15["PROPOSITION: heading toward ten workers on the main track; affected subject: the tram; magnitude or qualifier: ten"]
    n16["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n17["PROPOSITION: spare one worker; affected subject: one worker"]
    n18["PROPOSITION: ten workers"]
    n19["PROPOSITION: Dana cannot stop the tram."]
    n20["PROPOSITION: If she pulls the lever, the tram will kill that worker and spare the ten."]
    n21["PROPOSITION: If she does not, it will kill the ten and spare the one."]
    n22["PROPOSITION: Dana can pull a lever that redirects the tram onto a side track, where one worker stands."]
    n23["PROPOSITION: A runaway tram is heading toward ten workers on the main track."]
    n24["PROPOSITION: pulls the lever; affected subject: a lever"]
    n25["PROPOSITION: redirects the tram onto a side track; affected subject: the tram"]
    n26["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n27["PROPOSITION: spare ten workers; affected subject: ten workers"]
    n28["PROPOSITION: does not; affected subject: a lever"]
    n29["PROPOSITION: heading toward ten workers on the main track; affected subject: the tram; magnitude or qualifier: ten"]
    n30["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n31["CLAIM: deontological: A0 PROHIBITED"]
    n32["FRAMEWORK_COMMITMENT: do not intentionally kill the innocent"]
    n33["FRAMEWORK_COMMITMENT: Perfect negative duties outrank imperfect aid duties"]
    n34["CLAIM: deontological: A1 PERMISSIBLE"]
    n35["FRAMEWORK_COMMITMENT: refrain from killing an innocent person"]
    n36["FRAMEWORK_COMMITMENT: Omission that avoids killing satisfies higher-priority perfect duty"]
    n37["UNRESOLVED_CONFLICT: intended-as-means classification corrected to side-effect"]
    n32 -->|"SUPPORT"| n31
    n33 -->|"SUPPORT"| n31
    n35 -->|"SUPPORT"| n34
    n36 -->|"SUPPORT"| n34
```

Proposed review queue (at most two issues):

- intended-as-means classification corrected to side-effect
