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
    n24["PROPOSITION: If she pulls the lever"]
    n25["PROPOSITION: If she does not"]
    n26["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n27["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n28["PROPOSITION: ten workers are entrusted to the decision maker"]
    n29["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n30["PROPOSITION: spare ten workers; affected subject: ten workers"]
    n31["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n32["PROPOSITION: spare one worker; affected subject: one worker"]
    n33["PROPOSITION: If she pulls the lever, the tram will kill that worker and spare the ten."]
    n34["PROPOSITION: If she does not, it will kill the ten and spare the one."]
    n35["PROPOSITION: the tram will kill one worker; affected subject: one worker"]
    n36["PROPOSITION: it will kill ten workers; affected subject: ten workers"]
    n37["CLAIM: utilitarian: A0 assessment"]
    n38["CLAIM: utilitarian: A0 assessment"]
    n39["CLAIM: utilitarian: A0 assessment"]
    n40["CLAIM: utilitarian: A0 assessment"]
    n41["CLAIM: utilitarian: A1 assessment"]
    n42["CLAIM: utilitarian: A1 assessment"]
    n43["CLAIM: utilitarian: A1 assessment"]
    n44["CLAIM: utilitarian: A1 assessment"]
    n45["CLAIM: deontological: A0 CONFLICTED"]
    n46["FRAMEWORK_COMMITMENT: do not kill an innocent person"]
    n47["FRAMEWORK_COMMITMENT: Perfect negative duty overrides imperfect positive duty"]
    n48["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n49["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n50["CLAIM: deontological: A1 PERMISSIBLE"]
    n51["FRAMEWORK_COMMITMENT: refrain from killing"]
    n52["FRAMEWORK_COMMITMENT: Perfect duty not to kill prevails"]
    n53["UNRESOLVED_CONFLICT: dana&#x27;s duty &#x27;refrain from killing&#x27; protecting side-track worker ↔ dana&#x27;s competing duty &#x27;duty to rescue those in peril&#x27; protecting ten workers"]
    n54["UNRESOLVED_CONFLICT: A0 prohibited by do not kill an innocent person ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n55["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n56["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n57["CLAIM: care: A0 RESPONSIVE"]
    n58["FRAMEWORK_COMMITMENT: Control over danger imposes responsibility to act"]
    n59["FRAMEWORK_COMMITMENT: ENTRUSTED_RESPONSIBILITY"]
    n60["CLAIM: care: A1 NEGLECTFUL"]
    n61["FRAMEWORK_COMMITMENT: Responsibility ignored"]
    n62["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n63["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n64["CLAIM: virtue: A0 EXEMPLIFIES"]
    n65["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n66["CLAIM: virtue: A1 MIXED"]
    n67["CLAIM: rawlsian: A0 assessment"]
    n68["FRAMEWORK_COMMITMENT: MAXIMIN_PRIMARY_GOODS"]
    n69["CLAIM: rawlsian: A1 assessment"]
    n70["UNRESOLVED_CONFLICT: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n45 -->|"DEPENDENCY"| n26
    n45 -->|"DEPENDENCY"| n27
    n46 -->|"SUPPORT"| n45
    n47 -->|"SUPPORT"| n45
    n48 -->|"OBJECTION"| n45
    n49 -->|"OBJECTION"| n45
    n50 -->|"DEPENDENCY"| n26
    n50 -->|"DEPENDENCY"| n27
    n51 -->|"SUPPORT"| n50
    n52 -->|"SUPPORT"| n50
    n3 -->|"SUPPORT"| n57
    n57 -->|"DEPENDENCY"| n28
    n58 -->|"SUPPORT"| n57
    n59 -->|"SUPPORT"| n57
    n6 -->|"SUPPORT"| n60
    n60 -->|"DEPENDENCY"| n28
    n61 -->|"SUPPORT"| n60
    n59 -->|"SUPPORT"| n60
    n65 -->|"SUPPORT"| n64
    n65 -->|"SUPPORT"| n66
    n67 -->|"DEPENDENCY"| n35
    n67 -->|"DEPENDENCY"| n36
    n68 -->|"SUPPORT"| n67
    n69 -->|"DEPENDENCY"| n35
    n69 -->|"DEPENDENCY"| n36
    n68 -->|"SUPPORT"| n69
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
