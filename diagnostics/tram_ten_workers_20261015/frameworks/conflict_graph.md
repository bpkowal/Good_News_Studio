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
    n45["UNRESOLVED_CONFLICT: numeric reversal threshold lacks a scenario-grounded quantity"]
    n46["CLAIM: deontological: A0 CONFLICTED"]
    n47["FRAMEWORK_COMMITMENT: do not kill an innocent person"]
    n48["FRAMEWORK_COMMITMENT: Perfect negative duty overrides imperfect positive duty"]
    n49["UNRESOLVED_CONFLICT: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n50["UNRESOLVED_CONFLICT: resolved adjudication rests on unsupported decisive premises"]
    n51["CLAIM: deontological: A1 PERMISSIBLE"]
    n52["FRAMEWORK_COMMITMENT: refrain from killing"]
    n53["FRAMEWORK_COMMITMENT: Perfect duty not to kill prevails"]
    n54["UNRESOLVED_CONFLICT: dana&#x27;s duty &#x27;refrain from killing&#x27; protecting side-track worker ↔ dana&#x27;s competing duty &#x27;duty to rescue those in peril&#x27; protecting ten workers"]
    n55["UNRESOLVED_CONFLICT: A0 prohibited by do not kill an innocent person ↔ whether the harm-relation follows the admitted graph&#x27;s causal topology rather than a relabel chosen to fit the verdict"]
    n56["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: doing-harm classification lacks an agent-caused settled welfare harm on the protected party"]
    n57["UNRESOLVED_CONFLICT: A0 ADJUDICATION_CALIBRATION: resolved adjudication rests on unsupported decisive premises"]
    n58["CLAIM: care: A0 RESPONSIVE"]
    n59["FRAMEWORK_COMMITMENT: Control over danger imposes responsibility to act"]
    n60["FRAMEWORK_COMMITMENT: ENTRUSTED_RESPONSIBILITY"]
    n61["CLAIM: care: A1 NEGLECTFUL"]
    n62["FRAMEWORK_COMMITMENT: Responsibility ignored"]
    n63["UNRESOLVED_CONFLICT: A0 competing Care claim remains contested"]
    n64["UNRESOLVED_CONFLICT: A1 competing Care claim remains contested"]
    n65["CLAIM: virtue: A0 EXEMPLIFIES"]
    n66["FRAMEWORK_COMMITMENT: PRACTICAL_WISDOM"]
    n67["CLAIM: virtue: A1 MIXED"]
    n68["CLAIM: rawlsian: A0 assessment"]
    n69["FRAMEWORK_COMMITMENT: MAXIMIN_PRIMARY_GOODS"]
    n70["CLAIM: rawlsian: A1 assessment"]
    n71["UNRESOLVED_CONFLICT: decisive numbers are not tied to the affected subject or Rawlsian position"]
    n46 -->|"DEPENDENCY"| n26
    n46 -->|"DEPENDENCY"| n27
    n47 -->|"SUPPORT"| n46
    n48 -->|"SUPPORT"| n46
    n49 -->|"OBJECTION"| n46
    n50 -->|"OBJECTION"| n46
    n51 -->|"DEPENDENCY"| n26
    n51 -->|"DEPENDENCY"| n27
    n52 -->|"SUPPORT"| n51
    n53 -->|"SUPPORT"| n51
    n3 -->|"SUPPORT"| n58
    n58 -->|"DEPENDENCY"| n28
    n59 -->|"SUPPORT"| n58
    n60 -->|"SUPPORT"| n58
    n6 -->|"SUPPORT"| n61
    n61 -->|"DEPENDENCY"| n28
    n62 -->|"SUPPORT"| n61
    n60 -->|"SUPPORT"| n61
    n66 -->|"SUPPORT"| n65
    n66 -->|"SUPPORT"| n67
    n68 -->|"DEPENDENCY"| n35
    n68 -->|"DEPENDENCY"| n36
    n69 -->|"SUPPORT"| n68
    n70 -->|"DEPENDENCY"| n35
    n70 -->|"DEPENDENCY"| n36
    n69 -->|"SUPPORT"| n70
```

Proposed review queue (at most two issues):

- doing-harm classification lacks an agent-caused settled welfare harm on the protected party
- resolved adjudication rests on unsupported decisive premises
