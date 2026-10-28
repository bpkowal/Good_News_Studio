# Candidate arguments

Nonvoting argument overlays. These are not admitted world-state graphs.

## Means-Assessment

STAGED_ARGUMENT

Strongest remaining conclusion: Len does employ Ivo as a means, and the mechanism’s failure does not change that status because the means relation turns on Len’s instrumental intention, not causal success.

```mermaid
flowchart TB
n0["BELIEF / AFFIRMED / UNASSESSED: Len believes that having Ivo hold the contact plate will close the gate and avert flood damage."]
n1["PURPOSIVE_USE / AFFIRMED / UNASSESSED: Len purposefully asks Ivo to hold the plate for that flood-prevention end and Ivo voluntarily agrees."]
n2["PHYSICAL_CONTRIBUTION / AFFIRMED / UNASSESSED: The contact plate is in fact disconnected and cannot close the gate."]
n3["MAXIM / AFFIRMED / CONDITIONAL: If I believe that requesting someone’s help will avert serious harm, I may ask that person to perform the helpful act."]
n4["NORMATIVE_PREMISE / AFFIRMED / CONDITIONAL: One employs another person as a means whenever one intentionally involves that person in order to realize one’s own further end, as believed by the ac"]
n5["NORMATIVE_JUDGMENT / AFFIRMED / REQUIRES_REASSESSMENT: Therefore Len does employ Ivo as a means, and the later discovery that the mechanism was ineffective does not alter that employment status."]
n0 -->|PRACTICAL_REASONING / REQUIRES_REASSESSMENT| n5
n1 -->|PRACTICAL_REASONING / REQUIRES_REASSESSMENT| n5
n4 -->|PRACTICAL_REASONING / REQUIRES_REASSESSMENT| n5
```

Remaining questions: Does the posted rule barring untrained operators impose a stronger duty on Len than the permissibility inferred from his belief‐based maxim?; If so, how should conflicting duties—respect for Ivo’s autonomy versus adherence to the rule—be ranked in the absence of an explicit priority clause?

## Rule-vs-Consent Permissibility

STAGED_ARGUMENT

Strongest remaining conclusion: Even acknowledging that Len uses Ivo as a means, Len’s request is presumptively impermissible because it violates the standing safety rule whose moral authority is (absent further facts) not overridden.

```mermaid
flowchart TB
n0["SOURCE_RULE / AFFIRMED / UNASSESSED: A posted rule states that only trained staff should operate the controls and gives no emergency exception."]
n1["PURPOSIVE_USE / AFFIRMED / UNASSESSED: Len nevertheless enlists untrained volunteer Ivo to handle the control."]
n2["NORMATIVE_PREMISE / AFFIRMED / CONDITIONAL: One ought to comply with standing safety rules unless a clearly overriding moral duty is present."]
n3["NORMATIVE_PREMISE / AFFIRMED / CONDITIONAL: Obtaining a person’s voluntary and informed consent normally satisfies the duty not to treat that person merely as a means."]
n4["NORMATIVE_PREMISE / AFFIRMED / UNASSESSED: No explicit emergency override of the posted rule is given in the circumstances described."]
n5["NORMATIVE_JUDGMENT / AFFIRMED / CONDITIONAL: Accordingly, Len’s consent-based request remains presumptively impermissible because it violates the standing safety rule, whose authority is not defe"]
n6["NORMATIVE_JUDGMENT / AFFIRMED / REQUIRES_REASSESSMENT: Len does employ Ivo as a means, because Len believes that Ivo’s holding the contact plate will help achieve the end of closing the gate."]
n0 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n5
n1 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n5
n2 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n5
n4 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n5
n3 -->|NORM_PRIORITY / CONDITIONAL| n5
n2 -->|NORM_PRIORITY / CONDITIONAL| n5
n0 -->|PRACTICAL_REASONING / REQUIRES_REASSESSMENT| n6
n1 -->|PRACTICAL_REASONING / REQUIRES_REASSESSMENT| n6
```

Remaining questions: Does an imminent flood danger in fact create a stronger duty to avert harm that would override the posted rule?; Is Ivo’s consent sufficiently informed, given the unknown disconnection and any possible risks to him?
