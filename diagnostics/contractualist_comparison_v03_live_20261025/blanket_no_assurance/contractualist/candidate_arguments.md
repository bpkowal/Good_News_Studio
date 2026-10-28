# Candidate arguments

Nonvoting argument overlays. These are not admitted world-state graphs.

## contractualist

STAGED_PARTIAL_ARGUMENT

Critic's proposed remaining conclusion (check accepted graph below): Giving the blanket to Rowan (Action A) is uniquely defensible because the strongest reasonable rejection falls on the alternative principle that would give the blanket to Sasha.

```mermaid
flowchart TB
n0["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: If Ellis gives the blanket to Rowan, Rowan has an 80% chance of survival."]
n1["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: If Ellis gives the blanket to Sasha, Rowan will die."]
n2["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: If Ellis gives the blanket to Rowan, Sasha will die."]
n3["FACTUAL_PREMISE / AFFIRMED / UNASSESSED: If Ellis gives the blanket to Sasha, Sasha has a 60% chance of survival."]
n4["PRINCIPLE / AFFIRMED / CONDITIONAL: When only one person can receive a life-saving resource, it should be given to the person with the higher probability of survival."]
n5["PRINCIPLE / AFFIRMED / CONDITIONAL: When only one person can receive a life-saving resource, it should be given to the person whose prospects will improve the most (higher absolute gain)"]
n6["NORMATIVE_PREMISE / AFFIRMED / CONDITIONAL: A complaint is stronger when it involves a larger loss in probability of survival compared with what that person could have under an alternative princ"]
n7["NORMATIVE_JUDGMENT / AFFIRMED / REQUIRES_REASSESSMENT: Rowan’s complaint against principle PR2 (blanket to Sasha) is stronger than Sasha’s complaint against principle PR1 (blanket to Rowan)."]
n8["NORMATIVE_JUDGMENT / AFFIRMED / REQUIRES_REASSESSMENT: Principle PR1 is not reasonably rejectable; principle PR2 is reasonably rejectable."]
n9["NORMATIVE_JUDGMENT / AFFIRMED / REQUIRES_REASSESSMENT: Action A (give blanket to Rowan) is defensible under the least reasonably rejectable principle."]
n0 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n7
n1 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n7
n2 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n7
n3 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n7
n6 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n7
n7 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n8
n8 -->|NORM_APPLICATION / REQUIRES_REASSESSMENT| n9
```

Remaining questions: Whether a 20 % difference in lost survival chance is enough to make one complaint decisively stronger remains open to reasonable dispute.


## Binding issues (not moral verdicts)

[
  {
    "kind": "DETACHED_COMPARISON",
    "record_id": "C1"
  },
  {
    "kind": "VERDICT_USED_AS_PREMISE",
    "record_id": "STATEMENT:ce7ec42951b0143b54186925"
  }
]

## Comparison bookkeeping (not semantic validation)

{
  "version": "complaint-comparison/0.1",
  "findings": [],
  "status": "NO_STRUCTURAL_MISMATCH_FOUND",
  "matrix_authority": "MODEL_PROPOSED_NOT_SOURCE_VERIFIED",
  "semantic_support": "NOT_INDEPENDENTLY_ASSESSED",
  "moral_ranking_computed": false,
  "operative": false,
  "recommendation_status": "UNIQUE_DEFENSIBLE",
  "remaining_complaint_ids": [
    "R_Sasha_against_A"
  ]
}

## Burdens by option

| Burden / person | Option | Status |
| --- | --- | --- |
| Risk of death from hypothermia / Rowan | A | PRESENT |
| Risk of death from hypothermia / Rowan | B | PRESENT |
| Risk of death from hypothermia / Sasha | A | PRESENT |
| Risk of death from hypothermia / Sasha | B | PRESENT |


Full principles, reasons, comparisons and proposed attacks: [response.json](response.json).
